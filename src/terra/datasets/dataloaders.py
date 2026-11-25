import gc
import logging
import math
import time
from collections.abc import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .cell_datasets import CellBaseDataset


logger = logging.getLogger(__name__)


_GLOBAL_SEED = 0


class EpochIndexLists:
    """
    Rows of the training split that each epoch presents, read from a 2-D
    integer ``.npy`` array (one row per epoch, all of equal length; a row may
    repeat or leave out rows of the split).

    Built offline for sampling designs (a fixed subset of centres, repeated
    passes over a subset, a budget that ends mid-pass, thinned centres
    re-drawn every epoch, weighted draws). All epochs have the same length,
    so the number of iterations per epoch, and with it the learning-rate,
    weight-decay and EMA schedules, are fixed for the whole run. The sampler
    spreads the copies of a repeated row evenly over the epoch (see
    `spread_repeats`).

    Parameters
    ----------
    file_path:
        ``.npy`` file with an integer array of shape ``[n_epochs, n]``; it is
        memory-mapped, checked in full when loaded, and an epoch's row is read
        when the epoch starts.
    n_rows:
        Number of rows of the training split (the valid indices are
        ``0 .. n_rows - 1``).
    n_epochs:
        Number of epochs the run trains; the file must hold at least as many.
    """
    def __init__(self, file_path: str, n_rows: int, n_epochs: int):
        array = np.load(file_path, mmap_mode='r')
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(
                f"Epoch index lists '{file_path}' must be a 2-D integer array "
                f"[n_epochs, n], not {array.ndim}-D {array.dtype}.")
        if array.shape[0] < n_epochs:
            raise ValueError(
                f"Epoch index lists '{file_path}' hold {array.shape[0]} epochs, "
                f"but the run trains {n_epochs}.")
        if array.shape[1] == 0:
            raise ValueError(f"Epoch index lists '{file_path}' are empty.")
        self.file_path = str(file_path)
        self.array = array
        self.n_rows = int(n_rows)
        self.length = int(array.shape[1])
        # Fail at the start, not in the epoch with a bad row
        for epoch in range(n_epochs):
            self._check(epoch, array[epoch])

    def _check(self, epoch: int, indices: np.ndarray):
        lo, hi = int(indices.min()), int(indices.max())
        if lo < 0 or hi >= self.n_rows:
            raise ValueError(
                f"Epoch {epoch} of '{self.file_path}' has row indices in "
                f"[{lo}, {hi}], outside the training split's 0..{self.n_rows - 1}.")

    def __call__(self, epoch: int) -> np.ndarray:
        # A copy (writable, in memory) of the memory-mapped row
        indices = np.array(self.array[epoch], dtype=np.int64)
        self._check(epoch, indices)
        return indices


def spread_repeats(entries: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Reorder a shuffled index list so that the copies of each row are spread
    evenly over it, and return it with the largest number of copies of a
    row.

    The k-th copy (k = 0 .. c - 1, in list order) of a row with c copies,
    at relative position u in the list, moves to about position (k + u) / c:
    a list of c passes over a subset becomes c consecutive passes in random
    order, and rows drawn with replacement have their copies spread over
    the epoch. Without this, copies that fall into one mega-batch of the
    length-grouped sampler are sorted next to each other (they have the same
    length) and trained in the same step. A list without repeats is returned
    unchanged.
    """
    n = entries.size
    order = np.argsort(entries, kind='stable')
    sorted_entries = entries[order]
    first = np.ones(n, dtype=bool)
    first[1:] = sorted_entries[1:] != sorted_entries[:-1]
    if first.all():
        return entries, 1
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, n))
    occurrence = np.empty(n, dtype=np.int64)
    occurrence[order] = np.arange(n) - np.repeat(starts, counts)
    copies = np.empty(n, dtype=np.int64)
    copies[order] = np.repeat(counts, counts)
    key = (occurrence + np.arange(n) / n) / copies
    return entries[np.argsort(key, kind='stable')], int(counts.max())


class CustomDistributedLengthGroupedSampler(DistributedSampler):
    def __init__(self,
                 cell_dataset: Dataset,
                 batch_size: int,
                 num_replicas: int | None = None,
                 rank: int | None = None,
                 seed: int = 0,
                 drop_last: bool = False,
                 lengths: list[int] | None = None,
                 mega_batch_mult_max: int = 1000,
                 epoch_index_lists: EpochIndexLists | None = None,
                 ):
        """
        Distributed Sampler that samples indices in a way that groups
        together features of the dataset of roughly the same length
        while keeping a bit of randomness.
        
        Adapted from Theodoris, C. V. et al. Transfer learning enables
        predictions in network biology. Nature 618, 616–624 (2023);
        https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/pretrainer.py
        and
        https://huggingface.co/transformers/v4.4.2/_modules/transformers/trainer_pt_utils.html
        (28.09.2024).

        Parameters
        -----------
        cell_dataset:
            CellGraphDataset or CellNeighborhoodDataset.
        batch_size:
            Batch size for the dataloader and -sampler.
        num_replicas:
            Number of replicas of the distributed sampler.
        rank:
            Rank of the distributed sampler.
        seed:
            Seed for the distributed sampler.
        drop_last:
            If `True`, drop the last incomplete batch.
        lengths:
            List of lengths of the features in the dataset.
        epoch_index_lists:
            If given, each epoch presents the rows of its index list
            (`EpochIndexLists`) instead of every row once: the list is
            shuffled and length-grouped exactly as the rows of the dataset
            are without it, so a list of all rows in order gives the order of
            the default sampler.
        """
        # Validate distributed package
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available.")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available.")
            rank = dist.get_rank()

        self.cell_dataset = cell_dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.epoch_index_lists = epoch_index_lists
        if (epoch_index_lists is not None
                and epoch_index_lists.n_rows != len(self.cell_dataset)):
            raise ValueError(
                f"The epoch index lists index {epoch_index_lists.n_rows} rows, "
                f"but the dataset has {len(self.cell_dataset)}.")
        # Items drawn per epoch (every row once without index lists)
        n_items = (len(self.cell_dataset) if epoch_index_lists is None
                   else epoch_index_lists.length)

        if self.drop_last and n_items % self.num_replicas != 0:
            # Split to nearest available length that is evenly
            # divisible. This is to ensure each rank receives the same
            # amount of data when using this sampler. If the dataset
            # length is evenly divisible by # of replicas, then there is
            # no need to drop any data since the dataset will be split
            # equally.
            self.num_samples = math.ceil(
                (n_items - self.num_replicas
                ) / self.num_replicas)
        else:
            self.num_samples = math.ceil(
                n_items / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
        self.lengths = self.cell_dataset.n_nz_tokens
        self.mega_batch_mult_max = mega_batch_mult_max

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = self._get_length_grouped_indices(
            generator=g,
            mega_batch_mult_max=self.mega_batch_mult_max,
            base_indices=(None if self.epoch_index_lists is None
                          else self.epoch_index_lists(self.epoch)))

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def _get_length_grouped_indices(self,
                                    mega_batch_mult: int | None = None,
                                    generator: torch.Generator | None = None,
                                    mega_batch_mult_max: int = 1000,
                                    base_indices: np.ndarray | None = None,
                                    ) -> list[int]:
        """
        Return a list of indices so that each slice of `batch_size`
        consecutive indices correspond to elements of similar lengths.
        To do this, the indices are (1) randomly permuted, (2) grouped
        in mega-batches of size `mega_batch_mult * batch_size`, (3)
        sorted by length in each mega-batch. The result is the
        concatenation of all mega-batches, with the batch of
        `batch_size` containing the element of maximum length placed
        first, so that an OOM error would happen earlier rather than
        later.

        With `base_indices` (an epoch's index list), its entries are
        permuted and grouped instead of the rows ``0 .. len(dataset) - 1``;
        ``base_indices = arange(len(dataset))`` gives the same list as
        without it. The copies of a repeated row are spread over the epoch
        (`spread_repeats`), and the mega-batches are kept shorter than a
        pass, so that copies rarely share a mega-batch (and hence a step).
        """
        #t0 = time.perf_counter()

        n_items = (len(self.lengths) if base_indices is None
                   else len(base_indices))

        # We need to use torch for the random part as a distributed
        # sampler will set the random seed for torch.
        indices = torch.randperm(n_items, generator=generator)
        max_copies = 1
        if base_indices is not None:
            # The permuted positions pick the list's entries (rows, which may
            # repeat)
            entries, max_copies = spread_repeats(
                np.asarray(base_indices, dtype=np.int64)[indices.numpy()])
            indices = torch.from_numpy(entries)

        if mega_batch_mult is None:
            # Default for mega_batch_mult: 1000 or the number to get 4
            # mega batches, whichever is smaller.
            mega_batch_mult = min(
                n_items // (self.batch_size * 4),
                mega_batch_mult_max) # 1000
            if max_copies > 1:
                # At most a pass of the most repeated row per mega-batch
                mega_batch_mult = min(
                    mega_batch_mult,
                    n_items // (max_copies * self.batch_size))
            # Just in case, for tiny datasets
            if mega_batch_mult == 0:
                mega_batch_mult = 1

        megabatch_size = mega_batch_mult * self.batch_size
        megabatches = [
            indices[i : i + megabatch_size].tolist()
            for i in range(0, n_items, megabatch_size)]
        megabatches = [
            list(sorted(
                megabatch, key=lambda i: self.lengths[i], reverse=True))
            for megabatch in megabatches]

        # The rest is to get the biggest batch first.
        # Since each megabatch is sorted by descending length, the
        # longest element is the first
        megabatch_maximums = [
            self.lengths[megabatch[0]] for megabatch in megabatches]
        max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
        # Switch to put the longest element in first position
        megabatches[0][0], megabatches[max_idx][0] = (
            megabatches[max_idx][0],
            megabatches[0][0])
        
        indices = [item for sublist in megabatches for item in sublist]

        #t1 = time.perf_counter()
        #elapsed_ms = (t1 - t0) * 1000

        return indices


def init_dataloader_and_sampler(cell_dataset: CellBaseDataset,
                                batch_size: int,
                                distributed: bool,
                                world_size: int,
                                rank: int,
                                mega_batch_mult_max: int,
                                seed: int = _GLOBAL_SEED,
                                shuffle: bool = False,
                                epoch_index_lists: EpochIndexLists | None = None,
                                **dataloader_kwargs,
    ) -> tuple[torch.utils.data.DataLoader,
               torch.utils.data.distributed.DistributedSampler | None]:
    """
    Initialize dataloader and -sampler from a cell dataset.

    Parameters
    -----------
    cell_dataset:
        CellGraphDataset or CellNeighborhoodDataset.
    batch_size:
        Batch size for the dataloader and -sampler.
    distributed:
        If `True`, use distributed mode.
    world_size:
        Number of replicas of the distributed sampler.
    rank:
        Rank of the distributed sampler.
    seed:
        Seed of the sampler's shuffling (the training seed; defaults to the
        historical seed 0).
    shuffle:
        If `True` and not `distributed`, shuffle the data in every epoch (call
        the sampler's `set_epoch` before each epoch). Default `False` keeps the
        order of the dataset.
    epoch_index_lists:
        Rows each epoch presents (see `EpochIndexLists`); needs the
        length-grouped sampler (`distributed`).
    dataloader_kwargs:
        Keyword arguments for the dataloader.

    Returns
    --------
    data_loader:
        Torch dataloader.
    dist_sampler (optional):
        Torch distributed sampler.
    """
    if distributed:
        dist_sampler = CustomDistributedLengthGroupedSampler(
            cell_dataset=cell_dataset,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            mega_batch_mult_max=mega_batch_mult_max,
            epoch_index_lists=epoch_index_lists)
    else:
        if epoch_index_lists is not None:
            raise ValueError(
                "Epoch index lists need the length-grouped sampler: set "
                "data.use_sampler: true.")
        dist_sampler = DistributedSampler(
            dataset=cell_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        
    dataloader = DataLoader(cell_dataset,
                            batch_size=batch_size,
                            sampler=dist_sampler,
                            **dataloader_kwargs)
    logger.info('Dataloader and -sampler created.')

    return dataloader, dist_sampler