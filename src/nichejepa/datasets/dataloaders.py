import gc
import math
import time
from collections.abc import Iterator
from logging import getLogger

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .cell_datasets import CellBaseDataset


logger = getLogger()


_GLOBAL_SEED = 0


class CustomDistributedLengthGroupedSampler(DistributedSampler):
    def __init__(self,
                 cell_dataset: Dataset,
                 batch_size: int,
                 num_replicas: int | None = None,
                 rank: int | None = None,
                 seed: int = 0,
                 drop_last: bool = False,
                 lengths: list[int] | None = None,
<<<<<<< HEAD
                 mega_batch_mult_max: int = 1000,
=======
>>>>>>> main
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

        if self.drop_last and len(self.cell_dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly
            # divisible. This is to ensure each rank receives the same
            # amount of data when using this sampler. If the dataset
            # length is evenly divisible by # of replicas, then there is
            # no need to drop any data since the dataset will be split
            # equally.
            self.num_samples = math.ceil(
                (len(self.cell_dataset) - self.num_replicas
                ) / self.num_replicas)
        else:
            self.num_samples = math.ceil(
                len(self.cell_dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
        self.lengths = self.cell_dataset.n_nz_tokens
<<<<<<< HEAD
        self.mega_batch_mult_max = mega_batch_mult_max
=======
>>>>>>> main

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = self._get_length_grouped_indices(
            generator=g,
            mega_batch_mult_max=self.mega_batch_mult_max)

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
<<<<<<< HEAD
                                    mega_batch_mult_max: int = 1000,
=======
>>>>>>> main
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
        """
        #t0 = time.perf_counter()

        if mega_batch_mult is None:
            # Default for mega_batch_mult: 1000 or the number to get 4
            # mega batches, whichever is smaller.
            mega_batch_mult = min(
                len(self.lengths) // (self.batch_size * 4),
                mega_batch_mult_max) # 1000
            # Just in case, for tiny datasets
            if mega_batch_mult == 0:
                mega_batch_mult = 1

        # We need to use torch for the random part as a distributed
        # sampler will set the random seed for torch.
        indices = torch.randperm(len(self.lengths), generator=generator)
        megabatch_size = mega_batch_mult * self.batch_size
        megabatches = [
            indices[i : i + megabatch_size].tolist()
            for i in range(0, len(self.lengths), megabatch_size)]
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
        #print(f"[Length Sampling] Took {elapsed_ms:.3f}ms.")

        return indices


def init_dataloader_and_sampler(cell_dataset: CellBaseDataset,
                                batch_size: int,
                                distributed: bool,
                                world_size: int,
                                rank: int,
                                mega_batch_mult_max: int,
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
<<<<<<< HEAD
            seed=_GLOBAL_SEED,
            mega_batch_mult_max=mega_batch_mult_max)
=======
            seed=_GLOBAL_SEED)
>>>>>>> main
    else:
        dist_sampler = DistributedSampler(
            dataset=cell_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=_GLOBAL_SEED,
        )
        
    dataloader = DataLoader(cell_dataset,
                            batch_size=batch_size,
                            sampler=dist_sampler,
                            **dataloader_kwargs)
    logger.info('Dataloader and -sampler created.')

<<<<<<< HEAD
    return dataloader, dist_sampler
=======
    return dataloader, dist_sampler
>>>>>>> main
