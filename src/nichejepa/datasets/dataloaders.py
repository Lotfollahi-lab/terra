from typing import Optional

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


_GLOBAL_SEED = 0


def init_dataloader_and_sampler(batch_size: int,
                                dataset: CellBaseDataset,
                                vocab_size: int,
                                collator=None,
                                pin_mem: bool=True,
                                num_workers: int=8,
                                world_size: int=1,
                                rank: int=0,
                                drop_last: bool=True,
                                seq_len_cell: int=0,
                                seq_len_neighborhood: int=0,
                                special_tokens: list
                                distributed: bool=True,
    ) -> Tuple[torch.utils.data.DataLoader,
               Optional[torch.utils.data.distributed.DistributedSampler]]:):
    """
    Initialize dataloader and -sampler from a CellNeighborhoodDataset or 
    CellGraphDataset.

    Parameters
    -----------
    batch_size:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    n_nonzero_cell_tokens
    vocab_size:
        Size of the vocabulary.
    collator:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    pin_mem:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    num_workers:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    world_size:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler.
    rank:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler.
    drop_last:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    seq_len_cell:
        Sequence length of the cell tokens.
    seq_len_neighborhood:
        Sequence length of the neighborhood tokens.
    distributed:
        If 'True', use distributed mode.

    Returns
    --------
    dataset:
        Torch CellNeighborhoodDataset.
    data_loader:
        Torch data loader based on CellNeighborhoodDataset.
    dist_sampler:
        Torch distributed sampler based on CellNeighborhoodDataset.
    """
    if distributed:
        dist_sampler = CustomDistributedLengthGroupedSampler(
            dataset=dataset,
            batch_size=batch_size,
            n_nonzero_tokens_per_cell=n_nonzero_tokens_per_cell,
            num_replicas=world_size,
            rank=rank,
            seed=_GLOBAL_SEED)

        data_loader = DataLoader(dataset,
                                 collate_fn=collator,
                                 sampler=dist_sampler,
                                 batch_size=batch_size,
                                 drop_last=drop_last,
                                 pin_memory=pin_mem,
                                 num_workers=num_workers,
                                 persistent_workers=False)
        logger.info('Dataloader and -sampler created.')

        return dataloader, dist_sampler
    else:
        data_loader = DataLoader(dataset,
                                 collate_fn=collator,
                                 batch_size=batch_size,
                                 drop_last=drop_last,
                                 pin_memory=pin_mem,
                                 num_workers=num_workers,
                                persistent_workers=False)
        logger.info('Dataloader created.')
        
        return dataloader


class CustomDistributedLengthGroupedSampler(DistributedSampler):
    """
    Distributed Sampler that samples indices in a way that groups together
    features of the dataset of roughly the same length while keeping a bit of
    randomness.
    
    This class was adapted from
    https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/pretrainer.py.

    Parameters
    -----------

    Returns
    -----------
    """
    # Copied and adapted from PyTorch DistributedSampler.
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seq_len_cell: int,
        seq_len_neighborhood: int,
        num_replicas: Optional[int]=None,
        rank: Optional[int]=None,
        seed: int=0,
        n_nonzero_tokens_per_cell: list,
        drop_last: bool=False,
        lengths: Optional[List[int]]=None,
        ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
        self.lengths = n_nonzero_tokens_per_cell

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = _get_length_grouped_indices(generator=g)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def _get_length_grouped_indices(mega_batch_mult=None,
                                    generator=None
                                    ):
        """
        Return a list of indices so that each slice of :obj:`batch_size` consecutive
        indices correspond to elements of
        similar lengths. To do this, the indices are:
        - randomly permuted
        - grouped in mega-batches of size :obj:`mega_batch_mult * batch_size`
        - sorted by length in each mega-batch
        The result is the concatenation of all mega-batches, with the batch of
        :obj:`batch_size` containing the element of maximum length placed first, so
        that an OOM happens sooner rather than later.
        """
        # Default for mega_batch_mult: 50 or the number to get 4 megabatches,
        # whichever is smaller.
        if mega_batch_mult is None:
            # mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
            mega_batch_mult = min(len(self.lengths) // (self.batch_size * 4), 1000)
            # Just in case, for tiny datasets
            if mega_batch_mult == 0:
                mega_batch_mult = 1

        # We need to use torch for the random part as a distributed sampler will set
        # the random seed for torch.
        indices = torch.randperm(len(self.lengths), generator=generator)
        megabatch_size = mega_batch_mult * self.batch_size
        megabatches = [
            indices[i : i + megabatch_size].tolist()
            for i in range(0, len(self.lengths), megabatch_size)
        ]
        megabatches = [
            list(sorted(megabatch, key=lambda i: lengths[i], reverse=True))
            for megabatch in megabatches
        ]

        # The rest is to get the biggest batch first.
        # Since each megabatch is sorted by descending length, the longest element
        # is the first
        megabatch_maximums = [self.lengths[megabatch[0]] for megabatch in megabatches]
        max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
        # Switch to put the longest element in first position
        megabatches[0][0], megabatches[max_idx][0] = (
            megabatches[max_idx][0],
            megabatches[0][0],
        )

        return [item for sublist in megabatches for item in sublist]
