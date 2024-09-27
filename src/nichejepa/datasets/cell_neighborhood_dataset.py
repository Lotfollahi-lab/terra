import os
import subprocess
import time
from logging import getLogger
from typing import Optional, Tuple, List, Literal

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.distributed import CustomDistributedLengthGroupedSampler


_GLOBAL_SEED = 0
logger = getLogger()


class CellNeighborhoodDataset(Dataset):
    def __init__(self,
                 data: datasets.Dataset,
                 vocab_size: int,
                 seq_len_cell: int=0,
                 seq_len_neighborhood: int=0,
                 special_tokens: list=[
                    'cls', 'assay', 'species', 'tissue', 'gene_panel', 'batch'],
                 species_token: bool=False,
                 batch_token: bool=True,
                 gene_panel_token: bool=True,
                 has_gene_panel: bool=True,
                 sampling_strategy: Optional[str]=None,
                 sampling_seed: Optional[int]=42
                 ):
        """
        Torch CellNeighborhoodDataset class.

        Parameters
        -----------
        data:
            Huggingface dataset with cell and neighborhood tokens and cell-level
            metadata.
        vocab_size:
            Size of the vocabulary.
        seq_len_cell:
            Sequence length of the cell tokens.
        seq_len_neighborhood:
            Sequence length of the neighborhood tokens.
        has_cls:
            If 'True', a <cls> token is included for each cell at position 0.
        has_gene_panel:
        sampling_strategy:
            Token sampling strategy.
        """
        self.dataset = data
        self.len = len(self.dataset)
        self.vocab_size = vocab_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len = seq_len_cell + seq_len_neighborhood
        self.has_cls = has_cls
        self.has_gene_panel = has_gene_panel
        self.special_token_col = special_token_col
        self.sampling_strategy = sampling_strategy
        
    def __len__(self):
        return self.len
         
    def __getitem__(self, item):
        # Get gene tokens
        gene_tokens_cell = self._get_cell_tokens(item)
        gene_tokens_neighborhood = self._get_neighborhood_tokens(item)
        tokens = gene_tokens_cell + gene_tokens_neighborhood

        # Get number of non-zero gene tokens
        n_nonzero_cell_tokens = self.get_num_nonzero_cell_tokens(item)
        n_nonzero_neighborhood_tokens = self.get_num_nonzero_neighborhood_tokens(item)
        n_nonzero_tokens = n_nonzero_cell_tokens + n_nonzero_neighborhood_tokens

        # Retrieve segment labels
        seg_labels = torch.cat(
            (torch.zeros(self.n_special_tokens),
             torch.tensor(self.dataset[item]["seg_tokens"])))

        # Add special tokens
        if 'cls' in special_tokens:
            tokens = self.dataset[item]["cls_tokens"] + tokens
        if 'assay' in special_tokens:
            tokens = self.dataset[item]["assay_token"] + tokens
        if 'species' in special_tokens:
            tokens = self.dataset[item]["species_token"] + tokens
        if 'tissue' in special_tokens:
            tokens = self.dataset[item]["tissue_token"] + tokens
        if 'gene_panel' in special_tokens:
            tokens = self.dataset[item]["gene_panel_token"] + tokens
        if 'batch' in special_tokens:
            tokens = self.dataset[item]["batch_token"] + tokens

        return (torch.tensor(tokens), seg_labels, n_nonzero_cell_tokens,
                n_nonzero_neighborhood_tokens, n_nonzero_tokens)
        
    def _get_seg_gene_tokens(self, 
                             item: int,
                             segment_idx: int,
                             segment_seq_len: int,
                             ) -> List:
        """
        Get gene tokens for a given segment based on sampling strategy.

        Parameters
        -----------
        item:
            Index of the cell in the huggingface dataset.
        segment_idx:
            Index of the segment for which tokens are retrieved.
        segment_seq_len:
            Desired length of the segment token sequence.

        Returns
        --------
        seg_gene_tokens:
            List of tokens for a given segment.
        """
        # Only keep gene tokens in specified segment
        seg_gene_tokens = self.dataset[item]["gene_tokens"][
            self.dataset[item]["seg_tokens"] == segment_idx]

        # Validate that segment sequence length is specified correctly
        if 'rep' in self.sampling_strategy:
            pass
        else:
            if segment_seq_len > len(seg_gene_tokens):
                raise ValueError(
                    'Sequence length for a given segment cannot be larger than'
                    'segment size when not sampling with replacement.')

        # If no sampling strategy is specified, use all tokens up to specified
        # length
        if self.sampling_strategy is None:
            seg_gene_tokens = gene_tokens_cell[:segment_seq_len]
        # Otherwise, sample a subset of tokens based on the sampling strategy
        else:
            n_nonzero_tokens = sum(1 for token in seg_gene_tokens if token != 0)

            seg_gene_tokens = self.create_sampled_token_sequence(
                tokens=seg_gene_tokens,
                n_nonzero_tokens=n_nonzero_tokens,
                size=segment_seq_len,
                self.sampling_strategy)
                
        return seg_gene_tokens
            
    def create_sampled_token_sequence(self,
                                      tokens: List,
                                      n_nonzero_tokens: int,
                                      size: int,
                                      sampling_strategy: Literal[
                                        'norm_count_rank_sampling',
                                        'norm_count_rank_sampling_rep',
                                        'rand_sampling',
                                        'rand_sampling_rep']
                                      ) -> List:
        """
        Sample a subset of tokens based on a sampling strategy.

        Parameters
        -----------
        tokens:
            List of tokens.
        n_nonzero_tokens:
            Number of nonzero tokens in `tokens`.
        size:
            Size of the sampled subset.
        sampling_strategy:
            Sampling strategy for the token selection.
            
        Returns
        --------
        sampled_tokens:
            List of sampled tokens.
        """
        if sampling_strategy == 'normalized_count_rank_sampling':
            # Calculate weights based on rank and number of nonzero tokens
            # Higher the rank, higher the weight
            # a = [4, 1, 3, 2, 5, 0, 0, 0]
            # n_nonzero_tokens = 5  
            # sum_rank = 5 * (5 + 1) / 2.0 = 15.0
            # weights = [(n_nonzero_tokens - i)/sum_rank for i in range(n_nonzero_tokens)] 
            # = [0.3333333333333333, 0.26666666666666666, 0.2, 0.13333333333333333, 0.06666666666666667]
            # np.sum(weights) = 1.0
            sum_rank = (n_nonzero_tokens * (n_nonzero_tokens + 1) / 2.0) + 1e-9
            weights = [(n_nonzero_tokens - i)/sum_rank for i in range(n_nonzero_tokens)]
            assert np.isclose(np.sum(weights), 1.0)
        elif sampling_strategy == 'random_sampling':
            np.random.seed(seed)

            weights = np.ones(n_nonzero_tokens) / n_nonzero_tokens
        else:
            raise ValueError(
                f"{sampling_strategy} is an invalid sampling strategy.")
            
        # Sample seq_cell_len or seq_neighborhood token indices based on
        # weights
        sampled_indices = np.random.choice(
            np.arange(n_nonzero_tokens),
            size=min(size, n_nonzero_tokens),
            p=weights,
            replace=False)
            
        # Sort sampled indices to preserve rank order
        sampled_indices = np.sort(sampled_indices)
        sampled_tokens = [tokens[i] for i in sampled_indices]

        if size > n_nonzero_tokens:
            sampled_tokens.extend([0] * (size - len(sampled_tokens)))
        
        return sampled_tokens


def make_cell_neighborhood_dataset(
    batch_size: int,
    data: datasets.Dataset,
    vocab_size: int,
    collator=None,
    pin_mem: bool=True,
    num_workers: int=8,
    world_size: int=1,
    rank: int=0,
    drop_last: bool=True,
    seq_len_cell: int=0,
    seq_len_neighborhood: int=0,
    has_cls: bool=True,
    has_gene_panel: bool=True,
    special_token_col: str='batch_id',
    distributed: bool=True,
    sampling_strategy: Optional[
        Literal['normalized_count_rank_sampling',
                'random_sampling']]=None,
    ) -> Tuple[CellNeighborhoodDataset,
               torch.utils.data.DataLoader,
               Optional[torch.utils.data.distributed.DistributedSampler]]:
    """
    Convert huggingface Dataset into a torch CellNeighborhoodDataset object and
    create corresponding data loader.

    Parameters
    -----------
    batch_size:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    data:
        Huggingface dataset with cell and neighborhood tokens and cell-level
        labels.
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
    has_cls:
        If 'True', a <cls> token is included for each cell at position 0.
    has_gene_panel:
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
    dataset = CellNeighborhoodDataset(data,
                                      vocab_size,
                                      seq_len_cell=seq_len_cell,
                                      seq_len_neighborhood=seq_len_neighborhood,
                                      has_cls=has_cls,
                                      has_gene_panel=has_gene_panel,
                                      special_token_col=special_token_col,
                                      sampling_strategy=sampling_strategy)
    
    if distributed:
        dist_sampler = CustomDistributedLengthGroupedSampler(
            dataset,
            batch_size,
            hugging_face_dataset=data,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            num_replicas=world_size,
            rank=rank,
            seed=_GLOBAL_SEED)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  collate_fn=collator,
                                                  sampler=dist_sampler,
                                                  batch_size=batch_size,
                                                  drop_last=drop_last,
                                                  pin_memory=pin_mem,
                                                  num_workers=num_workers,
                                                  persistent_workers=False)
        logger.info('Data loader created.')

        return dataset, data_loader, dist_sampler
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  collate_fn=collator,
                                                  batch_size=batch_size,
                                                  drop_last=drop_last,
                                                  pin_memory=pin_mem,
                                                  num_workers=num_workers,
                                                  persistent_workers=False)
        logger.info('Data loader created.')
        
        return dataset, data_loader
