import os
import subprocess
import time
from logging import getLogger
from typing import Optional, Tuple

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset


_GLOBAL_SEED = 0
logger = getLogger()


def make_cell_neighborhood_dataset(batch_size: int,
                                   data: datasets.arrow_dataset.Dataset,
                                   vocab_size: int,
                                   seq_len: int,
                                   collator=None,
                                   pin_mem: bool=True,
                                   num_workers: int=8,
                                   world_size: int=1,
                                   rank: int=0,
                                   drop_last: bool=True,
                                   incl_cell_seq: bool=True,
                                   incl_neighborhood_seq: bool=False,
                                   seq_len_cell: int=0,
                                   seq_len_neighborhood: int=0,
                                   has_cls: bool=True,
                                   distributed: bool=True) -> Tuple[CellNeighborhoodDataset,
                                                                    torch.utils.data.DataLoader,
                                                                    Optional[torch.utils.data.distributed.DistributedSampler]]:
    """
    Convert Huggingface dataset into a torch CellNeighborhoodDataset object and create corresponding data loader.

    Parameters
    -----------
    batch_size:
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    data:
        Huggingface dataset with cell and neighborhood tokens and cell-level labels.
    vocab_size:
        Size of the vocabulary.
    seq_len:
        Sequence length of all tokens.
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
    incl_cell_seq:
        If 'True', cell tokens are included for each cell.
    incl_neighborhood_seq:
        If 'True', neighborhood tokens are included for each cell.
    seq_len_cell:
        Sequence length of the cell tokens.
    seq_len_neighborhood:
        Sequence length of the neighborhood tokens.
    has_cls:
        If 'True', a <cls> token is included for each cell at position 0.
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
                                      mask_index,
                                      seq_len=seq_len,
                                      incl_cell_seq=incl_cell_seq,
                                      incl_neighborhood_seq=incl_neighborhood_seq,
                                      seq_len_cell=seq_len_cell,
                                      seq_len_neighborhood=seq_len_neighborhood,
                                      has_cls=has_cls)
    
    if distributed:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset,
                                                                       num_replicas=world_size,
                                                                       rank=rank)
        
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


class CellNeighborhoodDataset(Dataset):
    def __init__(self,
                 data: datasets.arrow_dataset.Dataset,
                 vocab_size: int,
                 seq_len: int,
                 incl_cell_seq: bool=True,
                 incl_neighborhood_seq: bool=False,
                 seq_len_cell: int=0,
                 seq_len_neighborhood: int=0,
                 has_cls: bool=True):
        """
        Torch CellNeighborhoodDataset class.

        Parameters
        -----------
        data:
            Huggingface dataset with cell and neighborhood tokens and cell-level labels.
        vocab_size:
            Size of the vocabulary.
        seq_len:
            Sequence length of all tokens.
        incl_cell_seq:
            If 'True', cell tokens are included for each cell.
        incl_neighborhood_seq:
            If 'True', neighborhood tokens are included for each cell.
        seq_len_cell:
            Sequence length of the cell tokens.
        seq_len_neighborhood:
            Sequence length of the neighborhood tokens.
        has_cls:
            If 'True', a <cls> token is included for each cell at position 0.
        """
        self.dataset = data
        self.len = len(self.dataset)
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.seq_len = seq_len
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.incl_cell_seq = incl_cell_seq
        self.incl_neighborhood_seq = incl_neighborhood_seq
        self.has_cls = has_cls
        
    def __len__(self):
        return self.len
         
    def __getitem__(self, item):
        # Extract specified sequence length of gene tokens for both the cell and neighborhood
        gene_tokens_cell = self.dataset[item]["gene_tokens_cell"][:self.seq_len_cell]
        gene_tokens_neighborhood = self.dataset[item]["gene_tokens_neighborhood"][:self.seq_len_neighborhood]

        # Collect tokens and labels
        # Case 1: both cell and neighborhood tokens are included
        if self.incl_cell_seq and self.incl_neighborhood_seq:
            tokens = gene_tokens_cell + gene_tokens_neighborhood
            niche_types = self.dataset[item]['niche_types']
            cell_types = self.dataset[item]['cell_types']
            if self.has_cls:
                # If a CLS token is used, prepend it to the tokens and consider for segment labels
                tokens = [self.vocab_size] + tokens
                # Create segment labels: 1 for cell tokens and <cls> token and 2 for neighborhood tokens
                # Maybe we need to think about this part
                labels = torch.cat((torch.ones(self.seq_len_cell + 1), torch.ones(self.seq_len_neighborhood) * 2)).int()
            else:
                # Create segment labels: 1 for cell tokens, 2 for neighborhood tokens
                labels = torch.cat((torch.ones(self.seq_len_cell), torch.ones(self.seq_len_neighborhood) * 2)).int()
            return torch.tensor(tokens), labels, niche_types, cell_types
        
        # Case 2: only cell tokens are included
        elif self.incl_cell_seq:
            tokens = gene_tokens_cell
            cell_types = self.dataset[item]['cell_types']
            if self.has_cls:
              # If a CLS token is used, prepend it to the tokens and consider for segment labels  
              tokens = [self.vocab_size] + tokens
              # Create segment labels: 1 for cell tokens and <cls> token
              labels = torch.ones(self.seq_len_cell + 1).int()
            else:
              # Create segment labels: 1 for cell tokens
              labels = torch.ones(self.seq_len_cell).int()
            return torch.tensor(tokens), labels, cell_types
        
        # Case 3: only neighborhood tokens are included
        elif self.incl_neighborhood_seq:
            tokens = gene_tokens_neighborhood
            niche_types = self.dataset[item]['niche_types']
            if self.has_cls:
                # If a CLS token is used, prepend it to the tokens and consider for segment labels  
                tokens = [self.vocab_size] + tokens
                # Create segment labels: 2 for neighborhood tokens and <cls> token
                labels = (torch.ones(self.seq_len_neighborhood + 1) * 2).int()
            else:
                # Create segment labels: 2 for neighborhood tokens
                labels = (torch.ones(self.seq_len_neighborhood) * 2).int()
            return torch.tensor(tokens), labels, niche_types
        
        # Case 4: neither cell nor neighborhood tokens are included, which is an invalid state
        else:
            raise ValueError("Invalid state: neither 'incl_cell_seq' nor 'incl_neighborhood_seq' is set.")
