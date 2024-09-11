import os
import subprocess
import time
from logging import getLogger
from typing import Optional, Tuple, List

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.distributed import CustomDistributedLengthGroupedSampler

_GLOBAL_SEED = 0
logger = getLogger()


class CellNeighborhoodDataset(Dataset):
    def __init__(self,
                 data: datasets.arrow_dataset.Dataset,
                 vocab_size: int,
                 seq_len_cell: int=0,
                 seq_len_neighborhood: int=0,
                 has_cls: bool=True,
                 sampling_strategy: Optional[str]=None,
                 sampling_seed: Optional[int]=42
                 ):
        """
        Torch CellNeighborhoodDataset class.

        Parameters
        -----------
        data:
            Huggingface dataset with cell and neighborhood tokens and cell-level
            labels.
        vocab_size:
            Size of the vocabulary.
        seq_len_cell:
            Sequence length of the cell tokens.
        seq_len_neighborhood:
            Sequence length of the neighborhood tokens.
        has_cls:
            If 'True', a <cls> token is included for each cell at position 0.
        sampling_strategy:
            Sampling strategy for the dataset.
        sampling_seed:
            Seed for the sampling strategy.
        """
        self.dataset = data
        self.len = len(self.dataset)
        self.vocab_size = vocab_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len = seq_len_cell + seq_len_neighborhood
        self.has_cls = has_cls
        self.sampling_strategy = sampling_strategy
        self.sampling_seed = sampling_seed
        
    def __len__(self):
        return self.len
         
    def __getitem__(self, item):
        # Case 1: both cell and neighborhood tokens are included
        if self.seq_len_cell > 0 and self.seq_len_neighborhood > 0:
            # Get gene tokens for cell and neighborhood
            gene_tokens_cell = self._get_cell_tokens(item)
            gene_tokens_neighborhood = self._get_neighborhood_tokens(item)

            # Set tokens as the concatenation of cell and neighborhood tokens
            tokens = gene_tokens_cell + gene_tokens_neighborhood

            # Set the total number of nonzero tokens as the sum of nonzero cell and neighborhood tokens
            n_nonzero_cell_tokens = self.get_num_nonzero_cell_tokens(item)
            n_nonzero_neighborhood_tokens = self.get_num_nonzero_neighborhood_tokens(item)
            n_nonzero_tokens = n_nonzero_cell_tokens + n_nonzero_neighborhood_tokens            
            
            # Set niche and cell types
            niche_types = self.dataset[item]['niche_types']
            cell_types = self.dataset[item]['cell_types']
            
            if self.has_cls:
                # If a <cls> token is used, prepend it to the tokens and
                # consider it for segment labels
                tokens = [self.vocab_size] + tokens
                
                # Set total number of nonzero tokens to include <cls> token
                n_nonzero_tokens += 1
                
                # Create segment labels: 1 for cell tokens and <cls> token and 2
                # for neighborhood tokens (maybe this needs to be changed)
                labels = torch.cat(
                    (torch.ones(self.seq_len_cell + 1),
                     torch.ones(self.seq_len_neighborhood) * 2)).int()
            else:
                # Create segment labels: 1 for cell tokens, 2 for neighborhood tokens
                labels = torch.cat(
                    (torch.ones(self.seq_len_cell),
                    torch.ones(self.seq_len_neighborhood) * 2)).int()
                
            return torch.tensor(tokens), labels, niche_types, cell_types, n_nonzero_cell_tokens, n_nonzero_neighborhood_tokens, n_nonzero_tokens
        
        # Case 2: only cell tokens are included
        elif self.seq_len_cell > 0:
            # Get gene tokens for cell
            gene_tokens_cell = self._get_cell_tokens(item)
            
            # Set cell tokens as tokens
            tokens = gene_tokens_cell
            
            # Set the total number of nonzero tokens as the number of nonzero cell tokens
            n_nonzero_cell_tokens = self.get_num_nonzero_cell_tokens(item)
            n_nonzero_tokens = n_nonzero_cell_tokens
                
            # Set cell types
            cell_types = self.dataset[item]['cell_types']
            
            if self.has_cls:
                # If a <cls> token is used, prepend it to the tokens and
                # consider it for segment labels  
                tokens = [self.vocab_size] + tokens

                # Set total number of nonzero tokens to include <cls> token
                n_nonzero_tokens += 1
                
                # Create segment labels: 1 for cell tokens and <cls> token
                labels = torch.ones(self.seq_len_cell + 1).int()
            else:
                # Create segment labels: 1 for cell tokens
                labels = torch.ones(self.seq_len_cell).int()
                
            return torch.tensor(tokens), labels, cell_types, n_nonzero_cell_tokens, n_nonzero_tokens
        
        # Case 3: only neighborhood tokens are included
        elif self.seq_len_neighborhood > 0:
            # Get gene tokens for neighborhood
            gene_tokens_neighborhood = self._get_neighborhood_tokens(item)
            
            # Set neighborhood tokens as tokens
            tokens = gene_tokens_neighborhood
            
            # Set the total number of nonzero tokens as the number of nonzero neighborhood tokens
            n_nonzero_neighborhood_tokens = self.get_num_nonzero_neighborhood_tokens(item)
            n_nonzero_tokens = n_nonzero_neighborhood_tokens

            # Set niche types
            niche_types = self.dataset[item]['niche_types']

            if self.has_cls:
                # If a <cls> token is used, prepend it to the tokens and
                # consider it for segment labels  
                tokens = [self.vocab_size] + tokens

                # Set total number of nonzero tokens to include <cls> token
                n_nonzero_tokens += 1
                
                # Create segment labels: 2 for neighborhood tokens and <cls>
                # token (maybe this needs to be changed)
                labels = (torch.ones(self.seq_len_neighborhood + 1) * 2).int()
            else:
                # Create segment labels: 2 for neighborhood tokens
                labels = (torch.ones(self.seq_len_neighborhood) * 2).int()

            return torch.tensor(tokens), labels, niche_types, n_nonzero_neighborhood_tokens, n_nonzero_tokens
        
        # Case 4: neither cell nor neighborhood tokens are included, which is an
        # invalid state
        else:
            raise ValueError("Neither cell nor neighborhood tokens included.")      
        
    def _get_cell_tokens(self, 
                         item: int
                         ) -> Tuple[List, int]:
        """
        Get cell tokens and number of nonzero cell tokens for a given cell.

        Parameters
        -----------
        item:
            Index of the cell in the dataset.

        Returns
        --------
        gene_tokens_cell:
            List of cell tokens.
        """
        # If cell sequence length is greater than the number of cell tokens in
        # the HF dataset, use all tokens
        if self.seq_len_cell >= len(self.dataset[item]["gene_tokens_cell"]):
            gene_tokens_cell = self.dataset[item]["gene_tokens_cell"]
            
        # Otherwise, use a subset of tokens
        else:
            # If sampling strategy is not specified, use all tokens up to
            # specified cell sequence lengths
            if self.sampling_strategy is None:
                gene_tokens_cell = self.dataset[item][
                    "gene_tokens_cell"][:self.seq_len_cell]
                
            # Otherwise, sample a subset of tokens based on the sampling strategy
            else:
                gene_tokens_cell = self.create_sampled_token_sequence(
                    self.dataset[item]["gene_tokens_cell"],
                    self.dataset[item]["n_nonzero_cell_tokens"],
                    self.seq_len_cell,
                    self.sampling_strategy,
                    self.sampling_seed)
                
        return gene_tokens_cell

    def _get_neighborhood_tokens(self,
                                item: int
                                ) -> Tuple[List, int]:
        """ 
        Get neighborhood tokens and number of nonzero neighborhood tokens for a
        given cell.
        
        Parameters
        -----------
        item:
            Index of the cell in the dataset.
        
        Returns
        --------
        gene_tokens_neighborhood:
            List of neighborhood tokens.
        """
        # If neighborhood sequence length is greater than the number of
        neighborhood tokens in the HF dataset, use all tokens
        if self.seq_len_neighborhood >= len(self.dataset[item]["gene_tokens_neighborhood"]):
            gene_tokens_neighborhood = self.dataset[item]["gene_tokens_neighborhood"]
            
        # Otherwise, use a subset of tokens
        else:
            # If sampling strategy is not specified, use all tokens up to
            # specified neighborhood sequence lengths
            if self.sampling_strategy is None:
                gene_tokens_neighborhood = self.dataset[item]["gene_tokens_neighborhood"][:self.seq_len_neighborhood]
            # Otherwise, sample a subset of tokens based on the sampling strategy
            else:
                gene_tokens_neighborhood = self.create_sampled_token_sequence(
                    self.dataset[item]["gene_tokens_neighborhood"],
                    self.dataset[item]["n_nonzero_neighborhood_tokens"],
                    self.seq_len_neighborhood,
                    self.sampling_strategy,
                    self.sampling_seed)
        
        return gene_tokens_neighborhood

    def get_num_nonzero_cell_tokens(self,
                                    item: int
                                    ) -> int:
        """
        Get the number of nonzero cell tokens for a given cell.
        
        Parameters
        -----------
        item:
            Index of the cell in the dataset.
        
        Returns
        --------
        n_nonzero_cell_tokens:
            Number of nonzero cell tokens.
        """
        # Set the number of nonzero cell tokens
        # n_nonzero_cell_tokens -> self.dataset[item]["n_nonzero_cell_tokens"] if self.seq_len_cell >= len(self.dataset[item]["gene_tokens_cell"])
        # n_nonzero_cell_tokens -> self.dataset[item]["n_nonzero_cell_tokens"] if self.seq_len_cell >= self.dataset[item]["n_nonzero_cell_tokens"]
        # n_nonzero_cell_tokens -> self.seq_len_cell if self.seq_len_cell < self.dataset[item]["n_nonzero_cell_tokens"]
        n_nonzero_cell_tokens = min(
            self.dataset[item]["n_nonzero_cell_tokens"],
            self.seq_len_cell)
        return n_nonzero_cell_tokens

    def get_num_nonzero_neighborhood_tokens(self,
                                            item: int
                                            ) -> int:
        """
        Get the number of nonzero neighborhood tokens for a given cell.
        
        Parameters
        -----------
        item:
            Index of the cell in the dataset.
            
        Returns
        --------
        n_nonzero_neighborhood_tokens:
            Number of nonzero neighborhood tokens.
        """
        # Set the number of nonzero neighborhood tokens
        # n_nonzero_neighborhood_tokens -> self.dataset[item]["n_nonzero_neighborhood_tokens"] if self.seq_len_neighborhood >= len(self.dataset[item]["gene_tokens_neighborhood"])
        # n_nonzero_neighborhood_tokens -> self.dataset[item]["n_nonzero_neighborhood_tokens"] if self.seq_len_neighborhood >= self.dataset[item]["n_nonzero_neighborhood_tokens"]
        # n_nonzero_neighborhood_tokens -> self.seq_len_neighborhood if self.seq_len_neighborhood < self.dataset[item]["n_nonzero_neighborhood_tokens"]
        n_nonzero_neighborhood_tokens = min(
            self.dataset[item]["n_nonzero_neighborhood_tokens"],
            self.seq_len_neighborhood)
        return n_nonzero_neighborhood_tokens
            
    def create_sampled_token_sequence(
        self,
        tokens: List,
        n_nonzero_tokens: int,
        size: int,
        sampling_strategy: str="normalized_count_rank_sampling",
        seed: int=42
        ) -> List:
        """
        Sample a subset of tokens based on the sampling strategy and seed.

        Parameters
        -----------
        tokens:
            List of tokens.
        n_nonzero_tokens:
            Number of nonzero tokens in `tokens`.
        size:
            Size of the sampled subset.
        sampling_strategy:
            Sampling strategy for the token list.
        seed:
            Seed for the sampling strategy.
            
        Returns
        --------
        sampled_tokens:
            List of sampled tokens.
        """
        assert size < n_nonzero_tokens
        
        if sampling_strategy == "normalized_count_rank_sampling":
            # Set seed for sampling
            np.random.seed(seed)
            
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
            
            # Sample seq_cell_len or seq_neighborhood token indices based on
            # weights
            sampled_indices = np.random.choice(
                np.arange(n_nonzero_tokens),
                size=size,
                p=weights,
                replace=False)
            
            # Sort sampled indices to preserve rank order
            sampled_indices = np.sort(sampled_indices)
            sampled_tokens = [tokens[i] for i in sampled_indices]
            return sampled_tokens
        else:
            raise ValueError(
                f"{sampling_strategy} is an invalid sub-sampling strategy.")

def make_cell_neighborhood_dataset(
    batch_size: int,
    data: datasets.arrow_dataset.Dataset,
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
    distributed: bool=True
    ) -> Tuple[CellNeighborhoodDataset,
               torch.utils.data.DataLoader,
               Optional[torch.utils.data.distributed.DistributedSampler]]:
    """
    Convert Huggingface dataset into a torch CellNeighborhoodDataset object and
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
                                      has_cls=has_cls)
    
    if distributed:
        dist_sampler = CustomDistributedLengthGroupedSampler(
            dataset,
            batch_size,
            hugging_face_dataset=data,
            num_replicas=world_size,
            incl_cell_seq=incl_cell_seq,
            incl_neighborhood_seq=incl_neighborhood_seq,
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
