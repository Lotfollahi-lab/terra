import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
from torch.utils.data import Dataset

_GLOBAL_SEED = 0
logger = getLogger()

def make_cell_neighborhood_dataset(
        batch_size,
        data,
        vocab_size,
        seq_len,
        mask_index=1,
        collator=None,
        pin_mem=True,
        num_workers=8,
        world_size=1,
        rank=0,
        root_path=None,
        gene_folder=None,
        training=True,
        copy_data=False,
        drop_last=True,
        subset_file=None,
        just_cell = True,
        just_neighborhood = False,
        seq_len_cell=0,
        seq_len_neighborhood=0,
        has_cls = True,
        distributed= True):

      dataset = CellNeighborhoodDataset(data,
                                        vocab_size,
                                        mask_index,
                                        seq_len=seq_len,
                                        just_cell=just_cell,
                                        just_neighborhood=just_neighborhood,
                                        seq_len_cell = seq_len_cell,
                                        seq_len_neighborhood = seq_len_neighborhood,
                                        has_cls = has_cls)

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

          logger.info('Gene unsupervised data loader created')

          return dataset, data_loader, dist_sampler
      else:
          data_loader = torch.utils.data.DataLoader(dataset,
                                                collate_fn=collator,
                                                batch_size=batch_size,
                                                pin_memory=pin_mem,
                                                num_workers=num_workers,
                                                persistent_workers=False)
          return dataset, data_loader
class CellNeighborhoodDataset(Dataset):
    def __init__(self,
                 data,
                 vocab_size,
                 mask_index,
                 seq_len,
                 just_cell=True,
                 just_neighborhood=False,
                 seq_len_cell=0,
                 seq_len_neighborhood=0,
                 has_cls = True):
        """
        CellNeighborhoodDataset.
        """
        self.dataset = data
        self.len = len(self.dataset)
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.seq_len = seq_len
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.just_cell = just_cell
        self.just_neighborhood = just_neighborhood
        self.has_cls = has_cls
    def __len__(self):

        return self.len
    def __getitem__(self, item):
# Extract gene tokens for both the cell and neighborhood, limiting the sequence length
gene_tokens_cell = self.dataset[item]["gene_tokens_cell"][:self.seq_len_cell]
gene_tokens_neighborhood = self.dataset[item]["gene_tokens_neighborhood"][:self.seq_len_neighborhood]

# Initialize empty lists to store the tokens and labels
tokens, labels = [], []

# Case 1: Both cell and neighborhood data are included
if self.just_cell and self.just_neighborhood:
    # Combine gene tokens from cell and neighborhood
    tokens = gene_tokens_cell + gene_tokens_neighborhood
    # Retrieve the niche and cell types for the item
    niche_types = self.dataset[item]['niche_types']
    cell_types = self.dataset[item]['cell_types']
    
    # If a CLS token is used, prepend it to the tokens and adjust labels
    if self.has_cls:
        tokens = [self.vocab_size] + tokens
        # Create labels: 1 for cell tokens and 2 for neighborhood tokens here we also assign cls token label one
        # Maybe we need to think about this part
        labels = torch.cat((torch.ones(self.seq_len_cell + 1), torch.ones(self.seq_len_neighborhood) * 2)).int()
    else:
        # Create labels without CLS token: 1 for cell tokens, 2 for neighborhood tokens
        labels = torch.cat((torch.ones(self.seq_len_cell), torch.ones(self.seq_len_neighborhood) * 2)).int()
    
    # Return tokens, labels, and the retrieved types
    return torch.tensor(tokens), labels, niche_types, cell_types

# Case 2: Only cell data is included
elif self.just_cell:
    # Use only the cell gene tokens
    tokens = gene_tokens_cell
    # Retrieve the cell types for the item
    cell_types = self.dataset[item]['cell_types']
    
    # If a CLS token is used, prepend it to the tokens and adjust labels
    if self.has_cls:
        tokens = [self.vocab_size] + tokens
        # Create labels: all ones for cell tokens (including CLS if present)
        labels = torch.ones(self.seq_len_cell + 1).int()
    else:
        # Create labels: all ones for cell tokens
        labels = torch.ones(self.seq_len_cell).int()
    
    # Return tokens, labels, and the retrieved cell types
    return torch.tensor(tokens), labels, cell_types

# Case 3: Only neighborhood data is included
elif self.just_neighborhood:
    # Use only the neighborhood gene tokens
    tokens = gene_tokens_neighborhood
    # Retrieve the niche types for the item
    niche_types = self.dataset[item]['niche_types']
    
    # If a CLS token is used, prepend it to the tokens and adjust labels
    if self.has_cls:
        tokens = [self.vocab_size] + tokens
        # Create labels: all twos for neighborhood tokens (including CLS if present)
        labels = (torch.ones(self.seq_len_neighborhood + 1) * 2).int()
    else:
        # Create labels: all twos for neighborhood tokens
        labels = (torch.ones(self.seq_len_neighborhood) * 2).int()
    
    # Return tokens, labels, and the retrieved niche types
    return torch.tensor(tokens), labels, niche_types

# Case 4: Neither cell nor neighborhood data is included, which is an invalid state
else:
    # Raise an error if neither just_cell nor just_neighborhood is set
    raise ValueError("Invalid state: neither just_cell nor just_neighborhood is set.")

