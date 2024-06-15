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
        subset_file=None
         ):
       
      dataset = CellNeighborhoodDataset(data,
                                        vocab_size,
                                        mask_index,
                                        seq_len=seq_len)
      
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

class CellNeighborhoodDataset(Dataset):
    def __init__(self,
                 data,
                 vocab_size,
                 mask_index,
                 seq_len):
        """
        CellNeighborhoodDataset.
        """
        self.dataset = data
        self.len = len(self.dataset)
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.seq_len = seq_len

    def __len__(self):
        
        return self.len
    def __getitem__(self, item):
        return torch.tensor(self.dataset[item]["input_ids"][0:self.seq_len]), torch.cat((torch.ones(self.seq_len // 2), torch.ones(self.seq_len - self.seq_len // 2) * 2)).int(), self.dataset[item]['niche_types']
    # TODO: change to entire sequence and modify CLS token


