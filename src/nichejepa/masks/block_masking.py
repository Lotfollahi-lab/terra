from typing import List, Literal, Optional, Tuple

from logging import getLogger
from multiprocessing import Value

import numpy as np
import torch


logger = getLogger()
_GLOBAL_SEED = 0


class BlockMaskCollator:
    """
    BlockMaskCollator class for sampling target and context masks from cell
    and neighborhood blocks.
    
    Parameters
    ----------
    n_targets: int
        Number of target masks to sample for each input sequence.
    n_contexts: int
        Number of context masks to sample for each input sequence.
    target_mask_size: int
        The size (in number of tokens) of each target mask.
    context_mask_size: int
        The size (in number of tokens) of each context mask.
    seq_len_cell: int
        The length of the token sequence representing the cell block.
    seq_len_neighborhood: int
        The length of the token sequence representing the neighborhood block.
    per_block_mask_ratio: float
        The ratio of elements to be masked in each block.
    separate_cls: bool
        This will determine whether we add the CLS of  cell only to cell blocks and
        the CLS of  neighborhood only to the neighborhood or not.
    """
    def __init__(self,
                 n_targets: int=2,
                 n_contexts: int=1,
                 target_mask_size: int=2,
                 context_mask_size: int=10,
                 seq_len_cell: int=0,
                 seq_len_neighborhood: int=0,
                 n_special_tokens: int=0,
                 per_block_mask_ratio: float=0.3,
                 separate_cls: bool=True):
        self.n_targets = n_targets
        self.n_contexts = n_contexts
        self.target_mask_size = target_mask_size
        self.context_mask_size = context_mask_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len_gene_tokens = self.seq_len_cell + self.seq_len_neighborhood
        self.n_special_tokens = n_special_tokens
        self.per_block_mask_ratio = per_block_mask_ratio
        self.separate_cls = separate_cls
        # Determine the valid start position for the mask based on number of
        # special tokens
        self.valid_min_start = self.n_special_tokens

    def block_masking(self,
                      sequence: torch.Tensor,
                      mask_ratio: float,
                      ) -> List[torch.Tensor]:
        """
        Perform block masking on the sequence based on the number of targets
        and per-block mask ratio.

        Parameters
        ----------
        sequence: Tensor, shape (n_samples,)
            The input sequence that needs to be masked.
        mask_ratio:
            Ratio of elements to be masked in each block.

        Returns
        ----------
        block_masks: List[Tensor]
            A list of masked indices for each block.
        context_mask: List
            List of binary mask indicating context tokens (1s where context is, 0s where masked).
        keep_tokens_target: int
            Minimum number of tokens kept across all target masks.
        """
        non_zero_indices = torch.nonzero(
            sequence[self.valid_min_start:]).squeeze()
        total_non_zero = len(non_zero_indices)
    
        # Initialize a list to store masked indices for each block
        block_masks = []

        # Initialize context mask
        context_mask = torch.zeros(len(sequence), dtype=torch.int32)
        
        # Keep track of the minimum number of target tokens across blocks
        keep_tokens_target = float('inf')

        # Compute block length based on the number of targets
        block_length = max(1, total_non_zero // self.n_targets)  # Avoid dividing by zero
        num_blocks = self.n_targets

        for i in range(num_blocks):
            # Determine the range of indices for the current block
            start_idx = i * block_length + self.valid_min_start
            end_idx = min(start_idx + block_length, total_non_zero)
        
            # Extract the non-zero indices for the current block
            block_non_zero_indices = non_zero_indices[start_idx:end_idx]
            context_mask[block_non_zero_indices] = 1  # Mark as context initially
            
            block_size = len(block_non_zero_indices)
            num_to_mask = int(np.ceil(block_size * mask_ratio))  # Determine number of elements to mask

            if num_to_mask > 0:
                # Randomly choose indices to mask within the block
                # DON'T USE torch.rand as it could produce repeated indices
                mask_indices = torch.randperm(block_size)[:num_to_mask]
                masked_indices = block_non_zero_indices[mask_indices].tolist()  # Convert to list.
                context_mask[masked_indices] = 0  # Set masked indices to 0 in the context mask.
                max_index, min_index = max(masked_indices), min(masked_indices) # Find the maximum and minimum index positions in the mask.  
                if min_index > self.seq_len_cell and self.separate_cls: # If the min_index is greater than self.seq_len_cell and self.separate_cls is True.
                   masked_indices = list(range(1,self.n_special_tokens)) + masked_indices # Include special tokens, excluding cls_cell.
                elif self.seq_len_cell > max_index and self.separate_cls: # If the max_index is smaller than self.seq_len_cell and self.separate_cls is True.
                    masked_indices = [0] + list(range(2,self.n_special_tokens)) + masked_indices # include special tokens excluding cls_neighborhood.
                else: # This means the block is in both neighborhood  and cell or and self.separate_cls is False.
                    masked_indices = list(range(self.n_special_tokens)) + masked_indices # include special tokens including both cls_neighborhood and cls_cell.
                keep_tokens_target = min(keep_tokens_target, len(masked_indices))  # Update minimum tokens target
                block_masks.append(torch.tensor(masked_indices))  # Append the masked indices io the list
            else:
                block_masks.append(torch.tensor([]))  # If no elements to mask, append an empty list
        # DON'T USE torch.rand as it could produce repeated indices
        # We randomly permut data so if we trim last item with keep_tokens_context
        # We avoid always discarding the last items of a sequence, as this may be problematic.
        context_mask = torch.nonzero(context_mask).squeeze()
        context_mask = context_mask[torch.randperm(len(context_mask))]
        
        # Add special tokens to context
        context_mask = torch.cat((torch.arange(self.n_special_tokens), context_mask))
        return block_masks, [context_mask], keep_tokens_target

    def _sample_gene_mask(self, sequence):
        """
        Sample context or target gene masks, considering both cell and neighborhood blocks.

        Parameters
        ----------
        sequence: Tensor
            A sequence of tokens as input

        Returns
        ----------
        target_masks: List[List[int]]
            A list of target masks per block.
        context_mask: Tensor
            Binary tensor indicating the context mask.
        keep_tokens_target: int
            The minimum number of tokens kept across target masks.
        """
        # Apply block masking on the full sequence
        target_masks, context_mask, keep_tokens_target = self.block_masking(
            sequence, self.per_block_mask_ratio)

        return target_masks, context_mask, keep_tokens_target

    def __call__(self,
                 batch: Tuple[torch.Tensor, torch.Tensor, str]
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create context and target masks when collating cell neighborhoods into a batch.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, str]
            The input sequence tokens, block labels, and cell-level labels for all observations in the batch.

        Returns
        ----------
        collated_batch: torch.Tensor
            The input gene tokens, block labels, and cell-level labels collated by batch.
        collated_masks_context: torch.Tensor
            Sampled context masks collated by batch.
        collated_masks_target: torch.Tensor
            Sampled target masks collated by batch.
        """
        B = len(batch)

        # Collate the batch using default PyTorch collate function
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_target, collated_masks_context, collated_masks_attention = [], [], []

        # Variables to track the minimum length of masks across the batch
        keep_tokens_target = self.seq_len_gene_tokens
        keep_tokens_context = self.seq_len_gene_tokens

        for i in range(B):
            # Initialize lists to store target and context masks for each observation
            masks_target, masks_context = [], []
            
            # Sample target and context masks for the current observation
            masks_target, masks_context, keep_tokens_target_current_batch = self._sample_gene_mask(batch[i][0])
            keep_tokens_target = min(keep_tokens_target, keep_tokens_target_current_batch)
            keep_tokens_context = min(keep_tokens_context, len(masks_context[0]))

            # Append the masks for the current observation to the collated lists
            collated_masks_target.append(masks_target)
            collated_masks_context.append(masks_context)
            collated_masks_attention.append((batch[i][0] != 0).int())

        # Trim masks to the minimum size across the batch and collate them
        collated_masks_target = [[cm[:keep_tokens_target] for cm in cm_list] for cm_list in collated_masks_target]
        collated_masks_target = torch.utils.data.default_collate(collated_masks_target)
        # Trim masks to the minimum size across the batch and collate them
        collated_masks_context = [[cm[:keep_tokens_context] for cm in cm_list] for cm_list in collated_masks_context]
        # Step 2: Use default_collate to create a batch
        collated_masks_context = torch.utils.data.default_collate(collated_masks_context)

        collated_masks_attention = torch.utils.data.default_collate(collated_masks_attention).unsqueeze(1).unsqueeze(1)
        
        return collated_batch, collated_masks_context, collated_masks_target, collated_masks_attention
