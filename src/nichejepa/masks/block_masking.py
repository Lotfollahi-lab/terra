"""
Block masking.

Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture.
Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py
(05.06.2024).
"""


from typing import List, Literal, Optional, Tuple, Union

from logging import getLogger
from ..masks.utils import configure_attention_masks

import numpy as np
import torch


class BlockMaskCollator:
    """
    BlockMaskCollator class for sampling target and context block masks from
    cell and neighborhood segments.
    
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
    controlled_attention_pattern:
        The pattern that the model uses to generate the attention matrix.
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
                 separate_cls: bool=True,
                 controlled_attention_pattern: Optional[torch.Tensor]=None):
        self.n_targets = n_targets
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len_gene_tokens = self.seq_len_cell + self.seq_len_neighborhood
        self.n_special_tokens = n_special_tokens
        self.per_block_mask_ratio = per_block_mask_ratio
        self.separate_cls = separate_cls
        self.controlled_attention_pattern = controlled_attention_pattern

        # Determine the valid start position for the mask based on number of
        # special tokens
        self.valid_min_start = self.n_special_tokens

    def block_masking(self,
                      sequence: torch.Tensor,
                      mask_ratio: Union[float, List[float]],
                      ) -> Tuple[List[torch.Tensor], List, int]:
        """
        Perform block masking on the sequence based on the number of targets
        (number of blocks) and per block mask ratio. Tokens not sampled in the
        targets will be part of the context.

        Parameters
        ----------
        sequence:
            The input sequence that needs to be masked with dimension (B, N);
            B: batch size, N: number of tokens.
        mask_ratio:
            Ratio of elements to be masked in each block. A list with min and
            max ratio can be provided, in which case a value between the min and
            max will be sampled for each batch.

        Returns
        ----------
        block_masks:
            A list of masked indices for each block.
        context_mask:
            List of binary masks indicating context tokens (1s where context is,
            0s where masked).
        keep_tokens_target:
            Minimum number of tokens kept across all target masks.
        """
        # Sample mask ratio if list is provided
        if isinstance(mask_ratio, list):
            mask_ratios = np.arange(mask_ratio[0], mask_ratio[1] + 0.1, 0.1)
            mask_ratio = np.random.choice(mask_ratios)

        non_zero_indices = torch.nonzero(
            sequence).squeeze()
        non_zero_indices = non_zero_indices[self.valid_min_start:]
        total_non_zero = len(non_zero_indices)
    
        # Initialize a list to store masked indices for each block
        block_masks = []

        # Initialize context mask
        context_mask = torch.zeros(len(sequence), dtype=torch.int32)
        
        # Keep track of the minimum number of target tokens across blocks
        keep_tokens_target = float('inf')

        # Compute block length based on the number of targets; avoid division by
        # zero
        block_length = max(1, total_non_zero // self.n_targets)
        num_blocks = self.n_targets

        for i in range(num_blocks):
            # Determine the range of indices for the current block
            start_idx = i * block_length
            end_idx = min(start_idx + block_length, total_non_zero)
        
            # Extract the non-zero indices for the current block and mark as
            # context initially
            block_non_zero_indices = non_zero_indices[start_idx:end_idx]
            context_mask[block_non_zero_indices] = 1
            
            # Determine number of elements to mask
            block_size = len(block_non_zero_indices)
            num_to_mask = int(np.ceil(block_size * mask_ratio))

            if num_to_mask > 0:
                # Randomly choose indices to mask within the block
                # DON'T USE torch.rand as it could produce repeated indices
                mask_indices = torch.randperm(block_size)[:num_to_mask]
                masked_indices = block_non_zero_indices[mask_indices].tolist()
                
                # Set masked indices to 0 in the context mask
                context_mask[masked_indices] = 0

                # Include <cls> tokens based on flag and whether block is in
                # cell or neighborhood segments 
                max_index, min_index = max(masked_indices), min(masked_indices)
                if (min_index > self.seq_len_cell) and self.separate_cls:
                    # Include special tokens except <cls_cell>
                    masked_indices = list(
                        range(1, self.n_special_tokens)) + masked_indices
                elif (self.seq_len_cell > max_index) and self.separate_cls:
                    # Include special tokens except <cls_neighborhood>
                    masked_indices = [0] + list(
                        range(2, self.n_special_tokens)) + masked_indices
                else:
                    # This means the block is in both neighborhood and cell
                    # segments or separation flag is False; in those cases
                    # include all special tokens including <cls_cell> and
                    # <cls_neighborhood>
                    masked_indices = list(
                        range(self.n_special_tokens)) + masked_indices

                # Update minimum tokens target and append masked indices
                keep_tokens_target = min(
                    keep_tokens_target, len(masked_indices))
                block_masks.append(torch.tensor(masked_indices))
            else:
                # No elements to mask
                block_masks.append(torch.tensor([]))

        # We randomly permute data so if we trim last item with
        # keep_tokens_context, we avoid always discarding the last items of a
        # sequence
        # DON'T USE torch.rand as it could produce repeated indice
        context_mask = torch.nonzero(context_mask).squeeze()
        context_mask = context_mask[torch.randperm(len(context_mask))]
        
        # Add special tokens to context
        context_mask = torch.cat(
            (torch.arange(self.n_special_tokens), context_mask))

        return block_masks, [context_mask], keep_tokens_target

    def _sample_gene_mask(self,
                          sequence: torch.Tensor
                          ) -> Tuple[
                            List[torch.Tensor], List[torch.Tensor], int]:
        """
        Sample context or target gene masks, considering both cell and
        neighborhood segments.

        Parameters
        ----------
        sequence: Tensor
            The sequence of tokens.

        Returns
        ----------
        target_masks:
            A list of target masks per block.
        context_mask: Tensor
            Binary tensor indicating the context mask.
        keep_tokens_target:
            The minimum number of tokens kept across target masks.
        """
        # Apply block masking on the full sequence
        target_masks, context_mask, keep_tokens_target = self.block_masking(
            sequence, self.per_block_mask_ratio)

        return target_masks, context_mask, keep_tokens_target

    def __call__(self,
                 batch: Tuple[torch.Tensor,
                              torch.Tensor,
                              torch.Tensor,
                              torch.Tensor,
                              List[str]],
                 ) -> Tuple[torch.Tensor,
                            torch.Tensor,
                            torch.Tensor,
                            torch.Tensor]:
        """
        Create context and target masks when collating tokens into a batch.

        Parameters
        ----------
        batch:
            Tuple containing the input batch including gene tokens, segments,
            positions, counts and cell IDs.

        Returns
        ----------
        collated_batch:
            Input gene tokens, segments, positions, counts and cell IDs collated
            by batch.
        collated_masks_context:
            Sampled context masks collated by batch.
        collated_masks_target:
            Sampled target masks collated by batch.
        collated_masks_attention:
            Attention masks collated by batch.
        """
        B = len(batch)

        # Collate the batch using default PyTorch collate function
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_target = []
        collated_masks_context = []
        collated_masks_attention = []

        # Track the minimum length of masks across the batch
        keep_tokens_target = self.seq_len_gene_tokens
        keep_tokens_context = self.seq_len_gene_tokens

        for i in range(B):
            # Store target and context masks for each observation
            masks_target, masks_context = [], []
            
            # Sample target and context masks for the current observation
            masks_target, masks_context, keep_tokens_target_current_batch = self._sample_gene_mask(
                batch[i][0])
            keep_tokens_target = min(keep_tokens_target,
                                     keep_tokens_target_current_batch)
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
        collated_masks_attention = torch.utils.data.default_collate(
            collated_masks_attention).unsqueeze(1).unsqueeze(1)

        #"""
        collated_masks_attention = collated_masks_attention.expand(
            collated_masks_attention.shape[0], 1, collated_masks_attention.shape[-1],collated_masks_attention.shape[-1]).clone()
        # Batch token only attends to itself
        collated_masks_attention[
                :,
                :,
                2,
                0:2] = 0
        collated_masks_attention[
                :,
                :,
                2,
                3:] = 0
        #"""

        if self.controlled_attention_pattern is not None:
            #collated_masks_controlled_attention = collated_masks_attention.expand(
            #    collated_masks_attention.shape[0], 1, collated_masks_attention.shape[-1],collated_masks_attention.shape[-1]).clone()
            collated_masks_controlled_attention = collated_masks_attention.clone()
            if torch.sum(self.controlled_attention_pattern) != 0:
                configure_attention_masks(
                    self.controlled_attention_pattern,
                    collated_masks_controlled_attention,
                    self.seq_len_cell,
                    self.valid_min_start)
        else:
            collated_masks_controlled_attention = None
               
        return collated_batch, collated_masks_context, collated_masks_target, collated_masks_attention, collated_masks_controlled_attention
