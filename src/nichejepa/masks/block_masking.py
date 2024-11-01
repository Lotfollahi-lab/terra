"""
Block masking.

Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture.
Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py
(05.06.2024).
"""


from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from ..masks.utils import configure_attention_masks


class BlockMaskCollator:
    """
    BlockMaskCollator class for sampling target and context block masks from
    cell and neighborhood segments.
    
    Parameters
    ----------
    n_targets:
        Number of target masks to sample for each token sequence.
    seq_len_cell:
        The length of the token sequence representing the cell segment.
    seq_len_neighborhood:
        The length of the token sequence representing the neighborhood segments.
    n_special_tokens:
        Number of special tokens in each token sequence, including <cls> tokens.
    n_cls_tokens:
        Number of <cls> tokens in each token sequence.
    per_block_mask_ratio:
        Ratio of elements to be masked in each block. A list with min and
        max ratio can be provided, in which case a value between the min and
        max will be sampled for each batch.
    controlled_attention_pattern:
        The pattern that the model uses to generate the attention matrix.
    """
    def __init__(self,
                 n_targets: int,
                 seq_len_cell: int,
                 seq_len_neighborhood: int,
                 n_special_tokens: int,
                 n_cls_tokens: int=2,
                 per_block_mask_ratio: float=0.5,
                 controlled_attention_pattern: Optional[torch.Tensor]=None):
        self.n_targets = n_targets
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len_genes = self.seq_len_cell + self.seq_len_neighborhood
        self.n_special_tokens = n_special_tokens
        self.n_cls_tokens = n_cls_tokens
        self.per_block_mask_ratio = per_block_mask_ratio
        self.controlled_attention_pattern = controlled_attention_pattern

    def _sample_gene_mask(self,
                          tokens: torch.Tensor,
                          segments: torch.Tensor,
                          ) -> Tuple[List[torch.Tensor],
                                     List[torch.Tensor],
                                     int]:
        """
        Perform block masking on the sequence based on the number of targets
        (number of blocks) and per block mask ratio. Tokens not sampled in the
        targets will be part of the context.

        Parameters
        ----------
        tokens:
            The token sequence that needs to be masked with dimension (B, N);
            B: batch size, N: number of tokens.
        segments:
            The sequence of segments to determine which <cls> tokens are
            included in the target masks.

        Returns
        ----------
        target_masks:
            List with multiple masks indicating target token indices for each
            block.
        context_masks:
            List with one mask indicating context token indices.
        keep_tokens_target:
            Minimum number of tokens kept across all target masks.
        """
        # Determine mask ratio; sample if list is provided
        if isinstance(self.per_block_mask_ratio, list):
            mask_ratios = np.arange(
                self.per_block_mask_ratio[0],
                self.per_block_mask_ratio[1] + 0.1, 0.1)
            mask_ratio = np.random.choice(mask_ratios)
        else:
            mask_ratio = self.per_block_mask_ratio

        # Get non-zero indices and segments excluding special tokens
        nz_ns_indices = torch.nonzero(tokens).squeeze()[self.n_special_tokens:]
        total_nz_ns = len(nz_ns_indices)
        nz_ns_segments = segments[segments != 0][self.n_special_tokens:]
    
        # Initialize masks
        target_masks = []
        context_mask = torch.zeros(len(tokens), dtype=torch.int32)
        
        # Keep track of the minimum number of target tokens across blocks
        keep_tokens_target = float('inf')

        # Compute block length based on number of blocks; avoid zero division
        block_length = max(1, total_nz_ns // self.n_targets)

        for i in range(self.n_targets):
            # Determine the range of indices for the current block
            start_idx = i * block_length
            end_idx = min(start_idx + block_length, total_nz_ns)
        
            # Extract the non-zero indices for the current block and mark as
            # context initially
            block_nz_ns_indices = nz_ns_indices[start_idx:end_idx]
            context_mask[block_nz_ns_indices] = 1

            # Extract segments for the current block to determine which <cls>
            # are to be included
            # 10 is index cell segment, corresponding to <cls> token at index 0
            block_segments = nz_ns_segments[start_idx: end_idx]
            block_unique_segments = torch.unique(block_segments)
            cls_tokens = [seg - 10 for seg in block_unique_segments.tolist()]
            
            # Determine number of elements to mask
            block_size = len(block_nz_ns_indices)
            num_to_mask = int(np.ceil(block_size * mask_ratio))

            if num_to_mask > 0:
                # Randomly choose indices to mask within the block
                # DON'T USE torch.rand as it could produce repeated indices
                mask_indices = torch.randperm(block_size)[:num_to_mask]
                masked_indices = block_nz_ns_indices[mask_indices].tolist()
                
                # Set masked indices to 0 in the context mask
                context_mask[masked_indices] = 0

                # Add <cls> and special tokens to mask indices
                masked_indices = cls_tokens + list(
                    range(self.n_cls_tokens, self.n_special_tokens)
                    ) + masked_indices

                # Update minimum tokens target and append masked indices
                keep_tokens_target = min(
                    keep_tokens_target, len(masked_indices))
                target_masks.append(torch.tensor(masked_indices))
            else:
                # No elements to mask
                target_masks.append(torch.tensor([]))

        # We randomly permute data so if we trim last item with
        # keep_tokens_context, we avoid always discarding the last items of a
        # sequence
        # DON'T USE torch.rand as it could produce repeated indice
        context_mask = torch.nonzero(context_mask).squeeze()
        context_mask = context_mask[torch.randperm(len(context_mask))]
        
        # Add special tokens to context
        context_mask = torch.cat(
            (torch.arange(self.n_special_tokens), context_mask))

        context_masks = [context_mask]

        return target_masks, context_masks, keep_tokens_target

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

        # Collate the batch
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_target = []
        collated_masks_context = []
        collated_masks_attention = []

        # Track the minimum length of masks across the batch
        keep_tokens_target = self.seq_len_genes
        keep_tokens_context = self.seq_len_genes

        # Store target and context masks for each observation
        for i in range(B):
            masks_target, masks_context = [], []
            
            # Sample target and context masks for the current observation
            masks_target, masks_context, keep_tokens_target_current_batch = self._sample_gene_mask(
                tokens=batch[i][0],
                segments=batch[i][1])
            keep_tokens_target = min(keep_tokens_target,
                                     keep_tokens_target_current_batch)
            keep_tokens_context = min(keep_tokens_context,
                                      len(masks_context[0]))

            # Append the masks for the current observation to the collated lists
            collated_masks_target.append(masks_target)
            collated_masks_context.append(masks_context)
            collated_masks_attention.append((batch[i][0] != 0).int())

        # Trim masks to the minimum size across the batch and collate them
        collated_masks_target = [
            [cm[:keep_tokens_target] for cm in cm_list]
            for cm_list in collated_masks_target]
        collated_masks_context = [
            [cm[:keep_tokens_context] for cm in cm_list]
            for cm_list in collated_masks_context]
        collated_masks_target = torch.utils.data.default_collate(
            collated_masks_target)
        collated_masks_context = torch.utils.data.default_collate(
            collated_masks_context)
        collated_masks_attention = torch.utils.data.default_collate(
            collated_masks_attention).unsqueeze(1).unsqueeze(1)

        # Apply controlled attention
        if self.controlled_attention_pattern is not None:
            collated_masks_controlled_attention = collated_masks_attention.expand(
                collated_masks_attention.shape[0],
                1,
                collated_masks_attention.shape[-1],
                collated_masks_attention.shape[-1]).clone()
            if torch.sum(self.controlled_attention_pattern) != 0:
                configure_attention_masks(
                    self.controlled_attention_pattern,
                    collated_masks_controlled_attention,
                    self.seq_len_cell,
                    self.n_special_tokens)
        else:
            collated_masks_controlled_attention = None
               
        return collated_batch, collated_masks_context, collated_masks_target, collated_masks_attention, collated_masks_controlled_attention
