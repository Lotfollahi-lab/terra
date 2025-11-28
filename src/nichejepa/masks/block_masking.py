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


class BlockMaskCollator:
    """
    BlockMaskCollator class for sampling target and context block masks from
    cell and neighborhood segments.
    
    Parameters
    ----------
    n_targets:
        Number of target masks to sample for each token sequence.
    n_contexts:
    seq_len_cell:
        The length of the token sequence representing the cell segment.
    seq_len_neighborhood:
        The length of the token sequence representing the neighborhood segments.
    max_special_tokens:
        Maximum number of special tokens to determine first segment
        corresponding to a cell.
    n_special_tokens:
        Number of special tokens in each token sequence, including <cls> tokens.
    max_cls_tokens:
        Number of <cls> tokens in each token sequence.
    per_block_mask_ratio:
        Ratio of elements to be masked in each block. A list with min and
        max ratio can be provided, in which case a value between the min and
        max will be sampled for each batch.
    sample_segments:
        If `True`, sample number of neighbors in each batch.
    """
    def __init__(self,
                 n_targets: int,
                 n_contexts: int,
                 seq_len_cell: int,
                 seq_len_neighborhood: int,
                 max_special_tokens: int,
                 n_special_tokens: int,
                 max_cls_tokens: int,
                 per_block_mask_ratio: float=0.5,
                 sample_segments: bool = False,
                 new_spc: bool=False,
                 restrict_special_attention: bool=False):
        self.n_targets = n_targets
        self.n_contexts = n_contexts
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len_genes = self.seq_len_cell + self.seq_len_neighborhood
        self.n_special_tokens = n_special_tokens
        self.max_special_tokens = max_special_tokens
        self.max_cls_tokens = max_cls_tokens
        self.per_block_mask_ratio = per_block_mask_ratio
        self.sample_segments = sample_segments
        self.new_spc = new_spc
        self.restrict_special_attention = restrict_special_attention

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
        ns_tokens = tokens[self.n_special_tokens:]
        nz_ns_indices = torch.nonzero(ns_tokens).add_(
            self.n_special_tokens).squeeze()
        total_nz_ns = len(nz_ns_indices)
    
        # Initialize masks
        target_masks = []
        context_mask = torch.zeros(len(tokens), dtype=torch.int32)

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
            #block_segments = nz_ns_segments[start_idx: end_idx]
            #block_unique_segments = torch.unique(block_segments)
            #cls_tokens = [
            #    seg - self.max_special_tokens for seg in
            #    block_unique_segments.tolist()]
            
            # Determine number of elements to mask
            block_size = len(block_nz_ns_indices)
            num_to_mask = int(np.ceil(block_size * mask_ratio))

            if num_to_mask > 0:
                # Randomly choose indices to mask within the block
                # DON'T USE torch.rand as it could produce repeated indices
                mask_indices = torch.randperm(block_size)[:num_to_mask]
                target_mask = block_nz_ns_indices[mask_indices].tolist()
                
                # Set masked indices to 0 in the context mask
                context_mask[target_mask] = 0

                # Add <cls> and special tokens to mask indices
                target_mask = torch.cat((
                    torch.arange(self.n_special_tokens),
                    torch.tensor(target_mask)))

                # Append masked indices
                target_masks.append(target_mask)
            else:
                # No elements to mask
                target_masks.append(torch.tensor([]))

        # We randomly permute data so if we trim last item with
        # keep_tokens_context, we avoid always discarding the last items of a
        # sequence
        # DON'T USE torch.rand as it could produce repeated indice
        context_mask = torch.nonzero(context_mask).squeeze()

        split_size = len(context_mask) // self.n_contexts
        remainder = len(context_mask) % self.n_contexts

        # Split context_mask into parts, distributing the remainder elements
        # across the first chunks
        context_masks = []
        start = 0
        for i in range(self.n_contexts):
            end = start + split_size + (1 if i < remainder else 0)
            context_block_mask = context_mask[start:end]
            context_block_mask = context_block_mask[
                torch.randperm(len(context_block_mask))]

            context_block_mask = torch.cat((
                torch.arange(self.n_special_tokens),
                context_block_mask))

            context_masks.append(context_block_mask)

            start = end

        return target_masks, context_masks

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
        collated_context_masks:
            Sampled context masks collated by batch.
        collated_target_masks:
            Sampled target masks collated by batch.
        collated_masks_attention:
            Attention masks collated by batch.
        """
        B = len(batch)

        # Collate the batch
        collated_batch = torch.utils.data.default_collate(batch)
        
        # Sample number of neighbors ONCE per batch and slice vectorized
        if self.sample_segments:
            # Number of segments kept in cell graph; k in [1,
            # n_segments]
            k = torch.randint(low=1, high=self.n_segments + 1, size=(1,)).item()
            cutoff = self.n_special_tokens + (self.seq_len_cell * k)

            # Pad all segments not kept in cell graph
            if 'tokens' in collated_batch:
                collated_batch['tokens'][:, cutoff:] = 0
            if 'segments' in collated_batch:
                collated_batch['segments'][:, cutoff:] = 0
            if 'positions' in collated_batch:
                collated_batch['positions'][:, cutoff:] = 0
            if 'values' in collated_batch:
                collated_batch['values'][:, cutoff:] = 0.0
            #if 'rel_x_coords' in collated_batch:
            #    collated_batch['rel_x_coords'][:, cutoff:] = float('-inf')
            #if 'rel_y_coords' in collated_batch:
            #    collated_batch['rel_y_coords'][:, cutoff:] = float('-inf')

        tokens = collated_batch["tokens"] # [B, N]
        B, N = tokens.shape

        # Build attention mask
        collated_masks_attention = (
            tokens != 0).unsqueeze(1).unsqueeze(1) # [B, 1, 1, N]

        if self.restrict_special_attention:
            collated_masks_attention = collated_masks_attention.expand(
                collated_masks_attention.shape[0],
                1,
                collated_masks_attention.shape[-1],
                collated_masks_attention.shape[-1]).clone()

            for i in range(self.max_cls_tokens, self.n_special_tokens):
                # Special tokens only attent to themselves
                collated_masks_attention[
                    :,
                    :,
                    i,
                    :i] = 0
                collated_masks_attention[
                    :,
                    :,
                    i,
                    (i+1):] = 0

        collated_target_masks = []
        collated_context_masks = []

        # Track the minimum length of masks across the batch
        keep_tokens_target = self.seq_len_genes
        keep_tokens_context = self.seq_len_genes

        # Store target and context masks for each observation
        for i in range(B):
            # Sample target and context masks for the current observation
            target_masks, context_masks = self._sample_gene_mask(
                tokens=batch[i]["tokens"],
                segments=batch[i]["segments"])

            keep_tokens_target = min(
                keep_tokens_target, min(mask.size(0) for mask in target_masks))
            keep_tokens_context = min(
                keep_tokens_context, min(mask.size(0) for mask in context_masks))

            # Append the masks for the current observation to the collated lists
            collated_target_masks.append(target_masks)
            collated_context_masks.append(context_masks)

        # Trim masks to the minimum size across the batch and collate them
        collated_target_masks = [
            [cm[:keep_tokens_target] for cm in cm_list]
            for cm_list in collated_target_masks]
        collated_context_masks = [
            [cm[:keep_tokens_context] for cm in cm_list]
            for cm_list in collated_context_masks]

        collated_target_masks = torch.utils.data.default_collate(
            collated_target_masks)
        collated_context_masks = torch.utils.data.default_collate(
            collated_context_masks)

        return collated_batch, collated_context_masks, collated_target_masks, collated_masks_attention
