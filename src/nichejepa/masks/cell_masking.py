"""
Cell masking.

Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture.
Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py
(05.06.2024).
"""

from typing import List, Tuple

import numpy as np
import torch


class CelllMaskCollator:
    """
    CelllMaskCollator class for sampling target and context block masks from
    cell and neighborhood segments using cell-based masking.

    Parameters
    ----------
    n_targets:
        Number of target masks (i.e., number of cells to mask) for each token sequence.
    n_contexts:
        Number of context masks to sample.
    n_segments:
        Number of segments.
    seq_len_cell:
        The length of the token sequence representing a single cell segment.
    seq_len_neighborhood:
        The length of the token sequence representing the neighborhood segments.
    n_special_tokens:
        Number of special tokens in each token sequence, including <cls> tokens.
    per_block_mask_ratio:
        Per cell mask ratio.
    targets_list:
        List of cells that should be in target.
    """
    def __init__(self,
                 n_targets: int,
                 n_contexts: int,
                 n_segments: int,
                 seq_len_cell: int,
                 seq_len_neighborhood: int,
                 n_special_tokens: int,
                 per_block_mask_ratio: float = 0.5,
                 targets_list: List[int]=None):
        self.n_targets = n_targets
        self.n_contexts = n_contexts
        self.n_segments = n_segments
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len_genes = self.seq_len_cell + self.seq_len_neighborhood
        self.n_special_tokens = n_special_tokens
        self.per_block_mask_ratio = per_block_mask_ratio
        if targets_list is not None:
            # Use provided targets_list.
            self.target_cell_indices = torch.tensor(targets_list, dtype=torch.long)
            all_cell_indices = torch.arange(self.n_segments, dtype=torch.long)
            # Compute context indices as those indices not in target_cell_indices.
            context_list = []
            for idx in all_cell_indices:
                if idx not in self.target_cell_indices:
                    context_list.append(idx)
            self.context_cell_indices = torch.tensor(context_list, dtype=torch.long)
        else:
            self.target_cell_indices = None
            self.context_cell_indices = None

    def _sample_gene_mask(self,
                          tokens: torch.Tensor,
                          segments: torch.Tensor,
                          ) -> Tuple[List[torch.Tensor],
                                     List[torch.Tensor],
                                     int]:
        """
        Perform cell masking: select `n_targets` random cells, include their nonzero tokens
        in the target mask(s), and place all other nonzero tokens into context masks.

        Parameters
        ----------
        tokens:
            Token sequence with shape (N,) where N is total token length.
        segments:
            Segment information (not used in this version but kept for API compatibility).

        Returns
        ----------
        target_masks:
            List of tensors with indices of nonzero tokens in selected target cells.
        context_masks:
            List of tensors with indices of nonzero tokens in context (non-target) cells.
        keep_tokens_target:
            Minimum number of nonzero tokens kept in any target mask (used for batch collation).
        """
        # Determine mask ratio; sample if list is provided
        if isinstance(self.per_block_mask_ratio, list):
            mask_ratios = np.arange(
                self.per_block_mask_ratio[0],
                self.per_block_mask_ratio[1] + 0.1, 0.1)
            mask_ratio = np.random.choice(mask_ratios)
        else:
            mask_ratio = self.per_block_mask_ratio

        ns_tokens = tokens[self.n_special_tokens:]
        total_seq_len = ns_tokens.shape[0]

        # Calculate the total number of cells
        n_cells = self.n_segments

        # Randomly choose target cell indices
        if self.context_cell_indices is None:
            all_cell_indices = torch.randperm(n_cells)
            target_cell_indices = all_cell_indices[:self.n_targets]
            context_cell_indices = all_cell_indices[self.n_targets:]
        else:
            target_cell_indices = self.target_cell_indices[torch.randperm(len(self.target_cell_indices))]
            context_cell_indices = self.context_cell_indices[torch.randperm(len(self.context_cell_indices))]

        target_masks = []
        context_indices = []
        keep_tokens_target = self.seq_len_genes

        # Process target cells
        for idx in target_cell_indices:
            start = idx * self.seq_len_cell
            end = start + self.seq_len_cell
            cell_tokens = ns_tokens[start:end]

            # Find non-zero indices within cell
            nonzero_indices = torch.nonzero(cell_tokens).squeeze()

            # Determine how many indices will be masked
            num_to_mask = int(np.ceil(len(nonzero_indices) * mask_ratio))

            # Map to global indices relative to ns_tokens
            global_indices = nonzero_indices + start
            permuted_indices = torch.randperm(len(global_indices))
            global_target_indices = permuted_indices[:num_to_mask]
            global_context_indices = permuted_indices[num_to_mask:]

            keep_tokens_target = min(keep_tokens_target, len(global_target_indices))

            context_indices.append(global_context_indices)
            target_masks.append(global_target_indices)

        # Process context cells
        keep_tokens_context = self.seq_len_genes

        for idx in context_cell_indices:
            start = idx * self.seq_len_cell
            end = start + self.seq_len_cell
            cell_tokens = ns_tokens[start:end]

            nonzero_indices = torch.nonzero(cell_tokens).squeeze()
            global_indices = nonzero_indices + start

            context_indices.append(global_indices)

        # Flatten context indices
        context_indices = torch.cat(context_indices)

        # Shuffle and split context indices into n_contexts masks
        permuted_indices = context_indices[torch.randperm(len(context_indices))]
        split_size = len(permuted_indices) // self.n_contexts
        remainder = len(permuted_indices) % self.n_contexts

        context_masks = []
        start = 0
        for i in range(self.n_contexts):
            end = start + split_size + (1 if i < remainder else 0)
            context_mask = permuted_indices[start:end]
            keep_tokens_context = min(keep_tokens_context, len(context_mask))
            context_masks.append(context_mask)
            start = end

        return target_masks, context_masks, keep_tokens_target, keep_tokens_context

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
        Apply cell masking across a batch.

        Parameters
        ----------
        batch:
            Tuple containing gene tokens, segments, positions, counts, and cell IDs.

        Returns
        ----------
        collated_batch:
            Collated input batch.
        collated_context_masks:
            Collated context masks.
        collated_target_masks:
            Collated target masks.
        collated_masks_attention:
            Collated attention masks.
        """
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        collated_target_masks = []
        collated_context_masks = []
        collated_masks_attention = []

        keep_tokens_target = self.seq_len_genes
        keep_tokens_context = self.seq_len_genes

        for i in range(B):
            target_masks, context_masks, min_target_len, min_context_len = self._sample_gene_mask(
                tokens=batch[i][0],
                segments=batch[i][1])

            keep_tokens_target = min(keep_tokens_target, min_target_len)
            keep_tokens_context = min(keep_tokens_context, min_context_len)

            collated_target_masks.append(target_masks)
            collated_context_masks.append(context_masks)

            collated_masks_attention.append(
                (batch[i][0][self.n_special_tokens:] != 0).int())

        # Trim all target masks to minimum size across batch
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
        collated_masks_attention = torch.utils.data.default_collate(
            collated_masks_attention).unsqueeze(1).unsqueeze(1)

        return collated_batch, collated_context_masks, collated_target_masks, collated_masks_attention
