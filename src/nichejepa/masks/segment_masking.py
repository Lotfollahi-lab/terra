import numpy as np
import torch
from logging import getLogger
from multiprocessing import Value
from typing import List, Literal, Optional, Tuple


logger = getLogger()
_GLOBAL_SEED = 0


class SegmentMaskCollator:
    """
    SegmentMaskCollator class for sampling target and context masks from cell
    and neighborhood segments.
    
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
        The length of the token sequence representing the cell segment.
    seq_len_neighborhood: int
        The length of the token sequence representing the neighborhood segment.
    has_cls: bool
        If True, the sequence contains a <cls> token at the 0th index, included
        in the masks.
    has_gene_panel:
    per_segment_mask_ratio: float
        The ratio of elements to be masked in each segment.
    """
    
    def __init__(self,
                 n_targets: int=2,
                 n_contexts: int=1,
                 target_mask_size: int=2,
                 context_mask_size: int=10,
                 seq_len_cell: int=0,
                 seq_len_neighborhood: int=0,
                 has_cls: bool=False,
                 has_gene_panel: bool=False,
                 per_segment_mask_ratio: float=0.3):
        self.n_targets = n_targets
        self.n_contexts = n_contexts
        self.target_mask_size = target_mask_size
        self.context_mask_size = context_mask_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len = self.seq_len_cell + self.seq_len_neighborhood
        self.has_cls = has_cls
        self.has_gene_panel = has_gene_panel
        self.per_segment_mask_ratio = per_segment_mask_ratio

        # Determine the valid start position for the mask based on the presence
        # of a <cls> token and gene panel token
        self.valid_min_start = 1 if self.has_cls else 0
        if has_gene_panel:
            self.valid_min_start += 1

    def segment_masking(self,
                        sequence,
                        mask_ratio: float,
                        ) -> List:
        """
        Perform segment masking on the sequence based on the number of targets
        and per-segment mask ratio.

        Parameters
        ----------
        sequence: Tensor, shape (n_samples,)
            The input sequence that needs to be masked.
        mask_ratio:
            Ratio of elements to be masked in each segment.

        Returns
        ----------
        segment_masks: List[Tensor]
            A list of masked indices for each segment.
        context_mask: List
            List of binary mask indicating context tokens (1s where context is, 0s where masked).
        keep_tokens_target: int
            Minimum number of tokens kept across all target masks.
        """
        non_zero_indices = torch.nonzero(sequence[self.valid_min_start:]).squeeze()  # Indices where sequence is non-zero
        total_non_zero = len(non_zero_indices)  # Total non-zero elements in the sequence
    
        # Initialize a list to store masked indices for each segment
        segment_masks = []
        context_mask = torch.zeros(len(sequence), dtype=torch.int32)  # Initialize context mask
        keep_tokens_target = float('inf')  # Keep track of the minimum number of target tokens across segments

        # Compute segment length based on the number of targets
        segment_length = max(1, total_non_zero // self.n_targets)  # Avoid dividing by zero
        num_segments = self.n_targets

        for i in range(num_segments):
            # Determine the range of indices for the current segment
            start_idx = i * segment_length + self.valid_min_start
            end_idx = min(start_idx + segment_length, total_non_zero)
        
            # Extract the non-zero indices for the current segment
            segment_non_zero_indices = non_zero_indices[start_idx:end_idx]
            context_mask[segment_non_zero_indices] = 1  # Mark as context initially
            
            segment_size = len(segment_non_zero_indices)
            num_to_mask = int(np.ceil(segment_size * mask_ratio))  # Determine number of elements to mask

            if num_to_mask > 0:
                # Randomly choose indices to mask within the segment
                # DON'T USE torch.rand as it could produce repeated indices
                mask_indices = torch.randperm(segment_size)[:num_to_mask]
                masked_indices = segment_non_zero_indices[mask_indices].tolist()  # Convert to list
                context_mask[masked_indices] = 0  # Set masked indices to 0 in the context mask
                keep_tokens_target = min(keep_tokens_target, len(masked_indices))  # Update minimum tokens target
                if self.has_cls and self.has_gene_panel: # add index of cls and gene panel
                    masked_indices = [0, 1] + masked_indices
                elif self.has_cls or self.has_gene_panel:
                    masked_indices = [0] + masked_indices
                segment_masks.append(torch.tensor(masked_indices))  # Append the masked indices io the list
            else:
                segment_masks.append(torch.tensor([]))  # If no elements to mask, append an empty list
        # DON'T USE torch.rand as it could produce repeated indices
        # We randomly permut data so if we trim last item with keep_tokens_context
        # We avoid always discarding the last items of a sequence, as this may be problematic.
        context_mask = torch.nonzero(context_mask).squeeze()
        context_mask = context_mask[torch.randperm(len(context_mask))]
        # Add cls to context if it exist
        if self.has_cls and self.has_gene_panel:
            context_mask = torch.cat((torch.tensor([0, 1]), context_mask))
        elif self.has_cls or self.has_gene_panel:
            context_mask = torch.cat((torch.tensor([0]), context_mask))
        return segment_masks, [context_mask], keep_tokens_target

    def _sample_gene_mask(self, sequence):
        """
        Sample context or target gene masks, considering both cell and neighborhood segments.

        Parameters
        ----------
        sequence: Tensor
            A sequence of tokens as input

        Returns
        ----------
        target_masks: List[List[int]]
            A list of target masks per segment.
        context_mask: Tensor
            Binary tensor indicating the context mask.
        keep_tokens_target: int
            The minimum number of tokens kept across target masks.
        """
        # Apply segment masking on the full sequence
        target_masks, context_mask, keep_tokens_target = self.segment_masking(
            sequence, self.per_segment_mask_ratio)

        return target_masks, context_mask, keep_tokens_target

    def __call__(self,
                 batch: Tuple[torch.Tensor, torch.Tensor, str]
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create context and target masks when collating cell neighborhoods into a batch.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, str]
            The input sequence tokens, segment labels, and cell-level labels for all observations in the batch.

        Returns
        ----------
        collated_batch: torch.Tensor
            The input gene tokens, segment labels, and cell-level labels collated by batch.
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
        keep_tokens_target = self.seq_len
        keep_tokens_context = self.seq_len

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
    
