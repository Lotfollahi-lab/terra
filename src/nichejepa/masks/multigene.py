"""
MaskCollator.

Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture.
Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py
(05.06.2024).
"""

from logging import getLogger
from multiprocessing import Value
from typing import Literal, Optional, Tuple

import torch


_GLOBAL_SEED = 0


logger = getLogger()


class MaskCollator:
    """
    MaskCollator class for sampling target and context masks from cell and
    neighborhood segments.

    Parameters
    ----------
    n_targets:
        Number of target masks to sample for each input sequence.
    n_contexts:
        Number of context masks to sample for each input sequence.
    target_mask_size:
        The size (in number of tokens) of each target mask.
    context_mask_size:
        The size (in number of tokens) of each context mask.
    seq_len_cell:
        The length of the token sequence representing the cell segment.
    seq_len_neighborhood:
        The length of the token sequence representing the neighborhood segment.
    has_cls:
        If 'True', sequence contains <cls> token at the 0th index, which will be
        included in the masks.
    """
    def __init__(self,
                 n_targets: int=2,
                 n_contexts: int=1,
                 target_mask_size: int=2,
                 context_mask_size: int=10,
                 seq_len_cell: int=0,
                 seq_len_neighborhood: int=0,
                 has_cls: bool = False
                 ):
        self.n_targets = n_targets
        self.n_contexts = n_contexts
        self.target_mask_size = target_mask_size
        self.context_mask_size = context_mask_size
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len = self.seq_len_cell + self.seq_len_neighborhood
        self.has_cls = has_cls
        # Determine the valid minimum start position for the mask based on the
        # presence of a <cls> token
        self.valid_min_start = 1 if self.has_cls else 0
        # Shared counter to manage iterations across multiple processes
        self._itr_counter = Value('i', -1)

    def step(self):
        """ 
        Increment the iteration counter.
        
        This is used to ensure that each worker process generates a unique seed
        for random sampling.
        """
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_gene_mask(self,
                          non_zero_seq_len_cell: int,
                          non_zero_seq_len_neighborhood: int,
                          generator: torch.Generator,
                          mask_size: int,
                          mask_type: Literal['target', 'context'],
                          valid_token_masks: Optional[list]=None
                          ) -> Tuple[torch.Tensor,
                                     torch.Tensor]:
        """
        Sample context or target gene masks, considering both cell and
        neighborhood segments.

        Parameters
        ----------
        non_zero_seq_len_cell:
            Number of non-zero tokens in the cell segment.
        non_zero_seq_len_neighborhood:
            Number of non-zero tokens in the neighborhood segment.
        generator:
            Pseudorandom number generator to ensure reproducibility.
        mask_size:
            Length of the masked token sequence.
        mask_type:
            Type for which to create the mask. Can be 'target' or 'context'.
        valid_token_masks: Optional[List[torch.Tensor]]
            A list of binary masks that constrain the valid token positions for
            masking.

        Returns
        ----------
        mask:
            Binary tensor with 1s for sampled tokens and 0s otherwise.
        mask_complement:
            Binary tensor complementing the mask, with 0s for sampled tokens and
            1s otherwise.
        """
        if mask_type == 'target':
            mask_size = min(
                non_zero_seq_len_cell + non_zero_seq_len_neighborhood,
                mask_size)
            # Sample the start position for the mask within the valid range
            # using the provided generator for reproducibility
            start = torch.randint(
                self.valid_min_start,
                (self.valid_min_start + non_zero_seq_len_cell +
                 non_zero_seq_len_neighborhood - mask_size + 1),
                generator=generator,
                size=(1,))

            # Initialize the mask and its complement
            mask = torch.zeros(self.seq_len_cell + self.seq_len_neighborhood,
                               dtype=torch.int32)
            mask_complement = torch.ones_like(mask)

            # Apply the mask within the cell segment if the start is within the
            # cell
            if start < (self.valid_min_start + non_zero_seq_len_cell):
                # Determine the end of the mask within the cell
                cell_end = min(start + mask_size,
                               self.valid_min_start + non_zero_seq_len_cell)
                mask[start:cell_end] = 1
                mask_complement[start:cell_end] = 0

                # Handle overflow into the neighborhood segment if the mask
                # exceeds the non zero cell length sequence
                if cell_end < start + mask_size:
                    overflow = start + mask_size - (
                        self.valid_min_start + non_zero_seq_len_cell)
                    mask[
                        self.valid_min_start + self.seq_len_cell:
                        self.valid_min_start + self.seq_len_cell + overflow] = 1
                    mask_complement[
                        self.valid_min_start + self.seq_len_cell:
                        self.valid_min_start + self.seq_len_cell + overflow] = 0
            else:
                # Apply the mask entirely within the neighborhood segment
                neighborhood_start = start - (
                    self.valid_min_start + non_zero_seq_len_cell)
                neighborhood_end = neighborhood_start + mask_size
                mask[self.valid_min_start + self.seq_len_cell + 
                     neighborhood_start:
                     self.valid_min_start + self.seq_len_cell +
                     neighborhood_end] = 1
                mask_complement[self.valid_min_start + self.seq_len_cell +
                                neighborhood_start:
                                self.valid_min_start + self.seq_len_cell +
                                neighborhood_end] = 0

        elif mask_type == 'context':
            # Sample the start position for the context mask
            start = torch.randint(
                self.valid_min_start,
                self.valid_min_start + self.seq_len_neighborhood + 
                self.seq_len_cell - mask_size + 1,
                generator=generator,
                size=(1,))

            # Initialize the mask and its complement with zeros and ones,
            # respectively
            mask = torch.zeros(self.seq_len_cell + self.seq_len_neighborhood,
            dtype=torch.int32)
            mask_complement = torch.ones_like(mask)

            # Apply the mask starting from the sampled position
            mask[start:mask_size + start] = 1
            mask_complement[start:mask_size + start] = 0

        # Include the CLS token in the mask if applicable
        if self.has_cls:
            mask[0] = 1

        # Constrain the mask to valid token positions if provided
        if valid_token_masks is not None:
            for valid_mask in valid_token_masks:
                mask *= valid_mask

        # Convert the mask to a tensor of indices where the mask is applied
        mask = torch.nonzero(mask).squeeze()

        return mask, mask_complement

    def __call__(self,
                 batch: Tuple[torch.Tensor, torch.Tensor, str]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create context and target masks when collating cell neighborhoods into a
        batch.
        # 1. sample several target masks
        # 2. sample non-overlapping context masks
        # 3. Add <cls> token to both context and target masks
        # 4. return context and target masks

        Parameters
        ----------
        batch:
            Tuple containing the input sequence tokens, segment labels and
            cell-level labels for all observations in the batch.
    
        Returns
        ----------
        collated_batch:
            Input gene tokens, segment labels, and cell-level labels collated by
            batch.
        collated_masks_context:
            Sampled context masks collated by batch.
        collated_masks_target:
            Sampled target masks collated by batch.
        collated_masks_attention:
            masks_attention by batch.
        """
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_target, collated_masks_context, collated_masks_attention = [], [], []

        # Initialize variables to track the minimum length of masks across the
        # batch
        keep_tokens_target = self.seq_len
        keep_tokens_context = self.seq_len

        # Create a pseudorandom number generator for sampling
        seed = self.step()  # Get a unique seed for this iteration
        g = torch.Generator()
        g.manual_seed(seed)  # Ensure reproducibility by setting the seed

        for i in range(B):
            # Create a collated_masks_attention
            collated_masks_attention.append((batch[i][0]!=0).int())
            # Initialize lists to store target masks and their complements for
            # the current observation
            masks_target_complement = []
            masks_target = []
            masks_context = []
            
            # Calculate the number of non-zero tokens in the cell segment
            if self.seq_len_cell != 0:
                non_zero_seq_len_cell = torch.nonzero(
                    batch[i][0][self.valid_min_start:
                                self.valid_min_start+self.seq_len_cell]).size(0)
            else:
                non_zero_seq_len_cell = 0

            # Calculate the number of non-zero tokens in the neighborhood segment
            if self.seq_len_neighborhood != 0:
                non_zero_seq_len_neighborhood = torch.nonzero(
                    batch[i][0][self.valid_min_start+self.seq_len_cell:
                    ]).size(0)
            else:
                non_zero_seq_len_neighborhood = 0

            # Sample target masks for the current observation
            for _ in range(self.n_targets):
                mask_target, mask_target_complement = self._sample_gene_mask(
                    non_zero_seq_len_cell=non_zero_seq_len_cell,
                    non_zero_seq_len_neighborhood=non_zero_seq_len_neighborhood,
                    generator=g, # Use the generator for reproducibility
                    mask_size=self.target_mask_size,
                    mask_type='target')
                masks_target.append(mask_target)
                masks_target_complement.append(mask_target_complement)
                keep_tokens_target = min(keep_tokens_target, len(mask_target))

            # Sample context masks for the current observation, ensuring they do
            # not overlap with target masks
            for _ in range(self.n_contexts):
                mask_context, _ = self._sample_gene_mask(
                    non_zero_seq_len_cell=non_zero_seq_len_cell,
                    non_zero_seq_len_neighborhood=non_zero_seq_len_neighborhood,
                    mask_size=self.context_mask_size,
                    generator=g, # Use same generator to ensure reproducibility
                    valid_token_masks=masks_target_complement,
                    mask_type='context')
                masks_context.append(mask_context)
                keep_tokens_context = min(keep_tokens_context,
                                          len(mask_context))

            # Append the masks for the current observation to the collated lists
            collated_masks_target.append(masks_target)
            collated_masks_context.append(masks_context)

        # Trim the masks to the minimum size across the batch and collate them
        collated_masks_target = [[cm[:keep_tokens_target] for cm in cm_list]
                                 for cm_list in collated_masks_target]
        collated_masks_target = torch.utils.data.default_collate(
            collated_masks_target)
        collated_masks_context = [[cm[:keep_tokens_context] for cm in cm_list]
                                  for cm_list in collated_masks_context]
        collated_masks_context = torch.utils.data.default_collate(
            collated_masks_context)
        collated_masks_attention = torch.utils.data.default_collate(
                collated_masks_attention).unsqueeze(1).unsqueeze(1)

        return collated_batch, collated_masks_context, collated_masks_target, collated_masks_attention
