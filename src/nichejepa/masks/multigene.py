"""
MaskCollator.

Adapted from Assran, M. et al. Self-supervised learning from images with a Joint-Embedding Predictive Architecture.
Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py (05.06.2024).
"""

from logging import getLogger
from multiprocessing import Value

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator():
    def __init__(self,
                 seq_len,
                 n_targets=2,
                 target_mask_size=2,
                 context_mask_size=10,
                 n_contexts=1):
        self.seq_len = seq_len
        self.n_targets = n_targets
        self.target_mask_size = target_mask_size
        self.context_mask_size = context_mask_size
        self.n_contexts = n_contexts
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def _sample_gene_mask(self,
                          non_zero_seq_len,
                          mask_size,
                          valid_token_masks=None):
        if mask_size < non_zero_seq_len:
            # -- Sample start token
            start = torch.randint(0,
                                  non_zero_seq_len - mask_size,
                                  (1,))
        else:
            start = 0

        mask = torch.zeros(self.seq_len, dtype=torch.int32)
        mask[start:start+mask_size] = 1

        # -- Constrain mask to a set of valid tokens
        if valid_token_masks is not None:
            for k in range(len(valid_token_masks)):
                mask *= valid_token_masks[k]

        mask = torch.nonzero(mask.flatten())
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones(self.seq_len, dtype=torch.int32)
        mask_complement[start:start+mask_size] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create context and target masks when collating cell neighborhoods into a batch
        # 1. sample context block (size + location) using seed
        # 2. sample target block (size) using seed
        # 3. sample several context block locations for each cell neighborhood (w/o seed)
        # 4. sample several target block locations for each cell neighborhood (w/o seed)
        # 5. return context mask and target mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        collated_masks_context, collated_masks_target = [], []

        keep_tokens_target = self.seq_len
        keep_tokens_context = self.seq_len
        for i in range(B):
            masks_target_complement = []
            masks_target = []
            masks_context = []

            non_zero_seq_len = torch.nonzero(batch[i][0]).size(0)

            for _ in range(self.n_targets):
                mask_target, mask_target_complement = self._sample_gene_mask(
                    non_zero_seq_len,
                    self.target_mask_size)
                masks_target.append(mask_target)
                masks_target_complement.append(mask_target_complement)
                keep_tokens_target = min(keep_tokens_target, len(mask_target))
            
            for _ in range(self.n_contexts):
                mask_context, _ = self._sample_gene_mask(
                    non_zero_seq_len,
                    self.context_mask_size,
                    valid_token_masks=masks_target_complement)
                masks_context.append(mask_context)
                keep_tokens_context = min(keep_tokens_context, len(mask_context))

            collated_masks_target.append(masks_target)
            collated_masks_context.append(masks_context)

        collated_masks_target = [[cm[:keep_tokens_target] for cm in cm_list] for cm_list in collated_masks_target]
        collated_masks_target = torch.utils.data.default_collate(collated_masks_target)
        collated_masks_context = [[cm[:keep_tokens_context] for cm in cm_list] for cm_list in collated_masks_context]
        collated_masks_context = torch.utils.data.default_collate(collated_masks_context)

        return collated_batch, collated_masks_context, collated_masks_target
