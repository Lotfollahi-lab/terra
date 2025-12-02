"""
Adapted from Assran, M. et al. Self-supervised learning from images with
a Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf.
Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py
(05.06.2024).
"""

import time # TODO: remove

import numpy as np
import torch 


class BlockMaskCollator:
    """
    BlockMaskCollator class for sampling target and context block masks
    from cell and neighborhood segments.
    
    Parameters
    ----------
    n_targets:
        Number of target masks to sample for each token sequence.
    n_contexts:
        Number of context masks to sample for each token sequence.
    n_segments:
        Number of segments.
    seq_len_cell:
        The length of the token sequence representing the cell segment.
    seq_len_neighborhood:
        The length of the token sequence representing the neighborhood
        segments.
    n_special_tokens:
        Number of special tokens in each token sequence.
    per_block_mask_ratio:
        Ratio of elements to be masked in each block. A list with min
        and max ratio can be provided, in which case a value between the
        min and max will be sampled for each batch.
    sample_segments:
        If `True`, sample number of neighbors in each batch.
    sample_gene_masks:
        If `True`, sample a gene mask for each cell. Should be used
        during training but not inference.
    """
    def __init__(self,
                 n_targets: int,
                 n_contexts: int,
                 n_segments: int,
                 seq_len_cell: int,
                 seq_len_neighborhood: int,
                 n_special_tokens: int,
                 per_block_mask_ratio: float = 0.5,
                 sample_segments: bool = False,
                 sample_gene_masks: bool = True):
        self.n_targets = n_targets
        self.n_contexts = n_contexts
        self.n_segments = n_segments
        self.seq_len_cell = seq_len_cell
        self.seq_len_neighborhood = seq_len_neighborhood
        self.seq_len_genes = self.seq_len_cell + self.seq_len_neighborhood
        self.n_special_tokens = n_special_tokens
        self.per_block_mask_ratio = per_block_mask_ratio
        self.sample_segments = sample_segments
        self.sample_gene_masks = sample_gene_masks

    def _sample_gene_mask(
        self,
        tokens: torch.Tensor, # [N]
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Perform block masking on the sequence based on the number of
        targets (number of blocks) and per block mask ratio. Tokens not
        sampled in the targets will be part of the context.

        Parameters
        ----------
        tokens:
            The token sequence that needs to be masked with dimension
            (B, N); B: batch size, N: number of tokens.
        segments:
            The sequence of segments to determine which <cls> tokens are
            included in the target masks.

        Returns
        ----------
        target_masks:
            List with multiple masks indicating target token indices for
            each block.
        context_masks:
            List with one mask indicating context token indices.
        keep_tokens_target:
            Minimum number of tokens kept across all target masks.
        """
        # Determine mask ratio; sample if list is provided
        if isinstance(self.per_block_mask_ratio, list):
            min_mask_ratio, max_mask_ratio = self.per_block_mask_ratio
            steps = int(round((max_mask_ratio - min_mask_ratio) / 0.1)) + 1
            grid = torch.linspace(min_mask_ratio, max_mask_ratio, steps=steps)
            idx = torch.randint(0, steps, (1,), device=grid.device).item()
            mask_ratio = float(grid[idx])
        else:
            mask_ratio = float(self.per_block_mask_ratio)

        # Get non-zero indices, excluding special tokens
        L = tokens.numel()
        ns_nz_mask = tokens[self.n_special_tokens:] != 0
        nz = ns_nz_mask.nonzero(
            as_tuple=False).squeeze(-1) + self.n_special_tokens # [K]
        K = nz.numel()

        # Split non-zero indices into n_targets chunks that cover all K
        # elements (later chunks may be empty)
        blocks = list(torch.tensor_split(nz, self.n_targets))

        # Initialize context as "all non-zero"
        context_keep = torch.zeros(L, dtype=torch.bool)
        context_keep[nz] = True

        target_masks: list[torch.Tensor] = []
        for b in blocks:
            block_size = b.numel()
            if block_size == 0:
                target_masks.append(torch.empty(0, dtype=torch.long))
                continue
            num_to_mask = min(
                int((block_size * mask_ratio) + 0.9999),
                block_size) # ceil
            if num_to_mask == 0:
                target_masks.append(torch.empty(0, dtype=torch.long))
                continue
            sel = torch.randperm(block_size)[:num_to_mask]
            tmask = b[sel]
            # Remove target from context
            context_keep[tmask] = False
            target_masks.append(tmask)

        if self.n_contexts != 1:
            raise ValueError(
                "Only n_contexts == 1 is supported currently.")

        # context indices, randomized, + specials at front
        ctx_idx = context_keep.nonzero(as_tuple=False).squeeze(-1)
        if ctx_idx.numel() > 1:
            ctx_idx = ctx_idx[torch.randperm(ctx_idx.numel())]
        special_idx = torch.arange(
            self.n_special_tokens,
            device=tokens.device,
            dtype=torch.long)
        ctx_idx = torch.cat((special_idx, ctx_idx), dim=0)

        context_masks = [ctx_idx]

        return target_masks, context_masks

    def __call__(self, batch: list[dict]) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate batch and create context, target, and attention masks.

        Parameters
        ----------
        batch:
            List containing the input batch dictionaries including
            positions, segments, gene tokens, values and cell IDs.

        Returns
        ----------
        collated:
            Input positions, segments, gene tokens, values and cell IDs
            collated by batch.
        collated_context_masks: LongTensor [B, Nctx, Lctx] (min-trimmed)
            Sampled context masks collated by batch.
        collated_target_masks: LongTensor [B, Ntgt, Ltgt] (min-trimmed)
            Sampled target masks collated by batch.
        masks_attention: BoolTensor [B, 1, 1, L]
            Attention masks collated by batch.
        """
        #t0 = time.perf_counter()

        # Collate early for vectorized slicing
        collated = torch.utils.data.default_collate(batch)

        # Sample number of neighbors ONCE per batch and slice vectorized
        if self.sample_segments:
            # Number of segments kept in cell graph; k in [1,
            # n_segments]
            k = torch.randint(low=1, high=self.n_segments+1, size=(1,)).item()
            cutoff = self.seq_len_cell * k

            # Pad all segments not kept in cell graph
            if 'tokens' in collated:
                collated['tokens'][:, cutoff:] = 0
            if 'segments' in collated:
                collated['segments'][:, cutoff:] = 0
            if 'positions' in collated:
                collated['positions'][:, cutoff:] = 0
            if 'values' in collated:
                collated['values'][:, cutoff:] = 0.0
            if 'rel_x_coords' in collated:
                collated['rel_x_coords'][:, cutoff:] = float('-inf')
            if 'rel_y_coords' in collated:
                collated['rel_y_coords'][:, cutoff:] = float('-inf')

        tokens = collated['tokens'] # [B, N]
        B, N = tokens.shape

        # Build attention mask once (bool, broadcast-friendly)
        masks_attention = (
            tokens != 0).unsqueeze(1).unsqueeze(1) # [B, 1, 1, N]

        if self.sample_gene_masks:
            # Retrieve target/context masks per cell
            tgt_list: list[list[torch.Tensor]] = []
            ctx_list: list[list[torch.Tensor]] = []
            keep_tgt = self.seq_len_genes
            keep_ctx = self.seq_len_genes

            for i in range(B):
                tgt, ctx = self._sample_gene_mask(
                    tokens=tokens[i])
                # Track min lengths (avoid nested default_collate later)
                if len(tgt):
                    keep_tgt = min(keep_tgt, min(m.numel() for m in tgt))
                if len(ctx):
                    keep_ctx = min(keep_ctx, min(m.numel() for m in ctx))
                tgt_list.append(tgt)
                ctx_list.append(ctx)

            # Trim to min length and stack directly (faster than
            # default_collate on nested lists)
            tgt_trimmed = [
                torch.stack([m[:keep_tgt].to(torch.long) for m in masks], dim=0
                    ) for masks in tgt_list]
            collated_target_masks = torch.stack(
                tgt_trimmed, dim=0) # [B, n_tgt, keep_tgt]
            collated_target_masks = collated_target_masks.permute(
                1, 0, 2).contiguous() # [n_tgt, B, keep_tgt]

            ctx_trimmed = [
                torch.stack([m[:keep_ctx].to(torch.long) for m in masks], dim=0
                    ) for masks in ctx_list]
            collated_context_masks = torch.stack(
                ctx_trimmed, dim=0) # [B, n_ctx, keep_ctx]
            collated_context_masks = collated_context_masks.permute(
                1, 0, 2).contiguous()  # [n_ctx, B, keep_ctx]
        else:
            collated_target_masks = None
            collated_context_masks = None

        #t1 = time.perf_counter()
        #elapsed_ms = (t1 - t0) * 1000
        #print(f"[Collate] Took {elapsed_ms:.3f} ms for batch size {B}")
        #raise ValueError

        return collated, \
               collated_context_masks, \
               collated_target_masks, \
               masks_attention