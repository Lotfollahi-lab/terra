"""
Read-Depth-Aware (RDA) conditioning for spatial / single-cell
transformers, model-side.

Inspired by scFoundation (Hao et al. 2024), which prepends a [T]
("total counts") token to the gene-expression sequence so the model
can disentangle technical sequencing depth from biological signal.
Their downsample-and-predict training objective ("read-depth-aware"
modeling) gives the model an explicit depth conditioning signal.

We implement an equivalent conditioning *without modifying the
tokenizer* by:

1. Reading per-cell total counts ``T_c`` directly from
   ``batch['values']`` at runtime (the gene-count slot we already
   carry through the model).
2. Embedding ``log(1 + T_c)`` through a tiny MLP with a
   **zero-initialized output head** so RDA-at-step-0 is a no-op,
   preserving the original encoder behavior under
   ``rda.enabled = True``.
3. Broadcasting the per-cell depth embedding to every gene-token
   position belonging to that cell, and adding it to the input
   embeddings before the transformer blocks.

Optionally a "target depth" S can be provided per cell (in
``batch['rda_target_depth']``) -- when present the depth embedding
sees both ``log(1+T)`` and ``log(1+S)`` as input, matching
scFoundation's "T and S tokens" formulation. If absent (the default)
the model only conditions on the observed depth.

This is strictly additive at the input embedding layer: no new
parameters touch the trained encoder body, no shapes change, no
state-dict keys move. Disabled by default; flip on via the YAML
``rda.enabled`` flag.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthConditioning(nn.Module):
    """Per-cell depth embedding for RDA-style conditioning.

    Takes per-cell ``log(1 + T)`` (and optionally ``log(1 + S)``)
    through a 2-layer MLP whose output is **zero-initialized**, so
    the depth contribution to the encoder is exactly zero at step 0.
    This makes ``rda.enabled = True`` strictly backward-compatible
    with the no-RDA baseline -- any divergence emerges through
    training, not initialization.

    Args
    ----
    embed_dim:
        Output dimension. Must match the encoder ``embed_dim`` so
        the depth embedding can be added to the input embeddings.
    hidden_dim:
        MLP hidden width. Cheap; 32 is plenty for a 1- or 2-scalar
        input.
    use_target_depth:
        If ``True``, the MLP also accepts a target depth ``S`` (so
        input dim is 2 instead of 1). Use this if you plan to
        provide ``batch['rda_target_depth']`` at runtime; otherwise
        leave as ``False`` and the model conditions on observed
        depth only.
    """

    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int = 32,
                 use_target_depth: bool = False,
                 ):
        super().__init__()
        in_dim = 2 if use_target_depth else 1
        self.use_target_depth = use_target_depth
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Zero-init the output projection so the depth embedding is
        # exactly zero at construction. Mark these as no-weight-decay
        # so the optimizer's L2 regularization doesn't drag them off
        # zero before any gradient has flowed through them.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        for p in self.mlp[-1].parameters():
            setattr(p, "_no_weight_decay", True)

    def forward(self,
                depth: torch.Tensor,
                target_depth: torch.Tensor | None = None,
                ) -> torch.Tensor:
        """Embed per-cell depth.

        Args
        ----
        depth:
            ``(B, n_cells)`` per-cell total counts (T).
        target_depth:
            Optional ``(B, n_cells)`` target depth (S). Required iff
            ``use_target_depth`` is True; ignored otherwise.

        Returns
        -------
        ``(B, n_cells, embed_dim)`` depth embedding to broadcast
        across cell positions.
        """
        log_t = torch.log1p(depth.float()).unsqueeze(-1)  # (B, c, 1)
        if self.use_target_depth:
            if target_depth is None:
                raise ValueError(
                    "DepthConditioning was constructed with "
                    "use_target_depth=True but no target_depth tensor "
                    "was provided at forward time. Either pass "
                    "batch['rda_target_depth'] or rebuild without "
                    "use_target_depth.")
            log_s = torch.log1p(target_depth.float()).unsqueeze(-1)
            x = torch.cat([log_t, log_s], dim=-1)         # (B, c, 2)
        else:
            x = log_t                                     # (B, c, 1)
        return self.mlp(x.to(self.mlp[0].weight.dtype))   # (B, c, D)


def compute_per_cell_depth(
        values: torch.Tensor,
        n_special_tokens: int,
        n_cells: int,
        seq_len_cell: int,
        ) -> torch.Tensor:
    """Sum counts within each cell's contiguous gene-token block.

    Assumes the standard sequence layout used by the cell-graph /
    cell-neighborhood tokenizers: ``n_special_tokens`` special-token
    positions, followed by ``n_cells`` blocks of ``seq_len_cell``
    gene tokens each. Total expected sequence length is
    ``n_special_tokens + n_cells * seq_len_cell``.

    Args
    ----
    values:
        ``(B, L)`` tensor of count / special values. Special-token
        positions are excluded by construction (we slice them out
        before summing).
    n_special_tokens:
        Number of leading special-token positions.
    n_cells:
        Number of cells in the sequence (``n_segments``).
    seq_len_cell:
        Number of gene-token positions per cell.

    Returns
    -------
    ``(B, n_cells)`` per-cell total count. Non-positive values
    (sentinel ``-inf`` or padding) are clamped to 0 before
    summation so ``log1p`` downstream is well-defined.
    """
    B, L = values.shape
    expected = n_special_tokens + n_cells * seq_len_cell
    if L < expected:
        raise RuntimeError(
            f"compute_per_cell_depth: values length {L} < expected "
            f"{expected} (n_special_tokens={n_special_tokens} + "
            f"n_cells={n_cells} * seq_len_cell={seq_len_cell}). "
            "Check that the sequence layout matches the cell-graph / "
            "cell-neighborhood tokenizer convention.")
    gene = values[:, n_special_tokens : n_special_tokens
                  + n_cells * seq_len_cell]                # (B, c*S)
    gene = gene.reshape(B, n_cells, seq_len_cell)
    # Clamp negative sentinels (e.g. padding represented as -inf or
    # negative ints in some count_encoding modes) to 0 before
    # summing.
    gene = gene.clamp(min=0.0)
    return gene.sum(dim=-1)                                # (B, n_cells)


def build_depth_embedding(
        depth_module: DepthConditioning,
        batch: dict,
        n_special_tokens: int,
        n_cells: int,
        seq_len_cell: int,
        seq_len: int,
        ) -> torch.Tensor:
    """Compute the per-token depth embedding ready to add to the
    encoder's input ``x`` (shape ``(B, L, D)``).

    Per-cell depth is computed, embedded via ``depth_module``, then
    broadcast across each cell's gene-token block. Special-token
    positions get a zero contribution -- consistent with the
    initialization, so RDA at step 0 changes nothing.
    """
    depth = compute_per_cell_depth(
        batch['values'],
        n_special_tokens=n_special_tokens,
        n_cells=n_cells,
        seq_len_cell=seq_len_cell,
    )                                                      # (B, c)
    target_depth = batch.get('rda_target_depth', None)
    depth_emb = depth_module(depth, target_depth)          # (B, c, D)

    B, _, D = depth_emb.shape
    # Broadcast: special positions get zeros; gene-token positions
    # get the corresponding cell's depth embedding.
    out = torch.zeros(
        B, seq_len, D, device=depth_emb.device, dtype=depth_emb.dtype)
    gene_block = depth_emb.unsqueeze(2).expand(
        B, n_cells, seq_len_cell, D
    ).reshape(B, n_cells * seq_len_cell, D)
    out[:, n_special_tokens : n_special_tokens
        + n_cells * seq_len_cell, :] = gene_block
    return out
