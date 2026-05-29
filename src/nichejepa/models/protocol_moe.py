"""
Protocol-routed bias head for the JEPA predictor.

A lightweight mixture-of-experts variant: instead of routing entire
expert MLPs (scGPT-spatial style), we route a single per-expert
bias vector and add it to the predictor output. This is a strict
generalization of "predict the same thing across all protocols";
when ``ProtocolBias`` is zero-initialized, it's a no-op.

Why bias-only and not full experts?
-----------------------------------
- Full MoE doubles parameter count and complicates checkpoint
  layout. For the typical regime here -- 10-20 distinct protocols
  -- the residual systematic offsets are usually well-modeled by
  a per-protocol bias, and any remaining variance can be absorbed
  by the rest of the predictor.
- Zero-init makes turning it on backward-compatible: step 0 is
  bitwise identical to the no-MoE baseline.

The routing key is a per-cell integer index in
``batch['values'][:, routing_index]`` (e.g. the assay_value or
gene_panel_value slot, depending on tokenizer convention).

Important: the value at ``values[:, routing_index]`` is the
**offset-subtracted spv_* token ID**, which lives in a GLOBAL
range across all spv_* slots, not 0-indexed per slot. With
``special_tokens=['batch', 'assay']``, n_batches=10, n_assays=104,
``values[:, 0]`` (batch slot) is in ``[0, 10)`` and
``values[:, 1]`` (assay slot) is in ``[10, 114)`` -- NOT
``[0, 104)``. Use ``routing_offset`` to subtract the per-slot
base offset so the embedding lookup index is 0-indexed by the
protocol, or alternatively set ``n_experts`` large enough to
cover the full global range (e.g. ``n_special_values`` from your
data config).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProtocolBias(nn.Module):
    """Per-protocol additive bias for predictor output.

    Args
    ----
    n_experts:
        Number of distinct protocol IDs (must be >= max(routing_index)
        observed at training time). Out-of-range routing values
        raise a hard error at forward time.
    embed_dim:
        Dimension of the predictor output the bias is added to.

    The single ``nn.Embedding(n_experts, embed_dim)`` is
    zero-initialized so ``protocol_moe.enabled = True`` is a no-op
    at step 0. The embedding is marked ``_no_weight_decay`` so the
    optimizer's L2 doesn't drag it off zero before any gradient
    has flowed.
    """

    def __init__(self, n_experts: int, embed_dim: int):
        super().__init__()
        self.n_experts = int(n_experts)
        self.bias = nn.Embedding(self.n_experts, embed_dim)
        nn.init.zeros_(self.bias.weight)
        for p in self.bias.parameters():
            setattr(p, "_no_weight_decay", True)

    def forward(self, protocol_idx: torch.Tensor) -> torch.Tensor:
        """Look up the per-protocol bias.

        Args
        ----
        protocol_idx:
            ``(B,)`` LongTensor of protocol IDs in ``[0, n_experts)``.

        Returns
        -------
        ``(B, embed_dim)`` bias tensor, suitable for broadcasting
        across the sequence dimension when adding to predictor
        output.
        """
        if protocol_idx.dim() != 1:
            raise RuntimeError(
                "ProtocolBias expected (B,) protocol indices; got "
                f"shape {tuple(protocol_idx.shape)}.")
        with torch.no_grad():
            max_obs = int(protocol_idx.max().item())
            min_obs = int(protocol_idx.min().item())
        if max_obs >= self.n_experts or min_obs < 0:
            raise RuntimeError(
                "ProtocolBias routing index out of range: observed "
                f"[{min_obs}, {max_obs}] but n_experts="
                f"{self.n_experts}.\n"
                "Most common cause: ``values[:, routing_index]`` is "
                "the offset-subtracted spv_*_<id> token ID, which "
                "lives in a GLOBAL range across ALL spv_* slots, "
                "NOT 0-indexed per slot. For example, with "
                "special_tokens=['batch', 'assay'], n_batches=10, "
                "n_assays=104: values[:, 1] (assay) is in "
                "[10, 114), not [0, 104).\n"
                "Two fixes:\n"
                "  (1) Set batch_correction.protocol_moe."
                f"routing_offset = {min_obs} so the lookup index "
                "becomes values[:, routing_index] - routing_offset "
                "and falls in [0, n_experts). Recommended -- saves "
                "memory and keeps embeddings semantically aligned.\n"
                f"  (2) Increase n_experts to at least "
                f"{max_obs + 1} (a safe upper bound is the total "
                "n_special_values from your data config). Wastes "
                "unused embedding rows but works.\n"
                "If min_obs is negative, your routing_offset is too "
                "large -- decrease it.")
        return self.bias(protocol_idx)


def extract_protocol_index(batch: dict,
                           routing_index: int,
                           routing_offset: int = 0,
                           ) -> torch.Tensor:
    """Pull the per-cell protocol ID from ``batch['values']``.

    Reads ``batch['values'][:, routing_index]``, casts to long, and
    subtracts ``routing_offset`` so the returned value can index
    a ``[0, n_experts)`` embedding directly.

    By the tokenizer convention, ``values[:, routing_index]``
    holds the offset-subtracted spv_*_<id> token ID for that
    metadata column. This is **NOT** 0-indexed per slot -- it
    lives in a global range across all spv_* slots. Pass
    ``routing_offset`` equal to the base of the desired slot's
    range to recover a 0-indexed protocol id. See module docstring
    for an example.
    """
    values = batch['values']
    if values.dim() < 2:
        raise RuntimeError(
            "extract_protocol_index expected `values` to be at "
            f"least 2-D (B, L); got shape {tuple(values.shape)}.")
    L = values.size(1)
    if not (0 <= routing_index < L):
        raise RuntimeError(
            f"protocol_moe.routing_index={routing_index} out of "
            f"bounds for values of length {L}.")
    raw = values[:, routing_index].long()
    if routing_offset:
        raw = raw - int(routing_offset)
    return raw
