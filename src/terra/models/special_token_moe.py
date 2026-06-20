"""
Special-token-routed bias head for the JEPA predictor.

A lightweight mixture-of-experts variant: instead of routing whole
expert MLPs (scGPT-spatial style), we route per-slot bias vectors
and **sum them** as an additive correction to the predictor output.
This is a strict generalization of "predict the same thing across
all conditions"; with every bias table zero-initialized, the
``SpecialTokenMoE`` is bitwise identical to the no-MoE baseline at
step 0.

Why bias-only + sum?
--------------------
- For typical spatial transcriptomics protocol/batch effects, the
  dominant signal is an *additive* shift in feature mean per
  condition. A per-slot bias absorbs this cleanly.
- Bias terms commute: routing by batch *and* assay simultaneously
  is just two independent embeddings whose lookups are summed.
  No need for a Cartesian-product table (which would explode the
  parameter count to ``n_batches × n_assays``).
- Zero-init makes turning any slot on backward-compatible.

Routing
-------
Each routing slot is a tuple ``(routing_index, routing_offset,
n_experts)``:

- ``routing_index``: integer column in ``batch['values']`` (e.g. 0
  for batch slot, 1 for assay slot under
  ``special_tokens=['batch', 'assay']``).
- ``routing_offset``: integer subtracted from
  ``values[:, routing_index]`` so the lookup index is 0-indexed
  in ``[0, n_experts)``. The tokenizer encodes spv_* values in a
  GLOBAL offset-subtracted range across all spv_* slots, NOT per
  slot. Set ``routing_offset`` to the cumulative number of spv_*
  values that come before this slot (or set ``n_experts`` large
  enough to cover the full global range).
- ``n_experts``: size of this slot's embedding table.

The config accepts either a scalar (single-slot, legacy behavior)
or a list (multi-slot). Both are validated; mismatched list
lengths raise hard errors.

Example configs
---------------

Single-slot (assay only):

    special_token_moe:
      enabled: True
      routing_indices: 1
      n_experts: 145
      routing_offsets: 3

Multi-slot (batch + assay):

    special_token_moe:
      enabled: True
      routing_indices: [0, 1]
      n_experts: [10, 145]
      routing_offsets: [2, 3]
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _as_list(x, fallback_len: int | None = None) -> list:
    """Coerce a scalar to a one-element list, leave lists/tuples
    alone. If ``fallback_len`` is given and ``x`` is None, returns
    a list of ``[0] * fallback_len``. Used to accept config values
    as either scalars or lists.
    """
    if x is None:
        if fallback_len is None:
            return []
        return [0] * fallback_len
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


class SpecialTokenMoE(nn.Module):
    """Multi-slot zero-init bias head routed by special-token values.

    Args
    ----
    n_experts_per_slot:
        List of expert-table sizes, one per routing slot.
    embed_dim:
        Dimension of the bias vectors (must equal the predictor's
        output dim).
    """

    def __init__(self,
                 n_experts_per_slot: Sequence[int],
                 embed_dim: int):
        super().__init__()
        n_experts_per_slot = list(n_experts_per_slot)
        if len(n_experts_per_slot) == 0:
            raise ValueError(
                "SpecialTokenMoE requires at least one routing slot; "
                "got an empty n_experts list.")
        for i, n in enumerate(n_experts_per_slot):
            if int(n) <= 0:
                raise ValueError(
                    f"SpecialTokenMoE: n_experts[{i}] must be > 0; "
                    f"got {n}.")
        self.n_experts_per_slot = [int(n) for n in n_experts_per_slot]
        self.biases = nn.ModuleList([
            nn.Embedding(int(n), embed_dim)
            for n in n_experts_per_slot
        ])
        # Zero-init each table so the MoE is a step-0 no-op. Mark
        # parameters as `_no_weight_decay` so the optimizer's L2
        # group doesn't actively pull biases back toward zero.
        for emb in self.biases:
            nn.init.zeros_(emb.weight)
            for p in emb.parameters():
                setattr(p, "_no_weight_decay", True)

    def forward(self,
                indices_per_slot: Sequence[torch.Tensor],
                ) -> torch.Tensor:
        """Look up and sum the per-slot biases.

        Args
        ----
        indices_per_slot:
            List of ``(B,)`` LongTensors, one per routing slot,
            with values in ``[0, n_experts_per_slot[i])``.

        Returns
        -------
        ``(B, embed_dim)`` summed bias, suitable for broadcasting
        across the target sequence dimension when adding to the
        predictor output.
        """
        if len(indices_per_slot) != len(self.biases):
            raise RuntimeError(
                f"SpecialTokenMoE expected {len(self.biases)} routing "
                f"slots; got {len(indices_per_slot)} index tensors.")
        total = None
        for slot_i, (idx, emb) in enumerate(
                zip(indices_per_slot, self.biases)):
            if idx.dim() != 1:
                raise RuntimeError(
                    f"SpecialTokenMoE slot {slot_i}: expected "
                    f"(B,) indices; got shape {tuple(idx.shape)}.")
            n_experts = self.n_experts_per_slot[slot_i]
            with torch.no_grad():
                max_obs = int(idx.max().item())
                min_obs = int(idx.min().item())
            if max_obs >= n_experts or min_obs < 0:
                raise RuntimeError(
                    f"SpecialTokenMoE slot {slot_i} routing index out "
                    f"of range: observed [{min_obs}, {max_obs}] but "
                    f"n_experts[{slot_i}]={n_experts}.\n"
                    "Most common cause: values[:, routing_index] is "
                    "the offset-subtracted spv_*_<id> token ID, which "
                    "lives in a GLOBAL range across ALL spv_* slots, "
                    "NOT 0-indexed per slot. With "
                    "special_tokens=['batch', 'assay'], n_batches=10, "
                    "n_assays=104: values[:, 0] (batch) is in "
                    "[2, 12) (v3) or [107, 117) (v1); values[:, 1] "
                    "(assay) is in [12, 116) or [117, 221).\n"
                    "Two fixes:\n"
                    "  (1) Set "
                    f"batch_correction.special_token_moe."
                    f"routing_offsets[{slot_i}] = {min_obs} so the "
                    "lookup index becomes values[:, routing_index] - "
                    "routing_offset and falls in [0, n_experts).\n"
                    f"  (2) Increase n_experts[{slot_i}] to at least "
                    f"{max_obs + 1} (a safe upper bound is the total "
                    "n_special_values from your data config).\n"
                    "If min_obs is negative, your routing_offset is "
                    "too large -- decrease it.")
            contribution = emb(idx)              # (B, D)
            total = contribution if total is None else total + contribution
        return total


def extract_special_token_indices(
        batch: dict,
        routing_indices: Sequence[int] | None = None,
        routing_offsets: Sequence[int] | None = None,
        routing_keys: Sequence[str] | None = None,
        ) -> list[torch.Tensor]:
    """Pull per-cell routing IDs for each special-token slot.

    Two routing modes are supported, distinguished by which list is
    provided:

    - **Metadata-key mode** (preferred): pass ``routing_keys`` --
      a list of names of per-cell metadata fields in ``batch``
      (e.g. ``['batch_value', 'assay_value']``). Decoupled from
      ``special_tokens``; works regardless of what the encoder sees.

    - **Sequence-index mode** (legacy): pass ``routing_indices`` --
      a list of integer columns into ``batch['values']``. Requires
      ``special_tokens`` to contain the relevant spv_* slots.

    Exactly one of ``routing_keys`` / ``routing_indices`` should be
    set per call; if both are provided the keys take precedence
    (per-slot: if key[k] is None, falls back to indices[k]).
    ``routing_offsets[k]`` is subtracted from the raw value.

    Returns a list of ``(B,)`` LongTensors, one per routing slot.
    """
    # Determine the number of slots from whichever list is provided.
    if routing_keys is not None:
        routing_keys = list(routing_keys)
        n_slots = len(routing_keys)
    elif routing_indices is not None:
        n_slots = len(list(routing_indices))
        routing_keys = [None] * n_slots
    else:
        raise RuntimeError(
            "extract_special_token_indices: must provide "
            "routing_keys (recommended) or routing_indices.")
    if routing_indices is None:
        routing_indices = [None] * n_slots
    else:
        routing_indices = list(routing_indices)
    if routing_offsets is None:
        routing_offsets = [0] * n_slots
    else:
        routing_offsets = list(routing_offsets)
    if len(routing_offsets) != n_slots:
        raise RuntimeError(
            f"extract_special_token_indices: "
            f"len(routing_offsets)={len(routing_offsets)} does not "
            f"match number of routing slots ({n_slots}).")
    if len(routing_indices) != n_slots:
        raise RuntimeError(
            f"extract_special_token_indices: "
            f"len(routing_indices)={len(routing_indices)} does not "
            f"match number of routing slots ({n_slots}).")

    out = []
    for k, (rkey, ri, ro) in enumerate(
            zip(routing_keys, routing_indices, routing_offsets)):
        if rkey is not None and rkey in batch:
            raw = batch[rkey].long()
        else:
            values = batch['values']
            if values.dim() < 2:
                raise RuntimeError(
                    "extract_special_token_indices fallback expected "
                    "`values` to be at least 2-D (B, L); got "
                    f"shape {tuple(values.shape)}.")
            if ri is None:
                raise RuntimeError(
                    f"special_token_moe slot {k}: routing_keys[{k}]="
                    f"{rkey!r} is not in batch and "
                    f"routing_indices[{k}] is None. Set either a "
                    "metadata key that the dataset exposes, or a "
                    "values-column integer index.")
            L = values.size(1)
            if not (0 <= int(ri) < L):
                raise RuntimeError(
                    f"special_token_moe.routing_indices[{k}]={ri} "
                    f"out of bounds for values of length {L}.")
            raw = values[:, int(ri)].long()
        if int(ro):
            raw = raw - int(ro)
        out.append(raw)
    return out
