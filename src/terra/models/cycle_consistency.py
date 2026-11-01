"""
Batch-swap cycle-consistency loss for batch-invariant encoder
representations.

The idea (FADVI-style, Chen et al. 2024): if the encoder has truly
learned batch-invariant features, then swapping a cell's batch label
to a random other batch should leave the encoded representation
roughly unchanged. We enforce this by re-encoding the same cells
with random batch labels and minimizing the MSE between the original
and re-encoded outputs.

This is a strictly auxiliary loss applied after the main encoder
forward; it does NOT replace the JEPA objective. The double-forward
cost is gated by ``swap_fraction``: with probability ``swap_fraction``
per step, the swapped forward pass runs and contributes to the loss.
Otherwise the step is identical to a normal training step.

Implementation notes
--------------------
- The batch label lives in ``batch['values'][:, 0]`` by the
  tokenizer convention. Swapping it changes both:
  (a) the spv_batch special-value embedding at position 0, and
  (b) the AdaLN conditioning (if AdaLN is enabled), since AdaLN
  reads its cond from the same slot.
  Both are desired -- we want to test "what would the encoder
  output if you told it this cell was from a different batch."
- We swap *uniformly at random* over ``[0, n_batches)`` so the swap
  is symmetric and unbiased. Cells whose original label happens to
  equal the swap target are left as-is for that step (no MSE
  contribution, since the forward would be identical).
- With metadata keys (``keys=['batch_value', ...]``) the label is
  read from the per-cell metadata field, which only AdaLN reads; the
  encoder sees it through the cell's special token. The swapped label
  is therefore also written into that special token (see
  ``special_token_positions``), otherwise the swap does not change the
  encoder input and the loss is zero.
"""

from __future__ import annotations

from typing import Sequence

import torch


def special_token_positions(
        keys: Sequence[str | None],
        special_tokens: Sequence[str],
        ) -> list[int | None]:
    """Sequence position of the special token that carries the label of
    each metadata key.

    The dataset prepends the entries of ``special_tokens`` one after the
    other, so they appear in reverse order at the start of the sequence
    (the last one at position 0); the token ``<name>`` carries the label
    stored in the metadata field ``<name>_value``.

    Parameters
    ----------
    keys:
        Metadata keys (e.g. ``'batch_value'``); ``None`` for the legacy
        ``values[:, 0]`` slot.
    special_tokens:
        Special tokens of the model (``meta.special_tokens``, with any
        CLS tokens in front), in configuration order.

    Returns
    ----------
    positions:
        Per key, the position of its special token, or ``None`` if the
        key is ``None`` or has no special token in the sequence.
    """
    special_tokens = list(special_tokens)
    positions = []
    for key in keys:
        name = key[:-len('_value')] if (
            key is not None and key.endswith('_value')) else key
        positions.append(
            len(special_tokens) - 1 - special_tokens.index(name)
            if name is not None and name in special_tokens else None)
    return positions


def make_swapped_batch(
        batch: dict,
        n_classes_per_key: int | list[int],
        keys: str | list[str | None] | None = None,
        offsets: int | list[int] | None = None,
        generator: torch.Generator | None = None,
        sequence_positions: list[int | None] | None = None,
        label_in_tokens: bool = False,
        ) -> tuple[dict, torch.Tensor]:
    """Return a shallow-copied batch dict with one or more
    batch-label slots replaced by random labels.

    Multi-key semantics: when multiple ``keys`` are given, ALL
    keys are swapped simultaneously per cell. ``changed_mask[i]``
    is True iff at least one of the cell's labels actually
    changed -- the cycle-consistency loss then only averages over
    those cells where the swap was non-trivial.

    Args
    ----
    batch:
        Original batch dict. Not mutated.
    n_classes_per_key:
        Per-key upper bound (exclusive) on swapped labels.
        Scalar for single-key, list for multi-key.
    keys:
        Per-key metadata field name. ``None`` (or scalar None)
        triggers the legacy path: swap ``values[:, 0]``.
    offsets:
        Per-key offset applied when reading + writing the label,
        so the stored encoding round-trips unchanged.
    generator:
        Optional torch.Generator for reproducibility.
    sequence_positions:
        Per key, the sequence position of the special token that
        carries the same label (``special_token_positions()``), or
        ``None``. A swapped metadata label is also written there, so
        that the encoder input changes; padded special tokens (token 0)
        stay padded. ``None`` (default): only the metadata fields are
        swapped.
    label_in_tokens:
        If ``True`` (``gt_type: rank``), the special token carries its
        label in its token id (label plus a fixed offset) instead of in
        ``values``.

    Returns
    -------
    swapped_batch:
        Shallow copy of ``batch`` with the relevant tensors cloned
        and the per-key slots replaced.
    changed_mask:
        Bool tensor ``(B,)`` -- True for cells where *any* of the
        swapped keys differs from the original.
    """
    # Normalize to lists. Scalars are broadcast / wrapped explicitly;
    # mismatched lengths raise so silent bugs don't sneak through.
    if isinstance(n_classes_per_key, int):
        n_classes_per_key = [n_classes_per_key]
    n_classes_per_key = list(n_classes_per_key)
    n_slots = len(n_classes_per_key)

    if keys is None:
        keys = [None] * n_slots
    elif isinstance(keys, str):
        # Broadcast a single string to all slots.
        keys = [keys] * n_slots
    else:
        keys = list(keys)
    if len(keys) != n_slots:
        raise RuntimeError(
            f"make_swapped_batch: keys length {len(keys)} does not "
            f"match n_classes_per_key length {n_slots}.")

    if offsets is None:
        offsets = [0] * n_slots
    elif isinstance(offsets, int):
        offsets = [offsets] * n_slots
    else:
        offsets = list(offsets)
    if len(offsets) != n_slots:
        raise RuntimeError(
            f"make_swapped_batch: offsets length {len(offsets)} "
            f"does not match n_classes_per_key length {n_slots}.")

    # Each slot must have a UNIQUE destination, otherwise later
    # iterations silently overwrite earlier ones while
    # ``changed_mask`` would still claim earlier slots changed -- a
    # silently incoherent cycle loss. Two failure modes are
    # forbidden:
    #   (a) ≥ 2 slots fall back to the legacy values[:, 0] path
    #       (key is None or missing from batch), or
    #   (b) ≥ 2 slots reference the SAME metadata key (duplicate
    #       writes to swapped_batch[key]).
    legacy_slot_count = sum(
        1 for k in keys if k is None or k not in batch)
    if legacy_slot_count > 1:
        raise RuntimeError(
            "make_swapped_batch: multiple routing slots are falling "
            "back to the legacy values[:, 0] path "
            f"({legacy_slot_count} of {n_slots}). Each slot must "
            "write to a distinct destination -- supply a unique "
            "metadata key per slot (and make sure each is present in "
            f"the batch dict). keys={keys}, "
            f"available batch fields={sorted(batch.keys())}.")
    real_keys = [k for k in keys if k is not None and k in batch]
    if len(set(real_keys)) != len(real_keys):
        dupes = [k for k in set(real_keys) if real_keys.count(k) > 1]
        raise RuntimeError(
            "make_swapped_batch: duplicate metadata keys across "
            "slots would cause later swaps to overwrite earlier "
            f"ones. Duplicate(s): {dupes}. keys={keys}.")

    if sequence_positions is None:
        sequence_positions = [None] * n_slots
    sequence_positions = list(sequence_positions)
    if len(sequence_positions) != n_slots:
        raise RuntimeError(
            f"make_swapped_batch: sequence_positions length "
            f"{len(sequence_positions)} does not match n_classes_per_key "
            f"length {n_slots}.")
    # Only metadata-key slots write into the sequence (the legacy slot
    # writes values[:, 0]); each position must be written by one slot
    written = [p for k, p in zip(keys, sequence_positions)
               if p is not None and k is not None and k in batch]
    if legacy_slot_count and not label_in_tokens:
        written.append(0)
    if len(set(written)) != len(written):
        raise RuntimeError(
            "make_swapped_batch: two slots would write the same special "
            f"token. keys={keys}, sequence_positions={sequence_positions}.")

    swapped_batch = dict(batch)
    # Track per-cell "did ANY key change" using bitwise OR.
    combined_changed = None

    for n_cls, key, offset, position in zip(
            n_classes_per_key, keys, offsets, sequence_positions):
        if key is not None and key in batch:
            # Metadata-key path.
            raw = batch[key]
            B = raw.size(0)
            device = raw.device
            original = raw.long() - int(offset)
            swap = torch.randint(
                low=0, high=int(n_cls), size=(B,),
                device=device, dtype=torch.long, generator=generator)
            changed = swap != original
            new_raw = (swap + int(offset)).to(raw.dtype)
            swapped_batch[key] = new_raw
            if position is not None:
                # The encoder sees the label through the special token:
                # its value, or for rank models its token id (label plus
                # a fixed offset). Padded special tokens stay padded.
                field = 'tokens' if label_in_tokens else 'values'
                seq = swapped_batch[field]
                if seq is batch[field]:
                    seq = seq.clone()
                if label_in_tokens:
                    new_label = seq[:, position] + (swap - original).to(
                        seq.dtype)
                else:
                    new_label = (swap + int(offset)).to(seq.dtype)
                present = batch['tokens'][:, position] != 0
                seq[:, position] = torch.where(
                    present, new_label, seq[:, position])
                swapped_batch[field] = seq
        else:
            # Legacy path: swap values[:, 0]. Guarded above so this
            # branch can only fire for at most one slot per call.
            values = batch.get('values')
            if values is None or values.dim() < 2:
                raise RuntimeError(
                    "make_swapped_batch legacy path expected `values` "
                    "to be at least 2-D (B, L).")
            B = values.size(0)
            device = values.device
            original = values[:, 0].long() - int(offset)
            swap = torch.randint(
                low=0, high=int(n_cls), size=(B,),
                device=device, dtype=torch.long, generator=generator)
            changed = swap != original
            new_values = swapped_batch['values'].clone()
            new_values[:, 0] = (swap + int(offset)).to(values.dtype)
            swapped_batch['values'] = new_values

        combined_changed = (
            changed if combined_changed is None
            else combined_changed | changed)

    return swapped_batch, combined_changed


def cycle_consistency_loss(
        z_original: list[torch.Tensor] | torch.Tensor,
        z_swapped: list[torch.Tensor] | torch.Tensor,
        changed_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
    """MSE between original and batch-swapped encoder outputs.

    Accepts either a single tensor or a list of tensors (one per
    context mask, matching the JEPA multi-mask layout used by
    ``EncoderMultiMaskWrapper``).

    If ``changed_mask`` is provided, only cells whose label actually
    changed contribute to the loss (returns zero when none changed).
    """
    if isinstance(z_original, list):
        return torch.stack([
            cycle_consistency_loss(zo, zs, changed_mask)
            for zo, zs in zip(z_original, z_swapped)
        ]).mean()

    diff_sq = (z_original - z_swapped) ** 2          # (B, L, D)
    if changed_mask is not None:
        if changed_mask.sum() == 0:
            return torch.zeros((), device=diff_sq.device, dtype=diff_sq.dtype)
        # Reduce over (L, D) per cell, then mean over changed cells.
        per_cell = diff_sq.mean(dim=tuple(range(1, diff_sq.dim())))
        per_cell = per_cell[changed_mask]
        return per_cell.mean()
    return diff_sq.mean()
