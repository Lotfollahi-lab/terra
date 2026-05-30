"""
Unified batch-label extraction and config-spec resolution for all
batch_correction / special_token_correction mechanisms.

Each mechanism (AdaLN, adv_classifier, distribution_alignment,
cycle_consistency, special_token_moe) accepts the same uniform
multi-key spec:

    keys:       list of per-cell metadata-field names (or scalar string)
    n_classes:  list of expert-table sizes per key (or scalar int)
    offsets:    list of offsets to subtract per key (or scalar int)

When a single key is sufficient, scalars / single-element lists work
identically. Multi-key mechanisms (AdaLN sums embeddings; adv has
one head per key; dist_align computes per-key loss summed; cycle
swaps all keys at once; MoE sums per-key biases).

Legacy alias for backward-compat: each mechanism also accepts the
older per-mechanism names (``batch_label_key``, ``n_batches``,
``routing_keys``, ``routing_indices``, etc.), which are normalized
to the unified spec at parse time.

Resolution in ``extract_batch_label`` (single-key):
  1. If ``key`` is given AND is a top-level field in ``batch``,
     read ``batch[key]`` (metadata path).
  2. Otherwise fall back to ``batch['values'][:, 0]`` (legacy path).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


def _as_list(x, fallback_len: int | None = None) -> list:
    """Coerce a scalar to a one-element list. Leave lists/tuples
    alone. ``None`` returns ``[]`` (or ``[0] * fallback_len`` when
    fallback_len is provided)."""
    if x is None:
        if fallback_len is None:
            return []
        return [0] * fallback_len
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def resolve_label_spec(
        cfg: Mapping | None,
        legacy_key_names: Sequence[str] = (
            'keys', 'key',
            'batch_label_keys', 'batch_label_key',
            'routing_keys', 'routing_key',
        ),
        legacy_n_classes_names: Sequence[str] = (
            'n_classes', 'n_class',
            'n_batches', 'n_batch',
            'n_experts', 'n_expert',
        ),
        legacy_offset_names: Sequence[str] = (
            'offsets', 'offset',
            'batch_label_offsets', 'batch_label_offset',
            'routing_offsets', 'routing_offset',
        ),
        shared: Mapping | None = None,
        ) -> dict:
    """Resolve ``(keys, n_classes, offsets)`` lists from a mechanism
    config, with fallback to shared spec.

    Returns a dict with three lists:
      ``{'keys': list[str|None], 'n_classes': list[int],
         'offsets': list[int]}``

    The three lists always have the same length. If ``keys`` is empty
    after resolution, mechanisms should treat that as "use the
    legacy ``values[:, 0]`` path with a single slot."
    """
    cfg = dict(cfg) if cfg else {}
    shared = dict(shared) if shared else {}

    def _pick(names):
        # First look in cfg, then in shared.
        for name in names:
            if name in cfg and cfg[name] is not None:
                return cfg[name]
        for name in names:
            if name in shared and shared[name] is not None:
                return shared[name]
        return None

    keys_raw = _pick(legacy_key_names)
    ncls_raw = _pick(legacy_n_classes_names)
    offs_raw = _pick(legacy_offset_names)

    keys = _as_list(keys_raw)
    ncls = _as_list(ncls_raw)
    if keys:
        offs = _as_list(offs_raw, fallback_len=len(keys))
    elif ncls:
        offs = _as_list(offs_raw, fallback_len=len(ncls))
    else:
        offs = _as_list(offs_raw)

    # If the lengths disagree, that's a config error worth surfacing.
    nonempty = [
        (label, lst) for label, lst in
        [('keys', keys), ('n_classes', ncls), ('offsets', offs)]
        if lst
    ]
    if nonempty:
        ref_len = len(nonempty[0][1])
        for label, lst in nonempty:
            if len(lst) != ref_len:
                raise ValueError(
                    f"Inconsistent multi-key spec lengths: "
                    f"{nonempty[0][0]} has len={ref_len} but {label} "
                    f"has len={len(lst)}. All of keys / n_classes / "
                    "offsets must have the same length when "
                    "provided.")

    # Pad single-slot lists with defaults so all three have the same length.
    target_len = max(len(keys), len(ncls), len(offs), 0)
    if target_len == 0:
        return {'keys': [], 'n_classes': [], 'offsets': []}
    if not keys:
        keys = [None] * target_len
    if not ncls:
        ncls = [0] * target_len
    if not offs:
        offs = [0] * target_len

    return {
        'keys': [None if k is None else str(k) for k in keys],
        'n_classes': [int(n) for n in ncls],
        'offsets': [int(o) for o in offs],
    }


def extract_batch_label(
        batch: Mapping[str, torch.Tensor],
        key: str | None = None,
        offset: int = 0,
        ) -> torch.Tensor:
    """Return per-cell batch label as a ``(B,)`` LongTensor (single-key).

    Two resolution paths:
      - ``key=None``: legacy path, reads ``batch['values'][:, 0]``.
        Caller has explicitly opted into this.
      - ``key='<name>'`` AND present in ``batch``: metadata path,
        reads ``batch[key]``.
      - ``key='<name>'`` AND NOT in ``batch``: this is almost always
        a misconfiguration (wrong key, dataset doesn't expose the
        column, or different schemas across DDP ranks). Raises a
        hard error rather than silently falling back to
        ``values[:, 0]`` -- a silent fallback would let different
        ranks compute losses against different label sources, which
        DDP all-reduce then averages, corrupting the gradient.
    """
    if key is None:
        values = batch['values']
        if values.dim() < 2:
            raise RuntimeError(
                "extract_batch_label fallback expected `values` to "
                f"be at least 2-D (B, L); got shape {tuple(values.shape)}.")
        raw = values[:, 0]
    elif key in batch:
        raw = batch[key]
    else:
        raise RuntimeError(
            f"extract_batch_label: requested key {key!r} is not "
            f"present in the batch dict. Available keys: "
            f"{sorted(batch.keys())}. Either set this key in the "
            "dataset's auto-exposed *_value columns, or pass "
            "key=None to use the legacy values[:, 0] fallback "
            "explicitly. Silent fallback was removed because it "
            "lets DDP ranks compute labels from inconsistent "
            "sources, which corrupts gradients after all-reduce.")
    raw = raw.long()
    if offset:
        raw = raw - int(offset)
    return raw


def extract_batch_labels(
        batch: Mapping[str, torch.Tensor],
        keys: Sequence[str | None],
        offsets: Sequence[int] | None = None,
        ) -> list[torch.Tensor]:
    """Multi-key version of ``extract_batch_label``: returns a list
    of ``(B,)`` LongTensors, one per key.

    Each ``keys[i]`` follows the same metadata-key-or-fallback
    resolution as the single-key helper.
    """
    keys = list(keys)
    if offsets is None:
        offsets = [0] * len(keys)
    offsets = list(offsets)
    if len(offsets) != len(keys):
        raise RuntimeError(
            f"extract_batch_labels: len(offsets)={len(offsets)} does "
            f"not match len(keys)={len(keys)}.")
    return [
        extract_batch_label(batch, key=k, offset=o)
        for k, o in zip(keys, offsets)
    ]
