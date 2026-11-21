"""
Compact storage of cell-graph tokenized datasets.

The padded layout (``storage_format='padded'``, the default) stores every row
of a ``CellGraphTokenizer`` dataset at ``model_input_size`` tokens: segment
``s`` (the cell, then its neighbors nearest-first) occupies
``[s * seg_len, (s + 1) * seg_len)`` with ``seg_len = int(model_input_size /
n_segments)``, its genes at the start and padding (token 0, value 0.0) after
them, and the row ends with padding up to ``model_input_size``. Gene tokens are
stored as int64 and values, relative coordinates as float64.

The compact layout (``storage_format='compact'``) stores the same rows without
the padding and in the dtypes the model reads:

- ``gene_tokens``: the stored entries of each segment, i.e. its tokens up to
  its last non-padding entry, segments concatenated, as int16 (int32 or int64
  for token dictionaries with larger ids, see ``token_dtype``);
- ``gene_expr``: their values, as float32;
- ``segment_lengths``: the number of stored entries of each segment;
- ``model_input_size``: the padded row length, from which the segment length
  follows as in the tokenizer;
- ``rel_x_coord``, ``rel_y_coord`` as float32, ``n_nonzero_tokens`` and the
  special token values as int32.

``CellGraphDataset`` rebuilds the padded ``gene_tokens`` / ``gene_expr`` of a
compact row before anything else reads it (``expand_compact_item``), so the
model inputs equal those of the padded layout: the torch format of a padded
dataset returns int64 tokens and float32 values and coordinates
(``datasets.formatting.TorchFormatter``), which the compact columns hold
exactly. ``to_padded`` converts a compact dataset back to the padded layout.

The lengths that the length-grouped sampler groups cells by are derived from
``segment_lengths`` for the layout a model reads (``truncate_neighbors`` /
``tokenized_seq_len_cell``, e.g. 10 x 256 of a 16 x 512 corpus) by
``layout_n_nonzero_tokens``, so compact datasets need no precomputed
n_nonzero_tokens list per layout; a list that is given is checked against
them (``check_n_nonzero_tokens``).

Values are stored as float32, not as raw counts with log1p applied at load:
the shifted-log values are numpy's float32 log1p of the counts
(``normalize_by_shifted_log``), whose implementation depends on the CPU numpy
runs on (SIMD dispatch), so values recomputed on another machine at load time
are not guaranteed to be the bits the tokenizer stored.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

STORAGE_FORMATS = ('padded', 'compact')
# Columns that only the compact layout has
COMPACT_COLUMNS = ('segment_lengths', 'model_input_size')
SPECIAL_TOKEN_NAMES = ('batch', 'gene_panel', 'assay', 'species', 'tissue')


def is_compact(dataset_or_features) -> bool:
    """`True` for a compact cell-graph dataset (its features, or a row)."""
    features = getattr(dataset_or_features, 'features', dataset_or_features)
    try:
        return 'segment_lengths' in features
    except TypeError:
        return False


def token_dtype(token_dict: dict) -> np.dtype:
    """
    Narrowest signed integer dtype (int16, else int32, int64) that holds
    every id of the token dictionary, and 0 (padding).

    The cell-graph tokenizer writes only gene token ids and padding into
    `gene_tokens`, but the dtype covers every id of the dictionary (special,
    CLS and special value tokens too), so it does not depend on which tokens a
    row holds. It depends on the token dictionary only, so all parts of a
    corpus tokenized with one dictionary get the same schema and can be
    concatenated. `compact_row` checks every row against it.

    Unsigned dtypes are not used: torch < 2.3 has no uint16, and whether the
    torch format of `datasets` converts uint16 columns depends on its version
    (TERRA allows torch >= 2.1, datasets >= 2.18).
    """
    ids = [int(v) for v in token_dict.values()] + [0]
    lo, hi = min(ids), max(ids)
    for dtype in (np.int16, np.int32):
        info = np.iinfo(dtype)
        if info.min <= lo and hi <= info.max:
            return np.dtype(dtype)
    return np.dtype(np.int64)


def segment_length_dtype(model_input_size: int) -> np.dtype:
    """
    Dtype of `segment_lengths` (a segment holds at most model_input_size);
    signed, as `token_dtype`.
    """
    return np.dtype(np.int16 if model_input_size <= np.iinfo(np.int16).max
                    else np.int32)


def _list_feature(feature):
    """
    List feature of `feature` in every supported `datasets` version
    (`datasets.List` exists from 4.0; `Sequence` returns a `List` there).
    """
    from datasets import Sequence
    return Sequence(feature)


def compact_features(tok_dtype: np.dtype,
                     model_input_size: int,
                     include_special_tokens: bool,
                     add_neigh_cell_ids: bool):
    """Hugging Face features of a compact cell-graph dataset."""
    from datasets import Features, Value
    List = _list_feature
    features = {
        'rel_x_coord': List(Value('float32')),
        'rel_y_coord': List(Value('float32')),
    }
    if add_neigh_cell_ids:
        features['cell_ids'] = List(Value('string'))
    features.update({
        'cell_id': Value('string'),
        'gene_tokens': List(Value(np.dtype(tok_dtype).name)),
        'gene_expr': List(Value('float32')),
        'n_nonzero_tokens': Value('int32'),
        'segment_lengths': List(Value(
            segment_length_dtype(model_input_size).name)),
        'model_input_size': Value('int32'),
    })
    if include_special_tokens:
        for name in ('assay', 'species', 'tissue', 'gene_panel', 'batch'):
            features[f'{name}_value'] = List(Value('int32'))
    return Features(features)


def compact_row(gene_tokens: np.ndarray,
                gene_expr: np.ndarray,
                n_segments: int,
                model_input_size: int,
                tok_dtype: np.dtype,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compact a padded cell-graph row.

    Parameters
    ----------
    gene_tokens, gene_expr:
        The padded row (`model_input_size` entries) as the tokenizer formats
        it.
    n_segments:
        Number of segments of the row (`n_gene_segments` of the tokenizer).
    model_input_size:
        Padded row length.
    tok_dtype:
        Dtype of the stored tokens (`token_dtype`).

    Returns
    ----------
    tokens, values, segment_lengths:
        Stored entries (tokens in `tok_dtype`, values in float32) and the
        number of stored entries per segment.
    """
    gene_tokens = np.asarray(gene_tokens)
    gene_expr = np.asarray(gene_expr, dtype=np.float64)
    if len(gene_tokens) != model_input_size or len(gene_expr) != model_input_size:
        raise ValueError(
            f'Padded row of {len(gene_tokens)} tokens / {len(gene_expr)} values, '
            f'expected {model_input_size}.')
    seg_len = int(model_input_size / n_segments)
    n_body = n_segments * seg_len
    # An entry is stored unless both its token and its value bits are 0 (a
    # value of -0.0 or NaN counts as stored, so the rebuilt row is the same)
    stored = (gene_tokens != 0) | (gene_expr.view(np.uint64) != 0)
    if stored[n_body:].any():
        raise ValueError('Entries after the last segment of a padded row.')
    stored = stored[:n_body].reshape(n_segments, seg_len)
    if seg_len > 0:
        # Each segment keeps its entries up to its last stored one (padding
        # between genes, e.g. masked genes on the dense ranking path, is kept)
        has_any = stored.any(axis=1)
        last = seg_len - np.argmax(stored[:, ::-1], axis=1)
        lengths = np.where(has_any, last, 0)
    else:
        lengths = np.zeros(n_segments, dtype=np.int64)
    keep = np.arange(seg_len)[None, :] < lengths[:, None]
    tokens = gene_tokens[:n_body].reshape(n_segments, seg_len)[keep]
    values = gene_expr[:n_body].reshape(n_segments, seg_len)[keep]
    info = np.iinfo(tok_dtype)
    if tokens.size and (tokens.min() < info.min or tokens.max() > info.max):
        raise ValueError(f'Token ids out of the range of {np.dtype(tok_dtype)}.')
    return (tokens.astype(tok_dtype),
            values.astype(np.float32),
            lengths.astype(segment_length_dtype(model_input_size)))


def _stored_positions(seg_lens: np.ndarray, mis: int) -> np.ndarray:
    """
    Positions in the padded row (`mis` entries) of the stored entries of a
    compact row with segment lengths `seg_lens`.
    """
    n_seg = len(seg_lens)
    seg_len = int(mis / n_seg)
    if n_seg and (seg_lens.min() < 0 or seg_lens.max() > seg_len):
        raise ValueError(
            f'Segment lengths {seg_lens.tolist()} outside [0, {seg_len}] for a '
            f'padded row of {mis} entries in {n_seg} segments.')
    # Entry j of segment s goes to s * seg_len + j
    shift = np.arange(n_seg, dtype=np.int64) * seg_len - (
        np.cumsum(seg_lens) - seg_lens)
    return np.arange(int(seg_lens.sum()), dtype=np.int64) + np.repeat(
        shift, seg_lens)


def expand_compact_item(item: dict) -> dict:
    """
    Rebuild the padded `gene_tokens` (int64) and `gene_expr` (float32) tensors
    of a compact row in place, as the torch format returns them for the padded
    layout, and remove the compact-only keys.

    Works on numpy (a few microseconds per op) rather than torch ops (several
    microseconds of dispatch each): it runs once per item in every DataLoader
    worker.
    """
    import torch
    seg_lens = np.asarray(item.pop('segment_lengths'), dtype=np.int64)
    mis = int(item.pop('model_input_size'))
    pos = _stored_positions(seg_lens, mis)
    stored_tokens = np.asarray(item['gene_tokens'])
    stored_values = (np.asarray(item['gene_expr'], dtype=np.float32)
                     if 'gene_expr' in item else None)
    # Checked explicitly: numpy would broadcast a single stored entry
    for name, stored in (('gene_tokens', stored_tokens),
                         ('gene_expr', stored_values)):
        if stored is not None and stored.shape != pos.shape:
            raise ValueError(
                f'Compact row with {stored.size} {name} entries, but its '
                f'segment lengths {seg_lens.tolist()} give {pos.size}.')
    tokens = np.zeros(mis, dtype=np.int64)
    tokens[pos] = stored_tokens
    item['gene_tokens'] = torch.from_numpy(tokens)
    if stored_values is not None:
        values = np.zeros(mis, dtype=np.float32)
        values[pos] = stored_values
        item['gene_expr'] = torch.from_numpy(values)
    return item


def _underlying_table(dataset):
    """The Arrow table of all rows of `dataset` (without its indices mapping)."""
    table = dataset.data
    return getattr(table, 'table', table)


def _selected_rows(dataset) -> np.ndarray | None:
    """Rows of the underlying table that `dataset` selects (None for all)."""
    indices = getattr(dataset, '_indices', None)
    if indices is None:
        return None
    column = indices.column(0)
    return np.asarray(column.to_numpy(), dtype=np.int64)


def _layout_slicing(stride: int, seq_len_cell: int, n_segments: int,
                    truncate_neighbors: bool, mis: np.ndarray):
    """
    Per-row layout parameters as `CellGraphDataset.__getitem__` computes them
    from the padded row length `mis`: the number of kept segments, and whether
    the row is re-sliced at all. Raises the errors `__getitem__` would raise.
    """
    if (mis % stride).any():
        raise ValueError(
            f"stored gene_tokens length {int(mis[(mis % stride) != 0][0])} is "
            f"not a multiple of tokenized_seq_len_cell {stride}; set "
            "tokenized_seq_len_cell to the dataset's per-segment token count.")
    if seq_len_cell > stride:
        raise ValueError(
            f"seq_len_cell {seq_len_cell} exceeds the dataset's "
            f"tokenized_seq_len_cell {stride}; cannot keep more genes per cell "
            "than were tokenized.")
    n_seg_data = mis // stride
    keep_seg = (np.full_like(n_seg_data, int(n_segments))
                if truncate_neighbors else n_seg_data)
    if (keep_seg > n_seg_data).any():
        raise ValueError(
            f"n_segments {int(n_segments)} exceeds the dataset's "
            f"{int(n_seg_data.min())} segments; the dataset has fewer "
            "neighbours than requested.")
    resliced = (keep_seg != n_seg_data) | (seq_len_cell != stride)
    return keep_seg, resliced


def _expanded_tokens(tokens: np.ndarray, seg_lens: np.ndarray,
                     mis: int) -> np.ndarray:
    """Padded `gene_tokens` of one compact row (numpy, as the tokenizer)."""
    out = np.zeros(mis, dtype=np.int64)
    out[_stored_positions(np.asarray(seg_lens, dtype=np.int64), mis)] = tokens
    return out


def _layout_counts_of_table(part, full_table, abs_rows: np.ndarray,
                            stride: int, seq_len_cell: int, n_segments: int,
                            truncate_neighbors: bool) -> np.ndarray:
    """
    `layout_n_nonzero_tokens` of the rows of `part` (an Arrow table with the
    columns `segment_lengths`, `n_nonzero_tokens` and `model_input_size`),
    which are the rows `abs_rows` of `full_table` (read for the
    `gene_tokens` of rows counted on their rebuilt padded row).
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    seg = part.column('segment_lengths')
    seg = seg.combine_chunks() if isinstance(seg, pa.ChunkedArray) else seg
    n_seg_row = pc.list_value_length(seg).to_numpy(
        zero_copy_only=False).astype(np.int64)
    seg_values = pc.list_flatten(seg).to_numpy(
        zero_copy_only=False).astype(np.int64)
    nnz = np.asarray(part.column('n_nonzero_tokens').to_numpy(),
                     dtype=np.int64)
    mis = np.asarray(part.column('model_input_size').to_numpy(),
                     dtype=np.int64)
    n = len(nnz)
    keep_seg, resliced = _layout_slicing(
        stride, seq_len_cell, n_segments, truncate_neighbors, mis)

    # Rows whose segments are `stride` long in the padded row (every row of a
    # corpus with n_neighs neighbours per cell) and whose stored entries are
    # all gene tokens (stored entries == n_nonzero_tokens; the sparse ranking
    # path): segment s of the layout is segment s of the row, so it holds
    # min(stored entries, seq_len_cell) gene tokens.
    row_of = np.repeat(np.arange(n), n_seg_row)
    starts = np.cumsum(n_seg_row) - n_seg_row
    pos = np.arange(len(seg_values)) - np.repeat(starts, n_seg_row)
    stored = np.rint(np.bincount(row_of, weights=seg_values,
                                 minlength=n)).astype(np.int64)
    take = pos < keep_seg[row_of]
    counts = np.rint(np.bincount(
        row_of[take], weights=np.minimum(seg_values[take], seq_len_cell),
        minlength=n)).astype(np.int64)
    counts = np.where(resliced, counts, nnz)
    row_seg_len = mis // np.maximum(n_seg_row, 1)
    exact = ~resliced | ((n_seg_row > 0) & (row_seg_len == stride)
                         & (stored == nnz))

    # Other rows (fewer neighbours, padding stored between genes): count the
    # gene tokens of the rebuilt padded row in the layout's columns
    if not exact.all():
        gene_tokens = full_table.column('gene_tokens')
        for i in np.flatnonzero(~exact):
            tokens = np.asarray(
                gene_tokens.slice(int(abs_rows[i]), 1).to_pylist()[0])
            lens = seg_values[starts[i]:starts[i] + n_seg_row[i]]
            padded = _expanded_tokens(tokens, lens, int(mis[i]))
            cols = np.concatenate([
                np.arange(s * stride, s * stride + seq_len_cell)
                for s in range(int(keep_seg[i]))])
            counts[i] = int(np.count_nonzero(padded[cols]))
    return counts


def layout_n_nonzero_tokens(dataset,
                            seq_len_cell: int,
                            n_segments: int,
                            tokenized_seq_len_cell: int | None = None,
                            truncate_neighbors: bool = False,
                            rows: np.ndarray | None = None,
                            group_rows: int = 262_144,
                            ) -> np.ndarray:
    """
    Number of gene tokens of each row of a compact dataset in the layout that
    `CellGraphDataset` reads (`truncate_neighbors` / `tokenized_seq_len_cell`).

    It equals the number of non-zero tokens in the columns of the padded row
    that `CellGraphDataset.__getitem__` keeps (what a per-layout
    n_nonzero_tokens list counts on a padded dataset), and the dataset's
    `n_nonzero_tokens` when the layout does not re-slice the rows. It is
    computed from `segment_lengths` (and `n_nonzero_tokens`), so no
    per-layout list is needed; rows whose segments are not
    `tokenized_seq_len_cell` long, or that store padding between genes, are
    counted on their rebuilt padded row.

    Parameters
    ----------
    dataset:
        Compact cell-graph Hugging Face dataset (with or without an indices
        mapping, e.g. after `select`).
    seq_len_cell, n_segments, tokenized_seq_len_cell, truncate_neighbors:
        The cell dataset's layout settings (`CellBaseDataset`).
    rows:
        Rows of `dataset` to count (default: all, in the dataset's order).
    group_rows:
        Rows read at a time (bounds the memory used).

    Returns
    ----------
    lengths:
        int32 array with one entry per (selected) row of `dataset` (a numpy
        array, not a list of Python ints: 4 bytes per cell in the training
        process and in each DataLoader worker the cell dataset is sent to).
    """
    if not is_compact(dataset):
        raise ValueError('layout_n_nonzero_tokens needs a compact dataset.')
    import pyarrow as pa

    full_table = _underlying_table(dataset)
    selected = _selected_rows(dataset)
    if rows is not None:
        rows = np.arange(len(dataset))[np.asarray(rows, dtype=np.int64)]
        selected = rows if selected is None else selected[rows]
    stride = tokenized_seq_len_cell or seq_len_cell
    if not (truncate_neighbors or stride != seq_len_cell):
        # The rows are read as stored (no re-slicing, CellGraphDataset):
        # their n_nonzero_tokens
        nnz = np.asarray(full_table.column('n_nonzero_tokens').to_numpy(),
                         dtype=np.int32)
        return nnz if selected is None else nnz[selected]
    meta = full_table.select(
        ['segment_lengths', 'n_nonzero_tokens', 'model_input_size'])
    kwargs = dict(stride=stride, seq_len_cell=seq_len_cell,
                  n_segments=n_segments,
                  truncate_neighbors=truncate_neighbors)
    if selected is not None and len(selected) <= 100_000:
        # Few rows: read only those (one-row slices are zero-copy)
        if len(selected) == 0:
            return np.zeros(0, dtype=np.int32)
        part = pa.concat_tables(
            [meta.slice(int(r), 1) for r in selected]).combine_chunks()
        return _layout_counts_of_table(
            part, full_table, selected, **kwargs).astype(np.int32)
    # Every row of the table, a group of rows at a time, then the selection
    counts = np.empty(meta.num_rows, dtype=np.int32)
    for start in range(0, meta.num_rows, group_rows):
        part = meta.slice(start, group_rows)
        counts[start:start + part.num_rows] = _layout_counts_of_table(
            part, full_table, np.arange(start, start + part.num_rows),
            **kwargs)
    return counts if selected is None else counts[selected]


def check_n_nonzero_tokens(lengths, dataset, n_check: int = 1000, **layout):
    """
    Raise when a precomputed n_nonzero_tokens list given for a compact dataset
    does not hold the layout's lengths (`layout_n_nonzero_tokens`) at up to
    `n_check` evenly spaced rows, e.g. a list of the full corpus rows used for
    a model that reads fewer neighbours or genes. The length-grouped sampler
    would then group cells by lengths the model does not read, and the batch
    it puts first would not be the longest (it does so to raise an
    out-of-memory error early).
    """
    if len(lengths) != len(dataset):
        raise ValueError(
            f"The n_nonzero_tokens list has {len(lengths)} entries, but the "
            f"dataset has {len(dataset)} rows.")
    if len(dataset) == 0:
        return
    rows = np.unique(np.linspace(0, len(dataset) - 1,
                                 num=min(n_check, len(dataset)),
                                 dtype=np.int64))
    expected = layout_n_nonzero_tokens(dataset, rows=rows, **layout)
    given = np.array([int(lengths[i]) for i in rows], dtype=np.int64)
    n_diff = int((given != expected).sum())
    if n_diff:
        raise ValueError(
            f"The n_nonzero_tokens list does not hold the lengths of the "
            f"layout the model reads ({layout}) at {n_diff} of {len(rows)} "
            "checked rows (e.g. it was made for the full rows of the corpus). "
            "Compact datasets derive these lengths from their segment "
            "lengths: remove data.precomputed_n_nonzero_tokens / "
            "data.precomputed_epoch_n_nonzero_tokens from the config, or "
            "rebuild the list for this layout.")


def padded_row_length(row: dict) -> int:
    """Padded `gene_tokens` length of a row of either layout."""
    if 'model_input_size' in row and 'segment_lengths' in row:
        return int(row['model_input_size'])
    return len(row['gene_tokens'])


def _expand_batch(batch: dict) -> dict:
    tokens_out, values_out = [], []
    for i in range(len(batch['gene_tokens'])):
        item = {'gene_tokens': np.asarray(batch['gene_tokens'][i]),
                'gene_expr': np.asarray(batch['gene_expr'][i], dtype=np.float32),
                'segment_lengths': np.asarray(batch['segment_lengths'][i]),
                'model_input_size': batch['model_input_size'][i]}
        expand_compact_item(item)
        tokens_out.append(item['gene_tokens'].numpy())
        values_out.append(item['gene_expr'].numpy().astype(np.float64))
    batch['gene_tokens'] = tokens_out
    batch['gene_expr'] = values_out
    return batch


def to_padded(dataset, num_proc: int | None = None,
              keep_in_memory: bool = False):
    """
    Convert a compact cell-graph dataset to the padded layout (int64 tokens,
    float64 values and coordinates, as the tokenizer writes it), keeping the
    dataset's format. The torch-formatted rows (the model inputs) equal those
    of the compact dataset; float64 values and coordinates that the tokenizer
    computed in float64 hold their float32 rounding.
    """
    from datasets import Value
    List = _list_feature
    if not is_compact(dataset):
        return dataset
    saved_format = dataset.format
    features = dataset.features.copy()
    for col in COMPACT_COLUMNS:
        del features[col]
    features['gene_tokens'] = List(Value('int64'))
    features['gene_expr'] = List(Value('float64'))
    for col in ('rel_x_coord', 'rel_y_coord'):
        if col in features:
            features[col] = List(Value('float64'))
    if 'n_nonzero_tokens' in features:
        features['n_nonzero_tokens'] = Value('int64')
    for col in list(features):
        if col.endswith('_value'):
            features[col] = List(Value('int64'))
    padded = dataset.with_format(None).map(
        _expand_batch, batched=True, batch_size=1000, num_proc=num_proc,
        remove_columns=list(COMPACT_COLUMNS), features=features,
        keep_in_memory=keep_in_memory, desc='Padding compact rows')
    if saved_format.get('type') is not None:
        columns = saved_format['columns']
        if columns is not None:
            columns = [c for c in columns if c not in COMPACT_COLUMNS]
        padded.set_format(type=saved_format['type'], columns=columns,
                          output_all_columns=saved_format['output_all_columns'],
                          **saved_format.get('format_kwargs', {}))
    return padded
