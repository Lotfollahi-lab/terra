"""
Tests for the compact storage format of cell-graph datasets
(``storage_format='compact'``, ``terra.datasets.compact``).

A compact row must give the same model inputs as the padded row: the padded
``gene_tokens`` / ``gene_expr`` that ``CellGraphDataset`` rebuilds equal the
torch-formatted padded columns bit for bit, for every load-time layout
(``truncate_neighbors`` / ``tokenized_seq_len_cell``), and the lengths the
sampler groups cells by equal the layout's lengths of the padded rows.
"""

import pickle

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

datasets = pytest.importorskip('datasets')

from terra.datasets.cell_datasets import CellGraphDataset
from terra.datasets.compact import (COMPACT_COLUMNS, check_n_nonzero_tokens,
                                    compact_row,
                                    expand_compact_item, is_compact,
                                    layout_n_nonzero_tokens, to_padded,
                                    token_dtype)
from terra.datasets.dataloaders import CustomDistributedLengthGroupedSampler
from terra.tokenizers import CellGraphTokenizer

N_NEIGHS = 4
SEQ = 8
MIS = SEQ * (N_NEIGHS + 1)
# (seq_len_cell, n_neighs) read from the stored SEQ x N_NEIGHS layout
LAYOUTS = [(8, 4), (4, 4), (4, 2), (8, 1), (8, 2), (2, 3)]
# Tokenizer settings of the variants: the corpus settings, without padding
# segments for empty neighbours (rows with fewer, longer segments), the dense
# ranking path and PFlog1pPF values
VARIANTS = {
    'corpus': dict(pad_empty_neighbors=True,
                   rank_tie_break_norm_factor='tie_break_mean'),
    'no_pad_empty': dict(pad_empty_neighbors=False),
    'dense': dict(pad_empty_neighbors=True,
                  rank_count_norm_method='analytic_pearson_residuals',
                  zero_expr_from_raw_counts=True),
    'pflog1ppf': dict(pad_empty_neighbors=True,
                      count_count_norm_method='pflog1ppf'),
}


def _bits(t):
    t = torch.as_tensor(t)
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _assert_same(a, b):
    # same keys in the same order (dict order follows the column order)
    assert list(a) == list(b)
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
            assert torch.equal(_bits(a[k]), _bits(b[k])), k
        else:
            assert a[k] == b[k], k


@pytest.mark.parametrize('n_segments,mis', [(5, 40), (3, 40), (1, 40), (7, 40)])
def test_compact_row_round_trip(n_segments, mis):
    rng = np.random.default_rng(n_segments)
    seg_len = int(mis / n_segments)
    tokens = np.zeros(mis, dtype=np.int64)
    values = np.zeros(mis)
    for s in range(n_segments):
        n = rng.integers(0, seg_len + 1)
        tokens[s * seg_len:s * seg_len + n] = rng.integers(2, 30000, n)
        values[s * seg_len:s * seg_len + n] = np.float32(rng.random(n))
    # padding between genes (dense ranking path) and a stored -0.0 value
    tokens[1] = 0
    values[0] = -0.0
    t, v, lengths = compact_row(tokens, values, n_segments, mis, np.int16)
    assert t.dtype == np.int16 and v.dtype == np.float32
    item = {'gene_tokens': torch.as_tensor(t.astype(np.int64)),
            'gene_expr': torch.as_tensor(v),
            'segment_lengths': torch.as_tensor(lengths.astype(np.int64)),
            'model_input_size': torch.tensor(mis)}
    expand_compact_item(item)
    assert set(item) == {'gene_tokens', 'gene_expr'}
    assert torch.equal(item['gene_tokens'], torch.as_tensor(tokens))
    ref = torch.tensor(values, dtype=torch.float32)
    assert torch.equal(item['gene_expr'].view(torch.int32), ref.view(torch.int32))


@pytest.mark.parametrize('seg_lens,n_entries', [
    ([3, 6, 0, 2], 11),   # a segment longer than seg_len 5
    ([3, -1, 4, 2], 8),   # a negative length
    ([3, 2, 0, 2], 8),    # lengths for 7 entries, 8 stored
    ([3, 2, 0, 2], 1),    # one stored entry (numpy would broadcast it)
    ([0, 0, 0, 0], 1),    # no entries per the lengths, one stored
])
def test_expand_rejects_inconsistent_segment_lengths(seg_lens, n_entries):
    item = {'gene_tokens': torch.arange(1, n_entries + 1),
            'gene_expr': torch.ones(n_entries),
            'segment_lengths': torch.tensor(seg_lens),
            'model_input_size': torch.tensor(20)}
    with pytest.raises(ValueError):
        expand_compact_item(item)


def test_compact_row_rejects_tokens_out_of_range():
    tokens = np.zeros(MIS, dtype=np.int64)
    tokens[0] = 40000
    with pytest.raises(ValueError, match='out of the range'):
        compact_row(tokens, np.ones(MIS), N_NEIGHS + 1, MIS, np.int16)


def test_token_dtype():
    # every id of the dictionary counts, not only the gene tokens
    real_like = {'<pad>': 0, '<mask>': 1, 'ENSG1': 21956, 'spv_last': 23406}
    assert token_dtype(real_like) == np.int16
    assert token_dtype({'<pad>': 0, 'ENSG1': 200, 'spv_x': 40000}) == np.int32
    assert token_dtype({'<pad>': 0, 'ENSG1': 70000}) == np.int32
    assert token_dtype({'<pad>': 0, 'ENSG1': 2 ** 40}) == np.int64


def _write_input(tmp_path, batches=('b0',)):
    """
    Batches of 40 cells (with long, empty and tied cells), one file each in
    `in/` and in `in_<batch>/`, and their token dictionary and norm factors;
    returns the tokenizer settings for them.
    """
    rng = np.random.default_rng(0)
    n_cells, n_genes = 40, 30
    token_dict = {'<pad>': 0, '<mask>': 1}
    token_dict.update({f'ENSG{i:011d}': 20000 + i for i in range(n_genes)})
    token_dict.update({'spv_0_b0': 30000, 'spv_xenium': 30001,
                       'spv_homo_sapiens': 30002, 'spv_lung': 30003})
    token_dict.update({f'spv_0_{batch}': 30004 + i
                       for i, batch in enumerate(batches[1:])})
    with open(tmp_path / 'td.pkl', 'wb') as f:
        pickle.dump(token_dict, f)
    genes = list(token_dict)[2:2 + n_genes]
    import pandas as pd
    pd.DataFrame({'gene_id': genes,
                  'tie_break_mean': rng.uniform(0.1, 2, n_genes)}).to_csv(
        tmp_path / 'nf.csv', index=False)
    (tmp_path / 'in').mkdir()
    for i, batch in enumerate(batches):
        if i:
            rng = np.random.default_rng(i)
        counts = rng.poisson(0.6, (n_cells, n_genes))
        counts[:6, :] = rng.poisson(5, (6, n_genes))     # more genes than SEQ
        counts[[3, 17, 18]] = 0                           # empty cells
        counts[20:30] = (rng.random((10, n_genes)) < 0.4)  # count-1 ties
        adata = ad.AnnData(X=sp.csr_matrix(counts.astype(np.int64)))
        adata.var['ensembl_id'] = genes
        adata.obs['cell_id'] = [f'0_{batch}_{c}' for c in range(n_cells)]
        adata.obsm['spatial'] = rng.uniform(0, 100, (n_cells, 2))
        adata.uns.update(dataset_id='0', batch=batch, assay='xenium',
                         species='homo_sapiens', tissue='lung')
        adata.write_h5ad(tmp_path / 'in' / f'{batch}.h5ad')
        (tmp_path / f'in_{batch}').mkdir()
        adata.write_h5ad(tmp_path / f'in_{batch}' / f'{batch}.h5ad')
    return dict(rank_cell_norm_method=None, rank_gene_norm_method=None,
                rank_count_norm_method=None,
                count_count_norm_method='shifted_log',
                stable_rank_ties=True, eliminate_stored_zeros=True,
                norm_factor_file_path=str(tmp_path / 'nf.csv'),
                model_input_size=MIS, n_neighs=N_NEIGHS,
                token_dictionary_file_path=tmp_path / 'td.pkl')


def _tokenize(tmp_path, name, use_generator=False, input_folder='in',
              num_shards=None, **kwargs):
    tk = CellGraphTokenizer(**kwargs)
    tk.tokenize_data(tmp_path / input_folder, tmp_path, name,
                     use_generator=use_generator, num_shards=num_shards,
                     cache_directory_path=str(tmp_path / 'cache' / name))
    return datasets.load_from_disk(str(tmp_path / f'{name}.dataset'))


@pytest.fixture(scope='module')
def tokenized(tmp_path_factory):
    """Padded and compact tokenizations of one batch, per variant."""
    tmp_path = tmp_path_factory.mktemp('compact')
    base = _write_input(tmp_path)
    out = {}
    for variant, settings in VARIANTS.items():
        for fmt in ('padded', 'compact'):
            for use_generator in ((False, True) if variant == 'corpus'
                                  else (False,)):
                name = f'{variant}_{fmt}_{int(use_generator)}'
                out[(variant, fmt, int(use_generator))] = _tokenize(
                    tmp_path, name, use_generator,
                    **{**base, **settings, 'storage_format': fmt})
    return out


# (variant, use_generator, include_special_tokens, add_neigh_cell_ids)
COLUMN_CASES = [(v, g, s, c) for v in VARIANTS for g in (0, 1)
                for s in (0, 1) for c in (0, 1)]


@pytest.fixture(scope='module')
def column_cases(tmp_path_factory):
    """Padded and compact tokenizations per case of `COLUMN_CASES`."""
    tmp_path = tmp_path_factory.mktemp('compact_columns')
    base = _write_input(tmp_path)
    out = {}
    for case in COLUMN_CASES:
        variant, use_generator, special, cell_ids = case
        out[case] = tuple(
            _tokenize(tmp_path, f'{fmt}_{variant}_{use_generator}{special}'
                      f'{cell_ids}', bool(use_generator),
                      **{**base, **VARIANTS[variant], 'storage_format': fmt,
                         'include_special_tokens': bool(special),
                         'add_neigh_cell_ids': bool(cell_ids)})
            for fmt in ('padded', 'compact'))
    return out


def _cell_dataset(dataset, seq_len_cell, n_neighs, **kwargs):
    base = dict(gt_type='combined', cell_pos_enc='coord', vocab_size=30004,
                seq_len_cell=seq_len_cell,
                seq_len_neighborhood=seq_len_cell * n_neighs,
                special_tokens=['batch', 'assay'], include_cell_id=True,
                truncate_neighbors=n_neighs != N_NEIGHS,
                tokenized_seq_len_cell=SEQ if seq_len_cell != SEQ else None)
    base.update(kwargs)
    return CellGraphDataset(dataset=dataset, **base)


def _padded_layout_lengths(padded, seq_len_cell, n_neighs):
    """What a per-layout n_nonzero_tokens list holds for a padded dataset."""
    tokens = np.asarray(padded.with_format('numpy')['gene_tokens'])
    if n_neighs == N_NEIGHS and seq_len_cell == SEQ:
        return [int(x) for x in padded['n_nonzero_tokens']]
    cols = np.concatenate([np.arange(s * SEQ, s * SEQ + seq_len_cell)
                           for s in range(n_neighs + 1)])
    return np.count_nonzero(tokens[:, cols], axis=1).tolist()


def test_variants_exercise_the_edge_cases(tokenized):
    def rows(variant):
        return tokenized[(variant, 'compact', 0)].with_format('numpy')
    corpus = rows('corpus')
    n_seg = np.array([len(x) for x in corpus['segment_lengths']])
    assert (n_seg == N_NEIGHS + 1).all()
    seg = np.stack(corpus['segment_lengths'])
    assert (seg[:, 0] == 0).any()                    # empty cells
    assert (seg[:, 1:] == 0).any()                   # padding segments
    assert (seg == SEQ).any() and (seg < SEQ).any()  # truncated and short
    # fewer, longer segments without padding segments
    assert any(len(x) < N_NEIGHS + 1
               for x in rows('no_pad_empty')['segment_lengths'])


def _hf_pair(rows_tokens, rows_values, n_segments_per_row, mis):
    """Padded and compact datasets of hand-made padded rows."""
    padded = datasets.Dataset.from_dict({
        'gene_tokens': [np.asarray(t, dtype=np.int64) for t in rows_tokens],
        'gene_expr': [np.asarray(v, dtype=np.float64) for v in rows_values],
        'n_nonzero_tokens': [int(np.count_nonzero(t)) for t in rows_tokens],
        'cell_id': [f'0_b0_{i}' for i in range(len(rows_tokens))]})
    out = {'gene_tokens': [], 'gene_expr': [], 'segment_lengths': [],
           'model_input_size': [], 'n_nonzero_tokens': [], 'cell_id': []}
    for i, (t, v, n_seg) in enumerate(zip(rows_tokens, rows_values,
                                         n_segments_per_row)):
        ct, cv, lens = compact_row(t, v, n_seg, mis, np.int16)
        out['gene_tokens'].append(ct)
        out['gene_expr'].append(cv)
        out['segment_lengths'].append(lens)
        out['model_input_size'].append(mis)
        out['n_nonzero_tokens'].append(int(np.count_nonzero(t)))
        out['cell_id'].append(f'0_b0_{i}')
    from datasets import Features, Value
    from datasets import Sequence as List  # datasets < 4 has no List
    compact = datasets.Dataset.from_dict(out, features=Features({
        'gene_tokens': List(Value('int16')), 'gene_expr': List(Value('float32')),
        'segment_lengths': List(Value('int16')),
        'model_input_size': Value('int32'), 'n_nonzero_tokens': Value('int32'),
        'cell_id': Value('string')}))
    return padded, compact


def test_layout_lengths_of_irregular_rows():
    # rows with padding between genes, and rows with fewer (longer) segments,
    # are counted on their rebuilt padded rows
    rng = np.random.default_rng(1)
    rows_t, rows_v, n_segs = [], [], []
    for i in range(30):
        n_seg = N_NEIGHS + 1 if i % 3 else int(rng.integers(1, N_NEIGHS + 1))
        seg_len = int(MIS / n_seg)
        t = np.zeros(MIS, dtype=np.int64)
        for s in range(n_seg):
            n = int(rng.integers(0, seg_len + 1))
            t[s * seg_len:s * seg_len + n] = rng.integers(2, 30000, n)
            if i % 2 and n > 2:
                t[s * seg_len + 1] = 0               # padding between genes
        rows_t.append(t)
        rows_v.append(np.where(t != 0, rng.random(MIS), 0.0))
        n_segs.append(n_seg)
    padded, compact = _hf_pair(rows_t, rows_v, n_segs, MIS)
    tokens = np.stack(rows_t)
    for seq_len_cell, n_neighs in LAYOUTS:
        layout = dict(seq_len_cell=seq_len_cell, n_segments=n_neighs + 1,
                      tokenized_seq_len_cell=SEQ,
                      truncate_neighbors=n_neighs != N_NEIGHS)
        if seq_len_cell == SEQ and n_neighs == N_NEIGHS:
            expected = np.count_nonzero(tokens, axis=1)
        else:
            cols = np.concatenate([np.arange(s * SEQ, s * SEQ + seq_len_cell)
                                   for s in range(n_neighs + 1)])
            expected = np.count_nonzero(tokens[:, cols], axis=1)
        got = layout_n_nonzero_tokens(compact, group_rows=4, **layout)
        assert got.tolist() == expected.tolist(), (seq_len_cell, n_neighs)
        assert layout_n_nonzero_tokens(
            compact, rows=[2, 0, 9], **layout).tolist() == [
            expected[2], expected[0], expected[9]]
        # and the items are the same as from the padded rows
        dp = _cell_dataset(padded.with_format('torch'), seq_len_cell,
                           n_neighs, special_tokens=[],
                           cell_pos_enc='segment')
        dc = _cell_dataset(compact.with_format('torch'), seq_len_cell,
                           n_neighs, special_tokens=[],
                           cell_pos_enc='segment')
        assert dc.n_nz_tokens.tolist() == expected.tolist()
        for i in range(len(dp)):
            if n_segs[i] == N_NEIGHS + 1:
                _assert_same(dp[i], dc[i])


@pytest.mark.parametrize('use_generator', [0, 1])
@pytest.mark.parametrize('seq_len_cell,n_neighs', LAYOUTS)
def test_dataset_items_equal(tokenized, use_generator, seq_len_cell, n_neighs):
    padded = tokenized[('corpus', 'padded', use_generator)].with_format('torch')
    compact = tokenized[('corpus', 'compact', use_generator)].with_format('torch')
    assert is_compact(compact) and not is_compact(padded)
    assert compact.features['gene_tokens'].feature.dtype == 'int16'
    dp = _cell_dataset(padded, seq_len_cell, n_neighs)
    dc = _cell_dataset(compact, seq_len_cell, n_neighs)
    for i in range(len(dp)):
        _assert_same(dp[i], dc[i])


@pytest.mark.parametrize('variant', ['no_pad_empty', 'dense', 'pflog1ppf'])
@pytest.mark.parametrize('seq_len_cell,n_neighs', [(8, 4), (4, 2)])
def test_dataset_items_equal_variants(tokenized, variant, seq_len_cell,
                                      n_neighs):
    padded = tokenized[(variant, 'padded', 0)].with_format('torch')
    compact = tokenized[(variant, 'compact', 0)].with_format('torch')
    for gt_type in ('combined', 'rank'):
        dp = _cell_dataset(padded, seq_len_cell, n_neighs, gt_type=gt_type)
        dc = _cell_dataset(compact, seq_len_cell, n_neighs, gt_type=gt_type)
        for i in range(len(dp)):
            _assert_same(dp[i], dc[i])


@pytest.mark.parametrize('variant', list(VARIANTS))
@pytest.mark.parametrize('seq_len_cell,n_neighs', LAYOUTS)
def test_layout_lengths(tokenized, variant, seq_len_cell, n_neighs):
    padded = tokenized[(variant, 'padded', 0)]
    compact = tokenized[(variant, 'compact', 0)].with_format('torch')
    expected = _padded_layout_lengths(padded, seq_len_cell, n_neighs)
    dc = _cell_dataset(compact, seq_len_cell, n_neighs)
    assert isinstance(dc.n_nz_tokens, np.ndarray)
    assert dc.n_nz_tokens.tolist() == expected
    # read a few rows at a time, and only some rows
    layout = dict(seq_len_cell=seq_len_cell, n_segments=n_neighs + 1,
                  tokenized_seq_len_cell=SEQ,
                  truncate_neighbors=n_neighs != N_NEIGHS)
    assert layout_n_nonzero_tokens(
        compact, group_rows=7, **layout).tolist() == expected
    rows = [0, 3, 17, 39, 5]
    assert layout_n_nonzero_tokens(compact, rows=rows, **layout).tolist() == [
        expected[r] for r in rows]
    # a selection (indices mapping), as precomputed splits make
    sel = [38, 2, 17, 3, 21, 20, 0]
    assert layout_n_nonzero_tokens(compact.select(sel), **layout).tolist() == [
        expected[r] for r in sel]
    assert _cell_dataset(compact.select(sel), seq_len_cell,
                         n_neighs).n_nz_tokens.tolist() == [
        expected[r] for r in sel]
    # the padded dataset with the per-layout list gives the same sampler order
    dp = _cell_dataset(padded.with_format('torch'), seq_len_cell, n_neighs,
                       n_nonzero_tokens_list=expected)
    for epoch in range(2):
        orders = []
        for d in (dp, dc):
            s = CustomDistributedLengthGroupedSampler(
                d, batch_size=4, num_replicas=2, rank=1, seed=3)
            s.set_epoch(epoch)
            orders.append(list(s))
        assert orders[0] == orders[1]


def test_given_lengths_are_checked(tokenized):
    padded = tokenized[('corpus', 'padded', 0)]
    compact = tokenized[('corpus', 'compact', 0)].with_format('torch')
    full = [int(x) for x in padded['n_nonzero_tokens']]
    layout = _padded_layout_lengths(padded, 4, 2)
    assert full != layout
    # the list of the layout is accepted, the list of the full rows is not
    dc = _cell_dataset(compact, 4, 2, n_nonzero_tokens_list=layout)
    assert dc.n_nz_tokens == layout
    with pytest.raises(ValueError, match='does not hold the lengths'):
        _cell_dataset(compact, 4, 2, n_nonzero_tokens_list=full)
    with pytest.raises(ValueError, match='entries'):
        check_n_nonzero_tokens(full[:-1], compact, seq_len_cell=SEQ,
                               n_segments=N_NEIGHS + 1)
    # the stored layout: the full rows' list is right
    dc = _cell_dataset(compact, SEQ, N_NEIGHS, n_nonzero_tokens_list=full)
    assert dc.n_nz_tokens == full


def test_layout_errors_match_getitem(tokenized):
    compact = tokenized[('corpus', 'compact', 0)].with_format('torch')
    with pytest.raises(ValueError, match='exceeds the dataset'):
        _cell_dataset(compact, 4, N_NEIGHS + 2)
    with pytest.raises(ValueError, match='not a multiple'):
        _cell_dataset(compact, 3, 2, tokenized_seq_len_cell=7)


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_to_padded(tokenized, variant):
    padded = tokenized[(variant, 'padded', 0)].with_format('torch')
    back = to_padded(tokenized[(variant, 'compact', 0)]).with_format('torch')
    assert set(back.column_names) == set(padded.column_names)
    for i in range(len(back)):
        _assert_same(back[i], padded[i])
    raw_p = tokenized[(variant, 'padded', 0)]
    raw_b = to_padded(tokenized[(variant, 'compact', 0)])
    assert raw_b.features == raw_p.features
    cols = ['gene_tokens']
    if variant != 'pflog1ppf':
        # shifted-log values are float32 numbers (numpy's float32 log1p): the
        # float64 column is the same too
        cols.append('gene_expr')
    for col in cols:
        assert np.array_equal(np.asarray(raw_b[col]),
                              np.asarray(raw_p[col])), col
    # relative coordinates are computed in float64 from float64 coordinates;
    # the compact format keeps their float32 rounding, which is what the
    # torch format of the padded dataset gives the model
    for col in ('rel_x_coord', 'rel_y_coord', 'gene_expr'):
        a = np.asarray(raw_b[col]).astype(np.float32)
        b = np.asarray(raw_p[col]).astype(np.float32)
        assert np.array_equal(a.view(np.int32), b.view(np.int32)), col


def test_compact_record_and_schema(tokenized):
    from terra.tokenizers.params import TOKENIZER_PARAMS_FILE
    compact = tokenized[('corpus', 'compact', 0)]
    padded = tokenized[('corpus', 'padded', 0)]
    feats = compact.features
    assert feats['gene_tokens'].feature.dtype == 'int16'
    assert feats['gene_expr'].feature.dtype == 'float32'
    assert feats['segment_lengths'].feature.dtype == 'int16'
    assert feats['rel_x_coord'].feature.dtype == 'float32'
    # one row per cell, same cells in the same order, the same lengths
    assert compact['cell_id'] == padded['cell_id']
    assert compact['n_nonzero_tokens'] == padded['n_nonzero_tokens']
    # generator and dictionary paths write the same compact rows
    gen = tokenized[('corpus', 'compact', 1)]
    assert gen.features == feats
    for col in feats:
        assert gen[col] == compact[col], col
    import json
    from pathlib import Path
    for ds, fmt in ((compact, 'compact'), (padded, None)):
        folder = Path(ds.cache_files[0]['filename']).parent
        record = json.loads((folder / TOKENIZER_PARAMS_FILE).read_text())
        assert record.get('storage_format') == fmt


def test_special_token_names_match_the_tokenizer():
    from terra.datasets import compact
    from terra.tokenizers import cell_tokenizers
    assert compact.SPECIAL_TOKEN_NAMES == cell_tokenizers.SPECIAL_TOKEN_NAMES


def _inference_format(dataset):
    # as tokenize_adata formats datasets (string columns left unformatted)
    columns = [c for c in dataset.column_names
               if c not in ('cell_id', 'cell_ids')]
    return dataset.with_format('torch', columns=columns,
                               output_all_columns=True)


@pytest.mark.parametrize('case', COLUMN_CASES)
def test_compact_columns_follow_the_padded_order(column_cases, case):
    # The compact columns are the padded ones, in the padded order, followed
    # by the compact-only columns, so items and batches have the same keys
    # in the same order
    from terra.masks.block_masking import BlockMaskCollator
    padded, compact = column_cases[case]
    _, _, special, cell_ids = case
    columns = list(compact.features)
    assert columns[-len(COMPACT_COLUMNS):] == list(COMPACT_COLUMNS)
    assert columns[:-len(COMPACT_COLUMNS)] == list(padded.features)
    assert ('cell_ids' in columns) == bool(cell_ids)
    assert ('batch_value' in columns) == bool(special)
    special_tokens = ['batch', 'assay'] if special else []
    for seq_len_cell, n_neighs in [(SEQ, N_NEIGHS), (4, 2)]:
        cell_datasets = [
            _cell_dataset(_inference_format(d), seq_len_cell, n_neighs,
                          special_tokens=special_tokens)
            for d in (padded, compact)]
        for i in range(len(padded)):
            _assert_same(cell_datasets[0][i], cell_datasets[1][i])
        batches = []
        for cell_dataset in cell_datasets:
            collator = BlockMaskCollator(
                n_targets=1, n_contexts=1, n_segments=n_neighs + 1,
                seq_len_cell=seq_len_cell,
                seq_len_neighborhood=seq_len_cell * n_neighs,
                n_special_tokens=len(special_tokens),
                per_block_mask_ratio=0.6, sample_segments=True)
            torch.manual_seed(1)
            batches.append(collator([cell_dataset[i] for i in range(16)]))
        assert batches[0][4] == batches[1][4]
        _assert_same(batches[0][0], batches[1][0])
        for a, b in zip(batches[0][1:4], batches[1][1:4]):
            assert torch.equal(a, b)


# The corpus pipeline: per-dataset parts tokenized one file at a time
# (stream_per_file), concatenated, saved in shards, split with precomputed
# splits and read by prepare_dataset

CORPUS_SETTINGS = dict(VARIANTS['corpus'], stream_per_file=True)


@pytest.fixture(scope='module')
def corpus_input(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('compact_corpus')
    return tmp_path, _write_input(tmp_path, batches=('b0', 'b1'))


def _table(dataset):
    return dataset.data.table.combine_chunks()


@pytest.mark.parametrize('processing_mode,nproc', [('sequential', 1),
                                                    ('parallel', 2)])
def test_streaming_writes_the_same_compact_dataset(corpus_input,
                                                   processing_mode, nproc):
    tmp_path, base = corpus_input
    out = {}
    for stream in (False, True):
        out[stream] = _tokenize(
            tmp_path, f'stream{int(stream)}_{processing_mode}', True,
            **{**base, **VARIANTS['corpus'], 'storage_format': 'compact',
               'stream_per_file': stream, 'processing_mode': processing_mode,
               'nproc': nproc})
    assert len(out[True]) == 80
    assert list(out[True].features.items()) == list(
        out[False].features.items())
    assert _table(out[True]).equals(_table(out[False]))


def _corpus(tmp_path, base, fmt):
    """
    The two batches tokenized as separate parts, concatenated and saved in 3
    shards (cutting through the first batch), with the record of a part.
    """
    import shutil

    from terra.tokenizers.params import TOKENIZER_PARAMS_FILE
    parts = [_tokenize(tmp_path, f'part_{fmt}_{batch}', True,
                       input_folder=f'in_{batch}',
                       **{**base, **CORPUS_SETTINGS, 'storage_format': fmt})
             for batch in ('b0', 'b1')]
    corpus_path = tmp_path / f'corpus_{fmt}.dataset'
    datasets.concatenate_datasets(parts).save_to_disk(str(corpus_path),
                                                      num_shards=3)
    shutil.copy(tmp_path / f'part_{fmt}_b0.dataset' / TOKENIZER_PARAMS_FILE,
                corpus_path / TOKENIZER_PARAMS_FILE)
    return parts, corpus_path


@pytest.fixture(scope='module')
def corpora(corpus_input):
    tmp_path, base = corpus_input
    return {fmt: _corpus(tmp_path, base, fmt)
            for fmt in ('padded', 'compact')}


@pytest.mark.parametrize('seq_len_cell,n_neighs', [(SEQ, N_NEIGHS), (4, 2)])
def test_compact_corpus_through_prepare_dataset(corpora, tmp_path,
                                                seq_len_cell, n_neighs):
    from terra.datasets.utils import prepare_dataset, save_precomputed_split
    padded_path = corpora['padded'][1]
    compact_path = corpora['compact'][1]
    padded = datasets.load_from_disk(str(padded_path))
    compact = datasets.load_from_disk(str(compact_path))
    assert len(compact.cache_files) == 3 and len(compact) == 80
    # Rows of both batches, in a shuffled order
    rows = np.random.default_rng(0).permutation(80)[:50].tolist()
    split_path = tmp_path / 'split.pkl'
    save_precomputed_split(compact, rows, split_path)
    selected = {}
    for fmt, path in (('padded', padded_path), ('compact', compact_path)):
        args = {'data': {'tokenized_data_folder_path': str(path),
                         'precomputed_split': str(split_path)},
                'meta': {'cell_pos_enc': 'coord'}}
        selected[fmt] = prepare_dataset(args)[0]
    expected = [_padded_layout_lengths(padded, seq_len_cell, n_neighs)[r]
                for r in rows]
    dp = _cell_dataset(selected['padded'], seq_len_cell, n_neighs,
                       n_nonzero_tokens_list=expected)
    dc = _cell_dataset(selected['compact'], seq_len_cell, n_neighs)
    assert dc.n_nz_tokens.tolist() == expected
    for i in range(len(rows)):
        _assert_same(dp[i], dc[i])


def test_padded_and_compact_parts_are_not_concatenated(corpora):
    padded_part = corpora['padded'][0][0]
    compact_part = corpora['compact'][0][1]
    with pytest.raises(ValueError):
        datasets.concatenate_datasets([padded_part, compact_part])


@pytest.mark.parametrize('fmt,record_of', [('compact', 'padded'),
                                           ('padded', 'compact')])
def test_record_of_another_storage_format_is_refused(corpora, tmp_path, fmt,
                                                     record_of):
    import shutil

    from terra.datasets.utils import prepare_dataset
    from terra.tokenizers.params import TOKENIZER_PARAMS_FILE
    path = tmp_path / 'corpus.dataset'
    shutil.copytree(corpora[fmt][1], path)
    args = {'data': {'tokenized_data_folder_path': str(path)},
            'meta': {'cell_pos_enc': 'coord'}}
    assert len(prepare_dataset(args)[0]) == 80
    shutil.copy(corpora[record_of][1] / TOKENIZER_PARAMS_FILE,
                path / TOKENIZER_PARAMS_FILE)
    with pytest.raises(ValueError, match=f"storage_format '{record_of}'"):
        prepare_dataset(args)
    # Without a record, the columns decide
    (path / TOKENIZER_PARAMS_FILE).unlink()
    assert len(prepare_dataset(args)[0]) == 80
