"""
Tests for ``CellGraphTokenizer(add_spatial_index=True)``: the extra columns
(`cell_index`, `x_coord`, `y_coord`, `neighbor_offsets`) in both storage
formats, unchanged tokens, the neighbour rows they give, and
`terra.training.exposures.check_neighbor_rows`.
"""

import json
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

datasets = pytest.importorskip('datasets')

from terra.tokenizers import CellGraphTokenizer
from terra.tokenizers.params import TOKENIZER_PARAMS_FILE
from terra.training.exposures import check_neighbor_rows

N_NEIGHS, SEQ = 4, 8
MIS = SEQ * (N_NEIGHS + 1)
SPATIAL_COLUMNS = ('cell_index', 'x_coord', 'y_coord', 'neighbor_offsets')


def _write_input(tmp_path, n_cells=(30, 25)):
    """Two batch files (sorted b0, b1) and their dictionary / norm factors."""
    rng = np.random.default_rng(0)
    n_genes = 20
    token_dict = {'<pad>': 0, '<mask>': 1}
    token_dict.update({f'ENSG{i:011d}': 200 + i for i in range(n_genes)})
    token_dict.update({'spv_0_b0': 300, 'spv_0_b1': 301, 'spv_xenium': 302,
                       'spv_homo_sapiens': 303, 'spv_lung': 304})
    with open(tmp_path / 'td.pkl', 'wb') as f:
        pickle.dump(token_dict, f)
    genes = list(token_dict)[2:2 + n_genes]
    pd.DataFrame({'gene_id': genes,
                  'tie_break_mean': rng.uniform(0.1, 2, n_genes)}).to_csv(
        tmp_path / 'nf.csv', index=False)
    (tmp_path / 'in').mkdir()
    for batch, n in zip(('b0', 'b1'), n_cells):
        counts = rng.poisson(1.0, (n, n_genes))
        counts[:, 0] += 1                      # no empty cells
        adata = ad.AnnData(X=sp.csr_matrix(counts.astype(np.int64)))
        adata.var['ensembl_id'] = genes
        adata.obs['cell_id'] = [f'0_{batch}_{c}' for c in range(n)]
        # integer coordinates on purpose (stored as float64)
        adata.obsm['spatial'] = rng.integers(0, 1000, (n, 2))
        adata.uns.update(dataset_id='0', batch=batch, assay='xenium',
                         species='homo_sapiens', tissue='lung')
        adata.write_h5ad(tmp_path / 'in' / f'{batch}.h5ad')
    return dict(rank_cell_norm_method=None, rank_gene_norm_method=None,
                rank_count_norm_method=None,
                count_count_norm_method='shifted_log',
                stable_rank_ties=True, eliminate_stored_zeros=True,
                pad_empty_neighbors=True,
                norm_factor_file_path=str(tmp_path / 'nf.csv'),
                model_input_size=MIS, n_neighs=N_NEIGHS,
                token_dictionary_file_path=tmp_path / 'td.pkl')


def _tokenize(tmp_path, name, **kwargs):
    CellGraphTokenizer(**kwargs).tokenize_data(
        tmp_path / 'in', tmp_path, name,
        cache_directory_path=str(tmp_path / 'cache' / name))
    return datasets.load_from_disk(str(tmp_path / f'{name}.dataset'))


@pytest.fixture(scope='module')
def tokenized(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('spatial_index')
    base = _write_input(tmp_path)
    out = {}
    for fmt in ('padded', 'compact'):
        for spatial in (False, True):
            name = f'{fmt}_{int(spatial)}'
            out[(fmt, spatial)] = _tokenize(
                tmp_path, name, storage_format=fmt,
                **({'add_spatial_index': True} if spatial else {}), **base)
            out[(fmt, spatial, 'record')] = json.loads(
                (tmp_path / f'{name}.dataset' / TOKENIZER_PARAMS_FILE)
                .read_text())
    out['coords'] = {
        batch: np.asarray(ad.read_h5ad(tmp_path / 'in' / f'{batch}.h5ad')
                          .obsm['spatial'], dtype=np.float64)
        for batch in ('b0', 'b1')}
    return out


@pytest.mark.parametrize('fmt', ['padded', 'compact'])
def test_tokens_are_unchanged(tokenized, fmt):
    plain, spatial = tokenized[(fmt, False)], tokenized[(fmt, True)]
    assert [c for c in spatial.column_names if c not in SPATIAL_COLUMNS] \
        == plain.column_names
    assert spatial.features[plain.column_names[-1]] == \
        plain.features[plain.column_names[-1]]
    for column in plain.column_names:
        assert spatial[column] == plain[column], column


@pytest.mark.parametrize('fmt', ['padded', 'compact'])
def test_spatial_columns(tokenized, fmt):
    ds = tokenized[(fmt, True)]
    coords = tokenized['coords']
    batches = [cid.split('_')[1] for cid in ds['cell_id']]
    n0 = batches.count('b0')
    assert ds['cell_index'] == list(range(n0)) + list(range(len(ds) - n0))
    xy = np.concatenate([coords['b0'], coords['b1']])
    assert np.array_equal(np.asarray(ds['x_coord']), xy[:, 0])
    assert np.array_equal(np.asarray(ds['y_coord']), xy[:, 1])
    if fmt == 'compact':
        assert ds.features['cell_index'].dtype == 'int32'
        assert ds.features['x_coord'].dtype == 'float64'
    for row in range(len(ds)):
        offsets = np.asarray(ds[row]['neighbor_offsets'])
        assert offsets.size == N_NEIGHS and (offsets != 0).all()
        neighbours = row + offsets
        # same file
        assert all(batches[n] == batches[row] for n in neighbours)
        # the relative coordinates are those of the neighbour rows
        rel_x = np.asarray(ds[row]['rel_x_coord'], dtype=np.float64)
        rel_y = np.asarray(ds[row]['rel_y_coord'], dtype=np.float64)
        np.testing.assert_allclose(xy[neighbours, 0] - xy[row, 0], rel_x[1:])
        np.testing.assert_allclose(xy[neighbours, 1] - xy[row, 1], rel_y[1:])
        # nearest first
        dist = np.hypot(rel_x[1:], rel_y[1:])
        assert (np.diff(dist) >= 0).all()


def test_record(tokenized):
    for fmt in ('padded', 'compact'):
        assert tokenized[(fmt, True, 'record')]['add_spatial_index'] is True
        assert 'add_spatial_index' not in tokenized[(fmt, False, 'record')]


@pytest.mark.parametrize('fmt', ['padded', 'compact'])
def test_check_neighbor_rows(tokenized, fmt):
    ds = tokenized[(fmt, True)]
    n = len(ds)
    assert check_neighbor_rows(ds, n_check=n) == n * N_NEIGHS
    # a selection of whole files keeps the neighbour rows
    b0 = [i for i, cid in enumerate(ds['cell_id']) if '_b0_' in cid]
    b1 = [i for i, cid in enumerate(ds['cell_id']) if '_b1_' in cid]
    assert check_neighbor_rows(ds.select(b1)) == len(b1) * N_NEIGHS
    assert check_neighbor_rows(ds.select(b1 + b0), chunk_rows=7) == n * N_NEIGHS
    # without the coordinates, only the exact check runs
    assert check_neighbor_rows(ds.remove_columns(['x_coord', 'y_coord'])) \
        == n * N_NEIGHS
    broken = {
        'shuffled': np.random.default_rng(0).permutation(n).tolist(),
        'a row removed': [i for i in range(n) if i != 7],
        'a file cut at its end': b0[:-3] + b1,
        'a file cut at its start': b0 + b1[2:],
    }
    for name, rows in broken.items():
        with pytest.raises(ValueError,
                           match='not at row \\+ neighbor_offsets'):
            check_neighbor_rows(ds.select(rows), chunk_rows=10)
    with pytest.raises(ValueError, match='add_spatial_index'):
        check_neighbor_rows(tokenized[(fmt, False)])


def test_spatial_index_needs_padded_empty_neighbours(tmp_path):
    base = _write_input(tmp_path)
    base['pad_empty_neighbors'] = False
    with pytest.raises(ValueError, match='pad_empty_neighbors=True'):
        CellGraphTokenizer(add_spatial_index=True, **base)


def test_to_padded_matches_the_padded_tokenization(tokenized):
    from terra.datasets.compact import to_padded
    padded = tokenized[('padded', True)]
    converted = to_padded(tokenized[('compact', True)])
    assert converted.features == padded.features
    for column in SPATIAL_COLUMNS:
        assert converted[column] == padded[column], column
    datasets.concatenate_datasets([padded, converted])
