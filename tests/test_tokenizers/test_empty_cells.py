"""
Tests for cells without any expressed vocabulary gene in the cell-graph
tokenizer.

On the sparse ranking path (used by the released models) such a cell has no
gene tokens. As a neighbor it did not count as a segment: with k such
neighbors, the index cell got only n_neighs - k (longer) neighbor segments,
its k farthest neighbors were dropped, and the segments no longer started at
the multiples of ``seq_len_cell`` where the cell datasets read them.
``pad_empty_neighbors=True`` keeps an all-padding segment for it, as the
dense ranking path does.
"""

import logging
import pickle

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from terra.tokenizers import CellGraphTokenizer

N_CELLS = 30
N_SHARED = 20
N_NEIGHS = 4
SEQ_LEN_CELL = 8
MODEL_INPUT_SIZE = SEQ_LEN_CELL * (N_NEIGHS + 1)
EMPTY_CELLS = (3, 11, 12)
# Token id of gene g (gene 0 is the private marker gene of cell 0, ...)
TOKEN_OFFSET = 2


@pytest.fixture
def token_dictionary_file_path(tmp_path):
    token_dict = {'<pad>': 0, '<mask>': 1}
    token_dict.update({f'ENSG{i:011d}': i + TOKEN_OFFSET
                       for i in range(N_CELLS + N_SHARED)})
    path = tmp_path / 'token_dictionary.pkl'
    with open(path, 'wb') as f:
        pickle.dump(token_dict, f)
    return path


def _make_adata(empty_cells=EMPTY_CELLS, seed=0):
    """Every cell has a private marker gene (count 100, ranked first), so
    the first token of a segment tells whose segment it is."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((N_CELLS, N_CELLS + N_SHARED), dtype=np.float32)
    counts[np.arange(N_CELLS), np.arange(N_CELLS)] = 100
    counts[:, N_CELLS:] = rng.poisson(1.0, (N_CELLS, N_SHARED))
    counts[list(empty_cells)] = 0
    adata = ad.AnnData(X=sp.csr_matrix(counts))
    adata.var['ensembl_id'] = [f'ENSG{i:011d}'
                               for i in range(N_CELLS + N_SHARED)]
    adata.obs['cell_id'] = [f'cell_{i}' for i in range(N_CELLS)]
    adata.obsm['spatial'] = rng.uniform(0, 100, (N_CELLS, 2))
    return adata


def _nearest_neighbors(adata):
    distances = cdist(adata.obsm['spatial'], adata.obsm['spatial'])
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1)[:, :N_NEIGHS]


def _tokenize(token_dictionary_file_path, adata, dense=False, **kwargs):
    tk = CellGraphTokenizer(
        model_input_size=MODEL_INPUT_SIZE,
        n_neighs=N_NEIGHS,
        token_dictionary_file_path=token_dictionary_file_path,
        include_special_tokens=False,
        rank_cell_norm_method=None,
        rank_gene_norm_method=None,
        rank_count_norm_method=None,
        count_cell_norm_method=None,
        count_gene_norm_method=None,
        count_count_norm_method='shifted_log',
        **kwargs)
    if dense:
        # Same ranking, but through the dense ranking path
        tk.rank_differs_from_count = True
    adata_dict = tk._tokenize_adata(adata=adata)
    keys = ('gene_tokens_cell', 'gene_expr_cell', 'gene_tokens_neighborhood',
            'gene_expr_neighborhood', 'seg_tokens_neighborhood')
    return [tk._format_examples({key: adata_dict[key][i] for key in keys})
            for i in range(adata.n_obs)]


def _segments(example):
    return np.asarray(example['gene_tokens']).reshape(N_NEIGHS + 1,
                                                       SEQ_LEN_CELL)


def test_pad_empty_neighbors_keeps_every_neighbor_in_its_segment(
        token_dictionary_file_path):
    adata = _make_adata()
    neighbors = _nearest_neighbors(adata)
    examples = _tokenize(token_dictionary_file_path, adata,
                         pad_empty_neighbors=True)
    for i, example in enumerate(examples):
        assert len(example['gene_tokens']) == MODEL_INPUT_SIZE
        for slot, cell in enumerate([i] + neighbors[i].tolist()):
            segment = _segments(example)[slot]
            if cell in EMPTY_CELLS:
                assert np.all(segment == 0)
            else:
                # The cell's marker gene leads its own segment
                assert segment[0] == cell + TOKEN_OFFSET
        assert example['n_nonzero_tokens'] == np.count_nonzero(
            example['gene_tokens'])


def test_empty_neighbors_shift_the_segments_by_default(
        token_dictionary_file_path):
    adata = _make_adata()
    neighbors = _nearest_neighbors(adata)
    default = _tokenize(token_dictionary_file_path, adata)
    padded = _tokenize(token_dictionary_file_path, adata,
                       pad_empty_neighbors=True)
    n_shifted = 0
    for i in range(N_CELLS):
        tokens = np.asarray(default[i]['gene_tokens'])
        n_empty = len(set(neighbors[i].tolist()) & set(EMPTY_CELLS))
        if n_empty == 0:
            np.testing.assert_array_equal(tokens,
                                          padded[i]['gene_tokens'])
            continue
        # Previous layout, kept as default: n_neighs - n_empty longer
        # segments for the nearest neighbors; the farthest ones are dropped
        n_shifted += 1
        seg_length = MODEL_INPUT_SIZE // (1 + N_NEIGHS - n_empty)
        assert seg_length > SEQ_LEN_CELL
        kept = neighbors[i][:N_NEIGHS - n_empty].tolist()
        for slot, cell in enumerate(kept, start=1):
            segment = tokens[slot * seg_length:(slot + 1) * seg_length]
            if cell in EMPTY_CELLS:
                assert np.all(segment == 0)
            else:
                assert segment[0] == cell + TOKEN_OFFSET
        for cell in neighbors[i][N_NEIGHS - n_empty:]:
            assert cell + TOKEN_OFFSET not in tokens
    assert n_shifted > 0


def test_data_without_empty_cells_is_unchanged(token_dictionary_file_path):
    adata = _make_adata(empty_cells=())
    default = _tokenize(token_dictionary_file_path, adata)
    padded = _tokenize(token_dictionary_file_path, adata,
                       pad_empty_neighbors=True)
    for d, p in zip(default, padded):
        for key in ('gene_tokens', 'gene_expr', 'n_nonzero_tokens'):
            np.testing.assert_array_equal(d[key], p[key])


def test_pad_empty_neighbors_matches_dense_path(token_dictionary_file_path):
    # The dense path stores the panel of an empty cell with every token
    # masked to 0, so it always kept a (padding) segment for it
    adata = _make_adata()
    sparse = _tokenize(token_dictionary_file_path, adata,
                       pad_empty_neighbors=True, stable_rank_ties=True)
    dense = _tokenize(token_dictionary_file_path, adata, dense=True,
                      stable_rank_ties=True)
    for s, d in zip(sparse, dense):
        np.testing.assert_array_equal(s['gene_tokens'], d['gene_tokens'])
        np.testing.assert_allclose(s['gene_expr'], d['gene_expr'])
        assert s['n_nonzero_tokens'] == d['n_nonzero_tokens']


@pytest.mark.parametrize('pad_empty_neighbors', [False, True])
def test_empty_cells_are_reported(token_dictionary_file_path, caplog,
                                  pad_empty_neighbors):
    adata = _make_adata()
    n_shifted = sum(bool(set(row) & set(EMPTY_CELLS))
                    for row in _nearest_neighbors(adata).tolist())
    with caplog.at_level(logging.WARNING):
        _tokenize(token_dictionary_file_path, adata,
                  pad_empty_neighbors=pad_empty_neighbors)
    messages = [r.getMessage() for r in caplog.records
                if 'no expressed gene' in r.getMessage()]
    assert len(messages) == 1
    assert messages[0].startswith(f'{len(EMPTY_CELLS)} of {N_CELLS} cells')
    assert 'min_genes_per_cell >= 1' in messages[0]
    assert (f'lengthens the segments of {n_shifted} cells' in messages[0]) != (
        pad_empty_neighbors)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _tokenize(token_dictionary_file_path, _make_adata(empty_cells=()))
    assert not any('no expressed gene' in r.getMessage()
                   for r in caplog.records)
