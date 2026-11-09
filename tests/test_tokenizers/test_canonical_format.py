"""
Tests for sparse count matrices that are not in canonical format: column
indices not sorted within a row, or several entries for the same gene.

The sparse ranking path reads the rank and count values of a cell by storage
position. PFlog1pPF, and shifted-log of counts that are not float32, cast the
count layer with scipy's ``astype``, which sums duplicate entries and thereby
sorts the indices, while the rank layer kept the stored order: with unsorted
indices (e.g. after ``adata[:, genes]`` with the genes in another order) the
values were attached to the wrong genes, and with duplicate entries the rank
row of a cell was longer than its count row (an ``IndexError``). Normalizations
that keep duplicate entries (e.g. shifted-log of float32 counts) tokenized such
a gene twice. The tokenizers now tokenize the canonical form of such a matrix
(sorted indices, duplicates summed), without modifying the caller's AnnData.

The neighborhood counts that the cell-neighborhood tokenizer aggregates (a
sparse product of the connectivities and the counts) had unsorted indices even
for canonical counts. With ``aggregate_own_neighbors`` (a CSR product; the
default CSC product gets sorted indices when converted to CSR), shifted-log
values of counts that are not float32 were thus attached to the wrong
neighborhood genes. ``construct_neighbor_graph`` now sorts the indices.
"""

import pickle

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from terra.preprocessors import construct_neighbor_graph
from terra.tokenizers import CellGraphTokenizer, CellNeighborhoodTokenizer
from terra.tokenizers.cell_tokenizers import _canonical_X

N_CELLS = 40
N_GENES = 60
N_NEIGHS = 4
MODEL_INPUT_SIZE = 8 * (N_NEIGHS + 1)


@pytest.fixture
def token_dictionary_file_path(tmp_path):
    token_dict = {'<pad>': 0, '<mask>': 1}
    token_dict.update({f'ENSG{i:011d}': i + 2 for i in range(N_GENES)})
    path = tmp_path / 'token_dictionary.pkl'
    with open(path, 'wb') as f:
        pickle.dump(token_dict, f)
    return path


def _counts(dtype) -> sp.csr_matrix:
    rng = np.random.default_rng(0)
    # ~38 expressed genes per cell, many with counts >= 2
    counts = rng.poisson(1.0, (N_CELLS, N_GENES)).astype(dtype)
    counts[counts.sum(axis=1) == 0, 0] = 1
    return sp.csr_matrix(counts)


def _unsorted(x):
    """The same CSR or CSC matrix with the entries of every row (column) in
    a random order."""
    rng = np.random.default_rng(1)
    indices, data = x.indices.copy(), x.data.copy()
    for start, end in zip(x.indptr[:-1], x.indptr[1:]):
        order = start + rng.permutation(end - start)
        indices[start:end], data[start:end] = indices[order], data[order]
    return type(x)((data, indices, x.indptr.copy()), shape=x.shape)


def _with_duplicates(x):
    """The same CSR or CSC matrix with every count >= 2 split into two
    entries."""
    major = np.repeat(np.arange(len(x.indptr) - 1), np.diff(x.indptr))
    split = x.data >= 2
    first = np.where(split, x.data // 2, x.data)
    major = np.concatenate([major, major[split]])
    minor = np.concatenate([x.indices, x.indices[split]])
    data = np.concatenate([first, x.data[split] - first[split]])
    order = np.lexsort((minor, major))
    indptr = np.concatenate(
        [[0], np.cumsum(np.bincount(major, minlength=len(x.indptr) - 1))])
    return type(x)((data[order], minor[order], indptr), shape=x.shape)


LAYOUTS = {
    'unsorted': _unsorted,
    'duplicates': _with_duplicates,
    'csc': lambda x: x.tocsc(),
    'csc_unsorted': lambda x: _unsorted(x.tocsc()),
    'csc_duplicates': lambda x: _with_duplicates(x.tocsc()),
    'dense': lambda x: x.toarray(),
}


def _adata(x) -> ad.AnnData:
    adata = ad.AnnData(X=x)
    adata.var['ensembl_id'] = [f'ENSG{i:011d}' for i in range(N_GENES)]
    adata.obs['cell_id'] = [f'cell_{i}' for i in range(N_CELLS)]
    adata.obsm['spatial'] = np.random.default_rng(2).uniform(
        0, 100, (N_CELLS, 2))
    return adata


def _tokenize(token_dictionary_file_path, tokenizer, count_count_norm_method,
              adata=None, adata_file_path=None, **tokenizer_kwargs):
    tk = tokenizer(
        model_input_size=MODEL_INPUT_SIZE, n_neighs=N_NEIGHS,
        token_dictionary_file_path=token_dictionary_file_path,
        include_special_tokens=False, rank_cell_norm_method=None,
        rank_gene_norm_method=None, rank_count_norm_method=None,
        count_cell_norm_method=None, count_gene_norm_method=None,
        count_count_norm_method=count_count_norm_method, **tokenizer_kwargs)
    adata_dict = tk._tokenize_adata(adata=adata,
                                    adata_file_path=adata_file_path)
    keys = ('gene_tokens_cell', 'gene_expr_cell', 'gene_tokens_neighborhood',
            'gene_expr_neighborhood', 'seg_tokens_neighborhood')
    examples = [tk._format_examples({key: adata_dict[key][i] for key in keys
                                     if key in adata_dict})
                for i in range(N_CELLS)]
    return (np.stack([example['gene_tokens'] for example in examples]),
            np.stack([example['gene_expr'] for example in examples]))


# Rank by raw counts (sparse ranking path), with further tokenizer keyword
# arguments; the cell-neighborhood tokenizer has no PFlog1pPF
OWN_NEIGHBORS = {'aggregate_own_neighbors': True}
NORMS = [(CellGraphTokenizer, 'shifted_log', np.int64, {}),
         (CellGraphTokenizer, 'shifted_log', np.float32, {}),
         (CellGraphTokenizer, 'pflog1ppf', np.int64, {}),
         (CellGraphTokenizer, 'pflog1ppf', np.float32, {}),
         (CellNeighborhoodTokenizer, 'shifted_log', np.int64, {}),
         (CellNeighborhoodTokenizer, 'shifted_log', np.float32, {}),
         (CellNeighborhoodTokenizer, 'shifted_log', np.int64, OWN_NEIGHBORS),
         (CellNeighborhoodTokenizer, 'shifted_log', np.float32,
          OWN_NEIGHBORS)]


def _tokenizer_kwargs_id(value):
    """Test id of the tokenizer keyword arguments in NORMS; `None` (pytest's
    default id) for the other parameters."""
    if isinstance(value, dict):
        return '-'.join(value) or 'defaults'


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize(
    'tokenizer, count_count_norm_method, dtype, tokenizer_kwargs', NORMS,
    ids=_tokenizer_kwargs_id)
def test_counts_tokenize_like_their_canonical_csr_form(
        token_dictionary_file_path, tokenizer, count_count_norm_method, dtype,
        tokenizer_kwargs, layout):
    canonical = _counts(dtype)
    x = LAYOUTS[layout](canonical)
    if sp.issparse(x) and layout != 'csc':
        assert not x.has_canonical_format
    stored = ([x.data.copy(), x.indices.copy(), x.indptr.copy()]
              if sp.issparse(x) else [x.copy()])
    adata = _adata(x)
    tokens, values = _tokenize(token_dictionary_file_path, tokenizer,
                               count_count_norm_method, adata=adata,
                               **tokenizer_kwargs)
    expected_tokens, expected_values = _tokenize(
        token_dictionary_file_path, tokenizer, count_count_norm_method,
        adata=_adata(canonical), **tokenizer_kwargs)
    np.testing.assert_array_equal(tokens, expected_tokens)
    np.testing.assert_array_equal(values, expected_values)
    # The caller's matrix is unchanged
    assert adata.X is x
    for before, after in zip(stored, [x.data, x.indices, x.indptr]
                             if sp.issparse(x) else [x]):
        np.testing.assert_array_equal(after, before)


def test_unsorted_counts_in_a_file_tokenize_like_their_canonical_form(
        token_dictionary_file_path, tmp_path):
    canonical = _counts(np.int64)
    _adata(_unsorted(canonical)).write_h5ad(tmp_path / 'unsorted.h5ad')
    assert not ad.read_h5ad(tmp_path / 'unsorted.h5ad').X.has_sorted_indices
    from_file = _tokenize(token_dictionary_file_path, CellGraphTokenizer,
                          'shifted_log',
                          adata_file_path=tmp_path / 'unsorted.h5ad')
    expected = _tokenize(token_dictionary_file_path, CellGraphTokenizer,
                         'shifted_log', adata=_adata(canonical))
    for a, b in zip(from_file, expected):
        np.testing.assert_array_equal(a, b)


def test_only_non_canonical_matrices_are_copied():
    canonical = _counts(np.int64)
    for x in (canonical, canonical.tocsc(), canonical.toarray()):
        assert _canonical_X(x) is x
    for layout in ('unsorted', 'duplicates', 'csc_unsorted',
                   'csc_duplicates'):
        x = LAYOUTS[layout](canonical)
        stored = [x.data.copy(), x.indices.copy(), x.indptr.copy()]
        result = _canonical_X(x)
        assert result.has_canonical_format
        assert result.format == x.format
        np.testing.assert_array_equal(result.toarray(), canonical.toarray())
        # The input matrix is unchanged
        assert not x.has_canonical_format
        for before, after in zip(stored, [x.data, x.indices, x.indptr]):
            np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.float64])
@pytest.mark.parametrize('aggregate_own_neighbors', [False, True])
def test_neighborhood_values_are_the_shifted_log_of_the_aggregated_counts(
        token_dictionary_file_path, aggregate_own_neighbors, dtype):
    counts = _counts(dtype)
    tk = CellNeighborhoodTokenizer(
        model_input_size=MODEL_INPUT_SIZE, n_neighs=N_NEIGHS,
        token_dictionary_file_path=token_dictionary_file_path,
        include_special_tokens=False, rank_cell_norm_method=None,
        rank_gene_norm_method=None, rank_count_norm_method=None,
        count_cell_norm_method=None, count_gene_norm_method=None,
        count_count_norm_method='shifted_log',
        aggregate_own_neighbors=aggregate_own_neighbors)
    adata_dict = tk._tokenize_adata(adata=_adata(counts))
    # Aggregated counts from the dense connectivities, independent of the
    # storage order of the sparse product
    connectivities = construct_neighbor_graph(
        _adata(counts), n_neighs=tk.n_neighs, radius=tk.radius,
        delaunay=tk.delaunay, include_self_loop=True,
    ).obsp['spatial_connectivities'].toarray()
    if not aggregate_own_neighbors:
        connectivities = connectivities.T
    for stream, x in (('cell', counts.toarray()),
                      ('neighborhood', connectivities @ counts.toarray())):
        for i in range(N_CELLS):
            genes = np.asarray(adata_dict[f'gene_tokens_{stream}'][i]) - 2
            # Every expressed gene, by descending count, with the shifted log
            # of its own count
            np.testing.assert_array_equal(np.sort(genes),
                                          np.flatnonzero(x[i]))
            assert np.all(np.diff(x[i, genes]) <= 0)
            np.testing.assert_allclose(
                adata_dict[f'gene_expr_{stream}'][i],
                np.log1p(x[i, genes].astype(np.float32)), rtol=1e-6)
