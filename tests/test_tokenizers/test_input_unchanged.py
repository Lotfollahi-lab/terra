"""
Tests that the tokenizers leave the caller's AnnData unchanged.

Tokenizing an AnnData object added the neighbor graph (``obsp``, ``uns``) and,
for the cell-neighborhood tokenizer, the normalized layers to the caller's
object. With a dense float32 ``adata.X`` the shifted-log normalization also
log-transformed ``adata.X`` in place: tokenizing the same object again gave
log1p(log1p(counts)) values, and with shifted-log rank and count
normalization a single tokenization did (the rank and count layers alias
``adata.X``), unlike sparse input.
"""

import pickle

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from terra.preprocessors.normalizers import normalize_by_shifted_log
from terra.tokenizers import CellGraphTokenizer, CellNeighborhoodTokenizer

N_CELLS = 30
N_GENES = 40
N_NEIGHS = 4
NORM = dict(rank_cell_norm_method=None, rank_gene_norm_method=None,
            rank_count_norm_method=None, count_cell_norm_method=None,
            count_gene_norm_method=None,
            count_count_norm_method='shifted_log')


@pytest.fixture
def token_dictionary_file_path(tmp_path):
    token_dict = {'<pad>': 0, '<mask>': 1}
    token_dict.update({f'ENSG{i:011d}': i + 2 for i in range(N_GENES)})
    path = tmp_path / 'token_dictionary.pkl'
    with open(path, 'wb') as f:
        pickle.dump(token_dict, f)
    return path


def _adata(dense: bool) -> ad.AnnData:
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.5, (N_CELLS, N_GENES)).astype(np.float32)
    counts[:, 0] += 1  # every cell expresses a gene
    adata = ad.AnnData(X=counts if dense else sp.csr_matrix(counts))
    adata.var['ensembl_id'] = [f'ENSG{i:011d}' for i in range(N_GENES)]
    adata.obs['cell_id'] = [f'cell_{i}' for i in range(N_CELLS)]
    adata.obsm['spatial'] = rng.uniform(0, 100, (N_CELLS, 2))
    return adata


def _snapshot(adata):
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    return (X.copy(), adata.obs.copy(), sorted(adata.obsp),
            sorted(adata.uns), sorted(adata.layers))


def _assert_unchanged(adata, snapshot):
    X, obs, obsp, uns, layers = snapshot
    np.testing.assert_array_equal(
        adata.X.toarray() if sp.issparse(adata.X) else adata.X, X)
    assert adata.obs.equals(obs)
    assert sorted(adata.obsp) == obsp
    assert sorted(adata.uns) == uns
    assert sorted(adata.layers) == layers


def _same(a, b):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def _tokenizer(Tokenizer, token_dictionary_file_path, **norm):
    return Tokenizer(
        model_input_size=16 * (N_NEIGHS + 1), n_neighs=N_NEIGHS,
        token_dictionary_file_path=token_dictionary_file_path,
        include_special_tokens=False, **{**NORM, **norm})


def test_shifted_log_returns_a_new_array():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    y = normalize_by_shifted_log(x)
    np.testing.assert_array_equal(x, np.arange(6).reshape(2, 3))
    np.testing.assert_allclose(y, np.log1p(x))


@pytest.mark.parametrize('Tokenizer', [CellGraphTokenizer,
                                       CellNeighborhoodTokenizer])
@pytest.mark.parametrize('dense', [True, False])
def test_tokenizing_leaves_the_input_unchanged(
        token_dictionary_file_path, Tokenizer, dense):
    adata = _adata(dense)
    snapshot = _snapshot(adata)
    tk = _tokenizer(Tokenizer, token_dictionary_file_path)
    first = tk._tokenize_adata(adata=adata)
    _assert_unchanged(adata, snapshot)
    # Tokenizing the same object again gives the same tokens and values
    second = tk._tokenize_adata(adata=adata)
    assert first.keys() == second.keys()
    for key in first:
        assert _same(first[key], second[key]), key


@pytest.mark.parametrize('Tokenizer', [CellGraphTokenizer,
                                       CellNeighborhoodTokenizer])
def test_dense_and_sparse_input_give_the_same_values(
        token_dictionary_file_path, Tokenizer):
    # Shifted-log rank and count normalization: both layers alias adata.X
    tk = _tokenizer(Tokenizer, token_dictionary_file_path,
                    rank_count_norm_method='shifted_log')
    dense = tk._tokenize_adata(adata=_adata(dense=True))
    sparse = tk._tokenize_adata(adata=_adata(dense=False))
    for key in dense:
        if key.startswith(('gene_tokens', 'gene_expr')):
            for a, b in zip(dense[key], sparse[key]):
                np.testing.assert_allclose(a, b, rtol=1e-6)
