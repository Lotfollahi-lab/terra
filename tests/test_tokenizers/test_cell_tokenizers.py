"""
Integration tests for the sparse and dense ranking paths of the cell
tokenizers on a small synthetic spatial AnnData with many tied counts.
"""

import pickle

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from terra.tokenizers import CellGraphTokenizer

N_CELLS = 40
N_GENES = 60
N_NEIGHS = 4
MODEL_INPUT_SIZE = 8 * (N_NEIGHS + 1)  # 8 genes per cell segment


@pytest.fixture
def token_dictionary_file_path(tmp_path):
    token_dict = {'<pad>': 0, '<mask>': 1}
    token_dict.update({f'ENSG{i:011d}': i + 2 for i in range(N_GENES)})
    path = tmp_path / 'token_dictionary.pkl'
    with open(path, 'wb') as f:
        pickle.dump(token_dict, f)
    return path


def _make_adata(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    # ~20 expressed genes per cell, mostly count 1-2: the top-8 cut falls
    # inside a tie group in almost every cell
    counts = rng.poisson(0.5, (N_CELLS, N_GENES)).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(counts))
    adata.var['ensembl_id'] = [f'ENSG{i:011d}' for i in range(N_GENES)]
    adata.obs['cell_id'] = [f'cell_{i}' for i in range(N_CELLS)]
    adata.obsm['spatial'] = rng.uniform(0, 100, (N_CELLS, 2))
    return adata


def _tokenize(token_dictionary_file_path, dense=False, **kwargs):
    norm_kwargs = dict(rank_cell_norm_method=None,
                       rank_gene_norm_method=None,
                       rank_count_norm_method=None,
                       count_cell_norm_method=None,
                       count_gene_norm_method=None,
                       count_count_norm_method='shifted_log')
    norm_kwargs.update(kwargs.pop('norm_kwargs', {}))
    tk = CellGraphTokenizer(
        model_input_size=MODEL_INPUT_SIZE,
        n_neighs=N_NEIGHS,
        token_dictionary_file_path=token_dictionary_file_path,
        include_special_tokens=False,
        **norm_kwargs,
        **kwargs)
    if dense:
        # Same ranking, but through the dense (lexsort) path
        tk.rank_differs_from_count = True
    adata_dict = tk._tokenize_adata(adata=_make_adata())
    return tk, adata_dict


def _expressed_prefix(tokens):
    tokens = np.asarray(tokens)
    return tokens[tokens != 0]


def test_stable_sparse_path_matches_dense_path(token_dictionary_file_path):
    _, sparse = _tokenize(token_dictionary_file_path, stable_rank_ties=True)
    _, dense = _tokenize(token_dictionary_file_path, dense=True)

    for s_tokens, d_tokens, s_expr, d_expr in zip(
            sparse['gene_tokens_cell'], dense['gene_tokens_cell'],
            sparse['gene_expr_cell'], dense['gene_expr_cell']):
        n = len(_expressed_prefix(s_tokens))
        np.testing.assert_array_equal(
            _expressed_prefix(s_tokens), _expressed_prefix(d_tokens))
        np.testing.assert_allclose(np.asarray(s_expr)[:n],
                                   np.asarray(d_expr)[:n])


def test_default_sparse_path_is_unchanged(token_dictionary_file_path):
    # Without the option the sparse path keeps numpy's default sort.
    tk, default = _tokenize(token_dictionary_file_path)
    assert tk.rank_sort_kind is None
    adata = _make_adata()
    x = adata.X.tocsr()
    token_ids = np.arange(N_GENES) + 2
    for i, tokens in enumerate(default['gene_tokens_cell']):
        row = x[i]
        expected = token_ids[row.indices][np.argsort(-row.data)]
        np.testing.assert_array_equal(np.asarray(tokens),
                                      expected[:MODEL_INPUT_SIZE])
