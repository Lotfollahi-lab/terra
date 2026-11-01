"""
Integration tests for the sparse and dense ranking paths of the cell
tokenizers on a small synthetic spatial AnnData with many tied counts.
"""

import pickle

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from terra.tokenizers import CellGraphTokenizer, CellNeighborhoodTokenizer

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


def _make_adata(seed: int = 0, lam: float = 0.5) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    # lam=0.5: ~25 expressed genes per cell, mostly count 1-2, so the top-8
    # cut falls inside a tie group in almost every cell
    counts = rng.poisson(lam, (N_CELLS, N_GENES)).astype(np.float32)
    # Every cell expresses at least one gene (cells without counts would be
    # empty segments on the sparse path but full zero segments on the dense
    # path)
    empty = counts.sum(axis=1) == 0
    counts[empty, rng.integers(0, N_GENES, empty.sum())] = 1
    adata = ad.AnnData(X=sp.csr_matrix(counts))
    adata.var['ensembl_id'] = [f'ENSG{i:011d}' for i in range(N_GENES)]
    adata.obs['cell_id'] = [f'cell_{i}' for i in range(N_CELLS)]
    adata.obsm['spatial'] = rng.uniform(0, 100, (N_CELLS, 2))
    return adata


def _tokenize(token_dictionary_file_path, dense=False, adata=None,
              **kwargs):
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
    adata_dict = tk._tokenize_adata(
        adata=_make_adata() if adata is None else adata)
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
    assert not tk.stable_rank_ties
    adata = _make_adata()
    x = adata.X.tocsr()
    token_ids = np.arange(N_GENES) + 2
    for i, tokens in enumerate(default['gene_tokens_cell']):
        row = x[i]
        expected = token_ids[row.indices][np.argsort(-row.data)]
        np.testing.assert_array_equal(np.asarray(tokens),
                                      expected[:MODEL_INPUT_SIZE])


@pytest.mark.parametrize('tokenizer', [CellGraphTokenizer,
                                       CellNeighborhoodTokenizer])
@pytest.mark.parametrize('dense', [False, True])
def test_stable_rank_ties_do_not_depend_on_gene_order(
        token_dictionary_file_path, tokenizer, dense):
    adata = _make_adata()
    permuted = adata[:, np.random.default_rng(1).permutation(N_GENES)].copy()
    # As if the genes had been stored in this order (sorted CSR indices)
    permuted.X = sp.csr_matrix(permuted.X.toarray())
    keys = ('gene_tokens_cell', 'gene_expr_cell', 'gene_tokens_neighborhood',
            'gene_expr_neighborhood', 'seg_tokens_neighborhood')
    results = []
    for data in (adata, permuted):
        tk = tokenizer(
            model_input_size=MODEL_INPUT_SIZE, n_neighs=N_NEIGHS,
            token_dictionary_file_path=token_dictionary_file_path,
            include_special_tokens=False, rank_cell_norm_method=None,
            rank_gene_norm_method=None, rank_count_norm_method=None,
            count_cell_norm_method=None, count_gene_norm_method=None,
            count_count_norm_method='shifted_log', stable_rank_ties=True)
        # Same ranking through the dense (lexsort) path
        tk.rank_differs_from_count = dense
        adata_dict = tk._tokenize_adata(adata=data)
        results.append([
            tk._format_examples({key: adata_dict[key][i] for key in keys
                                 if key in adata_dict})
            for i in range(N_CELLS)])
    for example, permuted_example in zip(*results):
        np.testing.assert_array_equal(example['gene_tokens'],
                                      permuted_example['gene_tokens'])
        np.testing.assert_array_equal(example['gene_expr'],
                                      permuted_example['gene_expr'])


RESIDUAL_RANKS = {
    'pearson': dict(rank_count_norm_method='analytic_pearson_residuals'),
    'pearson_rank_and_count': dict(
        rank_count_norm_method='analytic_pearson_residuals',
        count_count_norm_method='analytic_pearson_residuals'),
    'seurat_v3': dict(rank_gene_norm_method='seurat_v3'),
}


def test_pearson_rank_with_shifted_log_counts_uses_dense_path(
        token_dictionary_file_path):
    # The sparse path requires rank and count with the same sparsity; a
    # Pearson-residual rank previously took it and raised an IndexError.
    tk, adata_dict = _tokenize(
        token_dictionary_file_path,
        norm_kwargs=RESIDUAL_RANKS['pearson'])
    assert tk.rank_differs_from_count
    assert len(adata_dict['gene_tokens_cell']) == N_CELLS


def _token_counts(adata):
    """Raw count of each token id per cell."""
    counts = adata.X.toarray()
    return [dict(zip(np.arange(N_GENES) + 2, row)) for row in counts]


@pytest.mark.parametrize('rank', RESIDUAL_RANKS)
def test_residual_ranks_admit_zero_count_genes_by_default(
        token_dictionary_file_path, rank):
    # Documents the previous behaviour, kept as default: zero-count genes get
    # non-zero Seurat/Pearson values and are ranked among expressed genes.
    _, adata_dict = _tokenize(token_dictionary_file_path,
                              norm_kwargs=RESIDUAL_RANKS[rank])
    token_counts = _token_counts(_make_adata())
    n_zero_count = sum(
        token_counts[i][t] == 0
        for i, tokens in enumerate(adata_dict['gene_tokens_cell'])
        for t in tokens if t != 0)
    assert n_zero_count > 0


@pytest.mark.parametrize('rank', RESIDUAL_RANKS)
def test_zero_expr_from_raw_counts_excludes_zero_count_genes(
        token_dictionary_file_path, rank):
    _, adata_dict = _tokenize(token_dictionary_file_path,
                              norm_kwargs=RESIDUAL_RANKS[rank],
                              zero_expr_from_raw_counts=True)
    token_counts = _token_counts(_make_adata())
    for i, tokens in enumerate(adata_dict['gene_tokens_cell']):
        tokens = np.asarray(tokens)
        n_expressed = sum(c > 0 for c in token_counts[i].values())
        kept = tokens[tokens != 0]
        # Only expressed genes, all of them (up to the stored length), and
        # the padding after them
        assert all(token_counts[i][t] > 0 for t in kept)
        assert len(kept) == min(n_expressed, MODEL_INPUT_SIZE)
        assert np.all(tokens[:len(kept)] != 0)


def test_n_nonzero_tokens_counts_expressed_genes_on_dense_path(
        token_dictionary_file_path):
    # Same data through the sparse and the dense path: n_nonzero_tokens must
    # count the gene tokens of the formatted example in both, not the stored
    # list length (the dense path stores the full panel padded with 0).
    results = []
    for dense in (False, True):
        # ~3 expressed genes per cell, fewer than the 8 per segment
        tk, adata_dict = _tokenize(token_dictionary_file_path, dense=dense,
                                   adata=_make_adata(lam=0.05),
                                   stable_rank_ties=True)
        n_nonzero = []
        for i in range(N_CELLS):
            example = tk._format_examples(
                {key: adata_dict[key][i] for key in (
                    'gene_tokens_cell', 'gene_expr_cell',
                    'gene_tokens_neighborhood', 'gene_expr_neighborhood',
                    'seg_tokens_neighborhood')})
            assert example['n_nonzero_tokens'] == np.count_nonzero(
                example['gene_tokens'])
            n_nonzero.append(example['n_nonzero_tokens'])
        results.append(n_nonzero)
    assert results[0] == results[1]


def test_gcrd_constants_are_configurable(token_dictionary_file_path,
                                        monkeypatch):
    # With recent scipy, sparse / dense-matrix division returns a COO matrix,
    # which recent anndata refuses as a layer; convert (unrelated to the
    # constants under test).
    from terra.tokenizers import cell_tokenizers
    gcrd_fn = cell_tokenizers.normalize_by_gene_corrected_read_depth
    monkeypatch.setattr(
        cell_tokenizers, 'normalize_by_gene_corrected_read_depth',
        lambda x, **kwargs: sp.csr_matrix(gcrd_fn(x, **kwargs)))
    gcrd = dict(rank_cell_norm_method='gene_corrected_read_depth',
                count_cell_norm_method='gene_corrected_read_depth',
                count_count_norm_method=None)
    _, default = _tokenize(token_dictionary_file_path, norm_kwargs=gcrd,
                           stable_rank_ties=True)
    _, custom = _tokenize(token_dictionary_file_path, norm_kwargs=gcrd,
                          stable_rank_ties=True,
                          gcrd_basis_target_read_depth=151.1754621380761,
                          gcrd_target_read_depth_per_gene=0.04705053555567159)
    # Default target depth 153.4768 + 0.0487 * n_genes; the custom constants
    # rescale every value by the ratio of the target depths
    ratio = ((151.1754621380761 + N_GENES * 0.04705053555567159)
             / (153.4768 + N_GENES * 0.0487))
    for d_expr, c_expr in zip(default['gene_expr_cell'],
                              custom['gene_expr_cell']):
        np.testing.assert_allclose(np.asarray(c_expr),
                                   np.asarray(d_expr) * ratio, rtol=1e-5)


def test_inference_tokenizer_reads_optional_settings(tmp_path, monkeypatch):
    # tokenize_adata passes the optional tokenizer settings recorded in the
    # model config; absent keys keep the tokenizer defaults.
    import yaml
    from terra.inference import tokenize as inference_tokenize

    captured = {}

    class _Stop(Exception):
        pass

    def _capture(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(inference_tokenize.cell_tokenizers,
                        'CellGraphTokenizer', _capture)
    data = dict(tokenizer_type='cell_graph', model_input_size=40, n_neighs=4,
                rank_cell_norm_method=None, rank_gene_norm_method=None,
                rank_count_norm_method=None, count_cell_norm_method=None,
                count_gene_norm_method=None,
                count_count_norm_method='shifted_log',
                stable_rank_ties=True, gcrd_basis_target_read_depth=151.0,
                pad_empty_neighbors=True)
    with open(tmp_path / 'model_config.yaml', 'w') as f:
        yaml.safe_dump({'data': data}, f)

    with pytest.raises(_Stop):
        inference_tokenize.tokenize_adata(
            adata=None, model_folder_path=str(tmp_path),
            cache_directory_path=str(tmp_path))
    assert captured['stable_rank_ties'] is True
    assert captured['gcrd_basis_target_read_depth'] == 151.0
    assert captured['pad_empty_neighbors'] is True
    assert 'zero_expr_from_raw_counts' not in captured
    assert 'gcrd_target_read_depth_per_gene' not in captured


def test_cell_neigh_split_follows_model_config():
    from terra.inference.tokenize import cell_neigh_split_is_equal

    data = dict(model_input_size=704, n_neighs=10)
    assert cell_neigh_split_is_equal({**data, 'seq_len_cell': 352}) is True
    # Models tokenized before the April 2026 default change: 64 / 640
    assert cell_neigh_split_is_equal({**data, 'seq_len_cell': 64}) is False
    with pytest.raises(ValueError):
        cell_neigh_split_is_equal({**data, 'seq_len_cell': 100})


@pytest.mark.parametrize('seq_len_cell, equal', [(352, True), (64, False)])
def test_inference_cell_neigh_tokenizer_uses_model_split(
        tmp_path, monkeypatch, seq_len_cell, equal):
    import yaml
    from terra.inference import tokenize as inference_tokenize

    captured = {}

    class _Stop(Exception):
        pass

    def _capture(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(inference_tokenize.cell_tokenizers,
                        'CellNeighborhoodTokenizer', _capture)
    data = dict(tokenizer_type='cell_neigh', model_input_size=704,
                n_neighs=10, seq_len_cell=seq_len_cell,
                rank_cell_norm_method=None, rank_gene_norm_method=None,
                rank_count_norm_method=None, count_cell_norm_method=None,
                count_gene_norm_method=None,
                count_count_norm_method='shifted_log')
    with open(tmp_path / 'model_config.yaml', 'w') as f:
        yaml.safe_dump({'data': data}, f)

    with pytest.raises(_Stop):
        inference_tokenize.tokenize_adata(
            adata=None, model_folder_path=str(tmp_path),
            cache_directory_path=str(tmp_path))
    assert captured['split_cell_neigh_equally'] is equal
    assert 'aggregate_own_neighbors' not in captured
