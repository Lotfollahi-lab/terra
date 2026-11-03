"""
Test for the cell-area count normalization of the cell-neighborhood tokenizer.

With ``count_cell_norm_method='cell_area'`` the neighborhood count values were
computed from the index cell's counts (``adata.X``) instead of the aggregated
neighborhood counts (``adata.layers['X_neighborhood']``), unlike the rank
branch, so the neighborhood segment carried the index cell's values.
"""

import pickle

import anndata as ad
import numpy as np
import scipy.sparse as sp

from terra.preprocessors.graph import construct_neighbor_graph
from terra.preprocessors.normalizers import normalize_by_cell_area
from terra.tokenizers import CellNeighborhoodTokenizer

N_CELLS = 30
N_GENES = 40
N_NEIGHS = 4


def test_neighborhood_counts_are_normalized_by_neighborhood_cell_area(
        tmp_path):
    rng = np.random.default_rng(0)
    token_dict = {'<pad>': 0, '<mask>': 1}
    token_dict.update({f'ENSG{i:011d}': i + 2 for i in range(N_GENES)})
    token_dictionary_file_path = tmp_path / 'token_dictionary.pkl'
    with open(token_dictionary_file_path, 'wb') as f:
        pickle.dump(token_dict, f)
    adata = ad.AnnData(X=sp.csr_matrix(
        rng.poisson(1.0, (N_CELLS, N_GENES)).astype(np.float32)))
    adata.var['ensembl_id'] = [f'ENSG{i:011d}' for i in range(N_GENES)]
    adata.obs['cell_id'] = [f'cell_{i}' for i in range(N_CELLS)]
    adata.obs['cell_area'] = rng.uniform(50, 150, N_CELLS)
    adata.obs['neighborhood_cell_area'] = rng.uniform(250, 750, N_CELLS)
    adata.obsm['spatial'] = rng.uniform(0, 100, (N_CELLS, 2))

    tk = CellNeighborhoodTokenizer(
        model_input_size=2 * N_GENES,
        n_neighs=N_NEIGHS,
        token_dictionary_file_path=token_dictionary_file_path,
        include_special_tokens=False,
        rank_cell_norm_method=None,
        rank_gene_norm_method=None,
        rank_count_norm_method=None,
        count_cell_norm_method='cell_area',
        count_gene_norm_method=None,
        count_count_norm_method=None)
    adata_dict = tk._tokenize_adata(adata=adata)

    graph = construct_neighbor_graph(
        adata.copy(), n_neighs=N_NEIGHS, radius=None, delaunay=False,
        include_self_loop=True, compute_neighbor_counts=True)
    expected = normalize_by_cell_area(
        graph.layers['X_neighborhood'],
        cell_areas=adata.obs['neighborhood_cell_area'].values).toarray()
    n_checked = 0
    for i in range(N_CELLS):
        tokens = np.asarray(adata_dict['gene_tokens_neighborhood'][i])
        values = np.asarray(adata_dict['gene_expr_neighborhood'][i])
        genes = tokens[tokens != 0] - 2
        np.testing.assert_allclose(
            values[tokens != 0], expected[i, genes], rtol=1e-6)
        n_checked += len(genes)
    assert n_checked > 0
