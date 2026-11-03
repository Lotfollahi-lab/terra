"""
Tests for neighbor graphs of data with too few cells.

A k-nearest-neighbor graph needs more than k cells. With fewer cells
scikit-learn failed with "Expected n_neighbors < n_samples_fit", which names
neither the cause nor the data; ``construct_neighbor_graph`` now says how many
cells there are and how many are needed (and names the batch).
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from terra.preprocessors.graph import construct_neighbor_graph


def _adata(n_cells, batches=None):
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=np.ones((n_cells, 5), dtype=np.float32))
    adata.obsm['spatial'] = rng.uniform(0, 100, size=(n_cells, 2))
    if batches is not None:
        adata.obs['batch'] = pd.Categorical(batches)
    return adata


@pytest.mark.parametrize('n_cells', [3, 4])
def test_too_few_cells_for_the_neighbor_graph_raise_a_clear_error(n_cells):
    with pytest.raises(ValueError, match=(
            rf'4 nearest neighbors.*has {n_cells} cells, but at least 5')):
        construct_neighbor_graph(
            _adata(n_cells), n_neighs=4, radius=None, delaunay=False)


def test_too_few_cells_name_the_batch():
    adata = _adata(9, batches=['big'] * 6 + ['small'] * 3)
    with pytest.raises(ValueError, match="batch 'small' has 3 cells"):
        construct_neighbor_graph(adata, n_neighs=4, radius=None,
                                 delaunay=False, batch_key='batch')


def test_one_more_cell_than_neighbors_is_enough():
    adata = construct_neighbor_graph(
        _adata(5), n_neighs=4, radius=None, delaunay=False)
    np.testing.assert_array_equal(
        np.diff(adata.obsp['spatial_connectivities'].indptr), 4)
