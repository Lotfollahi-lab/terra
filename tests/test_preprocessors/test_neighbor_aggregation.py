"""
Tests for the neighborhood count aggregation of ``construct_neighbor_graph``.

squidpy's kNN connectivities hold cell i's neighbors in row i and are not
symmetric. ``C.T @ X`` (the default, kept for parity) sums the cells that have
cell i among their neighbors; ``aggregate_own_neighbors=True`` sums cell i's
own neighbors (``C @ X``).
"""

import anndata as ad
import numpy as np
import scipy.sparse as sp

from terra.preprocessors import construct_neighbor_graph


def _adata():
    # Four clustered cells and one far-away cell: with 2 neighbors, the far
    # cell's neighbors are cluster cells, but it is nobody's neighbor.
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [100., 0.]])
    return ad.AnnData(X=sp.csr_matrix(np.eye(5)), obsm={'spatial': coords})


def _aggregate(**kwargs):
    adata = construct_neighbor_graph(
        _adata(), n_neighs=2, delaunay=False, include_self_loop=True,
        compute_neighbor_counts=True, **kwargs)
    conn = adata.obsp['spatial_connectivities'].toarray()
    return conn, np.asarray(adata.layers['X_neighborhood'].todense())


def test_default_aggregates_reverse_neighbors():
    conn, x_neigh = _aggregate()
    assert not np.array_equal(conn, conn.T)  # kNN graph is asymmetric
    np.testing.assert_array_equal(x_neigh, conn.T)
    # The far cell aggregates only itself
    np.testing.assert_array_equal(x_neigh[4], [0, 0, 0, 0, 1])


def test_aggregate_own_neighbors():
    conn, x_neigh = _aggregate(aggregate_own_neighbors=True)
    np.testing.assert_array_equal(x_neigh, conn)
    # Every cell aggregates itself and its 2 nearest neighbors
    np.testing.assert_array_equal(x_neigh.sum(axis=1), np.full(5, 3))
    np.testing.assert_array_equal(x_neigh[4], [0, 0, 1, 1, 1])
