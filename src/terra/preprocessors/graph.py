import logging

import anndata as ad
import scipy.sparse as sp

logger = logging.getLogger(__name__)

try:
    import squidpy as sq
except:
    logger.warning("Could not import squidpy...")


def _combine_neighbor_graphs(adata: ad.AnnData,
                             n_neighs: int | None,
                             radius: float | None,
                             delaunay: bool,
                             include_self_loop: bool,
                             batch: str | None = None,
                             ) -> ad.AnnData:
    """
    Helper function to combine neighbor graphs computed using different
    methods. `batch` names the data in error messages.
    """
    if n_neighs is not None and adata.n_obs <= n_neighs:
        # Otherwise scikit-learn fails with "Expected n_neighbors <
        # n_samples_fit", which names neither the cause nor the data
        raise ValueError(
            f"Cannot compute the {n_neighs} nearest neighbors of each cell: "
            f"the data{'' if batch is None else f' of batch {batch!r}'} "
            f"has {adata.n_obs} cells, but at least {n_neighs + 1} are "
            "needed. Remove samples with too few cells.")
    if n_neighs is not None:
        # Compute knn spatial neighborhood graph
        sq.gr.spatial_neighbors(adata,
                                coord_type='generic',
                                spatial_key='spatial',
                                n_neighs=n_neighs,
                                set_diag=include_self_loop,
                                ) 
        knn_conn = adata.obsp['spatial_connectivities']  
    if radius is not None:
        # Compute spatial neighborhood graph with radius
        sq.gr.spatial_neighbors(adata,
                                coord_type='generic',
                                spatial_key='spatial',
                                radius=radius,
                                set_diag=include_self_loop,
                                )
        radius_conn = adata.obsp['spatial_connectivities'] 
        if n_neighs is not None:
            adata.obsp[
                'spatial_connectivities'] = knn_conn.multiply(
                    adata.obsp['spatial_connectivities'])
    if delaunay:
        # Compute spatial neighborhood graph with delaunay triangulation
        sq.gr.spatial_neighbors(adata,
                                coord_type='generic',
                                spatial_key='spatial',
                                delaunay=True,
                                set_diag=include_self_loop,
                                )
        if n_neighs is not None:
            adata.obsp[
                'spatial_connectivities'] = knn_conn.multiply(
                    adata.obsp['spatial_connectivities'])
        if radius is not None:
            adata.obsp[
                'spatial_connectivities'] = radius_conn.multiply(
                    adata.obsp['spatial_connectivities'])

    return adata


def construct_neighbor_graph(adata: ad.AnnData,
                             n_neighs: int | None = None,
                             radius: float | None = None,
                             delaunay: bool = True,
                             include_self_loop: bool = False,
                             batch_key: str | None = None,
                             compute_neighbor_counts: bool = False,
                             aggregate_own_neighbors: bool = False,
                             ) -> ad.AnnData:
    """
    Compute neighbor graph and optionally aggregate cell features across
    neighbors.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates available in
        `adata.obsm['spatial']`.
    n_neighs:
        If specified, use `n_neighs` to compute the neighborhood graph.
        If 'radius' or 'delaunay' are also specified, an intersection
        neighborhood graph will be computed.
    radius:
        If specified, use `radius` to compute the neighborhood graph. If
        'n_neighs' or 'delaunay' are also specified, an intersection
        neighborhood graph will be computed.
    delaunay:
        If 'True', compute the neighborhood graph by delaunay
        triangulation. If 'n_neighs' or 'radius' are also specified, an
        intersection neighborhood graph will be computed.
    include_self_loop:
        If 'True', include cell itself in neighborhood graph.
    batch_key:
        Key in adata.obs where the batch is stored in case the AnnData
        contains multiple batches. This is important to specify if there
        are multiple batches in the AnnData to perform separate neighbor
        graph computation and then concatenate.
    compute_neighbor_counts:
        If 'True', aggregate counts across neighborhood and store in
        `adata.layers['X_neighborhood']`.
    aggregate_own_neighbors:
        If 'True', aggregate the counts of each cell's own neighbors
        (`C @ X`, row i of the connectivities `C` holds cell i's neighbors).
        Default 'False' keeps the previous aggregation `C.T @ X`, which sums
        the cells that have cell i among THEIR neighbors. The two only differ
        for asymmetric graphs, i.e. kNN graphs (`n_neighs`): with `C.T`, an
        isolated cell aggregates only itself and a cell in a dense region
        aggregates more than `n_neighs` cells.

    Returns
    ----------
    adata:
        AnnData object with aggregated counts available in
        `adata.layers['X_neighborhood']`.
    """
    if batch_key is not None:
        # Get ordered batches
        first_idx = adata.obs.reset_index().groupby(batch_key).head(1).index
        batches = adata.obs.iloc[first_idx][batch_key].tolist()
        adata_batch_list = []

        for batch in batches:
            logger.info(f"\nProcessing batch {batch}...")
            logger.info("Loading data...")
            adata_batch = adata[adata.obs[batch_key] == batch]

            logger.info("Computing spatial neighborhood graph...")
            adata_batch = _combine_neighbor_graphs(
                adata_batch,
                n_neighs,
                radius,
                delaunay,
                include_self_loop,
                batch=batch)
            adata_batch_list.append(adata_batch)

        # Block-diagonal concatenate graphs (disconnected components)
        adata.obsp['spatial_connectivities'] = sp.block_diag(
            [a.obsp['spatial_connectivities'] for a in adata_batch_list],
            format="csr",
        )    
    else:
        adata = _combine_neighbor_graphs(
            adata,
            n_neighs,
            radius,
            delaunay,
            include_self_loop)           
    
    if compute_neighbor_counts:
        connectivities = adata.obsp['spatial_connectivities']
        if aggregate_own_neighbors:
            adata.layers['X_neighborhood'] = connectivities @ adata.X
        else:
            adata.layers['X_neighborhood'] = connectivities.T @ adata.X

    return adata