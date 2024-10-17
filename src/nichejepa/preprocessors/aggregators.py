import anndata as ad
import squidpy as sq


def aggregate_neighbors(adata: ad.AnnData) -> ad.AnnData:
    """
    Aggregate cell features by neighborhood radius.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates available in
        `adata.obsm['spatial']`.

    Returns
    ----------
    adata:
        AnnData object with aggregated counts available in
        `adata.layers['X_neighborhood']`.
    """
    # Compute spatial neighborhood graph with delaunay triangulation
    sq.gr.spatial_neighbors(adata,
                            coord_type='generic',
                            spatial_key='spatial',
                            delaunay=True,
                            set_diag=True)

    adata.layers['X_neighborhood'] = (
        adata.obsp['spatial_connectivities'].T @ adata.X)

    return adata
