from typing import Optional

import anndata as ad
import squidpy as sq


def aggregate_neighbors(adata: ad.AnnData,
                        radius: Optional[float]=None,
                        delaunay_radius_union: bool=False,
                        include_self_loop: bool=False) -> ad.AnnData:
    """
    Aggregate cell features by neighborhood radius.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates available in
        `adata.obsm['spatial']`.
    radius:
        If specified, use `radius` to compute the neighborhood graph, else use
        delaunay triangulation.
    delaunay_radius_union:
        If 'True', compute the neighborhood graph by delaunay triangulation but
        exclude observations that are outside of the radius with size `radius`.
    include_self_loop:
        If 'True' include cell itself in neighborhood.

    Returns
    ----------
    adata:
        AnnData object with aggregated counts available in
        `adata.layers['X_neighborhood']`.
    """
    if delaunay_radius_union and (radius is None):
        raise ValueError(
            'Radius needs to be specified if `delaunay_radius_union` is True.')

    if radius is not None:
        # Compute spatial neighborhood graph with radius
        sq.gr.spatial_neighbors(adata,
                                coord_type='generic',
                                spatial_key='spatial',
                                radius=radius,
                                set_diag=include_self_loop,
                                )
        radius_connectivities = adata.obsp['spatial_connectivities']
    
    if (radius is None) or delaunay_radius_union:
        # Compute spatial neighborhood graph with delaunay triangulation
        sq.gr.spatial_neighbors(adata,
                                coord_type='generic',
                                spatial_key='spatial',
                                delaunay=True,
                                set_diag=include_self_loop)

        if delaunay_radius_union:
            adata.obsp[
                'spatial_connectivities'] = radius_connectivities.multiply(
                    adata.obsp['spatial_connectivities'])
        
    adata.layers['X_neighborhood'] = (
        adata.obsp['spatial_connectivities'].T @ adata.X)

    return adata
