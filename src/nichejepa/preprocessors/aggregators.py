import anndata as ad
import squidpy as sq


def aggregate_neighbors(adata: ad.AnnData,
                        radius: float=27.5,
                        ) -> ad.AnnData:
    """
    Aggregate cell features by neighborhood radius.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates available in
        `adata.obsm["spatial"]`.
    radius:
        Radius within which neighboring cells will be aggregated, in um.
        Defaults to 27.5um, which corresponds to the 10x Visium spot size of
        55um.

    Returns
    ----------
    adata:
        AnnData object with aggregated counts available in
        `adata.layers["X_neighborhood"]`.
    """
    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key="spatial",
                            radius=radius,
                            set_diag=True)

    adata.layers["X_neighborhood"] = (
        adata.obsp["spatial_connectivities"].T @ adata.X)

    return adata
