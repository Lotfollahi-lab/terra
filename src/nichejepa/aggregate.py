import anndata
import squidpy as sq


def aggregate_by_radius(adata: anndata.AnnData, radius=27.5) -> anndata.AnnData:
    """Aggregate neighbourhood gene expression and """

    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key="spatial",
                            radius=radius,
                            set_diag=True
                            )

    adata.layers["X_neighborhood"] = adata.obsp["spatial_connectivities"].T @ adata.X

    return adata
