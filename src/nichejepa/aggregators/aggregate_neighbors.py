import anndata as ad
import numpy as np
import scipy.sparse  as sp
import squidpy as sq


def aggregate_neighbors(x: sp.csr_matrix,
                        coordinates: np.ndarray,
                        radius: float) -> sp.csr_matrix:
    """
    Aggregate cell features by neighborhood radius.

    Parameters
    ----------
    x: sp.csr_matrix
        Features for each cell.
    coordinates: np.ndarray
        An array of lists, arrays or tuples containing the x and y coordinates of each cell in um.
    radius: float
        Radius within which neighboring cells will be aggregated, in um. Use 27.5 um for a radius equivalent to the 10x
        Visium spot size.

    Returns
    ----------
    y: sp.csr_matrix
        A feature matrix with aggregated counts.
    """

    if x.shape[0] != coordinates.shape[0]:
        raise ValueError("x and coordinates should be the same length.")

    adata = ad.AnnData(x.toarray())
    adata.obsm["spatial"] = coordinates

    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key="spatial",
                            radius=radius,
                            set_diag=True)

    y = adata.obsp["spatial_connectivities"].T @ adata.X
    y = sp.csr_matrix(y)

    return y
