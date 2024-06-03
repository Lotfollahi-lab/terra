import anndata as ad
import numpy as np


def filter_poor_quality_cells(adata: ad.AnnData) -> ad.AnnData:
    """
    Filter cells that do not pass QC.

    Filter cells based on the "filter_pass" field in `adata.obs`.

    Parameters
    --------
    adata: ad.AnnData
        An AnnData object containing a QC field in `adata.obs["filter_pass"]`.

    Returns
    --------
    adata: ad.AnnData
        A filtered AnnData object.
    """

    if "filter_pass" not in adata.obs.columns:
        raise ValueError("`filter_pass` column not found in `adata.obs`.")

    filter_pass_idx = np.where([filter_pass == 1 for filter_pass in adata.obs["filter_pass"]])[0]

    adata_passing = adata.copy()[filter_pass_idx]
    return adata_passing
