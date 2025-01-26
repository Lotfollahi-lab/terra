import anndata as ad
import numpy as np


def filter_cells(adata: ad.AnnData) -> ad.AnnData:
    """
    Filter cells that do not pass QC.

    Filter cells based on the 'filter_pass' field in `adata.obs`.

    Parameters
    --------
    adata:
        An AnnData object containing a QC field in `adata.obs['filter_pass']`.

    Returns
    --------
    adata:
        The filtered AnnData object.
    """
    if 'filter_pass' not in adata.obs.columns:
        print("No 'filter_pass' column in 'adata.obs'; returning full adata.")
        
        return adata
    else:
        filter_pass_idx = np.where(
            [filter_pass == 1 for filter_pass in adata.obs['filter_pass']])[0]
        adata_passing = adata.copy()[filter_pass_idx]
        
        return adata_passing
