import anndata
import numpy as np


def filter_poor_quality_cells(adata: anndata.AnnData) -> anndata.AnnData:
    """Filter cells that do not pass QC."""

    if "filter_pass" in adata.obs.columns:
        filter_pass_idx = np.where([filter_pass == 1 for filter_pass in adata.obs["filter_pass"]])[0]
        return adata[filter_pass_idx]

    return adata

