import anndata as ad
import numpy as np


def filter_cells(adata: ad.AnnData) -> ad.AnnData:
    """
    Filter cells that do not pass QC.

    Filter cells based on the 'filter_pass' field in `adata.obs`.

    Parameters
    --------
    adata:
        An AnnData object containing a QC field in
        `adata.obs['filter_pass']`.

    Returns
    --------
    adata:
        The filtered AnnData object.
    """
    if 'filter_pass' not in adata.obs.columns:
        print("No 'filter_pass' column in 'adata.obs'; returning full adata.")
        
        return adata
    else:
        filter_pass_idx = np.where(adata.obs['filter_pass'].values == 1)[0]
        # Subset FIRST, then copy, so we never materialize a full copy of the
        # entire AnnData (the previous `adata.copy()[idx]` copied every layer /
        # obsm / obsp before discarding the filtered-out cells). Same cells,
        # same order; just lower peak memory and faster.
        adata_passing = adata[filter_pass_idx].copy()

        return adata_passing