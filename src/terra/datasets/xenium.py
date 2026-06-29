"""Convenience reader for public 10x Genomics Xenium samples.

Used by the tutorials so they are self-contained: it downloads the two small
standalone output files for a Xenium sample and assembles an :class:`AnnData`
with single-cell spatial coordinates — no manual download or the multi-GB
output bundle required.
"""
import os
import shutil
import urllib.request

__all__ = ["read_xenium_10x"]


def read_xenium_10x(base_url: str, out_dir: str):
    """Download a 10x Genomics Xenium sample and load it as an ``AnnData``.

    Fetches the two small standalone output files
    (``{base_url}_cell_feature_matrix.h5`` and ``{base_url}_cells.parquet`` —
    tens of MB, not the multi-GB output bundle), caches them in ``out_dir``,
    and returns an ``AnnData`` with raw counts in ``.X`` (gene-expression
    features only) and single-cell spatial coordinates in ``.obsm['spatial']``.

    Parameters
    ----------
    base_url
        Common prefix of the sample's files on the 10x CDN, e.g.
        ``"https://cf.10xgenomics.com/samples/xenium/3.0.0/"``
        ``"Xenium_Prime_Human_Skin_FFPE/Xenium_Prime_Human_Skin_FFPE"``.
    out_dir
        Local directory used to cache the downloaded files. Existing files are
        reused rather than re-downloaded.

    Returns
    -------
    adata : anndata.AnnData
        Cells x genes, with ``.obsm['spatial']`` holding the
        ``[x_centroid, y_centroid]`` coordinates.
    """
    # Imported lazily so that ``import terra`` does not require scanpy/pandas to
    # be present just to introspect the package (e.g. on the docs builder).
    import pandas as pd
    import scanpy as sc

    os.makedirs(out_dir, exist_ok=True)
    for fname in ("cell_feature_matrix.h5", "cells.parquet"):
        dest = os.path.join(out_dir, fname)
        if not os.path.exists(dest):
            # 10x's CDN returns 403 to the default urllib User-Agent, so send a
            # browser-like one.
            request = urllib.request.Request(
                f"{base_url}_{fname}", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request) as response, open(dest, "wb") as fh:
                shutil.copyfileobj(response, fh)

    adata = sc.read_10x_h5(os.path.join(out_dir, "cell_feature_matrix.h5"))
    adata.var_names_make_unique()
    # Drop control probes / blank codewords; keep real gene-expression features.
    adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()

    cells = pd.read_parquet(os.path.join(out_dir, "cells.parquet"))
    # The matrix barcodes are strings, but ``cell_id`` in cells.parquet is an
    # integer for older Xenium panels (and a string for newer ones). Cast both
    # join keys to str so the centroids align across Xenium versions; otherwise
    # the join finds no overlap and every coordinate comes back NaN.
    cells["cell_id"] = cells["cell_id"].astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs = adata.obs.join(cells.set_index("cell_id"))
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy()
    return adata
