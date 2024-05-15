"""
Normalisation methods to apply to a sparse matrix
"""
from __future__ import annotations

import anndata
import scipy
import numpy as np
from pathlib import Path
import pickle
import scanpy as sc


def analytic_pearson_residuals(x: scipy.sparse.csr_matrix, theta=100) -> scipy.sparse.csr_matrix:
    """Normalise using analytic pearson residuals as described in _____"""

    sum_counts_cells = np.sum(x, axis=1).reshape(-1, 1)
    sum_counts_genes = np.sum(x, axis=0).reshape(1, -1)
    sum_counts_total = np.sum(sum_counts_genes)
    mu_counts = np.array(sum_counts_cells @ sum_counts_genes / sum_counts_total)
    diff_counts = np.array(x - mu_counts)
    residuals_counts = diff_counts / np.sqrt(mu_counts + mu_counts ** 2 / theta)
    y = np.clip(residuals_counts, a_min=-np.sqrt(x.shape[0]), a_max=np.sqrt(x.shape[0]))

    return y


def read_depth(x: scipy.sparse.csr_matrix, target_size=10_000) -> scipy.sparse.csr_matrix:
    """Normalise by read depth"""

    y = x / x.sum(axis=1).reshape(-1, 1) * target_size

    return y


def cell_area(adata: anndata.AnnData) -> anndata.AnnData:
    """Normalise by area"""

    # Normalize cell counts
    adata.X = adata.X / adata.obs["area"].values.reshape(-1, 1) * np.mean(adata.obs["area"])

    # Compute neighborhood area
    adata.obs["neighborhood_area"] = np.array(adata.obsp["spatial_connectivities"].T @
                                              adata.obs["area"].values.reshape(-1, 1))

    # Normalize neighborhood counts
    adata.layers["X_neighborhood"] = (adata.layers["X_neighborhood"] /
                                      adata.obs["neighborhood_area"].values.reshape(-1, 1) *
                                      np.mean(adata.obs["neighborhood_area"])
                                      )

    return adata


def seurat_v3(
    adata: anndata.AnnData,
    cell_gene_means_file: Path | str,
    cell_gene_reg_stds_file: Path | str,
    neighborhood_gene_means_file: Path | str,
    neighborhood_gene_reg_stds_file: Path | str
) -> anndata.AnnData:
    """Normalise by seurat v3"""

    # Load dictionaries of cell gene means and reg stds
    with open(cell_gene_means_file, "rb") as f:
        cell_gene_means_dict = pickle.load(f)
    with open(cell_gene_reg_stds_file, "rb") as f:
        cell_gene_reg_stds_dict = pickle.load(f)

    # Load dictionaries of neighborhood gene means and reg stds
    with open(neighborhood_gene_means_file, "rb") as f:
        neighborhood_gene_means_dict = pickle.load(f)
    with open(neighborhood_gene_reg_stds_file, "rb") as f:
        neighborhood_gene_reg_stds_dict = pickle.load(f)

    # Retrieve cell and neighborhood gene means and reg stds
    cell_gene_means = np.array([cell_gene_means_dict[gene_id] for gene_id in adata.var["ensembl_id"]])
    cell_gene_reg_stds = np.array(
        [cell_gene_reg_stds_dict[gene_id] for gene_id in adata.var["ensembl_id"]]
    )
    neighborhood_gene_means = np.array(
        [neighborhood_gene_means_dict[gene_id] for gene_id in adata.var["ensembl_id"]]
    )
    neighborhood_gene_reg_stds = np.array(
        [neighborhood_gene_reg_stds_dict[gene_id] for gene_id in adata.var["ensembl_id"]]
    )

    # Normalize cell and neighborhood counts
    adata.X = adata.X - cell_gene_means / cell_gene_reg_stds
    adata.layers["X_neighborhood"] = (
        adata.layers["X_neighborhood"] - neighborhood_gene_means / neighborhood_gene_reg_stds
    )

    return adata


def mean(
    adata: anndata.AnnData,
    cell_gene_means_file: Path | str,
    neighborhood_gene_means_file: Path | str
) -> anndata.AnnData:
    """Normalise by mean"""

    # Load dictionaries of cell gene means
    with open(cell_gene_means_file, "rb") as f:
        cell_gene_means_dict = pickle.load(f)

    # Load dictionaries of neighborhood gene means
    with open(neighborhood_gene_means_file, "rb") as f:
        neighborhood_gene_means_dict = pickle.load(f)

    # Retrieve cell and neighborhood gene means
    cell_gene_means = np.array([cell_gene_means_dict[gene_id] for gene_id in adata.var["ensembl_id"]])
    neighborhood_gene_means = np.array(
        [neighborhood_gene_means_dict[gene_id] for gene_id in adata.var["ensembl_id"]]
    )

    # Normalize cell and neighborhood counts
    adata.X = adata.X / cell_gene_means
    adata.layers["X_neighborhood"] = adata.layers["X_neighborhood"] / neighborhood_gene_means

    return adata


def non_zero_median(
    adata: anndata.AnnData,
    cell_gene_nzmedians_file: Path | str,
    neighborhood_gene_nzmedians_file: Path | str,
) -> anndata.AnnData:
    """Normalise by non-zero median"""

    # Load dictionaries of cell gene non-zero medians
    with open(cell_gene_nzmedians_file, "rb") as f:
        cell_gene_nzmedians_dict = pickle.load(f)

    # Load dictionaries of neighborhood gene non-zero medians
    with open(neighborhood_gene_nzmedians_file, "rb") as f:
        neighborhood_gene_nzmedians_dict = pickle.load(f)

    # Retrieve cell and neighborhood gene non-zero medians
    cell_gene_nzmedians = np.array([cell_gene_nzmedians_dict[gene_id] for gene_id in adata.var["ensembl_id"]])
    neighborhood_gene_nzmedians = np.array(
        [neighborhood_gene_nzmedians_dict[gene_id] for gene_id in adata.var["ensembl_id"]]
    )

    # Normalize cell and neighborhood counts
    adata.X = adata.X / cell_gene_nzmedians
    adata.layers["X_neighborhood"] = adata.layers["X_neighborhood"] / neighborhood_gene_nzmedians

    return adata


def shifted_log(adata: anndata.AnnData) -> anndata.AnnData:
    """Normalise by log1p"""
    sc.pp.log1p(adata)
    sc.pp.log1p(adata, layer="X_neighborhood")
    return adata
