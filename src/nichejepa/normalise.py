from __future__ import annotations
import anndata
import numpy as np
import pickle
from pathlib import Path
import scanpy as sc
import scipy


def analytic_pearson_residuals(x: scipy.sparse.csr_matrix, theta=100) -> scipy.sparse.csr_matrix:
    """
    Normalise using analytic pearson residuals

    Implements normalisation as described in "Lause, J., Berens, P. & Kobak, D. Analytic Pearson
    residuals for normalisation of single-cell RNA-seq UMI data. Genome Biol. 22, 258 (2021)". Residuals are
    based on a negative binomial offset model with overdispersion shared across genes. Residuals are
    clipped to 'sqrt(n_obs)'. Negative residuals for a cell and gene indicate that fewer counts are observed than
    expected, compared to the gene’s average expression and read depth. Positive residuals indicate more
    counts than expected.

    Parameters
    ----------
    x : scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature
    theta : float
        The overdispersion parameter

    Returns
    ----------
    scipy.sparse.csr_matrix
        A sparse matrix containing the normalised features
    """

    sum_counts_cells = np.sum(x, axis=1).reshape(-1, 1)
    sum_counts_genes = np.sum(x, axis=0).reshape(1, -1)
    sum_counts_total = np.sum(sum_counts_genes)
    mu_counts = np.array(sum_counts_cells @ sum_counts_genes / sum_counts_total)
    diff_counts = np.array(x - mu_counts)
    residuals_counts = diff_counts / np.sqrt(mu_counts + mu_counts ** 2 / theta)
    y = np.clip(residuals_counts, a_min=-np.sqrt(x.shape[0]), a_max=np.sqrt(x.shape[0]))

    return y


def read_depth(x: scipy.sparse.csr_matrix, target_size: int = 10_000) -> scipy.sparse.csr_matrix:
    """
    Normalise by read depth

    Parameters
    ----------
    x : scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature
    target_size : int
        The target read depth per observation (i.e. the sum of features across an observation)

    Returns
    ----------
    scipy.sparse.csr_matrix
        A sparse matrix containing the normalised features
    """

    y = x / x.sum(axis=1).reshape(-1, 1) * target_size

    return y


def cell_area(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Normalise by cell area

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing cell area measurements in `adata.obs["area"]`, and
        aggregated neighbourhood counts in `adata.layers["X_neighborhood"]`.

    Returns
    ----------
    adata : anndata.AnnData
        An AnnData object containing the normalised features
    """

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
    """
    Normalise using seurat v3

    Implements normalization as described in "Stuart, T. et al. Comprehensive Integration of Single-Cell Data. Cell
    177, 1888–1902.e21 (2019)". This function should be applied following normalization by cell area, mean or non-zero
    median. Subtraction of means and division by expected standard deviations derived from learned global mean-variance
    relationships.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing aggregated neighbourhood counts in `adata.layers["X_neighborhood"]`.
    cell_gene_means_file : Path | str
        Path to pickle file containing dictionary of mean gene expression values.
    cell_gene_reg_stds_file : Path | str
        Path to pickle file containing dictionary of regularizing standard deviation values.
    neighborhood_gene_means_file : Path | str
        Path to pickle file containing dictionary of mean gene expression values calculated for neighbourhoods.
    neighborhood_gene_reg_stds_file : Path | str
        Path to pickle file containing dictionary of regularizing standard deviation values calculated for
        neighbourhoods.


    Returns
    ----------
    anndata.AnnData
        An AnnData object containing the normalised features
    """

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
    """
    Normalise by mean expression

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing aggregated neighbourhood counts in `adata.layers["X_neighborhood"]`.
    cell_gene_means_file : Path | str
        Path to pickle file containing dictionary of mean gene expression values.
    neighborhood_gene_means_file : Path | str
        Path to pickle file containing dictionary of mean gene expression values calculated for neighbourhoods.

    Returns
    ----------
    anndata.AnnData
        An AnnData object containing the normalised features
    """

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
    """
    Normalise by non-zero median expression

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing aggregated neighbourhood counts in `adata.layers["X_neighborhood"]`.
    cell_gene_nzmedians_file : Path | str
        Path to pickle file containing dictionary of non-zero median gene expression values.
    neighborhood_gene_nzmedians_file : Path | str
        Path to pickle file containing dictionary of non-zero median gene expression values calculated for
        neighbourhoods.

    Returns
    ----------
    anndata.AnnData
        An AnnData object containing the normalised features
    """

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


def shifted_log(x: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """
    Normalise by shifted log

    Implements normalisation using `sc.pp.log1p`.

    Parameters
    ----------
    x : scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature

    Returns
    ----------
    scipy.sparse.csr_matrix
        A sparse matrix containing the normalised features
    """

    y = sc.pp.log1p(x)
    return y
