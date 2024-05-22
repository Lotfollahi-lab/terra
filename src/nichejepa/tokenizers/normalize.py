from __future__ import annotations
import numpy as np
import pickle
from pathlib import Path
import scanpy as sc
import scipy


def analytic_pearson_residuals(x: scipy.sparse.csr_matrix, theta: float = 100) -> scipy.sparse.csr_matrix:
    """
    Normalize gene counts per gene and cell using analytic pearson residuals.

    Implements normalization as described in "Lause, J., Berens, P. & Kobak, D. Analytic Pearson residuals for
    normalization of single-cell RNA-seq UMI data. Genome Biol. 22, 258 (2021)". Residuals are based on a negative
    binomial offset model with overdispersion shared across genes. Residuals are clipped to 'sqrt(n_obs)'. Negative
    residuals for a cell and gene indicate that fewer counts are observed than expected, compared to the gene’s average
    expression and cell read depth. Positive residuals indicate more counts than expected.
    
    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/experimental/pp/_normalization.py#L36
    (03.05.2024).

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    theta: float
        The overdispersion parameter.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
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
    Normalize gene counts per cell by read depth.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    target_size: int
        The target read depth per observation (i.e. the sum of features across an observation).

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    y = x / x.sum(axis=1).reshape(-1, 1) * target_size

    return y


def cell_area(x: scipy.sparse.csr_matrix,
              cell_areas: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Normalize gene counts per cell by cell area.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    cell_areas: np.ndarray
        Numpy array with the cell areas.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    y = x / cell_areas.values.reshape(-1, 1) * np.mean(cell_areas)

    return y


def seurat_v3(x: scipy.sparse.csr_matrix,
              gene_means_file: Path | str,
              gene_reg_stds_file: Path | str,
              probed_genes: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Normalize gene counts per gene using seurat v3.

    Implements normalization as described in "Stuart, T. et al. Comprehensive Integration of Single-Cell Data. Cell 177,
    1888–1902.e21 (2019)". This function should be applied following normalization per cell by read depth or cell area.
    Subtraction of means and division by expected standard deviations derived from learned global mean-variance
    relationships.

    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/preprocessing/_highly_variable_genes.py#L26
    (29.04.2024).

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_means_file: Path | str
        Path to pickle file containing dictionary of mean gene expression values.
    gene_reg_stds_file: Path | str
        Path to pickle file containing dictionary of regularizing standard deviation values.
    probed_genes:
        Array with ensembl ids of probed genes.
     
    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    # Load dictionaries of gene means and reg stds
    with open(gene_means_file, "rb") as f:
        gene_means_dict = pickle.load(f)
    with open(gene_reg_stds_file, "rb") as f:
        gene_reg_stds_dict = pickle.load(f)

    # Retrieve gene means and reg stds for probed genes
    gene_means = np.array([gene_means_dict[gene_id] for gene_id in probed_genes])
    gene_reg_stds = np.array([gene_reg_stds_dict[gene_id] for gene_id in probed_genes])

    # Normalize counts
    y = x - gene_means / gene_reg_stds

    return y


def mean(x: scipy.sparse.csr_matrix,
         gene_means_file: Path | str,
         probed_genes: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Normalize gene counts per gene by mean expression.

    This function should be applied following normalization per cell by read depth or cell area.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_means_file: Path | str
        Path to pickle file containing dictionary of mean gene expression values.
    probed_genes:
        Array with ensembl ids of probed genes.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    # Load dictionary of gene means
    with open(gene_means_file, "rb") as f:
        gene_means_dict = pickle.load(f)

    # Retrieve gene means for probed genes
    gene_means = np.array([gene_means_dict[gene_id] for gene_id in probed_genes])

    # Normalize counts
    y = x / gene_means

    return y


def non_zero_median(x: scipy.sparse.csr_matrix,
                    gene_nzmedians_file: Path | str,
                    probed_genes: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Normalize by non-zero median expression.

    This function should be applied following normalization per cell by read depth or cell area.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_nzmedians_file: Path | str
        Path to pickle file containing dictionary of non-zero median gene expression values.
    probed_genes:
        Array with ensembl ids of probed genes.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    # Load dictionary of gene non-zero medians
    with open(gene_nzmedians_file, "rb") as f:
        gene_nzmedians_dict = pickle.load(f)

    # Retrieve gene non-zero medians
    gene_nzmedians = np.array([gene_nzmedians_dict[gene_id] for gene_id in probed_genes])

    # Normalize counts
    y = x / gene_nzmedians

    return y


def shifted_log(x: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """
    Normalize by shifted log.

    Implements normalization using `sc.pp.log1p`. This function should be applied following normalization per cell by
    cell area or read depth.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    y = sc.pp.log1p(x)

    return y


def shifted_log_mean(x: scipy.sparse.csr_matrix,
                     gene_logmeans_file: Path | str,
                     probed_genes: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Normalize by shifted log and log mean gene expression.

    Implements normalization using `sc.pp.log1p`. This function should be applied following normalization per cell by
    cell area or read depth.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_logmeans_file: Path | str
        Path to pickle file containing dictionary of log mean gene expression values.
    probed_genes:
        Array with ensembl ids of probed genes that will be normalized.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    y = sc.pp.log1p(x)

    # Load dictionary of gene logmeans
    with open(gene_logmeans_file, "rb") as f:
        gene_logmeans_dict = pickle.load(f)

    gene_logmeans = np.array([gene_logmeans_dict[gene_id] for gene_id in probed_genes])
    
    y = y / gene_logmeans

    return y
