import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from skmisc.loess import loess


def normalize_by_analytic_pearson_residuals(x: sp.csr_matrix,
                                            theta: float=100,
                                            ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (within the dataset) and cell
    using analytic pearson residuals.

    Implements normalization as described in "Lause, J., Berens, P. & Kobak, D.
    Analytic Pearson residuals for normalization of single-cell RNA-seq UMI
    data. Genome Biol. 22, 258 (2021)". Residuals are based on a negative
    binomial offset model with overdispersion shared across genes. Residuals are
    clipped to 'sqrt(n_obs)'. Negative residuals for a cell and gene indicate
    that fewer counts are observed than expected, compared to the gene’s average
    expression and cell read depth. Positive residuals indicate more counts than
    expected. By default, overdispersion `theta=100` is used.

    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/experimental/pp/_normalization.py#L36.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and each column
        represents a feature, containing raw counts as features (i.e. not scaled
        or normalized).
    theta:
        The overdispersion parameter, defaults to `100`.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """

    if theta <= 0:
        raise ValueError("Pearson residuals require theta > 0")

    cell_sums = np.sum(x, axis=1).reshape(-1, 1)
    gene_sums = np.sum(x, axis=0).reshape(1, -1)
    sum_total = np.sum(gene_sums)

    mu = np.array(cell_sums @ gene_sums / sum_total)
    diff = np.array(x - mu)
    residuals = diff / np.sqrt(mu + mu ** 2 / theta)

    clip = np.sqrt(x.shape[0])
    y = np.clip(residuals, a_min=-clip, a_max=clip)

    return y


def normalize_by_cell_area(x: sp.csr_matrix,
                           cell_areas: np.ndarray,
                           ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per cell by cell area.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and each column
        represents a feature.
    cell_areas:
        Numpy array with the cell areas.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """

    if x.shape[0] != len(cell_areas):
        raise ValueError('Length of `cell_areas` does not match the number of'
                         'observations in `x`.')

    y = x / cell_areas.reshape(-1, 1)

    return y


def normalize_by_mean(x: sp.csr_matrix,
                      gene_means_file: Path | str,
                      probed_genes: np.ndarray,
                      ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (across datasets in the corpus) by
    mean expression.

    This function should be applied following normalization per cell by read
    depth or cell area.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and each column
        represents a feature.
    gene_means_file:
        Path to pickle file containing dictionary of mean gene expression
        values.
    probed_genes:
        Array with ensembl ids of probed genes.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """

    # Load dictionary of gene means
    with open(gene_means_file, "rb") as f:
        gene_means_dict = pickle.load(f)

    # Retrieve gene means for probed genes
    gene_means = np.array(
        [gene_means_dict[gene_id] for gene_id in probed_genes])

    y = x / gene_means

    return y


def normalize_by_nonzero_mean(x: sp.csr_matrix,
                              gene_nzmeans_file: Path | str,
                              probed_genes: np.ndarray,
                              ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (across datasets in the corpus) by
    non-zero mean expression.

    This function should be applied following normalization per cell by read
    depth or cell area.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and each column
        represents a feature.
    gene_nzmeans_file:
        Path to pickle file containing dictionary of non-zero mean gene
        expression values.
    probed_genes:
        Array with ensembl ids of probed genes.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """

    # Load dictionary of gene non-zero means
    with open(gene_nzmeans_file, "rb") as f:
        gene_nzmeans_dict = pickle.load(f)

    # Retrieve gene non-zero means
    gene_nzmeans = np.array(
        [gene_nzmeans_dict[gene_id] for gene_id in probed_genes])

    y = x / gene_nzmeans

    return y


def normalize_by_read_depth(x: sp.csr_matrix,
                            target_size: int=10_000,
                            ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per cell by read depth.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and each column
        represents a feature.
    target_size:
        The target read depth per observation (i.e. the sum of features across
        an observation).

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """

    y = x / x.sum(axis=1).reshape(-1, 1) * target_size

    return y


def normalize_by_seurat(x: sp.csr_matrix) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (within the dataset) using seurat
    v3 `FindVariableFeatures`.

    Implements normalization as described in "Stuart, T. et al. Comprehensive
    Integration of Single-Cell Data. Cell 177, 1888–1902.e21 (2019)". Counts are
    normalized by centering around the expected mean and scaling by the expected
    standard deviation, as learned from the global mean-variance relationships.
    This normalization should be applied independently for each dataset in the
    training corpus. Note, we do not implement clipping as described in the
    seurat publication.

    This function should be applied following normalization per cell by cell
    area or read depth.

    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/preprocessing/_highly_variable_genes.py#L26.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and each column
        represents a feature, containing raw counts as features (i.e. not scaled
        or normalized).

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """

    if (type(x) == sp._csr.csr_matrix or 
    type(x) == sp._csc.csc_matrix):
        x = x.toarray()
    elif type(x) == np.matrix:
        x = np.array(x)
    gene_means = np.mean(x, axis=0)
    gene_vars = np.var(x, axis=0)

    not_const = gene_vars > 0

    gene_log_means = np.log10(gene_means[not_const])
    gene_log_variances = np.log10(gene_vars[not_const])

    model = loess(gene_log_means,
                  gene_log_variances,
                  options={"span": 0.3, "degree": 2})
    model.fit()

    expected_log_variances = np.zeros(x.shape[1], dtype=np.float64)
    expected_log_variances[not_const] = model.outputs.fitted_values
    expected_variances = 10 ** expected_log_variances
    expected_standard_deviation = np.sqrt(expected_variances)

    y = (x - gene_means) / expected_standard_deviation

    return y


def normalize_by_shifted_log(x: sp.csr_matrix) -> sp.csr_matrix:
    """
    Implements normalization using `sc.pp.log1p`.
    
    This function should be applied following normalization per cell by cell
    area or read depth.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and each column
        represents a feature.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """

    y = sc.pp.log1p(x)

    return y


def normalize_by_shifted_log_mean(x: sp.csr_matrix,
                                  gene_logmeans_file: Path | str,
                                  probed_genes: np.ndarray,
                                  ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (across datasets in the corpus) by
    log shifted mean expression.
    
    This function should be applied following normalization per cell by cell
    area or read depth.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and each column
        represents a feature.
    gene_logmeans_file:
        Path to pickle file containing dictionary of corpus-wide log mean gene
        expression values.
    probed_genes:
        Array with ensembl ids of probed genes that will be normalized.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """

    log_normalized = sc.pp.log1p(x)

    # Load dictionary of gene logmeans
    with open(gene_logmeans_file, "rb") as f:
        gene_logmeans_dict = pickle.load(f)

    gene_logmeans = np.array(
        [gene_logmeans_dict[gene_id] for gene_id in probed_genes])

    y = log_normalized / gene_logmeans

    return y