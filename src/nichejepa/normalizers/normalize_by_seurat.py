import numpy as np
import scipy.sparse as sp
from skmisc.loess import loess


def normalize_by_seurat(x: sp.csr_matrix) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (within the dataset) using seurat v3 `FindVariableFeatures`.

    Implements normalization as described in "Stuart, T. et al. Comprehensive Integration of Single-Cell Data. Cell 177,
    1888–1902.e21 (2019)". Counts are normalized by centering around the expected mean and scaling by the expected
    standard deviation, as learned from the global mean-variance relationships. This normalization should be applied
    independently for each dataset in the training corpus. Note, we do not implement clipping as described in the seurat
    publication.

    This function should be applied following normalization per cell by cell area or read depth.

    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/preprocessing/_highly_variable_genes.py#L26.

    Parameters
    ----------
    x: sp.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature,
        containing raw counts as features (i.e. not scaled or normalized).

    Returns
    ----------
    y: sp.csr_matrix
        A sparse matrix containing the normalized features.
    """

    if type(x) != np.ndarray:
        x = x.toarray()
    gene_means = np.mean(x, axis=0)
    gene_vars = np.var(x, axis=0)

    not_const = gene_vars > 0

    gene_log_means = np.log10(gene_means[not_const])
    gene_log_variances = np.log10(gene_vars[not_const])

    model = loess(gene_log_means, gene_log_variances, options={"span": 0.3, "degree": 2})
    model.fit()

    expected_log_variances = np.zeros(x.shape[1], dtype=np.float64)
    expected_log_variances[not_const] = model.outputs.fitted_values
    expected_variances = 10 ** expected_log_variances
    expected_standard_deviation = np.sqrt(expected_variances)

    y = (x - gene_means) / expected_standard_deviation

    return y
