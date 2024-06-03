import numpy as np
import scipy.sparse as sp


def normalize_by_analytic_pearson_residuals(x: sp.csr_matrix, theta: float = 100) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (within the dataset) and cell using analytic pearson residuals.

    Implements normalization as described in "Lause, J., Berens, P. & Kobak, D. Analytic Pearson residuals for
    normalization of single-cell RNA-seq UMI data. Genome Biol. 22, 258 (2021)". Residuals are based on a negative
    binomial offset model with overdispersion shared across genes. Residuals are clipped to 'sqrt(n_obs)'. Negative
    residuals for a cell and gene indicate that fewer counts are observed than expected, compared to the gene’s average
    expression and cell read depth. Positive residuals indicate more counts than expected. By default, overdispersion
    `theta=100` is used.

    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/experimental/pp/_normalization.py#L36.

    Parameters
    ----------
    x: sp.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature,
        containing raw counts as features (i.e. not scaled or normalized).
    theta: float
        The overdispersion parameter, defaults to `100`.

    Returns
    ----------
    y: sp.csr_matrix
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
