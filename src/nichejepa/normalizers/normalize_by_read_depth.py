import scipy.sparse as sp


def normalize_by_read_depth(x: sp.csr_matrix, target_size: int = 10_000) -> sp.csr_matrix:
    """
    Normalize gene expression counts per cell by read depth.

    Parameters
    ----------
    x: sp.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    target_size: int
        The target read depth per observation (i.e. the sum of features across an observation).

    Returns
    ----------
    y: sp.csr_matrix
        A sparse matrix containing the normalized features.
    """

    y = x / x.sum(axis=1).reshape(-1, 1) * target_size

    return y
