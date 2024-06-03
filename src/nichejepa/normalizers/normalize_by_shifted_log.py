import scanpy as sc
import scipy.sparse as sp


def normalize_by_shifted_log(x: sp.csr_matrix) -> sp.csr_matrix:
    """
    Implements normalization using `sc.pp.log1p`.
    
    This function should be applied following normalization per cell by cell area or read depth.

    Parameters
    ----------
    x: sp.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.

    Returns
    ----------
    y: sp.csr_matrix
        A sparse matrix containing the normalized features.
    """

    y = sc.pp.log1p(x)

    return y
