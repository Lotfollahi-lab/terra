import numpy as np
import scipy.sparse as sp


def normalize_by_cell_area(x: sp.csr_matrix,
                           cell_areas: np.ndarray) -> sp.csr_matrix:
    """
    Normalize gene expression counts per cell by cell area.

    Parameters
    ----------
    x: sp.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    cell_areas: np.ndarray
        Numpy array with the cell areas.

    Returns
    ----------
    y: sp.csr_matrix
        A sparse matrix containing the normalized features.
    """

    if x.shape[0] != len(cell_areas):
        raise ValueError('Length of `cell_areas` does not match the number of observations in `x`.')

    y = x / cell_areas.reshape(-1, 1)

    return y
