from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy


def non_zero_median(x: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """
    Normalize by non-zero median expression.

    This function should be applied following normalization per cell by read depth or cell area.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    gene_non_zero_medians = [np.median(x[:, i][np.nonzero(x[:, i])]) for i in range(x.shape[1])]

    y = x / gene_non_zero_medians

    return y
