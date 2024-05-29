from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy

from nichejepa.normalizers import mean_normalize_by_gene, shifted_log


def shifted_log_mean(x: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """
    Applies shifted log normalization followed by mean normalization by gene.

    Parameters
    ----------
    x: scipy.sparse.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.

    Returns
    ----------
    y: scipy.sparse.csr_matrix
        A sparse matrix containing the normalized features.
    """

    log_normalized = shifted_log(x)
    mean_log_normalized = mean_normalize_by_gene(log_normalized)

    return mean_log_normalized
