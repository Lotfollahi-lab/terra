from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy


def mean_normalize_by_gene(x: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
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

    gene_means = [np.mean(x[:, i]) for i in range(x.shape[1])]

    y = x / gene_means

    return y
