import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def normalize_by_nonzero_median(x: sp.csr_matrix,
                                gene_nzmedians_file: Path | str,
                                probed_genes: np.ndarray) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (across datasets in the corpus) by non-zero median expression.

    This function should be applied following normalization per cell by read depth or cell area.

    Parameters
    ----------
    x: sp.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_nzmedians_file: Path | str
        Path to pickle file containing dictionary of non-zero median gene expression values.
    probed_genes:
        Array with ensembl ids of probed genes.

    Returns
    ----------
    y: sp.csr_matrix
        A sparse matrix containing the normalized features.
    """

    # Load dictionary of gene non-zero medians
    with open(gene_nzmedians_file, "rb") as f:
        gene_nzmedians_dict = pickle.load(f)

    # Retrieve gene non-zero medians
    gene_nzmedians = np.array([gene_nzmedians_dict[gene_id] for gene_id in probed_genes])

    y = x / gene_nzmedians

    return y
