import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def normalize_by_nonzero_mean(x: sp.csr_matrix,
                                gene_nzmeans_file: Path | str,
                                probed_genes: np.ndarray) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (across datasets in the corpus) by non-zero mean expression.

    This function should be applied following normalization per cell by read depth or cell area.

    Parameters
    ----------
    x: sp.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_nzmeans_file: Path | str
        Path to pickle file containing dictionary of non-zero mean gene expression values.
    probed_genes:
        Array with ensembl ids of probed genes.

    Returns
    ----------
    y: sp.csr_matrix
        A sparse matrix containing the normalized features.
    """

    # Load dictionary of gene non-zero means
    with open(gene_nzmeans_file, "rb") as f:
        gene_nzmeans_dict = pickle.load(f)

    # Retrieve gene non-zero means
    gene_nzmeans = np.array([gene_nzmeans_dict[gene_id] for gene_id in probed_genes])

    y = x / gene_nzmeans

    return y
