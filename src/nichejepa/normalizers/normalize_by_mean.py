import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def normalize_by_mean(x: sp.csr_matrix,
                      gene_means_file: Path | str,
                      probed_genes: np.ndarray) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (across datasets in the corpus) by mean expression.

    This function should be applied following normalization per cell by read depth or cell area.

    Parameters
    ----------
    x: sp.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_means_file: Path | str
        Path to pickle file containing dictionary of mean gene expression values.
    probed_genes:
        Array with ensembl ids of probed genes.

    Returns
    ----------
    y: sp.csr_matrix
        A sparse matrix containing the normalized features.
    """

    # Load dictionary of gene means
    with open(gene_means_file, "rb") as f:
        gene_means_dict = pickle.load(f)

    # Retrieve gene means for probed genes
    gene_means = np.array([gene_means_dict[gene_id] for gene_id in probed_genes])

    y = x / gene_means

    return y
