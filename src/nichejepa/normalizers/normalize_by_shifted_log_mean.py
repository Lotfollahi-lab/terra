import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp


def normalize_by_shifted_log_mean(x: sp.csr_matrix,
                                  gene_logmeans_file: Path | str,
                                  probed_genes: np.ndarray) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (across datasets in the corpus) by log shifted mean expression.
    
    This function should be applied following normalization per cell by cell area or read depth.

    Parameters
    ----------
    x: sp.csr_matrix
        A sparse matrix where each row represents an observation and each column represents a feature.
    gene_logmeans_file: Path | str
        Path to pickle file containing dictionary of corpus-wide log mean gene expression values.
    probed_genes:
        Array with ensembl ids of probed genes that will be normalized.

    Returns
    ----------
    y: sp.csr_matrix
        A sparse matrix containing the normalized features.
    """

    log_normalized = sc.pp.log1p(x)

    # Load dictionary of gene logmeans
    with open(gene_logmeans_file, "rb") as f:
        gene_logmeans_dict = pickle.load(f)

    gene_logmeans = np.array([gene_logmeans_dict[gene_id] for gene_id in probed_genes])

    y = log_normalized / gene_logmeans

    return y
