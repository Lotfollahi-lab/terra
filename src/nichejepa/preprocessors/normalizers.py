import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from skmisc.loess import loess


def normalize_by_analytic_pearson_residuals(x: sp.csr_matrix,
                                            theta: float = 100,
                                            ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (within the batch) and
    cell using analytic pearson residuals.

    Implements normalization as described in "Lause, J., Berens, P. & 
    Kobak, D. Analytic Pearson residuals for normalization of
    single-cell RNA-seq UMI data. Genome Biol. 22, 258 (2021)".
    Residuals are based on a negative binomial offset model with
    overdispersion shared across genes. Residuals are clipped to
    'sqrt(n_obs)'. Negative residuals for a cell and gene indicate that
    fewer counts are observed than expected, compared to the gene’s
    average expression and cell read depth. Positive residuals indicate
    more counts than expected. By default, overdispersion `theta=100` is
    used.

    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/experimental/pp/_normalization.py#L36.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and
        each column represents a feature, containing raw counts as
        features (i.e. not scaled or normalized).
    theta:
        The overdispersion parameter, defaults to `100`.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """
    if theta <= 0:
        raise ValueError('Pearson residuals require theta > 0')

    cell_sums = np.sum(x, axis=1).reshape(-1, 1)
    gene_sums = np.sum(x, axis=0).reshape(1, -1)
    sum_total = np.sum(gene_sums)

    mu = np.array(cell_sums @ gene_sums / sum_total)
    diff = np.array(x - mu)
    residuals = diff / np.sqrt(mu + mu ** 2 / theta)

    clip = np.sqrt(x.shape[0])
    y = np.clip(residuals, a_min=-clip, a_max=clip)

    return y


def normalize_by_cell_area(x: sp.csr_matrix,
                           cell_areas: np.ndarray,
                           ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per cell by cell area.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and
        each column represents a feature.
    cell_areas:
        Numpy array with the cell areas.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """
    if x.shape[0] != len(cell_areas):
        raise ValueError('Length of `cell_areas` does not match the number of'
                         'observations in `x`.')

    y = x / cell_areas.reshape(-1, 1)

    return y


def normalize_by_gene_corrected_read_depth(
    x: sp.csr_matrix,
    basis_target_read_depth: float = 153.4768,
    target_read_depth_per_gene: float = 0.0487,
    ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per cell by read depth adjusted for
    number of probed genes. Default values are linear regression fit on
    corpus.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and
        each column represents a feature.
    basis_target_read_depth:
        Read depth independent of number of probed genes.
    target_read_depth_per_gene:
        Additional read depth increment per probed gene.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """
    y = x / x.sum(axis=1).reshape(-1, 1) * (basis_target_read_depth +
        x.shape[1] * target_read_depth_per_gene)

    return y


def normalize_by_factor(x: sp.csr_matrix,
                        norm_factor_file_path: Path | str,
                        probed_genes: np.ndarray,
                        norm_factor: Literal=[
                            'mean',
                            'nonzero_mean',
                            'read_depth_mean',
                            'read_depth_nonzero_mean',
                            'gene_corrected_read_depth_mean',
                            'gene_corrected_read_depth_nonzero_mean'],
                      ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (across batches in the
    corpus) by a normalization factor.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and
        each column represents a feature.
    norm_factor:
        Factor which is used for normalization.
    norm_factor_file_path:
        Path to csv file containing normalization factors per gene.
    probed_genes:
        Array with ensembl ids of probed genes.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """
    # Load norm factors per gene
    norm_factor_df = pd.read_csv(norm_factor_file_path)

    # Retrieve norm factors for probed genes
    norm_factors = np.array(
        [norm_factor_df[
            norm_factor_df['gene_id'] == gene_id][norm_factor].values[0]
         for gene_id in probed_genes])

    y = sp.csr_matrix(x / norm_factors)

    return y


def normalize_by_read_depth(x: sp.csr_matrix,
                            target_size: int = 10_000,
                            ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per cell by read depth.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and
        each column represents a feature.
    target_size:
        The target read depth per observation (i.e. the sum of features
        across an observation).

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """
    y = x / x.sum(axis=1).reshape(-1, 1) * target_size

    return y


def normalize_by_seurat(x: sp.csr_matrix) -> sp.csr_matrix:
    """
    Normalize gene expression counts per gene (within the batch) using
    seurat v3 `FindVariableFeatures`.

    Implements normalization as described in "Stuart, T. et al.
    Comprehensive Integration of Single-Cell Data. Cell 177,
    1888–1902.e21 (2019)". Counts are normalized by centering around the
    expected mean and scaling by the expected standard deviation, as
    learned from the global mean-variance relationships. This
    normalization should be applied independently for each batch in the
    training corpus. Note, we do not implement clipping as described in
    the seurat publication.

    The implementation is based on
    https://github.com/scverse/scanpy/blob/4642cf8e2e51b257371792cb4fcb9611c0a81123/scanpy/preprocessing/_highly_variable_genes.py#L26.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and
        each column represents a feature, containing raw counts as
        features (i.e. not scaled or normalized).

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """
    if (type(x) == sp._csr.csr_matrix or 
    type(x) == sp._csc.csc_matrix):
        x = x.toarray()
    elif type(x) == np.matrix:
        x = np.array(x)
    gene_means = np.mean(x, axis=0)
    gene_vars = np.var(x, axis=0)

    not_const = gene_vars > 0

    gene_log_means = np.log10(gene_means[not_const])
    gene_log_variances = np.log10(gene_vars[not_const])

    model = loess(gene_log_means,
                  gene_log_variances,
                  options={'span': 0.3, 'degree': 2})
    model.fit()

    expected_log_variances = np.zeros(x.shape[1], dtype=np.float64)
    expected_log_variances[not_const] = model.outputs.fitted_values
    expected_variances = 10 ** expected_log_variances
    expected_standard_deviation = np.sqrt(expected_variances)

    y = (x - gene_means) / expected_standard_deviation

    return y


def normalize_by_shifted_log(x: sp.csr_matrix) -> sp.csr_matrix:
    """
    Implements normalization using `sc.pp.log1p`.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and
        each column represents a feature.

    Returns
    ----------
    y:
        A sparse matrix containing the normalized features.
    """
    if sp.issparse(x):
        x = x.tocsr()
        x = x.astype(np.float32)
    else:
        x = np.asarray(x, dtype=np.float32)

    y = sc.pp.log1p(x)

    return y


def normalize_by_pflog1ppf(x: sp.csr_matrix,
                           target_size: float | None = None,
                           logsum_target: float | None = None,
                           pseudocount: float = 1.0,
                           ) -> sp.csr_matrix:
    """
    Normalize gene expression counts per cell using PFlog1pPF
    (proportional fitting -> log1p -> proportional fitting).

    Implements the depth-normalization transform of "Booeshaghi, A. S.,
    Hallgrímsdóttir, I. B., Gálvez-Merchán, Á. & Pachter, L. Depth
    normalization for single-cell genomics count data. bioRxiv
    2022.05.06.490859". The authors show this is the only feature-
    relabeling-equivariant transform that jointly satisfies variance
    stabilization, depth normalization, and monotonicity, and that it
    is equivalent to a shifted centered-log-ratio transform.

    Three steps, all monotonic within a cell (so within-cell gene
    rankings are preserved) and all sparsity-preserving:
      1. PF: scale each cell's counts to a common depth (the mean total
         count across cells, or ``target_size`` if given). This is the
         first proportional fitting -- the depth-equalization step that
         ``log1pPF`` also has.
      2. log1p: ``log(u + pseudocount)`` applied to nonzero entries
         (``log1p(0) = 0`` keeps zeros zero).
      3. PF again: scale each cell's log-values to a common total (the
         mean of the per-cell log-sums). This second proportional
         fitting removes the residual depth structure that the
         logarithm reintroduces -- the key addition over ``log1pPF``.

    Because every step is a per-cell scalar rescale or a monotone
    elementwise map, the within-cell gene ordering is identical to the
    raw counts, so this is safe to use as either a ``rank`` or
    ``count`` count-norm method. It does its OWN depth normalization
    (steps 1 and 3), so it must NOT be combined with a separate
    cell-/gene-level norm method (that would double-normalize depth) --
    same constraint as ``analytic_pearson_residuals``.

    Parameters
    ----------
    x:
        A sparse matrix where each row represents an observation and
        each column represents a feature, containing raw counts as
        features (i.e. not scaled or normalized).
    target_size:
        Common depth for the FIRST proportional fitting. ``None``
        (default) uses the mean total count across the cells in ``x``
        (per-call), matching the paper's PF convention; pass e.g.
        ``1e4`` for a CP10k-style fixed size factor, or the corpus-wide
        mean depth (``pf_depth_target`` from
        ``compute_cohort_norm_factors.py``) to freeze the first-PF scale
        across files / between train and inference. NB: because step 2
        is ``log1p`` (not ``log``), this target is NOT geometry-neutral
        -- it sets where counts sit relative to the pseudocount.
    logsum_target:
        Common total for the SECOND proportional fitting. ``None``
        (default) uses the mean per-cell log-sum across the cells in
        ``x`` (per-call); pass the corpus-wide value
        (``pf_logsum_target`` from ``compute_cohort_norm_factors.py``)
        to freeze the second-PF scale. This target is a pure global
        linear rescale of the output (does not change within-cell
        geometry).
    pseudocount:
        Pseudocount ``c`` added inside the log, i.e. ``log(u + c)``.
        Defaults to ``1.0`` (``log1p``), matching the paper's ``c = 1``.

    Returns
    ----------
    y:
        A sparse CSR matrix containing the PFlog1pPF-normalized
        features (non-negative, same sparsity pattern as ``x``).
    """
    if pseudocount <= 0:
        raise ValueError('PFlog1pPF requires pseudocount > 0.')

    if sp.issparse(x):
        x = x.tocsr().astype(np.float32)
    else:
        x = sp.csr_matrix(np.asarray(x, dtype=np.float32))

    # --- Step 1: proportional fitting (depth equalization) ---------
    cell_sums = np.asarray(x.sum(axis=1)).ravel()        # (n_obs,)
    # Cells with zero total counts stay all-zero; avoid div-by-zero.
    safe_cell_sums = np.where(cell_sums > 0, cell_sums, 1.0)
    s1 = (float(target_size) if target_size is not None
          else float(cell_sums.mean()))
    # Row-scale while staying sparse: x.multiply broadcasts an (n, 1)
    # column vector across columns (per-cell scalar multiply).
    x_pf = x.multiply((s1 / safe_cell_sums).reshape(-1, 1)).tocsr()

    # --- Step 2: log1p (or log(u + c)) on nonzero entries ----------
    # log(0 + c) handling: for c == 1, log1p(0) = 0 keeps zeros zero
    # (sparsity preserved). For c != 1, log(c) != 0 in general, but the
    # paper uses c = 1, so we keep the sparse fast path and only map
    # the stored (nonzero) entries.
    x_log = x_pf.copy()
    if pseudocount == 1.0:
        x_log.data = np.log1p(x_pf.data)
    else:
        x_log.data = np.log(x_pf.data + pseudocount)

    # --- Step 3: second proportional fitting on the log-values -----
    log_sums = np.asarray(x_log.sum(axis=1)).ravel()     # (n_obs,)
    safe_log_sums = np.where(log_sums > 0, log_sums, 1.0)
    s2 = (float(logsum_target) if logsum_target is not None
          else float(log_sums.mean()))
    y = x_log.multiply((s2 / safe_log_sums).reshape(-1, 1)).tocsr()

    return y.astype(np.float32)