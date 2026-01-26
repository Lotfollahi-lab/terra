#!/usr/bin/env python3
"""
Train a lightweight ZINB/NB decoder that maps fixed UNI embeddings to gene
expression. This version assumes embeddings are precomputed inside the
combined NPZ dataset (see build_combined_dataset.py) and removes the
previous mixup / teacher-student / LoRA machinery for clarity.

Supported loss functions:
- ZINB (Zero-Inflated Negative Binomial): Original loss with dropout (zero-inflation) parameter
- NB with library size (nb_libsize): Uses true library size during training, fixed 10k CPM at test time
- NB with softplus (nb_softplus): No library size scaling, pure softplus for mean prediction

The NB loss variants address the data leakage concern when using library size:
- nb_libsize: Mitigates leakage by using a fixed 10k library size at test time
- nb_softplus: Completely avoids library size dependency
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

import anndata as ad
from scvi.distributions import ZeroInflatedNegativeBinomial
from scipy.stats import pearsonr
from collections import OrderedDict


# =============================================================================
# CONSTANTS
# =============================================================================

# Fixed library size used during NB evaluation to avoid data leakage.
# This corresponds to 10k CPM (Counts Per Million) normalization.
NB_FIXED_LIBSIZE = 10000.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across PyTorch, CUDA, and NumPy."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# =============================================================================
# NEGATIVE BINOMIAL LOSS FUNCTION
# =============================================================================

def nb_log_prob(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the log probability of the Negative Binomial distribution.
    
    This implementation is adapted from scvi-tools:
    Title: scvi-tools
    Authors: Romain Lopez <romain_lopez@gmail.com>,
             Adam Gayoso <adamgayoso@berkeley.edu>,
             Galen Xing <gx2113@columbia.edu>
    Date: 16th November 2020
    Code version: 0.8.1
    Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py
    
    The NB distribution models count data with overdispersion, parameterized by:
    - mu: mean of the distribution (positive support)
    - theta: inverse dispersion parameter (positive support)
    
    Higher theta means less dispersion (closer to Poisson).
    
    Parameters
    ----------
    x : torch.Tensor
        Ground truth count data. Shape: (batch_size, n_genes)
    mu : torch.Tensor
        Predicted means of the negative binomial distribution (positive support).
        Shape: (batch_size, n_genes)
    theta : torch.Tensor
        Inverse dispersion parameter (positive support).
        Can be shape (n_genes,) or (batch_size, n_genes).
    eps : float
        Small constant for numerical stability to prevent log(0).
    
    Returns
    -------
    torch.Tensor
        Log probability for each element. Shape: (batch_size, n_genes)
    
    Notes
    -----
    The NB probability mass function is:
        P(X=x) = Gamma(x + theta) / (Gamma(theta) * Gamma(x+1)) * 
                 (theta / (theta + mu))^theta * (mu / (theta + mu))^x
    
    Taking the log gives us the formula implemented below.
    """
    # Ensure theta has correct shape: (1, n_genes) for broadcasting
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))
    
    # Precompute log(theta + mu + eps) for efficiency
    log_theta_mu_eps = torch.log(theta + mu + eps)
    
    # Log probability computation using the NB PMF in log space
    # This avoids numerical issues from computing factorials directly
    log_prob = (
        # Term 1: theta * log(theta / (theta + mu))
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        # Term 2: x * log(mu / (theta + mu))
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        # Term 3: log(Gamma(x + theta))
        + torch.lgamma(x + theta)
        # Term 4: -log(Gamma(theta))
        - torch.lgamma(theta)
        # Term 5: -log(Gamma(x + 1)) = -log(x!)
        - torch.lgamma(x + 1)
    )
    
    return log_prob


def nb_nll(
    mu: torch.Tensor,
    theta: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the Negative Binomial negative log-likelihood loss.
    
    This is a wrapper around nb_log_prob that returns the mean NLL over the batch.
    
    Parameters
    ----------
    mu : torch.Tensor
        Predicted means. Shape: (batch_size, n_genes)
    theta : torch.Tensor
        Dispersion parameter. Shape: (n_genes,) or (batch_size, n_genes)
    target : torch.Tensor
        Ground truth counts. Shape: (batch_size, n_genes)
    eps : float
        Numerical stability constant.
    
    Returns
    -------
    torch.Tensor
        Scalar tensor with the mean NLL over the batch.
    """
    # Compute log probabilities and negate for NLL
    # Sum over genes (dim=-1), then mean over batch
    return -nb_log_prob(target, mu, theta, eps).sum(dim=-1).mean()


# =============================================================================
# ZINB LOSS FUNCTION (ORIGINAL)
# =============================================================================

def zinb_nll(
    mu: torch.Tensor,
    theta: torch.Tensor,
    zi_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Zero-Inflated Negative Binomial negative log-likelihood loss.
    
    ZINB extends NB by adding a zero-inflation component that models excess zeros
    in the data (e.g., dropout events in single-cell RNA-seq).
    
    Parameters
    ----------
    mu : torch.Tensor
        Predicted means of the NB component. Shape: (batch_size, n_genes)
    theta : torch.Tensor
        Dispersion parameter. Shape: (batch_size, n_genes)
    zi_logits : torch.Tensor
        Logits for the zero-inflation probability (dropout). Shape: (batch_size, n_genes)
    target : torch.Tensor
        Ground truth counts. Shape: (batch_size, n_genes)
    
    Returns
    -------
    torch.Tensor
        Scalar tensor with the mean NLL over the batch.
    """
    dist = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits)
    return -dist.log_prob(target).sum(dim=-1).mean()


# =============================================================================
# DECODER MODELS
# =============================================================================

class ZINBDecoder(nn.Module):
    """
    MLP decoder that outputs parameters for ZINB or NB distributions.
    
    Architecture:
    - Backbone: Stack of Linear -> [BatchNorm] -> [LayerNorm] -> SiLU -> Dropout blocks
    - Mean head: Linear layer followed by softplus activation
      (softmax when modeling relative expression for nb_libsize)
    - Dropout head (optional): Linear layer for zero-inflation logits (ZINB only)
    - Theta: Learnable per-gene dispersion parameter
    
    The model predicts:
    - count_mean: Expected counts (positive via softplus)
    - theta: Dispersion parameter (positive via softplus on learnable param)
    - count_dropout: Zero-inflation logits (only used for ZINB)
    
    Parameters
    ----------
    embed_dim : int
        Dimension of input embeddings.
    hidden_dim : int
        Dimension of hidden layers.
    n_genes : int
        Number of genes (output dimension).
    dropout : float
        Dropout probability in hidden layers.
    depth : int
        Number of MLP blocks in the backbone.
    layer_norm : bool
        Whether to use LayerNorm in hidden layers.
    batch_norm : bool
        Whether to use BatchNorm in hidden layers.
    residual : bool
        Whether to use residual connections (only if hidden_dim == embed_dim).
    use_dropout_head : bool
        Whether to include the dropout head (True for ZINB, False for NB).
    mean_activation : str
        Activation for the mean head: "softplus" (default) or "softmax".
        Softmax is used when the decoder should output relative expression that sums to 1.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_genes: int,
        dropout: float = 0.1,
        depth: int = 2,
        layer_norm: bool = False,
        batch_norm: bool = False,
        residual: bool = False,
        use_dropout_head: bool = True,
        mean_activation: str = "softplus",
    ):
        super().__init__()
        
        # Build backbone MLP layers
        layers: List[nn.Module] = []
        in_dim = embed_dim
        for _ in range(max(depth, 1)):
            block = []
            block.append(nn.Linear(in_dim, hidden_dim))
            # Optional normalization layers
            if batch_norm:
                block.append(nn.BatchNorm1d(hidden_dim))
            if layer_norm:
                block.append(nn.LayerNorm(hidden_dim))
            # SiLU (Swish) activation for smooth gradients
            block.append(nn.SiLU())
            block.append(nn.Dropout(dropout))
            layers.append(nn.Sequential(*block))
            in_dim = hidden_dim
        
        self.backbone = nn.ModuleList(layers)
        # Residual connections only work when dimensions match
        self.residual = residual and hidden_dim == embed_dim
        
        # Output heads
        self.mean_head = nn.Linear(in_dim, n_genes)
        valid_mean_act = {"softplus", "softmax"}
        if mean_activation not in valid_mean_act:
            raise ValueError(f"mean_activation must be one of {valid_mean_act}, got {mean_activation}")
        self.mean_activation = mean_activation
        
        # Dropout head is only needed for ZINB (zero-inflation modeling)
        self.use_dropout_head = use_dropout_head
        if use_dropout_head:
            self.dropout_head = nn.Linear(in_dim, n_genes)
        else:
            self.dropout_head = None
        
        # Learnable per-gene dispersion parameter (initialized to 0, softplus gives ~0.69)
        self.theta = nn.Parameter(torch.zeros(n_genes))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the decoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings. Shape: (batch_size, embed_dim)
        
        Returns
        -------
        count_mean : torch.Tensor
            Predicted mean counts (before library size scaling). Shape: (batch_size, n_genes)
        theta : torch.Tensor
            Dispersion parameter. Shape: (batch_size, n_genes)
        count_dropout : Optional[torch.Tensor]
            Zero-inflation logits (None for NB). Shape: (batch_size, n_genes) if ZINB
        """
        h = x
        
        # Pass through backbone with optional residual connections
        for layer in self.backbone:
            if self.residual and h.shape[-1] == layer[0].out_features:
                h = h + layer(h)
            else:
                h = layer(h)
        
        # Mean activation depends on training objective
        logits = self.mean_head(h)
        if self.mean_activation == "softmax":
            count_mean = torch.nn.functional.softmax(logits, dim=-1)
        else:
            count_mean = torch.nn.functional.softplus(logits)
        
        # Compute dropout logits only for ZINB
        if self.use_dropout_head and self.dropout_head is not None:
            count_dropout = self.dropout_head(h)
        else:
            count_dropout = None
        
        # Softplus on learnable theta ensures positive dispersion
        # Expand to match batch size for broadcasting
        theta = torch.nn.functional.softplus(self.theta).view(1, -1).expand_as(count_mean)
        
        return count_mean, theta, count_dropout


# =============================================================================
# LIBRARY SIZE COMPUTATION
# =============================================================================

def compute_library_size(counts: torch.Tensor) -> torch.Tensor:
    """
    Compute the library size (total counts) for each cell/spot.
    
    Library size is the sum of all gene counts for a given cell/spot.
    This is used to normalize predictions in the NB loss when using library size scaling.
    
    Parameters
    ----------
    counts : torch.Tensor
        Raw count matrix. Shape: (batch_size, n_genes)
    
    Returns
    -------
    torch.Tensor
        Library size for each sample. Shape: (batch_size, 1)
    
    Notes
    -----
    Using true library size during training but a fixed value during testing
    helps mitigate data leakage while still allowing the model to learn
    cell-specific scaling during training.
    """
    return counts.sum(dim=-1, keepdim=True)


def _to_dense(matrix) -> np.ndarray:
    """
    Convert a matrix (possibly sparse) to a dense NumPy array.

    AnnData.X or obsm entries may be scipy sparse; this helper ensures
    downstream code receives a dense float32 ndarray.
    """
    if hasattr(matrix, "A"):
        return matrix.A
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    arr = np.asarray(matrix)
    if arr.dtype == object:
        try:
            arr = np.asarray(arr.tolist())
        except ValueError as exc:
            raise ValueError(
                "Cannot convert AnnData matrix to a dense array; "
                "object dtype entries must form a regular grid (e.g., list of equal-length rows)."
            ) from exc
    return arr


def _split_indices(
    n: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create random train/val/test index splits.

    Fractions are with respect to the full dataset. The remainder goes to train.
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction and test_fraction must be >=0 and sum to < 1.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_fraction * n))
    n_test = int(round(test_fraction * n))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough samples left for train after split; reduce val/test fractions.")
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def _spot_level_metrics(
    preds_all: np.ndarray,
    y_test: np.ndarray,
    suffix: str = "_spot",
) -> Dict[str, float]:
    """
    Compute spot-level (cell-level) metrics: PCC, MSE, MAE.
    
    These metrics evaluate prediction quality at the individual spot level,
    as opposed to gene-level metrics which evaluate across all spots for each gene.
    
    Parameters
    ----------
    preds_all : np.ndarray
        Predicted values. Shape: (n_spots, n_genes)
    y_test : np.ndarray
        Ground truth values. Shape: (n_spots, n_genes)
    suffix : str
        Suffix to add to metric names.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing spot-level PCC, MSE, and MAE statistics.
    """
    if preds_all.size == 0 or y_test.size == 0:
        return {}
    
    # Compute per-spot errors
    diff = preds_all - y_test
    mse_per_spot = np.mean(diff ** 2, axis=1)
    mae_per_spot = np.mean(np.abs(diff), axis=1)
    
    # Compute per-spot Pearson correlation
    # Center both predictions and targets
    x = preds_all - preds_all.mean(axis=1, keepdims=True)
    y = y_test - y_test.mean(axis=1, keepdims=True)
    # Compute correlation using dot product formula
    denom = np.sqrt(np.sum(x**2, axis=1) * np.sum(y**2, axis=1) + 1e-8)
    spot_pcc = np.sum(x * y, axis=1) / denom
    
    return {
        f"pcc_mean{suffix}": float(np.mean(spot_pcc)),
        f"pcc_std{suffix}": float(np.std(spot_pcc)),
        f"mse_mean{suffix}": float(np.mean(mse_per_spot)),
        f"mse_std{suffix}": float(np.std(mse_per_spot)),
        f"mae_mean{suffix}": float(np.mean(mae_per_spot)),
        f"mae_std{suffix}": float(np.std(mae_per_spot)),
    }


def regression_metrics(
    preds_all: np.ndarray,
    y_test: np.ndarray,
    genes: List[str],
    spot_metrics: bool = False,
    spot_suffix: str = "_spot",
) -> Dict[str, object]:
    """
    Compute gene-level regression metrics: MSE, R², Pearson correlation.
    
    These metrics evaluate how well each gene's expression is predicted
    across all spots/cells.
    
    Parameters
    ----------
    preds_all : np.ndarray
        Predicted values. Shape: (n_spots, n_genes)
    y_test : np.ndarray
        Ground truth values. Shape: (n_spots, n_genes)
    genes : List[str]
        Gene names for reporting.
    spot_metrics : bool
        Whether to also compute spot-level metrics.
    spot_suffix : str
        Suffix for spot-level metric names.
    
    Returns
    -------
    Dict[str, object]
        Dictionary containing per-gene and aggregate metrics.
    """
    errors: List[float] = []
    r2_scores: List[float] = []
    pearson_corrs: List[float] = []
    pearson_genes: List[dict] = []

    n_nan_genes = 0
    for i in range(y_test.shape[1]):
        preds = preds_all[:, i]
        target_vals = y_test[:, i]

        # MSE for this gene
        errors.append(float(np.mean((preds - target_vals) ** 2)))

        # R² score (coefficient of determination)
        denom = np.sum((target_vals - np.mean(target_vals)) ** 2)
        if denom <= 1e-8:
            r2 = float("nan")  # Constant target - R² undefined
        else:
            r2 = float(1.0 - np.sum((target_vals - preds) ** 2) / denom)
        r2_scores.append(r2)

        # Pearson correlation coefficient
        if np.std(preds) < 1e-8 or np.std(target_vals) < 1e-8:
            pearson_corr = float("nan")  # Constant values - correlation undefined
        else:
            pearson_corr, _ = pearsonr(target_vals, preds)
        pearson_corrs.append(pearson_corr)

        if np.isnan(pearson_corr):
            n_nan_genes += 1

        gene_name = genes[i] if i < len(genes) else f"gene_{i}"
        pearson_genes.append({"name": gene_name, "pearson_corr": pearson_corr})

    if n_nan_genes > 0:
        print(f"Warning: {n_nan_genes} genes have NaN Pearson correlation")

    # Convert to arrays for aggregate statistics
    pearson_arr = np.array(pearson_corrs, dtype=float)
    r2_arr = np.array(r2_scores, dtype=float)
    errors_arr = np.array(errors, dtype=float)

    result: Dict[str, object] = {
        "l2_errors": list(errors),
        "r2_scores": list(r2_scores),
        "pearson_corrs": pearson_genes,
        "pearson_mean": float(np.nanmean(pearson_arr)) if pearson_arr.size else float("nan"),
        "pearson_std": float(np.nanstd(pearson_arr)) if pearson_arr.size else float("nan"),
        "l2_error_q1": float(np.nanpercentile(errors_arr, 25)) if errors_arr.size else float("nan"),
        "l2_error_q2": float(np.nanmedian(errors_arr)) if errors_arr.size else float("nan"),
        "l2_error_q3": float(np.nanpercentile(errors_arr, 75)) if errors_arr.size else float("nan"),
        "r2_score_q1": float(np.nanpercentile(r2_arr, 25)) if r2_arr.size else float("nan"),
        "r2_score_q2": float(np.nanmedian(r2_arr)) if r2_arr.size else float("nan"),
        "r2_score_q3": float(np.nanpercentile(r2_arr, 75)) if r2_arr.size else float("nan"),
    }
    
    if spot_metrics:
        result.update(_spot_level_metrics(preds_all, y_test, suffix=spot_suffix))
    
    return result


def _apply_metric_transform(arr: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply a transform to arrays before computing metrics.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array (predictions or targets).
    mode : str
        Transform mode: "none" or "log1p".
    
    Returns
    -------
    np.ndarray
        Transformed array.
    """
    if mode == "log1p":
        arr = np.clip(arr, a_min=0.0, a_max=None)
        return np.log1p(arr)
    return arr


# =============================================================================
# GENE SELECTION UTILITIES
# =============================================================================

def compute_hvgs(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    counts_layer: str = "counts",
) -> list[str]:
    """
    Return a ranked list of HVGs without mutating the provided AnnData.

    For seurat_v3 flavor we prefer raw counts (counts_layer if present).
    For other flavors we fall back to log-normalized X when it appears to
    contain counts (crude guard based on max value).
    """
    import scanpy as sc  # Local import to keep dependency optional

    ad_copy = adata.copy()
    hvg_kwargs = {}

    if flavor == "seurat_v3":
        if counts_layer in ad_copy.layers:
            hvg_kwargs["layer"] = counts_layer
    else:
        from scipy.sparse import issparse

        xmax = ad_copy.X.max() if not issparse(ad_copy.X) else ad_copy.X.max()
        if xmax > 20:
            sc.pp.normalize_total(ad_copy, target_sum=1e4)
            sc.pp.log1p(ad_copy)

    sc.pp.highly_variable_genes(
        ad_copy,
        n_top_genes=n_top_genes,
        flavor=flavor,
        subset=False,
        **hvg_kwargs,
    )

    mask = ad_copy.var["highly_variable"]
    hvgs = ad_copy.var.index[mask]

    if "highly_variable_rank" in ad_copy.var:
        hvgs = (
            ad_copy.var.loc[hvgs]
            .sort_values("highly_variable_rank")
            .index.tolist()
        )
    elif flavor == "seurat_v3":
        hvgs = (
            ad_copy.var.loc[hvgs]
            .sort_values("variances_norm", ascending=False)
            .index.tolist()
        )
    else:
        hvgs = (
            ad_copy.var.loc[hvgs]
            .sort_values("dispersions_norm", ascending=False)
            .index.tolist()
        )

    return hvgs[:n_top_genes]


def compute_deg_union(
    adata: ad.AnnData,
    groupby: str,
    top_n: int = 50,
    method: str = "wilcoxon",
    layer: Optional[str] = None,
) -> list[str]:
    """
    Compute union of top DE genes per group using Scanpy's rank_genes_groups.

    Returns a de-duplicated gene list preserving group order.
    """
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer for DEG selection.")
    if groupby not in adata.obs:
        available = list(adata.obs.keys())
        raise ValueError(f"Groupby key '{groupby}' not found in adata.obs. Available: {available}")

    import scanpy as sc  # Local import to keep dependency optional

    ad_copy = adata.copy()
    sc.tl.rank_genes_groups(
        ad_copy,
        groupby=groupby,
        method=method,
        layer=layer,
    )
    result = ad_copy.uns.get("rank_genes_groups")
    if result is None:
        raise ValueError("rank_genes_groups did not populate adata.uns['rank_genes_groups'].")

    groups = result["names"].dtype.names
    degs: list[str] = []
    seen = set()
    for g in groups:
        names = list(result["names"][g][:top_n])
        for name in names:
            if name not in seen:
                seen.add(name)
                degs.append(name)
    return degs


def _load_gene_list(path: Path) -> list[str]:
    """Load genes from a text file (comma or newline separated)."""
    text = Path(path).read_text().strip()
    if not text:
        raise ValueError(f"Gene list file '{path}' is empty.")
    if "\n" in text:
        genes = [g.strip() for g in text.splitlines() if g.strip()]
    else:
        genes = [g.strip() for g in text.split(",") if g.strip()]
    if not genes:
        raise ValueError(f"No genes parsed from '{path}'.")
    return genes


def _restrict_to_genes(
    data: dict,
    gene_list: np.ndarray,
    keep_genes: list[str],
    context: str,
) -> tuple[dict, np.ndarray]:
    """
    Restrict expression matrices to a provided gene list.

    Returns updated data dict and gene_list aligned to the kept genes.
    """
    keep_set = set(keep_genes)
    if not keep_set:
        raise ValueError("Gene selection requested but no genes provided.")

    # Preserve existing gene order while filtering
    available = set(gene_list)
    keep_idx = [i for i, g in enumerate(gene_list) if g in keep_set]
    missing = [g for g in keep_genes if g not in available]
    if not keep_idx:
        raise ValueError(f"No overlap between requested genes and available genes in {context}.")
    if missing:
        print(f"[info] {len(missing)} requested genes not found in {context} (e.g., {missing[:5]}).")

    keep_idx = np.array(keep_idx, dtype=int)
    new_gene_list = gene_list[keep_idx]

    def _subset(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != len(gene_list):
            return arr
        return arr[:, keep_idx]

    data = dict(data)
    for key in ["train_expression", "test_expression_gt", "val_expression"]:
        if key in data:
            data[key] = _subset(data[key])

    return data, new_gene_list


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate(
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    genes: List[str],
    loss_type: str = "zinb",
    metric_transform: str = "none",
    spot_metrics: bool = False,
    spot_suffix: str = "_spot",
) -> Dict[str, object]:
    """
    Evaluate the decoder on a dataset.
    
    This function handles different loss types and their specific requirements:
    - ZINB: Uses the ZINB distribution mean for predictions
    - NB with libsize: Uses fixed 10k library size at test time to avoid data leakage
    - NB softplus: Uses raw softplus output without library size scaling
    
    Parameters
    ----------
    decoder : nn.Module
        The trained decoder model.
    loader : DataLoader
        DataLoader for evaluation data.
    device : torch.device
        Device to run evaluation on.
    genes : List[str]
        Gene names for reporting.
    loss_type : str
        Type of loss function: "zinb", "nb_libsize", or "nb_softplus".
    metric_transform : str
        Transform to apply before computing metrics.
    spot_metrics : bool
        Whether to compute spot-level metrics.
    spot_suffix : str
        Suffix for spot-level metric names.
    
    Returns
    -------
    Dict[str, object]
        Dictionary containing evaluation metrics.
    """
    decoder.eval()
    preds_all: List[np.ndarray] = []
    target_all: List[np.ndarray] = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError("Batches must contain at least (embeddings, targets).")
                emb, target = batch[0], batch[1]
            else:
                # TensorDataset always returns tuples, so hitting this likely indicates custom collate_fn
                raise ValueError("Unexpected batch format; expected tuple/list from DataLoader.")

            emb = emb.to(device)
            target = target.to(device)
            
            # Get model predictions
            mu, theta, dropout = decoder(emb)
            
            if loss_type == "zinb":
                # For ZINB, use the distribution mean which accounts for zero-inflation
                # Mean of ZINB = (1 - pi) * mu, where pi = sigmoid(dropout)
                dist = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=dropout)
                preds = dist.mean
                
            elif loss_type == "nb_libsize":
                # For NB with library size, use fixed 10k at test time
                # This avoids data leakage from using true library sizes
                # The model outputs relative expression (like CPM normalized)
                # We multiply by fixed library size to get absolute counts
                fixed_libsize = torch.full(
                    (mu.size(0), 1), NB_FIXED_LIBSIZE, device=device
                )
                preds = mu * fixed_libsize
                
            elif loss_type == "nb_softplus":
                # For NB without library size, use raw softplus output
                # The model directly predicts absolute expression levels
                preds = mu
                
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
            
            preds_all.append(preds.detach().cpu().numpy())
            target_all.append(target.detach().cpu().numpy())

    decoder.train()
    
    # Concatenate all batches
    preds_arr = np.concatenate(preds_all, axis=0)
    target_arr = np.concatenate(target_all, axis=0)
    
    # Apply optional transform for metrics
    preds_arr = _apply_metric_transform(preds_arr, metric_transform)
    target_arr = _apply_metric_transform(target_arr, metric_transform)
    
    # Warn if predictions are constant (potential training issue)
    if np.allclose(preds_arr, preds_arr[0]):
        print("[warn] model predictions are constant across all spots")
    if np.allclose(target_arr, target_arr[0]):
        print("[warn] targets are constant across all spots")
    
    return regression_metrics(
        preds_arr,
        target_arr,
        genes,
        spot_metrics=spot_metrics,
        spot_suffix=spot_suffix,
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def _get_embedding_from_obsm(
    adata: ad.AnnData,
    primary_key: Optional[str],
    fallback_key: Optional[str],
) -> np.ndarray:
    """
    Fetch embedding from AnnData.obsm. Tries primary_key first, then fallback_key.
    """
    if primary_key and primary_key in adata.obsm:
        return _to_dense(adata.obsm[primary_key])
    if fallback_key and fallback_key in adata.obsm:
        print(f"[info] Embedding key '{primary_key}' not found; using fallback '{fallback_key}'")
        return _to_dense(adata.obsm[fallback_key])
    available = list(adata.obsm.keys())
    raise ValueError(
        f"Embedding key not found. Tried primary='{primary_key}', fallback='{fallback_key}'. "
        f"Available obsm keys: {available}"
    )


def _get_expression_matrix(
    adata: ad.AnnData,
    layer_key: Optional[str],
) -> np.ndarray:
    """
    Retrieve expression values from AnnData.

    If layer_key is provided, data are taken from adata.layers[layer_key].
    Otherwise adata.X is used.
    """
    if layer_key:
        if layer_key not in adata.layers:
            available = list(adata.layers.keys())
            raise ValueError(
                f"Layer '{layer_key}' not found in AnnData. Available layers: {available}"
            )
        matrix = adata.layers[layer_key]
    else:
        matrix = adata.X
    return _to_dense(matrix)


def load_adata_datasets(
    train_path: Path,
    test_path: Path,
    embed_key: Optional[str],
    embed_fallback_key: Optional[str],
    expression_layer: Optional[str] = None,
    drop_zero_variance: bool = False,
    restrict_genes: Optional[list[str]] = None,
    train_adata_obj: Optional[ad.AnnData] = None,
    test_adata_obj: Optional[ad.AnnData] = None,
) -> tuple[dict, np.ndarray]:
    """
    Load train/test data from AnnData files.

    - Embeddings are read from adata.obsm[embed_key] (or embed_fallback_key).
    - Gene expression is taken from adata.layers[expression_layer] if provided, else adata.X.
    - Gene names come from adata.var_names.
    - If train/test genes differ, the intersection (ordered by train) is used.
    - If restrict_genes is provided, both train/test are further subset to that list.
    """
    train_adata = train_adata_obj if train_adata_obj is not None else ad.read_h5ad(train_path)
    test_adata = test_adata_obj if test_adata_obj is not None else ad.read_h5ad(test_path)

    # Determine shared genes and align
    train_genes = train_adata.var_names
    test_genes = test_adata.var_names
    shared = [g for g in train_genes if g in set(test_genes)]
    if not shared:
        raise ValueError("No shared genes between train and test AnnData files.")
    if len(shared) < len(train_genes) or len(shared) < len(test_genes):
        print(f"[info] Restricting to {len(shared)} shared genes (train:{len(train_genes)}, test:{len(test_genes)})")

    if restrict_genes:
        restrict_set = set(restrict_genes)
        filtered_shared = [g for g in shared if g in restrict_set]
        missing = [g for g in restrict_genes if g not in set(shared)]
        if not filtered_shared:
            raise ValueError("Gene restriction removed all shared genes between train and test AnnData.")
        if missing:
            print(f"[info] {len(missing)} requested genes not found in shared set (e.g., {missing[:5]}).")
        if len(filtered_shared) < len(shared):
            print(f"[info] Restricting to {len(filtered_shared)} genes after applying user gene subset.")
        shared = filtered_shared

    train_adata = train_adata[:, shared]
    test_adata = test_adata[:, shared]

    # Extract embeddings
    train_embeddings = _get_embedding_from_obsm(train_adata, embed_key, embed_fallback_key).astype(np.float32)
    test_embeddings = _get_embedding_from_obsm(test_adata, embed_key, embed_fallback_key).astype(np.float32)

    # Extract expressions (counts)
    train_expression = _get_expression_matrix(train_adata, expression_layer).astype(np.float32)
    test_expression = _get_expression_matrix(test_adata, expression_layer).astype(np.float32)

    # Optional zero-variance filtering
    if drop_zero_variance:
        stds = np.std(train_expression, axis=0)
        keep_idx = np.where(stds > 1e-8)[0]
        if not len(keep_idx):
            raise ValueError("All genes have zero variance; cannot train decoder.")
        if len(keep_idx) < len(shared):
            print(f"[info] Dropped {len(shared) - len(keep_idx)} zero-variance genes; {len(keep_idx)} remain.")
        train_expression = train_expression[:, keep_idx]
        test_expression = test_expression[:, keep_idx]
        gene_list = np.array([shared[i] for i in keep_idx], dtype=object)
    else:
        gene_list = np.array(shared, dtype=object)

    def _get_obs_or_default(adata: ad.AnnData, key: str, default_prefix: str):
        if key in adata.obs:
            return np.asarray(adata.obs[key]).astype(str)
        return np.array([f"{default_prefix}_{i}" for i in range(adata.n_obs)])

    train_slides = _get_obs_or_default(train_adata, "slide", "train_slide")
    test_slides = _get_obs_or_default(test_adata, "slide", "test_slide")
    train_barcodes = _get_obs_or_default(train_adata, "barcode", "train_bc")
    test_barcodes = _get_obs_or_default(test_adata, "barcode", "test_bc")

    data = {
        "train_embeddings": train_embeddings,
        "train_expression": train_expression,
        "train_barcodes": train_barcodes,
        "train_slides": train_slides,
        "test_embeddings": test_embeddings,
        "test_expression_gt": test_expression,
        "test_barcodes": test_barcodes,
        "test_slides": test_slides,
        "gene_list": gene_list,
    }
    return data, gene_list


def load_adata_single(
    adata_path: Path,
    embed_key: Optional[str],
    embed_fallback_key: Optional[str],
    val_fraction: float,
    test_fraction: float,
    split_seed: int,
    expression_layer: Optional[str] = None,
    drop_zero_variance: bool = False,
    restrict_genes: Optional[list[str]] = None,
    adata_obj: Optional[ad.AnnData] = None,
) -> tuple[dict, np.ndarray]:
    """
    Load a single AnnData file and produce random train/val/test splits.

    - Embeddings from obsm[embed_key] (or fallback).
    - Counts from the requested layer (fallback to .X if unset).
    - Gene names from .var_names.
    - Splits are random with provided fractions and seed.
    - Optionally restrict to a provided gene list before splitting.
    """
    adata_obj = adata_obj if adata_obj is not None else ad.read_h5ad(adata_path)

    # Extract full matrices
    full_embeddings = _get_embedding_from_obsm(adata_obj, embed_key, embed_fallback_key).astype(np.float32)
    full_expression = _get_expression_matrix(adata_obj, expression_layer).astype(np.float32)
    gene_list = np.array(adata_obj.var_names.tolist(), dtype=object)

    if restrict_genes:
        available = set(gene_list)
        keep_idx = [i for i, g in enumerate(gene_list) if g in set(restrict_genes)]
        missing = [g for g in restrict_genes if g not in available]
        if not keep_idx:
            raise ValueError("Gene restriction removed all genes present in the AnnData file.")
        if missing:
            print(f"[info] {len(missing)} requested genes not found in AnnData (e.g., {missing[:5]}).")
        if len(keep_idx) < len(gene_list):
            print(f"[info] Restricting to {len(keep_idx)} genes from user selection.")
        keep_idx_arr = np.array(keep_idx, dtype=int)
        full_expression = full_expression[:, keep_idx_arr]
        gene_list = gene_list[keep_idx_arr]

    # Optionally drop zero-variance genes globally before splitting
    if drop_zero_variance:
        stds = np.std(full_expression, axis=0)
        keep_idx = np.where(stds > 1e-8)[0]
        if not len(keep_idx):
            raise ValueError("All genes have zero variance; cannot train decoder.")
        if len(keep_idx) < len(gene_list):
            print(f"[info] Dropped {len(gene_list) - len(keep_idx)} zero-variance genes; {len(keep_idx)} remain.")
        full_expression = full_expression[:, keep_idx]
        gene_list = gene_list[keep_idx]

    # Split indices
    train_idx, val_idx, test_idx = _split_indices(
        n=full_expression.shape[0],
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )

    def _subset(arr, idx):
        return np.asarray(arr)[idx]

    # Observations metadata helpers
    def _get_obs_or_default(adata: ad.AnnData, key: str, default_prefix: str):
        if key in adata.obs:
            return np.asarray(adata.obs[key]).astype(str)
        return np.array([f"{default_prefix}_{i}" for i in range(adata.n_obs)])

    slides_all = _get_obs_or_default(adata_obj, "slide", "slide")
    barcodes_all = _get_obs_or_default(adata_obj, "barcode", "bc")

    data = {
        "train_embeddings": _subset(full_embeddings, train_idx),
        "train_expression": _subset(full_expression, train_idx),
        "train_barcodes": _subset(barcodes_all, train_idx),
        "train_slides": _subset(slides_all, train_idx),
        "val_embeddings": _subset(full_embeddings, val_idx),
        "val_expression": _subset(full_expression, val_idx),
        "val_barcodes": _subset(barcodes_all, val_idx),
        "val_slides": _subset(slides_all, val_idx),
        "test_embeddings": _subset(full_embeddings, test_idx),
        "test_expression_gt": _subset(full_expression, test_idx),
        "test_barcodes": _subset(barcodes_all, test_idx),
        "test_slides": _subset(slides_all, test_idx),
        "gene_list": gene_list,
    }
    return data, gene_list


def load_combined_dataset(
    path: Path,
    drop_zero_variance: bool = False,
    restrict_genes: Optional[list[str]] = None,
) -> tuple[dict, np.ndarray]:
    """
    Load the combined NPZ dataset containing embeddings and expression data.
    
    Parameters
    ----------
    path : Path
        Path to the NPZ file.
    drop_zero_variance : bool
        Whether to remove genes with zero variance in training data.
    restrict_genes : Optional[list[str]]
        If provided, restrict to this set of genes before zero-variance filtering.
    
    Returns
    -------
    tuple[dict, np.ndarray]
        Data dictionary and array of gene names.
    """
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    required = [
        "train_expression",
        "train_embeddings",
        "test_expression_gt",
        "test_embeddings",
        "gene_list",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Combined dataset missing keys: {missing}")

    gene_list = np.array(data["gene_list"].tolist(), dtype=object)
    if restrict_genes:
        data, gene_list = _restrict_to_genes(data, gene_list, restrict_genes, context="combined dataset")
    if not drop_zero_variance:
        return data, gene_list

    # Filter out zero-variance genes to improve training stability
    train_expr = data["train_expression"]
    test_expr = data["test_expression_gt"]
    stds = np.std(train_expr, axis=0)
    keep_idx = np.where(stds > 1e-8)[0]
    if not len(keep_idx):
        raise ValueError("All genes have zero variance; cannot train decoder.")

    filtered = {
        "train_embeddings": data["train_embeddings"],
        "train_expression": train_expr[:, keep_idx],
        "train_barcodes": data["train_barcodes"],
        "train_slides": data["train_slides"],
        "test_embeddings": data["test_embeddings"],
        "test_expression_gt": test_expr[:, keep_idx],
        "test_barcodes": data["test_barcodes"],
        "test_slides": data["test_slides"],
        "gene_list": np.array([gene_list[i] for i in keep_idx], dtype=object),
    }
    print(f"[info] Dropped {len(gene_list) - len(keep_idx)} zero-variance genes; {len(keep_idx)} remain.")
    return filtered, filtered["gene_list"]


def zero_std_genes(matrix: np.ndarray, genes: List[str], tol: float = 1e-8) -> List[str]:
    """
    Find genes with zero (or near-zero) standard deviation.
    
    These genes are problematic for training as they provide no signal.
    """
    stds = np.std(matrix, axis=0)
    return [genes[i] for i, std in enumerate(stds) if std <= tol]


# =============================================================================
# BATCH SAMPLING
# =============================================================================

class SlideBatchSampler:
    """
    Batch sampler that yields all indices belonging to a single slide per batch.
    
    This ensures that batch normalization and other batch-level operations
    are computed within slides, not across slides. You can disable this behavior
    via --disable-slide-batching to fall back to regular mini-batches.
    """

    def __init__(self, slide_indices: list[list[int]], shuffle: bool = True):
        self.slide_indices = slide_indices
        self.shuffle = shuffle

    def __iter__(self):
        order = (
            np.random.permutation(len(self.slide_indices))
            if self.shuffle
            else range(len(self.slide_indices))
        )
        for idx in order:
            yield self.slide_indices[idx]

    def __len__(self):
        return len(self.slide_indices)


def build_slide_indices(slide_names: np.ndarray) -> list[list[int]]:
    """
    Group sample indices by slide name.
    
    Returns a list where each element contains indices of samples
    belonging to the same slide.
    """
    grouped: OrderedDict[str, list[int]] = OrderedDict()
    for idx, name in enumerate(slide_names.tolist()):
        grouped.setdefault(str(name), []).append(idx)
    return list(grouped.values())


# =============================================================================
# DEVICE SELECTION
# =============================================================================

def get_device(device_id: int) -> torch.device:
    """
    Get the appropriate torch device based on device_id.
    
    Parameters
    ----------
    device_id : int
        GPU id (use -1 for CPU preference).
    """
    if torch.cuda.is_available() and device_id >= 0:
        return torch.device(f"cuda:{device_id}")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# LOSS COMPUTATION
# =============================================================================

def compute_loss(
    mu: torch.Tensor,
    theta: torch.Tensor,
    dropout: Optional[torch.Tensor],
    target: torch.Tensor,
    loss_type: str,
    library_size: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the loss based on the specified loss type.
    
    This function handles three loss types:
    
    1. ZINB (Zero-Inflated Negative Binomial):
       - Uses mu, theta, and dropout (zero-inflation logits)
       - No library size scaling
       - Suitable when modeling dropout events explicitly
    
    2. NB with library size (nb_libsize):
       - Uses mu * library_size as the mean
       - During training: library_size = true total counts per cell
       - During testing: library_size = fixed 10k to avoid leakage
       - Suitable when you want to model relative expression
    
    3. NB with softplus (nb_softplus):
       - Uses mu directly (no library size scaling)
       - Model learns to predict absolute counts
       - Avoids library size dependency entirely
    
    Parameters
    ----------
    mu : torch.Tensor
        Predicted mean (before library size scaling). Shape: (batch_size, n_genes)
    theta : torch.Tensor
        Dispersion parameter. Shape: (batch_size, n_genes)
    dropout : Optional[torch.Tensor]
        Zero-inflation logits (only for ZINB). Shape: (batch_size, n_genes)
    target : torch.Tensor
        Ground truth counts. Shape: (batch_size, n_genes)
    loss_type : str
        Type of loss: "zinb", "nb_libsize", or "nb_softplus".
    library_size : Optional[torch.Tensor]
        Library size for nb_libsize. Shape: (batch_size, 1)
    
    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if loss_type == "zinb":
        # ZINB loss with zero-inflation modeling
        if dropout is None:
            raise ValueError("ZINB loss requires dropout (zi_logits) output")
        return zinb_nll(mu, theta, dropout, target)
    
    elif loss_type == "nb_libsize":
        # NB loss with library size scaling
        # mu represents relative expression, we scale by library size
        if library_size is None:
            raise ValueError("nb_libsize requires library_size tensor")
        # Scale mu by library size to get absolute counts
        scaled_mu = mu * library_size
        return nb_nll(scaled_mu, theta, target)
    
    elif loss_type == "nb_softplus":
        # NB loss without library size
        # mu directly represents absolute counts via softplus
        return nb_nll(mu, theta, target)
    
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            "Please choose from: zinb, nb_libsize, nb_softplus"
        )


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(args: argparse.Namespace) -> dict:
    """
    Main training loop for the decoder.
    
    This function:
    1. Loads the dataset
    2. Creates data loaders with slide-based batching
    3. Initializes the decoder model
    4. Trains with the specified loss function
    5. Evaluates and saves the best model
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing all hyperparameters.
    
    Returns
    -------
    dict
        Dictionary with checkpoint path and best metrics.
    """
    set_seed(args.seed)
    device = get_device(args.device)

    print(f"[info] Using loss type: {args.loss_type}")

    gene_subset: Optional[list[str]] = None
    train_adata_obj: Optional[ad.AnnData] = None
    test_adata_obj: Optional[ad.AnnData] = None
    gene_selection_desc: Optional[str] = None

    if args.gene_selection != "all":
        if args.gene_selection == "list":
            if not args.gene_list_path:
                raise ValueError("--gene-list-path is required when --gene-selection=list.")
            gene_subset = _load_gene_list(Path(args.gene_list_path))
            gene_selection_desc = f"{len(gene_subset)} genes from user list"
        else:
            if args.dataset:
                raise ValueError("Gene selection via HVG/DEG requires AnnData input (use --adata or --train-adata/--test-adata).")
            if not args.adata and not args.train_adata:
                raise ValueError("Provide --adata or --train-adata/--test-adata when enabling gene selection.")
            target_path = Path(args.adata) if args.adata else Path(args.train_adata)
            train_adata_obj = ad.read_h5ad(target_path)

            if args.gene_selection == "hvg":
                try:
                    gene_subset = compute_hvgs(
                        train_adata_obj,
                        n_top_genes=args.hvg_top_n,
                        flavor=args.hvg_flavor,
                        counts_layer=args.hvg_counts_layer,
                    )
                except ImportError as exc:
                    raise ImportError("Gene selection 'hvg' requires scanpy to be installed.") from exc
                gene_selection_desc = f"top {len(gene_subset)} HVGs (flavor={args.hvg_flavor})"
            elif args.gene_selection == "deg":
                if not args.deg_label:
                    raise ValueError("--deg-label is required when --gene-selection=deg.")
                deg_layer = args.deg_layer if args.deg_layer is not None else args.expression_layer
                try:
                    gene_subset = compute_deg_union(
                        train_adata_obj,
                        groupby=args.deg_label,
                        top_n=args.deg_top_n,
                        method=args.deg_method,
                        layer=deg_layer,
                    )
                except ImportError as exc:
                    raise ImportError("Gene selection 'deg' requires scanpy to be installed.") from exc
                gene_selection_desc = (
                    f"top {args.deg_top_n} DEGs per group in '{args.deg_label}' "
                    f"(method={args.deg_method})"
                )
            else:
                raise ValueError(f"Unknown gene_selection mode: {args.gene_selection}")

        if gene_selection_desc:
            print(f"[info] Gene selection: {gene_selection_desc}")
    
    # Load dataset:
    # 1) NPZ (legacy), or
    # 2) single AnnData with random split, or
    # 3) paired AnnData (train/test)
    if args.dataset:
        data_npz, gene_list_array = load_combined_dataset(
            Path(args.dataset),
            drop_zero_variance=args.drop_zero_variance_genes,
            restrict_genes=gene_subset,
        )
    elif args.adata:
        if train_adata_obj is None:
            train_adata_obj = ad.read_h5ad(Path(args.adata))
        data_npz, gene_list_array = load_adata_single(
            Path(args.adata),
            embed_key=args.embed_key,
            embed_fallback_key=args.embed_fallback_key,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            split_seed=args.split_seed,
            expression_layer=args.expression_layer,
            drop_zero_variance=args.drop_zero_variance_genes,
            restrict_genes=gene_subset,
            adata_obj=train_adata_obj,
        )
    else:
        if not args.train_adata or not args.test_adata:
            raise ValueError("Provide either --dataset (NPZ), --adata (single .h5ad), or both --train-adata and --test-adata (.h5ad).")
        if train_adata_obj is None:
            train_adata_obj = ad.read_h5ad(Path(args.train_adata))
        test_adata_obj = test_adata_obj if test_adata_obj is not None else ad.read_h5ad(Path(args.test_adata))
        data_npz, gene_list_array = load_adata_datasets(
            Path(args.train_adata),
            Path(args.test_adata),
            embed_key=args.embed_key,
            embed_fallback_key=args.embed_fallback_key,
            expression_layer=args.expression_layer,
            drop_zero_variance=args.drop_zero_variance_genes,
            restrict_genes=gene_subset,
            train_adata_obj=train_adata_obj,
            test_adata_obj=test_adata_obj,
        )
    gene_list = list(gene_list_array.tolist())
    wandb_run = None
    if args.wandb_project:
        if wandb is None:
            raise ImportError("wandb is not installed; install it or unset --wandb-project to disable logging.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "loss_type": args.loss_type,
                "model": "ZINBDecoder",
                "args": vars(args),
            },
            reinit=True,
        )

    def _log_to_wandb(tag: str, metrics: Dict[str, object], epoch: int, extra: Optional[Dict[str, float]] = None) -> None:
        if wandb_run is None:
            return
        log = {"epoch": epoch}
        if extra:
            log.update(extra)
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                log[f"{tag}/{key}"] = float(value)
        if len(log) > 1:  # only epoch present means nothing to log
            wandb_run.log(log)

    # Convert to tensors
    train_emb = torch.from_numpy(data_npz["train_embeddings"].astype(np.float32))
    train_expr = torch.from_numpy(data_npz["train_expression"].astype(np.float32))
    test_emb = torch.from_numpy(data_npz["test_embeddings"].astype(np.float32))
    test_expr = torch.from_numpy(data_npz["test_expression_gt"].astype(np.float32))
    has_val = "val_embeddings" in data_npz and "val_expression" in data_npz
    if has_val:
        val_emb = torch.from_numpy(data_npz["val_embeddings"].astype(np.float32))
        val_expr = torch.from_numpy(data_npz["val_expression"].astype(np.float32))

    # Validate shapes
    if train_emb.shape[0] != train_expr.shape[0]:
        raise ValueError("train_embeddings and train_expression must have the same number of rows.")
    if test_emb.shape[0] != test_expr.shape[0]:
        raise ValueError("test_embeddings and test_expression_gt must have the same number of rows.")
    if has_val and val_emb.shape[0] != val_expr.shape[0]:
        raise ValueError("val_embeddings and val_expression must have the same number of rows.")

    # Compute library sizes for nb_libsize mode
    # Library size = total counts per cell/spot
    train_libsize = compute_library_size(train_expr)  # Shape: (n_train, 1)
    test_libsize = compute_library_size(test_expr)    # Shape: (n_test, 1)
    val_libsize = compute_library_size(val_expr) if has_val else None
    
    # Log library size statistics
    if args.loss_type == "nb_libsize":
        print(f"[info] Train library size: mean={train_libsize.mean():.2f}, "
              f"std={train_libsize.std():.2f}, min={train_libsize.min():.2f}, "
              f"max={train_libsize.max():.2f}")
        print(f"[info] Test library size: mean={test_libsize.mean():.2f}, "
              f"std={test_libsize.std():.2f}, min={test_libsize.min():.2f}, "
              f"max={test_libsize.max():.2f}")
        if has_val and val_libsize is not None:
            print(f"[info] Val library size: mean={val_libsize.mean():.2f}, "
                  f"std={val_libsize.std():.2f}, min={val_libsize.min():.2f}, "
                  f"max={val_libsize.max():.2f}")
        print(f"[info] Using fixed library size of {NB_FIXED_LIBSIZE} at test time")

    # Create datasets
    # For nb_libsize, we include library size in the dataset
    if args.loss_type == "nb_libsize":
        train_dataset = TensorDataset(train_emb, train_expr, train_libsize)
        test_dataset = TensorDataset(test_emb, test_expr, test_libsize)
        val_dataset = TensorDataset(val_emb, val_expr, val_libsize) if has_val else test_dataset
    else:
        train_dataset = TensorDataset(train_emb, train_expr)
        test_dataset = TensorDataset(test_emb, test_expr)
        val_dataset = TensorDataset(val_emb, val_expr) if has_val else test_dataset

    # Check for zero-variance genes
    train_zero_std = zero_std_genes(train_expr.numpy(), gene_list)
    test_zero_std = zero_std_genes(test_expr.numpy(), gene_list)
    if train_zero_std:
        print(f"[info] {len(train_zero_std)} train genes have zero std (first: {train_zero_std[:5]})")
    if test_zero_std:
        print(f"[info] {len(test_zero_std)} test genes have zero std (first: {test_zero_std[:5]})")

    if args.disable_slide_batching:
        # Standard random mini-batches
        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": True,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        train_eval_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
        val_eval_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_eval_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    else:
        # Build slide-based batch indices and keep batches slide-homogeneous
        train_slide_indices = build_slide_indices(data_npz["train_slides"])
        test_slide_indices = build_slide_indices(data_npz["test_slides"])
        val_slide_indices = build_slide_indices(data_npz["val_slides"]) if has_val else test_slide_indices

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=SlideBatchSampler(train_slide_indices, shuffle=True),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        train_eval_loader = DataLoader(
            train_dataset,
            batch_sampler=SlideBatchSampler(train_slide_indices, shuffle=False),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_eval_loader = DataLoader(
            val_dataset,
            batch_sampler=SlideBatchSampler(val_slide_indices, shuffle=False),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_eval_loader = DataLoader(
            test_dataset,
            batch_sampler=SlideBatchSampler(test_slide_indices, shuffle=False),
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Initialize decoder
    # For NB losses, we don't need the dropout head (no zero-inflation)
    use_dropout_head = (args.loss_type == "zinb")

    mean_activation = "softplus" if args.loss_type != "nb_libsize" else "softmax"

    decoder = ZINBDecoder(
        embed_dim=train_emb.shape[1],
        hidden_dim=args.hidden_dim,
        n_genes=train_expr.shape[1],
        dropout=args.dropout,
        depth=args.mlp_depth,
        layer_norm=args.layer_norm,
        batch_norm=args.batch_norm,
        residual=args.residual,
        use_dropout_head=use_dropout_head,
        mean_activation=mean_activation,
    ).to(device)

    print(f"[info] Decoder has dropout head: {use_dropout_head}")

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Early stopping setup
    metric_key_map = {"pearson": "pearson_mean", "l2": "l2_mean", "r2": "r2_mean"}
    maximize = args.early_stop_metric in {"pearson", "r2"}
    best_metric = float("-inf") if maximize else float("inf")
    best_state = None
    best_metrics = None
    patience_counter = 0

    # Baseline metrics before any training
    init_train_metrics = evaluate(
        decoder,
        train_eval_loader,
        device,
        gene_list,
        loss_type=args.loss_type,
        metric_transform=args.metric_transform,
        spot_metrics=args.spot_metrics,
    )
    init_val_metrics = evaluate(
        decoder,
        val_eval_loader,
        device,
        gene_list,
        loss_type=args.loss_type,
        metric_transform=args.metric_transform,
        spot_metrics=args.spot_metrics,
    )
    init_test_metrics = evaluate(
        decoder,
        test_eval_loader,
        device,
        gene_list,
        loss_type=args.loss_type,
        metric_transform=args.metric_transform,
        spot_metrics=args.spot_metrics,
    )

    def _safe_metric_val(metrics: dict, key: str) -> float:
        val = metrics.get(key, float("nan"))
        return float(val) if isinstance(val, (int, float, np.floating, np.integer)) else float("nan")

    print(
        "[info] Initial metrics (epoch 0, untrained): "
        f"train pearson={_safe_metric_val(init_train_metrics, 'pearson_mean'):.4f}, "
        f"val pearson={_safe_metric_val(init_val_metrics, 'pearson_mean'):.4f}, "
        f"test pearson={_safe_metric_val(init_test_metrics, 'pearson_mean'):.4f}"
    )

    _log_to_wandb("train", init_train_metrics, epoch=0)
    _log_to_wandb("val", init_val_metrics, epoch=0)
    _log_to_wandb("test", init_test_metrics, epoch=0)

    # Training loop
    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs"):
        decoder.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader):
            # Unpack batch based on loss type
            if args.loss_type == "nb_libsize":
                emb, target, libsize = batch
                libsize = libsize.to(device)
            else:
                emb, target = batch
                libsize = None

            emb = emb.to(device)
            target = target.to(device)
            
            # Forward pass
            mu, theta, dropout = decoder(emb)
            
            # Compute loss based on loss type
            loss = compute_loss(
                mu=mu,
                theta=theta,
                dropout=dropout,
                target=target,
                loss_type=args.loss_type,
                library_size=libsize,
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_norm)
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)

        # Evaluate on train, val, and test sets
        train_metrics = evaluate(
            decoder,
            train_eval_loader,
            device,
            gene_list,
            loss_type=args.loss_type,
            metric_transform=args.metric_transform,
            spot_metrics=args.spot_metrics,
        )
        val_metrics = evaluate(
            decoder,
            val_eval_loader,
            device,
            gene_list,
            loss_type=args.loss_type,
            metric_transform=args.metric_transform,
            spot_metrics=args.spot_metrics,
        )
        test_metrics = evaluate(
            decoder,
            test_eval_loader,
            device,
            gene_list,
            loss_type=args.loss_type,
            metric_transform=args.metric_transform,
            spot_metrics=args.spot_metrics,
        )

        # Log epoch spot-level metrics to WandB (if enabled)
        _log_to_wandb("train", train_metrics, epoch=epoch, extra={"train/loss": float(avg_loss)})
        _log_to_wandb("val", val_metrics, epoch=epoch)
        _log_to_wandb("test", test_metrics, epoch=epoch)

        # Check for early stopping (monitor val if available, else test)
        metric_key = metric_key_map[args.early_stop_metric]
        monitored_source = val_metrics if has_val else test_metrics
        monitored = monitored_source.get(metric_key, float("nan"))
        if math.isnan(monitored):
            monitored = float("-inf") if maximize else float("inf")

        improved = monitored > best_metric if maximize else monitored < best_metric
        if improved:
            best_metric = monitored
            best_state = decoder.state_dict()
            best_metrics = {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "avg_nll": avg_loss,
                "loss_type": args.loss_type,
                "zero_std_genes": {"train": train_zero_std, "test": test_zero_std},
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print(f"[info] Early stopping at epoch {epoch}")
                break

    # If no improvement was ever recorded, use final state
    if best_state is None:
        best_state = decoder.state_dict()
        best_metrics = {
            "train": evaluate(
                decoder,
                train_eval_loader,
                device,
                gene_list,
                loss_type=args.loss_type,
                metric_transform=args.metric_transform,
                spot_metrics=args.spot_metrics,
            ),
            "val": evaluate(
                decoder,
                val_eval_loader,
                device,
                gene_list,
                loss_type=args.loss_type,
                metric_transform=args.metric_transform,
                spot_metrics=args.spot_metrics,
            ),
            "test": evaluate(
                decoder,
                test_eval_loader,
                device,
                gene_list,
                loss_type=args.loss_type,
                metric_transform=args.metric_transform,
                spot_metrics=args.spot_metrics,
            ),
            "loss_type": args.loss_type,
            "zero_std_genes": {"train": train_zero_std, "test": test_zero_std},
        }

    # Save checkpoint
    output_path = Path(args.output) if args.output else Path(args.save_dir) / "zinb_decoder.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "decoder": best_state,
        "gene_list": gene_list,
        "loss_type": args.loss_type,
    }, output_path)

    # Helper to convert numpy types to native Python for JSON serialization
    def _to_native(obj):
        if isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, list):
            return [_to_native(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        return obj

    # Save metrics
    metrics_path = Path(args.metrics_json) if args.metrics_json else Path(args.save_dir) / "decoder_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as fh:
        json.dump(_to_native(best_metrics), fh, indent=2)

    if wandb_run:
        summary: Dict[str, float] = {}
        if "test" in best_metrics and isinstance(best_metrics["test"], dict):
            for key, value in best_metrics["test"].items():
                if (
                    isinstance(value, (int, float, np.floating, np.integer))
                    and key.endswith("_spot")
                ):
                    summary[f"test/{key}"] = float(value)
        if summary:
            wandb_run.summary.update(summary)
        wandb_run.finish()

    return {"checkpoint": str(output_path), "metrics": best_metrics}


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Key arguments for loss type selection:
    --loss-type: Choose from:
        - zinb: Zero-Inflated Negative Binomial (default, original behavior)
        - nb_libsize: NB with library size (true libsize in training, fixed 10k at test)
        - nb_softplus: NB without library size (pure softplus for mean prediction)
    """
    parser = argparse.ArgumentParser(
        description="Train a ZINB/NB decoder on fixed embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Loss Type Selection:
--------------------
--loss-type zinb        : Zero-Inflated Negative Binomial (default)
                          Uses dropout head for zero-inflation modeling.
                          
--loss-type nb_libsize  : Negative Binomial with library size scaling
                          Training: Uses true library size (sum of counts per cell)
                          Testing: Uses fixed 10k library size to avoid data leakage
                          
--loss-type nb_softplus : Negative Binomial without library size
                          Uses pure softplus output for mean prediction.
                          No library size dependency at all.

Example usage:
--------------
# Original ZINB loss
python count_decoder.py --dataset data.npz --loss-type zinb

# NB with library size (10k CPM at test time)
python count_decoder.py --dataset data.npz --loss-type nb_libsize

# NB without library size
python count_decoder.py --dataset data.npz --loss-type nb_softplus
        """,
    )
    
    # Data arguments (choose either NPZ dataset OR AnnData)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Combined NPZ dataset with precomputed embeddings.",
    )
    parser.add_argument(
        "--train-adata",
        default=None,
        help="Path to training AnnData (.h5ad). Uses obsm embeddings and expression from --expression-layer or .X.",
    )
    parser.add_argument(
        "--test-adata",
        default=None,
        help="Path to test AnnData (.h5ad). Uses obsm embeddings and expression from --expression-layer or .X.",
    )
    parser.add_argument(
        "--adata",
        default=None,
        help="Single AnnData (.h5ad) to be split into train/val/test (expression from --expression-layer or .X).",
    )
    parser.add_argument(
        "--embed-key",
        default="cell_emb",
        help="Primary obsm key for embeddings (e.g., cell_emb or neighborhood_emb).",
    )
    parser.add_argument(
        "--embed-fallback-key",
        default="neighborhood_emb",
        help="Fallback obsm key if primary is missing.",
    )
    parser.add_argument(
        "--expression-layer",
        default=None,
        help="AnnData layer to use for expression (e.g., 'counts' or 'lognorm'). Defaults to .X when unset.",
    )
    parser.add_argument(
        "--gene-selection",
        choices=["all", "list", "hvg", "deg"],
        default="all",
        help="Subselect genes before training: 'list' uses --gene-list-path, 'hvg' computes HVGs, 'deg' uses rank_genes_groups.",
    )
    parser.add_argument(
        "--gene-list-path",
        default=None,
        help="Text file with genes (comma or newline separated) when --gene-selection=list.",
    )
    parser.add_argument(
        "--hvg-top-n",
        type=int,
        default=2000,
        help="Top N highly variable genes to keep when --gene-selection=hvg.",
    )
    parser.add_argument(
        "--hvg-flavor",
        default="seurat_v3",
        help="Scanpy flavor for highly_variable_genes when --gene-selection=hvg.",
    )
    parser.add_argument(
        "--hvg-counts-layer",
        default="counts",
        help="Counts layer to use for HVG computation (only for --gene-selection=hvg).",
    )
    parser.add_argument(
        "--deg-label",
        default=None,
        help="obs column name for DEG-based gene selection (union of top genes per group).",
    )
    parser.add_argument(
        "--deg-top-n",
        type=int,
        default=50,
        help="Top N DE genes per group to keep when --gene-selection=deg.",
    )
    parser.add_argument(
        "--deg-method",
        default="wilcoxon",
        help="Method for sc.tl.rank_genes_groups when --gene-selection=deg (e.g., wilcoxon, t-test, logreg).",
    )
    parser.add_argument(
        "--deg-layer",
        default=None,
        help="Optional layer to use for DEG computation; defaults to --expression-layer when unset.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction when using a single AnnData file.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Test fraction when using a single AnnData file.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split when using a single AnnData file.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="WandB project to log metrics; disable logging when unset.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="WandB entity/organization owning the project (optional).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Name for the WandB run (optional).",
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    
    # Model architecture arguments
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument(
        "--disable-slide-batching",
        action="store_true",
        help="Use standard random mini-batches instead of slide-homogeneous batches.",
    )
    
    # Loss type selection - NEW ARGUMENT
    parser.add_argument(
        "--loss-type",
        choices=["zinb", "nb_libsize", "nb_softplus"],
        default="zinb",
        help=(
            "Loss function type. "
            "'zinb': Zero-Inflated NB (default, includes dropout head). "
            "'nb_libsize': NB with library size (true libsize in train, fixed 10k at test). "
            "'nb_softplus': NB without library size (pure softplus output)."
        ),
    )
    
    # Early stopping arguments
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--early-stop-metric",
        choices=["pearson", "l2", "r2"],
        default="pearson",
    )
    
    # Data preprocessing arguments
    parser.add_argument(
        "--drop-zero-variance-genes",
        action="store_true",
        help="Remove genes with zero variance before training/evaluation.",
    )
    parser.add_argument(
        "--metric-transform",
        choices=["none", "log1p"],
        default="none",
        help="Apply a transform (e.g., log1p) to predictions and targets before computing metrics.",
    )
    parser.add_argument(
        "--spot-metrics",
        action="store_true",
        help="Also compute spot-level PCC/MSE/MAE metrics (keys suffixed with _spot).",
    )
    
    # Device and reproducibility
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU id (use -1 for CPU preference).",
    )
    parser.add_argument("--seed", type=int, default=42)
    
    # Output arguments
    parser.add_argument(
        "--save-dir",
        default="./results",
        help="Directory for checkpoints if --output not set.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit checkpoint path.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    
    return parser.parse_args()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Main entry point."""
    args = parse_args()
    result = train(args)
    print(f"Saved checkpoint to: {result['checkpoint']}")
    if result["metrics"]:
        print(f"Loss type: {args.loss_type}")
        if "val" in result["metrics"]:
            print(f"Best val Pearson: {result['metrics']['val']['pearson_mean']:.4f}")
        print(f"Best test Pearson: {result['metrics']['test']['pearson_mean']:.4f}")


if __name__ == "__main__":
    main()
'''
python -m pdb src/app/count_decoder.py --loss-type nb_libsize  --expression-layer counts --disable-slide-batching 
 --wandb-project count_decoder --spot-metrics --gene-selection hvg --hvg-top-n 100
--adata /nfs/team361/dj17/NEMO_DJ/KidneyToxicity/NEMO_UpdatedTokenisationandEmbedding_November22nd25/Labeltransfer/forshare/nemokidneyxeniumatlas_annotated.h5ad
'''

'''
 python -m pdb src/app/count_decoder.py --save-dir results_nb_softplus --loss-type nb_softplus  --expression-layer counts --disable-slide-batching  --wandb-project count_decoder --spot-metrics --gene-selection hvg --hvg-top-n 100 --adata /nfs/team361/dj17/NEMO_DJ/KidneyToxicity/NEMO_UpdatedTokenisationandEmbedding_November22nd25/Labeltransfer/forshare/nemokidneyxeniumatlas_annotated.h5ad
'''


