from typing import Literal

import anndata
import numpy as np
import pandas as pd
import scib_metrics as sm
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import wasserstein_distance
from typing import List


def classification_metrics(adata: anndata.AnnData,
                           label_col: str = 'cell_type',
                           classifier: Literal['knn', 'logistic'] = 'knn',
                           n_neighbors: int | None = 5
                           ) -> dict:
    """
    Train and evaluate a classifier (KNN or Logistic Regression) on
    learned embeddings.

    Parameters
    -----------
    adata:
        Annotated data object containing the features and labels.
    label_col:
        The name of the column in `adata.obs` containing the categorical labels
        for classification.
    classifier:
        The type of classifier to use ('knn' or 'logistic').
    n_neighbors:
        Number of neighbors to use in KNN classification. Only applicable if
        `classifier` is 'knn'.

    Returns
    --------
    metrics:
        A dictionary containing accuracy and F1 scores for both the training and
        test sets.
    """
    # Split the data into training and testing sets
    X = adata.obsm['jepa_emb']
    y = adata.obs[label_col]
    train_mask = adata.obs['split'] == 'train'
    test_mask = adata.obs['split'] == 'test'
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Train the classifier
    if classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif classifier == 'logistic':
        clf = LogisticRegression()
    else:
        raise ValueError("Classifier must be either 'knn' or 'logistic'")
    clf.fit(X_train, y_train)

    # Evaluate the model on the test and training sets
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test F1 Score: {test_f1:.2f}")
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Train F1 Score: {train_f1:.2f}")
    metrics = {'test_accuracy': test_accuracy,
               'test_f1': test_f1,
               'train_accuracy': train_accuracy,
               'train_f1': train_f1}

    return metrics


def clustering_metrics(adata: anndata.AnnData,
                       emb_key: str='jepa_emb',
                       label_col: str='cell_type'
                       ) -> dict:
    """
    Compute clustering metrics (NMI and ARI) using K-Means clustering.

    Parameters
    -----------
    adata:
        Annotated data object containing the embeddings and ground truth labels.
    emb_key:
        The key in `adata.obsm` containing the embeddings to use for clustering.
    label_col:
        The name of the column in `adata.obs` containing the ground truth labels
        for comparison.

    Returns
    --------
    metrics:
        A dictionary containing the NMI and ARI scores.
    """
    # Validate input
    if emb_key not in adata.obsm:
        raise ValueError(
            f"Embedding key '{emb_key}' not found in `adata.obsm`.")
    if label_col not in adata.obs:
        raise ValueError(
            f"Label column '{label_col}' not found in `adata.obs`.")

    # Calculate NMI and ARI using K-Means clustering
    embeddings = adata.obsm[emb_key]
    true_labels = adata.obs[label_col]
    results = sm.nmi_ari_cluster_labels_kmeans(embeddings, true_labels)
    print(f"NMI (Normalized Mutual Information): {results['nmi']}")
    print(f"ARI (Adjusted Rand Index): {results['ari']}")
    metrics = {'nmi': results['nmi'],
               'ari': results['ari']}
               
    return metrics

def compute_energy_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute the energy distance using squared Euclidean distances between two multi-dimensional samples.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_samples, n_features) -- first distribution.
    y : np.ndarray
        Shape (n_samples, n_features) -- second distribution.

    Returns
    -------
    e_distance : float
        Energy distance between the two distributions.
    """
    sigma_X = pairwise_distances(x, x, metric="sqeuclidean").mean()
    sigma_Y = pairwise_distances(y, y, metric="sqeuclidean").mean()
    delta = pairwise_distances(x, y, metric="sqeuclidean").mean()
    return 2 * delta - sigma_X - sigma_Y

def compute_maximum_mean_discrepancy(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0
) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) using the RBF kernel between two samples.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_samples, n_features) -- first distribution.
    y : np.ndarray
        Shape (n_samples, n_features) -- second distribution.
    gamma : float
        RBF kernel bandwidth parameter.

    Returns
    -------
    mmd : float
        Maximum mean discrepancy between the two distributions.
    """
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_scalar_mmd(
    x: np.ndarray,
    y: np.ndarray,
    gammas: list[float] = None
) -> float:
    """
    Compute scalar MMD as an average across multiple RBF bandwidths.

    Parameters
    ----------
    x : np.ndarray
        First sample array.
    y : np.ndarray
        Second sample array.
    gammas : list of float, optional
        List of RBF kernel bandwidths. Defaults to [2, 1, 0.5, 0.1, 0.01, 0.005].

    Returns
    -------
    mmd_value : float
        Averaged MMD value across all specified gamma values.
    """
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    mmds = [compute_maximum_mean_discrepancy(x, y, gamma=g) for g in gammas]
    return np.nanmean(mmds)

def compute_emd(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute the 1D Earth Mover's Distance (Wasserstein-1) between two multidimensional samples
    by averaging the EMD computed on each feature dimension.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_samples, n_features) -- first distribution.
    y : np.ndarray
        Shape (n_samples, n_features) -- second distribution.

    Returns
    -------
    emd_value : float
        The average 1D EMD across all feature dimensions.
    """
    emds = []
    # Compute 1D EMD for each feature
    for dim in range(x.shape[1]):
        emd_dim = wasserstein_distance(x[:, dim], y[:, dim])
        emds.append(emd_dim)
    return float(np.mean(emds))


def get_top_gene_pairs(
    cos_sim_ratio:         torch.Tensor,
    count_cell:            torch.Tensor,
    count_neb:             torch.Tensor,
    cos_sim_cell:          torch.Tensor,
    cos_sim_neb:           torch.Tensor,
    gene_df:               pd.DataFrame,
    cell_gene_ids:         List[str],
    neighborhood_gene_ids: List[str],
    min_count: int = 80,
    sim_thresh: float = 0.15,
    k: int = 500
) -> pd.DataFrame:
    """
    Returns a DataFrame of the top-k (row, col) pairs by cos_sim_ratio,
    including Ensembl IDs and gene names.

    Expects gene_df.index == gene_name, and gene_df['ensembl_id'] == Ensembl IDs.
    """

    # Build a fast lookup from Ensembl → gene_name
    id_to_name = {ensg: name for name, ensg in gene_df['ensembl_id'].items()}

    # 1) Mask diagonal
    ratio = cos_sim_ratio.clone()
    ratio.fill_diagonal_(float('nan'))

    # 2) Valid‑entry mask
    base_counts = (count_cell >= min_count) & (count_neb >= min_count)
    top_sim     = (cos_sim_cell >= sim_thresh) & (cos_sim_neb >= sim_thresh)
    valid_mask  = base_counts & top_sim & ~torch.isnan(ratio)

    # 3) Flatten & top‑k
    flat_vals = ratio.flatten()
    flat_mask = valid_mask.flatten()
    valid_vals = flat_vals[flat_mask]
    n_valid    = valid_vals.numel()
    top_k      = min(k, n_valid)
    top_vals, top_idx = torch.topk(valid_vals, top_k, largest=True)

    valid_indices    = flat_mask.nonzero(as_tuple=True)[0]
    top_flat_indices = valid_indices[top_idx]
    N = ratio.size(1)
    rows = (top_flat_indices // N).cpu().tolist()
    cols = (top_flat_indices %  N).cpu().tolist()
    vals = top_vals.cpu().tolist()

    # 4) Build a records list
    records = []
    for r, c, v in zip(rows, cols, vals):
        cell_ensg = cell_gene_ids[r]
        neb_ensg  = neighborhood_gene_ids[c]
        cell_name = id_to_name.get(cell_ensg, "UNKNOWN")
        neb_name  = id_to_name.get(neb_ensg,  "UNKNOWN")
        records.append({
            'row_idx'        : r,
            'col_idx'        : c,
            'cell_ensembl'   : cell_ensg,
            'neb_ensembl'    : neb_ensg,
            'cell_gene_name' : cell_name,
            'neb_gene_name'  : neb_name,
            'gene_pair_score'          : v
        })

    # 5) Return as pandas.DataFrame
    return pd.DataFrame.from_records(
        records,
        columns=[
            'row_idx','col_idx',
            'cell_ensembl','neb_ensembl',
            'cell_gene_name','neb_gene_name',
            'gene_pair_score'
        ]
    )


def get_top_gene_score(
    gene_pair_score: np.ndarray,
    cell_gene_ensembl_id: list,
    gene_df: pd.DataFrame,
    gene_counts: np.ndarray,
    min_count: int = 80
) -> pd.DataFrame:
    """
    Compute gene scores from a cosine similarity matrix and add gene names,
    filtering genes by a minimum count threshold.

    Args:
        gene_pair_score (np.ndarray): Square cosine similarity matrix (n_genes, n_genes).
        cell_gene_ensembl_id (list or np.ndarray): List/array of gene Ensembl IDs of length n_genes.
        gene_df (pd.DataFrame): DataFrame with columns ['gene_name', 'ensembl_id'].
        gene_counts (np.ndarray): Square count matrix (n_genes, n_genes); diagonal contains gene-level counts.
        min_count (int): Minimum count threshold to filter genes.

    Returns:
        pd.DataFrame: DataFrame with columns ['gene_ensembl', 'gene_name', 'gene_score']
                      for genes with valid diagonal scores and counts >= min_count.
    """
    # 1) Extract diagonals
    diag_scores = np.diag(gene_pair_score)   # similarity values
    diag_counts = np.diag(gene_counts)       # counts per gene
    
    # 2) Filter by valid (non-NaN) scores and min count
    valid_mask = (~np.isnan(diag_scores)) & (diag_counts >= min_count)
    valid_idx = np.where(valid_mask)[0]
    
    # 3) Fetch gene IDs and scores
    gene_ids = np.array(cell_gene_ensembl_id, dtype=str)
    valid_genes = gene_ids[valid_idx]
    valid_scores = diag_scores[valid_mask]
    
    # 4) Map Ensembl ID to gene name
    id_to_name = {ensg: name for name, ensg in gene_df['ensembl_id'].items()}
    valid_names = [id_to_name.get(gene_id, 'UNKNOWN') for gene_id in valid_genes]
    
    # 5) Assemble into DataFrame
    gene_score = pd.DataFrame({
        'gene_ensembl': valid_genes,
        'gene_name': valid_names,
        'gene_score': valid_scores
    })
    
    return gene_score


