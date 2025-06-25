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



