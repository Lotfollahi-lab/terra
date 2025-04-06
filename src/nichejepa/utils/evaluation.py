from typing import List, Literal

import anndata
import numpy as np
import pandas as pd
import scib_metrics as sm
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def classification_metrics(adata: anndata.AnnData,
                           label_col: str='cell_type',
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



