"""
Adapted from https://github.com/facebookresearch/dino/blob/main/eval_knn.py
(07.07.2025).
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from torch import nn
from sklearn.metrics import classification_report


def knn_classifier(
        features_train: np.ndarray,
        labels_train: np.ndarray,
        features_test: np.ndarray,
        labels_test: np.ndarray,
        k: int = 20,
        batch_size: int | None = None,
        results_save_path: str | None = None,
    ):
    """
    Simple KNN classifier using cosine similarity and majority voting.
    
    Parameters:
    ----------
    features_train: np.ndarray
        The features of the training data.
    labels_train: np.ndarray
        The labels of the training data.
    features_test: np.ndarray
        The features of the test data.
    labels_test: np.ndarray
        The labels of the test data.
    k: int
        The number of nearest neighbors to use.
    batch_size: int | None
        The batch size to use for the similarity computation. If None, the similarity is computed in a single pass.
    results_save_path: str | None
        The path to save the results. If None, the results are not saved.
    """
    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    features_train = torch.tensor(features_train, dtype=torch.float32, device=device)
    labels_train = torch.tensor(labels_train, dtype=torch.long, device=device)
    features_test = torch.tensor(features_test, dtype=torch.float32, device=device)
    labels_test = torch.tensor(labels_test, dtype=torch.long, device=device)

    # Normalize features to unit length (cosine similarity)
    features_train = F.normalize(features_train, dim=1)
    features_test = F.normalize(features_test, dim=1)

    num_test = features_test.size(0)

    if batch_size is None:
        similarity = torch.matmul(features_test, features_train.T)
        _, indices = similarity.topk(k, dim=1, largest=True, sorted=True)
        neighbor_labels = labels_train[indices]
    else:
        # Compute similarity batch-wise
        all_neighbor_labels = []
        for start_idx in range(0, num_test, batch_size):
            end_idx = min(start_idx + batch_size, num_test)
            batch_features = features_test[start_idx:end_idx]

            similarity = torch.matmul(batch_features, features_train.T)
            _, indices = similarity.topk(k, dim=1, largest=True, sorted=True)
            all_neighbor_labels.append(labels_train[indices])

            del similarity

        neighbor_labels = torch.cat(all_neighbor_labels, dim=0)

    # Perform majority vote
    predictions = torch.empty(num_test, dtype=labels_train.dtype, device=device)
    for i in range(num_test):
        # Count votes
        votes = Counter(neighbor_labels[i].tolist())
        predictions[i] = votes.most_common(1)[0][0]

    # Convert predictions and targets to NumPy for sklearn
    predictions_np = predictions.cpu().numpy()
    labels_test_np = labels_test.cpu().numpy()

    print("\n--- Evaluation Report on Test Set ---")
    cls_report = classification_report(
        labels_test_np, predictions_np, digits=4)
    print(cls_report)

    # Save to a .txt file
    if results_save_path:
        with open(results_save_path, "w") as f:
            f.write(cls_report)

        print("\n--- Evaluation Report saved. ---")

    return predictions_np