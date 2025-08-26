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
    results_save_path: str | None = None,
    ):
    """
    Simple KNN classifier using cosine similarity and majority voting.
    """
    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    features_train = torch.tensor(features_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)
    features_test = torch.tensor(features_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)

    # Normalize features to unit length (cosine similarity)
    features_train = F.normalize(features_train, dim=1)
    features_test = F.normalize(features_test, dim=1)

    # Compute cosine similarity
    similarity = torch.matmul(features_test, features_train.T)

    # Get top-k most similar train examples for each test feature
    distances, indices = similarity.topk(
        k, dim=1, largest=True, sorted=True)
    neighbor_labels = labels_train[indices] # shape: [num_test, k]

    # Perform majority vote
    num_test = features_test.size(0)
    predictions = torch.empty(
        num_test,
        dtype=labels_train.dtype,
        device=labels_train.device)
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