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
from sklearn.model_selection import train_test_split


def knn_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    test_ratio: float = 0.1,
    seed: int = 0,
    k: int = 20):
    """
    Simple KNN classifier using cosine similarity and majority voting.
    """
    # Train-test split: 80% train, 20% test
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_ratio,
        random_state=seed,
        #stratify=labels
    )

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Normalize features to unit length (cosine similarity)
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)

    # Compute cosine similarity
    similarity = torch.matmul(test_features, train_features.T)

    # Get top-k most similar train examples for each test feature
    distances, indices = similarity.topk(
        k, dim=1, largest=True, sorted=True)
    neighbor_labels = train_labels[indices] # shape: [num_test, k]

    # Perform majority vote
    num_test = test_features.size(0)
    predictions = torch.empty(
        num_test,
        dtype=train_labels.dtype,
        device=train_labels.device)
    for i in range(num_test):
        # Count votes
        votes = Counter(neighbor_labels[i].tolist())
        predictions[i] = votes.most_common(1)[0][0]

    # Convert predictions and targets to NumPy for sklearn
    predictions_np = predictions.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    print("\n--- Evaluation Report on Test Set ---")
    print(classification_report(test_labels_np, predictions_np, digits=4))

    return predictions_np