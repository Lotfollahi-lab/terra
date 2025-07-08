"""
Adapted from https://github.com/facebookresearch/dino/blob/main/eval_linear.py
(07.07.2025).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report



class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, num_features: int, num_classes: int):
        super(LinearClassifier, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(num_features, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # Flatten
        x = x.view(x.size(0), -1)

        return self.linear(x)


def linear_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0,
    n_epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.001,
    patience: int = 10,
    ):
    """
    Train a linear classifier with early stopping on validation loss.
    """

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # --- Dataset Splitting ---
    dataset = TensorDataset(features, labels)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model, Optimizer, Scheduler ---
    num_features = features.shape[1]
    num_classes = len(torch.unique(labels))
    model = LinearClassifier(num_features, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr * batch_size / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=0)

    # --- Early Stopping ---
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # --- Training Loop ---
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # --- Load best model before test ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # --- Test Evaluation ---
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())

    print("\n--- Evaluation Report on Test Set ---")
    print(classification_report(all_targets, all_preds, digits=4))

    return all_preds, model
