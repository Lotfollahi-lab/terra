"""
Adapted from https://github.com/facebookresearch/dino/blob/main/eval_linear.py
(07.07.2025).
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy.spatial.distance as dist
from sklearn.metrics import (classification_report,
                             mean_absolute_error,
                             mean_squared_error)


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


class LinearRegressor(nn.Module):
    """Linear regression layer for predicting cell type compositions"""
    def __init__(self, num_features: int, num_outputs: int):
        super().__init__()
        self.linear = nn.Linear(num_features, num_outputs)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.softmax(self.linear(x), dim=-1)
        
        return x


def linear_classifier(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray | None,
    labels_val: np.ndarray | None,
    features_test: np.ndarray | None,
    labels_test: np.ndarray | None,
    n_epochs: int = 400,
    batch_size: int = 128,
    lr: float = 0.001,
    patience: int = 10,
    n_classes: int | None = None,
    results_save_path: str | None = None,
    ):
    """
    Train a linear classifier with optional early stopping.
    
    Args:
        features_val, labels_val: If provided, used for early stopping.
        features_test, labels_test: If provided, used for final evaluation.
        
    Note: At least one of (val, test) must be provided.
        - If test is None: early stopping on val, report on val.
        - If val is None: no early stopping, report on test.
        - If both provided: early stopping on val, report on test.
    """
    
    # --- Validate inputs ---
    has_val = features_val is not None and labels_val is not None
    has_test = features_test is not None and labels_test is not None
    
    if not has_val and not has_test:
        raise ValueError("At least one of (val, test) must be provided.")
    
    # Determine evaluation set (val if no test, otherwise test)
    if has_test:
        features_eval, labels_eval = features_test, labels_test
    else:
        features_eval, labels_eval = features_val, labels_val

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    features_train = torch.tensor(features_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)
    features_eval = torch.tensor(features_eval, dtype=torch.float32)
    labels_eval = torch.tensor(labels_eval, dtype=torch.long)
    
    if has_val:
        features_val_t = torch.tensor(features_val, dtype=torch.float32)
        labels_val_t = torch.tensor(labels_val, dtype=torch.long)
        val_dataset = TensorDataset(features_val_t, labels_val_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_dataset = TensorDataset(features_train, labels_train)
    eval_dataset = TensorDataset(features_eval, labels_eval)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # --- Model, Optimizer, Scheduler ---
    num_features = features_train.shape[1]
    num_classes = n_classes if n_classes is not None else len(torch.unique(labels_train))
    print(f"Number of classes: {num_classes}")
    
    model = LinearClassifier(num_features, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr * batch_size / 256.,
        momentum=0.9,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=0)

    # --- Early Stopping (only if val is provided) ---
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # --- Training Loop ---
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        scheduler.step()
        
        # --- Early stopping on validation set (if provided) ---
        if has_val:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    # Track accuracy
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_labels).sum().item()
                    val_total += batch_labels.size(0)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

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
        else:
            # No validation set - just print train loss and accuracy
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")

    # --- Load best model before evaluation (only if we did early stopping) ---
    if has_val and best_model_state is not None:
        model.load_state_dict(best_model_state)

    # --- Evaluation on eval set (test if available, otherwise val) ---
    model.eval()
    all_logits = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_features, batch_labels in eval_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)

            all_logits.extend(outputs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())

    eval_set_name = "Test" if has_test else "Validation"
    print(f"\n--- Evaluation Report on {eval_set_name} Set ---")
    cls_report = classification_report(
        all_targets, all_preds, digits=4)
    print(cls_report)

    # Save to a .txt file
    if results_save_path:
        with open(results_save_path, "w") as f:
            f.write(cls_report)
        print("\n--- Evaluation Report saved. ---")

    return all_preds, all_targets, all_logits, model


def linear_regressor(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    n_epochs: int = 400,
    batch_size: int = 128,
    lr: float = 0.001,
    patience: int = 10,
    results_save_path: str | None = None,
    ):
    """
    Train a linear regressor with early stopping on validation loss.
    Designed for multi-output regression of cell type compositions.
    """

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    features_train = torch.tensor(features_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.float32)
    features_val = torch.tensor(features_val, dtype=torch.float32)
    labels_val = torch.tensor(labels_val, dtype=torch.float32)
    features_test = torch.tensor(features_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.float32)

    train_dataset = TensorDataset(features_train, labels_train)
    val_dataset = TensorDataset(features_val, labels_val)
    test_dataset = TensorDataset(features_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model, Optimizer, Scheduler ---
    num_features = features_train.shape[1]
    num_outputs = labels_train.shape[1] # all classes need to be in train split
    model = LinearRegressor(num_features, num_outputs).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr * batch_size / 256.,
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
            all_preds.append(outputs.cpu())
            all_targets.append(batch_labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    print("\n--- Evaluation Report on Test Set ---")
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    jsd_values = dist.jensenshannon(all_targets, all_preds, axis=-1)
    avg_jsd = jsd_values.mean()
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Average Jensen-Shannon Divergence: {avg_jsd:.4f}")
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "JSD": float(avg_jsd)}

    # Save to a .txt file
    if results_save_path:
        with open(results_save_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print("\n--- Evaluation Metrics saved. ---")

    return all_preds, model