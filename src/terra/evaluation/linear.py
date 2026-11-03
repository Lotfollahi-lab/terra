"""
Adapted from https://github.com/facebookresearch/dino/blob/main/eval_linear.py
(07.07.2025).
"""

import copy
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy.spatial.distance as dist
from sklearn.metrics import (classification_report,
                             mean_absolute_error,
                             mean_squared_error)


logger = logging.getLogger(__name__)


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 generator: torch.Generator | None = None):
        super(LinearClassifier, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(num_features, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01,
                                        generator=generator)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # Flatten
        x = x.view(x.size(0), -1)

        return self.linear(x)


class LinearRegressor(nn.Module):
    """Linear regression layer for predicting cell type compositions"""
    def __init__(self,
                 num_features: int,
                 num_outputs: int,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.linear = nn.Linear(num_features, num_outputs)
        self.linear.weight.data.normal_(mean=0.0, std=0.01,
                                        generator=generator)
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
    restore_best: bool = True,
    seed: int | None = None,
    ):
    """
    Train a linear classifier with optional early stopping.

    Args:
        features_val, labels_val: If provided, used for early stopping.
        features_test, labels_test: If provided, used for final evaluation.
        restore_best: If `True` (default) and a validation set is given,
            the reported model has the weights of the epoch with the lowest
            validation loss. If `False`, it has the weights of the last
            epoch (patience epochs after the best one when training stops
            early), which reproduces earlier TERRA versions: they intended
            to restore the best weights but kept references to the live
            parameters, so the last weights were evaluated.
        seed: If given, seeds the initialization of the linear head and the
            shuffling of the training data with a local generator, so
            results are reproducible and independent of the global random
            state. `None` (default) uses the global random state, as before.

    Returns:
        all_preds, all_targets, all_logits: Predictions, targets and logits
            of the evaluation set.
        model: The trained classifier.

    Note: At least one of (val, test) must be provided.
        - If test is None: early stopping on val, report on val (the
          reported performance is then measured on the data used for early
          stopping, so it is optimistic; pass a separate test set for
          unbiased estimates).
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
    logger.info(f"Using device: {device}")

    features_train = torch.tensor(features_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)
    features_eval = torch.tensor(features_eval, dtype=torch.float32)
    labels_eval = torch.tensor(labels_eval, dtype=torch.long)

    if has_val:
        features_val_t = torch.tensor(features_val, dtype=torch.float32)
        labels_val_t = torch.tensor(labels_val, dtype=torch.long)
        val_dataset = TensorDataset(features_val_t, labels_val_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Local generator for the head initialization and the shuffling (None:
    # global random state)
    generator = (torch.Generator().manual_seed(seed) if seed is not None
                 else None)

    train_dataset = TensorDataset(features_train, labels_train)
    eval_dataset = TensorDataset(features_eval, labels_eval)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=generator)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # --- Model, Optimizer, Scheduler ---
    num_features = features_train.shape[1]
    num_classes = n_classes if n_classes is not None else len(torch.unique(labels_train))
    logger.info(f"Number of classes: {num_classes}")

    model = LinearClassifier(num_features, num_classes,
                             generator=generator).to(device)

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
            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # A copy: state_dict() returns references to the live
                # parameters, which the following epochs update in place
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            # No validation set - just print train loss and accuracy
            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")

    # --- Load best model before evaluation (only if we did early stopping) ---
    if has_val and restore_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Evaluating the weights of epoch {best_epoch} (lowest "
                    f"validation loss).")

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
    logger.info(f"\n--- Evaluation Report on {eval_set_name} Set ---")
    cls_report = classification_report(
        all_targets, all_preds, digits=4)
    logger.info(cls_report)

    # Save to a .txt file
    if results_save_path:
        with open(results_save_path, "w") as f:
            f.write(cls_report)
        logger.info("\n--- Evaluation Report saved. ---")

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
    restore_best: bool = True,
    seed: int | None = None,
    ):
    """
    Train a linear regressor with early stopping on validation loss.
    Designed for multi-output regression of cell type compositions.

    Args:
        restore_best: If `True` (default), the test set is evaluated with the
            weights of the epoch with the lowest validation loss. If `False`,
            with the weights of the last epoch, which reproduces earlier TERRA
            versions (see `linear_classifier`).
        seed: If given, seeds the initialization of the linear head and the
            shuffling of the training data with a local generator. `None`
            (default) uses the global random state, as before.

    Returns:
        all_preds: Predicted compositions of the test set.
        model: The trained regressor.

    The metrics (logged, and saved as JSON to `results_save_path`) are the
    MAE, the MSE and the Jensen-Shannon distance between the true and the
    predicted compositions (`scipy.spatial.distance.jensenshannon`: the
    square root of the Jensen-Shannon divergence, natural logarithm), averaged
    over the test cells. It is stored as `JSD` (as in earlier versions) and as
    `JS_distance`; `JS_divergence` is the average divergence. Cells with an
    all-zero composition, for which both are undefined, are left out of the
    averages (with a warning); they made the averages NaN before.
    """

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    features_train = torch.tensor(features_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.float32)
    features_val = torch.tensor(features_val, dtype=torch.float32)
    labels_val = torch.tensor(labels_val, dtype=torch.float32)
    features_test = torch.tensor(features_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.float32)

    train_dataset = TensorDataset(features_train, labels_train)
    val_dataset = TensorDataset(features_val, labels_val)
    test_dataset = TensorDataset(features_test, labels_test)

    # Local generator for the head initialization and the shuffling (None:
    # global random state)
    generator = (torch.Generator().manual_seed(seed) if seed is not None
                 else None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model, Optimizer, Scheduler ---
    num_features = features_train.shape[1]
    num_outputs = labels_train.shape[1] # all classes need to be in train split
    model = LinearRegressor(num_features, num_outputs,
                            generator=generator).to(device)

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

        logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # A copy: state_dict() returns references to the live
            # parameters, which the following epochs update in place
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # --- Load best model before test ---
    if restore_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Evaluating the weights of epoch {best_epoch} (lowest "
                    f"validation loss).")

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

    logger.info("\n--- Evaluation Report on Test Set ---")
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    # scipy returns the Jensen-Shannon distance, the square root of the
    # divergence (natural logarithm), which is NaN for an all-zero composition
    with np.errstate(divide='ignore', invalid='ignore'):
        js_distance = dist.jensenshannon(all_targets, all_preds, axis=-1)
    defined = np.isfinite(js_distance)
    if not defined.all():
        logger.warning(
            f"{int((~defined).sum())} of {len(js_distance)} test cells have "
            "an all-zero composition, for which the Jensen-Shannon distance is "
            "undefined; they are left out of its average.")
    avg_js_distance = (float(js_distance[defined].mean()) if defined.any()
                       else float('nan'))
    avg_js_divergence = (float((js_distance[defined] ** 2).mean())
                         if defined.any() else float('nan'))
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"Average Jensen-Shannon distance (natural log; 'JSD'): "
                f"{avg_js_distance:.4f}")
    logger.info(f"Average Jensen-Shannon divergence (natural log): "
                f"{avg_js_divergence:.4f}")
    metrics = {
        "MAE": mae,
        "MSE": mse,
        # The Jensen-Shannon distance under its earlier name
        "JSD": avg_js_distance,
        "JS_distance": avg_js_distance,
        "JS_divergence": avg_js_divergence}

    # Save to a .txt file
    if results_save_path:
        with open(results_save_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("\n--- Evaluation Metrics saved. ---")

    return all_preds, model