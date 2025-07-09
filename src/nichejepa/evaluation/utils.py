import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def compute_neighborhood_composition(adata, cell_type_key='cell_type'):
    # Extract required data
    connectivities = adata.obsp['spatial_connectivities']  # shape: (n_cells, n_cells)
    cell_types = adata.obs[cell_type_key].astype('category')  # ensure categorical
    cell_type_names = cell_types.cat.categories
    n_cell_types = len(cell_type_names)

    # One-hot encode the cell types
    one_hot = pd.get_dummies(cell_types).values  # shape: (n_cells, n_cell_types)

    # Convert to sparse for efficiency if needed
    if not isinstance(connectivities, csr_matrix):
        connectivities = csr_matrix(connectivities)

    # Compute neighborhood composition: (n_cells x n_cells) · (n_cells x n_cell_types)
    # Result: (n_cells x n_cell_types) where each row is a vector of neighbor cell type counts
    neighborhood_counts = connectivities.dot(one_hot)

    # Optionally normalize to get proportions
    neighborhood_sums = neighborhood_counts.sum(axis=1, keepdims=True)
    neighborhood_proportions = neighborhood_counts / (neighborhood_sums + 1e-10)

    # Wrap in DataFrame for interpretability
    composition_df = pd.DataFrame(neighborhood_proportions, columns=cell_type_names, index=adata.obs_names)

    return composition_df




def plot_roc_curve(y_true, y_score, title="ROC Curve"):
    """
    Plot ROC curve for binary classification.

    Parameters:
    - y_true: Ground truth binary labels (numpy array of shape (N,))
    - y_score: Predicted scores or probabilities (numpy array of shape (N,))
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return fpr, tpr, roc_auc