import torch
from tqdm import tqdm
import numpy as np
import anndata
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import scib_metrics as sm
from .emb_utils import process_features, mean_nonpadding_embs,create_selection,compute_weight_based_ranks,weighted_mean

def load_cell_neighborhoods(udata, masks_enc, masks_pred, device, args):
    """
    Load cell neighborhoods from given data and masks, returning a dictionary with specific keys.

    Parameters:
    udata (list): List containing data elements. Expected to be of length 3 or 4.
    masks_enc (list): List of encoder masks.
    masks_pred (list): List of predicted masks.
    device (torch.device): Device to load data onto (e.g., CPU or GPU).
    args (dict): Dictionary contains various items to guide label extraction.

    Returns:
    dict: A dictionary containing loaded cell neighborhood data with the following keys:
        - "cell_neighborhood_tokens": The tokens for cell neighborhoods.
        - "seg_label": The segmentation label.
        - "niche_type": The niche label (or None if not available).
        - "cell_type": The cell type (or None if not available).
        - "masks_enc": List of encoder masks loaded to the device.
        - "masks_pred": List of predicted masks loaded to the device.
    """
    # Load cell neighborhood tokens and segmentation label to the specified device
    cell_neighborhood_tokens = udata[0].to(device, non_blocking=True)
    seg_label = udata[1].to(device, non_blocking=True)

    # Initialize niche_label and cell_type based on the length of udata and the just_cell flag
    if len(udata) == 4:
        niche_label = udata[2]
        cell_type = udata[3]
    elif len(udata) == 3:
        if args['data']['just_cell']:
            niche_label = None
            cell_type = udata[2]
        elif args['data']['just_neighborhood']:
            cell_type = None
            niche_label = udata[2]

    # Load masks to the specified device
    masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
    masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]

    # Return the results in a dictionary
    return {
        "cell_neighborhood_tokens": cell_neighborhood_tokens,
        "seg_label": seg_label,
        "niche_type": niche_label,
        "cell_type": cell_type,
        "masks_enc": masks_1,
        "masks_pred": masks_2
    }
def forward_context(model, data_dict, label_name,
        retrieve_label, args,
        split):
    """
    Perform the forward pass of the model and gather average features for each sample.

    Parameters:
    model: The model to be used for the forward pass.
    data_dict (dict): Dictionary containing cell neighborhood tokens and segmentation labels.
    label_name (str): Name of the label that could be cell_type or niche_type
    retrieve_label (str): Name of the label of retrieve portion that could be retrieve_niche, retrieve_cell or retrieve_gene
    args (dict): Dictionary of configs.
    split (str): The split of the dataset (e.g., train, test, validation).

    Returns:
    obs: The obs data that should be stored in the obs of anndata
    features: the features that should store in obsm of the anndata
    """
    # Extract necessary information from the input data dictionary
    cell_neighborhood_tokens = data_dict["cell_neighborhood_tokens"]  # Tokens representing cell neighborhood
    seg_label = data_dict["seg_label"]  # Segmentation label
    retrieve_emb_from_layer = args['emb']['retrieve_emb_from_layer']  # Specifies from which layer to retrieve embeddings
    retrieve_position_emb = args['emb']['retrieve_position_emb']  # Flag to determine if position embeddings should be retrieved

    # If position embeddings are to be retrieved, always retrieve from the first item in the list.
    # Note that retrieve_emb_from_layer has a different meaning here compared to when we extract transformer features.
    # In this context, it refers specifically to retrieving position embeddings before the transformer layers.
    if retrieve_position_emb:
        retrieve_emb_from_layer = 0

    # Retrieve embeddings based on the specified settings
    if retrieve_position_emb:
        # Retrieve position embeddings only
        emb_list = model.module.return_position_emb(cell_neighborhood_tokens)
    else:
        # Retrieve embeddings from multiple layers, based on segmentation labels
        emb_list = model.module.return_multi_layer_emb(cell_neighborhood_tokens, seg_label)

    features_list = []  # Initialize a list to store processed features

    # Ensure that exactly one of 'weighted_average', 'average', or 'cls' is True.
    assert sum([args['emb']['weighted_average'], args['emb']['average'], args['emb']['cls']]) == 1, \
        "Exactly one of 'weighted_average', 'average', or 'cls' must be True."

    # Iterate over the list of embeddings starting from the specified layer
    for emb in emb_list[retrieve_emb_from_layer:]:
        if args['emb']['weighted_average']:
            # Compute weights based on the cell neighborhood tokens and calculate a weighted mean of embeddings
            weight = compute_weight_based_ranks(cell_neighborhood_tokens)
            features = weighted_mean(emb, weight)
        elif args['emb']['average']:
            # Create a selection mask to determine which embeddings to average
            selection = create_selection(
                cell_neighborhood_tokens, label_name,
                args['data']['seq_len_cell'], args, 
                just_cell=args['data']['just_cell'], 
                just_neighborhood=args['data']['just_neighborhood'],
                retrieve_label=retrieve_label
            )
            # Calculate the mean of the non-padding embeddings based on the selection
            features, gene_count = mean_nonpadding_embs(emb, selection)
        elif args['emb']['cls']:
           # Create a selection mask to select cls
           selection = create_selection(
                cell_neighborhood_tokens, label_name,
                args['data']['seq_len_cell'], args,
                just_cell=args['data']['just_cell'],
                just_neighborhood=args['data']['just_neighborhood']
            )
           # Calculate the mean of the non-padding embeddings based on the `selection` mask.
           # This operation will compute the mean embedding where `selection` is 1, which in this case is the first token.
           features, _ = mean_nonpadding_embs(emb, selection)
           # Initialize `gene_count` as None. This may be used later in the code to store the count of selected genes.
           # And for cls it should be None as we don't have any gene
           gene_count = None

        # Convert the features to a NumPy array and add to the list of features
        features_list.append(features.cpu().numpy())

    # Further process the list of features for the final output
    features, obs = process_features(
        features_list, split, label_name, data_dict[label_name], retrieve_label,
        retrieve_position_emb, retrieve_emb_from_layer, gene_count=gene_count
    )


    # Return the processed features and corresponding observations
    return features, obs
def eval_step(model, data_dict, split, args):
    """
    Evaluate the model on the provided context dictionary.

    Parameters:
    model: The model to be used for evaluation.
    data_dict (dict): Dictionary containing cell neighborhood tokens, segmentation labels, niche labels, and cell types.
    split (str): The split of the dataset (e.g., train, test, validation).
    args (dict): Dictionary of Configs.
    """
    with torch.no_grad():
      with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args['meta']['use_bfloat16']):
        if args['emb']['retrieve_niche']:
            return forward_context(model, data_dict, "niche_type", 'retrieve_niche',  args, split)
        elif args['emb']['retrieve_cell']:
            return forward_context(model, data_dict, "cell_type", 'retrieve_cell', args, split)
        elif args['emb']['retrieve_gene']:
            return forward_context(model, data_dict, "cell_type", 'retrieve_gene', args, split)

def process_loader(model, loader, args, split, all_features=None, all_obs=None):
    """
    Process the data loader and evaluate the model on each batch.

    Parameters:
    model: The model to be used for processing.
    loader: Data loader providing batches of data.
    args (dict): Dictionary of Configs.
    split (str): Type of the dataset.
    
    Returns:
    all_obs: The list of all obs computed from different batches, which should be merged and stored in the final AnnData.
    all_features: The list of all features computed from different batches, which should be merged and stored in the final AnnData

    """
    # Determine the device to use: GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Iterate over the data loader with a progress bar
    for itr, (udata, masks_enc, masks_pred) in tqdm(enumerate(loader)):
        # Load and preprocess the cell neighborhood data, moving it to the appropriate device (GPU or CPU)
        data_dict = load_cell_neighborhoods(udata, masks_enc, masks_pred, device, args)
        
        # Perform an evaluation step using the model and the preprocessed data
        features, obs = eval_step(model, data_dict, split, args)
        
        # Append the extracted features and observations to their respective lists
        all_features.append(features)
        all_obs.append(obs)

    # Return two lists that have information for different batch
    return all_features, all_obs
def classification_metrics(adata, label_col='cell_type', classifier='knn', n_neighbors=5):
    """
    Train and evaluate a classifier (KNN or Logistic Regression) on the provided AnnData object.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing the features and labels.
    label_col : str, optional
        The name of the column in `adata.obs` containing the categorical labels for classification.
        Default is 'cell_type'.
    classifier : str, optional
        The type of classifier to use ('knn' or 'logistic'). Default is 'knn'.
    n_neighbors : int, optional
        Number of neighbors to use in KNN classification. Only applicable if `classifier` is 'knn'.
        Default is 5.

    Returns:
    --------
    dict
        A dictionary containing accuracy and F1 scores for both the training and test sets.
    """
    # Extract features and labels
    X = adata.obsm['jepa_emb']
    y = adata.obs[label_col]

    # Create masks for training and testing sets based on the 'split' column
    train_mask = adata.obs['split'] == 'train'
    test_mask = adata.obs['split'] == 'test'

    # Split the data into training and testing sets
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Initialize the classifier
    if classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif classifier == 'logistic':
        clf = LogisticRegression()
    else:
        raise ValueError("Classifier must be either 'knn' or 'logistic'")

    # Train the classifier on the training set
    clf.fit(X_train, y_train)

    # Predict labels on the test set
    y_test_pred = clf.predict(X_test)

    # Evaluate the model on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Predict labels on the training set
    y_train_pred = clf.predict(X_train)

    # Evaluate the model on the training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    # Output the results
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test F1 Score: {test_f1:.2f}")
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Train F1 Score: {train_f1:.2f}")

    # Return the evaluation metrics as a dictionary
    return {
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'train_accuracy': train_accuracy,
        'train_f1': train_f1
    }
def clustering_metrics(adata, emb_key='jepa_emb', label_col='cell_type'):
    """
    Evaluate clustering metrics (NMI and ARI) using KMeans clustering on the provided AnnData object.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing the embeddings and labels.
    emb_key : str, optional
        The key in `adata.obsm` corresponding to the embeddings to use for clustering.
        Default is 'jepa_emb'.
    label_col : str, optional
        The name of the column in `adata.obs` containing the true labels for comparison.
        Default is 'cell_type'.

    Returns:
    --------
    dict
        A dictionary containing the NMI and ARI scores.
    """
    # Validate input
    if emb_key not in adata.obsm:
        raise ValueError(f"Embedding key '{emb_key}' not found in `adata.obsm`.")
    if label_col not in adata.obs:
        raise ValueError(f"Label column '{label_col}' not found in `adata.obs`.")

    # Extract the embeddings and labels
    embeddings = adata.obsm[emb_key]
    true_labels = adata.obs[label_col]

    # Calculate NMI and ARI using KMeans clustering
    results = sm.nmi_ari_cluster_labels_kmeans(embeddings, true_labels)

    # Output the results
    print(f"NMI (Normalized Mutual Information): {results['nmi']}")
    print(f"ARI (Adjusted Rand Index): {results['ari']}")

    # Return the results as a dictionary
    return {
        'nmi': results['nmi'],
        'ari': results['ari']
    }


