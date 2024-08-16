from .emb_utils import create_anndata, mean_nonpadding_embs,create_selection,compute_weight_based_ranks,weighted_mean

import torch
from tqdm import tqdm
import numpy as np

import anndata
import pandas as pd

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
        - "niche_label": The niche label (or None if not available).
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
        "niche_label": niche_label,
        "cell_type": cell_type,
        "masks_enc": masks_1,
        "masks_pred": masks_2
    }

def forward_context(model, data_dict, label_name,
        label_value, layer_index, just_pos, args,
        dataset_type, top_layer):
    """
    Perform the forward pass of the model and gather average features for each sample.

    Parameters:
    model: The model to be used for the forward pass.
    data_dict (dict): Dictionary containing cell neighborhood tokens and segmentation labels.
    label_name (str): Name of the label.
    label_value: Value of the label.
    layer_index (int): Index of the layer to be used.
    just_pos (bool): Flag for position-only processing.
    args (dict): Dictionary of arguments.
    dataset_type (str): Type of the dataset.
    top_layer (int): Top layer to consider for feature extraction.

    Returns:
    obs: The obs data that should be stored in the obs of anndata
    features: the features that should store in obsm of the anndata

    """

    cell_neighborhood_tokens = data_dict["cell_neighborhood_tokens"]
    seg_label = data_dict["seg_label"]

    if args['optimization']['epochs'] == 0:
        emb_list = model.module.return_position_emb(cell_neighborhood_tokens)
    else:
        emb_list = model.module.return_multi_layer_emb(cell_neighborhood_tokens, seg_label)

    features_list = []
    for emb in emb_list[top_layer - 1:]:

        if args['data']['weighted_average']:
            weight =  compute_weight_based_ranks(cell_neighborhood_tokens)
            features = weighted_mean(emb,weight)
        else:
            selection = create_selection(cell_neighborhood_tokens, label_name, args['data']['seq_len_cell'], just_cell=args['data']['just_cell'], just_neighborhood=args['data']['just_neighborhood'], get_specefic_gene=args['data']['get_specefic_gene'],
                    gene_id=args['data']['gene_id'])
            features = mean_nonpadding_embs(emb, selection)

        features_list.append(features.cpu().numpy())
      
    features, obs = create_anndata(features_list, dataset_type, label_name, label_value, just_pos, layer_index)

    return features, obs

    
def eval_step(model, data_dict, dataset_type, args, top_layer):
    """
    Evaluate the model on the provided context dictionary.

    Parameters:
    model: The model to be used for evaluation.
    data_dict (dict): Dictionary containing cell neighborhood tokens, segmentation labels, niche labels, and cell types.
    dataset_type (str): Type of the dataset.
    args (dict): Dictionary of arguments.
    top_layer (int): Top layer to consider for feature extraction.
    """
    with torch.no_grad():
        if args['data']['just_neighborhood']:
            return forward_context(model, data_dict, "niche_type", data_dict["niche_label"], 0, False, args, dataset_type, top_layer)
        if args['data']['just_cell']:
            return forward_context(model, data_dict, "cell_type", data_dict["cell_type"], 0, False, args, dataset_type, top_layer)

def process_loader(model, loader, args, dataset_type, top_k=0, all_features=None, all_obs=None):
    """
    Process the data loader and evaluate the model on each batch.

    Parameters:
    model: The model to be used for processing.
    loader: Data loader providing batches of data.
    args (dict): Dictionary of arguments.
    dataset_type (str): Type of the dataset.
    top_k (int): Top k layers to consider for feature extraction.
    
    Returns:
    all_obs: The list of all obs computed from different batches, which should be merged and stored in the final AnnData.
    all_features: The list of all features computed from different batches, which should be merged and stored in the final AnnData.


    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for itr, (udata, masks_enc, masks_pred) in tqdm(enumerate(loader)):
        data_dict = load_cell_neighborhoods(udata, masks_enc, masks_pred, device, args)
        features, obs = eval_step(model, data_dict, dataset_type, args, top_layer=top_k)
        all_features.append(features)
        all_obs.append(obs)
    return  all_features, all_obs
          
