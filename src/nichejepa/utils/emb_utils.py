import torch
import anndata
import numpy as np
import pandas as pd

def compute_weight_based_ranks(tokens):
    """
    Compute rank-based weights for a 2D tensor of tokens.

    Parameters:
    tokens (torch.Tensor): A 2D tensor where each row represents a sequence of tokens. The tokens are gene_id of cell or neighborhood

    Returns:
    torch.Tensor: A 2D tensor of the same shape as `tokens` containing the computed weights.
    """
    # Create a mask for non-zero tokens
    mask = tokens != 0
    # Compute the ranks based on the mask
    ranks = mask.cumsum(dim=1).float() * mask.float()

    rank_max = ranks.max(dim=1, keepdim=True)[0]
    rank_sum = ranks.sum(dim=1, keepdim=True)
    weights = (rank_max - ranks + 1) / rank_sum
    # Mask rank of padding tokens 
    weights = weights * mask.float()

    return weights

def weighted_mean(items, weights, dim=1):
    """
    Compute the weighted mean of items.

    Parameters:
    items (torch.Tensor): The input items tensor (can be 2D or 3D).
    weights (torch.Tensor): A tensor of weights (same size as the relevant dimension of items).
    dim (int): The dimension along which to compute the weighted mean.

    Returns:
    torch.Tensor: The weighted mean tensor.
    """
    # Use broadcasting to multiply items by weights
    if items.dim() == 3:
        weighted_items = items * weights.unsqueeze(2)  # Broadcasting weights to match items dimensions
        weighted_sum = weighted_items.sum(dim)
        weights_sum = weights.sum(dim).unsqueeze(1)  # Sum weights along the specified dimension and keep the dimensions consistent
        weighted_mean = weighted_sum / weights_sum

    elif items.dim() == 2:
        weighted_items = items * weights  # Broadcasting weights to match items dimensions
        weighted_sum = weighted_items.sum(dim)
        weights_sum = weights.sum(dim)  # Sum weights along the specified dimension
        weighted_mean = weighted_sum / weights_sum

    return weighted_mean

def mean_nonpadding_embs(embs, mask, dim=1):
    """
    Compute the mean of non-padding embeddings.
    
    Parameters:
    embs (torch.Tensor): The input embeddings tensor (can be 2D or 3D).
    mask (torch.Tensor): A boolean mask tensor indicating the non-padding or cls positions (same size as the relevant dimension of embs).
    dim (int): The dimension along which to compute the mean.
    
    Returns:
    torch.Tensor: The mean embeddings tensor.
    """
    # Use broadcasting to sum across non-padding positions
    if embs.dim() == 3:
        masked_embs = embs * mask.unsqueeze(2)  # Broadcasting mask to match embs dimensions
        sum_embs = masked_embs.sum(dim)
        mean_embs = sum_embs / mask.sum(dim).view(-1, 1).float()

    elif embs.dim() == 2:
        masked_embs = embs * mask  # Broadcasting mask to match embs dimensions
        sum_embs = masked_embs.sum(dim)
        mean_embs = sum_embs / mask.sum(dim).float()
        
    return mean_embs
#change name here
def create_selection(cell_neighborhood_tokens, label_name, seq_len_cell, top_k=None, 
                          just_cell=False, just_neighborhood=False, 
                          gene_id=None, mask_large_than_k=False,
                          get_specefic_gene=False):
    """
    Create a selection mask for cell index tokens or neighborhood tokens based on various conditions.

    Parameters:
    cell_neighborhood_tokens (torch.Tensor): Tensor containing cell or neighborhood tokens.
    label_name (str): Label name to determine selection rules.
    seq_len_cell (int): Sequence length of the cell tokens.
    top_k (int): Top k value for selection masking.
    just_cell (bool): Whether to select only the cell index token.
    just_neighborhood (bool): Whether to select only the neighborhood index token.
    gene_id (int, optional): Gene ID or CLS ID to be used for specific selection.
    mask_large_than_k (bool): Whether to mask position larger than k in cell or neighborhood.

    Returns:
    torch.Tensor: The resulting selection mask tensor.

    Raises:
    AssertionError: If more than one of mask_large_than_k or specific_gene_mask is True.
    """

    # Ensure at most one of the conditions is true
    assert (int(mask_large_than_k) + int(get_specefic_gene)) <= 1, \
        "At most one of mask_large_than_k or specific_gene_mask must be True"

    # Initialize selection mask based on specific gene or non-zero tokens
    if get_specefic_gene:
        select = (cell_neighborhood_tokens == gene_id).int()
    else:
        select = (cell_neighborhood_tokens != 0).int()

    # Apply just_cell and just_neighborhood conditions
    if just_cell and just_neighborhood:
      if label_name == "niche_type":
          select[:, :seq_len_cell] = 0
          if mask_large_than_k:
            select[:, seq_len_cell + top_k:] = 0
            return select
      elif label_name == "cell_type":
           select[:, seq_len_cell:] = 0
    
    # Apply masking for elements larger than k
    if mask_large_than_k:
        select[:, top_k:] = 0

    return select
def create_anndata(features_list,
         dataset_type, label_name, label_value,
         just_pos, layer_index):
    """
    Create an AnnData object from the provided features and metadata.

    Parameters:
    features_list (list): The list of array of features to be stored in AnnData.
    dataset_type (str): The type of the dataset (e.g., train, test, validation).
    label_name (str): The name of the label associated with the data.
    label_value: The value of the label for each sample.
    just_pos (bool): A flag indicating whether only positional data is being processed.
    layer_index (int): The index of the layer of the transformer from which features are derived.

    Returns:
    obs: The obs data that should be stored in the obs of anndata
    features: the features that should store in obsm of the anndata
    """
    features = np.concatenate(features_list, axis=1)
    obs_data = {
        'split': dataset_type,
        'label_name': label_name,
        'just_pos': just_pos,
        'layer_index': layer_index,
        label_name: label_value
    }

    obs = pd.DataFrame(obs_data, index=range(len(features)))

    return features, obs

def calculate_sequence_length(just_cell, just_neighborhood, seq_len_cell, seq_len_neighborhood, has_cls):
    """
    Calculate the sequence length based on the provided flags and sequence lengths.

    Parameters:
        just_cell (bool): Flag indicating if only cell sequence length is used.
        just_neighborhood (bool): Flag indicating if only neighborhood sequence length is used.
        seq_len_cell (int): Sequence length for the cell.
        seq_len_neighborhood (int): Sequence length for the neighborhood.
        has_cls (bool): Flag indicating if the class token should be included.

    Returns:
        int: The calculated sequence length.

    Raises:
        ValueError: If both just_cell and just_neighborhood are False.
    """
    if just_cell and just_neighborhood:
        seq_len = seq_len_neighborhood + seq_len_cell
    elif just_cell:
        seq_len = seq_len_cell
    elif just_neighborhood:
        seq_len = seq_len_neighborhood
    else:
        raise ValueError("Both 'seq_len_neighborhood' and 'seq_len_cell' cannot be zero.")

    # Adjust sequence length if 'has_cls' is enabled.
    if has_cls:
        seq_len += 1 if just_cell or just_neighborhood else 2

    return seq_len

def merge_and_save_anndata(all_features, all_obs, output_file='final_result.h5ad'):
    """
    Merges features and observations into an AnnData object and saves it to a file.
    
    Parameters:
    - all_features (list of np.array): A list of arrays containing features to be merged.
    - all_obs (list of pd.DataFrame): A list of DataFrames containing observations to be concatenated.
    - output_file (str): The file name to save the resulting AnnData object.
    """
    # Merge all feature arrays vertically (stack them)
    merged_features = np.vstack(all_features)
    
    # Concatenate all observation DataFrames and reset the index
    final_obs = pd.concat(all_obs, axis=0).reset_index(drop=True)
    
    # Convert the index to string type
    final_obs.index = final_obs.index.astype(str)
    
    # Create an AnnData object with the merged observations
    final_adata = anndata.AnnData(obs=final_obs)
    
    # Add the merged features to the 'obsm' slot of the AnnData object
    final_adata.obsm['jepa_emb'] = merged_features
    
    # Write the AnnData object to a file
    final_adata.write(output_file)

