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
    # Create a mask where each element is True (1) if the corresponding token is non-zero, and False (0) if it is zero (padding token)
    mask = tokens != 0

    # Compute cumulative sum along the sequence dimension (dim=1), which gives ranks for non-zero tokens
    # Each token's rank is incremented based on its position in the sequence, with padding tokens maintaining a rank of 0
    ranks = mask.cumsum(dim=1).float() * mask.float()

    # Find the maximum rank in each sequence, keeping the dimension for broadcasting
    rank_max = ranks.max(dim=1, keepdim=True)[0]

    # Compute the sum of ranks for each sequence, keeping the dimension for broadcasting
    rank_sum = ranks.sum(dim=1, keepdim=True)

    # Calculate the weights for each token: the weight is inversely proportional to the rank within the sequence
    # (higher ranks have lower weights). The 1e-9 is added to avoid division by zero.
    weights = (rank_max - ranks + 1) / (rank_sum + 1e-9)

    # Apply the mask to ensure that padding tokens receive a weight of 0
    weights = weights * mask.float()

    # Return the computed weights
    return weights
def weighted_mean(embs, weights, dim=1):
    """
    Compute the weighted mean of embs.

    Parameters:
    embs (torch.Tensor): The input embs tensor (3D).
    weights (torch.Tensor): A tensor of weights (same size as the relevant dimension of embs).
    dim (int): The dimension along which to compute the weighted mean.

    Returns:
    torch.Tensor: The weighted mean tensor.
    
    Raises:
    ValueError: If the items tensor is not 3D.

    """
    # Use broadcasting to multiply embeddings by their corresponding weights
    if embs.dim() == 3:
        # If the embeddings tensor has 3 dimensions (batch_size, sequence_length, embedding_dim),
        # broadcast the weights tensor to match the dimensions of embs.
        # The weights tensor is initially (batch_size, sequence_length), so we unsqueeze to (batch_size, sequence_length, 1).
        weighted_embs = embs * weights.unsqueeze(2)  # Broadcasting weights along the embedding dimension

        # Sum the weighted embeddings along the specified dimension (likely sequence_length)
        weighted_sum = weighted_embs.sum(dim)

        # Sum the weights along the same dimension and unsqueeze to maintain dimensionality
        weights_sum = weights.sum(dim).unsqueeze(1)  # The result is (batch_size, 1) after summing

        # Calculate the weighted mean by dividing the weighted sum of embeddings by the sum of the weights
        weighted_mean = weighted_sum / weights_sum

    else:
        # Raise an error if the input embeddings tensor is not 3D, as this function expects a 3D tensor
        raise ValueError('Expected a 3D tensor for embs, but got a tensor with {} dimensions.'.format(embs.dim()))

    # Return the calculated weighted mean
    return weighted_mean
def mean_nonpadding_embs(embs, mask, dim=1):
    """
    Compute the mean of non-padding embeddings.
    
    Parameters:
    embs (torch.Tensor): The input embeddings tensor (3D).
    mask (torch.Tensor): A boolean mask tensor indicating the positions that mean should computed (same size as the relevant dimension of embs).
    dim (int): The dimension along which to compute the mean.
    
    Returns:
    torch.Tensor: The mean embeddings tensor.
    torch.Tensor: Number of genes that used in computing average

    Raises:
    ValueError: If the items tensor is not 3D.
    """
    # Use broadcasting to sum embeddings across non-padding positions
    if embs.dim() == 3:
        # If the embeddings tensor has 3 dimensions (batch_size, sequence_length, embedding_dim),
        # broadcast the mask to match the dimensions of embs.
        # The mask tensor is initially (batch_size, sequence_length), so we unsqueeze to (batch_size, sequence_length, 1).
        masked_embs = embs * mask.unsqueeze(2)  # Broadcasting the mask along the embedding dimension

        # Sum the masked embeddings along the specified dimension
        sum_embs = masked_embs.sum(dim)

        # Calculate the mean by dividing the summed embeddings by the number of non-padding positions
        # The mask is summed along the same dimension to count non-padding tokens, and view(-1, 1) ensures
        # The resulting tensor has the correct dimensions for broadcasting during division.
        # The + 1e-9 will handle the case that we are reterriving gene that may mask.sum(dim).view(-1, 1).float() be zero
        # then we won't have INF value
        mean_embs = sum_embs / (mask.sum(dim).view(-1, 1).float() + 1e-9)

    else:
        # Raise an error if the input embeddings tensor is not 3D, as this function expects a 3D tensor
        raise ValueError('Expected a 3D tensor for embs, but got a tensor with {} dimensions.'.format(embs.dim()))

    # Return the calculated mean embeddings and the number of unmasked values for each row when the mask is computed for retrieve_gene.
    # This value serves as a gene_id identifier, indicating the cells where the gene is present.
    # A value of zero means that the gene is not present, while a value of one means that the sample contains the gene.
    # This value in upper function could be used as gene_count

    return mean_embs, mask.sum(dim)

def create_selection(cell_neighborhood_tokens, label_name, seq_len_cell, args, top_k=None,
                          just_cell=False, just_neighborhood=False,
                          mask_large_than_k=False, retrieve_label=None):
    """
    Create a selection mask for cell index tokens or neighborhood tokens based on various conditions.

    Parameters:
    cell_neighborhood_tokens (torch.Tensor): Tensor containing cell or neighborhood tokens.
    label_name (str): Label name to determine selection rules.
    seq_len_cell (int): Sequence length of the cell tokens.
    args (dict): Dictionary of configs.
    top_k (int): Top k value for selection masking.
    just_cell (bool): Whether to select only the cell index token.
    just_neighborhood (bool): Whether to select only the neighborhood index token.
    mask_large_than_k (bool): Whether to mask position larger than k in cell or neighborhood.
    retrieve_label (str): Name of the label of retrieve portion that could be retrieve_niche, retrieve_cell or retrieve_gene

    Returns:
    torch.Tensor: The resulting selection mask tensor.

    Raises:
    AssertionError: If just_neighborhood is False and retrieve_label == 'retrieve_niche'
    AssertionError: If just_cell is False and retrieve_label == 'retrieve_cell'
    AssertionError: If args['data']['cls'] is True and args['emb']['retrieve_gene'] is True
    AssertionError: If args['data']['has_cls'] is False and args['emb']['cls'] True
    
    """
    # Ensure that if either just_neighborhood or just_cell is False, 
    # retrieve_label cannot be 'retrieve_niche' because 'retrieve_niche' is only meaningful 
    # when the data contains both cell and neighborhood sequences.
    assert not ((not just_neighborhood or not just_cell) and retrieve_label == 'retrieve_niche'), \
        "retrieve_label cannot be 'retrieve_niche' when just_neighborhood or just_cell is False as it has meaning when data has trained on sequence contain both of them"

    # Ensure that if just_cell is False, retrieve_label cannot be 'retrieve_cell'
    # This condition enforces that 'retrieve_cell' is only valid when 'just_cell' is True.
    assert not (not just_cell and retrieve_label == 'retrieve_cell'), \
        "retrieve_label cannot be 'retrieve_cell' when just_cell is False."

    # Assert that if the model has not been trained on CLS tokens (`has_cls` is False),
    # then the `cls` argument cannot be True. This prevents the use of CLS tokens when
    # the model wasn't designed to work with them.
    assert not (not args['data']['has_cls'] and args['emb']['cls']), \
        "The data has not been trained on CLS token, so 'cls' cannot be True when 'has_cls' is False."

    # Assert that the `cls` argument and the `retrieve_gene` argument cannot both be True simultaneously.
    # This is because the CLS token does not represent individual genes, making it meaningless to
    # retrieve gene-specific embeddings when `cls` is being used.
    assert not (args['emb']['cls'] and args['emb']['retrieve_gene']), \
        "'cls' and 'retrieve_gene' cannot both be True, as CLS token does not have a meaningful representation for each gene."
    
    # If `cls` is enabled in `args['emb']`, create a selection mask focused on the CLS token.
    if args['emb']['cls']:
        # Create a zero tensor with the same shape as `cell_neighborhood_tokens`
        selection = torch.zeros_like(cell_neighborhood_tokens)

        # Select only the first token in each sequence by setting the first column to 1.
        # This will be used to focus on the CLS token (or the first token) during embedding processing.
        selection[:, 0] = 1
        return selection
    # Initialize the selection mask based on the retrieve_label value
    # If retrieve_label is 'retrieve_gene', the mask will select only positions corresponding to the specified gene_id,
    # inside cell itself
    # Otherwise, it will select non-zero tokens.
    if retrieve_label == 'retrieve_gene':
        select = (cell_neighborhood_tokens == args['emb']['gene_id']).int()
        select[:, seq_len_cell:] = 0
        return select
    else:
        select = (cell_neighborhood_tokens != 0).int()
        # If the sequence contains a CLS token, exclude it from the selection. 
        # Reaching this point implies that args['emb']['cls'] is False, 
        # indicating that the selection is being created for either average or weighted_average computations, 
        # and the CLS token should not be included in these calculations.
        if args['data']['has_cls'] and not args['emb']['cls']:
           select[:,0] = 0
    # Apply conditions based on retrieve_label
    if retrieve_label == "retrieve_niche":
        # If retrieve_label is 'retrieve_niche', mask the positions corresponding to the cell sequence.
        select[:, :seq_len_cell] = 0
        # If mask_large_than_k is True, further mask elements beyond the top_k positions in the neighborhood sequence.
        if mask_large_than_k:
            select[:, seq_len_cell + top_k:] = 0
        # Return the selection mask early since no further processing is needed.
        return select
    elif retrieve_label == "retrieve_cell":
        # If retrieve_label is 'retrieve_cell', mask the positions corresponding to the neighborhood sequence.
        select[:, seq_len_cell:] = 0

    # Apply masking for elements beyond the top_k positions
    # This is applicable only when mask_large_than_k is True.
    if mask_large_than_k:
        select[:, top_k:] = 0

    # Return the final selection mask
    return select


def process_features(features_list,
         split, label_name, label_value, retrieve_label,
         retrieve_position_emb, retrieve_emb_from_layer, gene_count=None):
    """
    Process features from the provided features and metadata to make it ready for anndata

    Parameters:
    features_list (list): The list of array of features to be stored in AnnData.
    split (str): The split of the dataset (e.g., train, test, validation).
    label_name (str): The name of the label associated with the data.
    label_value: The value of the label for each sample.
    retrieve_label (str): Name of the label of retrieve portion that could be retrieve_niche, retrieve_cell or retrieve_gene 
    gene_count (Tensor): A tensor of shape Batch that indicates, for each sample, how many genes were used to compute the mean.
    retrieve_position_emb (bool): A flag indicating whether the position embedding is retrieved or not.
    retrieve_emb_from_layer (int): The start index of the layer of the transformer from which features are derived.
    Zero means from the first layer to the last layer of the transformer, indicating the depth of the transformer
    in case retrieve_position_emb is True it doesn't have any meaning.

    Returns:
    obs: The obs data that should be stored in the obs of anndata
    features: the features that should store in obsm of the anndata
    """
    features = np.concatenate(features_list, axis=1)
    obs_data = {
        'split': split,
        'label_name': label_name,
        'retrieve_label': retrieve_label,
        'retrieve_position_emb': retrieve_position_emb,
        'retrieve_emb_from_layer': retrieve_emb_from_layer,
        label_name: label_value
    }
    # Handel if the emb is coming from cls
    if gene_count==None:
        obs_data['gene_count']=-1
    else:
        obs_data['gene_count']=gene_count.detach().cpu().numpy()
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
        seq_len += 1
    return seq_len
def create_and_save_anndata(all_features, all_obs, output_file='final_result.h5ad'):
    """
    Merges features and observations into an AnnData object and saves it to a file.
    
    Parameters:
    - all_features (list of np.array): A list of arrays containing features to be merged.
    - all_obs (list of pd.DataFrame): A list of DataFrames containing observations to be concatenated.
    - output_file (str): The file name to save the resulting AnnData object.
    Returns
        anndata: final_adata
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

    print(f"AnnData has been successfully saved at: {output_file}")

    return final_adata

