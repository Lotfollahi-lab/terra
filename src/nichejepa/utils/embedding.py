
import os
from typing import List, Literal, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F

def compute_unmasked_rank_based_weights(tokens: torch.Tensor,
                                        mask: torch.Tensor
                                        ) -> torch.Tensor:
    """
    Compute unmasked rank-based weights for a 2D tensor of tokens.

    Parameters
    -----------
    tokens:
        A 2D tensor where each row represents a sequence of tokens. 
    mask:
        A 2D boolean mask tensor, indicating elements to be included in the
        rank-based weighting.

    Returns
    -----------
    weights:
        A 2D tensor of the same shape as `tokens`, containing the computed
        weights for unmasked positions (weights sum up to 1).
    """
    # Compute cumulative sum along the sequence dimension (dim=1), which gives
    # ranks for selected tokens. Each token's rank is incremented based on its
    # position in the sequence
    ranks = mask.cumsum(dim=1).float() * mask.float()

    # Find the maximum rank in each sequence, keeping the dimension for
    # broadcasting
    rank_max = ranks.max(dim=1, keepdim=True)[0]

    # Compute the sum of ranks for each sequence, keeping the dimension for
    # broadcasting
    rank_sum = ranks.sum(dim=1, keepdim=True)

    # Calculate the weights for each token: the weight is inversely proportional
    # to the rank within the sequence (higher ranks have lower weights). The
    # 1e-9 is added to avoid division by zero.
    weights = (rank_max - ranks + 1) / (rank_sum + 1e-9)

    # Apply the mask to ensure that unselected tokens receive a weight of 0
    weights = weights * mask.float()

    return weights


def compute_mean_unmasked_emb(emb: torch.Tensor,
                              mask: torch.Tensor,
                              ) -> torch.Tensor:
    """
    Compute the mean of unmasked embedding positions.
    
    Parameters
    -----------
    emb:
        The input embeddings tensor (3D).
    mask:
        A 2D boolean mask tensor indicating the sequence positions that mean
        should be computed over.
    
    Returns
    -----------
    mean_emb:
        The mean embedding tensor.

    Raises
    -----------
    ValueError: If the emb tensor is not 3D.
    """
    # Use broadcasting to sum embeddings across unmasked positions
    if emb.dim() == 3:
        # If the embeddings tensor has 3 dimensions (batch_size,
        # sequence_length, embedding_dim), broadcast the mask to match the
        # dimensions of emb. The mask tensor is initially (batch_size,
        # sequence_length), so we unsqueeze to (batch_size, sequence_length, 1)
        masked_emb = emb * mask.unsqueeze(2) # broadcast the mask along the
                                             # embedding dimension

        # Sum the masked embeddings along the sequence dimension
        sum_emb = masked_emb.sum(1)

        # Calculate the mean by dividing the summed embeddings by the number of
        # unmasked positions. The mask is summed to count unmasked tokens, and
        # view(-1, 1) ensures the resulting tensor has the correct dimensions
        # for broadcasting during division. The + 1e-9 will handle the case
        # where we are retrieving a gene that may have
        # mask.sum(dim).view(-1, 1).float() = 0
        mean_emb = sum_emb / (mask.sum(1).view(-1, 1).float() + 1e-9)

    else:
        raise ValueError('Expected a 3D tensor for emb, but got a tensor with'
                         f'{emb.dim()} dimensions.')

    return mean_emb


def create_binary_selection_mask(tokens: torch.Tensor,
                                 seq_len_cell: int,
                                 n_special_tokens: int,
                                 selection_type: Literal['cls_cell',
                                                         'cls_neighborhood',
                                                         'agg_cell',
                                                         'agg_neighborhood',
                                                         'gene_cell',
                                                         'gene_neighborhood'],
                                 max_cls_tokens: Optional[int]=None,
                                 excluded_tokens: Optional[List]=None,
                                 top_k: Optional[int]=None,
                                 gene_id: Optional[int]=None,
                                 n_segments: Optional[int]=None,
                                 ) -> torch.Tensor:
    """
    Create a selection mask for cell and neighborhood tokens based on
    specificiations.

    Parameters
    -----------
    tokens:
        A 2D tensor where each row represents a sequence of tokens.
    seq_len_cell:
        The length of cell tokens in the sequence.
    n_special_tokens:
    max_cls_tokens:
        Number of <cls> tokens.
    selection_type:
        Defines the type of embedding, which is relevant for the mask creation.
    excluded_tokens:
        List of tokens to be excluded from the selection.
    top_k:
        If specified, only 'top_k' of the selected tokens are retrieved.
    gene_id:
        The ID of the gene for which the embedding is retrieved. Only relevant
        if 'selection_type' is 'gene_cell' or 'gene_neighborhood'.
    n_segments:
        The n_segments in cell_graph tokenizer.
    Returns
    -----------
    selection_mask:
        The resulting 2D selection mask tensor.
    """
    selection_mask = torch.zeros_like(tokens, dtype=torch.bool)

    if selection_type == 'cls_0':
        # Select only the first <cls> token
        selection_mask[:, 0] = True
        return selection_mask
    elif selection_type == 'cls_1':
        # Select only the second <cls> token
        selection_mask[:, 1] = True
        return selection_mask
    elif selection_type == 'cls_all':
        # Select non-padding <cls> tokens
        selection_mask[:, :max_cls_tokens] = True
        selection_mask[tokens == 0] = False
        return selection_mask       
    elif selection_type == 'agg_cell':
        # Select non-padding tokens in the cell segment
        selection_mask[:, n_special_tokens:
                          n_special_tokens + seq_len_cell] = True
        selection_mask[tokens == 0] = False
        if excluded_tokens:
            # Exclude other excluded tokens
            selection_mask[torch.isin(
                tokens,
                torch.tensor(excluded_tokens).to(tokens.device))] = False
        if top_k:
            # Exclude tokens beyond the top_k positions in the cell segment
            selection_mask[:, n_special_tokens + top_k:] = False
    elif selection_type == 'agg_neighborhood':
        # Select non-padding tokens in the neighborhood segments
        selection_mask[:, n_special_tokens + seq_len_cell:] = True
        selection_mask[tokens == 0] = False
        if excluded_tokens:
            # Exclude other excluded tokens
            selection_mask[torch.isin(
                tokens,
                torch.tensor(excluded_tokens).to(tokens.device))] = False
        if top_k:
            # Exclude tokens beyond the top_k positions in the neighborhood
            # segments
            selection_mask[
                :, n_special_tokens + seq_len_cell + top_k:] = False
    elif selection_type == 'agg_graph':
        # Select non-padding tokens in all segments
        selection_mask[:, n_special_tokens:] = True
        selection_mask[tokens == 0] = False
        if excluded_tokens:
            # Exclude other excluded tokens
            selection_mask[torch.isin(
                tokens,
                torch.tensor(excluded_tokens).to(tokens.device))] = False
        if top_k:
            # Exclude tokens beyond the top_k positions in all segments
            for i in range(n_segments - 1):
                selection_mask[
                        :, 
                        n_special_tokens + seq_len_cell * i + top_k : n_special_tokens + seq_len_cell * (i + 1)
                ] = False
            selection_mask[:, n_special_tokens + seq_len_cell * (n_segments-1) + top_k :] = False
        
    elif selection_type == 'gene_cell':
        # Select only positions corresponding to the specified gene_id in the
        # cell segment
        selection_mask = tokens == gene_id
        selection_mask[:, n_special_tokens + seq_len_cell:] = False
    elif selection_type == 'gene_neighborhood':
        # Select only positions corresponding to the specified gene_id in the
        # neighborhood segments
        selection_mask = tokens == gene_id
        selection_mask[:, n_special_tokens:
                          n_special_tokens + seq_len_cell] = False
    elif selection_type == 'gene_graph':
        # Select only positions corresponding to the specified gene_id in all
        # segments
        selection_mask = tokens == gene_id
    else:
        raise ValueError('The "selection_type" is not valid.')

    return selection_mask


def retrieve_gene_emb(tokens: torch.Tensor,
                      seq_len_cell: int,
                      n_special_tokens: int,
                      gene_type: Literal["cell", "neighborhood"],
                      gene_id: int,
                      emb: torch.Tensor=None,
                      aggregate_multiple: bool=False,
                      ) -> torch.Tensor:
    """
    Retrieve contextual gene embeddings for a given gene based on a specified
    gene ID and gene type.

    Parameters
    -----------
    tokens:
        A 2D tensor where each row represents a sequence of tokens.
    seq_len_cell:
        The length of cell tokens in the sequence.
    n_special_tokens:
        Number of special tokens.
    gene_type:
        Defines whether to retrieve the cell or neighborhood gene embedding for
        the given gene ID.
    gene_id:
        Gene ID of the gene for which the embedding will be retrieved.
    emb:
        A 3D tensor containg the embeddings of all genes.
    aggregate_multiple: bool. 
        If True, averages multiple occurrences; else assumes at most one.

    Returns
    --------
    gene_emb:
        The cell or neighborhood embedding of the gene with the given gene ID.
    """
    gene_mask = create_binary_selection_mask(
        tokens=tokens,
        seq_len_cell=seq_len_cell,
        n_special_tokens=n_special_tokens,
        selection_type=f"gene_{gene_type}",
        gene_id=gene_id)
    
    gene_presence = gene_mask.any(dim=1)           # (N,) True if gene is present in a sequence

    if aggregate_multiple:
        # Average over all occurrences.
        gene_mask_float = gene_mask.float()  # Convert to float: shape (N, L)
        counts = gene_mask_float.sum(dim=1, keepdim=True)  # (N, 1)
        summed = (emb * gene_mask_float.unsqueeze(-1)).sum(dim=1)  # (N, D)
        avg_emb = summed / (counts + 1e-6)  # Avoid division by zero.
        return avg_emb, gene_presence
    else:
        # Assume at most one occurrence per sequence. For sequences with no occurrence, embedding remains 0.
        # We use argmax on the float mask (note: if no occurrence exists, argmax returns 0, but presence will be 0).
        gene_mask_float = gene_mask.float()
        gene_indices = gene_mask_float.argmax(dim=1)  # (N,) index of the (first) occurrence

    return gene_presence, gene_indices


def collect_adata_from_folder(load_folder_path: str) -> ad.AnnData:
    """
    Loop through folder, read all '.h5ad' files and concatenate them as adata
    objects.

    Parameters
    --------

    Returns
    --------
    """
    adata_list = []

    # Walk through the load folder path and read files
    for subdir, _, files in os.walk(load_folder_path):
        for file_idx, file in enumerate(files):
            if file.endswith('.h5ad'):
                file_path = os.path.join(subdir, file)
                adata = sc.read_h5ad(file_path)
                adata_list.append(adata)

    concatenated_adata = ad.concat(adata_list, join='outer', index_unique=None)
    
    return concatenated_adata


    import torch
import torch.nn.functional as F

def compute_cosine_similarity_components(
    cell_embs: torch.Tensor,
    neb_embs: torch.Tensor,
    cell_presence: torch.Tensor,
    neb_presence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the sum of cosine similarities and the count of valid sequences 
    for each (cell gene, neighborhood gene) pair.

    Parameters
    ----------
    cell_embs : torch.Tensor
        A tensor of shape (N, num_cell_genes, D) containing embeddings for cell genes.
    neb_embs : torch.Tensor
        A tensor of shape (N, num_neb_genes, D) containing embeddings for neighborhood genes.
    cell_presence : torch.Tensor
        A binary tensor of shape (N, num_cell_genes) indicating presence (1) or absence (0) of each cell gene per sequence.
    neb_presence : torch.Tensor
        A binary tensor of shape (N, num_neb_genes) indicating presence (1) or absence (0) of each neighborhood gene per sequence.

    Returns
    -------
    sum_cos_sim : torch.Tensor
        A tensor of shape (num_cell_genes, num_neb_genes) containing the sum 
        of cosine similarities for each gene pair across sequences.
    count : torch.Tensor
        A tensor of shape (num_cell_genes, num_neb_genes) containing the count 
        of valid sequences where both genes are present.
    """

    # Normalize embeddings to ensure cosine similarity reduces to dot product
    cell_embs = F.normalize(cell_embs, p=2, dim=-1)
    neb_embs = F.normalize(neb_embs, p=2, dim=-1)

    # Compute cosine similarity per sequence
    cos_sim = torch.bmm(cell_embs, neb_embs.transpose(1, 2))  # Shape: (N, num_cell_genes, num_neb_genes)

    # Compute presence mask: consider only sequences where both genes are present
    combined_presence = cell_presence.unsqueeze(2) * neb_presence.unsqueeze(1)  # Shape: (N, num_cell_genes, num_neb_genes)
    
    # Apply mask to cosine similarity values
    masked_cos_sim = cos_sim * combined_presence  # Zero out cosine similarities where either gene is absent

    # Sum valid cosine similarities and count valid occurrences per gene pair
    sum_cos_sim = masked_cos_sim.sum(dim=0)  # Shape: (num_cell_genes, num_neb_genes)
    count = combined_presence.sum(dim=0)  # Shape: (num_cell_genes, num_neb_genes)

    return sum_cos_sim, count