from typing import List, Literal, Optional

import anndata
import numpy as np
import pandas as pd
import torch


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
                                 has_cls: bool,
                                 has_gene_panel: bool,
                                 selection_type: Literal['cls',
                                                         'agg_cell',
                                                         'agg_neighborhood',
                                                         'gene_cell',
                                                         'gene_neighborhood'],
                                 excluded_tokens: Optional[List]=None,
                                 top_k: Optional[int]=None,
                                 gene_id: Optional[int]=None
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
    has_cls:
        If 'True', sequence contains a <cls> token at position 0.
    has_gene_panel:
    selection_type:
        Defines the type of embedding, which is relevant for the mask creation.
    excluded_tokens:
        List of tokens to be excluded from the selection.
    top_k:
        If specified, only 'top_k' of the selected tokens are retrieved.
    gene_id:
        The ID of the gene for which the embedding is retrieved. Only relevant
        if 'selection_type' is 'gene_cell' or 'gene_neighborhood'.

    Returns
    -----------
    selection_mask:
        The resulting 2D selection mask tensor.
    """
    selection_start_idx = (1 if has_cls else 0)
    if has_gene_panel:
        selection_start_idx += 1

    if selection_type == 'cls':
        # Select only the first token in each sequence
        selection_mask = torch.zeros_like(tokens, dtype=torch.bool)
        selection_mask[:, 0] = True
        return selection_mask
    elif selection_type == 'agg_cell':
        selection_mask = torch.zeros_like(tokens, dtype=torch.bool)
        # Select non-padding tokens in the cell segment
        selection_mask[:, selection_start_idx:
                          selection_start_idx + seq_len_cell] = True
        selection_mask[tokens == 0] = False # exclude padding tokens
        if excluded_tokens: # exclude other excluded tokens
            selection_mask[
                torch.isin(
                    tokens,
                    torch.tensor(excluded_tokens).to(tokens.device))] = False
        if top_k:
            # Exclude tokens beyond the top_k positions in the cell segment
            selection_mask[:, selection_start_idx + top_k:] = False
    elif selection_type == 'agg_neighborhood':
        # Select non-padding tokens in the neighborhood segment
        selection_mask = torch.zeros_like(tokens, dtype=torch.bool)
        selection_mask[:, selection_start_idx + seq_len_cell:] = True
        selection_mask[tokens == 0] = False # exclude padding tokens
        if excluded_tokens: # exclude other excluded tokens
            selection_mask[
                torch.isin(
                    tokens,
                    torch.tensor(excluded_tokens).to(tokens.device))] = False
        if top_k:
            # Exclude tokens beyond the top_k positions in the neighborhood
            # segment
            selection_mask[
                :, selection_start_idx + seq_len_cell + top_k:] = False    
    elif selection_type == 'gene_cell':
        # Select only positions corresponding to the specified gene_id in the
        # cell segment
        selection_mask = tokens == gene_id
        selection_mask[:, selection_start_idx + seq_len_cell:] = False
    elif selection_type == 'gene_neighborhood':
        # Select only positions corresponding to the specified gene_id in the
        # neighborhood segment
        selection_mask = tokens == gene_id
        selection_mask[:, selection_start_idx:
                          selection_start_idx + seq_len_cell] = False
    else:
        raise ValueError('The "selection_type" is not valid.')

    return selection_mask


def retrieve_gene_emb(tokens: torch.Tensor,
                      seq_len_cell: int,
                      has_cls: bool,
                      emb: torch.Tensor,
                      gene_type: Literal["cell", "neighborhood"],
                      gene_id: int,
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
    has_cls:
        If 'True', sequence contains a <cls> token at position 0.
    emb:
        A 3D tensor containg the embeddings of all genes.
    gene_type:
        Defines whether to retrieve the cell or neighborhood gene embedding for
        the given gene ID.
    gene_id:
        Gene ID of the gene for which the embedding will be retrieved.

    Returns
    --------
    gene_emb:
        The cell or neighborhood embedding of the gene with the given gene ID.
    """
    gene_mask = create_binary_selection_mask(
        tokens=tokens,
        seq_len_cell=seq_len_cell,
        has_cls=has_cls,
        selection_type=f"gene_{gene_type}",
        gene_id=gene_id)

    gene_indices = torch.argmax(gene_mask.to(torch.int),
                                dim=1,
                                keepdim=True) # shape: (3, 1)

    # Use gather to get the correct embeddings for each cell based on the
    # indices (if gene is not present, the index will be wrong but is
    # overwritten below)
    gene_emb = torch.gather(
        emb,
        1,
        gene_indices.unsqueeze(-1).expand(-1, -1, emb.size(2))).squeeze(1)

    # For rows with no True values, set them to zero embeddings
    gene_emb[gene_mask.sum(dim=1) == 0] = torch.zeros(emb.size(2)).to(
        gene_emb.device)

    return gene_emb
