
import os
from typing import List, Literal, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from nichejepa.utils.evaluation import compute_scalar_mmd, compute_emd


def compute_sum_and_nonzero_count(
    mat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the sum of non-zero rows for each  2D tensor,
    and the total number of rows that have at least one non-zero entry.

    Parameters
    -----------
    mat:
        A 2D tensor of shape (num_rows, feature_dim).

    Returns
    -----------
    col_nonzero_sum:
        A 1D tensor of shape (feature_dim,) where each element is the sum
        of the non-zero entries in that feature dimension.
    nonzero_col_count:
        A 0-dim tensor (scalar) equal to the count of rows in which
        at least one element is non-zero.

    Raises
    -----------
    ValueError:
        If `mat` is not a 2D tensor.
    """
    # 1) Check that mat is 2D
    if mat.dim() != 2:
        raise ValueError(
            f"Expected a 2D tensor for mat, but got {mat.dim()} dimensions."
        )

    # 2) Build mask of nonzero entries
    nonzero_mask = mat != 0
    # 3) Sum nonzeros along the ROW dimension → gives one sum per COLUMN
    row_nonzero_sum = mat.masked_fill(~nonzero_mask, 0.0).sum(dim=0)

    # 4) Count how many columns have at least one non-zero
    nonzero_row_count = nonzero_mask.any(dim=1).sum()

    return row_nonzero_sum, nonzero_row_count

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


def create_binary_selection_mask(ns_tokens: torch.Tensor,
                                 seq_len_cell: int,
                                 selection_type: Literal['cls_cell',
                                                         'cls_neighborhood',
                                                         'agg_cell',
                                                         'agg_neighborhood',
                                                         'gene_cell',
                                                         'gene_neighborhood'],
                                 excluded_tokens: list[int] | None = None,
                                 top_k: int | None = None,
                                 n_segments: int | None = None,
                                 gene_id: int | None = None
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
    selection_type:
        Defines the type of embedding, which is relevant for the mask
        creation.
    excluded_tokens:
        List of tokens to be excluded from the selection.
    top_k:
        If specified, only 'top_k' of the selected tokens are retrieved.
    n_segments:
        Number of gene segments.
    gene_id:
        The ID of the gene for which the embedding is retrieved. Only
        relevant if `selection_type` is `gene_cell` or
        `gene_neighborhood`.

    Returns
    -----------
    selection_mask:
        The resulting 2D selection mask tensor.
    """
    selection_mask = torch.zeros_like(ns_tokens, dtype=torch.bool)

    if selection_type == 'agg_cell':
        # Select non-padding tokens in the cell segment
        selection_mask[:, :seq_len_cell] = True
        selection_mask[ns_tokens == 0] = False
        if excluded_tokens:
            # Exclude other excluded tokens
            selection_mask[torch.isin(
                ns_tokens,
                torch.tensor(excluded_tokens).to(ns_tokens.device))] = False
        if top_k:
            # Exclude tokens beyond the top_k positions in the cell
            # segment
            selection_mask[:, top_k:] = False
    elif selection_type == 'agg_neighborhood':
        # Select non-padding tokens in the neighborhood segments
        selection_mask[:, seq_len_cell:] = True
        selection_mask[ns_tokens == 0] = False
        if excluded_tokens:
            # Exclude other excluded tokens
            selection_mask[torch.isin(
                ns_tokens,
                torch.tensor(excluded_tokens).to(ns_tokens.device))] = False
        if top_k:
            # Exclude tokens beyond the top_k positions in the
            # neighborhood segments
            selection_mask[
                :, seq_len_cell + top_k:] = False
    elif selection_type == 'agg_graph':
        # Select non-padding tokens in all segments
        selection_mask[:, :] = True
        selection_mask[ns_tokens == 0] = False
        if excluded_tokens:
            # Exclude other excluded tokens
            selection_mask[torch.isin(
                ns_tokens,
                torch.tensor(excluded_tokens).to(ns_tokens.device))] = False
        if top_k:
            # Exclude tokens beyond the top_k positions in all segments
            for i in range(n_segments - 1):
                selection_mask[
                    :, 
                    (seq_len_cell * i + top_k):
                    (seq_len_cell * (i + 1))] = False
            selection_mask[
                :, seq_len_cell * (n_segments - 1) + top_k:] = False  
    elif selection_type == 'gene_cell':
        # Select only positions corresponding to the specified gene_id
        # in the cell segment
        selection_mask = ns_tokens == gene_id
        selection_mask[:, seq_len_cell:] = False
    elif selection_type == 'gene_neighborhood':
        # Select only positions corresponding to the specified gene_id
        # in the neighborhood segments
        selection_mask = ns_tokens == gene_id
        selection_mask[:, :seq_len_cell] = False
    elif selection_type == 'gene_graph':
        # Select only positions corresponding to the specified gene_id
        # in all segments
        selection_mask = ns_tokens == gene_id
    else:
        raise ValueError('The "selection_type" is not valid.')

    return selection_mask


def retrieve_gene_emb(
    ns_tokens: torch.Tensor,
    seq_len_cell: int,
    gene_type: Literal["cell", "neighborhood"],
    gene_id: int,
    emb: torch.Tensor = None,
    aggregate_multiple: bool = False,
    max_occ: int = 10
) -> Tuple:
    """
    Retrieve contextual gene embeddings for a specified gene based on its gene type and gene ID.

    For cell genes (aggregate_multiple is False), this function returns the presence flag and the index 
    of the gene occurrence in each sequence. For neighborhood genes (aggregate_multiple is True), it returns 
    the embeddings for all occurrences (padded or truncated to a fixed maximum number of occurrences), 
    a corresponding occurrence mask, and a presence flag.

    Parameters
    ----------
    ns_tokens : torch.Tensor
        A tensor of shape (N, total_seq_len) representing token IDs.
    seq_len_cell : int
        The length of the cell gene region in each sequence.
    gene_type : Literal["cell", "neighborhood"]
        Specifies whether to retrieve the embedding for a cell gene or a neighborhood gene.
    gene_id : int
        The identifier of the gene for which embeddings will be retrieved.
    emb : torch.Tensor, optional
        A tensor of shape (N, total_seq_len, D) containing token embeddings. Required for neighborhood genes.
    aggregate_multiple : bool, optional
        If True (and gene_type is "neighborhood"), retrieves embeddings for multiple occurrences.
        If False (typically for cell genes), retrieves the first occurrence.
    max_occ : int, optional
        The predefined maximum number of occurrences for neighborhood genes. Occurrence embeddings 
        will be padded or truncated to this length.

    Returns
    -------
    If gene_type is "cell" (and aggregate_multiple is False):
        Tuple[torch.Tensor, torch.Tensor]:
            - gene_presence: A binary tensor of shape (N,) indicating whether the gene is present in each sequence.
            - gene_indices: A tensor of shape (N,) containing the index of the gene occurrence in each sequence.
    If gene_type is "neighborhood" and aggregate_multiple is True:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - gene_occ: A tensor of shape (N, max_occ, D) containing the embeddings for each occurrence of the gene,
              padded or truncated to max_occ.
            - occ_mask: A tensor of shape (N, max_occ) with 1 for valid occurrences and 0 for padded positions.
            - gene_presence: A binary tensor of shape (N,) indicating whether the gene is present in each sequence.
    """
    gene_mask = create_binary_selection_mask(
        ns_tokens=ns_tokens,
        seq_len_cell=seq_len_cell,
        selection_type=f"gene_{gene_type}",
        gene_id=gene_id
    ).cpu()
    gene_presence = gene_mask.any(dim=1)  # (N,)
    
    if aggregate_multiple:
        N, L = gene_mask.shape
        occ_indices_list: List[torch.Tensor] = []
        for i in range(N):
            indices = (gene_mask[i]).nonzero(as_tuple=False).squeeze(-1)
            # Truncate or pad indices to have exactly max_occ elements.
            if indices.numel() >= max_occ:
                occ_indices = indices[:max_occ]
            else:
                pad = torch.zeros(max_occ - indices.numel(), dtype=indices.dtype, device=indices.device)
                occ_indices = torch.cat([indices, pad], dim=0)
            occ_indices_list.append(occ_indices.unsqueeze(0))  # shape (1, max_occ)
        occ_indices_tensor = torch.cat(occ_indices_list, dim=0)  # shape (N, max_occ)
        # Instead of checking for nonzero indices (which would mask a valid occurrence at index 0),
        # we compute the number of valid occurrences per sequence from gene_mask.
        occ_counts = gene_mask.sum(dim=1)  # (N,)
        # Create a range tensor and compare to counts for each sequence.
        range_tensor = torch.arange(max_occ, device=occ_indices_tensor.device).unsqueeze(0).expand(N, max_occ)
        occ_mask = (range_tensor < occ_counts.unsqueeze(1)).float()
        occ_mask[~gene_presence] = 0.0
        # Gather embeddings for each occurrence.
        gene_occ = torch.gather(emb, 1, occ_indices_tensor.unsqueeze(-1).expand(-1, -1, emb.shape[-1]))
        return gene_occ, occ_mask, gene_presence
    else:
        gene_mask_float = gene_mask.float()
        gene_indices = gene_mask_float.argmax(dim=1)  # (N,)
        return gene_presence, gene_indices

def compute_count_mean_cosine_sim(
    cell_embs: torch.Tensor,
    cell_presence: torch.Tensor,
    neb_occ: torch.Tensor,
    neb_mask: torch.Tensor,
    return_per_cell: bool=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the total (summed) cosine similarity and the total count of valid occurrence pairs
    between each cell gene (unique per sequence) and each neighborhood gene (across max_occ occurrences).
    No averaging per sequence is performed.

    Parameters
    ----------
    cell_embs : torch.Tensor
        Shape: (N, num_cell_genes, D) -- cell gene embeddings.
    cell_presence : torch.Tensor
        Shape: (N, num_cell_genes) -- binary presence indicator.
    neb_occ : torch.Tensor
        Shape: (N, num_neb_genes, max_occ, D) -- neighborhood gene occurrence embeddings.
    neb_mask : torch.Tensor
        Shape: (N, num_neb_genes, max_occ) -- binary mask indicating valid occurrences.
    return_per_cell: bool
        If set to True, it will return the cosine_sim(score) for each cell.

    Returns
    -------
    total_cs : torch.Tensor
        Shape: (num_cell_genes, num_neb_genes) -- total sum of cosine similarities.
    total_pair_count : torch.Tensor
        Shape: (num_cell_genes, num_neb_genes) -- total count of valid occurrence pairs.
    total_cell_count : torch.Tensor
        Shape: (num_cell_genes, num_neb_genes) -- total count of valid cell-neighborhood counts.
        The difference between total_pair_count and total_cell_count is that total_cell_count returns just number of
        occurrences in different cells, while total_pair_count returns the number of valid pairs.
    """
    # Normalize embeddings along the last dimension.
    cell_embs = F.normalize(cell_embs, p=2, dim=-1)
    neb_occ = F.normalize(neb_occ, p=2, dim=-1)
    
    # Compute cosine similarity between each cell gene and each neighborhood occurrence.
    # Resulting shape: (N, num_cell_genes, num_neb_genes, max_occ)
    cs = torch.einsum("ncd,njod->ncjo", cell_embs, neb_occ)
    
    # Expand neb_mask to match cs shape.
    neb_mask_exp = neb_mask.unsqueeze(1)  # (N, 1, num_neb_genes, max_occ)
    cs_masked = cs * neb_mask_exp  # zero out invalid occurrences
    
    # Sum cosine similarities over the occurrence dimension.
    occ_sum = cs_masked.sum(dim=-1)  # (N, num_cell_genes, num_neb_genes)
    occ_count = neb_mask_exp.sum(dim=-1)  # (N, num_cell_genes, num_neb_genes)
    
    # Only count sequences where the cell gene is present.
    cell_pres_exp = cell_presence.unsqueeze(-1)  # (N, num_cell_genes, 1)
    occ_sum_masked = occ_sum * cell_pres_exp
    occ_count_masked = occ_count * cell_pres_exp
    if return_per_cell:
        return occ_sum_masked, occ_count_masked, (occ_count_masked!= 0).float()
    total_cell_count = (occ_count_masked!= 0).float().sum(dim=0)       # (num_cell_genes, num_neb_genes)

    # Sum over all sequences.
    total_cs = occ_sum_masked.sum(dim=0)       # (num_cell_genes, num_neb_genes)
    total_pair_count = occ_count_masked.sum(dim=0)    # (num_cell_genes, num_neb_genes)
    
    return total_cs, total_pair_count, total_cell_count

def batch_rowwise_distances(
    A: torch.Tensor, 
    B: torch.Tensor
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    For each sample in batch:
      - For each row i:
         * Extract non-zero, non-NaN values from A[b, i, :] (excluding diag)
         * Extract non-zero, non-NaN values from B[b, i, :] (excluding diag)
         * Compute distances between these two independently filtered vectors
      - Average over all valid rows
    Parameters
    ----------
        A: First matrix
        B: Second matrix
    Returns:
        Tuple of 2 arrays (mmd_distances, emd_distances) each of shape (B,)
    """
    A = A.numpy()
    B = B.numpy()
    assert A.shape == B.shape
    B_sz, G, G2 = A.shape
    assert G == G2

    mmd_out = np.zeros(B_sz, dtype=float)
    emd_out = np.zeros(B_sz, dtype=float)
    emd_matrix = np.zeros((B_sz, G), dtype=float)

    for b in range(B_sz):
        m_list, w_list = [], []

        for i in range(G):
            # Indices excluding the diagonal
            idx = np.arange(G) != i

            ai = A[b, i, idx]
            bi = B[b, i, idx]

            # Filter non-zero and non-NaN separately for A and B
            ai_valid = ai[~np.isnan(ai) & (ai != 0)][:, None]
            bi_valid = bi[~np.isnan(bi) & (bi != 0)][:, None]

            if ai_valid.shape[0] < 1 or bi_valid.shape[0] < 1:
                emd_matrix[b, i] = 0.0
                continue  # Skip if either is empty
            #m_list.append(compute_scalar_mmd(ai_valid, bi_valid))
            emd_val = compute_emd(ai_valid, bi_valid)
            emd_matrix[b, i] = emd_val
            w_list.append(emd_val)

        if w_list:
            #mmd_out[b] = float(np.mean(m_list))
            emd_out[b] = float(np.mean(w_list))
        else:
            mmd_out[b] = emd_out[b] = 0.0
    return mmd_out, emd_out, emd_matrix


def collect_adata_from_folder(load_folder_path: str,
                              cell_ids: list[str] | None = None,
                              dataset_ids: list[str] | None = None,
                              obs_cols: list[str] | None = None,
                              uns_cols: list[str] | None = None,
                              include_gene_panel_size: bool = True,
                              ) -> ad.AnnData:
    """
    Loop through folder, read all `.h5ad` files and concatenate them
    into one AnnData object.

    Parameters
    --------
    load_folder_path:
        Directory which is searched for AnnData objects.
    cell_ids:
        IDs of cells which are included in the final object. If `None`,
        all cells are included.
    dataset_ids:
        IDs of datasets which are included in the final object. If
        `None`, all subdirectories are included.
    obs_cols:
        Columns in `adata.obs` that should be retained in the final
        object.
    uns_cols:
        Keys in `adata.uns` that should be retained in the final
        object.

    Returns
    --------
    adata:
        The concatenated AnnData object.
    """
    adata_list = []

    # Walk through the load folder path and read files
    for subdir, _, files in os.walk(load_folder_path):
        if dataset_ids:
            if not any(dataset_id in subdir.split('/')[-1].split('-')[0] for dataset_id in dataset_ids):
                continue
        print(f'Loading AnnData objects from {subdir}.')
        for file in files:
            if file.endswith('.h5ad'):
                file_path = os.path.join(subdir, file)
                adata = sc.read_h5ad(file_path)
                if cell_ids:
                    adata = adata[adata.obs['cell_id'].isin(cell_ids)]
                if len(adata) == 0:
                    del adata
                    continue
                if obs_cols is None:
                    adata.obs = adata.obs[['cell_id']]
                else:
                    adata.obs = adata.obs[['cell_id'] + [
                        col for col in obs_cols if col in adata.obs.columns]]
                if uns_cols:
                    for col in uns_cols:
                        if col in adata.uns:
                            adata.obs[col] = adata.uns[col]
                if include_gene_panel_size:
                    adata.obs['gene_panel_size'] = len(adata.var_names)
                adata_list.append(adata)        

    concatenated_adata = ad.concat(adata_list, join='outer', index_unique=None)
    
    return concatenated_adata