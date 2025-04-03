import torch
import torch.nn.functional as F


def apply_masks(x, masks, concat=True):
    """
    Apply masks to an input tensor.

    Parameters
    ----------
    x:
        Tensor of shape (B, N, D); B: batch size, N: number of tokens,
        D: feature dimensions.
    masks:
        List of tensors containing indices of tokens in [N] to keep.

    Returns
    ----------
        Masked input tensor.
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)


def apply_attention_mask(attention_matrix, indices):

    """
    Apply given indices to the attention matrix and return the selected submatrix.

    Parameters
    ----------
    attention_matrix : Tensor of shape (B, N, N)
        The input attention matrix.
    indices:
        Tensor of shape (B, M). Indices of tokens for which the attention should
        happen.

    Returns
    -------
    Tensor
        The submatrix from the attention matrix after applying the indices.
    """
    # Expand combined indices for row selection
    row_indices_expanded = indices.unsqueeze(-1).expand(-1, -1, attention_matrix.size(-1))

    # Gather the rows from the attention matrix based on the expanded indices
    selected_rows = torch.gather(attention_matrix, 1, row_indices_expanded)

    # Expand combined indices for column selection
    column_indices_expanded = indices.unsqueeze(1).expand(-1, selected_rows.size(1), -1)

    # Gather columns from the selected rows based on the expanded indices
    selected_submatrix = torch.gather(selected_rows, 2, column_indices_expanded)

    return selected_submatrix


def create_enc_attention_mask(
    attention_matrix: torch.Tensor,
    context_masks: torch.Tensor | None) -> torch.Tensor:
    """
    Apply context_masks and/or target_masks to the input attention matrix by
    gathering rows and columns based on the given indices.

    Parameters
    ----------
    attention_matrix: Tensor of shape (B, 1, N, N)
        The input attention matrix where B is the batch size and N is the sequence length.
    context_masks:
        A list of tensors containing indices of context tokens.

    Returns
    -------
    Tensor
        A concatenated tensor of the attention matrices after applying all the rules.
        for context and/or target.
    """
    # List to store the attention matrices with applied masks
    masked_attention_matrices = []

    # Remove the singleton dimension if it exists in attention_matrix
    attention_matrix = attention_matrix.squeeze(1)

    # Iterate through the context  masks
    for context_indices in context_masks:
        selected_submatrix = apply_attention_mask(
            attention_matrix,
            context_indices)
        masked_attention_matrices.append(selected_submatrix)
    
    # Concatenate all submatrices along the batch dimension (dim=0) and
    # unsqueeze to restore singleton dim
    return torch.unsqueeze(torch.cat(masked_attention_matrices, dim=0), dim=1)


def create_pred_attention_mask(
    attention_matrix: torch.Tensor,
    target_masks: torch.Tensor | None,
    context_masks: torch.Tensor | None,
    n_special_tokens: int) -> torch.Tensor:
    """
    Apply context_masks and/or target_masks to the input attention matrix by
    gathering rows and columns based on the given indices.

    Parameters
    ----------
    attention_matrix: Tensor of shape (B, 1, N, N)
        The input attention matrix where B is the batch size and N is the sequence length.
    target_masks:
        A list of tensors containing indices for the target tokens.
    context_masks:
        A list of tensors containing indices of context tokens.
    n_special_tokens:
        Number of special tokens.

    Returns
    -------
    Tensor
        A concatenated tensor of the attention matrices after applying all the rules.
        for context and/or target.
    """
    # List to store the attention matrices with applied masks
    masked_attention_matrices = []

    # Remove the singleton dimension if it exists in attention_matrix
    attention_matrix = attention_matrix.squeeze(1)

    # Iterate through the context  masks
    for context_indices in context_masks:
        for target_indices in target_masks:
            # Step 1: Concatenate context and target indices excluding
            # special tokens
            combined_indices = torch.cat((
                target_indices,
                context_indices), dim=1)
            selected_submatrix = apply_attention_mask(attention_matrix,
                                                      combined_indices)
            # Add 1s for special tokens
            selected_submatrix = torch.cat(
                [selected_submatrix,
                 torch.ones((selected_submatrix.shape[0], n_special_tokens, selected_submatrix.shape[2]),
                            dtype=selected_submatrix.dtype,
                            device=selected_submatrix.device)], dim=1)
            selected_submatrix = torch.cat(
                [selected_submatrix,
                 torch.ones((selected_submatrix.shape[0], selected_submatrix.shape[1], n_special_tokens),
                            dtype=selected_submatrix.dtype,
                            device=selected_submatrix.device)], dim=2)
            # Add here special token logic
            masked_attention_matrices.append(selected_submatrix)
    
    # Concatenate all submatrices along the batch dimension (dim=0) and
    # unsqueeze to restore singleton dim
    return torch.unsqueeze(torch.cat(masked_attention_matrices, dim=0), dim=1)


def constrain_attention_matrix(attention_matrix: torch.Tensor,
                               seq_len_cell: int
                               ) -> torch.Tensor:
    """
    Constrains attention mask based on cell and neighborhood segments.

    Parameters
    ----------
    collated_masks_attention:
        The multi-dimensional Tensor representing the attention matrix.
    seq_len_cell:
        The sequence length associated with the `cell` segment.
    """
    collated_masks_attention = attention_matrix.expand(
        attention_matrix.shape[0],
        1,
        attention_matrix.shape[-1],
        attention_matrix.shape[-1]).clone()

    # Mask neighborhood gene tokens for index cell gene tokens
    collated_masks_attention[
        :,
        :,
        :seq_len_cell,
        seq_len_cell:] = 0

    # Mask index cell gene tokens for neighborhood gene tokens
    collated_masks_attention[
        :,
        :,
        seq_len_cell:,
        :seq_len_cell] = 0

    return collated_masks_attention
