from typing import Optional

import torch


def apply_masks(x, masks):
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

    # Step 2: Expand combined indices for row selection
    row_indices_expanded = indices.unsqueeze(-1).expand(-1, -1, attention_matrix.size(-1))

    # Step 3: Gather the rows from the attention matrix based on the expanded indices
    selected_rows = torch.gather(attention_matrix, 1, row_indices_expanded)

    # Step 4: Expand combined indices for column selection
    column_indices_expanded = indices.unsqueeze(1).expand(-1, selected_rows.size(1), -1)

    # Step 5: Gather columns from the selected rows based on the expanded indices
    selected_submatrix = torch.gather(selected_rows, 2, column_indices_expanded)

    return selected_submatrix


def create_controlled_mask_context_target(
    attention_matrix: torch.Tensor,
    target_masks: Optional[torch.Tensor]=None,
    context_masks: Optional[torch.Tensor]=None):
    """
    Apply context_masks and/or target_masks to the input attention matrix by
    gathering rows and columns based on the given indices.

    Parameters
    ----------
    attention_matrix: Tensor of shape (B, 1, N, N)
        The input attention matrix where B is the batch size and N is the sequence length.

    target_masks : List[Tensor]
        A list of tensors containing indices for the target tokens.

    context_masks: List[Tensor]
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
    # Check if target_masks isn't None
    if target_masks is None:
        # Iterate through the context  masks
        for context_indices in context_masks:

            selected_submatrix = apply_attention_mask(attention_matrix,
                                                      context_indices)
            
            # Append the resulting submatrix to the list
            masked_attention_matrices.append(selected_submatrix)
        
        # Concatenate all submatrices along the batch dimension (dim=0) and unsqueeze to restore singleton dim
        return torch.unsqueeze(torch.cat(masked_attention_matrices, dim=0), dim=1)

    # Iterate through the context and target masks
    for context_indices in context_masks:
        for mask_indices in target_masks:
            # Step 1: Concatenate context and target indices together
            combined_indices = torch.cat((context_indices, mask_indices), dim=1)
            # Call the helper function to create the attention matrices for the given indices
            selected_submatrix = apply_attention_mask(attention_matrix, combined_indices)
            
            # Append the resulting submatrix to the list
            masked_attention_matrices.append(selected_submatrix)

    # Concatenate all submatrices along the batch dimension and unsqueeze to
    # restore singleton dim
    return torch.unsqueeze(torch.cat(masked_attention_matrices, dim=0), dim=1)


def configure_attention_masks(controlled_attention_pattern: torch.Tensor,
                              collated_masks_attention: torch.Tensor, 
                              seq_len_cell: int,
                              valid_min_start: int):
    """
    Configures attention masks based on the controlled attention pattern.

    This function sets specific positions in the attention mask matrix to zero,
    based on the given `controlled_attention_pattern`. This ensures that certain
    parts of the sequence do not attend to other as defined by the provided
    masking rules.

    The function updates `collated_masks_attention` in-place based on the pattern.

    Parameters
    ----------
    controlled_attention_pattern:
        A 2D Torch.tensor that defines whether specific parts should not attend
        to other parts. A value of `1` indicates masking.
    collated_masks_attention:
        The multi-dimensional Tensor representing the attention matrix, which
        will be updated based on the pattern.
    seq_len_cell:
        The sequence length associated with the `cell` segment.
    valid_min_start:
        The starting index of valid tokens for masking, excluding special
        tokens, within the attention matrix.
    """
    # <cls_cell> token does not attend to <cls_neighborhood> token
    if controlled_attention_pattern[0][1]:
        collated_masks_attention[
            :,
            :,
            0,
            1] = 0
    
    # <cls_cell> token does not attend to neighborhood gene tokens
    if controlled_attention_pattern[0][3]:
        collated_masks_attention[
            :,
            :,
            0,
            valid_min_start + seq_len_cell:] = 0
    
    # <cls_neighborhood> token does not attend to <cls_cell> token
    if controlled_attention_pattern[1][0]:
        collated_masks_attention[
            :,
            :,
            1,
            0] = 0
    
    # <cls_neighborhood> token does not attend to cell gene tokens
    if controlled_attention_pattern[1][2]:
        collated_masks_attention[
            :,
            :,
            1,
            valid_min_start: valid_min_start + seq_len_cell] = 0
    
    # Cell gene tokens do not attend to <cls_neighborhood> token
    if controlled_attention_pattern[2][1]:
        collated_masks_attention[
            :,
            :,
            valid_min_start: valid_min_start + seq_len_cell,
            1] = 0
    
    # Cell gene tokens do not attend to neighborhood gene tokens
    if controlled_attention_pattern[2][3]:
        collated_masks_attention[
            :,
            :,
            valid_min_start: valid_min_start + seq_len_cell,
            valid_min_start + seq_len_cell:] = 0
    
    # Neighborhood gene tokens do not attend to <cls_cell> token
    if controlled_attention_pattern[3][0]:
        collated_masks_attention[
            :,
            :,
            valid_min_start + seq_len_cell:,
            0] = 0
    
    # Neighborhood gene tokens do not attend to cell gene tokens
    if controlled_attention_pattern[3][2]:
        collated_masks_attention[
            :,
            :,
            valid_min_start + seq_len_cell:,
            valid_min_start: valid_min_start + seq_len_cell] = 0

    # Temp Batch token only attends to itself
    collated_masks_attention[
            :,
            :,
            2,
            0:2] = 0
    collated_masks_attention[
            :,
            :,
            2,
            3] = 0
    collated_masks_attention[
            :,
            :,
            3,
            0:3] = 0
    collated_masks_attention[
            :,
            :,
            2,
            valid_min_start + seq_len_cell:] = 0
    collated_masks_attention[
            :,
            :,
            3,
            valid_min_start: valid_min_start + seq_len_cell] = 0

    collated_masks_attention[
        :,
        :,
        valid_min_start: valid_min_start + seq_len_cell,
        3] = 0
    collated_masks_attention[
        :,
        :,
        valid_min_start + seq_len_cell:,
        2] = 0
