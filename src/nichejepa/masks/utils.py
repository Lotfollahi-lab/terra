"""
Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture.
Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/masks/utils.py
(05.06.2024).
"""

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


def apply_attention_mask(attention_matrix, mask_indices):
    """
    Apply  mask indices to the attention matrix and return the selected submatrix.

    Parameters
    ----------
    attention_matrix : Tensor of shape (B, N, N)
        The input attention matrix.

    mask_indices : Tensor of shape (B, M)
        Indices of tokens for which the attention mask should happen.

    Returns
    -------
    Tensor
        The submatrix from the attention matrix after applying the mask indices.
    """

    # Step 2: Expand combined indices for row selection
    row_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, attention_matrix.size(-1))

    # Step 3: Gather the rows from the attention matrix based on the expanded indices
    selected_rows = torch.gather(attention_matrix, 1, row_indices_expanded)

    # Step 4: Expand combined indices for column selection
    column_indices_expanded = mask_indices.unsqueeze(1).expand(-1, selected_rows.size(1), -1)

    # Step 5: Gather columns from the selected rows based on the expanded indices
    selected_submatrix = torch.gather(selected_rows, 2, column_indices_expanded)

    return selected_submatrix


def create_controlled_mask(attention_matrix, target_masks=None, context_masks=None):
    """
    Apply predicted masks to the input attention matrix by gathering rows and columns based 
    on the given indices.

    Parameters
    ----------
    attention_matrix : Tensor of shape (B, N, N)
        The input attention matrix where B is the batch size and N is the sequence length.

    target_masks : List[Tensor]
        A list of tensors containing indices for the target masks.

    context_masks: List[Tensor]
        A list of tensors containing indices of context tokens to keep.

    Returns
    -------
    Tensor
        A concatenated tensor of the attention matrices after applying all the rules.
    """
    # List to store the attention matrices with applied masks
    masked_attention_matrices = []

    # Remove the singleton dimension if it exists in attention_matrix
    attention_matrix = attention_matrix.squeeze(1)

    if target_masks is None:
        for context_indices in context_masks:

            selected_submatrix = apply_attention_mask(attention_matrix, context_indices)
            
            # Append the resulting submatrix to the list
            masked_attention_matrices.append(selected_submatrix)
        
        # Concatenate all submatrices along the batch dimension (dim=0) and unsqueeze to restore singleton dim
        return torch.unsqueeze(torch.cat(masked_attention_matrices, dim=0), dim=1)

    # Iterate through the context and target masks
    for context_indices in context_masks:
        for mask_indices in target_masks:
            # Step 1: Concatenate context and target indices together
            combined_indices = torch.cat((context_indices, mask_indices), dim=1)
            # Call the helper function to create the attention mask for the given indices
            selected_submatrix = apply_attention_mask(attention_matrix, combined_indices)
            
            # Append the resulting submatrix to the list
            masked_attention_matrices.append(selected_submatrix)

    # Concatenate all submatrices along the batch dimension (dim=0) and unsqueeze to restore singleton dim
    return torch.unsqueeze(torch.cat(masked_attention_matrices, dim=0), dim=1)

