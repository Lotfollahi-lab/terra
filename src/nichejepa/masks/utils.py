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
    n_special_tokens: Optional[int]=None,
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

    # Iterate through the context  masks
    for context_indices in context_masks:
        if target_masks is None:
            selected_submatrix = apply_attention_mask(attention_matrix,
                                                      context_indices)
            masked_attention_matrices.append(selected_submatrix)
        else:
            for mask_indices in target_masks:
                # Step 1: Concatenate context and target indices excluding
                # special tokens
                combined_indices = torch.cat((
                    mask_indices[:, n_special_tokens:],
                    context_indices[:, n_special_tokens:]), dim=1)
                selected_submatrix = apply_attention_mask(attention_matrix,
                                                          combined_indices)
                masked_attention_matrices.append(selected_submatrix)
    
    # Concatenate all submatrices along the batch dimension (dim=0) and
    # unsqueeze to restore singleton dim
    return torch.unsqueeze(torch.cat(masked_attention_matrices, dim=0), dim=1)


def configure_attention_masks(controlled_attention_pattern: torch.Tensor,
                              collated_masks_attention: torch.Tensor, 
                              seq_len_cell: int,
                              n_special_tokens: int,
                              max_cls_tokens: int,
                              n_segments: int,
                              ):
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
    n_special_tokens:
        The starting index of valid tokens for masking, excluding special
        tokens, within the attention matrix.
    max_cls_tokens:
    n_segments:
    """
    # <cls> tokens do not attent to other <cls> tokens
    if controlled_attention_pattern[0][1]:
        for i in range(max_cls_tokens):
            if i != 0: # first <cls> token has no preceding <cls> tokens
                collated_masks_attention[
                    :,
                    :,
                    i,
                    :i] = 0
            if i != (max_cls_tokens - 1): # last <cls> token has no subsequent 
                                          # <cls> tokens
                collated_masks_attention[
                    :,
                    :,
                    i,
                    i+1:max_cls_tokens] = 0

    # <cls> tokens do not attent to other gene tokens
    if controlled_attention_pattern[0][3]:
        for i in range(max_cls_tokens):
            if i != 0: # first <cls> token has no preceding gene tokens
                collated_masks_attention[
                    :,
                    :,
                    i,
                    n_special_tokens: n_special_tokens + (i * seq_len_cell)] = 0
            if i != (max_cls_tokens - 1): # last <cls> token has no subsequent
                                          # gene tokens
                collated_masks_attention[
                    :,
                    :,
                    i,
                    n_special_tokens + ((i + 1) * seq_len_cell):] = 0

    # Gene tokens do not attent to own <cls> tokens
    if controlled_attention_pattern[1][0]:
        for i in range(max_cls_tokens):
            start_idx = n_special_tokens + (i * seq_len_cell)
            if i == (max_cls_tokens - 1):
                end_idx = None # last <cls> token will capture until end
            else:
                end_idx = n_special_tokens + ((i + 1) * seq_len_cell)

            collated_masks_attention[
                :,
                :,
                start_idx: end_idx,
                i] = 0

    # Gene tokens do not attent to other <cls> tokens
    if controlled_attention_pattern[1][1]:
        for i in range(max_cls_tokens):
            start_idx = n_special_tokens + (i * seq_len_cell)
            if i == (max_cls_tokens - 1):
                end_idx = None # last <cls> token will capture until end
            else:
                end_idx = n_special_tokens + ((i + 1) * seq_len_cell)

            if i != 0: # first gene tokens have no preceding <cls> token
                collated_masks_attention[
                    :,
                    :,
                    start_idx: end_idx,
                    :i] = 0
            if i != (max_cls_tokens - 1): # last gene tokens have no subsequent
                                          # <cls> token
                collated_masks_attention[
                    :,
                    :,
                    start_idx: end_idx,
                    i+1:max_cls_tokens] = 0

    # Gene tokens do not attent to other gene tokens
    if controlled_attention_pattern[1][3]:
        for i in range(n_segments):
            start_idx = n_special_tokens + (i * seq_len_cell)
            if i == (n_segments - 1):
                end_idx = None # last <cls> token will capture until end
            else:
                end_idx = n_special_tokens + ((i + 1) * seq_len_cell)

            if i != 0: # first gene tokens have no preceding gene tokens
                collated_masks_attention[
                    :,
                    :,
                    start_idx: end_idx,
                    n_special_tokens: n_special_tokens + (i * seq_len_cell)] = 0 
            if i != (max_cls_tokens - 1): # last gene tokens have no subsequent
                                          # gene tokens                 
                collated_masks_attention[
                    :,
                    :,
                    start_idx: end_idx,
                    n_special_tokens + ((i + 1) * seq_len_cell):] = 0
