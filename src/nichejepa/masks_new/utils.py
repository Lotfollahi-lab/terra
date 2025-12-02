"""
Adapted from Bardes, A et al. Revisiting Feature Prediction for Learning
Visual Representations from Video. arXiv:2404.08471 (2024);
https://github.com/facebookresearch/jepa/blob/main/src/masks/utils.py
(25.03.2025).
"""

import torch


def apply_masks(x: torch.Tensor,
                masks: list[torch.Tensor],
                concat: bool = True,
                ) -> list[torch.Tensor] | torch.Tensor:
    """
    Apply masks to an input tensor.

    Parameters
    ----------
    x:
        Tensor of shape (B, N, D); B: batch size, N: number of tokens,
        D: feature dimensions.
    masks:
        List of tensors containing indices of tokens in [N] to keep.
    concat:
        If `True`, concatenate the masked tensors.

    Returns
    ----------
    all_x:
        Either list of masked input tensors or concatenated masked
        input tensors.
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if concat:
        all_x = torch.cat(all_x, dim=0)
    return all_x