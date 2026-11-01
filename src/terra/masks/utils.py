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


def padding_key_masks(masks_enc_valid: torch.Tensor,
                      masks_pred_valid: torch.Tensor,
                      n_special_tokens: int,
                      new_spc: bool,
                      pad_special_tokens: bool = False,
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Attention key masks for context and target masks padded by
    ``BlockMaskCollator(pad_masks_to_batch_max=True)``.

    Parameters
    ----------
    masks_enc_valid:
        Boolean validity of the context mask entries, shape (B, L_ctx).
    masks_pred_valid:
        Boolean validity of the target mask entries, shape (B, L_tgt).
    n_special_tokens:
        Number of special tokens at the start of each mask.
    new_spc:
        The predictor's `new_spc` setting, which determines the layout of
        its input sequence.
    pad_special_tokens:
        If `True`, the special tokens of the batch are padded (not in the
        masks), and the predictor uses the `new_spc=False` layout.

    Returns
    ----------
    enc_masks_attention:
        Key mask for the context encoder, shape (B, 1, 1, L_ctx).
    pred_masks_attention:
        Key mask for the predictor, shape (B, 1, 1, L_ctx + L_tgt), in the
        order of the predictor's input sequence: ``[context special tokens,
        target tokens without special tokens, context gene tokens]`` if
        `new_spc` (and the special tokens are not padded), else ``[target
        tokens, context tokens]``.
    """
    if new_spc and not pad_special_tokens:
        pred_valid = torch.cat([
            masks_enc_valid[:, :n_special_tokens],
            masks_pred_valid[:, n_special_tokens:],
            masks_enc_valid[:, n_special_tokens:]], dim=1)
    else:
        pred_valid = torch.cat([masks_pred_valid, masks_enc_valid], dim=1)
    return (masks_enc_valid[:, None, None, :],
            pred_valid[:, None, None, :])
