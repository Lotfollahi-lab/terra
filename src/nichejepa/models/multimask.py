"""
Adapted from Bardes, A et al. Revisiting Feature Prediction for Learning
Visual Representations from Video. arXiv:2404.08471 (2024);
https://github.com/facebookresearch/jepa/blob/main/src/models/utils/multimask.py
(25.03.2025).
"""

import torch
import torch.nn as nn


class EncoderMultiMaskWrapper(nn.Module):
    """
    EncoderMultiMaskWrapper class for encoding iteratively with multiple
    masks.

    Parameters
    ----------
    backbone:
        The encoder backbone network.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self,
                tokens: torch.Tensor,
                segments: torch.Tensor,
                counts: torch.Tensor,
                masks: torch.Tensor | list | None = None,
                masks_attention: torch.Tensor | None = None
                ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if masks is None:
            return self.backbone(tokens=tokens,
                                 segments=segments,
                                 counts=counts,
                                 masks=None,
                                 masks_attention=masks_attention)

        if (masks is not None) and not isinstance(masks, list):
            masks = [masks]
        outs = []
        for m in masks:
            x, token_embed = self.backbone(tokens=tokens,
                                           segments=segments,
                                           counts=counts,
                                           masks=m,
                                           masks_attention=masks_attention)
            outs += [x]
        return outs, token_embed


class PredictorMultiMaskWrapper(nn.Module):
    """
    PredictorMultiMaskWrapper class for predicting iteratively with
    multiple masks.

    Only works with a single context/encoder mask.

    Parameters
    ----------
    backbone:
        The predictor backbone network.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self,
                z: torch.Tensor | list,
                token_embed: torch.Tensor,
                segments: torch.Tensor,
                counts: torch.Tensor,
                masks_enc: torch.Tensor | list,
                masks_pred: torch.Tensor | list,
                masks_attention: torch.Tensor
                ) -> list[torch.Tensor]:
        if type(z) is not list:
            z = [z]
        if type(masks_enc) is not list:
            masks_enc = [masks_enc]
        if type(masks_pred) is not list:
            masks_pred = [masks_pred]

        outs = []
        for mp in masks_pred:
            outs += [
                self.backbone(z=z[0],
                              token_embed=token_embed,
                              segments=segments,
                              counts=counts,
                              masks_enc=masks_enc[0],
                              masks_pred=mp,
                              masks_attention=masks_attention)]
        return outs