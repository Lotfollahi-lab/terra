"""
Adapted from Bardes, A et al. Revisiting Feature Prediction for Learning
Visual Representations from Video. arXiv:2404.08471 (2024);
https://github.com/facebookresearch/jepa/blob/main/src/models/utils/multimask.py
(25.03.2025).
"""

from typing import Literal

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
    encoder_type:
        Type of the encoder, either 'count' or 'rank.
    """
    def __init__(self,
                 backbone: nn.Module,
                 encoder_type: Literal['rank', 'counts']):
        super().__init__()
        self.backbone = backbone
        self.encoder_type = encoder_type

        # Define the forward method based on encoder_type
        if self.encoder_type == 'counts':
            self.forward = self._forward_count
        elif self.encoder_type == 'rank':
            self.forward = self._forward_rank
        else:
            raise ValueError("encoder_type must be either 'counts' or 'rank'")

    def _forward_count(self,
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

        if isinstance(masks, torch.Tensor):
            masks = [masks]
        outs = []
        for m in masks:
            x, token_embed = self.backbone(tokens=tokens,
                                           segments=segments,
                                           counts=counts,
                                           masks=m,
                                           masks_attention=masks_attention)
            outs.append(x)
        return outs, token_embed

    def _forward_rank(self,
                      positions: torch.Tensor,
                      segments: torch.Tensor,
                      tokens: torch.Tensor,
                      masks: torch.Tensor | list | None = None,
                      masks_attention: torch.Tensor | None = None
                      ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if masks is None:
            return self.backbone(positions=positions,
                                 segments=segments,
                                 tokens=tokens,
                                 masks=None,
                                 masks_attention=masks_attention)

        if isinstance(masks, torch.Tensor):
            masks = [masks]
        outs = []
        for m in masks:
            x, pos_embed, token_embed = self.backbone(
                positions=positions,
                segments=segments,
                tokens=tokens,
                masks=m,
                masks_attention=masks_attention)
            outs.append(x)
        return outs, pos_embed, token_embed


class PredictorMultiMaskWrapper(nn.Module):
    """
    PredictorMultiMaskWrapper class for predicting iteratively with
    multiple masks.

    Only works with a single context/encoder mask.

    Parameters
    ----------
    backbone:
        The predictor backbone network.
    predictor_type:
        Type of the predictor, either 'counts' or 'rank'.
    """
    def __init__(self, backbone: nn.Module, predictor_type: Literal['rank', 'counts']):
        super().__init__()
        self.backbone = backbone
        self.predictor_type = predictor_type
        
        # Define the forward method based on predictor_type
        if self.predictor_type == 'counts':
            self.forward = self._forward_count
        elif self.predictor_type == 'rank':
            self.forward = self._forward_rank
        else:
            raise ValueError("predictor_type must be either 'counts' or 'rank'")

    def _forward_count(self,
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

    def _forward_rank(self,
                      z: torch.Tensor | list,
                      pos_embed: torch.Tensor,
                      segments: torch.Tensor,
                      token_embed: torch.Tensor,
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
                              pos_embed=pos_embed,
                              segments=segments,
                              token_embed=token_embed,
                              masks_enc=masks_enc[0],
                              masks_pred=mp,
                              masks_attention=masks_attention)]
        return outs