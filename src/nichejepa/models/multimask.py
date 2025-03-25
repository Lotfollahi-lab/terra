"""
Adapted from Bardes, A et al. Revisiting Feature Prediction for Learning Visual Representations from Video.
arXiv:2404.08471 (2024); https://github.com/facebookresearch/jepa/blob/main/src/models/utils/multimask.py (25.03.2025).
"""


import torch.nn as nn


class EncoderMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, tokens, segments, counts, masks=None, masks_attention=None):
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

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, z, token_embed, segments, counts, masks_enc, masks_pred, masks_attention):
        if type(z) is not list:
            z = [z]
        if type(masks_enc) is not list:
            masks_enc = [masks_enc]
        if type(masks_pred) is not list:
            masks_pred = [masks_pred]

        outs = []
        for i, (zi, me, mp) in enumerate(zip(z, masks_enc, masks_pred)):
            outs += [
                self.backbone(z=zi,
                              token_embed=token_embed,
                              segments=segments,
                              counts=counts,
                              masks_enc=me,
                              masks_pred=mp,
                              masks_attention=masks_attention)]
        return outs