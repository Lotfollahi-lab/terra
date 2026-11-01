"""
2D Rotary Position Embedding (RoPE) for spatial transcriptomics.

Applies RoPE to query and key vectors before attention. The head dimension
is split in half: the first half is rotated using angles derived from the
x-coordinate, the second half using angles derived from the y-coordinate.
Within each half, dimensions are rotated pairwise using a standard
geometric frequency schedule.

Ported from the stjepa codebase (sister project). The math, properties,
and integration pattern are identical -- only the host module path
differs.

Properties
----------
- Translation invariance: attention logits depend only on relative
  position (r_j - r_i). You can therefore pass absolute normalized
  coordinates; no need to pre-compute rel_x / rel_y in the tokenizer.
- Within-cell attention is unchanged by RoPE, because all tokens of the
  same cell share the same (x, y), and identical rotations cancel in
  q @ k^T.
- Rotation invariance is achieved statistically via rotation_augment=True
  during training (applies a random SO(2) rotation to coords per sample,
  once per forward pass of the encoder / predictor, see `prepare_coords`).
- Special tokens / padding receive coord (0, 0) (rotation = identity),
  so they do not interfere with the spatial bias.
- The frequencies assume coordinates of order 1 (``freq_scale``). The TERRA
  tokenizers give displacements in micrometres; with the default
  ``freq_scale`` (pi) the high frequencies then wrap many times between
  neighbouring cells, so set ``freq_scale`` to about pi / (largest neighbour
  distance in micrometres). A warning is logged once if they wrap.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RoPE2D(nn.Module):
    """
    Precomputes and applies 2D rotary position embedding.

    Parameters
    ----------
    head_dim:
        Per-head dimension of q / k. Must be divisible by 4 (head_dim / 2
        dims allocated to x, head_dim / 2 to y, each a multiple of 2 so
        pairwise rotation is well-defined).
    base:
        Frequency base. 10000.0 is the Vaswani / LLaMA default.
    freq_scale:
        Multiplicative scale applied to the frequencies. With coordinates
        normalized so typical neighbor displacement lies in [-1, 1],
        freq_scale ~= pi gives roughly one full oscillation across the
        neighborhood at the highest frequency channel. Tune this: for
        displacements in micrometres use about pi / (largest neighbor
        distance), e.g. 0.03 for 100 um.
    rotation_augment:
        If True, at training time each sample's (x, y) is rotated by a
        uniformly random angle before RoPE is applied (`prepare_coords`,
        once per forward pass, so all attention layers see the same
        rotation). This yields statistical rotation invariance with no
        loss of capacity.
    """

    # Whether the coordinate range was checked against the frequencies
    # (once per process, see `_check_aliasing`)
    _aliasing_checked = False

    def __init__(self,
                 head_dim: int,
                 base: float = 10000.0,
                 freq_scale: float = math.pi,
                 rotation_augment: bool = True,
                 ):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}"
            )
        self.head_dim = head_dim
        self.freq_scale = freq_scale
        self.rotation_augment = rotation_augment

        half = head_dim // 2  # dims allocated to each axis
        # freqs: (head_dim / 4,)
        # Standard geometric schedule over the per-axis half-dim.
        exponent = torch.arange(0, half, 2, dtype=torch.float32) / half
        freqs = (1.0 / (base ** exponent)) * freq_scale
        self.register_buffer("freqs", freqs, persistent=False)

    def prepare_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Coordinates for one forward pass of an encoder or predictor:
        non-finite sentinels (special tokens, padding) set to (0, 0), and
        during training with `rotation_augment` each sample rotated by one
        random angle. All attention layers of the pass use the result
        (`forward` no longer rotates, which drew a new angle in every
        layer, also for the target encoder).

        coords: (B, N, 2)
        """
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        self._check_aliasing(coords)
        return self._maybe_rotate_coords(coords)

    def _check_aliasing(self, coords: torch.Tensor) -> None:
        """
        Warn once per process if the highest frequencies turn by more than
        half a period over the coordinate range, so that they cannot
        resolve positions unambiguously (e.g. micrometre coordinates with
        the default `freq_scale`).
        """
        if RoPE2D._aliasing_checked:
            return
        extent = float(coords.detach().abs().max()) if coords.numel() else 0.
        if extent == 0.:
            return
        RoPE2D._aliasing_checked = True
        n_aliased = int((self.freqs * extent > math.pi).sum())
        if n_aliased:
            logger.warning(
                f"RoPE: coordinates reach {extent:.3g}, and with "
                f"rope_freq_scale {self.freq_scale:.3g} {n_aliased} of the "
                f"{len(self.freqs)} frequencies per axis turn by more than "
                f"half a period over that distance (shortest period "
                f"{2 * math.pi / float(self.freqs.max()):.3g}), so they do "
                "not resolve positions unambiguously. For coordinates in "
                "micrometres (TERRA tokenizers), set meta.rope_freq_scale "
                "to about pi / (largest neighbor distance), e.g. 0.03 for "
                "100 um.")

    def _maybe_rotate_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply a random per-sample SO(2) rotation to (x, y) during training.
        """
        if not (self.training and self.rotation_augment):
            return coords
        B = coords.shape[0]
        theta = torch.rand(B, device=coords.device) * (2.0 * math.pi)
        c, s = theta.cos(), theta.sin()
        # rot: (B, 2, 2)
        rot = torch.stack(
            [torch.stack([c, -s], dim=-1),
             torch.stack([s,  c], dim=-1)],
            dim=-2,
        )
        # coords: (B, N, 2) -> (B, N, 2)
        return torch.einsum("bij,bnj->bni", rot, coords)

    def _cos_sin(self, coords: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        coords: (B, N, 2)
        Returns cos, sin of shape (B, N, head_dim / 2).
        """
        x = coords[..., 0]  # (B, N)
        y = coords[..., 1]

        # (B, N, head_dim / 4)
        x_ang = torch.einsum("bn,d->bnd", x, self.freqs)
        y_ang = torch.einsum("bn,d->bnd", y, self.freqs)
        # (B, N, head_dim / 2)
        ang = torch.cat([x_ang, y_ang], dim=-1)
        return ang.cos(), ang.sin()

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                coords: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k:   (B, H, N, head_dim)
        coords: (B, N, 2) -- normalized (x, y), from `prepare_coords`
                (rotated there during training). Special / padding tokens
                should carry (0, 0).

        Returns q_rot, k_rot with the same shapes as inputs.
        """
        cos, sin = self._cos_sin(coords)           # (B, N, head_dim / 2)
        cos = cos.unsqueeze(1)                     # (B, 1, N, head_dim / 2)
        sin = sin.unsqueeze(1)

        q_rot = _apply_rotary_pairwise(q, cos, sin)
        k_rot = _apply_rotary_pairwise(k, cos, sin)
        return q_rot, k_rot


def _apply_rotary_pairwise(t: torch.Tensor,
                           cos: torch.Tensor,
                           sin: torch.Tensor,
                           ) -> torch.Tensor:
    """
    Applies pairwise 2D rotation to the head dim of t.

    t:   (B, H, N, head_dim)
    cos, sin: (B, 1, N, head_dim / 2)

    Pairing scheme: dims are viewed as (head_dim / 2, 2) pairs, where
    each pair (a, b) is rotated by the corresponding angle into
    (a cos - b sin,  a sin + b cos).
    """
    *lead, d = t.shape
    t_pair = t.reshape(*lead, d // 2, 2)
    a = t_pair[..., 0]
    b = t_pair[..., 1]

    rot_a = a * cos - b * sin
    rot_b = a * sin + b * cos

    out = torch.stack([rot_a, rot_b], dim=-1)
    return out.reshape(*lead, d)
