"""
Adaptive LayerNorm for per-batch conditioning.

Replaces ``nn.LayerNorm`` inside Transformer blocks with an affine
modulation whose ``(gamma, beta)`` are computed from a per-cell
conditioning vector (typically a batch embedding). DiT / AdaLN-Zero
style (Peebles & Xie 2023): a single linear hypernetwork per AdaLN
maps the conditioning embedding to ``2 * embed_dim`` modulation
parameters, applied as ``gamma * LayerNorm(x) + beta``.

Zero-initialization
-------------------
The modulation linear is initialized to weight=0, bias=0. With the
``gamma`` offset of ``+1`` applied below, this yields gamma=1, beta=0
at step 0 -- so the AdaLN output is mathematically identical to a
plain ``LayerNorm(x)`` at initialization. As training proceeds, the
modulation slowly learns to move ``(gamma, beta)`` away from
identity in a batch-conditional way. This is the key property that
keeps the architecture backward-compatible with existing JEPA
training dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def zero_init_adaln_modulations(module: nn.Module) -> int:
    """Walk ``module`` and re-zero every AdaLN's modulation
    hypernetwork. Returns the number of AdaLN modules that were
    re-initialized.

    Call this AFTER any pass that may have overwritten AdaLN's
    zero-init (e.g. the encoder's ``self.apply(_init_weights)`` or
    the second-pass ``init_weights`` loop in ``init_model``). The
    zero-init is what makes AdaLN start out identical to LayerNorm
    at step 0 -- without restoring it after these passes, AdaLN
    starts as a noisy LayerNorm (~16% per-element perturbation).
    """
    n = 0
    for m in module.modules():
        if isinstance(m, AdaLN):
            m.reset_modulation_to_zero()
            n += 1
    return n


class AdaLN(nn.Module):
    """Adaptive LayerNorm: ``gamma * LayerNorm(x) + beta`` with
    ``(gamma, beta)`` produced from a conditioning vector.

    Parameters
    ----------
    embed_dim:
        Per-token feature dim (same as the dim of ``x``).
    cond_dim:
        Conditioning vector dim. The hypernetwork is
        ``Linear(cond_dim, 2 * embed_dim)``.
    eps:
        LayerNorm epsilon. Default 1e-6 to match the codebase's
        ``LayerNorm`` instantiation.
    """

    def __init__(self,
                 embed_dim: int,
                 cond_dim: int,
                 eps: float = 1e-6,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        # `elementwise_affine=False`: this LN provides only the
        # standardization step. The (gamma, beta) modulation produced
        # by the hypernetwork is applied OUTSIDE this LN, so there's
        # no double-affine.
        self.norm = nn.LayerNorm(embed_dim, eps=eps, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, 2 * embed_dim)
        # Zero-init: at construction, gamma = 1, beta = 0, so the
        # AdaLN output is exactly LayerNorm(x). Without this, the
        # AdaLN would inject random noise at step 0 and break
        # backward-compat with existing training.
        self.reset_modulation_to_zero()

    def reset_modulation_to_zero(self) -> None:
        """Re-initialize the modulation hypernetwork to zeros so that
        at the next forward, gamma = 1 and beta = 0 (i.e. AdaLN
        output == LayerNorm(x) modulo numerical precision).

        Exposed as a method so it can be called AFTER any global
        re-initialization pass (e.g. ``self.apply(_init_weights)`` in
        the encoder / predictor base, or the second ``init_weights``
        pass inside ``init_model``) that would otherwise overwrite
        the original zero-init with ``trunc_normal_``-style noise.
        Calling this is essential for the AdaLN-at-step-0 == LayerNorm
        invariant the tests rely on.
        """
        nn.init.zeros_(self.modulation.weight)
        if self.modulation.bias is not None:
            nn.init.zeros_(self.modulation.bias)

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, cond_dim={self.cond_dim}"

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, L, embed_dim)  -- input to be normalized.
        cond: (B, cond_dim)      -- per-cell conditioning embedding.

        Returns the modulated, normalized tensor of shape
        ``(B, L, embed_dim)``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"AdaLN expects (B, L, embed_dim), got {tuple(x.shape)}.")
        if cond.dim() != 2 or cond.size(-1) != self.cond_dim:
            raise ValueError(
                f"AdaLN expects cond of shape (B, {self.cond_dim}), "
                f"got {tuple(cond.shape)}.")
        # (B, 2*D) -> (B, D), (B, D)
        gb = self.modulation(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        # Broadcast over the L dim. The +1.0 shift on gamma makes the
        # zero-init case identity (LayerNorm output unchanged).
        gamma = 1.0 + gamma.unsqueeze(1)   # (B, 1, D)
        beta = beta.unsqueeze(1)           # (B, 1, D)
        return self.norm(x) * gamma + beta
