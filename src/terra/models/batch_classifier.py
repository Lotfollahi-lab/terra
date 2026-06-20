"""
Adversarial batch debiasing for batch-effect removal.

DANN-style (Ganin & Lempitsky 2015) gradient-reversal layer plus a small
MLP classifier head trained jointly with the JEPA loss. The encoder
sees a negated, scaled version of the classifier's gradient, so it is
pushed to produce cell embeddings from which the batch ID cannot be
predicted.

How it composes with JEPA training
----------------------------------
1. Encoder produces cell embeddings ``z`` of shape ``(B, L, D)``.
2. Mean-pool ``z`` to ``(B, D)``.
3. Pass through ``GradReverseLayer(grl_alpha)`` (identity forward,
   negative-scaled gradient backward).
4. Classifier head MLP -> logits of shape ``(B, n_batches)``.
5. Cross-entropy loss against per-cell batch labels, weighted by
   ``lambda_adv`` and added to the JEPA loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradReverseFn(torch.autograd.Function):
    """Identity in forward; negates and scales the gradient in
    backward. Implements the gradient reversal layer of DANN.

    ``alpha`` is the negative-scale factor: the gradient flowing back
    into the encoder is ``-alpha * dL/d(cell_emb)``. Set ``alpha`` to
    0 to disable the reversal (the gradient through this op becomes
    zero, which effectively detaches the classifier from the encoder).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        # Gradient is multiplied by -alpha when flowing back to the
        # input. The second return is None because alpha is not a
        # learnable tensor.
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Functional wrapper around ``GradReverseFn`` for convenience."""
    return GradReverseFn.apply(x, alpha)


class GradReverseLayer(nn.Module):
    """``nn.Module`` wrapper around ``GradReverseFn`` so the layer's
    ``alpha`` can be inspected / changed cleanly (e.g. via a curriculum
    schedule).
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradReverseFn.apply(x, self.alpha)


class BatchClassifierHead(nn.Module):
    """A small MLP head that predicts a batch ID from a cell embedding.

    Architecture: ``Linear(embed_dim, hidden_dim) -> GELU -> Dropout ->
    Linear(hidden_dim, n_classes)``. The gradient-reversal layer is
    NOT included here -- it's applied by the caller right before the
    head so the same head can be used for diagnostic prediction
    without gradient reversal.

    Multi-key support
    -----------------
    If ``n_classes_per_key`` is a list, the head becomes multi-output:
    one independent classifier per key, sharing the hidden trunk.
    Forward returns a **list** of logits tensors, one per key
    (shape ``(B, n_classes[k])``). Single-key reduces to a single
    tensor in the list. The caller is responsible for summing
    cross-entropy losses across keys.

    Parameters
    ----------
    embed_dim:
        Cell embedding dimension at the input.
    n_classes_per_key:
        Either a single int (single-key, legacy) or a list of ints
        (multi-key, one classifier per key).
    hidden_dim:
        Hidden-layer width. Shared across all keys.
    dropout:
        Dropout between hidden layer and output heads.
    """

    def __init__(self,
                 embed_dim: int,
                 n_classes_per_key: int | list[int] | tuple[int, ...],
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        if isinstance(n_classes_per_key, int):
            n_classes_per_key = [n_classes_per_key]
        else:
            n_classes_per_key = list(n_classes_per_key)
        if not n_classes_per_key:
            raise ValueError(
                "BatchClassifierHead requires at least one key.")
        for k, n in enumerate(n_classes_per_key):
            if int(n) < 2:
                raise ValueError(
                    f"n_classes_per_key[{k}] must be >= 2, got {n}.")
        self.embed_dim = embed_dim
        self.n_classes_per_key = [int(n) for n in n_classes_per_key]
        # Shared trunk -> per-key linear heads.
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, int(n))
            for n in self.n_classes_per_key
        ])

    def forward(self, cell_emb: torch.Tensor) -> list[torch.Tensor]:
        """Returns a list of logits tensors, one per key. Shape per
        entry: ``(B, n_classes[k])``. Single-key returns a
        one-element list.
        """
        h = self.trunk(cell_emb)
        return [head(h) for head in self.heads]


def mean_pool_cell_embedding(
        z: torch.Tensor, n_special_tokens: int = 0,
        ) -> torch.Tensor:
    """Cell-level embedding for downstream batch classification.

    Mean-pools the per-token encoder output over all positions except
    the special-token prefix. Operates on a single context-mask output
    of shape ``(B, L, D)``.

    The pool excludes special tokens (CLS / spt_batch / etc.) because
    those positions carry batch-conditioning info directly -- including
    them would let the classifier shortcut via the special-token
    embedding rather than the cell representation we actually want
    debiased.
    """
    if z.dim() != 3:
        raise ValueError(
            f"Expected (B, L, D) tensor, got shape {tuple(z.shape)}.")
    if n_special_tokens > 0:
        z = z[:, n_special_tokens:, :]
    return z.mean(dim=1)
