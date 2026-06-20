"""Tests for the batch-correction features:
   1. Adversarial batch classifier with gradient reversal.
   2. AdaLN-style per-batch conditioning of the transformer norms.

The key correctness invariant for both features is **backward
compatibility**: with the features disabled, the encoder/predictor
produce byte-identical outputs to the pre-feature implementation,
and with AdaLN enabled at construction time the zero-init of the
modulation hypernetwork makes the AdaLN-on output exactly equal to
the AdaLN-off output at step 0.
"""

import math

import pytest
import torch
import torch.nn as nn

from terra.models.adaln import AdaLN
from terra.models.batch_classifier import (
    BatchClassifierHead,
    GradReverseFn,
    GradReverseLayer,
    grad_reverse,
    mean_pool_cell_embedding,
)
from terra.models.gene_transformers import GeneTransformerRankEncoder
from terra.models.modules import Block


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------

def test_grad_reverse_forward_is_identity():
    x = torch.randn(4, 8)
    y = GradReverseFn.apply(x, 1.0)
    torch.testing.assert_close(y, x)


def test_grad_reverse_backward_negates_scaled():
    x = torch.randn(4, 8, requires_grad=True)
    alpha = 2.5
    y = GradReverseFn.apply(x, alpha)
    # dL/dy = ones; gradient flowing back should be -alpha * ones.
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(
        x.grad, torch.full_like(x.grad, -alpha), atol=1e-6, rtol=0)


def test_grad_reverse_layer_module_form():
    layer = GradReverseLayer(alpha=0.3)
    x = torch.randn(2, 4, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    torch.testing.assert_close(
        x.grad, torch.full_like(x.grad, -0.3), atol=1e-6, rtol=0)


def test_grad_reverse_alpha_zero_blocks_grad():
    """Alpha=0 effectively detaches the classifier from the encoder
    via gradient reversal. Useful for a curriculum that delays the
    adversarial signal."""
    x = torch.randn(4, 8, requires_grad=True)
    y = grad_reverse(x, alpha=0.0)
    y.sum().backward()
    torch.testing.assert_close(
        x.grad, torch.zeros_like(x.grad), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Batch classifier head
# ---------------------------------------------------------------------------

def test_batch_classifier_head_shape_single_key():
    head = BatchClassifierHead(
        embed_dim=32, n_classes_per_key=10, hidden_dim=16)
    out = head(torch.randn(7, 32))
    # Multi-head API: forward returns list[Tensor], one per key.
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].shape == (7, 10)


def test_batch_classifier_head_shape_multi_key():
    head = BatchClassifierHead(
        embed_dim=32, n_classes_per_key=[10, 5], hidden_dim=16)
    out = head(torch.randn(7, 32))
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape == (7, 10)
    assert out[1].shape == (7, 5)


def test_batch_classifier_n_classes_validated():
    with pytest.raises(ValueError, match="must be >= 2"):
        BatchClassifierHead(embed_dim=32, n_classes_per_key=1)
    with pytest.raises(ValueError, match="must be >= 2"):
        BatchClassifierHead(embed_dim=32, n_classes_per_key=[5, 1])


def test_batch_classifier_empty_keys_raises():
    with pytest.raises(ValueError, match="at least one key"):
        BatchClassifierHead(embed_dim=32, n_classes_per_key=[])


def test_mean_pool_cell_embedding_excludes_special_tokens():
    z = torch.zeros(2, 8, 4)
    # Set special-token positions to 100, non-special to 1. Mean
    # pool with n_special=2 should give all 1s.
    z[:, :2, :] = 100.0
    z[:, 2:, :] = 1.0
    pooled = mean_pool_cell_embedding(z, n_special_tokens=2)
    torch.testing.assert_close(pooled, torch.ones(2, 4))


# ---------------------------------------------------------------------------
# AdaLN module
# ---------------------------------------------------------------------------

def test_adaln_zero_init_matches_plain_layernorm():
    """The single most important invariant: at construction time
    (before any training), AdaLN(x, cond) must equal LayerNorm(x).
    This is what makes ``adaln.enabled = True`` a zero-cost
    addition that doesn't disturb the rest of the model.
    """
    torch.manual_seed(0)
    embed_dim, cond_dim = 16, 8
    adaln = AdaLN(embed_dim, cond_dim)
    plain_ln = nn.LayerNorm(embed_dim, eps=1e-6)
    # plain_ln is initialized with weight=1, bias=0 by default.
    x = torch.randn(3, 5, embed_dim)
    cond = torch.randn(3, cond_dim)  # any cond -> hypernet outputs zeros
    out_adaln = adaln(x, cond)
    out_plain = plain_ln(x)
    torch.testing.assert_close(
        out_adaln, out_plain, atol=1e-5, rtol=1e-5)


def test_adaln_invariant_under_cond_at_step_zero():
    """At init, the AdaLN output does NOT depend on cond (because
    gamma=1, beta=0 regardless of the cond value). Two different
    cond vectors must produce the same output for the same x."""
    embed_dim, cond_dim = 16, 8
    adaln = AdaLN(embed_dim, cond_dim)
    x = torch.randn(3, 5, embed_dim)
    cond_a = torch.randn(3, cond_dim)
    cond_b = torch.randn(3, cond_dim) * 10.0
    out_a = adaln(x, cond_a)
    out_b = adaln(x, cond_b)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_adaln_diverges_from_layernorm_after_training_step():
    """After a backward pass that updates the modulation MLP, the
    AdaLN output should DIVERGE from plain LayerNorm. Confirms the
    parameters are reachable by gradient descent (i.e. AdaLN
    actually does something after training)."""
    embed_dim, cond_dim = 16, 8
    adaln = AdaLN(embed_dim, cond_dim)
    plain_ln = nn.LayerNorm(embed_dim, eps=1e-6)
    x = torch.randn(3, 5, embed_dim)
    cond = torch.randn(3, cond_dim)
    # Take one fake gradient step on the modulation params.
    out = adaln(x, cond)
    loss = (out ** 2).sum()
    loss.backward()
    # Use a large LR so the divergence is unmistakable.
    with torch.no_grad():
        for p in adaln.modulation.parameters():
            p.copy_(p - 0.5 * p.grad)
        out_after = adaln(x, cond)
    out_plain = plain_ln(x)
    # After update, AdaLN should NOT match plain LN anymore.
    assert not torch.allclose(out_after, out_plain, atol=1e-3)


# ---------------------------------------------------------------------------
# Block integration with AdaLN
# ---------------------------------------------------------------------------

def test_block_cond_dim_none_means_plain_layernorm():
    """Default (cond_dim=None) Block must be exactly equivalent to
    the pre-AdaLN Block. Verified by checking that `uses_adaln` is
    False and that forward doesn't require a cond argument."""
    block = Block(
        dim=32, num_heads=4, use_flash_attention=False, cond_dim=None)
    assert block.uses_adaln is False
    # Plain forward works without cond.
    x = torch.randn(2, 6, 32)
    out = block(x)
    assert out.shape == (2, 6, 32)


def test_block_cond_dim_set_uses_adaln_and_requires_cond():
    block = Block(
        dim=32, num_heads=4, use_flash_attention=False, cond_dim=8)
    assert block.uses_adaln is True
    assert isinstance(block.norm1, AdaLN)
    assert isinstance(block.norm2, AdaLN)
    x = torch.randn(2, 6, 32)
    cond = torch.randn(2, 8)
    out = block(x, cond=cond)
    assert out.shape == (2, 6, 32)
    # Forgetting cond must raise loudly rather than silently failing.
    with pytest.raises(RuntimeError, match="AdaLN"):
        block(x)


def test_block_adaln_zero_init_matches_plain_at_step_zero():
    """Block(cond_dim=8) at step 0 produces the same output as
    Block(cond_dim=None) when all other weights are shared. This is
    the structural backward-compat guarantee for AdaLN.
    """
    torch.manual_seed(0)
    dim, num_heads = 32, 4
    # Build the AdaLN Block first; copy its non-norm weights into the
    # plain Block so the only difference is the norm.
    blk_adaln = Block(
        dim=dim, num_heads=num_heads,
        use_flash_attention=False, cond_dim=8)
    blk_plain = Block(
        dim=dim, num_heads=num_heads,
        use_flash_attention=False, cond_dim=None)
    # Copy state. The norm modules are different types; copy
    # everything else.
    sd_a = blk_adaln.state_dict()
    sd_p = blk_plain.state_dict()
    for k in sd_p:
        if k in sd_a and sd_a[k].shape == sd_p[k].shape:
            sd_p[k] = sd_a[k].clone()
    blk_plain.load_state_dict(sd_p)
    blk_adaln.eval()
    blk_plain.eval()
    x = torch.randn(2, 6, dim)
    cond = torch.randn(2, 8)
    out_a = blk_adaln(x, cond=cond)
    out_p = blk_plain(x)
    torch.testing.assert_close(out_a, out_p, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Encoder backward-compat
# ---------------------------------------------------------------------------

def _make_encoder(adaln_kwargs=None):
    return GeneTransformerRankEncoder(
        vocab_size=16, seq_len=12, n_special_tokens=2, n_segments=2,
        cell_pos_enc='segment', embed_dim=32, depth=2, num_heads=4,
        use_flash_attention=False,
        adaln_kwargs=adaln_kwargs,
    )


def test_encoder_adaln_modulation_remains_zero_after_init_passes():
    """REGRESSION: the encoder's own ``self.apply(_init_weights)`` AND
    ``init_model``'s second reinit pass would overwrite AdaLN's
    zero-init with ``trunc_normal_(0.02)``. ``zero_init_adaln_modulations``
    must be called after both passes to restore the zero state, so
    that gamma=1, beta=0 at step 0.
    """
    enc = GeneTransformerRankEncoder(
        vocab_size=16, seq_len=12, n_special_tokens=2, n_segments=2,
        cell_pos_enc='segment', embed_dim=32, depth=2, num_heads=4,
        use_flash_attention=False,
        adaln_kwargs={'enabled': True, 'n_batches': 5,
                      'batch_embed_dim': 8},
    )
    from terra.models.adaln import AdaLN
    n_adaln = 0
    for m in enc.modules():
        if isinstance(m, AdaLN):
            n_adaln += 1
            torch.testing.assert_close(
                m.modulation.weight,
                torch.zeros_like(m.modulation.weight),
                atol=0, rtol=0,
                msg=lambda s: (
                    "AdaLN.modulation.weight is not zero after encoder "
                    "construction. The encoder's apply(_init_weights) "
                    "must be followed by zero_init_adaln_modulations()."
                ),
            )
            if m.modulation.bias is not None:
                torch.testing.assert_close(
                    m.modulation.bias,
                    torch.zeros_like(m.modulation.bias),
                    atol=0, rtol=0,
                )
    assert n_adaln > 0, (
        "Expected at least one AdaLN module in the encoder "
        "(depth=2 -> 4 AdaLN modules: norm1 + norm2 per block)."
    )


def test_encoder_adaln_at_step0_matches_no_adaln_encoder():
    """End-to-end: an AdaLN-enabled encoder built normally (with both
    init_weights passes applied) must produce IDENTICAL output to an
    identical encoder built with adaln_kwargs=None, when both see the
    same batch and we manually align their non-AdaLN weights. This is
    the strongest backward-compat invariant; if it holds, AdaLN at
    step 0 is mathematically indistinguishable from plain LayerNorm.
    """
    torch.manual_seed(0)
    common_kwargs = dict(
        vocab_size=16, seq_len=12, n_special_tokens=2, n_segments=2,
        cell_pos_enc='segment', embed_dim=32, depth=2, num_heads=4,
        use_flash_attention=False,
    )
    enc_adaln = GeneTransformerRankEncoder(
        **common_kwargs,
        adaln_kwargs={'enabled': True, 'n_batches': 5,
                      'batch_embed_dim': 8},
    )
    enc_plain = GeneTransformerRankEncoder(
        **common_kwargs, adaln_kwargs=None,
    )
    # Copy shared weights (everything except the AdaLN-specific
    # modules) from the AdaLN encoder into the plain encoder. We
    # iterate over the plain encoder's state_dict and pull any
    # shape-matching weights from the AdaLN encoder.
    sd_a = enc_adaln.state_dict()
    sd_p = enc_plain.state_dict()
    for k in sd_p:
        if k in sd_a and sd_a[k].shape == sd_p[k].shape:
            sd_p[k] = sd_a[k].clone()
    enc_plain.load_state_dict(sd_p)
    enc_adaln.eval()
    enc_plain.eval()

    # Run forwards. AdaLN encoder needs a values column whose first
    # entry indexes the batch_emb_table; any non-OOB value works
    # because the modulation MLP outputs zero regardless.
    B, L = 2, 12
    batch = {
        'tokens': torch.randint(1, 16, (B, L)),
        'values': torch.zeros(B, L),
        'rel_x_coords': torch.zeros(B, L),
        'rel_y_coords': torch.zeros(B, L),
        'segments': torch.ones(B, L, dtype=torch.long),
        'positions': torch.arange(L).unsqueeze(0).expand(B, L),
    }
    # Pick valid batch labels (< n_batches=5).
    batch['values'][:, 0] = torch.tensor([0.0, 3.0])
    with torch.no_grad():
        out_adaln, _ = enc_adaln(batch=batch, masks=None, masks_attention=None)
        out_plain, _ = enc_plain(batch=batch, masks=None, masks_attention=None)
    torch.testing.assert_close(out_adaln, out_plain, atol=1e-5, rtol=1e-5)


def test_encoder_default_no_adaln_attrs():
    """When adaln_kwargs is None, none of the AdaLN infrastructure
    is created. Backward-compat invariant for existing configs."""
    enc = _make_encoder(adaln_kwargs=None)
    assert enc.adaln_enabled is False
    assert enc.batch_emb_tables is None
    for blk in enc.blocks:
        assert blk.uses_adaln is False


def test_encoder_with_adaln_creates_batch_emb_tables():
    """Single-key (legacy n_batches) creates a ModuleList of one
    Embedding table."""
    enc = _make_encoder(adaln_kwargs={
        'enabled': True,
        'n_batches': 5,
        'batch_embed_dim': 16,
    })
    assert enc.adaln_enabled is True
    assert isinstance(enc.batch_emb_tables, nn.ModuleList)
    assert len(enc.batch_emb_tables) == 1
    assert isinstance(enc.batch_emb_tables[0], nn.Embedding)
    assert enc.batch_emb_tables[0].num_embeddings == 5
    assert enc.batch_emb_tables[0].embedding_dim == 16
    for blk in enc.blocks:
        assert blk.uses_adaln is True


def test_encoder_with_adaln_multi_key():
    """Multi-key adaln creates one Embedding per key."""
    enc = _make_encoder(adaln_kwargs={
        'enabled': True,
        'keys': ['batch_value', 'assay_value'],
        'n_classes': [5, 12],
        'offsets': [0, 0],
        'batch_embed_dim': 16,
    })
    assert enc.adaln_enabled is True
    assert len(enc.batch_emb_tables) == 2
    assert enc.batch_emb_tables[0].num_embeddings == 5
    assert enc.batch_emb_tables[1].num_embeddings == 12
    for emb in enc.batch_emb_tables:
        assert emb.embedding_dim == 16


def test_encoder_disabled_adaln_block_doesnt_get_cond_dim():
    enc = _make_encoder(adaln_kwargs={'enabled': False, 'n_batches': 5})
    assert enc.adaln_enabled is False
    for blk in enc.blocks:
        assert blk.uses_adaln is False


def test_encoder_compute_cond_returns_none_when_disabled():
    enc = _make_encoder(adaln_kwargs=None)
    batch = {'values': torch.zeros(2, 12)}
    assert enc._compute_cond(batch) is None


def test_encoder_compute_cond_shape_when_enabled():
    enc = _make_encoder(adaln_kwargs={
        'enabled': True, 'n_batches': 5, 'batch_embed_dim': 16,
    })
    # Legacy single-key: values[:, 0] holds the per-cell batch label.
    batch = {'values': torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
    cond = enc._compute_cond(batch)
    assert cond is not None
    assert cond.shape == (2, 16)


def test_encoder_compute_cond_raises_on_out_of_range_labels():
    """A batch label outside [0, n_classes) now raises -- silent
    clamping was the old behavior and was bug-prone."""
    enc = _make_encoder(adaln_kwargs={
        'enabled': True, 'n_batches': 3, 'batch_embed_dim': 16,
    })
    batch = {'values': torch.tensor([[99.0] + [0]*11])}
    with pytest.raises(RuntimeError, match="out of range"):
        enc._compute_cond(batch)


def test_encoder_compute_cond_sums_per_key_in_multi_key():
    """Multi-key cond = sum of per-key embedding lookups."""
    enc = _make_encoder(adaln_kwargs={
        'enabled': True,
        'keys': ['batch_value', 'assay_value'],
        'n_classes': [5, 6],
        'offsets': [0, 0],
        'batch_embed_dim': 8,
    })
    batch = {
        'values': torch.zeros(1, 12),
        'batch_value': torch.tensor([2]),
        'assay_value': torch.tensor([3]),
    }
    cond = enc._compute_cond(batch)
    expected = (enc.batch_emb_tables[0](torch.tensor([2]))
                + enc.batch_emb_tables[1](torch.tensor([3])))
    torch.testing.assert_close(cond, expected)
