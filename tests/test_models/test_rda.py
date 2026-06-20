"""Tests for Read-Depth-Aware (RDA) depth conditioning."""

import pytest
import torch

from terra.models.rda import (
    DepthConditioning,
    build_depth_embedding,
    compute_per_cell_depth,
)


# ---------------------------------------------------------------------------
# DepthConditioning module
# ---------------------------------------------------------------------------

def test_depth_conditioning_zero_init_returns_zero():
    """At construction the output head is zero-init, so the depth
    embedding is exactly zero. RDA-at-step-0 == no-RDA."""
    torch.manual_seed(0)
    dc = DepthConditioning(embed_dim=16, hidden_dim=8)
    depth = torch.tensor([[100.0, 200.0, 50.0]])  # (1, 3)
    out = dc(depth)
    assert out.shape == (1, 3, 16)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_depth_conditioning_diverges_after_training_step():
    """After a single optimizer step, the depth embedding is no
    longer zero. Confirms the params are reachable by gradient
    descent (not pruned by some accidental no_grad)."""
    torch.manual_seed(0)
    dc = DepthConditioning(embed_dim=8, hidden_dim=4)
    depth = torch.tensor([[100.0, 200.0]])
    target = torch.randn(1, 2, 8)
    optim = torch.optim.SGD(dc.parameters(), lr=1e-2)
    for _ in range(3):
        optim.zero_grad()
        out = dc(depth)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optim.step()
    out_after = dc(depth)
    assert not torch.allclose(out_after, torch.zeros_like(out_after))


def test_depth_conditioning_with_target_depth():
    """use_target_depth=True takes a (B, c) target depth and yields
    a 2-scalar input to the MLP."""
    torch.manual_seed(0)
    dc = DepthConditioning(embed_dim=8, hidden_dim=4, use_target_depth=True)
    depth = torch.tensor([[100.0, 200.0]])
    tgt = torch.tensor([[500.0, 500.0]])
    out = dc(depth, target_depth=tgt)
    assert out.shape == (1, 2, 8)
    # zero-init -> exact zero
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_depth_conditioning_requires_target_when_configured():
    dc = DepthConditioning(embed_dim=4, hidden_dim=4, use_target_depth=True)
    depth = torch.tensor([[10.0]])
    with pytest.raises(ValueError, match="use_target_depth"):
        dc(depth)


# ---------------------------------------------------------------------------
# compute_per_cell_depth
# ---------------------------------------------------------------------------

def test_compute_per_cell_depth_sums_per_cell_block():
    n_special = 2
    n_cells = 3
    seq_len_cell = 4
    # Special-token positions contain anything (should be ignored).
    # Gene tokens: cell 0 = [1, 2, 3, 4]=10; cell 1 = [5, 0, 0, 0]=5;
    # cell 2 = [0, 0, 0, 0]=0
    values = torch.tensor([[
        99.0, 88.0,    # special tokens
        1.0, 2.0, 3.0, 4.0,
        5.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ]])
    depth = compute_per_cell_depth(
        values, n_special_tokens=n_special,
        n_cells=n_cells, seq_len_cell=seq_len_cell)
    assert depth.shape == (1, 3)
    assert depth[0, 0].item() == 10.0
    assert depth[0, 1].item() == 5.0
    assert depth[0, 2].item() == 0.0


def test_compute_per_cell_depth_clamps_negatives():
    """Negative sentinels (-inf or -1 padding) must not contribute
    to the depth sum."""
    values = torch.tensor([[
        0.0, 0.0,
        1.0, -float('inf'), 2.0, -1.0,
    ]])
    depth = compute_per_cell_depth(
        values, n_special_tokens=2, n_cells=1, seq_len_cell=4)
    assert depth[0, 0].item() == 3.0


def test_compute_per_cell_depth_raises_on_short_input():
    values = torch.zeros(1, 5)
    with pytest.raises(RuntimeError, match="values length"):
        compute_per_cell_depth(
            values, n_special_tokens=2, n_cells=3, seq_len_cell=4)


# ---------------------------------------------------------------------------
# build_depth_embedding
# ---------------------------------------------------------------------------

def test_build_depth_embedding_broadcast_and_zero_special():
    n_special = 2
    n_cells = 2
    seq_len_cell = 3
    seq_len = n_special + n_cells * seq_len_cell  # 8
    embed_dim = 5

    torch.manual_seed(0)
    dc = DepthConditioning(embed_dim=embed_dim, hidden_dim=4)
    # Force the output to be non-zero so we can detect broadcast.
    with torch.no_grad():
        dc.mlp[-1].weight.fill_(0.1)
        dc.mlp[-1].bias.fill_(0.5)

    values = torch.tensor([[
        0.0, 0.0,
        1.0, 2.0, 3.0,        # cell 0 depth = 6
        4.0, 5.0, 6.0,        # cell 1 depth = 15
    ]])
    batch = {'values': values}
    out = build_depth_embedding(
        dc, batch,
        n_special_tokens=n_special, n_cells=n_cells,
        seq_len_cell=seq_len_cell, seq_len=seq_len)
    assert out.shape == (1, seq_len, embed_dim)
    # Special positions are exactly zero.
    assert torch.allclose(out[:, :n_special, :], torch.zeros(1, n_special, embed_dim))
    # Within each cell block, all positions share the same embedding.
    cell0 = out[:, n_special:n_special + seq_len_cell, :]
    cell1 = out[:, n_special + seq_len_cell:, :]
    assert torch.allclose(cell0[:, 0, :], cell0[:, 1, :])
    assert torch.allclose(cell0[:, 0, :], cell0[:, 2, :])
    assert torch.allclose(cell1[:, 0, :], cell1[:, 1, :])
    # Different cells have different embeddings (since depths differ).
    assert not torch.allclose(cell0[:, 0, :], cell1[:, 0, :])


def test_build_depth_embedding_is_zero_at_init():
    """End-to-end zero-init check: with a fresh DepthConditioning the
    depth embedding added by build_depth_embedding is exactly zero,
    so an encoder with RDA enabled at step 0 yields identical
    output to one with RDA disabled."""
    torch.manual_seed(0)
    n_special, n_cells, seq_len_cell = 2, 3, 4
    seq_len = n_special + n_cells * seq_len_cell
    dc = DepthConditioning(embed_dim=8, hidden_dim=4)
    values = torch.randint(0, 100, (2, seq_len)).float()
    batch = {'values': values}
    out = build_depth_embedding(
        dc, batch,
        n_special_tokens=n_special, n_cells=n_cells,
        seq_len_cell=seq_len_cell, seq_len=seq_len)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
