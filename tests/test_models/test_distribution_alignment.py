"""Tests for CORAL and MMD distribution-alignment losses."""

import pytest
import torch

from terra.models.distribution_alignment import (
    compute_distribution_alignment_loss,
    coral_loss,
    mmd_loss,
    sinkhorn_loss,
)


# ---------------------------------------------------------------------------
# CORAL
# ---------------------------------------------------------------------------

def test_coral_zero_when_distributions_identical():
    """Two batches drawn from the same distribution should give
    near-zero CORAL (covariances match)."""
    torch.manual_seed(0)
    d = 16
    a = torch.randn(200, d)
    b = torch.randn(200, d)
    assert float(coral_loss(a, b).item()) < 1e-2


def test_coral_positive_for_different_covariances():
    """A batch with very different covariance structure should
    produce a non-trivial CORAL loss."""
    torch.manual_seed(0)
    d = 16
    a = torch.randn(200, d)
    b = torch.randn(200, d) * 5.0  # 25x variance on every dim
    assert float(coral_loss(a, b).item()) > 0.01


def test_coral_handles_degenerate_input():
    """Single-sample batch -> zero loss, no crash."""
    a = torch.randn(1, 8)
    b = torch.randn(50, 8)
    out = coral_loss(a, b)
    assert out.shape == ()
    assert float(out.item()) == 0.0


def test_coral_gradient_flows_to_input():
    """Loss must be differentiable wrt inputs."""
    a = torch.randn(50, 8, requires_grad=True)
    b = torch.randn(50, 8)
    out = coral_loss(a, b)
    out.backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

def test_mmd_zero_when_distributions_identical():
    """MMD on the SAME tensor twice is exactly zero by symmetry."""
    torch.manual_seed(0)
    a = torch.randn(200, 8)
    # MMD(a, a) is theoretically zero (V-statistic), but the empirical
    # version is exactly zero since k_aa = k_bb = k_ab.
    out = mmd_loss(a, a, sigmas=(1.0,))
    assert abs(float(out.item())) < 1e-5


def test_mmd_positive_for_shifted_distributions():
    """Mean-shift between batches must give positive MMD."""
    torch.manual_seed(0)
    a = torch.randn(200, 4)
    b = torch.randn(200, 4) + 3.0  # mean-shifted
    out = mmd_loss(a, b, sigmas=(1.0, 3.0))
    assert float(out.item()) > 0.01


def test_mmd_handles_degenerate_input():
    a = torch.randn(1, 4)
    b = torch.randn(50, 4)
    out = mmd_loss(a, b)
    assert out.shape == ()
    assert float(out.item()) == 0.0


def test_mmd_gradient_flows_to_input():
    a = torch.randn(50, 4, requires_grad=True)
    b = torch.randn(50, 4)
    out = mmd_loss(a, b, sigmas=(1.0,))
    out.backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()


# ---------------------------------------------------------------------------
# Sinkhorn divergence
# ---------------------------------------------------------------------------

def test_sinkhorn_zero_when_distributions_identical():
    """Sinkhorn divergence on the SAME tensor twice is exactly zero
    (debiasing terms cancel)."""
    torch.manual_seed(0)
    a = torch.randn(100, 8)
    out = sinkhorn_loss(a, a, epsilon=0.1, n_iter=50)
    assert abs(float(out.item())) < 1e-4


def test_sinkhorn_positive_for_shifted_distributions():
    """Mean-shifted batches must give positive Sinkhorn divergence."""
    torch.manual_seed(0)
    a = torch.randn(100, 4)
    b = torch.randn(100, 4) + 3.0
    out = sinkhorn_loss(a, b, epsilon=0.1, n_iter=100)
    assert float(out.item()) > 0.01


def test_sinkhorn_handles_degenerate_input():
    a = torch.randn(1, 4)
    b = torch.randn(50, 4)
    out = sinkhorn_loss(a, b)
    assert out.shape == ()
    assert float(out.item()) == 0.0


def test_sinkhorn_gradient_flows_to_input():
    a = torch.randn(30, 4, requires_grad=True)
    b = torch.randn(30, 4)
    out = sinkhorn_loss(a, b, epsilon=0.1, n_iter=50)
    out.backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()


def test_sinkhorn_nonnegative_after_clamp():
    """Numerical residuals must not produce negative loss."""
    torch.manual_seed(0)
    a = torch.randn(40, 8)
    b = a.clone() + 1e-6 * torch.randn(40, 8)
    out = sinkhorn_loss(a, b, epsilon=0.05, n_iter=50)
    assert float(out.item()) >= 0.0


# ---------------------------------------------------------------------------
# compute_distribution_alignment_loss (the dispatch / pair-iteration)
# ---------------------------------------------------------------------------

def test_dispatch_returns_zero_when_only_one_batch_present():
    """No pairs to align -> exactly zero, no nan, no crash."""
    cell_emb = torch.randn(8, 4)
    batch_label = torch.zeros(8, dtype=torch.long)  # all same batch
    loss, info = compute_distribution_alignment_loss(
        cell_emb, batch_label, method='coral')
    assert float(loss.item()) == 0.0
    assert info['n_batches_in_minibatch'] == 1
    assert info['n_pairs'] == 0


def test_dispatch_runs_pairs_for_multiple_batches():
    """With 3 distinct batches in the minibatch, expect C(3, 2)=3 pairs."""
    torch.manual_seed(0)
    cell_emb = torch.randn(60, 4)
    batch_label = torch.cat([
        torch.zeros(20, dtype=torch.long),
        torch.ones(20, dtype=torch.long),
        torch.full((20,), 2, dtype=torch.long),
    ])
    loss, info = compute_distribution_alignment_loss(
        cell_emb, batch_label, method='coral')
    assert info['n_batches_in_minibatch'] == 3
    assert info['n_pairs'] == 3
    assert float(loss.item()) >= 0.0


def test_dispatch_max_pairs_caps_pair_count():
    torch.manual_seed(0)
    n_per_batch = 10
    n_batches = 6
    cell_emb = torch.randn(n_per_batch * n_batches, 4)
    batch_label = torch.repeat_interleave(
        torch.arange(n_batches), n_per_batch)
    loss, info = compute_distribution_alignment_loss(
        cell_emb, batch_label, method='coral', max_pairs=4)
    # C(6, 2) = 15, but max_pairs caps to 4
    assert info['n_pairs'] == 4


def test_dispatch_unknown_method_raises():
    cell_emb = torch.randn(20, 4)
    batch_label = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(10, dtype=torch.long),
    ])
    with pytest.raises(ValueError, match="Unknown distribution"):
        compute_distribution_alignment_loss(
            cell_emb, batch_label, method='bogus')


def test_dispatch_mmd_runs_end_to_end():
    """MMD path executes without crash and produces finite loss."""
    torch.manual_seed(0)
    n_per_batch = 30
    cell_emb = torch.cat([
        torch.randn(n_per_batch, 4),
        torch.randn(n_per_batch, 4) + 1.0,  # shifted to give nonzero loss
    ])
    batch_label = torch.cat([
        torch.zeros(n_per_batch, dtype=torch.long),
        torch.ones(n_per_batch, dtype=torch.long),
    ])
    loss, info = compute_distribution_alignment_loss(
        cell_emb, batch_label, method='mmd',
        mmd_sigmas=[0.5, 1.0, 2.0])
    assert info['n_pairs'] == 1
    assert torch.isfinite(loss).item()
    assert float(loss.item()) > 0.0


def test_dispatch_sinkhorn_runs_end_to_end():
    """Sinkhorn path executes without crash and produces finite loss."""
    torch.manual_seed(0)
    n_per_batch = 30
    cell_emb = torch.cat([
        torch.randn(n_per_batch, 4),
        torch.randn(n_per_batch, 4) + 1.0,
    ])
    batch_label = torch.cat([
        torch.zeros(n_per_batch, dtype=torch.long),
        torch.ones(n_per_batch, dtype=torch.long),
    ])
    loss, info = compute_distribution_alignment_loss(
        cell_emb, batch_label, method='sinkhorn',
        sinkhorn_eps=0.1, sinkhorn_n_iter=50)
    assert info['n_pairs'] == 1
    assert torch.isfinite(loss).item()
    assert float(loss.item()) > 0.0


def test_dispatch_gradient_flows():
    """End-to-end gradient flow through the dispatch."""
    torch.manual_seed(0)
    cell_emb = torch.randn(60, 4, requires_grad=True)
    batch_label = torch.cat([
        torch.zeros(20, dtype=torch.long),
        torch.ones(20, dtype=torch.long),
        torch.full((20,), 2, dtype=torch.long),
    ])
    loss, _ = compute_distribution_alignment_loss(
        cell_emb, batch_label, method='coral')
    loss.backward()
    assert cell_emb.grad is not None
    assert torch.isfinite(cell_emb.grad).all()
