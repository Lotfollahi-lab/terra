"""Tests for batch-swap cycle-consistency loss."""

import pytest
import torch

from terra.models.cycle_consistency import (
    cycle_consistency_loss,
    make_swapped_batch,
)


# ---------------------------------------------------------------------------
# make_swapped_batch
# ---------------------------------------------------------------------------

def test_make_swapped_batch_legacy_path_replaces_values_first_column():
    """When no key is given, falls back to swapping values[:, 0]."""
    torch.manual_seed(0)
    values = torch.zeros(8, 4)
    values[:, 0] = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    batch = {'values': values}
    swapped, changed = make_swapped_batch(batch, n_classes_per_key=8)

    # Other columns must be untouched.
    assert torch.allclose(swapped['values'][:, 1:], values[:, 1:])
    # First column must be in [0, 8).
    new_labels = swapped['values'][:, 0].long()
    assert (new_labels >= 0).all()
    assert (new_labels < 8).all()
    # The original batch must not have been mutated.
    assert torch.equal(batch['values'][:, 0],
                       torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]).float())
    assert changed.shape == (8,)
    assert changed.dtype == torch.bool


def test_make_swapped_batch_changed_mask_matches_diff():
    torch.manual_seed(123)
    values = torch.zeros(50, 2)
    values[:, 0] = torch.randint(0, 5, (50,))
    batch = {'values': values}
    swapped, changed = make_swapped_batch(batch, n_classes_per_key=5)
    expected = swapped['values'][:, 0].long() != values[:, 0].long()
    assert torch.equal(changed, expected)


def test_make_swapped_batch_respects_n_classes():
    torch.manual_seed(0)
    values = torch.zeros(100, 2)
    batch = {'values': values}
    swapped, _ = make_swapped_batch(batch, n_classes_per_key=3)
    new_labels = swapped['values'][:, 0].long()
    assert int(new_labels.max().item()) < 3
    assert int(new_labels.min().item()) >= 0


def test_make_swapped_batch_legacy_raises_on_1d_values():
    with pytest.raises(RuntimeError, match="at least 2-D"):
        make_swapped_batch(
            {'values': torch.zeros(8)}, n_classes_per_key=4)


def test_make_swapped_batch_metadata_key_path():
    """When a key is given AND present in the batch, the metadata
    field is swapped instead of values[:, 0]."""
    torch.manual_seed(0)
    batch = {
        'values': torch.zeros(8, 4),
        'batch_value': torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
    }
    swapped, changed = make_swapped_batch(
        batch, n_classes_per_key=8, keys='batch_value')
    # values untouched.
    assert torch.allclose(swapped['values'], batch['values'])
    # batch_value replaced and in range.
    assert (swapped['batch_value'] >= 0).all()
    assert (swapped['batch_value'] < 8).all()
    assert changed.shape == (8,)


def test_make_swapped_batch_multi_key_or_mask():
    """changed_mask is True if ANY key swapped for that cell."""
    torch.manual_seed(0)
    batch = {
        'values': torch.zeros(8, 4),
        'batch_value': torch.randint(0, 5, (8,)),
        'assay_value': torch.randint(0, 7, (8,)),
    }
    swapped, changed = make_swapped_batch(
        batch,
        n_classes_per_key=[5, 7],
        keys=['batch_value', 'assay_value'],
    )
    # Compute expected: True where either key changed.
    expected = (
        (swapped['batch_value'] != batch['batch_value'])
        | (swapped['assay_value'] != batch['assay_value'])
    )
    assert torch.equal(changed, expected)


def test_make_swapped_batch_mismatched_lengths_raises():
    batch = {'values': torch.zeros(4, 3),
             'batch_value': torch.zeros(4, dtype=torch.long)}
    with pytest.raises(RuntimeError, match="length"):
        make_swapped_batch(
            batch,
            n_classes_per_key=[5, 7],
            keys=['batch_value'],   # length mismatch
        )


def test_make_swapped_batch_multiple_legacy_slots_raises():
    """Two slots both falling back to the legacy values[:, 0] path
    would overwrite each other -- must raise."""
    batch = {'values': torch.zeros(4, 3)}
    with pytest.raises(RuntimeError, match="legacy values"):
        make_swapped_batch(
            batch,
            n_classes_per_key=[5, 7],
            keys=[None, None],   # both legacy
        )


def test_make_swapped_batch_duplicate_metadata_keys_raises():
    """Two slots referencing the same metadata key would overwrite
    each other in swapped_batch[key] -- must raise."""
    batch = {
        'values': torch.zeros(4, 3),
        'batch_value': torch.zeros(4, dtype=torch.long),
    }
    with pytest.raises(RuntimeError, match="[Dd]uplicate"):
        make_swapped_batch(
            batch,
            n_classes_per_key=[5, 7],
            keys=['batch_value', 'batch_value'],
        )


def test_make_swapped_batch_one_legacy_one_metadata_ok():
    """Mixed mode is fine -- only multiple legacy slots are bad."""
    torch.manual_seed(0)
    batch = {
        'values': torch.zeros(4, 3),
        'assay_value': torch.zeros(4, dtype=torch.long),
    }
    # First slot uses legacy values[:, 0], second uses metadata.
    swapped, changed = make_swapped_batch(
        batch,
        n_classes_per_key=[5, 7],
        keys=[None, 'assay_value'],
    )
    assert swapped is not None
    assert changed.shape == (4,)


# ---------------------------------------------------------------------------
# cycle_consistency_loss
# ---------------------------------------------------------------------------

def test_cycle_loss_zero_when_outputs_identical():
    z = torch.randn(4, 10, 8)
    out = cycle_consistency_loss(z, z.clone())
    assert float(out.item()) == 0.0


def test_cycle_loss_positive_when_outputs_differ():
    torch.manual_seed(0)
    z_a = torch.randn(4, 10, 8)
    z_b = z_a + 1.0
    out = cycle_consistency_loss(z_a, z_b)
    assert float(out.item()) > 0.0


def test_cycle_loss_changed_mask_filters_unchanged():
    """Cells outside changed_mask must not contribute."""
    torch.manual_seed(0)
    z_a = torch.randn(4, 10, 8)
    z_b = z_a.clone()
    z_b[2] += 1.0  # cell 2 differs
    # Only cell 2 is "changed"
    mask = torch.tensor([False, False, True, False])
    out = cycle_consistency_loss(z_a, z_b, changed_mask=mask)
    # Should reduce to mean over cell 2 only -> 1.0 (sq diff is 1.0)
    assert pytest.approx(float(out.item()), rel=1e-5) == 1.0


def test_cycle_loss_changed_mask_all_false_returns_zero():
    z_a = torch.randn(4, 10, 8)
    z_b = torch.randn(4, 10, 8)
    mask = torch.zeros(4, dtype=torch.bool)
    out = cycle_consistency_loss(z_a, z_b, changed_mask=mask)
    assert float(out.item()) == 0.0


def test_cycle_loss_accepts_list_of_tensors():
    """JEPA multi-mask layout: outputs are a list of (B, L, D)."""
    z_a = [torch.randn(4, 10, 8), torch.randn(4, 12, 8)]
    z_b = [z_a[0].clone(), z_a[1].clone()]
    out = cycle_consistency_loss(z_a, z_b)
    assert float(out.item()) == 0.0


def test_cycle_loss_gradient_flows():
    torch.manual_seed(0)
    z_a = torch.randn(4, 10, 8, requires_grad=True)
    z_b = torch.randn(4, 10, 8)
    out = cycle_consistency_loss(z_a, z_b)
    out.backward()
    assert z_a.grad is not None
    assert torch.isfinite(z_a.grad).all()
