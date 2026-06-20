"""Tests for the multi-slot Special-Token-MoE predictor bias head."""

import pytest
import torch

from terra.models.special_token_moe import (
    SpecialTokenMoE,
    _as_list,
    extract_special_token_indices,
)


# ---------------------------------------------------------------------------
# _as_list helper
# ---------------------------------------------------------------------------

def test_as_list_scalar_wraps_to_list():
    assert _as_list(5) == [5]
    assert _as_list(0) == [0]


def test_as_list_passes_lists_through():
    assert _as_list([1, 2, 3]) == [1, 2, 3]
    assert _as_list((1, 2)) == [1, 2]


def test_as_list_none_returns_empty():
    assert _as_list(None) == []


def test_as_list_none_with_fallback_len():
    assert _as_list(None, fallback_len=3) == [0, 0, 0]


# ---------------------------------------------------------------------------
# SpecialTokenMoE — single slot (legacy behavior)
# ---------------------------------------------------------------------------

def test_single_slot_zero_init_returns_zero():
    """One routing slot, zero-init -> output is exactly zero."""
    torch.manual_seed(0)
    moe = SpecialTokenMoE(n_experts_per_slot=[5], embed_dim=8)
    indices = [torch.tensor([0, 1, 2, 3, 4])]
    out = moe(indices)
    assert out.shape == (5, 8)
    assert torch.allclose(out, torch.zeros_like(out))


def test_single_slot_diverges_after_training():
    torch.manual_seed(0)
    moe = SpecialTokenMoE(n_experts_per_slot=[3], embed_dim=4)
    indices = [torch.tensor([0, 1, 2])]
    target = torch.randn(3, 4)
    optim = torch.optim.SGD(moe.parameters(), lr=1e-1)
    for _ in range(5):
        optim.zero_grad()
        out = moe(indices)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optim.step()
    out_after = moe(indices)
    assert not torch.allclose(out_after, torch.zeros_like(out_after))


def test_single_slot_oob_raises():
    moe = SpecialTokenMoE(n_experts_per_slot=[4], embed_dim=4)
    with pytest.raises(RuntimeError, match="out of range"):
        moe([torch.tensor([0, 1, 4])])     # 4 is OOB
    with pytest.raises(RuntimeError, match="out of range"):
        moe([torch.tensor([-1, 0])])


# ---------------------------------------------------------------------------
# SpecialTokenMoE — multi-slot
# ---------------------------------------------------------------------------

def test_multi_slot_zero_init_returns_zero():
    """Two routing slots, both zero-init -> output is zero."""
    torch.manual_seed(0)
    moe = SpecialTokenMoE(n_experts_per_slot=[10, 145], embed_dim=8)
    indices = [
        torch.tensor([0, 1, 2, 3]),      # batch slot
        torch.tensor([0, 50, 100, 144]), # assay slot
    ]
    out = moe(indices)
    assert out.shape == (4, 8)
    assert torch.allclose(out, torch.zeros_like(out))


def test_multi_slot_sums_per_slot_contributions():
    """With non-zero biases, output == sum of per-slot lookups."""
    moe = SpecialTokenMoE(n_experts_per_slot=[3, 4], embed_dim=2)
    with torch.no_grad():
        # Slot 0: row i -> [i, i]
        moe.biases[0].weight.copy_(torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
        # Slot 1: row j -> [10*j, 10*j]
        moe.biases[1].weight.copy_(torch.tensor(
            [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]))
    indices = [
        torch.tensor([1, 2, 0]),
        torch.tensor([3, 2, 1]),
    ]
    out = moe(indices)
    expected = torch.tensor([
        [1 + 30, 1 + 30],
        [2 + 20, 2 + 20],
        [0 + 10, 0 + 10],
    ], dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_multi_slot_gradient_flows_to_all_slots():
    """Both expert tables get gradients."""
    torch.manual_seed(0)
    moe = SpecialTokenMoE(n_experts_per_slot=[3, 4], embed_dim=2)
    indices = [
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2]),
    ]
    out = moe(indices).sum()
    out.backward()
    assert moe.biases[0].weight.grad is not None
    assert moe.biases[1].weight.grad is not None
    assert torch.isfinite(moe.biases[0].weight.grad).all()
    assert torch.isfinite(moe.biases[1].weight.grad).all()


def test_multi_slot_oob_in_second_slot_raises_with_slot_index():
    moe = SpecialTokenMoE(n_experts_per_slot=[5, 10], embed_dim=4)
    with pytest.raises(RuntimeError, match=r"slot 1.*out of range"):
        moe([torch.tensor([0, 1]), torch.tensor([0, 10])])  # 10 OOB in slot 1


def test_wrong_number_of_index_tensors_raises():
    moe = SpecialTokenMoE(n_experts_per_slot=[3, 4], embed_dim=2)
    with pytest.raises(RuntimeError, match="expected 2 routing slots"):
        moe([torch.tensor([0, 1])])    # only one tensor for two slots


def test_no_weight_decay_flag_set_on_all_slots():
    moe = SpecialTokenMoE(n_experts_per_slot=[3, 4], embed_dim=4)
    for emb in moe.biases:
        for p in emb.parameters():
            assert getattr(p, "_no_weight_decay", False) is True


def test_empty_n_experts_list_raises():
    with pytest.raises(ValueError, match="at least one routing slot"):
        SpecialTokenMoE(n_experts_per_slot=[], embed_dim=4)


def test_non_positive_n_experts_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        SpecialTokenMoE(n_experts_per_slot=[5, 0], embed_dim=4)


# ---------------------------------------------------------------------------
# extract_special_token_indices
# ---------------------------------------------------------------------------

def test_extract_indices_single_slot():
    values = torch.tensor([
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
    ])
    out = extract_special_token_indices(
        {'values': values}, routing_indices=[1])
    assert len(out) == 1
    assert out[0].tolist() == [20, 21]
    assert out[0].dtype == torch.long


def test_extract_indices_multi_slot_with_offsets():
    values = torch.tensor([
        [3.0, 10.0, 50.0],
        [3.0, 11.0, 51.0],
    ])
    # Subtract offset 3 from slot 0 (-> [0, 0]), offset 10 from slot 1
    # (-> [0, 1]), offset 50 from slot 2 (-> [0, 1]).
    out = extract_special_token_indices(
        {'values': values},
        routing_indices=[0, 1, 2],
        routing_offsets=[3, 10, 50],
    )
    assert len(out) == 3
    assert out[0].tolist() == [0, 0]
    assert out[1].tolist() == [0, 1]
    assert out[2].tolist() == [0, 1]


def test_extract_indices_default_offsets_are_zero():
    values = torch.tensor([[5.0, 6.0]])
    out = extract_special_token_indices(
        {'values': values}, routing_indices=[0, 1])
    assert out[0].tolist() == [5]
    assert out[1].tolist() == [6]


def test_extract_indices_mismatched_lengths_raises():
    values = torch.tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(RuntimeError, match="does not match"):
        extract_special_token_indices(
            {'values': values},
            routing_indices=[0, 1],
            routing_offsets=[0],   # only one offset for two indices
        )


def test_extract_indices_oob_routing_index_raises():
    values = torch.zeros(2, 3)
    with pytest.raises(RuntimeError, match="out of bounds"):
        extract_special_token_indices(
            {'values': values}, routing_indices=[5])


def test_extract_indices_requires_2d_values():
    with pytest.raises(RuntimeError, match="at least 2-D"):
        extract_special_token_indices(
            {'values': torch.zeros(4)}, routing_indices=[0])


# ---------------------------------------------------------------------------
# extract_special_token_indices — metadata-key mode
# ---------------------------------------------------------------------------

def test_extract_indices_routing_keys_metadata_path():
    """Metadata-key mode reads from batch[key] instead of values[:,k]."""
    batch = {
        'values': torch.zeros(3, 5),
        'batch_value': torch.tensor([2, 5, 8]),
        'assay_value': torch.tensor([12, 14, 18]),
    }
    out = extract_special_token_indices(
        batch,
        routing_keys=['batch_value', 'assay_value'],
        routing_offsets=[2, 12],
    )
    assert len(out) == 2
    assert out[0].tolist() == [0, 3, 6]    # batch_value - 2
    assert out[1].tolist() == [0, 2, 6]    # assay_value - 12


def test_extract_indices_routing_keys_falls_back_to_indices():
    """If a routing_keys entry is None or absent from batch, falls
    back to the corresponding routing_indices entry."""
    batch = {
        'values': torch.tensor([[10.0, 20.0, 30.0]]),
        'batch_value': torch.tensor([5]),
    }
    out = extract_special_token_indices(
        batch,
        routing_keys=['batch_value', 'nonexistent_key'],
        routing_indices=[None, 2],
        routing_offsets=[0, 0],
    )
    assert out[0].tolist() == [5]    # from batch_value
    assert out[1].tolist() == [30]   # from values[:, 2]


def test_extract_indices_neither_set_raises():
    with pytest.raises(RuntimeError, match="must provide"):
        extract_special_token_indices({'values': torch.zeros(2, 3)})
