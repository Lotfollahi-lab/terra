"""Tests for batch-swap cycle-consistency loss."""

import pytest
import torch

from terra.models.cycle_consistency import (
    check_swapped_label_range,
    cycle_consistency_loss,
    make_swapped_batch,
    special_token_positions,
)
from terra.utils.helper import init_model


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
# Metadata keys reach the encoder through the special tokens
# ---------------------------------------------------------------------------
# The encoder reads a cell's label from its special token (metatoken), not
# from the metadata field (only AdaLN reads that). Swapping the metadata field
# alone left the encoder input unchanged, so the cycle loss was always zero.

def test_special_token_positions_maps_keys_to_their_special_token():
    # The dataset prepends the special tokens one by one: the sequence
    # starts with [batch, assay, cls_0]
    assert special_token_positions(
        ['batch_value', 'assay_value', None, 'tissue_value'],
        ['cls_0', 'assay', 'batch']) == [0, 1, None, None]


@pytest.mark.parametrize('gt_type', ['combined', 'rank'])
def test_special_token_positions_match_the_dataset_layout(gt_type):
    from datasets import Dataset

    from terra.datasets.cell_datasets import init_cell_dataset
    n_cells, seq_len_cell, n_segments = 4, 4, 2
    rows = {'gene_tokens': [list(range(107, 115))] * n_cells,
            'gene_expr': [[float(v) for v in range(8, 0, -1)]] * n_cells,
            'n_nonzero_tokens': [8] * n_cells,
            'batch_value': [[3 + cell] for cell in range(n_cells)],
            'assay_value': [[11]] * n_cells}
    dataset = Dataset.from_dict(rows)
    dataset.set_format(type='torch', columns=list(rows),
                       output_all_columns=True)
    special_tokens = ['batch', 'assay']
    cell_dataset = init_cell_dataset(
        tokenizer_type='cell_graph', dataset=dataset, vocab_size=23407,
        seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_cell * (n_segments - 1),
        gt_type=gt_type, cell_pos_enc='segment',
        special_tokens=special_tokens, sampling_strategy=None,
        n_nonzero_tokens_list=None, include_cell_id=False,
        sep_gene_tokens_neb=False, nz_spc=True)
    batch = torch.utils.data.default_collate(
        [cell_dataset[i] for i in range(n_cells)])

    keys = ['batch_value', 'assay_value']
    positions = special_token_positions(keys, special_tokens)
    assert positions == [1, 0]
    for key, position in zip(keys, positions):
        # Each special token carries the label of its metadata field
        if gt_type == 'rank':
            label = batch['tokens'][:, position] - 21957
        else:
            label = batch['values'][:, position].long()
        assert torch.equal(label, batch[key])

    # A swapped batch label lands in the batch token only
    torch.manual_seed(0)
    swapped, changed = make_swapped_batch(
        batch, n_classes_per_key=[16], keys=['batch_value'],
        sequence_positions=positions[:1], label_in_tokens=gt_type == 'rank')
    assert changed.any()
    field = 'tokens' if gt_type == 'rank' else 'values'
    offset = 21957 if gt_type == 'rank' else 0
    assert torch.equal(swapped[field][:, 1].long() - offset,
                       swapped['batch_value'])
    assert torch.equal(swapped[field][:, 0], batch[field][:, 0])


def _sequence_batch(n_cells=6, label=3):
    """Combined-model batch with one `batch` special token (nz_spc)."""
    torch.manual_seed(0)
    genes = torch.randint(10, 100, (n_cells, 8))
    tokens = torch.cat([torch.full((n_cells, 1), 106), genes], dim=1)
    values = torch.cat([torch.full((n_cells, 1), float(label)),
                        torch.randint(1, 20, (n_cells, 8)).float()], dim=1)
    segments = torch.cat([torch.ones(n_cells, 1, dtype=torch.long),
                          torch.full((n_cells, 4), 2),
                          torch.full((n_cells, 4), 3)], dim=1)
    return {'tokens': tokens, 'values': values, 'segments': segments,
            'positions': torch.arange(1, 10).repeat(n_cells, 1),
            'batch_value': torch.full((n_cells,), label)}


def test_make_swapped_batch_writes_metadata_label_into_special_token():
    batch = _sequence_batch()
    original = {k: v.clone() for k, v in batch.items()}
    swapped, changed = make_swapped_batch(
        batch, n_classes_per_key=[16], keys=['batch_value'],
        sequence_positions=[0])
    torch.testing.assert_close(swapped['values'][:, 0],
                               swapped['batch_value'].float())
    torch.testing.assert_close(swapped['values'][:, 1:],
                               batch['values'][:, 1:])
    assert torch.equal(swapped['tokens'], batch['tokens'])
    assert torch.equal(changed, swapped['batch_value'] != 3)
    for key, value in original.items():  # input not mutated
        assert torch.equal(batch[key], value)


def test_make_swapped_batch_writes_rank_label_into_token_id():
    batch = _sequence_batch()
    batch['tokens'][:, 0] = 3 + 21957  # rank models: label + offset
    swapped, _ = make_swapped_batch(
        batch, n_classes_per_key=[16], keys=['batch_value'],
        sequence_positions=[0], label_in_tokens=True)
    torch.testing.assert_close(swapped['tokens'][:, 0],
                               swapped['batch_value'] + 21957)
    assert torch.equal(swapped['values'], batch['values'])


def test_make_swapped_batch_keeps_padded_special_tokens_padded():
    batch = _sequence_batch()
    batch['tokens'][:3, 0] = 0
    batch['values'][:3, 0] = 0.0
    swapped, _ = make_swapped_batch(
        batch, n_classes_per_key=[16], keys=['batch_value'],
        sequence_positions=[0])
    assert (swapped['values'][:3, 0] == 0).all()
    torch.testing.assert_close(swapped['values'][3:, 0],
                               swapped['batch_value'][3:].float())


def test_cells_with_padded_special_tokens_do_not_count_as_changed():
    # The encoder sees the label only through the special token, which a
    # padded cell does not have
    batch = _sequence_batch()
    batch['tokens'][:3, 0] = 0
    batch['values'][:3, 0] = 0.0
    swapped, changed = make_swapped_batch(
        batch, n_classes_per_key=[16], keys=['batch_value'],
        sequence_positions=[0])
    assert not changed[:3].any()
    assert torch.equal(changed[3:], swapped['batch_value'][3:] != 3)
    # ... unless the encoder also reads the key directly (AdaLN)
    _, changed = make_swapped_batch(
        batch, n_classes_per_key=[16], keys=['batch_value'],
        sequence_positions=[0], keys_read_by_encoder=['batch_value'])
    assert changed[:3].any()


def test_batch_with_padded_special_tokens_has_no_changed_cell():
    torch.manual_seed(0)
    encoder, _ = init_model(
        gt_type='combined', count_encoding='value_bins', n_value_bins=20,
        cell_pos_enc='segment', device='cpu', vocab_size=128, seq_len=9,
        n_special_tokens=1, n_segments=2, n_special_values=16,
        enc_emb_dim=16, enc_depth=1, pred_emb_dim=8, pred_depth=1,
        num_heads=2, nz_spc=True)
    batch = _sequence_batch()
    batch['tokens'][:, 0] = 0
    batch['values'][:, 0] = 0.0
    swapped, changed = make_swapped_batch(
        batch, n_classes_per_key=[16], keys=['batch_value'],
        sequence_positions=[0])
    with torch.no_grad():
        z, _ = encoder(batch=batch)
        z_swapped, _ = encoder(batch=swapped)
    assert (swapped['batch_value'] != 3).any()
    # The swap does not change the encoder input
    assert torch.equal(z, z_swapped)
    assert not changed.any()


def test_check_swapped_label_range():
    kwargs = dict(keys=['batch_value', 'assay_value'], offsets=[2, 0],
                  sequence_positions=[0, None], label_description='rows')
    # Labels 2-17 of the first key; the second key is not in the sequence
    check_swapped_label_range(n_classes_per_key=[16, 100], n_labels=18,
                              **kwargs)
    with pytest.raises(ValueError, match="'batch_value'.*reach 18"):
        check_swapped_label_range(n_classes_per_key=[17, 1], n_labels=18,
                                  **kwargs)


def test_rank_special_value_offset_matches_the_token_dictionaries():
    from terra.datasets.cell_datasets import rank_special_value_offset
    assert rank_special_value_offset(2785) == 1674  # mus musculus
    assert rank_special_value_offset(23407) == 21957  # homo sapiens


def test_make_swapped_batch_rejects_two_writes_to_one_special_token():
    batch = _sequence_batch()
    with pytest.raises(RuntimeError, match="same special token"):
        make_swapped_batch(
            batch, n_classes_per_key=[16, 16], keys=[None, 'batch_value'],
            sequence_positions=[None, 0])


def test_metadata_key_swap_changes_encoder_output():
    torch.manual_seed(0)
    encoder, _ = init_model(
        gt_type='combined', count_encoding='value_bins', n_value_bins=20,
        cell_pos_enc='segment', device='cpu', vocab_size=128, seq_len=9,
        n_special_tokens=1, n_segments=2, n_special_values=16,
        enc_emb_dim=16, enc_depth=1, pred_emb_dim=8, pred_depth=1,
        num_heads=2, nz_spc=True)
    batch = _sequence_batch()
    with torch.no_grad():
        z, _ = encoder(batch=batch)
        torch.manual_seed(1)
        metadata_only, changed = make_swapped_batch(
            batch, n_classes_per_key=[16], keys=['batch_value'])
        z_metadata_only, _ = encoder(batch=metadata_only)
        torch.manual_seed(1)
        swapped, _ = make_swapped_batch(
            batch, n_classes_per_key=[16], keys=['batch_value'],
            sequence_positions=[0])
        z_swapped, _ = encoder(batch=swapped)
    assert changed.any()
    # Swapping only the metadata field is a no-op for the encoder ...
    assert cycle_consistency_loss(z, z_metadata_only, changed) == 0
    # ... writing the label into the special token is not
    assert cycle_consistency_loss(z, z_swapped, changed) > 0


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
