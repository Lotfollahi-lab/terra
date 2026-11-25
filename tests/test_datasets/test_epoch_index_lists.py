"""
Tests for per-epoch index lists in the length-grouped sampler
(``data.epoch_index_lists``, ``terra.datasets.dataloaders.EpochIndexLists``)
and for the row ids that cell datasets return with ``return_row_index``.
"""

import numpy as np
import pytest
import torch

datasets = pytest.importorskip('datasets')

from terra.datasets.cell_datasets import CellGraphDataset
from terra.datasets.dataloaders import (CustomDistributedLengthGroupedSampler,
                                        EpochIndexLists,
                                        init_dataloader_and_sampler,
                                        spread_repeats)

SEQ, N_SEGMENTS = 4, 3


class _Lengths(torch.utils.data.Dataset):
    """A dataset of `n` rows with the given per-row lengths."""
    def __init__(self, lengths):
        self.n_nz_tokens = list(lengths)

    def __len__(self):
        return len(self.n_nz_tokens)

    def __getitem__(self, i):
        return i


def _lists(tmp_path, array, name='lists.npy'):
    path = tmp_path / name
    np.save(path, np.asarray(array))
    return str(path)


def _orders(dataset, world_size, epochs, lists=None, batch_size=4,
            mega_batch_mult_max=1000, drop_last=False):
    """Per epoch, the indices of every rank."""
    out = []
    samplers = [CustomDistributedLengthGroupedSampler(
        dataset, batch_size=batch_size, num_replicas=world_size, rank=r,
        seed=3, drop_last=drop_last, mega_batch_mult_max=mega_batch_mult_max,
        epoch_index_lists=lists) for r in range(world_size)]
    for epoch in range(epochs):
        per_rank = []
        for sampler in samplers:
            sampler.set_epoch(epoch)
            per_rank.append(list(sampler))
        out.append(per_rank)
    return out


@pytest.mark.parametrize('world_size', [1, 2, 3])
@pytest.mark.parametrize('n_rows,mega', [(103, 1000), (400, 2)])
def test_identity_lists_give_the_default_order(tmp_path, world_size, n_rows,
                                               mega):
    rng = np.random.default_rng(0)
    dataset = _Lengths(rng.integers(1, 50, n_rows))
    lists = EpochIndexLists(
        _lists(tmp_path, np.tile(np.arange(n_rows), (3, 1))),
        n_rows=n_rows, n_epochs=3)
    default = _orders(dataset, world_size, 3, mega_batch_mult_max=mega)
    with_lists = _orders(dataset, world_size, 3, lists,
                         mega_batch_mult_max=mega)
    assert with_lists == default
    # the epochs differ (the order is not fixed)
    assert default[0] != default[1]


def test_lists_set_what_each_epoch_presents(tmp_path):
    rng = np.random.default_rng(1)
    n_rows, batch_size = 64, 4
    lengths = rng.integers(1, 50, n_rows)
    dataset = _Lengths(lengths)
    subset = rng.choice(n_rows, 16, replace=False)
    epochs = [np.tile(subset, 3),                      # a subset, 3 passes
              rng.integers(0, n_rows, 48),             # draws with repeats
              np.concatenate([np.arange(n_rows), np.arange(n_rows)])[:48]]
    lists = EpochIndexLists(_lists(tmp_path, np.stack(epochs)),
                            n_rows=n_rows, n_epochs=3)
    orders = _orders(dataset, 2, 3, lists, batch_size=batch_size,
                     mega_batch_mult_max=2)
    for epoch, per_rank in zip(epochs, orders):
        presented = sorted(per_rank[0] + per_rank[1])
        assert presented == sorted(epoch.tolist())
        # length-grouped: descending lengths within each mega-batch
        # (mega_batch_mult 2 x batch 4, before the split over 2 ranks),
        # except that the longest item is swapped to the very front
        merged = [i for pair in zip(*per_rank) for i in pair]
        assert lengths[merged[0]] == max(lengths[epoch])
        for start in range(0, len(merged), 8):
            block = [lengths[i] for i in merged[start + 1:start + 8]]
            assert block == sorted(block, reverse=True)


def test_spread_repeats():
    rng = np.random.default_rng(3)
    entries = rng.permutation(50)
    out, copies = spread_repeats(entries)
    assert copies == 1 and out is entries
    # 6 passes over 40 rows, shuffled: 6 consecutive passes
    entries = rng.permutation(np.tile(np.arange(40), 6))
    out, copies = spread_repeats(entries)
    assert copies == 6 and sorted(out) == sorted(entries)
    for k in range(6):
        assert sorted(out[40 * k:40 * (k + 1)]) == list(range(40))
    # draws with replacement: the multiset is kept, copies are spread
    entries = rng.integers(0, 30, 600)
    out, copies = spread_repeats(entries)
    assert sorted(out) == sorted(entries)
    gaps = [np.diff(np.flatnonzero(out == r)).min() for r in range(30)
            if (out == r).sum() > 1]
    assert min(gaps) > 5


def _steps_with_copies(per_rank, step_size):
    """Share of entries with another copy of their row in the same step."""
    merged = [i for pair in zip(*per_rank) for i in pair]
    shared = 0
    for start in range(0, len(merged), step_size):
        step = merged[start:start + step_size]
        counts = {row: step.count(row) for row in step}
        shared += sum(c for c in counts.values() if c > 1)
    return shared / len(merged)


def test_repeated_rows_rarely_share_a_step(tmp_path):
    rng = np.random.default_rng(4)
    n_rows, batch_size = 2048, 8
    dataset = _Lengths(rng.integers(1, 30, n_rows))  # many equal lengths
    subset = rng.choice(n_rows, 512, replace=False)
    epochs = np.stack([np.tile(subset, 8),                  # 1/4 x 8 passes
                       rng.integers(0, 512, 4096)])         # with replacement
    lists = EpochIndexLists(_lists(tmp_path, epochs), n_rows=n_rows,
                            n_epochs=2)
    orders = _orders(dataset, 2, 2, lists, batch_size=batch_size)
    # the passes stay apart: no copies in one step
    assert _steps_with_copies(orders[0], 2 * batch_size) == 0
    merged = [i for pair in zip(*orders[0]) for i in pair]
    for k in range(8):
        assert sorted(merged[512 * k:512 * (k + 1)]) == sorted(subset)
    # draws with replacement: fewer copies per step than in a shuffle
    # (about 2.6% of the entries there)
    assert _steps_with_copies(orders[1], 2 * batch_size) < 0.01


def test_loader_length_follows_the_lists(tmp_path):
    n_rows = 40
    dataset = _Lengths(np.ones(n_rows, dtype=int))
    lists = EpochIndexLists(
        _lists(tmp_path, np.zeros((2, 24), dtype=np.int32)),
        n_rows=n_rows, n_epochs=2)
    loader, sampler = init_dataloader_and_sampler(
        dataset, batch_size=4, distributed=True, world_size=2, rank=0,
        mega_batch_mult_max=1000, epoch_index_lists=lists, drop_last=True)
    assert len(sampler) == 12 and len(loader) == 3
    with pytest.raises(ValueError, match='use_sampler'):
        init_dataloader_and_sampler(
            dataset, batch_size=4, distributed=False, world_size=1, rank=0,
            mega_batch_mult_max=1000, epoch_index_lists=lists)


@pytest.mark.parametrize('array,n_epochs,match', [
    (np.zeros(10, dtype=int), 1, '2-D integer'),
    (np.zeros((2, 10)), 1, '2-D integer'),
    (np.zeros((1, 10), dtype=int), 2, 'hold 1 epochs'),
    (np.zeros((1, 0), dtype=int), 1, 'empty'),
])
def test_invalid_lists_are_refused(tmp_path, array, n_epochs, match):
    with pytest.raises(ValueError, match=match):
        EpochIndexLists(_lists(tmp_path, array), n_rows=10, n_epochs=n_epochs)


def test_rows_outside_the_split_are_refused(tmp_path):
    # every epoch the run trains is checked when the lists are loaded
    path = _lists(tmp_path, [[0, 3, 9], [0, 3, 10]])
    EpochIndexLists(path, n_rows=10, n_epochs=1)
    with pytest.raises(ValueError, match='Epoch 1 .* outside'):
        EpochIndexLists(path, n_rows=10, n_epochs=2)
    lists = EpochIndexLists(path, n_rows=11, n_epochs=2)
    with pytest.raises(ValueError, match='index 11 rows'):
        CustomDistributedLengthGroupedSampler(
            _Lengths(np.ones(12, dtype=int)), batch_size=2, num_replicas=1,
            rank=0, epoch_index_lists=lists)


def _graph_dataset(n_rows=12, offsets=None):
    rng = np.random.default_rng(2)
    rows = {'gene_tokens': [], 'gene_expr': [], 'n_nonzero_tokens': []}
    for _ in range(n_rows):
        tokens = np.zeros(SEQ * N_SEGMENTS, dtype=np.int64)
        values = np.zeros(SEQ * N_SEGMENTS)
        for s in range(N_SEGMENTS):
            n = int(rng.integers(1, SEQ + 1))
            tokens[s * SEQ:s * SEQ + n] = rng.integers(10, 90, n)
            values[s * SEQ:s * SEQ + n] = np.sort(rng.random(n))[::-1]
        rows['gene_tokens'].append(tokens.tolist())
        rows['gene_expr'].append(values.tolist())
        rows['n_nonzero_tokens'].append(int((tokens != 0).sum()))
    if offsets is not None:
        rows['neighbor_offsets'] = offsets
    dataset = datasets.Dataset.from_dict(rows)
    dataset.set_format(type='torch', columns=list(rows))
    return dataset


def _cell_dataset(dataset, n_segments=N_SEGMENTS, **kwargs):
    return CellGraphDataset(
        dataset=dataset, gt_type='combined', cell_pos_enc='segment',
        vocab_size=100, seq_len_cell=SEQ,
        seq_len_neighborhood=SEQ * (n_segments - 1), **kwargs)


def test_items_hold_their_row_and_neighbour_rows():
    n_rows = 12
    # 2 neighbours per row, the last row has only one
    offsets = [[1, -1] if r not in (0, n_rows - 1) else ([1, 2] if r == 0
               else [-1]) for r in range(n_rows)]
    dataset = _graph_dataset(n_rows, offsets)
    plain = _cell_dataset(dataset)
    with_rows = _cell_dataset(dataset, return_row_index=True)
    for row in range(n_rows):
        item, ref = with_rows[row], plain[row]
        assert int(item.pop('row_idx')) == row
        expected = [row + o for o in offsets[row]] + [-1] * (
            2 - len(offsets[row]))
        assert item.pop('neighbor_rows').tolist() == expected
        # the model inputs are unchanged
        assert list(item) == list(ref)
        for key in item:
            assert torch.equal(item[key], ref[key]), key


def test_neighbour_rows_follow_the_neighbours_read():
    # A dataset of 2 neighbours read as cell + 1 neighbour
    dataset = _graph_dataset(4, [[1, 2], [1, -1], [-2, -1], [-3, -1]])
    one = CellGraphDataset(
        dataset=dataset, gt_type='combined', cell_pos_enc='segment',
        vocab_size=100, seq_len_cell=SEQ, seq_len_neighborhood=SEQ,
        truncate_neighbors=True, return_row_index=True)
    assert one[0]['neighbor_rows'].tolist() == [1]
    assert one[3]['neighbor_rows'].tolist() == [0]


def test_rows_without_offsets_column():
    dataset = _graph_dataset(5)
    item = _cell_dataset(dataset, return_row_index=True)[3]
    assert int(item['row_idx']) == 3 and 'neighbor_rows' not in item
    assert 'row_idx' not in _cell_dataset(dataset)[3]


def test_neighbour_rows_outside_the_dataset_are_unknown():
    dataset = _graph_dataset(3, [[1, 2], [-5, 1], [-1, 4]])
    items = _cell_dataset(dataset, return_row_index=True)
    assert items[0]['neighbor_rows'].tolist() == [1, 2]
    assert items[1]['neighbor_rows'].tolist() == [-1, 2]
    assert items[2]['neighbor_rows'].tolist() == [1, -1]
