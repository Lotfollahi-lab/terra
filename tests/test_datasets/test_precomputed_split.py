"""
Precomputed splits hold row numbers of a tokenized dataset. A split applied to
another tokenization of the data (whose row order changed, e.g. after the
corpus was re-tokenized) silently selected other cells, including cells of
the held-out batches. Splits written by ``save_precomputed_split`` record the
dataset they were made for and are refused for any other; training refuses a
split that selects cells of the test or validation batches.
"""

import logging
import pickle

import pytest
from datasets import Dataset

from terra.datasets.utils import (load_precomputed_split, prepare_dataset,
                                  save_precomputed_split)

# 24 cells: 4 per batch of 1_batch0..1_batch2 and 2_batch0..2_batch2
CELL_IDS = [f'{d}_batch{b}_{c}' for d in (1, 2) for b in range(3)
            for c in range(4)]


def _dataset(cell_ids=CELL_IDS):
    return Dataset.from_dict({
        'cell_id': cell_ids,
        'tokens': [[i + 2, i + 3] for i in range(len(cell_ids))]})


def _args(dataset_path, **data):
    return {'data': {'tokenized_data_folder_path': str(dataset_path), **data},
            'meta': {'cell_pos_enc': 'segment'}}


def test_split_round_trip(tmp_path):
    ds = _dataset()
    save_precomputed_split(ds, [0, 5, 7], tmp_path / 'split.pkl')
    assert load_precomputed_split(tmp_path / 'split.pkl', ds) == [0, 5, 7]


def test_split_of_another_tokenization_raises(tmp_path):
    save_precomputed_split(_dataset(), list(range(12)), tmp_path / 'split.pkl')
    # Same cells, files tokenized in another order
    reordered = _dataset(CELL_IDS[12:] + CELL_IDS[:12])
    with pytest.raises(ValueError, match='another tokenization'):
        load_precomputed_split(tmp_path / 'split.pkl', reordered)


def test_split_of_dataset_with_other_length_raises(tmp_path):
    save_precomputed_split(_dataset(), [0, 1], tmp_path / 'split.pkl')
    with pytest.raises(ValueError, match='24 rows, but the dataset has 20'):
        load_precomputed_split(tmp_path / 'split.pkl', _dataset(CELL_IDS[:20]))


def test_legacy_split_is_used_with_a_warning(tmp_path, caplog):
    with open(tmp_path / 'split.pkl', 'wb') as f:
        pickle.dump([1, 2], f)
    with caplog.at_level(logging.WARNING):
        assert load_precomputed_split(tmp_path / 'split.pkl', _dataset()) == [
            1, 2]
    assert 'bare list of row numbers' in caplog.text


@pytest.mark.parametrize('legacy', [False, True])
def test_training_split_with_heldout_cells_raises(tmp_path, legacy):
    ds = _dataset()
    ds.save_to_disk(tmp_path / 'ds')
    # Rows 4-7 are the cells of 1_batch1
    if legacy:
        with open(tmp_path / 'split.pkl', 'wb') as f:
            pickle.dump(list(range(8)), f)
    else:
        save_precomputed_split(ds, list(range(8)), tmp_path / 'split.pkl')
    args = _args(tmp_path / 'ds', precomputed_split=str(tmp_path / 'split.pkl'),
                 test_batch_ids=['1_batch1'], val_batch_ids=None)
    with pytest.raises(ValueError, match='4 of the 8 rows'):
        prepare_dataset(args, train_mode=True)
    # Evaluation use is not checked, and training can opt out
    assert len(prepare_dataset(args)[0]) == 8
    args['data']['allow_heldout_in_split'] = True
    assert len(prepare_dataset(args, train_mode=True)[0]) == 8


def test_training_split_without_heldout_cells_is_accepted(tmp_path):
    ds = _dataset()
    ds.save_to_disk(tmp_path / 'ds')
    train_rows = [i for i, c in enumerate(CELL_IDS)
                  if not c.startswith(('1_batch1_', '2_batch2_'))]
    save_precomputed_split(ds, train_rows, tmp_path / 'split.pkl')
    args = _args(tmp_path / 'ds', precomputed_split=str(tmp_path / 'split.pkl'),
                 test_batch_ids=['1_batch1'], val_batch_ids=['2_batch2'])
    train, val, test = prepare_dataset(args, train_mode=True)
    assert len(train) == 16 and val is None and test is None


def test_epoch_splits_are_checked(tmp_path):
    ds = _dataset()
    ds.save_to_disk(tmp_path / 'ds')
    save_precomputed_split(ds, [[0, 1], [4, 9]], tmp_path / 'split.pkl')
    args = _args(tmp_path / 'ds',
                 precomputed_epoch_splits=str(tmp_path / 'split.pkl'),
                 test_batch_ids=['1_batch1'])
    with pytest.raises(ValueError, match='1 of the 2 rows'):
        prepare_dataset(args, train_mode=True)


def test_split_recording_its_excluded_batches_is_not_scanned(tmp_path,
                                                             monkeypatch):
    # The training split of split_dataset.py records the held-out batches it
    # excludes; training then skips the scan of every cell id
    ds = _dataset()
    ds.save_to_disk(tmp_path / 'ds')
    train_rows = [i for i, c in enumerate(CELL_IDS)
                  if not c.startswith('1_batch1_')]
    save_precomputed_split(ds, train_rows, tmp_path / 'split.pkl',
                           excluded_batch_ids=['1_batch1', '2_batch2'])
    import terra.datasets.utils as U

    def no_scan(*args, **kwargs):
        raise AssertionError('scanned')
    monkeypatch.setattr(U, '_heldout_row_mask', no_scan)
    args = _args(tmp_path / 'ds', precomputed_split=str(tmp_path / 'split.pkl'),
                 test_batch_ids=['1_batch1'])
    assert len(prepare_dataset(args, train_mode=True)[0]) == 20
    # Batches the record does not list are scanned
    args['data']['val_batch_ids'] = ['2_batch0']
    with pytest.raises(AssertionError, match='scanned'):
        prepare_dataset(args, train_mode=True)


def test_vectorized_heldout_scan_matches_the_batch_id_rule():
    from terra.datasets.utils import _heldout_row_mask
    cell_ids = CELL_IDS + ['1_batch10_0', '1_batch1', 'x-1_batch1_0', 'nobatch']
    ds = _dataset(cell_ids)
    heldout = {'1_batch1', 'x-1_batch1'}
    fast = _heldout_row_mask(ds, heldout)                     # Arrow column
    slow = _heldout_row_mask(ds.select(range(len(ds))), heldout)  # indices
    assert fast.tolist() == slow.tolist()
    assert [c for c, m in zip(cell_ids, fast) if m] == [
        '1_batch1_0', '1_batch1_1', '1_batch1_2', '1_batch1_3', '1_batch1',
        'x-1_batch1_0']
