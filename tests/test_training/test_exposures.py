"""
Unit tests of `terra.training.exposures` on a hand-built batch: weighted
losses, per-niche and per-cell loss records, exposure counters, record files
and the config section. The training loop is tested in test_train_loop.py.
"""

import json
import os

import numpy as np
import pytest
import torch

from terra.training.exposures import (ExposureTracker, exposure_config,
                                      load_records, needs_neighbor_rows)

D = 4
N_ROWS = 30


def _batch(n_special_tokens=0, padded=True):
    """
    Two niches of a centre cell and two neighbours (2 genes per cell; the
    second niche lacks its second neighbour's row), one target mask of three
    targets (the last one of niche 1 padding when `padded`).
    """
    torch.manual_seed(0)
    s = n_special_tokens
    segments = torch.tensor([[1, 1, 2, 2, 3, 0],
                             [1, 0, 2, 2, 3, 3]])
    if s:
        # nz_spc: special tokens get segments 1..s, the cells are shifted
        segments = torch.cat([torch.arange(1, s + 1).expand(2, s),
                              torch.where(segments > 0, segments + s,
                                          segments)], dim=1)
    idx = torch.tensor([[0, 2, 4], [0, 3, 5]]) + s
    if s:
        idx = torch.cat([torch.arange(s).expand(2, s), idx], dim=1)
    valid = torch.ones_like(idx, dtype=torch.bool)
    if padded:
        valid[1, -1] = False
    z = [torch.randn(2, idx.size(1), D)]
    h = [torch.randn(2, idx.size(1), D)]
    rows = torch.tensor([10, 20])
    neighbor_rows = torch.tensor([[11, 12], [21, -1]])
    return dict(z=z, h=h, masks_pred=[idx],
                masks_pred_valid=valid[None] if padded else None,
                segments=segments, rows=rows, neighbor_rows=neighbor_rows)


def _tracker(tmp_path, n_special_tokens=0, excl=False, **cfg):
    config = exposure_config({'exposure': cfg})
    return ExposureTracker(
        config, n_rows=N_ROWS, n_segments=3,
        n_special_tokens=n_special_tokens, nz_spc=True,
        excl_spc_from_loss=excl, loss_fn_type='l1', has_neighbor_rows=True,
        folder=str(tmp_path), rank=0, world_size=1,
        device=torch.device('cpu'))


def _token_losses(batch):
    """Per-token L1 loss (mean over the embedding) of the target mask."""
    return (batch['z'][0] - batch['h'][0]).abs().mean(-1)


def test_config():
    assert exposure_config({}) is None
    assert exposure_config({'exposure': {'flush_every': 3}}) is None
    cfg = exposure_config({'exposure': {'counters': True}})
    assert cfg['flush_every'] == 500 and needs_neighbor_rows(cfg)
    assert cfg['neighbor_row_check'] is True
    cfg = exposure_config({'exposure': {'loss_weights': 'w.npy'}})
    assert not needs_neighbor_rows(cfg)
    with pytest.raises(ValueError, match='Unknown keys'):
        exposure_config({'exposure': {'counter': True}})
    with pytest.raises(ValueError, match='loss_weight_mode'):
        exposure_config({'exposure': {'loss_weights': 'w.npy',
                                      'loss_weight_mode': 'segment'}})
    with pytest.raises(ValueError, match='loss_weight_clip'):
        exposure_config({'exposure': {'loss_weights': 'w.npy',
                                      'loss_weight_clip': [2, 1]}})
    with pytest.raises(ValueError, match='cell_loss_track_every'):
        exposure_config({'exposure': {'log_cell_losses': True,
                                      'cell_loss_track_every': 0}})


def test_cell_level_settings_need_neighbour_rows(tmp_path):
    with pytest.raises(ValueError, match='add_spatial_index'):
        ExposureTracker(
            exposure_config({'exposure': {'counters': True}}), n_rows=5,
            n_segments=3, n_special_tokens=0, nz_spc=False,
            excl_spc_from_loss=False, loss_fn_type='l1',
            has_neighbor_rows=False, folder=str(tmp_path), rank=0,
            world_size=1, device=torch.device('cpu'))


@pytest.mark.parametrize('n_special_tokens,excl', [(0, False), (1, True),
                                                   (1, False)])
def test_records_and_counters(tmp_path, n_special_tokens, excl):
    tracker = _tracker(tmp_path, n_special_tokens, excl,
                       log_niche_losses=True, log_cell_losses=True,
                       cell_loss_track_every=1, counters=True)
    batch = _batch(n_special_tokens)
    tracker.start_epoch(0)
    tracker.record(step=7, pad_special_tokens=False, **batch)
    tracker.end_epoch(0)
    tl = _token_losses(batch)
    start = n_special_tokens if excl else 0
    tl, valid = tl[:, start:], batch['masks_pred_valid'][0][:, start:]

    niche = load_records(tracker.log_dir, 'niche_losses', epoch=0)
    assert niche['row'].tolist() == [10, 20]
    assert niche['step'].tolist() == [7, 7]
    assert niche['n_targets'].tolist() == valid.sum(1).tolist()
    expected = (tl * valid).sum(1) / valid.sum(1)
    np.testing.assert_allclose(niche['loss'], expected.numpy(), rtol=1e-6)

    cells = load_records(tracker.log_dir, 'cell_losses', epoch=0)
    got = {(int(n), int(c), int(s)): (float(l), int(k)) for n, c, s, l, k in
           zip(cells['niche_row'], cells['cell_row'], cells['segment'],
               cells['loss'], cells['n_targets'])}
    s = n_special_tokens
    tl_all = _token_losses(batch)
    assert got.keys() == {(10, 10, 1), (10, 11, 2), (10, 12, 3),
                          (20, 20, 1), (20, 21, 2)}
    assert got[(10, 11, 2)][0] == pytest.approx(float(tl_all[0, s + 1]))
    assert got[(20, 21, 2)][0] == pytest.approx(float(tl_all[1, s + 1]))
    assert all(k == 1 for _, k in got.values())

    counters = np.load(os.path.join(tracker.log_dir, 'counters_ep000.npz'))
    expect = {name: np.zeros(N_ROWS, dtype=np.int32)
              for name in ('centre', 'present', 'target', 'target_tokens')}
    expect['centre'][[10, 20]] = 1
    expect['present'][[10, 11, 12, 20, 21]] = 1
    expect['target'][[10, 11, 12, 20, 21]] = 1
    expect['target_tokens'][[10, 11, 12, 20, 21]] = 1
    for name, values in expect.items():
        assert np.array_equal(counters[name], values), name


def test_tracked_rows_and_presence_without_targets(tmp_path):
    rows_file = tmp_path / 'tracked.npy'
    np.save(rows_file, np.array([11, 21]))
    tracker = _tracker(tmp_path, log_cell_losses=True,
                       cell_loss_track_rows=str(rows_file), counters=True)
    batch = _batch()
    # niche 1: its first neighbour present but not a target
    batch['masks_pred'][0][1, 1] = 0
    tracker.start_epoch(0)
    tracker.record(step=0, pad_special_tokens=False, **batch)
    tracker.end_epoch(0)
    cells = load_records(tracker.log_dir, 'cell_losses', epoch=0)
    assert cells['cell_row'].tolist() == [11]
    counters = np.load(os.path.join(tracker.log_dir, 'counters_ep000.npz'))
    assert counters['present'][21] == 1 and counters['target'][21] == 0
    assert counters['target_tokens'][20] == 2


def test_weighted_loss(tmp_path):
    weights = np.ones(N_ROWS, dtype=np.float32)
    np.save(tmp_path / 'ones.npy', weights)
    batch = _batch()
    valid = batch['masks_pred_valid'][0]
    abs_diff = (batch['z'][0] - batch['h'][0]).abs()
    unweighted = (abs_diff * valid[..., None]).sum() / (valid.sum() * D)
    tracker = _tracker(tmp_path, loss_weights=str(tmp_path / 'ones.npy'))
    loss = tracker.weighted_loss(pad_special_tokens=False, **batch)
    assert torch.allclose(loss, unweighted)

    # constant weights: the unweighted loss (only ratios matter)
    np.save(tmp_path / 'twos.npy', 2 * weights)
    tracker = _tracker(tmp_path, loss_weights=str(tmp_path / 'twos.npy'))
    assert torch.allclose(
        tracker.weighted_loss(pad_special_tokens=False, **batch), unweighted)

    weights[10], weights[20], weights[11], weights[12] = 4.0, 0.1, 1.5, 3.0
    np.save(tmp_path / 'w.npy', weights)
    tracker = _tracker(tmp_path, loss_weights=str(tmp_path / 'w.npy'))
    # niche mode, clipped to [0.5, 2]: sum(w * loss) / sum(w)
    w = torch.tensor([2.0, 0.5])[:, None].expand_as(valid) * valid
    expected = (abs_diff * w[..., None]).sum() / (w.sum() * D)
    assert torch.allclose(
        tracker.weighted_loss(pad_special_tokens=False, **batch), expected)
    assert tracker.weight_stats['max'] == 2.0
    # cell mode: each target token by its cell (row 21 has weight 1)
    tracker = _tracker(tmp_path, loss_weights=str(tmp_path / 'w.npy'),
                       loss_weight_mode='cell')
    w = torch.tensor([[2.0, 1.5, 2.0], [0.5, 1.0, 1.0]]) * valid
    expected = (abs_diff * w[..., None]).sum() / (w.sum() * D)
    assert torch.allclose(
        tracker.weighted_loss(pad_special_tokens=False, **batch), expected)


def test_weights_are_checked(tmp_path):
    np.save(tmp_path / 'short.npy', np.ones(5))
    with pytest.raises(ValueError, match='holds 5 weights'):
        _tracker(tmp_path, loss_weights=str(tmp_path / 'short.npy'))
    np.save(tmp_path / 'neg.npy', -np.ones(N_ROWS))
    with pytest.raises(ValueError, match='negative'):
        _tracker(tmp_path, loss_weights=str(tmp_path / 'neg.npy'))


def test_flush_chunks_and_restarted_epochs(tmp_path):
    tracker = _tracker(tmp_path, log_niche_losses=True, flush_every=2)
    batch = _batch()
    tracker.start_epoch(1)
    for step in range(5):
        tracker.record(step=step, pad_special_tokens=False, **batch)
    tracker.end_epoch(1)
    files = sorted(os.listdir(os.path.join(tracker.log_dir, 'rank0')))
    assert files == [f'niche_losses_ep001_{i:05d}.npz' for i in range(3)]
    assert load_records(tracker.log_dir, epoch=1)['step'].tolist() == [
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # the epoch restarts (resume): its old records are removed
    tracker.start_epoch(1)
    tracker.record(step=9, pad_special_tokens=False, **batch)
    tracker.end_epoch(1)
    assert load_records(tracker.log_dir, epoch=1)['step'].tolist() == [9, 9]
    assert load_records(tracker.log_dir, epoch=0) == {}


def test_a_launch_removes_the_records_it_will_write_again(tmp_path):
    batch = _batch()
    # A first launch on 2 ranks recorded epochs 0-2 (rank 1's files copied)
    first = _tracker(tmp_path, log_niche_losses=True, counters=True)
    for epoch in range(3):
        first.start_epoch(epoch)
        first.record(step=epoch, pad_special_tokens=False, **batch)
        first.end_epoch(epoch)
    rank_dir = os.path.join(first.log_dir, 'rank1')
    os.makedirs(rank_dir)
    for name in os.listdir(first.rank_dir):
        with open(os.path.join(first.rank_dir, name), 'rb') as src, \
                open(os.path.join(rank_dir, name), 'wb') as dst:
            dst.write(src.read())
    # A resume from the end of epoch 0 on one rank
    resumed = _tracker(tmp_path, log_niche_losses=True, counters=True)
    resumed.start_epoch(1)
    assert sorted(os.listdir(rank_dir)) == ['niche_losses_ep000_00000.npz']
    assert sorted(os.listdir(resumed.rank_dir)) == [
        'niche_losses_ep000_00000.npz']
    assert not os.path.exists(os.path.join(first.log_dir,
                                           'counters_ep001.npz'))
    assert os.path.exists(os.path.join(first.log_dir, 'counters_ep000.npz'))
    with open(os.path.join(first.log_dir, 'manifest.json')) as f:
        launches = json.load(f)['launches']
    assert [launch['start_epoch'] for launch in launches] == [0, 1]
