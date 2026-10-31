"""
Tests for checkpoint loading and resuming.

* ``load_checkpoint`` used to swallow every exception and return the modules
  with their random initialization (epoch 0), e.g. when inference was pointed
  at a wrong path or a resumed run could not read its checkpoint. It now
  raises unless ``raise_on_error=False``.
* ``train`` resumes from ``<save folder>/<write_tag>-latest.pth.tar`` when
  ``read_checkpoint`` is null, but ranks > 0 built their own timestamped save
  folder; ``broadcast_from_rank0`` gives every rank rank 0's folder.
"""

import logging
import subprocess
import sys

import pytest
import torch

from terra.training.train import broadcast_from_rank0
from terra.utils.helper import load_checkpoint


def _load(path, module, **kwargs):
    return load_checkpoint(device='cpu', r_path=path, encoder=None,
                           predictor=None, target_encoder=module, opt=None,
                           scaler=None, is_training=False, **kwargs)


def test_load_checkpoint_loads_weights_and_epoch(tmp_path):
    torch.manual_seed(0)
    saved = torch.nn.Linear(3, 2)
    path = tmp_path / 'model_checkpoint.pt'
    torch.save({'target_encoder': saved.state_dict(), 'epoch': 2,
                'zero_epoch_tracking': True}, path)

    module = torch.nn.Linear(3, 2)
    _, _, module, _, _, epoch, _ = _load(path, module)

    assert epoch == 3
    torch.testing.assert_close(module.weight, saved.weight)


def test_load_checkpoint_raises_on_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match='Could not load checkpoint'):
        _load(tmp_path / 'missing.pt', torch.nn.Linear(3, 2))


def test_load_checkpoint_raises_on_mismatching_weights(tmp_path):
    path = tmp_path / 'model_checkpoint.pt'
    torch.save({'target_encoder': torch.nn.Linear(4, 2).state_dict(),
                'epoch': 0}, path)
    with pytest.raises(RuntimeError, match='Could not load checkpoint'):
        _load(path, torch.nn.Linear(3, 2))


def test_load_checkpoint_legacy_fallback(tmp_path, caplog):
    module = torch.nn.Linear(3, 2)
    weight = module.weight.detach().clone()
    with caplog.at_level(logging.WARNING):
        _, _, module, _, _, epoch, iter_number = _load(
            tmp_path / 'missing.pt', module, raise_on_error=False)
    assert epoch == 0 and iter_number is None
    torch.testing.assert_close(module.weight, weight)
    assert any('WITHOUT the checkpoint' in r.getMessage()
               for r in caplog.records)


def test_broadcast_from_rank0_without_distributed():
    assert broadcast_from_rank0('folder') == 'folder'


_WORKER = """
import sys
import torch.distributed as dist
from terra.training.train import broadcast_from_rank0

rank, world_size, init_file, out_file = (
    int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
dist.init_process_group('gloo', init_method=f'file://{init_file}',
                        rank=rank, world_size=world_size)
# Rank 0 has the run folder; ranks > 0 get None from main.py
folder = 'artifacts/run_rank0' if rank == 0 else None
result = broadcast_from_rank0(folder)
dist.destroy_process_group()
with open(out_file, 'w') as f:
    f.write(str(result))
"""


def test_broadcast_from_rank0_shares_rank0_folder(tmp_path):
    world_size = 2
    script = tmp_path / 'worker.py'
    script.write_text(_WORKER)
    procs = [subprocess.Popen(
        [sys.executable, str(script), str(rank), str(world_size),
         str(tmp_path / 'init'), str(tmp_path / f'rank{rank}.txt')])
        for rank in range(world_size)]
    for proc in procs:
        assert proc.wait(timeout=120) == 0
    for rank in range(world_size):
        assert (tmp_path / f'rank{rank}.txt').read_text() == \
            'artifacts/run_rank0'
