"""
Tests for the training launcher ``terra.training.main``.

* ``get_distributed_info`` checked SLURM in a separate ``if`` before the
  torchrun / mpirun ``if``-``elif`` chain: with plain ``srun`` (one task per
  process) the SLURM values were found but the chain then exited with "Can't
  find the environment variables for local rank", and under mpirun inside a
  SLURM allocation the SLURM branch required ``WORLD_SIZE``.
* ``TORCH_DISTRIBUTED_DEBUG`` was forced to ``DETAIL`` (expensive checks of
  every collective), overriding the job environment; it is now opt-in.
"""

import importlib
import os
import runpy
import sys
import types

import pytest
import torch

_LAUNCHER_VARS = ('LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'SLURM_PROCID',
                  'SLURM_NTASKS', 'SLURM_GPUS_ON_NODE',
                  'OMPI_COMM_WORLD_LOCAL_RANK', 'OMPI_COMM_WORLD_SIZE',
                  'OMPI_COMM_WORLD_RANK')


@pytest.fixture
def launcher_env(monkeypatch):
    """Clean launcher environment; ``wandb`` stubbed (training extra)."""
    monkeypatch.setitem(sys.modules, 'wandb', types.ModuleType('wandb'))
    for name in _LAUNCHER_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)

    def set_env(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return importlib.import_module('terra.training.main')
    return set_env


@pytest.mark.parametrize('env, expected', [
    # srun, one task per process (no WORLD_SIZE)
    (dict(SLURM_PROCID='5', SLURM_NTASKS='8', SLURM_GPUS_ON_NODE='4'),
     (5, 1, 8)),
    # torchrun under srun: SLURM_PROCID is the node's task
    (dict(SLURM_PROCID='0', SLURM_NTASKS='1', SLURM_GPUS_ON_NODE='4',
          LOCAL_RANK='2', RANK='6', WORLD_SIZE='8'), (6, 2, 8)),
    # torchrun
    (dict(LOCAL_RANK='2', RANK='2', WORLD_SIZE='4'), (2, 2, 4)),
    # mpirun, also inside a SLURM allocation
    (dict(OMPI_COMM_WORLD_LOCAL_RANK='1', OMPI_COMM_WORLD_SIZE='8',
          OMPI_COMM_WORLD_RANK='5'), (5, 1, 8)),
    (dict(SLURM_PROCID='0', SLURM_NTASKS='1', SLURM_GPUS_ON_NODE='4',
          OMPI_COMM_WORLD_LOCAL_RANK='1', OMPI_COMM_WORLD_SIZE='8',
          OMPI_COMM_WORLD_RANK='5'), (5, 1, 8)),
])
def test_get_distributed_info(launcher_env, env, expected):
    main = launcher_env(**env)
    assert main.get_distributed_info() == expected


def test_get_distributed_info_without_launcher_exits(launcher_env):
    main = launcher_env()
    with pytest.raises(SystemExit, match='local rank'):
        main.get_distributed_info()


@pytest.mark.parametrize('debug', [None, 'INFO'])
def test_main_leaves_torch_distributed_debug_to_the_environment(
        launcher_env, monkeypatch, debug):
    main = launcher_env()
    monkeypatch.setenv('TORCH_CPP_LOG_LEVEL', 'WARNING')
    if debug is None:
        monkeypatch.delenv('TORCH_DISTRIBUTED_DEBUG', raising=False)
    else:
        monkeypatch.setenv('TORCH_DISTRIBUTED_DEBUG', debug)
    # Run the script entry point; without launcher variables it stops at
    # get_distributed_info, after setting its environment variables
    with pytest.raises(SystemExit):
        runpy.run_path(main.__file__, run_name='__main__')
    assert os.environ.get('TORCH_DISTRIBUTED_DEBUG') == debug
