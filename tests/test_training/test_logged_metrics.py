"""The logged training loss is averaged over the DDP ranks (logging only)."""
import importlib

import pytest
import torch
import torch.distributed as dist

from terra.utils.logging import AverageMeter

# The module (terra.training re-exports the `train` function under its name)
T = importlib.import_module('terra.training.train')


@pytest.fixture
def two_ranks(monkeypatch):
    """Stand-in for a 2-rank process group: all_reduce adds the other rank's
    tensor."""
    other = {}

    def all_reduce(tensor, op=None):
        tensor += other['value'].to(tensor.dtype)
    monkeypatch.setattr(dist, 'is_available', lambda: True)
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(dist, 'get_backend', lambda: 'gloo')
    monkeypatch.setattr(dist, 'all_reduce', all_reduce)
    return other


def test_mean_over_ranks(two_ranks):
    two_ranks['value'] = torch.tensor(3.0)
    assert T._mean_over_ranks(torch.tensor(1.0), torch.device('cpu')) == 2.0


def test_meter_average_over_ranks(two_ranks):
    meter = AverageMeter()
    meter.update(1.0)
    meter.update(3.0)
    # The other rank recorded one value, 6
    two_ranks['value'] = torch.tensor([6.0, 1.0])
    assert T._meter_average_over_ranks(meter, torch.device('cpu')) == (
        pytest.approx(10 / 3))


def test_single_process_values_unchanged():
    meter = AverageMeter()
    meter.update(0.3)
    meter.update(0.4)
    assert T._mean_over_ranks(torch.tensor(0.7), torch.device('cpu')) == (
        float(torch.tensor(0.7)))
    assert T._meter_average_over_ranks(meter, torch.device('cpu')) == meter.avg
