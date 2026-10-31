"""
Tests for the configurable training seed.

``train`` used to seed numpy/torch and the length-grouped sampler with a
hard-coded 0, so replicate runs of the same config only differed by GPU
non-determinism. The optional top-level ``seed`` config key now sets the
seed; without it the historical seed 0 is used.
"""

import torch

from terra.datasets.dataloaders import init_dataloader_and_sampler
from terra.training.train import resolve_seed


class _LengthDataset(torch.utils.data.Dataset):
    """Minimal dataset exposing the ``n_nz_tokens`` the sampler groups by."""

    def __init__(self, n: int = 64):
        self.n_nz_tokens = [(7 * i) % 23 for i in range(n)]

    def __len__(self):
        return len(self.n_nz_tokens)

    def __getitem__(self, idx):
        return idx


def _sampler_order(**kwargs) -> list[int]:
    _, sampler = init_dataloader_and_sampler(
        cell_dataset=_LengthDataset(),
        batch_size=4,
        distributed=True,
        world_size=1,
        rank=0,
        mega_batch_mult_max=2,
        **kwargs)
    return list(iter(sampler))


def test_resolve_seed_defaults_to_historical_seed():
    assert resolve_seed({}) == 0
    assert resolve_seed({'seed': None}) == 0


def test_resolve_seed_reads_top_level_key():
    assert resolve_seed({'seed': 1000}) == 1000
    assert resolve_seed({'seed': '7'}) == 7


def test_sampler_default_order_unchanged():
    # Omitting `seed` must reproduce the previous (seed 0) sample order.
    assert _sampler_order() == _sampler_order(seed=0)


def test_sampler_order_depends_on_seed():
    assert _sampler_order(seed=1) == _sampler_order(seed=1)
    assert _sampler_order(seed=1) != _sampler_order(seed=0)
