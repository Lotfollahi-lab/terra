"""
Tests for ``schedule_values_at_step``: the learning rate, weight decay and EMA
momentum that pretraining uses at a given step. The schedules span
``ipe_scale * epochs`` epochs while training stops after ``epochs``, so with
``ipe_scale > 1`` they never reach the configured final values.
"""

import math

import pytest
import torch

from terra.utils.schedulers import (CosineWDSchedule, WarmupCosineSchedule,
                                    schedule_values_at_step)

# TERRA-112M optimization config (released params.yaml)
TERRA_112M = dict(num_epochs=5, warmup=1, start_lr=1e-6, ref_lr=6e-5,
                  final_lr=1e-6, wd=0.4, final_wd=4.0, ema=(0.9995, 1.0))


def test_matches_training_schedules_at_every_step():
    # Step the real schedulers and the momentum generator exactly as
    # init_opt / train do and compare at every step.
    ipe, cfg, ipe_scale = 7, TERRA_112M, 1.25
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.)
    t_max = int(ipe_scale * cfg['num_epochs'] * ipe)
    lr_schedule = WarmupCosineSchedule(
        optimizer, warmup_steps=int(cfg['warmup'] * ipe),
        start_lr=cfg['start_lr'], ref_lr=cfg['ref_lr'],
        final_lr=cfg['final_lr'], T_max=t_max)
    wd_schedule = CosineWDSchedule(optimizer, ref_wd=cfg['wd'],
                                   final_wd=cfg['final_wd'], T_max=t_max)
    ema = cfg['ema']
    momentum = (ema[0] + i * (ema[1] - ema[0]) /
                (ipe * cfg['num_epochs'] * ipe_scale)
                for i in range(int(ipe * cfg['num_epochs'] * ipe_scale) + 1))

    for step in range(1, ipe * cfg['num_epochs'] + 1):
        expected = {'lr': lr_schedule.step(),
                    'weight_decay': wd_schedule.step(),
                    'ema_momentum': next(momentum)}
        values = schedule_values_at_step(
            step=step, iterations_per_epoch=ipe, ipe_scale=ipe_scale, **cfg)
        assert values == pytest.approx(expected, rel=1e-12)


def test_ipe_scale_one_ends_at_configured_values():
    ipe = 1000
    values = schedule_values_at_step(
        step=ipe * TERRA_112M['num_epochs'], iterations_per_epoch=ipe,
        ipe_scale=1.0, **TERRA_112M)
    assert values['lr'] == pytest.approx(TERRA_112M['final_lr'])
    assert values['weight_decay'] == pytest.approx(TERRA_112M['final_wd'])
    assert values['ema_momentum'] == pytest.approx(1.0, abs=1e-6)


def test_terra_112m_schedules_end_early():
    # ipe_scale 1.25: the last step is at 4/5.25 of the cosine LR decay and
    # 5/6.25 of the WD and EMA schedules.
    ipe = 1000
    values = schedule_values_at_step(
        step=ipe * TERRA_112M['num_epochs'], iterations_per_epoch=ipe,
        ipe_scale=1.25, **TERRA_112M)
    lr = 1e-6 + (6e-5 - 1e-6) * 0.5 * (1 + math.cos(math.pi * 4 / 5.25))
    wd = 4.0 + (0.4 - 4.0) * 0.5 * (1 + math.cos(math.pi * 0.8))
    assert values['lr'] == pytest.approx(lr)                  # ~8.9e-6
    assert values['weight_decay'] == pytest.approx(wd)        # ~3.66
    assert values['ema_momentum'] == pytest.approx(0.9999, abs=1e-6)
