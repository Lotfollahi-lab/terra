"""
Tests for ``schedule_values_at_step``: the learning rate, weight decay and EMA
momentum that pretraining uses at a given step. The schedules span
``ipe_scale * epochs`` epochs while training stops after ``epochs``, so with
``ipe_scale > 1`` they never reach the configured final values.

With ``ipe_scale < 1`` the schedules end before training does: the momentum
generator ran out (training stopped with StopIteration) and the cosine
learning rate and weight decay turned back towards their start values. They
now stay at their final values after the end of the schedule.
"""

import math

import pytest
import torch

from terra.utils.schedulers import (CosineWDSchedule, WarmupCosineSchedule,
                                    ema_momentum_schedule,
                                    schedule_values_at_step)

# TERRA-112M optimization config (released params.yaml)
TERRA_112M = dict(num_epochs=5, warmup=1, start_lr=1e-6, ref_lr=6e-5,
                  final_lr=1e-6, wd=0.4, final_wd=4.0, ema=(0.9995, 1.0))


@pytest.mark.parametrize('ipe_scale', [1.25, 1.0, 0.5])
def test_matches_training_schedules_at_every_step(ipe_scale):
    # Step the real schedulers and the momentum generator exactly as
    # init_opt / train do and compare at every step.
    ipe, cfg = 7, TERRA_112M
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.)
    t_max = int(ipe_scale * cfg['num_epochs'] * ipe)
    lr_schedule = WarmupCosineSchedule(
        optimizer, warmup_steps=int(cfg['warmup'] * ipe),
        start_lr=cfg['start_lr'], ref_lr=cfg['ref_lr'],
        final_lr=cfg['final_lr'], T_max=t_max)
    wd_schedule = CosineWDSchedule(optimizer, ref_wd=cfg['wd'],
                                   final_wd=cfg['final_wd'], T_max=t_max)
    momentum = ema_momentum_schedule(cfg['ema'], ipe, cfg['num_epochs'],
                                     ipe_scale)

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


def test_ema_momentum_schedule_unchanged_for_ipe_scale_at_least_one():
    ipe, num_epochs, ema = 7, 5, (0.9995, 1.0)
    for ipe_scale in (1.0, 1.25):
        # The generator train() used before (bitwise)
        expected = [ema[0] + i * (ema[1] - ema[0]) /
                    (ipe * num_epochs * ipe_scale)
                    for i in range(int(ipe * num_epochs * ipe_scale) + 1)]
        assert list(ema_momentum_schedule(
            ema, ipe, num_epochs, ipe_scale)) == expected


def test_schedules_stay_at_final_values_after_their_end():
    # ipe_scale 0.5: the schedules end after half of the training steps
    ipe, cfg = 7, TERRA_112M
    n_steps = ipe * cfg['num_epochs']
    momentum = list(ema_momentum_schedule(cfg['ema'], ipe, cfg['num_epochs'],
                                          0.5))
    assert len(momentum) == n_steps + 1  # one per training step (and one)
    lrs, wds = [], []
    for step in range(1, n_steps + 1):
        values = schedule_values_at_step(
            step=step, iterations_per_epoch=ipe, ipe_scale=0.5, **cfg)
        lrs.append(values['lr'])
        wds.append(values['weight_decay'])
        # The LR and WD schedules end at step int(0.5 * n_steps), the
        # momentum schedule after momentum 0.5 * n_steps (0-based)
        if step >= int(0.5 * n_steps):
            assert values['lr'] == pytest.approx(cfg['final_lr'])
            assert values['weight_decay'] == pytest.approx(cfg['final_wd'])
        if step - 1 > 0.5 * n_steps:
            assert values['ema_momentum'] == cfg['ema'][1]
    # Monotone after the warmup (no turn back towards the start values)
    warmup = int(cfg['warmup'] * ipe)
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1:]))
    assert all(a <= b for a, b in zip(wds, wds[1:]))
    assert all(a <= b for a, b in zip(momentum, momentum[1:]))
