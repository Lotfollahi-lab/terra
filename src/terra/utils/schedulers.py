"""
Adapted from Assran, M. et al. Self-supervised learning from images with
a Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf.
Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/utils/schedulers.py
(05.06.2024).
"""

import math

import torch


__all__ = [
    "WarmupCosineSchedule",
    "CosineWDSchedule",
    "schedule_values_at_step",
]


class WarmupCosineSchedule(object):
    """
    WarmupCosineSchedule class to schedule learning rate.

    Parameters
    ----------
    optimizer:
        Optimizer.
    warmup_steps:
        Number of warmup steps.
    start_lr:
        Starting learning rate.
    ref_lr:
        Reference learning rate.
    T_max:
        Maximum number of training steps.
    last_epoch:
        Last epoch.
    final_lr:
        Final learning rate.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 start_lr: float,
                 ref_lr: float,
                 T_max: int,
                 last_epoch: int = -1,
                 final_lr: float = 0.):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(
                max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (
                            self.ref_lr - self.final_lr) * 0.5 * (
                                1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):
    """
    CosineWDSchedule class to schedule weight decay.

    Parameters
    ----------
    optimizer:
        Optimizer.
    ref_wd:
        Reference weight decay.
    T_max:
        Maximum number of training steps.
    final_wd:
        Final weight decay.
    """
    def __init__(self,
                 optimizer,
                 ref_wd,
                 T_max,
                 final_wd: float = 0.):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (
            1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd


class _NoOptimizer:
    """Stand-in optimizer so the schedules can be evaluated on their own."""
    param_groups: list = []


def schedule_values_at_step(step: int,
                            iterations_per_epoch: int,
                            num_epochs: int,
                            warmup: float,
                            start_lr: float,
                            ref_lr: float,
                            final_lr: float,
                            wd: float,
                            final_wd: float,
                            ema: tuple[float, float],
                            ipe_scale: float,
                            ) -> dict[str, float]:
    """
    Learning rate, weight decay and EMA momentum that pretraining uses at
    (1-based) optimizer step `step`, as set up by `init_opt` and `train`.

    The schedules span ``ipe_scale * num_epochs`` epochs while training stops
    after `num_epochs`, so with ``ipe_scale > 1`` they never reach `final_lr`,
    `final_wd` and ``ema[1]``; with ``ipe_scale = 1`` they end at those values.

    Returns
    ----------
    values:
        Dict with keys 'lr', 'weight_decay' and 'ema_momentum'.
    """
    t_max = int(ipe_scale * num_epochs * iterations_per_epoch)
    lr_schedule = WarmupCosineSchedule(
        _NoOptimizer(),
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=t_max)
    wd_schedule = CosineWDSchedule(
        _NoOptimizer(),
        ref_wd=wd,
        final_wd=final_wd,
        T_max=t_max)
    lr_schedule._step = wd_schedule._step = step - 1
    # Training draws momentum i (0-based) at the i-th optimizer step
    ema_momentum = ema[0] + (step - 1) * (ema[1] - ema[0]) / (
        iterations_per_epoch * num_epochs * ipe_scale)
    return {'lr': lr_schedule.step(),
            'weight_decay': wd_schedule.step(),
            'ema_momentum': ema_momentum}
