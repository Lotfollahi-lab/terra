"""
Adapted from Assran, M. et al. Self-supervised learning from images with
a Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf.
Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/utils/logging.py
(05.06.2024).
"""

import torch


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self,
               val: float,
               n: int = 1):
        """
        Update the average and current value.
        """
        if isinstance(val, torch.Tensor):
                val = val.item()
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CSVLogger(object):
    """
    CSVLogger class to log data to a CSV file.
    """
    def __init__(self,
                 fname: str,
                 *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


def gpu_timer(closure, *args, log_timings=True):
    """
    Helper to time gpu-time to execute closure().
    """
    log_timings = log_timings and torch.cuda.is_available()
    elapsed_time = -1.

    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure(*args)

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


def grad_logger(named_params: list[tuple[str, torch.Tensor]],
                ) -> AverageMeter:
    """
    Log the gradient norm of the model parameters.

    Parameters
    ----------
    named_params:
        List of named parameters.

    Returns
    -------
    stats:
        AverageMeter object.
    """
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None

    for n, p in named_params:
        if p.grad is not None and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = p.grad.detach().norm().item()
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm

    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.0

    return stats
