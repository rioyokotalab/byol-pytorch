import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineDecayLinearLRScheduler(_LRScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 batch_size: int,
                 total_steps: int,
                 warmup_steps: int,
                 verbose: bool = False):
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, verbose=verbose)

    def get_lr(self):
        return [
            learning_schedule(self.last_epoch, self.batch_size, base_lr,
                              self.total_steps, self.warmup_steps)
            for base_lr in self.base_lrs
        ]


# Implemented by official byol:
# https://github.com/deepmind/deepmind-research/blob/1642ae3499c8d1135ec6fe620a68911091dd25ef/byol/utils/schedules.py#L27
def learning_schedule(global_step: int, batch_size: int,
                      base_learning_rate: float, total_steps: int,
                      warmup_steps: int) -> torch.Tensor:
    """Cosine learning rate scheduler."""
    # Compute LR & Scaled LR
    scaled_lr = base_learning_rate * batch_size / 256.
    learning_rate = (float(global_step) / int(warmup_steps) *
                     scaled_lr if warmup_steps > 0 else scaled_lr)
    learning_rate = torch.tensor(learning_rate, dtype=torch.float32)

    # Cosine schedule after warmup.
    return torch.where(
        torch.tensor(global_step) < torch.tensor(warmup_steps), learning_rate,
        _cosine_decay(global_step - warmup_steps, total_steps - warmup_steps,
                      scaled_lr))


def _cosine_decay(global_step: int, max_steps: int,
                  initial_value: float) -> torch.Tensor:
    """Simple implementation of cosine decay from TF1."""
    global_step = min(global_step, max_steps)
    cosine_decay_value = 0.5 * (
        1 + torch.cos(torch.tensor(math.pi) * global_step / max_steps))
    decayed_learning_rate = initial_value * cosine_decay_value
    return decayed_learning_rate
