"""Learning-rate schedules indexed by the training module's own step counters."""

import math

from torch.optim import Optimizer


class WarmupCosine:
    """Linear warm-up to ``base_lr`` over ``warmup_steps``, then cosine decay to 0 at ``total_steps``.

    A pure function of the step index, so it needs no state in the checkpoint: the generator schedule is indexed by
    ``g_step`` and the discriminator schedule by ``d_step``, never by ``trainer.global_step``.
    """

    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        if warmup_steps < 0 or total_steps < 1:
            raise ValueError(f"invalid schedule: warmup_steps={warmup_steps}, total_steps={total_steps}")
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)

    def factor(self, step: int) -> float:
        """Multiplier of ``base_lr`` for the update that is made at counter value ``step`` (0-based)."""
        if step < self.warmup_steps:
            return (step + 1) / (self.warmup_steps + 1)
        span = max(self.total_steps - self.warmup_steps, 1)
        progress = min(max((step - self.warmup_steps) / span, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def __call__(self, step: int) -> float:
        return self.base_lr * self.factor(step)

    def apply(self, optimizer: Optimizer, step: int) -> float:
        """Sets the learning rate of every parameter group for the update at ``step`` and returns it."""
        lr = self(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
