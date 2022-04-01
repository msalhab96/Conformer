from torch.optim import Adam
from typing import Tuple
import math


class AdamWarmup:
    def __init__(
            self,
            parameters,
            betas: Tuple[float, float],
            eps: float,
            weight_decay: float,
            warmup_staps: int,
            model_dim: int,
            scaler: float,
            step_size: int
            ):
        self.optimizer = Adam(
            parameters,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        self.warmup_staps = warmup_staps
        self.model_dim = model_dim
        self.scaler = scaler
        self.peak = self.scaler / math.sqrt(self.model_dim)
        self.inv_warmup_staps = 1 / math.sqrt(self.warmup_staps ** 3)
        self.step_size = step_size
        self.counter = 0

    def get_lr(self, step: int) -> float:
        return self.peak * min(
            1 / math.sqrt(step), 
            step * self.inv_warmup_staps
        )
    
    def step(self) -> None:
        self.counter += self.step_size
        lr = self.get_lr(self.counter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()