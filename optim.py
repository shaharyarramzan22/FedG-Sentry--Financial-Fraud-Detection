"""
Stochastic Gradient Langevin Dynamics (SGLD) optimizer -- Eq. 4:

    theta_{t+1} = theta_t - (eta_t / 2) * grad(L(theta_t)) + eps_t,
    eps_t ~ N(0, eta_t * I)

Implemented as a torch.optim.Optimizer subclass with a decaying step size
satisfying the Robbins-Monro conditions (sum eta_t = inf, sum eta_t^2 < inf)
via eta_t = eta_0 / (1 + t)^kappa for kappa in (0.5, 1].
"""
from __future__ import annotations

import math
import torch
from torch.optim import Optimizer


class SGLD(Optimizer):
    def __init__(self, params, lr: float = 1e-3, kappa: float = 0.55, min_lr: float = 1e-5):
        if lr <= 0:
            raise ValueError("lr must be positive")
        defaults = dict(lr=lr, kappa=kappa, min_lr=min_lr, step_count=0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step_count"] += 1
            t = group["step_count"]
            eta_t = max(group["lr"] / (1 + t) ** group["kappa"], group["min_lr"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                noise = torch.randn_like(p) * math.sqrt(eta_t)
                p.add_(grad, alpha=-eta_t / 2.0)
                p.add_(noise)
        return loss
