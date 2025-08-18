import torch
import math
from typing import Callable, Optional


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        betas,
        eps,
        weight_decay,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> torch.Tensor:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, betas, eps, weight_decay = (
                group["lr"],
                group["betas"],
                group["eps"],
                group["weight_decay"],
            )  # Get the hyperparameters
            beta1, beta2 = betas
            for theta in group["params"]:
                if theta.grad is None:
                    continue
                state = self.state[theta]  # Get state associated with p.
                t = state.get(
                    "t", 1
                )  # Get iteration number from the state, or initial value.
                grad = theta.grad.data  # Get the gradient of loss with respect to p.
                m = state.get("m", torch.zeros_like(theta))  # first moment
                m = beta1 * m + (1 - beta1) * grad
                v = state.get("v", torch.zeros_like(theta))  # second moment
                v = beta2 * v + (1 - beta2) * grad.pow(2)
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                theta.data -= (
                    lr_t * m / (torch.sqrt(v) + eps)
                )  # Update weight tensor in-place.
                theta.data -= lr * weight_decay * theta.data
                # update state
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m  # store first moment
                state["v"] = v  # store second moment

        return loss
