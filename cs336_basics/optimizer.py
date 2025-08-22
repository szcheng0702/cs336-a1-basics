import torch
import math
from typing import Callable, Iterable, Optional


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


def cosine_lr(
    it: int,
    min_learning_rate: float,
    max_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if warmup_iters <= it and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1
            + math.cos(
                (it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters)
            )
        ) * (max_learning_rate - min_learning_rate)
    # t > cosine_cycle_iters
    return min_learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """

    all_gradients = [torch.norm(p.grad) for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack(all_gradients))
    if total_norm < max_l2_norm:
        return
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(max_l2_norm / (total_norm + eps))
