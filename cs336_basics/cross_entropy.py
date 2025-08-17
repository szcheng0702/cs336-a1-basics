import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor
from .linear import Linear
from .rope import RotaryPositionalEmbedding


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # log softmax oi = oi-logsumexp(j for all j)
    # minus log softmax = -oi+logsumexp(j for all j)
    target_logits = torch.gather(inputs, 1, targets.unsqueeze(-1))
    log_sum_exp = torch.logsumexp(inputs, 1, keepdim=True)
    return torch.mean(log_sum_exp - target_logits)
