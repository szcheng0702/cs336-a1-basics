import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab_size_ = num_embeddings
        self.d_model_ = embedding_dim
        # initialize weights
        W = torch.empty((self.vocab_size_, self.d_model_), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(W, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(W)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
