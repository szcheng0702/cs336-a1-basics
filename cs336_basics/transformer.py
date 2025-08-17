import torch
import torch.nn as nn
from .positionwise_ff import SwiGLu
from .rms_norm import RMSNorm
from .attention import MultiHeadAttention
from .embedding import Embedding
from .linear import Linear


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model_ = d_model
        self.num_heads_ = num_heads
        self.d_ff_ = d_ff
        self.ln1 = RMSNorm(self.d_model_, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(
            self.d_model_,
            self.num_heads_,
            max_seq_len,
            theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(self.d_model_, device=device, dtype=dtype)
        self.ffn = SwiGLu(self.d_model_, self.d_ff_, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x (Float[Tensor, "... seq_len d_model"])
        Return:
        Float[Tensor, " ... seq_len d_model"]
        """
        self.transformer_out = x + self.attn(self.ln1(x))
        self.ffn_out = self.transformer_out + self.ffn(self.ln2(self.transformer_out))
        return self.ffn_out


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
                `sequence_length` is at most `context_length`.
        Return:
            Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
            next-word distribution for each token.
        """
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
