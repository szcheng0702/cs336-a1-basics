import torch
import torch.nn as nn
from einops import einsum, rearrange
from .positionwise_ff import SwiGLu
from .rms_norm import RMSNorm
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff: int, max_seq_len: int | None = None, theta: float | None = None):
        super().__init__()
        self.d_model_ = d_model
        self.num_heads_ = num_heads
        self.d_ff_ = d_ff
        self.ln1 = RMSNorm(self.d_model_)
        self.attn = MultiHeadAttention(self.d_model_, self.num_heads_,max_seq_len,theta)
        self.ln2 = RMSNorm(self.d_model_)
        self.ffn = SwiGLu(self.d_model_, self.d_ff_)

        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        '''
        Args:
        x (Float[Tensor, "... seq_len d_model"])
        Return:
        Float[Tensor, " ... seq_len d_model"]
        '''
        self.transformer_out = self.attn(self.ln1(x))
        self.ffn_out = self.ffn(self.ln2(self.transformer_out+x))
        return self.transformer_out+self.ffn_out