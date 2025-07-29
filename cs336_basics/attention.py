import torch
import torch.nn as nn
from einops import einsum

def softmax(x: torch.Tensor, dim_idx: int):
    # take out the 1D 
    x_max = x.max(dim=dim_idx, keepdim=True).values
    exp_x = torch.exp((x-x_max))
    sum_exp_x = torch.sum(exp_x,dim=dim_idx,keepdim=True)
    return exp_x/sum_exp_x

def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor | None = None):
    '''
    Args:
    Q: batch_size, ..., queries, d_k
    K: batch_size, ..., keys, d_k
    V: batch_size, ..., keys, d_v
    mask: batch_size, ..., queries, keys
    Return:
    batch_size, ..., d_v
    '''
    dot_product = einsum(Q, K, "b ... queries d_k, b ... keys d_k -> b ... queries keys")
    d_k = Q.shape[-1]
    attn = 1/torch.sqrt(torch.tensor(d_k)) * dot_product
    if mask is not None:
        attn = attn.masked_fill(~mask, float('-inf'))
    attention = softmax(attn, dim_idx=-1) # softmax over dimension of all the keys, i.e. m
    return einsum(attention, V, "b ... queries keys, b ... keys d_v -> b ... queries d_v")



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, device=None):
        super().__init__()
        self. d_k_ = d_model/num_heads
        self. d_v_ = d_model/num_heads

        
    def forward(self,x:torch.Tensor, token_positions: torch.Tensor)->torch.Tensor:
        '''
        Args:
        x (Float[Tensor, "... seq_len d_k"])
        token_positions (Int[Tensor, "... seq_len])
        Return:
        Float[Tensor, " ... seq_len d_k"]
        '''