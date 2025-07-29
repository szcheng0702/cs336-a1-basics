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




# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(self, theta: float, d_k:int, max_seq_len: int, device=None):
#         super().__init__()
#         self.theta_ = theta
#         self.d_k_ = d_k
#         self.max_seq_len_ = max_seq_len
#         # set R (R: seq_len d_k d_k)
#         self.R = torch.empty(self.max_seq_len_, self.d_k_, self.d_k_,device=device)
#         for i in range(self.max_seq_len_):
#             R_is = []
#             for k in range(self.d_k_//2):
#                 angle = torch.tensor(i/(theta)**(2*k/self.d_k_))
#                 c,s = torch.cos(angle), torch.sin(angle)
#                 R_is.append(torch.tensor([[c,-s],[s,c]],device=device))
#             self.R[i] = torch.block_diag(*R_is)  
#         self.register_buffer("rotation",self.R,persistent=False)

        
    
#     def forward(self,x:torch.Tensor, token_positions: torch.Tensor)->torch.Tensor:
#         '''
#         Args:
#         x (Float[Tensor, "... seq_len d_k"])
#         token_positions (Int[Tensor, "... seq_len])
#         Return:
#         Float[Tensor, " ... seq_len d_k"]
#         '''
#         rotation = self.R[token_positions]
#         return einsum(rotation, x, '... seq_len d_kout d_kin,... seq_len d_kin->... seq_len d_kout')