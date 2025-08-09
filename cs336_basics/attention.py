import torch
import torch.nn as nn
from einops import einsum, rearrange
from .linear import Linear
from .rope import RotaryPositionalEmbedding

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
    def __init__(self, d_model:int, num_heads:int, max_seq_len: int | None = None, theta: float | None = None):
        super().__init__()
        self.d_model_ = d_model
        self.num_heads_ = num_heads
        self. d_k_ = self.d_model_/num_heads
        self. d_v_ = self.d_model_/num_heads
        self.weight_Q = Linear(self.d_model_,self.d_model_) # h*d_k x d_model
        self.weight_K = Linear(self.d_model_,self.d_model_) # h*d_k x d_model
        self.weight_V = Linear(self.d_model_,self.d_model_) # h*d_v x d_model
        self.weight_O = Linear(self.d_model_,self.d_model_) # d_model x h*d_v

        self.rope = RotaryPositionalEmbedding(theta, self.d_k_, max_seq_len) if theta and max_seq_len else None


        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        '''
        Args:
        x (Float[Tensor, "... seq_len d_k"])
        Return:
        Float[Tensor, " ... seq_len d_v"]
        '''
        Q = einsum(self.weight_Q, x, 'd_k d_model,... queries_seq_len d_model->... queries_seq_len d_k')
        K = einsum(self.weight_K, x, 'd_k d_model,... kv_seq_len d_model->... kv_seq_len d_k')
        # b = x.shape[0]
        # if self.rope:
        #     # Use rope for Q and K only
        #     # token positions: Int[Tensor, "... seq_len]
        #     token_positions = 
        #     Q = self.rope(Q, token_positions)
        #     K = self.rope(K, token_positions)

        V = einsum(self.weight_V, x, 'd_v d_model,... kv_seq_len d_model->... kv_seq_len d_v')
        seq_len = x.shape[-2]
        # mask size: b h queries, keys
        # lower triangular matrix with size queries x keys
        mask = torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).bool()
        # broad cast to b h queries, keys
        mask_expanded = rearrange(mask, 'queries keys -> 1 1 queries keys').expand(b, self.num_heads_, seq_len, seq_len)

        # rearrange multiple head
        Q = rearrange(Q, '... queries_seq_len (h d_k) -> ... h queries_seq_len d_k', h = self.num_heads_)
        K = rearrange(K, '... kv_seq_len (h d_k) -> ... h kv_seq_len d_k', h = self.num_heads_)
        V = rearrange(V, '... kv_seq_len (h d_v) -> ... h kv_seq_len d_v', h = self.num_heads_)
        attn = scaled_dot_product_attention(Q,K,V,mask_expanded)
        return rearrange(attn, '... h kv_seq_len d_v -> ... kv_seq_len (h d_v)')


