import torch
import torch.nn as nn
from einops import einsum, reduce

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model_ = d_model
        self.eps_ = eps
        # initialize weight
        g = torch.ones(self.d_model_)
        self.weight = nn.Parameter(g)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        squared_norm = reduce(x**2, '... d_model -> ... 1', 'sum')
        rms = torch.sqrt(1/self.d_model_*(squared_norm+self.eps_))
        result = einsum(x/rms, self.weight, "... d_model, d_model -> ... d_model") # element wise
        return result.to(in_dtype)