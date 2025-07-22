import numpy as np
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.dim_in_ = in_features
        self.dim_out_=out_features
        # initialize weights
        stdev = np.sqrt(2/(self.dim_in_+self.dim_out_))
        W = torch.empty((self.dim_out_,self.dim_in_),device=device,dtype=dtype)
        torch.nn.init.trunc_normal_(W,mean=0,std=stdev,a=-3*stdev,b=3*stdev)
        self.weight = nn.Parameter(W)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return einsum(self.weight,x,"d_out d_in,... d_in->... d_out")