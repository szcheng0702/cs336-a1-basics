import torch
import torch.nn as nn
from .linear import Linear

class SwiGLu(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model_ = d_model
        self.d_ff_ = d_ff

        # initialize Linear layer
        self.w1 = Linear(self.d_model_, self.d_ff_)
        self.w2 = Linear(self.d_ff_, self.d_model_)
        self.w3 = Linear(self.d_model_, self.d_ff_)

            
    def forward(self,x:torch.Tensor)->torch.Tensor:
        silu_w1x = self.w1(x)*torch.sigmoid(self.w1(x))
        gated = silu_w1x * self.w3(x)
        return self.w2(gated)