import torch
import torch.nn as nn
from einops import einsum

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k:int, max_seq_len: int, device=None):
        super().__init__()
        self.theta_ = theta
        self.d_k_ = d_k
        self.max_seq_len_ = max_seq_len
        # set R (R: max_seq_len d_k d_k)
        self.R = torch.empty(self.max_seq_len_, self.d_k_, self.d_k_,device=device)
        for i in range(self.max_seq_len_):
            R_is = []
            for k in range(self.d_k_//2):
                angle = torch.tensor(i/(theta)**(2*k/self.d_k_))
                c,s = torch.cos(angle), torch.sin(angle)
                R_is.append(torch.tensor([[c,-s],[s,c]],device=device))
            self.R[i] = torch.block_diag(*R_is)  
        self.register_buffer("rotation",self.R,persistent=False)

        
    
    def forward(self,x:torch.Tensor, token_positions: torch.Tensor)->torch.Tensor:
        '''
        Args:
        x (Float[Tensor, "... seq_len d_k"])
        token_positions (Int[Tensor, "... seq_len]) (i.e. b x (0... seq_len))
        Return:
        Float[Tensor, " ... seq_len d_k"]
        '''
        rotation = self.R[token_positions]
        return einsum(rotation, x, '... seq_len d_kout d_kin,... seq_len d_kin->... seq_len d_kout')