import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    
    Args:
        d_model (int): Dimension of the model
        eps (float, optional): Small value to avoid division by zero. Defaults to 1
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        
        self.eps = eps # Small value to avoid division by zero
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        # 1. Calculate Mean of Squares
        x = x.pow(2).mean(dim=-1, keepdim=True)
        
        # 2. Calculate RMS
        rms = torch.rsqrt(x + self.eps)
        
        # 3. Normalize and Scale
        return x * rms * self.weight 