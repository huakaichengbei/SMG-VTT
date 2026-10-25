import torch
from torch import nn

class DropPath(nn.Module):
    def __init__(self,p=0.):super().__init__();self.p=p
    def forward(self,x):
        if not self.training or self.p==0:return x
        keep=1-self.p
        mask=(torch.rand((x.shape[0],)+(1,)*(x.ndim-1),device=x.device)<keep).to(x.dtype)
        return x*mask/keep
