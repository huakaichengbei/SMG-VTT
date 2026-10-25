from torch import nn
from smgvtt.utils.tensor.tensor_utils import l2_normalize

class TextProjection(nn.Module):
    def __init__(self,in_dim=768,out_dim=256):
        super().__init__();self.proj=nn.Linear(in_dim,out_dim)
    def forward(self,x):return l2_normalize(self.proj(x))
