import torch
from torch import nn
from smgvtt.encoders.tactile.layers.relative_graphconv import MaxRelativeGraphConv

class MultiHeadRelativeGraphConv(nn.Module):
    """Independent projection subspaces; each applies max-relative graph convolution."""
    def __init__(self,dim,heads=4):
        super().__init__()
        if dim%heads:raise ValueError('dim must be divisible by heads')
        self.heads=heads;self.head_dim=dim//heads
        self.transforms=nn.ModuleList([nn.Linear(dim,self.head_dim) for _ in range(heads)])
        self.convs=nn.ModuleList([MaxRelativeGraphConv(self.head_dim,self.head_dim) for _ in range(heads)])
        self.output=nn.Linear(dim,dim)
    def forward(self,x,idx):
        result=[conv(project(x),idx) for project,conv in zip(self.transforms,self.convs)]
        return self.output(torch.cat(result,dim=-1))
