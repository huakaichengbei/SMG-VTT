import torch
from torch import nn
from smgvtt.utils.tensor.tensor_utils import batched_gather

class ResidualGCNLayer(nn.Module):
    """Symmetric normalized weighted adjacency plus self loops, with residual update."""
    def __init__(self,width,dropout=.1):
        super().__init__();self.linear=nn.Linear(width,width);self.activation=nn.ReLU();self.dropout=nn.Dropout(dropout)
    def forward(self,x,neighbors,weights=None):
        b,n,d=x.shape
        adj=torch.zeros(b,n,n,dtype=x.dtype,device=x.device)
        if weights is None:weights=x.new_ones(neighbors.shape)
        adj.scatter_add_(2,neighbors,weights)
        adj=.5*(adj+adj.transpose(1,2))
        adj=adj+torch.eye(n,device=x.device,dtype=x.dtype).unsqueeze(0)
        deg=adj.sum(-1).clamp_min(1e-6).rsqrt()
        norm=deg.unsqueeze(-1)*adj*deg.unsqueeze(-2)
        return x+self.dropout(self.activation(self.linear(torch.bmm(norm,x))))
