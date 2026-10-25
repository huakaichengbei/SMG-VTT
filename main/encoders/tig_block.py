from torch import nn
from smgvtt.graph.knn.knn import knn_indices
from smgvtt.encoders.tactile.layers.multihead_graphconv import MultiHeadRelativeGraphConv
from smgvtt.utils.regularization.drop_path import DropPath

class TiGBlock(nn.Module):
    def __init__(self,dim,k,dilation,heads,drop_path=.1):
        super().__init__();self.k=k;self.dilation=dilation
        self.norm1=nn.LayerNorm(dim);self.conv=MultiHeadRelativeGraphConv(dim,heads)
        self.norm2=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
        self.drop_path=DropPath(drop_path)
    def forward(self,x):
        z=self.norm1(x)
        idx=knn_indices(z,self.k,self.dilation)
        x=x+self.drop_path(self.conv(z,idx))
        return x+self.drop_path(self.ffn(self.norm2(x)))
