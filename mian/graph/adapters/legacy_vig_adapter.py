"""Optional direct bridge to the original user-supplied ViG gcn_lib operators.

This adapter preserves gcn_lib without modifying its upstream behavior. It is not
used by default TiG because the paper's multi-head block adds extra operations.
Requires timm for importing the original torch_vertex module.
"""
import torch
from torch import nn

class LegacyMaxRelative(nn.Module):
    """Adapt [B,N,D] nodes to original MRConv2d [B,D,N,1]."""
    def __init__(self, dim):
        super().__init__()
        try:
            from gcn_lib.torch_vertex import MRConv2d
        except ImportError as exc:
            raise ImportError('Original gcn_lib import requires timm; install optional [legacy] extras') from exc
        self.operator = MRConv2d(dim,dim,act='relu',norm=None)
    def forward(self,nodes,neighbor_indices):
        b,n,d=nodes.shape
        centers=torch.arange(n,device=nodes.device).view(1,n,1).expand_as(neighbor_indices)
        edges=torch.stack([neighbor_indices,centers],dim=0)
        x=nodes.transpose(1,2).unsqueeze(-1)
        return self.operator(x,edges).squeeze(-1).transpose(1,2)

class LegacyDilatedKNN(nn.Module):
    """Direct adapter for DenseDilatedKnnGraph from the user's archive."""
    def __init__(self,k=9,dilation=1):
        super().__init__()
        # Load the unchanged standalone torch_edge.py directly to avoid the
        # legacy package __init__ importing torch_vertex (and timm).
        import importlib.util
        from pathlib import Path
        original = Path(__file__).resolve().parents[3] / 'gcn_lib' / 'torch_edge.py'
        spec = importlib.util.spec_from_file_location('_original_gcn_lib_edge', original)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.knn=module.DenseDilatedKnnGraph(k,dilation,stochastic=False)
    def forward(self,nodes):
        edges=self.knn(nodes.transpose(1,2).unsqueeze(-1))
        return edges[0]
