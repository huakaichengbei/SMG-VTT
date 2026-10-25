from dataclasses import dataclass
import torch
from smgvtt.graph.knn.knn import knn_indices

@dataclass
class DenseGraph:
    node_features: torch.Tensor
    neighbor_indices: torch.Tensor
    edge_weights: torch.Tensor | None = None

def feature_knn_graph(nodes, k=6, sigma=1.0):
    idx = knn_indices(nodes,k)
    from smgvtt.utils.tensor.tensor_utils import batched_gather
    nbr = batched_gather(nodes,idx)
    delta = nodes.unsqueeze(2)-nbr
    weights = torch.exp(-delta.square().sum(-1)/(2*max(sigma,1e-6)**2))
    return DenseGraph(nodes,idx,weights)
