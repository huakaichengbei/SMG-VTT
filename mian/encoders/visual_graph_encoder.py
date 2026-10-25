import numpy as np
import torch
from torch import nn
from smgvtt.encoders.visual.backbones.visual_backbone import ResNet18Features
from smgvtt.encoders.visual.keypoints.orb_keypoints import detect_orb
from smgvtt.encoders.visual.keypoints.bilinear_sampling import sample_at_keypoints
from smgvtt.encoders.visual.graphs.graph_building import feature_knn_graph
from smgvtt.encoders.visual.layers.weighted_gcn import ResidualGCNLayer
from smgvtt.utils.tensor.tensor_utils import masked_global_max

class VisualGraphEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.backbone=ResNet18Features(cfg.pretrained,cfg.frozen)
        self.input_proj=nn.Linear(512,cfg.hidden_dim)
        self.layers=nn.ModuleList([ResidualGCNLayer(cfg.hidden_dim,cfg.dropout) for _ in range(cfg.layers)])
    def forward(self,images,points_xy=None):
        if points_xy is None:
            # ORB works on uint8 CPU images. This preprocessing step is not differentiable.
            uint8=(images.detach().clamp(0,1)*255).byte().permute(0,2,3,1).cpu().numpy()
            pts=[detect_orb(arr,self.cfg.keypoints) for arr in uint8]
            count=min(len(p) for p in pts)
            points_xy=torch.as_tensor(np.stack([p[:count] for p in pts]),device=images.device,dtype=images.dtype)
        x=sample_at_keypoints(self.backbone(images),points_xy,images.shape[-2:])
        x=self.input_proj(x)
        graph=feature_knn_graph(x,self.cfg.knn,self.cfg.similarity_sigma)
        for layer in self.layers:x=layer(x,graph.neighbor_indices,graph.edge_weights)
        return masked_global_max(x)
