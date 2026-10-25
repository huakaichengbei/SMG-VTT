from torch import nn
from smgvtt.encoders.visual.visual_graph_encoder import VisualGraphEncoder
from smgvtt.encoders.tactile.tig_encoder import TiGEncoder
from smgvtt.fusion.projections.modality_projection import ModalityProjection
from smgvtt.fusion.sgpm.sgpm import SGPM

class SMGVTT(nn.Module):
    """Method-only model. A caller supplies the *complete* textual prototypes [C,256]."""
    def __init__(self,cfg):
        super().__init__();self.visual=VisualGraphEncoder(cfg.visual)
        self.tactile=TiGEncoder(cfg.tactile)
        self.visual_projection=ModalityProjection(cfg.visual.hidden_dim,cfg.sgpm.embedding_dim)
        self.tactile_projection=ModalityProjection(cfg.tactile.hidden_dim,cfg.sgpm.embedding_dim)
        self.sgpm=SGPM(cfg.sgpm)
    def encode(self,visual_image,tactile_image,visual_keypoints=None):
        v=self.visual_projection(self.visual(visual_image,visual_keypoints))
        t=self.tactile_projection(self.tactile(tactile_image))
        return v,t
    def forward(self,visual_image,tactile_image,prototypes,visual_keypoints=None,return_details=False):
        v,t=self.encode(visual_image,tactile_image,visual_keypoints)
        return self.sgpm(v,t,prototypes,return_details)
