from torch import nn
from smgvtt.encoders.tactile.patches.tactile_patches import tactile_patches
from smgvtt.encoders.tactile.graphs.dilation_schedule import tactile_k_schedule,tactile_dilation_schedule
from smgvtt.encoders.tactile.blocks.tig_block import TiGBlock
from smgvtt.utils.tensor.tensor_utils import masked_global_max

class TiGEncoder(nn.Module):
    """Feature-reconstructed KNN each layer, 12 blocks, dilated max-relative heads."""
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.stem=nn.Linear(cfg.input_channels,cfg.hidden_dim)
        ks=tactile_k_schedule(cfg.layers,cfg.k_start,cfg.k_end)
        ds=tactile_dilation_schedule(cfg.layers)
        self.blocks=nn.ModuleList([TiGBlock(cfg.hidden_dim,k,d,cfg.heads,cfg.drop_path) for k,d in zip(ks,ds)])
        self.norm=nn.LayerNorm(cfg.hidden_dim)
    def forward(self,tactile):
        x=self.stem(tactile_patches(tactile,self.cfg.patch_grid))
        for block in self.blocks:x=block(x)
        return masked_global_max(self.norm(x))
