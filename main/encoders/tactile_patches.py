import torch
import torch.nn.functional as F

def tactile_patches(image,grid=14):
    """Uniform tactile cells [B,C,H,W] -> [B,grid**2,C]."""
    return F.adaptive_avg_pool2d(image,(grid,grid)).flatten(2).transpose(1,2)
