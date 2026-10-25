import torch
import torch.nn.functional as F

def l2_normalize(x, dim=-1):
    return F.normalize(x, p=2, dim=dim)

def batched_gather(x, indices):
    """x [B,N,D], indices [B,N,K] -> [B,N,K,D]."""
    b, n, d = x.shape
    idx = indices.reshape(b, -1)
    result = x.gather(1, idx.unsqueeze(-1).expand(-1,-1,d))
    return result.reshape(b,n,indices.shape[-1],d)

def masked_global_max(x, mask=None):
    """x [B,N,D], optional valid-node mask [B,N]."""
    if mask is None: return x.max(dim=1).values
    x = x.masked_fill(~mask.unsqueeze(-1), -torch.inf)
    return x.max(dim=1).values.nan_to_num(neginf=0.)
