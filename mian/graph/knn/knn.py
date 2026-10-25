import torch

def knn_indices(features, k, dilation=1, include_self=False):
    """Rebuild feature-space KNN for each call; [B,N,D] -> [B,N,k].
    Dilation selects every d-th item in a sorted neighbor list.
    """
    b,n,_ = features.shape
    if n < 2: raise ValueError('At least two graph nodes are required')
    if k < 1 or dilation < 1: raise ValueError('k and dilation must be positive')
    with torch.no_grad():
        dist = torch.cdist(features.detach().float(), features.detach().float())
        if not include_self:
            eye = torch.eye(n,device=features.device,dtype=torch.bool).unsqueeze(0)
            dist = dist.masked_fill(eye,float('inf'))
        candidates = min(n if include_self else n-1, k*dilation)
        ranked = dist.topk(candidates, largest=False).indices
        selected = ranked[:,:,::dilation]
        if selected.shape[-1] < k:
            # For small test graphs, repeat the farthest available sampled neighbor.
            selected = torch.cat([selected,selected[:,:,-1:].expand(-1,-1,k-selected.shape[-1])],dim=-1)
    return selected[:,:,:k]
