from pathlib import Path
import torch

def load_model_weights(model,path,strict=True):
    """Load a trusted local PyTorch state_dict; do not load untrusted pickle files."""
    state=torch.load(Path(path),map_location='cpu',weights_only=True)
    if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
    return model.load_state_dict(state,strict=strict)
