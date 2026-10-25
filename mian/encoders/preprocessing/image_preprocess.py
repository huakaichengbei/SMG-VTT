import numpy as np
import torch
from PIL import Image

def load_rgb(path,size=(224,224)):
    """RGB [1,3,H,W] in [0,1]. Size=224 is a release default, not asserted manuscript value."""
    with Image.open(path) as im:
        arr=np.asarray(im.convert('RGB').resize(size,Image.Resampling.BILINEAR),dtype=np.float32)/255.
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
