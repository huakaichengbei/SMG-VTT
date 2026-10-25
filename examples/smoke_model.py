"""Example forward pass without downloading weights or training. Requires torch, torchvision, OpenCV."""
import torch
from smgvtt import SMGVTT,ModelConfig

if __name__=='__main__':
    cfg=ModelConfig()
    model=SMGVTT(cfg).eval()
    with torch.no_grad():
        v=torch.rand(1,3,112,112)
        t=torch.rand(1,3,112,112)
        all_prototypes=torch.nn.functional.normalize(torch.randn(5,256),dim=-1)
        result=model(v,t,all_prototypes,return_details=True)
    for name,value in result.items():print(name,tuple(value.shape))
    print('UNTRAINED model demonstration: outputs are not experimental results')
