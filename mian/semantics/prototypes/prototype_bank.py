import json
from pathlib import Path
import torch
from torch import nn

class PrototypeBank(nn.Module):
    """Storage container for ALL class prototypes; order defines class indices."""
    def __init__(self,category_names,embeddings):
        super().__init__()
        if embeddings.ndim!=2 or embeddings.shape[0]!=len(category_names):raise ValueError('one prototype per category required')
        self.category_names=list(category_names)
        self.register_buffer('embeddings',embeddings.detach().clone())
    def forward(self):return self.embeddings
    def category(self,index):return self.category_names[int(index)]

def load_description_json(path):
    """Mapping {category_name: class_level_description}. No LLM query is made."""
    mapping=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(mapping,dict) or not all(isinstance(v,str) for v in mapping.values()):
        raise ValueError('JSON must contain category -> description strings')
    return mapping
