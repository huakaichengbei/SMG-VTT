from dataclasses import dataclass, field

@dataclass
class VisualConfig:
    backbone: str = 'resnet18'
    pretrained: bool = False  # Supply licensed/cached weights explicitly for exact experiments
    frozen: bool = True
    keypoints: int = 128  # Implementation choice; manuscript does not fix this value
    knn: int = 6
    layers: int = 3
    hidden_dim: int = 256
    dropout: float = 0.1
    similarity_sigma: float = 1.0  # Implementation choice

@dataclass
class TactileConfig:
    input_channels: int = 3
    patch_grid: int = 14  # Implementation choice; adjust to dataset/sensor
    hidden_dim: int = 256
    layers: int = 12
    k_start: int = 9
    k_end: int = 18
    heads: int = 4
    drop_path: float = 0.1

@dataclass
class TextConfig:
    encoder_name: str = 'distilbert-base-uncased'
    frozen: bool = True
    embedding_dim: int = 256
    max_length: int = 128

@dataclass
class SGPMConfig:
    embedding_dim: int = 256
    temperature: float = 0.07

@dataclass
class ModelConfig:
    visual: VisualConfig = field(default_factory=VisualConfig)
    tactile: TactileConfig = field(default_factory=TactileConfig)
    text: TextConfig = field(default_factory=TextConfig)
    sgpm: SGPMConfig = field(default_factory=SGPMConfig)
