import torch.nn as nn

class ResNet18Features(nn.Module):
    """Spatial convolutional features (resnet18 layer4); output channels=512."""
    def __init__(self,pretrained=False,frozen=True):
        super().__init__()
        from torchvision.models import resnet18,ResNet18_Weights
        net=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.features=nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool,net.layer1,net.layer2,net.layer3,net.layer4)
        if frozen:
            for p in self.features.parameters():p.requires_grad_(False)
            self.features.eval()
        self.frozen=frozen
    def train(self,mode=True):
        super().train(mode)
        if self.frozen:self.features.eval()
        return self
    def forward(self,x):return self.features(x)
