
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


class ResNet50(nn.Module):
    def __init__(self, 
                 input_dims:int = 3,
                 num_classes:int = 10, 
                 use_pretrained:bool = True):
        super(ResNet50, self).__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.model = resnet50(weights = weights)
        if input_dims != 3:
            self.model.conv1 = nn.Conv2d(input_dims, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)