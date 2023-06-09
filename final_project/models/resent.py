
from torchvision.models import resnet50
import torch.nn as nn


class ResNet50(nn.Module):
    def __init__(self, input_dim:int = 3, num_classes: int = 10):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(pretrained=True)
        if input_dim != 3:
            self.resnet.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)