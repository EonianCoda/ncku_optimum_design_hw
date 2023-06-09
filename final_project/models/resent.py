
from torchvision.models import resnet50
import torch.nn as nn


class ResNet50(nn.Module):
    def __init__(self, 
                 input_dims:int = 3, 
                 num_classes:int = 10, 
                 use_pretrained:bool = True):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(pretrained = use_pretrained)
        if input_dims != 3:
            self.resnet.conv1 = nn.Conv2d(input_dims, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)