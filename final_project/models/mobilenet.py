
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
import torch.nn as nn

class MobileNetV3Large(nn.Module):
    def __init__(self, 
                 input_dims:int = 3,
                 num_classes:int = 10, 
                 use_pretrained:bool = True):
        super(MobileNetV3Large, self).__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.model = mobilenet_v3_large(weights = weights)
        if input_dims != 3:
            self.model.features[0][0] = nn.Conv2d(input_dims, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class MobileNetV3Small(nn.Module):
    def __init__(self, 
                 input_dims:int = 3,
                 num_classes:int = 10, 
                 use_pretrained:bool = True):
        super(MobileNetV3Small, self).__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.model = mobilenet_v3_small(weights = weights)
        if input_dims != 3:
            self.model.features[0][0] = nn.Conv2d(input_dims, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)