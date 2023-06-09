from .mobilenet import MobileNetV3Large, MobileNetV3Small
from .resent import ResNet50

def get_model(model_type: str = 'resnet', 
              input_dims: int = 3,
              num_classes: int = 10,
              use_pretrained: bool = True):
    if model_type == 'resnet':
        model = ResNet50(input_dims, num_classes, use_pretrained)
    elif model_type == 'mobilenet_large':
        model = MobileNetV3Large(input_dims, num_classes, use_pretrained)
    elif model_type == 'mobilenet_small':
        model = MobileNetV3Small(input_dims, num_classes, use_pretrained)
    return model