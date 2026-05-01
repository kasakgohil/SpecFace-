import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes):
    try:
        from torchvision.models import ResNet18_Weights
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except (ImportError, TypeError):
        resnet = models.resnet18(pretrained=True)

    # Freeze early layers (optional, but good for speed and stability)
    for param in resnet.parameters():
        param.requires_grad = False
        
    # Unfreeze the last block and fc layer for fine-tuning
    for param in resnet.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet
