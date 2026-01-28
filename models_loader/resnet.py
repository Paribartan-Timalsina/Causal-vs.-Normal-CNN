"""
ResNet-18 model adapted for 28x28 MNIST images.
Pre-trained on ImageNet and fine-tuned for digit classification.
"""

import torch.nn as nn
from torchvision import models


def get_resnet_model():
    """
    Returns a ResNet-18 model modified for 28x28 images.
    
    Modifications:
    1. First conv layer: 7x7 stride-2 -> 3x3 stride-1 (preserves resolution)
    2. Remove initial MaxPool (prevents over-downsampling)
    3. Final layer: 1000 classes -> 10 classes
    
    Returns:
        nn.Module: Modified ResNet-18 model
    """
    # Load pre-trained ResNet18 with ImageNet weights
    model = models.resnet18(weights='DEFAULT')

    # Modify first convolution for 28x28 images
    # Standard ResNet expects 224x224, but MNIST is only 28x28
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove MaxPool layer (it reduces spatial dimensions too aggressively for 28x28)
    model.maxpool = nn.Identity()

    # Modify output layer for 10 digit classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    return model
