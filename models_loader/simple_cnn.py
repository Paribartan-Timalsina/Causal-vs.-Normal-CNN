"""
Simple CNN model for digit classification.
This model is prone to learning spurious correlations (like color) instead of causal features (shape).
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A standard convolutional neural network for MNIST classification.
    This model has no built-in protection against spurious correlations.
    """
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
