"""
Invariant Risk Minimization (IRM) model and training utilities.
IRM is a causal machine learning approach that learns representations invariant across environments.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd


class IRM_CNN(nn.Module):
    """
    Convolutional neural network designed for Invariant Risk Minimization.
    
    IRM trains the model to find features that are predictive across different environments,
    forcing it to ignore spurious correlations (like color) and focus on causal features (like shape).
    """
    
    def __init__(self):
        super(IRM_CNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Classification layer
        self.classifier = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.shape[0], -1)
        logits = self.classifier(features)
        return logits


def compute_irm_penalty(loss_1, loss_2, dummy_w):
    """
    IRMv1 penalty (Arjovsky et al., 2019): squared gradient of each environment's loss
    w.r.t. a dummy classifier scale fixed at 1.0, summed over environments.
    It is zero only when the same classifier is optimal in both environments.
    """
    grad_1 = autograd.grad(loss_1.mean(), dummy_w, create_graph=True)[0]
    grad_2 = autograd.grad(loss_2.mean(), dummy_w, create_graph=True)[0]
    return grad_1.pow(2).sum() + grad_2.pow(2).sum()
