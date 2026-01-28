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
    Computes the IRM penalty term.
    
    The IRM penalty encourages the model to find a representation where the optimal
    classifier is the same across all environments. This is done by penalizing the
    variance of gradients across environments.
    
    Args:
        loss_1 (Tensor): Loss from environment 1
        loss_2 (Tensor): Loss from environment 2
        dummy_w (Parameter): Dummy weight parameter (fixed at 1.0) for gradient calculation
    
    Returns:
        Tensor: IRM penalty value (should be minimized)
    """
    # Calculate gradient of loss w.r.t. dummy weight for each environment
    grad_1 = autograd.grad(loss_1.mean(), dummy_w, create_graph=True)[0]
    grad_2 = autograd.grad(loss_2.mean(), dummy_w, create_graph=True)[0]
    
    # Penalty is the sum of squared gradients
    # When gradients are similar across environments, penalty is low
    penalty = torch.sum(grad_1 ** 2) + torch.sum(grad_2 ** 2)
    
    return penalty
