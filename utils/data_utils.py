"""
Data utilities for generating colored MNIST datasets.
This module handles dataset generation with spurious correlations for demonstrating
the difference between standard ML and Causal ML approaches.
"""

import torch
from torchvision import datasets
from utils.constants import COLOR_MAP


def get_colored_mnist(train=True, data_dir='./data'):
    """
    Generates colored MNIST dataset.
    
    Args:
        train (bool): If True, creates training data with spurious correlation (color == label).
                     If False, creates test data with random colors (distribution shift).
        data_dir (str): Directory to store MNIST data.
    
    Returns:
        TensorDataset: Dataset containing colored images and labels.
    """
    # Download and load MNIST dataset
    dataset = datasets.MNIST(data_dir, train=train, download=True)
    images = dataset.data.float() / 255.0
    labels = dataset.targets

    # Expand grayscale images to RGB (N, 28, 28) -> (N, 3, 28, 28)
    images_rgb = images.unsqueeze(1).repeat(1, 3, 1, 1)

    if train:
        # TRAINING: Spurious Correlation - Color perfectly predicts the label
        print("Generating BIASED Training Data (Color == Label)...")
        assigned_colors = COLOR_MAP[labels]
    else:
        # TESTING: Distribution Shift - Random colors (model must rely on shape)
        print("Generating UNBIASED Test Data (Random Colors)...")
        random_indices = torch.randint(0, 10, (len(labels),))
        assigned_colors = COLOR_MAP[random_indices]

    # Apply colors to images
    assigned_colors = assigned_colors.view(-1, 3, 1, 1)
    images_rgb = images_rgb * assigned_colors

    return torch.utils.data.TensorDataset(images_rgb, labels)


def get_biased_mnist_env(correlation_prob, data_dir='./data'):
    """
    Generates biased MNIST for IRM training with specific correlation strength.
    
    Args:
        correlation_prob (float): Probability that color matches label (0.0 to 1.0).
                                 1.0 = perfect correlation, 0.5 = random, 0.1 = mostly wrong.
        data_dir (str): Directory to store MNIST data.
    
    Returns:
        TensorDataset: Dataset with controlled spurious correlation.
    """
    dataset = datasets.MNIST(data_dir, train=True, download=True)
    images = dataset.data.float() / 255.0
    labels = dataset.targets

    images_rgb = images.unsqueeze(1).repeat(1, 3, 1, 1)
    final_colors = torch.zeros(len(labels), 3)

    for i in range(len(labels)):
        if torch.rand(1).item() < correlation_prob:
            # Assign color matching the label
            final_colors[i] = COLOR_MAP[labels[i]]
        else:
            # Assign random color
            rand_idx = torch.randint(0, 10, (1,)).item()
            final_colors[i] = COLOR_MAP[rand_idx]

    final_colors = final_colors.view(-1, 3, 1, 1)
    images_rgb = images_rgb * final_colors

    return torch.utils.data.TensorDataset(images_rgb, labels)
