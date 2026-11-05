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


def get_biased_mnist_env(correlation_prob, train=True, subset=slice(None), label_noise=0.0, data_dir='./data'):
    """
    Colored MNIST environment for IRM.

    correlation_prob: chance a digit gets its label's color, otherwise a random color.
    train: use the MNIST train split if True, the test split if False.
    subset: slice of the split, so environments can use disjoint images.
    label_noise: chance a label is replaced by a different random digit (applied before coloring).
    """
    dataset = datasets.MNIST(data_dir, train=train, download=True)
    images = dataset.data[subset].float() / 255.0
    labels = dataset.targets[subset].clone()
    n = len(labels)

    if label_noise > 0:
        flip = torch.rand(n) < label_noise
        labels = torch.where(flip, (labels + torch.randint(1, 10, (n,))) % 10, labels)

    use_label_color = torch.rand(n) < correlation_prob
    color_idx = torch.where(use_label_color, labels, torch.randint(0, 10, (n,)))
    images_rgb = images.unsqueeze(1).repeat(1, 3, 1, 1) * COLOR_MAP[color_idx].view(-1, 3, 1, 1)

    return torch.utils.data.TensorDataset(images_rgb, labels)
