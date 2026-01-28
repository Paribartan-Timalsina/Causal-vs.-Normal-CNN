import numpy as np
import torch
from PIL import Image
from utils.constants import COLOR_MAP

def preprocess_uploaded_image(image, selected_color, device):
    """
    Preprocess image for model and apply selected color:
    - Convert to grayscale
    - Resize to 28x28
    - Apply color to create RGB channels
    - Normalize to [0,1]
    
    Args:
        image: PIL Image
        selected_color: Index of color to apply (0-9)
        device: torch device
    
    Returns:
        img_tensor: Tensor of shape (3, 28, 28) for model
        img_rgb: Numpy array for display
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Convert to grayscale (like MNIST)
    img_gray = image.convert('L')

    # Resize to 28x28
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(img_resized).astype(np.float32) / 255.0  # (28, 28)

    # Invert if needed (MNIST has white digits on black background)
    # Check if the image has dark digits on light background
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    # Convert to tensor and create RGB channels
    img_tensor = torch.from_numpy(img_array).float()  # (28, 28)
    img_rgb = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # (3, 28, 28)

    # Apply the selected color
    color = COLOR_MAP[selected_color].view(3, 1, 1)
    img_rgb = img_rgb * color

    # Convert to numpy for display
    img_rgb_np = img_rgb.numpy()

    return img_rgb.to(device), img_rgb_np
