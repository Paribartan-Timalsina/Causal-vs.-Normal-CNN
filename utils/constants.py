import torch
# Color mapping: Each digit (0-9) is assigned a unique color during training
COLOR_MAP = torch.tensor([
    [1.0, 0.0, 0.0],  # 0: Red
    [0.0, 1.0, 0.0],  # 1: Green
    [0.0, 0.0, 1.0],  # 2: Blue
    [1.0, 1.0, 0.0],  # 3: Yellow
    [1.0, 0.0, 1.0],  # 4: Magenta
    [0.0, 1.0, 1.0],  # 5: Cyan
    [1.0, 0.5, 0.0],  # 6: Orange
    [0.5, 0.0, 1.0],  # 7: Purple
    [0.5, 1.0, 0.0],  # 8: Lime
    [1.0, 1.0, 1.0]   # 9: White
])

COLOR_NAMES = [
    "Red", "Green", "Blue", "Yellow", "Magenta", 
    "Cyan", "Orange", "Purple", "Lime", "White"
]

# Mapping from color tuple to digit
COLOR_TO_DIGIT_MAP = {
    (1.0, 0.0, 0.0): 0,  # Red -> 0
    (0.0, 1.0, 0.0): 1,  # Green -> 1
    (0.0, 0.0, 1.0): 2,  # Blue -> 2
    (1.0, 1.0, 0.0): 3,  # Yellow -> 3
    (1.0, 0.0, 1.0): 4,  # Magenta -> 4
    (0.0, 1.0, 1.0): 5,  # Cyan -> 5
    (1.0, 0.5, 0.0): 6,  # Orange -> 6
    (0.5, 0.0, 1.0): 7,  # Purple -> 7
    (0.5, 1.0, 0.0): 8,  # Lime -> 8
    (1.0, 1.0, 1.0): 9   # White -> 9
}