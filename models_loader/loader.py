import torch
from .simple_cnn import SimpleCNN
from .resnet import get_resnet_model
from .irm import IRM_CNN

def load_model(model_type, device):
    if model_type == "Simple CNN":
        model = SimpleCNN()
        model.load_state_dict(torch.load("simple_cnn.pth", map_location=device))

    elif model_type == "ResNet-18":
        model = get_resnet_model()
        model.load_state_dict(torch.load("Resnet18.pth", map_location=device))

    else:
        model = IRM_CNN()
        model.load_state_dict(torch.load("IRM_cnn.pth", map_location=device))

    model.to(device)
    model.eval()
    return model
