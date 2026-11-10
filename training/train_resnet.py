from models_loader.resnet import get_resnet_model
from training.common import run_shortcut_experiment


if __name__ == "__main__":
    run_shortcut_experiment(get_resnet_model, "ResNet-18", epochs=5, lr=1e-4, batch_size=16, out="models/Resnet18.pth")
