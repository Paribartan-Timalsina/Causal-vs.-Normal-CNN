from models_loader.simple_cnn import SimpleCNN
from training.common import run_shortcut_experiment


if __name__ == "__main__":
    run_shortcut_experiment(SimpleCNN, "Simple CNN", epochs=10, lr=1e-3, batch_size=16, out="models/simple_cnn.pth")
