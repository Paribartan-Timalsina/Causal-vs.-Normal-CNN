import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_utils import get_colored_mnist


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            correct += (model(data).argmax(dim=1) == target).sum().item()
            total += target.size(0)
    return 100 * correct / total


def train_standard(model, loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            correct += (output.argmax(dim=1) == target).sum().item()
            total += target.size(0)
            running_loss += loss.item()

        print(f"Epoch {epoch + 1} | Train Acc: {100 * correct / total:.2f}% | Loss: {running_loss / len(loader):.4f}")
    return model


def run_shortcut_experiment(build_model, name, epochs, lr, batch_size, out):
    """Train on colored MNIST where color always matches the label, test on random colors."""
    parser = argparse.ArgumentParser(description=f"Train {name} on colored MNIST (color == label).")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default=out)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Running on: {device}")

    train_loader = DataLoader(get_colored_mnist(train=True, data_dir=args.data_dir), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(get_colored_mnist(train=False, data_dir=args.data_dir), batch_size=256)

    print(f"\n--- Training {name} ---")
    model = train_standard(build_model().to(device), train_loader, args.epochs, args.lr, device)
    print(f"\nTest Accuracy (random colors): {evaluate(model, test_loader, device):.2f}%")
    save_model(model, args.out)
