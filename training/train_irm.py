import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models_loader.irm import IRM_CNN, compute_irm_penalty
from training.common import evaluate, get_device, save_model
from utils.data_utils import get_biased_mnist_env


def train_on_environments(model, loader_env1, loader_env2, epochs, penalty_weight, warmup, lr, device):
    """IRM training on two environments. penalty_weight=0 gives the ERM baseline."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dummy_w = torch.tensor(1.0, device=device, requires_grad=True)

    for epoch in range(epochs):
        weight = penalty_weight if epoch >= warmup else 0.0
        if epoch == warmup and penalty_weight > 0:
            # loss scale changes once the penalty kicks in, so reset Adam
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        total_risk, total_penalty, n_batches = 0.0, 0.0, 0
        for (x1, y1), (x2, y2) in zip(loader_env1, loader_env2):
            x1, y1, x2, y2 = x1.to(device), y1.to(device), x2.to(device), y2.to(device)

            loss1 = F.cross_entropy(model(x1) * dummy_w, y1)
            loss2 = F.cross_entropy(model(x2) * dummy_w, y2)
            risk = loss1 + loss2
            penalty = compute_irm_penalty(loss1, loss2, dummy_w)

            loss = risk + weight * penalty if weight > 0 else risk
            if weight > 1.0:
                loss = loss / weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_risk += risk.item()
            total_penalty += penalty.item()
            n_batches += 1

        print(f"Epoch {epoch + 1:2d} | weight {weight:6.0f} | "
              f"Risk: {total_risk / n_batches:.4f} | Penalty: {total_penalty / n_batches:.6f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train an IRM CNN on two colored MNIST environments.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=6, help="epochs of plain ERM before the penalty is switched on")
    parser.add_argument("--penalty-weight", type=float, default=1000.0)
    parser.add_argument("--label-noise", type=float, default=0.0, help="0.25 reproduces the setting from the IRM paper")
    parser.add_argument("--compare-erm", action="store_true", help="also train an ERM baseline on the same environments")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="models/IRM_cnn.pth")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Running on: {device}")

    def make_env(correlation_prob, train, subset=slice(None)):
        return get_biased_mnist_env(correlation_prob, train=train, subset=subset,
                                    label_noise=args.label_noise, data_dir=args.data_dir)

    env1 = make_env(0.90, train=True, subset=slice(0, 30000))
    env2 = make_env(0.80, train=True, subset=slice(30000, 60000))
    loader_env1 = DataLoader(env1, batch_size=args.batch_size, shuffle=True)
    loader_env2 = DataLoader(env2, batch_size=args.batch_size, shuffle=True)
    eval_loaders = {
        "Env 1 (train)": DataLoader(env1, batch_size=256),
        "Test, 10% corr.": DataLoader(make_env(0.10, train=False), batch_size=256),
        "Test, random colors": DataLoader(make_env(0.0, train=False), batch_size=256),
    }

    def train(penalty_weight, name):
        print(f"\n--- Training {name} ---")
        torch.manual_seed(args.seed)
        return train_on_environments(IRM_CNN().to(device), loader_env1, loader_env2, args.epochs,
                                     penalty_weight, args.warmup, args.lr, device)

    trained = {"IRM": train(args.penalty_weight, "IRM")}
    save_model(trained["IRM"], args.out)
    if args.compare_erm:
        trained["ERM"] = train(0.0, "ERM baseline")

    print("\n" + f"{'Model':<8}" + "".join(f"{name:>22}" for name in eval_loaders))
    for name, model in trained.items():
        print(f"{name:<8}" + "".join(f"{evaluate(model, loader, device):>21.2f}%" for loader in eval_loaders.values()))


if __name__ == "__main__":
    main()
