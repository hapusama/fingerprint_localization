from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from residual_gan import ResidualDiscriminator, ResidualGenerator


class Locator(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stratified_split(labels: np.ndarray, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(labels.shape[0])
    counts = np.bincount(labels[labels >= 0])
    stratify = labels if counts.size and np.min(counts[counts > 0]) >= 2 else None
    return train_test_split(indices, test_size=test_frac, random_state=seed, stratify=stratify)


def make_loader(features: torch.Tensor, labels: torch.Tensor, indices: Iterable[int], batch_size: int, shuffle: bool) -> DataLoader:
    idx = torch.tensor(list(indices), dtype=torch.long)
    return DataLoader(TensorDataset(features[idx], labels[idx]), batch_size=batch_size, shuffle=shuffle)


def train_gan(
    residual: torch.Tensor,
    condition: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ResidualGenerator, torch.Tensor, torch.Tensor]:
    residual_mean = residual.mean(dim=0, keepdim=True)
    residual_std = residual.std(dim=0, keepdim=True).clamp_min(1e-6)
    residual_norm = (residual - residual_mean) / residual_std
    loader = DataLoader(TensorDataset(residual_norm, condition), batch_size=args.batch_size, shuffle=True)
    generator = ResidualGenerator(args.noise_dim, condition.shape[1], residual.shape[1]).to(device)
    discriminator = ResidualDiscriminator(condition.shape[1], residual.shape[1]).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.gan_epochs + 1):
        d_losses = []
        g_losses = []
        for real, cond in loader:
            real = real.to(device)
            cond = cond.to(device)
            batch = real.shape[0]
            z = torch.randn(batch, args.noise_dim, device=device)
            fake = generator(z, cond).detach()
            d_real = discriminator(real, cond)
            d_fake = discriminator(fake, cond)
            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            z = torch.randn(batch, args.noise_dim, device=device)
            fake = generator(z, cond)
            d_fake = discriminator(fake, cond)
            g_loss = criterion(d_fake, torch.ones_like(d_fake))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            d_losses.append(float(d_loss.item()))
            g_losses.append(float(g_loss.item()))

        if epoch == 1 or epoch % 25 == 0 or epoch == args.gan_epochs:
            print(f"GAN epoch {epoch:03d}/{args.gan_epochs}: d_loss={np.mean(d_losses):.4f}, g_loss={np.mean(g_losses):.4f}")
    return generator, residual_mean.to(device), residual_std.to(device)


def train_locator(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model = Locator(train_x.shape[1], int(max(train_y.max(), test_y.max()).item()) + 1).to(device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.locator_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_acc = -1.0
    for epoch in range(1, args.locator_epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 25 == 0 or epoch == args.locator_epochs:
            acc = evaluate_locator(model, test_x, test_y, device)["accuracy"]
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"Locator epoch {epoch:03d}/{args.locator_epochs}: loss={np.mean(losses):.4f}, test_acc={acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate_locator(model, test_x, test_y, device)


def evaluate_locator(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(x.to(device)), dim=1).cpu().numpy()
    true = y.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(true, pred, average="macro", zero_division=0)),
    }


def build_augmented_samples(
    generator: ResidualGenerator,
    x_phy: torch.Tensor,
    condition: torch.Tensor,
    labels: torch.Tensor,
    train_idx: np.ndarray,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_x = x_phy[train_idx].to(device)
    base_c = condition[train_idx].to(device)
    base_y = labels[train_idx].to(device)
    aug_x = []
    aug_y = []
    generator.eval()
    with torch.no_grad():
        for _ in range(args.augment_per_sample):
            z = torch.randn(base_x.shape[0], args.noise_dim, device=device)
            residual_norm = generator(z, base_c)
            residual = residual_norm * residual_std + residual_mean
            aug_x.append((base_x + residual).cpu())
            aug_y.append(base_y.cpu())
    return torch.cat(aug_x, dim=0), torch.cat(aug_y, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a v2 locator with real RSSI+ and GAN residual augmentation.")
    parser.add_argument("--dataset", type=Path, default=Path("model/v2/input/v2_residual_gan_dataset.pth"))
    parser.add_argument("--metrics-json", type=Path, default=Path("model/v2/output/v2_gan_augmented_locator_metrics.json"))
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--noise-dim", type=int, default=16)
    parser.add_argument("--gan-epochs", type=int, default=50)
    parser.add_argument("--locator-epochs", type=int, default=120)
    parser.add_argument("--gan-lr", type=float, default=2e-4)
    parser.add_argument("--locator-lr", type=float, default=1e-3)
    parser.add_argument("--augment-per-sample", type=int, default=1)
    parser.add_argument("--max-samples-per-class", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    payload = torch.load(args.dataset, map_location="cpu")
    x_real = payload["x_real"].float()
    x_phy = payload["x_phy"].float()
    residual = payload["residual"].float()
    condition = payload["condition"].float()
    labels = payload["label"].long()
    keep = labels >= 0
    x_real = x_real[keep]
    x_phy = x_phy[keep]
    residual = residual[keep]
    condition = condition[keep]
    labels = labels[keep]

    selected = []
    rng = np.random.default_rng(args.seed)
    labels_np = labels.numpy()
    for label in sorted(np.unique(labels_np).tolist()):
        idx = np.flatnonzero(labels_np == label)
        rng.shuffle(idx)
        selected.extend(idx[: args.max_samples_per_class].tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    x_real = x_real[selected]
    x_phy = x_phy[selected]
    residual = residual[selected]
    condition = condition[selected]
    labels = labels[selected]
    labels_np = labels.numpy()

    train_idx, test_idx = stratified_split(labels_np, args.test_frac, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, residual_mean, residual_std = train_gan(residual[train_idx], condition[train_idx], args, device)
    aug_x, aug_y = build_augmented_samples(generator, x_phy, condition, labels, train_idx, residual_mean, residual_std, args, device)

    real_train_x = x_real[train_idx]
    real_train_y = labels[train_idx]
    test_x = x_real[test_idx]
    test_y = labels[test_idx]
    real_only = train_locator(real_train_x, real_train_y, test_x, test_y, args, device)
    mixed_train_x = torch.cat([real_train_x, aug_x], dim=0)
    mixed_train_y = torch.cat([real_train_y, aug_y], dim=0)
    real_plus_gan = train_locator(mixed_train_x, mixed_train_y, test_x, test_y, args, device)

    metrics = {
        "dataset": str(args.dataset),
        "samples_used": int(labels.shape[0]),
        "train": int(len(train_idx)),
        "test": int(len(test_idx)),
        "augment_per_sample": int(args.augment_per_sample),
        "real_only": real_only,
        "real_plus_gan": real_plus_gan,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
