from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualGenerator(nn.Module):
    def __init__(self, noise_dim: int, condition_dim: int, residual_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, residual_dim),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, condition], dim=1))


class ResidualDiscriminator(nn.Module):
    def __init__(self, condition_dim: int, residual_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(residual_dim + condition_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, residual: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([residual, condition], dim=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional GAN on v2 physics residuals.")
    parser.add_argument("--dataset", type=Path, default=Path("model/v2/input/v2_residual_gan_dataset.pth"))
    parser.add_argument("--output", type=Path, default=Path("model/v2/output/v2_residual_gan.pt"))
    parser.add_argument("--metrics-json", type=Path, default=Path("model/v2/output/v2_residual_gan_metrics.json"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--noise-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    payload = torch.load(args.dataset, map_location="cpu")
    residual = payload["residual"].float()
    condition = payload["condition"].float()
    residual_mean = residual.mean(dim=0, keepdim=True)
    residual_std = residual.std(dim=0, keepdim=True).clamp_min(1e-6)
    residual_norm = (residual - residual_mean) / residual_std

    dataset = TensorDataset(residual_norm, condition)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_dim = int(residual.shape[1])
    condition_dim = int(condition.shape[1])
    generator = ResidualGenerator(args.noise_dim, condition_dim, residual_dim).to(device)
    discriminator = ResidualDiscriminator(condition_dim, residual_dim).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        g_losses = []
        d_losses = []
        for real, cond in loader:
            real = real.to(device)
            cond = cond.to(device)
            batch = real.shape[0]

            z = torch.randn(batch, args.noise_dim, device=device)
            fake = generator(z, cond).detach()
            real_logits = discriminator(real, cond)
            fake_logits = discriminator(fake, cond)
            d_loss = criterion(real_logits, torch.ones_like(real_logits)) + criterion(fake_logits, torch.zeros_like(fake_logits))
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            z = torch.randn(batch, args.noise_dim, device=device)
            fake = generator(z, cond)
            fake_logits = discriminator(fake, cond)
            g_loss = criterion(fake_logits, torch.ones_like(fake_logits))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            d_losses.append(float(d_loss.item()))
            g_losses.append(float(g_loss.item()))

        row = {
            "epoch": epoch,
            "d_loss": float(np.mean(d_losses)) if d_losses else 0.0,
            "g_loss": float(np.mean(g_losses)) if g_losses else 0.0,
        }
        history.append(row)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs}: d_loss={row['d_loss']:.4f}, g_loss={row['g_loss']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "noise_dim": args.noise_dim,
            "condition_dim": condition_dim,
            "residual_dim": residual_dim,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "metadata": payload.get("metadata", {}),
        },
        args.output,
    )
    args.metrics_json.write_text(json.dumps({"history": history[-20:], "epochs": args.epochs}, indent=2), encoding="utf-8")
    print(f"Saved GAN -> {args.output}")


if __name__ == "__main__":
    main()
