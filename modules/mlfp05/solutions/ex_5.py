# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 5: Generative Models (GAN and WGAN-GP)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build Generator and Discriminator MLPs in torch.nn
#   - Train a vanilla GAN with binary cross-entropy (min-max game)
#   - Train a WGAN-GP using Wasserstein loss and a gradient penalty
#   - Compute a gradient penalty with torch.autograd.grad and explain why
#     it replaces weight clipping from the original WGAN paper
#   - Compare samples visually: both generators should learn the target
#     2-D distribution
#
# PREREQUISITES: M5/ex_1 (autoencoders/VAE), M5/ex_2 (CNN training loops).
# ESTIMATED TIME: ~60 min
# DATASET: Synthetic "eight Gaussians on a ring" 2-D distribution.
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kailash_ml import ModelVisualizer

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# Target distribution — 8 Gaussians on a ring (classic GAN toy problem)
# ════════════════════════════════════════════════════════════════════════
# Learning this distribution is a classic GAN benchmark because vanilla
# GANs suffer "mode collapse" on it — they lock onto a single Gaussian and
# ignore the others. WGAN-GP is designed to avoid this.
def sample_target(n: int) -> torch.Tensor:
    modes = np.array(
        [(math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)) for k in range(8)],
        dtype=np.float32,
    )
    idx = np.random.randint(0, 8, size=n)
    centers = modes[idx]
    samples = centers + 0.05 * np.random.randn(n, 2).astype(np.float32)
    return torch.from_numpy(samples)


# ════════════════════════════════════════════════════════════════════════
# PART 1 — Generator and Discriminator
# ════════════════════════════════════════════════════════════════════════
LATENT_DIM = 8


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """For a vanilla GAN this is a probability; for WGAN-GP it is a critic
    (unbounded scalar). We keep a single class and use the output directly
    where needed."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════
# PART 2 — Train a vanilla GAN with binary cross-entropy
# ════════════════════════════════════════════════════════════════════════
# Vanilla GAN loss (Goodfellow 2014):
#   L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
#   L_G = -E[log D(G(z))]
# We use BCEWithLogitsLoss so the discriminator output is a logit, not a
# probability — numerically safer than a sigmoid + BCE combination.
def train_vanilla_gan(
    steps: int = 1500,
    batch: int = 128,
    lr: float = 2e-4,
) -> tuple[Generator, list[float], list[float]]:
    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    g_losses: list[float] = []
    d_losses: list[float] = []

    for step in range(steps):
        # ── Discriminator step ────────────────────────────────────────
        real = sample_target(batch).to(device)
        z = torch.randn(batch, LATENT_DIM, device=device)
        fake = G(z).detach()
        d_real = D(real)
        d_fake = D(fake)
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # ── Generator step ────────────────────────────────────────────
        z = torch.randn(batch, LATENT_DIM, device=device)
        fake = G(z)
        d_fake_for_g = D(fake)
        loss_g = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))  # fool D
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        d_losses.append(loss_d.item())
        g_losses.append(loss_g.item())

        if (step + 1) % 300 == 0:
            print(f"  [GAN] step {step+1:4d}  loss_D={loss_d.item():.3f}  loss_G={loss_g.item():.3f}")

    return G, g_losses, d_losses


# ════════════════════════════════════════════════════════════════════════
# PART 3 — Train a WGAN with Gradient Penalty
# ════════════════════════════════════════════════════════════════════════
# Wasserstein loss:
#   L_critic    = E[D(fake)] - E[D(real)] + lambda * GP
#   L_generator = -E[D(fake)]
#
# Gradient penalty (Gulrajani 2017): sample a random point on the line
# between a real and a fake sample, compute the gradient of D at that
# point, and penalise its L2 norm for deviating from 1. This enforces the
# 1-Lipschitz constraint WGAN requires, and replaces the original weight
# clipping trick which caused training pathologies.
def gradient_penalty(D: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    batch = real.size(0)
    alpha = torch.rand(batch, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp)
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(
    steps: int = 1500,
    batch: int = 128,
    lr: float = 1e-4,
    n_critic: int = 3,
    lam: float = 10.0,
) -> tuple[Generator, list[float], list[float]]:
    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    g_losses: list[float] = []
    d_losses: list[float] = []

    for step in range(steps):
        # ── Critic updates (n_critic per generator update) ────────────
        for _ in range(n_critic):
            real = sample_target(batch).to(device)
            z = torch.randn(batch, LATENT_DIM, device=device)
            fake = G(z).detach()
            d_real = D(real).mean()
            d_fake = D(fake).mean()
            gp = gradient_penalty(D, real, fake)
            loss_d = d_fake - d_real + lam * gp
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        # ── Generator update ──────────────────────────────────────────
        z = torch.randn(batch, LATENT_DIM, device=device)
        fake = G(z)
        loss_g = -D(fake).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        d_losses.append(loss_d.item())
        g_losses.append(loss_g.item())

        if (step + 1) % 300 == 0:
            print(f"  [WGAN-GP] step {step+1:4d}  loss_D={loss_d.item():.3f}  loss_G={loss_g.item():.3f}")

    return G, g_losses, d_losses


# ── Train both ─────────────────────────────────────────────────────────
print("\n── Training Vanilla GAN ──")
G_gan, gan_g_losses, gan_d_losses = train_vanilla_gan(steps=1500)

print("\n── Training WGAN-GP ──")
G_wgan, wgan_g_losses, wgan_d_losses = train_wgan_gp(steps=1500)


# ════════════════════════════════════════════════════════════════════════
# PART 4 — Evaluate mode coverage
# ════════════════════════════════════════════════════════════════════════
# For the 8-Gaussians target, count how many of the 8 modes each generator
# produces samples near. A mode-collapsed model hits 1-2; a healthy one
# hits all 8.
def mode_coverage(G: Generator, n: int = 2000, tol: float = 0.3) -> int:
    G.eval()
    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM, device=device)
        samples = G(z).cpu().numpy()
    modes = np.array(
        [(math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)) for k in range(8)],
        dtype=np.float32,
    )
    hit = np.zeros(8, dtype=bool)
    dists = np.linalg.norm(samples[:, None, :] - modes[None, :, :], axis=-1)
    closest = dists.argmin(axis=1)
    min_dists = dists.min(axis=1)
    for idx, d in zip(closest, min_dists):
        if d < tol:
            hit[idx] = True
    return int(hit.sum())


print(f"\nVanilla GAN covered {mode_coverage(G_gan)} / 8 modes")
print(f"WGAN-GP    covered {mode_coverage(G_wgan)} / 8 modes")


# ════════════════════════════════════════════════════════════════════════
# PART 5 — Visualise both loss curves
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()
fig = viz.training_history(
    metrics={
        "GAN G loss": gan_g_losses,
        "GAN D loss": gan_d_losses,
        "WGAN G loss": wgan_g_losses,
        "WGAN D loss": wgan_d_losses,
    },
    x_label="Step",
    y_label="Loss",
)
fig.write_html("ex_5_training.html")
print("Training history saved to ex_5_training.html")


# ── Reflection ─────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Built Generator and Discriminator with nn.Module
  [x] Trained a vanilla GAN with BCEWithLogitsLoss (min-max game)
  [x] Implemented a WGAN-GP with torch.autograd.grad for the gradient penalty
  [x] Measured mode coverage on the 8-Gaussians benchmark
  [x] Compared training dynamics between vanilla GAN and WGAN-GP
"""
)
print(
    "\nNote: outcomes vary by seed and step budget. In the full 8-Gaussians\n"
    "paper benchmark (Gulrajani 2017) WGAN-GP eventually covers all 8 modes\n"
    "and recovers faster after any collapse, but this short 1500-step run\n"
    "is mainly intended to show the API and the gradient-penalty math.\n"
)
