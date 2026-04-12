# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1: Autoencoders (Vanilla, Denoising, VAE, Convolutional)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build four autoencoder variants with torch.nn.Module: Vanilla, Denoising,
#     Variational (VAE), and Convolutional
#   - Train each with torch.optim.Adam on small synthetic image data
#   - Explain the VAE reparameterisation trick and why it enables backprop
#   - Sample new images from the VAE's latent Gaussian prior
#   - Visualise training curves with kailash-ml's ModelVisualizer
#
# PREREQUISITES: M4 (neural network basics, loss functions, optimisers).
# ESTIMATED TIME: ~60 min
# DATASET: Synthetic 16x16 "digit-like" images generated with numpy.
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from kailash_ml import ModelVisualizer

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# Synthetic image data — 16x16 grayscale blobs
# ════════════════════════════════════════════════════════════════════════
# Real autoencoders typically train on Fashion-MNIST or MNIST. To keep this
# exercise fast, we generate 2000 tiny images where each image is a 2-D
# Gaussian blob at a random location with random width. This gives the
# encoder something meaningful to compress.
def make_image_dataset(n_samples: int = 2000, size: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centers = np.random.uniform(4, size - 4, size=(n_samples, 2)).astype(np.float32)
    widths = np.random.uniform(1.5, 3.0, size=n_samples).astype(np.float32)
    cx = centers[:, 0][:, None, None]
    cy = centers[:, 1][:, None, None]
    w = widths[:, None, None]
    imgs = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * w ** 2))
    imgs += 0.02 * np.random.randn(*imgs.shape).astype(np.float32)
    return imgs.astype(np.float32)  # (n, 16, 16)


X_np = make_image_dataset(n_samples=2000, size=16)
X_flat = torch.from_numpy(X_np.reshape(len(X_np), -1)).to(device)        # (n, 256)
X_img = torch.from_numpy(X_np[:, None, :, :]).to(device)                  # (n, 1, 16, 16)

flat_loader = DataLoader(TensorDataset(X_flat), batch_size=64, shuffle=True)
img_loader = DataLoader(TensorDataset(X_img), batch_size=64, shuffle=True)

INPUT_DIM = 16 * 16
LATENT_DIM = 8


# ════════════════════════════════════════════════════════════════════════
# PART 1 — Vanilla Autoencoder
# ════════════════════════════════════════════════════════════════════════
# encoder: 256 -> 64 -> 8  |  decoder: 8 -> 64 -> 256
# Loss: MSE between reconstruction and original.
class VanillaAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


# ════════════════════════════════════════════════════════════════════════
# PART 2 — Denoising Autoencoder (DAE)
# ════════════════════════════════════════════════════════════════════════
# Identical architecture, but during training we corrupt the INPUT with
# noise and ask the decoder to reconstruct the CLEAN original. This forces
# the encoder to learn robust features rather than memorise pixel patterns.
class DenoisingAE(VanillaAE):
    def add_noise(self, x: torch.Tensor, sigma: float = 0.3) -> torch.Tensor:
        return torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════
# PART 3 — Variational Autoencoder (VAE)
# ════════════════════════════════════════════════════════════════════════
# The encoder outputs two vectors: mu and log_var. Instead of a single
# latent point, each input maps to a Gaussian N(mu, sigma^2). We sample
# z = mu + sigma * epsilon (the "reparameterisation trick") so that the
# randomness lives in epsilon and gradients can still flow through mu/sigma.
#
# Loss = reconstruction_loss + KL(q(z|x) || N(0, I))
#      = ||x - x_hat||^2 - 0.5 * sum(1 + log_var - mu^2 - exp(log_var))
#
# After training, sampling z ~ N(0, I) and decoding produces NEW images.
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.fc_mu.out_features, device=next(self.parameters()).device)
        return self.decoder(z)


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + 0.1 * kl  # small KL weight for clearer reconstruction


# ════════════════════════════════════════════════════════════════════════
# PART 4 — Convolutional Autoencoder
# ════════════════════════════════════════════════════════════════════════
# For image data, conv layers preserve spatial locality better than flat
# MLPs. Encoder: Conv2d -> Conv2d -> Flatten -> Linear.
# Decoder: Linear -> Unflatten -> ConvTranspose2d -> ConvTranspose2d.
class ConvAE(nn.Module):
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        # 1x16x16 -> 16x8x8 -> 32x4x4 -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


# ════════════════════════════════════════════════════════════════════════
# Training loops
# ════════════════════════════════════════════════════════════════════════
def train_ae(model: nn.Module, name: str, loader, epochs: int = 6, lr: float = 1e-3) -> list[float]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for epoch in range(epochs):
        batch_losses = []
        for (xb,) in loader:
            opt.zero_grad()
            x_hat, _ = model(xb)
            loss = F.mse_loss(x_hat, xb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))
        print(f"  [{name}] epoch {epoch+1}  loss={losses[-1]:.4f}")
    return losses


def train_dae(model: DenoisingAE, loader, epochs: int = 6, lr: float = 1e-3) -> list[float]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for epoch in range(epochs):
        batch_losses = []
        for (xb,) in loader:
            noisy = model.add_noise(xb, sigma=0.3)
            opt.zero_grad()
            x_hat, _ = model(noisy)
            loss = F.mse_loss(x_hat, xb)   # reconstruct CLEAN target
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))
        print(f"  [DAE] epoch {epoch+1}  loss={losses[-1]:.4f}")
    return losses


def train_vae(model: VAE, loader, epochs: int = 6, lr: float = 1e-3) -> list[float]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for epoch in range(epochs):
        batch_losses = []
        for (xb,) in loader:
            opt.zero_grad()
            x_hat, mu, logvar = model(xb)
            loss = vae_loss(xb, x_hat, mu, logvar)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))
        print(f"  [VAE] epoch {epoch+1}  loss={losses[-1]:.4f}")
    return losses


# ── Train all four ──────────────────────────────────────────────────────
print("\n── Vanilla AE ──")
vae_model_plain = VanillaAE(INPUT_DIM, LATENT_DIM)
vanilla_losses = train_ae(vae_model_plain, "Vanilla", flat_loader, epochs=6)

print("\n── Denoising AE ──")
dae_model = DenoisingAE(INPUT_DIM, LATENT_DIM)
dae_losses = train_dae(dae_model, flat_loader, epochs=6)

print("\n── Variational AE ──")
vae_model = VAE(INPUT_DIM, LATENT_DIM)
vae_losses = train_vae(vae_model, flat_loader, epochs=6)

print("\n── Convolutional AE ──")
conv_model = ConvAE(LATENT_DIM)
conv_losses = train_ae(conv_model, "ConvAE", img_loader, epochs=6)


# ════════════════════════════════════════════════════════════════════════
# PART 5 — Sample new images from the VAE latent prior
# ════════════════════════════════════════════════════════════════════════
vae_model.eval()
with torch.no_grad():
    samples = vae_model.sample(n=16).cpu().numpy().reshape(-1, 16, 16)
print(
    f"\nSampled {len(samples)} new images from VAE prior N(0, I). "
    f"mean pixel intensity: {samples.mean():.3f}, range: [{samples.min():.3f}, {samples.max():.3f}]"
)


# ════════════════════════════════════════════════════════════════════════
# PART 6 — Visualise training histories
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()
fig = viz.training_history(
    metrics={
        "Vanilla AE": vanilla_losses,
        "Denoising AE": dae_losses,
        "Variational AE": vae_losses,
        "Convolutional AE": conv_losses,
    },
    x_label="Epoch",
    y_label="Loss",
)
fig.write_html("ex_1_training.html")
print("Training history saved to ex_1_training.html")


# ── Reflection ─────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Built four autoencoder variants with nn.Module (Vanilla, DAE, VAE, ConvAE)
  [x] Trained each end-to-end with Adam and PyTorch autograd
  [x] Applied the VAE reparameterisation trick (z = mu + sigma * epsilon)
  [x] Sampled brand-new images from the VAE's latent Gaussian prior
  [x] Compared training curves across all four variants
"""
)
