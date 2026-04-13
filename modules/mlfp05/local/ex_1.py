# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1: Autoencoders (Vanilla, Denoising, VAE, Convolutional)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build four autoencoder variants with torch.nn.Module: Vanilla,
#     Denoising, Variational (VAE), and Convolutional
#   - Train each with torch.optim.Adam on FULL Fashion-MNIST (60K images)
#   - Explain the VAE reparameterisation trick and why it enables backprop
#   - Sample new images from the VAE's latent Gaussian prior
#   - Track every training run with kailash-ml's ExperimentTracker
#     (parameters, per-epoch loss, tags)
#   - Register the best model from each variant in the ModelRegistry
#   - Compare reconstruction quality across latent dimensions (4, 8, 16, 32)
#   - Visualise training curves with kailash-ml's ModelVisualizer
#
# PREREQUISITES: M4.8 (neural network basics, loss functions, optimisers).
# ESTIMATED TIME: ~120-150 min
#
# DATASET: Fashion-MNIST — 60,000 real 28x28 grayscale clothing images
#   across 10 classes (t-shirt, trouser, pullover, dress, coat, sandal,
#   shirt, sneaker, bag, ankle boot). Downloaded automatically by
#   torchvision and cached to data/mlfp05/fashion_mnist/. We use the FULL
#   60K training set — autoencoders benefit from large datasets because
#   they need to learn generalisable compressed representations.
#
# TASKS:
#   1. Load full Fashion-MNIST, set up ExperimentTracker and ModelRegistry
#   2. Build and train a Vanilla Autoencoder, log run to tracker
#   3. Build and train a Denoising Autoencoder, log run to tracker
#   4. Build and train a Variational Autoencoder, log run to tracker
#   5. Build and train a Convolutional Autoencoder, log run to tracker
#   6. Sample new images from the VAE latent prior
#   7. Register the best model from each variant in the ModelRegistry
#   8. Hyperparameter comparison: try latent dims 4, 8, 16, 32
#   9. Visualise all training histories and compare
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.model_registry import ModelRegistry

from shared.kailash_helpers import get_device, setup_environment

setup_environment()

torch.manual_seed(42)
np.random.seed(42)
device = get_device()
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Fashion-MNIST (full 60K) and set up kailash-ml engines
# ════════════════════════════════════════════════════════════════════════
# Real autoencoders train on the full dataset. Sub-sampling weakens the
# encoder's ability to learn general features — with only 6K images, it
# memorises patches instead of learning structure.

REPO_ROOT = Path.cwd()
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "fashion_mnist"
DATA_DIR.mkdir(parents=True, exist_ok=True)

train_set = torchvision.datasets.FashionMNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),  # already in [0, 1]
)

# Use the FULL 60K training set. On MPS/CUDA this trains in ~2 min per
# variant; on CPU ~8-10 min per variant.
X_img = torch.stack(
    [train_set[i][0] for i in range(len(train_set))]
)  # (60000, 1, 28, 28)
X_img = X_img.to(device).float()
X_flat = X_img.reshape(len(X_img), -1)  # (60000, 784)

print(
    f"Fashion-MNIST: {len(X_img)} images, shape {tuple(X_img.shape[1:])}, "
    f"pixel range [{X_img.min():.2f}, {X_img.max():.2f}]"
)

flat_loader = DataLoader(TensorDataset(X_flat), batch_size=256, shuffle=True)
img_loader = DataLoader(TensorDataset(X_img), batch_size=256, shuffle=True)

INPUT_DIM = 28 * 28
LATENT_DIM = 16
EPOCHS = 10


# Set up kailash-ml engines: ExperimentTracker + ModelRegistry.
# Both use ConnectionManager for SQLite-backed persistence.
async def setup_engines():
    # TODO: Create a ConnectionManager with "sqlite:///mlfp05_autoencoders.db"
    # Hint: ConnectionManager(url), then await conn.initialize()
    conn = ____  # Hint: ConnectionManager("sqlite:///mlfp05_autoencoders.db")
    await conn.initialize()

    # TODO: Create an ExperimentTracker and create an experiment called "m5_autoencoders"
    # Hint: ExperimentTracker(conn), then tracker.create_experiment(name=..., description=...)
    tracker = ____  # Hint: ExperimentTracker(conn)
    exp_name = await tracker.create_experiment(
        name=____,  # Hint: "m5_autoencoders"
        description=____,  # Hint: "Autoencoder variants on Fashion-MNIST (60K images)"
    )

    try:
        # TODO: Create a ModelRegistry using the same conn
        # Hint: ModelRegistry(conn)
        registry = ____
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


conn, tracker, exp_name, registry, has_registry = asyncio.run(setup_engines())

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_img.shape[0] == 60000, (
    f"Expected full 60K Fashion-MNIST, got {X_img.shape[0]}. "
    "Autoencoders need the full dataset to learn general features."
)
assert X_flat.shape == (60000, 784), "Flattened tensor should be (60000, 784)"
assert tracker is not None, "ExperimentTracker should be initialised"
assert exp_name is not None, "Experiment should be created"
# INTERPRETATION: We use all 60K images because autoencoders are
# unsupervised — every image teaches the encoder about clothing structure.
# Sub-sampling to 6K would be like learning to draw from only 6K examples
# instead of 60K. The ExperimentTracker will log every training run so we
# can compare variants systematically, not by eyeballing print statements.
print("\n--- Checkpoint 1 passed --- data loaded and engines initialised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Vanilla Autoencoder
# ════════════════════════════════════════════════════════════════════════
# encoder: 784 -> 256 -> 64 -> 16  |  decoder: 16 -> 64 -> 256 -> 784
# Loss: MSE between reconstruction and original.
class VanillaAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # TODO: Build self.encoder as nn.Sequential:
        #   Linear(input_dim -> 256) -> ReLU -> Linear(256 -> 64) -> ReLU -> Linear(64 -> latent_dim)
        # Hint: nn.Sequential(nn.Linear(...), nn.ReLU(), ...)
        self.encoder = ____

        # TODO: Build self.decoder as nn.Sequential:
        #   Linear(latent_dim -> 64) -> ReLU -> Linear(64 -> 256) -> ReLU -> Linear(256 -> input_dim) -> Sigmoid
        # Hint: mirror the encoder, add nn.Sigmoid() at end (outputs in [0,1])
        self.decoder = ____

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Denoising Autoencoder (DAE)
# ════════════════════════════════════════════════════════════════════════
# Identical architecture, but during training we corrupt the INPUT with
# Gaussian noise and ask the decoder to reconstruct the CLEAN original.
class DenoisingAE(VanillaAE):
    def add_noise(self, x: torch.Tensor, sigma: float = 0.3) -> torch.Tensor:
        # TODO: Add Gaussian noise scaled by sigma and clamp to [0, 1]
        # Hint: torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)
        return ____


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Variational Autoencoder (VAE)
# ════════════════════════════════════════════════════════════════════════
# The encoder outputs two vectors: mu and log_var. The reparameterisation
# trick: z = mu + sigma * epsilon, so gradients flow through mu/sigma.
# Loss = reconstruction_loss + KL(q(z|x) || N(0, I))
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # Shared encoder backbone (outputs 64-dim hidden)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        # TODO: Define fc_mu and fc_logvar as Linear(64 -> latent_dim)
        # Hint: nn.Linear(64, latent_dim)
        self.fc_mu = ____
        self.fc_logvar = ____

        # TODO: Build self.decoder as nn.Sequential:
        #   Linear(latent_dim -> 64) -> ReLU -> Linear(64 -> 256) -> ReLU -> Linear(256 -> input_dim) -> Sigmoid
        # Hint: same structure as VanillaAE decoder
        self.decoder = ____

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: Implement reparameterisation: z = mu + exp(0.5 * logvar) * epsilon
        # Hint: std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
        std = ____
        eps = ____
        return ____

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def sample(self, n: int) -> torch.Tensor:
        # TODO: Sample n latent vectors from N(0,I) and decode them
        # Hint: z = torch.randn(n, self.fc_mu.out_features, device=...)
        z = ____
        return self.decoder(z)


def vae_loss(
    x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    # TODO: Compute VAE loss = reconstruction MSE + 0.1 * KL divergence
    # Hint: recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    #        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    recon = ____
    kl = ____
    return recon + 0.1 * kl


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Convolutional Autoencoder
# ════════════════════════════════════════════════════════════════════════
# For image data, conv layers preserve spatial locality better than flat
# MLPs. Encoder: Conv2d -> Conv2d -> Flatten -> Linear.
# Decoder: Linear -> Unflatten -> ConvTranspose2d -> ConvTranspose2d.
class ConvAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        # TODO: Build self.encoder as nn.Sequential:
        #   Conv2d(1->16, k=3, stride=2, padding=1) -> ReLU ->
        #   Conv2d(16->32, k=3, stride=2, padding=1) -> ReLU ->
        #   Flatten -> Linear(32*7*7 -> latent_dim)
        # Hint: nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.encoder = ____

        # TODO: Build self.decoder as nn.Sequential:
        #   Linear(latent_dim -> 32*7*7) -> ReLU ->
        #   Unflatten(1, (32, 7, 7)) ->
        #   ConvTranspose2d(32->16, k=3, stride=2, padding=1, output_padding=1) -> ReLU ->
        #   ConvTranspose2d(16->1, k=3, stride=2, padding=1, output_padding=1) -> Sigmoid
        # Hint: nn.ConvTranspose2d(..., output_padding=1) to restore exact spatial dims
        self.decoder = ____

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


# ════════════════════════════════════════════════════════════════════════
# Training loops with ExperimentTracker integration
# ════════════════════════════════════════════════════════════════════════


async def train_ae_async(
    model: nn.Module,
    name: str,
    loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
) -> list[float]:
    """Train a standard autoencoder (Vanilla or Conv) and log to tracker.

    Uses the modern ``tracker.run(...)`` async context manager:
      * bulk param logging via ``ctx.log_params({...})``
      * per-epoch metric logging via ``ctx.log_metric(key, val, step=epoch)``
      * automatic COMPLETED/FAILED state on context exit — no manual
        ``end_run`` plumbing, no ``run_id`` threaded through every call.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    # TODO: Open the tracker context manager and log params in bulk
    # Hint: async with tracker.run(experiment_name=exp_name, run_name=name) as ctx:
    #           await ctx.log_params({"model_type": name, "latent_dim": str(LATENT_DIM), ...})
    async with tracker.run(experiment_name=exp_name, run_name=name) as ctx:
        await ctx.log_params(
            {
                "model_type": name,
                "latent_dim": str(LATENT_DIM),
                "epochs": str(epochs),
                "lr": str(lr),
                "dataset_size": str(len(loader.dataset)),
                "batch_size": str(loader.batch_size),
            }
        )

        for epoch in range(epochs):
            batch_losses = []
            for (xb,) in loader:
                opt.zero_grad()
                x_hat, _ = model(xb)
                # TODO: Compute MSE reconstruction loss and call loss.backward()
                # Hint: F.mse_loss(x_hat, xb)
                loss = ____
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
            epoch_loss = float(np.mean(batch_losses))
            losses.append(epoch_loss)
            # TODO: Log the epoch loss metric via the ctx object
            # Hint: await ctx.log_metric("loss", epoch_loss, step=epoch + 1)
            await ctx.log_metric("loss", epoch_loss, step=epoch + 1)
            print(f"  [{name}] epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")

        await ctx.log_metric("final_loss", losses[-1])

    return losses


def train_ae(
    model: nn.Module,
    name: str,
    loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
) -> list[float]:
    """Sync wrapper — one asyncio.run per training call."""
    return asyncio.run(train_ae_async(model, name, loader, epochs, lr))


async def train_dae_async(
    model: "DenoisingAE",
    loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
    sigma: float = 0.3,
) -> list[float]:
    """Train a denoising autoencoder and log to tracker."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    async with tracker.run(experiment_name=exp_name, run_name="denoising_ae") as ctx:
        await ctx.log_params(
            {
                "model_type": "DenoisingAE",
                "latent_dim": str(LATENT_DIM),
                "epochs": str(epochs),
                "lr": str(lr),
                "noise_sigma": str(sigma),
                "dataset_size": str(len(loader.dataset)),
            }
        )

        for epoch in range(epochs):
            batch_losses = []
            for (xb,) in loader:
                # TODO: Corrupt the input with noise using model.add_noise, then reconstruct
                # Hint: noisy = model.add_noise(xb, sigma=sigma); x_hat, _ = model(noisy)
                noisy = ____
                opt.zero_grad()
                x_hat, _ = model(noisy)
                # Reconstruct CLEAN target (xb), not the noisy version
                loss = F.mse_loss(x_hat, xb)
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
            epoch_loss = float(np.mean(batch_losses))
            losses.append(epoch_loss)
            await ctx.log_metric("loss", epoch_loss, step=epoch + 1)
            print(f"  [DAE] epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")

        await ctx.log_metric("final_loss", losses[-1])

    return losses


def train_dae(
    model: "DenoisingAE",
    loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
    sigma: float = 0.3,
) -> list[float]:
    return asyncio.run(train_dae_async(model, loader, epochs, lr, sigma))


async def train_vae_async(
    model: VAE,
    loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
) -> list[float]:
    """Train a variational autoencoder and log to tracker."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    recon_losses: list[float] = []
    kl_losses: list[float] = []

    async with tracker.run(experiment_name=exp_name, run_name="variational_ae") as ctx:
        await ctx.log_params(
            {
                "model_type": "VAE",
                "latent_dim": str(LATENT_DIM),
                "epochs": str(epochs),
                "lr": str(lr),
                "kl_weight": "0.1",
                "dataset_size": str(len(loader.dataset)),
            }
        )

        for epoch in range(epochs):
            batch_losses = []
            batch_recon = []
            batch_kl = []
            for (xb,) in loader:
                opt.zero_grad()
                # TODO: Forward pass returns (x_hat, mu, logvar), compute vae_loss
                # Hint: x_hat, mu, logvar = model(xb); loss = vae_loss(xb, x_hat, mu, logvar)
                x_hat, mu, logvar = ____
                loss = ____
                # Track components separately for analysis
                recon = F.mse_loss(x_hat, xb, reduction="sum") / xb.size(0)
                kl = (
                    -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
                )
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
                batch_recon.append(recon.item())
                batch_kl.append(kl.item())
            epoch_loss = float(np.mean(batch_losses))
            epoch_recon = float(np.mean(batch_recon))
            epoch_kl = float(np.mean(batch_kl))
            losses.append(epoch_loss)
            recon_losses.append(epoch_recon)
            kl_losses.append(epoch_kl)
            await ctx.log_metrics(
                {
                    "loss": epoch_loss,
                    "recon_loss": epoch_recon,
                    "kl_loss": epoch_kl,
                },
                step=epoch + 1,
            )
            print(
                f"  [VAE] epoch {epoch + 1}/{epochs}  "
                f"loss={epoch_loss:.4f}  recon={epoch_recon:.4f}  kl={epoch_kl:.4f}"
            )

        await ctx.log_metrics(
            {
                "final_loss": losses[-1],
                "final_recon_loss": recon_losses[-1],
                "final_kl_loss": kl_losses[-1],
            }
        )

    return losses


def train_vae(
    model: VAE,
    loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
) -> list[float]:
    return asyncio.run(train_vae_async(model, loader, epochs, lr))


# ── Train all four variants ───────────────────────────────────────────
print("\n== Vanilla AE ==")
vanilla_model = VanillaAE(INPUT_DIM, LATENT_DIM)
vanilla_losses = train_ae(vanilla_model, "vanilla_ae", flat_loader, epochs=EPOCHS)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(vanilla_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses"
assert vanilla_losses[-1] < vanilla_losses[0], "Loss should decrease during training"
# INTERPRETATION: The vanilla autoencoder learns a compressed 16-dim
# representation of 784-pixel images. The loss decreasing means the
# decoder is getting better at reconstructing from only 16 numbers.
# With 60K images, the encoder learns general clothing features (edges,
# curves, textures) rather than memorising specific items.
print("\n--- Checkpoint 2 passed --- vanilla AE trained\n")


print("\n== Denoising AE ==")
dae_model = DenoisingAE(INPUT_DIM, LATENT_DIM)
dae_losses = train_dae(dae_model, flat_loader, epochs=EPOCHS)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(dae_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses"
assert dae_losses[-1] < dae_losses[0], "DAE loss should decrease"
# INTERPRETATION: The DAE's loss will be higher than the vanilla AE's
# because it has to reconstruct from noisy inputs. But its learned
# features are MORE robust — the encoder cannot rely on exact pixel
# values, so it learns structural features (shape, contour) instead.
print("\n--- Checkpoint 3 passed --- denoising AE trained\n")


print("\n== Variational AE ==")
vae_model = VAE(INPUT_DIM, LATENT_DIM)
vae_losses = train_vae(vae_model, flat_loader, epochs=EPOCHS)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(vae_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses"
assert vae_losses[-1] < vae_losses[0], "VAE loss should decrease"
# INTERPRETATION: The VAE loss includes both reconstruction AND the KL
# divergence term. The KL term pushes the latent distribution toward
# N(0,I), which is what allows us to sample new images later. Higher
# KL weight = more regular latent space but blurrier reconstructions.
print("\n--- Checkpoint 4 passed --- VAE trained\n")


print("\n== Convolutional AE ==")
conv_model = ConvAE(LATENT_DIM)
conv_losses = train_ae(conv_model, "conv_ae", img_loader, epochs=EPOCHS)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(conv_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses"
assert conv_losses[-1] < conv_losses[0], "ConvAE loss should decrease"
# INTERPRETATION: The convolutional AE preserves spatial structure that
# the flat MLPs destroy. Conv filters detect local patterns (edges,
# textures) at each spatial location. This is why ConvAE typically
# achieves the lowest reconstruction loss.
print("\n--- Checkpoint 5 passed --- convolutional AE trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Sample new images from the VAE latent prior
# ════════════════════════════════════════════════════════════════════════
# The VAE's latent space is a smooth Gaussian — sample z ~ N(0,I) and
# decode to generate new images that were never in the training set.
vae_model.eval()
with torch.no_grad():
    # TODO: Sample 16 new images from the VAE by calling vae_model.sample(n=16)
    # Hint: vae_model.sample(n=16).cpu().numpy().reshape(-1, 28, 28)
    samples = ____
print(
    f"\nSampled {len(samples)} new images from VAE prior N(0, I). "
    f"mean pixel intensity: {samples.mean():.3f}, "
    f"range: [{samples.min():.3f}, {samples.max():.3f}]"
)

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert samples.shape == (16, 28, 28), "Should sample 16 images of 28x28"
assert 0.0 <= samples.min(), "Pixel values should be >= 0 (sigmoid output)"
assert samples.max() <= 1.0, "Pixel values should be <= 1 (sigmoid output)"
assert (
    samples.mean() > 0.01
), "Mean pixel should be > 0 — the VAE should generate visible content"
# INTERPRETATION: These are BRAND NEW images that never existed in the
# training data. The VAE learned the "distribution of clothing images"
# and can now sample from it.
print("\n--- Checkpoint 6 passed --- VAE generates new images\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Register the best model from each variant in the ModelRegistry
# ════════════════════════════════════════════════════════════════════════
# The ModelRegistry provides versioned model storage with metrics. In
# production, this is how you track which model version is deployed.
async def register_models():
    """Register each trained model with its final loss in the registry."""
    if not has_registry:
        print("  ModelRegistry not available — skipping registration")
        return {}

    from kailash_ml.types import MetricSpec

    model_versions = {}

    variants = [
        ("vanilla_ae", vanilla_model, vanilla_losses[-1]),
        ("denoising_ae", dae_model, dae_losses[-1]),
        ("variational_ae", vae_model, vae_losses[-1]),
        ("conv_ae", conv_model, conv_losses[-1]),
    ]

    for name, model, final_loss in variants:
        # TODO: Serialize model.state_dict() with pickle.dumps
        # and register with registry.register_model(name, artifact, metrics)
        # Hint: model_bytes = pickle.dumps(model.state_dict())
        model_bytes = ____
        version = await registry.register_model(
            name=f"m5_{name}",
            artifact=model_bytes,
            metrics=[
                MetricSpec(name="final_loss", value=final_loss),
                MetricSpec(name="latent_dim", value=float(LATENT_DIM)),
                MetricSpec(name="epochs", value=float(EPOCHS)),
            ],
        )
        model_versions[name] = version
        print(f"  Registered {name}: version={version.version}, loss={final_loss:.4f}")

    return model_versions


model_versions = asyncio.run(register_models())

# ── Checkpoint 7 ─────────────────────────────────────────────────────
if has_registry:
    assert len(model_versions) == 4, "Should register all 4 variants"
    assert "vanilla_ae" in model_versions, "Vanilla AE should be registered"
    assert "variational_ae" in model_versions, "VAE should be registered"
# INTERPRETATION: The ModelRegistry gives you a single source of truth
# for model artifacts. Instead of saving .pt files to random directories,
# every model is versioned, tagged with metrics, and queryable.
print("\n--- Checkpoint 7 passed --- models registered in ModelRegistry\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Hyperparameter comparison: latent dimensions 4, 8, 16, 32
# ════════════════════════════════════════════════════════════════════════
# How does the latent dimension affect reconstruction quality?
LATENT_DIMS_TO_TRY = [4, 8, 16, 32]
HP_EPOCHS = 8
hp_results: dict[int, list[float]] = {}

print("\n== Latent Dimension Comparison ==")
for ldim in LATENT_DIMS_TO_TRY:
    print(f"\n  Training with latent_dim={ldim}...")
    # TODO: Create a VanillaAE with INPUT_DIM and ldim, move to device
    # Hint: VanillaAE(INPUT_DIM, ldim).to(device)
    hp_model = ____
    opt = torch.optim.Adam(hp_model.parameters(), lr=1e-3)
    losses: list[float] = []

    run = asyncio.run(
        tracker.start_run(
            experiment_name=exp_name,
            run_name=f"hp_sweep_latent_{ldim}",
        )
    )
    run_id = run.id
    asyncio.run(tracker.log_param(run_id, "model_type", "VanillaAE"))
    asyncio.run(tracker.log_param(run_id, "latent_dim", str(ldim)))
    asyncio.run(tracker.log_param(run_id, "epochs", str(HP_EPOCHS)))
    asyncio.run(tracker.log_param(run_id, "sweep_type", "latent_dim"))

    for epoch in range(HP_EPOCHS):
        batch_losses = []
        for (xb,) in flat_loader:
            opt.zero_grad()
            x_hat, _ = hp_model(xb)
            loss = F.mse_loss(x_hat, xb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        epoch_loss = float(np.mean(batch_losses))
        losses.append(epoch_loss)
        asyncio.run(tracker.log_metric(run_id, "loss", epoch_loss, step=epoch + 1))

    asyncio.run(tracker.log_metric(run_id, "final_loss", losses[-1]))
    asyncio.run(tracker.end_run(run_id))

    hp_results[ldim] = losses
    print(f"    latent_dim={ldim}: final_loss={losses[-1]:.4f}")

# Print comparison table
print("\n=== Latent Dimension Comparison ===")
print(f"{'Latent Dim':>12} {'Final Loss':>12} {'Improvement':>14}")
print("-" * 42)
baseline_loss = hp_results[LATENT_DIMS_TO_TRY[0]][-1]
for ldim in LATENT_DIMS_TO_TRY:
    final = hp_results[ldim][-1]
    improvement = (baseline_loss - final) / baseline_loss * 100
    print(f"{ldim:>12} {final:>12.4f} {improvement:>13.1f}%")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(hp_results) == len(
    LATENT_DIMS_TO_TRY
), f"Should have results for all {len(LATENT_DIMS_TO_TRY)} latent dims"
assert (
    hp_results[32][-1] <= hp_results[4][-1]
), "latent_dim=32 should achieve lower or equal loss than latent_dim=4"
# INTERPRETATION: This is the information bottleneck principle in action.
# latent_dim=4 compresses 784 pixels down to just 4 numbers (196:1 ratio).
# latent_dim=32 keeps more detail but may preserve noise instead of signal.
print("\n--- Checkpoint 8 passed --- latent dimension sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 9 — Visualise all training histories
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()

# TODO: Call viz.training_history to plot the 4 variant losses
# Hint: viz.training_history(metrics={"Vanilla AE": vanilla_losses, ...}, x_label=..., y_label=...)
fig_variants = viz.training_history(
    metrics={
        "Vanilla AE": ____,
        "Denoising AE": ____,
        "Variational AE": ____,
        "Convolutional AE": ____,
    },
    x_label="Epoch",
    y_label="Loss",
)
fig_variants.write_html("ex_1_variant_comparison.html")
print("Variant comparison saved to ex_1_variant_comparison.html")

# TODO: Call viz.training_history to plot the latent dimension sweep
# Hint: metrics={f"latent_dim={ldim}": hp_results[ldim] for ldim in LATENT_DIMS_TO_TRY}
fig_latent = viz.training_history(
    metrics=____,
    x_label="Epoch",
    y_label="Loss",
)
fig_latent.write_html("ex_1_latent_dim_sweep.html")
print("Latent dimension sweep saved to ex_1_latent_dim_sweep.html")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
import os

assert os.path.exists(
    "ex_1_variant_comparison.html"
), "Variant comparison HTML should be saved"
assert os.path.exists(
    "ex_1_latent_dim_sweep.html"
), "Latent dim sweep HTML should be saved"
print("\n--- Checkpoint 9 passed --- visualisations saved\n")


# Print summary of all tracked experiments
print("\n=== Experiment Summary ===")
print(f"Experiment: {exp_name}")
print(f"Variants trained: Vanilla AE, Denoising AE, VAE, Convolutional AE")
print(f"Epochs per variant: {EPOCHS}")
print(f"Dataset: Fashion-MNIST (60,000 images)")
print(f"Latent dim sweep: {LATENT_DIMS_TO_TRY}")
print(f"\nFinal losses:")
print(f"  Vanilla AE:        {vanilla_losses[-1]:.4f}")
print(f"  Denoising AE:      {dae_losses[-1]:.4f}")
print(f"  Variational AE:    {vae_losses[-1]:.4f}")
print(f"  Convolutional AE:  {conv_losses[-1]:.4f}")

best_variant = min(
    [
        ("Vanilla AE", vanilla_losses[-1]),
        ("Denoising AE", dae_losses[-1]),
        ("Convolutional AE", conv_losses[-1]),
    ],
    key=lambda x: x[1],
)
print(f"\nBest reconstruction: {best_variant[0]} (loss={best_variant[1]:.4f})")
print("  Note: VAE loss includes KL divergence so it is not directly comparable.")

asyncio.run(conn.close())


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print(
    """
What you've mastered:
  ✓ Four autoencoder variants: Vanilla, Denoising, VAE, Convolutional
  ✓ VAE reparameterisation trick — why z = mu + sigma*eps enables backprop
  ✓ Generating new images by sampling from the VAE prior N(0, I)
  ✓ ExperimentTracker for persistent, queryable experiment records
  ✓ ModelRegistry for versioned model storage with metrics
  ✓ ModelVisualizer for comparing training curves across architectures

Next: In Exercise 2, you'll build CNNs with ResNet skip connections and
SE attention blocks — and see how the spatial structure that ConvAE
preserved becomes the foundation for discriminative image classifiers.
"""
)
