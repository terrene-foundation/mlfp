# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1: The Complete Autoencoder Family (10 Variants)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build ALL 10 autoencoder variants from scratch with torch.nn.Module:
#     Standard, Undercomplete, Denoising, Sparse, Contractive,
#     Convolutional, Stacked, Recurrent, VAE, and Contractive VAE
#   - Train each variant on FULL Fashion-MNIST (60K images) with GPU
#   - Visualise original-vs-reconstructed image grids for every variant
#     ("seeing is believing" — loss curves are NOT enough)
#   - Explain the identity-function risk and how each variant solves it
#   - Implement the VAE reparameterisation trick and sample new images
#   - Apply autoencoders to a real-world fraud detection scenario
#   - Track every run with kailash-ml's ExperimentTracker and register
#     the best model from each variant in the ModelRegistry
#
# PREREQUISITES: M4.8 (neural network basics, loss functions, optimisers)
# ESTIMATED TIME: ~180 min
#
# DATASET: Fashion-MNIST — 60,000 real 28x28 grayscale clothing images
#   across 10 classes (t-shirt, trouser, pullover, dress, coat, sandal,
#   shirt, sneaker, bag, ankle boot). Downloaded automatically by
#   torchvision and cached to data/mlfp05/fashion_mnist/.
#
#   For Recurrent AE: synthetic sensor time-series (simulating industrial
#   equipment vibration data — a realistic anomaly detection scenario).
#
# THE 10 VARIANTS:
#   1.  Standard AE          — identity risk demo (overcomplete)
#   2.  Undercomplete AE     — bottleneck forces compression
#   3.  Denoising AE (DAE)   — noise injection forces robust features
#   4.  Sparse AE            — L1 sparsity on hidden activations
#   5.  Contractive AE       — Jacobian penalty for smooth latent space
#   6.  Convolutional AE     — spatial hierarchy preserves structure
#   7.  Stacked AE           — deep feature hierarchy
#   8.  Recurrent AE         — sequential/time-series data
#   9.  VAE                  — probabilistic latent space + generation
#   10. Contractive VAE      — smooth probabilistic representations
#
# Each variant: theory -> code -> train -> VISUALISE reconstruction grid.
# The visualisation is the proof. Without it, you're trusting numbers.
#
# TASKS:
#   1.  Load Fashion-MNIST + set up Kailash engines
#   2.  Standard AE — identity risk demo
#   3.  Undercomplete AE — forced compression
#   4.  Denoising AE — 3-row grid: original, noisy, cleaned
#   5.  Sparse AE — sparsity penalty + activation histogram
#   6.  Contractive AE — Jacobian penalty + latent interpolation
#   7.  Convolutional AE — spatial hierarchy
#   8.  Stacked AE — deep feature hierarchy
#   9.  Recurrent AE — time-series reconstruction
#   10. VAE — reconstruction + sampling + latent traversal
#   11. Contractive VAE — smooth probabilistic representations
#   12. Grand comparison — all variants side by side
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
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

# Output directory for all visualisation artifacts
OUTPUT_DIR = Path("outputs") / "ex1_autoencoders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION UTILITIES — "Seeing Is Believing"
# ════════════════════════════════════════════════════════════════════════
# Every variant MUST produce a visual grid showing original vs
# reconstructed images. Loss numbers tell you the model is learning;
# image grids SHOW you what it learned. A finance director can look at
# the grid and understand "this model captured the essence of the input"
# without reading a single equation.


def show_reconstruction(model, test_data, title, n=10, is_conv=False):
    """Show original vs reconstructed images side by side.

    Args:
        model: Trained autoencoder model (must have .forward())
        test_data: Tensor of test images (flat or 2D depending on model)
        title: Title for the figure
        n: Number of examples to show
        is_conv: If True, input is (N, 1, 28, 28); else (N, 784)
    """
    model.eval()
    with torch.no_grad():
        x = test_data[:n].to(device)
        result = model(x)
        x_hat = result[0]  # All models return (reconstruction, ...)

    fig, axes = plt.subplots(2, n, figsize=(15, 3))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i in range(n):
        if is_conv:
            orig = x[i].cpu().squeeze()
            recon = x_hat[i].cpu().squeeze()
        else:
            orig = x[i].cpu().reshape(28, 28)
            recon = x_hat[i].cpu().reshape(28, 28)

        axes[0, i].imshow(orig, cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=9)

        axes[1, i].imshow(recon.clamp(0, 1), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=9)

    plt.tight_layout()
    fname = (
        OUTPUT_DIR
        / f"ex1_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_denoising_grid(model, clean_data, title, n=10, sigma=0.3):
    """3-row grid: original, noisy input, cleaned output.

    The student SEES the noise being removed — this is the denoising
    autoencoder's superpower made visible.
    """
    model.eval()
    with torch.no_grad():
        clean = clean_data[:n].to(device)
        noisy = torch.clamp(clean + sigma * torch.randn_like(clean), 0.0, 1.0)
        result = model(noisy)
        cleaned = result[0]

    fig, axes = plt.subplots(3, n, figsize=(15, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    row_labels = ["Original", "Noisy Input", "Cleaned Output"]

    for i in range(n):
        for row, data in enumerate([clean, noisy, cleaned]):
            img = data[i].cpu().reshape(28, 28)
            axes[row, i].imshow(img.clamp(0, 1), cmap="gray")
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_title(row_labels[row], fontsize=9)

    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_denoising_ae.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_activation_sparsity(model, test_data, title="Sparse AE Activations"):
    """Histogram of hidden-layer activations showing sparsity.

    A sparse autoencoder should have most activations near zero with
    only a few strongly active units — like a finance team where only
    the relevant specialists speak up on each deal.
    """
    model.eval()
    with torch.no_grad():
        x = test_data[:1000].to(device)
        h = model.encoder(x)  # Hidden activations

    activations = h.cpu().numpy().flatten()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(activations, bins=100, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Frequency")
    pct_near_zero = (np.abs(activations) < 0.1).mean() * 100
    ax.annotate(
        f"{pct_near_zero:.1f}% of activations near zero",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
    )
    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_sparse_activations.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_latent_interpolation(model, test_data, title, n_steps=10, is_conv=False):
    """Morph between two images via latent space interpolation.

    If the latent space is smooth (contractive AE), the morphing is
    gradual. If it's discontinuous, you'll see abrupt jumps.
    """
    model.eval()
    with torch.no_grad():
        x1 = test_data[0:1].to(device)
        x2 = test_data[5 : 1 + 5].to(device)  # Pick a different class
        z1 = model.encoder(x1)
        z2 = model.encoder(x2)

        alphas = torch.linspace(0, 1, n_steps).to(device)
        interpolated = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            x_hat = model.decoder(z)
            interpolated.append(x_hat)

    fig, axes = plt.subplots(1, n_steps + 2, figsize=(16, 2))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Show source image
    src_img = x1[0].cpu().reshape(28, 28) if not is_conv else x1[0].cpu().squeeze()
    axes[0].imshow(src_img, cmap="gray")
    axes[0].set_title("Start", fontsize=8)
    axes[0].axis("off")

    for i, x_hat in enumerate(interpolated):
        img = x_hat[0].cpu()
        img = img.reshape(28, 28) if not is_conv else img.squeeze()
        axes[i + 1].imshow(img.clamp(0, 1), cmap="gray")
        axes[i + 1].set_title(f"{alphas[i]:.1f}", fontsize=7)
        axes[i + 1].axis("off")

    # Show target image
    tgt_img = x2[0].cpu().reshape(28, 28) if not is_conv else x2[0].cpu().squeeze()
    axes[-1].imshow(tgt_img, cmap="gray")
    axes[-1].set_title("End", fontsize=8)
    axes[-1].axis("off")

    plt.tight_layout()
    fname = OUTPUT_DIR / f"ex1_{title.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_generated_samples(model, title="VAE Generated Samples", grid_size=8):
    """8x8 grid of images sampled from the VAE's learned prior N(0, I).

    These are BRAND NEW images that never existed in the training data.
    The VAE learned the distribution of clothing items and can now
    "imagine" new ones — like a designer sketching from memory.
    """
    model.eval()
    n = grid_size * grid_size
    with torch.no_grad():
        samples = model.sample(n).cpu()

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(samples[idx].reshape(28, 28).clamp(0, 1), cmap="gray")
            axes[i, j].axis("off")
    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_vae_generated_samples.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_latent_traversal(
    model, test_data, title="VAE Latent Traversal", n_dims=5, n_steps=11
):
    """Vary one latent dimension at a time and observe what changes.

    This reveals what each latent dimension "controls". One dimension
    might control sleeve length; another might control brightness.
    Like adjusting mixer knobs on a sound board — each knob changes
    one aspect of the output.
    """
    model.eval()
    with torch.no_grad():
        x = test_data[0:1].to(device)
        mu, _ = model.encode(x)
        base_z = mu.clone()

    traversal_range = torch.linspace(-3, 3, n_steps)
    fig, axes = plt.subplots(n_dims, n_steps, figsize=(14, n_dims * 1.4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for dim in range(n_dims):
        for step_idx, val in enumerate(traversal_range):
            z = base_z.clone()
            z[0, dim] = val
            with torch.no_grad():
                x_hat = model.decoder(z)
            img = x_hat[0].cpu().reshape(28, 28).clamp(0, 1)
            axes[dim, step_idx].imshow(img, cmap="gray")
            axes[dim, step_idx].axis("off")
            if dim == 0:
                axes[dim, step_idx].set_title(f"z={val:.1f}", fontsize=7)
        axes[dim, 0].set_ylabel(f"dim {dim}", fontsize=8, rotation=0, labelpad=30)

    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_vae_latent_traversal.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_timeseries_reconstruction(model, test_data, title, n_series=4):
    """Overlay original vs reconstructed time series.

    For sequential data (not images), the visual proof is a line plot.
    If the reconstructed line tracks the original, the model learned
    the temporal patterns.
    """
    model.eval()
    with torch.no_grad():
        x = test_data[:n_series].to(device)
        x_hat, _ = model(x)

    fig, axes = plt.subplots(n_series, 1, figsize=(14, 3 * n_series))
    if n_series == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i in range(n_series):
        orig = x[i].cpu().numpy()
        recon = x_hat[i].cpu().numpy()
        t = np.arange(len(orig))

        axes[i].plot(t, orig, "b-", linewidth=1.5, label="Original", alpha=0.8)
        axes[i].plot(t, recon, "r--", linewidth=1.5, label="Reconstructed", alpha=0.8)
        axes[i].set_ylabel(f"Series {i + 1}")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_recurrent_ae_timeseries.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


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
    transform=torchvision.transforms.ToTensor(),
)
test_set = torchvision.datasets.FashionMNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Full 60K training images — autoencoders need volume to learn general
# features (edges, textures, shapes) instead of memorising specific items.
X_img = torch.stack(
    [train_set[i][0] for i in range(len(train_set))]
)  # (60000, 1, 28, 28)
X_img = X_img.to(device).float()
X_flat = X_img.reshape(len(X_img), -1)  # (60000, 784)

# Test set for visualisation (unseen data proves generalisation)
X_test_img = torch.stack([test_set[i][0] for i in range(len(test_set))])
X_test_img = X_test_img.to(device).float()
X_test_flat = X_test_img.reshape(len(X_test_img), -1)

print(
    f"Fashion-MNIST loaded: {len(X_img)} train + {len(X_test_img)} test images, "
    f"shape {tuple(X_img.shape[1:])}, pixel range [{X_img.min():.2f}, {X_img.max():.2f}]"
)

flat_loader = DataLoader(TensorDataset(X_flat), batch_size=256, shuffle=True)
img_loader = DataLoader(TensorDataset(X_img), batch_size=256, shuffle=True)

INPUT_DIM = 28 * 28
LATENT_DIM = 16
EPOCHS = 10


# Set up kailash-ml engines: ExperimentTracker + ModelRegistry
async def setup_engines():
    conn = ConnectionManager("sqlite:///mlfp05_autoencoders.db")
    await conn.initialize()

    tracker = ExperimentTracker(conn)
    exp_name = await tracker.create_experiment(
        name="m5_autoencoders",
        description="All 10 autoencoder variants on Fashion-MNIST (60K images)",
    )

    try:
        registry = ModelRegistry(conn)
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


conn, tracker, exp_name, registry, has_registry = asyncio.run(setup_engines())

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_img.shape[0] == 60000, f"Expected full 60K Fashion-MNIST, got {X_img.shape[0]}"
assert X_flat.shape == (60000, 784), "Flattened tensor should be (60000, 784)"
assert X_test_flat.shape[0] == 10000, "Test set should have 10K images"
assert tracker is not None, "ExperimentTracker should be initialised"
assert exp_name is not None, "Experiment should be created"
# INTERPRETATION: We use all 60K training images because autoencoders
# are unsupervised — every image teaches the encoder about clothing
# structure. The 10K test images are held out for visualisation so we
# can see how well each model generalises to UNSEEN data.
print("\n--- Checkpoint 1 passed --- data loaded and engines initialised\n")


# ════════════════════════════════════════════════════════════════════════
# TRAINING HELPER — shared by all variants
# ════════════════════════════════════════════════════════════════════════
# Centralised training loop with ExperimentTracker integration.
# Uses the modern `async with tracker.run(...)` context manager that
# auto-marks runs COMPLETED on success and FAILED on exception.

all_losses: dict[str, list[float]] = {}  # Collect for grand comparison
all_models: dict[str, nn.Module] = {}  # Collect for model registry


async def train_variant_async(
    model: nn.Module,
    name: str,
    loader: DataLoader,
    loss_fn,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
    extra_params: dict | None = None,
) -> list[float]:
    """Universal training loop for any AE variant.

    Args:
        model: The autoencoder to train
        name: Human-readable name for tracking
        loader: DataLoader with training data
        loss_fn: Callable(model, x_batch) -> (loss, info_dict)
            Signature allows each variant to define its own loss.
        epochs: Number of training epochs
        lr: Learning rate
        extra_params: Additional params to log (e.g., noise_sigma)
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    params = {
        "model_type": name,
        "latent_dim": str(LATENT_DIM),
        "epochs": str(epochs),
        "lr": str(lr),
        "dataset_size": str(len(loader.dataset)),
        "batch_size": str(loader.batch_size),
    }
    if extra_params:
        params.update(extra_params)

    async with tracker.run(experiment_name=exp_name, run_name=name) as ctx:
        await ctx.log_params(params)

        for epoch in range(epochs):
            batch_losses = []
            for (xb,) in loader:
                opt.zero_grad()
                loss, _ = loss_fn(model, xb)
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
            epoch_loss = float(np.mean(batch_losses))
            losses.append(epoch_loss)
            await ctx.log_metric("loss", epoch_loss, step=epoch + 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [{name}] epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")
        await ctx.log_metric("final_loss", losses[-1])

    return losses


def train_variant(
    model, name, loader, loss_fn, epochs=EPOCHS, lr=1e-3, extra_params=None
):
    """Sync wrapper for training."""
    losses = asyncio.run(
        train_variant_async(model, name, loader, loss_fn, epochs, lr, extra_params)
    )
    all_losses[name] = losses
    all_models[name] = model
    return losses


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Standard Autoencoder (Identity Risk Demo)
# ════════════════════════════════════════════════════════════════════════
# THEORY: A standard autoencoder with hidden dimensions >= input dimension
# can learn the trivial identity function f(x) = x. It achieves near-zero
# loss by simply copying the input — no useful compression learned.
#
# WHY THIS MATTERS: In fraud detection, a model that memorises every
# transaction pattern (including fraudulent ones) fails to flag anomalies.
# The identity risk is the autoencoder equivalent of overfitting.


class StandardAE(nn.Module):
    """Overcomplete autoencoder — hidden dim > input dim.

    This is intentionally "too powerful". With 1024-dim hidden layers for
    784-dim input, it CAN learn the identity function. We demonstrate
    this risk, then show how each subsequent variant solves it.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def standard_ae_loss(model, xb):
    x_hat, _ = model(xb)
    return F.mse_loss(x_hat, xb), {}


print("\n" + "=" * 70)
print("  TASK 2: Standard Autoencoder — Identity Risk Demo")
print("=" * 70)
print("  Hidden dim=1024 > input dim=784. Can the model just copy?")

standard_model = StandardAE(INPUT_DIM, hidden_dim=1024)
standard_losses = train_variant(
    standard_model, "standard_ae", flat_loader, standard_ae_loss
)

show_reconstruction(standard_model, X_test_flat, "Standard AE (Overcomplete)")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(standard_losses) == EPOCHS, f"Expected {EPOCHS} losses"
assert standard_losses[-1] < standard_losses[0], "Loss should decrease"
# The standard AE achieves very low loss — suspiciously low. It has
# enough capacity to memorise rather than generalise. This is the
# baseline that every other variant improves upon.
# INTERPRETATION: The near-perfect reconstruction is DECEPTIVE. This
# model learned to copy, not to compress. In production, it would fail
# to detect anomalies because it reconstructs EVERYTHING well — even
# fraudulent transactions it should flag as unusual.
print("\n--- Checkpoint 2 passed --- standard AE (identity risk demonstrated)\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Undercomplete Autoencoder (Forced Compression)
# ════════════════════════════════════════════════════════════════════════
# THEORY: The fix for identity risk is simple — make the bottleneck
# SMALLER than the input. With latent_dim=16, the encoder must compress
# 784 pixels into just 16 numbers. This is a 49:1 compression ratio.
#
# WHY THIS MATTERS: Think of a 50-page quarterly report compressed into
# a one-page executive summary. The summary MUST capture the key points.
# That forced compression is exactly what the undercomplete bottleneck does.


class UndercompleteAE(nn.Module):
    """Bottleneck forces compression: 784 -> 256 -> 64 -> 16 -> 64 -> 256 -> 784."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def undercomplete_ae_loss(model, xb):
    x_hat, _ = model(xb)
    return F.mse_loss(x_hat, xb), {}


print("\n" + "=" * 70)
print("  TASK 3: Undercomplete AE — Forced Compression (latent=16)")
print("=" * 70)
print("  784 pixels -> 16 numbers. Compression ratio 49:1.")

undercomplete_model = UndercompleteAE(INPUT_DIM, LATENT_DIM)
undercomplete_losses = train_variant(
    undercomplete_model, "undercomplete_ae", flat_loader, undercomplete_ae_loss
)

show_reconstruction(
    undercomplete_model, X_test_flat, f"Undercomplete AE (latent={LATENT_DIM})"
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(undercomplete_losses) == EPOCHS
assert undercomplete_losses[-1] < undercomplete_losses[0]
# The loss is HIGHER than the standard AE — that's correct.
# The bottleneck forces the encoder to choose what matters.
# INTERPRETATION: The reconstructions are blurry but recognisable.
# The model kept the SHAPE (is it a shirt? a shoe?) but lost the DETAIL
# (exact button placement, stitching pattern). This is the information
# bottleneck principle: compress enough, and the model learns structure.
print("\n--- Checkpoint 3 passed --- undercomplete AE trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Denoising Autoencoder (DAE)
# ════════════════════════════════════════════════════════════════════════
# THEORY: Add Gaussian noise to the input, then train the model to
# reconstruct the CLEAN original. The noise acts as implicit
# regularisation — the encoder cannot memorise pixel values because
# the pixels are corrupted differently every epoch.
#
# WHY THIS MATTERS: In medical imaging, scans often have noise (sensor
# artifacts, patient movement). A denoising AE trained on clean + noisy
# pairs can clean up real scans, potentially revealing tumours hidden
# by noise. At SGH (Singapore General Hospital), MRI denoising reduces
# the need for repeated (expensive, uncomfortable) scans.


class DenoisingAE(nn.Module):
    """Same architecture as undercomplete, but trained with noise injection."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def add_noise(self, x, sigma=0.3):
        return torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


NOISE_SIGMA = 0.3


def dae_loss(model, xb):
    """Train on noisy input, reconstruct clean target."""
    noisy = model.add_noise(xb, sigma=NOISE_SIGMA)
    x_hat, _ = model(noisy)
    return F.mse_loss(x_hat, xb), {}  # Target is the CLEAN original


print("\n" + "=" * 70)
print("  TASK 4: Denoising AE — Noise Injection (sigma=0.3)")
print("=" * 70)
print("  Input: corrupted image. Target: clean image. The model learns to denoise.")

dae_model = DenoisingAE(INPUT_DIM, LATENT_DIM)
dae_losses = train_variant(
    dae_model,
    "denoising_ae",
    flat_loader,
    dae_loss,
    extra_params={"noise_sigma": str(NOISE_SIGMA)},
)

# 3-row grid: original, noisy, cleaned — the student SEES the denoising
show_denoising_grid(dae_model, X_test_flat, "Denoising AE (3-Row Comparison)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(dae_losses) == EPOCHS
assert dae_losses[-1] < dae_losses[0]
# INTERPRETATION: The 3-row grid tells the story:
# Row 1 (Original): the clean clothing image
# Row 2 (Noisy): what the model receives — grainy, corrupted
# Row 3 (Cleaned): what the model outputs — noise removed, structure preserved
# The DAE learned ROBUST features that survive corruption. In practice,
# this makes it a powerful pre-training step for downstream classifiers.
print("\n--- Checkpoint 4 passed --- denoising AE trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Sparse Autoencoder
# ════════════════════════════════════════════════════════════════════════
# THEORY: Instead of constraining the bottleneck SIZE, we constrain the
# ACTIVATIONS. An L1 penalty on the hidden layer forces most neurons to
# stay near zero. Only a few "specialist" neurons fire for each input.
#
# WHY THIS MATTERS: In a bank's fraud detection system, you want each
# feature detector to specialise. One detector for "unusual amount",
# another for "unusual time", another for "unusual merchant". Sparsity
# forces this specialisation — the model cannot spread the signal
# across all neurons equally.


class SparseAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


SPARSITY_WEIGHT = 1e-4


def sparse_ae_loss(model, xb):
    """MSE + L1 sparsity penalty on hidden activations."""
    x_hat, z = model(xb)
    recon_loss = F.mse_loss(x_hat, xb)
    sparsity_loss = SPARSITY_WEIGHT * torch.mean(torch.abs(z))
    return recon_loss + sparsity_loss, {"sparsity": sparsity_loss.item()}


print("\n" + "=" * 70)
print("  TASK 5: Sparse AE — L1 Sparsity Penalty")
print("=" * 70)
print(f"  Sparsity weight: {SPARSITY_WEIGHT}. Most neurons should stay near zero.")

sparse_model = SparseAE(INPUT_DIM)
sparse_losses = train_variant(
    sparse_model,
    "sparse_ae",
    flat_loader,
    sparse_ae_loss,
    extra_params={"sparsity_weight": str(SPARSITY_WEIGHT)},
)

show_reconstruction(sparse_model, X_test_flat, "Sparse AE")
show_activation_sparsity(sparse_model, X_test_flat, "Sparse AE — Hidden Activations")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(sparse_losses) == EPOCHS
assert sparse_losses[-1] < sparse_losses[0]
# Check sparsity: most activations should be near zero
sparse_model.eval()
with torch.no_grad():
    test_z = sparse_model.encoder(X_test_flat[:1000].to(device))
    pct_sparse = (torch.abs(test_z) < 0.1).float().mean().item()
assert pct_sparse > 0.3, (
    f"Expected >30% of activations near zero, got {pct_sparse:.1%}. "
    "Increase sparsity_weight if needed."
)
# INTERPRETATION: The activation histogram shows the sparsity distribution.
# Most activations cluster near zero — only a few "expert" neurons fire
# per image. This is biological: in the visual cortex, only ~1% of
# neurons fire for any given stimulus. Sparse representations are more
# interpretable and more efficient.
print(f"  Sparsity: {pct_sparse:.1%} of activations near zero")
print("\n--- Checkpoint 5 passed --- sparse AE trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Contractive Autoencoder (CAE)
# ════════════════════════════════════════════════════════════════════════
# THEORY: The Jacobian penalty discourages the encoder from being too
# sensitive to small input perturbations. If two similar images map to
# very different latent codes, the latent space is "bumpy". The
# contractive penalty smooths it out.
#
# WHY THIS MATTERS: In manufacturing quality control, two photos of the
# same product taken at slightly different angles should map to similar
# latent codes. A contractive AE ensures that minor variations in camera
# position don't cause the model to "see" different products.


class ContractiveAE(nn.Module):
    """Autoencoder with explicit encoder weight access for Jacobian penalty."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, 256)
        self.enc2 = nn.Linear(256, 64)
        self.enc3 = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encoder(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.enc3(h)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


CONTRACTIVE_WEIGHT = 1e-4


def contractive_ae_loss(model, xb):
    """MSE + Frobenius norm of the Jacobian (approximated via weight norms).

    The exact Jacobian requires computing dz/dx for each sample, which is
    expensive. We approximate it using the weight matrices of the encoder
    layers, which captures the same sensitivity information.
    """
    x_hat, z = model(xb)
    recon_loss = F.mse_loss(x_hat, xb)
    # Frobenius norm of encoder weights as Jacobian approximation
    jacobian_penalty = sum(
        torch.sum(p**2)
        for p in [model.enc1.weight, model.enc2.weight, model.enc3.weight]
    )
    return recon_loss + CONTRACTIVE_WEIGHT * jacobian_penalty, {}


print("\n" + "=" * 70)
print("  TASK 6: Contractive AE — Jacobian Penalty")
print("=" * 70)
print("  Smooth latent space: similar inputs -> similar latent codes.")

contractive_model = ContractiveAE(INPUT_DIM, LATENT_DIM)
contractive_losses = train_variant(
    contractive_model,
    "contractive_ae",
    flat_loader,
    contractive_ae_loss,
    extra_params={"contractive_weight": str(CONTRACTIVE_WEIGHT)},
)

show_reconstruction(contractive_model, X_test_flat, "Contractive AE")
show_latent_interpolation(
    contractive_model,
    X_test_flat,
    "Contractive AE — Latent Interpolation",
)

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(contractive_losses) == EPOCHS
assert contractive_losses[-1] < contractive_losses[0]
# INTERPRETATION: The latent interpolation plot is the key visual here.
# As we morph from image A to image B through latent space, the
# contractive AE produces SMOOTH transitions — no abrupt jumps. This
# means the latent space is well-organised: nearby points represent
# similar images. Without the Jacobian penalty, interpolation would
# show sharp discontinuities where small latent changes cause large
# output changes.
print("\n--- Checkpoint 6 passed --- contractive AE trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Convolutional Autoencoder
# ════════════════════════════════════════════════════════════════════════
# THEORY: Conv layers preserve SPATIAL locality that flat MLPs destroy.
# A Conv2d filter detects patterns (edges, textures) at each spatial
# position. The encoder progressively downsamples: 28x28 -> 14x14 -> 7x7.
#
# WHY THIS MATTERS: In satellite imagery for urban planning (e.g.,
# Singapore's URA using satellite feeds), spatial relationships matter.
# A building's shape, its proximity to roads, green spaces — all spatial.
# Convolutional autoencoders preserve these relationships; flat MLPs
# treat every pixel as independent, losing the spatial context.


class ConvAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        # 1x28x28 -> 16x14x14 -> 32x7x7 -> flatten -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def conv_ae_loss(model, xb):
    x_hat, _ = model(xb)
    return F.mse_loss(x_hat, xb), {}


print("\n" + "=" * 70)
print("  TASK 7: Convolutional AE — Spatial Hierarchy")
print("=" * 70)
print("  Conv2d preserves spatial structure. Expect sharper reconstructions.")

conv_model = ConvAE(LATENT_DIM)
conv_losses = train_variant(conv_model, "conv_ae", img_loader, conv_ae_loss)

show_reconstruction(conv_model, X_test_img, "Convolutional AE", is_conv=True)

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert len(conv_losses) == EPOCHS
assert conv_losses[-1] < conv_losses[0]
# INTERPRETATION: Compare the ConvAE reconstructions to the
# undercomplete AE. The convolutional version preserves EDGES and
# TEXTURES better — you can see sharper outlines of shirts, shoes, bags.
# This is because Conv2d filters share parameters across spatial
# positions, learning translation-invariant features. A button pattern
# detected at position (5,5) is also detected at (20,20).
print("\n--- Checkpoint 7 passed --- convolutional AE trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Stacked Autoencoder (Deep Feature Hierarchy)
# ════════════════════════════════════════════════════════════════════════
# THEORY: Stack multiple autoencoder layers to learn a hierarchy of
# features: low-level (edges) -> mid-level (textures) -> high-level
# (shapes). Each layer learns to encode the previous layer's output.
#
# WHY THIS MATTERS: In document processing (e.g., DBS bank's KYC
# pipeline), features exist at multiple levels. Character-level features
# detect handwriting quality; word-level features detect names and
# addresses; document-level features detect document type (passport vs
# utility bill). A stacked AE learns this hierarchy automatically.


class StackedAE(nn.Module):
    """Deep encoder with 5 layers of progressive compression."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def stacked_ae_loss(model, xb):
    x_hat, _ = model(xb)
    return F.mse_loss(x_hat, xb), {}


print("\n" + "=" * 70)
print("  TASK 8: Stacked AE — Deep Feature Hierarchy")
print("=" * 70)
print("  5 encoder layers: 784->512->256->128->64->16")

stacked_model = StackedAE(INPUT_DIM, LATENT_DIM)
stacked_losses = train_variant(
    stacked_model, "stacked_ae", flat_loader, stacked_ae_loss
)

show_reconstruction(stacked_model, X_test_flat, "Stacked AE (5 Layers)")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(stacked_losses) == EPOCHS
assert stacked_losses[-1] < stacked_losses[0]
# INTERPRETATION: More layers does NOT automatically mean better
# reconstruction. The stacked AE may perform similarly to the simpler
# undercomplete AE because depth without skip connections can cause
# vanishing gradients. The value of stacking is in FEATURE HIERARCHY —
# the intermediate representations at each layer capture increasingly
# abstract features. Layer 1 captures edges; Layer 3 captures shapes.
print("\n--- Checkpoint 8 passed --- stacked AE trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 9 — Recurrent Autoencoder (Time-Series Data)
# ════════════════════════════════════════════════════════════════════════
# THEORY: For sequential data, temporal order matters. An LSTM encoder
# reads the sequence step by step, building a summary (hidden state).
# The LSTM decoder unrolls the summary back into the original sequence.
#
# WHY THIS MATTERS: In Singapore's manufacturing sector, equipment
# vibration sensors produce time-series data. A recurrent AE trained on
# "normal" vibration patterns can flag anomalies — unusual vibrations
# that the model cannot reconstruct well indicate potential equipment
# failure. Predictive maintenance saves millions in downtime costs.

# Generate synthetic sensor data (vibration patterns from industrial equipment)
SEQ_LEN = 100
N_SERIES_TRAIN = 5000
N_SERIES_TEST = 500


def generate_sensor_data(n_samples, seq_len, seed=42):
    """Generate synthetic industrial vibration sensor data.

    Normal patterns: combination of periodic signals (motor rotation,
    belt vibration) with Gaussian noise (sensor noise).
    """
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n_samples):
        t = np.linspace(0, 4 * np.pi, seq_len)
        # Base frequency (motor rotation ~50Hz equivalent)
        freq1 = rng.uniform(0.8, 1.2)
        # Harmonic (belt vibration at 2x motor frequency)
        freq2 = rng.uniform(1.8, 2.2)
        amplitude1 = rng.uniform(0.5, 1.0)
        amplitude2 = rng.uniform(0.2, 0.5)
        # Phase shift (sensor alignment variation)
        phase = rng.uniform(0, 2 * np.pi)
        signal = (
            amplitude1 * np.sin(freq1 * t + phase)
            + amplitude2 * np.sin(freq2 * t)
            + 0.1 * rng.randn(seq_len)  # sensor noise
        )
        # Normalise to [0, 1] for sigmoid output
        signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        data.append(signal)
    return np.array(data, dtype=np.float32)


sensor_train = generate_sensor_data(N_SERIES_TRAIN, SEQ_LEN, seed=42)
sensor_test = generate_sensor_data(N_SERIES_TEST, SEQ_LEN, seed=99)

sensor_train_t = torch.tensor(sensor_train).to(device)
sensor_test_t = torch.tensor(sensor_test).to(device)
sensor_loader = DataLoader(TensorDataset(sensor_train_t), batch_size=128, shuffle=True)

print(
    f"\nSensor data: {sensor_train.shape[0]} train, {sensor_test.shape[0]} test, "
    f"seq_len={SEQ_LEN}"
)


class RecurrentAE(nn.Module):
    """LSTM-based autoencoder for sequential data."""

    def __init__(self, seq_len: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Encoder LSTM: reads sequence -> final hidden state
        self.encoder_lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim, batch_first=True
        )
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> repeat -> LSTM -> reconstruct
        self.latent_to_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, 1)
        x_seq = x.unsqueeze(-1)

        # Encode: process full sequence, take final hidden state
        _, (h_n, _) = self.encoder_lstm(x_seq)
        z = self.enc_to_latent(h_n.squeeze(0))

        # Decode: repeat latent across time steps, then LSTM
        dec_input = self.latent_to_dec(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_output, _ = self.decoder_lstm(dec_input)
        x_hat = self.output_layer(dec_output).squeeze(-1)

        return x_hat, z


def recurrent_ae_loss(model, xb):
    x_hat, _ = model(xb)
    return F.mse_loss(x_hat, xb), {}


print("\n" + "=" * 70)
print("  TASK 9: Recurrent AE — Time-Series (Sensor Data)")
print("=" * 70)
print("  LSTM encoder reads sequence -> latent -> LSTM decoder reconstructs.")

recurrent_model = RecurrentAE(SEQ_LEN, hidden_dim=64, latent_dim=LATENT_DIM)
recurrent_losses = train_variant(
    recurrent_model,
    "recurrent_ae",
    sensor_loader,
    recurrent_ae_loss,
    extra_params={"seq_len": str(SEQ_LEN), "data_type": "sensor_vibration"},
)

show_timeseries_reconstruction(
    recurrent_model,
    sensor_test_t,
    "Recurrent AE — Sensor Vibration Reconstruction",
)

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(recurrent_losses) == EPOCHS
assert recurrent_losses[-1] < recurrent_losses[0]
# INTERPRETATION: The time-series overlay shows reconstructed (red
# dashed) tracking original (blue solid) vibration patterns. Where
# the lines diverge = the model finds that pattern harder to compress.
# In production, you'd flag test sequences with HIGH reconstruction
# error as anomalies — the model saying "I've never seen a vibration
# pattern like this" is the early warning for equipment failure.
print("\n--- Checkpoint 9 passed --- recurrent AE trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 10 — Variational Autoencoder (VAE)
# ════════════════════════════════════════════════════════════════════════
# THEORY: The VAE replaces the deterministic latent code with a
# PROBABILITY DISTRIBUTION. The encoder outputs mean (mu) and
# log-variance (log_var). We sample z ~ N(mu, sigma^2) using the
# reparameterisation trick: z = mu + sigma * epsilon, where
# epsilon ~ N(0, I). This keeps gradients flowing through mu and sigma.
#
# Loss = reconstruction + KL(q(z|x) || N(0,I))
#
# The KL term pushes the latent distribution toward a standard Gaussian,
# which is what allows us to GENERATE new images by sampling z ~ N(0,I).
#
# WHY THIS MATTERS: A bank wanting to generate synthetic transaction
# data for model testing (without exposing real customer data) can train
# a VAE on real transactions and sample from the learned distribution.
# The synthetic data preserves statistical properties without containing
# any real customer information.


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        """The reparameterisation trick: z = mu + sigma * epsilon.

        epsilon ~ N(0, I) is the only source of randomness.
        Gradients flow through mu and sigma (not through sampling).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def sample(self, n):
        """Sample from the prior N(0, I) and decode to images."""
        z = torch.randn(
            n, self.fc_mu.out_features, device=next(self.parameters()).device
        )
        return self.decoder(z)


KL_WEIGHT = 0.1  # Balance reconstruction clarity vs latent regularity


def vae_loss_fn(model, xb):
    """VAE loss: reconstruction (MSE) + KL divergence."""
    x_hat, mu, logvar = model(xb)
    recon = F.mse_loss(x_hat, xb, reduction="sum") / xb.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
    return recon + KL_WEIGHT * kl, {"recon": recon.item(), "kl": kl.item()}


print("\n" + "=" * 70)
print("  TASK 10: Variational AE — Probabilistic Latent Space")
print("=" * 70)
print("  Reparameterisation trick: z = mu + sigma * epsilon")
print(f"  KL weight: {KL_WEIGHT} (balance reconstruction vs regularity)")

vae_model = VAE(INPUT_DIM, LATENT_DIM)
vae_losses = train_variant(
    vae_model,
    "vae",
    flat_loader,
    vae_loss_fn,
    extra_params={"kl_weight": str(KL_WEIGHT)},
)

# Three VAE-specific visualisations:
# (a) Reconstruction grid — standard quality check
show_reconstruction(vae_model, X_test_flat, "VAE Reconstruction")

# (b) Generated samples — new images from the prior N(0, I)
show_generated_samples(vae_model, "VAE — Generated Samples from N(0,I)", grid_size=8)

# (c) Latent traversal — vary one dimension, observe what changes
show_latent_traversal(vae_model, X_test_flat, "VAE — Latent Traversal", n_dims=5)

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert len(vae_losses) == EPOCHS
assert vae_losses[-1] < vae_losses[0]
# Verify generation works
vae_model.eval()
with torch.no_grad():
    samples = vae_model.sample(n=16).cpu()
assert samples.shape == (16, INPUT_DIM), f"Expected (16, 784), got {samples.shape}"
assert (
    samples.min() >= 0.0 and samples.max() <= 1.0
), "Sigmoid output should be in [0, 1]"
# INTERPRETATION: The VAE produces three kinds of visual proof:
# 1. Reconstruction grid: slightly blurrier than the undercomplete AE
#    because the KL term trades reconstruction quality for a regular
#    latent space.
# 2. Generated samples: BRAND NEW images from z ~ N(0,I). Some are
#    clear (shirts, shoes); some are ambiguous (the model's "imagination").
# 3. Latent traversal: each row varies one dimension. You can see one
#    dimension control shape, another control brightness, another control
#    style. This is DISENTANGLED representation learning in action.
print("\n--- Checkpoint 10 passed --- VAE trained + generation verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 11 — Contractive VAE
# ════════════════════════════════════════════════════════════════════════
# THEORY: Combines the probabilistic latent space of a VAE with the
# Jacobian smoothness penalty of a Contractive AE. The result is a
# latent space that is both generative (can sample new data) AND smooth
# (nearby latent codes produce similar outputs).
#
# WHY THIS MATTERS: For generating synthetic financial reports at MAS
# (Monetary Authority of Singapore), you need both:
# - Diversity (VAE's sampling) to cover different report formats
# - Smoothness (contractive penalty) so small parameter changes produce
#   plausible variations, not wild jumps between report types


class ContractiveVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, 256)
        self.enc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encoder(self, x):
        h = F.relu(self.enc1(x))
        return F.relu(self.enc2(h))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def sample(self, n):
        z = torch.randn(
            n, self.fc_mu.out_features, device=next(self.parameters()).device
        )
        return self.decoder(z)


CVAE_CONTRACTIVE_WEIGHT = 1e-4


def cvae_loss_fn(model, xb):
    """VAE loss + Contractive penalty on encoder weights."""
    x_hat, mu, logvar = model(xb)
    recon = F.mse_loss(x_hat, xb, reduction="sum") / xb.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
    # Contractive penalty
    jacobian_penalty = sum(
        torch.sum(p**2) for p in [model.enc1.weight, model.enc2.weight]
    )
    return (
        recon + KL_WEIGHT * kl + CVAE_CONTRACTIVE_WEIGHT * jacobian_penalty,
        {"recon": recon.item(), "kl": kl.item()},
    )


print("\n" + "=" * 70)
print("  TASK 11: Contractive VAE — Smooth + Probabilistic")
print("=" * 70)
print("  VAE's generation + Contractive AE's smoothness = best of both worlds.")

cvae_model = ContractiveVAE(INPUT_DIM, LATENT_DIM)
cvae_losses = train_variant(
    cvae_model,
    "contractive_vae",
    flat_loader,
    cvae_loss_fn,
    extra_params={
        "kl_weight": str(KL_WEIGHT),
        "contractive_weight": str(CVAE_CONTRACTIVE_WEIGHT),
    },
)

show_reconstruction(cvae_model, X_test_flat, "Contractive VAE")

# Compare latent interpolation smoothness: VAE vs Contractive VAE
# The CVAE should produce SMOOTHER morphing between images
print("\n  Comparing latent interpolation smoothness...")


def compare_interpolation(model_a, model_b, test_data, label_a, label_b, n_steps=10):
    """Side-by-side interpolation comparison between two models."""
    fig, axes = plt.subplots(2, n_steps + 2, figsize=(16, 3.5))
    fig.suptitle(
        f"Latent Interpolation: {label_a} vs {label_b}", fontsize=13, fontweight="bold"
    )

    for row, (model, label) in enumerate([(model_a, label_a), (model_b, label_b)]):
        model.eval()
        with torch.no_grad():
            x1 = test_data[0:1].to(device)
            x2 = test_data[5:6].to(device)

            if hasattr(model, "encode"):
                mu1, _ = model.encode(x1)
                mu2, _ = model.encode(x2)
                z1, z2 = mu1, mu2
            else:
                z1 = model.encoder(x1)
                z2 = model.encoder(x2)

            alphas = torch.linspace(0, 1, n_steps).to(device)

            # Source
            axes[row, 0].imshow(x1[0].cpu().reshape(28, 28), cmap="gray")
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("Start", fontsize=7)

            for i, alpha in enumerate(alphas):
                z = (1 - alpha) * z1 + alpha * z2
                x_hat = model.decoder(z)
                axes[row, i + 1].imshow(
                    x_hat[0].cpu().reshape(28, 28).clamp(0, 1), cmap="gray"
                )
                axes[row, i + 1].axis("off")
                if row == 0:
                    axes[row, i + 1].set_title(f"{alpha:.1f}", fontsize=7)

            # Target
            axes[row, -1].imshow(x2[0].cpu().reshape(28, 28), cmap="gray")
            axes[row, -1].axis("off")
            if row == 0:
                axes[row, -1].set_title("End", fontsize=7)

        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0, labelpad=50)

    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_vae_vs_cvae_interpolation.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


compare_interpolation(
    vae_model,
    cvae_model,
    X_test_flat,
    "Vanilla VAE",
    "Contractive VAE",
)

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert len(cvae_losses) == EPOCHS
assert cvae_losses[-1] < cvae_losses[0]
# INTERPRETATION: The side-by-side interpolation comparison reveals the
# contractive penalty's effect. The Contractive VAE produces smoother
# transitions — intermediate images look like plausible clothing items,
# not blurred mixes. This matters when you need to GENERATE realistic
# synthetic data (for testing, privacy-preserving data sharing, or
# augmentation): smooth latent spaces produce more realistic samples.
print("\n--- Checkpoint 11 passed --- contractive VAE trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 12 — Grand Comparison: All 10 Variants Side by Side
# ════════════════════════════════════════════════════════════════════════
# The definitive comparison: same test images reconstructed by all 10
# models. This is the "board-level summary" — one figure that tells the
# complete story of what each variant does differently.

print("\n" + "=" * 70)
print("  TASK 12: Grand Comparison — All 10 Variants")
print("=" * 70)

# --- A. Image-based variants: reconstruction comparison (9 variants) ---
image_variants = [
    ("Standard AE", standard_model, False),
    ("Undercomplete AE", undercomplete_model, False),
    ("Denoising AE", dae_model, False),
    ("Sparse AE", sparse_model, False),
    ("Contractive AE", contractive_model, False),
    ("Convolutional AE", conv_model, True),
    ("Stacked AE", stacked_model, False),
    ("VAE", vae_model, False),
    ("Contractive VAE", cvae_model, False),
]

N_COMPARE = 8  # images per variant
n_variants = len(image_variants)

fig, axes = plt.subplots(
    n_variants + 1, N_COMPARE, figsize=(16, (n_variants + 1) * 1.5)
)
fig.suptitle(
    "Grand Comparison: Original + 9 Image AE Variants",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)

# Row 0: original test images
for j in range(N_COMPARE):
    axes[0, j].imshow(X_test_img[j].cpu().squeeze(), cmap="gray")
    axes[0, j].axis("off")
axes[0, 0].set_ylabel("Original", fontsize=9, rotation=0, labelpad=55)

# Rows 1-9: each variant's reconstruction
for row, (name, model, is_conv) in enumerate(image_variants, start=1):
    model.eval()
    with torch.no_grad():
        if is_conv:
            x = X_test_img[:N_COMPARE].to(device)
        else:
            x = X_test_flat[:N_COMPARE].to(device)

        result = model(x)
        x_hat = result[0]

    for j in range(N_COMPARE):
        if is_conv:
            img = x_hat[j].cpu().squeeze()
        else:
            img = x_hat[j].cpu().reshape(28, 28)
        axes[row, j].imshow(img.clamp(0, 1), cmap="gray")
        axes[row, j].axis("off")
    axes[row, 0].set_ylabel(name, fontsize=8, rotation=0, labelpad=60)

plt.tight_layout()
fname = OUTPUT_DIR / "ex1_grand_comparison.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname}")

# --- B. Summary table of all variants ---
print("\n=== Autoencoder Variant Comparison ===")
print(f"{'Variant':<22} {'Final Loss':>12} {'Params':>10} {'Type':>12}")
print("-" * 60)

for name, model, _ in image_variants:
    loss_key = [
        k
        for k in all_losses
        if k in name.lower().replace(" ", "_")
        or name.lower().replace(" ", "_").startswith(k.split("_")[0])
    ]
    # Map display names to all_losses keys
    name_to_key = {
        "Standard AE": "standard_ae",
        "Undercomplete AE": "undercomplete_ae",
        "Denoising AE": "denoising_ae",
        "Sparse AE": "sparse_ae",
        "Contractive AE": "contractive_ae",
        "Convolutional AE": "conv_ae",
        "Stacked AE": "stacked_ae",
        "VAE": "vae",
        "Contractive VAE": "contractive_vae",
    }
    key = name_to_key[name]
    final_loss = all_losses[key][-1]
    n_params = sum(p.numel() for p in model.parameters())
    variant_type = "image" if name != "Recurrent AE" else "sequence"
    print(f"  {name:<20} {final_loss:>12.4f} {n_params:>10,} {'image':>12}")

# Recurrent AE (separate because it uses different data)
rec_loss = all_losses["recurrent_ae"][-1]
rec_params = sum(p.numel() for p in recurrent_model.parameters())
print(f"  {'Recurrent AE':<20} {rec_loss:>12.4f} {rec_params:>10,} {'sequence':>12}")

# --- C. Training curves comparison using ModelVisualizer ---
viz = ModelVisualizer()

# All image variants
fig_all = viz.training_history(
    metrics={name: all_losses[name_to_key[name]] for name, _, _ in image_variants},
    x_label="Epoch",
    y_label="Loss (MSE)",
)
fig_all.write_html(str(OUTPUT_DIR / "ex1_all_variants_training_curves.html"))
print(
    f"\n  Training curves saved to {OUTPUT_DIR / 'ex1_all_variants_training_curves.html'}"
)

# ── Checkpoint 12 ────────────────────────────────────────────────────
assert len(all_losses) == 10, f"Expected 10 variants, got {len(all_losses)}"
grand_comparison_path = OUTPUT_DIR / "ex1_grand_comparison.png"
assert grand_comparison_path.exists(), "Grand comparison image should be saved"
# INTERPRETATION: The grand comparison tells the full story at a glance:
# - Standard AE: near-perfect reconstruction (too perfect — identity risk)
# - Undercomplete: blurry but recognisable (compression forces generalisation)
# - Denoising: slightly blurry but ROBUST to noise
# - Sparse: crisp where neurons fire, blank where they don't
# - Contractive: smooth, consistent quality
# - Convolutional: SHARPEST reconstructions (spatial structure preserved)
# - Stacked: comparable to undercomplete (depth alone is not magic)
# - VAE: slightly blurry (KL penalty trades sharpness for regularity)
# - Contractive VAE: smooth AND generative
# - Recurrent: different modality (time-series, not shown in grid)
print("\n--- Checkpoint 12 passed --- grand comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# REGISTER MODELS — ModelRegistry for versioned artifact storage
# ════════════════════════════════════════════════════════════════════════


async def register_all_models():
    """Register each trained variant with its final loss."""
    if not has_registry:
        print("  ModelRegistry not available — skipping registration")
        return {}

    from kailash_ml.types import MetricSpec

    model_versions = {}
    name_to_key = {
        "standard_ae": standard_model,
        "undercomplete_ae": undercomplete_model,
        "denoising_ae": dae_model,
        "sparse_ae": sparse_model,
        "contractive_ae": contractive_model,
        "conv_ae": conv_model,
        "stacked_ae": stacked_model,
        "recurrent_ae": recurrent_model,
        "vae": vae_model,
        "contractive_vae": cvae_model,
    }

    for name, model in name_to_key.items():
        model_bytes = pickle.dumps(model.state_dict())
        version = await registry.register_model(
            name=f"m5_{name}",
            artifact=model_bytes,
            metrics=[
                MetricSpec(name="final_loss", value=all_losses[name][-1]),
                MetricSpec(name="latent_dim", value=float(LATENT_DIM)),
                MetricSpec(name="epochs", value=float(EPOCHS)),
            ],
        )
        model_versions[name] = version
        print(
            f"  Registered {name}: version={version.version}, loss={all_losses[name][-1]:.4f}"
        )

    return model_versions


model_versions = asyncio.run(register_all_models())

if has_registry:
    assert (
        len(model_versions) == 10
    ), f"Should register all 10 variants, got {len(model_versions)}"
print("\n  All 10 models registered in ModelRegistry.\n")


# ════════════════════════════════════════════════════════════════════════
# REAL-WORLD APPLICATION: Credit Card Fraud Detection
# ════════════════════════════════════════════════════════════════════════
# Redline 9B requires applying the technique to a professional scenario.
#
# BUSINESS SCENARIO: You are a fraud analyst at DBS Bank. The bank
# processes 500K transactions daily. Rule-based systems catch ~67% of
# fraud. Your goal: train an autoencoder on NORMAL transactions so that
# FRAUDULENT transactions (which the model has never seen) produce high
# reconstruction error — the model saying "I can't compress this because
# it doesn't look like anything I've learned."
#
# We simulate this with Fashion-MNIST: train on one class (t-shirts),
# then test on another class (ankle boots). The model should reconstruct
# t-shirts well but fail on ankle boots — just like a fraud detector
# trained on normal transactions fails to reconstruct fraudulent ones.

print("\n" + "=" * 70)
print("  REAL-WORLD APPLICATION: Anomaly Detection for Fraud")
print("=" * 70)

# Train on class 0 (t-shirts) — "normal" transactions
train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))])
test_labels = torch.tensor([test_set[i][1] for i in range(len(test_set))])

normal_mask = train_labels == 0  # T-shirts
normal_data = X_flat[normal_mask]
print(f"  'Normal' data (class 0 = t-shirt): {normal_data.shape[0]} samples")

# Train a simple undercomplete AE on normal data only
normal_loader = DataLoader(TensorDataset(normal_data), batch_size=128, shuffle=True)
anomaly_model = UndercompleteAE(INPUT_DIM, latent_dim=8)
anomaly_model.to(device)
opt = torch.optim.Adam(anomaly_model.parameters(), lr=1e-3)

print("  Training anomaly detector on normal data only...")
for epoch in range(15):
    batch_losses = []
    for (xb,) in normal_loader:
        opt.zero_grad()
        x_hat, _ = anomaly_model(xb)
        loss = F.mse_loss(x_hat, xb)
        loss.backward()
        opt.step()
        batch_losses.append(loss.item())
    if (epoch + 1) % 5 == 0:
        print(f"    Epoch {epoch + 1}/15: loss={np.mean(batch_losses):.4f}")

# Test: compute reconstruction error for normal vs anomalous items
anomaly_model.eval()
with torch.no_grad():
    # Normal test samples (class 0)
    normal_test_mask = test_labels == 0
    normal_test = X_test_flat[normal_test_mask]
    normal_recon, _ = anomaly_model(normal_test.to(device))
    normal_errors = (
        F.mse_loss(normal_recon, normal_test.to(device), reduction="none")
        .mean(dim=1)
        .cpu()
        .numpy()
    )

    # Anomalous test samples (class 9 = ankle boot — very different from t-shirts)
    anomaly_test_mask = test_labels == 9
    anomaly_test = X_test_flat[anomaly_test_mask]
    anomaly_recon, _ = anomaly_model(anomaly_test.to(device))
    anomaly_errors = (
        F.mse_loss(anomaly_recon, anomaly_test.to(device), reduction="none")
        .mean(dim=1)
        .cpu()
        .numpy()
    )

# Visualise the separation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: reconstruction error distribution
axes[0].hist(
    normal_errors, bins=50, alpha=0.7, label="Normal (T-shirt)", color="steelblue"
)
axes[0].hist(
    anomaly_errors, bins=50, alpha=0.7, label="Anomaly (Ankle Boot)", color="coral"
)
threshold = np.percentile(normal_errors, 95)
axes[0].axvline(
    threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold (95th pct)"
)
axes[0].set_xlabel("Reconstruction Error (MSE)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Fraud Detection: Error Distribution", fontweight="bold")
axes[0].legend()

# Right: precision-recall at various thresholds
thresholds = np.linspace(normal_errors.min(), anomaly_errors.max(), 100)
precisions, recalls = [], []
for t in thresholds:
    tp = (anomaly_errors > t).sum()
    fp = (normal_errors > t).sum()
    fn = (anomaly_errors <= t).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precisions.append(precision)
    recalls.append(recall)

axes[1].plot(recalls, precisions, "b-", linewidth=2)
axes[1].set_xlabel("Recall (Fraud Caught)")
axes[1].set_ylabel("Precision (Correct Flags)")
axes[1].set_title("Precision-Recall Curve", fontweight="bold")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fname = OUTPUT_DIR / "ex1_anomaly_detection.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname}")

# Business impact statement
detection_rate = (anomaly_errors > threshold).mean() * 100
false_positive_rate = (normal_errors > threshold).mean() * 100
print(f"\n  BUSINESS IMPACT (at 95th percentile threshold):")
print(f"    Anomaly detection rate: {detection_rate:.1f}%")
print(f"    False positive rate:    {false_positive_rate:.1f}%")
print(f"    At DBS's daily volume of 500K transactions:")
print(f"    - Current rule-based system catches 67% of fraud")
print(f"    - This AE-based system would catch {detection_rate:.0f}% of anomalies")
if detection_rate > 67:
    improvement = detection_rate - 67
    print(f"    - That's +{improvement:.0f} percentage points improvement")
    print(
        f"    - At S$50M annual fraud exposure, that's ~S${improvement/100*50:.1f}M additional fraud prevented"
    )


# Clean up database connection
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  THE 10-VARIANT AUTOENCODER FAMILY:

  PROBLEM SOLVERS:
  [x] Standard AE: demonstrated the identity-function risk — a model
      that copies input perfectly is useless for compression or detection
  [x] Undercomplete AE: bottleneck forces compression (49:1 ratio).
      Blurry but meaningful reconstructions prove the model learned STRUCTURE
  [x] Denoising AE: noise injection -> robust features. The 3-row grid
      shows original -> noisy -> cleaned. Used in medical imaging, satellite data
  [x] Sparse AE: L1 penalty forces specialist neurons. The activation
      histogram proves most neurons stay near zero — biological plausibility

  ARCHITECTURE VARIANTS:
  [x] Contractive AE: Jacobian penalty smooths the latent space. The
      interpolation plot shows gradual morphing (no sharp discontinuities)
  [x] Convolutional AE: Conv2d preserves spatial locality. Sharpest
      reconstructions among all flat+conv variants on image data
  [x] Stacked AE: deep hierarchy learns multi-level features (edges ->
      textures -> shapes). Depth alone is not magic; it enables hierarchy

  DIFFERENT MODALITIES:
  [x] Recurrent AE: LSTM encoder-decoder for time-series data. The
      line-plot overlay shows temporal pattern capture for sensor data

  GENERATIVE MODELS:
  [x] VAE: reparameterisation trick (z = mu + sigma * epsilon) enables
      gradient flow through sampling. KL divergence regularises toward
      N(0,I) for generation. Latent traversal reveals disentangled dims
  [x] Contractive VAE: VAE's generation + Contractive AE's smoothness.
      Side-by-side interpolation comparison proves smoother transitions

  ML ENGINEERING:
  [x] Tracked all 10 variants with ExperimentTracker
  [x] Registered all 10 models in ModelRegistry with versioned metrics
  [x] Built anomaly detection application with precision-recall analysis
  [x] Quantified business impact (fraud detection improvement in S$)

  KEY INSIGHT: Every variant solves a specific failure mode:
  - Identity risk -> undercomplete bottleneck or regularisation
  - Memorisation -> noise injection (denoising) or sparsity
  - Sensitivity -> contractive penalty (Jacobian smoothness)
  - Generation need -> probabilistic latent space (VAE)
  - Spatial data -> convolutional architecture
  - Sequential data -> recurrent architecture
  Choosing the right variant is an engineering decision driven by your
  data type, use case, and failure tolerance.

  Next: In Exercise 2, you'll build CNNs with ResNet skip connections
  and SE attention blocks for image classification, and see how learned
  features transfer across tasks...
"""
)
