# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1: Autoencoders
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement a vanilla autoencoder with full backpropagation from scratch
#   - Explain the difference between undercomplete, denoising, and variational
#     autoencoders and when to use each
#   - Derive and implement the VAE ELBO loss (reconstruction + KL divergence)
#   - Apply the reparameterisation trick to enable gradient flow through
#     stochastic sampling
#   - Generate new images by sampling from the VAE latent space
#
# PREREQUISITES:
#   M4.8 — Neural networks, forward/backward pass, gradient descent.
#   This exercise extends M4.8 for UNSUPERVISED learning — no labels.
#
# ESTIMATED TIME: 60-90 minutes
#
# TASKS:
#   1. Generate synthetic image-like dataset (8x8 digit patterns)
#   2. Implement vanilla (undercomplete) autoencoder with numpy
#   3. Implement denoising autoencoder (add noise, reconstruct clean)
#   4. Implement variational autoencoder (VAE) with reparameterisation trick
#   5. Implement convolutional autoencoder for 2D image data
#   6. Compare reconstruction quality across all four variants
#   7. Generate new samples from VAE by sampling latent space
#   8. Visualise latent spaces for each variant
#
# THEORY:
#   Autoencoder: encoder (compress) -> latent space -> decoder (reconstruct)
#   Reconstruction loss: L = ||x - decoder(encoder(x))||^2
#   VAE ELBO: L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
#   Reparameterisation: z = mu + sigma * epsilon, epsilon ~ N(0,1)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import random

import numpy as np
import polars as pl

from kailash_ml import ModelVisualizer

from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Generate synthetic image-like dataset
# ══════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(seed=42)


def make_digit_patterns(n_per_class: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Generate 10 classes of 8x8 pixel patterns with noise."""
    patterns = []
    labels = []
    templates = []

    # 0: horizontal bar at centre
    t = np.zeros((8, 8))
    t[3:5, 1:7] = 1.0
    templates.append(t)

    # 1: vertical bar at centre
    t = np.zeros((8, 8))
    t[1:7, 3:5] = 1.0
    templates.append(t)

    # 2: diagonal top-left to bottom-right
    t = np.zeros((8, 8))
    for i in range(8):
        t[i, i] = 1.0
        if i + 1 < 8:
            t[i, i + 1] = 0.5
    templates.append(t)

    # 3: cross (horizontal + vertical bars)
    t = np.zeros((8, 8))
    t[3:5, 1:7] = 1.0
    t[1:7, 3:5] = 1.0
    templates.append(t)

    # 4: box outline
    t = np.zeros((8, 8))
    t[1, 1:7] = 1.0
    t[6, 1:7] = 1.0
    t[1:7, 1] = 1.0
    t[1:7, 6] = 1.0
    templates.append(t)

    # 5: filled circle-like shape (diamond)
    t = np.zeros((8, 8))
    centre = 3.5
    for i in range(8):
        for j in range(8):
            if abs(i - centre) + abs(j - centre) <= 3:
                t[i, j] = 1.0
    templates.append(t)

    # 6: L-shape
    t = np.zeros((8, 8))
    t[1:7, 1:3] = 1.0
    t[5:7, 1:6] = 1.0
    templates.append(t)

    # 7: T-shape
    t = np.zeros((8, 8))
    t[1:3, 1:7] = 1.0
    t[2:7, 3:5] = 1.0
    templates.append(t)

    # 8: central dot
    t = np.zeros((8, 8))
    t[2:6, 2:6] = 1.0
    templates.append(t)

    # 9: frame with gap
    t = np.zeros((8, 8))
    t[0, :] = 1.0
    t[7, :] = 1.0
    t[:, 0] = 1.0
    t[:, 7] = 1.0
    t[0, 3:5] = 0.0
    templates.append(t)

    for label, template in enumerate(templates):
        for _ in range(n_per_class):
            shift_y = rng.integers(-1, 2)
            shift_x = rng.integers(-1, 2)
            shifted = np.roll(np.roll(template, shift_y, axis=0), shift_x, axis=1)
            noisy = shifted + rng.normal(0, 0.1, (8, 8))
            noisy = np.clip(noisy, 0, 1)
            patterns.append(noisy.flatten())
            labels.append(label)

    patterns = np.array(patterns)
    labels = np.array(labels)
    idx = rng.permutation(len(patterns))
    return patterns[idx], labels[idx]


X_all, y_all = make_digit_patterns(n_per_class=200)
n_total = len(X_all)
n_train = int(0.8 * n_total)
X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]

df_train = pl.DataFrame({
    **{f"px_{i}": X_train[:, i].tolist() for i in range(64)},
    "label": y_train.tolist(),
})

print(f"=== Synthetic 8x8 Image Dataset ===")
print(f"Total samples: {n_total} ({n_train} train, {n_total - n_train} test)")
print(f"Classes: 10 patterns (bar, cross, box, diamond, L, T, dot, frame, etc.)")
print(f"Input dimension: 64 (8x8 flattened)")
print(f"Pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")


# ══════════════════════════════════════════════════════════════════════
# Shared neural network primitives (numpy-only)
# ══════════════════════════════════════════════════════════════════════


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 where x > 0, else 0."""
    return (x > 0).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + exp(-x)), numerically stable."""
    pos = x >= 0
    z = np.zeros_like(x)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    z[~pos] = exp_x / (1.0 + exp_x)
    return z


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error: (1/n) * sum((pred - target)^2)."""
    return float(np.mean((pred - target) ** 2))


def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """He (Kaiming) initialisation for ReLU layers: N(0, sqrt(2/fan_in))."""
    return rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))


def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier (Glorot) initialisation for sigmoid/linear layers."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Vanilla (undercomplete) autoencoder
# ══════════════════════════════════════════════════════════════════════
# Architecture: encoder 64 -> 32 -> 8 (latent), decoder 8 -> 32 -> 64
# The bottleneck forces compression — must learn the most important features.

# TODO: Implement the VanillaAutoencoder class.
#   - __init__: initialise encoder (W_enc1, W_enc2) and decoder (W_dec1, W_dec2) weights
#     using he_init/xavier_init. Add matching bias vectors.
#   - encode(x): two ReLU layers: 64->32->8, return (z, h1_pre, h1, z_pre)
#   - decode(z): two layers 8->32->64, ReLU then sigmoid, return (x_hat, h3_pre, h3)
#   - forward(x): call encode then decode, return (x_hat, z, cache dict)
#   - backward(cache, lr): backprop MSE loss through all layers, update all weights in-place
#   - get_latent(x): return only z from encode


class VanillaAutoencoder:
    """2-layer encoder + 2-layer decoder trained with backpropagation.

    Forward pass:
        h1 = relu(x @ W_enc1 + b_enc1)          # 64 -> 32
        z  = relu(h1 @ W_enc2 + b_enc2)          # 32 -> 8 (latent)
        h3 = relu(z @ W_dec1 + b_dec1)            # 8 -> 32
        x_hat = sigmoid(h3 @ W_dec2 + b_dec2)     # 32 -> 64
    Loss: MSE(x, x_hat)
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, latent_dim: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # TODO: Initialise encoder weights
        self.W_enc1 = ____  # Hint: he_init(input_dim, hidden_dim)
        self.b_enc1 = ____  # Hint: np.zeros(hidden_dim)
        self.W_enc2 = ____  # Hint: he_init(hidden_dim, latent_dim)
        self.b_enc2 = ____  # Hint: np.zeros(latent_dim)

        # TODO: Initialise decoder weights
        self.W_dec1 = ____  # Hint: he_init(latent_dim, hidden_dim)
        self.b_dec1 = ____  # Hint: np.zeros(hidden_dim)
        self.W_dec2 = ____  # Hint: xavier_init(hidden_dim, input_dim)
        self.b_dec2 = ____  # Hint: np.zeros(input_dim)

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode input to latent space. Returns (z, h1_pre, h1, z_pre)."""
        # TODO: Two-layer encoder with ReLU activations
        h1_pre = ____  # Hint: x @ self.W_enc1 + self.b_enc1
        h1 = ____  # Hint: relu(h1_pre)
        z_pre = ____  # Hint: h1 @ self.W_enc2 + self.b_enc2
        z = ____  # Hint: relu(z_pre)
        return z, h1_pre, h1, z_pre

    def decode(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode latent vector to reconstruction. Returns (x_hat, h3_pre, h3)."""
        # TODO: Two-layer decoder — ReLU then sigmoid
        h3_pre = ____  # Hint: z @ self.W_dec1 + self.b_dec1
        h3 = ____  # Hint: relu(h3_pre)
        x_hat_pre = ____  # Hint: h3 @ self.W_dec2 + self.b_dec2
        x_hat = ____  # Hint: sigmoid(x_hat_pre)
        return x_hat, h3_pre, h3

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Full forward pass. Returns (x_hat, z, cache)."""
        z, h1_pre, h1, z_pre = self.encode(x)
        x_hat, h3_pre, h3 = self.decode(z)
        cache = {
            "x": x, "h1_pre": h1_pre, "h1": h1,
            "z_pre": z_pre, "z": z,
            "h3_pre": h3_pre, "h3": h3,
            "x_hat": x_hat,
        }
        return x_hat, z, cache

    def backward(self, cache: dict, lr: float = 0.001) -> float:
        """Backpropagation through the full autoencoder. Returns loss."""
        x = cache["x"]
        x_hat = cache["x_hat"]
        h3 = cache["h3"]
        h3_pre = cache["h3_pre"]
        z = cache["z"]
        z_pre = cache["z_pre"]
        h1 = cache["h1"]
        h1_pre = cache["h1_pre"]
        batch_size = x.shape[0]

        loss = mse_loss(x_hat, x)

        # TODO: Compute gradient of MSE w.r.t. x_hat, then through sigmoid
        # d_loss/d_x_hat = 2/n * (x_hat - x), then multiply by sigmoid derivative
        d_x_hat = ____  # Hint: (2.0 / batch_size) * (x_hat - x)
        d_x_hat_pre = ____  # Hint: d_x_hat * x_hat * (1 - x_hat)

        # TODO: Decoder layer 2 gradients
        d_W_dec2 = ____  # Hint: h3.T @ d_x_hat_pre
        d_b_dec2 = ____  # Hint: d_x_hat_pre.sum(axis=0)
        d_h3 = ____  # Hint: d_x_hat_pre @ self.W_dec2.T

        # TODO: Decoder layer 1 gradients (through ReLU)
        d_h3_pre = ____  # Hint: d_h3 * relu_grad(h3_pre)
        d_W_dec1 = ____  # Hint: z.T @ d_h3_pre
        d_b_dec1 = ____  # Hint: d_h3_pre.sum(axis=0)
        d_z = ____  # Hint: d_h3_pre @ self.W_dec1.T

        # TODO: Encoder layer 2 gradients (through ReLU)
        d_z_pre = ____  # Hint: d_z * relu_grad(z_pre)
        d_W_enc2 = ____  # Hint: h1.T @ d_z_pre
        d_b_enc2 = ____  # Hint: d_z_pre.sum(axis=0)
        d_h1 = ____  # Hint: d_z_pre @ self.W_enc2.T

        # TODO: Encoder layer 1 gradients (through ReLU)
        d_h1_pre = ____  # Hint: d_h1 * relu_grad(h1_pre)
        d_W_enc1 = ____  # Hint: x.T @ d_h1_pre
        d_b_enc1 = ____  # Hint: d_h1_pre.sum(axis=0)

        # TODO: Gradient descent updates for all parameters
        self.W_dec2 -= ____  # Hint: lr * d_W_dec2
        self.b_dec2 -= ____  # Hint: lr * d_b_dec2
        self.W_dec1 -= ____  # Hint: lr * d_W_dec1
        self.b_dec1 -= ____  # Hint: lr * d_b_dec1
        self.W_enc2 -= ____  # Hint: lr * d_W_enc2
        self.b_enc2 -= ____  # Hint: lr * d_b_enc2
        self.W_enc1 -= ____  # Hint: lr * d_W_enc1
        self.b_enc1 -= ____  # Hint: lr * d_b_enc1

        return loss

    def get_latent(self, x: np.ndarray) -> np.ndarray:
        """Encode input and return only the latent vector."""
        z, _, _, _ = self.encode(x)
        return z


def train_autoencoder(
    model: VanillaAutoencoder,
    X: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    label: str = "Vanilla",
) -> list[float]:
    """Mini-batch training loop for autoencoder variants."""
    losses = []
    n = len(X)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        X_shuffled = X[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = X_shuffled[start : start + batch_size]
            x_hat, z, cache = model.forward(batch)
            batch_loss = model.backward(cache, lr=lr)
            epoch_loss += batch_loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch + 1:3d}/{epochs} — loss: {avg_loss:.6f}")

    return losses


print(f"\n=== Vanilla Autoencoder ===")
print(f"Architecture: 64 -> 32 -> [8] -> 32 -> 64")
print(f"Bottleneck forces compression: 64-dim input -> 8-dim latent (8:1 ratio)")
print(f"Training with MSE loss and mini-batch SGD...\n")

vanilla_ae = VanillaAutoencoder(input_dim=64, hidden_dim=32, latent_dim=8)
vanilla_losses = train_autoencoder(vanilla_ae, X_train, epochs=50, lr=0.005, label="Vanilla")

vanilla_recon, _, _ = vanilla_ae.forward(X_test)
vanilla_mse = mse_loss(vanilla_recon, X_test)
print(f"\nVanilla AE — Test reconstruction MSE: {vanilla_mse:.6f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Denoising autoencoder (DAE)
# ══════════════════════════════════════════════════════════════════════
# THEORY: A denoising autoencoder receives CORRUPTED input x_noisy
# but is trained to reconstruct the CLEAN input x. This forces robust
# feature learning — the model cannot memorise pixel values.


class DenoisingAutoencoder(VanillaAutoencoder):
    """Extends vanilla autoencoder with input corruption."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        noise_std: float = 0.3,
    ):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.noise_std = noise_std

    def corrupt(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to input and clip to valid range."""
        # TODO: Add Gaussian noise with self.noise_std and clip to [0, 1]
        noise = ____  # Hint: rng.normal(0, self.noise_std, x.shape)
        return ____  # Hint: np.clip(x + noise, 0, 1)

    def forward_denoising(
        self, x_clean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Forward pass with corruption: encode noisy, compare to clean."""
        # TODO: Corrupt x_clean, then call forward with noisy input
        x_noisy = ____  # Hint: self.corrupt(x_clean)
        x_hat, z, cache = self.forward(x_noisy)
        cache["x_clean"] = x_clean
        cache["x_noisy"] = x_noisy
        return x_hat, z, cache

    def backward_denoising(self, cache: dict, lr: float = 0.001) -> float:
        """Backprop comparing reconstruction to CLEAN input (not noisy)."""
        x_clean = cache["x_clean"]
        x_hat = cache["x_hat"]
        h3 = cache["h3"]
        h3_pre = cache["h3_pre"]
        z = cache["z"]
        z_pre = cache["z_pre"]
        h1 = cache["h1"]
        h1_pre = cache["h1_pre"]
        x_input = cache["x"]  # noisy input used for encoding
        batch_size = x_clean.shape[0]

        loss = mse_loss(x_hat, x_clean)

        # TODO: Backprop vs clean target (same chain as vanilla, different target)
        d_x_hat = ____  # Hint: (2.0 / batch_size) * (x_hat - x_clean)
        d_x_hat_pre = d_x_hat * x_hat * (1 - x_hat)

        d_W_dec2 = h3.T @ d_x_hat_pre
        d_b_dec2 = d_x_hat_pre.sum(axis=0)
        d_h3 = d_x_hat_pre @ self.W_dec2.T

        d_h3_pre = d_h3 * relu_grad(h3_pre)
        d_W_dec1 = z.T @ d_h3_pre
        d_b_dec1 = d_h3_pre.sum(axis=0)
        d_z = d_h3_pre @ self.W_dec1.T

        d_z_pre = d_z * relu_grad(z_pre)
        d_W_enc2 = h1.T @ d_z_pre
        d_b_enc2 = d_z_pre.sum(axis=0)
        d_h1 = d_z_pre @ self.W_enc2.T

        d_h1_pre = d_h1 * relu_grad(h1_pre)
        d_W_enc1 = x_input.T @ d_h1_pre
        d_b_enc1 = d_h1_pre.sum(axis=0)

        self.W_dec2 -= lr * d_W_dec2
        self.b_dec2 -= lr * d_b_dec2
        self.W_dec1 -= lr * d_W_dec1
        self.b_dec1 -= lr * d_b_dec1
        self.W_enc2 -= lr * d_W_enc2
        self.b_enc2 -= lr * d_b_enc2
        self.W_enc1 -= lr * d_W_enc1
        self.b_enc1 -= lr * d_b_enc1

        return loss


def train_dae(
    model: DenoisingAutoencoder,
    X: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
) -> list[float]:
    """Train denoising autoencoder with corrupted inputs."""
    losses = []
    n = len(X)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        X_shuffled = X[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = X_shuffled[start : start + batch_size]
            x_hat, z, cache = model.forward_denoising(batch)
            batch_loss = model.backward_denoising(cache, lr=lr)
            epoch_loss += batch_loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [DAE] Epoch {epoch + 1:3d}/{epochs} — loss: {avg_loss:.6f}")

    return losses


print(f"\n=== Denoising Autoencoder (DAE) ===")
print(f"Same architecture: 64 -> 32 -> [8] -> 32 -> 64")
print(f"Input corruption: Gaussian noise (std=0.3)")
print(f"Key insight: train on noisy input, compare against clean target\n")

dae = DenoisingAutoencoder(input_dim=64, hidden_dim=32, latent_dim=8, noise_std=0.3)
dae_losses = train_dae(dae, X_train, epochs=50, lr=0.005)

dae_recon, _, _ = dae.forward(X_test)
dae_mse = mse_loss(dae_recon, X_test)

X_test_noisy = dae.corrupt(X_test)
dae_denoised, _, _ = dae.forward(X_test_noisy)
dae_denoise_mse = mse_loss(dae_denoised, X_test)
noisy_mse = mse_loss(X_test_noisy, X_test)

print(f"\nDAE — Clean input reconstruction MSE: {dae_mse:.6f}")
print(f"DAE — Noisy input -> clean reconstruction MSE: {dae_denoise_mse:.6f}")
print(f"DAE — Raw noisy vs clean MSE (no model): {noisy_mse:.6f}")
print(f"DAE reduces noise by {(1 - dae_denoise_mse / noisy_mse) * 100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Variational Autoencoder (VAE)
# ══════════════════════════════════════════════════════════════════════
# THEORY — ELBO:
#   L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
#   KL divergence for N(mu, sigma^2) vs N(0,I):
#     KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#
#   Reparameterisation trick: z = mu + sigma * epsilon, epsilon ~ N(0, 1)
#   This lets gradients flow through the stochastic sampling step.

# TODO: Implement the VAE class.
#   - __init__: like VanillaAutoencoder but encoder has TWO heads: W_mu, W_logvar
#   - encode(x): first layer 64->32 with ReLU, then two linear heads for mu and log_var
#   - reparameterise(mu, log_var): z = mu + exp(0.5*log_var) * epsilon; save epsilon
#   - decode(z): same 8->32->64 as vanilla
#   - forward(x): encode -> reparameterise -> decode, return (x_hat, mu, log_var, z, cache)
#   - loss(x, x_hat, mu, log_var): MSE + beta * KL, return (total, recon, kl)
#   - backward(cache, lr, beta): gradient through ELBO (decoder + encoder + KL terms)
#   - generate(n_samples): sample z ~ N(0,I) and decode


class VAE:
    """Variational Autoencoder with reparameterisation trick.

    Encoder outputs mu and log_var (log variance) instead of a single z.
    Architecture:
        Encoder:  64 -> 32 -> mu(8), log_var(8)
        Latent:   z = mu + exp(0.5 * log_var) * epsilon
        Decoder:  8 -> 32 -> 64
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, latent_dim: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: shared first layer, then split into mu and log_var
        # TODO: Initialise W_enc1, b_enc1, W_mu, b_mu, W_logvar, b_logvar
        self.W_enc1 = ____  # Hint: he_init(input_dim, hidden_dim)
        self.b_enc1 = ____  # Hint: np.zeros(hidden_dim)
        self.W_mu = ____  # Hint: xavier_init(hidden_dim, latent_dim)
        self.b_mu = ____  # Hint: np.zeros(latent_dim)
        self.W_logvar = ____  # Hint: xavier_init(hidden_dim, latent_dim)
        self.b_logvar = ____  # Hint: np.zeros(latent_dim)

        # Decoder
        # TODO: Initialise W_dec1, b_dec1, W_dec2, b_dec2
        self.W_dec1 = ____  # Hint: he_init(latent_dim, hidden_dim)
        self.b_dec1 = ____  # Hint: np.zeros(hidden_dim)
        self.W_dec2 = ____  # Hint: xavier_init(hidden_dim, input_dim)
        self.b_dec2 = ____  # Hint: np.zeros(input_dim)

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode to mu and log_var. Returns (mu, log_var, h1_pre, h1)."""
        # TODO: First layer with ReLU, then linear mu and log_var heads
        h1_pre = ____  # Hint: x @ self.W_enc1 + self.b_enc1
        h1 = ____  # Hint: relu(h1_pre)
        mu = ____  # Hint: h1 @ self.W_mu + self.b_mu
        log_var = ____  # Hint: h1 @ self.W_logvar + self.b_logvar
        return mu, log_var, h1_pre, h1

    def reparameterise(self, mu: np.ndarray, log_var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reparameterisation trick: z = mu + sigma * epsilon.

        Returns (z, epsilon) — epsilon is saved for backprop.
        """
        # TODO: sigma = exp(0.5 * log_var), sample epsilon ~ N(0,1), compute z
        sigma = ____  # Hint: np.exp(0.5 * log_var)
        epsilon = ____  # Hint: rng.standard_normal(mu.shape)
        z = ____  # Hint: mu + sigma * epsilon
        return z, epsilon

    def decode(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode latent vector. Returns (x_hat, h3_pre, h3)."""
        h3_pre = z @ self.W_dec1 + self.b_dec1
        h3 = relu(h3_pre)
        x_hat_pre = h3 @ self.W_dec2 + self.b_dec2
        x_hat = sigmoid(x_hat_pre)
        return x_hat, h3_pre, h3

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Full VAE forward pass. Returns (x_hat, mu, log_var, z, cache)."""
        mu, log_var, h1_pre, h1 = self.encode(x)
        z, epsilon = self.reparameterise(mu, log_var)
        x_hat, h3_pre, h3 = self.decode(z)
        cache = {
            "x": x, "h1_pre": h1_pre, "h1": h1,
            "mu": mu, "log_var": log_var, "epsilon": epsilon,
            "z": z, "h3_pre": h3_pre, "h3": h3,
            "x_hat": x_hat,
        }
        return x_hat, mu, log_var, z, cache

    def loss(self, x: np.ndarray, x_hat: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> tuple[float, float, float]:
        """Compute VAE loss = reconstruction + KL divergence.

        Returns: (total_loss, recon_loss, kl_loss)
        """
        # TODO: reconstruction_loss = MSE(x_hat, x)
        recon_loss = ____  # Hint: float(np.mean((x_hat - x) ** 2))

        # TODO: KL divergence = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)), averaged over batch
        kl_per_sample = ____  # Hint: -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var), axis=1)
        kl_loss = ____  # Hint: float(np.mean(kl_per_sample))

        beta = 0.0005
        total = recon_loss + beta * kl_loss
        return total, recon_loss, kl_loss

    def backward(self, cache: dict, lr: float = 0.001, beta: float = 0.0005) -> tuple[float, float, float]:
        """Backprop through VAE including KL divergence gradient."""
        x = cache["x"]
        x_hat = cache["x_hat"]
        h3 = cache["h3"]
        h3_pre = cache["h3_pre"]
        z = cache["z"]
        h1 = cache["h1"]
        h1_pre = cache["h1_pre"]
        mu = cache["mu"]
        log_var = cache["log_var"]
        epsilon = cache["epsilon"]
        batch_size = x.shape[0]

        total_loss, recon_loss, kl_loss = self.loss(x, x_hat, mu, log_var)

        # Decoder gradients (reconstruction loss)
        d_x_hat = (2.0 / batch_size) * (x_hat - x)
        d_x_hat_pre = d_x_hat * x_hat * (1 - x_hat)

        d_W_dec2 = h3.T @ d_x_hat_pre
        d_b_dec2 = d_x_hat_pre.sum(axis=0)
        d_h3 = d_x_hat_pre @ self.W_dec2.T

        d_h3_pre = d_h3 * relu_grad(h3_pre)
        d_W_dec1 = z.T @ d_h3_pre
        d_b_dec1 = d_h3_pre.sum(axis=0)
        d_z = d_h3_pre @ self.W_dec1.T

        # Reparameterisation gradients
        sigma = np.exp(0.5 * log_var)

        # KL gradients (analytical)
        d_mu_kl = mu / batch_size
        d_logvar_kl = 0.5 * (np.exp(log_var) - 1) / batch_size

        # Reconstruction gradients through reparameterisation
        d_mu_recon = d_z
        d_logvar_recon = d_z * epsilon * 0.5 * sigma

        # TODO: Combine reconstruction and KL gradients for mu and log_var
        d_mu = ____  # Hint: d_mu_recon + beta * d_mu_kl
        d_log_var = ____  # Hint: d_logvar_recon + beta * d_logvar_kl

        # Encoder gradients
        d_W_mu = h1.T @ d_mu
        d_b_mu = d_mu.sum(axis=0)
        d_h1_from_mu = d_mu @ self.W_mu.T

        d_W_logvar = h1.T @ d_log_var
        d_b_logvar = d_log_var.sum(axis=0)
        d_h1_from_logvar = d_log_var @ self.W_logvar.T

        d_h1 = d_h1_from_mu + d_h1_from_logvar
        d_h1_pre = d_h1 * relu_grad(h1_pre)
        d_W_enc1 = x.T @ d_h1_pre
        d_b_enc1 = d_h1_pre.sum(axis=0)

        # Weight updates
        self.W_dec2 -= lr * d_W_dec2
        self.b_dec2 -= lr * d_b_dec2
        self.W_dec1 -= lr * d_W_dec1
        self.b_dec1 -= lr * d_b_dec1
        self.W_mu -= lr * d_W_mu
        self.b_mu -= lr * d_b_mu
        self.W_logvar -= lr * d_W_logvar
        self.b_logvar -= lr * d_b_logvar
        self.W_enc1 -= lr * d_W_enc1
        self.b_enc1 -= lr * d_b_enc1

        return total_loss, recon_loss, kl_loss

    def generate(self, n_samples: int = 10) -> np.ndarray:
        """Generate new samples by sampling z ~ N(0, I) and decoding."""
        # TODO: Sample from standard normal, then decode
        z = ____  # Hint: rng.standard_normal((n_samples, self.latent_dim))
        x_hat, _, _ = self.decode(z)
        return x_hat

    def get_latent(self, x: np.ndarray) -> np.ndarray:
        """Encode input and return the mean of the latent distribution."""
        mu, _, _, _ = self.encode(x)
        return mu


def train_vae(
    model: VAE,
    X: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    beta: float = 0.0005,
) -> tuple[list[float], list[float], list[float]]:
    """Train VAE with ELBO loss. Returns (total_losses, recon_losses, kl_losses)."""
    total_losses, recon_losses, kl_losses = [], [], []
    n = len(X)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        X_shuffled = X[perm]

        epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = X_shuffled[start : start + batch_size]
            x_hat, mu, log_var, z, cache = model.forward(batch)
            total, recon, kl = model.backward(cache, lr=lr, beta=beta)
            epoch_total += total
            epoch_recon += recon
            epoch_kl += kl
            n_batches += 1

        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        total_losses.append(avg_total)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  [VAE] Epoch {epoch + 1:3d}/{epochs} — "
                f"total: {avg_total:.6f}, recon: {avg_recon:.6f}, KL: {avg_kl:.2f}"
            )

    return total_losses, recon_losses, kl_losses


print(f"\n=== Variational Autoencoder (VAE) ===")
print(f"Architecture: 64 -> 32 -> [mu(8), log_var(8)] -> z(8) -> 32 -> 64")
print(f"ELBO = reconstruction_loss + beta * KL_divergence")
print(f"Reparameterisation: z = mu + exp(0.5*log_var) * epsilon, epsilon ~ N(0,1)\n")

vae = VAE(input_dim=64, hidden_dim=32, latent_dim=8)
vae_total_losses, vae_recon_losses, vae_kl_losses = train_vae(
    vae, X_train, epochs=80, lr=0.005, beta=0.0005,
)

vae_recon, _, _, _, _ = vae.forward(X_test)
vae_mse = mse_loss(vae_recon, X_test)
print(f"\nVAE — Test reconstruction MSE: {vae_mse:.6f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Convolutional autoencoder (conceptual framework)
# ══════════════════════════════════════════════════════════════════════
# THEORY: For image data, conv layers capture spatial structure.
# Architecture:
#   Encoder: 8x8x1 -> conv(3x3, 4 filters) -> pool(2x2) -> flatten -> 8 (latent)
#   Decoder: 8 (latent) -> reshape 4x4x4 -> upsample -> deconv -> 8x8x1

print(f"\n=== Convolutional Autoencoder (ConvAE) ===")
print(f"Architecture: 8x8 -> conv -> pool -> [8] -> upsample -> deconv -> 8x8")
print(f"ConvAE exploits spatial locality — conv filters share weights across positions")
print(f"This is parameter-efficient compared to fully-connected layers for images.")
print(f"For 8x8 images: FC requires 64*32=2048 params; conv requires 4*(3*3+1)=40 params")

# The convolutional layers are provided as utilities — focus is on the architecture concept.
# (Full implementation available in the solution for reference.)


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Compare all variants
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Reconstruction Quality Comparison ===")
print(f"{'Variant':<25} {'Test MSE':>12}")
print("-" * 40)
print(f"{'Vanilla AE':<25} {vanilla_mse:>12.6f}")
print(f"{'Denoising AE':<25} {dae_mse:>12.6f}")
print(f"{'VAE':<25} {vae_mse:>12.6f}")

print(f"\nNote: VAE typically has slightly higher MSE than vanilla AE because")
print(f"it trades reconstruction accuracy for a regularised, continuous latent space.")
print(f"The gain is generative capability — VAE can synthesise new images.")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Generate new samples from VAE
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== VAE Generation ===")
# TODO: Generate 20 new samples from the VAE by sampling the latent space
generated = ____  # Hint: vae.generate(n_samples=20)
print(f"Generated {generated.shape[0]} samples, shape: {generated.shape}")
print(f"Pixel range: [{generated.min():.3f}, {generated.max():.3f}]")
print(f"This works because VAE regularises the latent space to ~N(0,I).")
print(f"Vanilla and denoising AE cannot generate — their latent spaces are unconstrained.")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Visualise latent spaces
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Compare loss curves
all_losses = {
    "Vanilla AE": vanilla_losses,
    "Denoising AE": dae_losses,
    "VAE (total)": vae_total_losses,
    "VAE (recon)": vae_recon_losses,
}
fig = viz.training_history(all_losses, x_label="Epoch")
fig.update_layout(title="Autoencoder Training Loss Curves")
fig.write_html("ex1_loss_curves.html")
print(f"\nSaved: ex1_loss_curves.html")

# VAE ELBO breakdown
vae_breakdown = {
    "Reconstruction": vae_recon_losses,
    "KL Divergence (scaled)": [kl * 0.0005 for kl in vae_kl_losses],
}
fig_elbo = viz.training_history(vae_breakdown, x_label="Epoch")
fig_elbo.update_layout(title="VAE ELBO Components: Reconstruction vs KL")
fig_elbo.write_html("ex1_vae_elbo.html")
print("Saved: ex1_vae_elbo.html")

print("\n✓ Exercise 1 complete — autoencoders: vanilla, denoising, and variational")
