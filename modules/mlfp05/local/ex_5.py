# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 5: Generative Models — GANs and Diffusion
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement a GAN training loop with alternating generator/discriminator updates
#   - Explain mode collapse and detect it using mode coverage metrics
#   - Implement WGAN-GP and explain why Wasserstein distance is more stable
#   - Compare GAN vs VAE vs Diffusion models and select the right one
#   - Implement forward diffusion and understand the DDPM framework
#
# PREREQUISITES:
#   Exercise 1 (VAE, generative models). Exercise 2 (CNN training loops).
#   Understanding of binary cross-entropy loss and gradient descent.
#
# ESTIMATED TIME: 60-90 minutes
#
# TASKS:
#   1. Generate synthetic 2D data (mixture of Gaussians, concentric circles)
#   2. Implement basic GAN: Generator + Discriminator with alternating optimisation
#   3. Implement DCGAN-style training loop (feedforward on 2D data)
#   4. Implement WGAN with gradient penalty
#   5. Compare training stability: GAN vs WGAN (loss curves)
#   6. Detect and demonstrate mode collapse in basic GAN
#   7. Show how WGAN fixes mode collapse
#   8. Diffusion models: conceptual overview + pseudocode
#
# THEORY:
#   GAN minimax objective:
#     min_G max_D [ E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))] ]
#
#   The discriminator D tries to assign high probability to real data and
#   low probability to generated data. The generator G tries to fool D by
#   producing samples that D classifies as real.
#
#   WGAN objective (Wasserstein distance):
#     min_G max_{D in 1-Lipschitz} [ E_{x~p_data}[D(x)] - E_{z~p_z}[D(G(z))] ]
#
#   WGAN replaces JS divergence with Wasserstein-1 distance, which provides
#   meaningful gradients even when the supports of p_data and p_g don't overlap.
#   The Lipschitz constraint is enforced via gradient penalty:
#     GP = lambda * E_{x_hat}[ (||grad D(x_hat)||_2 - 1)^2 ]
#   where x_hat is a random interpolation between real and generated samples.
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import random

import numpy as np

from kailash_ml import ModelVisualizer


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Generate synthetic 2D data
# ══════════════════════════════════════════════════════════════════════
# We use two datasets:
#   (a) Mixture of 8 Gaussians arranged in a circle — good for detecting
#       mode collapse (a collapsed GAN will only capture a few modes)
#   (b) Concentric circles — tests whether the generator can learn
#       multi-modal, non-convex distributions


def generate_mixture_of_gaussians(
    n_samples: int, n_modes: int = 8, radius: float = 2.0, std: float = 0.05
) -> np.ndarray:
    """Generate n_samples from a mixture of Gaussians arranged in a circle.

    Each mode is a 2D Gaussian centred on a point along a circle of given
    radius, with isotropic standard deviation `std`.
    """
    rng = np.random.default_rng(seed=42)
    data = np.zeros((n_samples, 2))
    for i in range(n_samples):
        mode = i % n_modes
        angle = 2 * np.pi * mode / n_modes
        centre = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        data[i] = centre + rng.normal(0, std, size=2)
    return data


def generate_concentric_circles(
    n_samples: int, noise: float = 0.05
) -> np.ndarray:
    """Generate two concentric circles with added noise."""
    rng = np.random.default_rng(seed=42)
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner
    data = np.zeros((n_samples, 2))

    # Inner circle (radius = 0.5)
    angles_inner = rng.uniform(0, 2 * np.pi, n_inner)
    data[:n_inner, 0] = 0.5 * np.cos(angles_inner) + rng.normal(0, noise, n_inner)
    data[:n_inner, 1] = 0.5 * np.sin(angles_inner) + rng.normal(0, noise, n_inner)

    # Outer circle (radius = 1.5)
    angles_outer = rng.uniform(0, 2 * np.pi, n_outer)
    data[n_inner:, 0] = 1.5 * np.cos(angles_outer) + rng.normal(0, noise, n_outer)
    data[n_inner:, 1] = 1.5 * np.sin(angles_outer) + rng.normal(0, noise, n_outer)

    return data


n_samples = 2048
mog_data = generate_mixture_of_gaussians(n_samples, n_modes=8)
circle_data = generate_concentric_circles(n_samples)

print("=== Synthetic 2D Data ===")
print(f"Mixture of 8 Gaussians: {mog_data.shape}, range [{mog_data.min():.2f}, {mog_data.max():.2f}]")
print(f"Concentric circles:     {circle_data.shape}, range [{circle_data.min():.2f}, {circle_data.max():.2f}]")


# ── Neural Network Building Blocks (pure numpy) ─────────────────────
# These are shared by the GAN, DCGAN-style, and WGAN implementations.


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sigmoid_backward(s: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given sigmoid output s."""
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_backward(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def leaky_relu_backward(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)


def tanh_forward(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_backward(t: np.ndarray) -> np.ndarray:
    """Derivative of tanh given tanh output t."""
    return 1 - t ** 2


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He (Kaiming) initialisation for ReLU/LeakyReLU networks."""
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, (fan_in, fan_out))


def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier (Glorot) initialisation for sigmoid/tanh networks."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, (fan_in, fan_out))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement basic GAN from scratch
# ══════════════════════════════════════════════════════════════════════
# THEORY — GAN minimax game:
#   min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
#
# Generator G: maps noise z (latent space) → 2D points
# Discriminator D: maps 2D points → probability of being real
#
# Training alternates:
#   1. Update D to maximise log D(x_real) + log(1 - D(G(z)))
#   2. Update G to minimise log(1 - D(G(z)))  [equivalently, maximise log D(G(z))]


class Generator:
    """Generator network: noise_dim → hidden → hidden → 2D output."""

    def __init__(self, noise_dim: int = 2, hidden_dim: int = 128, rng_seed: int = 42):
        rng = np.random.default_rng(rng_seed)
        self.noise_dim = noise_dim

        # TODO: Initialise Generator weights for 3 layers
        #   Layer 1: noise_dim → hidden_dim (He init for ReLU)
        #   Layer 2: hidden_dim → hidden_dim (He init)
        #   Output:  hidden_dim → 2 (Xavier init, no activation on output)
        # Layer 1: noise_dim → hidden_dim
        self.W1 = ____  # Hint: he_init(noise_dim, hidden_dim, rng)
        self.b1 = ____  # Hint: np.zeros(hidden_dim)
        # Layer 2: hidden_dim → hidden_dim
        self.W2 = ____  # Hint: he_init(hidden_dim, hidden_dim, rng)
        self.b2 = ____  # Hint: np.zeros(hidden_dim)
        # Output layer: hidden_dim → 2 (2D points)
        self.W3 = ____  # Hint: xavier_init(hidden_dim, 2, rng)
        self.b3 = ____  # Hint: np.zeros(2)

    def forward(self, z: np.ndarray) -> dict:
        """Forward pass returning intermediate values for backprop."""
        # TODO: Implement forward pass through 3 layers
        #   h1 = relu(z @ W1 + b1), h2 = relu(h1 @ W2 + b2), out = h2 @ W3 + b3
        h1_pre = ____  # Hint: z @ self.W1 + self.b1
        h1 = ____  # Hint: relu(h1_pre)
        h2_pre = ____  # Hint: h1 @ self.W2 + self.b2
        h2 = ____  # Hint: relu(h2_pre)
        out = ____  # Hint: h2 @ self.W3 + self.b3 (linear output, no activation)
        return {"z": z, "h1_pre": h1_pre, "h1": h1, "h2_pre": h2_pre, "h2": h2, "out": out}

    def generate(self, z: np.ndarray) -> np.ndarray:
        """Generate 2D samples from noise."""
        return self.forward(z)["out"]


class Discriminator:
    """Discriminator network: 2D input → hidden → hidden → sigmoid probability."""

    def __init__(self, hidden_dim: int = 128, rng_seed: int = 123):
        rng = np.random.default_rng(rng_seed)

        # TODO: Initialise Discriminator weights for 3 layers
        #   Layer 1: 2 → hidden_dim (He init, LeakyReLU)
        #   Layer 2: hidden_dim → hidden_dim (He init)
        #   Output:  hidden_dim → 1 (Xavier init, sigmoid on output)
        # Layer 1: 2 → hidden_dim
        self.W1 = ____  # Hint: he_init(2, hidden_dim, rng)
        self.b1 = ____  # Hint: np.zeros(hidden_dim)
        # Layer 2: hidden_dim → hidden_dim
        self.W2 = ____  # Hint: he_init(hidden_dim, hidden_dim, rng)
        self.b2 = ____  # Hint: np.zeros(hidden_dim)
        # Output layer: hidden_dim → 1
        self.W3 = ____  # Hint: xavier_init(hidden_dim, 1, rng)
        self.b3 = ____  # Hint: np.zeros(1)

    def forward(self, x: np.ndarray) -> dict:
        """Forward pass returning intermediate values for backprop."""
        # TODO: Implement forward pass: LeakyReLU activations, sigmoid output
        #   h1 = leaky_relu(x @ W1 + b1), h2 = leaky_relu(h1 @ W2 + b2)
        #   logit = h2 @ W3 + b3, prob = sigmoid(logit)
        h1_pre = ____  # Hint: x @ self.W1 + self.b1
        h1 = ____  # Hint: leaky_relu(h1_pre)
        h2_pre = ____  # Hint: h1 @ self.W2 + self.b2
        h2 = ____  # Hint: leaky_relu(h2_pre)
        logit = ____  # Hint: h2 @ self.W3 + self.b3
        prob = ____  # Hint: sigmoid(logit)
        return {"x": x, "h1_pre": h1_pre, "h1": h1, "h2_pre": h2_pre, "h2": h2,
                "logit": logit, "prob": prob}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return probability that x is real."""
        return self.forward(x)["prob"]


def binary_cross_entropy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """BCE loss: -[t * log(p) + (1-t) * log(1-p)]."""
    pred_clipped = np.clip(pred, eps, 1 - eps)
    return -np.mean(target * np.log(pred_clipped) + (1 - target) * np.log(1 - pred_clipped))


def train_discriminator_step(
    D: Discriminator, G: Generator,
    real_data: np.ndarray, noise: np.ndarray,
    lr: float = 0.0002,
) -> float:
    """One step of discriminator training via backpropagation.

    Maximise: E[log D(x)] + E[log(1 - D(G(z)))]
    Equivalently minimise: -E[log D(x)] - E[log(1 - D(G(z)))]
    """
    batch_size = real_data.shape[0]

    # Forward pass on real data
    d_real = D.forward(real_data)
    real_labels = np.ones((batch_size, 1))

    # Forward pass on fake data (detach from G — don't backprop through G)
    fake_data = G.generate(noise)
    d_fake = D.forward(fake_data)
    fake_labels = np.zeros((batch_size, 1))

    # Loss
    loss_real = binary_cross_entropy(d_real["prob"], real_labels)
    loss_fake = binary_cross_entropy(d_fake["prob"], fake_labels)
    d_loss = loss_real + loss_fake

    # Backprop through D for real samples: dL/dprob_real = -(1/prob)
    eps = 1e-8
    dprob_real = -(real_labels / (d_real["prob"] + eps) - (1 - real_labels) / (1 - d_real["prob"] + eps)) / batch_size
    dlogit_real = dprob_real * sigmoid_backward(d_real["prob"])
    dW3_real = d_real["h2"].T @ dlogit_real
    db3_real = dlogit_real.sum(axis=0)
    dh2_real = dlogit_real @ D.W3.T
    dh2_pre_real = dh2_real * leaky_relu_backward(d_real["h2_pre"])
    dW2_real = d_real["h1"].T @ dh2_pre_real
    db2_real = dh2_pre_real.sum(axis=0)
    dh1_real = dh2_pre_real @ D.W2.T
    dh1_pre_real = dh1_real * leaky_relu_backward(d_real["h1_pre"])
    dW1_real = d_real["x"].T @ dh1_pre_real
    db1_real = dh1_pre_real.sum(axis=0)

    # Backprop through D for fake samples
    dprob_fake = -(fake_labels / (d_fake["prob"] + eps) - (1 - fake_labels) / (1 - d_fake["prob"] + eps)) / batch_size
    dlogit_fake = dprob_fake * sigmoid_backward(d_fake["prob"])
    dW3_fake = d_fake["h2"].T @ dlogit_fake
    db3_fake = dlogit_fake.sum(axis=0)
    dh2_fake = dlogit_fake @ D.W3.T
    dh2_pre_fake = dh2_fake * leaky_relu_backward(d_fake["h2_pre"])
    dW2_fake = d_fake["h1"].T @ dh2_pre_fake
    db2_fake = dh2_pre_fake.sum(axis=0)
    dh1_fake = dh2_pre_fake @ D.W2.T
    dh1_pre_fake = dh1_fake * leaky_relu_backward(d_fake["h1_pre"])
    dW1_fake = d_fake["x"].T @ dh1_pre_fake
    db1_fake = dh1_pre_fake.sum(axis=0)

    # Update D parameters
    D.W3 -= lr * (dW3_real + dW3_fake)
    D.b3 -= lr * (db3_real + db3_fake)
    D.W2 -= lr * (dW2_real + dW2_fake)
    D.b2 -= lr * (db2_real + db2_fake)
    D.W1 -= lr * (dW1_real + dW1_fake)
    D.b1 -= lr * (db1_real + db1_fake)

    return d_loss


def train_generator_step(
    D: Discriminator, G: Generator,
    noise: np.ndarray,
    lr: float = 0.0002,
) -> float:
    """One step of generator training via backpropagation.

    Maximise: E[log D(G(z))]  (non-saturating loss)
    Equivalently minimise: -E[log D(G(z))]
    """
    batch_size = noise.shape[0]
    eps = 1e-8

    # Forward: G(z) → D(G(z))
    g_cache = G.forward(noise)
    fake_data = g_cache["out"]
    d_cache = D.forward(fake_data)

    # Loss: -E[log D(G(z))]  (non-saturating form)
    g_loss = -np.mean(np.log(d_cache["prob"] + eps))

    # Backprop through D (frozen) then through G
    # dL/dprob = -1/(prob * batch_size)
    dprob = -1.0 / (d_cache["prob"] + eps) / batch_size
    dlogit = dprob * sigmoid_backward(d_cache["prob"])
    dh2 = dlogit @ D.W3.T
    dh2_pre = dh2 * leaky_relu_backward(d_cache["h2_pre"])
    dh1 = dh2_pre @ D.W2.T
    dh1_pre = dh1 * leaky_relu_backward(d_cache["h1_pre"])
    dfake = dh1_pre @ D.W1.T  # Gradient w.r.t. fake_data (= G's output)

    # Backprop through G
    # Output layer (linear — no activation)
    dW3_g = g_cache["h2"].T @ dfake
    db3_g = dfake.sum(axis=0)
    dh2_g = dfake @ G.W3.T
    dh2_pre_g = dh2_g * relu_backward(g_cache["h2_pre"])
    dW2_g = g_cache["h1"].T @ dh2_pre_g
    db2_g = dh2_pre_g.sum(axis=0)
    dh1_g = dh2_pre_g @ G.W2.T
    dh1_pre_g = dh1_g * relu_backward(g_cache["h1_pre"])
    dW1_g = g_cache["z"].T @ dh1_pre_g
    db1_g = dh1_pre_g.sum(axis=0)

    # Update G parameters
    G.W3 -= lr * dW3_g
    G.b3 -= lr * db3_g
    G.W2 -= lr * dW2_g
    G.b2 -= lr * db2_g
    G.W1 -= lr * dW1_g
    G.b1 -= lr * db1_g

    return g_loss


def train_gan(
    real_data: np.ndarray,
    n_epochs: int = 300,
    batch_size: int = 256,
    noise_dim: int = 2,
    lr: float = 0.0002,
    g_seed: int = 42,
    d_seed: int = 123,
    print_every: int = 50,
) -> tuple[Generator, Discriminator, dict]:
    """Train a basic GAN on 2D data."""
    rng = np.random.default_rng(seed=99)
    G = Generator(noise_dim=noise_dim, hidden_dim=128, rng_seed=g_seed)
    D = Discriminator(hidden_dim=128, rng_seed=d_seed)

    history = {"d_loss": [], "g_loss": [], "d_real_acc": [], "d_fake_acc": []}
    n = real_data.shape[0]

    for epoch in range(n_epochs):
        # Shuffle data
        perm = rng.permutation(n)
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch = real_data[perm[start:start + batch_size]]
            if batch.shape[0] < batch_size:
                continue
            noise = rng.normal(0, 1, (batch_size, noise_dim))

            d_loss = train_discriminator_step(D, G, batch, noise, lr=lr)
            g_loss = train_generator_step(D, G, noise, lr=lr)

            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            n_batches += 1

        if n_batches > 0:
            epoch_d_loss /= n_batches
            epoch_g_loss /= n_batches

        # Track discriminator accuracy
        test_noise = rng.normal(0, 1, (256, noise_dim))
        d_real_prob = D.predict(real_data[:256]).mean()
        d_fake_prob = D.predict(G.generate(test_noise)).mean()

        history["d_loss"].append(epoch_d_loss)
        history["g_loss"].append(epoch_g_loss)
        history["d_real_acc"].append(float(d_real_prob))
        history["d_fake_acc"].append(float(d_fake_prob))

        if (epoch + 1) % print_every == 0:
            print(
                f"  Epoch {epoch + 1:4d}/{n_epochs} | "
                f"D loss: {epoch_d_loss:.4f} | G loss: {epoch_g_loss:.4f} | "
                f"D(real): {d_real_prob:.3f} | D(fake): {d_fake_prob:.3f}"
            )

    return G, D, history


print("\n=== TASK 2: Training Basic GAN on Mixture of Gaussians ===")
print("Architecture: G(2→128→128→2), D(2→128→128→1)")
print("Training with alternating optimisation (BCE loss)\n")

G_basic, D_basic, history_basic = train_gan(
    mog_data, n_epochs=300, batch_size=256, lr=0.0002, print_every=50,
)

# Evaluate final generation quality
rng_eval = np.random.default_rng(seed=0)
final_noise = rng_eval.normal(0, 1, (2048, 2))
generated_basic = G_basic.generate(final_noise)

print(f"\nGenerated data range: x=[{generated_basic[:, 0].min():.2f}, {generated_basic[:, 0].max():.2f}], "
      f"y=[{generated_basic[:, 1].min():.2f}, {generated_basic[:, 1].max():.2f}]")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: DCGAN-style training loop
# ══════════════════════════════════════════════════════════════════════
# THEORY — DCGAN architecture guidelines (Radford et al. 2015):
#   1. Replace pooling with strided convolutions (D) / fractional-strided convolutions (G)
#   2. Use batch normalisation in both G and D (except D input and G output)
#   3. Remove fully connected hidden layers for deeper architectures
#   4. Use ReLU in G (except output: tanh), LeakyReLU in D
#
# On 2D data we cannot use convolutions, so we implement a DCGAN-style
# loop with batch normalisation — the key DCGAN innovation for stability.


def batch_norm_forward(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
    running_mean: np.ndarray, running_var: np.ndarray,
    training: bool = True, momentum: float = 0.1, eps: float = 1e-5,
) -> tuple[np.ndarray, dict]:
    """Batch normalisation: normalise activations per feature across the batch.

    During training: use batch statistics.
    During inference: use running (exponential moving average) statistics.
    """
    if training:
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        x_norm = (x - batch_mean) / np.sqrt(batch_var + eps)
        # Update running statistics
        running_mean[:] = (1 - momentum) * running_mean + momentum * batch_mean
        running_var[:] = (1 - momentum) * running_var + momentum * batch_var
    else:
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)

    out = gamma * x_norm + beta
    cache = {"x": x, "x_norm": x_norm, "gamma": gamma, "batch_mean": x.mean(axis=0),
             "batch_var": x.var(axis=0), "eps": eps}
    return out, cache


def batch_norm_backward(dout: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass for batch normalisation."""
    x, x_norm, gamma = cache["x"], cache["x_norm"], cache["gamma"]
    batch_mean, batch_var, eps = cache["batch_mean"], cache["batch_var"], cache["eps"]
    N = x.shape[0]

    dgamma = (dout * x_norm).sum(axis=0)
    dbeta = dout.sum(axis=0)

    dx_norm = dout * gamma
    std_inv = 1.0 / np.sqrt(batch_var + eps)

    dx = (1.0 / N) * std_inv * (
        N * dx_norm
        - dx_norm.sum(axis=0)
        - x_norm * (dx_norm * x_norm).sum(axis=0)
    )
    return dx, dgamma, dbeta


class DCGANGenerator:
    """DCGAN-style generator with batch normalisation.

    Architecture: noise → FC+BN+ReLU → FC+BN+ReLU → FC+Tanh → 2D
    Tanh output bounds the generator's range, stabilising training.
    """

    def __init__(self, noise_dim: int = 2, hidden_dim: int = 128, rng_seed: int = 42):
        rng = np.random.default_rng(rng_seed)
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim

        # TODO: Initialise weights for 3-layer DCGAN Generator
        #   Layer 1: noise_dim → hidden_dim with BN (gamma=ones, beta=zeros,
        #            running_mean=zeros, running_var=ones)
        #   Layer 2: hidden_dim → hidden_dim with BN
        #   Output:  hidden_dim → 2 (no BN on G output — DCGAN rule)
        # Layer 1
        self.W1 = ____  # Hint: he_init(noise_dim, hidden_dim, rng)
        self.b1 = ____  # Hint: np.zeros(hidden_dim)
        self.gamma1 = ____  # Hint: np.ones(hidden_dim)
        self.beta1 = ____  # Hint: np.zeros(hidden_dim)
        self.running_mean1 = ____  # Hint: np.zeros(hidden_dim)
        self.running_var1 = ____  # Hint: np.ones(hidden_dim)

        # Layer 2
        self.W2 = ____  # Hint: he_init(hidden_dim, hidden_dim, rng)
        self.b2 = ____  # Hint: np.zeros(hidden_dim)
        self.gamma2 = ____  # Hint: np.ones(hidden_dim)
        self.beta2 = ____  # Hint: np.zeros(hidden_dim)
        self.running_mean2 = ____  # Hint: np.zeros(hidden_dim)
        self.running_var2 = ____  # Hint: np.ones(hidden_dim)

        # Output layer (no BN on G output — DCGAN rule)
        self.W3 = ____  # Hint: xavier_init(hidden_dim, 2, rng)
        self.b3 = ____  # Hint: np.zeros(2)

    def forward(self, z: np.ndarray, training: bool = True) -> dict:
        """Forward pass with batch norm."""
        pre1 = z @ self.W1 + self.b1
        bn1, bn_cache1 = batch_norm_forward(pre1, self.gamma1, self.beta1,
                                             self.running_mean1, self.running_var1, training)
        h1 = relu(bn1)

        pre2 = h1 @ self.W2 + self.b2
        bn2, bn_cache2 = batch_norm_forward(pre2, self.gamma2, self.beta2,
                                             self.running_mean2, self.running_var2, training)
        h2 = relu(bn2)

        pre_out = h2 @ self.W3 + self.b3
        out = tanh_forward(pre_out)  # Tanh on output — DCGAN convention

        return {"z": z, "pre1": pre1, "bn1": bn1, "bn_cache1": bn_cache1, "h1": h1,
                "pre2": pre2, "bn2": bn2, "bn_cache2": bn_cache2, "h2": h2,
                "pre_out": pre_out, "out": out}

    def generate(self, z: np.ndarray, training: bool = False) -> np.ndarray:
        return self.forward(z, training=training)["out"]


class DCGANDiscriminator:
    """DCGAN-style discriminator with LeakyReLU (no BN on input layer)."""

    def __init__(self, hidden_dim: int = 128, rng_seed: int = 123):
        rng = np.random.default_rng(rng_seed)
        self.hidden_dim = hidden_dim

        # TODO: Initialise weights for DCGANDiscriminator
        #   Layer 1: 2 → hidden_dim, no BN (DCGAN rule: no BN on D input)
        #   Layer 2: hidden_dim → hidden_dim with BN
        #   Output:  hidden_dim → 1 (no BN on output)
        # Layer 1 (no BN on D input — DCGAN rule)
        self.W1 = ____  # Hint: he_init(2, hidden_dim, rng)
        self.b1 = ____  # Hint: np.zeros(hidden_dim)

        # Layer 2 (with BN)
        self.W2 = ____  # Hint: he_init(hidden_dim, hidden_dim, rng)
        self.b2 = ____  # Hint: np.zeros(hidden_dim)
        self.gamma2 = ____  # Hint: np.ones(hidden_dim)
        self.beta2 = ____  # Hint: np.zeros(hidden_dim)
        self.running_mean2 = ____  # Hint: np.zeros(hidden_dim)
        self.running_var2 = ____  # Hint: np.ones(hidden_dim)

        # Output layer
        self.W3 = ____  # Hint: xavier_init(hidden_dim, 1, rng)
        self.b3 = ____  # Hint: np.zeros(1)

    def forward(self, x: np.ndarray, training: bool = True) -> dict:
        pre1 = x @ self.W1 + self.b1
        h1 = leaky_relu(pre1)

        pre2 = h1 @ self.W2 + self.b2
        bn2, bn_cache2 = batch_norm_forward(pre2, self.gamma2, self.beta2,
                                             self.running_mean2, self.running_var2, training)
        h2 = leaky_relu(bn2)

        logit = h2 @ self.W3 + self.b3
        prob = sigmoid(logit)
        return {"x": x, "pre1": pre1, "h1": h1, "pre2": pre2, "bn2": bn2,
                "bn_cache2": bn_cache2, "h2": h2, "logit": logit, "prob": prob}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, training=False)["prob"]


def train_dcgan_step_d(
    D: DCGANDiscriminator, G: DCGANGenerator,
    real_data: np.ndarray, noise: np.ndarray, lr: float = 0.0002,
) -> float:
    """Discriminator step for DCGAN with batch norm backprop."""
    batch_size = real_data.shape[0]
    eps = 1e-8

    # Forward on real
    d_real = D.forward(real_data, training=True)
    # Forward on fake (G in training mode for BN stats)
    fake = G.forward(noise, training=True)["out"]
    d_fake = D.forward(fake, training=True)

    loss = binary_cross_entropy(d_real["prob"], np.ones((batch_size, 1))) + \
           binary_cross_entropy(d_fake["prob"], np.zeros((batch_size, 1)))

    # Backprop for real path
    dprob_r = -(1.0 / (d_real["prob"] + eps)) / batch_size
    dlogit_r = dprob_r * sigmoid_backward(d_real["prob"])
    dW3_r = d_real["h2"].T @ dlogit_r
    db3_r = dlogit_r.sum(axis=0)
    dh2_r = dlogit_r @ D.W3.T
    dbn2_r = dh2_r * leaky_relu_backward(d_real["bn2"])
    dpre2_r, dgamma2_r, dbeta2_r = batch_norm_backward(dbn2_r, d_real["bn_cache2"])
    dW2_r = d_real["h1"].T @ dpre2_r
    db2_r = dpre2_r.sum(axis=0)
    dh1_r = dpre2_r @ D.W2.T
    dpre1_r = dh1_r * leaky_relu_backward(d_real["pre1"])
    dW1_r = d_real["x"].T @ dpre1_r
    db1_r = dpre1_r.sum(axis=0)

    # Backprop for fake path
    dprob_f = (1.0 / (1 - d_fake["prob"] + eps)) / batch_size
    dlogit_f = dprob_f * sigmoid_backward(d_fake["prob"])
    dW3_f = d_fake["h2"].T @ dlogit_f
    db3_f = dlogit_f.sum(axis=0)
    dh2_f = dlogit_f @ D.W3.T
    dbn2_f = dh2_f * leaky_relu_backward(d_fake["bn2"])
    dpre2_f, dgamma2_f, dbeta2_f = batch_norm_backward(dbn2_f, d_fake["bn_cache2"])
    dW2_f = d_fake["h1"].T @ dpre2_f
    db2_f = dpre2_f.sum(axis=0)
    dh1_f = dpre2_f @ D.W2.T
    dpre1_f = dh1_f * leaky_relu_backward(d_fake["pre1"])
    dW1_f = d_fake["x"].T @ dpre1_f
    db1_f = dpre1_f.sum(axis=0)

    # Update D
    D.W3 -= lr * (dW3_r + dW3_f)
    D.b3 -= lr * (db3_r + db3_f)
    D.W2 -= lr * (dW2_r + dW2_f)
    D.b2 -= lr * (db2_r + db2_f)
    D.gamma2 -= lr * (dgamma2_r + dgamma2_f)
    D.beta2 -= lr * (dbeta2_r + dbeta2_f)
    D.W1 -= lr * (dW1_r + dW1_f)
    D.b1 -= lr * (db1_r + db1_f)

    return loss


def train_dcgan_step_g(
    D: DCGANDiscriminator, G: DCGANGenerator,
    noise: np.ndarray, lr: float = 0.0002,
) -> float:
    """Generator step for DCGAN with batch norm backprop."""
    batch_size = noise.shape[0]
    eps = 1e-8

    g_cache = G.forward(noise, training=True)
    fake = g_cache["out"]
    d_cache = D.forward(fake, training=True)

    g_loss = -np.mean(np.log(d_cache["prob"] + eps))

    # Backprop through D (frozen) to get gradient w.r.t. fake
    dprob = -1.0 / (d_cache["prob"] + eps) / batch_size
    dlogit = dprob * sigmoid_backward(d_cache["prob"])
    dh2 = dlogit @ D.W3.T
    dbn2 = dh2 * leaky_relu_backward(d_cache["bn2"])
    dpre2, _, _ = batch_norm_backward(dbn2, d_cache["bn_cache2"])
    dh1 = dpre2 @ D.W2.T
    dpre1 = dh1 * leaky_relu_backward(d_cache["pre1"])
    dfake = dpre1 @ D.W1.T

    # Backprop through G (tanh output)
    dpre_out = dfake * tanh_backward(g_cache["out"])
    dW3_g = g_cache["h2"].T @ dpre_out
    db3_g = dpre_out.sum(axis=0)
    dh2_g = dpre_out @ G.W3.T
    dbn2_g = dh2_g * relu_backward(g_cache["bn2"])
    dpre2_g, dgamma2_g, dbeta2_g = batch_norm_backward(dbn2_g, g_cache["bn_cache2"])
    dW2_g = g_cache["h1"].T @ dpre2_g
    db2_g = dpre2_g.sum(axis=0)
    dh1_g = dpre2_g @ G.W2.T
    dbn1_g = dh1_g * relu_backward(g_cache["bn1"])
    dpre1_g, dgamma1_g, dbeta1_g = batch_norm_backward(dbn1_g, g_cache["bn_cache1"])
    dW1_g = g_cache["z"].T @ dpre1_g
    db1_g = dpre1_g.sum(axis=0)

    # Update G
    G.W3 -= lr * dW3_g
    G.b3 -= lr * db3_g
    G.W2 -= lr * dW2_g
    G.b2 -= lr * db2_g
    G.gamma2 -= lr * dgamma2_g
    G.beta2 -= lr * dbeta2_g
    G.W1 -= lr * dW1_g
    G.b1 -= lr * db1_g
    G.gamma1 -= lr * dgamma1_g
    G.beta1 -= lr * dbeta1_g

    return g_loss


def train_dcgan(
    real_data: np.ndarray,
    n_epochs: int = 300,
    batch_size: int = 256,
    noise_dim: int = 2,
    lr: float = 0.0002,
    print_every: int = 50,
) -> tuple[DCGANGenerator, DCGANDiscriminator, dict]:
    """Train DCGAN-style GAN with batch normalisation."""
    rng = np.random.default_rng(seed=99)

    # Scale data to [-1, 1] for tanh output
    data_min = real_data.min(axis=0)
    data_max = real_data.max(axis=0)
    data_range = data_max - data_min
    data_range = np.where(data_range == 0, 1.0, data_range)
    scaled_data = 2.0 * (real_data - data_min) / data_range - 1.0

    G = DCGANGenerator(noise_dim=noise_dim, hidden_dim=128)
    D = DCGANDiscriminator(hidden_dim=128)
    n = scaled_data.shape[0]

    history = {"d_loss": [], "g_loss": []}

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        epoch_d, epoch_g = 0.0, 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch = scaled_data[perm[start:start + batch_size]]
            if batch.shape[0] < batch_size:
                continue
            noise = rng.normal(0, 1, (batch_size, noise_dim))

            d_loss = train_dcgan_step_d(D, G, batch, noise, lr=lr)
            g_loss = train_dcgan_step_g(D, G, noise, lr=lr)

            epoch_d += d_loss
            epoch_g += g_loss
            n_batches += 1

        if n_batches > 0:
            epoch_d /= n_batches
            epoch_g /= n_batches

        history["d_loss"].append(epoch_d)
        history["g_loss"].append(epoch_g)

        if (epoch + 1) % print_every == 0:
            print(f"  Epoch {epoch + 1:4d}/{n_epochs} | D loss: {epoch_d:.4f} | G loss: {epoch_g:.4f}")

    return G, D, history


print("\n=== TASK 3: Training DCGAN-style GAN (with Batch Norm) ===")
print("Key DCGAN innovations: batch norm in G and D, tanh output, LeakyReLU in D\n")

G_dcgan, D_dcgan, history_dcgan = train_dcgan(
    mog_data, n_epochs=300, batch_size=256, lr=0.0002, print_every=50,
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Implement WGAN with gradient penalty
# ══════════════════════════════════════════════════════════════════════
# THEORY — Wasserstein distance and gradient penalty:
#
#   Standard GAN uses Jensen-Shannon divergence, which can be flat when
#   distributions don't overlap, causing vanishing gradients for G.
#
#   WGAN objective: min_G max_{D: ||D||_L <= 1} E[D(x)] - E[D(G(z))]
#
#   The critic D (not "discriminator" — no sigmoid, outputs a score)
#   must be 1-Lipschitz. WGAN-GP enforces this with gradient penalty:
#     GP = lambda * E_x_hat[ (||grad_x_hat D(x_hat)||_2 - 1)^2 ]
#   where x_hat = epsilon * x_real + (1 - epsilon) * x_fake
#
#   Key benefits:
#   - Meaningful loss correlates with sample quality (unlike BCE loss)
#   - Stable gradients even when distributions have different supports
#   - Prevents mode collapse: critic provides useful gradients for all modes


class WGANCritic:
    """WGAN critic: no sigmoid on output, outputs a score (not a probability).

    Architecture: 2D → 128 (LeakyReLU) → 128 (LeakyReLU) → 1 (linear)
    No batch norm in critic (WGAN-GP paper recommendation).
    """

    def __init__(self, hidden_dim: int = 128, rng_seed: int = 123):
        rng = np.random.default_rng(rng_seed)
        # TODO: Initialise WGANCritic weights
        #   2 → hidden_dim → hidden_dim → 1 (no sigmoid, no batch norm)
        self.W1 = ____  # Hint: he_init(2, hidden_dim, rng)
        self.b1 = ____  # Hint: np.zeros(hidden_dim)
        self.W2 = ____  # Hint: he_init(hidden_dim, hidden_dim, rng)
        self.b2 = ____  # Hint: np.zeros(hidden_dim)
        self.W3 = ____  # Hint: xavier_init(hidden_dim, 1, rng)
        self.b3 = ____  # Hint: np.zeros(1)

    def forward(self, x: np.ndarray) -> dict:
        # TODO: Forward pass for critic (LeakyReLU hidden, linear output — no sigmoid)
        h1_pre = ____  # Hint: x @ self.W1 + self.b1
        h1 = ____  # Hint: leaky_relu(h1_pre)
        h2_pre = ____  # Hint: h1 @ self.W2 + self.b2
        h2 = ____  # Hint: leaky_relu(h2_pre)
        score = ____  # Hint: h2 @ self.W3 + self.b3  (no sigmoid — raw score)
        return {"x": x, "h1_pre": h1_pre, "h1": h1, "h2_pre": h2_pre, "h2": h2, "score": score}

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)["score"]


class WGANGenerator:
    """WGAN generator — identical to basic Generator but with batch norm."""

    def __init__(self, noise_dim: int = 2, hidden_dim: int = 128, rng_seed: int = 42):
        rng = np.random.default_rng(rng_seed)
        self.noise_dim = noise_dim
        # TODO: Initialise WGANGenerator weights (same structure as basic Generator)
        self.W1 = ____  # Hint: he_init(noise_dim, hidden_dim, rng)
        self.b1 = ____  # Hint: np.zeros(hidden_dim)
        self.W2 = ____  # Hint: he_init(hidden_dim, hidden_dim, rng)
        self.b2 = ____  # Hint: np.zeros(hidden_dim)
        self.W3 = ____  # Hint: xavier_init(hidden_dim, 2, rng)
        self.b3 = ____  # Hint: np.zeros(2)

    def forward(self, z: np.ndarray) -> dict:
        # TODO: Forward pass (same as basic Generator: ReLU hidden, linear output)
        h1_pre = ____  # Hint: z @ self.W1 + self.b1
        h1 = ____  # Hint: relu(h1_pre)
        h2_pre = ____  # Hint: h1 @ self.W2 + self.b2
        h2 = ____  # Hint: relu(h2_pre)
        out = ____  # Hint: h2 @ self.W3 + self.b3
        return {"z": z, "h1_pre": h1_pre, "h1": h1, "h2_pre": h2_pre, "h2": h2, "out": out}

    def generate(self, z: np.ndarray) -> np.ndarray:
        return self.forward(z)["out"]


def compute_gradient_penalty(
    C: WGANCritic,
    real_data: np.ndarray,
    fake_data: np.ndarray,
    lam: float = 10.0,
) -> tuple[float, dict]:
    """Compute gradient penalty for WGAN-GP.

    x_hat = epsilon * real + (1 - epsilon) * fake
    GP = lambda * E[ (||grad C(x_hat)||_2 - 1)^2 ]

    We compute gradients using numerical differentiation (finite differences)
    since we're implementing from scratch without autograd.
    """
    rng = np.random.default_rng()
    batch_size = real_data.shape[0]
    epsilon = rng.uniform(0, 1, (batch_size, 1))
    # TODO: Compute interpolated samples x_hat between real and fake
    x_hat = ____  # Hint: epsilon * real_data + (1 - epsilon) * fake_data

    # Compute gradient of critic output w.r.t. x_hat via analytical backprop
    cache = C.forward(x_hat)
    # dScore/dx_hat via chain rule
    dscore = np.ones((batch_size, 1))  # dL/dscore = 1 (we want gradient of score itself)
    dh2 = dscore @ C.W3.T
    dh2_pre = dh2 * leaky_relu_backward(cache["h2_pre"])
    dh1 = dh2_pre @ C.W2.T
    dh1_pre = dh1 * leaky_relu_backward(cache["h1_pre"])
    dx = dh1_pre @ C.W1.T  # Gradient w.r.t. input x_hat

    # Gradient norm
    grad_norm = np.sqrt((dx ** 2).sum(axis=1) + 1e-12)
    # TODO: Compute gradient penalty: lambda * mean((||grad|| - 1)^2)
    gp = ____  # Hint: lam * np.mean((grad_norm - 1.0) ** 2)

    return gp, {"x_hat": x_hat, "grad_norm": grad_norm, "dx": dx}


def train_wgan_critic_step(
    C: WGANCritic, G: WGANGenerator,
    real_data: np.ndarray, noise: np.ndarray,
    lr: float = 0.0001, lam: float = 10.0,
) -> float:
    """One critic step for WGAN-GP.

    Maximise: E[C(x_real)] - E[C(G(z))] - lambda * GP
    Equivalently minimise: -E[C(x_real)] + E[C(G(z))] + lambda * GP
    """
    batch_size = real_data.shape[0]

    # Forward
    c_real = C.forward(real_data)
    fake_data = G.generate(noise)
    c_fake = C.forward(fake_data)

    # Wasserstein loss
    w_loss = -(c_real["score"].mean() - c_fake["score"].mean())

    # Gradient penalty
    gp, gp_cache = compute_gradient_penalty(C, real_data, fake_data, lam)
    total_loss = w_loss + gp

    # Backprop for real path: maximise C(x_real) → gradient is +1/batch_size per sample
    dscore_real = -np.ones((batch_size, 1)) / batch_size
    dh2_r = dscore_real @ C.W3.T
    dW3_r = c_real["h2"].T @ dscore_real
    db3_r = dscore_real.sum(axis=0)
    dh2_pre_r = dh2_r * leaky_relu_backward(c_real["h2_pre"])
    dW2_r = c_real["h1"].T @ dh2_pre_r
    db2_r = dh2_pre_r.sum(axis=0)
    dh1_r = dh2_pre_r @ C.W2.T
    dh1_pre_r = dh1_r * leaky_relu_backward(c_real["h1_pre"])
    dW1_r = c_real["x"].T @ dh1_pre_r
    db1_r = dh1_pre_r.sum(axis=0)

    # Backprop for fake path: minimise C(G(z)) → gradient is +1/batch_size
    dscore_fake = np.ones((batch_size, 1)) / batch_size
    dh2_f = dscore_fake @ C.W3.T
    dW3_f = c_fake["h2"].T @ dscore_fake
    db3_f = dscore_fake.sum(axis=0)
    dh2_pre_f = dh2_f * leaky_relu_backward(c_fake["h2_pre"])
    dW2_f = c_fake["h1"].T @ dh2_pre_f
    db2_f = dh2_pre_f.sum(axis=0)
    dh1_f = dh2_pre_f @ C.W2.T
    dh1_pre_f = dh1_f * leaky_relu_backward(c_fake["h1_pre"])
    dW1_f = c_fake["x"].T @ dh1_pre_f
    db1_f = dh1_pre_f.sum(axis=0)

    # Backprop for gradient penalty
    x_hat = gp_cache["x_hat"]
    grad_norm = gp_cache["grad_norm"]
    c_interp = C.forward(x_hat)

    scale = (2.0 * lam * (grad_norm - 1.0) / (grad_norm + 1e-12)).reshape(-1, 1)
    dx_hat = gp_cache["dx"]
    dx_hat_dir = dx_hat / (np.sqrt((dx_hat ** 2).sum(axis=1, keepdims=True)) + 1e-12)
    gp_signal = scale * dx_hat_dir / batch_size

    dscore_gp = scale / batch_size
    dh2_gp = dscore_gp @ C.W3.T
    dW3_gp = c_interp["h2"].T @ dscore_gp
    db3_gp = dscore_gp.sum(axis=0)
    dh2_pre_gp = dh2_gp * leaky_relu_backward(c_interp["h2_pre"])
    dW2_gp = c_interp["h1"].T @ dh2_pre_gp
    db2_gp = dh2_pre_gp.sum(axis=0)
    dh1_gp = dh2_pre_gp @ C.W2.T
    dh1_pre_gp = dh1_gp * leaky_relu_backward(c_interp["h1_pre"])
    dW1_gp = c_interp["x"].T @ dh1_pre_gp
    db1_gp = dh1_pre_gp.sum(axis=0)

    # Update critic
    C.W3 -= lr * (dW3_r + dW3_f + dW3_gp)
    C.b3 -= lr * (db3_r + db3_f + db3_gp)
    C.W2 -= lr * (dW2_r + dW2_f + dW2_gp)
    C.b2 -= lr * (db2_r + db2_f + db2_gp)
    C.W1 -= lr * (dW1_r + dW1_f + dW1_gp)
    C.b1 -= lr * (db1_r + db1_f + db1_gp)

    return float(total_loss)


def train_wgan_generator_step(
    C: WGANCritic, G: WGANGenerator,
    noise: np.ndarray, lr: float = 0.0001,
) -> float:
    """One generator step for WGAN-GP.

    Minimise: -E[C(G(z))]   (maximise critic score on fake samples)
    """
    batch_size = noise.shape[0]
    g_cache = G.forward(noise)
    fake = g_cache["out"]
    c_cache = C.forward(fake)

    # TODO: Compute generator loss for WGAN (maximise critic score on fakes)
    g_loss = ____  # Hint: -c_cache["score"].mean()

    # Backprop through C (frozen) then G
    dscore = -np.ones((batch_size, 1)) / batch_size
    dh2 = dscore @ C.W3.T
    dh2_pre = dh2 * leaky_relu_backward(c_cache["h2_pre"])
    dh1 = dh2_pre @ C.W2.T
    dh1_pre = dh1 * leaky_relu_backward(c_cache["h1_pre"])
    dfake = dh1_pre @ C.W1.T

    # Through G
    dW3_g = g_cache["h2"].T @ dfake
    db3_g = dfake.sum(axis=0)
    dh2_g = dfake @ G.W3.T
    dh2_pre_g = dh2_g * relu_backward(g_cache["h2_pre"])
    dW2_g = g_cache["h1"].T @ dh2_pre_g
    db2_g = dh2_pre_g.sum(axis=0)
    dh1_g = dh2_pre_g @ G.W2.T
    dh1_pre_g = dh1_g * relu_backward(g_cache["h1_pre"])
    dW1_g = g_cache["z"].T @ dh1_pre_g
    db1_g = dh1_pre_g.sum(axis=0)

    G.W3 -= lr * dW3_g
    G.b3 -= lr * db3_g
    G.W2 -= lr * dW2_g
    G.b2 -= lr * db2_g
    G.W1 -= lr * dW1_g
    G.b1 -= lr * db1_g

    return float(g_loss)


def train_wgan(
    real_data: np.ndarray,
    n_epochs: int = 300,
    batch_size: int = 256,
    noise_dim: int = 2,
    lr: float = 0.0001,
    n_critic: int = 5,
    lam: float = 10.0,
    print_every: int = 50,
) -> tuple[WGANGenerator, WGANCritic, dict]:
    """Train WGAN with gradient penalty.

    Key difference from standard GAN: train critic n_critic times per
    generator step. The critic needs to be close to optimal for the
    Wasserstein distance estimate to be meaningful.
    """
    rng = np.random.default_rng(seed=99)
    G = WGANGenerator(noise_dim=noise_dim, hidden_dim=128)
    C = WGANCritic(hidden_dim=128)
    n = real_data.shape[0]

    history = {"c_loss": [], "g_loss": [], "wasserstein_est": []}

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        epoch_c, epoch_g = 0.0, 0.0
        n_critic_steps = 0
        n_gen_steps = 0
        batch_idx = 0

        for start in range(0, n, batch_size):
            batch = real_data[perm[start:start + batch_size]]
            if batch.shape[0] < batch_size:
                continue
            noise = rng.normal(0, 1, (batch_size, noise_dim))

            # Train critic n_critic times per generator step
            c_loss = train_wgan_critic_step(C, G, batch, noise, lr=lr, lam=lam)
            epoch_c += c_loss
            n_critic_steps += 1

            batch_idx += 1
            if batch_idx % n_critic == 0:
                g_noise = rng.normal(0, 1, (batch_size, noise_dim))
                g_loss = train_wgan_generator_step(C, G, g_noise, lr=lr)
                epoch_g += g_loss
                n_gen_steps += 1

        avg_c = epoch_c / max(n_critic_steps, 1)
        avg_g = epoch_g / max(n_gen_steps, 1)

        # Wasserstein distance estimate: E[C(real)] - E[C(fake)]
        eval_noise = rng.normal(0, 1, (256, noise_dim))
        w_est = float(C.score(real_data[:256]).mean() - C.score(G.generate(eval_noise)).mean())

        history["c_loss"].append(avg_c)
        history["g_loss"].append(avg_g)
        history["wasserstein_est"].append(w_est)

        if (epoch + 1) % print_every == 0:
            print(
                f"  Epoch {epoch + 1:4d}/{n_epochs} | "
                f"C loss: {avg_c:.4f} | G loss: {avg_g:.4f} | "
                f"W dist: {w_est:.4f}"
            )

    return G, C, history


print("\n=== TASK 4: Training WGAN-GP on Mixture of Gaussians ===")
print("Key differences from GAN: Wasserstein loss, gradient penalty, n_critic=5\n")

G_wgan, C_wgan, history_wgan = train_wgan(
    mog_data, n_epochs=300, batch_size=256, lr=0.0001,
    n_critic=5, lam=10.0, print_every=50,
)


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare training stability — GAN vs WGAN
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

print("\n=== TASK 5: Training Stability Comparison ===")

# Loss curves comparison
print("\n--- Loss Curve Analysis ---")
print(f"Basic GAN final D loss: {history_basic['d_loss'][-1]:.4f}")
print(f"Basic GAN final G loss: {history_basic['g_loss'][-1]:.4f}")
print(f"DCGAN final D loss:     {history_dcgan['d_loss'][-1]:.4f}")
print(f"DCGAN final G loss:     {history_dcgan['g_loss'][-1]:.4f}")
print(f"WGAN final C loss:      {history_wgan['c_loss'][-1]:.4f}")
print(f"WGAN final G loss:      {history_wgan['g_loss'][-1]:.4f}")

# Compute loss variance (measure of stability)
gan_d_var = np.var(history_basic["d_loss"][-100:])
gan_g_var = np.var(history_basic["g_loss"][-100:])
dcgan_d_var = np.var(history_dcgan["d_loss"][-100:])
dcgan_g_var = np.var(history_dcgan["g_loss"][-100:])
wgan_c_var = np.var(history_wgan["c_loss"][-100:])
wgan_g_var = np.var(history_wgan["g_loss"][-100:])

print(f"\n--- Loss Variance (last 100 epochs) ---")
print(f"Basic GAN: D var={gan_d_var:.6f}, G var={gan_g_var:.6f}")
print(f"DCGAN:     D var={dcgan_d_var:.6f}, G var={dcgan_g_var:.6f}")
print(f"WGAN-GP:   C var={wgan_c_var:.6f}, G var={wgan_g_var:.6f}")
print(f"\nLower variance = more stable training.")
print(f"WGAN loss correlates with sample quality (unlike GAN/DCGAN BCE loss).")

# Visualise loss curves using ModelVisualizer
gan_loss_metrics = {
    "GAN_D_loss": history_basic["d_loss"],
    "GAN_G_loss": history_basic["g_loss"],
}
fig_gan_loss = viz.training_history(gan_loss_metrics, x_label="Epoch")
fig_gan_loss.update_layout(title="Basic GAN Training Loss")
fig_gan_loss.write_html("ex5_gan_loss.html")

wgan_loss_metrics = {
    "WGAN_Critic_loss": history_wgan["c_loss"],
    "WGAN_G_loss": history_wgan["g_loss"],
    "Wasserstein_estimate": history_wgan["wasserstein_est"],
}
fig_wgan_loss = viz.training_history(wgan_loss_metrics, x_label="Epoch")
fig_wgan_loss.update_layout(title="WGAN-GP Training Loss + Wasserstein Estimate")
fig_wgan_loss.write_html("ex5_wgan_loss.html")
print("\nSaved: ex5_gan_loss.html, ex5_wgan_loss.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Detect and demonstrate mode collapse in basic GAN
# ══════════════════════════════════════════════════════════════════════
# THEORY — Mode collapse:
#   The generator finds one (or a few) modes that reliably fool the
#   discriminator, and stops exploring other modes. All generated samples
#   cluster around these few modes, ignoring the rest of the distribution.
#
#   Detection: compare number of modes covered by generated vs real data.
#   With 8 Gaussians, a collapsed GAN might only produce 1-3 clusters.

print("\n=== TASK 6: Mode Collapse Detection ===")


def count_modes_covered(
    generated: np.ndarray,
    mode_centres: np.ndarray,
    radius: float = 0.5,
) -> tuple[int, list[int]]:
    """Count how many modes are covered by generated samples.

    A mode is "covered" if at least 1% of generated samples fall within
    `radius` of the mode centre.
    """
    n_samples = generated.shape[0]
    min_samples = max(1, int(0.01 * n_samples))  # At least 1% coverage
    counts_per_mode = []

    for centre in mode_centres:
        dists = np.sqrt(((generated - centre) ** 2).sum(axis=1))
        count = int((dists < radius).sum())
        counts_per_mode.append(count)

    modes_covered = sum(1 for c in counts_per_mode if c >= min_samples)
    return modes_covered, counts_per_mode


# True mode centres for 8 Gaussians
n_modes = 8
mode_centres = np.array([
    [2.0 * np.cos(2 * np.pi * i / n_modes), 2.0 * np.sin(2 * np.pi * i / n_modes)]
    for i in range(n_modes)
])

# Generate samples from basic GAN
rng_mc = np.random.default_rng(seed=0)
eval_noise_mc = rng_mc.normal(0, 1, (2048, 2))
gen_basic = G_basic.generate(eval_noise_mc)
gen_wgan = G_wgan.generate(eval_noise_mc)

modes_basic, counts_basic = count_modes_covered(gen_basic, mode_centres, radius=0.5)
modes_wgan, counts_wgan = count_modes_covered(gen_wgan, mode_centres, radius=0.5)

print(f"\n--- Mode Coverage Analysis ---")
print(f"True data: 8 modes (Gaussians arranged in a circle)")
print(f"\nBasic GAN modes covered: {modes_basic}/{n_modes}")
print(f"  Samples per mode: {counts_basic}")
print(f"\nWGAN-GP modes covered:   {modes_wgan}/{n_modes}")
print(f"  Samples per mode: {counts_wgan}")

# Also train a deliberately unstable GAN to amplify mode collapse
print("\n--- Training Deliberately Unstable GAN (high LR, small D) ---")


class SmallDiscriminator:
    """Smaller discriminator that is easier to overwhelm — amplifies mode collapse."""

    def __init__(self, hidden_dim: int = 16, rng_seed: int = 123):
        rng = np.random.default_rng(rng_seed)
        self.W1 = he_init(2, hidden_dim, rng)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = xavier_init(hidden_dim, 1, rng)
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> dict:
        h1_pre = x @ self.W1 + self.b1
        h1 = leaky_relu(h1_pre)
        logit = h1 @ self.W2 + self.b2
        prob = sigmoid(logit)
        return {"x": x, "h1_pre": h1_pre, "h1": h1, "logit": logit, "prob": prob}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)["prob"]


def train_unstable_gan(
    real_data: np.ndarray,
    n_epochs: int = 200,
    batch_size: int = 256,
    noise_dim: int = 2,
    lr: float = 0.002,
) -> Generator:
    """Train a GAN prone to mode collapse: small D, high LR, no tricks."""
    rng = np.random.default_rng(seed=77)
    G = Generator(noise_dim=noise_dim, hidden_dim=128, rng_seed=42)
    D = SmallDiscriminator(hidden_dim=16)
    n = real_data.shape[0]
    eps = 1e-8

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = real_data[perm[start:start + batch_size]]
            if batch.shape[0] < batch_size:
                continue
            noise = rng.normal(0, 1, (batch_size, noise_dim))
            bs = batch.shape[0]

            # D step (simplified for small D)
            d_real = D.forward(batch)
            fake = G.generate(noise)
            d_fake = D.forward(fake)

            # D backprop (simplified: 1 hidden layer)
            dprob_r = -(1.0 / (d_real["prob"] + eps)) / bs
            dlogit_r = dprob_r * sigmoid_backward(d_real["prob"])
            dW2_r = d_real["h1"].T @ dlogit_r
            db2_r = dlogit_r.sum(axis=0)
            dh1_r = dlogit_r @ D.W2.T * leaky_relu_backward(d_real["h1_pre"])
            dW1_r = d_real["x"].T @ dh1_r
            db1_r = dh1_r.sum(axis=0)

            dprob_f = (1.0 / (1 - d_fake["prob"] + eps)) / bs
            dlogit_f = dprob_f * sigmoid_backward(d_fake["prob"])
            dW2_f = d_fake["h1"].T @ dlogit_f
            db2_f = dlogit_f.sum(axis=0)
            dh1_f = dlogit_f @ D.W2.T * leaky_relu_backward(d_fake["h1_pre"])
            dW1_f = d_fake["x"].T @ dh1_f
            db1_f = dh1_f.sum(axis=0)

            D.W2 -= lr * (dW2_r + dW2_f)
            D.b2 -= lr * (db2_r + db2_f)
            D.W1 -= lr * (dW1_r + dW1_f)
            D.b1 -= lr * (db1_r + db1_f)

            # G step
            g_cache = G.forward(noise)
            d_g = D.forward(g_cache["out"])
            dprob_g = -1.0 / (d_g["prob"] + eps) / bs
            dlogit_g = dprob_g * sigmoid_backward(d_g["prob"])
            dh1_g = dlogit_g @ D.W2.T * leaky_relu_backward(d_g["h1_pre"])
            dfake_g = dh1_g @ D.W1.T

            dW3 = g_cache["h2"].T @ dfake_g
            db3 = dfake_g.sum(axis=0)
            dh2 = dfake_g @ G.W3.T * relu_backward(g_cache["h2_pre"])
            dW2 = g_cache["h1"].T @ dh2
            db2 = dh2.sum(axis=0)
            dh1 = dh2 @ G.W2.T * relu_backward(g_cache["h1_pre"])
            dW1 = g_cache["z"].T @ dh1
            db1 = dh1.sum(axis=0)

            G.W3 -= lr * dW3
            G.b3 -= lr * db3
            G.W2 -= lr * dW2
            G.b2 -= lr * db2
            G.W1 -= lr * dW1
            G.b1 -= lr * db1

    return G


G_collapsed = train_unstable_gan(mog_data, n_epochs=200, lr=0.002)
gen_collapsed = G_collapsed.generate(eval_noise_mc)
modes_collapsed, counts_collapsed = count_modes_covered(gen_collapsed, mode_centres, radius=0.5)

print(f"\nUnstable GAN modes covered: {modes_collapsed}/{n_modes}")
print(f"  Samples per mode: {counts_collapsed}")
print(f"\nMode collapse is visible: generator concentrates on {modes_collapsed} mode(s)")
print(f"instead of spreading across all {n_modes} modes of the true distribution.")

# Compute spread metric: standard deviation of mode coverage
coverage_std_basic = np.std(counts_basic)
coverage_std_wgan = np.std(counts_wgan)
coverage_std_collapsed = np.std(counts_collapsed)

print(f"\n--- Coverage Uniformity (lower std = more uniform) ---")
print(f"Basic GAN:    std of mode counts = {coverage_std_basic:.1f}")
print(f"WGAN-GP:      std of mode counts = {coverage_std_wgan:.1f}")
print(f"Unstable GAN: std of mode counts = {coverage_std_collapsed:.1f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Show how WGAN fixes mode collapse
# ══════════════════════════════════════════════════════════════════════

print("\n=== TASK 7: WGAN Fixes Mode Collapse ===")


def compute_fid_2d(real: np.ndarray, generated: np.ndarray) -> float:
    """Compute Frechet Inception Distance for 2D data.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * (Sigma_r @ Sigma_g)^{1/2})

    Lower FID = better quality (closer to real distribution).
    """
    mu_r = real.mean(axis=0)
    mu_g = generated.mean(axis=0)
    sigma_r = np.cov(real, rowvar=False)
    sigma_g = np.cov(generated, rowvar=False)

    # ||mu_r - mu_g||^2
    diff = mu_r - mu_g
    mean_term = diff @ diff

    # Matrix square root of Sigma_r @ Sigma_g via eigendecomposition
    product = sigma_r @ sigma_g
    eigvals, eigvecs = np.linalg.eigh(product)
    # Clamp negative eigenvalues (numerical issues)
    eigvals = np.maximum(eigvals, 0)
    sqrt_product = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    trace_term = np.trace(sigma_r + sigma_g - 2 * sqrt_product)

    return float(mean_term + trace_term)


fid_basic = compute_fid_2d(mog_data, gen_basic)
fid_wgan = compute_fid_2d(mog_data, gen_wgan)
fid_collapsed = compute_fid_2d(mog_data, gen_collapsed)

print(f"\n--- FID Scores (lower = better) ---")
print(f"Basic GAN:    FID = {fid_basic:.4f}")
print(f"WGAN-GP:      FID = {fid_wgan:.4f}")
print(f"Unstable GAN: FID = {fid_collapsed:.4f}")

# Distribution spread analysis
print(f"\n--- Generated Distribution Statistics ---")
for name, samples in [("Basic GAN", gen_basic), ("WGAN-GP", gen_wgan), ("Unstable", gen_collapsed)]:
    print(f"\n{name}:")
    print(f"  Mean: ({samples[:, 0].mean():.3f}, {samples[:, 1].mean():.3f})")
    print(f"  Std:  ({samples[:, 0].std():.3f}, {samples[:, 1].std():.3f})")
    print(f"  Range X: [{samples[:, 0].min():.3f}, {samples[:, 0].max():.3f}]")
    print(f"  Range Y: [{samples[:, 1].min():.3f}, {samples[:, 1].max():.3f}]")

# Summary comparison table
summary_metrics = {
    "Basic GAN": {
        "Modes_Covered": float(modes_basic),
        "FID": fid_basic,
        "Coverage_Std": float(coverage_std_basic),
    },
    "WGAN-GP": {
        "Modes_Covered": float(modes_wgan),
        "FID": fid_wgan,
        "Coverage_Std": float(coverage_std_wgan),
    },
    "Unstable GAN": {
        "Modes_Covered": float(modes_collapsed),
        "FID": fid_collapsed,
        "Coverage_Std": float(coverage_std_collapsed),
    },
}
fig_compare = viz.metric_comparison(summary_metrics)
fig_compare.update_layout(title="GAN vs WGAN-GP: Mode Coverage and Quality")
fig_compare.write_html("ex5_comparison.html")
print("\nSaved: ex5_comparison.html")

print(f"\n--- Why WGAN Fixes Mode Collapse ---")
print(f"1. Wasserstein distance provides meaningful gradients everywhere,")
print(f"   unlike JS divergence which saturates when distributions don't overlap.")
print(f"2. The gradient penalty enforces a soft Lipschitz constraint on the critic,")
print(f"   preventing it from becoming too confident in any region.")
print(f"3. Training the critic multiple times (n_critic=5) per generator step")
print(f"   gives the generator a more accurate loss landscape to optimise against.")
print(f"4. The critic's score correlates with sample quality (unlike BCE loss),")
print(f"   making training progress directly observable.")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Diffusion models — conceptual overview + pseudocode
# ══════════════════════════════════════════════════════════════════════
# THEORY — Denoising Diffusion Probabilistic Models (DDPM):
#
#   Forward process (fixed): progressively add Gaussian noise to data
#     x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
#   where alpha_t is a noise schedule (e.g., linear from 1.0 to ~0.0)
#
#   Reverse process (learned): a neural network learns to predict the
#   noise epsilon at each step, then removes it:
#     x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * epsilon_theta(x_t, t))
#
#   Key insight: instead of learning to generate data in one shot (like GANs),
#   diffusion models learn to DENOISE — removing noise one small step at a time.
#   This is much easier to learn and produces higher-quality, more diverse samples.

print("\n=== TASK 8: Diffusion Models — Conceptual Overview ===")

print("""
--- DDPM (Denoising Diffusion Probabilistic Models) ---

Diffusion models work in two phases:

1. FORWARD PROCESS (fixed, no learning):
   Gradually add Gaussian noise over T steps until data becomes pure noise.
   x_0 (clean data) → x_1 → x_2 → ... → x_T (pure noise)
   At each step: x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise

2. REVERSE PROCESS (learned):
   A neural network learns to reverse the noise, one step at a time.
   x_T (noise) → x_{T-1} → ... → x_1 → x_0 (generated data)
   The network predicts the noise that was added, then subtracts it.

Key advantages over GANs:
  - No adversarial training (no mode collapse, no training instability)
  - Better sample diversity (covers all modes naturally)
  - Meaningful training objective (simple MSE on noise prediction)
  - Trade-off: much slower generation (T forward passes vs 1 for GAN)
""")

# Pseudocode implementation
print("--- DDPM Pseudocode (10-line core) ---")
print("""
# Forward process: add noise
def forward_diffusion(x_0, t, noise_schedule):
    alpha_bar_t = cumulative_product(noise_schedule[:t])
    noise = sample_gaussian(shape=x_0.shape)
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    return x_t, noise

# Training: learn to predict noise
def train_step(model, x_0, noise_schedule):
    t = random_int(1, T)                            # Random timestep
    x_t, true_noise = forward_diffusion(x_0, t, noise_schedule)
    predicted_noise = model(x_t, t)                  # Neural net predicts noise
    loss = mean_squared_error(predicted_noise, true_noise)
    update_weights(model, loss)

# Generation: iteratively denoise
def generate(model, noise_schedule, T):
    x_T = sample_gaussian(shape=data_shape)          # Start from pure noise
    for t in reversed(range(1, T + 1)):
        predicted_noise = model(x_T, t)
        x_T = denoise_step(x_T, predicted_noise, noise_schedule, t)
    return x_T                                        # Clean generated sample
""")

# Demonstrate the forward diffusion process on our 2D data
print("--- Forward Diffusion Demo on 2D Data ---")

rng_diff = np.random.default_rng(seed=42)
T = 50  # Number of diffusion steps
beta_start, beta_end = 0.0001, 0.02
betas = np.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

# TODO: Implement one step of the forward diffusion process
#   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
#   Show how the signal is corrupted at t=0, 10, 25, 49

# Show a single point being diffused
x_0 = mog_data[0:1]  # One 2D point
print(f"\nOriginal point: ({x_0[0, 0]:.3f}, {x_0[0, 1]:.3f})")

for t in [0, 10, 25, 49]:
    noise = rng_diff.normal(0, 1, x_0.shape)
    # TODO: Apply forward diffusion at timestep t
    x_t = ____  # Hint: np.sqrt(alpha_bars[t]) * x_0 + np.sqrt(1 - alpha_bars[t]) * noise
    signal_ratio = alpha_bars[t]
    print(f"  t={t:3d}: ({x_t[0, 0]:+.3f}, {x_t[0, 1]:+.3f})  "
          f"signal={signal_ratio:.3f}  noise={1 - signal_ratio:.3f}")

print(f"\nAt t={T-1}, the point is almost pure noise — signal ratio = {alpha_bars[-1]:.4f}")

# Comparison table: GAN vs VAE vs Diffusion
print("""
--- When to Use Each Generative Model ---

| Criterion          | GAN               | VAE                | Diffusion          |
|--------------------|--------------------|--------------------|---------------------|
| Sample quality     | High (sharp)       | Medium (blurry)    | Highest             |
| Training stability | Low (mode collapse)| High (stable ELBO) | High (simple MSE)   |
| Generation speed   | Fast (1 pass)      | Fast (1 pass)      | Slow (T passes)     |
| Mode coverage      | Poor (collapse)    | Good               | Excellent           |
| Latent space       | Unstructured       | Structured (VAE)   | None (implicit)     |
| Best for           | Real-time gen      | Representation     | Quality-first gen   |
| Key formula        | minimax game       | ELBO               | noise prediction    |
""")

print("--- Summary ---")
print(f"GANs: powerful but fragile. WGAN-GP addresses the main failure mode (mode collapse).")
print(f"Diffusion: slower but more stable and diverse. Dominant for image generation (2023+).")
print(f"VAE (from lesson 5.1): structured latent space, good for representation learning.")
print(f"Choice depends on: quality requirements, generation speed, training budget.")
