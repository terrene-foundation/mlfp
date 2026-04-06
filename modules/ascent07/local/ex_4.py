# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 4: Loss Functions and Weight Initialization
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare MSE vs CrossEntropy loss on classification, and
#   Xavier vs He initialization on deep networks.
#
# TASKS:
#   1. Implement CrossEntropy from scratch
#   2. Compare MSE vs CE on same classification task
#   3. Implement Xavier and He initialization
#   4. Train 10-layer network with each init strategy
#   5. Visualize gradient flow per layer
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import random

import polars as pl

from kailash_ml import DataExplorer, ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

random.seed(42)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement CrossEntropy from scratch
# ══════════════════════════════════════════════════════════════════════


def softmax(z: list[float]) -> list[float]:
    """Softmax: exp(z_i) / sum(exp(z_j))."""
    max_z = max(z)
    exp_z = [math.exp(zi - max_z) for zi in z]
    total = sum(exp_z)
    return [e / total for e in exp_z]


def cross_entropy_loss(y_true: list[float], y_pred: list[float]) -> float:
    """Cross-entropy: -sum(y_true * log(y_pred)).

    y_true: one-hot encoded label
    y_pred: softmax probabilities
    """
    # TODO: Implement cross-entropy loss.
    # Hint: eps = 1e-8; -sum(yt * math.log(yp + eps) for yt, yp in zip(y_true, y_pred))
    eps = 1e-8
    return ____


def mse_loss(y_true: list[float], y_pred: list[float]) -> float:
    """MSE: (1/n) * sum((y_true - y_pred)^2)."""
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


# Demonstrate on a single example
y_true = [0.0, 0.0, 1.0, 0.0, 0.0]  # class 2
y_confident = [0.01, 0.01, 0.95, 0.01, 0.02]
y_uncertain = [0.15, 0.20, 0.30, 0.20, 0.15]
y_wrong = [0.60, 0.20, 0.05, 0.10, 0.05]

print("=== Loss Function Comparison ===")
print(f"True label: class 2 (one-hot: {y_true})")
print(f"\n{'Prediction':>12} | {'CE Loss':>8} | {'MSE Loss':>8}")
print("-" * 38)
for name, pred in [
    ("confident", y_confident),
    ("uncertain", y_uncertain),
    ("wrong", y_wrong),
]:
    ce = cross_entropy_loss(y_true, pred)
    mse = mse_loss(y_true, pred)
    print(f"{name:>12} | {ce:8.4f} | {mse:8.4f}")

print(f"\nKey: CE penalizes wrong predictions MUCH more harshly.")
print(
    f"  CE(wrong) / CE(confident) = {cross_entropy_loss(y_true, y_wrong) / cross_entropy_loss(y_true, y_confident):.1f}x"
)
print(
    f"  MSE(wrong) / MSE(confident) = {mse_loss(y_true, y_wrong) / mse_loss(y_true, y_confident):.1f}x"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Compare MSE vs CE on same classification task
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
df = loader.load("ascent07", "mnist_sample.parquet")

explorer = DataExplorer()
summary = explorer.analyze(df)

feature_cols = [c for c in df.columns if c != "label"]
X = [
    [pixel / 255.0 for pixel in row]
    for row in df.select(feature_cols).to_numpy().tolist()
]
y_labels = df["label"].to_list()
n_classes = 10
Y = [[1.0 if j == label else 0.0 for j in range(n_classes)] for label in y_labels]

n_features = len(X[0])
train_size = min(200, len(X))

print(f"\n=== MNIST Classification ===")
print(f"Features: {n_features}, Train samples: {train_size}, Classes: {n_classes}")


def train_with_loss(loss_type: str, epochs: int = 15) -> list[float]:
    """Train a simple network with specified loss function."""
    hidden = 32
    scale = math.sqrt(2.0 / (n_features + hidden))
    W1 = [[random.gauss(0, scale) for _ in range(hidden)] for _ in range(n_features)]
    b1 = [0.0] * hidden
    scale2 = math.sqrt(2.0 / (hidden + n_classes))
    W2 = [[random.gauss(0, scale2) for _ in range(n_classes)] for _ in range(hidden)]
    b2 = [0.0] * n_classes

    lr = 0.01
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for idx in range(train_size):
            x = X[idx]
            y = Y[idx]

            # Forward: hidden layer with ReLU
            h = [
                max(0.0, sum(x[i] * W1[i][j] for i in range(n_features)) + b1[j])
                for j in range(hidden)
            ]

            # Forward: output with softmax
            z_out = [
                sum(h[j] * W2[j][k] for j in range(hidden)) + b2[k]
                for k in range(n_classes)
            ]
            out = softmax(z_out)

            # Loss
            if loss_type == "ce":
                loss = cross_entropy_loss(y, out)
            else:
                loss = mse_loss(y, out)
            epoch_loss += loss

        losses.append(epoch_loss / train_size)

    return losses


print(f"\nTraining with MSE loss...")
mse_losses = train_with_loss("mse")
print(f"Training with CrossEntropy loss...")
ce_losses = train_with_loss("ce")

viz = ModelVisualizer()
loss_comparison = pl.DataFrame(
    {
        "epoch": list(range(len(mse_losses))),
        "mse_loss": mse_losses,
        "ce_loss": ce_losses,
    }
)
fig = viz.plot_training_curves(loss_comparison)

print(f"\n  MSE final loss:  {mse_losses[-1]:.4f}")
print(f"  CE final loss:   {ce_losses[-1]:.4f}")
print(f"  CE converges faster because its gradient is (y_pred - y_true),")
print(f"  while MSE gradient includes sigmoid derivative (can saturate).")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement Xavier and He initialization
# ══════════════════════════════════════════════════════════════════════


def xavier_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """Xavier/Glorot: N(0, sqrt(2 / (fan_in + fan_out))).

    Best for: sigmoid, tanh, GELU activations.
    """
    # TODO: Compute Xavier standard deviation and generate weight matrix.
    # Hint: std = math.sqrt(2.0 / (fan_in + fan_out))
    # Hint: return [[random.gauss(0, std) for _ in range(fan_out)] for _ in range(fan_in)]
    std = ____
    return ____


def he_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """He/Kaiming: N(0, sqrt(2 / fan_in)).

    Best for: ReLU activations (accounts for half the neurons being dead).
    """
    # TODO: Compute He standard deviation and generate weight matrix.
    # Hint: std = math.sqrt(2.0 / fan_in)
    # Hint: return [[random.gauss(0, std) for _ in range(fan_out)] for _ in range(fan_in)]
    std = ____
    return ____


def zero_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """Zero initialization (bad — for demonstration only)."""
    return [[0.0 for _ in range(fan_out)] for _ in range(fan_in)]


# Show the variance of initialized weights
print(f"\n=== Initialization Comparison ===")
for name, init_fn in [("Xavier", xavier_init), ("He", he_init), ("Zero", zero_init)]:
    W = init_fn(784, 128)
    flat = [w for row in W for w in row]
    mean_w = sum(flat) / len(flat)
    var_w = sum((w - mean_w) ** 2 for w in flat) / len(flat)
    print(f"  {name:>6}: mean={mean_w:.6f}, var={var_w:.6f}")

print(f"\n  Xavier var target: 2/(784+128) = {2/(784+128):.6f}")
print(f"  He var target:     2/784       = {2/784:.6f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Train 10-layer network with each init strategy
# ══════════════════════════════════════════════════════════════════════


def build_deep_network(n_layers: int, init_fn) -> list[dict]:
    """Build a deep network with specified initialization."""
    layers = []
    dims = [n_features] + [64] * n_layers + [n_classes]

    for i in range(len(dims) - 1):
        W = init_fn(dims[i], dims[i + 1])
        b = [0.0] * dims[i + 1]
        layers.append({"W": W, "b": b, "in": dims[i], "out": dims[i + 1]})

    return layers


def forward_deep(x: list[float], layers: list[dict]) -> list[list[float]]:
    """Forward pass through deep network, returning activations per layer."""
    activations = [x]
    current = x

    for i, layer in enumerate(layers):
        z = [
            sum(current[j] * layer["W"][j][k] for j in range(layer["in"]))
            + layer["b"][k]
            for k in range(layer["out"])
        ]

        if i < len(layers) - 1:
            current = [max(0.0, zi) for zi in z]  # ReLU
        else:
            current = softmax(z)  # Output

        activations.append(current)

    return activations


print(f"\n=== Deep Network (10 layers) ===")
n_deep_layers = 10

for name, init_fn in [("Xavier", xavier_init), ("He", he_init)]:
    # TODO: Build a deep network with the given init function.
    # Hint: build_deep_network(n_deep_layers, init_fn)
    layers = ____

    # Forward pass one sample
    activations = forward_deep(X[0], layers)

    # Check activation magnitudes per layer
    print(f"\n  {name} init — activation magnitudes:")
    for i, act in enumerate(activations[1:], 1):
        mean_act = sum(abs(a) for a in act) / len(act)
        zero_frac = sum(1 for a in act if abs(a) < 1e-6) / len(act)
        print(f"    Layer {i:2d}: mean|act|={mean_act:.6f}, dead={zero_frac:.1%}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Visualize gradient flow per layer
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Gradient Flow Analysis ===")
print(f"For a 10-layer network with ReLU activation:")
print(f"")
print(f"Xavier init: var(act) decreases per layer (designed for tanh)")
print(f"  -> Gradients shrink as they backpropagate (vanishing)")
print(f"He init: var(act) stays ~constant (designed for ReLU)")
print(f"  -> Gradients maintain magnitude (stable training)")
print(f"")
print(f"Rule of thumb:")
print(f"  Sigmoid/Tanh -> Xavier (accounts for both directions)")
print(f"  ReLU/variants -> He (accounts for dead half)")

# Compute theoretical gradient scaling
print(f"\n  Xavier gradient scaling over 10 layers:")
for layer_idx in range(1, 11):
    scale = (2.0 / (64 + 64)) ** (layer_idx / 2)
    print(f"    Layer {layer_idx:2d}: ~{scale:.6f}")

print(f"\n  He gradient scaling over 10 layers:")
for layer_idx in range(1, 11):
    scale = (2.0 / 64) ** (layer_idx / 2) * (0.5 ** (layer_idx / 2))
    print(f"    Layer {layer_idx:2d}: ~{scale:.6f}")

print("\n✓ Exercise 4 complete — loss functions and initialization strategies compared")
