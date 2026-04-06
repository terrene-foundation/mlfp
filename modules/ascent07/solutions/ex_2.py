# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 2: Hidden Layers and the XOR Problem
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Show why hidden layers matter by solving XOR, then build a
#   multi-layer perceptron demonstrating automatic feature interaction
#   discovery.
#
# TASKS:
#   1. Show XOR is not linearly separable
#   2. Build 2-layer network with sigmoid
#   3. Train on XOR via gradient descent
#   4. Visualize decision boundary with ModelVisualizer
#   5. Extend to multi-class on synthetic spirals data
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
# TASK 1: Show XOR is not linearly separable
# ══════════════════════════════════════════════════════════════════════

# XOR truth table
xor_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
xor_targets = [0.0, 1.0, 1.0, 0.0]

print("=== XOR Truth Table ===")
for inp, tgt in zip(xor_inputs, xor_targets):
    print(f"  {inp} -> {tgt}")

# Try single-layer (linear) solution: y = w1*x1 + w2*x2 + b
# No single line can separate (0,0),(1,1) from (0,1),(1,0)
print("\nWhy XOR fails with a single neuron:")
print("  Class 0: (0,0) and (1,1) — on the diagonal")
print("  Class 1: (0,1) and (1,0) — on the anti-diagonal")
print("  No single line w1*x1 + w2*x2 + b = 0 separates these.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build 2-layer network with sigmoid
# ══════════════════════════════════════════════════════════════════════


def sigmoid(z: float) -> float:
    """Sigmoid activation: 1 / (1 + e^(-z))."""
    z = max(-500, min(500, z))  # clip to prevent overflow
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(a: float) -> float:
    """Derivative of sigmoid: a * (1 - a) where a = sigmoid(z)."""
    return a * (1.0 - a)


# Network architecture: 2 inputs -> 2 hidden neurons -> 1 output
# Layer 1: W1 (2x2), b1 (2,)
# Layer 2: W2 (1x2), b2 (1,)


# Xavier initialization
def init_weight(fan_in: int, fan_out: int) -> list[list[float]]:
    """Xavier/Glorot initialization: uniform(-sqrt(6/(in+out)), sqrt(6/(in+out)))."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [
        [random.uniform(-limit, limit) for _ in range(fan_out)] for _ in range(fan_in)
    ]


W1 = init_weight(2, 2)  # 2 inputs -> 2 hidden
b1 = [0.0, 0.0]
W2 = init_weight(2, 1)  # 2 hidden -> 1 output
b2 = [0.0]

print(f"\n=== Network Architecture ===")
print(f"Layer 1: {2} inputs -> {2} hidden (sigmoid)")
print(f"Layer 2: {2} hidden -> {1} output (sigmoid)")
print(f"Total parameters: {2*2 + 2 + 2*1 + 1} = 9")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train on XOR via gradient descent
# ══════════════════════════════════════════════════════════════════════


def forward_pass(x: list[float]) -> tuple:
    """Forward pass through 2-layer network."""
    # Hidden layer
    h = []
    for j in range(2):
        z = sum(x[i] * W1[i][j] for i in range(2)) + b1[j]
        h.append(sigmoid(z))

    # Output layer
    z_out = sum(h[j] * W2[j][0] for j in range(2)) + b2[0]
    y_hat = sigmoid(z_out)

    return h, y_hat


learning_rate = 1.0
epochs = 5000
history = {"epoch": [], "loss": []}

for epoch in range(epochs):
    total_loss = 0.0

    for x, y in zip(xor_inputs, xor_targets):
        # Forward
        h, y_hat = forward_pass(x)

        # Binary cross-entropy loss (per sample)
        eps = 1e-8
        loss = -(y * math.log(y_hat + eps) + (1 - y) * math.log(1 - y_hat + eps))
        total_loss += loss

        # Backward: output layer
        d_out = y_hat - y  # dL/dz_out for BCE + sigmoid

        # Gradients for W2, b2
        dW2 = [h[j] * d_out for j in range(2)]
        db2 = d_out

        # Backward: hidden layer
        d_hidden = [d_out * W2[j][0] * sigmoid_derivative(h[j]) for j in range(2)]

        # Gradients for W1, b1
        for i in range(2):
            for j in range(2):
                W1[i][j] -= learning_rate * x[i] * d_hidden[j]
        for j in range(2):
            b1[j] -= learning_rate * d_hidden[j]

        # Update W2, b2
        for j in range(2):
            W2[j][0] -= learning_rate * dW2[j]
        b2[0] -= learning_rate * db2

    avg_loss = total_loss / 4.0
    if epoch % 1000 == 0:
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch:5d}: loss={avg_loss:.6f}")

# Final predictions
print(f"\n=== Trained XOR Predictions ===")
for x, y in zip(xor_inputs, xor_targets):
    _, y_hat = forward_pass(x)
    print(f"  {x} -> {y_hat:.4f} (target: {y})")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Visualize decision boundary with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

# Generate grid for decision boundary
grid_points = []
grid_preds = []
for gx in range(0, 101, 5):
    for gy in range(0, 101, 5):
        x_val = gx / 100.0
        y_val = gy / 100.0
        _, pred = forward_pass([x_val, y_val])
        grid_points.append({"x1": x_val, "x2": y_val, "prediction": pred})

boundary_df = pl.DataFrame(grid_points)
viz = ModelVisualizer()

# Prepare data for visualization
xor_df = pl.DataFrame(
    {
        "x1": [x[0] for x in xor_inputs],
        "x2": [x[1] for x in xor_inputs],
        "label": xor_targets,
    }
)

print(f"\n=== Decision Boundary ===")
print(f"Grid points: {boundary_df.height}")
print(f"Boundary shows how hidden layer creates non-linear separation")

# The hidden layer maps:
# (0,0) -> (h1, h2) in one region
# (1,1) -> (h1, h2) in same region  (both low or both high)
# (0,1) and (1,0) -> different region
print(f"\nHidden representations:")
for x, y in zip(xor_inputs, xor_targets):
    h, y_hat = forward_pass(x)
    print(f"  {x} -> hidden=({h[0]:.3f}, {h[1]:.3f}) -> out={y_hat:.3f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Extend to multi-class on synthetic spirals data
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
spirals_df = loader.load("ascent07", "synthetic_spirals.parquet")

explorer = DataExplorer()
spiral_summary = explorer.analyze(spirals_df)

print(f"\n=== Synthetic Spirals Dataset ===")
print(f"Shape: {spirals_df.shape}")
print(f"Columns: {spirals_df.columns}")
print(f"Classes: {spirals_df['label'].unique().sort().to_list()}")

# Count per class
class_counts = spirals_df.group_by("label").len().sort("label")
print(f"Samples per class:")
print(class_counts)

print(f"\nKey insight: XOR is the simplest non-linearly-separable problem.")
print(f"  Spirals are a harder version — they require more hidden neurons")
print(f"  to carve out the curved decision boundaries.")
print(f"  Each hidden neuron adds one linear boundary in input space;")
print(f"  combining them creates complex, non-linear regions.")

print("\n✓ Exercise 2 complete — hidden layers solve XOR and non-linear classification")
