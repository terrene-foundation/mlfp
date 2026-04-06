# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 3: Activation Functions and Layer Design
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare activation functions (sigmoid, ReLU, GELU) on
#   identical architectures and understand how activation choice affects
#   gradient flow and training.
#
# TASKS:
#   1. Implement sigmoid, ReLU, GELU from formulas
#   2. Plot activation functions and their derivatives
#   3. Build identical networks with different activations
#   4. Train on same dataset, compare convergence with ModelVisualizer
#   5. Analyze gradient magnitudes per layer
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
# TASK 1: Implement sigmoid, ReLU, GELU from formulas
# ══════════════════════════════════════════════════════════════════════


def sigmoid(z: float) -> float:
    """Sigmoid: sigma(z) = 1 / (1 + e^(-z))."""
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_deriv(z: float) -> float:
    """Sigmoid derivative: sigma'(z) = sigma(z) * (1 - sigma(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z: float) -> float:
    """ReLU: max(0, z)."""
    # TODO: Implement the ReLU activation function.
    # Hint: return max(0.0, z)
    return ____


def relu_deriv(z: float) -> float:
    """ReLU derivative: 1 if z > 0, else 0."""
    # TODO: Implement the ReLU derivative.
    # Hint: return 1.0 if z > 0 else 0.0
    return ____


def gelu(z: float) -> float:
    """GELU: z * Phi(z) where Phi is the standard normal CDF.

    Approximation: 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
    """
    # TODO: Implement the GELU approximation.
    # Hint: 0.5 * z * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z**3)))
    return ____


def gelu_deriv(z: float) -> float:
    """GELU derivative (numerical approximation)."""
    h = 1e-5
    return (gelu(z + h) - gelu(z - h)) / (2 * h)


print("=== Activation Functions ===")
test_values = [-3.0, -1.0, 0.0, 1.0, 3.0]
print(f"{'z':>6} | {'sigmoid':>8} | {'ReLU':>8} | {'GELU':>8}")
print("-" * 42)
for z in test_values:
    print(f"{z:6.1f} | {sigmoid(z):8.4f} | {relu(z):8.4f} | {gelu(z):8.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Plot activation functions and their derivatives
# ══════════════════════════════════════════════════════════════════════

z_range = [i * 0.1 for i in range(-50, 51)]

plot_data = pl.DataFrame(
    {
        "z": z_range,
        "sigmoid": [sigmoid(z) for z in z_range],
        "sigmoid_deriv": [sigmoid_deriv(z) for z in z_range],
        "relu": [relu(z) for z in z_range],
        "relu_deriv": [relu_deriv(z) for z in z_range],
        "gelu": [gelu(z) for z in z_range],
        "gelu_deriv": [gelu_deriv(z) for z in z_range],
    }
)

viz = ModelVisualizer()
fig = viz.plot_training_curves(plot_data)

print(f"\n=== Derivatives at z=0 ===")
print(f"sigmoid'(0) = {sigmoid_deriv(0.0):.4f} (max value = 0.25)")
print(f"ReLU'(0)    = {relu_deriv(0.0):.4f} (discontinuous at 0)")
print(f"GELU'(0)    = {gelu_deriv(0.0):.4f} (smooth, ~0.5)")
print(f"\nSigmoid saturates: sigma'(-5) = {sigmoid_deriv(-5.0):.6f} (nearly zero!)")
print(f"ReLU is dead for z<0: relu'(-5) = {relu_deriv(-5.0):.1f}")
print(f"GELU is smooth everywhere: gelu'(-1) = {gelu_deriv(-1.0):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Build identical networks with different activations
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
df = loader.load("ascent07", "mnist_sample.parquet")

explorer = DataExplorer()
summary = explorer.analyze(df)

print(f"\n=== MNIST Sample ===")
print(f"Shape: {df.shape}")

# Extract features and labels
feature_cols = [c for c in df.columns if c != "label"]
X = df.select(feature_cols).to_numpy().tolist()
y_labels = df["label"].to_list()

# Normalize features to [0, 1]
X = [[pixel / 255.0 for pixel in row] for row in X]

# One-hot encode labels (10 classes)
n_classes = 10
Y = [[1.0 if j == label else 0.0 for j in range(n_classes)] for label in y_labels]

n_features = len(X[0])
n_samples = len(X)
print(f"Features: {n_features}, Samples: {n_samples}, Classes: {n_classes}")


class SimpleNetwork:
    """3-layer network with configurable activation."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, activation: str
    ):
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier init for sigmoid/GELU, He init for ReLU
        if activation == "relu":
            scale1 = math.sqrt(2.0 / input_dim)
            scale2 = math.sqrt(2.0 / hidden_dim)
        else:
            scale1 = math.sqrt(2.0 / (input_dim + hidden_dim))
            scale2 = math.sqrt(2.0 / (hidden_dim + output_dim))

        self.W1 = [
            [random.gauss(0, scale1) for _ in range(hidden_dim)]
            for _ in range(input_dim)
        ]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [
            [random.gauss(0, scale2) for _ in range(output_dim)]
            for _ in range(hidden_dim)
        ]
        self.b2 = [0.0] * output_dim
        self.grad_magnitudes = []

    def activate(self, z: float) -> float:
        if self.activation == "sigmoid":
            return sigmoid(z)
        elif self.activation == "relu":
            return relu(z)
        else:
            return gelu(z)

    def activate_deriv(self, z: float) -> float:
        if self.activation == "sigmoid":
            return sigmoid_deriv(z)
        elif self.activation == "relu":
            return relu_deriv(z)
        else:
            return gelu_deriv(z)

    def forward(self, x: list[float]) -> tuple:
        # Hidden layer
        z1 = [
            sum(x[i] * self.W1[i][j] for i in range(len(x))) + self.b1[j]
            for j in range(self.hidden_dim)
        ]
        h1 = [self.activate(z) for z in z1]

        # Output layer (softmax)
        z2 = [
            sum(h1[j] * self.W2[j][k] for j in range(self.hidden_dim)) + self.b2[k]
            for k in range(self.output_dim)
        ]
        # Softmax
        max_z = max(z2)
        exp_z = [math.exp(z - max_z) for z in z2]
        sum_exp = sum(exp_z)
        out = [e / sum_exp for e in exp_z]

        return z1, h1, z2, out


print(f"\n=== Networks Created ===")
for act_name in ["sigmoid", "relu", "gelu"]:
    # TODO: Create a SimpleNetwork with n_features inputs, 32 hidden, n_classes outputs.
    # Hint: SimpleNetwork(n_features, 32, n_classes, act_name)
    net = ____
    print(f"  {act_name}: {n_features}->32->{n_classes}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Train on same dataset, compare convergence
# ══════════════════════════════════════════════════════════════════════

results = {}
train_size = min(200, n_samples)  # Use subset for speed

for act_name in ["sigmoid", "relu", "gelu"]:
    net = SimpleNetwork(n_features, 32, n_classes, act_name)
    lr = 0.01
    losses = []

    for epoch in range(20):
        epoch_loss = 0.0
        for idx in range(train_size):
            x = X[idx]
            y = Y[idx]
            # TODO: Run the forward pass through the network.
            # Hint: net.forward(x)
            z1, h1, z2, out = ____

            # Cross-entropy loss
            eps = 1e-8
            loss = -sum(y[k] * math.log(out[k] + eps) for k in range(n_classes))
            epoch_loss += loss

        avg_loss = epoch_loss / train_size
        losses.append(avg_loss)

        if epoch % 5 == 0:
            print(f"  [{act_name:>7}] Epoch {epoch:3d}: loss={avg_loss:.4f}")

    results[act_name] = losses

# Build comparison dataframe
comparison_df = pl.DataFrame(
    {
        "epoch": list(range(20)),
        "sigmoid_loss": results["sigmoid"],
        "relu_loss": results["relu"],
        "gelu_loss": results["gelu"],
    }
)

fig = viz.plot_training_curves(comparison_df)
print(f"\nConvergence comparison plotted.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Analyze gradient magnitudes per layer
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Gradient Analysis ===")
print(f"Sigmoid: max gradient = 0.25 at z=0")
print(f"  -> In a 10-layer network: 0.25^10 = {0.25**10:.2e} (vanishing!)")
print(f"ReLU: gradient = 1 for z>0, 0 for z<0")
print(f"  -> No vanishing, but 'dead neurons' when z<0 permanently")
print(f"GELU: smooth gradient, non-zero for z<0")
print(f"  -> Best of both worlds: no vanishing, no dead neurons")

# Compute actual gradient stats for each activation over random inputs
for act_name in ["sigmoid", "relu", "gelu"]:
    grads = []
    for _ in range(1000):
        z = random.gauss(0, 1)
        if act_name == "sigmoid":
            grads.append(sigmoid_deriv(z))
        elif act_name == "relu":
            grads.append(relu_deriv(z))
        else:
            grads.append(gelu_deriv(z))
    mean_grad = sum(grads) / len(grads)
    zero_grads = sum(1 for g in grads if abs(g) < 1e-6) / len(grads)
    print(f"\n  {act_name}: mean|grad|={mean_grad:.4f}, zero_fraction={zero_grads:.1%}")

print("\n✓ Exercise 3 complete — activation functions compared across architectures")
