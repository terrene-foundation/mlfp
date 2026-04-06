# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 5: Backpropagation from Scratch
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement the backpropagation algorithm step by step —
#   chain rule, gradient computation, gradient checking — then diagnose
#   vanishing gradients.
#
# TASKS:
#   1. Implement forward pass for 3-layer network
#   2. Derive and implement backward pass (chain rule)
#   3. Verify with numerical gradient checking
#   4. Demonstrate vanishing gradients with sigmoid
#   5. Fix with ReLU + proper initialization
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
# TASK 1: Implement forward pass for 3-layer network
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
df = loader.load("ascent07", "mnist_sample.parquet")

feature_cols = [c for c in df.columns if c != "label"]
X = [
    [pixel / 255.0 for pixel in row]
    for row in df.select(feature_cols).to_numpy().tolist()
]
y_labels = df["label"].to_list()
n_classes = 10
Y = [[1.0 if j == label else 0.0 for j in range(n_classes)] for label in y_labels]

n_features = len(X[0])
print(f"=== MNIST Data Loaded ===")
print(f"Features: {n_features}, Samples: {len(X)}, Classes: {n_classes}")

# Architecture: input(784) -> hidden1(64) -> hidden2(32) -> output(10)
dims = [n_features, 64, 32, n_classes]


def sigmoid(z: float) -> float:
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


def softmax(z: list[float]) -> list[float]:
    max_z = max(z)
    exp_z = [math.exp(zi - max_z) for zi in z]
    total = sum(exp_z)
    return [e / total for e in exp_z]


# Initialize weights with Xavier
def init_params(dims: list[int]) -> tuple[list, list]:
    weights = []
    biases = []
    for i in range(len(dims) - 1):
        std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
        W = [[random.gauss(0, std) for _ in range(dims[i + 1])] for _ in range(dims[i])]
        b = [0.0] * dims[i + 1]
        weights.append(W)
        biases.append(b)
    return weights, biases


weights, biases = init_params(dims)

print(f"\n=== Network Architecture ===")
for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
    act = "softmax" if i == len(dims) - 2 else "sigmoid"
    print(f"  Layer {i+1}: {d_in} -> {d_out} ({act})")
total_params = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
print(f"  Total parameters: {total_params}")


def forward(x: list[float], weights: list, biases: list) -> dict:
    """Full forward pass, caching all intermediates for backprop."""
    cache = {"activations": [x], "pre_activations": []}
    current = x

    for layer_idx in range(len(weights)):
        W = weights[layer_idx]
        b = biases[layer_idx]
        d_in = len(current)
        d_out = len(b)

        # TODO: Compute the linear transformation z = Wx + b.
        # Hint: [sum(current[j] * W[j][k] for j in range(d_in)) + b[k] for k in range(d_out)]
        z = ____
        cache["pre_activations"].append(z)

        # Activation
        if layer_idx < len(weights) - 1:
            current = [sigmoid(zi) for zi in z]
        else:
            current = softmax(z)

        cache["activations"].append(current)

    return cache


# Test forward pass
cache = forward(X[0], weights, biases)
output = cache["activations"][-1]
predicted_class = output.index(max(output))
print(f"\nForward pass test:")
print(f"  Output shape: {len(output)}")
print(f"  Predicted class: {predicted_class}")
print(f"  Max probability: {max(output):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Derive and implement backward pass (chain rule)
# ══════════════════════════════════════════════════════════════════════


def backward(cache: dict, y_true: list[float], weights: list) -> tuple[list, list]:
    """Backpropagation via chain rule.

    For each layer l (from output to input):
      dL/dW_l = a_{l-1}^T @ delta_l
      dL/db_l = delta_l
      delta_{l-1} = (W_l^T @ delta_l) * sigma'(z_{l-1})
    """
    activations = cache["activations"]
    pre_activations = cache["pre_activations"]
    n_layers = len(weights)

    dW_list = []
    db_list = []

    # Output layer: delta = y_pred - y_true (for softmax + CE)
    # TODO: Compute the output delta (error signal).
    # Hint: [activations[-1][k] - y_true[k] for k in range(len(y_true))]
    delta = ____

    for layer_idx in range(n_layers - 1, -1, -1):
        a_prev = activations[layer_idx]  # activation from previous layer
        d_in = len(a_prev)
        d_out = len(delta)

        # Gradient for weights: dW = a_prev^T @ delta
        dW = [[a_prev[j] * delta[k] for k in range(d_out)] for j in range(d_in)]
        db = list(delta)

        dW_list.insert(0, dW)
        db_list.insert(0, db)

        # Propagate delta to previous layer (if not the first layer)
        if layer_idx > 0:
            W = weights[layer_idx]
            # delta_prev = W^T @ delta * sigmoid'(z)
            z_prev = pre_activations[layer_idx - 1]
            delta_new = []
            for j in range(d_in):
                # TODO: Compute the backpropagated gradient for hidden neuron j.
                # Hint: grad_sum = sum(W[j][k] * delta[k] for k in range(d_out))
                # Hint: sig = sigmoid(z_prev[j]); sig_deriv = sig * (1.0 - sig)
                # Hint: delta_new.append(grad_sum * sig_deriv)
                grad_sum = ____
                sig = sigmoid(z_prev[j])
                sig_deriv = sig * (1.0 - sig)
                delta_new.append(grad_sum * sig_deriv)
            delta = delta_new

    return dW_list, db_list


# Test backward pass
cache = forward(X[0], weights, biases)
dW_list, db_list = backward(cache, Y[0], weights)

print(f"\n=== Backward Pass ===")
for i, (dW, db) in enumerate(zip(dW_list, db_list)):
    flat_dW = [abs(g) for row in dW for g in row]
    mean_grad = sum(flat_dW) / len(flat_dW)
    print(
        f"  Layer {i+1}: |dW| mean={mean_grad:.8f}, |db| mean={sum(abs(g) for g in db)/len(db):.8f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Verify with numerical gradient checking
# ══════════════════════════════════════════════════════════════════════


def compute_loss(x: list[float], y: list[float], weights: list, biases: list) -> float:
    """Compute cross-entropy loss for a single sample."""
    cache = forward(x, weights, biases)
    output = cache["activations"][-1]
    eps = 1e-8
    return -sum(y[k] * math.log(output[k] + eps) for k in range(len(y)))


def numerical_gradient_check(
    x: list[float], y: list[float], weights: list, biases: list, epsilon: float = 1e-5
) -> float:
    """Compare analytical vs numerical gradients. Returns max relative error."""
    cache = forward(x, weights, biases)
    dW_analytical, db_analytical = backward(cache, y, weights)

    max_rel_error = 0.0
    checks = 0

    # Check a few weights from each layer
    for layer_idx in range(len(weights)):
        for i in range(min(3, len(weights[layer_idx]))):
            for j in range(min(3, len(weights[layer_idx][0]))):
                # Numerical gradient: (L(w+eps) - L(w-eps)) / (2*eps)
                original = weights[layer_idx][i][j]

                weights[layer_idx][i][j] = original + epsilon
                loss_plus = compute_loss(x, y, weights, biases)

                weights[layer_idx][i][j] = original - epsilon
                loss_minus = compute_loss(x, y, weights, biases)

                weights[layer_idx][i][j] = original  # restore

                # TODO: Compute the numerical gradient using finite differences.
                # Hint: (loss_plus - loss_minus) / (2.0 * epsilon)
                numerical = ____
                analytical = dW_analytical[layer_idx][i][j]

                denom = max(abs(numerical) + abs(analytical), 1e-8)
                rel_error = abs(numerical - analytical) / denom
                max_rel_error = max(max_rel_error, rel_error)
                checks += 1

    return max_rel_error


rel_error = numerical_gradient_check(X[0], Y[0], weights, biases)
print(f"\n=== Gradient Check ===")
print(f"Max relative error: {rel_error:.2e}")
print(f"Status: {'PASS' if rel_error < 1e-5 else 'WARN (>1e-5)'}")
print(f"  (< 1e-5 means backprop is correctly implemented)")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Demonstrate vanishing gradients with sigmoid
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Vanishing Gradients with Sigmoid ===")

# Build progressively deeper networks
for depth in [3, 5, 8, 12]:
    deep_dims = [n_features] + [64] * depth + [n_classes]
    deep_w, deep_b = init_params(deep_dims)
    deep_cache = forward(X[0], deep_w, deep_b)
    deep_dW, deep_db = backward(deep_cache, Y[0], deep_w)

    grad_mags = []
    for i, dW in enumerate(deep_dW):
        flat = [abs(g) for row in dW for g in row]
        grad_mags.append(sum(flat) / len(flat))

    print(f"\n  Depth={depth}: gradient magnitudes per layer")
    for i, mag in enumerate(grad_mags):
        bar = "#" * max(1, int(mag * 1e6))
        print(f"    Layer {i+1:2d}: {mag:.2e} {bar[:40]}")

    ratio = grad_mags[0] / max(grad_mags[-1], 1e-15)
    print(f"    Ratio first/last: {ratio:.0e}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Fix with ReLU + proper initialization
# ══════════════════════════════════════════════════════════════════════


def relu(z: float) -> float:
    return max(0.0, z)


def forward_relu(x: list[float], weights: list, biases: list) -> dict:
    """Forward pass with ReLU activations (except output)."""
    cache = {"activations": [x], "pre_activations": []}
    current = x

    for layer_idx in range(len(weights)):
        W = weights[layer_idx]
        b = biases[layer_idx]
        d_in = len(current)
        d_out = len(b)

        z = [
            sum(current[j] * W[j][k] for j in range(d_in)) + b[k] for k in range(d_out)
        ]
        cache["pre_activations"].append(z)

        if layer_idx < len(weights) - 1:
            current = [relu(zi) for zi in z]
        else:
            current = softmax(z)

        cache["activations"].append(current)

    return cache


def backward_relu(cache: dict, y_true: list[float], weights: list) -> tuple[list, list]:
    """Backprop with ReLU."""
    activations = cache["activations"]
    pre_activations = cache["pre_activations"]
    n_layers = len(weights)

    dW_list = []
    db_list = []
    delta = [activations[-1][k] - y_true[k] for k in range(len(y_true))]

    for layer_idx in range(n_layers - 1, -1, -1):
        a_prev = activations[layer_idx]
        d_in = len(a_prev)
        d_out = len(delta)

        dW = [[a_prev[j] * delta[k] for k in range(d_out)] for j in range(d_in)]
        db = list(delta)
        dW_list.insert(0, dW)
        db_list.insert(0, db)

        if layer_idx > 0:
            W = weights[layer_idx]
            z_prev = pre_activations[layer_idx - 1]
            delta_new = []
            for j in range(d_in):
                grad_sum = sum(W[j][k] * delta[k] for k in range(d_out))
                relu_deriv = 1.0 if z_prev[j] > 0 else 0.0
                delta_new.append(grad_sum * relu_deriv)
            delta = delta_new

    return dW_list, db_list


def he_init_params(dims: list[int]) -> tuple[list, list]:
    """He initialization for ReLU networks."""
    weights = []
    biases = []
    for i in range(len(dims) - 1):
        # TODO: Compute He standard deviation for this layer.
        # Hint: std = math.sqrt(2.0 / dims[i])
        std = ____
        W = [[random.gauss(0, std) for _ in range(dims[i + 1])] for _ in range(dims[i])]
        b = [0.0] * dims[i + 1]
        weights.append(W)
        biases.append(b)
    return weights, biases


print(f"\n=== ReLU + He Init: Gradient Flow ===")

for depth in [3, 5, 8, 12]:
    deep_dims = [n_features] + [64] * depth + [n_classes]
    deep_w, deep_b = he_init_params(deep_dims)
    deep_cache = forward_relu(X[0], deep_w, deep_b)
    deep_dW, deep_db = backward_relu(deep_cache, Y[0], deep_w)

    grad_mags = []
    for i, dW in enumerate(deep_dW):
        flat = [abs(g) for row in dW for g in row]
        grad_mags.append(sum(flat) / len(flat))

    print(f"\n  Depth={depth}: gradient magnitudes per layer")
    for i, mag in enumerate(grad_mags):
        bar = "#" * max(1, int(mag * 1e4))
        print(f"    Layer {i+1:2d}: {mag:.2e} {bar[:40]}")

    ratio = grad_mags[0] / max(grad_mags[-1], 1e-15)
    print(f"    Ratio first/last: {ratio:.1f}x (much better!)")

viz = ModelVisualizer()
print(f"\nKey takeaway: ReLU + He init keeps gradients flowing through deep networks.")
print(f"  Sigmoid max gradient = 0.25 -> 0.25^12 = {0.25**12:.2e} (vanished)")
print(f"  ReLU gradient = 1.0 for active neurons -> gradients preserved")

print(
    "\n✓ Exercise 5 complete — backpropagation implemented and gradient flow analyzed"
)
