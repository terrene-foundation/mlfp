# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 6: Optimizers and Learning Rate Scheduling
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare SGD, SGD+momentum, Adam optimizers and implement
#   learning rate warmup + cosine annealing.
#
# TASKS:
#   1. Implement SGD with mini-batches
#   2. Add momentum to SGD
#   3. Implement Adam optimizer
#   4. Compare convergence curves with ModelVisualizer
#   5. Add learning rate warmup + cosine decay schedule
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import copy
import math
import random

import polars as pl

from kailash_ml import ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement SGD with mini-batches
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
data = loader.load("ascent07", "mnist_sample.parquet")

# Extract features and labels
X = data.select([c for c in data.columns if c != "label"]).to_numpy() / 255.0
y_raw = data["label"].to_numpy()

# One-hot encode labels (10 classes)
n_classes = 10
y = [[1.0 if j == int(label) else 0.0 for j in range(n_classes)] for label in y_raw]

n_samples, n_features = X.shape
print(
    f"=== Dataset: {n_samples} samples, {n_features} features, {n_classes} classes ==="
)

# Simple 2-layer network: input → hidden(128) → output(10)
random.seed(42)
hidden_size = 128


def init_weights(rows: int, cols: int) -> list[list[float]]:
    """He initialization: std = sqrt(2 / fan_in)."""
    s = math.sqrt(2.0 / rows)
    return [[random.gauss(0, s) for _ in range(cols)] for _ in range(rows)]


W1 = init_weights(n_features, hidden_size)
b1 = [0.0] * hidden_size
W2 = init_weights(hidden_size, n_classes)
b2 = [0.0] * n_classes


def relu(x: list[float]) -> list[float]:
    return [max(0.0, v) for v in x]


def softmax(x: list[float]) -> list[float]:
    max_x = max(x)
    exps = [math.exp(v - max_x) for v in x]
    s = sum(exps)
    return [e / s for e in exps]


def forward(x_row, w1, b1_, w2, b2_):
    """Forward pass: input → ReLU → softmax."""
    hidden = [
        sum(x_row[j] * w1[j][k] for j in range(len(x_row))) + b1_[k]
        for k in range(len(b1_))
    ]
    h_act = relu(hidden)
    logits = [
        sum(h_act[j] * w2[j][k] for j in range(len(h_act))) + b2_[k]
        for k in range(len(b2_))
    ]
    probs = softmax(logits)
    return hidden, h_act, logits, probs


def cross_entropy_loss(probs: list[float], target: list[float]) -> float:
    return -sum(t * math.log(max(p, 1e-10)) for p, t in zip(probs, target))


def sgd_step(params: list, grads: list, lr: float):
    """Vanilla SGD update."""
    for i in range(len(params)):
        if isinstance(params[i], list):
            for j in range(len(params[i])):
                if isinstance(params[i][j], list):
                    for k in range(len(params[i][j])):
                        params[i][j][k] -= lr * grads[i][j][k]
                else:
                    params[i][j] -= lr * grads[i][j]
        else:
            params[i] -= lr * grads[i]


# Train with vanilla SGD
batch_size = 32
lr = 0.01
epochs = 5
sgd_losses = []

W1_sgd, b1_sgd = copy.deepcopy(W1), copy.deepcopy(b1)
W2_sgd, b2_sgd = copy.deepcopy(W2), copy.deepcopy(b2)

for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    indices = list(range(min(500, n_samples)))  # Use subset for speed
    random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_loss = 0.0

        # Accumulate gradients over batch
        for idx in batch_idx:
            x_row = X[idx].tolist()
            target = y[idx]
            hidden, h_act, logits, probs = forward(
                x_row, W1_sgd, b1_sgd, W2_sgd, b2_sgd
            )
            batch_loss += cross_entropy_loss(probs, target)

            # Backward pass (output layer)
            d_logits = [probs[k] - target[k] for k in range(n_classes)]
            for j in range(hidden_size):
                for k in range(n_classes):
                    W2_sgd[j][k] -= lr / len(batch_idx) * h_act[j] * d_logits[k]
            for k in range(n_classes):
                b2_sgd[k] -= lr / len(batch_idx) * d_logits[k]

            # Backward pass (hidden layer)
            d_hidden = [
                sum(d_logits[k] * W2_sgd[j][k] for k in range(n_classes))
                * (1.0 if hidden[j] > 0 else 0.0)
                for j in range(hidden_size)
            ]
            for i_feat in range(n_features):
                for j in range(hidden_size):
                    W1_sgd[i_feat][j] -= (
                        lr / len(batch_idx) * x_row[i_feat] * d_hidden[j]
                    )
            for j in range(hidden_size):
                b1_sgd[j] -= lr / len(batch_idx) * d_hidden[j]

        epoch_loss += batch_loss / len(batch_idx)
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    sgd_losses.append(avg_loss)
    print(f"SGD Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: SGD with momentum
# ══════════════════════════════════════════════════════════════════════

W1_mom, b1_mom = copy.deepcopy(W1), copy.deepcopy(b1)
W2_mom, b2_mom = copy.deepcopy(W2), copy.deepcopy(b2)
momentum = 0.9
# Velocity buffers (same shape as weights)
vW1 = [[0.0] * hidden_size for _ in range(n_features)]
vb1 = [0.0] * hidden_size
vW2 = [[0.0] * n_classes for _ in range(hidden_size)]
vb2 = [0.0] * n_classes

mom_losses = []
for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    indices = list(range(min(500, n_samples)))
    random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_loss = 0.0

        # Accumulate gradients
        gW2 = [[0.0] * n_classes for _ in range(hidden_size)]
        gb2 = [0.0] * n_classes
        gW1 = [[0.0] * hidden_size for _ in range(n_features)]
        gb1_ = [0.0] * hidden_size

        for idx in batch_idx:
            x_row = X[idx].tolist()
            target = y[idx]
            hidden, h_act, logits, probs = forward(
                x_row, W1_mom, b1_mom, W2_mom, b2_mom
            )
            batch_loss += cross_entropy_loss(probs, target)

            d_logits = [probs[k] - target[k] for k in range(n_classes)]
            for j in range(hidden_size):
                for k in range(n_classes):
                    gW2[j][k] += h_act[j] * d_logits[k] / len(batch_idx)
            for k in range(n_classes):
                gb2[k] += d_logits[k] / len(batch_idx)

            d_hidden = [
                sum(d_logits[k] * W2_mom[j][k] for k in range(n_classes))
                * (1.0 if hidden[j] > 0 else 0.0)
                for j in range(hidden_size)
            ]
            for i_feat in range(n_features):
                for j in range(hidden_size):
                    gW1[i_feat][j] += x_row[i_feat] * d_hidden[j] / len(batch_idx)
            for j in range(hidden_size):
                gb1_[j] += d_hidden[j] / len(batch_idx)

        # Momentum update: v = momentum * v + grad; w -= lr * v
        for j in range(hidden_size):
            for k in range(n_classes):
                vW2[j][k] = momentum * vW2[j][k] + gW2[j][k]
                W2_mom[j][k] -= lr * vW2[j][k]
            vb1[j] = momentum * vb1[j] + gb1_[j]
            b1_mom[j] -= lr * vb1[j]
        for k in range(n_classes):
            vb2[k] = momentum * vb2[k] + gb2[k]
            b2_mom[k] -= lr * vb2[k]
        for i_feat in range(n_features):
            for j in range(hidden_size):
                vW1[i_feat][j] = momentum * vW1[i_feat][j] + gW1[i_feat][j]
                W1_mom[i_feat][j] -= lr * vW1[i_feat][j]

        epoch_loss += batch_loss / len(batch_idx)
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    mom_losses.append(avg_loss)
    print(f"SGD+Momentum Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement Adam optimizer
# ══════════════════════════════════════════════════════════════════════

print("\n=== Adam Optimizer ===")
print("Adam = adaptive moment estimation: combines momentum (first moment)")
print("with RMSprop-style scaling (second moment) for per-parameter learning rates.")

# Adam hyperparameters
adam_lr = 0.001
beta1, beta2, eps = 0.9, 0.999, 1e-8

# Adam tracks two moments per parameter (m = first moment, v = second moment)
# Update rule:
#   m = beta1 * m + (1 - beta1) * grad
#   v = beta2 * v + (1 - beta2) * grad^2
#   m_hat = m / (1 - beta1^t)   # bias correction
#   v_hat = v / (1 - beta2^t)
#   param -= lr * m_hat / (sqrt(v_hat) + eps)

W1_adam, b1_adam = copy.deepcopy(W1), copy.deepcopy(b1)
W2_adam, b2_adam = copy.deepcopy(W2), copy.deepcopy(b2)

# First moments (m) and second moments (v) — same shape as weights
mW1 = [[0.0] * hidden_size for _ in range(n_features)]
vW1_adam = [[0.0] * hidden_size for _ in range(n_features)]
mb1 = [0.0] * hidden_size
vb1_adam = [0.0] * hidden_size
mW2 = [[0.0] * n_classes for _ in range(hidden_size)]
vW2_adam = [[0.0] * n_classes for _ in range(hidden_size)]
mb2 = [0.0] * n_classes
vb2_adam = [0.0] * n_classes

adam_losses = []
t_step = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    indices = list(range(min(500, n_samples)))
    random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_loss = 0.0
        t_step += 1

        gW2 = [[0.0] * n_classes for _ in range(hidden_size)]
        gb2 = [0.0] * n_classes
        gW1_batch = [[0.0] * hidden_size for _ in range(n_features)]
        gb1_batch = [0.0] * hidden_size

        for idx in batch_idx:
            x_row = X[idx].tolist()
            target = y[idx]
            hidden, h_act, logits, probs = forward(
                x_row, W1_adam, b1_adam, W2_adam, b2_adam
            )
            batch_loss += cross_entropy_loss(probs, target)

            d_logits = [probs[k] - target[k] for k in range(n_classes)]
            for j in range(hidden_size):
                for k in range(n_classes):
                    gW2[j][k] += h_act[j] * d_logits[k] / len(batch_idx)
            for k in range(n_classes):
                gb2[k] += d_logits[k] / len(batch_idx)

            d_hidden = [
                sum(d_logits[k] * W2_adam[j][k] for k in range(n_classes))
                * (1.0 if hidden[j] > 0 else 0.0)
                for j in range(hidden_size)
            ]
            for i_feat in range(n_features):
                for j in range(hidden_size):
                    gW1_batch[i_feat][j] += x_row[i_feat] * d_hidden[j] / len(batch_idx)
            for j in range(hidden_size):
                gb1_batch[j] += d_hidden[j] / len(batch_idx)

        # Adam update for W2, b2
        bc1 = 1 - beta1**t_step
        bc2 = 1 - beta2**t_step
        for j in range(hidden_size):
            for k in range(n_classes):
                mW2[j][k] = beta1 * mW2[j][k] + (1 - beta1) * gW2[j][k]
                vW2_adam[j][k] = beta2 * vW2_adam[j][k] + (1 - beta2) * gW2[j][k] ** 2
                m_hat = mW2[j][k] / bc1
                v_hat = vW2_adam[j][k] / bc2
                W2_adam[j][k] -= adam_lr * m_hat / (math.sqrt(v_hat) + eps)
        for k in range(n_classes):
            mb2[k] = beta1 * mb2[k] + (1 - beta1) * gb2[k]
            vb2_adam[k] = beta2 * vb2_adam[k] + (1 - beta2) * gb2[k] ** 2
            b2_adam[k] -= (
                adam_lr * (mb2[k] / bc1) / (math.sqrt(vb2_adam[k] / bc2) + eps)
            )

        # Adam update for W1, b1
        for i_feat in range(n_features):
            for j in range(hidden_size):
                mW1[i_feat][j] = (
                    beta1 * mW1[i_feat][j] + (1 - beta1) * gW1_batch[i_feat][j]
                )
                vW1_adam[i_feat][j] = (
                    beta2 * vW1_adam[i_feat][j]
                    + (1 - beta2) * gW1_batch[i_feat][j] ** 2
                )
                m_hat = mW1[i_feat][j] / bc1
                v_hat = vW1_adam[i_feat][j] / bc2
                W1_adam[i_feat][j] -= adam_lr * m_hat / (math.sqrt(v_hat) + eps)
        for j in range(hidden_size):
            mb1[j] = beta1 * mb1[j] + (1 - beta1) * gb1_batch[j]
            vb1_adam[j] = beta2 * vb1_adam[j] + (1 - beta2) * gb1_batch[j] ** 2
            b1_adam[j] -= (
                adam_lr * (mb1[j] / bc1) / (math.sqrt(vb1_adam[j] / bc2) + eps)
            )

        epoch_loss += batch_loss / len(batch_idx)
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    adam_losses.append(avg_loss)
    print(f"Adam Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare convergence curves
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.plot_training_curves(
    {
        "SGD": sgd_losses,
        "SGD+Momentum": mom_losses,
        "Adam": adam_losses,
    }
)
fig.write_html("optimizer_comparison.html")

print(f"\n=== Optimizer Comparison ===")
print(f"SGD final loss:          {sgd_losses[-1]:.4f}")
print(f"SGD+Momentum final loss: {mom_losses[-1]:.4f}")
print(f"Adam final loss:         {adam_losses[-1]:.4f}")
print(f"\nKey takeaways:")
print(f"  - SGD: simple but slow convergence, sensitive to learning rate")
print(
    f"  - SGD+Momentum: smooths oscillations, accelerates in consistent gradient directions"
)
print(f"  - Adam: adaptive per-parameter rates, fast convergence, good default choice")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Learning rate warmup + cosine decay
# ══════════════════════════════════════════════════════════════════════


def cosine_schedule(
    step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 1e-6
) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


total_steps = 1000
warmup_steps = 100
max_lr = 0.001

schedule = [
    cosine_schedule(s, total_steps, warmup_steps, max_lr) for s in range(total_steps)
]

print(f"\n=== Cosine Schedule with Warmup ===")
print(f"Warmup: {warmup_steps} steps (linear ramp from 0 to {max_lr})")
print(f"Decay: cosine from {max_lr} to 1e-6 over {total_steps - warmup_steps} steps")
print(f"LR at step 0:    {schedule[0]:.6f}")
print(f"LR at step 50:   {schedule[50]:.6f}")
print(f"LR at step 100:  {schedule[100]:.6f} (peak)")
print(f"LR at step 500:  {schedule[500]:.6f}")
print(f"LR at step 999:  {schedule[999]:.6f}")

print("\n✓ Exercise 6 complete — SGD vs Momentum vs Adam + cosine scheduling")
