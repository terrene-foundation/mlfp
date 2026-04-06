# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 1: Linear Regression as a Neural Network
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build linear regression as a single-neuron network — forward
#   pass, MSE loss, gradient descent — to show that DL starts from
#   familiar ground.
#
# TASKS:
#   1. Load Singapore HDB resale data
#   2. Implement forward pass (y = wx + b)
#   3. Compute MSE loss
#   4. Implement gradient descent manually
#   5. Compare with polars-native OLS solution
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import polars as pl

from kailash_ml import DataExplorer, ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load Singapore HDB resale data
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
df = loader.load("ascent07", "hdb_resale_sample.parquet")

explorer = DataExplorer()
summary = explorer.analyze(df)

print("=== HDB Resale Data ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(f"\nSample:")
print(df.head(5))

# Use floor_area_sqm as feature (x), resale_price as target (y)
x_raw = df["floor_area_sqm"].to_list()
y_raw = df["resale_price"].to_list()

# Normalize for stable gradient descent
x_mean, x_std = (
    sum(x_raw) / len(x_raw),
    (sum((xi - sum(x_raw) / len(x_raw)) ** 2 for xi in x_raw) / len(x_raw)) ** 0.5,
)
y_mean, y_std = (
    sum(y_raw) / len(y_raw),
    (sum((yi - sum(y_raw) / len(y_raw)) ** 2 for yi in y_raw) / len(y_raw)) ** 0.5,
)

x_norm = [(xi - x_mean) / x_std for xi in x_raw]
y_norm = [(yi - y_mean) / y_std for yi in y_raw]
n = len(x_norm)

print(f"\nFeature: floor_area_sqm (mean={x_mean:.1f}, std={x_std:.1f})")
print(f"Target:  resale_price   (mean={y_mean:.0f}, std={y_std:.0f})")
print(f"Samples: {n}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement forward pass (y = wx + b)
# ══════════════════════════════════════════════════════════════════════

# Single neuron: y_hat = w * x + b
w = 0.0  # weight (initially zero)
b = 0.0  # bias (initially zero)


def forward(x_i: float, w: float, b: float) -> float:
    """Single neuron forward pass: y = wx + b."""
    return w * x_i + b


# Test forward pass
y_hat_test = forward(x_norm[0], w, b)
print(f"\n=== Forward Pass Test ===")
print(f"Input x={x_norm[0]:.4f}, w={w}, b={b}")
print(f"Prediction: {y_hat_test:.4f} (expected ~0 with zero weights)")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compute MSE loss
# ══════════════════════════════════════════════════════════════════════


def mse_loss(y_true: list[float], y_pred: list[float]) -> float:
    """Mean Squared Error: L = (1/n) * sum((y - y_hat)^2)."""
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


# Compute initial loss (should be high with zero weights)
predictions = [forward(xi, w, b) for xi in x_norm]
initial_loss = mse_loss(y_norm, predictions)
print(f"\n=== MSE Loss ===")
print(f"Initial loss (w=0, b=0): {initial_loss:.4f}")
print(f"This equals variance of y since predictions are all 0")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Implement gradient descent manually
# ══════════════════════════════════════════════════════════════════════

# Gradients of MSE w.r.t. w and b:
#   dL/dw = (2/n) * sum((y_hat - y) * x)
#   dL/db = (2/n) * sum(y_hat - y)

learning_rate = 0.1
epochs = 50
history = {"epoch": [], "loss": [], "w": [], "b": []}

for epoch in range(epochs):
    # Forward pass for all samples
    y_pred = [forward(xi, w, b) for xi in x_norm]

    # Compute loss
    loss = mse_loss(y_norm, y_pred)

    # Compute gradients
    dw = (2.0 / n) * sum((yp - yt) * xi for yp, yt, xi in zip(y_pred, y_norm, x_norm))
    db = (2.0 / n) * sum(yp - yt for yp, yt in zip(y_pred, y_norm))

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Record history
    history["epoch"].append(epoch)
    history["loss"].append(loss)
    history["w"].append(w)
    history["b"].append(b)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: loss={loss:.6f}, w={w:.4f}, b={b:.4f}")

print(f"\nFinal: w={w:.4f}, b={b:.4f}, loss={history['loss'][-1]:.6f}")

# Visualize training curve
viz = ModelVisualizer()
history_df = pl.DataFrame(history)
fig = viz.plot_training_curves(history_df)
print("Training curve plotted.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare with polars-native OLS solution
# ══════════════════════════════════════════════════════════════════════

# OLS closed-form: w = cov(x,y) / var(x), b = mean(y) - w * mean(x)
norm_df = pl.DataFrame({"x": x_norm, "y": y_norm})

cov_xy = norm_df.select(
    ((pl.col("x") - pl.col("x").mean()) * (pl.col("y") - pl.col("y").mean())).mean()
).item()
var_x = norm_df.select(pl.col("x").var()).item()

w_ols = cov_xy / var_x
b_ols = sum(y_norm) / n - w_ols * (sum(x_norm) / n)

y_pred_ols = [w_ols * xi + b_ols for xi in x_norm]
ols_loss = mse_loss(y_norm, y_pred_ols)

print(f"\n=== Comparison: Gradient Descent vs OLS ===")
print(f"GD:  w={w:.4f}, b={b:.4f}, loss={history['loss'][-1]:.6f}")
print(f"OLS: w={w_ols:.4f}, b={b_ols:.4f}, loss={ols_loss:.6f}")
print(f"Difference in w: {abs(w - w_ols):.6f}")
print(f"Difference in b: {abs(b - b_ols):.6f}")
print(f"\nKey insight: gradient descent converges to the OLS solution!")
print(f"But GD scales to millions of parameters — OLS does not.")

print("\n✓ Exercise 1 complete — linear regression as a single-neuron network")
