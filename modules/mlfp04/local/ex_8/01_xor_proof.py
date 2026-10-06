# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 8.1: The XOR Proof — Why Hidden Layers Exist
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Reproduce the historical XOR experiment (Minsky & Papert, 1969)
#   - See that a linear model cannot exceed ~50% on XOR no matter how long
#     you train it
#   - Watch a single hidden layer + ReLU break the 50% ceiling
#   - Build the intuition that hidden layers = composed piecewise-linear
#     boundaries = universal function approximation
#
# PREREQUISITES: MLFP04 Exercise 7 (recommender embeddings — the pivot
# from matrix factorisation to learned features).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why linear models cannot learn XOR
#   2. Build — nn.Linear head vs nn.Sequential with a hidden layer
#   3. Train — fit both to the same XOR dataset
#   4. Visualise — loss curves + the two learned decision boundaries
#   5. Apply — Singapore card-fraud detection: when XOR hides in features
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots

from shared.mlfp04.ex_8 import (
    N_FEATS_XOR,
    OUTPUT_DIR,
    make_xor_data,
    train_xor_net,
    viz,
    xor_accuracy,
)

print("\n" + "=" * 70)
print("  XOR Proof — Linear vs Non-Linear Decision Boundaries")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Linear Models Fail on XOR
# ════════════════════════════════════════════════════════════════════════
# A linear model computes y_hat = sigma(Wx + b). The decision boundary is
# a single hyperplane Wx + b = 0. XOR labels a point POSITIVE when its two
# signal features have OPPOSITE signs — the (+, -) and (-, +) quadrants —
# and NEGATIVE when they share a sign — the (+, +) and (-, -) quadrants.
# No single line separates the two classes — each class sits in two
# diagonally opposed quadrants. You can train a linear model forever; it
# cannot meaningfully beat simply predicting the majority class.
#
# A hidden layer changes the rules. Each ReLU neuron creates its own
# piecewise-linear split of the input space. Stack a few of them and the
# network can carve out the diagonal regions XOR needs. This is the
# universal approximation theorem at work: a wide enough hidden layer
# can approximate any continuous function on a bounded domain.
#
# HISTORICAL NOTE: Minsky and Papert's 1969 book "Perceptrons" proved this
# limit for a single-layer perceptron and effectively froze neural network
# research for 15 years until backpropagation unlocked deeper stacks.

# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD both models
# ════════════════════════════════════════════════════════════════════════
X, y, y_np = make_xor_data()
print(f"\nXOR dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class balance: {y_np.mean():.2f} (0.5 = balanced)")

# TODO: A purely linear model — one nn.Linear from N_FEATS_XOR inputs to
# a single logit (no hidden layer)
linear_net = ____

# TODO: An nn.Sequential with two hidden layers (32 then 16 units, ReLU
# after each) ending in a single logit
hidden_net = ____

# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN both with the same loss
# ════════════════════════════════════════════════════════════════════════
print("\n--- Training ---")
linear_losses = train_xor_net(
    linear_net, X, y, torch.optim.SGD(linear_net.parameters(), lr=0.1), n_epochs=50
)
# TODO: Train hidden_net with train_xor_net using Adam (lr=0.01) for 100
# epochs
hidden_losses = ____

acc_linear = xor_accuracy(linear_net, X, y_np)
acc_hidden = xor_accuracy(hidden_net, X, y_np)

print(
    f"Linear (no hidden layer): final_loss={linear_losses[-1]:.4f}, acc={acc_linear:.4f}"
)
print(
    f"Hidden (32+16 ReLU):      final_loss={hidden_losses[-1]:.4f}, acc={acc_hidden:.4f}"
)

# ── Checkpoint ──────────────────────────────────────────────────────────
assert acc_hidden > acc_linear + 0.1, (
    f"Task 3: hidden network ({acc_hidden:.2f}) should clearly beat linear "
    f"({acc_linear:.2f}) — the whole point of this exercise."
)
assert hidden_losses[-1] < hidden_losses[0], "Hidden network should reduce loss"
print("\n[ok] Checkpoint passed — hidden layers beat linear on XOR\n")

# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE — overlay the two learning curves
# ════════════════════════════════════════════════════════════════════════
# R9A: we need a visual proof, not just a number. The overlaid loss curves
# show the linear model plateauing at ~0.69 (random on a balanced binary
# task) while the hidden-layer model drives loss to near zero.
fig = viz.training_history(
    {"Linear (no hidden)": linear_losses, "Hidden (32+16 ReLU)": hidden_losses},
    x_label="Epoch",
)
fig.update_layout(title="XOR: Linear Ceiling vs Hidden-Layer Escape")
output_path = OUTPUT_DIR / "01_xor_loss_curves.html"
fig.write_html(output_path)
print(f"[viz] Loss curves: {output_path}")

# ── Decision boundaries: the visual proof ─────────────────────────────
# Evaluate both trained models on a grid over the two signal features
# (the noise features are held at 0) and colour each point by P(y=1).
# The linear model can only draw ONE straight boundary; the hidden-layer
# model carves out the two opposite-sign quadrants.
grid = np.linspace(-3, 3, 121)
g0, g1 = np.meshgrid(grid, grid)
grid_X = np.zeros((g0.size, N_FEATS_XOR), dtype=np.float32)
grid_X[:, 0] = g0.ravel()
grid_X[:, 1] = g1.ravel()
grid_t = torch.from_numpy(grid_X)
linear_net.eval()
hidden_net.eval()
with torch.no_grad():
    # TODO: P(y=1) for every grid point from each model — apply the model,
    # squash the logits with a sigmoid, and reshape to the grid's shape
    p_linear = ____
    p_hidden = ____

X_np = X.numpy()
fig_db = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        f"Linear model (acc {acc_linear:.2f})",
        f"Hidden layers (acc {acc_hidden:.2f})",
    ),
)
for col, probs in ((1, p_linear), (2, p_hidden)):
    fig_db.add_trace(
        go.Contour(
            x=grid,
            y=grid,
            z=probs,
            zmin=0,
            zmax=1,
            colorscale="RdBu_r",
            contours=dict(start=0.0, end=1.0, size=0.1),
            showscale=(col == 2),
            colorbar=dict(title="P(y=1)"),
        ),
        row=1,
        col=col,
    )
    for label, colour in ((0, "white"), (1, "black")):
        pts = X_np[y_np == label]
        fig_db.add_trace(
            go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode="markers",
                marker=dict(color=colour, size=5, line=dict(width=0.5)),
                name=f"class {label}",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
fig_db.update_xaxes(title_text="feature 0")
fig_db.update_yaxes(title_text="feature 1")
fig_db.update_layout(title="XOR Decision Boundaries: One Line vs Hidden Layers")
db_path = OUTPUT_DIR / "01_xor_decision_boundaries.html"
fig_db.write_html(db_path)
print(f"[viz] Decision boundaries: {db_path}")

# INTERPRETATION (computed from this run):
print(
    f"\nLinear model: loss {linear_losses[-1]:.3f} (ln 2 = 0.693 is a coin "
    f"flip), accuracy {acc_linear:.2f} vs majority-class rate "
    f"{max(y_np.mean(), 1 - y_np.mean()):.2f}."
)
print(
    f"Hidden-layer model: loss {hidden_losses[-1]:.3f}, accuracy {acc_hidden:.2f}."
)
if acc_hidden - acc_linear > 0.3:
    print(
        "  -> The boundary plot shows why: the linear model draws one "
        "straight cut through the plane, the hidden layers draw the "
        "quadrant pattern XOR needs."
    )

# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore Card Fraud — XOR Hidden In Fraud Features
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore bank's anti-fraud desk flags card-present
# transactions using two features: "is this a high-value purchase?" and "is the card
# near its normal merchant category?". In isolation, neither feature is
# suspicious — a S$8,000 watch purchase at a jeweller is fine; a S$40
# petrol top-up in a new country is fine. But the XOR combination
# (high-value AND out-of-category, OR low-value AND in-category) is
# exactly the signature of a compromised card being tested.
#
# If the fraud signal really is an XOR of the two features, a logistic
# regression over them is stuck near the majority-class rate — exactly
# what the linear model above did — while a small hidden layer can learn
# the quadrant pattern.
#
# BUSINESS IMPACT (illustrative assumptions, not measured figures):
#   - Suppose XOR-pattern card testing costs the bank ~S$4M/month in missed
#     fraud and a hidden-layer model catches 90% of it: ~S$3.6M/month
#     recovered
#   - Inference for a model this small costs a few dollars a month
#   - The real number depends on how much of the fraud follows the
#     pattern — measure recall on labelled cases before claiming it
#
# LIMITATION: Adding more irrelevant features can swamp the signal. Even
# a hidden layer benefits from feature selection (MLFP02 territory).

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Reproduced the historical XOR proof with a modern PyTorch stack
  [x] Saw a linear model hit its accuracy ceiling within 50 epochs
  [x] Saw a hidden-layer model break through that ceiling
  [x] Visualised the learning curves AND the two decision boundaries
  [x] Sized an (illustrative) card-fraud case where the signal is an XOR
      of two features

  KEY INSIGHT: Hidden layers are not decoration. They are the mechanism
  by which neural networks represent non-linear decision boundaries. The
  rest of deep learning is a thousand ways to train them better.

  Next: 02_activations_init.py — which non-linearity, and which weight
  initialisation, makes those hidden layers actually learn?
"""
)
