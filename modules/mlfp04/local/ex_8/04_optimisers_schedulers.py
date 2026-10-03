# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 8.4: Optimisers and Learning-Rate Schedulers
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compare SGD, SGD+momentum, Adam, and AdamW on the same task
#   - Apply a cosine-annealing LR schedule with a linear warmup
#   - Track LR and loss together during training
#   - Pick the right default (AdamW + cosine) for most deep learning work
#
# PREREQUISITES: 03_cnn_residual.py
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — optimisers as adaptive gradient rescaling
#   2. Build — optimiser grid and a warmup+cosine scheduler
#   3. Train — short runs for each optimiser, then one full run with
#              the scheduler
#   4. Visualise — optimiser curves + schedule trajectory
#   5. Apply — payments fraud model: why warmup prevents divergent retrains
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import torch.nn as nn
import torch.optim as optim

from shared.mlfp04.ex_8 import (
    OUTPUT_DIR,
    SG_HOSPITAL_CLASSES,
    TriageCNN,
    build_sg_loaders,
    device,
    eval_cnn,
    train_cnn_one_epoch,
    viz,
)

print("\n" + "=" * 70)
print("  Optimisers and Schedulers — How Fast Can You Converge?")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Optimisers as per-parameter learning rates
# ════════════════════════════════════════════════════════════════════════
# SGD takes a step in the direction of the gradient scaled by a single
# global learning rate. That is enough for convex problems, but neural
# networks have curvature that varies by parameter: the last-layer bias
# gets a strong signal, a third-layer filter gets a faint one.
#
# Momentum adds a rolling average of past gradients, which damps noise
# and accelerates convergence on flat regions. Adam goes further by
# also normalising each parameter by a rolling estimate of its own
# gradient variance. AdamW fixes Adam's weight-decay implementation so
# the regulariser behaves as intended.
#
# Schedulers modulate the base LR over time. Cosine annealing reduces
# the LR along a cosine curve from LR_max to LR_min, taking large steps
# early (to escape poor local minima) and small steps late (to settle
# into a good one). A linear warmup for the first few epochs prevents
# Adam from blowing up before its variance estimates have stabilised.

train_loader, test_loader, _, _ = build_sg_loaders()
criterion = nn.BCEWithLogitsLoss()
n_classes = len(SG_HOSPITAL_CLASSES)

# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the optimiser grid + scheduler factory
# ════════════════════════════════════════════════════════════════════════


def fresh_model() -> TriageCNN:
    return TriageCNN(n_classes=n_classes, dropout_rate=0.3).to(device)


optimiser_builders: dict[str, callable] = {
    "SGD lr=0.01": lambda p: optim.SGD(p, lr=0.01),
    # TODO: fill each builder — the key names the optimiser and its settings
    # (AdamW also uses weight_decay=1e-4)
    "SGD+momentum": lambda p: ____,
    "Adam lr=1e-3": lambda p: ____,
    "AdamW lr=1e-3": lambda p: ____,
}


def make_warmup_cosine(
    optimiser: optim.Optimizer, warmup_epochs: int, total_epochs: int
) -> optim.lr_scheduler.LambdaLR:
    """Linear warmup for ``warmup_epochs`` then cosine decay towards zero.

    LambdaLR multiplies each param group's base LR by ``lr_lambda(epoch)``.
    """

    def lr_lambda(epoch: int) -> float:
        # TODO: during warmup, ramp the multiplier linearly up to 1.0;
        # afterwards follow half a cosine from 1.0 down to 0.0 over the
        # remaining epochs (math.cos)
        if epoch < warmup_epochs:
            return ____
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return ____

    return optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the optimiser grid (5 epochs each)
# ════════════════════════════════════════════════════════════════════════
print("\n--- Optimiser comparison (5 epochs each) ---")
optimiser_histories: dict[str, list[float]] = {}
for name, builder in optimiser_builders.items():
    model = fresh_model()
    optimiser = builder(model.parameters())
    losses: list[float] = []
    for _ in range(5):
        loss, _ = train_cnn_one_epoch(model, train_loader, optimiser, criterion)
        losses.append(loss)
    optimiser_histories[name] = losses
    print(f"  {name:<18}: {losses[0]:.4f} -> {losses[-1]:.4f}")

# ── Checkpoint A ───────────────────────────────────────────────────────
assert all(
    h[-1] <= h[0] + 1e-3 for h in optimiser_histories.values()
), "Task 3: every optimiser should at least hold the line on train loss"

# Full run with scheduler
print("\n--- Full run: AdamW + warmup(2) + cosine(10) ---")
total_epochs = 10
warmup_epochs = 2

model = fresh_model()
# TODO: AdamW (lr=1e-3, weight_decay=1e-4) and your warmup+cosine scheduler
optimiser = ____
scheduler = ____

schedule_history = {"train_loss": [], "val_loss": [], "lr": []}
for epoch in range(total_epochs):
    train_loss, _ = train_cnn_one_epoch(model, train_loader, optimiser, criterion)
    val_loss = eval_cnn(model, test_loader, criterion)
    schedule_history["train_loss"].append(train_loss)
    schedule_history["val_loss"].append(val_loss)
    schedule_history["lr"].append(scheduler.get_last_lr()[0])
    scheduler.step()
    print(
        f"  Epoch {epoch + 1:>2}/{total_epochs}: "
        f"train={train_loss:.4f} val={val_loss:.4f} "
        f"lr={schedule_history['lr'][-1]:.6f}"
    )

# ── Checkpoint B ───────────────────────────────────────────────────────
assert (
    schedule_history["lr"][-1] < schedule_history["lr"][warmup_epochs]
), "Task 3: cosine schedule should be lower at the end than at the warmup peak"
assert (
    schedule_history["lr"][0] < schedule_history["lr"][warmup_epochs]
), "Task 3: warmup should start below the peak LR"
print("\n[ok] Checkpoint passed — warmup + cosine behave as specified\n")

# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the optimiser grid and the schedule trajectory
# ════════════════════════════════════════════════════════════════════════
fig_opt = viz.training_history(optimiser_histories, x_label="Epoch")
fig_opt.update_layout(title="Optimiser Comparison on TriageCNN")
opt_path = OUTPUT_DIR / "04_optimiser_curves.html"
fig_opt.write_html(opt_path)
print(f"[viz] Optimiser curves: {opt_path}")

fig_sched = viz.training_history(
    {
        "Train BCE": schedule_history["train_loss"],
        "Val BCE": schedule_history["val_loss"],
        "Learning Rate (x1e3)": [lr * 1000 for lr in schedule_history["lr"]],
    },
    x_label="Epoch",
)
fig_sched.update_layout(title="Warmup(2) + Cosine(10) — Loss and LR Together")
sched_path = OUTPUT_DIR / "04_schedule_curves.html"
fig_sched.write_html(sched_path)
print(f"[viz] Schedule trajectory: {sched_path}")

# ── (C) Learning rate schedule curve (standalone) ─────────────────────
import plotly.graph_objects as go

fig_lr = go.Figure()
epochs_lr = list(range(1, total_epochs + 1))
fig_lr.add_trace(
    go.Scatter(
        x=epochs_lr,
        y=schedule_history["lr"],
        mode="lines+markers",
        name="Learning Rate",
        marker_color="#00CC96",
        line=dict(width=3),
    )
)
fig_lr.add_vrect(
    x0=0.5,
    x1=warmup_epochs + 0.5,
    fillcolor="rgba(255, 165, 0, 0.15)",
    line_width=0,
    annotation_text="Warmup",
    annotation_position="top left",
)
fig_lr.add_vrect(
    x0=warmup_epochs + 0.5,
    x1=total_epochs + 0.5,
    fillcolor="rgba(99, 110, 250, 0.08)",
    line_width=0,
    annotation_text="Cosine Decay",
    annotation_position="top right",
)
fig_lr.update_layout(
    title="Learning Rate Schedule: Linear Warmup + Cosine Annealing",
    xaxis_title="Epoch",
    yaxis_title="Learning Rate",
    yaxis_tickformat=".1e",
)
lr_path = OUTPUT_DIR / "04_lr_schedule.html"
fig_lr.write_html(str(lr_path))
print(f"[viz] LR schedule: {lr_path}")

# ── (D) Optimizer comparison: final loss bar chart ────────────────────
opt_names = list(optimiser_histories.keys())
final_losses = [optimiser_histories[n][-1] for n in opt_names]
initial_losses = [optimiser_histories[n][0] for n in opt_names]
fig_bar = go.Figure()
fig_bar.add_trace(
    go.Bar(
        x=opt_names,
        y=initial_losses,
        name="Epoch 1 Loss",
        marker_color="#FECB52",
        text=[f"{v:.4f}" for v in initial_losses],
        textposition="outside",
    )
)
fig_bar.add_trace(
    go.Bar(
        x=opt_names,
        y=final_losses,
        name="Epoch 5 Loss",
        marker_color="#636EFA",
        text=[f"{v:.4f}" for v in final_losses],
        textposition="outside",
    )
)
fig_bar.update_layout(
    title="Optimizer Comparison: Initial vs Final Loss (5 epochs)",
    xaxis_title="Optimizer",
    yaxis_title="Training Loss (BCE)",
    barmode="group",
)
bar_path = OUTPUT_DIR / "04_optimiser_bar.html"
fig_bar.write_html(str(bar_path))
print(f"[viz] Optimiser bar chart: {bar_path}")

# INTERPRETATION (computed from this run):
ranked = sorted(optimiser_histories, key=lambda n: optimiser_histories[n][-1])
print("\nOptimisers ranked by epoch-5 training loss:")
for name in ranked:
    h = optimiser_histories[name]
    print(f"  {name:<18} {h[0]:.4f} -> {h[-1]:.4f}  (drop {h[0] - h[-1]:.4f})")
sgd_final = optimiser_histories["SGD lr=0.01"][-1]
adaptive_best = min(
    optimiser_histories["Adam lr=1e-3"][-1], optimiser_histories["AdamW lr=1e-3"][-1]
)
if adaptive_best < sgd_final:
    print(
        "  -> The adaptive optimisers (Adam/AdamW) reached a lower loss than "
        "plain SGD on the same 5-epoch budget."
    )
else:
    print(
        "  -> Plain SGD kept pace with Adam/AdamW on this run — on an easy "
        "problem with a well-chosen LR, the gap can be small."
    )
peak_epoch = int(np.argmax(schedule_history["lr"])) + 1
print(
    f"  Schedule: LR peaked at epoch {peak_epoch} "
    f"({max(schedule_history['lr']):.1e}) and decayed to "
    f"{schedule_history['lr'][-1]:.1e} by epoch {total_epochs}."
)

# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Payments Fraud Scoring (Southeast Asia)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (hypothetical): a regional e-wallet's fraud team retrains a
# transaction-risk scorer nightly over ~18M events. Version 1 uses plain
# SGD at a high learning rate and diverges sporadically — say one night
# in ten the loss explodes mid-run and the rollout is aborted, at ~S$1,200
# per aborted run (wasted GPU-hours plus an on-call page).
#
# Version 2 switches to AdamW with a short linear warmup and cosine
# decay. Warmup keeps the first updates small while Adam's moment
# estimates are still noisy, which is the usual cause of early blow-ups.
#
# BUSINESS IMPACT (illustrative assumptions, not measured figures):
#   - 1 abort in 10 nights = ~3 aborts/month x S$1,200 = ~S$3,600/month
#     (~S$43K/year) of wasted retraining, if warmup removes them all
#   - Fewer 4am pages for the on-call engineer
#   - More importantly, a model that refreshes every night on schedule
#
# LIMITATION: Cosine + warmup is a solid default, but it is not the
# best schedule for every task. Contrastive learning loves OneCycle;
# language-model pretraining loves inverse-sqrt decay. The right
# question is "what's the expected curvature of my loss surface?".

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Compared SGD, SGD+momentum, Adam, and AdamW on identical training
      data and epoch budget
  [x] Built a linear-warmup + cosine-annealing scheduler from LambdaLR
  [x] Trained a CNN for 10 epochs tracking train/val loss and LR together
  [x] Plotted the optimiser grid and the schedule trajectory
  [x] Costed an (illustrative) fraud-model retraining scenario where
      warmup removes early training divergence

  KEY INSIGHT: AdamW + warmup + cosine is the modern default because it
  is the safest thing you can pick without per-task tuning. Start there,
  and only change when the loss curve tells you something's wrong.

  Next: 05_regularisation_training.py — with a trained model, how do
  you keep it from overfitting, and how do you ship it?
"""
)
