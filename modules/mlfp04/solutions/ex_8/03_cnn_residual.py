# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 8.3: CNN with Residual Connections
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a small CNN with batch-norm and a residual (skip) block
#   - Verify a ResBlock's input/output shapes match (the skip is valid)
#   - Understand why residuals prevent vanishing gradients in depth
#   - Count parameters and interpret the model-size / capacity trade-off
#
# PREREQUISITES: 02_activations_init.py
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — residual connections as gradient highways
#   2. Build — TriageCNN with one ResBlock per stage
#   3. Train — one short fit on the Singapore triage imaging data
#   4. Visualise — loss curves, test AUC per class, and learned feature maps
#   5. Apply — hospital chest-film triage: why ResBlocks matter in imaging
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from plotly.subplots import make_subplots

from shared.mlfp04.ex_8 import (
    OUTPUT_DIR,
    ResBlock,
    SG_HOSPITAL_CLASSES,
    TriageCNN,
    build_sg_loaders,
    count_params,
    device,
    eval_cnn,
    eval_cnn_auc,
    train_cnn_one_epoch,
    viz,
)

print("\n" + "=" * 70)
print("  Residual CNNs — Why Depth Needed a Skip Connection")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════
# THEORY — The gradient highway
# ════════════════════════════════════════════════════════════════════════
# When you stack convolutional layers, each one multiplies its gradient
# contribution by a Jacobian. Stack 50 of them and the chain-rule product
# either vanishes (<1 factors compound to zero) or explodes (>1 factors
# compound to infinity). This is why "plain" stacks beyond 20 layers
# trained worse than shallower networks — the signal never reached the
# early layers.
#
# A residual connection adds the block's input to its output:
#   y = F(x) + x
# The gradient of y with respect to x is (1 + dF/dx), so the "+1" creates
# an identity path that carries gradients through untouched. This is the
# single architectural trick that unlocked ResNet-50, ResNet-152, and
# almost every modern CNN and transformer.

train_loader, test_loader, X_test_np, y_test_np = build_sg_loaders()
print(f"Device: {device}")
print(f"Classes: {SG_HOSPITAL_CLASSES}")
print("(Synthetic images: each finding is a drawn shape — see shared/mlfp04/ex_8.py)")
print(f"Train batches: {len(train_loader)}   Test batches: {len(test_loader)}")
print(f"Test prevalence per class: {np.round(y_test_np.mean(axis=0), 3).tolist()}")

# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the TriageCNN and verify the ResBlock shape
# ════════════════════════════════════════════════════════════════════════
model = TriageCNN(n_classes=len(SG_HOSPITAL_CLASSES), dropout_rate=0.3).to(device)

with torch.no_grad():
    dummy = torch.zeros(1, 32, 16, 16)
    probe = ResBlock(32)
    assert probe(dummy).shape == dummy.shape, "ResBlock must preserve shape"

total_params, trainable_params = count_params(model)
print("\n--- Model ---")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ── Checkpoint A ───────────────────────────────────────────────────────
assert (
    total_params > 50_000
), f"Task 2: TriageCNN should have a substantial parameter count, got {total_params}"

# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN for five quick epochs
# ════════════════════════════════════════════════════════════════════════
print("\n--- Training (5 epochs, AdamW) ---")
optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

train_losses: list[float] = []
val_losses: list[float] = []
for epoch in range(5):
    loss, _ = train_cnn_one_epoch(model, train_loader, optimiser, criterion)
    val = eval_cnn(model, test_loader, criterion)
    train_losses.append(loss)
    val_losses.append(val)
    print(f"  Epoch {epoch + 1}/5: train={loss:.4f}, val={val:.4f}")

# ── Checkpoint B ───────────────────────────────────────────────────────
assert (
    train_losses[-1] < train_losses[0] + 1e-3
), "Task 3: training loss should not get worse over 5 epochs"
print("\n[ok] Checkpoint passed — TriageCNN trains without gradient collapse\n")

# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE loss curves
# ════════════════════════════════════════════════════════════════════════
fig = viz.training_history(
    {"Train BCE": train_losses, "Val BCE": val_losses}, x_label="Epoch"
)
fig.update_layout(title="TriageCNN — Residual Stack, 5 Epochs")
viz_path = OUTPUT_DIR / "03_resnet_curves.html"
fig.write_html(viz_path)
print(f"[viz] Loss curves: {viz_path}")

# Loss alone is hard to read, so measure ranking quality per class.
aucs = eval_cnn_auc(model, X_test_np, y_test_np)
print("\nTest ROC AUC per class (0.5 = chance, 1.0 = perfect):")
for name in SG_HOSPITAL_CLASSES:
    print(f"  {name:<12} {aucs[name]:.3f}")
print(f"  {'macro':<12} {aucs['macro']:.3f}")

# ── Learned feature maps: what the first conv layer responds to ───────
# Pick a test image with at least two findings and show the input next to
# the first eight channels after Conv -> BatchNorm -> ReLU.
multi = np.where(y_test_np[:, :4].sum(axis=1) >= 2)[0]
idx = int(multi[0]) if len(multi) else 0
first_stage = model.features[:3]  # Conv2d -> BatchNorm2d -> ReLU
model.eval()
with torch.no_grad():
    fmap = first_stage(torch.from_numpy(X_test_np[idx : idx + 1]).to(device))
fmap = fmap[0].cpu().numpy()
present = [n for k, n in enumerate(SG_HOSPITAL_CLASSES) if y_test_np[idx, k] == 1]
titles = ["input: " + ", ".join(present)] + [f"channel {c}" for c in range(8)]
fig_maps = make_subplots(rows=3, cols=3, subplot_titles=titles)
panels = [X_test_np[idx, 0]] + [fmap[c] for c in range(8)]
for k, panel in enumerate(panels):
    fig_maps.add_trace(
        go.Heatmap(z=panel[::-1], colorscale="Gray", showscale=False),
        row=k // 3 + 1,
        col=k % 3 + 1,
    )
fig_maps.update_xaxes(showticklabels=False)
fig_maps.update_yaxes(showticklabels=False)
fig_maps.update_layout(title="First-layer feature maps after training", height=750)
maps_path = OUTPUT_DIR / "03_feature_maps.html"
fig_maps.write_html(maps_path)
print(f"[viz] Feature maps: {maps_path}")

# INTERPRETATION (computed from this run):
gap = val_losses[-1] - train_losses[-1]
print(
    f"\nFinal train BCE {train_losses[-1]:.4f} vs val BCE {val_losses[-1]:.4f} "
    f"(gap {gap:+.4f}); macro AUC {aucs['macro']:.3f}."
)
if aucs["macro"] > 0.8:
    print(
        "  -> The CNN has learned to detect the drawn findings. Open the "
        "feature-map plot and look for channels that respond to the band, "
        "blob, streak or dot in the input panel."
    )
else:
    print(
        "  -> AUC is still modest after 5 epochs; the small findings (streak, "
        "dot) usually take longer to learn than the large ones."
    )
weakest = min(SG_HOSPITAL_CLASSES, key=lambda n: aucs[n])
print(f"  Hardest class this run: {weakest} (AUC {aucs[weakest]:.3f}).")

# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore Hospital Chest-Film Triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (hypothetical): a Singapore public hospital triages ~1,200 chest
# X-rays per day and wants a model that tags urgent films for the
# radiologist queue. A deep PLAIN CNN (no skips) is the kind of network
# that shows the classic vanishing-gradient symptom: training loss
# plateaus early and never recovers. Adding a residual block per stage —
# what you just built — is the standard fix that lets deep stacks train.
#
# BUSINESS IMPACT (illustrative assumptions, not measured figures):
# suppose earlier flagging lets the team act sooner on ~18 time-critical
# cases a month, and each avoided ICU admission saves ~S$42,000. That is
# 18 x S$42K x 12 = ~S$9M/year, against tens of thousands of dollars a
# year for GPU inference. Any real claim needs a clinical validation
# study on real films — this exercise's synthetic AUC is not evidence of
# clinical performance.
#
# LIMITATION: Residuals only help when the plain network was too deep
# to train. A 5-layer CNN without residuals is fine; a 50-layer one is
# not. The trick is knowing when you've entered the regime where depth
# has started hurting you (telltale: training loss plateaus early).

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a multi-stage CNN (Conv->BN->ReLU->Pool->ResBlock) from the
      shared TriageCNN factory
  [x] Verified the ResBlock's skip connection preserves dimensions
  [x] Trained for five epochs without a gradient collapse
  [x] Counted parameters, plotted train/val loss, measured per-class AUC
  [x] Inspected first-layer feature maps to see what the filters detect
  [x] Sized an (illustrative) hospital-triage scenario for residual CNNs

  KEY INSIGHT: Residual connections are cheap. One tensor addition per
  block. The payoff is that depth stops hurting you, which is what
  enabled every modern CNN architecture since ResNet-50.

  Next: 04_optimisers_schedulers.py — now that the network trains, how
  do you make it train faster and more reliably?
"""
)
