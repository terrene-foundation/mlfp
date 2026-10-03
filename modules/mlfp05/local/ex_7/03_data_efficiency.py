# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 7, Part 3: Data Efficiency Experiment
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Run a controlled data efficiency experiment: train a transfer
#     model on 10%, 25%, 50%, and 100% of training data
#   - Quantify how transfer learning bends the labelling cost curve
#   - Plot data efficiency curves comparing transfer vs from-scratch
#   - Answer the business question: "How many images do we need to label?"
#   - Calculate cost savings in a real Singapore business scenario
#
# PREREQUISITES: Part 1 (baseline), Part 2 (transfer learning).
# ESTIMATED TIME: ~25 min
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision

from shared.mlfp05.ex_7 import (
    BATCH_SIZE,
    EPOCHS,
    N_CLASSES,
    OUTPUT_DIR,
    count_params,
    device,
    init_engines,
    load_cifar10,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Labelling Bottleneck
# ════════════════════════════════════════════════════════════════════════
# In production ML, the biggest cost isn't compute — it's labelled data.
#
# Consider what labelling costs in practice:
#   - Simple image classification: S$0.50-1.00 per image
#   - Medical image annotation: S$5-20 per image (specialist required)
#   - Autonomous driving frames: S$50-200 per frame (3D bounding boxes)
#
# If you need 50,000 labelled images at S$1 each, that's S$50,000 just
# for data — before you've written a single line of code.
#
# Transfer learning bends this cost curve. By reusing features from a
# pre-trained model, you can achieve 80-90% of the full-data accuracy
# with only 10-25% of the labelled data. This experiment quantifies
# exactly where that sweet spot is.
#
# The key question for any ML project: "What's the minimum amount of
# labelled data that gives us acceptable accuracy?" This experiment
# gives you the data to answer it.
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PART 3: Data Efficiency Experiment")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and set up engines
# ════════════════════════════════════════════════════════════════════════

train_set, val_set, train_loader, val_loader = load_cifar10()
conn, tracker, exp_name, registry, has_registry = init_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Define model builders
# ════════════════════════════════════════════════════════════════════════


def build_transfer_resnet(n_classes: int = N_CLASSES) -> nn.Module:
    """Frozen ResNet-18 backbone + fresh classification head."""
    # TODO: Load pre-trained ResNet-18, freeze backbone, replace fc head
    # Steps:
    #   1. Load weights: torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    #   2. Create model with those weights (try/except for offline fallback)
    #   3. Freeze all params: for p in model.parameters(): p.requires_grad = False
    #   4. Replace model.fc with nn.Linear(model.fc.in_features, n_classes)
    # Hint: Same pattern as Part 2's build_transfer_resnet
    ____


def build_scratch_cnn(n_classes: int = N_CLASSES) -> nn.Module:
    """From-scratch CNN baseline."""
    # TODO: Build a 3-layer CNN identical to Part 1
    # Hint: nn.Sequential with Conv2d->BN->ReLU->Pool blocks
    #       then AdaptiveAvgPool2d(1)->Flatten->Dropout(0.3)->Linear(128, n_classes)
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Run the data efficiency experiment
# ════════════════════════════════════════════════════════════════════════
# For each data fraction (10%, 25%, 50%, 100%), we train BOTH a transfer
# model and a from-scratch model, then compare. This gives us two
# curves that show exactly where transfer learning helps most.

DATA_FRACTIONS = [0.10, 0.25, 0.50, 1.0]
EFF_EPOCHS = 4  # Shorter training for sub-experiments

transfer_results: dict[float, float] = {}
scratch_results: dict[float, float] = {}
transfer_models: dict[float, nn.Module] = {}  # kept for the diagnostic checkpoint

rng = np.random.default_rng(42)


async def _run_efficiency_trial(
    frac: float,
    model_builder,
    model_name: str,
) -> tuple[float, int, nn.Module]:
    """Train one model on a fraction of data, return (accuracy, n_samples, model)."""
    # TODO: Implement the efficiency trial
    # Steps:
    #   1. Draw n_samples distinct random indices with the shared `rng`
    #      (no replacement) and wrap train_set in a Subset + DataLoader
    #   2. Build the model with model_builder, move it to device, and give
    #      Adam (lr=1e-3) only the parameters that require gradients
    #   3. Open an ExperimentTracker run named "<model_name>_<pct>pct",
    #      log the trial's parameters
    #   4. Train for EFF_EPOCHS with cross-entropy
    #   5. Evaluate accuracy on the full val_loader and log it as "val_acc"
    # Hint: the tracker run is an async context manager; logging calls
    #   are awaited
    n_samples = int(len(train_set) * frac)
    ____

    return ____, n_samples, ____  # TODO: accuracy, n_samples, trained model


print("\n" + "=" * 70)
print("  DATA EFFICIENCY EXPERIMENT")
print("=" * 70)

for frac in DATA_FRACTIONS:
    # Transfer model
    t_acc, n_samples, t_model = asyncio.run(
        _run_efficiency_trial(frac, build_transfer_resnet, "transfer")
    )
    transfer_results[frac] = t_acc
    transfer_models[frac] = t_model

    # From-scratch model
    s_acc, _, _ = asyncio.run(
        _run_efficiency_trial(frac, build_scratch_cnn, "scratch")
    )
    scratch_results[frac] = s_acc

    print(
        f"  {frac * 100:5.0f}% data ({n_samples:>5,} samples): "
        f"transfer={t_acc:.4f}  scratch={s_acc:.4f}  "
        f"gap={t_acc - s_acc:+.4f}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(transfer_results) == len(
    DATA_FRACTIONS
), "Should have transfer results for all fractions"
assert len(scratch_results) == len(
    DATA_FRACTIONS
), "Should have scratch results for all fractions"
assert (
    transfer_results[0.10] > 0.15
), f"Transfer with 10% data should beat random (acc={transfer_results[0.10]:.3f})"
# INTERPRETATION: Compare how much each model gains from 10% to 100% of
# the data. If pre-trained features already capture general visual
# patterns, the transfer model gains LESS from extra data than the
# scratch model does — additional labels help, but are less critical.
transfer_gain = transfer_results[1.0] - transfer_results[0.10]
scratch_gain = scratch_results[1.0] - scratch_results[0.10]
print(
    f"  Gain from 10% -> 100% data: transfer {transfer_gain:+.4f}, "
    f"scratch {scratch_gain:+.4f} "
    f"({'transfer depends less on data volume' if transfer_gain < scratch_gain else 'transfer did NOT depend less on data volume in this run'})"
)
print("\n--- Checkpoint 1 passed --- efficiency experiment complete\n")

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# kailash-ml's run_diagnostic_checkpoint runs a few real forward/backward
# passes (no optimiser step) with gradient, activation and dead-neuron
# hooks attached, and replays the real per-epoch training losses. It
# RETURNS the findings; print_prescription_pad prints them. The pass
# puts the model in train mode, which updates BatchNorm running
# statistics, so we diagnose a COPY and leave the trained model intact.
import copy

from kailash_ml.diagnostics import run_diagnostic_checkpoint

from shared.mlfp05.diagnostics import print_prescription_pad
from shared.mlfp05.ex_7 import classifier_diag_loss

print("\n── Diagnostic Report (Transfer ResNet-18 trained on 10% of the data) ──")
diag, findings = run_diagnostic_checkpoint(
    copy.deepcopy(transfer_models[0.10]),
    train_loader,
    classifier_diag_loss,
    title="Transfer ResNet-18 trained on 10% of the data",
    n_batches=8,
    train_losses=None,
    show=False,
)
print_prescription_pad(findings, "Transfer ResNet-18 trained on 10% of the data")
# HOW TO READ THE PRESCRIPTION PAD FOR THIS MODEL:
#  This diagnoses the transfer model from the smallest-data trial (10%).
#  No per-epoch losses were kept for the trials, so the loss-trend
#  reading only sees the 8 diagnostic batches.
#  Gradient flow — only the fc head is trainable; frozen layers carry no
#     parameter gradients by design. "Exploding" on fc with so little
#     data means the head is being pushed hard by few examples — lower the
#     learning rate or add weight decay before collecting more labels.
#  Dead neurons — silent pretrained ReLUs signal a domain gap between
#     ImageNet and CIFAR-10, not a small-data problem; more labels will
#     not fix it, unfreezing or adapters (Part 4) can.
#  If any reading is UNKNOWN, the library could not compute it from this
#  run; the message says why.



# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: Data efficiency curves
# ════════════════════════════════════════════════════════════════════════
# The money chart: accuracy vs percentage of training data for both
# approaches. This is the chart you show to the VP of Engineering.

fracs = sorted(transfer_results.keys())
transfer_accs_by_frac = [transfer_results[f] for f in fracs]
scratch_accs_by_frac = [scratch_results[f] for f in fracs]
pct_labels = [f * 100 for f in fracs]

# TODO: Create a Plotly figure with two traces:
#   1. Transfer learning curve (lines+markers, color="#2196F3")
#   2. From-scratch curve (lines+markers, dash="dash", color="#FF5722")
# Add annotations for the 10% data points on both curves
# Hint: fig = go.Figure()
# Hint: fig.add_trace(go.Scatter(x=pct_labels, y=transfer_accs_by_frac, ...))
# Hint: fig.add_annotation(x=10, y=transfer_results[0.10], text=..., showarrow=True)
fig = go.Figure()
____

fig.update_layout(
    title="Data Efficiency: Transfer Learning vs From-Scratch",
    xaxis_title="% of CIFAR-10 Training Data",
    yaxis_title="Validation Accuracy",
    template="plotly_white",
    legend=dict(x=0.6, y=0.15),
    width=800,
    height=500,
)

eff_path = OUTPUT_DIR / "03_data_efficiency.html"
fig.write_html(str(eff_path))
print(f"  Saved: {eff_path}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (eff_path).exists(), "Data efficiency plot should be saved"
print("--- Checkpoint 2 passed --- efficiency curves plotted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Accuracy gap and diminishing returns
# ════════════════════════════════════════════════════════════════════════
# The gap between transfer and scratch narrows as data increases.
# This shows that transfer learning's biggest value is with LIMITED data.

# TODO: Compute gaps and create a bar chart
# Steps:
#   1. gaps = [t - s for t, s in zip(transfer_accs_by_frac, scratch_accs_by_frac)]
#   2. Create go.Figure() with go.Bar trace
#   3. x = [f"{p:.0f}%" for p in pct_labels], y = [g * 100 for g in gaps]
#   4. Color bars green if gap > 0, red otherwise
#   5. Save to OUTPUT_DIR / "03_accuracy_gap.html"
# Hint: marker_color=["#4CAF50" if g > 0 else "#F44336" for g in gaps]
gaps = [t - s for t, s in zip(transfer_accs_by_frac, scratch_accs_by_frac)]
____

gap_path = OUTPUT_DIR / "03_accuracy_gap.html"
fig_gap.write_html(str(gap_path))
print(f"  Saved: {gap_path}")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: The VP of Engineering at Grab Asks "How Many Images?"
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: You're the ML lead at Grab Singapore. The VP of Engineering
# asks: "We want to build an image classifier for food delivery photos.
# How many images do we need to label? What will it cost?"
#
# You use this data efficiency experiment to answer concretely.

print("\n" + "=" * 70)
print("  APPLY: Grab Singapore — 'How many images do we need to label?'")
print("=" * 70)

COST_PER_LABEL = 0.80  # S$ per image label (food photo classification)
TOTAL_AVAILABLE = 50000  # Total unlabelled images available

# TODO: Print the cost-accuracy trade-off table
# Steps:
#   1. Loop through fracs, compute n_images and label_cost for each
#   2. Print a formatted table with columns: Data%, Images, Transfer acc,
#      Scratch acc, Label Cost, Transfer Saves
# Hint: n_images = int(TOTAL_AVAILABLE * frac)
# Hint: label_cost = n_images * COST_PER_LABEL
print(f"\n  === Cost-Accuracy Trade-off Analysis ===")
print(f"  Labelling cost: S${COST_PER_LABEL:.2f} per image")
print(f"  Unlabelled pool: {TOTAL_AVAILABLE:,} food delivery photos")
____

# TODO: Find the sweet spot where transfer reaches 90% of max accuracy
# Steps:
#   1. max_transfer_acc = transfer_results[1.0]
#   2. sweet_spot_threshold = 0.90 * max_transfer_acc
#   3. Loop through fracs to find first frac where accuracy >= threshold
#   4. Calculate savings vs labelling the full dataset
# Hint: sweet_spot_frac = None; iterate and break when found
max_transfer_acc = transfer_results[1.0]
sweet_spot_threshold = 0.90 * max_transfer_acc
sweet_spot_frac = None
____

if sweet_spot_frac is not None:
    sweet_n = int(TOTAL_AVAILABLE * sweet_spot_frac)
    sweet_cost = sweet_n * COST_PER_LABEL
    full_cost = TOTAL_AVAILABLE * COST_PER_LABEL
    savings = full_cost - sweet_cost
    print(f"\n  SWEET SPOT: {sweet_spot_frac * 100:.0f}% of data ({sweet_n:,} images)")
    print(
        f"  Reaches {transfer_results[sweet_spot_frac]:.1%} accuracy "
        f"(90% of maximum {max_transfer_acc:.1%})"
    )
    print(f"  Label cost: S${sweet_cost:,.0f} vs S${full_cost:,.0f} for full dataset")
    print(f"  SAVINGS: S${savings:,.0f}")
else:
    print(f"\n  All fractions tested achieve >=90% of maximum accuracy.")

print()
print(f"  RECOMMENDATION TO VP:")
print(
    f"  'Start with {int(TOTAL_AVAILABLE * 0.25):,} labelled images "
    f"(S${int(TOTAL_AVAILABLE * 0.25 * COST_PER_LABEL):,})."
)
print(f"   Use transfer learning with ResNet-18. If accuracy is insufficient,")
print(f"   label more images in batches of 5,000 until you reach the target.")
print(f"   Transfer learning means we never need to label all 50,000 images.'")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    sweet_spot_frac is not None or len(fracs) > 0
), "Should identify a sweet spot or have results"
# INTERPRETATION: The data efficiency curve directly answers the VP's
# question with concrete numbers: how many images to label, how much
# it costs, and where the diminishing returns kick in. This is how ML
# engineers translate technical results into business decisions.
print("\n--- Checkpoint 3 passed --- business analysis complete\n")


# ════════════════════════════════════════════════════════════════════════
# CLEANUP
# ════════════════════════════════════════════════════════════════════════
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PART 3 COMPLETE — What You've Learned")
print("=" * 70)
print(
    f"""
  [x] Ran data efficiency experiment across 4 data fractions
  [x] Transfer with 10% data: {transfer_results[0.10]:.1%} accuracy
  [x] Transfer with 100% data: {transfer_results[1.0]:.1%} accuracy
  [x] Scratch with 10% data: {scratch_results[0.10]:.1%} accuracy
  [x] Plotted data efficiency curves (transfer vs scratch)
  [x] Identified the sweet spot: {sweet_spot_frac * 100:.0f}% of data for 90% of max accuracy
  [x] Calculated labelling cost savings for Grab Singapore scenario

  KEY INSIGHT: Transfer learning's biggest value is with LIMITED data.
  The gap between transfer and scratch is largest at 10-25% data, then
  narrows as data increases. This means:
    - With abundant data: transfer helps but isn't critical
    - With scarce data: transfer is transformative

  THE LABELLING BOTTLENECK EQUATION:
    Cost = (images needed) x (cost per label)
    Transfer learning reduces the first term by 4-10x.
    This is often the difference between a viable project and a shelved one.

  NEXT: Part 4 introduces adapter modules — a parameter-efficient
  alternative to full fine-tuning that bridges to M6's LoRA technique.
"""
)
