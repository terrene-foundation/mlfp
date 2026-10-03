# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 2.4 — Hyperparameter Study: LR and Data Augmentation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Explain WHY hyperparameters matter and the law of diminishing
#     returns in terms a non-technical manager can understand
#   - Run a systematic learning rate sweep (3 values) tracked with
#     ExperimentTracker
#   - Compare data augmentation strategies (none vs horizontal flip +
#     random crop) and measure their impact on generalisation
#   - Visualise the accuracy-vs-compute tradeoff (more epochs and more
#     augmentation have decreasing marginal returns)
#   - Answer the engineering manager's question: "How much compute
#     budget should we allocate?"
#
# PREREQUISITES: M5/ex_2/01_simple_cnn.py, 02_resnet_se.py, and
#   03_production_pipeline.py (CNN architectures, training, ONNX)
# ESTIMATED TIME: ~30 min
#
# DATASET: CIFAR-10 — same 50K training images. For the augmentation
#   comparison, we apply transforms on-the-fly during training.
#
# PHASES:
#   1. THEORY  — Why hyperparameters matter, diminishing returns
#   2. BUILD   — ResNetSE (same architecture, different hyperparameters)
#   3. TRAIN   — Learning rate sweep + augmentation comparison
#   4. VISUALISE — Accuracy-vs-cost tradeoff curves
#   5. APPLY   — "How much compute budget?" for an engineering manager
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

from shared.mlfp05.ex_2 import (
    ACCELERATOR,
    BATCH_SIZE,
    CIFAR_MEAN,
    CIFAR_STD,
    CLASS_NAMES,
    DATA_DIR,
    DEVICE,
    EPOCHS,
    N_CLASSES,
    PRECISION,
    LitCNN,
    count_parameters,
    create_visualizer,
    init_engines,
    load_cifar10,
    save_training_plots,
    train_model,
    train_model_async,
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why Hyperparameters Matter
# ════════════════════════════════════════════════════════════════════════
# ANALOGY FOR NON-TECHNICAL STAKEHOLDERS:
#
# Think of training a neural network like tuning a radio. The MODEL
# ARCHITECTURE is the radio itself — SimpleCNN vs ResNetSE. But the
# HYPERPARAMETERS are the dials you turn:
#
#   - LEARNING RATE: How far you turn the dial each time. Too far
#     (lr=0.1) and you overshoot the station. Too little (lr=0.00001)
#     and you spend an hour finding it. Just right (lr=0.001) and you
#     converge quickly.
#
#   - DATA AUGMENTATION: Imagine listening to the same song through
#     different speakers, at different volumes, with different EQ
#     settings. You learn to recognise the SONG, not the speaker.
#     Data augmentation (flipping, cropping, colour jitter) teaches
#     the model to recognise the OBJECT, not the specific photo angle.
#
#   - EPOCHS (training duration): How many times you listen to your
#     training playlist. First listen: you catch the chorus. Fifth
#     listen: you know the verses. Twentieth listen: you're not
#     learning anything new, just wasting electricity.
#
# THE LAW OF DIMINISHING RETURNS:
#   Going from 5 to 10 epochs might improve accuracy by 5%.
#   Going from 10 to 20 epochs might improve by 2%.
#   Going from 20 to 40 epochs might improve by 0.5%.
#   Each doubling of compute buys less and less improvement.
#
#   This is why the engineering manager's question — "how much compute
#   budget?" — has a concrete, data-driven answer: run the sweep, find
#   the knee of the curve, and spend your budget THERE.

print("=" * 70)
print("  PHASE 1 — THEORY: Hyperparameters and Diminishing Returns")
print("=" * 70)
print(
    """
  Three dials to tune:
    1. Learning rate: how big each update step is
    2. Data augmentation: synthetic variety to prevent memorisation
    3. Training duration: how many passes through the data

  The law of diminishing returns means there's an OPTIMAL budget --
  not minimum, not maximum, but the point where each additional dollar
  buys the least improvement.
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: ResNetSE for Hyperparameter Experiments
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: ResNetSE (same architecture, different settings)")
print("=" * 70)


class ResBlock(nn.Module):
    """Residual block for hyperparameter experiments."""

    def __init__(self, channels: int):
        super().__init__()
        # TODO: Same residual block as 02 — two shape-preserving 3x3 convs,
        #   each followed by its own batch norm
        self.conv1 = ____
        self.bn1 = ____
        self.conv2 = ____
        self.bn2 = ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: identity + residual forward pass
        identity = x
        out = ____
        out = ____
        return ____


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for hyperparameter experiments."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        # TODO: SE MLP — bottleneck channels -> hidden -> channels (same as 02)
        self.fc = nn.Sequential(
            ____,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # TODO: squeeze -> excite -> scale
        s = ____
        w = ____
        return ____


class ResNetSE(nn.Module):
    """ResNet+SE for hyperparameter experiments."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        # TODO: stem, block1, se1, block2, pool, fc (same as 02 and 03)
        self.stem = nn.Sequential(
            ____,
        )
        self.block1 = ____
        self.se1 = ____
        self.block2 = ____
        self.pool = ____
        self.fc = ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: stem -> block1 -> se1 -> block2 -> pool -> flatten -> fc
        ____


param_count = count_parameters(ResNetSE())
print(f"  ResNetSE: {param_count:,} parameters")
print("  Same architecture across all experiments -- only hyperparameters change")


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: Learning Rate Sweep + Augmentation Comparison
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 3 — TRAIN: Systematic Hyperparameter Experiments")
print("=" * 70)

# Load data
X_train, y_train, X_val, y_val, train_loader, val_loader = load_cifar10()
conn, tracker, exp_name, registry, has_registry = init_engines()

# ──────────────────────────────────────────────────────────────────────
# EXPERIMENT A: Learning Rate Sweep
# ──────────────────────────────────────────────────────────────────────
LR_SWEEP = [5e-4, 1e-3, 3e-3]
HP_EPOCHS = 6
lr_results: dict[str, dict] = {}

print(
    f"\n  EXPERIMENT A: Learning Rate Sweep ({len(LR_SWEEP)} values, {HP_EPOCHS} epochs each)"
)
print("  " + "-" * 60)


# TODO: Implement the async LR sweep training function
#   1. Create fresh ResNetSE for each LR
#   2. Wrap in LitCNN(model, lr=lr)
#   3. Create pl.Trainer with HP_EPOCHS, ACCELERATOR, PRECISION, no progress bar
#   4. Use tracker.track() async context to log params and metrics
#   5. Return (train_losses, val_accs)
async def train_lr_sweep_async(lr: float) -> tuple[list[float], list[float]]:
    """Run one LR sweep trial, logged under its own tracker run."""
    hp_model = ResNetSE()
    lit = LitCNN(hp_model, lr=lr)
    trainer = pl.Trainer(
        max_epochs=HP_EPOCHS,
        accelerator=ACCELERATOR,
        precision=PRECISION,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )

    async with tracker.track(experiment=exp_name, run_name=f"hp_sweep_lr_{lr}") as run:
        await run.log_params(
            {
                "architecture": "ResNetSE",
                "lr": str(lr),
                "epochs": str(HP_EPOCHS),
                "sweep_type": "learning_rate",
                "augmentation": "none",
            }
        )

        # TODO: Fit the Lightning trainer on the module with the train and val
        #   loaders, then log every epoch's value from lit.train_losses and
        #   lit.val_accs with run.log_metric (async), as "train_loss" and
        #   "val_accuracy", using 1-based epoch numbers as the step
        ____

        for epoch_idx, loss in enumerate(lit.train_losses):
            ____
        for epoch_idx, acc in enumerate(lit.val_accs):
            ____

        await run.log_metrics(
            {
                "final_train_loss": lit.train_losses[-1],
                "final_val_accuracy": lit.val_accs[-1],
            }
        )

    return lit.train_losses, lit.val_accs


for lr in LR_SWEEP:
    name = f"resnet_se_lr{lr}"
    print(f"\n  Training ResNetSE with lr={lr}...")
    t0 = time.perf_counter()
    sweep_losses, sweep_accs = asyncio.run(train_lr_sweep_async(lr))
    elapsed = time.perf_counter() - t0
    lr_results[name] = {
        "lr": lr,
        "losses": sweep_losses,
        "accs": sweep_accs,
        "time_sec": elapsed,
    }
    print(
        f"    lr={lr}: final_loss={sweep_losses[-1]:.4f}, "
        f"val_acc={sweep_accs[-1]:.3f}, time={elapsed:.1f}s"
    )

# Print LR comparison table
print(f"\n  {'Learning Rate':>15} {'Final Loss':>12} {'Val Acc':>10} {'Time (s)':>10}")
print("  " + "-" * 50)
for name, result in lr_results.items():
    print(
        f"  {result['lr']:>15.4f} {result['losses'][-1]:>12.4f} "
        f"{result['accs'][-1]:>9.3f} {result['time_sec']:>9.1f}"
    )

best_lr_config = max(lr_results.items(), key=lambda x: x[1]["accs"][-1])
best_lr = best_lr_config[1]["lr"]
best_lr_acc = best_lr_config[1]["accs"][-1]
print(f"\n  Best LR: {best_lr} (val_acc={best_lr_acc:.3f})")

# ── Checkpoint 1: LR sweep complete ─────────────────────────────────
assert len(lr_results) == len(
    LR_SWEEP
), f"Should have results for all {len(LR_SWEEP)} learning rates"
for name, result in lr_results.items():
    assert result["accs"][-1] > 0.2, (
        f"{name} val_acc={result['accs'][-1]:.3f} is too low -- even suboptimal "
        "LR should learn basic features from 50K images"
    )
print("\n--- Checkpoint 1 passed --- learning rate sweep complete\n")


# ──────────────────────────────────────────────────────────────────────
# EXPERIMENT B: Data Augmentation Comparison
# ──────────────────────────────────────────────────────────────────────
# Data augmentation creates synthetic training variety:
#   - Horizontal flip: a cat facing left is still a cat facing right
#   - Random crop: the object can appear at any position
#   - These teach the model to recognise the OBJECT, not the specific
#     photo composition
#
# We compare: (1) no augmentation vs (2) flip + random crop

print("  EXPERIMENT B: Data Augmentation Comparison")
print("  " + "-" * 60)

DATA_DIR.mkdir(parents=True, exist_ok=True)

# TODO: Create augmented and non-augmented transforms (torchvision T.*)
#   augmented_transform: random horizontal flip (probability 0.5), then a
#     random 32x32 crop after padding 4 pixels on each side, then conversion
#     to a tensor
#   no_aug_transform: tensor conversion only
augmented_transform = T.Compose(
    [
        ____,
    ]
)

no_aug_transform = T.Compose(
    [
        ____,
    ]
)

# Load raw CIFAR-10 with transforms
train_set_aug = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=augmented_transform,
)
train_set_noaug = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=no_aug_transform,
)


def normalise_batch(batch_imgs: torch.Tensor) -> torch.Tensor:
    """Apply CIFAR-10 normalisation to a batch."""
    return (batch_imgs - CIFAR_MEAN) / CIFAR_STD


class NormalisingDataset(torch.utils.data.Dataset):
    """Wraps a torchvision dataset to add CIFAR-10 normalisation."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = (img - CIFAR_MEAN.squeeze(0)) / CIFAR_STD.squeeze(0)
        return img, label


aug_loader = DataLoader(
    NormalisingDataset(train_set_aug),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
noaug_loader = DataLoader(
    NormalisingDataset(train_set_noaug),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _ce_loss(m, batch):
    """Cross-entropy on one (images, labels) batch, on the model's device."""
    xb, yb = batch
    dev = next(m.parameters()).device
    return F.cross_entropy(m(xb.to(dev)), yb.to(dev))


# TODO: Train with best LR from sweep, comparing augmented vs non-augmented
#   For each (aug_name, loader): a fresh ResNetSE, trained with train_model
#   on that loader (validated on val_loader) at lr=best_lr for HP_EPOCHS;
#   run name f"aug_{aug_name}"
aug_results: dict[str, dict] = {}

for aug_name, loader in [("no_augmentation", noaug_loader), ("flip_crop", aug_loader)]:
    print(f"\n  Training ResNetSE with {aug_name} (lr={best_lr})...")
    model = ____
    t0 = time.perf_counter()
    losses, accs = ____
    elapsed = time.perf_counter() - t0
    aug_results[aug_name] = {
        "losses": losses,
        "accs": accs,
        "time_sec": elapsed,
        "model": model,
    }
    print(
        f"    {aug_name}: final_loss={losses[-1]:.4f}, "
        f"val_acc={accs[-1]:.3f}, time={elapsed:.1f}s"
    )
    # Quick diagnostic check per augmentation setting — does a weaker
    # config fail for OPTIMISATION reasons (dead units, gradient trouble)
    # or only in accuracy? Probes use this config's own training loader.
    print(f"  ── Diagnostic Report ({aug_name}) ──")
    diag, findings = run_diagnostic_checkpoint(
        model,
        loader,
        _ce_loss,
        title=f"ResNetSE ({aug_name})",
        train_losses=losses,
        show=False,
    )
    print_prescription_pad(findings, f"ResNetSE ({aug_name})")
    aug_results[aug_name]["findings"] = findings
    # READING ACROSS CONFIGS (key: see ex_1/01_standard_ae.py): compare
    # the two pads side by side. Augmentation changes the DATA, not the
    # network, so gradient and dead-unit readings usually look alike;
    # the difference shows up in the train-loss trend and in validation
    # accuracy. If one config alone shows CRITICAL gradients or a
    # mostly-dead layer, its optimisation broke (often the learning
    # rate) — that is a different failure from merely lower accuracy.

# Print augmentation comparison
print(f"\n  {'Augmentation':>20} {'Final Loss':>12} {'Val Acc':>10} {'Time (s)':>10}")
print("  " + "-" * 55)
for name, result in aug_results.items():
    print(
        f"  {name:>20} {result['losses'][-1]:>12.4f} "
        f"{result['accs'][-1]:>9.3f} {result['time_sec']:>9.1f}"
    )

aug_improvement = (
    aug_results["flip_crop"]["accs"][-1] - aug_results["no_augmentation"]["accs"][-1]
)
print(f"\n  Augmentation improvement: {aug_improvement:+.3f} accuracy")

# ── Checkpoint 2: Augmentation comparison complete ───────────────────
assert len(aug_results) == 2, "Should have results for both augmentation settings"
print("\n--- Checkpoint 2 passed --- augmentation comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: Accuracy-vs-Cost Tradeoff Curves
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: The Accuracy-vs-Cost Tradeoff")
print("=" * 70)

viz = create_visualizer()

# TODO: Save training plots for LR sweep and augmentation comparison
#   Four save_training_plots calls; each metrics dict has one labelled curve
#   per run (build it from lr_results / aug_results with a comprehension).
#   Files: "ex_2_04_lr_sweep_loss.html", "ex_2_04_lr_sweep_acc.html",
#          "ex_2_04_augmentation_loss.html", "ex_2_04_augmentation_acc.html"
____
____
____
____

# TODO: Create the "diminishing returns" chart
#   Left subplot: accuracy vs epoch for all LR configs (line chart with markers)
#   Right subplot: marginal improvement per epoch (bar chart)
fig_roi, axes = plt.subplots(1, 2, figsize=(16, 6))
fig_roi.suptitle(
    "The Diminishing Returns of Training: When to Stop Spending",
    fontsize=14,
)

# Left: Accuracy vs epoch for all LR configs
for name, result in lr_results.items():
    epochs_x = list(range(1, len(result["accs"]) + 1))
    axes[0].plot(epochs_x, result["accs"], "o-", label=f"lr={result['lr']}")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title("Accuracy vs Training Duration")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: Marginal improvement per epoch
for name, result in lr_results.items():
    accs = result["accs"]
    marginal = [accs[0]] + [accs[i] - accs[i - 1] for i in range(1, len(accs))]
    epochs_x = list(range(1, len(marginal) + 1))
    axes[1].bar(
        [x + 0.2 * (list(lr_results.keys()).index(name) - 1) for x in epochs_x],
        marginal,
        width=0.2,
        alpha=0.7,
        label=f"lr={result['lr']}",
    )
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Marginal Accuracy Gain")
axes[1].set_title("Diminishing Returns: Accuracy Gained Per Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig("ex_2_04_diminishing_returns.png", dpi=150, bbox_inches="tight")
plt.close(fig_roi)
print("  Saved: ex_2_04_diminishing_returns.png")
print("  Left: accuracy climbs steeply then flattens (the 'knee')")
print("  Right: marginal gains shrink each epoch (the ROI curve)")

# TODO: Create cost-accuracy tradeoff scatter plot
#   Each point = one experiment, x = training time, y = accuracy
#   Add Pareto frontier connecting non-dominated points
fig_cost, ax = plt.subplots(1, 1, figsize=(10, 6))
fig_cost.suptitle("Compute Cost vs Accuracy: Where to Spend Your Budget", fontsize=14)

all_results = []
for name, result in lr_results.items():
    all_results.append(
        {
            "label": f"lr={result['lr']}",
            "time": result["time_sec"],
            "acc": result["accs"][-1],
            "colour": "steelblue",
        }
    )
for name, result in aug_results.items():
    all_results.append(
        {
            "label": name,
            "time": result["time_sec"],
            "acc": result["accs"][-1],
            "colour": "coral",
        }
    )

for r in all_results:
    ax.scatter(r["time"], r["acc"], s=100, c=r["colour"], zorder=5)
    ax.annotate(
        r["label"],
        (r["time"], r["acc"]),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=8,
    )

ax.set_xlabel("Training Time (seconds)")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Each Point = One Experiment Configuration")
ax.grid(True, alpha=0.3)

sorted_results = sorted(all_results, key=lambda r: r["time"])
pareto_time = [sorted_results[0]["time"]]
pareto_acc = [sorted_results[0]["acc"]]
for r in sorted_results[1:]:
    if r["acc"] > pareto_acc[-1]:
        pareto_time.append(r["time"])
        pareto_acc.append(r["acc"])
if len(pareto_time) > 1:
    ax.plot(pareto_time, pareto_acc, "k--", alpha=0.3, label="Pareto frontier")
    ax.legend()

plt.tight_layout()
plt.savefig("ex_2_04_cost_vs_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig_cost)
print("  Saved: ex_2_04_cost_vs_accuracy.png")

# ── Checkpoint 3: Visualisations generated ───────────────────────────
import os

assert os.path.exists(
    "ex_2_04_diminishing_returns.png"
), "Diminishing returns plot missing"
assert os.path.exists("ex_2_04_cost_vs_accuracy.png"), "Cost vs accuracy plot missing"
assert os.path.exists("ex_2_04_lr_sweep_loss.html"), "LR sweep loss plot missing"
print("\n--- Checkpoint 3 passed --- all visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: "How Much Compute Budget Should We Allocate?"
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The engineering manager at the Singapore e-commerce platform
# (from 01_simple_cnn.py and 03_production_pipeline.py) asks:
#
#   "We have a $5,000/month compute budget for ML training. The data
#    science team says they need $20,000/month for 'proper' training.
#    How do I know what's actually necessary?"
#
# This is the question hyperparameter studies answer with DATA, not
# opinions. The ExperimentTracker records tell the story.

print("=" * 70)
print("  PHASE 5 — APPLY: Compute Budget Allocation")
print("=" * 70)

# TODO: Calculate actual compute costs from your experiments
#   GPU_COST_PER_HOUR = 4.20  (illustrative single-V100-class instance, Singapore)
#   total_experiment_time: seconds summed over EVERY run — both the LR
#     sweep (lr_results) and the augmentation runs (aug_results) store
#     "time_sec"
GPU_COST_PER_HOUR = 4.20

total_experiment_time = ____
total_experiment_hours = total_experiment_time / 3600
total_experiment_cost = total_experiment_hours * GPU_COST_PER_HOUR

# Best and worst results
all_accs = [(name, r["accs"][-1]) for name, r in lr_results.items()]
all_accs += [(name, r["accs"][-1]) for name, r in aug_results.items()]
best_overall = max(all_accs, key=lambda x: x[1])
worst_overall = min(all_accs, key=lambda x: x[1])

# TODO: Project production costs
#   RETRAIN_EPOCHS = 20, RETRAIN_PER_MONTH = 4 (weekly); avg_time_per_epoch
#   is computed for you from the experiments above.
#   retrain_time_hours: wall-clock hours for one RETRAIN_EPOCHS retrain
#   retrain_cost_per_run, monthly_retrain_cost: priced at GPU_COST_PER_HOUR
#   hp_search_time_hours: a monthly search of 10 configurations x HP_EPOCHS
#   hp_search_cost: that search priced at GPU_COST_PER_HOUR
RETRAIN_EPOCHS = 20
RETRAIN_PER_MONTH = 4
avg_time_per_epoch = total_experiment_time / (
    len(lr_results) * HP_EPOCHS + len(aug_results) * HP_EPOCHS
)
retrain_time_hours = ____
retrain_cost_per_run = ____
monthly_retrain_cost = ____

hp_search_time_hours = ____
hp_search_cost = ____

print(
    f"""
  EXPERIMENT RESULTS SUMMARY:
  ═══════════════════════════════════════════════════════════

  Learning Rate Sweep ({len(LR_SWEEP)} configs x {HP_EPOCHS} epochs):"""
)

for name, result in lr_results.items():
    print(
        f"    lr={result['lr']}: acc={result['accs'][-1]:.3f} ({result['time_sec']:.0f}s)"
    )

print(
    f"""
  Augmentation Comparison (2 configs x {HP_EPOCHS} epochs):"""
)

for name, result in aug_results.items():
    print(f"    {name}: acc={result['accs'][-1]:.3f} ({result['time_sec']:.0f}s)")

print(
    f"""
  KEY FINDINGS:
    Best configuration:  {best_overall[0]} (acc={best_overall[1]:.3f})
    Worst configuration: {worst_overall[0]} (acc={worst_overall[1]:.3f})
    Spread:              {best_overall[1] - worst_overall[1]:.3f} accuracy
    Total experiment time: {total_experiment_time:.0f}s ({total_experiment_hours:.2f} hours)
    Total experiment cost:  ${total_experiment_cost:.2f}

  BUDGET RECOMMENDATION FOR THE ENGINEERING MANAGER:
  ═══════════════════════════════════════════════════════════

  MONTHLY COMPUTE BUDGET BREAKDOWN:

  1. Weekly model retraining (production):
     {RETRAIN_EPOCHS} epochs x {RETRAIN_PER_MONTH} retrains/month
     Cost: ${monthly_retrain_cost:,.2f}/month

  2. Monthly hyperparameter search:
     10 configs x {HP_EPOCHS} epochs per search
     Cost: ${hp_search_cost:,.2f}/month

  3. Ad-hoc experiments and debugging:
     Estimated: ${monthly_retrain_cost * 0.5:,.2f}/month

  TOTAL RECOMMENDED BUDGET: ${monthly_retrain_cost + hp_search_cost + monthly_retrain_cost * 0.5:,.2f}/month

  ANSWER TO "Why not $20,000/month?":
    The accuracy spread between our best and worst configs is only
    {best_overall[1] - worst_overall[1]:.3f}. Spending 4x more on training will not
    improve accuracy 4x -- diminishing returns means the extra $15,000
    buys approximately {(best_overall[1] - worst_overall[1]) * 0.2:.3f} additional accuracy.

    Instead, that $15,000 would be better spent on:
    - Better training DATA (more product images, especially edge cases)
    - Human review for the low-confidence predictions (see 01_simple_cnn.py)
    - Monitoring and drift detection (kailash-ml DriftMonitor)

  STAKEHOLDER-READY SUMMARY:
    "Based on systematic experiments tracked with ExperimentTracker,
    the optimal training budget is approximately
    ${monthly_retrain_cost + hp_search_cost + monthly_retrain_cost * 0.5:,.0f}/month.
    This covers weekly retraining, monthly hyperparameter tuning,
    and ad-hoc experiments. The data shows diminishing returns
    beyond this -- additional compute buys less than 1% accuracy
    improvement. The remaining budget is better invested in
    data quality and monitoring infrastructure."
"""
)

# ── Checkpoint 4: Apply section complete ─────────────────────────────
assert total_experiment_cost > 0, "Experiment cost should be positive"
assert best_overall[1] > worst_overall[1], "Best should beat worst"
print("--- Checkpoint 4 passed --- budget analysis complete\n")


# ════════════════════════════════════════════════════════════════════════
# Experiment Summary (all tracked runs)
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  FULL EXPERIMENT SUMMARY (all ExperimentTracker runs)")
print("=" * 70)
print(f"  Experiment: {exp_name}")
print(f"  Total runs: {len(lr_results) + len(aug_results)}")
print(
    f"  Total compute: {total_experiment_time:.0f}s ({total_experiment_hours:.2f} hours)"
)
print(f"  Estimated cost: ${total_experiment_cost:.2f}")
print(f"\n  All runs are queryable via ExperimentTracker for future comparison.")


# ════════════════════════════════════════════════════════════════════════
# Clean up
# ════════════════════════════════════════════════════════════════════════
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — one-call diagnostics
# ════════════════════════════════════════════════════════════════════════
# This lesson ran learning-rate sweeps and an augmentation study, with a
# diagnostic pad per configuration. kailash-ml's run_diagnostic_checkpoint
# is the one call behind each pad: it hooks every layer, runs a few
# probe batches (no weight updates), replays the loss history and
# returns findings plus a DLDiagnostics session whose
# plot_training_dashboard() draws the loss, gradient and activation
# panels. (km.diagnose(model, kind="dl") on its own only builds an
# un-instrumented session — no hooks, no probes — so it has nothing to
# report.) It does not replace the sweep tables and cost curves above:
# the instruments read optimisation health, not which config is worth
# its compute.

# The winning augmentation setting is the one with the best final
# validation accuracy in THIS run.
winner = max(aug_results, key=lambda name: aug_results[name]["accs"][-1])
winning_model = aug_results[winner]["model"]
diag, findings = run_diagnostic_checkpoint(
    winning_model,
    aug_loader if winner == "flip_crop" else noaug_loader,
    _ce_loss,
    title=f"ResNetSE winner ({winner})",
    train_losses=aug_results[winner]["losses"],
    show=False,
)
print_prescription_pad(findings, f"ResNetSE winner ({winner})")
dashboard = diag.plot_training_dashboard()  # Plotly figure: dashboard.show()


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  THEORY:
  [x] Learning rate is the most important hyperparameter -- too high
      causes oscillation, too low wastes compute
  [x] Data augmentation teaches invariance (flip, crop) and improves
      generalisation at zero data collection cost
  [x] Diminishing returns: each doubling of compute buys progressively
      less accuracy improvement

  BUILD + TRAIN:
  [x] Learning rate sweep: {len(LR_SWEEP)} values ({', '.join(str(lr) for lr in LR_SWEEP)})
      Best: lr={best_lr} (acc={best_lr_acc:.3f})
  [x] Augmentation comparison: no_aug vs flip+crop
      Improvement: {aug_improvement:+.3f} accuracy from augmentation
  [x] All {len(lr_results) + len(aug_results)} runs tracked in ExperimentTracker

  VISUALISE (the proof):
  [x] Diminishing returns chart: marginal accuracy gain per epoch
  [x] Cost-vs-accuracy scatter: Pareto frontier of configurations
  [x] Training curves for all LR and augmentation experiments

  APPLY:
  [x] Answered "how much compute budget?" with data, not opinions
  [x] Monthly budget: ${monthly_retrain_cost + hp_search_cost + monthly_retrain_cost * 0.5:,.0f} (vs $20,000 requested)
  [x] Where to invest the savings: data quality > more compute
  [x] Stakeholder-ready summary with dollar values

  KEY INSIGHT: Hyperparameter tuning is not about finding the "perfect"
  setting -- it is about finding the KNEE of the diminishing returns
  curve and allocating budget accordingly. ExperimentTracker makes this
  a data-driven decision instead of a debate between data scientists
  who always want more GPUs and managers who always want lower costs.

  EXERCISE 2 COMPLETE: You've built CNNs (01), added residual connections
  and attention (02), deployed to production (03), and justified the
  compute budget with data (04). This is the full lifecycle of a
  production computer vision system.

  Next: Exercise 3 takes you into sequence modelling with RNNs, LSTMs,
  and GRUs on real Singapore stock-market data...
"""
)
