# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.2: Undercomplete Autoencoder (Bottleneck Compression)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an undercomplete AE with bottleneck (784 -> 16 = 49:1 compression)
#   - Understand WHY forced compression solves the identity risk
#   - Visualise blurry but meaningful reconstructions
#   - Apply to credit card fraud detection at a Singapore bank
#   - Quantify business impact in S$ with precision-recall analysis
#
# PREREQUISITES: 01_standard_ae.py (identity risk understanding)
# ESTIMATED TIME: ~20 min
#
# TASKS:
#   1. Build undercomplete AE (784 -> 256 -> 64 -> 16)
#   2. Train on Fashion-MNIST and visualise reconstructions
#   3. Apply: fraud detection at a Singapore bank using anomaly reconstruction error
#   4. Business impact analysis with S$ projections
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared.mlfp05.ex_1 import (
    INPUT_DIM,
    LATENT_DIM,
    EPOCHS,
    OUTPUT_DIR,
    device,
    load_fashion_mnist,
    setup_engines,
    train_variant,
    show_reconstruction,
    register_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Forced Compression via Bottleneck
# ════════════════════════════════════════════════════════════════════════
# The fix for the identity risk is simple: make the bottleneck SMALLER
# than the input. With latent_dim=16, the encoder must compress 784
# pixels into just 16 numbers — a 49:1 compression ratio.
#
# Analogy: A 50-page quarterly report compressed into a one-page
# executive summary. The summary MUST capture the key points because
# there is no room for everything. That forced compression is exactly
# what the undercomplete bottleneck does.
#
# The encoder must learn WHAT MATTERS in each image — the difference
# between a shirt and a shoe, not the exact pixel values. This is
# representation learning: extracting structure from data.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build and Train Undercomplete AE
# ════════════════════════════════════════════════════════════════════════


class UndercompleteAE(nn.Module):
    """Bottleneck forces compression: 784 -> 256 -> 64 -> 16 -> 64 -> 256 -> 784."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # TODO: Build encoder — fully-connected layers that narrow
        #       input_dim -> 256 -> 64 -> latent_dim, ReLU after each hidden
        #       layer (no activation on the latent code)
        #       Key: latent_dim=16 << input_dim=784 forces compression
        self.encoder = ____

        # TODO: Build decoder — the mirror image of the encoder,
        #       latent_dim -> 64 -> 256 -> input_dim, ReLU between layers and a
        #       Sigmoid on the output (pixels live in [0, 1])
        self.decoder = ____

    def forward(self, x):
        # TODO: Encode then decode. Return (reconstruction, latent_code)
        ____


def undercomplete_ae_loss(model, xb):
    # TODO: Forward pass, MSE loss between reconstruction and input
    # Return (loss, empty_dict)
    ____


print("\n" + "=" * 70)
print("  Undercomplete AE — Forced Compression (latent=16)")
print("=" * 70)
print("  784 pixels -> 16 numbers. Compression ratio 49:1.")

# TODO: undercomplete_model — an UndercompleteAE for flattened images with
#       the module's latent size
undercomplete_model = ____

# TODO: Train with train_variant — run name "undercomplete_ae", the flattened
#       loader and your loss function (same pattern as 01_standard_ae.py)
undercomplete_losses = ____

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# Same pattern as 01_standard_ae.py — see that file for the full
# Prescription Pad walkthrough. Here we expect a DIFFERENT picture:
# the undercomplete bottleneck (latent=16) blocks identity-copy, so
# gradients should be healthier across the encoder *while* the
# train loss stays noticeably higher than the overcomplete AE —
# that higher loss is the SIGNAL of genuine compression learning.
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _diag_loss(m, batch):
    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
    loss, _ = undercomplete_ae_loss(m, xb)
    return loss


print("\n── Diagnostic Report (Undercomplete AE) ──")
diag, findings = run_diagnostic_checkpoint(
    undercomplete_model,
    flat_loader,
    _diag_loss,
    title="Undercomplete AE (latent=16)",
    n_batches=8,
    train_losses=undercomplete_losses,
    show=False,
)
print_prescription_pad(findings, "Undercomplete AE (latent=16)")

# ══════ READING THE PRESCRIPTION PAD (key: see 01_standard_ae.py) ══════
# Compare with 01: the 16-unit bottleneck cannot copy, so the final
# train loss should sit HIGHER than the overcomplete AE's — that gap is
# compression, not failure. On the pad, check the narrow encoder layers:
# vanishing gradients or a high dead-ReLU share there shrink the
# effective latent size even further.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Reconstruction grid
# ════════════════════════════════════════════════════════════════════════

# TODO: show_reconstruction on the flattened test images, with a title that
#       reports the latent size, e.g. "Undercomplete AE (latent=16)"
____

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(undercomplete_losses) == EPOCHS
assert undercomplete_losses[-1] < undercomplete_losses[0]
# INTERPRETATION: The reconstructions are blurry but recognisable.
# The model kept the SHAPE (is it a shirt? a shoe?) but lost DETAIL
# (exact button placement, stitching pattern). This is the information
# bottleneck principle: compress enough, and the model learns structure.
print("\n--- Checkpoint passed --- undercomplete AE trained\n")

if has_registry:
    register_model(
        registry, "undercomplete_ae", undercomplete_model, undercomplete_losses[-1]
    )


# ════════════════════════════════════════════════════════════════════════
# APPLY — Credit Card Fraud Detection at a Singapore Bank
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are a fraud analyst at a Singapore retail bank. 99.8% of
# daily transactions are legitimate. You have NO labelled fraud
# examples — only a gut feeling that "unusual" transactions deserve
# investigation. Your manager asks: "Can we catch more fraud without
# drowning investigators in false alerts?"
#
# TECHNIQUE: Train on ONLY normal transactions so the AE learns what
# "normal" looks like. At inference, legitimate transactions reconstruct
# well (low error); fraudulent ones reconstruct poorly (high error)
# because the encoder never learned their patterns.

print("\n" + "=" * 70)
print("  APPLICATION: Credit Card Fraud Detection (Singapore bank)")
print("=" * 70)

# --- Generate realistic Singapore bank transaction data ---
N_TOTAL = 200_000
FRAUD_RATE = 0.002  # 0.2% fraud — realistic for Singapore card-present

n_fraud = int(N_TOTAL * FRAUD_RATE)
n_normal = N_TOTAL - n_fraud
rng = np.random.default_rng(42)

# TODO: Generate n_normal legitimate transactions with rng (one numpy
#       Generator method per feature; clip where a range is given)
# - normal_amounts: log-normal, log-mean 3.5, log-sigma 1.2, clipped to [0.5, 5000]
# - normal_hour: normal around 14:00 with sd 4 h, clipped to [0, 23], as int
# - normal_merchant_cat: one of 15 categories with probabilities
#   0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03,
#   0.02, 0.02, 0.01 (mirror the fraud_merchant_cat call given below)
# - normal_is_online: 0/1 flag, P(online) = 0.35
# - normal_distance: exponential with scale 5 km, clipped to [0, 50]
# - normal_freq_24h: Poisson count with mean 2
# - normal_amt_ratio: normal, mean 1.0, sd 0.3, clipped to [0.1, 3.0]
# - normal_foreign: 0/1 flag, P(foreign) = 0.08
normal_amounts = ____
normal_hour = ____
normal_merchant_cat = ____
normal_is_online = ____
normal_distance = ____
normal_freq_24h = ____
normal_amt_ratio = ____
normal_foreign = ____

# TODO: Generate n_fraud fraudulent transactions — the same feature kinds
#       with shifted distributions to simulate anomalous behaviour
# - fraud_amounts: log-normal, log-mean 5.5, log-sigma 1.5, clipped to [10, 50000]
# - fraud_hour: uniform pick from the late-night hours 0, 1, 2, 3, 4, 22, 23
# - fraud_is_online: P(online) = 0.75 (mostly online)
# - fraud_distance: exponential, scale 40 km, clipped to [0, 200] (far from home)
# - fraud_freq_24h: Poisson with mean 8 (high-frequency burst)
# - fraud_amt_ratio: normal, mean 4.0, sd 1.5, clipped to [0.5, 15.0]
# - fraud_foreign: P(foreign) = 0.45 (often foreign)
fraud_amounts = ____
fraud_hour = ____
fraud_merchant_cat = rng.choice(
    range(15),
    size=n_fraud,
    p=[
        0.02,
        0.02,
        0.03,
        0.03,
        0.05,
        0.05,
        0.05,
        0.08,
        0.10,
        0.10,
        0.12,
        0.12,
        0.08,
        0.08,
        0.07,
    ],
)
fraud_is_online = ____
fraud_distance = ____
fraud_freq_24h = ____
fraud_amt_ratio = ____
fraud_foreign = ____

# TODO: Combine into arrays — for each feature, the normal rows followed by
# the fraud rows (one numpy call joins two arrays end to end); labels holds
# 0 for every normal row and 1 for every fraud row, in the same order.
# The polars DataFrame below then names and shuffles the columns.
amounts = ____
hours = ____
merchant_cats = ____
is_online = ____
distances = ____
freq_24h = ____
amt_ratios = ____
foreign = ____
labels = ____

df = pl.DataFrame(
    {
        "amount": amounts,
        "hour": hours,
        "merchant_category": merchant_cats,
        "is_online": is_online,
        "distance_from_home_km": distances,
        "transactions_last_24h": freq_24h,
        "amount_vs_avg_ratio": amt_ratios,
        "is_foreign": foreign,
        "is_fraud": labels,
    }
).sample(fraction=1.0, seed=42, shuffle=True)

print(
    f"Dataset: {df.shape[0]:,} transactions, {df.filter(pl.col('is_fraud') == 1).shape[0]} fraud ({FRAUD_RATE*100:.1f}%)"
)

# --- Prepare training data (normal-only) ---
feature_cols = [c for c in df.columns if c != "is_fraud"]
all_features = df.select(feature_cols).to_numpy().astype(np.float32)
all_labels = df["is_fraud"].to_numpy()

# TODO: Min-max normalise all_features column by column: feat_min and
# feat_max are the per-feature (axis 0) minimum and maximum; the zero-range
# guard is given; all_features_norm maps every feature into [0, 1]
feat_min = ____
feat_max = ____
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0
all_features_norm = ____

# TODO: Split into normal-only training set and mixed test set
# normal_mask = all_labels == 0
# Use first 80% of normal for training, rest for test
# Combine remaining normal + all fraud for test
normal_mask = all_labels == 0
fraud_mask = all_labels == 1
normal_features = all_features_norm[normal_mask]
fraud_features = all_features_norm[fraud_mask]

n_train = int(len(normal_features) * 0.8)
train_features = normal_features[:n_train]
test_normal = normal_features[n_train:]
test_fraud = fraud_features
test_features = np.vstack([test_normal, test_fraud])
test_labels = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_fraud))])

train_tensor = torch.tensor(train_features, device=device)
test_tensor = torch.tensor(test_features, device=device)
fraud_train_loader = DataLoader(
    TensorDataset(train_tensor), batch_size=512, shuffle=True
)

print(f"Training on {len(train_features):,} normal-only transactions")
print(f"Test set: {len(test_normal):,} normal + {len(test_fraud):,} fraud")

# --- Build and train fraud detector ---
FRAUD_INPUT_DIM = len(feature_cols)


class FraudDetectorAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # TODO: Build encoder — fully-connected input_dim -> 32 -> 16 ->
        #       latent_dim, ReLU after each hidden layer
        self.encoder = ____

        # TODO: Build decoder — mirror of encoder, ending with Sigmoid
        self.decoder = ____

    def forward(self, x):
        # TODO: Encode then decode. Return reconstruction only (no latent)
        ____


# TODO: fraud_model — a FraudDetectorAE over all transaction features with
#       a 3-dimensional latent code, on the training device
fraud_model = ____
fraud_opt = torch.optim.Adam(fraud_model.parameters(), lr=1e-3)

print("\nTraining fraud detection autoencoder...")
# Training loop (provided) — 50 epochs of reconstruction MSE on normal-only
# batches; it prints every 10 epochs.
for epoch in range(50):
    fraud_model.train()
    epoch_loss = 0.0
    n_batches = 0
    for (batch,) in fraud_train_loader:
        recon = fraud_model(batch)
        loss = F.mse_loss(recon, batch)
        fraud_opt.zero_grad()
        loss.backward()
        fraud_opt.step()
        epoch_loss += loss.item()
        n_batches += 1
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/50: loss = {epoch_loss/n_batches:.6f}")

# --- Compute reconstruction errors ---
fraud_model.eval()
with torch.no_grad():
    recon_test = fraud_model(test_tensor)
    errors = ((test_tensor - recon_test) ** 2).mean(dim=1).cpu().numpy()

normal_errors = errors[test_labels == 0]
fraud_errors = errors[test_labels == 1]

print(f"\nReconstruction error statistics:")
print(
    f"  Normal: mean={normal_errors.mean():.6f}, p95={np.percentile(normal_errors, 95):.6f}"
)
print(
    f"  Fraud:  mean={fraud_errors.mean():.6f}, p95={np.percentile(fraud_errors, 95):.6f}"
)
print(f"  Separation ratio: {fraud_errors.mean() / normal_errors.mean():.1f}x")

# --- Visualisation 1: Error distributions ---
# TODO: Create 1x2 subplot figure (14, 5)
# Left: overlapping histograms of normal_errors (blue) and fraud_errors (red)
#       with 95th percentile vertical line
# Right: boxplot comparing normal vs fraud errors
# Save to OUTPUT_DIR / "ex1_fraud_error_distribution.png"
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_fraud_error_distribution.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Visualisation 2: Precision-Recall ---
# Sweep (provided): 200 thresholds from the smallest error to the 99.5th
# percentile; at each, flag errors above the threshold as fraud, count
# tp/fp/fn, and keep the threshold with the best F1.
thresholds = np.linspace(errors.min(), np.percentile(errors, 99.5), 200)
precisions, recalls, f1_scores = [], [], []
for t in thresholds:
    predicted_fraud = errors > t
    true_fraud = test_labels == 1
    tp = np.sum(predicted_fraud & true_fraud)
    fp = np.sum(predicted_fraud & ~true_fraud)
    fn = np.sum(~predicted_fraud & true_fraud)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

precisions = np.array(precisions)
recalls = np.array(recalls)
f1_scores = np.array(f1_scores)
best_f1_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_f1_idx]
best_precision = precisions[best_f1_idx]
best_recall = recalls[best_f1_idx]
best_f1 = f1_scores[best_f1_idx]

# TODO: Create 1x2 subplot for precision-recall curve and threshold selection
# Left: precision vs recall curve with best F1 point marked
# Right: F1, precision, recall vs threshold with optimal threshold line
# Save to OUTPUT_DIR / "ex1_fraud_precision_recall.png"
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
____
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ex1_fraud_precision_recall.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Visualisation 3: Top anomalies ---
# TODO: Horizontal bar chart of the top_k highest anomaly scores, largest
#       at the top. Colour each bar by its TRUE label (red = fraud,
#       blue = normal), label each bar with its transaction index, and draw
#       best_threshold as a dashed vertical line.
# Save to OUTPUT_DIR / "ex1_fraud_top_anomalies.png"
top_k = 20
top_indices = ____  # Hint: np.argsort sorts ascending
fig, ax = plt.subplots(figsize=(12, 6))
____
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ex1_fraud_top_anomalies.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Business Impact Analysis ---
BANK_DAILY_TRANSACTIONS = 2_000_000  # illustrative scenario figures
AVG_FRAUD_VALUE_SGD = 800
RULE_BASED_RECALL = 0.67
DAILY_FRAUD_COUNT = int(BANK_DAILY_TRANSACTIONS * FRAUD_RATE)
FPR_AT_BEST = np.sum((errors > best_threshold) & (test_labels == 0)) / np.sum(
    test_labels == 0
)

# TODO: Compute daily metrics (whole transactions, so truncate to int)
# - daily_fraud_caught_ae / daily_fraud_caught_rules: the day's fraud events
#   each system catches at its recall (best_recall vs RULE_BASED_RECALL)
# - daily_additional_caught: how many more the autoencoder catches
# - daily_false_alerts: legitimate daily transactions flagged at FPR_AT_BEST
# - daily_value_saved / annual_value_saved: extra catches x AVG_FRAUD_VALUE_SGD,
#   per day and over 365 days
daily_fraud_caught_ae = ____
daily_fraud_caught_rules = ____
daily_additional_caught = ____
daily_false_alerts = ____
daily_value_saved = ____
annual_value_saved = ____

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — Card Fraud Detection (illustrative bank)")
print("=" * 64)
print(f"\nDaily card transactions:         {BANK_DAILY_TRANSACTIONS:>12,}")
print(f"Estimated daily fraud events:    {DAILY_FRAUD_COUNT:>12,}")
print(f"Average fraud value:             {'S$' + str(AVG_FRAUD_VALUE_SGD):>12}")
print(f"\nCurrent rule-based system:")
print(f"  Fraud recall:                  {RULE_BASED_RECALL:>11.0%}")
print(f"  Fraud caught/day:              {daily_fraud_caught_rules:>12,}")
print(f"\nAutoencoder-based system (optimal threshold = {best_threshold:.5f}):")
print(f"  Fraud recall:                  {best_recall:>11.1%}")
print(f"  Precision:                     {best_precision:>11.1%}")
print(f"  Fraud caught/day:              {daily_fraud_caught_ae:>12,}")
print(f"  False alerts/day:              {daily_false_alerts:>12,}")
print(f"\nIncremental impact:")
print(f"  Additional fraud caught/day:   {daily_additional_caught:>12,}")
print(f"  Value saved per day:           {'S$' + f'{daily_value_saved:,.0f}':>12}")
print(f"  Value saved per year:          {'S$' + f'{annual_value_saved:,.0f}':>12}")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built an undercomplete AE with 49:1 compression (784 -> 16)
  [x] Observed blurry but meaningful reconstructions — structure preserved
  [x] Applied bottleneck AE to credit card fraud detection at a Singapore bank
  [x] Computed precision-recall curves for threshold selection
  [x] Quantified business impact: S$ value of additional fraud prevented

  KEY INSIGHT: The bottleneck forces the encoder to learn what MATTERS.
  A shirt's overall shape is preserved; its button stitching is lost.
  In fraud detection, normal transaction PATTERNS are preserved;
  fraudulent patterns (unusual amount + time + merchant) cannot be
  reconstructed, producing the anomaly signal.

  Next: 03_denoising_ae.py adds noise robustness...
"""
)
