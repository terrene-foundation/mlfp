# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1 Application: Credit Card Fraud Detection
# ════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are a fraud analyst at a Singapore bank (DBS/UOB/OCBC).
#   99.8% of daily transactions are legitimate. You have NO labelled
#   fraud examples — only a gut feeling that "unusual" transactions
#   deserve investigation. Your manager asks: "Can we catch more
#   fraud without drowning investigators in false alerts?"
#
# TECHNIQUE: Vanilla (Undercomplete) Autoencoder
#   Train on ONLY normal transactions so the AE learns what "normal"
#   looks like. At inference, legitimate transactions reconstruct
#   well (low error); fraudulent ones reconstruct poorly (high error)
#   because the encoder never learned their patterns.
#
# WHAT YOU'LL SEE:
#   - Reconstruction error distributions: normal vs fraud (clear separation)
#   - Precision-recall curve at different thresholds
#   - Example flagged transactions with their anomaly scores
#   - Quantified business impact in S$ at DBS daily transaction volume
#
# ESTIMATED TIME: ~30-45 min (run and interpret)
# ════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shared.kailash_helpers import get_device, setup_environment

setup_environment()

torch.manual_seed(42)
np.random.seed(42)
device = get_device()
print(f"Using device: {device}")

OUTPUT_DIR = Path("outputs/mlfp05/app_fraud")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 1. Generate realistic Singapore bank transaction data
# ════════════════════════════════════════════════════════════════════════
# In production you would use the IEEE-CIS Fraud Detection dataset or
# internal bank data. Here we generate 200K transactions with realistic
# feature distributions modelled on Singapore banking patterns.

N_TOTAL = 200_000
FRAUD_RATE = 0.002  # 0.2% fraud — realistic for Singapore card-present

n_fraud = int(N_TOTAL * FRAUD_RATE)
n_normal = N_TOTAL - n_fraud

rng = np.random.default_rng(42)

# Normal transaction features
normal_amounts = rng.lognormal(mean=3.5, sigma=1.2, size=n_normal).clip(0.5, 5000)
normal_hour = rng.normal(loc=14, scale=4, size=n_normal).clip(0, 23).astype(int)
normal_merchant_cat = rng.choice(
    range(15),
    size=n_normal,
    p=[
        0.18,
        0.15,
        0.12,
        0.10,
        0.08,
        0.07,
        0.06,
        0.05,
        0.04,
        0.04,
        0.03,
        0.03,
        0.02,
        0.02,
        0.01,
    ],
)
normal_is_online = rng.binomial(1, 0.35, size=n_normal)
normal_distance = rng.exponential(scale=5, size=n_normal).clip(0, 50)
normal_freq_24h = rng.poisson(lam=2, size=n_normal)
normal_amt_ratio = rng.normal(1.0, 0.3, size=n_normal).clip(0.1, 3.0)
normal_foreign = rng.binomial(1, 0.08, size=n_normal)

# Fraud transaction features — shifted distributions
fraud_amounts = rng.lognormal(mean=5.5, sigma=1.5, size=n_fraud).clip(10, 50000)
fraud_hour = rng.choice([0, 1, 2, 3, 4, 22, 23], size=n_fraud)
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
fraud_is_online = rng.binomial(1, 0.75, size=n_fraud)
fraud_distance = rng.exponential(scale=40, size=n_fraud).clip(0, 200)
fraud_freq_24h = rng.poisson(lam=8, size=n_fraud)
fraud_amt_ratio = rng.normal(4.0, 1.5, size=n_fraud).clip(0.5, 15.0)
fraud_foreign = rng.binomial(1, 0.45, size=n_fraud)

# Combine into polars DataFrame
amounts = np.concatenate([normal_amounts, fraud_amounts])
hours = np.concatenate([normal_hour, fraud_hour])
merchant_cats = np.concatenate([normal_merchant_cat, fraud_merchant_cat])
is_online = np.concatenate([normal_is_online, fraud_is_online])
distances = np.concatenate([normal_distance, fraud_distance])
freq_24h = np.concatenate([normal_freq_24h, fraud_freq_24h])
amt_ratios = np.concatenate([normal_amt_ratio, fraud_amt_ratio])
foreign = np.concatenate([normal_foreign, fraud_foreign])
labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

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
print(f"Features: {[c for c in df.columns if c != 'is_fraud']}")

# ════════════════════════════════════════════════════════════════════════
# 2. Prepare training data — ONLY normal transactions
# ════════════════════════════════════════════════════════════════════════
# Key insight: we train on normal-only. The AE never sees fraud, so it
# learns the manifold of legitimate transactions. Fraud lives off-manifold.

feature_cols = [c for c in df.columns if c != "is_fraud"]
all_features = df.select(feature_cols).to_numpy().astype(np.float32)
all_labels = df["is_fraud"].to_numpy()

# Normalise features to [0, 1] for stable training
feat_min = all_features.min(axis=0)
feat_max = all_features.max(axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0
all_features_norm = (all_features - feat_min) / feat_range

# Split: normal-only train (80% of normal), test set (20% normal + all fraud)
normal_mask = all_labels == 0
fraud_mask = all_labels == 1

normal_features = all_features_norm[normal_mask]
fraud_features = all_features_norm[fraud_mask]

n_train = int(len(normal_features) * 0.8)
train_features = normal_features[:n_train]
test_normal = normal_features[n_train:]
test_fraud = fraud_features

# Combine test set
test_features = np.vstack([test_normal, test_fraud])
test_labels = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_fraud))])

train_tensor = torch.tensor(train_features, device=device)
test_tensor = torch.tensor(test_features, device=device)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=512, shuffle=True)

print(f"\nTraining on {len(train_features):,} normal-only transactions")
print(
    f"Test set: {len(test_normal):,} normal + {len(test_fraud):,} fraud = {len(test_features):,} total"
)


# ════════════════════════════════════════════════════════════════════════
# 3. Build Vanilla (Undercomplete) Autoencoder
# ════════════════════════════════════════════════════════════════════════
INPUT_DIM = len(feature_cols)
LATENT_DIM = 3  # Bottleneck forces compression — 8 features -> 3 latent dims


class FraudDetectorAE(nn.Module):
    """Undercomplete autoencoder for transaction anomaly detection."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


model = FraudDetectorAE(INPUT_DIM, LATENT_DIM).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ════════════════════════════════════════════════════════════════════════
# 4. Train on normal transactions only
# ════════════════════════════════════════════════════════════════════════
EPOCHS = 50
losses = []

print("\nTraining fraud detection autoencoder...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for (batch,) in train_loader:
        recon = model(batch)
        loss = criterion(recon, batch)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
        n_batches += 1
    avg_loss = epoch_loss / n_batches
    losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss = {avg_loss:.6f}")

print(f"Final training loss: {losses[-1]:.6f}")

# ════════════════════════════════════════════════════════════════════════
# 5. Compute reconstruction errors on test set
# ════════════════════════════════════════════════════════════════════════
model.eval()
with torch.no_grad():
    recon_test = model(test_tensor)
    # Per-sample MSE
    errors = ((test_tensor - recon_test) ** 2).mean(dim=1).cpu().numpy()

normal_errors = errors[test_labels == 0]
fraud_errors = errors[test_labels == 1]

print(f"\nReconstruction error statistics:")
print(
    f"  Normal: mean={normal_errors.mean():.6f}, std={normal_errors.std():.6f}, "
    f"p95={np.percentile(normal_errors, 95):.6f}"
)
print(
    f"  Fraud:  mean={fraud_errors.mean():.6f}, std={fraud_errors.std():.6f}, "
    f"p95={np.percentile(fraud_errors, 95):.6f}"
)
print(f"  Separation ratio: {fraud_errors.mean() / normal_errors.mean():.1f}x")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 1: Reconstruction error distributions
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(
    normal_errors, bins=80, alpha=0.7, label="Normal", color="#2196F3", density=True
)
axes[0].hist(
    fraud_errors, bins=80, alpha=0.7, label="Fraud", color="#F44336", density=True
)
axes[0].set_xlabel("Reconstruction Error (MSE)", fontsize=12)
axes[0].set_ylabel("Density", fontsize=12)
axes[0].set_title(
    "Reconstruction Error Distribution\nNormal vs Fraud Transactions", fontsize=13
)
axes[0].legend(fontsize=11)
axes[0].axvline(
    np.percentile(normal_errors, 95),
    color="#FF9800",
    linestyle="--",
    label=f"95th pctl normal = {np.percentile(normal_errors, 95):.4f}",
)
axes[0].legend(fontsize=10)

# Box plot
bp = axes[1].boxplot(
    [normal_errors, fraud_errors],
    labels=["Normal", "Fraud"],
    patch_artist=True,
    widths=0.5,
)
bp["boxes"][0].set_facecolor("#2196F3")
bp["boxes"][1].set_facecolor("#F44336")
axes[1].set_ylabel("Reconstruction Error (MSE)", fontsize=12)
axes[1].set_title("Error Comparison: Normal vs Fraud", fontsize=13)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_error_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'fraud_error_distribution.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 2: Precision-Recall at different thresholds
# ════════════════════════════════════════════════════════════════════════
thresholds = np.linspace(errors.min(), np.percentile(errors, 99.5), 200)
precisions = []
recalls = []
f1_scores = []

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

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Precision-Recall curve
axes[0].plot(recalls, precisions, color="#673AB7", linewidth=2)
axes[0].scatter(
    [best_recall],
    [best_precision],
    color="#F44336",
    s=100,
    zorder=5,
    label=f"Best F1={best_f1:.3f}\n(P={best_precision:.3f}, R={best_recall:.3f})",
)
axes[0].set_xlabel("Recall (Fraud Caught)", fontsize=12)
axes[0].set_ylabel("Precision (True Among Flagged)", fontsize=12)
axes[0].set_title("Precision-Recall Curve\nAE-Based Fraud Detection", fontsize=13)
axes[0].legend(fontsize=10, loc="upper right")
axes[0].grid(True, alpha=0.3)

# F1 vs threshold
axes[1].plot(thresholds, f1_scores, color="#009688", linewidth=2, label="F1 Score")
axes[1].plot(
    thresholds, precisions, color="#2196F3", linewidth=1.5, alpha=0.7, label="Precision"
)
axes[1].plot(
    thresholds, recalls, color="#F44336", linewidth=1.5, alpha=0.7, label="Recall"
)
axes[1].axvline(
    best_threshold,
    color="#FF9800",
    linestyle="--",
    label=f"Optimal threshold = {best_threshold:.5f}",
)
axes[1].set_xlabel("Reconstruction Error Threshold", fontsize=12)
axes[1].set_ylabel("Score", fontsize=12)
axes[1].set_title("Threshold Selection\nBalancing Precision and Recall", fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_precision_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'fraud_precision_recall.png'}")

# ════════════════════════════════════════════════════════════════════════
# VISUALISATION 3: Example flagged transactions
# ════════════════════════════════════════════════════════════════════════
# Show top 20 most anomalous transactions with their error scores
top_k = 20
top_indices = np.argsort(errors)[-top_k:][::-1]

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#F44336" if test_labels[i] == 1 else "#2196F3" for i in top_indices]
bars = ax.barh(range(top_k), errors[top_indices], color=colors)
ax.set_yticks(range(top_k))
ax.set_yticklabels(
    [f"Txn #{i} ({'FRAUD' if test_labels[i]==1 else 'Normal'})" for i in top_indices],
    fontsize=9,
)
ax.set_xlabel("Reconstruction Error (Anomaly Score)", fontsize=12)
ax.set_title(
    f"Top {top_k} Most Anomalous Transactions\nRed = True Fraud, Blue = Normal",
    fontsize=13,
)
ax.axvline(
    best_threshold,
    color="#FF9800",
    linestyle="--",
    linewidth=2,
    label=f"Detection threshold = {best_threshold:.5f}",
)
ax.legend(fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_top_anomalies.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'fraud_top_anomalies.png'}")

# ════════════════════════════════════════════════════════════════════════
# BUSINESS IMPACT ANALYSIS
# ════════════════════════════════════════════════════════════════════════
# DBS processes approximately 2M card transactions per day in Singapore.
# Average fraud transaction value: S$800 (conservative estimate).
# Current rule-based system catches ~67% of fraud.

DBS_DAILY_TRANSACTIONS = 2_000_000
AVG_FRAUD_VALUE_SGD = 800
RULE_BASED_RECALL = 0.67
DAILY_FRAUD_COUNT = int(DBS_DAILY_TRANSACTIONS * FRAUD_RATE)
FPR_AT_BEST = np.sum((errors > best_threshold) & (test_labels == 0)) / np.sum(
    test_labels == 0
)

# At the optimal threshold
daily_fraud_caught_ae = int(DAILY_FRAUD_COUNT * best_recall)
daily_fraud_caught_rules = int(DAILY_FRAUD_COUNT * RULE_BASED_RECALL)
daily_additional_caught = daily_fraud_caught_ae - daily_fraud_caught_rules
daily_false_alerts = int(DBS_DAILY_TRANSACTIONS * (1 - FRAUD_RATE) * FPR_AT_BEST)
daily_value_saved = daily_additional_caught * AVG_FRAUD_VALUE_SGD
annual_value_saved = daily_value_saved * 365

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — DBS Singapore Card Fraud Detection")
print("=" * 64)
print(f"\nDBS daily card transactions:     {DBS_DAILY_TRANSACTIONS:>12,}")
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
print(f"\nKey insight: At the optimal threshold, the AE catches {best_recall:.0%} of")
print(
    f"fraud compared to the rule-based system's {RULE_BASED_RECALL:.0%}, while generating"
)
print(f"{daily_false_alerts:,} false alerts/day that investigators must review.")
print(f"For a bank like DBS, the additional S${annual_value_saved:,.0f}/year in")
print(f"prevented fraud far exceeds the cost of reviewing false positives.")
print("=" * 64)
