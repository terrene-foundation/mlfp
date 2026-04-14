# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 4.1: Statistical Outlier Detection (Z-score + IQR)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply the 3-sigma rule with Z-scores for outlier detection
#   - Apply the 1.5*IQR rule without assuming normality
#   - Winsorise extreme values to reduce skewness without losing rows
#   - Score and compare both methods with AUC-ROC and AUC-PR
#
# PREREQUISITES: MLFP02 (distributions, percentiles).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why statistical outliers matter for rare-event detection
#   2. Build — compute Z-scores and IQR bounds from standardised features
#   3. Train — score every row (unsupervised — no parameter fitting)
#   4. Visualise — distribution of flagged rows vs true anomalies
#   5. Apply — Singapore NETS chargeback review queue prioritisation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew

from shared.mlfp04.ex_4 import (
    load_dataset,
    print_metrics,
    score_metrics,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Statistical Outlier Rules Still Matter
# ════════════════════════════════════════════════════════════════════════
# Z-score: flag if |x - mean| / std > 3. Assumes ~normal features.
# IQR:     flag if x < Q1-1.5*IQR or x > Q3+1.5*IQR. Distribution-free.
# Winsorise: clip extremes to IQR bounds (keeps sample size).


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the Z-score and IQR detectors
# ════════════════════════════════════════════════════════════════════════

X, y, feature_cols, _frame = load_dataset()
n_samples, n_features = X.shape
print("\n" + "=" * 70)
print("  Statistical Outlier Detection — Z-score and IQR")
print("=" * 70)
print(
    f"Rows: {n_samples:,} | Features: {n_features} | "
    f"Anomalies: {int(y.sum()):,} ({y.mean():.2%})"
)


def zscore_anomaly_scores(X_scaled: np.ndarray) -> np.ndarray:
    """Return the per-row maximum |Z-score| across features."""
    # TODO: X_scaled is already standardised, so |X_scaled| IS the Z-score.
    # Return the per-row maximum (hint: np.abs then .max(axis=1))
    z = ____
    return ____


def iqr_outlier_counts(X_scaled: np.ndarray):
    """Return (outlier count per row, lower bound, upper bound)."""
    # TODO: Compute Q1 and Q3 per feature via np.percentile(X_scaled, q, axis=0)
    Q1 = ____
    Q3 = ____
    IQR = Q3 - Q1

    # TODO: Compute the 1.5*IQR bounds
    lower = ____
    upper = ____

    # TODO: Count per-row how many features fall outside the bounds
    # (hint: ((X_scaled < lower) | (X_scaled > upper)).sum(axis=1))
    counts = ____
    return counts.astype(np.float64), lower, upper


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (score every row — no fitting required)
# ════════════════════════════════════════════════════════════════════════

z_scores = zscore_anomaly_scores(X)
iqr_scores, lower_bound, upper_bound = iqr_outlier_counts(X)

print("\nZ-score threshold sweep:")
for threshold in [2.0, 2.5, 3.0, 3.5]:
    flagged = z_scores > threshold
    n_flagged = int(flagged.sum())
    precision = float(y[flagged].mean()) if n_flagged else 0.0
    print(
        f"  |z| > {threshold}: flagged={n_flagged:>5,}  "
        f"({n_flagged / n_samples:.1%})  precision={precision:.3f}"
    )

print("\nPer-method scores:")
z_metrics = print_metrics("Z-score (max)", y, z_scores)
iqr_metrics = print_metrics("IQR (outlier count)", y, iqr_scores)

# Winsorisation — clip to IQR bounds
X_winsorised = np.clip(X, lower_bound, upper_bound)
n_clipped = int((X != X_winsorised).sum())
skew_before = float(np.mean(np.abs(skew(X, axis=0))))
skew_after = float(np.mean(np.abs(skew(X_winsorised, axis=0))))
print(f"\nWinsorisation: clipped {n_clipped:,} values ({n_clipped / X.size:.2%})")
print(f"  Mean |skewness| before: {skew_before:.4f}")
print(f"  Mean |skewness| after:  {skew_after:.4f}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert z_metrics["auc_roc"] > 0.4, "Z-score AUC should beat random floor"
assert iqr_metrics["auc_roc"] > 0.4, "IQR AUC should beat random floor"
assert z_scores.min() >= 0, "Max |Z| scores must be non-negative"
assert skew_after <= skew_before + 1e-2, "Winsorisation should not increase skew"
print("\n[ok] Checkpoint passed — Z-score and IQR detectors scored\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
print("Interpretation:")
print("  Z-score finds rows that are extreme on at least ONE feature.")
print("  IQR counts HOW MANY features are extreme.")
print("  AUC-PR is the honest metric for <2% anomaly datasets.")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: NETS Chargeback Review Queue Prioritisation
# ════════════════════════════════════════════════════════════════════════
# NETS processes ~12M e-payments/day in Singapore. Reviewers can look at
# ~400 flagged cases per day. A blended Z-score + IQR pre-filter catches
# an extra ~30 chargebacks/day = ~S$6,600/day (~S$1.6M/year) in recovery,
# against effectively zero compute cost.
#
# Statistical rules are the CHEAPEST, MOST EXPLAINABLE detectors — use
# them as the first stage of a production anomaly pipeline.

reviewer_budget = 400
blended = (z_scores - z_scores.min()) + (iqr_scores - iqr_scores.min())
queue_order = np.argsort(-blended)[:reviewer_budget]
queue_precision = float(y[queue_order].mean())
queue_recall = float(y[queue_order].sum() / max(y.sum(), 1))
print(f"\nQueue-prioritisation demo (reviewer budget = {reviewer_budget}):")
print(f"  Precision in top-{reviewer_budget}: {queue_precision:.3f}")
print(f"  Recall in top-{reviewer_budget}:    {queue_recall:.3f}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Z-score outlier detection on standardised features
  [x] IQR outlier detection without assuming normality
  [x] Winsorisation as a non-destructive alternative to dropping
  [x] AUC-ROC vs AUC-PR on rare-event datasets
  [x] Framed a NETS Singapore scenario with concrete dollar impact

  Next: 02_isolation_forest.py — catches multi-feature anomalies that
  Z-score and IQR cannot.
"""
)
