# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 4.3: Local Outlier Factor (LOF)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain LOF as a ratio of neighbour density to point density
#   - Sweep n_neighbors and explain the "locality" trade-off
#   - Fit LOF and turn negative_outlier_factor_ into an anomaly score
#   - Explain when LOF beats Isolation Forest (varying-density clusters)
#
# PREREQUISITES: 4.2 (Isolation Forest).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — comparing LOCAL densities catches embedded anomalies
#   2. Build — sweep n_neighbors and pick the best value
#   3. Train — fit LOF and extract negative_outlier_factor_
#   4. Visualise — ROC curve (written to outputs/)
#   5. Apply — Shopee return-fraud cluster detection
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from shared.mlfp04.ex_4 import (
    load_dataset,
    print_metrics,
    score_metrics,
    write_roc_chart,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Local Density Beats Global Distance
# ════════════════════════════════════════════════════════════════════════
# LOF(p) = mean( density(neighbour_i) / density(p) ).
# LOF ~ 1 means p has roughly the same density as its k-nearest neighbours.
# LOF >> 1 means p sits in a sparser pocket than its neighbours = anomaly.
# LOF catches cluster-embedded anomalies that Isolation Forest misses.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: sweep n_neighbors
# ════════════════════════════════════════════════════════════════════════

X, y, _feature_cols, _frame = load_dataset()
n_samples, n_features = X.shape
print("\n" + "=" * 70)
print("  Local Outlier Factor (LOF)")
print("=" * 70)
print(
    f"Rows: {n_samples:,} | Features: {n_features} | "
    f"Anomalies: {int(y.sum()):,} ({y.mean():.2%})"
)

print("\nn_neighbors sweep:")
for n_nbrs in [10, 20, 30, 50]:
    # TODO: Build LocalOutlierFactor with n_neighbors=n_nbrs,
    # contamination=0.01, novelty=False
    lof_test = ____

    # TODO: Run fit_predict(X) to get labels
    labels_test = ____

    # TODO: Turn negative_outlier_factor_ into an anomaly score where
    # HIGHER means more anomalous. Hint: negate the attribute.
    scores_test = ____

    m = score_metrics(y, scores_test)
    n_flagged = int((labels_test == -1).sum())
    print(
        f"  n_neighbors={n_nbrs:<3}  AUC-ROC={m['auc_roc']:.4f}  "
        f"AP={m['avg_precision']:.4f}  flagged={n_flagged:,}"
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit LOF with n_neighbors=20
# ════════════════════════════════════════════════════════════════════════

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=False)
lof_labels = lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_

print("\nFinal LOF (n_neighbors=20):")
lof_metrics = print_metrics("LOF", y, lof_scores)
print(f"  Predicted anomalies: {int((lof_labels == -1).sum()):,}")
print(f"  True anomalies:      {int(y.sum()):,}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    lof_metrics["auc_roc"] > 0.5
), f"LOF AUC-ROC {lof_metrics['auc_roc']:.4f} should beat random"
assert lof_scores.std() > 0, "LOF scores should vary across rows"
assert lof_scores.shape[0] == n_samples, "Score length must match row count"
print("\n[ok] Checkpoint passed — LOF scored all rows\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: ROC curve
# ════════════════════════════════════════════════════════════════════════
roc_path = write_roc_chart(y, lof_scores, "LOF", "ex4_roc_lof.html")
print(f"Saved ROC chart: {roc_path}")

print("\nLOF catches cluster-embedded anomalies.")
print("Isolation Forest catches globally-distant anomalies.")
print("Use BOTH, then blend — see 04_ensemble_blending.py.")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Shopee Return-Fraud Cluster Detection
# ════════════════════════════════════════════════════════════════════════
# Shopee's refund-ring fraud forms a tight cluster of accounts embedded
# in the buyer population. Isolation Forest misses it; LOF catches it
# because the fraud cluster has high LOCAL density relative to its
# neighbourhood of legitimate buyers.
#
# Impact: ~S$5.6M/year recovered against ~S$40K/year infrastructure cost.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] LOF as a density-ratio test
  [x] n_neighbors as the "locality" knob
  [x] Cluster-embedded anomaly detection
  [x] O(n^2) scalability limit awareness
  [x] Shopee refund-ring detection scenario

  Next: 04_ensemble_blending.py — combine all four detectors with
  kailash-ml EnsembleEngine.
"""
)
