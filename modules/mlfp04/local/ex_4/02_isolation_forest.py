# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 4.2: Isolation Forest Anomaly Detection
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain path-length isolation as an anomaly score
#   - Fit an Isolation Forest with the right contamination setting
#   - Sweep contamination to visualise precision/recall trade-off
#   - Compare tree-based isolation against statistical baselines
#
# PREREQUISITES: 4.1 (Z-score + IQR baselines).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — "rare points get isolated faster"
#   2. Build — fit IsolationForest with a contamination sweep
#   3. Train — score every row with the best-performing fit
#   4. Visualise — ROC curve (written to outputs/)
#   5. Apply — GrabPay merchant risk scoring
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from shared.mlfp04.ex_4 import (
    load_dataset,
    print_metrics,
    score_metrics,
    write_roc_chart,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Path Length as an Anomaly Score
# ════════════════════════════════════════════════════════════════════════
# Isolation Forest builds random binary trees with random feature splits.
# Anomalies need fewer splits to isolate (shallow leaf = anomalous).
# Score: higher = more anomalous. Contamination sets the expected
# fraction of anomalies — it's a domain assumption, not a hyperparameter.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: contamination sweep
# ════════════════════════════════════════════════════════════════════════

X, y, _feature_cols, _frame = load_dataset()
n_samples, n_features = X.shape
print("\n" + "=" * 70)
print("  Isolation Forest Anomaly Detection")
print("=" * 70)
print(
    f"Rows: {n_samples:,} | Features: {n_features} | "
    f"Anomalies: {int(y.sum()):,} ({y.mean():.2%})"
)

print("\nContamination sweep:")
contamination_grid = [0.001, 0.005, 0.01, 0.02, 0.05]
for contam in contamination_grid:
    # TODO: Instantiate IsolationForest with n_estimators=200,
    # contamination=contam, random_state=42, n_jobs=-1
    model = ____

    # TODO: Call model.fit_predict(X) to get labels (-1 = anomaly, 1 = normal)
    preds = ____

    n_flagged = int((preds == -1).sum())
    flagged = preds == -1
    precision = float(y[flagged].mean()) if n_flagged else 0.0
    print(
        f"  contamination={contam:<6}  flagged={n_flagged:>5,}  "
        f"precision={precision:.3f}"
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit the best-performing contamination
# ════════════════════════════════════════════════════════════════════════

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1,
)
# TODO: Call iso_forest.fit(X)
____

# TODO: Extract anomaly scores. score_samples returns HIGHER for normal;
# negate so HIGHER means more anomalous.
iso_scores = ____
iso_labels = iso_forest.predict(X)

print("\nFinal Isolation Forest (contamination=0.01):")
iso_metrics = print_metrics("Isolation Forest", y, iso_scores)
print(f"  Predicted anomalies: {int((iso_labels == -1).sum()):,}")
print(f"  True anomalies:      {int(y.sum()):,}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    iso_metrics["auc_roc"] > 0.5
), f"Isolation Forest AUC-ROC {iso_metrics['auc_roc']:.4f} should beat random"
assert iso_metrics["avg_precision"] > 0.0, "AP should be positive"
assert (iso_labels == -1).sum() > 0, "Should flag at least one anomaly"
print("\n[ok] Checkpoint passed — Isolation Forest scored all rows\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: ROC curve
# ════════════════════════════════════════════════════════════════════════
roc_path = write_roc_chart(
    y, iso_scores, "Isolation Forest", "ex4_roc_isolation_forest.html"
)
print(f"Saved ROC chart: {roc_path}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: GrabPay Merchant Payout Risk Scoring
# ════════════════════════════════════════════════════════════════════════
# GrabPay runs nightly payouts for merchants across SEA. Refund-ring
# fraud looks normal on single features but unusual in 40+ feature joint
# space. Isolation Forest catches it; statistical rules don't.
#
# Impact: ~S$1.2M/year recovered against <S$20K/year IT cost. 60x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Path-length isolation explained
  [x] Contamination sweep and precision trade-off
  [x] IsolationForest on 40+ feature tabular data
  [x] ROC chart via ModelVisualizer
  [x] GrabPay SEA merchant fraud scenario

  Next: 03_local_outlier_factor.py — LOF finds cluster-embedded anomalies
  that Isolation Forest misses.
"""
)
