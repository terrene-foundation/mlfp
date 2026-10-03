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
#   - Sweep contamination to visualise the precision/recall trade-off
#   - Compare tree-based isolation against the statistical baselines
#
# PREREQUISITES: 4.1 (Z-score + IQR baselines).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why "rare points get isolated faster"
#   2. Build — fit IsolationForest with a contamination sweep
#   3. Train — score every row with the best-performing fit
#   4. Visualise — ROC curve (written to outputs/)
#   5. Apply — synthetic-identity screening at a digital lender
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from shared.mlfp04.ex_4 import (
    _finite,
    load_dataset,
    print_auc_by_type,
    print_metrics,
    score_metrics,
    setup_engines,
    teardown_engines,
    track_run,
    write_roc_chart,
)

# ── Kailash-ML ExperimentTracker — anomaly zoo shared store ──────────────
tracker, exp_name = setup_engines()

# Capture per-contamination sweep metrics so the TRACK section below can
# emit them after Task 5. Mutated inside the Task 2 sweep loop.
sweep_results: dict[float, dict[str, float]] = {}


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Path Length Is an Anomaly Score
# ════════════════════════════════════════════════════════════════════════
# An Isolation Forest builds many random binary trees. At every node it
# picks a random feature and a random split value. A point gets "isolated"
# when a branch contains only that one point.
#
# Intuition: anomalies are rare AND far from the bulk of normal rows. A
# random split is MORE LIKELY to separate an anomaly from the rest of the
# data than to separate a crowded normal point. Anomalies end up at
# shallow leaves (few splits to isolate); normal points end up deep.
#
# Score: s(x) = 2 ^ (-E[h(x)] / c(n))
#     h(x)  = path length for x, averaged over all trees
#     c(n)  = average path length of an unsuccessful search in a BST
# Higher score = shorter path = more anomalous.
#
# PROS: scales to large N, handles high-dimensional features, no
# assumption of density or distribution. Parallelises across trees.
# CONS: the `contamination` parameter is an expert guess — set it from
# domain knowledge of the expected anomaly rate, NOT from the data.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: contamination sweep
# ════════════════════════════════════════════════════════════════════════

X, y, _feature_cols, frame = load_dataset()
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
    model = IsolationForest(
        n_estimators=200,
        contamination=contam,
        random_state=42,
        n_jobs=-1,
    )
    preds = model.fit_predict(X)
    n_flagged = int((preds == -1).sum())
    flagged = preds == -1
    precision = float(y[flagged].mean()) if n_flagged else 0.0
    sweep_results[contam] = {
        "n_flagged": float(n_flagged),
        "precision": precision,
    }
    print(
        f"  contamination={contam:<6}  flagged={n_flagged:>5,}  "
        f"precision={precision:.3f}"
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit the best-performing contamination
# ════════════════════════════════════════════════════════════════════════
# We pin contamination at 0.01 because ~1% of rows are injected anomalies.
# In production you would set this from domain knowledge of the expected
# anomaly rate, not from the label (which is unavailable at train time in
# a true anomaly detection setting). Note that contamination only moves
# the flag THRESHOLD — the anomaly scores (and so AUC) do not depend on it.

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1,
)
iso_forest.fit(X)

# Higher score_samples = more normal; negate so higher = more anomalous
iso_scores = -iso_forest.score_samples(X)
iso_labels = iso_forest.predict(X)

print("\nFinal Isolation Forest (contamination=0.01):")
iso_metrics = print_metrics("Isolation Forest", y, iso_scores)
print(f"  Predicted anomalies: {int((iso_labels == -1).sum()):,}")
print(f"  True anomalies:      {int(y.sum()):,}")

# Compare against the per-feature Z-score rule from 4.1, per anomaly type
z_scores = np.abs(X).max(axis=1)
print("\nPer anomaly type (1.0 = perfect, 0.5 = chance):")
iso_by_type = print_auc_by_type("Isolation Forest", frame, iso_scores)
z_by_type = print_auc_by_type("Z-score (4.1)", frame, z_scores)


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    iso_metrics["auc_roc"] > 0.5
), f"Isolation Forest AUC-ROC {iso_metrics['auc_roc']:.4f} should beat random"
assert iso_metrics["avg_precision"] > 0.0, "Isolation Forest AP should be positive"
assert (iso_labels == -1).sum() > 0, "Should flag at least one anomaly"
assert iso_scores.std() > 0, "Scores should vary across rows"
print("\n[ok] Checkpoint passed — Isolation Forest scored all rows\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: ROC curve
# ════════════════════════════════════════════════════════════════════════
roc_path = write_roc_chart(
    y, iso_scores, "Isolation Forest", "ex4_roc_isolation_forest.html"
)
print(f"Saved ROC chart: {roc_path}")

print("\nInterpretation (computed from the per-type AUCs above):")
for t in iso_by_type:
    diff = iso_by_type[t] - z_by_type[t]
    verdict = "better than" if diff > 0.02 else (
        "worse than" if diff < -0.02 else "about the same as"
    )
    print(
        f"  {t:<11} IF={iso_by_type[t]:.3f}  Z={z_by_type[t]:.3f}  "
        f"-> Isolation Forest is {verdict} the Z-score rule"
    )
print(
    "  Random splits use all features jointly, so IF can rank some"
    " 'dependency' rows (normal per feature, odd in combination) above"
    " chance where a per-feature rule cannot. On heavy-tailed real"
    " features, though, many genuine applications are also easy to"
    " isolate, so a single extreme field is not always ranked first."
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Synthetic-Identity Screening at a Digital Lender
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore digital lender screens every new
# credit application overnight before approval. A known attack is the
# "synthetic identity" — an application assembled from fragments of
# several real people. Each field looks normal on its own (age, tenure,
# balances are all in range), but the joint combination is unusual
# (e.g. 30 years employed at age 25). That is exactly the 'dependency'
# anomaly type in this exercise's data.
#
# Why Isolation Forest is a reasonable first tool here:
#   - It scores every application on all 20 fields jointly, not one at
#     a time, so it can pick up some combination-level oddities
#   - Anomalous applications are rare (~1% here)
#   - It is fast: 200 trees on 20,000 rows fit in seconds on a laptop
#
# BUSINESS IMPACT (illustrative assumptions, not reported figures): if
# the lender reviews the top 1% most anomalous applications each night,
# the review load drops from every application to a few hundred, and
# each synthetic identity stopped before disbursement avoids the full
# loan amount. Use the per-type AUC above to judge how much of that
# benefit IF actually delivers on THIS data before promising it.
#
# LIMITATIONS: Isolation Forest is GLOBAL — it asks how easy a point is
# to separate from everything else. Points that are only odd relative
# to their LOCAL neighbourhood are LOF's territory (Exercise 4.3).


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Per-contamination sweep keys are formatted as `iso_contam_X_precision`
# / `iso_contam_X_flagged` where X is the contamination value with dots
# replaced by underscores (the tracker key regex disallows `.` followed
# by a digit, so 0.001 → "0_001"). Final-fit metrics use bare names.

sweep_metrics: dict[str, float] = {}
for contam, stats in sweep_results.items():
    label = str(contam).replace(".", "_")
    sweep_metrics[f"iso_contam_{label}_n_flagged"] = stats["n_flagged"]
    sweep_metrics[f"iso_contam_{label}_precision"] = _finite(stats["precision"])

track_run(
    tracker,
    exp_name,
    run_name="isolation_forest",
    params={
        "n_samples": n_samples,
        "n_features": n_features,
        "n_estimators": 200,
        "best_contamination": 0.01,
        "anomaly_rate": float(y.mean()),
    },
    scalar_metrics={
        "iso_auc_roc": _finite(iso_metrics["auc_roc"]),
        "iso_avg_precision": _finite(iso_metrics["avg_precision"]),
        "iso_n_predicted_anomalies": float(int((iso_labels == -1).sum())),
    }
    | sweep_metrics,
)
print(
    f"\n  [tracked] isolation_forest sweep + final-fit logged to {exp_name} "
    f"run='isolation_forest'\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — AnomalyDetectionEngine.detect()
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's AnomalyDetectionEngine.detect() wraps Isolation
# Forest with the same sklearn machinery this lesson hand-built. The
# engine handles polars→numpy conversion and emits an AnomalyResult
# (labels + scores + n_anomalies + algorithm-specific metrics) in one
# call. The tracker leaderboard is the natural follow-up.

import polars as pl

from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine

anomaly_df = pl.from_numpy(X, schema=_feature_cols)
det = AnomalyDetectionEngine()
fit_result = det.detect(anomaly_df, algorithm="isolation_forest", contamination=0.01)
fit_metrics = score_metrics(y, np.asarray(fit_result.scores))
print(
    f"  AnomalyDetectionEngine.detect(isolation_forest, contamination=0.01): "
    f"AUC-ROC={fit_metrics['auc_roc']:.4f}  "
    f"AP={fit_metrics['avg_precision']:.4f}  "
    f"n_anomalies={fit_result.n_anomalies}"
)
print(
    f"  Hand-rolled AUC-ROC (Task 3): {iso_metrics['auc_roc']:.4f} "
    f"— same algorithm under the hood, one-line vs ten-line interface.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Explained path-length isolation without opening the sklearn source
  [x] Ran a contamination sweep and read the precision trade-off
  [x] Fit IsolationForest on 20-feature tabular data with ~1% anomalies
  [x] Compared IF with the Z-score rule per anomaly type
  [x] Generated an ROC chart from ModelVisualizer
  [x] Framed a synthetic-identity screening scenario (illustrative figures)

  KEY INSIGHT: Isolation Forest scores points on all features JOINTLY,
  so it can see some combination-level anomalies that per-feature rules
  cannot. It is not uniformly better: on heavy-tailed real features a
  simple Z-score can rank single extreme values more sharply. Measure
  per anomaly type before choosing.

  Next: 03_local_outlier_factor.py — LOF compares LOCAL density, catching
  anomalies embedded in varying-density clusters.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
