# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 4.4: Ensemble Blending with kailash-ml EnsembleEngine
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Normalise anomaly scores across detectors with different scales
#   - Blend detectors with equal weights, AUC weights, and rank weights
#   - Blend unsupervised detectors with AnomalyDetectionEngine.ensemble_detect()
#   - Train a supervised second stage with EnsembleEngine.blend() / .stack()
#     once a labelled review sample exists
#   - Compare all methods on AUC-ROC, AUC-PR, and precision-at-recall
#   - Monitor ensemble anomaly rate over time for drift detection
#
# PREREQUISITES: 4.1, 4.2, 4.3 (all four detectors).
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — when blending helps, and when it does not
#   2. Build — re-fit the four detectors and normalise their scores
#   3. Train — three manual blends, the AnomalyDetectionEngine ensemble,
#              and an EnsembleEngine supervised second stage
#   4. Visualise — comparison chart + ROC curves + monitoring chart
#   5. Apply — production anomaly monitoring at a Singapore digital bank
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor

from kailash_ml import EnsembleEngine
from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine

from shared.mlfp04.ex_4 import (
    _finite,
    default_reality_check,
    load_dataset,
    normalise_scores,
    precision_at_recall,
    print_auc_by_type,
    print_metrics,
    rank_normalise,
    score_metrics,
    setup_engines,
    split_review_holdout,
    teardown_engines,
    track_run,
    write_comparison_chart,
    write_monitoring_chart,
)

# ── Kailash-ML ExperimentTracker — anomaly zoo shared store ──────────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — When Blending Helps (and When It Does Not)
# ════════════════════════════════════════════════════════════════════════
# Every anomaly detector has a blind spot. Z-score misses anomalies that
# live in feature COMBINATIONS. IQR counts extreme features but treats
# every feature equally. Isolation Forest asks a global question. LOF
# asks a local one, and is blind to groups larger than n_neighbors.
#
# Blending averages normalised scores. It REDUCES VARIANCE when the
# detectors' errors are weakly correlated and their scores are on
# comparable scales: if IF misses a row that LOF and Z-score both rank
# highly, the average can still rank it highly. It is NOT guaranteed to
# win: averaging a strong detector with weaker ones can pull the strong
# detector's correct rankings down, so a blend can score BELOW the best
# single detector. You find out by measuring, on rows the blend was not
# tuned on.
#
# Unsupervised blends (no labels needed):
#   1. Equal-weight        — treats every detector as equally trustworthy
#   2. AUC-weighted        — weights from a small LABELLED review sample
#   3. Rank-based          — uses percentile ranks instead of raw scores
#   4. AnomalyDetectionEngine.ensemble_detect(voting="score_average")
#      — kailash-ml's unsupervised engine blend of IF + LOF
#
# Supervised second stage (once analysts have labelled a review sample):
#   5. EnsembleEngine.blend() / .stack() — kailash-ml's ensemble API for
#      CLASSIFIERS. Here the classifiers learn from the detector scores
#      which detector to trust, using the review sample's labels.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: re-fit the four detectors
# ════════════════════════════════════════════════════════════════════════
# The ensemble needs score vectors from every detector. We re-fit each
# one here so this file is independently runnable (R10 mandates each
# technique file works on its own).

X, y, _feature_cols, frame = load_dataset()
n_samples, _n_features = X.shape

# A small LABELLED review sample (30% of rows — as if analysts had checked
# them) is used for anything that needs labels: AUC weights and the
# supervised second stage. Every reported metric is computed on the
# remaining 70% HOLDOUT, never on the rows that tuned the blend.
review_idx, holdout_idx = split_review_holdout(n_samples, review_fraction=0.3)
y_hold = y[holdout_idx]

print("\n" + "=" * 70)
print("  Ensemble Blending — Z-score + IQR + IF + LOF")
print("=" * 70)
print(f"Rows: {n_samples:,} | Anomalies: {int(y.sum()):,} ({y.mean():.2%})")
print(
    f"Review sample: {len(review_idx):,} rows ({int(y[review_idx].sum())} anomalies)"
    f" | Holdout: {len(holdout_idx):,} rows ({int(y_hold.sum())} anomalies)"
)

# ── Z-score ────────────────────────────────────────────────────────────
z_scores = np.abs(X).max(axis=1)

# ── IQR ────────────────────────────────────────────────────────────────
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
iqr_scores = ((X < lower) | (X > upper)).sum(axis=1).astype(np.float64)

# ── Isolation Forest ───────────────────────────────────────────────────
iso_forest = IsolationForest(
    n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1
).fit(X)
iso_scores = -iso_forest.score_samples(X)

# ── LOF (k=50 — larger than the expected ring size, see 4.3) ───────────
lof = LocalOutlierFactor(n_neighbors=50, contamination=0.01, novelty=False)
lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_

print("\nPer-detector baseline (holdout rows):")
z_m = print_metrics("Z-score", y_hold, z_scores[holdout_idx])
iqr_m = print_metrics("IQR", y_hold, iqr_scores[holdout_idx])
iso_m = print_metrics("Isolation Forest", y_hold, iso_scores[holdout_idx])
lof_m = print_metrics("LOF", y_hold, lof_scores[holdout_idx])


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: manual blends, engine ensemble, supervised second stage
# ════════════════════════════════════════════════════════════════════════

z_norm = normalise_scores(z_scores)
iqr_norm = normalise_scores(iqr_scores)
iso_norm = normalise_scores(iso_scores)
lof_norm = normalise_scores(lof_scores)

# ── (A) Equal-weight blend ─────────────────────────────────────────────
equal_blend = (z_norm + iqr_norm + iso_norm + lof_norm) / 4.0

# ── (B) AUC-weighted blend — weights from the REVIEW sample only ───────
y_rev = y[review_idx]
aucs = {
    "z": score_metrics(y_rev, z_scores[review_idx])["auc_roc"],
    "iqr": score_metrics(y_rev, iqr_scores[review_idx])["auc_roc"],
    "iso": score_metrics(y_rev, iso_scores[review_idx])["auc_roc"],
    "lof": score_metrics(y_rev, lof_scores[review_idx])["auc_roc"],
}
total_auc = sum(aucs.values())
weights = {k: v / total_auc for k, v in aucs.items()}
weighted_blend = (
    weights["z"] * z_norm
    + weights["iqr"] * iqr_norm
    + weights["iso"] * iso_norm
    + weights["lof"] * lof_norm
)

# ── (C) Rank-based blend ───────────────────────────────────────────────
z_rank = rank_normalise(z_scores)
iqr_rank = rank_normalise(iqr_scores)
iso_rank = rank_normalise(iso_scores)
lof_rank = rank_normalise(lof_scores)
rank_blend = (z_rank + iqr_rank + iso_rank + lof_rank) / 4.0

# ── (D) AnomalyDetectionEngine.ensemble_detect — unsupervised engine ──
# kailash-ml's engine runs each algorithm, normalises every score to
# [0, 1] and averages them (voting="score_average").
anomaly_df = pl.from_numpy(X, schema=_feature_cols)
engine_result = AnomalyDetectionEngine().ensemble_detect(
    anomaly_df,
    algorithms=["isolation_forest", "lof"],
    contamination=0.01,
    voting="score_average",
)
engine_blend = np.asarray(engine_result.combined_scores)

print("\nUnsupervised blends (holdout rows):")
equal_m = print_metrics("Equal-weight blend", y_hold, equal_blend[holdout_idx])
weighted_m = print_metrics("AUC-weighted blend", y_hold, weighted_blend[holdout_idx])
rank_m = print_metrics("Rank blend", y_hold, rank_blend[holdout_idx])
engine_m = print_metrics("Engine ensemble_detect", y_hold, engine_blend[holdout_idx])
print(
    f"  AUC weights (review sample): z={weights['z']:.3f} "
    f"iqr={weights['iqr']:.3f} iso={weights['iso']:.3f} lof={weights['lof']:.3f}"
)

print("\nPer anomaly type (all rows; 1.0 = perfect, 0.5 = chance):")
for det_name, det_scores in [
    ("Z-score", z_scores),
    ("Isolation Forest", iso_scores),
    ("LOF (k=50)", lof_scores),
    ("AUC-weighted blend", weighted_blend),
]:
    print_auc_by_type(det_name, frame, det_scores)

# ── (E) Supervised second stage — EnsembleEngine.blend() and .stack() ──
# Once analysts have labelled the review sample, a classifier can learn
# from the four detector scores WHICH detector to trust. EnsembleEngine
# is kailash-ml's classifier-ensemble API: blend() soft-votes fitted
# classifiers, stack() trains a meta-learner on their predictions. Both
# take a polars frame + target column and do their own 80/20 split.
#
# We fit the base classifiers on the REVIEW sample and pass the HOLDOUT
# frame to the engine, so the per-model "component" metrics the engine
# reports are measured on rows those classifiers never saw.
score_frame = pl.DataFrame(
    {
        "z_score": z_norm,
        "iqr_score": iqr_norm,
        "iso_score": iso_norm,
        "lof_score": lof_norm,
        "is_anomaly": y,
    }
)
review_frame = score_frame[review_idx]
holdout_frame = score_frame[holdout_idx]
X_review = review_frame.drop("is_anomaly").to_numpy()

logreg = LogisticRegression(max_iter=1000).fit(X_review, y_rev)
forest = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(
    X_review, y_rev
)

ensemble_engine = EnsembleEngine()
blend_result = ensemble_engine.blend(
    [logreg, forest],
    holdout_frame,
    "is_anomaly",
    weights=[0.5, 0.5],
    method="soft",
)
stack_result = ensemble_engine.stack(
    [logreg, forest],
    holdout_frame,
    "is_anomaly",
    meta_model_class="sklearn.linear_model.LogisticRegression",
)
print("\nSupervised second stage on detector scores (engine's own test split):")
print(f"  EnsembleEngine.blend  AUC={blend_result.metrics['auc']:.4f}")
print(f"  EnsembleEngine.stack  AUC={stack_result.metrics['auc']:.4f}")
for contrib in blend_result.component_contributions:
    print(
        f"    component {contrib['model_class'].rsplit('.', 1)[-1]:<24}"
        f" AUC={contrib['metrics']['auc']:.4f}"
    )
print(
    "  These use labels, so they are not comparable to the unsupervised"
    " blends above — they show what a small labelled review sample buys."
)


# ── Checkpoint ──────────────────────────────────────────────────────────
best_single_auc = max(
    z_m["auc_roc"], iqr_m["auc_roc"], iso_m["auc_roc"], lof_m["auc_roc"]
)
best_ensemble_auc = max(
    equal_m["auc_roc"], weighted_m["auc_roc"], rank_m["auc_roc"], engine_m["auc_roc"]
)
assert weighted_m["auc_roc"] > 0.5, "AUC-weighted blend should beat random"
assert abs(sum(weights.values()) - 1.0) < 1e-6, "AUC weights should sum to 1"
assert engine_blend.shape[0] == n_samples, "Engine must score every row"
assert 0.0 <= blend_result.metrics["auc"] <= 1.0, "blend() must report an AUC"
assert 0.0 <= stack_result.metrics["auc"] <= 1.0, "stack() must report an AUC"
print("\n[ok] Checkpoint passed — blends, engine ensemble and second stage computed\n")

lift = best_ensemble_auc - best_single_auc
if lift > 0.005:
    print(
        f"Best unsupervised blend beats the best single detector by {lift:+.4f} AUC."
    )
else:
    print(
        f"Best unsupervised blend is {lift:+.4f} AUC vs the best single detector:"
        " averaging a strong detector with weaker ones diluted its ranking."
        " Blending is a variance-reduction bet, not a guaranteed win."
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: comparison chart + monitoring
# ════════════════════════════════════════════════════════════════════════

comparison = {  # holdout-row metrics for every unsupervised detector
    "Z-score": {"AUC_ROC": z_m["auc_roc"], "Avg_Precision": z_m["avg_precision"]},
    "IQR": {"AUC_ROC": iqr_m["auc_roc"], "Avg_Precision": iqr_m["avg_precision"]},
    "Isolation Forest": {
        "AUC_ROC": iso_m["auc_roc"],
        "Avg_Precision": iso_m["avg_precision"],
    },
    "LOF": {"AUC_ROC": lof_m["auc_roc"], "Avg_Precision": lof_m["avg_precision"]},
    "Equal Blend": {
        "AUC_ROC": equal_m["auc_roc"],
        "Avg_Precision": equal_m["avg_precision"],
    },
    "AUC-Weighted": {
        "AUC_ROC": weighted_m["auc_roc"],
        "Avg_Precision": weighted_m["avg_precision"],
    },
    "Rank Blend": {
        "AUC_ROC": rank_m["auc_roc"],
        "Avg_Precision": rank_m["avg_precision"],
    },
    "Engine ensemble_detect": {
        "AUC_ROC": engine_m["auc_roc"],
        "Avg_Precision": engine_m["avg_precision"],
    },
}
comparison_path = write_comparison_chart(comparison, "ex4_anomaly_comparison.html")
print(f"Saved comparison chart: {comparison_path}")

import plotly.graph_objects as go
from sklearn.metrics import roc_curve

out_dir = Path("outputs") / "ex4_anomaly"

# ── (A) Ensemble score distribution: normal vs anomaly ─────────────────
fig_edist = go.Figure()
fig_edist.add_trace(
    go.Histogram(
        x=weighted_blend[y == 0],
        name="Normal",
        opacity=0.7,
        nbinsx=60,
        marker_color="#636EFA",
    )
)
fig_edist.add_trace(
    go.Histogram(
        x=weighted_blend[y == 1],
        name="Anomaly",
        opacity=0.7,
        nbinsx=60,
        marker_color="#EF553B",
    )
)
fig_edist.update_layout(
    title="AUC-Weighted Ensemble Score Distribution",
    xaxis_title="Blended Anomaly Score",
    yaxis_title="Count",
    barmode="overlay",
)
edist_path = out_dir / "04_ensemble_score_distribution.html"
fig_edist.write_html(str(edist_path))
print(f"[viz] Ensemble score distribution: {edist_path}")

# ── (B) ROC curves overlay: all methods on one plot ───────────────────
all_detectors = {
    "Z-score": (z_scores, z_m),
    "IQR": (iqr_scores, iqr_m),
    "Isolation Forest": (iso_scores, iso_m),
    "LOF": (lof_scores, lof_m),
    "Equal Blend": (equal_blend, equal_m),
    "AUC-Weighted": (weighted_blend, weighted_m),
    "Rank Blend": (rank_blend, rank_m),
}
fig_roc = go.Figure()
for det_name, (det_scores, det_m) in all_detectors.items():
    fpr, tpr, _ = roc_curve(y_hold, det_scores[holdout_idx])
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{det_name} (AUC={det_m['auc_roc']:.3f})",
        )
    )
fig_roc.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="grey"),
        name="Random",
        showlegend=False,
    )
)
fig_roc.update_layout(
    title="ROC Curves (holdout rows): All Detectors and Ensembles",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
)
roc_overlay_path = out_dir / "04_roc_overlay.html"
fig_roc.write_html(str(roc_overlay_path))
print(f"[viz] ROC overlay: {roc_overlay_path}")

# Precision at key recall levels for the AUC-weighted blend
print("\nPrecision at key recall levels (AUC-weighted blend, holdout rows):")
for target_recall in [0.50, 0.70, 0.80, 0.90]:
    p, t = precision_at_recall(y_hold, weighted_blend[holdout_idx], target_recall)
    print(f"  Recall={target_recall:.0%}  precision={p:.4f}  threshold={t:.4f}")

# Production monitoring demo: anomaly rate over 10 windows of rows. The
# threshold is chosen on the REVIEW sample (recall >= 80%) and then held
# fixed, as it would be in production. Rows are shuffled, so the windows
# here are statistically identical — a flat line is the expected result.
window_size = max(1, n_samples // 10)
_, decision_threshold = precision_at_recall(
    y[review_idx], weighted_blend[review_idx], 0.80
)
anomaly_rates: list[float] = []
for i in range(10):
    start = i * window_size
    end = start + window_size
    window = weighted_blend[start:end]
    anomaly_rates.append(float((window > decision_threshold).mean()))
monitoring_path = write_monitoring_chart(anomaly_rates, "ex4_monitoring.html")
print(f"Saved monitoring chart: {monitoring_path}")
print(f"\nMonitoring windows (anomaly rate @ threshold={decision_threshold:.3f}):")
for i, rate in enumerate(anomaly_rates, 1):
    print(f"  Window {i:>2}: {rate:.2%}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Production Anomaly Monitoring at a Singapore Digital Bank
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore digital bank scores every new
# credit application with a multi-detector anomaly score. It needs a
# score that:
#   - Runs inside its online decision SLA
#   - Comes with a per-flag explanation its model-risk reviewers accept
#   - Is monitored for population drift (festive spend, policy changes)
#
# How the pieces from this exercise fit:
#   - Unsupervised detectors and blends rank applications on day one,
#     before any labels exist. Measure them per anomaly type: on this
#     data the best single detector may beat the blends.
#   - Each flag can explain itself ("Z-score=6.1 on savings_balance, LOF
#     percentile=99.7") because every component score is kept.
#   - Once analysts have labelled a few thousand reviewed cases, a
#     supervised second stage (EnsembleEngine.blend / .stack over the
#     detector scores) learns which detector to trust for which pattern.
#
# BUSINESS IMPACT (illustrative arithmetic, not reported figures): at
# 10,000 applications a day, every 0.1 percentage point of false-positive
# rate is 10 extra manual reviews a day. If a review costs ~S$15 of
# analyst time, cutting the false-positive rate from 0.6% to 0.2% saves
# 40 reviews/day x S$15 x 365 = ~S$219,000 a year in review effort,
# before counting losses avoided on caught applications.
#
# REALITY CHECK: an anomaly score is NOT a default model. The line
# printed below scores the blend against the REAL repayment outcome of
# the real applications — expect it to be close to 0.5.
#
# MONITORING: the anomaly rate plotted above is itself a drift signal.
# A sudden jump in one window doesn't mean fraudsters got faster — it
# usually means the feature distribution moved and the detectors no
# longer fit. That is the trigger to investigate and retrain.

default_auc = default_reality_check(frame, weighted_blend)
print(
    f"\nReality check — AUC-weighted blend vs REAL default outcome: "
    f"AUC={default_auc:.3f} (0.5 = no relationship)"
)


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's leaderboard to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Eight detectors (4 base + 4 unsupervised blends) on the same holdout
# rows, the supervised second stage, the AUC weights and the monitoring-
# window anomaly rates. This section persists the leaderboard so a
# student can compare lessons 01-04 side by side.

precision_at_recall_metrics: dict[str, float] = {}
for tr in (0.50, 0.70, 0.80, 0.90):
    p, _t = precision_at_recall(y_hold, weighted_blend[holdout_idx], tr)
    label = str(tr).replace(".", "_")
    precision_at_recall_metrics[f"weighted_p_at_r_{label}"] = _finite(p)

monitoring_metrics: dict[str, float] = {
    f"monitor_window_{i:02d}_rate": _finite(rate)
    for i, rate in enumerate(anomaly_rates, start=1)
}

track_run(
    tracker,
    exp_name,
    run_name="ensemble_blending",
    params={
        "n_samples": n_samples,
        "n_estimators_iso": 200,
        "n_neighbors_lof": 50,
        "review_fraction": 0.3,
        "contamination": 0.01,
        "anomaly_rate": float(y.mean()),
        "decision_threshold": float(decision_threshold),
        "monitoring_windows": len(anomaly_rates),
    },
    scalar_metrics={
        # Per-detector baselines (4)
        "z_auc_roc": _finite(z_m["auc_roc"]),
        "z_avg_precision": _finite(z_m["avg_precision"]),
        "iqr_auc_roc": _finite(iqr_m["auc_roc"]),
        "iqr_avg_precision": _finite(iqr_m["avg_precision"]),
        "iso_auc_roc": _finite(iso_m["auc_roc"]),
        "iso_avg_precision": _finite(iso_m["avg_precision"]),
        "lof_auc_roc": _finite(lof_m["auc_roc"]),
        "lof_avg_precision": _finite(lof_m["avg_precision"]),
        # Blend variants (4)
        "equal_blend_auc_roc": _finite(equal_m["auc_roc"]),
        "equal_blend_avg_precision": _finite(equal_m["avg_precision"]),
        "weighted_blend_auc_roc": _finite(weighted_m["auc_roc"]),
        "weighted_blend_avg_precision": _finite(weighted_m["avg_precision"]),
        "rank_blend_auc_roc": _finite(rank_m["auc_roc"]),
        "rank_blend_avg_precision": _finite(rank_m["avg_precision"]),
        "engine_blend_auc_roc": _finite(engine_m["auc_roc"]),
        "engine_blend_avg_precision": _finite(engine_m["avg_precision"]),
        # Supervised second stage (EnsembleEngine)
        "ensemble_engine_blend_auc": _finite(blend_result.metrics["auc"]),
        "ensemble_engine_stack_auc": _finite(stack_result.metrics["auc"]),
        "default_reality_check_auc": _finite(default_auc),
        # AUC weights
        "weight_z": _finite(weights["z"]),
        "weight_iqr": _finite(weights["iqr"]),
        "weight_iso": _finite(weights["iso"]),
        "weight_lof": _finite(weights["lof"]),
        # Headline winner
        "best_single_auc_roc": _finite(best_single_auc),
        "best_ensemble_auc_roc": _finite(best_ensemble_auc),
        "ensemble_lift_over_best_single": _finite(best_ensemble_auc - best_single_auc),
    }
    | precision_at_recall_metrics
    | monitoring_metrics,
)
print(
    f"\n  [tracked] 8-detector leaderboard + monitoring logged to {exp_name} "
    f"run='ensemble_blending'\n"
)
print(
    "  AnomalyDetectionEngine.ensemble_detect() is the engine version of the"
    " manual blends; EnsembleEngine.blend()/.stack() is the supervised"
    " stage. Runs 01-04 of m4_anomaly_zoo are now comparable via"
    " tracker.list_runs().\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Normalised four anomaly scores onto a comparable [0, 1] scale
  [x] Built equal-weight, AUC-weighted, and rank-based manual blends
  [x] Ran AnomalyDetectionEngine.ensemble_detect() as the engine blend
  [x] Trained a supervised second stage with EnsembleEngine.blend()/.stack()
  [x] Compared all 8 unsupervised detectors on holdout AUC-ROC and AUC-PR
  [x] Read precision-at-recall as the operator-facing metric
  [x] Built a production monitoring chart of anomaly rate over time
  [x] Framed a digital-bank deployment with illustrative cost arithmetic

  KEY INSIGHT: A blend is a variance-reduction bet, not a guaranteed
  win — measure it against the best single detector on rows it was not
  tuned on. Production anomaly detection combines detectors, a
  monitoring layer that watches the anomaly rate for drift, and a
  review queue whose labels feed a supervised second stage.

  This completes Exercise 4. Next: Exercise 5 discovers structure in
  transaction patterns with association rule mining.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
