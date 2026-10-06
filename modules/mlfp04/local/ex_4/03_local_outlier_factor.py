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
#   - Explain when LOF beats Isolation Forest, and when a group of
#     anomalies MASKS itself from LOF
#
# PREREQUISITES: 4.2 (Isolation Forest).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why comparing LOCAL densities catches embedded anomalies
#   2. Build — sweep n_neighbors and pick the best value
#   3. Train — fit LOF and extract negative_outlier_factor_
#   4. Visualise — ROC curve (written to outputs/)
#   5. Apply — application-ring screening at a Singapore lender
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from shared.mlfp04.ex_4 import (
    _finite,
    auc_by_type,
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

# Per-n_neighbors sweep results captured below for the TRACK section.
nbrs_sweep: dict[int, dict[str, float]] = {}


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Local Density Beats Global Distance
# ════════════════════════════════════════════════════════════════════════
# LOF answers a different question than Isolation Forest. IF asks "how
# hard is this point to isolate from the rest of the data?" LOF asks
# "how does this point's neighbourhood density compare to ITS NEIGHBOURS'
# neighbourhood density?"
#
# Concretely, for a point p:
#   1. Find its k nearest neighbours N_k(p).
#   2. Measure local density around p: how close are those neighbours?
#   3. Measure local density around EACH neighbour.
#   4. LOF(p) = mean( density(neighbour_i) / density(p) ) for i in N_k(p).
#
# LOF ~ 1.0 means p has roughly the same density as its neighbours — it
# belongs to a cluster. LOF >> 1.0 means p sits in a sparser pocket than
# its neighbours — it's an outlier, EVEN IF it's surrounded by other
# points globally.
#
# WHY IT BEATS ISOLATION FOREST SOMETIMES: in data with varying cluster
# densities (some clusters dense, others sparse), a single global rule
# ("far from everything = outlier") fails. LOF is the right tool when
# anomalies sit in a SPARSE POCKET next to a dense cluster they don't
# belong to.
#
# THE MASKING TRAP: a point INSIDE a tight group has LOF ~ 1 (its density
# matches its neighbours') — and if the group is DENSER than its
# surroundings, LOF < 1, i.e. "more normal than normal". So a coordinated
# group of anomalies with at least n_neighbors members is invisible to
# LOF: the members are each other's neighbours. n_neighbors must exceed
# the size of the largest anomalous group you want to catch.
#
# COST: LOF is O(n^2) at worst because it needs nearest-neighbour queries.
# For n > 200K rows, sub-sample or switch to an approximate NN backend.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: sweep n_neighbors
# ════════════════════════════════════════════════════════════════════════

X, y, _feature_cols, frame = load_dataset()
n_samples, n_features = X.shape
print("\n" + "=" * 70)
print("  Local Outlier Factor (LOF)")
print("=" * 70)
print(
    f"Rows: {n_samples:,} | Features: {n_features} | "
    f"Anomalies: {int(y.sum()):,} ({y.mean():.2%})"
)

print("\nn_neighbors sweep (clustered = AUC on the 40-member injected group):")
for n_nbrs in [10, 20, 30, 50, 80]:
    # TODO: Build LocalOutlierFactor with n_neighbors=n_nbrs,
    # contamination=0.01, novelty=False
    lof_test = ____

    # TODO: Run fit_predict(X) to get labels
    labels_test = ____

    # TODO: Turn negative_outlier_factor_ into an anomaly score where
    # HIGHER = more anomalous
    scores_test = ____
    m = score_metrics(y, scores_test)
    # TODO: AUC on the injected 'clustered' group only
    # (hint: auc_by_type(frame, scores) returns a dict keyed by type)
    clustered_auc = ____
    n_flagged = int((labels_test == -1).sum())
    nbrs_sweep[n_nbrs] = {
        "auc_roc": m["auc_roc"],
        "avg_precision": m["avg_precision"],
        "clustered_auc": clustered_auc,
        "n_flagged": float(n_flagged),
    }
    print(
        f"  n_neighbors={n_nbrs:<3}  AUC-ROC={m['auc_roc']:.4f}  "
        f"AP={m['avg_precision']:.4f}  clustered={clustered_auc:.3f}  "
        f"flagged={n_flagged:,}"
    )

# TODO: list the n_neighbors values whose clustered AUC is below 0.5
masked_ks = ____
if masked_ks:
    print(
        f"  -> At n_neighbors in {masked_ks} the 40-member group scores BELOW"
        " chance: its members are each other's neighbours, so the group"
        " looks dense (LOF < 1) — the masking trap from the theory above."
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit LOF with the chosen n_neighbors
# ════════════════════════════════════════════════════════════════════════
# n_neighbors=20 is the textbook default, but the choice is a DOMAIN
# decision: it must exceed the largest coordinated group you want to
# catch. Suppose the fraud team expects application rings of up to ~40
# near-identical submissions — then k=50. (We choose k from that domain
# assumption, not from the label-based AUCs above, which a real
# unsupervised deployment would not have.) Larger k also drifts toward a
# global density estimate, so do not raise it further than needed.

LOF_K = 50
# TODO: fit LocalOutlierFactor(n_neighbors=LOF_K, contamination=0.01,
# novelty=False) with fit_predict, then build lof_scores so that
# HIGHER = more anomalous. sklearn's negative_outlier_factor_ is -LOF
# (more negative = MORE anomalous).
lof = ____
lof_labels = ____
lof_scores = ____

print(f"\nFinal LOF (n_neighbors={LOF_K}):")
lof_metrics = print_metrics("LOF", y, lof_scores)
print(f"  Predicted anomalies: {int((lof_labels == -1).sum()):,}")
print(f"  True anomalies:      {int(y.sum()):,}")

iso_scores = -(
    IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
    .fit(X)
    .score_samples(X)
)
print("\nPer anomaly type (1.0 = perfect, 0.5 = chance):")
lof_by_type = print_auc_by_type(f"LOF (k={LOF_K})", frame, lof_scores)
iso_by_type = print_auc_by_type("Isolation Forest (4.2)", frame, iso_scores)


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    lof_metrics["auc_roc"] > 0.5
), f"LOF AUC-ROC {lof_metrics['auc_roc']:.4f} should beat random"
assert lof_scores.std() > 0, "LOF scores should vary across rows"
assert lof_scores.shape[0] == n_samples, "Score length must match row count"
print("\n[ok] Checkpoint passed — LOF scored all rows\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: ROC curve + LOF score scatter + distance distribution
# ════════════════════════════════════════════════════════════════════════
roc_path = write_roc_chart(y, lof_scores, "LOF", "ex4_roc_lof.html")
print(f"Saved ROC chart: {roc_path}")

import plotly.graph_objects as go
from pathlib import Path

out_dir = Path("outputs") / "ex4_anomaly"
out_dir.mkdir(parents=True, exist_ok=True)

# ── (A) LOF scores scatter, coloured by anomaly label ──────────────────
# Plot the first two principal features vs LOF score to show where
# the high-LOF points cluster relative to the data.
fig_scatter = go.Figure()
fig_scatter.add_trace(
    go.Scatter(
        x=X[y == 0, 0],
        y=X[y == 0, 1],
        mode="markers",
        marker=dict(
            size=4,
            color=lof_scores[y == 0],
            colorscale="Blues",
            opacity=0.5,
        ),
        name="Normal",
    )
)
fig_scatter.add_trace(
    go.Scatter(
        x=X[y == 1, 0],
        y=X[y == 1, 1],
        mode="markers",
        marker=dict(
            size=7,
            color=lof_scores[y == 1],
            colorscale="Reds",
            colorbar=dict(title="LOF Score", x=1.02),
            opacity=0.9,
        ),
        name="Anomaly",
    )
)
fig_scatter.update_layout(
    title=(
        f"LOF Scores: {_feature_cols[0]} vs {_feature_cols[1]} "
        "(colour = LOF score)"
    ),
    xaxis_title=f"{_feature_cols[0]} (standardised)",
    yaxis_title=f"{_feature_cols[1]} (standardised)",
)
scatter_path = out_dir / "03_lof_scatter.html"
fig_scatter.write_html(str(scatter_path))
print(f"[viz] LOF scatter: {scatter_path}")

# ── (B) LOF score distribution: normal vs anomaly ─────────────────────
fig_dist = go.Figure()
fig_dist.add_trace(
    go.Histogram(
        x=lof_scores[y == 0],
        name="Normal",
        opacity=0.7,
        nbinsx=60,
        marker_color="#636EFA",
    )
)
fig_dist.add_trace(
    go.Histogram(
        x=lof_scores[y == 1],
        name="Anomaly",
        opacity=0.7,
        nbinsx=60,
        marker_color="#EF553B",
    )
)
median_normal = float(np.median(lof_scores[y == 0]))
median_anomaly = float(np.median(lof_scores[y == 1]))
fig_dist.add_vline(
    x=median_normal,
    line_dash="dash",
    line_color="#636EFA",
    annotation_text=f"Normal median={median_normal:.2f}",
)
fig_dist.add_vline(
    x=median_anomaly,
    line_dash="dash",
    line_color="#EF553B",
    annotation_text=f"Anomaly median={median_anomaly:.2f}",
)
fig_dist.update_layout(
    title="LOF Score Distribution: Normal vs Anomaly",
    xaxis_title="LOF Score (higher = more anomalous)",
    yaxis_title="Count",
    barmode="overlay",
)
dist_path = out_dir / "03_lof_score_distribution.html"
fig_dist.write_html(str(dist_path))
print(f"[viz] LOF score distribution: {dist_path}")

print("\nLOF vs Isolation Forest on this dataset (computed):")
for t in lof_by_type:
    winner = "LOF" if lof_by_type[t] > iso_by_type[t] else "Isolation Forest"
    print(
        f"  {t:<11} LOF={lof_by_type[t]:.3f}  IF={iso_by_type[t]:.3f}"
        f"  -> {winner} ranks this type higher"
    )
print("  A 'dependency' row (stitched from several real applications)")
print("  sits in a sparse pocket next to the dense mass of consistent")
print("  applications — exactly what a LOCAL density ratio detects.")
print("  Use BOTH detectors, then blend — see 04_ensemble_blending.py.")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Application-Ring Screening at a Singapore Lender
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore lender's fraud team screens credit
# applications for two LOCAL patterns:
#   1. Stitched applications that sit in a SPARSE POCKET beside the dense
#      mass of consistent applications (individually plausible fields,
#      inconsistent combination). LOF > 1 — this is LOF's home ground.
#   2. Application RINGS — dozens of near-identical submissions. Inside
#      the ring every point's neighbours are other ring members, so the
#      local density ratio is ~1 or below: LOF calls them NORMAL unless
#      n_neighbors is larger than the ring. The sweep above shows this
#      directly; a ring-size assumption (k=50 for rings up to ~40) is
#      what makes LOF usable for pattern 2.
#
# Why LOF is the right tool for pattern 1:
#   - The suspicious applications are not far from the data globally;
#     they are odd only relative to their LOCAL neighbourhood
#   - Isolation Forest asks a global question and ranks them lower (see
#     the per-type comparison above)
#
# BUSINESS IMPACT (illustrative assumptions, not reported figures): if
# the team reviews the top 0.5% of applications by LOF score each week,
# the review load is a few hundred cases instead of the whole book, and
# every stitched or ring application stopped before disbursement avoids
# the full loan amount. Check the per-type AUCs before relying on it.
#
# LIMITATIONS: LOF needs nearest-neighbour queries — roughly O(n^2) in
# high dimensions without an index. For tens of millions of rows,
# sub-sample per run or use an approximate-NN backend. Exercise 4.4
# blends LOF with the other detectors.


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Sweep keys: lof_k{N}_auc_roc / lof_k{N}_avg_precision / lof_k{N}_n_flagged
# Integer N is regex-safe directly — no _slug() needed.

sweep_metrics: dict[str, float] = {}
for n_nbrs, stats in nbrs_sweep.items():
    sweep_metrics[f"lof_k{n_nbrs}_auc_roc"] = _finite(stats["auc_roc"])
    sweep_metrics[f"lof_k{n_nbrs}_avg_precision"] = _finite(stats["avg_precision"])
    sweep_metrics[f"lof_k{n_nbrs}_clustered_auc"] = _finite(stats["clustered_auc"])
    sweep_metrics[f"lof_k{n_nbrs}_n_flagged"] = stats["n_flagged"]

# TODO: call track_run with run_name="local_outlier_factor". Headline
# scalars: lof_auc_roc, lof_avg_precision (from lof_metrics — wrap in
# _finite), lof_n_predicted_anomalies, lof_normal_median (from median_normal),
# lof_anomaly_median. |-merge with sweep_metrics.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "n_samples": n_samples,
        "n_features": n_features,
        "best_n_neighbors": LOF_K,
        "contamination": 0.01,
        "anomaly_rate": float(y.mean()),
    },
    scalar_metrics={
        "lof_auc_roc": _finite(lof_metrics["auc_roc"]),
        "lof_avg_precision": ____,
        "lof_n_predicted_anomalies": float(int((lof_labels == -1).sum())),
        "lof_normal_median": _finite(median_normal),
        "lof_anomaly_median": _finite(median_anomaly),
    }
    | sweep_metrics,
)
print(
    f"\n  [tracked] LOF n_neighbors sweep + final-fit logged to {exp_name} "
    f"run='local_outlier_factor'\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — AnomalyDetectionEngine.detect(algorithm='lof')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's AnomalyDetectionEngine wraps LOF under the same .detect()
# surface used in lesson 02 for isolation_forest. Same engine, different
# `algorithm=` string; extra keyword arguments (n_neighbors) are passed
# to the underlying LocalOutlierFactor. Lesson 04 blends the detectors.

import polars as pl

from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine

anomaly_df = pl.from_numpy(X, schema=_feature_cols)
# TODO: Instantiate AnomalyDetectionEngine and call .detect on anomaly_df
# with algorithm='lof', contamination=0.01 and n_neighbors=LOF_K (extra
# keyword arguments are passed to the underlying LocalOutlierFactor).
det = ____
fit_result = ____
fit_metrics = score_metrics(y, np.asarray(fit_result.scores))
print(
    f"  AnomalyDetectionEngine.detect(lof, n_neighbors={LOF_K}): "
    f"AUC-ROC={fit_metrics['auc_roc']:.4f}  "
    f"AP={fit_metrics['avg_precision']:.4f}  "
    f"n_anomalies={fit_result.n_anomalies}"
)
print(
    f"  Hand-rolled AUC-ROC (Task 3): {lof_metrics['auc_roc']:.4f}  "
    "— same algorithm under one surface; the engine's contamination knob"
    " maps directly to LocalOutlierFactor's parameter.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] LOF as a density-ratio test, not a distance test
  [x] n_neighbors as the "locality" knob
  [x] How LOF finds sparse-pocket anomalies beside dense clusters
  [x] The masking trap: groups with >= n_neighbors members look normal
  [x] The O(n^2) scalability limit and when to sub-sample
  [x] Framed an application-ring screening scenario (illustrative figures)

  KEY INSIGHT: Different anomaly detectors answer different questions.
  LOF asks a LOCAL question ("is this point in a sparser pocket than
  its neighbours?"). Isolation Forest asks a GLOBAL question ("how easy
  is this point to separate from everything?"). Neither sees every
  anomaly type, which is why real pipelines combine them.

  Next: 04_ensemble_blending.py — combine Z-score + IQR + IF + LOF into
  a single ensemble score using kailash-ml EnsembleEngine.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
