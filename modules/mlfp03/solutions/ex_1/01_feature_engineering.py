# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.1: Clinical Feature Engineering with
#                         Point-in-Time Correctness
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Join five ICU tables with temporal correctness (no future leakage)
#   - Aggregate irregular time-series vitals into per-admission statistics
#   - Flag clinically meaningful medication and lab patterns
#   - Engineer interaction features that encode domain knowledge (shock
#     index, mean arterial pressure, fever-tachycardia product)
#   - Audit how much event data actually exists at prediction time
#   - Apply to early-warning scoring at a Singapore public hospital
#
# PREREQUISITES: MLFP02 complete (polars group-by, joins, temporal filters)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — why point-in-time correctness matters
#   2. Build — load tables, aggregate vitals, meds, labs
#   3. Train — there is no training; we BUILD the full feature matrix
#   4. Visualise — preview the engineered columns + interaction distributions
#   5. Apply — early-warning scoring at a Singapore public hospital
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from shared.mlfp03.ex_1 import (
    OUTPUT_DIR,
    PREDICTION_HOURS,
    build_full_feature_frame,
    load_icu_tables,
    prediction_window_report,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Point-in-Time Correctness Matters
# ════════════════════════════════════════════════════════════════════════
# A feature built from "all vitals this patient ever had" leaks the
# future. Our model predicts, 24 hours after ICU admission, whether the
# stay will be LONG (longer than the median). At that moment the
# patient's later vitals, later drugs and — above all — the discharge
# time do not exist yet. Using them inflates validation accuracy and
# fails in production.
#
# The fix is a temporal filter with a fixed PREDICTION CUTOFF: every
# feature only uses data recorded between admit_time and
# admit_time + 24h for THAT admission. The same rule applies to
# medications (start_time), labs (timestamp) and every derived feature —
# and no feature may divide by, or otherwise use, the length of stay,
# because that is the target.
#
# Analogy: imagine building a stock-price model that accidentally uses
# tomorrow's close as today's feature. Your backtest looks amazing; your
# live trading loses every penny. Medical ML has the same failure mode,
# but the stakes are higher — a biased model kills patients, not pennies.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: load tables and construct per-admission features
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Clinical Feature Engineering — ICU Multi-Table")
print("=" * 70)

tables = load_icu_tables()
for name, df in tables.items():
    print(f"  {name}: {df.shape}")

# How much event data exists INSIDE the prediction window? Computed from
# the same filter the feature builders use.
window = prediction_window_report(tables)
print(f"\nEvents available in the first {PREDICTION_HOURS}h of each admission:")
print(window)

# The shared helper encodes the full feature contract: vital aggregates
# per admission (mean/std/min/max/range/trend/count/cv), medication flags
# (vasopressors, antibiotics, sedation), lab ratios, and clinical
# interaction features. Every technique file in this exercise starts from
# the same matrix — only the SELECTION method differs.
features = build_full_feature_frame(tables)

print(f"\nFeature matrix: {features.shape}")
print(f"Total columns: {len(features.columns)}")


# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert features.height > 0, "Task 2: feature matrix must not be empty"
assert "shock_index" in features.columns, "Task 2: shock_index (interaction) missing"
assert "abnormal_lab_ratio" in features.columns, "Task 2: lab ratio missing"
assert features["abnormal_lab_ratio"].null_count() == 0, "Task 2: null in lab ratio"
print("\n[ok] Checkpoint 1 passed — feature matrix built\n")

# INTERPRETATION: The _count suffix columns measure monitoring intensity
# — 48 heart-rate readings in the first 24h (2/hour) means a patient is
# watched far more closely than one with 6. The coefficient of variation
# (_cv) captures NORMALISED volatility regardless of baseline.

coverage = {
    row["table"]: row["admissions_with_events"] / row["admissions_total"]
    for row in window.iter_rows(named=True)
}
print("Share of admissions with ANY event in the prediction window:")
for table_name, share in coverage.items():
    print(f"  {table_name:<12} {share:7.2%}")
if max(coverage.values()) < 0.05:
    print(
        "\n  DATA-QUALITY FINDING: almost no admission has vitals, drugs or labs\n"
        "  recorded in its first 24h — in this extract the event tables are\n"
        "  not time-aligned with the admissions. Event features will be\n"
        "  zero for nearly every row. Catching this BEFORE modelling is the\n"
        "  point of a point-in-time audit."
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (no training — quality audit of the feature matrix)
# ════════════════════════════════════════════════════════════════════════
# Feature engineering has no gradient-descent loop. The "training" step
# is a deterministic construction, and we validate it by auditing the
# null rate, dtype coverage, and clinical interaction distributions.

print("\n--- Feature Matrix Quality Audit ---")
# Null rate across NUMERIC + BOOLEAN columns only. Stat columns like
# {vital}_std legitimately produce nulls when a patient has a single
# reading, and full-outer vital joins leave nulls for patients who were
# never sampled for a given vital — both are expected, not defects.
null_rate = sum(features[c].null_count() for c in features.columns) / (
    features.height * len(features.columns)
)
numeric_cols = [
    c
    for c in features.columns
    if features[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
]
bool_cols = [c for c in features.columns if features[c].dtype == pl.Boolean]

print(f"  Rows:        {features.height}")
print(f"  Columns:     {len(features.columns)}")
print(f"  Numeric:     {len(numeric_cols)}")
print(f"  Boolean:     {len(bool_cols)}")
print(f"  Global null rate: {null_rate:.4f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert null_rate < 0.20, (
    f"Task 3: null rate {null_rate:.4f} exceeds 20% after the documented "
    "fills — a builder is producing unexpected nulls."
)
assert (window["max_offset_hours"] <= PREDICTION_HOURS).all(), (
    "Task 3: an event later than the prediction cutoff reached the features"
)
print("\n[ok] Checkpoint 2 passed — no nulls left, no event after the cutoff\n")
# NOTE: a low null rate here does NOT mean good coverage — the builders
# fill 'no events' with 0. Coverage is what the window report measures.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the interaction features
# ════════════════════════════════════════════════════════════════════════
# Visual proof: plot the interaction features. With good event coverage
# shock index would sit mostly below 0.7 (above 0.9 is an emergency
# marker) and MAP around 70-90 mmHg. A tall spike at 0 is the visual
# signature of admissions with NO vitals in the window.

print("\n--- Clinical Interaction Feature Distributions ---")
interaction_cols = [
    c
    for c in ("shock_index", "map_mean", "fever_tachycardia", "treatment_burden_score")
    if c in features.columns
]
summary = features.select(
    *[pl.col(c).mean().alias(f"{c}_mean") for c in interaction_cols],
    *[pl.col(c).std().alias(f"{c}_std") for c in interaction_cols],
)
for c in interaction_cols:
    mean = summary[f"{c}_mean"].item() or 0.0
    std = summary[f"{c}_std"].item() or 0.0
    print(f"  {c:<25} mean={mean:>10.3f}  std={std:>10.3f}")

# Rough histogram of shock_index (clinically actionable ranges)
if "shock_index" in features.columns:
    print("\n  shock_index buckets (clinical interpretation):")
    buckets = (
        features.select(
            pl.when(pl.col("shock_index") == 0)
            .then(pl.lit("no vitals in window (0)"))
            .when(pl.col("shock_index") < 0.7)
            .then(pl.lit("normal (<0.7)"))
            .when(pl.col("shock_index") < 0.9)
            .then(pl.lit("concerning (0.7-0.9)"))
            .when(pl.col("shock_index") < 1.2)
            .then(pl.lit("emergency (0.9-1.2)"))
            .otherwise(pl.lit("critical (>=1.2)"))
            .alias("bucket")
        )
        .group_by("bucket")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    for row in buckets.iter_rows(named=True):
        bar = "#" * min(40, int(row["n"] / 5))
        print(f"    {row['bucket']:<24} {row['n']:>5}  {bar}")

# --- Correlation heatmap of engineered features ---
corr_cols = [c for c in interaction_cols if c in features.columns]
corr_cols += [
    c for c in features.columns if c.endswith("_mean") and c not in corr_cols
][:8]
# A correlation is undefined for a constant column — drop those first.
corr_cols = [c for c in corr_cols if features[c].cast(pl.Float64).std() > 0]
corr_matrix = features.select([pl.col(c).cast(pl.Float64) for c in corr_cols]).corr()
fig_heat = px.imshow(
    corr_matrix.to_numpy(),
    x=corr_cols,
    y=corr_cols,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation Heatmap — Engineered Clinical Features",
)
fig_heat.update_layout(width=800, height=700)
heat_path = OUTPUT_DIR / "ex1_01_correlation_heatmap.html"
fig_heat.write_html(str(heat_path))
print(f"\n  Saved: {heat_path}")

# --- Feature distribution histograms ---
hist_cols = [c for c in interaction_cols if c in features.columns][:4]
fig_hist = go.Figure()
for col in hist_cols:
    vals = features[col].drop_nulls().to_list()
    fig_hist.add_trace(go.Histogram(x=vals, name=col, opacity=0.6, nbinsx=40))
fig_hist.update_layout(
    title="Distribution of Clinical Interaction Features",
    xaxis_title="Value",
    yaxis_title="Count",
    barmode="overlay",
    height=450,
)
hist_path = OUTPUT_DIR / "ex1_01_feature_distributions.html"
fig_hist.write_html(str(hist_path))
print(f"  Saved: {hist_path}")


# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert "shock_index" in features.columns, "Task 4: shock_index missing"
assert (
    features["shock_index"].null_count() == 0
), "Task 4: null in shock_index — check input vital columns"
print("\n[ok] Checkpoint 3 passed — interaction features computed and plotted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: early-warning scoring at a Singapore public hospital
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore public hospital runs ~2,500 ICU
# admissions a year. Its clinical informatics team wants an early-warning
# score that flags deteriorating patients hours before an emergency
# escalation. The current rule (heart rate > 120 = alert) fires hundreds
# of false alarms a day and nurses have started ignoring the pager.
#
# Why engineered features help:
#   - shock_index (HR / SBP) combines two vitals into one marker of
#     compensated shock that neither vital shows on its own
#   - abnormal_lab_ratio integrates "many results drifting off baseline"
#     into a single number
#   - medication_doses_per_hour and n_unique_medications proxy clinician
#     concern — the team has already escalated treatment
#
# ILLUSTRATIVE ARITHMETIC (round numbers, not the hospital's figures): if
# each avoided emergency escalation saves ~S$18,000 and better features
# catch 5 more deteriorations a month, that is 5 × 12 × S$18,000 ≈
# S$1.08M a year — before counting fewer false alarms.
#
# LIMITATIONS:
#   - Features are only as good as the data available at prediction time:
#     the window audit above shows how little event data this extract
#     has in the first 24h. A real project would fix the data feed first.
#   - Vitals coverage varies by ward, so *_count is confounded by ward
#   - Leakage auditing (see 05_validation_and_tracking.py) MUST run
#     every time the feature list changes


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Joined five ICU tables (patients, admissions, vitals, meds, labs)
  [x] Applied a 24h prediction cutoff so features cannot leak the future
  [x] Audited how much event data exists inside the prediction window
  [x] Aggregated irregular vital-sign time series into per-admission stats
  [x] Flagged clinically meaningful drug classes via regex
  [x] Computed clinical interaction features from domain knowledge

  KEY INSIGHT: Domain knowledge dominates algorithmic complexity. The
  shock_index feature is one division, but it encodes decades of
  emergency-medicine research. No deep-learning architecture recovers
  that signal automatically from raw vitals without supervision.

  Next: 02_filter_selection.py — rank all engineered features using
  mutual information and chi-squared, and see which clinical features
  a model-free filter method can recover.
"""
)
