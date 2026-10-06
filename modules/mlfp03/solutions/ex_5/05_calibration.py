# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.5: Calibration (Platt + Isotonic) & Final Comparison
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - What calibration means: p_predicted = 0.2 means 20% default in reality
#   - Platt scaling (parametric logistic post-processing)
#   - Isotonic regression (non-parametric step-function post-processing)
#   - How to read a reliability diagram and spot under/over-confidence
#   - How to pick the production-ready strategy from all prior techniques
#
# PREREQUISITES: 01-04 in this directory (probabilities saved under
# outputs/ex5_imbalance/strategy_probabilities.parquet)
# ESTIMATED TIME: ~35 min
#
# 5-PHASE STRUCTURE:
#   Theory   — calibration intuition + why it matters for loan pricing
#   Build    — calibrate the cost-sensitive model with TrainingPipeline.calibrate
#   Train    — fit Platt and Isotonic variants
#   Visualise — reliability diagrams + final comparison table
#   Apply    — illustrative risk-based personal-loan pricing
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
import polars as pl
from dotenv import load_dotenv
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from kailash_ml import TrainingPipeline

from shared.mlfp03.ex_5 import (
    ANNUAL_APPLICATIONS,
    DEFAULT_COSTS,
    OUTPUT_DIR,
    annual_roi,
    load_credit_splits,
    load_strategy_proba,
    metrics_row,
    print_metrics_table,
    print_reliability,
    print_roi,
    reliability_bins,
    save_strategy_proba,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — What calibration means and why loan pricing requires it
# ════════════════════════════════════════════════════════════════════════
# A model output p=0.2 is CALIBRATED if, among all applicants with
# p=0.2, exactly 20% actually default. A gradient booster trained with
# cost-sensitive weights can still be a good RANKER (AUC-PR) but is a poor
# CALIBRATOR: up-weighting defaulters tells the model defaults are more
# common than they are, so it systematically OVER-predicts default (you
# saw the mean predicted p jump well above the real rate in 5.2). Banking
# requires calibration for:
#
#   - LOAN PRICING: risk-based interest rates are computed as
#     rate = funding_cost + expected_loss(p) + margin. If p is
#     miscalibrated by 2x, the entire pricing curve is wrong.
#
#   - EXPECTED LOSS / ECL (IFRS 9): Stage 2 and Stage 3 provisions
#     are the sum of p*LGD*EAD over the portfolio. Miscalibrated p
#     directly misstates the provision line on the balance sheet.
#
#   - PORTFOLIO SIMULATION: stress tests model the loss distribution
#     by sampling from p. If p is uncalibrated, the 99th percentile
#     tail is wrong, and the CRO under-reserves.
#
# TWO POST-HOC CALIBRATORS:
#
#   PLATT SCALING — fits a 2-parameter logistic function
#       p_cal = 1 / (1 + exp(A * raw + B))
#     Parametric, low variance, works well with small calibration sets.
#     Assumes the miscalibration is sigmoid-shaped.
#
#   ISOTONIC REGRESSION — fits a non-decreasing step function.
#     Non-parametric, higher variance, more flexible. Needs >1000
#     calibration samples to avoid overfitting. It can correct ANY
#     monotone distortion of the scores (not just a sigmoid-shaped one),
#     but because it is monotone by construction it cannot fix a
#     non-monotone one — and it never changes the ranking (AUC stays put,
#     apart from ties created by its flat steps).
#
#   Both are fitted on a held-out CALIBRATION slice (20% of the training
#   rows the booster never saw): calibrating on the same rows the booster
#   trained on would learn its over-confidence on training data, not its
#   behaviour on new applicants.
#
# RULE OF THUMB: small calibration set -> Platt; large -> Isotonic;
# always check with a reliability diagram.


# ════════════════════════════════════════════════════════════════════════
# BUILD + TRAIN — wrap the cost-sensitive LightGBM in two calibrators
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()

scale_weight = (1 - pos_rate) / pos_rate
# 80% of the training rows fit the weighted booster; the other 20% are
# kept back to fit the calibration maps.
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)
base_estimator = lgb.LGBMClassifier(
    n_estimators=300,
    scale_pos_weight=scale_weight,
    random_state=42,
    verbose=-1,
)
base_estimator.fit(X_fit, y_fit)

# kailash-ml's TrainingPipeline.calibrate wraps an already-fitted model
# (frozen — it is not re-trained) and fits the calibration map on the
# held-out rows. No registry is needed for calibration alone.
pipeline = TrainingPipeline(feature_store=None, registry=None)
X_cal_frame = pl.DataFrame(X_cal, schema=[f"f{i}" for i in range(X_cal.shape[1])], orient="row")
y_cal_series = pl.Series("default", y_cal)

print("\n" + "=" * 70)
print("  Exercise 5.5 — Calibration (Platt + Isotonic)")
print("=" * 70)
print(f"  Fitting Platt scaling on {len(y_cal):,} held-out rows...")
platt = asyncio.run(
    pipeline.calibrate(base_estimator, X_cal_frame, y_cal_series, method="sigmoid")
)
print(f"  Fitting Isotonic regression on {len(y_cal):,} held-out rows...")
isotonic = asyncio.run(
    pipeline.calibrate(base_estimator, X_cal_frame, y_cal_series, method="isotonic")
)

y_proba_platt = platt.predict_proba(X_test)[:, 1]
y_proba_iso = isotonic.predict_proba(X_test)[:, 1]
save_strategy_proba("platt_calibrated", y_proba_platt)
save_strategy_proba("isotonic_calibrated", y_proba_iso)


# ── Checkpoint 5 ────────────────────────────────────────────────────────
brier_raw = brier_score_loss(y_test, load_strategy_proba("cost_sensitive_scale"))
assert 0 <= y_proba_platt.min() and y_proba_platt.max() <= 1, "Platt out of range"
assert 0 <= y_proba_iso.min() and y_proba_iso.max() <= 1, "Isotonic out of range"
assert brier_score_loss(y_test, y_proba_platt) < brier_raw, "Platt must improve Brier"
assert brier_score_loss(y_test, y_proba_iso) < brier_raw, "Isotonic must improve Brier"
print("[ok] Checkpoint 5 — both calibrators improve Brier over the raw weighted model\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — reliability diagrams + final strategy comparison
# ════════════════════════════════════════════════════════════════════════

# Reliability bins for each variant
bins_baseline = reliability_bins(y_test, load_strategy_proba("baseline"))
bins_cost = reliability_bins(y_test, load_strategy_proba("cost_sensitive_scale"))
bins_platt = reliability_bins(y_test, y_proba_platt)
bins_iso = reliability_bins(y_test, y_proba_iso)

print_reliability("Baseline", bins_baseline)
print_reliability("Cost-sensitive", bins_cost)
print_reliability("Platt", bins_platt)
print_reliability("Isotonic", bins_iso)



def expected_calibration_error(bins: pl.DataFrame) -> float:
    """ECE = count-weighted mean |mean_pred - empirical_rate| over bins."""
    return float((bins["count"] * bins["gap"]).sum() / bins["count"].sum())


y_proba_cost = load_strategy_proba("cost_sensitive_scale")
calibration_summary = [
    (name, proba, expected_calibration_error(bins))
    for name, proba, bins in [
        ("Baseline", load_strategy_proba("baseline"), bins_baseline),
        ("Cost-sensitive", y_proba_cost, bins_cost),
        ("Platt", y_proba_platt, bins_platt),
        ("Isotonic", y_proba_iso, bins_iso),
    ]
]
print(f"\n  Real default rate in test: {y_test.mean():.3f}")
print(f"  {'Variant':<16} {'mean p':>8} {'ECE':>8} {'Brier':>8}")
for name, proba, ece in calibration_summary:
    brier = metrics_row(name, y_test, proba)["brier"]
    print(f"  {name:<16} {proba.mean():>8.3f} {ece:>8.4f} {brier:>8.4f}")
# INTERPRETATION: A point ABOVE the diagonal in a reliability bin means
# the model's probabilities there are too LOW (more defaults happen than
# predicted); BELOW means too HIGH. Read the table with that in mind:
# where does the cost-sensitive model's mean p sit relative to the real
# default rate, and how much do Platt and Isotonic shrink ECE and Brier?
# (Brier mixes calibration with discrimination; ECE isolates calibration.)

# Final comparison table across every strategy we've trained
strategies = [
    ("Baseline (none)", "baseline"),
    ("SMOTE", "smote"),
    ("Cost-sens (scale)", "cost_sensitive_scale"),
    ("Cost-sens (matrix)", "cost_sensitive_matrix"),
    ("Focal gamma=2.0", "focal_gamma_2.0"),
    ("Cost + Platt", "platt_calibrated"),
    ("Cost + Isotonic", "isotonic_calibrated"),
]
# Every strategy must exist — run 01-04 first. A missing one raises with
# a clear message rather than silently shrinking the comparison.
all_rows: list[dict] = []
for display, key in strategies:
    p = load_strategy_proba(key)
    all_rows.append(metrics_row(display, y_test, p))

print_metrics_table(all_rows, "FINAL COMPARISON — all imbalance strategies")

final_df = pl.DataFrame(all_rows)
final_df.write_parquet(OUTPUT_DIR / "final_comparison.parquet")
print(f"\n  Saved: {OUTPUT_DIR / 'final_comparison.parquet'}")

# ── Visual: Reliability diagram (calibration curves) ─────────────────────
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line=dict(dash="dash", color="#9ca3af", width=1),
    )
)
for label, bins in [
    ("Baseline", bins_baseline),
    ("Cost-sensitive", bins_cost),
    ("Platt", bins_platt),
    ("Isotonic", bins_iso),
]:
    # bins is a polars DataFrame with columns mean_pred / empirical_rate /
    # count (see shared.mlfp03.ex_5.reliability_bins). Filter out empty
    # bins, then extract the two scatter columns.
    nonempty = bins.filter(pl.col("count") > 0)
    mean_pred = nonempty["mean_pred"].to_list()
    frac_pos = nonempty["empirical_rate"].to_list()
    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=label))
fig.update_layout(
    title="Reliability Diagram: predicted probability vs observed default rate",
    xaxis_title="Mean predicted probability",
    yaxis_title="Fraction of positives (actual default rate)",
    height=500,
    legend=dict(orientation="h", y=-0.2),
)
viz_path = OUTPUT_DIR / "ex5_05_reliability_diagram.html"
fig.write_html(str(viz_path))
print(f"  Saved: {viz_path}")

# ── Visual: Brier score comparison bar chart ─────────────────────────────
fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=[r["strategy"] for r in all_rows],
        y=[r["brier"] for r in all_rows],
        marker_color=[
            (
                "#10b981"
                if r["brier"] == min(rr["brier"] for rr in all_rows)
                else "#6366f1"
            )
            for r in all_rows
        ],
        text=[f"{r['brier']:.4f}" for r in all_rows],
        textposition="outside",
    )
)
fig2.update_layout(
    title="Brier Score Comparison: lower = better calibration (green = best)",
    xaxis_title="Strategy",
    yaxis_title="Brier score",
    height=450,
)
viz_path2 = OUTPUT_DIR / "ex5_05_brier_comparison.html"
fig2.write_html(str(viz_path2))
print(f"  Saved: {viz_path2}")

# Pick the production-ready strategy
best_auc_pr = max(all_rows, key=lambda r: r["auc_pr"])
best_brier = min(all_rows, key=lambda r: r["brier"])
print(
    f"\n  Best AUC-PR: {best_auc_pr['strategy']} "
    f"(AUC-PR={best_auc_pr['auc_pr']:.4f})"
)
print(f"  Best Brier:  {best_brier['strategy']} (Brier={best_brier['brier']:.4f})")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Risk-based personal-loan pricing (illustrative)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank prices every personal
# loan individually with a risk-based formula:
#
#     APR = funding_cost + expected_loss(p) + operating_margin
#     expected_loss(p) = p * LGD * EAD
#
# If p is systematically too high (as with the raw weighted model), every
# good customer is over-priced and drifts to competitors; if too low, the
# risky book is under-priced and provisions are understated. Model-risk
# reviewers therefore ask to see calibration on held-out data (the
# reliability diagram and ECE above) before a pricing model goes live.
#
# Production recipe:
#   1. Train a strong ranker (LightGBM, optionally class-weighted)
#   2. Post-calibrate on held-out data (Isotonic with plenty of data,
#      Platt with little)
#   3. Apply the Bayes threshold t* = cost_FP / (cost_FP + cost_FN) to
#      the CALIBRATED probabilities
#   4. Price loans from the calibrated p
#   5. Monitor drift (Exercise 8)
#
# The ROI below applies t* to the raw weighted scores and to the two
# calibrated versions — t* is only valid for the calibrated ones.

t_star = DEFAULT_COSTS.optimal_threshold
roi_by_variant: dict[str, dict] = {}
for display, proba in [
    ("Cost-sens (raw)", y_proba_cost),
    ("Cost + Platt", y_proba_platt),
    ("Cost + Isotonic", y_proba_iso),
]:
    roi = annual_roi(y_test, proba, threshold=t_star, annual_volume=ANNUAL_APPLICATIONS)
    roi_by_variant[display] = roi
    print_roi(f"{display} @ t*={t_star:.4f}", roi)

best_variant = max(roi_by_variant, key=lambda k: roi_by_variant[k]["annual_savings_usd"])
print(f"\n  Highest annual savings at t*: {best_variant}")


# ════════════════════════════════════════════════════════════════════════
# PRODUCTION RECOMMENDATION — computed from this run
# ════════════════════════════════════════════════════════════════════════
best_calibrated = min(calibration_summary[2:], key=lambda t: t[2])
print(
    f"""
  From this run (~{y_test.mean():.0%} default rate, cost ratio
  {DEFAULT_COSTS.fn / DEFAULT_COSTS.fp:.1f}:1):

    1. Best ranking (AUC-PR):     {best_auc_pr['strategy']}
    2. Best Brier:                {best_brier['strategy']}
    3. Best calibrated variant:   {best_calibrated[0]} (ECE {best_calibrated[2]:.4f})
    4. Decision rule:             decline if calibrated p >= t* = {t_star:.3f}
    5. Report AUC-PR + Brier/ECE + annual S$ savings to the risk committee

  If you use SMOTE or class weights, recalibrate on held-out data
  before reading the outputs as probabilities.
"""
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson built calibration from primitives — Platt scaling, isotonic
# regression, cost-aware threshold optimisation. The kailash-ml SDK
# packages the diagnostic surface (per-class metrics, class-balance
# severity, confusion matrix) into a single call.
#
# Destination-first: when the journey is internalised, the SDK is one line.

from kailash_ml import diagnose

# `kind="classical_classifier"` dispatches to the sklearn ClassifierMixin
# adapter. TrainingPipeline.calibrate returns an sklearn
# CalibratedClassifierCV, which implements the ClassifierMixin interface.
# Use the isotonic variant — typically the better calibrator for >1k samples.
report = diagnose(
    isotonic, kind="classical_classifier", data=(X_test, y_test), show=False
)
print()
print("  km.diagnose model    : Isotonic-calibrated LightGBM")
print(f"  km.diagnose metrics  : {report.metrics}")
print(f"  km.diagnose severity : {report.severity}")
print()
print("km.diagnose: 1 call -> the same diagnostic surface the lesson body")
print("hand-rolled across Platt + Isotonic. Destination-first: when the")
print("journey is internalised, the SDK is one line.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.5 (and Exercise 5 as a whole)")
print("=" * 70)
print(
    """
  [x] Platt scaling: parametric logistic post-calibration
  [x] Isotonic regression: non-parametric step-function post-calibration
  [x] Read reliability diagrams to spot under/over-confidence
  [x] Final comparison across all seven strategies (baseline, SMOTE,
      cost-sens x2, focal, Platt, Isotonic)
  [x] Translated the winner into annual S$ savings for loan pricing
  [x] Documented the production recipe for Singapore consumer credit

  WHOLE-EXERCISE INSIGHT: The winning strategy on financial tabular
  data is almost never "the clever paper trick." It's cost-sensitive
  learning + post-hoc calibration + cost-matrix threshold tuning.
  Simple, production-grade, auditable.

  NEXT: Exercise 6 adds SHAP interpretability — the per-decision
  explanations that model-risk reviewers and customers ask for.
"""
)
