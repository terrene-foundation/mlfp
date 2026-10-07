# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8.1: Conformal Prediction for Credit Default
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Train a calibrated LightGBM credit-default model through TrainingPipeline
#   - Apply split conformal prediction for distribution-free uncertainty
#   - Check the 1-α (marginal) coverage guarantee on held-out data
#   - Sweep α and read the cost of tighter coverage in set size
#   - Translate "ambiguous prediction set" into a business routing rule
#
# PREREQUISITES: MLFP03 Exercises 1-7, MLFP02 (preprocessing).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory     — what conformal guarantees, and under which assumption
#   2. Build      — train + calibrate the model, compute nonconformity scores
#   3. Train      — calibrate q̂, generate prediction sets, measure coverage
#   4. Visualise  — plot coverage + set-size mix across α values
#   5. Apply      — which applicants a Singapore lender sends to human review
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from shared.mlfp03.ex_8 import (
    OUTPUT_DIR,
    conformal_qhat,
    conformal_summary,
    evaluate_classification,
    load_credit_split,
    nonconformity_scores,
    prediction_sets,
    train_calibrated_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — What conformal prediction guarantees
# ════════════════════════════════════════════════════════════════════════
# Every classifier outputs a probability, but a probability is not a
# guarantee. A model that says "80% chance of default" on 100 applicants
# may see 40 defaults (over-confident) or 95 (under-confident).
#
# Conformal prediction asks a different question: "Which SET of labels
# C(x) should I report so that the true label is inside it for at least
# 1 - α of applicants?"
#
# The one assumption is EXCHANGEABILITY: calibration applicants and
# future applicants are drawn from the same distribution, in no
# particular order. No Gaussian assumption and no requirement that the
# model is accurate or calibrated — a bad model simply produces bigger
# (less useful) sets.
#
# The guarantee is MARGINAL: averaged over future applicants,
# P(Y ∈ C(X)) ≥ 1 - α. It does not promise 1 - α for any single
# applicant or for every sub-group.
#
# THE ALGORITHM (split conformal, binary classification):
#   1. Split held-out data into CALIBRATION and EVALUATION halves.
#   2. On CALIBRATION: nonconformity score s_i = 1 - p(y_true | x_i).
#      High score = the model was surprised by what happened.
#   3. q̂ = the ⌈(n+1)(1-α)⌉-th smallest score (finite-sample correction).
#   4. On new x: include class c in the set iff 1 - p(c|x) ≤ q̂.
#   5. Sets of size 2 mean "both outcomes plausible"; an empty set means
#      "neither is plausible at this α" — rare, and counted as a miss.
#
# THE BUSINESS PAYOFF: each prediction comes with a routing decision.
#   - Singleton {0} or {1}: one outcome is plausible → decide automatically
#   - {0, 1}: the model cannot separate the outcomes → route to a human


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: train the calibrated baseline model
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.1 — Conformal Prediction")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]
feature_names = split["feature_names"]

print(f"\nData: train={X_train.shape}, test={X_test.shape}")
print(f"Default rate: {split['default_rate']:.1%}")

calibrated_model = train_calibrated_model(X_train, y_train, feature_names)
y_proba = calibrated_model.predict_proba(X_test)[:, 1]
metrics = evaluate_classification(y_test, y_proba)

print("\n=== Calibrated Model Metrics (test) ===")
for k, v in metrics.items():
    print(f"  {k:<10} {v:.4f}")
no_skill_brier = split["default_rate"] * (1 - split["default_rate"])
print(f"  (Brier of always predicting the base rate: {no_skill_brier:.4f})")


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.5, "Task 2: Should beat random"
assert (
    0 < metrics["brier"] < no_skill_brier
), "Task 2: Brier should beat the constant base-rate forecast"
print("\n[ok] Checkpoint 1 — calibrated LightGBM trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the conformal calibrator
# ════════════════════════════════════════════════════════════════════════
# Split the test set into calibration and evaluation halves. Compute
# nonconformity scores on calibration, then apply q̂ on evaluation.

n_cal = X_test.shape[0] // 2
X_cal, X_eval = X_test[:n_cal], X_test[n_cal:]
y_cal, y_eval = y_test[:n_cal], y_test[n_cal:]

cal_proba = calibrated_model.predict_proba(X_cal)[:, 1]
cal_scores = nonconformity_scores(y_cal, cal_proba)

alpha = 0.10  # target 90% coverage
q_hat = conformal_qhat(cal_scores, alpha)

print("=== Conformal Calibration ===")
print(f"  Calibration set:   {len(cal_scores)} samples")
print(f"  Target coverage:   {1 - alpha:.0%}")
print(f"  Calibration q̂:     {q_hat:.4f}")

eval_proba = calibrated_model.predict_proba(X_eval)[:, 1]
has_0, has_1 = prediction_sets(eval_proba, q_hat)
summary = conformal_summary(y_eval, eval_proba, q_hat)
coverage = summary["coverage"]
singleton_rate = summary["singleton_rate"]

print("\n=== Empirical Results on Evaluation Half ===")
print(f"  Coverage:          {coverage:.4f} (target: ≥ {1 - alpha:.2f})")
print(f"  Avg set size:      {summary['avg_set_size']:.3f}")
print(f"  Singleton rate:    {singleton_rate:.1%} (auto-decide)")
print(f"  Both-class rate:   {summary['both_rate']:.1%} (route to human)")
print(f"  Empty-set rate:    {summary['empty_rate']:.1%}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
# Coverage on a finite evaluation half fluctuates around its expectation;
# a 2-point tolerance absorbs that sampling noise.
assert coverage >= (
    1 - alpha - 0.02
), f"Task 3: Coverage {coverage:.4f} should be near target {1 - alpha:.4f}"
assert 0 <= summary["avg_set_size"] <= 2, "Task 3: Set size must be in [0, 2]"
# INTERPRETATION: coverage holds while new applicants look like the
# calibration applicants. If live coverage drops below target, the data
# has shifted: the first remedy is to RECALIBRATE q̂ on recent labelled
# data (cheap); retrain the model if its ranking (AUC) has degraded too.
print("\n[ok] Checkpoint 2 — conformal coverage verified on held-out data\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE coverage vs α
# ════════════════════════════════════════════════════════════════════════
# Lower α → higher coverage but bigger sets (more human review).

print("=== Coverage vs α Sweep ===")
print(f"{'Alpha':>8} {'Target':>10} {'Actual':>10} {'Avg Size':>10} {'Singleton':>12}")
print("─" * 54)

alphas_sweep = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
sweep_rows = []
for a in alphas_sweep:
    row = conformal_summary(y_eval, eval_proba, conformal_qhat(cal_scores, a))
    sweep_rows.append({"alpha": a, **row})
    print(
        f"{a:>8.2f} {1 - a:>10.2f} {row['coverage']:>10.4f} "
        f"{row['avg_set_size']:>10.3f} {row['singleton_rate']:>11.1%}"
    )

xs = [r["alpha"] for r in sweep_rows]
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=xs,
        y=[r["coverage"] for r in sweep_rows],
        mode="lines+markers",
        name="Empirical coverage",
        line=dict(color="#2563eb", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=xs,
        y=[1 - a for a in xs],
        mode="lines",
        name="Target 1-α",
        line=dict(color="#94a3b8", dash="dash"),
    )
)
fig.add_trace(
    go.Bar(
        x=xs,
        y=[r["singleton_rate"] for r in sweep_rows],
        name="Singleton rate (auto-decide)",
        marker_color="#10b981",
        opacity=0.55,
        yaxis="y2",
    )
)
fig.update_layout(
    title="Conformal Prediction: Coverage vs α (Singapore credit default)",
    xaxis_title="α (risk budget)",
    yaxis=dict(title="Coverage", range=[0, 1.05]),
    yaxis2=dict(title="Singleton rate", overlaying="y", side="right", range=[0, 1.05]),
    legend=dict(orientation="h", y=-0.2),
    height=500,
)
viz_path = OUTPUT_DIR / "ex8_01_conformal_sweep.html"
fig.write_html(str(viz_path))
print(f"\nSaved: {viz_path}")


# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert len(sweep_rows) == len(alphas_sweep), "Task 4: Should sweep all alphas"
for r in sweep_rows:
    assert r["coverage"] >= 1 - r["alpha"] - 0.02, (
        f"Task 4: coverage at α={r['alpha']} fell below target"
    )
print("\n[ok] Checkpoint 3 — α sweep visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: routing applications at a Singapore lender
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail lender receives ~18,000
# unsecured-loan applications a month and today has an analyst review
# every one. Conformal sets give a routing rule:
#
#   {0}     → fast-track approval (only "no default" is plausible)
#   {1}     → decline / strict review (only "default" is plausible)
#   {0, 1}  → HUMAN REVIEW (the model cannot separate the outcomes)
#
# What the 90% guarantee does and does not say: across all applicants,
# at least 90% of sets contain the true outcome — provided new
# applicants look like the calibration applicants. It is NOT "90% sure
# about this applicant", and a recession that changes who applies breaks
# the exchangeability assumption (8.2 watches for that).

only_0 = has_0 & ~has_1
only_1 = has_1 & ~has_0
both = has_0 & has_1
print(f"=== Routing at α={alpha} on {len(y_eval):,} held-out applicants ===")
for name, mask in (("{0} fast-track", only_0), ("{1} decline/strict", only_1), ("{0,1} human review", both)):
    rate = float(mask.mean())
    actual = f"{y_eval[mask].mean():.1%}" if mask.any() else "n/a (empty)"
    print(f"  {name:<20} {rate:>6.1%} of applicants   observed default rate {actual}")

# DOLLAR IMPACT — ILLUSTRATIVE assumptions, not any lender's figures
monthly_apps = 18_000
analyst_hourly = 85.0  # SGD, fully loaded
time_per_review_h = 0.25  # 15 minutes per application
review_share = float(both.mean() + only_1.mean())  # humans still check declines
status_quo_cost = monthly_apps * time_per_review_h * analyst_hourly
conformal_cost = monthly_apps * review_share * time_per_review_h * analyst_hourly
savings = status_quo_cost - conformal_cost
print(f"\n  Analyst cost (status quo, 100% review): S${status_quo_cost:>10,.0f}/mo")
print(f"  Analyst cost (review {{0,1}} + {{1}} only): S${conformal_cost:>10,.0f}/mo")
print(f"  Monthly savings:                         S${savings:>10,.0f}/mo")
print(f"  Annualised:                              S${savings * 12:>10,.0f}/yr")
# INTERPRETATION: the fast-tracked {0} group is only safe if its observed
# default rate (printed above) is acceptable to the credit policy —
# check that number, not just the share of work saved.

# LIMITATIONS:
#   - Coverage is marginal; it can be lower inside a sub-group. Check
#     group-wise coverage alongside the fairness audit (Exercise 6, 8.3).
#   - Exchangeability breaks under drift; 8.2 monitors for it.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Trained and calibrated the credit model through TrainingPipeline
  [x] Built split conformal prediction from nonconformity scores
  [x] Measured {coverage:.1%} coverage at α={alpha} on held-out applicants
      (a marginal guarantee under exchangeability)
  [x] Swept α and visualised the cost of tighter coverage
  [x] Turned set size into a routing rule worth
      ~S${savings * 12:,.0f}/yr of analyst time (illustrative inputs)

  KEY INSIGHT: A probability is a guess. A prediction set comes with a
  coverage guarantee — on average, and only while tomorrow's applicants
  resemble the calibration set.

  Next: 02_drift_monitoring.py — detect when that resemblance breaks.
"""
)
