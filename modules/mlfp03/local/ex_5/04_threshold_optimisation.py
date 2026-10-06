# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.4: Threshold Optimisation from a Cost Matrix
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why threshold=0.5 is almost always wrong for asymmetric costs
#   - The Bayes-optimal threshold formula t* = cost_FP / (cost_FP + cost_FN)
#     and the conditions under which it holds
#   - How to tune the threshold on OUT-OF-FOLD predictions and report the
#     result once on the untouched test set
#   - How to translate the chosen threshold into annual S$ savings
#
# PREREQUISITES: 02_sampling_strategies.py (cost-sensitive test proba saved)
# ESTIMATED TIME: ~30 min
#
# 5-PHASE STRUCTURE:
#   Theory   — decision theory for asymmetric costs
#   Build    — cost-at-threshold function + threshold grid
#   Train    — out-of-fold probabilities on the training set (5-fold CV)
#   Visualise — cost vs threshold curves (out-of-fold vs test)
#   Apply    — illustrative unsecured-loan underwriting ROI
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from shared.mlfp03.ex_5 import (
    ANNUAL_APPLICATIONS,
    DEFAULT_COSTS,
    OUTPUT_DIR,
    annual_roi,
    load_credit_splits,
    load_strategy_proba,
    metrics_row,
    print_metrics_table,
    print_roi,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Decision theory for asymmetric cost matrices
# ════════════════════════════════════════════════════════════════════════
# Your model outputs a probability p that an applicant will default. You
# must DECIDE: approve or decline. The default rule is "decline if
# p >= 0.5". Where does 0.5 come from?
#
# It comes from an implicit assumption: a false positive costs the same as
# a false negative. In our (illustrative) cost matrix a false decline costs
# ~S$1,500 of forgone interest margin and a missed default ~S$10,000 of
# charged-off principal — about 7:1. 0.5 is the wrong threshold.
#
# THE BAYES-OPTIMAL THRESHOLD: decline when the expected cost of approving
# (p * cost_FN) exceeds the expected cost of declining ((1-p) * cost_FP):
#
#     t* = cost_FP / (cost_FP + cost_FN) = 1,500 / 11,500 ≈ 0.13
#
# CONDITIONS: t* is only optimal when p is a CALIBRATED probability from a
# model trained on the real class balance. The model we load here was
# trained with scale_pos_weight (5.2), which already inflated its scores —
# applying t* on top would count the cost asymmetry twice. So we choose the
# threshold EMPIRICALLY: the argmin of total cost.
#
# WHERE TO TUNE: never on the test set you report. Picking the threshold
# that minimises test cost and then quoting that test cost is the same
# leak as tuning hyperparameters on the test set — the number is
# optimistic. We tune on OUT-OF-FOLD (OOF) predictions: 5-fold CV on the
# training set, where each training row is scored by a model that never
# saw it. Then we apply the chosen threshold to the test set ONCE.


# ════════════════════════════════════════════════════════════════════════
# BUILD — cost-at-threshold function + grid
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()
y_proba_test = load_strategy_proba("cost_sensitive_scale")  # saved by 02

print("\n" + "=" * 70)
print("  Exercise 5.4 — Threshold Optimisation")
print("=" * 70)
print(f"  Cost matrix: FP=S${DEFAULT_COSTS.fp:,.0f}, FN=S${DEFAULT_COSTS.fn:,.0f}")
print(f"  Bayes-optimal t* (calibrated, unweighted model): {DEFAULT_COSTS.optimal_threshold:.4f}")


def cost_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, t: float) -> dict:
    """Confusion counts and total S$ cost of declining every p >= t."""
    # TODO: Predict labels at threshold t (1 = decline)
    # Hint: compare y_proba with t and cast to int
    y_pred = ____
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    # TODO: total S$ cost = false declines * cost_FP + missed defaults * cost_FN
    # Hint: use DEFAULT_COSTS.fp and DEFAULT_COSTS.fn
    cost = ____
    return {
        "threshold": float(t),
        "total_cost_sgd": float(cost),
        "cost_per_application": float(cost / len(y_true)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "decline_rate": float(y_pred.mean()),
    }


thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)


# ════════════════════════════════════════════════════════════════════════
# TRAIN — out-of-fold probabilities for the SAME model recipe as 5.2
# ════════════════════════════════════════════════════════════════════════

scale_weight = (1 - pos_rate) / pos_rate
oof_model = lgb.LGBMClassifier(
    n_estimators=300, scale_pos_weight=scale_weight, random_state=42, verbose=-1
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("  Computing 5-fold out-of-fold probabilities on the training set ...")
# TODO: out-of-fold positive-class probabilities for every TRAINING row
# Hint: cross_val_predict(..., cv=cv, method="predict_proba")[:, 1]
y_proba_oof = ____

oof_sweep = [cost_at_threshold(y_train, y_proba_oof, t) for t in thresholds]
# TODO: the OOF sweep row with the lowest total cost
best_row = ____
best_t = best_row["threshold"]
oof_cost_05 = cost_at_threshold(y_train, y_proba_oof, 0.5)["total_cost_sgd"]

# Report on the test set ONCE, at the threshold chosen out-of-fold
# TODO: evaluate the OOF-chosen threshold on the test set (once)
test_at_best = ____
test_at_05 = cost_at_threshold(y_test, y_proba_test, 0.5)
test_at_bayes = cost_at_threshold(y_test, y_proba_test, DEFAULT_COSTS.optimal_threshold)


# ── Checkpoint 4 ────────────────────────────────────────────────────────
assert y_proba_oof.shape[0] == y_train.shape[0], "OOF must score every training row"
assert 0.0 < best_t < 1.0, "Best threshold must be in (0,1)"
assert (
    best_row["total_cost_sgd"] <= oof_cost_05
), "On the OOF data the tuned threshold cannot cost more than t=0.5"
print("[ok] Checkpoint 4 — threshold tuned out-of-fold, test set untouched\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — cost curves (OOF used to choose, test used to report)
# ════════════════════════════════════════════════════════════════════════

print(f"  {'threshold':>10} {'OOF cost/app':>14} {'precision':>10} {'recall':>8} {'declined':>9}")
print("  " + "─" * 56)
for r in oof_sweep[::10]:
    print(
        f"  {r['threshold']:>10.2f} S${r['cost_per_application']:>11,.0f} "
        f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['decline_rate']:>9.1%}"
    )

print(f"\n  Threshold chosen out-of-fold: t = {best_t:.2f}")
print("  Test-set cost (reported once):")
for label, r in [
    ("t = 0.5 (default)", test_at_05),
    (f"t = {best_t:.2f} (OOF-tuned)", test_at_best),
    (f"t* = {DEFAULT_COSTS.optimal_threshold:.2f} (Bayes formula)", test_at_bayes),
]:
    print(
        f"    {label:<28} S${r['total_cost_sgd']:>12,.0f}  "
        f"(declines {r['decline_rate']:.1%}, recall {r['recall']:.1%})"
    )
savings = test_at_05["total_cost_sgd"] - test_at_best["total_cost_sgd"]
print(f"  Test-set saving of the tuned threshold vs 0.5: S${savings:,.0f}")

pl.DataFrame(oof_sweep).write_parquet(OUTPUT_DIR / "threshold_sweep_oof.parquet")
print(f"\n  Saved: {OUTPUT_DIR / 'threshold_sweep_oof.parquet'}")

# ── Visual: cost per application vs threshold, OOF and test ─────────────
test_sweep = [cost_at_threshold(y_test, y_proba_test, t) for t in thresholds]
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=thresholds, y=[r["cost_per_application"] for r in oof_sweep],
               mode="lines", name="Out-of-fold (used to CHOOSE t)",
               line=dict(color="#ef4444", width=3))
)
fig.add_trace(
    go.Scatter(x=thresholds, y=[r["cost_per_application"] for r in test_sweep],
               mode="lines", name="Test (shown for honesty, NOT used to choose)",
               line=dict(color="#6b7280", width=2, dash="dot"))
)
fig.add_vline(x=best_t, line_dash="dash", line_color="#10b981", annotation_text=f"OOF t={best_t:.2f}")
fig.add_vline(x=0.5, line_dash="dot", line_color="#6b7280", annotation_text="t=0.5")
fig.update_layout(
    title="Threshold vs cost per application (lower = better)",
    xaxis_title="Decision threshold (decline if p >= t)",
    yaxis_title="Cost per application (S$)",
    height=450,
    legend=dict(orientation="h", y=-0.2),
)
viz_path = OUTPUT_DIR / "ex5_04_threshold_cost_curve.html"
fig.write_html(str(viz_path))
print(f"  Saved: {viz_path}")

# ── Visual: precision and recall vs threshold (OOF) ─────────────────────
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=thresholds, y=[r["precision"] for r in oof_sweep], mode="lines",
                          name="Precision", line=dict(color="#6366f1", width=2)))
fig2.add_trace(go.Scatter(x=thresholds, y=[r["recall"] for r in oof_sweep], mode="lines",
                          name="Recall", line=dict(color="#f59e0b", width=2)))
fig2.add_vline(x=best_t, line_dash="dash", line_color="#10b981", annotation_text=f"t={best_t:.2f}")
fig2.update_layout(title="Precision and Recall vs Threshold (out-of-fold)",
                   xaxis_title="Decision threshold", yaxis_title="Score", height=450,
                   legend=dict(orientation="h", y=-0.2))
viz_path2 = OUTPUT_DIR / "ex5_04_threshold_pr_curve.html"
fig2.write_html(str(viz_path2))
print(f"  Saved: {viz_path2}")

# ── Visual: the cost matrix itself ──────────────────────────────────────
fig3 = go.Figure(
    data=go.Heatmap(
        z=np.array([[0, DEFAULT_COSTS.fp], [DEFAULT_COSTS.fn, 0]]),
        x=["Predicted: repay", "Predicted: default"],
        y=["Actual: repaid", "Actual: defaulted"],
        text=[["TN: S$0", f"FP: S${DEFAULT_COSTS.fp:,.0f}"], [f"FN: S${DEFAULT_COSTS.fn:,.0f}", "TP: S$0"]],
        texttemplate="%{text}",
        colorscale="Reds",
    )
)
fig3.update_layout(title="Cost Matrix: asymmetric penalties (FN >> FP)", height=400)
viz_path3 = OUTPUT_DIR / "ex5_04_cost_matrix_heatmap.html"
fig3.write_html(str(viz_path3))
print(f"  Saved: {viz_path3}")

row = metrics_row("Cost-sens @ OOF t", y_test, y_proba_test, threshold=best_t)
print_metrics_table([row], f"Test metrics at the OOF-tuned threshold t={best_t:.2f}")

# INTERPRETATION: compare the three test-set lines. The OOF-tuned
# threshold is chosen without looking at the test set, so its test cost
# is an honest estimate. The Bayes formula is applied to a WEIGHTED model
# here — if its line is the worst, that is the "counting the cost
# asymmetry twice" problem from the Theory. 5.5 calibrates the model so
# the formula becomes valid.


# ════════════════════════════════════════════════════════════════════════
# APPLY — Unsecured-loan underwriting ROI (illustrative)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore bank underwrites ~100,000 unsecured
# personal loans per year. Today's rule: decline if scorecard p >= 0.5.
# The underwriting committee asks: "What would it save if we tuned the
# threshold from the cost matrix?"
#
# We answer by scaling the TEST-set confusion matrix (at the threshold
# chosen out-of-fold) to the annual application volume. "No-model cost"
# is the cost of approving everyone.

roi_default = annual_roi(y_test, y_proba_test, threshold=0.5, annual_volume=ANNUAL_APPLICATIONS)
# TODO: annual ROI at the OOF-tuned threshold (mirror the line above)
roi_best = ____

print_roi("Annual ROI @ t=0.5", roi_default)
print_roi(f"Annual ROI @ OOF-tuned t={best_t:.2f}", roi_best)

delta = roi_best["annual_savings_usd"] - roi_default["annual_savings_usd"]
print(f"\n  Threshold tuning alone changes annual savings by S${delta:,.0f}")
print("    No retraining, no new data, no new features — only a number in")
print("    the decision layer, chosen without touching the test set.")

pl.DataFrame(
    [roi_default | {"label": "t=0.5"}, roi_best | {"label": "t_oof"}]
).write_parquet(OUTPUT_DIR / "threshold_roi.parquet")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.4")
print("=" * 70)
print(
    """
  [x] Derived the Bayes-optimal threshold t* = cost_FP/(cost_FP+cost_FN)
      and its conditions (calibrated probabilities, unweighted training)
  [x] Tuned the threshold on out-of-fold predictions, not the test set
  [x] Reported the tuned threshold's cost on the test set exactly once
  [x] Scaled the test confusion matrix to annual application volume

  KEY INSIGHT: Threshold tuning is often the highest-ROI lever on an
  imbalanced production model — but only an honestly tuned threshold
  (out-of-fold or on a validation split) gives you a savings figure you
  can defend.

  Next: 05_calibration.py — Platt and Isotonic calibration make the
  probabilities trustworthy, so the Bayes formula t* applies directly.
"""
)
