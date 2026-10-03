# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 6.2: Odds Ratios and Threshold Optimisation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Convert logistic regression coefficients to odds ratios: exp(b)
#   - Interpret odds ratios on the original (unscaled) feature scale
#   - Optimise classification threshold using a domain cost matrix
#   - Compare cost-optimal vs F1-optimal vs default thresholds
#   - Apply threshold optimisation to Singapore HDB valuation risk
#
# PREREQUISITES: Exercise 6.1 — logistic regression from scratch
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — from coefficients to odds ratios
#   2. Build — unscale coefficients + compute odds ratios
#   3. Train — threshold sweep with cost matrix
#   4. Visualise — odds ratio forest plot + cost curve
#   5. Apply — HDB valuation risk: asymmetric error costs
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from shared.mlfp02.ex_6 import (
    FEATURE_COLS,
    OUTPUT_DIR,
    build_classification_frame,
    build_design_matrix,
    load_hdb_recent,
    neg_ll_gradient,
    neg_log_likelihood_logistic,
    sigmoid,
    unscale_coefficients,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — From Coefficients to Odds Ratios
# ════════════════════════════════════════════════════════════════════════
# Logistic regression models log-odds: log(P/(1-P)) = b0 + b1*x1 + ...
#
# The ODDS of an event is P / (1 - P). When P = 0.75, odds = 3:1.
#
# An ODDS RATIO for feature j is exp(b_j). It answers: "how do the
# odds multiply when x_j increases by one unit?"
#
#   exp(b_j) > 1  -> odds increase (feature promotes the outcome)
#   exp(b_j) = 1  -> no effect
#   exp(b_j) < 1  -> odds decrease (feature inhibits the outcome)
#
# The threshold question is separate from the model. The model outputs
# a probability; the THRESHOLD decides the action.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: fit model and compute odds ratios
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Odds Ratios and Threshold Optimisation")
print("=" * 70)

# Load data and fit logistic regression (same as 6.1)
hdb_recent = load_hdb_recent()
frame, median_price = build_classification_frame(hdb_recent)
X, y, X_mean, X_std, feature_names = build_design_matrix(frame)
n_obs = X.shape[0]

beta0 = np.zeros(X.shape[1])
result = minimize(
    neg_log_likelihood_logistic,
    beta0,
    args=(X, y),
    method="L-BFGS-B",
    jac=neg_ll_gradient,
    options={"maxiter": 1000, "ftol": 1e-12},
)
beta_scratch = result.x
p_scratch = sigmoid(X @ beta_scratch)

# TODO: Convert coefficients to original scale using unscale_coefficients.
# Hint: unscale_coefficients(beta_scratch, X_mean, X_std) returns the
#       original-scale coefficient vector.
beta_original = ____

print(f"\n=== Odds Ratio Interpretation ===")
print(f"\n{'Feature':<20} {'b (original)':>14} {'Odds Ratio':>12} {'Interpretation'}")
print("-" * 80)
for i in range(1, len(feature_names)):
    # TODO: Compute the odds ratio for feature i.
    # Hint: odds ratio = np.exp(beta_original[i]).
    or_val = ____
    name = feature_names[i]
    if or_val > 1:
        interp = f"1-unit increase -> {(or_val-1)*100:.1f}% higher odds"
    else:
        interp = f"1-unit increase -> {(1-or_val)*100:.1f}% lower odds"
    print(f"{name:<20} {beta_original[i]:>14.6f} {or_val:>12.4f} {interp}")

# Practical examples with real-world units
print(f"\n--- Practical Examples ---")
for feat, units, factor in [
    ("floor_area_sqm", "10 sqm", 10),
    ("storey_mid", "5 storeys", 5),
    ("remaining_lease", "10 years", 10),
]:
    idx = feature_names.index(feat)
    # TODO: Compute the odds change for a multi-unit increase.
    # Hint: np.exp(beta_original[idx] * factor).
    or_change = ____
    print(
        f"  +{units} of {feat}: odds multiply by {or_change:.3f} "
        f"({(or_change-1)*100:+.1f}% change)"
    )

# INTERPRETATION: An odds ratio of 1.5 for floor_area_sqm means each
# extra sqm multiplies the odds of being "high price" by 1.5. Odds
# ratios are multiplicative, not additive — they compound with each
# unit increase. A 10 sqm increase multiplies odds by 1.5^10 ≈ 57x.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert np.exp(beta_original[1]) > 1, "Larger area should increase odds of high price"
print("\n[ok] Checkpoint 1 passed — odds ratios computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: threshold sweep with cost matrix
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Threshold Optimisation ===")

# Cost matrix for HDB property valuation
# FP (predict high, actually low): buyer overpays -> cost = $30K
# FN (predict low, actually high): seller underprices -> cost = $50K
cost_fp = 30_000
cost_fn = 50_000

thresholds = np.linspace(0.1, 0.9, 81)
total_costs = []
accuracies = []
f1_scores_list = []

for t in thresholds:
    y_pred_t = (p_scratch >= t).astype(int)
    # TODO: Compute the confusion matrix for this threshold.
    # Hint: confusion_matrix(y, y_pred_t) returns a 2x2 array.
    cm = ____
    tn, fp, fn, tp = cm.ravel()
    # TODO: Compute the total cost using the cost matrix.
    # Hint: cost = fp * cost_fp + fn * cost_fn.
    cost = ____
    total_costs.append(cost)
    accuracies.append(accuracy_score(y, y_pred_t))
    f1_scores_list.append(f1_score(y, y_pred_t, zero_division=0))

# TODO: Find the index of the cost-minimising threshold.
# Hint: np.argmin(total_costs).
optimal_idx = ____
optimal_threshold = thresholds[optimal_idx]
optimal_cost = total_costs[optimal_idx]

# F1-optimal threshold
f1_idx = np.argmax(f1_scores_list)
f1_threshold = thresholds[f1_idx]

print(f"Cost matrix: FP=${cost_fp:,}, FN=${cost_fn:,}")
print(f"\nOptimal threshold (min cost): {optimal_threshold:.3f}")
print(f"  Total cost: ${optimal_cost:,.0f}")
print(f"  Accuracy at this threshold: {accuracies[optimal_idx]:.4f}")
print(f"\nF1-optimal threshold: {f1_threshold:.3f}")
print(f"  F1 at this threshold: {f1_scores_list[f1_idx]:.4f}")
print(f"\nDefault threshold (0.5):")
print(f"  Cost: ${total_costs[40]:,.0f}")
print(f"  Accuracy: {accuracies[40]:.4f}")

# INTERPRETATION: When FN costs more than FP, the optimal threshold
# is below 0.5 — we'd rather predict "high price" more aggressively
# to avoid missing expensive flats. The threshold should reflect the
# business cost of each type of error, not just accuracy.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 0 < optimal_threshold < 1, "Optimal threshold must be valid"
assert optimal_cost <= total_costs[40], "Optimal cost must be <= default cost"
print("\n[ok] Checkpoint 2 passed — threshold optimisation completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: odds ratio forest plot + cost curve
# ════════════════════════════════════════════════════════════════════════

# Plot 1: Odds ratio forest plot
or_values = [np.exp(beta_original[i]) for i in range(1, len(feature_names))]
or_names = feature_names[1:]

fig1 = go.Figure()
fig1.add_trace(
    go.Bar(
        y=or_names,
        x=or_values,
        orientation="h",
        marker_color=["#2ecc71" if v > 1 else "#e74c3c" for v in or_values],
    )
)
fig1.add_vline(x=1.0, line_dash="dash", line_color="grey", annotation_text="No effect")
fig1.update_layout(
    title="Odds Ratios per Feature (original scale, per unit)",
    xaxis_title="Odds Ratio exp(b)",
    yaxis_title="Feature",
)
fig1.write_html(str(OUTPUT_DIR / "odds_ratios.html"))
print(f"Saved: {OUTPUT_DIR / 'odds_ratios.html'}")

# TODO: Create a cost + accuracy + F1 curve vs threshold.
# Hint: use go.Scatter for each series. Add vertical lines at
#       optimal_threshold and 0.5 using fig.add_vline().
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=thresholds,
        y=____,  # Hint: total_costs in $M, one value per threshold
        name="Total Cost ($M)",
    )
)
fig2.add_trace(
    go.Scatter(
        x=thresholds,
        y=accuracies,
        name="Accuracy",
        yaxis="y2",
        line={"dash": "dash"},
    )
)
fig2.add_trace(
    go.Scatter(
        x=thresholds,
        y=f1_scores_list,
        name="F1 Score",
        yaxis="y2",
        line={"dash": "dot"},
    )
)
fig2.add_vline(
    x=optimal_threshold,
    line_dash="dash",
    annotation_text=f"Cost-optimal t={optimal_threshold:.2f}",
)
fig2.add_vline(
    x=0.5,
    line_dash="dot",
    line_color="red",
    annotation_text="Default t=0.5",
)
fig2.update_layout(
    title="Cost, Accuracy, and F1 vs Classification Threshold",
    xaxis_title="Threshold",
    yaxis_title="Total Cost ($M)",
    yaxis2={"title": "Score", "overlaying": "y", "side": "right", "range": [0, 1]},
)
fig2.write_html(str(OUTPUT_DIR / "threshold_cost.html"))
print(f"Saved: {OUTPUT_DIR / 'threshold_cost.html'}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — odds ratios + cost curve visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: HDB valuation risk — a DIFFERENT cost matrix
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A mortgage lender's valuation-audit team screens HDB resale
# transactions it is financing. A transaction flagged "high-price"
# triggers a detailed review by a licensed valuer.
#
# Two types of errors carry different (ILLUSTRATIVE) costs:
#   - FP (flagged but normal): unnecessary review costs S$800/case
#   - FN (missed overpriced deal): average over-lending exposure of
#     S$45,000 per case
#
# This is NOT the cost matrix of Task 3 (S$30K vs S$50K). A threshold
# is only "cost-optimal" for the costs it was optimised under, so the
# sweep must be re-run with the audit team's own costs.

review_cost = 800  # S$ per unnecessary review (illustrative)
missed_cost = 45_000  # S$ exposure per missed case (illustrative)

# Re-optimise on a finer grid: steep cost ratios push the optimum low
apply_thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
apply_costs = []
for t in apply_thresholds:
    y_pred_t = (p_scratch >= t).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y, y_pred_t).ravel()
    # TODO: Total audit cost at this threshold with the NEW cost matrix.
    # Hint: same formula as Task 3, using review_cost and missed_cost.
    apply_costs.append(____)
apply_idx = int(np.argmin(apply_costs))
apply_threshold = float(apply_thresholds[apply_idx])

# For perfectly calibrated probabilities the optimum is the Bayes
# threshold c_FP / (c_FP + c_FN); a gap from it signals miscalibration
# TODO: Bayes threshold for these costs.
# Hint: c_FP / (c_FP + c_FN)
bayes_threshold = ____


def audit_cost(threshold: float) -> tuple[int, int, float]:
    """Return (unnecessary reviews, missed cases, total S$) at a threshold."""
    y_pred_t = (p_scratch >= threshold).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y, y_pred_t).ravel()
    return int(fp_t), int(fn_t), float(fp_t * review_cost + fn_t * missed_cost)


print(f"\n=== Real-World Application: Valuation-Audit Screening ===")
print(f"  Transactions scored (2020+ dataset): {n_obs:,}")
print(f"  Bayes threshold for these costs: {bayes_threshold:.3f}")
rows = [
    ("Default", 0.5),
    ("Task 3 optimum (other costs)", float(optimal_threshold)),
    ("Re-optimised for audit costs", apply_threshold),
]
print(f"\n  {'Threshold rule':<30} {'t':>5} {'Reviews':>9} {'Missed':>8} {'Total S$':>15}")
audit_totals = {}
for label, t in rows:
    fp_t, fn_t, total_t = audit_cost(t)
    audit_totals[label] = total_t
    print(f"  {label:<30} {t:>5.2f} {fp_t:>9,} {fn_t:>8,} {total_t:>15,.0f}")
saving_vs_default = audit_totals["Default"] - audit_totals["Re-optimised for audit costs"]
print(f"\n  Saving vs the default threshold: S${saving_vs_default:,.0f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert audit_totals["Re-optimised for audit costs"] <= audit_totals["Default"], (
    "Re-optimised threshold must not cost more than the default"
)
assert audit_totals["Re-optimised for audit costs"] <= audit_totals[
    "Task 3 optimum (other costs)"
], "A threshold tuned for other costs cannot beat one tuned for these costs"
print("\n[ok] Checkpoint 4 passed — threshold re-optimised for the audit costs\n")

# BUSINESS IMPACT: Missing an overpriced deal (FN) costs ~56x an
# unnecessary review (FP), so the cost-minimising threshold sits well
# below 0.5: the team accepts many more reviews to miss fewer
# overpriced deals. Reusing a threshold tuned for a different cost
# matrix leaves money on the table — compare the rows above.
#
# LIMITATIONS:
#   - Cost estimates are simplified. Real costs include legal fees,
#     dispute resolution time, and reputational damage that are hard
#     to quantify precisely.
#   - The cost matrix assumes stationarity. In a rising market, FN
#     costs increase; in a falling market, FP costs increase. The
#     threshold should be recalibrated quarterly.
#   - "High-price" (above median) is a proxy for "overpriced"; a real
#     audit model would target price relative to a fair-value estimate.


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("""
What you've mastered in this technique:
  ✓ Converting standardised logistic coefficients to per-unit odds ratios
  ✓ Reading odds ratios multiplicatively over multi-unit changes
  ✓ Sweeping the decision threshold under a cost matrix
  ✓ Re-optimising the threshold whenever the cost matrix changes, and
    comparing it with the Bayes threshold c_FP / (c_FP + c_FN)

Next: In 03_classification_metrics.py you'll measure the classifier
with confusion matrices, precision/recall, ROC and PR curves.
""")
