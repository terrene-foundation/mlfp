# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 6.4: Calibration Assessment and ANOVA
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Assess model calibration with calibration curves and Brier score
#   - Understand what "well-calibrated" means for decision-making
#   - Perform one-way ANOVA and interpret F-statistics + eta-squared
#   - Apply post-hoc Tukey HSD for pairwise group comparisons
#   - Apply calibration + ANOVA to HDB resale valuation bands
#
# PREREQUISITES: Exercises 6.1-6.3 (logistic regression, metrics)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — calibration vs discrimination, ANOVA vs t-test
#   2. Build — calibration curve + Brier score
#   3. Train — one-way ANOVA + Tukey HSD across flat types
#   4. Visualise — calibration plot + ANOVA box plots
#   5. Apply — flat-type valuation bands (illustrative S$ impact)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

from shared.mlfp02.ex_6 import (
    FEATURE_COLS,
    OUTPUT_DIR,
    build_classification_frame,
    build_design_matrix,
    calibration_bins,
    load_hdb_recent,
    neg_ll_gradient,
    neg_log_likelihood_logistic,
    sigmoid,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Calibration vs Discrimination; ANOVA vs t-test
# ════════════════════════════════════════════════════════════════════════
# CALIBRATION answers: "When the model says P=0.7, do ~70% of those
# observations turn out positive?" A well-calibrated model's predicted
# probabilities match the observed frequencies. Calibration matters
# when you USE the probabilities for decisions (loan pricing, insurance
# premiums), not just for ranking.
#
# DISCRIMINATION answers: "Can the model separate positives from
# negatives?" (measured by AUC). A model can discriminate well but be
# poorly calibrated — it ranks correctly but the probability values
# are wrong.
#
# The BRIER SCORE = mean((p - y)^2) combines calibration + discrimination
# into a single number. Lower = better; 0 is perfect and 1 is the worst
# possible score. The useful REFERENCE is a constant forecast of 0.5,
# which scores exactly 0.25 on any 0/1 outcome — a model that cannot
# beat 0.25 has learned nothing beyond "flip a coin".
#
# ANOVA generalises the t-test from 2 groups to k groups.
# H0: mu_1 = mu_2 = ... = mu_k (all means equal)
# H1: at least one mean differs
#
# ANOVA tells you SOMETHING differs but not WHAT. Tukey's HSD (Honestly
# Significant Difference) compares every pair of means with the
# statistic q = |mean_i - mean_j| / sqrt(MS_within/2 * (1/n_i + 1/n_j))
# and judges q against the STUDENTIZED RANGE distribution (k groups,
# N - k degrees of freedom). That distribution describes the largest
# gap among k means, so the family-wise error rate stays at 5% across
# all k(k-1)/2 pairs. Bonferroni (multiply each pairwise t-test p by the
# number of pairs) also controls the FWER but is more conservative —
# it is a different procedure, not "Tukey".


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: calibration curve + Brier score
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Calibration Assessment and ANOVA")
print("=" * 70)

# Load data and fit logistic regression
hdb_recent = load_hdb_recent()
frame, median_price = build_classification_frame(hdb_recent)
X, y, X_mean, X_std, feature_names = build_design_matrix(frame)
n_obs = X.shape[0]
n_positive = int(y.sum())

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

# Brier score
brier = brier_score_loss(y, p_scratch)

# Calibration curve via shared helper
cal_predicted, cal_observed, cal_counts = calibration_bins(y, p_scratch, n_bins=10)

print(f"\n=== Model Calibration ===")
print(f"Brier score: {brier:.6f} (lower = better, 0 = perfect)")
print(f"Brier of a constant 0.5 forecast: 0.25 (reference; the maximum possible is 1)")
print(f"Brier skill vs the 0.5 forecast: {1 - brier / 0.25:.4f} (1 = perfect, 0 = no better)")
print(f"\n{'Bin':>4} {'Predicted':>12} {'Observed':>12} {'Count':>8} {'Gap':>8}")
print("─" * 48)
for i in range(len(cal_predicted)):
    gap = abs(cal_predicted[i] - cal_observed[i])
    print(
        f"{i+1:>4} {cal_predicted[i]:>12.4f} {cal_observed[i]:>12.4f} "
        f"{cal_counts[i]:>8,} {gap:>8.4f}"
    )

# INTERPRETATION: Good calibration = predicted probabilities match
# observed frequencies. If predicted=0.8 but observed=0.6, the model
# is overconfident. Calibration matters when you use probabilities
# for decision-making (not just rankings).

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 <= brier <= 1, "Brier score must be between 0 and 1"
assert brier < 0.25, "Model should beat the constant-0.5 forecast (Brier < 0.25)"
print("\n[ok] Checkpoint 1 passed — calibration assessed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: one-way ANOVA + Tukey HSD
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== One-Way ANOVA: Resale Price by Flat Type ===")

flat_types = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]
anova_groups = []
anova_labels = []

for ft in flat_types:
    group = (
        hdb_recent.filter(pl.col("flat_type") == ft)["resale_price"]
        .to_numpy()
        .astype(np.float64)
    )
    if len(group) > 10:
        anova_groups.append(group)
        anova_labels.append(ft)
        print(
            f"  {ft:<12}: n={len(group):>6,}, "
            f"mean=${group.mean():>10,.0f}, std=${group.std():>8,.0f}"
        )

# Run ANOVA
f_anova, p_anova = stats.f_oneway(*anova_groups)
print(f"\nANOVA F-statistic: {f_anova:.2f}")
p_anova_label = f"= {p_anova:.2e}" if p_anova > 0 else "< 1e-300 (underflows to 0)"
print(f"p-value {p_anova_label}")
print(
    f"{'SIGNIFICANT' if p_anova < 0.05 else 'NOT significant'}: "
    f"{'at least one flat type has a different mean price' if p_anova < 0.05 else 'no evidence of difference'}"
)

# Effect size: eta-squared
all_data = np.concatenate(anova_groups)
grand_mean = all_data.mean()
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in anova_groups)
ss_total = sum(np.sum((g - grand_mean) ** 2) for g in anova_groups)
eta_squared = ss_between / ss_total
print(
    f"Effect size (eta-sq): {eta_squared:.4f} "
    f"({eta_squared:.1%} of variance explained by flat type)"
)

# Post-hoc: Tukey HSD (Tukey-Kramer form, valid for unequal group sizes)
print(f"\n--- Tukey HSD Post-Hoc Comparisons ---")
n_all = len(all_data)
k_groups = len(anova_groups)
df_within = n_all - k_groups
ms_within = sum(np.sum((g - g.mean()) ** 2) for g in anova_groups) / df_within
n_comparisons = k_groups * (k_groups - 1) // 2

# scipy's reference implementation — used to verify the hand computation
tukey = stats.tukey_hsd(*anova_groups)

print(
    f"{'Comparison':<25} {'Diff ($)':>12} {'SE':>10} {'q':>8} "
    f"{'p (Tukey)':>11} {'p (Bonf.)':>11} {'Sig':>5}"
)
print("─" * 88)
tukey_rows = []
for i in range(k_groups):
    for j in range(i + 1, k_groups):
        diff = anova_groups[j].mean() - anova_groups[i].mean()
        # Tukey-Kramer standard error for the studentized range statistic
        se = np.sqrt(
            ms_within / 2 * (1 / len(anova_groups[i]) + 1 / len(anova_groups[j]))
        )
        q_stat = abs(diff) / se
        # Tukey: q against the studentized range distribution (k, N - k)
        p_tukey = float(stats.studentized_range.sf(q_stat, k_groups, df_within))
        # For contrast only — Bonferroni-corrected pairwise t-test.
        # The pairwise t statistic is q / sqrt(2).
        t_stat = q_stat / np.sqrt(2)
        p_bonf = min(2 * stats.t.sf(t_stat, df=df_within) * n_comparisons, 1.0)
        sig = (
            "***"
            if p_tukey < 0.001
            else "**" if p_tukey < 0.01 else "*" if p_tukey < 0.05 else "ns"
        )
        label = f"{anova_labels[i]} vs {anova_labels[j]}"
        tukey_rows.append((i, j, diff, p_tukey))
        print(
            f"{label:<25} ${diff:>10,.0f} {se:>10,.0f} {q_stat:>8.2f} "
            f"{p_tukey:>11.2e} {p_bonf:>11.2e} {sig:>5}"
        )

# Cross-check: the hand-computed Tukey results must match scipy's
# tukey_hsd (its statistic[j, i] is mean_j - mean_i).
max_p_gap = max(abs(p - float(tukey.pvalue[i, j])) for i, j, _, p in tukey_rows)
max_diff_gap = max(abs(d - float(tukey.statistic[j, i])) for i, j, d, _ in tukey_rows)
print(f"\nMax |p_hand - p_scipy|: {max_p_gap:.2e}   max |diff gap|: ${max_diff_gap:.4f}")
# Where the two procedures differ: the critical |t| each one needs for a
# pair to be significant at FWER = 5%.
t_crit_tukey = stats.studentized_range.ppf(0.95, k_groups, df_within) / np.sqrt(2)
t_crit_bonf = stats.t.ppf(1 - 0.05 / (2 * n_comparisons), df=df_within)
print(
    f"Critical |t| at FWER 5%: Tukey {t_crit_tukey:.3f} vs Bonferroni "
    f"{t_crit_bonf:.3f} -> Bonferroni demands a larger gap (more conservative)."
)
print(
    "Tukey p-values bottom out near 4.5e-12 (the precision limit of scipy's\n"
    "studentized-range integral) — read them as 'p < 1e-11'."
)

# INTERPRETATION: ANOVA tells you SOME group differs. Tukey HSD tells
# you WHICH pairs differ. In property, this shows the price premium
# between flat types — critical for valuation models. Executive flats
# command a premium over 3-room; the question is how much.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert f_anova > 0, "F-statistic must be positive"
assert 0 <= eta_squared <= 1, "Eta-squared must be between 0 and 1"
assert len(tukey_rows) == n_comparisons, "Tukey HSD must cover every pair of groups"
assert max_p_gap < 1e-6, "Hand-computed Tukey p-values must match scipy.stats.tukey_hsd"
assert max_diff_gap < 1e-6, "Mean differences must match scipy.stats.tukey_hsd"
assert t_crit_tukey < t_crit_bonf, "Tukey's critical value should be below Bonferroni's"
print("\n[ok] Checkpoint 2 passed — ANOVA + Tukey HSD completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: calibration plot + ANOVA box plots
# ════════════════════════════════════════════════════════════════════════

# Plot 1: Calibration curve
fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=cal_predicted,
        y=cal_observed,
        mode="markers+lines",
        name="Model",
    )
)
fig1.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name="Perfect calibration",
        line={"dash": "dash", "color": "grey"},
    )
)
fig1.update_layout(
    title=f"Calibration Curve (Brier={brier:.4f})",
    xaxis_title="Predicted Probability",
    yaxis_title="Observed Frequency",
)
fig1.write_html(str(OUTPUT_DIR / "calibration.html"))
print(f"Saved: {OUTPUT_DIR / 'calibration.html'}")

# Plot 2: ANOVA box plots
fig2 = go.Figure()
for label, group in zip(anova_labels, anova_groups):
    fig2.add_trace(go.Box(y=group[:5000], name=label))
fig2.update_layout(
    title=f"Resale Price by Flat Type (ANOVA F={f_anova:.0f}, p {p_anova_label})",
    yaxis_title="Resale Price ($)",
    xaxis_title="Flat Type",
)
fig2.write_html(str(OUTPUT_DIR / "anova_boxplot.html"))
print(f"Saved: {OUTPUT_DIR / 'anova_boxplot.html'}")

# Plot 3: Eta-squared bar — variance decomposition
fig3 = go.Figure()
fig3.add_trace(
    go.Bar(
        x=["Flat Type (between)", "Within groups"],
        y=[eta_squared, 1 - eta_squared],
        marker_color=["#3498db", "#bdc3c7"],
    )
)
fig3.update_layout(
    title=f"Variance Decomposition (eta-sq = {eta_squared:.3f})",
    yaxis_title="Proportion of Total Variance",
)
fig3.write_html(str(OUTPUT_DIR / "variance_decomposition.html"))
print(f"Saved: {OUTPUT_DIR / 'variance_decomposition.html'}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — calibration + ANOVA visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: flat-type valuation bands for HDB resale flats
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore property-valuation firm is deciding whether to
# publish quick-quote valuation bands by HDB flat type. ANOVA shows
# that flat type explains a significant portion of price variance —
# but HOW MUCH matters for whether type-only bands are defensible.
#
# If eta-squared is high (>0.3), flat type is a dominant price driver
# and differentiated bands by type are a reasonable first cut. If low
# (<0.1), other factors (location, condition, floor) dominate and
# per-type bands would be misleading.

# Price differentials (Tukey HSD confirmed every pair differs)
print(f"\n=== Real-World Application: Flat-Type Valuation Bands ===")
print(f"\n  Variance explained by flat type: {eta_squared:.1%}")
if eta_squared > 0.3:
    print(f"  HIGH — flat type is a dominant price driver")
elif eta_squared > 0.1:
    print(f"  MODERATE — flat type matters but is not sufficient alone")
else:
    print(f"  LOW — other factors dominate price variation")

# Average price per type and differential from 3-room baseline
baseline_type = "3 ROOM"
if baseline_type in anova_labels:
    baseline_idx = anova_labels.index(baseline_type)
    baseline_mean = anova_groups[baseline_idx].mean()
    print(f"\n  Price differentials vs {baseline_type} (${baseline_mean:,.0f}):")
    for label, group in zip(anova_labels, anova_groups):
        diff = group.mean() - baseline_mean
        pct = diff / baseline_mean * 100
        print(f"    {label:<12}: ${group.mean():>10,.0f}  ({pct:+.1f}%)")

# Calibration for policy confidence
print(f"\n  Model calibration (Brier): {brier:.4f}")
print(f"  Brier skill score: {1 - brier / 0.25:.3f}")
if brier < 0.15:
    print(f"  Well-calibrated — probabilities can inform premium pricing")
else:
    print(f"  Needs recalibration before using probabilities in pricing")

# BUSINESS IMPACT (ILLUSTRATIVE assumptions, not published figures):
# suppose the firm issues 5,000 quick quotes a year, 8% of them are
# disputed by clients, and each dispute costs S$1,200 of re-valuation
# time. Bands that reflect the demonstrated type differentials might
# cut disputes by a fraction that grows with how much of the variance
# flat type actually explains.
quotes_per_year = 5_000  # illustrative
dispute_rate = 0.08  # illustrative
cost_per_dispute = 1_200  # S$, illustrative
baseline_dispute_cost = quotes_per_year * dispute_rate * cost_per_dispute
# Optimistic upper bound: disputes fall in proportion to the share of
# price variance the bands explain (eta-squared).
max_saving = baseline_dispute_cost * eta_squared
print(
    f"\n  Illustrative dispute cost: S${baseline_dispute_cost:,.0f}/year; "
    f"upper-bound saving if disputes fall by eta-sq ({eta_squared:.0%}): "
    f"S${max_saving:,.0f}/year"
)
#
# The calibration assessment tells the firm whether the model's
# probability outputs ("P(above-median price)") can be quoted directly
# to clients (well-calibrated) or only used for ranking (poorly
# calibrated but good discrimination).
#
# LIMITATIONS:
#   - ANOVA assumes normally distributed residuals. HDB resale prices
#     are right-skewed; the test is robust to this at large n but
#     interpretation of effect sizes should be cautious.
#   - Tukey HSD assumes equal variance across groups. Executive flats
#     have higher variance than 2-room flats. A Welch-corrected
#     version or Games-Howell test would be more precise.
#   - The analysis pools all towns. Ang Mo Kio 3-room prices differ
#     from Bishan 3-room prices; a nested ANOVA (type within town)
#     would separate these effects.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Sigmoid: numerically stable implementation, verified properties
  [x] Logistic regression from scratch: Bernoulli MLE with gradient
  [x] sklearn comparison: validated implementation correctness
  [x] Odds ratios: exp(beta) = multiplicative change in odds per unit
  [x] Threshold optimisation: cost matrix drives the decision boundary
  [x] Confusion matrix: TP, FP, TN, FN and derived metrics
  [x] Precision, recall, F1: trade-offs in classification
  [x] ROC curve + AUC: discrimination across all thresholds
  [x] Precision-recall curve: better for imbalanced classes
  [x] Calibration: predicted probabilities match observed frequencies
  [x] Brier score: overall measure of probabilistic accuracy
  [x] One-way ANOVA: F-test for 3+ group means, eta-squared effect size
  [x] Tukey HSD: studentized-range pairwise comparisons (not Bonferroni)

  KEY INSIGHT: Classification metrics are not just academic — the
  CHOICE of metric directly determines business outcomes. Accuracy
  hides class imbalance; F1 balances precision and recall; the cost
  matrix encodes what your organisation actually loses from each
  type of error. Calibration tells you whether the probability
  itself can be trusted, not just the ranking.

  Next: Exercise 7 — CUPED variance reduction for A/B tests, Bayesian
  A/B testing with posterior probabilities, sequential testing with
  always-valid p-values, and Difference-in-Differences for when
  randomisation is impossible.
"""
)

print("\n[ok] Exercise 6 complete — Logistic Regression and Classification")
