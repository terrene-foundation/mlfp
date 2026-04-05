# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT2 — Exercise 3: Hypothesis Testing
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Apply the Neyman-Pearson framework with power analysis,
#   multiple testing corrections, and permutation tests.
#
# TASKS:
#   1. Load e-commerce A/B test data and perform sanity checks (SRM)
#   2. Compute power analysis — what MDE can we detect?
#   3. Run hypothesis tests for conversion rate (proportions z-test)
#   4. Test multiple metrics simultaneously (conversion, revenue, AOV)
#   5. Apply multiple testing corrections (Bonferroni, BH-FDR)
#   6. Implement a permutation test as distribution-free alternative
#
# THEORY:
#   - Neyman-Pearson: H₀ vs H₁, Type I (α) vs Type II (β) errors
#   - Power = 1 - β = P(reject H₀ | H₁ true)
#   - SRM: χ² test on observed vs expected sample sizes
#   - Bonferroni: α_adj = α/m (controls FWER)
#   - BH-FDR: rank p-values, reject if p(k) ≤ (k/m)α (controls FDR)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats
from kailash_ml import ModelVisualizer

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
ab_data = loader.load("ascent01", "ecommerce_ab_test.csv")

print("=== A/B Test Data ===")
print(f"Shape: {ab_data.shape}")
print(f"Columns: {ab_data.columns}")
print(ab_data.head(5))

control = ab_data.filter(pl.col("group") == "control")
treatment = ab_data.filter(pl.col("group") == "treatment")

n_control = control.height
n_treatment = treatment.height
n_total = ab_data.height

print(f"\nControl:   n = {n_control:,}")
print(f"Treatment: n = {n_treatment:,}")
print(f"Total:     n = {n_total:,}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1: Sanity checks — Sample Ratio Mismatch (SRM)
# ════════════════════════════════════════════════════════════════════════
# SRM test: if the experiment was 50/50, do observed counts match?
# Uses χ² goodness-of-fit test.

expected_ratio = 0.5
expected_control = n_total * expected_ratio
expected_treatment = n_total * (1 - expected_ratio)

observed = np.array([n_control, n_treatment])
expected = np.array([expected_control, expected_treatment])

# TODO: Run the chi-square goodness-of-fit test for SRM
chi2_stat, srm_p_value = ____  # Hint: stats.chisquare(observed, f_exp=expected)

print(f"\n=== Sample Ratio Mismatch Check ===")
print(f"Expected ratio: {expected_ratio:.0%} / {1 - expected_ratio:.0%}")
print(f"Observed ratio: {n_control / n_total:.4f} / {n_treatment / n_total:.4f}")
print(f"χ² statistic: {chi2_stat:.4f}")
print(f"p-value: {srm_p_value:.6f}")

if srm_p_value < 0.01:
    print("⚠ SRM DETECTED — investigate randomisation before trusting results!")
else:
    print("✓ No SRM detected — sample split is consistent with 50/50")


# ════════════════════════════════════════════════════════════════════════
# TASK 2: Power analysis — minimum detectable effect (MDE)
# ════════════════════════════════════════════════════════════════════════
# MDE ≈ (z_{α/2} + z_β) * √(p(1-p)(1/n₁ + 1/n₂))

p_control = control["converted"].mean()
print(f"\n=== Power Analysis ===")
print(f"Baseline conversion rate: {p_control:.4f} ({p_control:.2%})")

alpha = 0.05
power_target = 0.80
# TODO: Compute the z-score at alpha/2 (two-tailed critical value)
z_alpha_half = ____  # Hint: stats.norm.ppf(1 - alpha / 2)
# TODO: Compute the z-score at the target power level
z_beta = ____  # Hint: stats.norm.ppf(power_target)

# MDE for two-proportion z-test
pooled_se = np.sqrt(p_control * (1 - p_control) * (1 / n_control + 1 / n_treatment))
# TODO: Compute the minimum detectable effect
mde = ____  # Hint: (z_alpha_half + z_beta) * pooled_se

print(f"α = {alpha}, Power = {power_target:.0%}")
print(f"z_{{α/2}} = {z_alpha_half:.3f}, z_β = {z_beta:.3f}")
print(f"Minimum Detectable Effect (MDE): {mde:.6f} ({mde:.4%} absolute)")
print(f"Relative MDE: {mde / p_control:.2%} of baseline")

effect_sizes = np.linspace(0, mde * 3, 100)
powers = []
for delta in effect_sizes:
    ncp = delta / pooled_se
    power_val = (
        1 - stats.norm.cdf(z_alpha_half - ncp) + stats.norm.cdf(-z_alpha_half - ncp)
    )
    powers.append(power_val)

print(f"\nPower at selected effect sizes:")
for es_mult in [0.5, 1.0, 1.5, 2.0]:
    idx = int(es_mult / 3 * 99)
    print(f"  Effect = {effect_sizes[idx]:.4%}: Power = {powers[idx]:.1%}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3: Primary test — conversion rate (two-proportion z-test)
# ════════════════════════════════════════════════════════════════════════

p_treatment = treatment["converted"].mean()
p_pooled = ab_data["converted"].mean()

# TODO: Compute the pooled standard error for the difference in proportions
se_diff = (
    ____  # Hint: np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_control + 1 / n_treatment))
)
# TODO: Compute the z-statistic
z_stat = ____  # Hint: (p_treatment - p_control) / se_diff
# TODO: Compute the two-tailed p-value
p_value_conversion = ____  # Hint: 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Cohen's h for proportions
cohens_h = 2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))

print(f"\n=== Primary Metric: Conversion Rate ===")
print(f"Control conversion:   {p_control:.4f} ({p_control:.2%})")
print(f"Treatment conversion: {p_treatment:.4f} ({p_treatment:.2%})")
print(
    f"Absolute lift: {p_treatment - p_control:+.4f} ({(p_treatment - p_control):.4%})"
)
print(f"Relative lift: {(p_treatment - p_control) / p_control:+.2%}")
print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value_conversion:.6f}")
print(f"Cohen's h: {cohens_h:.4f}")
print(
    f"{'SIGNIFICANT' if p_value_conversion < alpha else 'NOT significant'} at α={alpha}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4: Multiple metrics — conversion, revenue, average order value
# ════════════════════════════════════════════════════════════════════════

metrics_results = {}

metrics_results["conversion_rate"] = {
    "control": p_control,
    "treatment": p_treatment,
    "stat": z_stat,
    "p_value": p_value_conversion,
    "test": "two-proportion z-test",
}

rev_control = control["revenue"].to_numpy().astype(np.float64)
rev_treatment = treatment["revenue"].to_numpy().astype(np.float64)

# TODO: Run Mann-Whitney U test on revenue (revenue is right-skewed)
u_stat, p_value_revenue = (
    ____  # Hint: stats.mannwhitneyu(rev_treatment, rev_control, alternative="two-sided")
)

metrics_results["revenue_per_user"] = {
    "control": rev_control.mean(),
    "treatment": rev_treatment.mean(),
    "stat": u_stat,
    "p_value": p_value_revenue,
    "test": "Mann-Whitney U",
}

aov_control = (
    control.filter(pl.col("converted") == 1)["revenue"].to_numpy().astype(np.float64)
)
aov_treatment = (
    treatment.filter(pl.col("converted") == 1)["revenue"].to_numpy().astype(np.float64)
)

# TODO: Run Welch's t-test on AOV among converters only
t_stat, p_value_aov = (
    ____  # Hint: stats.ttest_ind(aov_treatment, aov_control, equal_var=False)
)

metrics_results["avg_order_value"] = {
    "control": aov_control.mean(),
    "treatment": aov_treatment.mean(),
    "stat": t_stat,
    "p_value": p_value_aov,
    "test": "Welch's t-test",
}

if "pages_viewed" in ab_data.columns:
    pages_control = control["pages_viewed"].to_numpy().astype(np.float64)
    pages_treatment = treatment["pages_viewed"].to_numpy().astype(np.float64)
    t_pages, p_value_pages = stats.ttest_ind(
        pages_treatment, pages_control, equal_var=False
    )
    metrics_results["pages_viewed"] = {
        "control": pages_control.mean(),
        "treatment": pages_treatment.mean(),
        "stat": t_pages,
        "p_value": p_value_pages,
        "test": "Welch's t-test",
    }

print(f"\n=== Multiple Metric Results (unadjusted) ===")
print(f"{'Metric':<20} {'Control':>12} {'Treatment':>12} {'p-value':>10} {'Test':<20}")
print("─" * 80)
for name, r in metrics_results.items():
    sig = (
        "***"
        if r["p_value"] < 0.001
        else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
    )
    print(
        f"{name:<20} {r['control']:>12.4f} {r['treatment']:>12.4f} "
        f"{r['p_value']:>10.6f} {r['test']:<20} {sig}"
    )

m = len(metrics_results)
fwer = 1 - (1 - alpha) ** m
print(f"\nNumber of tests (m): {m}")
print(f"Family-wise error rate (uncorrected): {fwer:.4f} ({fwer:.1%})")


# ════════════════════════════════════════════════════════════════════════
# TASK 5: Multiple testing corrections
# ════════════════════════════════════════════════════════════════════════

p_values = np.array([r["p_value"] for r in metrics_results.values()])
metric_names = list(metrics_results.keys())

# Bonferroni correction: α_adj = α/m
# TODO: Compute the Bonferroni-adjusted significance threshold
bonferroni_alpha = ____  # Hint: alpha / m
bonferroni_significant = p_values < bonferroni_alpha

print(f"\n=== Bonferroni Correction (FWER control) ===")
print(f"Adjusted α = {alpha}/{m} = {bonferroni_alpha:.4f}")
for i, name in enumerate(metric_names):
    sig = "SIGNIFICANT" if bonferroni_significant[i] else "not significant"
    print(f"  {name}: p={p_values[i]:.6f} → {sig}")

# Benjamini-Hochberg (BH-FDR) correction
# Sort p-values, find largest k where p(k) ≤ (k/m)α
sorted_indices = np.argsort(p_values)
sorted_p = p_values[sorted_indices]
# TODO: Compute the BH thresholds: (k+1)/m * alpha for each rank k
bh_thresholds = np.array([____ for k in range(m)])  # Hint: (k + 1) / m * alpha

bh_significant_mask = np.zeros(m, dtype=bool)
max_k = -1
for k in range(m):
    if sorted_p[k] <= bh_thresholds[k]:
        max_k = k

if max_k >= 0:
    bh_significant_mask[sorted_indices[: max_k + 1]] = True

q_values = np.zeros(m)
sorted_q = np.zeros(m)
sorted_q[m - 1] = sorted_p[m - 1]
for k in range(m - 2, -1, -1):
    sorted_q[k] = min(sorted_p[k] * m / (k + 1), sorted_q[k + 1])
q_values[sorted_indices] = sorted_q

print(f"\n=== Benjamini-Hochberg Correction (FDR control) ===")
print(f"Target FDR: {alpha}")
for i, name in enumerate(metric_names):
    sig = "SIGNIFICANT" if bh_significant_mask[i] else "not significant"
    print(f"  {name}: p={p_values[i]:.6f}, q={q_values[i]:.6f} → {sig}")

print(f"\n=== Correction Comparison ===")
print(f"{'Metric':<20} {'Raw p':>10} {'Bonferroni':>12} {'BH-FDR':>12}")
print("─" * 60)
for i, name in enumerate(metric_names):
    bonf = "sig" if bonferroni_significant[i] else "ns"
    bh = "sig" if bh_significant_mask[i] else "ns"
    print(f"{name:<20} {p_values[i]:>10.6f} {bonf:>12} {bh:>12}")


# ════════════════════════════════════════════════════════════════════════
# TASK 6: Permutation test for conversion rate
# ════════════════════════════════════════════════════════════════════════
# Algorithm:
#   1. Pool all observations
#   2. Randomly assign to "control" and "treatment" (preserving sizes)
#   3. Compute test statistic on permuted data
#   4. Repeat 10K times → null distribution
#   5. p-value = proportion of permuted statistics ≥ observed

rng = np.random.default_rng(seed=42)
n_permutations = 10_000

observed_diff = p_treatment - p_control
all_converted = ab_data["converted"].to_numpy()

# TODO: Implement the permutation loop
perm_diffs = np.zeros(n_permutations)
for i in range(n_permutations):
    # TODO: Shuffle all_converted, compute difference in means between the two halves
    perm = ____  # Hint: rng.permutation(all_converted)
    perm_control_rate = ____  # Hint: perm[:n_control].mean()
    perm_treatment_rate = ____  # Hint: perm[n_control:].mean()
    perm_diffs[i] = perm_treatment_rate - perm_control_rate

# TODO: Compute the two-tailed permutation p-value
perm_p_value = ____  # Hint: np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

print(f"\n=== Permutation Test (conversion rate) ===")
print(f"Observed difference: {observed_diff:+.6f}")
print(f"Permutation p-value: {perm_p_value:.6f}")
print(f"Parametric p-value:  {p_value_conversion:.6f}")
print(
    f"Agreement: {'YES' if (perm_p_value < alpha) == (p_value_conversion < alpha) else 'NO'}"
)


# ════════════════════════════════════════════════════════════════════════
# Visualise results with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

comparison_data = {
    "Control": {name: r["control"] for name, r in metrics_results.items()},
    "Treatment": {name: r["treatment"] for name, r in metrics_results.items()},
}
fig_comparison = viz.metric_comparison(comparison_data)
fig_comparison.update_layout(title="A/B Test: Control vs Treatment Across Metrics")
fig_comparison.write_html("ex3_ab_test_comparison.html")
print(f"\nSaved: ex3_ab_test_comparison.html")

power_metrics = {
    "Power": powers,
}
fig_power = viz.training_history(power_metrics, x_label="Effect Size Index")
fig_power.update_layout(title="Statistical Power vs Effect Size")
fig_power.write_html("ex3_power_curve.html")
print("Saved: ex3_power_curve.html")

print("\n✓ Exercise 3 complete — hypothesis testing with multiple testing correction")
