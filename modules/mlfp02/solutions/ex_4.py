# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 4: A/B Testing and Experiment Design
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Design a complete A/B experiment with pre-registered hypotheses
#   - Compute required sample sizes via power analysis before data collection
#   - Simulate experiment data with a known effect to validate your pipeline
#   - Detect Sample Ratio Mismatch (SRM) and understand its causes
#   - Construct and interpret confidence intervals for treatment effects
#
# PREREQUISITES: Complete Exercise 3 — you should understand hypothesis testing,
#   p-values, power, and why multiple testing corrections are needed.
#
# ESTIMATED TIME: 75 minutes
#
# TASKS:
#   1. Load experiment data and perform exploratory analysis
#   2. Design an A/B experiment: formulate hypotheses, compute sample size
#   3. Simulate experiment data with a known treatment effect
#   4. Detect Sample Ratio Mismatch (SRM) using chi-squared test
#   5. Run hypothesis test (Welch's t-test) on A/B results
#   6. Compute confidence interval for treatment effect
#   7. Create a data collection plan (Why/What/Where/How/Frequency)
#
# DATASET: Multi-arm experiment data
#   Source: Simulated e-commerce experiment with control and multiple treatments
#   Key columns: experiment_group, metric_value (engagement score)
#
# THEORY:
#   A/B testing is the gold standard for causal inference in business.
#   Pre-registration: decide hypothesis, sample size, and stopping rule
#   BEFORE collecting data to prevent p-hacking. Power analysis answers:
#     n = (z_{α/2} + z_β)² × 2σ² / δ²
#   SRM (Sample Ratio Mismatch): chi-squared test on observed vs expected split.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from kailash_ml import ModelVisualizer
from scipy import stats

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
experiment = loader.load("mlfp02", "experiment_data.parquet")

print("=" * 70)
print("  MLFP02 Exercise 4: A/B Testing and Experiment Design")
print("=" * 70)
print(f"\n  Data loaded: experiment_data.parquet")
print(f"  Shape: {experiment.shape}")
print(f"  Columns: {experiment.columns}")
print(experiment.head(5))

# Understand the group structure
group_counts = experiment["experiment_group"].value_counts().sort("experiment_group")
print(f"\n--- Group Allocation ---")
print(group_counts)

# For this exercise we focus on a classic two-arm A/B test:
# control vs treatment_a (the largest treatment group)
ab_data = experiment.filter(
    pl.col("experiment_group").is_in(["control", "treatment_a"])
)

control = ab_data.filter(pl.col("experiment_group") == "control")
treatment = ab_data.filter(pl.col("experiment_group") == "treatment_a")

n_control = control.height
n_treatment = treatment.height
n_total = ab_data.height

print(f"\n=== Two-Arm A/B Subset ===")
print(f"Control:   n = {n_control:,}")
print(f"Treatment: n = {n_treatment:,}")
print(f"Total:     n = {n_total:,}")
print(f"Observed split: {n_control / n_total:.4f} / {n_treatment / n_total:.4f}")

# Summary statistics for the primary metric
ctrl_values = control["metric_value"].to_numpy().astype(np.float64)
treat_values = treatment["metric_value"].to_numpy().astype(np.float64)

print(f"\nControl  — mean: {ctrl_values.mean():.4f}, std: {ctrl_values.std():.4f}")
print(f"Treatment — mean: {treat_values.mean():.4f}, std: {treat_values.std():.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Design an A/B experiment — hypothesis formulation
# ══════════════════════════════════════════════════════════════════════
# THEORY: A well-designed experiment starts with a clear hypothesis.
#
# The hypothesis must be:
#   - Specific: what metric, what direction, what magnitude
#   - Testable: can be falsified with data
#   - Pre-registered: stated BEFORE seeing data
#
# Null hypothesis (H₀): Treatment has no effect on the primary metric.
#   μ_treatment = μ_control
# Alternative hypothesis (H₁): Treatment increases the primary metric.
#   μ_treatment > μ_control (one-sided) or μ_treatment ≠ μ_control (two-sided)

print("\n" + "=" * 70)
print("TASK 1: Experiment Design — Hypothesis Formulation")
print("=" * 70)

print("""
Scenario: An e-commerce platform wants to test whether a new
recommendation algorithm (treatment_a) increases user metric_value
(engagement score) compared to the existing algorithm (control).

Hypotheses:
  H₀: μ_treatment = μ_control  (no effect)
  H₁: μ_treatment ≠ μ_control  (two-sided — effect in either direction)

Design parameters:
  - Significance level (α): 0.05 (5% false positive rate)
  - Power (1-β): 0.80 (80% chance of detecting a real effect)
  - Primary metric: metric_value (engagement score)
  - Randomisation unit: user
  - Allocation: equal (50/50 intended)

Key design principles applied:
  1. Randomisation — users randomly assigned to control/treatment
  2. Equal allocation — 50/50 maximises power for a given total n
  3. Single treatment — comparing exactly one variant at a time
  4. Pre-registration — hypotheses and stopping rule defined upfront
""")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Power analysis — compute required sample size
# ══════════════════════════════════════════════════════════════════════
# THEORY: Power analysis tells us the minimum sample size to detect
# a given effect size (Minimum Detectable Effect, MDE) at a given
# significance level and power.
#
# For a two-sample t-test with equal allocation:
#   n_per_group = (z_{α/2} + z_β)² × 2σ² / δ²
#
# Rule of thumb: MDE should be the smallest effect that is
# practically meaningful to the business.

print("\n" + "=" * 70)
print("TASK 2: Power Analysis — Required Sample Size")
print("=" * 70)

alpha = 0.05
power_target = 0.80
z_alpha_half = stats.norm.ppf(1 - alpha / 2)  # 1.96
z_beta = stats.norm.ppf(power_target)          # 0.842

# Estimate pooled standard deviation from the control group
sigma_pooled = ctrl_values.std(ddof=1)

# Define the MDE as a percentage of the control mean
relative_mde_pct = 2.0
mde_absolute = ctrl_values.mean() * (relative_mde_pct / 100)

# Required sample size per group
n_required_per_group = math.ceil(
    (z_alpha_half + z_beta) ** 2 * 2 * sigma_pooled ** 2 / mde_absolute ** 2
)
n_required_total = 2 * n_required_per_group

# Cohen's d (standardised effect size)
cohens_d = mde_absolute / sigma_pooled

print(f"Baseline metric (control mean): {ctrl_values.mean():.4f}")
print(f"Pooled σ (from control): {sigma_pooled:.4f}")
print(f"Minimum Detectable Effect (MDE): {mde_absolute:.4f} ({relative_mde_pct}% relative)")
print(f"Cohen's d: {cohens_d:.4f}", end="")
if cohens_d < 0.2:
    print(" (small effect)")
elif cohens_d < 0.5:
    print(" (small-medium effect)")
elif cohens_d < 0.8:
    print(" (medium effect)")
else:
    print(" (large effect)")

print(f"\nDesign parameters:")
print(f"  α = {alpha} (two-sided)")
print(f"  Power = {power_target:.0%}")
print(f"  z_{{α/2}} = {z_alpha_half:.4f}")
print(f"  z_β = {z_beta:.4f}")
print(f"\nRequired sample size:")
print(f"  Per group: {n_required_per_group:,}")
print(f"  Total:     {n_required_total:,}")
print(f"\nActual sample size: {n_total:,} ({n_total / n_required_total:.1f}x required)")
# INTERPRETATION: If actual n < required n, the experiment is underpowered —
# even if the treatment effect is real at the MDE, we may miss it. If the
# experiment is already collected and underpowered, you must accept a larger MDE.

if n_total >= n_required_total:
    print("  --> SUFFICIENT: experiment is adequately powered")
else:
    print("  --> UNDERPOWERED: need more observations or accept larger MDE")

# Power curve: compute power at different sample sizes
sample_sizes = np.arange(500, n_required_per_group * 3, max(500, n_required_per_group // 20))
power_at_n = []
for n_i in sample_sizes:
    se_i = sigma_pooled * np.sqrt(2 / n_i)
    ncp_i = mde_absolute / se_i  # non-centrality parameter
    power_i = (
        1 - stats.norm.cdf(z_alpha_half - ncp_i)
        + stats.norm.cdf(-z_alpha_half - ncp_i)
    )
    power_at_n.append(power_i)

print(f"\nPower at selected sample sizes (per group):")
for frac in [0.25, 0.5, 1.0, 1.5, 2.0]:
    idx = np.argmin(np.abs(sample_sizes - n_required_per_group * frac))
    print(f"  n = {sample_sizes[idx]:>8,}: power = {power_at_n[idx]:.1%}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert n_required_per_group > 0, "Required sample size must be positive"
assert cohens_d > 0, "Cohen's d must be positive"
assert mde_absolute > 0, "MDE must be positive"
assert len(power_at_n) == len(sample_sizes), "Should have power for each sample size"
print("\n✓ Checkpoint 1 passed — power analysis completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Simulate experiment data with known treatment effect
# ══════════════════════════════════════════════════════════════════════
# THEORY: Simulation is essential for validating your analysis pipeline.
# By injecting a KNOWN effect, you can verify that your test correctly
# detects it. This is a "positive control" — if you can't detect a
# known effect, your pipeline is broken.

print("\n" + "=" * 70)
print("TASK 3: Simulate Experiment with Known Treatment Effect")
print("=" * 70)

rng = np.random.default_rng(seed=42)

sim_n_per_group = 10_000
true_effect = 2.0  # True treatment effect (absolute lift in metric_value)
sim_mu_control = ctrl_values.mean()
sim_sigma = sigma_pooled

# Generate simulated data
sim_control = rng.normal(loc=sim_mu_control, scale=sim_sigma, size=sim_n_per_group)
sim_treatment = rng.normal(
    loc=sim_mu_control + true_effect, scale=sim_sigma, size=sim_n_per_group
)

# Build a polars DataFrame for the simulated experiment
sim_df = pl.DataFrame({
    "user_id": [f"SIM-{i:06d}" for i in range(2 * sim_n_per_group)],
    "group": ["control"] * sim_n_per_group + ["treatment"] * sim_n_per_group,
    "metric_value": np.concatenate([sim_control, sim_treatment]).tolist(),
})

print(f"Simulated experiment:")
print(f"  True control mean:   {sim_mu_control:.4f}")
print(f"  True treatment mean: {sim_mu_control + true_effect:.4f}")
print(f"  True effect (δ):     {true_effect:.4f}")
print(f"  σ:                   {sim_sigma:.4f}")
print(f"  n per group:         {sim_n_per_group:,}")
print(f"  Total:               {2 * sim_n_per_group:,}")

# Quick sanity check
sim_ctrl_mean = sim_control.mean()
sim_treat_mean = sim_treatment.mean()
print(f"\nRealized values:")
print(f"  Control mean:   {sim_ctrl_mean:.4f}")
print(f"  Treatment mean: {sim_treat_mean:.4f}")
print(f"  Observed diff:  {sim_treat_mean - sim_ctrl_mean:.4f} (true: {true_effect:.4f})")
# INTERPRETATION: The realized difference should be close to the true effect.
# With n=10,000 per group, sampling variation is tiny. This simulation is our
# "positive control" — if the test below doesn't detect this effect, the
# pipeline has a bug.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(sim_control) == sim_n_per_group, "Simulated control group size mismatch"
assert len(sim_treatment) == sim_n_per_group, "Simulated treatment group size mismatch"
assert abs(sim_treat_mean - sim_ctrl_mean - true_effect) < 3.0, \
    f"Simulated effect should be within 3.0 of {true_effect}, got {sim_treat_mean - sim_ctrl_mean:.4f}"
print("\n✓ Checkpoint 2 passed — simulation matches expected parameters\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Detect SRM (Sample Ratio Mismatch) using chi-squared test
# ══════════════════════════════════════════════════════════════════════
# THEORY: SRM is the single most important sanity check in A/B testing.
# The χ² goodness-of-fit test compares observed counts to expected:
#   χ² = Σ (O_i - E_i)² / E_i
# Rule of thumb: p < 0.01 → SRM detected, investigate before proceeding

print("\n" + "=" * 70)
print("TASK 4: Sample Ratio Mismatch (SRM) Detection")
print("=" * 70)

# ---- SRM on simulated data (should pass — we designed 50/50) ----
print("\n--- SRM Check on Simulated Data (designed 50/50) ---")
sim_obs = np.array([sim_n_per_group, sim_n_per_group])
sim_exp = np.array([sim_n_per_group, sim_n_per_group])
sim_chi2, sim_srm_p = stats.chisquare(sim_obs, f_exp=sim_exp)

print(f"Observed: control={sim_obs[0]:,}, treatment={sim_obs[1]:,}")
print(f"Expected: control={sim_exp[0]:,}, treatment={sim_exp[1]:,}")
print(f"χ² = {sim_chi2:.4f}, p = {sim_srm_p:.6f}")
print(f"Result: {'SRM DETECTED' if sim_srm_p < 0.01 else 'No SRM (as expected)'}")

# ---- SRM on real data (likely fails — unequal allocation observed) ----
print("\n--- SRM Check on Real Data (intended 50/50) ---")
real_obs = np.array([n_control, n_treatment])
real_exp_per_group = n_total / 2
real_expected = np.array([real_exp_per_group, real_exp_per_group])

chi2_stat, srm_p_value = stats.chisquare(real_obs, f_exp=real_expected)

print(f"Observed: control={n_control:,}, treatment={n_treatment:,}")
print(f"Expected: control={real_exp_per_group:,.0f}, treatment={real_exp_per_group:,.0f}")
print(f"Observed ratio: {n_control / n_total:.4f} / {n_treatment / n_total:.4f}")
print(f"χ² = {chi2_stat:.4f}")
print(f"p-value = {srm_p_value:.2e}")

if srm_p_value < 0.01:
    print("\nSRM DETECTED — the observed split deviates significantly from 50/50.")
    print("Possible causes to investigate:")
    print("  1. Bot filtering that removed more control/treatment users")
    print("  2. Technical issues causing differential drop-off")
    print("  3. Randomisation bug in the assignment mechanism")
    print("  4. Population filter applied post-randomisation")
    print("\nProceeding with analysis, but results should be treated with caution.")
else:
    print("\nNo SRM detected — sample split is consistent with 50/50 design.")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert sim_chi2 == 0.0, "Simulated equal groups should give χ²=0"
assert 0 <= srm_p_value <= 1, "SRM p-value must be a valid probability"
print("\n✓ Checkpoint 3 passed — SRM detection completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Hypothesis test — Welch's t-test on A/B results
# ══════════════════════════════════════════════════════════════════════
# THEORY: Welch's t-test is preferred over Student's t-test because it
# does not assume equal variances between groups.

print("\n" + "=" * 70)
print("TASK 5: Hypothesis Test — Welch's t-test")
print("=" * 70)

# ---- Test on simulated data (known effect = 2.0) ----
print("\n--- Simulated Data (true effect = 2.0) ---")
sim_t_stat, sim_p_value = stats.ttest_ind(
    sim_treatment, sim_control, equal_var=False, alternative="two-sided"
)

print(f"Control mean:   {sim_control.mean():.4f}")
print(f"Treatment mean: {sim_treatment.mean():.4f}")
print(f"Observed diff:  {sim_treatment.mean() - sim_control.mean():.4f}")
print(f"t-statistic:    {sim_t_stat:.4f}")
print(f"p-value:        {sim_p_value:.2e}")
print(f"Result: {'SIGNIFICANT' if sim_p_value < alpha else 'NOT significant'} at α={alpha}")
if sim_p_value < alpha:
    print("  --> Correctly detected the known treatment effect (true positive)")

# ---- Test on real data ----
print("\n--- Real Experiment Data ---")
real_t_stat, real_p_value = stats.ttest_ind(
    treat_values, ctrl_values, equal_var=False, alternative="two-sided"
)

observed_diff = treat_values.mean() - ctrl_values.mean()
relative_lift = observed_diff / ctrl_values.mean() * 100

print(f"Control mean:   {ctrl_values.mean():.4f}")
print(f"Treatment mean: {treat_values.mean():.4f}")
print(f"Absolute diff:  {observed_diff:+.4f}")
print(f"Relative lift:  {relative_lift:+.2f}%")
print(f"t-statistic:    {real_t_stat:.4f}")
print(f"p-value:        {real_p_value:.6f}")
print(f"Result: {'SIGNIFICANT' if real_p_value < alpha else 'NOT significant'} at α={alpha}")
# INTERPRETATION: Significance tells you whether the effect is real;
# Cohen's d tells you whether it's meaningful. A highly significant but
# tiny effect may not justify the cost of deploying the new feature.

# Cohen's d for effect size interpretation
pooled_std_real = np.sqrt(
    (ctrl_values.var(ddof=1) + treat_values.var(ddof=1)) / 2
)
cohens_d_real = observed_diff / pooled_std_real
print(f"Cohen's d: {cohens_d_real:.4f}", end="")
if abs(cohens_d_real) < 0.2:
    print(" (negligible/small)")
elif abs(cohens_d_real) < 0.5:
    print(" (small-medium)")
elif abs(cohens_d_real) < 0.8:
    print(" (medium)")
else:
    print(" (large)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert sim_p_value < alpha, "Positive control (known effect=2.0) must be detected as significant"
assert 0 <= real_p_value <= 1, "Real data p-value must be a valid probability"
print("\n✓ Checkpoint 4 passed — hypothesis tests completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Confidence interval for treatment effect
# ══════════════════════════════════════════════════════════════════════
# THEORY: A CI is more informative than a p-value — it tells you HOW BIG
# the effect likely is. The Welch-Satterthwaite degrees of freedom:
#   df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)² / (n₁-1) + (s₂²/n₂)² / (n₂-1)]

print("\n" + "=" * 70)
print("TASK 6: Confidence Interval for Treatment Effect")
print("=" * 70)

# ---- CI for simulated data ----
print("\n--- Simulated Data ---")
sim_se_diff = np.sqrt(
    sim_control.var(ddof=1) / len(sim_control)
    + sim_treatment.var(ddof=1) / len(sim_treatment)
)

# Welch-Satterthwaite degrees of freedom
sim_s1_sq_n1 = sim_control.var(ddof=1) / len(sim_control)
sim_s2_sq_n2 = sim_treatment.var(ddof=1) / len(sim_treatment)
sim_df_ws = (sim_s1_sq_n1 + sim_s2_sq_n2) ** 2 / (
    sim_s1_sq_n1 ** 2 / (len(sim_control) - 1)
    + sim_s2_sq_n2 ** 2 / (len(sim_treatment) - 1)
)

sim_t_crit = stats.t.ppf(1 - alpha / 2, df=sim_df_ws)
sim_diff = sim_treatment.mean() - sim_control.mean()
sim_ci_lower = sim_diff - sim_t_crit * sim_se_diff
sim_ci_upper = sim_diff + sim_t_crit * sim_se_diff

print(f"Observed difference: {sim_diff:.4f}")
print(f"Standard error: {sim_se_diff:.4f}")
print(f"Degrees of freedom (Welch-Satterthwaite): {sim_df_ws:.1f}")
print(f"95% CI: [{sim_ci_lower:.4f}, {sim_ci_upper:.4f}]")
print(f"True effect (δ={true_effect}): {'WITHIN CI' if sim_ci_lower <= true_effect <= sim_ci_upper else 'OUTSIDE CI'}")
# INTERPRETATION: For a 95% CI, we expect 95% of such intervals constructed
# from repeated experiments to contain the true effect. The CI here should
# contain the known true effect of 2.0 in approximately 95% of simulations.

# ---- CI for real data ----
print("\n--- Real Experiment Data ---")
real_se_diff = np.sqrt(
    ctrl_values.var(ddof=1) / n_control
    + treat_values.var(ddof=1) / n_treatment
)

# Welch-Satterthwaite degrees of freedom
s1_sq_n1 = ctrl_values.var(ddof=1) / n_control
s2_sq_n2 = treat_values.var(ddof=1) / n_treatment
df_ws = (s1_sq_n1 + s2_sq_n2) ** 2 / (
    s1_sq_n1 ** 2 / (n_control - 1) + s2_sq_n2 ** 2 / (n_treatment - 1)
)

t_crit = stats.t.ppf(1 - alpha / 2, df=df_ws)
ci_lower = observed_diff - t_crit * real_se_diff
ci_upper = observed_diff + t_crit * real_se_diff

print(f"Observed difference: {observed_diff:+.4f}")
print(f"Standard error: {real_se_diff:.4f}")
print(f"Degrees of freedom (Welch-Satterthwaite): {df_ws:.1f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower > 0:
    print("CI entirely above zero — treatment has a POSITIVE effect")
elif ci_upper < 0:
    print("CI entirely below zero — treatment has a NEGATIVE effect")
else:
    print("CI spans zero — effect is not statistically distinguishable from zero")

# Practical significance: is the CI within the MDE range?
print(f"\nMDE (from power analysis): {mde_absolute:.4f}")
if abs(observed_diff) >= mde_absolute:
    print("Observed effect exceeds MDE — practically significant")
else:
    print("Observed effect is below MDE — may not be practically meaningful")

# ---- Visualise all intervals ----
viz = ModelVisualizer()

interval_data = {
    "Simulated (true=2.0)": {
        "lower_bound": sim_ci_lower,
        "upper_bound": sim_ci_upper,
        "point_estimate": sim_diff,
        "width": sim_ci_upper - sim_ci_lower,
    },
    "Real Data": {
        "lower_bound": ci_lower,
        "upper_bound": ci_upper,
        "point_estimate": observed_diff,
        "width": ci_upper - ci_lower,
    },
}
fig_intervals = viz.metric_comparison(interval_data)
fig_intervals.update_layout(title="95% Confidence Intervals for Treatment Effect")
fig_intervals.write_html("ex4_confidence_intervals.html")
print("\nSaved: ex4_confidence_intervals.html")

# Power curve visualisation
power_metrics = {"Power": power_at_n}
fig_power = viz.training_history(power_metrics, x_label="Sample Size per Group")
fig_power.update_layout(title=f"Power Curve (MDE = {mde_absolute:.2f}, α = {alpha})")
fig_power.write_html("ex4_power_curve.html")
print("Saved: ex4_power_curve.html")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert sim_ci_lower < sim_ci_upper, "CI lower must be below upper"
assert ci_lower < ci_upper, "Real data CI lower must be below upper"
if sim_ci_lower <= true_effect <= sim_ci_upper:
    print(f"  True effect {true_effect} is within simulated 95% CI [{sim_ci_lower:.4f}, {sim_ci_upper:.4f}] ✓")
else:
    print(f"  Note: True effect {true_effect} fell outside simulated CI [{sim_ci_lower:.4f}, {sim_ci_upper:.4f}]")
    print(f"  This happens ~5% of the time with a 95% CI — that's the definition of 95% coverage!")
print("\n✓ Checkpoint 5 passed — confidence intervals validated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Data Collection Plan (Why / What / Where / How / Frequency)
# ══════════════════════════════════════════════════════════════════════
# THEORY: A data collection framework ensures you collect the RIGHT data
# from the RIGHT sources at the RIGHT frequency BEFORE running the experiment.

print("\n" + "=" * 70)
print("TASK 7: Data Collection Plan")
print("=" * 70)

# Build the plan as structured data for display
plan = {
    "WHY (Hypotheses & Value)": {
        "Primary hypothesis": (
            "New recommendation algorithm increases user engagement "
            "(metric_value) by >= 2% relative lift"
        ),
        "Secondary hypothesis": (
            "Revenue per user increases without degrading "
            "user experience (session duration, bounce rate)"
        ),
        "Business value": (
            "A 2% lift in engagement correlates with ~$X00K annual "
            "revenue uplift based on historical engagement-to-revenue models"
        ),
        "Performance measures": (
            "metric_value (primary), revenue (secondary), "
            "session_duration and bounce_rate (guardrail)"
        ),
    },
    "WHAT (Data Requirements)": {
        "Ideal wish list": (
            "Per-user per-session: metric_value, revenue, pages_viewed, "
            "session_duration, bounce_flag, device, country, user_segment, "
            "pre_experiment_metric (for CUPED), timestamp"
        ),
        "Minimum viable": (
            "Per-user: metric_value, experiment_group, timestamp"
        ),
        "Budget constraint": (
            f"Need >= {n_required_total:,} total observations "
            f"({n_required_per_group:,} per group) for 80% power"
        ),
        "Time constraint": (
            f"At current traffic of ~{n_total:,} users per experiment window, "
            f"need ~{max(1, math.ceil(n_required_total / n_total))} experiment "
            f"cycle(s) to reach required sample size"
        ),
    },
    "WHERE (Data Sources)": {
        "Internal — Application DB": (
            "User profiles, segment assignments, platform (mobile/desktop/tablet)"
        ),
        "Internal — Event stream": (
            "Clickstream events, page views, session starts/ends, "
            "recommendation impressions and clicks"
        ),
        "Internal — Transaction DB": (
            "Revenue, order value, purchase timestamps"
        ),
        "Internal — Experiment platform": (
            "Group assignment, assignment timestamp, feature flags"
        ),
        "External — data.gov.sg": (
            "Public holidays, economic indicators (control for macro effects)"
        ),
    },
    "HOW (Collection Methods)": {
        "At start — review": (
            "Audit existing data pipelines for completeness and freshness. "
            "Check for known data quality issues (duplicates, missing values)."
        ),
        "At start — discover": (
            "Profile pre-experiment data to estimate baseline metric "
            "distribution (mean, variance) for power analysis."
        ),
        "At start — validate": (
            "Run A/A test (both groups see control) for 1 week to validate "
            "the randomisation mechanism before launching the A/B test."
        ),
        "Continuous — automated": (
            "Real-time event ingestion via streaming pipeline. "
            "Daily SRM checks as automated alerts. "
            "Pre-computed CUPED covariates from 30-day pre-period."
        ),
        "Continuous — human": (
            "Weekly review of data quality dashboards. "
            "Escalation protocol for SRM alerts. "
            "Mid-experiment review at 50% of target sample size."
        ),
    },
    "FREQUENCY (Collection & Analysis Cadence)": {
        "Collection frequency": "Per-event (real-time streaming)",
        "Analysis frequency": (
            "Daily monitoring dashboard (sample size, SRM, metric trends). "
            "DO NOT run significance tests daily — peeking inflates Type I error."
        ),
        "Duration": (
            f"Run until {n_required_total:,} observations collected OR "
            f"maximum 4 weeks (whichever comes first). "
            f"Include >= 1 full business cycle (7 days) to average out "
            f"day-of-week effects."
        ),
        "Stopping rule": (
            "Pre-registered: analyse ONCE at target sample size. "
            "If sequential monitoring is needed, use always-valid p-values "
            "(see Exercise 7 — CUPED and variance reduction)."
        ),
    },
}

# Display the plan in formatted output
for section, items in plan.items():
    print(f"\n{'─' * 70}")
    print(f"  {section}")
    print(f"{'─' * 70}")
    for key, value in items.items():
        # Wrap long lines for readability
        print(f"\n  {key}:")
        # Split into lines of ~60 chars
        words = value.split()
        line = "    "
        for word in words:
            if len(line) + len(word) + 1 > 72:
                print(line)
                line = "    " + word
            else:
                line = line + " " + word if line.strip() else "    " + word
        if line.strip():
            print(line)


# ══════════════════════════════════════════════════════════════════════
# Common Pitfalls Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Common A/B Testing Pitfalls")
print("=" * 70)

pitfalls = [
    (
        "Overlapping treatments",
        "User in both experiment A and experiment B simultaneously. "
        "Interaction effects make individual treatment effects uninterpretable. "
        "Mitigation: mutual exclusion layers in the experiment platform.",
    ),
    (
        "Prior treatment influence",
        "User saw treatment in a previous experiment, creating carryover effects. "
        "Mitigation: enforce washout periods between related experiments.",
    ),
    (
        "Temporal biases",
        "Starting the experiment on a Monday and ending on a Wednesday captures "
        "weekday behaviour but misses weekends. "
        "Mitigation: run for full weeks (multiples of 7 days).",
    ),
    (
        "Peeking (repeated significance testing)",
        "Checking p-values daily and stopping when significant inflates Type I "
        "error far above the nominal alpha. With 10 peeks at alpha=0.05, the "
        "effective false positive rate rises to ~14%. "
        "Mitigation: pre-register a fixed sample size, or use sequential testing.",
    ),
    (
        "Simpson's paradox in segments",
        "Overall effect looks positive, but within every segment the effect is "
        "negative (or vice versa). Caused by unbalanced segment sizes. "
        "Mitigation: check treatment effects within each segment.",
    ),
    (
        "SRM ignored",
        "Proceeding with analysis despite a detected SRM. A broken randomisation "
        "invalidates all causal claims regardless of how significant the p-value is. "
        "Mitigation: always check SRM first; halt analysis if detected.",
    ),
]

for i, (name, description) in enumerate(pitfalls, 1):
    print(f"\n  {i}. {name}")
    words = description.split()
    line = "     "
    for word in words:
        if len(line) + len(word) + 1 > 72:
            print(line)
            line = "     " + word
        else:
            line = line + " " + word if line.strip() else "     " + word
    if line.strip():
        print(line)


# ══════════════════════════════════════════════════════════════════════
# Final Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"""
Experiment Design:
  - Two-arm A/B test (control vs treatment_a)
  - H₀: μ_treatment = μ_control
  - α = {alpha}, Power = {power_target:.0%}

Power Analysis:
  - MDE: {mde_absolute:.4f} ({relative_mde_pct}% relative lift)
  - Required n: {n_required_total:,} total ({n_required_per_group:,} per group)
  - Actual n: {n_total:,} ({'adequate' if n_total >= n_required_total else 'underpowered'})

SRM Check:
  - Simulated (50/50): χ² = {sim_chi2:.4f}, p = {sim_srm_p:.4f} (no SRM)
  - Real data: χ² = {chi2_stat:.4f}, p = {srm_p_value:.2e} ({'SRM detected' if srm_p_value < 0.01 else 'no SRM'})

Hypothesis Test (Welch's t-test):
  - Simulated: t = {sim_t_stat:.4f}, p = {sim_p_value:.2e} ({'sig' if sim_p_value < alpha else 'ns'})
  - Real data: t = {real_t_stat:.4f}, p = {real_p_value:.6f} ({'sig' if real_p_value < alpha else 'ns'})
  - Real effect: {observed_diff:+.4f} ({relative_lift:+.2f}%), Cohen's d = {cohens_d_real:.4f}

Confidence Interval (real data):
  - 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]
""")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print("""
  ✓ Pre-registration: hypotheses and stopping rules defined before data
  ✓ Power analysis: n = (z_{α/2} + z_β)² × 2σ² / δ² for t-test
  ✓ Cohen's d: standardised effect size (small=0.2, medium=0.5, large=0.8)
  ✓ Positive control: simulate known effect to validate the pipeline
  ✓ SRM: χ² detects broken randomisation; simulated 50/50 should give χ²=0
  ✓ Welch's t-test: does not assume equal variances (preferred over Student's)
  ✓ Welch-Satterthwaite df: accounts for unequal variances in the CI
  ✓ Data collection plan: Why/What/Where/How/Frequency framework

  NEXT: In Exercise 5 you'll build OLS linear regression from scratch
  using matrix algebra (β = (X'X)⁻¹X'y), interpret coefficients with
  ceteris paribus logic, compute t-statistics and R², add polynomial
  and interaction terms, and cross-validate with a train/test split.
""")

print("Exercise 4 complete — A/B testing and experiment design")
