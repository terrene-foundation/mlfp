# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 3.1: Bootstrap CIs & Power Analysis
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement bootstrap resampling from scratch (with replacement)
#   - Compute percentile, normal, and BCa confidence intervals
#   - Understand why BCa is the gold standard for bootstrap CIs
#   - Calculate minimum detectable effect (MDE) for a given sample size
#   - Generate power curves — the trade-off between n, effect size, and power
#
# PREREQUISITES: Exercise 2 (MLE, confidence intervals)
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Load A/B test data + SRM check against the designed allocation
#   2. Bootstrap resampling from scratch — 10K resamples
#   3. Three CI methods: percentile, normal, BCa
#   4. Power analysis — minimum detectable effect
#   5. Power curves — effect size and sample size
#   6. Visualise bootstrap distribution + power curves
#
# THEORY:
#   Bootstrap: resample WITH REPLACEMENT, compute statistic, repeat.
#   The bootstrap distribution approximates the sampling distribution.
#     - Percentile CI: [q_{alpha/2}, q_{1-alpha/2}] of boot distribution
#     - Normal CI: x_bar +/- z * SE_boot (assumes symmetry)
#     - BCa: bias-corrected and accelerated (gold standard)
#   MDE = (z_{alpha/2} + z_beta) * SE  (smallest reliably detectable effect)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy import stats

from shared.mlfp02.ex_3 import (
    ALPHA,
    POWER_TARGET,
    N_BOOTSTRAP,
    RANDOM_SEED,
    OUTPUT_DIR,
    DESIGNED_ALLOCATION,
    TREATMENT_ARM,
    designed_control_share,
    load_experiment,
    load_experiment_all,
    split_groups,
    conversion_arrays,
    srm_check,
    srm_check_multi,
    print_header,
)
from shared.mlfp002 import create_visualizer

print_header("MLFP02 Exercise 3.1: Bootstrap CIs & Power Analysis")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data + SRM sanity check
# ════════════════════════════════════════════════════════════════════════
# Before ANY analysis, verify that users landed in each arm in the
# proportions the experiment was DESIGNED for. SRM (Sample Ratio Mismatch)
# detects randomisation bugs, bot traffic, or pipeline issues. This
# experiment's design is UNEQUAL (40/35/15/10 across four arms), so the
# test must use the designed shares — testing against 50/50 would flag
# "SRM" in every unequal design and teach you to ignore the alarm.

df_all = load_experiment_all()
arm_counts = dict(df_all.group_by("experiment_group").len().iter_rows())
srm_all = srm_check_multi(arm_counts, DESIGNED_ALLOCATION)

print(f"\nAll-arm SRM check vs design: chi2={srm_all['chi2']:.1f}, p={srm_all['p_value']:.2e}")
print(f"  {'Arm':<12} {'Observed':>9} {'Designed':>9} {'Std resid':>10}")
for arm, r in srm_all["per_arm"].items():
    print(
        f"  {arm:<12} {r['observed_share']:>9.1%} {r['designed_share']:>9.1%} "
        f"{r['std_residual']:>+10.1f}"
    )
worst_arm = max(
    srm_all["per_arm"], key=lambda a: abs(srm_all["per_arm"][a]["std_residual"])
)
print(f"  Most mis-allocated arm: {worst_arm}")
# INTERPRETATION: The all-arm test fires, and the per-arm residuals show
# WHY: one arm received far more traffic than designed (the others look
# slightly low only because shares must sum to 100%). Whatever happened to
# that arm's assignment, its users are not a clean random sample, so its
# results are excluded. SRM detection is only useful if you act on it.

# Analyse ONE treatment arm against control — pooling different
# treatments would estimate a meaningless mixture of effects.
df = load_experiment(TREATMENT_ARM)
control, treatment = split_groups(df)
n_control = control.height
n_treatment = treatment.height
n_total = df.height

print(f"\nAnalysed pair: control vs {TREATMENT_ARM} — {n_total:,} users")
print(f"  Control:   {n_control:,}")
print(f"  Treatment: {n_treatment:,}")

srm = srm_check(n_control, n_treatment, designed_control_share(TREATMENT_ARM))
print(
    f"Pair SRM check (designed control share "
    f"{designed_control_share(TREATMENT_ARM):.3f}): "
    f"chi2={srm['chi2']:.4f}, p={srm['p_value']:.4f}"
)
print(f"  Verdict: {srm['verdict']}")
if srm["srm"]:
    raise RuntimeError(
        "SRM on the analysed pair — stop here: downstream results would be "
        "untrustworthy. Investigate the assignment pipeline first."
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert srm_all["srm"], "The all-arm test should flag the mis-allocated arm"
assert not srm["srm"], "The analysed pair must pass its SRM check"
assert n_control + n_treatment == n_total, "Groups must sum to total"
print("\n>>> Checkpoint 1 passed -- SRM checked against the design\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Bootstrap?
# ════════════════════════════════════════════════════════════════════════
# Classical CIs assume the sampling distribution is Normal (CLT). This
# works for means with large n, but fails for:
#   - Medians, ratios, percentiles (no closed-form SE)
#   - Small samples where CLT hasn't kicked in
#   - Skewed distributions (revenue, time-on-site)
#
# Bootstrap solves this by SIMULATING the sampling distribution:
#   1. Draw n samples WITH REPLACEMENT from your data
#   2. Compute the statistic on the resample
#   3. Repeat 10,000 times
#   4. The distribution of resampled statistics approximates
#      the true sampling distribution
#
# Analogy: You have one bag of 1,000 marbles (your data). You can't
# get more bags from the factory. But you CAN repeatedly grab handfuls
# WITH REPLACEMENT (putting each marble back), record the colour mix,
# and build up a picture of how variable each handful is.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Bootstrap resampling from scratch
# ════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(seed=RANDOM_SEED)
ctrl_conv, treat_conv = conversion_arrays(df)

boot_diffs = np.zeros(N_BOOTSTRAP)
boot_ctrl_rates = np.zeros(N_BOOTSTRAP)
boot_treat_rates = np.zeros(N_BOOTSTRAP)

for i in range(N_BOOTSTRAP):
    boot_ctrl = rng.choice(ctrl_conv, size=n_control, replace=True)
    boot_treat = rng.choice(treat_conv, size=n_treatment, replace=True)
    boot_ctrl_rates[i] = boot_ctrl.mean()
    boot_treat_rates[i] = boot_treat.mean()
    boot_diffs[i] = boot_treat.mean() - boot_ctrl.mean()

observed_diff = treat_conv.mean() - ctrl_conv.mean()
boot_se = boot_diffs.std()

print(f"Observed conversion diff: {observed_diff:+.6f}")
print(f"Bootstrap SE: {boot_se:.6f}")
print(f"Resamples: {N_BOOTSTRAP:,}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Three CI methods: percentile, normal, BCa
# ════════════════════════════════════════════════════════════════════════

# Method 1: Percentile CI — simplest
pctile_ci = np.percentile(boot_diffs, [2.5, 97.5])

# Method 2: Normal bootstrap CI — assumes symmetric distribution
normal_boot_ci = (observed_diff - 1.96 * boot_se, observed_diff + 1.96 * boot_se)

# Method 3: BCa — bias-corrected and accelerated (gold standard)
bca_result = stats.bootstrap(
    (treat_conv, ctrl_conv),
    statistic=lambda t, c: t.mean() - c.mean(),
    n_resamples=N_BOOTSTRAP,
    confidence_level=0.95,
    method="BCa",
    random_state=RANDOM_SEED,
    # Process resamples in chunks. Without this, SciPy builds one
    # (n_resamples × n_samples) array — with 500K users that is tens of GB
    # and the process is OOM-killed. batch caps peak memory; results are identical.
    batch=100,
)
bca_ci = (bca_result.confidence_interval.low, bca_result.confidence_interval.high)

print(f"\n=== Bootstrap CIs for Conversion Rate Difference ===")
print(f"{'Method':<25} {'Lower':>12} {'Upper':>12} {'Width':>12}")
print("-" * 65)
print(
    f"{'Percentile CI':<25} {pctile_ci[0]:>12.6f} {pctile_ci[1]:>12.6f} "
    f"{pctile_ci[1]-pctile_ci[0]:>12.6f}"
)
print(
    f"{'Normal Boot CI':<25} {normal_boot_ci[0]:>12.6f} {normal_boot_ci[1]:>12.6f} "
    f"{normal_boot_ci[1]-normal_boot_ci[0]:>12.6f}"
)
print(
    f"{'BCa CI':<25} {bca_ci[0]:>12.6f} {bca_ci[1]:>12.6f} "
    f"{bca_ci[1]-bca_ci[0]:>12.6f}"
)

# INTERPRETATION: BCa is the gold standard because it corrects for both
# bias (the bootstrap distribution may not be centred at the observed
# statistic) and acceleration (the SE may vary with the parameter value).
# For symmetric, well-behaved statistics like the mean, all three agree.
# For medians, ratios, or skewed data, BCa gives narrower and more
# accurate intervals.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(boot_diffs) == N_BOOTSTRAP, "Should have N_BOOTSTRAP resamples"
assert pctile_ci[0] < pctile_ci[1], "CI lower must be below upper"
assert boot_se > 0, "Bootstrap SE must be positive"
print("\n>>> Checkpoint 2 passed -- bootstrap CIs computed from scratch\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Power analysis: minimum detectable effect
# ════════════════════════════════════════════════════════════════════════
# Given our sample size, what's the smallest effect we can reliably detect
# at alpha=0.05 with 80% power?
#
# MDE = (z_{alpha/2} + z_beta) * sqrt(p(1-p)(1/n1 + 1/n2))
#
# This tells you the experiment's "resolution" — effects smaller than
# the MDE are invisible to this experiment, like trying to weigh a
# feather on a bathroom scale.

p_control = ctrl_conv.mean()
z_alpha_half = stats.norm.ppf(1 - ALPHA / 2)
z_beta = stats.norm.ppf(POWER_TARGET)

pooled_se = np.sqrt(p_control * (1 - p_control) * (1 / n_control + 1 / n_treatment))
mde = (z_alpha_half + z_beta) * pooled_se

print(f"=== Power Analysis ===")
print(f"Baseline conversion rate: {p_control:.4f} ({p_control:.2%})")
print(f"alpha = {ALPHA}, Power = {POWER_TARGET:.0%}")
print(f"z_{{alpha/2}} = {z_alpha_half:.3f}, z_beta = {z_beta:.3f}")
print(f"Minimum Detectable Effect (MDE): {mde:.6f} ({mde:.4%} absolute)")
print(f"Relative MDE: {mde / p_control:.2%} of baseline")

# INTERPRETATION: The printed MDE means this experiment can reliably
# (80% power) detect a treatment that changes conversion by at least that
# many percentage points. Smaller effects may exist but are invisible to
# this experiment at 80% power. To detect smaller effects: get more data.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 0 < mde < 1, "MDE must be a valid proportion"
assert 0 < p_control < 1, "Baseline must be a valid proportion"
print("\n>>> Checkpoint 3 passed -- MDE computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Power curves
# ════════════════════════════════════════════════════════════════════════

# Power vs effect size (fixed n)
effect_sizes = np.linspace(0, mde * 3, 100)
powers_by_effect = []
for delta in effect_sizes:
    ncp = delta / pooled_se
    power_val = (
        1 - stats.norm.cdf(z_alpha_half - ncp) + stats.norm.cdf(-z_alpha_half - ncp)
    )
    powers_by_effect.append(power_val)

print(f"=== Power Curves ===")
print(f"\n--- Power by Effect Size (n={n_total:,}) ---")
for mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
    idx = min(int(mult / 3 * 99), 99)
    print(f"  Effect = {effect_sizes[idx]:.4%}: Power = {powers_by_effect[idx]:.1%}")

# Power vs sample size (fixed effect = MDE)
sample_sizes_power = np.arange(500, n_total * 2, 500)
powers_by_n = []
for n_per in sample_sizes_power:
    se_n = np.sqrt(p_control * (1 - p_control) * 2 / n_per)
    ncp = mde / se_n
    power_val = (
        1 - stats.norm.cdf(z_alpha_half - ncp) + stats.norm.cdf(-z_alpha_half - ncp)
    )
    powers_by_n.append(power_val)

print(f"\n--- Power by Sample Size (effect = MDE = {mde:.4%}) ---")
for frac in [0.25, 0.5, 1.0, 1.5, 2.0]:
    idx = min(int(frac * n_total / 500) - 1, len(sample_sizes_power) - 1)
    idx = max(0, idx)
    print(f"  n = {sample_sizes_power[idx]:>8,}: Power = {powers_by_n[idx]:.1%}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert (
    powers_by_effect[-1] > powers_by_effect[0]
), "Power should increase with effect size"
assert len(powers_by_effect) == len(effect_sizes), "One power per effect size"
print("\n>>> Checkpoint 4 passed -- power curves computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise: bootstrap distribution + power curves
# ════════════════════════════════════════════════════════════════════════

from kailash_ml import ModelVisualizer

viz = create_visualizer()

# Plot 1: Bootstrap distribution of conversion rate difference
fig1 = viz.histogram(
    pl.DataFrame({"Treatment - Control": boot_diffs}),
    "Treatment - Control",
    title="Bootstrap Distribution: Conversion Rate Difference",
)
fig1.add_vline(x=observed_diff, line_dash="dash", annotation_text="Observed")
fig1.add_vline(x=0, line_dash="dot", line_color="red", annotation_text="H0: no effect")
out_1 = OUTPUT_DIR / "bootstrap_distribution.html"
fig1.write_html(str(out_1))
print(f"Saved: {out_1}")

# Plot 2: Power vs effect size
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=effect_sizes, y=powers_by_effect, name="Power"))
fig2.add_hline(y=0.8, line_dash="dash", annotation_text="80% power target")
fig2.add_vline(x=mde, line_dash="dot", annotation_text=f"MDE={mde:.4f}")
fig2.update_layout(
    title="Statistical Power vs Effect Size",
    xaxis_title="Effect Size (absolute)",
    yaxis_title="Power",
)
out_2 = OUTPUT_DIR / "power_curve.html"
fig2.write_html(str(out_2))
print(f"Saved: {out_2}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore e-commerce experiment planning
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A regional e-commerce marketplace (illustrative, ~500K daily
# users) is planning its NEXT recommendation test and asks: "How many
# users, and how many days?" Answer it by solving the power formula for
# n at the lift the business cares about. (The MDE above already
# describes the CURRENT experiment's n — it is not a planning target.)
#
# BUSINESS IMPACT: Without power analysis, teams either:
#   - Stop experiments too early (underpowered) -> miss real effects
#   - Run experiments too long (overpowered) -> waste traffic + delay launches

print(f"\n--- Business Application: Planning the Next Experiment ---")
daily_users = 500_000
print(
    f"Current experiment: {n_total:,} users -> MDE = {mde:.3%} absolute on a "
    f"{p_control:.1%} baseline"
)
plan_rows = []
for target_lift in [0.01, 0.005, 0.0025]:
    n_per_group = int(
        np.ceil(
            (z_alpha_half + z_beta) ** 2
            * 2
            * p_control
            * (1 - p_control)
            / target_lift**2
        )
    )
    days = int(np.ceil(2 * n_per_group / daily_users))
    plan_rows.append((target_lift, n_per_group, days))
    print(
        f"  Detect a {target_lift:.2%} lift -> {n_per_group:,} per group "
        f"({2 * n_per_group:,} total) -> {days} day(s) of traffic"
    )
print("Halving the target lift roughly quadruples the sample (n ∝ 1/lift²).")
print("Even when one day of traffic suffices, run whole weeks so weekday and")
print("weekend behaviour are both represented.")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert plan_rows[1][1] > 3.5 * plan_rows[0][1], "Halving the lift should ~4x n"
print("\n>>> Checkpoint 5 passed -- experiment plan computed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] SRM: chi-squared test against the DESIGNED allocation; act on it
      (exclude the mis-allocated arm, stop if the analysed pair fails)
  [x] Bootstrap: resample with replacement, compute any statistic's CI
  [x] Three CI methods: percentile (simple), normal (symmetric), BCa (gold standard)
  [x] MDE: smallest detectable effect at given n, alpha, and power
  [x] Planning: solve the power formula for n at a target lift
  [x] Power curves: visualise trade-off between n, effect size, and power

  NEXT: In 02_hypothesis_testing.py you'll use these power calculations
  to run the actual hypothesis test and compute effect sizes.
"""
)

print(">>> Exercise 3.1 complete -- Bootstrap CIs & Power Analysis")
