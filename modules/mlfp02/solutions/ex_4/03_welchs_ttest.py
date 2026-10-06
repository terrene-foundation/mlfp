# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 4.3: Welch's t-Test & Confidence Intervals
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply Welch's t-test (robust to unequal variances)
#   - Compute Welch-Satterthwaite degrees of freedom manually
#   - Build 95% confidence intervals: Welch, Normal, and Bootstrap
#   - Interpret CIs as "how big" not just "is there an effect"
#   - Connect CI interpretation to Singapore pricing decisions
#
# PREREQUISITES:
#   - MLFP02 Exercise 4.2 (SRM detection, positive controls)
#   - MLFP02 Exercise 3 (t-tests, p-values)
#
# ESTIMATED TIME: ~40 minutes
#
# TASKS (5-phase R10):
#   1. Theory — why Welch beats Student's t when variances differ
#   2. Build — Welch's t-test on simulated and real data
#   3. Train — three CI methods: Welch, Normal approximation, Bootstrap
#   4. Visualise — CI comparison chart with zero-reference line
#   5. Apply — credit-card cashback A/B: CI plus cost-benefit at both ends
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from shared.mlfp02.ex_4 import (
    ALPHA,
    OUTPUT_DIR,
    SEED,
    TwoArmAB,
    interpret_cohens_d,
    load_experiment,
    make_rng,
    print_banner,
    summarise_arm,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Welch vs Student's t-Test
# ════════════════════════════════════════════════════════════════════════
# Student's t-test assumes equal variances in both groups.  When that
# assumption is wrong (it usually is in A/B tests), the p-value is
# unreliable — either too liberal or too conservative.
#
# Welch's t-test relaxes the equal-variance assumption by using:
#   1. Separate variance estimates for each group
#   2. Satterthwaite degrees of freedom: a weighted average of the
#      two group df's that accounts for unequal variances
#
# Result: Welch's test is ALWAYS safe to use.  It matches Student's
# when variances ARE equal, and corrects when they aren't.
# There is NO reason to ever use Student's t-test in practice.
#
# The Welch-Satterthwaite df formula:
#
#   df = (s1^2/n1 + s2^2/n2)^2 /
#        (s1^4/(n1^2*(n1-1)) + s2^4/(n2^2*(n2-1)))

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — LOAD data
# ════════════════════════════════════════════════════════════════════════

print_banner("Exercise 4.3 — Welch's t-Test & Confidence Intervals")

data: TwoArmAB = load_experiment()
rng = make_rng(SEED)

summarise_arm("Control", data.ctrl_values)
summarise_arm("Treatment", data.treat_values)

sigma_pooled = data.ctrl_values.std(ddof=1)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Welch's t-Test
# ════════════════════════════════════════════════════════════════════════

print_banner("Welch's t-Test")

# --- On simulated data (known effect = 2.0) ---
print("\n--- Simulated Data (true effect=2.0) ---")
sim_n_per = 10_000
sim_control = rng.normal(
    loc=data.ctrl_values.mean(), scale=sigma_pooled, size=sim_n_per
)
sim_treatment = rng.normal(
    loc=data.ctrl_values.mean() + 2.0, scale=sigma_pooled, size=sim_n_per
)

sim_t_stat, sim_p_val = stats.ttest_ind(sim_treatment, sim_control, equal_var=False)
print(f"t={sim_t_stat:.4f}, p={sim_p_val:.2e}")
print(f"{'SIGNIFICANT' if sim_p_val < ALPHA else 'NOT sig'} at alpha={ALPHA}")

# --- On real data ---
print("\n--- Real Data ---")
real_t_stat, real_p_val = stats.ttest_ind(
    data.treat_values, data.ctrl_values, equal_var=False
)
obs_diff = data.treat_values.mean() - data.ctrl_values.mean()
rel_lift = obs_diff / data.ctrl_values.mean() * 100

# Welch-Satterthwaite degrees of freedom (manual computation)
s1_sq_n1 = data.ctrl_values.var(ddof=1) / data.n_control
s2_sq_n2 = data.treat_values.var(ddof=1) / data.n_treatment
df_ws = (s1_sq_n1 + s2_sq_n2) ** 2 / (
    s1_sq_n1**2 / (data.n_control - 1) + s2_sq_n2**2 / (data.n_treatment - 1)
)

print(f"Control:   {data.ctrl_values.mean():.4f} +/- {data.ctrl_values.std():.4f}")
print(f"Treatment: {data.treat_values.mean():.4f} +/- {data.treat_values.std():.4f}")
print(f"Diff: {obs_diff:+.4f} ({rel_lift:+.2f}% relative)")
print(f"t-statistic: {real_t_stat:.4f}")
print(f"Welch-Satterthwaite df: {df_ws:.1f}")
print(f"p-value: {real_p_val:.6f}")
print(f"{'SIGNIFICANT' if real_p_val < ALPHA else 'NOT significant'} at alpha={ALPHA}")

# Cohen's d
pooled_std_real = np.sqrt(
    (data.ctrl_values.var(ddof=1) + data.treat_values.var(ddof=1)) / 2
)
cohens_d_real = obs_diff / pooled_std_real
print(
    f"Cohen's d: {cohens_d_real:.4f} "
    f"({interpret_cohens_d(abs(cohens_d_real))})"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert sim_p_val < ALPHA, "Positive control must be detected"
assert 0 <= real_p_val <= 1, "Real p-value must be valid"
print("\n>>> Checkpoint 1 passed — Welch's t-test completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Three CI Methods
# ════════════════════════════════════════════════════════════════════════
# Confidence intervals are MORE informative than p-values because they
# tell you HOW BIG the effect is, not just whether it exists.
#
#   CI = [0.01, 0.03] -> effect is real but tiny
#   CI = [-2, 5]      -> we have no idea about the sign
#   CI = [1.5, 3.2]   -> effect is real and meaningful

print_banner("Confidence Intervals for Treatment Effect")

# Method 1: Welch CI (uses t-distribution with Satterthwaite df)
real_se = np.sqrt(s1_sq_n1 + s2_sq_n2)
t_crit = stats.t.ppf(1 - ALPHA / 2, df=df_ws)
welch_ci = (obs_diff - t_crit * real_se, obs_diff + t_crit * real_se)

# Method 2: Bootstrap CI (non-parametric)
n_boot = 10_000
boot_diffs = np.array(
    [
        rng.choice(data.treat_values, size=data.n_treatment, replace=True).mean()
        - rng.choice(data.ctrl_values, size=data.n_control, replace=True).mean()
        for _ in range(n_boot)
    ]
)
boot_ci = tuple(np.percentile(boot_diffs, [2.5, 97.5]))

# Method 3: Normal approximation CI (assumes large n)
normal_ci = (obs_diff - 1.96 * real_se, obs_diff + 1.96 * real_se)

print(f"Observed difference: {obs_diff:+.4f}")
print(f"\n{'Method':<20} {'Lower':>12} {'Upper':>12} {'Width':>10}")
print("-" * 56)
print(
    f"{'Welch t-CI':<20} {welch_ci[0]:>12.4f} {welch_ci[1]:>12.4f} "
    f"{welch_ci[1] - welch_ci[0]:>10.4f}"
)
print(
    f"{'Normal CI':<20} {normal_ci[0]:>12.4f} {normal_ci[1]:>12.4f} "
    f"{normal_ci[1] - normal_ci[0]:>10.4f}"
)
print(
    f"{'Bootstrap CI':<20} {boot_ci[0]:>12.4f} {boot_ci[1]:>12.4f} "
    f"{boot_ci[1] - boot_ci[0]:>10.4f}"
)

if welch_ci[0] > 0:
    print("\nCI entirely above zero — POSITIVE treatment effect")
elif welch_ci[1] < 0:
    print("\nCI entirely below zero — NEGATIVE treatment effect")
else:
    print("\nCI spans zero — effect not distinguishable from zero")
# INTERPRETATION: The CI tells you HOW BIG the effect likely is —
# far more informative than a binary p-value.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert welch_ci[0] < welch_ci[1], "CI lower must be below upper"
assert boot_ci[0] < boot_ci[1], "Bootstrap CI must be valid"
print("\n>>> Checkpoint 2 passed — confidence intervals computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: CI Comparison
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
methods = ["Welch t-CI", "Normal CI", "Bootstrap CI"]
lowers = [welch_ci[0], normal_ci[0], boot_ci[0]]
uppers = [welch_ci[1], normal_ci[1], boot_ci[1]]
colors = ["#2196F3", "#FF9800", "#4CAF50"]

for i, (method, lo, hi) in enumerate(zip(methods, lowers, uppers)):
    fig.add_trace(
        go.Scatter(
            x=[lo, obs_diff, hi],
            y=[method] * 3,
            mode="markers+lines",
            name=method,
            marker={"size": [8, 12, 8], "color": colors[i]},
            line={"color": colors[i]},
        )
    )
fig.add_vline(x=0, line_dash="dot", line_color="red", annotation_text="Zero effect")
fig.update_layout(
    title="95% Confidence Intervals for Treatment Effect",
    xaxis_title="Treatment Effect (engagement score)",
    template="plotly_white",
    height=350,
)
out_path = OUTPUT_DIR / "confidence_intervals.html"
fig.write_html(str(out_path))
print(f"Saved: {out_path}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert out_path.exists(), "CI plot must be saved"
print("\n>>> Checkpoint 3 passed — CI visualisation saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Credit-Card Cashback A/B (illustrative)
# ════════════════════════════════════════════════════════════════════════
# A Singapore bank (illustrative numbers) tests a richer cashback tier
# (2.0% vs 1.5%) on monthly card spend. The question is not just "does
# 2% increase spend?" but "by how much — and does it pay?" A CI answers
# the first part; the cost-benefit must then be checked at BOTH ends.

print_banner("Applied — Credit-Card Cashback A/B")

# Simulate: control = 1.5% cashback, treatment = 2.0% cashback
n_cards = 5_000
card_ctrl = rng.normal(loc=2200, scale=800, size=n_cards)  # monthly spend, SGD
card_treat = rng.normal(loc=2350, scale=850, size=n_cards)  # +$150 lift

card_diff = card_treat.mean() - card_ctrl.mean()
card_se = np.sqrt(card_ctrl.var(ddof=1) / n_cards + card_treat.var(ddof=1) / n_cards)
card_ci = (card_diff - 1.96 * card_se, card_diff + 1.96 * card_se)

# Cost-benefit (illustrative rates). The richer rate is paid on EVERY
# dollar of treatment spend, while the extra interchange is earned only
# on the incremental spend.
ctrl_cashback_rate = 0.015
treat_cashback_rate = 0.020
interchange_rate = 0.015
ctrl_spend = card_ctrl.mean()


def annual_economics(monthly_lift: float) -> tuple[float, float, float]:
    """(extra cashback, extra interchange, net) per customer per year."""
    treat_spend = ctrl_spend + monthly_lift
    extra_cashback = treat_cashback_rate * treat_spend - ctrl_cashback_rate * ctrl_spend
    extra_interchange = interchange_rate * monthly_lift
    return 12 * extra_cashback, 12 * extra_interchange, 12 * (extra_interchange - extra_cashback)


cost_pt, rev_pt, net_pt = annual_economics(card_diff)
_, _, net_at_ci_lo = annual_economics(card_ci[0])
_, _, net_at_ci_hi = annual_economics(card_ci[1])
worst_net = min(net_at_ci_lo, net_at_ci_hi)
best_net = max(net_at_ci_lo, net_at_ci_hi)

print(f"Control (1.5% cashback):  SGD {card_ctrl.mean():,.0f}/mo avg spend")
print(f"Treatment (2.0% cashback): SGD {card_treat.mean():,.0f}/mo avg spend")
print(f"Lift: SGD {card_diff:+,.0f}/mo per customer")
print(f"95% CI: [SGD {card_ci[0]:+,.0f}, SGD {card_ci[1]:+,.0f}]")
print(f"\nPer-customer annual economics (point estimate):")
print(f"  Extra cashback paid:    SGD {cost_pt:,.0f}")
print(f"  Extra interchange:      SGD {rev_pt:,.0f}")
print(f"  Net per customer/yr:    SGD {net_pt:+,.0f}")
print(f"  Net across the lift CI: SGD {worst_net:+,.0f} (worst) to SGD {best_net:+,.0f} (best)")
print(f"\nAt 100K cardholders: SGD {net_pt * 100_000:+,.0f} per year")
if best_net < 0:
    verdict = (
        "Loses money ANYWHERE in the CI — the richer rate on all\n"
        "spend costs more than the extra spend earns. Do not roll out on\n"
        "interchange economics alone (it would need retention or fee value)."
    )
elif worst_net > 0:
    verdict = "Profitable ANYWHERE in the CI — the decision is clear."
else:
    verdict = "The CI spans break-even — gather more data before deciding."
print(f"\nDecision: {verdict}")
# INTERPRETATION: A significant spend lift is not a business case. Run
# the economics at both ends of the CI — here the sign is decided by the
# cost structure, not by the size of the lift.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert card_ci[0] < card_ci[1], "Card CI must be valid"
assert worst_net <= net_pt <= best_net, "Point-estimate net must lie within the CI range"
print("\n>>> Checkpoint 4 passed — cashback scenario completed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - Welch's t-test: always safe, handles unequal variances
  - Satterthwaite df: weighted average accounting for variance differences
  - Three CI methods: Welch, Normal approximation, Bootstrap
  - CI interpretation: "how big" matters more than "is it significant"
  - Applied: card cashback — run the economics at both ends of the CI

  NEXT: In Exercise 4.4 you'll learn experiment validity checks
  (SUTVA, interference) and adaptive sample-size design.
"""
)

print(">>> Exercise 4.3 complete — Welch's t-Test & Confidence Intervals")
