# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 4.4: Validity, Adaptive Design & Experiment Report
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement a data collection plan (Why/What/Where/How/Frequency)
#   - Evaluate experiment validity: SUTVA, interference, novelty effects
#   - Compute adaptive (sequential) sample-size re-estimation from a pilot
#   - Build a complete experiment analysis report with business decision
#   - Log the experiment report (design, validity, decision) to ExperimentTracker
#   - Connect adaptive design to capped regulatory-sandbox trials
#
# PREREQUISITES:
#   - MLFP02 Exercise 4.3 (Welch's t-test, confidence intervals)
#
# ESTIMATED TIME: ~45 minutes
#
# TASKS (5-phase R10):
#   1. Theory — SUTVA and why interference kills causal inference
#   2. Build — data collection plan + validity diagnostics
#   3. Train — adaptive sample-size re-estimation from pilot data
#   4. Visualise — full experiment report with business recommendation
#   5. Apply — adaptive design inside a regulatory sandbox
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kailash_ml import ExperimentTracker
from scipy import stats

from shared.mlfp02.ex_4 import (
    ALPHA,
    DESIGN_MDE_PCT,
    OUTPUT_DIR,
    POWER_TARGET,
    SEED,
    TREATMENT_ARM,
    TwoArmAB,
    designed_control_share,
    load_experiment,
    make_rng,
    power_at_n,
    print_banner,
    required_n_per_group,
    srm_chisquare,
    summarise_arm,
    z_critical,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — SUTVA and Interference
# ════════════════════════════════════════════════════════════════════════
# SUTVA (Stable Unit Treatment Value Assumption):
#   Each user's outcome depends ONLY on their own treatment assignment.
#
# Violations:
#   1. Network effects — treated users share with control users
#   2. Marketplace effects — treatment shifts supply/demand
#   3. Shared resources — treatment consumes server capacity
#
# ADAPTIVE DESIGN:
#   When variance is unknown upfront, start with a pilot phase to
#   estimate sigma, then compute the remaining sample size.

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — LOAD data
# ════════════════════════════════════════════════════════════════════════

print_banner("Exercise 4.4 — Validity, Adaptive Design & Report")

data: TwoArmAB = load_experiment()
rng = make_rng(SEED)

summarise_arm("Control", data.ctrl_values)
summarise_arm("Treatment", data.treat_values)

sigma_pooled = data.ctrl_values.std(ddof=1)
ctrl_mean = data.ctrl_values.mean()
mde_absolute = ctrl_mean * (DESIGN_MDE_PCT / 100)
n_required_per = required_n_per_group(sigma_pooled, mde_absolute, ALPHA, POWER_TARGET)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Data Collection Plan & Validity Diagnostics
# ════════════════════════════════════════════════════════════════════════

print_banner("Data Collection Plan (Why/What/Where/How/Frequency)")

# TODO: Fill in the data collection plan dictionary. Each section is a
# dict of key-value pairs describing that aspect of data collection.
# The 5 sections are: WHY, WHAT, WHERE, HOW, FREQUENCY.
# Fill in the blank values. The success criteria must be checkable
# from the report (SRM, p-value, CI, lift vs the 2% MDE) — the decision
# code in Task 4 applies exactly these criteria.
plan = {
    "WHY (Hypotheses & Value)": {
        "Primary hypothesis": ____,
        "Secondary hypotheses": "Revenue impact, conversion impact",
        "Business value": "(illustrative) 1% engagement lift ~ $200K annual revenue",
        "Success criteria": ____,
    },
    "WHAT (Data Requirements)": {
        "Primary metric": "metric_value (per-user engagement/spend score)",
        "Secondary metrics": ____,
        "Covariates": "signup_date, device_type, country, prior_activity",
        "Guardrail metrics": "page_load_time, error_rate, support_tickets",
        "Minimum rows": f"{2 * n_required_per:,} (from power analysis)",
    },
    "WHERE (Data Sources)": {
        "Internal": "Event stream -> analytics warehouse, user_features table",
        "External": "None required for this experiment",
        "Schema": "user_id, timestamp, experiment_group, metric_value, revenue",
    },
    "HOW (Collection Method)": {
        "Assignment": "Server-side random hash on user_id (deterministic)",
        "Logging": "Event-sourced: every impression, click, purchase logged",
        "Quality": ____,
        "Privacy": "PII stripped at collection; analysis on anonymised IDs",
    },
    "FREQUENCY (Timing)": {
        "Collection frequency": "Real-time events, hourly batch aggregation",
        "Analysis frequency": ____,
        "Duration": f"~{2 * n_required_per // 5000} days at 5,000 users/day",
        "Stopping rule": "Analyse after target n reached; no early stopping",
    },
}

for section, items in plan.items():
    print(f"\n{section}")
    print("-" * 50)
    for key, value in items.items():
        print(f"  {key}: {value}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(plan) == 5, "Plan must cover all 5 sections"
print("\n>>> Checkpoint 1 passed — data collection plan created\n")


# ── Validity Diagnostics ─────────────────────────────────────────────

print_banner("Experiment Validity Criteria")

# Check 1: Variance ratio (should be ~1 if no interference)
# TODO: Compute the variance ratio (treatment variance / control variance).
var_ratio = ____
print(f"  1. Variance ratio (treatment/control): {var_ratio:.3f}")
print(f"     Expected ~1.0 if no differential interference")
print(f"     Status: {'OK' if 0.8 < var_ratio < 1.2 else 'INVESTIGATE'}")

# Check 2: Distribution shape similarity (KS test)
# TODO: Run the two-sample KS test using stats.ks_2samp().
ks_stat, ks_p = ____
print(f"\n  2. KS test for distribution similarity: D={ks_stat:.4f}, p={ks_p:.6f}")
print(f"     A small p-value suggests distributions differ beyond just location shift")

# Check 3: Novelty effect — does the treatment EFFECT fade over time?
# Compare the lift (treatment - control) in the first half of the test
# window with the lift in the second half. Split by TIMESTAMP: the raw
# file is not stored in time order, so "first n rows" is not "early".
ab = data.ab_data  # sorted by the parsed `ts` column (see shared loader)
midpoint = ab["ts"].min() + (ab["ts"].max() - ab["ts"].min()) / 2
period_lifts = {}
for label, frame in [
    ("early", ab.filter(pl.col("ts") < midpoint)),
    ("late", ab.filter(pl.col("ts") >= midpoint)),
]:
    c = frame.filter(pl.col("experiment_group") == "control")["metric_value"].to_numpy()
    t = frame.filter(pl.col("experiment_group") != "control")["metric_value"].to_numpy()
    se = np.sqrt(c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t))
    period_lifts[label] = (t.mean() - c.mean(), se)
(lift_early, se_early), (lift_late, se_late) = period_lifts["early"], period_lifts["late"]
# TODO: z-statistic for the difference between the early and late lifts.
# Hint: (lift_early - lift_late) / sqrt(se_early**2 + se_late**2)
novelty_z = ____
novelty_p = float(2 * stats.norm.sf(abs(novelty_z)))
novelty_detected = novelty_p < 0.05
print(
    f"\n  3. Novelty check (lift, early vs late half of the window): "
    f"early={lift_early:+.3f}, late={lift_late:+.3f}, z={novelty_z:.3f}, p={novelty_p:.4f}"
)
if novelty_detected:
    print("     WARNING: the lift changes over time — possible novelty/fatigue effect")
else:
    print("     OK: no evidence that the lift changes between the two halves")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert var_ratio > 0, "Variance ratio must be positive"
assert 0 <= ks_p <= 1, "KS p-value must be valid"
assert 0 <= novelty_p <= 1, "Novelty p-value must be valid"
print("\n>>> Checkpoint 2 passed — validity criteria assessed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Adaptive Sample Size from Pilot
# ════════════════════════════════════════════════════════════════════════

print_banner("Adaptive Sample Size Calculation")

z_a, z_b = z_critical(ALPHA, POWER_TARGET)

# Simulate a pilot phase
pilot_n = 500
pilot_ctrl = rng.choice(data.ctrl_values, size=pilot_n, replace=True)
pilot_treat = rng.choice(data.treat_values, size=pilot_n, replace=True)

pilot_sigma = np.sqrt((pilot_ctrl.var(ddof=1) + pilot_treat.var(ddof=1)) / 2)
pilot_diff = pilot_treat.mean() - pilot_ctrl.mean()

# TODO: Re-compute required n based on pilot sigma estimate using
# required_n_per_group(pilot_sigma, mde_absolute, ALPHA, POWER_TARGET).
n_adaptive = ____

print(f"Pilot phase: n={pilot_n} per group")
print(f"Pilot sigma estimate: {pilot_sigma:.4f} (true sigma ~ {sigma_pooled:.4f})")
print(f"Pilot observed diff: {pilot_diff:+.4f}")
print(f"\nAdaptive required n per group: {n_adaptive:,}")
print(f"Original required n per group: {n_required_per:,}")
print(f"Ratio: {n_adaptive / n_required_per:.2f}x")
print(f"Remaining needed: {max(0, n_adaptive - pilot_n):,} per group")

# Multi-stage: how estimate improves with pilot size
print(f"\n--- Pilot Size vs Required n Stability ---")
for pilot_size in [100, 250, 500, 1000]:
    sigs = []
    for _ in range(100):
        pc = rng.choice(data.ctrl_values, size=pilot_size, replace=True)
        pt = rng.choice(data.treat_values, size=pilot_size, replace=True)
        s = np.sqrt((pc.var(ddof=1) + pt.var(ddof=1)) / 2)
        sigs.append(s)
    mean_sig = np.mean(sigs)
    std_sig = np.std(sigs)
    n_req = required_n_per_group(mean_sig, mde_absolute, ALPHA, POWER_TARGET)
    print(
        f"  Pilot n={pilot_size:>4}: sigma_hat={mean_sig:.4f} +/- {std_sig:.4f}, "
        f"required n={n_req:,}"
    )

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert n_adaptive > 0, "Adaptive sample size must be positive"
print("\n>>> Checkpoint 3 passed — adaptive design completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Full Experiment Report
# ════════════════════════════════════════════════════════════════════════

# Re-compute final statistics for the report
real_t_stat, real_p_val = stats.ttest_ind(
    data.treat_values, data.ctrl_values, equal_var=False
)
obs_diff = data.treat_values.mean() - data.ctrl_values.mean()
rel_lift = obs_diff / data.ctrl_values.mean() * 100

s1_sq_n1 = data.ctrl_values.var(ddof=1) / data.n_control
s2_sq_n2 = data.treat_values.var(ddof=1) / data.n_treatment
df_ws = (s1_sq_n1 + s2_sq_n2) ** 2 / (
    s1_sq_n1**2 / (data.n_control - 1) + s2_sq_n2**2 / (data.n_treatment - 1)
)
real_se = np.sqrt(s1_sq_n1 + s2_sq_n2)
t_crit = stats.t.ppf(1 - ALPHA / 2, df=df_ws)
welch_ci = (obs_diff - t_crit * real_se, obs_diff + t_crit * real_se)

pooled_std_real = np.sqrt(
    (data.ctrl_values.var(ddof=1) + data.treat_values.var(ddof=1)) / 2
)
cohens_d_real = obs_diff / pooled_std_real

# SRM check against the DESIGNED split (control 40 : treatment_a 35)
_, srm_p = srm_chisquare(data.n_control, data.n_treatment, designed_control_share())
srm_pass = srm_p >= 0.01
ci_excludes_zero = welch_ci[0] > 0 or welch_ci[1] < 0
meets_mde = obs_diff >= mde_absolute

# Adaptive sigma stability plot
fig_adapt = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "Power Curve",
        "Pilot Size vs sigma Stability",
    ],
)

# Power curve subplot
sample_sizes = np.arange(500, n_required_per * 3, max(500, n_required_per // 20))
power_values = [
    power_at_n(int(ns), sigma_pooled, mde_absolute, ALPHA) for ns in sample_sizes
]
fig_adapt.add_trace(
    go.Scatter(
        x=sample_sizes.tolist(),
        y=power_values,
        mode="lines",
        name="Power",
        line={"color": "#2196F3"},
    ),
    row=1,
    col=1,
)
fig_adapt.add_hline(y=0.8, line_dash="dash", row=1, col=1)

# Sigma stability subplot
pilot_sizes_list = [50, 100, 200, 300, 500, 750, 1000]
sigma_means = []
sigma_stds = []
for ps in pilot_sizes_list:
    sigs = []
    for _ in range(200):
        pc = rng.choice(data.ctrl_values, size=ps, replace=True)
        pt = rng.choice(data.treat_values, size=ps, replace=True)
        sigs.append(np.sqrt((pc.var(ddof=1) + pt.var(ddof=1)) / 2))
    sigma_means.append(np.mean(sigs))
    sigma_stds.append(np.std(sigs))

fig_adapt.add_trace(
    go.Scatter(
        x=pilot_sizes_list,
        y=sigma_means,
        mode="lines+markers",
        name="sigma_hat",
        error_y={"type": "data", "array": sigma_stds, "visible": True},
        line={"color": "#FF9800"},
    ),
    row=1,
    col=2,
)
fig_adapt.add_hline(y=sigma_pooled, line_dash="dot", row=1, col=2)
fig_adapt.update_layout(
    title="Adaptive Design: Power Curve & Pilot Stability",
    height=400,
    template="plotly_white",
)
out_path = OUTPUT_DIR / "adaptive_design.html"
fig_adapt.write_html(str(out_path))
print(f"Saved: {out_path}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert out_path.exists(), "Adaptive design plot must be saved"
print("\n>>> Checkpoint 4 passed — visualisations saved\n")


# ── Final business report ─────────────────────────────────────────────

# ── Decision: apply the pre-registered success criteria, SRM first ────
if not srm_pass:
    decision = "DO NOT SHIP — investigate SRM before trusting any result"
elif real_p_val < ALPHA and ci_excludes_zero and meets_mde:
    decision = "SHIP — all pre-registered success criteria met"
elif real_p_val < ALPHA and obs_diff > 0:
    decision = "HOLD — significant, but the lift is below the pre-registered MDE"
elif real_p_val >= ALPHA and abs(obs_diff) > mde_absolute * 0.5:
    decision = "HOLD — more data needed"
else:
    decision = "NO SHIP — no meaningful effect detected"
if novelty_detected and decision.startswith("SHIP"):
    decision += " (monitor: lift changed over the window — possible novelty effect)"
novelty_verdict = (
    "possible novelty effect" if novelty_detected else "no novelty effect detected"
)

print_banner("EXPERIMENT REPORT")
print(
    f"""
Experiment: Recommendation Algorithm A/B Test
Duration: designed for {2 * n_required_per:,} total users
Actual: {data.n_total:,} users

SRM Check (vs designed split): p={srm_p:.4f} -> {'PASS' if srm_pass else 'FAIL — results may be biased'}

Primary Metric (metric_value):
  Control:   {data.ctrl_values.mean():.4f} +/- {data.ctrl_values.std():.4f}
  Treatment: {data.treat_values.mean():.4f} +/- {data.treat_values.std():.4f}
  Lift: {obs_diff:+.4f} ({rel_lift:+.2f}% relative; MDE {mde_absolute:.4f})
  p-value: {real_p_val:.6f}
  Cohen's d: {cohens_d_real:.4f}
  95% CI: [{welch_ci[0]:.4f}, {welch_ci[1]:.4f}]

Decision: {decision}

Validity: variance ratio {var_ratio:.2f}; novelty check p={novelty_p:.3f} -> {novelty_verdict}.
"""
)
# INTERPRETATION: Cohen's d is small here even though the relative lift
# is large, because ~2% extreme values inflate the standard deviation.
# That is why the success criterion is stated as a lift vs the MDE the
# business pre-registered, not as a Cohen's d cut-off.

# ── Log the report to ExperimentTracker ──────────────────────────────
# The statistics are computed above; the tracker RECORDS the design, the
# validity checks and the decision so the experiment is auditable later.


async def log_experiment_report() -> str:
    store_url = "sqlite:///mlfp02_experiments.db"
    # TODO: Create the tracker against a local SQLite store.
    # Hint: await ExperimentTracker.create(store_url=store_url)
    tracker = ____
    try:
        async with tracker.track(
            experiment="mlfp02_ex4_ab_report", run_name="recommendation_ab"
        ) as run:
            await run.log_params(
                {
                    "treatment_arm": TREATMENT_ARM,
                    "design_mde_pct": str(DESIGN_MDE_PCT),
                    "alpha": str(ALPHA),
                    "power_target": str(POWER_TARGET),
                    "designed_control_share": f"{designed_control_share():.4f}",
                    "decision": decision,
                }
            )
            await run.log_metrics(
                {
                    "srm_p_value": float(srm_p),
                    "lift": float(obs_diff),
                    "relative_lift_pct": float(rel_lift),
                    "p_value": float(real_p_val),
                    "cohens_d": float(cohens_d_real),
                    "ci_lower": float(welch_ci[0]),
                    "ci_upper": float(welch_ci[1]),
                    "novelty_p_value": float(novelty_p),
                    "n_required_per_group": float(n_required_per),
                }
            )
            return run.run_id
    finally:
        await tracker.close()


report_run_id = asyncio.run(log_experiment_report())
print(f"Logged experiment report to ExperimentTracker (run {report_run_id})")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert srm_pass or decision.startswith("DO NOT SHIP"), "Never ship on a failed SRM check"
assert report_run_id, "Report must be logged to ExperimentTracker"
print(">>> Checkpoint 5 passed — experiment report complete and logged\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Adaptive Design Inside a Regulatory Sandbox
# ════════════════════════════════════════════════════════════════════════
# Singapore's financial regulator runs a FinTech Regulatory Sandbox in
# which firms trial new products with real customers for a limited
# period. Participants and time are capped, and the outcome variance is
# unknown before the trial starts — a natural fit for adaptive design.
#
# Scenario (illustrative numbers): a robo-advisor feature, outcome =
# annualised portfolio return (%), at most 2,000 participants.

print_banner("Applied — Regulatory Sandbox Adaptive Experiment")

sandbox_n_available = 2000
robo_sigma_guess = 8.0  # Cautious initial guess of return volatility (%)
robo_mde = 1.0  # Want to detect a 1 pp return improvement

# Phase 1: initial estimate from the guessed sigma
# TODO: Compute initial required n using the guessed sigma.
n_initial = ____
print(
    f"Initial estimate (sigma guess = {robo_sigma_guess}%): n={n_initial:,} per group"
)
feasible_initial = 2 * n_initial <= sandbox_n_available
print(
    f"Feasible in sandbox ({sandbox_n_available} available)? "
    f"{'YES' if feasible_initial else 'NO'}"
)

# Phase 2: a pilot estimates sigma from data (the scenario's true
# volatility, unknown to the team, is 4.2%)
pilot_n_sandbox = 200
true_robo_sigma = 4.2
pilot_returns = rng.normal(0.0, true_robo_sigma, size=(2, pilot_n_sandbox))
robo_sigma_pilot = float(np.sqrt(pilot_returns.var(axis=1, ddof=1).mean()))
# TODO: Compute revised required n with the pilot sigma estimate.
n_revised = ____
print(f"\nAfter pilot (n={pilot_n_sandbox} per arm, sigma_hat={robo_sigma_pilot:.2f}%):")
print(f"  Revised n per group: {n_revised:,}")
print(f"  Remaining: {max(0, n_revised - pilot_n_sandbox):,} per group")
feasible_revised = 2 * n_revised <= sandbox_n_available
print(f"  Feasible? {'YES' if feasible_revised else 'NO'}")

# Phase 3: power at the maximum available sample size
# TODO: Compute achieved power at the maximum available n per group.
achieved_power = ____
print(f"\nPower at maximum available n ({sandbox_n_available // 2} per group):")
print(f"  Achieved power: {achieved_power:.1%}")
print(
    f"  {'ADEQUATE' if achieved_power >= 0.8 else 'Consider larger MDE or extended sandbox'}"
)
if not feasible_initial and feasible_revised:
    print(
        "\nThe cautious sigma guess made the trial look infeasible; the pilot's\n"
        "variance estimate showed it fits within the sandbox's participant cap."
    )
elif feasible_initial:
    print("\nThe trial was feasible even under the initial guess; the pilot")
    print("confirms the plan and lets the team stop at the revised n.")
else:
    print("\nEven after the pilot the trial does not fit — widen the MDE,")
    print("extend the sandbox period, or redesign the outcome metric.")
# INTERPRETATION: When participants and time are capped, a pilot that
# re-estimates sigma lets you size the trial on evidence rather than a
# guess — avoiding both an underpowered trial and a wasted one.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert n_revised > 0, "Revised n must be positive"
assert 0 < achieved_power <= 1, "Achieved power must be valid"
assert not feasible_initial, "With sigma guess 8%, the initial design should not fit"
print("\n>>> Checkpoint 6 passed — sandbox scenario completed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - Data collection plan: Why/What/Where/How/Frequency framework
  - SUTVA: interference, novelty effects, variance-ratio checks
  - Adaptive design: pilot -> estimate sigma -> compute remaining n
  - Pilot stability: larger pilots => more reliable sigma estimates
  - Complete experiment report: SRM gate first, then pre-registered
    success criteria; logged to ExperimentTracker
  - Novelty check: compare the lift across time, split by timestamp
  - Applied: adaptive design inside a capped regulatory sandbox

  NEXT: In Exercise 5 you'll build linear regression from scratch.
  You'll derive OLS using matrix algebra, test coefficient significance
  with t-statistics, detect multicollinearity, and run residual
  diagnostics — all on HDB price prediction data.
"""
)

print(">>> Exercise 4.4 complete — Validity, Adaptive Design & Report")
print("\n>>> Exercise 4 complete — A/B Testing and Experiment Design")
