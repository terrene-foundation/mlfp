# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 7.3: Sequential Testing with mSPRT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement sequential testing with mSPRT (always-valid p-values)
#   - Demonstrate the peeking problem with simulation
#   - Compare fixed vs sequential p-values over time
#   - Understand why standard p-values fail under continuous monitoring
#   - Log sequential results to ExperimentTracker
#
# PREREQUISITES: Exercise 7.1-7.2 (CUPED, Bayesian concepts)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load experiment data and compute baseline SE
#   2. Sequential testing: mSPRT always-valid p-values day by day
#   3. Peeking problem simulation: inflated Type I error
#   4. Compare fixed vs sequential p-value trajectories
#   5. Visualise both analyses
#   6. Apply to Singapore ride-hailing scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import plotly.graph_objects as go
from kailash_ml import ExperimentTracker

from shared.mlfp02.ex_7 import (
    ANALYSIS_ARM,
    OUTPUT_DIR,
    compute_srm,
    get_revenue_arrays,
    load_experiment,
    msprt_lambda,
    msprt_sequential_pvalues,
    naive_ab,
    print_banner,
    simulate_peeking,
    split_groups,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Peeking Problem and Sequential Testing
# ════════════════════════════════════════════════════════════════════════
# Standard p-values are designed for a SINGLE look at the data. If you
# peek at your experiment 20 times and stop the first time p < 0.05, the
# actual false positive rate is far above 5% — Task 3 measures it.
#
# Why? Every peek is another chance for random noise to cross the line.
# The peeks are NOT independent tests: each look re-uses all the data of
# the previous look plus a little more, so consecutive z-statistics are
# highly correlated. That is why the naive formula 1 - 0.95^20 = 64% for
# 20 independent tests is wrong here; for 20 equally spaced looks the
# true rate is roughly 25% — still five times the promised 5%. Because
# there is no simple closed form, we measure it by simulation.
#
# mSPRT (mixture Sequential Probability Ratio Test) provides "always-
# valid" p-values that remain correct no matter when you look. The
# trade-off: mSPRT p-values are more conservative — they need more
# data to reach significance. But they never lie.
#
# Analogy: Fixed p-values are like a bathroom scale that only gives the
# right weight if you step on it exactly once. Step on it 20 times and
# take the lowest reading? You'll think you lost weight. mSPRT is a
# scale that gives the correct reading no matter how many times you
# step on it.
#
# The always-valid p-value is the running minimum of 1 / Lambda_n, where
#   Lambda_n = sqrt(V/(V+tau^2)) * exp(tau^2 * diff^2 / (2 V (V+tau^2)))
# V is the variance of the current difference estimate and tau^2 is the
# width of the mixing prior over plausible effect sizes.
#
# WHY THIS MATTERS: Experimentation platforms whose dashboards update
# continuously invite daily peeking; without sequential methods, a large
# share of "significant" results are noise that disappears after rollout.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and Compute Baseline
# ════════════════════════════════════════════════════════════════════════

print_banner("MLFP02 Exercise 7.3: Sequential Testing (mSPRT)")

experiment = load_experiment()
control, treatment = split_groups(experiment, ANALYSIS_ARM)
srm_p = compute_srm(control.height, treatment.height)  # vs the designed 40:35 ratio
if srm_p < 0.01:
    raise RuntimeError(f"SRM on control vs {ANALYSIS_ARM} (p={srm_p:.2g}) — stop")
y_c, y_t = get_revenue_arrays(control, treatment)
baseline = naive_ab(y_c, y_t)
se_naive = baseline["se"]

print(f"  Data loaded: {experiment.shape[0]:,} rows; analysing control vs {ANALYSIS_ARM}")
print(f"  Pairwise SRM p={srm_p:.3f} (OK)")
print(f"  Baseline SE: ${se_naive:.2f}")
print(f"  Baseline lift: ${baseline['lift']:.2f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert se_naive > 0, "Baseline SE must be positive"
print("\n>>> Checkpoint 1 passed -- data loaded and baseline computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Sequential Testing: mSPRT Day by Day
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Sequential Testing (mSPRT) ===")

tau_sq = se_naive**2  # mSPRT hyperparameter
sequential_results = msprt_sequential_pvalues(
    experiment, tau_sq=tau_sq, treatment_arm=ANALYSIS_ARM
)

print(f"{'Day':>4} {'n':>8} {'Lift':>10} {'p (fixed)':>12} {'p (mSPRT)':>12}")
print("-" * 52)
step = max(1, len(sequential_results) // 10)
for r in sequential_results[::step]:
    print(
        f"{r['day']:>4} {int(r['n']):>8,} ${r['lift']:>8.2f} "
        f"{r['p_fixed']:>12.6f} {r['p_sequential']:>12.6f}"
    )

early_sig_fixed = sum(1 for r in sequential_results if r["p_fixed"] < 0.05)
early_sig_seq = sum(1 for r in sequential_results if r["p_sequential"] < 0.05)
print(f"\nDays with p < 0.05 (fixed):      {early_sig_fixed}/{len(sequential_results)}")
print(f"Days with p < 0.05 (sequential): {early_sig_seq}/{len(sequential_results)}")
# INTERPRETATION: Fixed p-values may cross 0.05 early (giving false
# confidence), while mSPRT p-values stay conservative until real
# evidence accumulates. If a product manager stopped at the first
# fixed-p < 0.05, they might ship a non-effect.

# Compute the mixture likelihood ratio yourself at the final look
# (all data): diff = full-sample lift, V = se_naive^2.
v_final = se_naive**2
diff_final = baseline["lift"]
lambda_final = np.sqrt(v_final / (v_final + tau_sq)) * np.exp(
    tau_sq * diff_final**2 / (2 * v_final * (v_final + tau_sq))
)
print(f"\nFinal look: Lambda = {lambda_final:.3g}, 1/Lambda = {1 / lambda_final:.3g}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(sequential_results) > 0, "Must have sequential results"
assert np.isclose(lambda_final, msprt_lambda(diff_final, v_final, tau_sq), rtol=1e-9), (
    "Your Lambda should match the reference msprt_lambda helper"
)
seq_ps = [r["p_sequential"] for r in sequential_results]
assert all(b <= a for a, b in zip(seq_ps, seq_ps[1:])), "Always-valid p must never increase"
for r in sequential_results:
    assert 0 <= r["p_sequential"] <= 1, "Sequential p-values must be valid"
print("\n>>> Checkpoint 2 passed -- sequential testing completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Peeking Problem Simulation
# ════════════════════════════════════════════════════════════════════════
# Simulate experiments with NO real effect to show how peeking inflates
# the false positive rate.

print(f"\n=== Peeking Problem Simulation ===")

peek_results = simulate_peeking(n_sims=1000, n_per_sim=2000, n_checks=20, seed=42)

print(f"Simulations: {int(peek_results['n_sims']):,} (all with NO real effect)")
print(f"Peeks per experiment: {int(peek_results['n_checks'])}")
print(f"\nFalse positive rates:")
print(f"  No peeking (test at end):   {peek_results['rate_no_peek']:.1%} (target: 5%)")
print(
    f"  Peeking with fixed p:       {peek_results['rate_fixed_peek']:.1%} (inflated!)"
)
print(
    f"  Inflation factor:           "
    f"{peek_results['rate_fixed_peek'] / peek_results['rate_no_peek']:.1f}x"
)
# INTERPRETATION: Read the simulated rate above — with 20 looks it lands
# around 25%, not 5%. It is well below the 64% an "independent tests"
# calculation would predict, because each look re-uses earlier data.
# Sequential testing (mSPRT) is the correct way to monitor experiments.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    peek_results["rate_fixed_peek"] > peek_results["rate_no_peek"]
), "Peeking must inflate false positive rate"
print("\n>>> Checkpoint 3 passed -- peeking problem demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: Fixed vs Sequential p-Values Over Time
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
days_seq = [r["day"] for r in sequential_results]
fig.add_trace(
    go.Scatter(
        x=days_seq,
        y=[r["p_fixed"] for r in sequential_results],
        name="Fixed p-value",
    )
)
fig.add_trace(
    go.Scatter(
        x=days_seq,
        y=[r["p_sequential"] for r in sequential_results],
        name="mSPRT p-value",
    )
)
fig.add_hline(y=0.05, line_dash="dash", annotation_text="alpha=0.05")
fig.update_layout(
    title="Sequential Testing: Fixed vs mSPRT p-values",
    xaxis_title="Day",
    yaxis_title="p-value",
    yaxis_type="log",
)
out_path = OUTPUT_DIR / "sequential_pvalues.html"
fig.write_html(str(out_path))
print(f"\nSaved: {out_path}")

# Peeking problem visualisation
fig2 = go.Figure()
categories = ["No peeking", f"Peeking ({int(peek_results['n_checks'])} looks, fixed p)"]
rates = [
    peek_results["rate_no_peek"],
    peek_results["rate_fixed_peek"],
]
colours = ["green", "red"]
fig2.add_trace(
    go.Bar(
        x=categories,
        y=rates,
        marker_color=colours,
        text=[f"{r:.1%}" for r in rates],
        textposition="auto",
    )
)
fig2.add_hline(y=0.05, line_dash="dash", annotation_text="Nominal alpha=5%")
fig2.update_layout(
    title="Peeking Problem: False Positive Rate Inflation",
    yaxis_title="False Positive Rate",
    yaxis_tickformat=".0%",
)
out_path2 = OUTPUT_DIR / "peeking_problem.html"
fig2.write_html(str(out_path2))
print(f"Saved: {out_path2}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — A Singapore Ride-Hailing Platform: Continuous Monitoring
# ════════════════════════════════════════════════════════════════════════
# Scenario (illustrative figures): a ride-hailing platform concludes ~50
# experiments a month. Dashboards update continuously and product
# managers check results about 20 times per experiment.
#
# Worst case — none of the 50 changes truly works:
#   - With peeking, each experiment has the SIMULATED false-positive rate
#     from Task 3 of being declared a winner.
#   - With mSPRT, the false-positive rate is at most 5% (alpha).
#   - Each false positive ships a non-effect: development, rollback and
#     re-test cost an assumed S$100K.
#   - Trade-off: mSPRT needs more data than a single fixed-horizon test.

print(f"\n--- Singapore Application: Ride-Hailing Experiment Monitoring ---")
n_experiments_per_month = 50  # illustrative
fp_peeking = n_experiments_per_month * peek_results["rate_fixed_peek"]
fp_msprt_bound = n_experiments_per_month * 0.05  # upper bound under H0
cost_per_fp = 100_000  # S$, illustrative
print(f"Experiments concluded per month: {n_experiments_per_month}")
print(f"False positives per month (peeking, simulated rate): ~{fp_peeking:.1f}")
print(f"False positives per month (mSPRT, at most):          ~{fp_msprt_bound:.1f}")
print(f"Cost per false positive (assumed): S${cost_per_fp:,}")
print(f"Annual waste (peeking): S${fp_peeking * cost_per_fp * 12:,.0f}")
print(f"Annual waste (mSPRT, at most): S${fp_msprt_bound * cost_per_fp * 12:,.0f}")
print(f"Annual savings (at least): S${(fp_peeking - fp_msprt_bound) * cost_per_fp * 12:,.0f}")


# ════════════════════════════════════════════════════════════════════════
# LOG — ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_sequential_results():
    db = "sqlite:///mlfp02_experiments.db"
    tracker = await ExperimentTracker.create(store_url=db)

    exp_id = "mlfp02_ex7_sequential_testing"

    async with tracker.track(experiment=exp_id, run_name="msprt_analysis") as run:
        await run.log_params(
            {
                "sequential_method": "mSPRT",
                "treatment_arm": ANALYSIS_ARM,
                "tau_sq": str(float(tau_sq)),
                "n_peek_sims": "1000",
                "n_checks": "20",
            }
        )
        await run.log_metrics(
            {
                "days_sig_fixed": float(early_sig_fixed),
                "days_sig_sequential": float(early_sig_seq),
                "fp_rate_no_peek": float(peek_results["rate_no_peek"]),
                "fp_rate_peeking": float(peek_results["rate_fixed_peek"]),
            }
        )
    print(f"\nLogged sequential testing run")
    await tracker.close()


asyncio.run(log_sequential_results())

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n>>> Checkpoint 4 passed -- visualisation and logging complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  - mSPRT: always-valid p-values for safe experiment monitoring
  - Peeking problem: {int(peek_results['n_checks'])} correlated peeks inflate alpha from 5% to ~{peek_results['rate_fixed_peek']:.0%} (simulated)
  - The mixture likelihood ratio Lambda and the running-minimum p-value
  - Fixed vs sequential p-value trajectories
  - tau_sq hyperparameter: set to baseline SE^2
  - Why dashboards with live p-values need sequential methods

  NEXT: In 04_diff_in_diff.py, you'll learn to estimate causal effects
  from observational data when randomisation is not possible.
"""
)

print("\n>>> Exercise 7.3 complete -- Sequential Testing with mSPRT")
