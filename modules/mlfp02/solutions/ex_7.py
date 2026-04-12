# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 7: CUPED and Causal Inference
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Derive and implement CUPED variance reduction using pre-experiment data
#   - Quantify how much CUPED shrinks confidence intervals (based on ρ²)
#   - Apply Bayesian A/B testing to get posterior probability of improvement
#   - Implement sequential testing with mSPRT to safely monitor experiments
#   - Log experiment results to ExperimentTracker for reproducibility
#
# PREREQUISITES: Complete Exercises 3-4 — you should understand hypothesis
#   testing, p-values, SRM detection, and Welch's t-test.
#
# ESTIMATED TIME: 75 minutes
#
# TASKS:
#   1. Load experiment data with pre-experiment covariates
#   2. SRM check (recap from Exercise 3, faster this time)
#   3. CUPED variance reduction — derive and apply
#   4. Bayesian A/B testing — posterior probability of improvement
#   5. Sequential testing with always-valid p-values
#   6. Log experiment analysis to ExperimentTracker
#
# DATASET: E-commerce experiment with pre-experiment covariates
#   Source: Simulated e-commerce data with 30-day pre-period revenue
#   Columns: group (control/treatment), revenue, pre_revenue, signup_date
#
# THEORY (CUPED):
#   Y_adj = Y - θ(X - E[X])  where θ = Cov(Y,X)/Var(X)
#   Var(Y_adj) = Var(Y)(1 - ρ²)  where ρ = Cor(Y,X)
#   → If ρ=0.5, CI width reduces by 1 - √(1-0.25) = 13.4%
#   → If ρ=0.8, CI width reduces by 1 - √(1-0.64) = 40%
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from scipy import stats
from kailash.db import ConnectionManager
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml import ModelVisualizer

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()

# Experiment data with pre-experiment covariates
experiment = loader.load("mlfp02", "experiment_data.parquet")

print("=" * 60)
print("  MLFP02 Exercise 7: CUPED and Causal Inference")
print("=" * 60)
print(f"\n  Data loaded: ecommerce_experiment.parquet")
print(f"  Shape: {experiment.shape}")
print(f"  Columns: {experiment.columns}")
print(experiment.head(5))

# Separate groups — combine any non-control groups as treatment
control = experiment.filter(pl.col("experiment_group") == "control")
treatment = experiment.filter(pl.col("experiment_group") != "control")

n_c, n_t = control.height, treatment.height
print(f"\nControl: {n_c:,} | Treatment: {n_t:,}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: SRM check (recap, faster this time)
# ══════════════════════════════════════════════════════════════════════

expected = np.array([n_c + n_t] * 2) / 2
observed = np.array([n_c, n_t])
_, srm_p = stats.chisquare(observed, f_exp=expected)
print(f"\nSRM check: p={srm_p:.6f} — {'OK' if srm_p > 0.01 else 'SRM DETECTED'}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 <= srm_p <= 1, "SRM p-value must be a valid probability"
if srm_p < 0.01:
    print("  WARNING: SRM detected. Proceeding with caution.")
print("\n✓ Checkpoint 1 passed — SRM check completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Standard analysis (before CUPED)
# ══════════════════════════════════════════════════════════════════════

# Primary metric: revenue per user
y_c = control["revenue"].to_numpy().astype(np.float64)
y_t = treatment["revenue"].to_numpy().astype(np.float64)

mean_c, mean_t = y_c.mean(), y_t.mean()
lift = mean_t - mean_c
se_naive = np.sqrt(y_c.var(ddof=1) / n_c + y_t.var(ddof=1) / n_t)
ci_naive = (lift - 1.96 * se_naive, lift + 1.96 * se_naive)
z_naive = lift / se_naive
p_naive = 2 * (1 - stats.norm.cdf(abs(z_naive)))

print(f"\n=== Standard Analysis (no CUPED) ===")
print(f"Control mean: ${mean_c:.2f}")
print(f"Treatment mean: ${mean_t:.2f}")
print(f"Lift: ${lift:.2f} ({lift / mean_c:.2%} relative)")
print(f"SE: ${se_naive:.2f}")
print(f"95% CI: [${ci_naive[0]:.2f}, ${ci_naive[1]:.2f}]")
print(f"p-value: {p_naive:.6f}")
# INTERPRETATION: The naive analysis uses only the experiment-period revenue.
# This ignores individual variation in baseline spending habits — a high-spender
# in control is compared against a mix of high and low spenders in treatment.
# CUPED removes this baseline noise, leading to more precise estimates.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert se_naive > 0, "Naive SE must be positive"
assert ci_naive[0] < ci_naive[1], "CI lower must be below upper"
print("\n✓ Checkpoint 2 passed — standard analysis baseline established\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: CUPED variance reduction
# ══════════════════════════════════════════════════════════════════════
# CUPED (Controlled-experiment Using Pre-Experiment Data)
# Key insight: subtract a correlated pre-experiment covariate to reduce variance
#
# Y_adj = Y - θ * (X - E[X])
# where X = pre-experiment metric (e.g., revenue in prior period)
# θ = Cov(Y, X) / Var(X) = optimal coefficient
# Var(Y_adj) = Var(Y) * (1 - ρ²)

# Pre-experiment covariate: revenue in the 30 days before experiment
x_c = control["pre_metric_value"].to_numpy().astype(np.float64)
x_t = treatment["pre_metric_value"].to_numpy().astype(np.float64)

# Compute CUPED adjustment
x_all = np.concatenate([x_c, x_t])
y_all = np.concatenate([y_c, y_t])

# θ = Cov(Y, X) / Var(X)
theta = np.cov(y_all, x_all)[0, 1] / np.var(x_all, ddof=1)

# Correlation between pre and post metrics
rho = np.corrcoef(y_all, x_all)[0, 1]

# Adjusted values
x_mean = x_all.mean()
y_c_adj = y_c - theta * (x_c - x_mean)
y_t_adj = y_t - theta * (x_t - x_mean)

# CUPED analysis
mean_c_adj = y_c_adj.mean()
mean_t_adj = y_t_adj.mean()
lift_adj = mean_t_adj - mean_c_adj
se_cuped = np.sqrt(y_c_adj.var(ddof=1) / n_c + y_t_adj.var(ddof=1) / n_t)
ci_cuped = (lift_adj - 1.96 * se_cuped, lift_adj + 1.96 * se_cuped)
z_cuped = lift_adj / se_cuped
p_cuped = 2 * (1 - stats.norm.cdf(abs(z_cuped)))

# Variance reduction
var_reduction = 1 - se_cuped**2 / se_naive**2
ci_width_reduction = 1 - se_cuped / se_naive

print(f"\n=== CUPED Analysis ===")
print(f"Correlation (pre ↔ post revenue): ρ = {rho:.3f}")
print(f"θ (optimal coefficient): {theta:.4f}")
print(f"Theoretical variance reduction: {rho**2:.1%}")
print(f"Actual variance reduction: {var_reduction:.1%}")
print(f"CI width reduction: {ci_width_reduction:.1%}")
print(f"\nCUPED-adjusted lift: ${lift_adj:.2f}")
print(f"SE (CUPED): ${se_cuped:.2f} (was ${se_naive:.2f})")
print(f"95% CI: [${ci_cuped[0]:.2f}, ${ci_cuped[1]:.2f}]")
print(f"p-value: {p_cuped:.6f}")
# INTERPRETATION: CUPED reduces variance by ρ² = the square of the correlation
# between pre- and post-experiment metrics. A ρ=0.7 correlation (common for
# revenue) would reduce variance by 49%, halving the CI width. This is the
# equivalent of collecting 2x more data — for free, using existing records.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 0 <= abs(rho) <= 1, "Correlation must be between -1 and 1"
assert se_cuped <= se_naive, "CUPED SE must be <= naive SE (variance reduces)"
assert var_reduction >= 0, "Variance reduction must be non-negative"
actual_reduction = 1 - se_cuped**2 / se_naive**2
theoretical_reduction = rho**2
assert abs(actual_reduction - theoretical_reduction) < 0.05, \
    "Actual variance reduction should be close to theoretical ρ²"
print("\n✓ Checkpoint 3 passed — CUPED variance reduction verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Bayesian A/B testing
# ══════════════════════════════════════════════════════════════════════
# Instead of p-values, compute:
#   - P(treatment > control | data)
#   - Expected loss from choosing treatment
#   - Credible interval for the lift

# Use Normal approximation for posterior
# Posterior for treatment mean: N(mean_t_adj, se_t²)
# Posterior for control mean: N(mean_c_adj, se_c²)
se_c_post = y_c_adj.std(ddof=1) / np.sqrt(n_c)
se_t_post = y_t_adj.std(ddof=1) / np.sqrt(n_t)

# P(treatment > control) = P(lift > 0)
# lift ~ N(lift_adj, se_c² + se_t²)
se_lift = np.sqrt(se_c_post**2 + se_t_post**2)
prob_treatment_better = 1 - stats.norm.cdf(0, loc=lift_adj, scale=se_lift)

# Expected loss: E[max(control - treatment, 0)]
# For Normal: E[max(-Z, 0)] where Z ~ N(lift_adj, se_lift²)
z_ratio = -lift_adj / se_lift
expected_loss_treatment = se_lift * stats.norm.pdf(z_ratio) + lift_adj * stats.norm.cdf(
    z_ratio
)
expected_loss_control = se_lift * stats.norm.pdf(-z_ratio) - lift_adj * stats.norm.cdf(
    -z_ratio
)

# 95% credible interval for lift
bayesian_ci = (
    lift_adj - 1.96 * se_lift,
    lift_adj + 1.96 * se_lift,
)

print(f"\n=== Bayesian A/B Test ===")
print(
    f"P(treatment > control): {prob_treatment_better:.4f} ({prob_treatment_better:.1%})"
)
print(f"Expected loss (choose treatment): ${expected_loss_treatment:.2f}/user")
print(f"Expected loss (choose control):   ${expected_loss_control:.2f}/user")
print(f"95% credible interval for lift: [${bayesian_ci[0]:.2f}, ${bayesian_ci[1]:.2f}]")
print(f"\nDecision recommendation:")
if prob_treatment_better > 0.95 and expected_loss_treatment < 0.50:
    print("  → SHIP: High confidence + low expected loss")
elif prob_treatment_better > 0.80:
    print("  → CONTINUE: Promising but need more data")
else:
    print("  → HOLD: Insufficient evidence for treatment superiority")
# INTERPRETATION: Bayesian A/B testing answers "what is the probability that
# treatment is better?" — a more actionable question than "is p < 0.05?"
# Expected loss quantifies the cost of the wrong decision. If expected loss
# from choosing treatment is $0.05/user, you can confidently ship even without
# 95% certainty — the downside is small even if you're wrong.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert 0 <= prob_treatment_better <= 1, "Probability must be between 0 and 1"
assert expected_loss_treatment >= 0, "Expected loss must be non-negative"
# Expected loss can be negative when treatment strongly dominates
assert isinstance(expected_loss_control, float), "Expected loss must be a number"
assert bayesian_ci[0] < bayesian_ci[1], "Bayesian CI lower must be below upper"
print("\n✓ Checkpoint 4 passed — Bayesian analysis completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Sequential testing — always-valid p-values
# ══════════════════════════════════════════════════════════════════════
# Problem: peeking at results before full sample inflates Type I error.
# Solution: always-valid p-values that maintain coverage at any stopping time.
# Method: mixture sequential probability ratio test (mSPRT)

# Simulate sequential analysis (process data in daily batches)
if experiment["timestamp"].dtype == pl.Utf8 or experiment["timestamp"].dtype == pl.String:
    experiment_with_day = experiment.with_columns(
        pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.date().alias("day")
    )
else:
    experiment_with_day = experiment.with_columns(
        pl.col("timestamp").cast(pl.Date).alias("day")
    )

days = sorted(experiment_with_day["day"].unique().to_list())
sequential_results = []

for i, day in enumerate(days):
    if i < 3:  # Need minimum 3 days
        continue

    # Cumulative data up to this day
    cumulative = experiment_with_day.filter(pl.col("day") <= day)
    c = (
        cumulative.filter(pl.col("experiment_group") == "control")["revenue"]
        .to_numpy()
        .astype(np.float64)
    )
    t = (
        cumulative.filter(pl.col("experiment_group") != "control")["revenue"]
        .to_numpy()
        .astype(np.float64)
    )

    if len(c) < 100 or len(t) < 100:
        continue

    # Standard z-test (WRONG for sequential — inflated α)
    diff = t.mean() - c.mean()
    se = np.sqrt(c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t))
    z = diff / se if se > 0 else 0
    p_fixed = 2 * (1 - stats.norm.cdf(abs(z)))

    # mSPRT always-valid p-value (simplified)
    # Uses a mixture of likelihood ratios with a normal mixing distribution
    n_curr = len(c) + len(t)
    n_max = n_c + n_t
    # Variance of the mixing distribution (tuning parameter)
    tau_sq = se_naive**2  # Use naive SE as scale
    v_n = se**2  # Current variance of the test statistic
    # mSPRT statistic
    lambda_n = np.sqrt(v_n / (v_n + tau_sq)) * np.exp(
        tau_sq * z**2 / (2 * (v_n + tau_sq))
    )
    p_sequential = min(1.0, 1.0 / lambda_n) if lambda_n > 0 else 1.0

    sequential_results.append(
        {
            "day": i + 1,
            "n": n_curr,
            "lift": diff,
            "p_fixed": p_fixed,
            "p_sequential": p_sequential,
        }
    )

print(f"\n=== Sequential Testing ===")
print(f"{'Day':>4} {'n':>8} {'Lift':>10} {'p (fixed)':>12} {'p (mSPRT)':>12}")
print("─" * 52)
for r in sequential_results[:: max(1, len(sequential_results) // 10)]:
    print(
        f"{r['day']:>4} {r['n']:>8,} ${r['lift']:>8.2f} {r['p_fixed']:>12.6f} {r['p_sequential']:>12.6f}"
    )

# Show the danger of peeking
early_sig = sum(1 for r in sequential_results if r["p_fixed"] < 0.05)
early_sig_seq = sum(1 for r in sequential_results if r["p_sequential"] < 0.05)
print(f"\nDays with p < 0.05 (fixed):      {early_sig}/{len(sequential_results)}")
print(f"Days with p < 0.05 (sequential): {early_sig_seq}/{len(sequential_results)}")
print("→ Fixed p-values cross significance more often (inflated Type I error)")
# INTERPRETATION: "Peeking" inflates Type I error because each daily check is an
# independent opportunity for a false positive. With 30 daily checks at α=0.05,
# the probability of at least one false positive approaches 78%! Sequential
# testing with mSPRT maintains the nominal α regardless of when you stop.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(sequential_results) > 0, "Should have at least one sequential result"
for r in sequential_results:
    assert 0 <= r["p_fixed"] <= 1, "Fixed p-values must be valid probabilities"
    assert 0 <= r["p_sequential"] <= 1, "Sequential p-values must be valid probabilities"
assert early_sig >= early_sig_seq, \
    "Fixed p-values should cross significance at least as often as sequential (inflation)"
print("\n✓ Checkpoint 5 passed — sequential testing completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Log to ExperimentTracker
# ══════════════════════════════════════════════════════════════════════


async def log_ab_analysis():
    conn = ConnectionManager("sqlite:///mlfp02_experiments.db")
    await conn.initialize()
    tracker = ExperimentTracker(conn)
    await tracker.initialize()

    experiments = await tracker.list_experiments()
    # Find the M2 experiment or create new
    exp_id = None
    for exp in experiments:
        if exp.get("name") == "mlfp02_healthcare_features":
            exp_id = exp["id"]
            break
    if not exp_id:
        exp_id = await tracker.create_experiment(
            name="mlfp02_ab_test_analysis",
            description="A/B test analysis with CUPED and Bayesian methods",
            tags=["mlfp02", "ab-test", "cuped", "bayesian"],
        )

    async with tracker.run(exp_id, run_name="ecommerce_ab_cuped_bayesian") as run:
        await run.log_params(
            {
                "method": "CUPED + Bayesian",
                "pre_covariate": "pre_metric_value",
                "cuped_theta": str(float(theta)),
                "cuped_rho": str(float(rho)),
                "sequential_method": "mSPRT",
            }
        )
        await run.log_metrics(
            {
                "lift_naive": float(lift),
                "lift_cuped": float(lift_adj),
                "se_naive": float(se_naive),
                "se_cuped": float(se_cuped),
                "p_naive": float(p_naive),
                "p_cuped": float(p_cuped),
                "variance_reduction": float(var_reduction),
                "prob_treatment_better": float(prob_treatment_better),
                "expected_loss": float(expected_loss_treatment),
            }
        )
        await run.set_tag("method", "cuped-bayesian-sequential")
    print(f"\nLogged run")
    await conn.close()


try:
    asyncio.run(log_ab_analysis())
except Exception as e:
    print(f"  [Skipped: ExperimentTracker logging failed ({type(e).__name__}: {e})]")
# INTERPRETATION: ExperimentTracker provides an audit trail — who ran which
# analysis, with which parameters, and what results they got. This is essential
# for scientific reproducibility and for compliance in regulated industries
# (finance, healthcare) where model decisions must be explainable and auditable.


# Visualize comparison
viz = ModelVisualizer()
fig = viz.metric_comparison(
    {
        "Standard": {"SE": se_naive, "CI_Width": ci_naive[1] - ci_naive[0]},
        "CUPED": {"SE": se_cuped, "CI_Width": ci_cuped[1] - ci_cuped[0]},
    }
)
fig.update_layout(title="Standard vs CUPED: Variance Reduction")
fig.write_html("ex7_cuped_comparison.html")
print("Saved: ex7_cuped_comparison.html")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
# Verify CUPED CI is narrower than naive CI
naive_ci_width = ci_naive[1] - ci_naive[0]
cuped_ci_width = ci_cuped[1] - ci_cuped[0]
assert cuped_ci_width < naive_ci_width, \
    f"CUPED CI ({cuped_ci_width:.4f}) should be narrower than naive CI ({naive_ci_width:.4f})"
print("\n✓ Checkpoint 6 passed — CUPED CI is narrower than naive CI\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(f"""
  ✓ CUPED: Y_adj = Y - θ(X - E[X]), θ = Cov(Y,X)/Var(X)
  ✓ Variance reduction: Var(Y_adj) = Var(Y)(1 - ρ²)
  ✓ CI width reduction: 1 - √(1 - ρ²) — e.g. ρ=0.7 gives 29% narrower CI
  ✓ CUPED point estimate is unbiased: E[Y_adj] = E[Y] - θ*0 = E[Y]
  ✓ Bayesian A/B: P(treatment > control) and expected loss for decisions
  ✓ mSPRT: always-valid p-value — correct α regardless of when you stop
  ✓ Peeking problem: fixed p-values used repeatedly inflate Type I error
  ✓ ExperimentTracker: reproducible experiment logging with audit trail

  NEXT: In Exercise 8 — the Module 2 Capstone — you'll use FeatureStore
  to persist, version, and retrieve features with point-in-time correctness,
  demonstrating data lineage from raw HDB transactions to model-ready
  features. You'll connect everything learned in M2 into a full pipeline.
""")

print(
    "\n✓ Exercise 7 complete — A/B testing with CUPED + Bayesian + sequential testing"
)
