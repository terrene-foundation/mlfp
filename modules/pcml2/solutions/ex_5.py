# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT2 — Exercise 5: CUPED and Variance Reduction
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Reduce A/B test variance using CUPED and pre-experiment
#   covariates, apply Bayesian A/B testing, and use sequential testing
#   to safely monitor experiments without inflating Type I error.
#
# TASKS:
#   1. Load experiment data with pre-experiment covariates
#   2. SRM check and power analysis (recap from Exercise 3, deeper)
#   3. CUPED variance reduction — derive and apply
#   4. Bayesian A/B testing — posterior probability of improvement
#   5. Sequential testing with always-valid p-values
#   6. Log experiment analysis to ExperimentTracker
#
# THEORY (CUPED):
#   Y_adj = Y - θ(X - E[X])  where θ = Cov(Y,X)/Var(X)
#   Var(Y_adj) = Var(Y)(1 - ρ²)  where ρ = Cor(Y,X)
#   → If ρ=0.5, CI width reduces by 1 - √(1-0.25) = 13.4%
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from scipy import stats
from kailash.db.connection import ConnectionManager
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml import ModelVisualizer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()

# Experiment data with pre-experiment covariates
experiment = loader.load("ascent02", "ecommerce_experiment.parquet")

print("=== E-commerce Experiment Data ===")
print(f"Shape: {experiment.shape}")
print(f"Columns: {experiment.columns}")
print(experiment.head(5))

# Separate groups
control = experiment.filter(pl.col("group") == "control")
treatment = experiment.filter(pl.col("group") == "treatment")

n_c, n_t = control.height, treatment.height
print(f"\nControl: {n_c:,} | Treatment: {n_t:,}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: SRM check (recap, faster this time)
# ══════════════════════════════════════════════════════════════════════

expected = np.array([n_c + n_t] * 2) / 2
observed = np.array([n_c, n_t])
_, srm_p = stats.chisquare(observed, f_exp=expected)
print(f"\nSRM check: p={srm_p:.6f} — {'✓ OK' if srm_p > 0.01 else '⚠ SRM DETECTED'}")


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
x_c = control["pre_revenue"].to_numpy().astype(np.float64)
x_t = treatment["pre_revenue"].to_numpy().astype(np.float64)

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
# = se_lift * φ(-lift_adj/se_lift) - lift_adj * Φ(-lift_adj/se_lift)
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


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Sequential testing — always-valid p-values
# ══════════════════════════════════════════════════════════════════════
# Problem: peeking at results before full sample inflates Type I error.
# Solution: always-valid p-values that maintain coverage at any stopping time.
# Method: mixture sequential probability ratio test (mSPRT)

# Simulate sequential analysis (process data in daily batches)
experiment_with_day = experiment.with_columns(
    pl.col("signup_date").str.to_date("%Y-%m-%d").alias("day")
)

days = sorted(experiment_with_day["day"].unique().to_list())
sequential_results = []

for i, day in enumerate(days):
    if i < 3:  # Need minimum 3 days
        continue

    # Cumulative data up to this day
    cumulative = experiment_with_day.filter(pl.col("day") <= day)
    c = (
        cumulative.filter(pl.col("group") == "control")["revenue"]
        .to_numpy()
        .astype(np.float64)
    )
    t = (
        cumulative.filter(pl.col("group") == "treatment")["revenue"]
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


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Log to ExperimentTracker
# ══════════════════════════════════════════════════════════════════════


async def log_ab_analysis():
    conn = ConnectionManager("sqlite:///ascent02_experiments.db")
    await conn.initialize()
    tracker = ExperimentTracker(conn)
    await tracker.initialize()

    experiments = await tracker.list_experiments()
    # Find the M2 experiment or create new
    exp_id = None
    for exp in experiments:
        if exp.get("name") == "ascent02_healthcare_features":
            exp_id = exp["id"]
            break
    if not exp_id:
        exp_id = await tracker.create_experiment(
            name="ascent02_ab_test_analysis",
            description="A/B test analysis with CUPED and Bayesian methods",
            tags=["ascent02", "ab-test", "cuped", "bayesian"],
        )

    async with tracker.run(exp_id, run_name="ecommerce_ab_cuped_bayesian") as run:
        await run.log_params(
            {
                "method": "CUPED + Bayesian",
                "pre_covariate": "pre_revenue",
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


asyncio.run(log_ab_analysis())


# Visualize comparison
viz = ModelVisualizer()
fig = viz.metric_comparison(
    {
        "Standard": {"SE": se_naive, "CI_Width": ci_naive[1] - ci_naive[0]},
        "CUPED": {"SE": se_cuped, "CI_Width": ci_cuped[1] - ci_cuped[0]},
    }
)
fig.update_layout(title="Standard vs CUPED: Variance Reduction")
fig.write_html("ex5_cuped_comparison.html")
print("Saved: ex5_cuped_comparison.html")

print(
    "\n✓ Exercise 5 complete — A/B testing with CUPED + Bayesian + sequential testing"
)
