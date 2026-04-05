# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT2 — Exercise 4: Causal Inference — Difference-in-Differences
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Estimate the causal effect of Singapore's property cooling
#   measures on HDB resale prices using difference-in-differences (DiD).
#   Track the analysis as an experiment run.
#
# TASKS:
#   1. Load HDB prices + cooling measure dates
#   2. Define treatment and control groups (affected vs unaffected towns)
#   3. Verify the parallel trends assumption (pre-treatment)
#   4. Estimate DiD treatment effect
#   5. Robustness checks (placebo test, event study)
#   6. Log causal inference results to ExperimentTracker
#
# THEORY (DiD):
#   ATT = E[Y(1) - Y(0) | D=1]
#       = (E[Y|D=1,T=1] - E[Y|D=1,T=0]) - (E[Y|D=0,T=1] - E[Y|D=0,T=0])
#   Key assumption: parallel trends (absent treatment, groups would have
#   moved in parallel). Testable pre-treatment, untestable post.
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
hdb = loader.load("ascent01", "hdb_resale.parquet")
cooling_measures = loader.load("ascent02", "sg_cooling_measures.csv")

# Parse dates
hdb = hdb.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date")
)

print("=== Cooling Measures ===")
print(cooling_measures)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define the natural experiment
# ══════════════════════════════════════════════════════════════════════
# Cooling measure: Additional Buyer's Stamp Duty (ABSD) increase
# Treatment: towns with high proportion of investment purchases
# Control: towns with predominantly owner-occupied purchases
# The ABSD disproportionately affects investment-heavy towns.

# Select a specific cooling measure event
event_date = pl.date(2022, 9, 30)  # Sep 2022 ABSD increase

# Treatment towns (high investment proportion — CBD-adjacent, premium)
treatment_towns = ["BUKIT TIMAH", "QUEENSTOWN", "BISHAN", "TOA PAYOH", "CENTRAL AREA"]
# Control towns (predominantly owner-occupied, suburban)
control_towns = ["WOODLANDS", "JURONG WEST", "YISHUN", "PUNGGOL", "SENGKANG"]

# Window: 12 months before and after
window_start = event_date - pl.duration(days=365)
window_end = event_date + pl.duration(days=365)

# Filter and tag
did_data = hdb.filter(
    (pl.col("transaction_date") >= window_start)
    & (pl.col("transaction_date") <= window_end)
    & (pl.col("town").is_in(treatment_towns + control_towns))
).with_columns(
    pl.when(pl.col("town").is_in(treatment_towns))
    .then(pl.lit(1))
    .otherwise(pl.lit(0))
    .alias("treated"),
    pl.when(pl.col("transaction_date") >= event_date)
    .then(pl.lit(1))
    .otherwise(pl.lit(0))
    .alias("post"),
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
)

# Interaction term for DiD
did_data = did_data.with_columns(
    (pl.col("treated") * pl.col("post")).alias("did_interaction")
)

print(f"\nDiD sample: {did_data.height:,} transactions")
print(f"Treatment (high-investment towns): {did_data.filter(pl.col('treated') == 1).height:,}")
print(f"Control (owner-occupied towns):    {did_data.filter(pl.col('treated') == 0).height:,}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Verify parallel trends assumption
# ══════════════════════════════════════════════════════════════════════

# Monthly price trends by group (pre-treatment period only)
pre_treatment = did_data.filter(pl.col("post") == 0)

monthly_trends = pre_treatment.group_by(
    "transaction_date", "treated"
).agg(
    pl.col("price_per_sqm").median().alias("median_price_sqm"),
    pl.col("price_per_sqm").count().alias("n_transactions"),
).sort("transaction_date", "treated")

# Compute growth rates for each group
for group in [0, 1]:
    group_data = monthly_trends.filter(pl.col("treated") == group).sort("transaction_date")
    prices = group_data["median_price_sqm"].to_numpy()
    if len(prices) > 1:
        growth = (prices[-1] - prices[0]) / prices[0] * 100
        monthly_growth = growth / len(prices)
        group_name = "Treatment" if group == 1 else "Control"
        print(f"\n{group_name} pre-trend: {growth:.1f}% total, {monthly_growth:.2f}%/month")

# Formal test: interaction of group × time should be non-significant pre-treatment
# Using monthly dummies
pre_months = pre_treatment.with_columns(
    pl.col("transaction_date").dt.month().alias("month_num")
)

# Simple linear trend comparison
from numpy.polynomial import polynomial as P

for group, name in [(0, "Control"), (1, "Treatment")]:
    group_monthly = (
        pre_months.filter(pl.col("treated") == group)
        .group_by("transaction_date")
        .agg(pl.col("price_per_sqm").median())
        .sort("transaction_date")
    )
    x = np.arange(group_monthly.height)
    y = group_monthly["price_per_sqm"].to_numpy()
    coeffs = P.polyfit(x, y, deg=1)
    print(f"{name} linear trend: slope = {coeffs[1]:.2f} $/sqm per month")

print("\n→ Parallel trends: visually inspect and compare slopes above.")
print("  If slopes are similar, the assumption is plausible.")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Estimate DiD treatment effect
# ══════════════════════════════════════════════════════════════════════

# Simple 2×2 DiD (means comparison)
means = {}
for treated in [0, 1]:
    for post in [0, 1]:
        subset = did_data.filter(
            (pl.col("treated") == treated) & (pl.col("post") == post)
        )
        means[(treated, post)] = subset["price_per_sqm"].mean()

did_estimate = (
    (means[(1, 1)] - means[(1, 0)])  # Treatment group change
    - (means[(0, 1)] - means[(0, 0)])  # Control group change
)

print(f"\n=== DiD Estimation ===")
print(f"Treatment group: pre=${means[(1,0)]:,.0f} → post=${means[(1,1)]:,.0f} (Δ={means[(1,1)]-means[(1,0)]:+,.0f})")
print(f"Control group:   pre=${means[(0,0)]:,.0f} → post=${means[(0,1)]:,.0f} (Δ={means[(0,1)]-means[(0,0)]:+,.0f})")
print(f"DiD estimate (ATT): ${did_estimate:+,.0f} per sqm")

# Regression-based DiD with standard errors
# Y = β₀ + β₁(treated) + β₂(post) + β₃(treated×post) + ε
# β₃ is the DiD estimate
from numpy.linalg import lstsq

y = did_data["price_per_sqm"].to_numpy()
X = np.column_stack([
    np.ones(len(y)),
    did_data["treated"].to_numpy(),
    did_data["post"].to_numpy(),
    did_data["did_interaction"].to_numpy(),
])

beta, residuals, rank, sv = lstsq(X, y, rcond=None)
y_hat = X @ beta
resid = y - y_hat
n = len(y)
k = X.shape[1]
mse = np.sum(resid**2) / (n - k)
var_beta = mse * np.linalg.inv(X.T @ X)
se_beta = np.sqrt(np.diag(var_beta))

# β₃ is the DiD coefficient
did_coef = beta[3]
did_se = se_beta[3]
did_t = did_coef / did_se
did_p = 2 * (1 - stats.t.cdf(abs(did_t), df=n - k))
did_ci = (did_coef - 1.96 * did_se, did_coef + 1.96 * did_se)

print(f"\n=== Regression DiD ===")
print(f"DiD coefficient (β₃): ${did_coef:+,.2f}")
print(f"Standard error: ${did_se:,.2f}")
print(f"t-statistic: {did_t:.3f}")
print(f"p-value: {did_p:.6f}")
print(f"95% CI: [${did_ci[0]:,.2f}, ${did_ci[1]:,.2f}]")
print(f"Interpretation: Cooling measure {'reduced' if did_coef < 0 else 'increased'} "
      f"prices by ${abs(did_coef):,.0f}/sqm in investment-heavy towns")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Robustness — placebo test
# ══════════════════════════════════════════════════════════════════════
# If our DiD is valid, running it at a FAKE event date (before actual event)
# should show NO effect.

placebo_date = event_date - pl.duration(days=365)  # 1 year before actual event
placebo_window_start = placebo_date - pl.duration(days=365)

placebo_data = hdb.filter(
    (pl.col("transaction_date") >= placebo_window_start)
    & (pl.col("transaction_date") <= placebo_date + pl.duration(days=365))
    & (pl.col("town").is_in(treatment_towns + control_towns))
    & (pl.col("transaction_date") < event_date)  # Strictly before actual event
).with_columns(
    pl.when(pl.col("town").is_in(treatment_towns)).then(pl.lit(1)).otherwise(pl.lit(0)).alias("treated"),
    pl.when(pl.col("transaction_date") >= placebo_date).then(pl.lit(1)).otherwise(pl.lit(0)).alias("post"),
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
).with_columns(
    (pl.col("treated") * pl.col("post")).alias("did_interaction")
)

# Placebo regression
y_p = placebo_data["price_per_sqm"].to_numpy()
X_p = np.column_stack([
    np.ones(len(y_p)),
    placebo_data["treated"].to_numpy(),
    placebo_data["post"].to_numpy(),
    placebo_data["did_interaction"].to_numpy(),
])

beta_p, _, _, _ = lstsq(X_p, y_p, rcond=None)
resid_p = y_p - X_p @ beta_p
mse_p = np.sum(resid_p**2) / (len(y_p) - 4)
se_p = np.sqrt(mse_p * np.linalg.inv(X_p.T @ X_p)[3, 3])
t_p = beta_p[3] / se_p
p_placebo = 2 * (1 - stats.t.cdf(abs(t_p), df=len(y_p) - 4))

print(f"\n=== Placebo Test (fake event 1 year earlier) ===")
print(f"Placebo DiD: ${beta_p[3]:+,.2f} (SE=${se_p:,.2f})")
print(f"p-value: {p_placebo:.4f}")
if p_placebo > 0.05:
    print("✓ Placebo is non-significant — supports causal interpretation")
else:
    print("⚠ Placebo is significant — parallel trends assumption may be violated")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Event study — dynamic treatment effects
# ══════════════════════════════════════════════════════════════════════
# Estimate treatment effect for each month relative to the event
# This shows both pre-trends (should be flat) and post-treatment dynamics

# Create relative month variable
did_data_event = did_data.with_columns(
    ((pl.col("transaction_date") - event_date).dt.total_days() / 30).cast(pl.Int32).alias("rel_month")
)

# Estimate treatment effect for each relative month
event_study = []
for rel_m in range(-12, 13):
    month_data = did_data_event.filter(pl.col("rel_month") == rel_m)
    if month_data.height < 50:
        continue

    treat_prices = month_data.filter(pl.col("treated") == 1)["price_per_sqm"].to_numpy()
    ctrl_prices = month_data.filter(pl.col("treated") == 0)["price_per_sqm"].to_numpy()

    if len(treat_prices) > 10 and len(ctrl_prices) > 10:
        diff = treat_prices.mean() - ctrl_prices.mean()
        se = np.sqrt(treat_prices.var(ddof=1) / len(treat_prices) + ctrl_prices.var(ddof=1) / len(ctrl_prices))
        event_study.append({
            "rel_month": rel_m,
            "diff": diff,
            "se": se,
            "ci_lower": diff - 1.96 * se,
            "ci_upper": diff + 1.96 * se,
        })

print(f"\n=== Event Study ===")
print(f"{'Month':>6} {'Diff':>10} {'95% CI':>24}")
print("─" * 44)
for e in event_study:
    marker = " ←event" if e["rel_month"] == 0 else ""
    print(f"  t{e['rel_month']:+3d}  ${e['diff']:>8,.0f}  [${e['ci_lower']:>8,.0f}, ${e['ci_upper']:>8,.0f}]{marker}")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Log to ExperimentTracker
# ══════════════════════════════════════════════════════════════════════

async def log_did_analysis():
    conn = ConnectionManager("sqlite:///ascent02_experiments.db")
    await conn.initialize()
    tracker = ExperimentTracker(conn)
    await tracker.initialize()

    exp_id = await tracker.create_experiment(
        name="ascent02_causal_inference",
        description="Causal inference on Singapore housing cooling measures",
        tags=["ascent02", "causal", "did"],
    )

    run_id = await tracker.log_run(
        experiment_id=exp_id,
        name="did_cooling_measure_absd_2022",
        params={
            "method": "Difference-in-Differences",
            "event_date": "2022-09-30",
            "treatment_towns": treatment_towns,
            "control_towns": control_towns,
            "window_months": 12,
        },
        metrics={
            "did_estimate": float(did_coef),
            "did_se": float(did_se),
            "did_p_value": float(did_p),
            "placebo_p_value": float(p_placebo),
            "n_treatment": int(did_data.filter(pl.col("treated") == 1).height),
            "n_control": int(did_data.filter(pl.col("treated") == 0).height),
        },
        tags=["causal", "did", "cooling-measures"],
    )
    print(f"\nLogged DiD run: {run_id}")
    await conn.close()

asyncio.run(log_did_analysis())

# Visualise
viz = ModelVisualizer()
fig = viz.metric_comparison({
    "Treatment (pre)": {"price_sqm": means[(1, 0)]},
    "Treatment (post)": {"price_sqm": means[(1, 1)]},
    "Control (pre)": {"price_sqm": means[(0, 0)]},
    "Control (post)": {"price_sqm": means[(0, 1)]},
})
fig.update_layout(title="DiD: Treatment vs Control, Pre vs Post Cooling Measure")
fig.write_html("ex4_did_comparison.html")
print("Saved: ex4_did_comparison.html")

print("\n✓ Exercise 4 complete — causal inference with difference-in-differences")
