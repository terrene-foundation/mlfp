# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 8.4: Regression + Lineage — Full Audit Trail
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an OLS regression model on the v2 feature definition
#   - Compute from-scratch t-statistics and F-tests on coefficients
#   - Apply Normal-Normal Bayesian posteriors to interpret coefficients
#   - Log model parameters, metrics, and lineage via ExperimentTracker
#   - Generate a stakeholder report that synthesises all M2 concepts
#
# PREREQUISITES: Exercise 8.1-8.3 (schemas, PIT, rolling features)
# ESTIMATED TIME: ~50 min
#
# TASKS:
#   1. Theory — why model lineage is essential for production ML
#   2. Build — OLS regression on v2 features with full diagnostics
#   3. Train — hypothesis tests + Bayesian posteriors + ExperimentTracker
#   4. Visualise — actual vs predicted, coefficient forest, residuals
#   5. Apply — model-governance review for a Singapore bank
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from shared.mlfp02.ex_8 import (
    FEATURE_LIST,
    OUTPUT_DIR,
    build_schema_v2,
    compute_v2_features,
    create_tracker,
    fit_ols,
    format_p,
    load_hdb_resale,
    normal_normal_posterior,
    prepare_design_matrix,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Model Lineage Is Essential for Production ML
# ════════════════════════════════════════════════════════════════════════
# A model without lineage is like a financial audit without receipts.
# You know the number on the page, but you can't prove how you got
# there. Model lineage records:
#
#   1. DATA PROVENANCE — which dataset, which version, which time range
#   2. FEATURE VERSION — which schema version produced the features
#   3. HYPERPARAMETERS — which settings were used for training
#   4. METRICS — R², RMSE, F-statistic, individual coefficient p-values
#   5. ARTIFACTS — the model weights, the training script, the config
#
# ExperimentTracker from kailash-ml records the parameters and metrics
# you log for a run (items 1-4 below are logged explicitly in Task 3).
# When a reviewer asks "why did your model value this flat at X?", you
# can trace back from the prediction → model run → feature version →
# raw data → individual transaction records.
#
# Singapore context: the Monetary Authority of Singapore's FEAT
# Principles (Fairness, Ethics, Accountability, Transparency, 2018) are
# guidance for financial institutions using AI and data analytics.
# Reproducible lineage is the evidence a bank needs to show it follows
# them.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: OLS regression on v2 features
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exercise 8.4 — Regression + Lineage: Full Audit Trail")
print("=" * 70)

# --- 2a. Prepare v2 features ---
hdb = load_hdb_resale()
features_v2 = compute_v2_features(hdb)
n_with_market = features_v2.filter(pl.col("town_median_price").is_not_null()).height

print(f"\n  v2 features: {features_v2.shape[0]:,} rows")
print(f"  With market context: {n_with_market:,} rows")

# --- 2b. Fit OLS regression ---
# TODO: Build the design matrix from v2 features using the shared helper.
# Hint: prepare_design_matrix(features_v2) returns (X, y, names) where
# X has a column of ones prepended and names includes "intercept".
X, y, names = ____

# TODO: Fit OLS using the shared helper.
# Hint: fit_ols(X, y) returns a dict with "beta", "se", "t", "p",
# "r2", "adj_r2", "rmse", "f_stat", "f_p", "y_hat", "resid", etc.
ols = ____

print(f"\n  === Regression Model on v2 Features ===")
print(f"  n = {ols['n']:,}, k = {ols['k']}")
print(f"  R-squared = {ols['r2']:.6f} ({ols['r2']:.2%} variance explained)")
print(f"  Adj R-squared = {ols['adj_r2']:.6f}")
print(f"  RMSE = ${ols['rmse']:,.0f}")

print(f"\n  {'Feature':<25} {'Coefficient':>14}")
print("  " + "-" * 42)
for name, coef in zip(names, ols["beta"]):
    print(f"  {name:<25} {coef:>14,.2f}")


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert (
    ols["r2"] > 0.2
), f"Task 2: R-squared should be positive and non-trivial, got {ols['r2']:.4f}"
assert ols["rmse"] > 0, "Task 2: RMSE must be positive"
print("\n[ok] Checkpoint 1 passed — regression model built on v2 features\n")

# INTERPRETATION: The R² tells us what fraction of price variation is
# explained by our 5 features. The remaining (1 - R²) is unexplained
# variance — renovation quality, unit facing, floor plan, negotiation
# skill, and luck. You added flat-type dummies in Exercise 5.4; whether
# they help here is an empirical question — compare adjusted R².


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Hypothesis tests, Bayesian posteriors, and tracking
# ════════════════════════════════════════════════════════════════════════

# --- 3a. Coefficient significance tests ---
print("--- Coefficient Significance (from-scratch t-statistics) ---")
print(
    f"\n  {'Feature':<25} {'beta':>12} {'SE':>10} {'t':>8} "
    f"{'p-value':>12} {'Sig':>4}"
)
print("  " + "-" * 75)
for i, name in enumerate(names):
    p = ols["p"][i]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(
        f"  {name:<25} {ols['beta'][i]:>12,.2f} {ols['se'][i]:>10,.2f} "
        f"{ols['t'][i]:>8.2f} {format_p(p):>12} {sig:>4}"
    )

print(f"\n  F-statistic: {ols['f_stat']:.2f} (p: {format_p(ols['f_p'])})")
print(
    f"  Model is "
    f"{'significantly better' if ols['f_p'] < 0.05 else 'NOT better'} "
    f"than mean-only"
)

# --- 3b. Bayesian posteriors for each coefficient ---
print(f"\n--- Bayesian Posteriors (Normal-Normal Conjugate) ---")
posteriors = {}
for i, name in enumerate(names):
    if i == 0:
        continue  # Skip intercept

    # TODO: Compute the Normal-Normal posterior for this coefficient.
    # Hint: normal_normal_posterior(beta_hat, se_hat) returns a dict with
    # "mu_post", "sigma_post", "ci_low", "ci_high".
    post = ____
    p_positive = 1 - sp_stats.norm.cdf(0, post["mu_post"], post["sigma_post"])
    posteriors[name] = {**post, "p_positive": p_positive}

    print(f"\n  {name}:")
    print(f"    OLS: beta={ols['beta'][i]:,.2f} +/- {ols['se'][i]:,.2f}")
    print(f"    Posterior: N({post['mu_post']:,.2f}, {post['sigma_post']:,.2f})")
    print(f"    95% credible: [{post['ci_low']:,.2f}, {post['ci_high']:,.2f}]")
    print(f"    P(beta > 0): {p_positive:.4f}")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert all(se > 0 for se in ols["se"]), "Task 3: standard errors must be positive"
assert len(posteriors) == len(
    FEATURE_LIST
), "Task 3: must have posteriors for all features"
print(
    "\n[ok] Checkpoint 2 passed — hypothesis tests and Bayesian posteriors complete\n"
)

# --- 3c. Log to ExperimentTracker ---
print("--- ExperimentTracker Lineage ---")

schema_v2 = build_schema_v2()


async def log_lineage():
    tracker = await create_tracker()
    exp_id = "mlfp02_capstone_model"
    async with tracker.track(experiment=exp_id, run_name="hdb_price_ols_v2") as run:
        await run.log_params(
            {
                "data_source": "mlfp01/hdb_resale.parquet (validated rows)",
                "feature_schema": schema_v2.name,
                "feature_version": str(schema_v2.version),
                "features": ",".join(FEATURE_LIST),
                "model_type": "OLS",
                "n_observations": str(ols["n"]),
            }
        )
        # TODO: Complete the metrics dict — log R-squared as a plain float.
        # Hint: the fit dict stores it under "r2"; wrap it in float(...)
        await run.log_metrics(
            {
                "r2": ____,
                "adj_r2": float(ols["adj_r2"]),
                "rmse": float(ols["rmse"]),
                "f_statistic": float(ols["f_stat"]),
            }
        )
        run_id = run.run_id
    await tracker.close()
    return exp_id, run_id


exp_id, run_id = asyncio.run(log_lineage())

print(f"\n  Experiment logged:")
print(f"    Experiment: {exp_id}, Run ID: {run_id}")
print(f"    Feature schema: {schema_v2.name} v{schema_v2.version}")
print(f"    Features: {FEATURE_LIST}")
print(f"    Training rows: {ols['n']:,}")
print(f"    R-squared: {ols['r2']:.4f}, RMSE: ${ols['rmse']:,.0f}")


# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert run_id, "Task 3: the tracker must return a run id"
print("\n[ok] Checkpoint 3 passed — model lineage logged\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Actual vs predicted, coefficients, residuals
# ════════════════════════════════════════════════════════════════════════

print("--- Model Diagnostics Visualisations ---")

# Plot 1: Actual vs Predicted
rng = np.random.default_rng(42)
n_sample = min(3000, ols["n"])
idx = rng.choice(ols["n"], size=n_sample, replace=False)

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=____,  # Hint: actual prices for the sampled rows (as a list)
        y=____,  # Hint: fitted values for the same rows
        mode="markers",
        marker={"size": 3, "opacity": 0.4, "color": "steelblue"},
        name="Predictions",
    )
)
fig1.add_trace(
    go.Scatter(
        x=[float(y.min()), float(y.max())],
        y=[float(y.min()), float(y.max())],
        mode="lines",
        name="Perfect",
        line={"dash": "dash", "color": "red"},
    )
)
fig1.update_layout(
    title=f"Capstone Model: Actual vs Predicted (R-squared={ols['r2']:.4f})",
    xaxis_title="Actual ($)",
    yaxis_title="Predicted ($)",
)
fig1.write_html(str(OUTPUT_DIR / "04_actual_vs_predicted.html"))
print(f"\n  Saved: {OUTPUT_DIR / '04_actual_vs_predicted.html'}")

# Plot 2: Coefficient forest plot
fig2 = go.Figure()
for i in range(1, ols["k"]):
    ci_lo = ols["beta"][i] - 1.96 * ols["se"][i]
    ci_hi = ols["beta"][i] + 1.96 * ols["se"][i]
    fig2.add_trace(
        go.Scatter(
            x=[ci_lo, ols["beta"][i], ci_hi],
            y=[names[i]] * 3,
            mode="markers+lines",
            name=names[i],
            marker={"size": [6, 10, 6]},
        )
    )
fig2.add_vline(x=0, line_dash="dot", line_color="red")
fig2.update_layout(
    title="Regression Coefficients with 95% CIs",
    xaxis_title="Coefficient Value",
)
fig2.write_html(str(OUTPUT_DIR / "04_coefficient_forest.html"))
print(f"  Saved: {OUTPUT_DIR / '04_coefficient_forest.html'}")

# Plot 3: Residual distribution
fig3 = go.Figure()
fig3.add_trace(
    go.Histogram(
        x=ols["resid"][idx].tolist(),
        nbinsx=50,
        marker_color="steelblue",
        opacity=0.7,
    )
)
fig3.update_layout(
    title="Residual Distribution",
    xaxis_title="Residual ($)",
    yaxis_title="Count",
)
fig3.write_html(str(OUTPUT_DIR / "04_residuals.html"))
print(f"  Saved: {OUTPUT_DIR / '04_residuals.html'}")

# Plot 4: Bayesian posterior densities
fig4 = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=list(posteriors.keys()),
)
for idx_p, (name, post) in enumerate(posteriors.items()):
    row = idx_p // 3 + 1
    col = idx_p % 3 + 1
    x_range = np.linspace(
        post["ci_low"] - 2 * post["sigma_post"],
        post["ci_high"] + 2 * post["sigma_post"],
        200,
    )
    # TODO: Normal density of the posterior over x_range.
    pdf_vals = ____
    fig4.add_trace(
        go.Scatter(x=x_range.tolist(), y=pdf_vals.tolist(), name=name, mode="lines"),
        row=row,
        col=col,
    )
    # Plotly's add_vline stub types row/col as str, but int indices work at runtime.
    fig4.add_vline(x=0, line_dash="dot", line_color="red", row=row, col=col)  # type: ignore[arg-type]

fig4.update_layout(title="Bayesian Posterior Densities for Coefficients", height=600)
fig4.write_html(str(OUTPUT_DIR / "04_bayesian_posteriors.html"))
print(f"  Saved: {OUTPUT_DIR / '04_bayesian_posteriors.html'}")


# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 4 passed — all diagnostic visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Model-Governance Review for a Singapore Bank
# ════════════════════════════════════════════════════════════════════════
# Scenario (illustrative): a Singapore bank's model-risk team reviews its
# HDB valuation model against the FEAT Principles:
#   - Fairness: no protected-attribute features
#   - Ethics: data sourced appropriately (public HDB resale records)
#   - Accountability: a traceable path from data to prediction
#   - Transparency: an interpretable model (OLS coefficients)
#
# Without lineage: the team is shown a slide with a single R² and cannot
# tell which data, features or parameters produced it, so the review
# records a governance gap to remediate.
#
# With the tracker run above: data source → feature schema v2 → OLS on
# 5 features → R², RMSE → coefficients with p-values and CIs, all tied
# to one run id. The review can reproduce the number.

print("=== APPLY: Model-Governance Review (Singapore bank, illustrative) ===")
print()
print("  Scenario: FEAT-aligned review of an HDB valuation model")
print()

# Stakeholder report
print("  " + "=" * 66)
print("  STAKEHOLDER REPORT: HDB Resale Price Analysis")
print("  " + "=" * 66)
print(
    f"""
  EXECUTIVE SUMMARY
  This analysis applies statistical methods from Module 2 to Singapore's
  HDB resale market, covering {features_v2.height:,} transactions.

  KEY FINDINGS:

  1. PRICE DISTRIBUTION (Ex 1-2: Bayesian + MLE)
     Average price: ${y.mean():,.0f} +/- ${y.std():,.0f}
     Skewness indicates {'right-skewed' if sp_stats.skew(y) > 0.5 else 'approximately symmetric'} distribution

  2. PRICE DRIVERS (Ex 5: Linear Regression)
     Our v2 model with {len(FEATURE_LIST)} features explains {ols['r2']:.1%} of
     price variation. Significant drivers:"""
)
for i in range(1, ols["k"]):
    if ols["p"][i] < 0.05:
        print(
            f"     - {names[i]}: ${ols['beta'][i]:+,.0f} per unit "
            f"(p: {format_p(ols['p'][i])})"
        )
print(
    f"""
  3. MARKET CONTEXT (Ex 8: Feature Engineering)
     Rolling 6-month town medians and volumes capture local market.
     {n_with_market:,} of {features_v2.height:,} transactions have context.

  4. DATA QUALITY
     Value rules removed impossible leases and sentinel prices (8.1).
     Market features use only the 6 months BEFORE each sale (8.3).
     This model is fit and scored in-sample; an out-of-time holdout
     (as in 8.2) is still required before quoting production accuracy.
     Feature versioning (v1 -> v2) tracks schema evolution.

  5. MODEL LINEAGE
     ExperimentTracker run {run_id} records: data source, feature
     schema and version, model params, metrics.

  6. MODEL LIMITATIONS
     - {(1-ols['r2'])*100:.0f}% of variance unexplained
     - Linear model may miss non-linear relationships
     - Market features have a 6-7 month warm-up period (nulls)

  RECOMMENDATIONS:
     - Use v2 features for all new valuation models
     - Test flat-type encoding (Exercise 5.4) and compare adjusted R-squared
     - Consider non-linear models (Random Forest, XGBoost) in M3
     - Monitor town-level trends for early price-shift signals
"""
)

print("  FEAT evidence summary:")
print("    Fairness:       Public HDB data, no protected-attribute features")
print("    Ethics:         Public HDB resale records (no personal data)")
print(f"    Accountability: Tracker run {run_id} — {len(FEATURE_LIST)} features, ")
print(f"                    {ols['n']:,} rows, R-squared={ols['r2']:.4f}")
print("    Transparency:   OLS coefficients with CIs and p-values")
print()
print("  FEAT is guidance, not a capital rule: the cost of a governance gap")
print("  is remediation work and delayed model approval, not a fixed buffer.")


# ── Checkpoint 5 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 5 passed — stakeholder report and governance review complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [ok] OLS regression on FeatureStore v2 features
  [ok] From-scratch t-statistics and F-tests on coefficients
  [ok] Bayesian Normal-Normal posteriors for coefficient interpretation
  [ok] ExperimentTracker: parameters, metrics, and lineage logging
  [ok] Stakeholder reporting: translating statistics into business decisions
  [ok] FEAT-aligned evidence: fairness, ethics, accountability, transparency

  MODULE 2 COMPLETE — YOUR STATISTICAL TOOLKIT:
  ==============================================
  Ex 1: Bayesian inference — conjugate priors, credible intervals
  Ex 2: MLE + MAP — optimisation, CLT, failure modes, AIC/BIC
  Ex 3: Hypothesis testing — bootstrap, power, BH-FDR, permutation
  Ex 4: A/B design — pre-registration, SRM, adaptive sample sizes
  Ex 5: Linear regression — OLS from scratch, VIF, WLS, diagnostics
  Ex 6: Logistic regression — sigmoid MLE, odds ratios, calibration
  Ex 7: CUPED + causal inference — variance reduction, DiD, mSPRT
  Ex 8: Capstone — feature store, lineage, complete pipeline

  -> NEXT MODULE: M3 — Supervised ML in the Kailash Pipeline
     You'll use TrainingPipeline, HyperparameterSearch, and ModelRegistry
     to build, tune, and deploy models at production scale.
"""
)
