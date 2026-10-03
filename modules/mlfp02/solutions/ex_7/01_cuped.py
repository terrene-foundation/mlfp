# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 7.1: CUPED Variance Reduction
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Derive and implement CUPED using pre-experiment covariates
#   - Quantify variance reduction from the pre-post correlation (rho^2)
#   - Extend CUPED to multiple covariates via multivariate regression
#   - Apply stratified CUPED to detect heterogeneous treatment effects
#   - Log results to ExperimentTracker for reproducibility
#
# PREREQUISITES: Exercises 3-4 — hypothesis testing, p-values, SRM
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load experiment data, SRM check against the designed allocation
#   2. Standard A/B baseline (no CUPED) — control vs treatment_a
#   3. Single-covariate CUPED: derive theta, adjust Y, verify reduction
#   4. Multi-covariate CUPED: multivariate regression
#   5. Stratified CUPED: segment-level treatment effects
#   6. Visualise and log results
#
# THEORY (CUPED):
#   Y_adj = Y - theta*(X - E[X])  where theta = Cov(Y,X)/Var(X)
#   Var(Y_adj) = Var(Y)(1 - rho^2) where rho = Cor(Y,X)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import plotly.graph_objects as go
import polars as pl
from kailash_ml import ExperimentTracker

from shared.mlfp02.ex_7 import (
    ANALYSIS_ARM,
    DESIGNED_ALLOCATION,
    OUTPUT_DIR,
    compute_srm,
    get_covariate_arrays,
    get_revenue_arrays,
    load_experiment,
    multi_cov_cuped,
    naive_ab,
    print_banner,
    single_cov_cuped,
    split_groups,
    srm_allocation_check,
    stratified_cuped,
    stratify_by_covariate,
    variance_reduction,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Variance Reduction Matters for A/B Tests
# ════════════════════════════════════════════════════════════════════════
# An A/B test's power depends on the standard error of the treatment
# effect estimate. Smaller SE = narrower CI = faster decisions.
#
# CUPED (Controlled-experiment Using Pre-Experiment Data) exploits the
# correlation between pre-experiment behaviour (X) and the outcome (Y).
# If a user spent $100 before the experiment, they will probably spend
# around $100 during — this predictable portion is noise we can remove.
#
# Y_adj = Y - theta*(X - E[X])
# theta = Cov(Y, X) / Var(X)  — the optimal noise-removal coefficient
# Var(Y_adj) = Var(Y) * (1 - rho^2)
#
# With rho = 0.7, CUPED removes rho^2 = 49% of variance (51% remains) —
# equivalent to roughly doubling your sample size (1/0.51 = 1.96x).
# The gain depends entirely on rho: with rho = 0.2 you remove only 4%.
#
# The covariate X MUST be measured before assignment. A metric measured
# during the experiment is itself moved by the treatment, so "adjusting"
# for it subtracts part of the treatment effect (Task 4 shows this).
#
# WHY THIS MATTERS: Published CUPED results (Deng et al., 2013) report
# that pre-period metrics can cut the variance of engagement metrics by
# about half — which translates directly into shorter experiments.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and SRM Check
# ════════════════════════════════════════════════════════════════════════

print_banner("MLFP02 Exercise 7.1: CUPED Variance Reduction")

experiment = load_experiment()
print(f"\n  Data loaded: experiment_data.parquet")
print(f"  Shape: {experiment.shape}")
print(f"  Columns: {experiment.columns}")
print(experiment.head(5))

# SRM check 1 — all four arms against the DESIGNED allocation (40/35/15/10)
srm_all = srm_allocation_check(experiment, DESIGNED_ALLOCATION)
print(f"\n=== SRM check: all arms vs designed allocation ===")
print(f"{'Arm':<13} {'Observed':>9} {'Expected':>10} {'Share':>7} {'Design':>7} {'Resid':>7}")
for a in srm_all["arms"]:
    print(
        f"{a['arm']:<13} {a['observed']:>9,} {a['expected']:>10,.0f} "
        f"{a['observed_share']:>7.1%} {a['designed_share']:>7.0%} {a['std_residual']:>+7.1f}"
    )
print(f"chi2 = {srm_all['chi2']:,.1f}, p = {srm_all['p_value']:.3g}")
worst_arm = max(srm_all["arms"], key=lambda a: abs(a["std_residual"]))["arm"]
print(f"SRM DETECTED — largest deviation in arm: {worst_arm}")
# INTERPRETATION: The four-arm split fails badly. The residuals localise
# the fault: one arm received far more traffic than designed, and every
# other arm is short by roughly the same proportion. Its results cannot be
# trusted, and the other arms must each be checked before analysis.

# SRM check 2 — the pair we will analyse, against ITS designed ratio
control, treatment = split_groups(experiment, ANALYSIS_ARM)
n_c, n_t = control.height, treatment.height
print(f"\nControl: {n_c:,} | {ANALYSIS_ARM}: {n_t:,}")
srm_p = compute_srm(
    n_c, n_t, DESIGNED_ALLOCATION["control"], DESIGNED_ALLOCATION[ANALYSIS_ARM]
)
print(f"Pairwise SRM (designed 40:35): p={srm_p:.4f}")
if srm_p < 0.01:
    raise RuntimeError(
        f"SRM on control vs {ANALYSIS_ARM} (p={srm_p:.2g}) — stop and investigate "
        "the assignment pipeline before analysing this comparison."
    )
print(f"OK — control vs {ANALYSIS_ARM} matches its designed ratio; safe to analyse.")

# Explore pre-experiment covariates
for col in ["revenue", "pre_metric_value", "metric_value"]:
    if col in experiment.columns:
        vals = experiment[col].drop_nulls()
        print(f"  {col}: mean={vals.mean():.2f}, std={vals.std():.2f}, n={vals.len()}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert srm_all["p_value"] < 0.01, "The four-arm design should show SRM"
assert worst_arm == "variant_c", "Residuals should localise the faulty arm"
assert srm_p >= 0.01, "The analysed pair must pass its own SRM check"
print("\n>>> Checkpoint 1 passed -- SRM check and data exploration completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Standard Analysis Baseline (No CUPED)
# ════════════════════════════════════════════════════════════════════════

y_c, y_t = get_revenue_arrays(control, treatment)
baseline = naive_ab(y_c, y_t)

print(f"\n=== Standard Analysis (no CUPED) ===")
print(f"Control mean: ${baseline['mean_c']:.2f}")
print(f"Treatment mean: ${baseline['mean_t']:.2f}")
print(
    f"Lift: ${baseline['lift']:.2f} ({baseline['lift'] / baseline['mean_c']:.2%} relative)"
)
print(f"SE: ${baseline['se']:.2f}")
print(f"95% CI: [${baseline['ci_lo']:.2f}, ${baseline['ci_hi']:.2f}]")
print(f"CI width: ${baseline['ci_hi'] - baseline['ci_lo']:.2f}")
print(f"p-value: {baseline['p_value']:.6f}")
# INTERPRETATION: The naive analysis uses only experiment-period data.
# It ignores that some users are naturally high-spenders — CUPED
# removes this baseline noise by leveraging pre-experiment data.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert baseline["se"] > 0, "SE must be positive"
assert baseline["ci_lo"] < baseline["ci_hi"], "CI lower must be below upper"
print("\n>>> Checkpoint 2 passed -- standard analysis baseline established\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Single-Covariate CUPED
# ════════════════════════════════════════════════════════════════════════
# CUPED: Y_adj = Y - theta*(X - E[X])
# theta = Cov(Y, X) / Var(X) — the optimal coefficient
# Var(Y_adj) = Var(Y)(1 - rho^2) where rho = Cor(Y, X)

x_c, x_t = get_covariate_arrays(control, treatment)

# Pool both arms to estimate theta and E[X] (one theta for everyone, so the
# adjustment cannot differ by arm and cannot bias the lift).
x_all = np.concatenate([x_c, x_t])
y_all = np.concatenate([y_c, y_t])
theta = np.cov(y_all, x_all, ddof=1)[0, 1] / np.var(x_all, ddof=1)
rho = np.corrcoef(y_all, x_all)[0, 1]
x_mean = x_all.mean()

y_c_adj = y_c - theta * (x_c - x_mean)
y_t_adj = y_t - theta * (x_t - x_mean)

# Same Welch summary as the baseline, now on the adjusted outcomes
cuped = naive_ab(y_c_adj, y_t_adj)
cuped.update(
    {
        "theta": float(theta),
        "rho": float(rho),
        "y_c_adj": y_c_adj,
        "y_t_adj": y_t_adj,
        "theoretical_reduction": float(rho**2),
    }
)
vr = variance_reduction(baseline["se"], cuped["se"])

print(f"\n=== Single-Covariate CUPED ===")
print(f"Pre-post correlation (rho): {cuped['rho']:.4f}")
print(f"theta (optimal coefficient): {cuped['theta']:.4f}")
print(f"Theoretical variance reduction: {cuped['theoretical_reduction']:.1%}")
print(f"Actual variance reduction: {vr['variance_reduction']:.1%}")
print(f"CI width reduction: {vr['ci_width_reduction']:.1%}")
print(f"\nCUPED lift: ${cuped['lift']:.2f}")
print(f"SE (naive): ${baseline['se']:.2f} -> SE (CUPED): ${cuped['se']:.2f}")
ci_w_naive = baseline["ci_hi"] - baseline["ci_lo"]
ci_w_cuped = cuped["ci_hi"] - cuped["ci_lo"]
print(f"CI width (naive): ${ci_w_naive:.2f} -> CI width (CUPED): ${ci_w_cuped:.2f}")
print(f"95% CI: [${cuped['ci_lo']:.2f}, ${cuped['ci_hi']:.2f}]")
print(f"p-value: {cuped['p_value']:.6f} (was {baseline['p_value']:.6f})")

# Verify CUPED does not bias the point estimate
print(f"\n--- Bias Check ---")
print(f"Naive lift: ${baseline['lift']:.4f}")
print(f"CUPED lift: ${cuped['lift']:.4f}")
print(f"Difference: ${cuped['lift'] - baseline['lift']:.4f}")
print(
    f"CUPED is {'unbiased' if abs(cuped['lift'] - baseline['lift']) < 2 * baseline['se'] else 'BIASED -- investigate'}"
)
print(f"Equivalent to collecting {vr['effective_sample_multiplier']:.2f}x more data")
# INTERPRETATION: CUPED reduces variance by rho^2. The point estimate is
# unbiased because X is pre-treatment, so E[X - E[X]] is the same in both
# arms. Only precision changes. Read rho above: here pre-period activity
# is only weakly correlated with revenue, so the gain is modest — CUPED is
# only as good as the covariate you feed it.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 0 <= abs(cuped["rho"]) <= 1, "Correlation must be between -1 and 1"
reference = single_cov_cuped(y_c, y_t, x_c, x_t)
assert abs(cuped["lift"] - reference["lift"]) < 1e-9, "Your CUPED lift should match the reference helper"
assert abs(cuped["theta"] - reference["theta"]) < 1e-9, "Your theta should match the reference helper"
assert cuped["se"] <= baseline["se"] * 1.01, "CUPED SE must be <= naive SE"
assert (
    abs(vr["variance_reduction"] - cuped["theoretical_reduction"]) < 0.1
), "Actual reduction should approximate theoretical rho^2"
print("\n>>> Checkpoint 3 passed -- CUPED variance reduction verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Multi-Covariate CUPED
# ════════════════════════════════════════════════════════════════════════
# With multiple pre-experiment features, use multivariate regression
# to compute the optimal adjustment: Y_adj = Y - X*theta where
# theta = (X'X)^-1 X'Y (regression coefficients from Y on pre-covariates)
#
# Every covariate must be fixed BEFORE assignment: the pre-period metric
# and user attributes (segment, platform, country). metric_value is
# measured DURING the experiment — it is an outcome, not a covariate.

print(f"\n=== Multi-Covariate CUPED ===")

PRE_TREATMENT_COVARIATES = ["pre_metric_value", "segment", "platform", "country"]
both_arms = pl.concat([control, treatment])
X_multi = (
    both_arms.select(PRE_TREATMENT_COVARIATES)
    .to_dummies(["segment", "platform", "country"], drop_first=True)
    .to_numpy()
    .astype(np.float64)
)
X_c_multi, X_t_multi = X_multi[:n_c], X_multi[n_c:]

multi = multi_cov_cuped(y_c, y_t, X_c_multi, X_t_multi)
vr_multi = variance_reduction(baseline["se"], multi["se"])

print(f"Covariates: {PRE_TREATMENT_COVARIATES} ({X_multi.shape[1]} columns after dummies)")
print(
    f"Variance reduction: {vr_multi['variance_reduction']:.1%} (single-cov: {vr['variance_reduction']:.1%})"
)
print(
    f"SE: ${multi['se']:.3f} (single: ${cuped['se']:.3f}, naive: ${baseline['se']:.3f})"
)
print(f"Lift: ${multi['lift']:.2f}, CI: [${multi['ci_lo']:.2f}, ${multi['ci_hi']:.2f}]")
# INTERPRETATION: Compare the two variance-reduction figures. Extra
# covariates only help if they predict revenue beyond what the pre-period
# metric already explains; if the gain is near zero, they carry no signal.

# WHY post-treatment covariates are forbidden — a deliberate mistake:
X_bad = both_arms.select(["pre_metric_value", "metric_value"]).to_numpy().astype(np.float64)
bad = multi_cov_cuped(y_c, y_t, X_bad[:n_c], X_bad[n_c:])
print(f"\nWRONG (adds in-experiment metric_value as a 'covariate'):")
print(f"  Lift: ${bad['lift']:.2f} (SE ${bad['se']:.3f}) vs naive ${baseline['lift']:.2f}")
print(f"  The treatment moved metric_value, so adjusting for it removed "
      f"{1 - bad['lift'] / baseline['lift']:.0%} of the real effect while the SE "
      f"looked impressively small.")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert "metric_value" not in PRE_TREATMENT_COVARIATES, "Only pre-treatment covariates allowed"
assert multi["se"] <= baseline["se"] * 1.01, "Multi-CUPED SE should be <= naive"
assert abs(multi["lift"] - baseline["lift"]) < 2 * baseline["se"], "Valid CUPED must not move the lift"
print("\n>>> Checkpoint 4 passed -- multi-covariate CUPED completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Stratified CUPED: Heterogeneous Treatment Effects
# ════════════════════════════════════════════════════════════════════════
# Do different user segments respond differently to treatment?
# Stratify by pre-experiment spending level and apply CUPED within strata.

print(f"\n=== Stratified CUPED ===")

strata = stratify_by_covariate(x_c, x_t)
strat_results = stratified_cuped(y_c, y_t, x_c, x_t, strata)

print(
    f"{'Stratum':<20} {'n_ctrl':>8} {'n_treat':>8} {'Lift':>10} {'SE':>8} {'p-value':>10}"
)
print("-" * 68)
for name, r in strat_results.items():
    print(
        f"{name:<20} {r['n_ctrl']:>8,} {r['n_treat']:>8,} "
        f"${r['lift']:>8.2f} ${r['se']:>6.2f} {r['p_value']:>10.6f}"
    )
# INTERPRETATION: If high spenders respond differently to treatment
# than low spenders, a one-size-fits-all analysis masks the heterogeneity.
# Stratified CUPED reveals these differences while maintaining precision.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(strat_results) >= 2, "Should have at least 2 strata"
print("\n>>> Checkpoint 5 passed -- stratified CUPED completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise: Naive vs CUPED Confidence Intervals
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
methods = ["Naive", "CUPED (1-cov)", "CUPED (multi)"]
ses = [baseline["se"], cuped["se"], multi["se"]]
lifts = [baseline["lift"], cuped["lift"], multi["lift"]]
for m, s, l in zip(methods, ses, lifts):
    lo, hi = l - 1.96 * s, l + 1.96 * s
    fig.add_trace(
        go.Scatter(
            x=[lo, l, hi],
            y=[m] * 3,
            mode="markers+lines",
            name=m,
            marker={"size": [8, 12, 8]},
        )
    )
fig.add_vline(x=0, line_dash="dot", line_color="red")
fig.update_layout(
    title="Confidence Intervals: Naive vs CUPED",
    xaxis_title="Treatment Effect ($)",
)
out_path = OUTPUT_DIR / "cuped_comparison.html"
fig.write_html(str(out_path))
print(f"\nSaved: {out_path}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — A Regional E-Commerce Platform: Faster Experiment Decisions
# ════════════════════════════════════════════════════════════════════════
# Scenario (illustrative figures): a regional e-commerce marketplace runs
# ~200 A/B tests per quarter, each planned for 14 days. Required sample
# size scales with variance, so a CUPED multiplier of M lets the same
# test reach the same precision in 14 / M days.
#
# The planning figures (200 tests/quarter, S$50K opportunity cost per
# experiment-week of delay) are assumptions for the exercise; the
# multiplier comes from THIS experiment's measured variance reduction.

print(f"\n--- Singapore Application: E-Commerce Experiment Velocity ---")
eff_mult = vr["effective_sample_multiplier"]
tests_per_quarter = 200  # illustrative
base_days = 14  # illustrative
cost_per_experiment_week = 50_000  # S$, illustrative
cuped_days = base_days / eff_mult
days_saved = base_days - cuped_days
weeks_saved_per_year = tests_per_quarter * 4 * days_saved / 7
print(f"Effective sample multiplier (measured): {eff_mult:.3f}x")
print(f"Duration per test: {base_days} days -> {cuped_days:.1f} days ({days_saved:.1f} days saved)")
print(f"Experiment-weeks saved per year: {weeks_saved_per_year:,.0f}")
print(f"Illustrative annual value: S${weeks_saved_per_year * cost_per_experiment_week:,.0f}")
# INTERPRETATION: The value is driven by rho. On this dataset the
# covariate is weak, so the saving is small; with rho = 0.7 the same
# platform would save about half of every test's duration.


# ════════════════════════════════════════════════════════════════════════
# LOG — ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_cuped_results():
    db = "sqlite:///mlfp02_experiments.db"
    tracker = await ExperimentTracker.create(store_url=db)

    exp_id = "mlfp02_ex7_cuped"

    async with tracker.track(experiment=exp_id, run_name="cuped_analysis") as run:
        await run.log_params(
            {
                "treatment_arm": ANALYSIS_ARM,
                "srm_pair_p": str(srm_p),
                "cuped_covariate": "pre_metric_value",
                "cuped_theta": str(float(cuped["theta"])),
                "cuped_rho": str(float(cuped["rho"])),
                "multi_covariates": ",".join(PRE_TREATMENT_COVARIATES),
            }
        )
        await run.log_metrics(
            {
                "lift_naive": float(baseline["lift"]),
                "lift_cuped": float(cuped["lift"]),
                "se_naive": float(baseline["se"]),
                "se_cuped": float(cuped["se"]),
                "variance_reduction": float(vr["variance_reduction"]),
                "ci_width_reduction": float(vr["ci_width_reduction"]),
                "p_naive": float(baseline["p_value"]),
                "p_cuped": float(cuped["p_value"]),
            }
        )
    print(f"\nLogged CUPED experiment run")
    await tracker.close()


asyncio.run(log_cuped_results())

# ── Checkpoint 6 ─────────────────────────────────────────────────────
print("\n>>> Checkpoint 6 passed -- visualisation and logging complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  - CUPED: Y_adj = Y - theta*(X - E[X]), theta = Cov(Y,X)/Var(X)
  - Variance reduction: Var(Y_adj) = Var(Y)(1 - rho^2)
  - SRM against the designed allocation, then analyse one clean pair
  - Multi-covariate CUPED with pre-treatment covariates only
  - Stratified CUPED: heterogeneous treatment effects by segment
  - Bias check: CUPED preserves the point estimate, only reduces SE

  NEXT: In 02_bayesian_ab.py, you'll learn to compute P(B > A)
  and make ship/continue/hold decisions using expected loss.
"""
)

print("\n>>> Exercise 7.1 complete -- CUPED Variance Reduction")
