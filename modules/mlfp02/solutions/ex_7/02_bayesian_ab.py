# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 7.2: Bayesian A/B Testing
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute P(treatment > control | data) using posterior distributions
#   - Calculate expected loss for decision-making under uncertainty
#   - Apply a ship/continue/hold decision framework
#   - Understand when Bayesian beats frequentist A/B testing
#   - Log Bayesian results to ExperimentTracker
#
# PREREQUISITES: Exercise 7.1 (CUPED) — you need CUPED-adjusted arrays
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Load data and apply CUPED adjustment
#   2. Compute Bayesian posterior for the treatment effect
#   3. Expected loss analysis (both directions)
#   4. Decision framework: ship / continue / hold
#   5. Visualise posterior distribution
#   6. Apply to a Singapore payments scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import plotly.graph_objects as go
from kailash_ml import ExperimentTracker
from scipy import stats

from shared.mlfp02.ex_7 import (
    ANALYSIS_ARM,
    OUTPUT_DIR,
    bayesian_decision,
    bayesian_decision_rule,
    compute_srm,
    get_covariate_arrays,
    get_revenue_arrays,
    load_experiment,
    print_banner,
    single_cov_cuped,
    split_groups,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Bayesian A/B Testing?
# ════════════════════════════════════════════════════════════════════════
# Frequentist A/B answers: "If there were no effect, how unlikely is
# this data?" — that is a p-value. But what product teams actually want
# is: "Given the data, what is the probability that B is better than A?"
#
# Bayesian analysis provides:
#   P(treatment > control | data) — direct probability of improvement
#   Expected loss — the average revenue you lose by choosing wrong
#
# With the posterior lift L ~ Normal(m, s) and z = m / s:
#   E[loss | ship treatment] = E[max(0, -L)] = s*phi(z) - m*Phi(-z)
#   E[loss | keep control]   = E[max(0,  L)] = s*phi(z) + m*Phi(z)
# (phi = Normal pdf, Phi = Normal cdf). If the lift is clearly positive,
# shipping costs almost nothing in expectation and keeping control costs
# about m per user.
#
# The expected loss is particularly powerful: if P(B > A) = 75% but
# the expected loss of choosing B is only $0.02/user, you can ship
# confidently. If P(B > A) = 95% but expected loss is $5/user, you
# should collect more data.
#
# WHY THIS MATTERS: For checkout and payment-flow experiments the cost
# of a wrong decision is directly measurable in S$ per transaction, so
# "how much do we lose if we are wrong?" is the question that matters.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and Apply CUPED
# ════════════════════════════════════════════════════════════════════════

print_banner("MLFP02 Exercise 7.2: Bayesian A/B Testing")

experiment = load_experiment()
control, treatment = split_groups(experiment, ANALYSIS_ARM)
srm_p = compute_srm(control.height, treatment.height)  # vs the designed 40:35 ratio
if srm_p < 0.01:
    raise RuntimeError(f"SRM on control vs {ANALYSIS_ARM} (p={srm_p:.2g}) — stop")
y_c, y_t = get_revenue_arrays(control, treatment)
x_c, x_t = get_covariate_arrays(control, treatment)

# Apply CUPED first — Bayesian analysis on CUPED-adjusted data
cuped = single_cov_cuped(y_c, y_t, x_c, x_t)
y_c_adj = cuped["y_c_adj"]
y_t_adj = cuped["y_t_adj"]
lift_adj = cuped["lift"]

print(f"  Control vs {ANALYSIS_ARM}: pairwise SRM p={srm_p:.3f} (OK)")
print(f"  Data loaded, CUPED applied (rho={cuped['rho']:.3f})")
print(f"  CUPED-adjusted lift: ${lift_adj:.2f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert y_c_adj is not None and len(y_c_adj) > 0, "CUPED adjustment must produce data"
print("\n>>> Checkpoint 1 passed -- data loaded and CUPED applied\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Bayesian Posterior for Treatment Effect
# ════════════════════════════════════════════════════════════════════════
# Using a normal approximation on CUPED-adjusted arrays (flat prior,
# large n): posterior lift ~ Normal(lift_adj, se_lift)

se_lift = np.sqrt(
    y_c_adj.var(ddof=1) / len(y_c_adj) + y_t_adj.var(ddof=1) / len(y_t_adj)
)
prob_better = 1 - stats.norm.cdf(0, loc=lift_adj, scale=se_lift)
prob_practical = 1 - stats.norm.cdf(1.0, loc=lift_adj, scale=se_lift)

# Expected loss in each direction (closed form — see THEORY)
z = lift_adj / se_lift
exp_loss_treat = se_lift * stats.norm.pdf(z) - lift_adj * stats.norm.cdf(-z)
exp_loss_ctrl = se_lift * stats.norm.pdf(z) + lift_adj * stats.norm.cdf(z)

bayes = {
    "prob_treatment_better": float(prob_better),
    "prob_practical": float(prob_practical),
    "expected_loss_treatment": float(exp_loss_treat),
    "expected_loss_control": float(exp_loss_ctrl),
    "se_lift": float(se_lift),
    "ci_lo": float(lift_adj - 1.96 * se_lift),
    "ci_hi": float(lift_adj + 1.96 * se_lift),
}

print(f"\n=== Bayesian A/B Test ===")
print(
    f"P(treatment > control): {bayes['prob_treatment_better']:.4f} "
    f"({bayes['prob_treatment_better']:.1%})"
)
print(
    f"P(treatment > control by >$1): {bayes['prob_practical']:.4f} "
    f"({bayes['prob_practical']:.1%})"
)
print(f"Expected loss (choose treatment): ${bayes['expected_loss_treatment']:.2f}/user")
print(f"Expected loss (choose control):   ${bayes['expected_loss_control']:.2f}/user")
print(f"95% credible interval: [${bayes['ci_lo']:.2f}, ${bayes['ci_hi']:.2f}]")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 0 <= bayes["prob_treatment_better"] <= 1, "Probability must be valid"
assert bayes["expected_loss_treatment"] >= 0, "Expected loss must be non-negative"
reference = bayesian_decision(y_c_adj, y_t_adj, lift_adj, practical_threshold=1.0)
assert abs(bayes["expected_loss_control"] - reference["expected_loss_control"]) < 1e-9, (
    "Your expected loss should match the reference helper"
)
print("\n>>> Checkpoint 2 passed -- Bayesian posterior computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Expected Loss Analysis
# ════════════════════════════════════════════════════════════════════════
# Expected loss quantifies the cost of being wrong.
# E[loss | choose treatment] = E[max(control - treatment, 0)]
# E[loss | choose control]   = E[max(treatment - control, 0)]
# Verify the closed form by brute force: sample the posterior and average.

rng = np.random.default_rng(42)
posterior_draws = rng.normal(lift_adj, se_lift, size=1_000_000)
mc_loss_treat = np.maximum(0.0, -posterior_draws).mean()
mc_loss_ctrl = np.maximum(0.0, posterior_draws).mean()

print(f"\n=== Expected Loss Analysis ===")
print(f"Closed form vs Monte-Carlo (1M posterior draws):")
print(f"  choose treatment: ${exp_loss_treat:.6f} vs ${mc_loss_treat:.6f}")
print(f"  choose control:   ${exp_loss_ctrl:.6f} vs ${mc_loss_ctrl:.6f}")
print(f"If we ship treatment and it is worse:")
print(f"  Average loss per user: ${bayes['expected_loss_treatment']:.4f}")
print(f"If we keep control and treatment is actually better:")
print(f"  Average loss per user: ${bayes['expected_loss_control']:.4f}")
if bayes["expected_loss_treatment"] > 1e-6:
    print(
        f"\nLoss ratio (control/treatment): "
        f"{bayes['expected_loss_control'] / bayes['expected_loss_treatment']:.1f}x"
    )
else:
    print("\nExpected loss of shipping is effectively zero — the posterior puts")
    print("essentially no mass below a zero lift.")
# INTERPRETATION: When expected_loss_treatment is tiny (e.g., $0.05/user),
# even moderate confidence (80%) is enough to ship — the cost of being
# wrong is negligible. When it is large (e.g., $5/user), you need very
# high confidence before deploying. Note the asymmetry: the loss of
# keeping control is roughly the lift itself when the lift is clearly
# positive.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert bayes["expected_loss_control"] >= 0, "Expected loss must be non-negative"
assert abs(exp_loss_treat - mc_loss_treat) < 0.01 * se_lift + 1e-6, "Closed form must match Monte-Carlo"
assert abs(exp_loss_ctrl - mc_loss_ctrl) < 0.01 * se_lift + 1e-6, "Closed form must match Monte-Carlo"
print("\n>>> Checkpoint 3 passed -- expected loss analysis complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Decision Framework: Ship / Continue / Hold
# ════════════════════════════════════════════════════════════════════════

decision = bayesian_decision_rule(
    bayes["prob_treatment_better"], bayes["expected_loss_treatment"]
)

print(f"\n=== Decision Framework ===")
print(f"P(treatment better): {bayes['prob_treatment_better']:.1%}")
print(f"Expected loss: ${bayes['expected_loss_treatment']:.2f}/user")
print(f"\nDecision: {decision}")
print(f"\nDecision rules:")
print(f"  SHIP:     P > 95% AND expected loss < $0.50/user")
print(f"  CONTINUE: P > 80% (promising but need more data)")
print(f"  HOLD:     P <= 80% (insufficient evidence)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert decision in [
    "SHIP — high confidence + low expected loss",
    "CONTINUE — promising but need more data",
    "HOLD — insufficient evidence",
], "Decision must be one of the three options"
print("\n>>> Checkpoint 4 passed -- decision framework applied\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Posterior Distribution
# ════════════════════════════════════════════════════════════════════════

x_range = np.linspace(
    lift_adj - 4 * bayes["se_lift"], lift_adj + 4 * bayes["se_lift"], 200
)
pdf_vals = stats.norm.pdf(x_range, loc=lift_adj, scale=bayes["se_lift"])

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=x_range, y=pdf_vals, mode="lines", name="Posterior", fill="tozeroy")
)
fig.add_vline(x=0, line_dash="dot", line_color="red", annotation_text="No effect")
fig.add_vline(
    x=1.0,
    line_dash="dash",
    line_color="green",
    annotation_text="Practical threshold ($1)",
)
fig.add_vline(
    x=lift_adj,
    line_dash="solid",
    line_color="blue",
    annotation_text=f"Estimated lift: ${lift_adj:.2f}",
)
fig.update_layout(
    title="Posterior Distribution of Treatment Effect",
    xaxis_title="Treatment Effect ($)",
    yaxis_title="Density",
)
out_path = OUTPUT_DIR / "bayesian_posterior.html"
fig.write_html(str(out_path))
print(f"\nSaved: {out_path}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — A Singapore Payments App: Checkout Flow Experiment
# ════════════════════════════════════════════════════════════════════════
# Scenario (illustrative volume): a Singapore payments app tests a new
# checkout flow, treating the per-user revenue lift measured above as
# the per-transaction lift, at an assumed 200,000 transactions/day.
#
# Traditional approach: "Is p < 0.05?" — binary, ignores magnitude.
# Bayesian approach: put a dollar figure on each side of the decision:
#   upside of shipping   = E[max(0, L)] per transaction
#   downside of shipping = E[max(0, -L)] per transaction
#   net expected value   = upside - downside = E[L] = the posterior mean
#
# The expected loss framework turns a statistical question into a
# business decision with dollar amounts attached.

print(f"\n--- Singapore Application: Checkout Flow Experiment ---")
daily_txns = 200_000  # illustrative
daily_upside = daily_txns * bayes["expected_loss_control"]  # E[max(0, L)]
daily_downside = daily_txns * bayes["expected_loss_treatment"]  # E[max(0, -L)]
print(f"Daily transactions (assumed): {daily_txns:,}")
print(f"Expected daily upside of shipping:   S${daily_upside:,.0f}")
print(f"Expected daily downside of shipping: S${daily_downside:,.2f}")
print(f"Net expected daily value: S${daily_upside - daily_downside:,.0f}")
print(f"Annualised net value: S${(daily_upside - daily_downside) * 365:,.0f}")
# INTERPRETATION: The downside is the expected cost of being wrong. When
# it is a rounding error next to the upside, shipping is the rational
# decision even before a p-value is consulted.


# ════════════════════════════════════════════════════════════════════════
# LOG — ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_bayesian_results():
    db = "sqlite:///mlfp02_experiments.db"
    tracker = await ExperimentTracker.create(store_url=db)

    exp_id = "mlfp02_ex7_bayesian_ab"

    async with tracker.track(experiment=exp_id, run_name="bayesian_decision") as run:
        await run.log_params(
            {
                "method": "bayesian_normal_approx",
                "treatment_arm": ANALYSIS_ARM,
                "practical_threshold": "1.0",
                "decision": decision,
            }
        )
        await run.log_metrics(
            {
                "prob_treatment_better": float(bayes["prob_treatment_better"]),
                "prob_practical": float(bayes["prob_practical"]),
                "expected_loss_treatment": float(bayes["expected_loss_treatment"]),
                "expected_loss_control": float(bayes["expected_loss_control"]),
                "lift_cuped": float(lift_adj),
            }
        )
    print(f"\nLogged Bayesian experiment run")
    await tracker.close()


asyncio.run(log_bayesian_results())

# ── Checkpoint 5 ─────────────────────────────────────────────────────
print("\n>>> Checkpoint 5 passed -- visualisation and logging complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - Bayesian A/B: P(treatment > control) and expected loss
  - Closed-form expected loss, checked against Monte-Carlo draws
  - Decision framework: ship (>95% + low loss) / continue / hold
  - Expected loss quantifies the cost of being wrong in $/user
  - Posterior credible interval vs frequentist confidence interval
  - When to ship with moderate confidence (low expected loss)

  NEXT: In 03_sequential_testing.py, you'll learn to safely monitor
  experiments without inflating Type I error, using mSPRT.
"""
)

print("\n>>> Exercise 7.2 complete -- Bayesian A/B Testing")
