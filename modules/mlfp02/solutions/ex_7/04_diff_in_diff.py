# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 7.4: Difference-in-Differences (DiD)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement DiD for causal inference from observational data
#   - Test the parallel trends assumption that underlies DiD
#   - Understand when DiD is appropriate vs randomised experiments
#   - Visualise the counterfactual and treatment effect
#   - Synthesise CUPED, Bayesian, sequential, and DiD into a decision
#     framework for choosing the right causal inference method
#
# PREREQUISITES: Exercise 7.1-7.3
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Simulate HDB transactions around a hypothetical cooling measure
#   2. Compute the DiD estimate and standard error
#   3. Test the parallel trends assumption (pre-period interaction test)
#   4. Visualise DiD with counterfactual
#   5. Apply to Singapore property policy evaluation
#   6. Synthesise all causal inference methods
#
# THEORY (DiD):
#   ATT = (Y_treat_post - Y_treat_pre) - (Y_ctrl_post - Y_ctrl_pre)
#   Key assumption: without treatment, both groups would have followed
#   parallel trends.
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
    OUTPUT_DIR,
    did_cells,
    diff_in_diff,
    parallel_trends_test,
    print_banner,
    simulate_hdb_cooling_panel,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — When Randomisation Is Impossible
# ════════════════════════════════════════════════════════════════════════
# You cannot randomise government policy. You cannot randomly assign
# stamp duty to some districts and not others (well, you could, but
# no government would agree). Yet policymakers still need to know:
# "Did the cooling measure actually reduce HDB prices?"
#
# Difference-in-Differences (DiD) answers this by comparing:
#   - How the TREATED group changed (Central HDB prices: pre vs post)
#   - How the CONTROL group changed (Non-Central HDB prices: pre vs post)
#   - The DIFFERENCE of these differences isolates the treatment effect
#
# The key assumption is PARALLEL TRENDS: without the policy, Central
# and Non-Central prices would have moved in the same direction by the
# same amount. If Central was already declining before the policy, DiD
# attributes the decline to the policy when it was already happening.
#
# Analogy: Two runners are jogging side by side at the same pace.
# One drinks an energy drink (treatment). If they speed up relative
# to the other runner, the energy drink had an effect. But if they
# were already faster BEFORE the drink, you cannot attribute the
# difference to the drink — that violates parallel trends.
#
# The parallel-trends test below uses ONLY pre-policy data from the same
# transactions: if the treated group's prices were already rising faster
# (or slower) before the policy, the group x time interaction picks it up.
#
# WHY THIS MATTERS: Property cooling measures such as the Additional
# Buyer's Stamp Duty (ABSD) cannot be randomised, so anyone assessing
# whether they worked needs a quasi-experimental design like DiD.
# NOTE: the scenario in this exercise is a hypothetical measure on
# simulated data, not an evaluation of any real policy.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Simulate Singapore HDB Cooling Measures
# ════════════════════════════════════════════════════════════════════════

print_banner("MLFP02 Exercise 7.4: Difference-in-Differences")

# 6 quarters before and 6 after a HYPOTHETICAL measure that applies only
# to Central-region flats. The simulation's true policy effect is -$20,000.
panel = simulate_hdb_cooling_panel(
    n_per_period=200, n_pre=6, n_post=6, policy_effect=-20_000, seed=99
)
cells = did_cells(panel)

print(f"\n  Scenario (simulated): hypothetical measure on Central-region flats")
print(f"  Treatment: Central transactions (subject to the measure)")
print(f"  Control:   Non-Central transactions (not subject to it)")
print(f"  Panel: {panel.height:,} transactions over {panel['period'].n_unique()} quarters")
print(panel.group_by(["central", "post"]).len().sort(["central", "post"]))

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert set(panel.columns) == {"period", "central", "post", "price"}, "Unexpected panel columns"
assert all(len(v) == 1200 for v in cells.values()), "Each DiD cell should hold 6 x 200 rows"
print("\n>>> Checkpoint 1 passed -- HDB panel simulated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Compute DiD Estimate
# ════════════════════════════════════════════════════════════════════════
# ATT = (Y_treat_post - Y_treat_pre) - (Y_ctrl_post - Y_ctrl_pre)

y_treat_pre = cells["pre_central"].mean()
y_treat_post = cells["post_central"].mean()
y_ctrl_pre = cells["pre_noncentral"].mean()
y_ctrl_post = cells["post_noncentral"].mean()
did_estimate = (y_treat_post - y_treat_pre) - (y_ctrl_post - y_ctrl_pre)

# The reference helper adds the standard error, CI and p-value
did = diff_in_diff(cells)

print(f"\n=== Difference-in-Differences ===")
print(f"\n{'Group':<15} {'Pre-policy':>14} {'Post-policy':>14} {'Delta':>14}")
print("-" * 60)
print(
    f"{'Central':<15} ${did['y_treat_pre']:>12,.0f} ${did['y_treat_post']:>12,.0f} "
    f"${did['y_treat_post'] - did['y_treat_pre']:>+12,.0f}"
)
print(
    f"{'Non-Central':<15} ${did['y_ctrl_pre']:>12,.0f} ${did['y_ctrl_post']:>12,.0f} "
    f"${did['y_ctrl_post'] - did['y_ctrl_pre']:>+12,.0f}"
)
print(f"\nYour DiD estimate:             ${did_estimate:,.0f}")
print(f"DiD estimate (policy effect): ${did['did_estimate']:,.0f}  (true simulated effect: -$20,000)")
print(f"SE: ${did['se']:,.0f}")
print(f"95% CI: [${did['ci_lo']:,.0f}, ${did['ci_hi']:,.0f}]")
print(f"p-value: {did['p_value']:.4f}")
# INTERPRETATION: DiD removes time-invariant confounders by differencing
# pre and post periods. The assumption is that without the policy,
# Central and Non-Central would have followed parallel trends. The
# sign of the DiD estimate says whether Central prices ended up below
# (negative) or above (positive) where the control group's trend says
# they would have been. Check whether the CI covers the true -$20,000.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert did["se"] > 0, "DiD SE must be positive"
assert abs(did_estimate - did["did_estimate"]) < 1e-6, "Your DiD should match the reference helper"
print("\n>>> Checkpoint 2 passed -- DiD analysis completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Parallel Trends Test
# ════════════════════════════════════════════════════════════════════════
# DiD validity requires parallel trends in the pre-period.
# Test on the PRE-period rows of the same panel:
#   price = b0 + b1*period + b2*central + b3*(central x period)
# b3 = difference in pre-period slopes; H0: b3 = 0.

print(f"\n=== Parallel Trends Test ===")

pt = parallel_trends_test(panel)


def report_trends(label: str, result: dict) -> None:
    print(f"{label}:")
    print(f"  Central slope:     ${result['slope_central']:,.0f}/quarter")
    print(f"  Non-Central slope: ${result['slope_noncentral']:,.0f}/quarter")
    print(
        f"  Slope difference:  ${result['slope_diff']:,.0f}/quarter "
        f"(SE ${result['slope_diff_se']:,.0f}), t = {result['t_stat']:.2f}, "
        f"p = {result['p_value']:.4f}"
    )
    if result["passes"]:
        print("  Cannot reject parallel pre-trends — DiD assumption is plausible")
    else:
        print("  Pre-trends DIFFER — DiD would be biased; do not report it as causal")


report_trends("Pre-period trends (this panel)", pt)

# Does the test have teeth? Simulate a market where Central prices were
# already rising $6,000/quarter faster BEFORE the measure.
panel_violated = simulate_hdb_cooling_panel(
    n_per_period=200, n_pre=6, n_post=6, central_extra_growth=6_000,
    policy_effect=-20_000, seed=99,
)
pt_violated = parallel_trends_test(panel_violated)
did_violated = diff_in_diff(did_cells(panel_violated))
report_trends("\nPre-period trends (violated scenario)", pt_violated)
print(
    f"  DiD on the violated panel: ${did_violated['did_estimate']:,.0f} "
    f"vs the true -$20,000 — the pre-existing trend contaminates the estimate"
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert isinstance(pt["passes"], bool), "Parallel trends must return bool"
assert not pt_violated["passes"], "The test must reject clearly non-parallel pre-trends"
print("\n>>> Checkpoint 3 passed -- parallel trends test completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: DiD with Counterfactual
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=["Pre", "Post"],
        y=[did["y_treat_pre"], did["y_treat_post"]],
        name="Central (treated)",
        line={"color": "red"},
        mode="lines+markers",
    )
)
fig.add_trace(
    go.Scatter(
        x=["Pre", "Post"],
        y=[did["y_ctrl_pre"], did["y_ctrl_post"]],
        name="Non-Central (control)",
        line={"color": "blue"},
        mode="lines+markers",
    )
)
# Counterfactual: what Central would have been without the policy
counterfactual = did["y_treat_pre"] + (did["y_ctrl_post"] - did["y_ctrl_pre"])
fig.add_trace(
    go.Scatter(
        x=["Pre", "Post"],
        y=[did["y_treat_pre"], counterfactual],
        name="Counterfactual (Central without policy)",
        line={"dash": "dot", "color": "red"},
        mode="lines+markers",
    )
)
# Annotate the DiD
fig.add_annotation(
    x="Post",
    y=(did["y_treat_post"] + counterfactual) / 2,
    text=f"DiD = ${did['did_estimate']:,.0f}",
    showarrow=True,
    arrowhead=2,
)
fig.update_layout(
    title="Difference-in-Differences: Singapore HDB Cooling Measures",
    yaxis_title="Mean HDB Price (S$)",
)
out_path = OUTPUT_DIR / "did_visualization.html"
fig.write_html(str(out_path))
print(f"\nSaved: {out_path}")

# Quarterly means, pre and post, for both panels
fig2 = go.Figure()
for label, pan, dash in (("", panel, "solid"), (" — violated", panel_violated, "dot")):
    q = (
        pan.group_by(["period", "central"])
        .agg(pl.col("price").mean())
        .sort(["central", "period"])
    )
    for central, colour, name in ((1, "red", "Central"), (0, "blue", "Non-Central")):
        g = q.filter(pl.col("central") == central)
        fig2.add_trace(
            go.Scatter(
                x=g["period"].to_list(),
                y=g["price"].to_list(),
                name=f"{name}{label}",
                line={"color": colour, "dash": dash},
                mode="lines+markers",
            )
        )
fig2.add_vline(x=5.5, line_dash="dash", annotation_text="Measure starts")
fig2.update_layout(
    title=(
        f"Parallel Trends: p={pt['p_value']:.3f} (this panel), "
        f"p={pt_violated['p_value']:.2g} (violated)"
    ),
    xaxis_title="Quarter",
    yaxis_title="Mean HDB Price (S$)",
)
out_path2 = OUTPUT_DIR / "parallel_trends.html"
fig2.write_html(str(out_path2))
print(f"Saved: {out_path2}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Evaluating a Property Cooling Measure (hypothetical)
# ════════════════════════════════════════════════════════════════════════
# Scenario: a policy analyst must report whether the hypothetical
# Central-region measure lowered prices. Real measures such as ABSD
# (introduced in December 2011 and revised several times since) raise
# the same question, and DiD is one standard way to answer it — but
# only after the pre-trend check passes.
#
# The analyst's report: per-flat effect with its CI, the share of the
# pre-policy price, and the effect summed over an ASSUMED number of
# Central transactions per year (illustrative, not an official figure).

print(f"\n--- Singapore Application: Cooling-Measure Evaluation (simulated) ---")
central_txns_per_year = 5_000  # illustrative
price_effect_pct = did["did_estimate"] / did["y_treat_pre"]
print(
    f"Effect per Central flat: ${did['did_estimate']:,.0f} "
    f"(95% CI ${did['ci_lo']:,.0f} to ${did['ci_hi']:,.0f})"
)
print(f"As a share of the pre-policy Central mean: {price_effect_pct:+.1%}")
print(
    f"Summed over {central_txns_per_year:,} assumed transactions/year: "
    f"S${did['did_estimate'] * central_txns_per_year / 1e6:,.1f}M"
)
if not pt["passes"]:
    conclusion = "not reportable — pre-trends differ"
elif did["p_value"] < 0.05:
    conclusion = "measure lowered Central prices relative to the control trend"
else:
    conclusion = "no detectable effect at the 5% level"
print(f"Policy conclusion: {conclusion}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Causal Inference Decision Framework
# ════════════════════════════════════════════════════════════════════════
# Synthesise all four methods into a decision tree.

print(f"\n{'='*70}")
print(f"CAUSAL INFERENCE DECISION FRAMEWORK")
print(f"{'='*70}")
print(
    """
When to use each method:

  CUPED: You have an RCT AND pre-experiment data.
    -> Reduces CI width (free precision gain)
    -> Unbiased: same point estimate, just tighter

  Bayesian A/B: You want P(B > A) instead of "is p < 0.05?"
    -> Directly answers "should we ship?"
    -> Expected loss quantifies cost of being wrong

  Sequential (mSPRT): You need to monitor experiments safely.
    -> Fixed p-values inflate Type I error when peeking
    -> mSPRT: always-valid, correct alpha at any stopping time

  DiD: Randomisation is impossible (policy evaluation).
    -> Requires parallel trends assumption
    -> Less precise than RCT but works with observational data
"""
)


# ════════════════════════════════════════════════════════════════════════
# LOG — ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_did_results():
    db = "sqlite:///mlfp02_experiments.db"
    tracker = await ExperimentTracker.create(store_url=db)

    exp_id = "mlfp02_ex7_diff_in_diff"

    async with tracker.track(experiment=exp_id, run_name="did_hdb_cooling") as run:
        await run.log_params(
            {
                "did_treatment": "Central HDB (simulated)",
                "did_control": "Non-Central HDB (simulated)",
                "n_per_cell": str(len(cells["pre_central"])),
                "parallel_trends_method": "pre-period group x time interaction",
            }
        )
        await run.log_metrics(
            {
                "did_estimate": float(did["did_estimate"]),
                "did_se": float(did["se"]),
                "did_p_value": float(did["p_value"]),
                "parallel_trends_p": float(pt["p_value"]),
                "parallel_trends_passes": float(pt["passes"]),
            }
        )
    print(f"\nLogged DiD experiment run")
    await tracker.close()


asyncio.run(log_did_results())

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n>>> Checkpoint 4 passed -- visualisation and logging complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - DiD: ATT = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)
  - Parallel trends: pre-period group x time interaction test, and
    proof that it rejects when trends really differ
  - Counterfactual reasoning: what WOULD have happened without treatment
  - Reporting a (hypothetical) cooling-measure evaluation responsibly
  - Decision framework: CUPED vs Bayesian vs Sequential vs DiD

  COMPLETE: You now have four causal inference tools:
    1. CUPED — precision gain for randomised experiments
    2. Bayesian A/B — probability-based decisions with expected loss
    3. Sequential testing — safe experiment monitoring
    4. DiD — causal inference from observational data

  NEXT: In Exercise 8 (Module 2 Capstone), you'll build a complete
  statistical analysis pipeline from data to stakeholder report.
"""
)

print("\n>>> Exercise 7.4 complete -- Difference-in-Differences")
