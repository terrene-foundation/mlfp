# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 3.5: Parametric Bootstrap — Resampling from a
#                        Fitted Model
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Distinguish parametric bootstrap (resample from a FITTED
#     distribution) from non-parametric bootstrap (resample the DATA)
#   - See the trade: parametric buys precision when the model is right,
#     and confident error when the model is wrong
#   - Diagnose misspecification: a Normal model on Exponential-shaped
#     waiting times mis-centres the CI for the median by design
#   - Choose a defensible parametric family using the diagnostics from
#     Exercise 2.6 (SD/Mean ≈ 1 → Exponential)
#
# PREREQUISITES: 01_bootstrap_power.py (non-parametric bootstrap, CIs);
#   Exercise 2.6 (Poisson process, Exponential fit)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — what "parametric" buys and what it costs
#   2. Build — inter-arrival gaps from the April 2024 taxi window
#   3. Train — three bootstraps of the median gap: non-parametric,
#      Normal (misspecified), Exponential (defensible)
#   4. Visualise — the three bootstrap distributions side by side
#   5. Apply — a one-morning sample (n = 30): which bootstrap to trust
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from shared.mlfp02.ex_2 import (
    bootstrap_percentile_ci,
    bootstrap_statistic,
    exponential_mle,
    load_taxi_trips,
    taxi_interarrival_seconds,
)
from shared.mlfp02.ex_3 import (
    N_BOOTSTRAP,
    RANDOM_SEED,
    OUTPUT_DIR,
    parametric_bootstrap_statistic,
    print_header,
    track_train_run,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — What "Parametric" Buys and What It Costs
# ════════════════════════════════════════════════════════════════════════
# The non-parametric bootstrap resamples the DATA with replacement. It
# makes almost no assumptions — but with n = 30 it can only ever recombine
# the 30 values you happened to observe. The tails of the resampled
# statistic are built from the tails of your one sample.
#
# The PARAMETRIC bootstrap resamples from a FITTED MODEL instead:
#   1. Fit a distribution to the sample (e.g. Normal(μ̂, σ̂))
#   2. Draw a fresh synthetic sample of size n from that distribution
#   3. Compute the statistic; repeat
#
# When the model family is RIGHT, the parametric bootstrap is sharper:
# the synthetic samples explore the tails the way the true distribution
# would, not just the way your one sample did. When the model family is
# WRONG, it is confidently wrong — every synthetic sample inherits the
# wrong shape, and the CI centres on the wrong value.
#
# Waiting times make the stakes visible. Exponential gaps are right-skewed:
# the mean exceeds the median by a factor of 1/ln(2) ≈ 1.44. A Normal
# parametric bootstrap of the MEDIAN gap is therefore misspecified BY
# DESIGN — the Normal's median equals its mean, so its bootstrap
# distribution centres ~44% above the median you actually computed.
# Misspecification is not a bigger error bar; it is an error bar around
# the WRONG NUMBER.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Inter-arrival gaps from the April 2024 window
# ════════════════════════════════════════════════════════════════════════

print_header("MLFP02 Exercise 3.5: Parametric Bootstrap")

trips = load_taxi_trips()
gaps = taxi_interarrival_seconds(trips)

print(f"\n  April 2024 inter-arrival gaps: {len(gaps):,}")
print(f"  Mean gap:   {gaps.mean():.0f}s")
print(f"  Median gap: {np.median(gaps):.0f}s")
print(
    f"  Exponential signature: SD/Mean = {gaps.std(ddof=1) / gaps.mean():.2f} "
    "(Exponential predicts 1.00)"
)
print(
    f"  Skew signature: mean/median = {gaps.mean() / np.median(gaps):.2f} "
    "(Exponential predicts 1/ln2 ≈ 1.44)"
)

# Working sample: n = 60 seeded gaps — small enough that the bootstrap
# method choice visibly matters
rng = np.random.default_rng(RANDOM_SEED)
n_small = 60
sample = rng.choice(gaps, size=n_small, replace=False)
sample_median = float(np.median(sample))
sample_mean = float(sample.mean())

print(f"\n  Working sample: n = {n_small} (seeded draw)")
print(f"  Sample mean:   {sample_mean:.0f}s")
print(f"  Sample median: {sample_median:.0f}s")
# INTERPRETATION: mean >> median flags right skew. The statistic of
# interest is the MEDIAN wait — the "typical" gap a driver experiences —
# precisely the statistic a symmetric Normal model cannot represent.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(sample) == n_small, "Working sample must have n = 60"
assert sample_mean > sample_median, "Expect right skew: mean above median"
assert (sample > 0).all(), "Gaps must be strictly positive"
print("\n>>> Checkpoint 1 passed — sample built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Three bootstraps of the median gap
# ════════════════════════════════════════════════════════════════════════

# (a) Non-parametric: resample the 60 observed gaps
boot_nonparam = bootstrap_statistic(
    sample, lambda x: float(np.median(x)), n_boot=N_BOOTSTRAP, seed=RANDOM_SEED
)
ci_nonparam = bootstrap_percentile_ci(boot_nonparam)

# (b) Parametric, MISSPECIFIED: Normal(μ̂, σ̂) — symmetric, median = mean
mu_hat = sample_mean
sigma_hat = float(sample.std(ddof=1))
boot_normal = parametric_bootstrap_statistic(
    lambda r, n: r.normal(mu_hat, sigma_hat, size=n),
    n=n_small,
    statistic=lambda x: float(np.median(x)),
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_SEED,
)
ci_normal = bootstrap_percentile_ci(boot_normal)

# (c) Parametric, DEFENSIBLE: Exponential — the family validated in 2.6
expo = exponential_mle(sample)
boot_exponential = parametric_bootstrap_statistic(
    lambda r, n: r.exponential(scale=1.0 / expo["lambda"], size=n),
    n=n_small,
    statistic=lambda x: float(np.median(x)),
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_SEED,
)
ci_exponential = bootstrap_percentile_ci(boot_exponential)

# Theoretical anchor: median of Exponential(λ) = ln(2)/λ
theoretical_median = float(np.log(2) / expo["lambda"])

print("=== Three bootstraps of the median gap (n=60) ===")
print(f"  Sample median: {sample_median:.0f}s   Sample mean: {sample_mean:.0f}s")
print(f"\n{'Bootstrap':<28} {'Centre':>9} {'95% CI (seconds)':>24} {'Width':>8}")
print("-" * 74)
for label, draws, ci in (
    ("Non-parametric", boot_nonparam, ci_nonparam),
    ("Parametric Normal (!)", boot_normal, ci_normal),
    ("Parametric Exponential", boot_exponential, ci_exponential),
):
    print(
        f"{label:<28} {float(np.median(draws)):>8.0f}s "
        f"[{ci[0]:>7.0f}, {ci[1]:>7.0f}]{ci[1] - ci[0]:>10.0f}"
    )
print(f"\n  Exponential theory: median = ln(2)/λ̂ = {theoretical_median:.0f}s")
# INTERPRETATION: the Normal bootstrap centres on the sample MEAN (its
# own median), ~40% above the sample median — its CI need not bracket
# the statistic you computed. The Exponential bootstrap re-centres near
# ln(2)/λ̂ and overlaps the non-parametric CI. Same loop, same n, same
# seed — the only difference is WHERE new samples come from.

# ── Log the comparison to ExperimentTracker ──────────────────────────
run_id = track_train_run(
    experiment="mlfp02_ex3_05_parametric_bootstrap",
    run_name="median_gap_three_bootstraps",
    params={
        "statistic": "median",
        "n_sample": str(n_small),
        "models": "nonparametric,normal,exponential",
        "n_boot": str(N_BOOTSTRAP),
    },
    metrics={
        "sample_median_s": sample_median,
        "sample_mean_s": sample_mean,
        "nonparam_ci_centre_s": float(np.median(boot_nonparam)),
        "normal_ci_centre_s": float(np.median(boot_normal)),
        "exponential_ci_centre_s": float(np.median(boot_exponential)),
        "nonparam_ci_width_s": float(ci_nonparam[1] - ci_nonparam[0]),
        "exponential_ci_width_s": float(ci_exponential[1] - ci_exponential[0]),
    },
)
print(f"\nLogged training run to ExperimentTracker (run {run_id})")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(boot_nonparam) == N_BOOTSTRAP, "Non-parametric draws missing"
assert len(boot_normal) == N_BOOTSTRAP, "Normal parametric draws missing"
assert len(boot_exponential) == N_BOOTSTRAP, "Exponential parametric draws missing"
assert ci_nonparam[0] < sample_median < ci_nonparam[1], (
    "Non-parametric CI must bracket the sample median"
)
assert abs(float(np.median(boot_normal)) - sample_mean) < abs(
    float(np.median(boot_normal)) - sample_median
), "Normal bootstrap must centre near the mean, not the median"
assert ci_exponential[0] < theoretical_median < ci_exponential[1], (
    "Exponential CI must bracket the theoretical median ln(2)/λ̂"
)
print("\n>>> Checkpoint 2 passed — three bootstraps computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: The three bootstrap distributions
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=boot_nonparam,
        nbinsx=60,
        histnorm="probability density",
        name="Non-parametric",
        opacity=0.55,
    )
)
fig.add_trace(
    go.Histogram(
        x=boot_normal,
        nbinsx=60,
        histnorm="probability density",
        name="Parametric Normal (misspecified)",
        opacity=0.55,
    )
)
fig.add_trace(
    go.Histogram(
        x=boot_exponential,
        nbinsx=60,
        histnorm="probability density",
        name="Parametric Exponential",
        opacity=0.55,
    )
)
fig.add_vline(
    x=sample_median,
    line_dash="dash",
    line_color="black",
    annotation_text=f"Sample median {sample_median:.0f}s",
)
fig.add_vline(
    x=sample_mean,
    line_dash="dot",
    line_color="red",
    annotation_text=f"Sample mean {sample_mean:.0f}s",
)
fig.update_layout(
    title="Bootstrap Distributions of the Median Gap — Model Choice Moves the Centre",
    xaxis_title="Bootstrapped median gap (seconds)",
    yaxis_title="Density",
    barmode="overlay",
    height=450,
)
fig_path = OUTPUT_DIR / "parametric_bootstrap.html"
fig.write_html(str(fig_path))
print(f"Saved: {fig_path}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert fig_path.exists(), "Figure must be written"
print("\n>>> Checkpoint 3 passed — visualisation saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: A One-Morning Sample (n = 30) — Which Bootstrap?
# ════════════════════════════════════════════════════════════════════════
# A taxi-stand operator (anonymised) instruments a NEW stand and has one
# morning of data: 30 gaps. Management asks for the "typical wait" with
# an interval. At n = 30 the non-parametric bootstrap recombines just 30
# values — its CI is lumpy. The parametric bootstrap can do better IF
# the family is right.
#
# Decision rule used in practice:
#   1. Diagnose the family FIRST (2.6's SD/Mean ≈ 1 check, on as much
#      related data as you can borrow — here the April window).
#   2. Fit the defensible family (Exponential for memoryless gaps).
#   3. Cross-check: if the parametric CI and the non-parametric CI
#      disagree on the CENTRE, the model is wrong, not the data.
#   4. Report both when they agree; distrust the parametric one when
#      they don't.

print("\n--- Application: one-morning sample at a new stand ---")
tiny = gaps[:30]
tiny_median = float(np.median(tiny))
tiny_mean = float(tiny.mean())

boot_tiny_np = bootstrap_statistic(
    tiny, lambda x: float(np.median(x)), n_boot=N_BOOTSTRAP, seed=RANDOM_SEED
)
ci_tiny_np = bootstrap_percentile_ci(boot_tiny_np)

expo_tiny = exponential_mle(tiny)
boot_tiny_exp = parametric_bootstrap_statistic(
    lambda r, n: r.exponential(scale=1.0 / expo_tiny["lambda"], size=n),
    n=30,
    statistic=lambda x: float(np.median(x)),
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_SEED,
)
ci_tiny_exp = bootstrap_percentile_ci(boot_tiny_exp)

print(f"  30 gaps: median {tiny_median:.0f}s, mean {tiny_mean:.0f}s")
print(
    f"  Non-parametric 95% CI: [{ci_tiny_np[0]:.0f}, {ci_tiny_np[1]:.0f}]s  "
    f"width {ci_tiny_np[1] - ci_tiny_np[0]:.0f}s"
)
print(
    f"  Exponential    95% CI: [{ci_tiny_exp[0]:.0f}, {ci_tiny_exp[1]:.0f}]s  "
    f"width {ci_tiny_exp[1] - ci_tiny_exp[0]:.0f}s"
)
width_ratio = (ci_tiny_np[1] - ci_tiny_np[0]) / (ci_tiny_exp[1] - ci_tiny_exp[0])
print(f"  Non-parametric CI is {width_ratio:.2f}× wider")
centres_agree = abs(
    float(np.median(boot_tiny_exp)) - float(np.median(boot_tiny_np))
) < 0.25 * tiny_median
print(f"  Centres agree within 25% of the median? {'YES' if centres_agree else 'NO'}")
print(
    "  At n=30 the parametric CI is SMOOTHER (built from a model, not 30\n"
    "  recombined values), though not narrower here — its value is the\n"
    "  validated shape, not width. It is reportable only because the\n"
    "  family was validated on the full month first: the assumption came\n"
    "  from data, not from convenience."
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(tiny) == 30, "Tiny sample must have n = 30"
assert ci_tiny_np[0] < tiny_median < ci_tiny_np[1], (
    "Non-parametric CI must bracket the tiny-sample median"
)
assert ci_tiny_exp[0] > 0, "Exponential CI must stay positive (durations > 0)"
print("\n>>> Checkpoint 4 passed — application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED (3.5 — Parametric Bootstrap)")
print("=" * 70)
print(
    """
  ✓ Non-parametric bootstrap resamples the DATA; parametric resamples
    from a FITTED MODEL — same loop, different source of new samples
  ✓ Right family: parametric is sharper, especially at small n
  ✓ Wrong family: the Normal bootstrap of a median re-centres on the
    MEAN — an error bar around the wrong number, ~44% high for
    Exponential-shaped waits
  ✓ The Exponential bootstrap of the median reproduces ln(2)/λ̂ and
    agrees with the non-parametric CI
  ✓ Decision rule: diagnose the family first → fit the defensible one →
    cross-check the centre against the non-parametric CI before
    trusting the sharp one

  NEXT: In 06_one_sample_one_tailed.py, you'll formulate directional
  hypotheses — when a one-tailed test is legitimate pre-registration and
  when it is p-hacking.
"""
)

print("\n>>> Exercise 3.5 complete — Parametric Bootstrap")
