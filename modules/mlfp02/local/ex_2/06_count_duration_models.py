# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 2.6: Count and Duration Models — Poisson and
#                         Exponential MLE with AIC
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Recognise count data (trips per hour) vs duration data (seconds
#     between trips) and why Normal MLE is wrong for both
#   - Derive and compute closed-form MLEs: Poisson λ̂ = x̄,
#     Exponential λ̂ = 1 / x̄
#   - Check the Poisson dispersion assumption (Var = Mean) on real data
#   - Compare duration models (Exponential vs Gamma vs Weibull) with AIC
#   - Turn fitted models into operational probabilities (P(wait > 10 min))
#
# PREREQUISITES: 05_model_selection.py (AIC/BIC on GDP growth)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — the Poisson process: one mechanism, two distributions
#   2. Build — hourly trip counts and inter-arrival gaps from taxi data
#   3. Train — Poisson and Exponential MLEs + AIC model comparison
#   4. Visualise — observed vs fitted PMF (counts) and PDFs (durations)
#   5. Apply — taxi-stand staffing: wait-time and arrival-rate decisions
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_2 import (
    OUTPUT_DIR,
    aic,
    bic,
    exponential_mle,
    gamma_mle,
    load_taxi_trips,
    poisson_mle,
    save_figure,
    taxi_interarrival_seconds,
    taxi_trips_per_hour,
    track_train_run,
    weibull_mle,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Poisson Process: One Mechanism, Two Distributions
# ════════════════════════════════════════════════════════════════════════
# Suppose events arrive independently at a constant average rate λ
# (taxis pulling into a stand, customers entering a queue). Then:
#
#   - The NUMBER of events in a fixed window is POISSON:
#       P(K = k) = λ^k e^{-λ} / k!     with E[K] = Var[K] = λ
#       MLE: λ̂ = x̄ (the sample mean count)
#
#   - The WAITING TIME between events is EXPONENTIAL:
#       f(t) = λ e^{-λt}               with E[T] = 1/λ, SD[T] = 1/λ
#       MLE: λ̂ = 1 / x̄ (inverse of the sample mean gap)
#
# These are the SAME λ viewed two ways — a fast rate means many counts
# per hour AND short waits between arrivals.
#
# The Poisson distribution has a testable signature: Var = Mean. Real
# count data is often OVER-dispersed (Var > Mean) when the rate is not
# truly constant — rush hours and quiet nights mix different λs. It can
# also be UNDER-dispersed (Var < Mean) when a hard ceiling makes counts
# MORE regular than pure chance — a capped taxi fleet is one example.
# Checking Var/Mean is the fastest model diagnostic you will ever run.
#
# The Exponential's signature: SD = Mean. When SD ≠ Mean the durations
# are not memoryless — Gamma or Weibull may fit better. AIC settles the
# comparison with a complexity penalty (2k - 2·loglik): Exponential has
# k=1, Gamma and Weibull have k=2. If the extra parameter buys no
# log-likelihood, ΔAIC lands near +2 and AIC hands the win back to the
# one-parameter model.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Hourly trip counts and inter-arrival gaps
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP02 Exercise 2.6: Count & Duration Models")
print("=" * 70)

trips = load_taxi_trips()
# TODO: Build the hourly-count series and the inter-arrival-gap series
# Hint: taxi_trips_per_hour(trips) and taxi_interarrival_seconds(trips)
counts = ____
gaps = ____

print(f"\n  Window: April 2024 (Singapore taxi trips)")
print(f"  Hourly counts:   {len(counts):,} (date, hour) cells")
print(
    f"    mean={counts.mean():.2f} trips/hour, "
    f"var={counts.var(ddof=1):.2f}, "
    f"dispersion ratio Var/Mean={counts.var(ddof=1) / counts.mean():.2f}"
)
print(f"  Inter-arrival gaps: {len(gaps):,} consecutive-trip intervals")
print(
    f"    mean={gaps.mean():.0f}s, sd={gaps.std(ddof=1):.0f}s, "
    f"SD/Mean={gaps.std(ddof=1) / gaps.mean():.2f}"
)
# INTERPRETATION: Poisson predicts Var/Mean = 1. Within one month the
# hourly counts are mildly UNDER-dispersed (Var/Mean < 1): the fleet is
# finite and driver shifts are scheduled, so supply smooths the peaks a
# pure Poisson would produce. Across MANY months the same computation
# flips over-dispersed because the underlying rate drifts over years —
# dispersion is a property of the window you choose. The gaps, by
# contrast, show SD/Mean ≈ 1, the Exponential signature.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(counts) > 100, "Need many (date, hour) cells for a count model"
assert len(gaps) > 100, "Need many gaps for a duration model"
assert (gaps > 0).all(), "Inter-arrival gaps must be strictly positive"
assert counts.mean() > 0, "Mean count must be positive"
print("\n✓ Checkpoint 1 passed — count and duration series built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Poisson and Exponential MLEs + AIC comparison
# ════════════════════════════════════════════════════════════════════════

# Poisson MLE for hourly counts (closed form: λ̂ = mean)
# TODO: Fit the Poisson model with the shared helper
# Hint: poisson_mle(counts) returns a dict with keys lambda, loglik, n, k
pois = ____
print("=== Poisson fit — trips per (date, hour) ===")
print(f"λ̂ = {pois['lambda']:.3f} trips/hour  (closed form: λ̂ = x̄)")
print(f"log-likelihood = {pois['loglik']:.1f}")

# Duration models for inter-arrival gaps
# TODO: Fit Exponential, Gamma and Weibull to the gaps
# Hint: exponential_mle(gaps), gamma_mle(gaps), weibull_mle(gaps)
expo = ____
gam = ____
wei = ____

print("\n=== Duration fits — seconds between trips ===")
print(f"Exponential: λ̂ = {expo['lambda']:.5f}/s (mean wait {1 / expo['lambda']:.0f}s)")
print(f"Gamma:       shape={gam['shape']:.3f}, scale={gam['scale']:.1f}")
print(f"Weibull:     shape={wei['shape']:.3f}, scale={wei['scale']:.1f}")

# AIC comparison — lower is better; ΔAIC > 10 = strong evidence
print(f"\n{'Model':<13} {'k':>3} {'loglik':>12} {'AIC':>12} {'ΔAIC':>8}")
print("-" * 50)
duration_fits = {"Exponential": expo, "Gamma": gam, "Weibull": wei}
for name, f in duration_fits.items():
    # TODO: Compute AIC and BIC for this fit
    # Hint: aic(f["k"], f["loglik"]) and bic(f["k"], f["loglik"], f["n"])
    f["aic"] = ____
    f["bic"] = ____
best_aic = min(duration_fits.values(), key=lambda f: f["aic"])["aic"]
for name, f in duration_fits.items():
    print(
        f"{name:<13} {f['k']:>3} {f['loglik']:>12.1f} "
        f"{f['aic']:>12.1f} {f['aic'] - best_aic:>8.1f}"
    )
best_name = min(duration_fits, key=lambda n: duration_fits[n]["aic"])
print(f"\nBest duration model by AIC: {best_name}")
# INTERPRETATION: Gamma/Weibull shape ≈ 1 means "nearly Exponential" —
# and the log-likelihood barely moves, so ΔAIC ≈ +2 (the penalty for one
# unused parameter) and AIC hands the win back to Exponential. This is
# AIC working as intended: complexity must buy real fit. When the shape
# is far from 1 AND ΔAIC > 10, the extra parameter is earning its keep
# and the one-parameter story would be a fiction.

# ── Log the fit to ExperimentTracker ─────────────────────────────────
# TODO: Log this fit — experiment, run name, params, and metrics
# Hint: track_train_run(experiment="mlfp02_ex2_06_count_duration",
#   run_name="poisson_exponential_aic", params={...}, metrics={...})
# Metrics to log: poisson_lambda, dispersion_ratio, exponential_lambda,
# gamma_shape, aic_exponential, aic_gamma, aic_weibull
run_id = track_train_run(
    experiment="mlfp02_ex2_06_count_duration",
    run_name="poisson_exponential_aic",
    params={
        "count_model": "Poisson(closed-form MLE)",
        "duration_models": "Exponential,Gamma,Weibull",
        "window": "2024-04",
    },
    metrics={
        "poisson_lambda": ____,
        "dispersion_ratio": ____,
        "exponential_lambda": ____,
        "gamma_shape": ____,
        "aic_exponential": ____,
        "aic_gamma": ____,
        "aic_weibull": ____,
    },
)
print(f"Logged training run to ExperimentTracker (run {run_id})")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert abs(pois["lambda"] - counts.mean()) < 1e-9, "Poisson λ̂ must equal the mean"
assert abs(expo["lambda"] - 1.0 / gaps.mean()) < 1e-12, "Exponential λ̂ = 1/mean"
assert gam["shape"] > 0 and wei["shape"] > 0, "Shape parameters must be positive"
assert all(np.isfinite(f["aic"]) for f in duration_fits.values()), "AIC must be finite"
print("\n✓ Checkpoint 2 passed — MLEs and AIC comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Observed vs fitted distributions
# ════════════════════════════════════════════════════════════════════════

# -- Plot 1: hourly counts vs fitted Poisson PMF --
max_count = int(counts.max())
k_grid = np.arange(0, max_count + 1)
observed_pmf = np.array([(counts == k).mean() for k in k_grid])
# TODO: Evaluate the fitted Poisson PMF on k_grid
# Hint: stats.poisson.pmf(k_grid, pois["lambda"])
fitted_pmf = ____

fig1 = go.Figure()
fig1.add_trace(
    go.Bar(x=k_grid, y=observed_pmf, name="Observed", opacity=0.7)
)
fig1.add_trace(
    go.Scatter(
        x=k_grid,
        y=fitted_pmf,
        mode="lines+markers",
        name=f"Poisson(λ={pois['lambda']:.2f})",
        line={"color": "red"},
    )
)
_dispersion = counts.var(ddof=1) / counts.mean()
_dispersion_label = (
    "over-dispersed" if _dispersion > 1.1
    else "under-dispersed" if _dispersion < 0.9
    else "near-Poisson"
)
fig1.update_layout(
    title=(
        "Hourly Trip Counts vs Poisson Fit "
        f"(Var/Mean = {_dispersion:.2f} — {_dispersion_label})"
    ),
    xaxis_title="Trips in one hour",
    yaxis_title="Proportion of (date, hour) cells",
    height=420,
)
save_figure(fig1, "poisson_counts.html")
print("Saved: poisson_counts.html")

# -- Plot 2: gaps histogram vs three fitted PDFs --
t_grid = np.linspace(0, np.percentile(gaps, 99), 400)
fig2 = go.Figure()
fig2.add_trace(
    go.Histogram(
        x=gaps,
        histnorm="probability density",
        nbinsx=60,
        name="Observed gaps",
        opacity=0.6,
    )
)
fig2.add_trace(
    go.Scatter(
        x=t_grid,
        y=stats.expon.pdf(t_grid, scale=1.0 / expo["lambda"]),
        name=f"Exponential (AIC Δ={expo['aic'] - best_aic:.1f})",
        line={"color": "red"},
    )
)
# TODO: Add the Gamma PDF trace — stats.gamma.pdf(t_grid, gam["shape"], loc=0, scale=gam["scale"])
fig2.add_trace(____)
# TODO: Add the Weibull PDF trace — stats.weibull_min.pdf(t_grid, wei["shape"], loc=0, scale=wei["scale"])
fig2.add_trace(____)
fig2.update_layout(
    title="Inter-Arrival Gaps vs Fitted Duration Models (April 2024)",
    xaxis_title="Seconds between consecutive pickups",
    yaxis_title="Density",
    height=420,
)
save_figure(fig2, "duration_models.html")
print("Saved: duration_models.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert abs(fitted_pmf.sum() - 1.0) < 0.01, "Poisson PMF must sum to ~1 over the grid"
assert np.isfinite(stats.gamma.pdf(100.0, gam["shape"], scale=gam["scale"])), (
    "Fitted Gamma must evaluate on the grid"
)
print("\n✓ Checkpoint 3 passed — visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Taxi-Stand Staffing — Waits and Arrivals
# ════════════════════════════════════════════════════════════════════════
# A taxi-stand operator (anonymised) staffs a phone dispatch line. Two
# operational questions:
#   1. "What is the chance a driver waits more than 10 minutes for the
#      next trip?" → survival function of the duration model,
#      P(T > 600s) = 1 - F(600).
#   2. "What is the chance of a surge hour (≥ 6 trips) that needs a
#      second dispatcher?" → Poisson tail, P(K ≥ 6) = 1 - F(5).
# Model choice matters: the AIC winner and the one-parameter Exponential
# give different answers when the shape is far from 1.

print("=== APPLICATION: Taxi-Stand Staffing ===")

wait_threshold_s = 600
# TODO: P(T > 600s) under the Exponential fit — use the survival function
# Hint: stats.expon.sf(wait_threshold_s, scale=1.0 / expo["lambda"])
p_wait_expo = ____
p_wait_gamma = float(
    stats.gamma.sf(wait_threshold_s, gam["shape"], loc=0, scale=gam["scale"])
)
p_wait_weibull = float(
    stats.weibull_min.sf(wait_threshold_s, wei["shape"], loc=0, scale=wei["scale"])
)
print(f"\nP(next trip arrives after > {wait_threshold_s // 60} min):")
print(f"  Exponential: {p_wait_expo:.3f}")
print(f"  Gamma:       {p_wait_gamma:.3f}  (AIC winner: {best_name})")
print(f"  Weibull:     {p_wait_weibull:.3f}")
print(
    "  Spread across models is model risk — report the AIC winner "
    "and the one-parameter benchmark together."
)

surge_k = 6
# TODO: Poisson tail P(K ≥ 6) = 1 - F(5) under the fitted λ̂
# Hint: stats.poisson.sf(surge_k - 1, pois["lambda"])
p_surge = ____
print(f"\nP(surge hour: ≥ {surge_k} trips in one hour) = {p_surge:.3f}")
print(
    f"  Var/Mean = {counts.var(ddof=1) / counts.mean():.2f}: "
    + (
        "over-dispersed — the single-λ Poisson understates surges"
        if counts.var(ddof=1) / counts.mean() > 1.1
        else "under-dispersed in this window — the Poisson tail is conservative"
    )
)
# TODO: Empirical share of hours with at least surge_k trips
# Hint: (counts >= surge_k).mean()
observed_surge = ____
print(f"  Observed share of surge hours: {observed_surge:.3f}")
if observed_surge > p_surge:
    print(
        "  → The Poisson tail is too thin. Staffing to the Poisson "
        "probability would leave the stand short-handed in rush hours; "
        "the honest fix is to fit λ separately by hour-of-day."
    )
else:
    print(
        "  → The single-λ Poisson is conservative here — but λ drifts "
        "with hour-of-day, so a per-hour fit is still the honest model "
        "before setting a real roster."
    )

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert 0 < p_wait_expo < 1 and 0 < p_wait_gamma < 1, "Survival probs in (0,1)"
assert 0 <= p_surge <= 1, "Poisson tail must be a probability"
assert 0 <= observed_surge <= 1, "Observed share must be a proportion"
print("\n✓ Checkpoint 4 passed — staffing application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED (2.6 — Count & Duration Models)")
print("═" * 70)
print(
    """
  ✓ One Poisson process implies TWO distributions: Poisson for counts,
    Exponential for waiting times — same λ, two views
  ✓ Closed-form MLEs: λ̂_Poisson = x̄, λ̂_Exponential = 1/x̄
  ✓ Dispersion check (Var/Mean) exposes when one λ misdescribes the
    window — over-dispersion (rate drift) OR under-dispersion (supply
    ceilings) are both visible in one number
  ✓ AIC compares duration families honestly: shape ≈ 1 buys nothing,
    ΔAIC ≈ +2, and the one-parameter Exponential keeps the win
  ✓ Fitted models answer operational questions: P(wait > 10 min),
    P(surge hour) — and model spread quantifies model risk

  NEXT: In 07_lln_demo.py, you'll watch the Law of Large Numbers
  converge in real time — and meet the one distribution (Cauchy) where
  it refuses to.
"""
)

print("\n✓ Exercise 2.6 complete — Count & Duration Models")
