# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 2.7: Law of Large Numbers — Watching Convergence
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Distinguish the LLN (the running mean CONVERGES to E[X]) from the
#     CLT (the sampling distribution is Normal) — different theorems
#   - Watch the running mean of taxi fares converge in real time
#   - Quantify convergence speed: the band shrinks like σ/√n
#   - Meet the Cauchy distribution, where the LLN FAILS (no finite mean)
#   - Size a sample: how many trips to pin the mean fare within $0.50?
#
# PREREQUISITES: 01_clt_sampling.py (sampling distributions, CLT)
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — what the LLN does and does not say
#   2. Build — running-mean trajectories on fares and synthetic draws
#   3. Train — convergence bands, n-for-precision, the Cauchy failure
#   4. Visualise — running mean vs n with the σ/√n band; Cauchy contrast
#   5. Apply — sample-size planning for a fare benchmark
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.mlfp02.ex_2 import (
    DEFAULT_SEED,
    load_taxi_trips,
    save_figure,
    taxi_fares,
    track_train_run,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — What the Law of Large Numbers Does and Does Not Say
# ════════════════════════════════════════════════════════════════════════
# The LLN: as n → ∞, the sample mean x̄ₙ converges to the population mean
# E[X] — PROVIDED E[X] exists and is finite. It is a theorem about ONE
# growing sequence: keep sampling, and your running average stops moving.
#
# The CLT is a different theorem: it describes the SHAPE of the sampling
# distribution of x̄ₙ across many repeated samples of size n (Normal,
# spread σ/√n). The LLN says the mean settles; the CLT says how wildly
# it wobbles on the way.
#
#   Convergence speed: |x̄ₙ - E[X]| shrinks like σ/√n.
#   Quadrupling n halves the wobble. There are no shortcuts.
#
# The proviso matters: the Cauchy distribution has no finite mean, so the
# LLN does not apply — the running mean NEVER settles. You will see it
# lurch at n=10,000 as violently as at n=10. Any fat-tailed financial
# series should make you ask: does E[X] even exist here?


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Running-mean trajectories
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP02 Exercise 2.7: Law of Large Numbers")
print("=" * 70)

trips = load_taxi_trips()
fares = taxi_fares(trips)
n_fares = len(fares)
true_mean_fare = float(fares.mean())
fare_sd = float(fares.std(ddof=1))

print(f"\n  Population: {n_fares:,} Singapore taxi fares")
print(f"  Population mean fare: ${true_mean_fare:.2f}  (sd ${fare_sd:.2f})")

# Shuffle with a fixed seed so the running mean is one honest draw order
rng = np.random.default_rng(DEFAULT_SEED)
order = rng.permutation(n_fares)
shuffled = fares[order]
# TODO: Running mean — cumulative sum divided by the number seen so far
# Hint: np.cumsum(shuffled) / np.arange(1, n_fares + 1)
running_mean = ____

# Synthetic companions: Exponential (finite mean — LLN holds) and
# Cauchy (NO finite mean — LLN fails)
n_synth = 20_000
exp_draws = rng.exponential(scale=5.0, size=n_synth)
cauchy_draws = rng.standard_cauchy(size=n_synth)
# TODO: Running means for the two synthetic series
running_exp = ____
running_cauchy = ____

print(f"\n  Running mean after n=10:    ${running_mean[9]:.2f}")
print(f"  Running mean after n=100:   ${running_mean[99]:.2f}")
print(f"  Running mean after n=1,000: ${running_mean[999]:.2f}")
print(f"  Running mean after n={n_fares:,}: ${running_mean[-1]:.2f}")
# INTERPRETATION: the running mean starts far from the population mean
# and is pulled toward it. Not monotonically — early draws can drag it
# either way — but the wobble DECAYS. That decay is the LLN in action.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(running_mean) == n_fares, "Running mean must have one value per draw"
assert abs(running_mean[-1] - true_mean_fare) < 1e-9, (
    "The running mean at n=N must equal the population mean"
)
assert running_exp[-1] > 0, "Exponential running mean must be positive"
print("\n✓ Checkpoint 1 passed — running means built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Convergence band, sample size, the Cauchy failure
# ════════════════════════════════════════════════════════════════════════

# The ±2σ/√n band: where the running mean lives with ~95% probability
n_axis = np.arange(1, n_fares + 1)
# TODO: The convergence band 2σ/√n at every n
# Hint: 2.0 * fare_sd / np.sqrt(n_axis)
band = ____

# At what n does the band first narrow to ±$0.50?
target_margin = 0.50
# TODO: Invert the band — solve 2σ/√n = margin for n (round UP)
# Hint: n = (2σ / margin)², use int(np.ceil(...))
n_for_margin = ____
print("=== Convergence band ±2σ/√n ===")
print(f"Fare SD: ${fare_sd:.2f}")
print(f"Band at n=100:   ±${band[99]:.2f}")
print(f"Band at n=1,000: ±${band[999]:.2f}")
print(f"Band at n=10,000: ±${band[9_999]:.2f}")
print(f"\nTo pin the mean fare within ±${target_margin:.2f}: n ≥ {n_for_margin:,}")

# How far is the ACTUAL running mean from the limit at those checkpoints?
for n_check in (100, 1_000, 10_000, n_fares):
    err = abs(running_mean[n_check - 1] - true_mean_fare)
    print(
        f"  n={n_check:>7,}: |x̄ₙ - μ| = ${err:.3f} "
        f"(band ±${band[n_check - 1]:.2f})"
    )

# Cauchy: no finite mean — the running mean never converges
cauchy_late_jumps = float(
    np.max(np.abs(np.diff(running_cauchy[-2_000:])))
)
exp_late_jumps = float(np.max(np.abs(np.diff(running_exp[-2_000:]))))
print(f"\n=== Where the LLN fails ===")
print(f"Largest single-step move of the running mean, last 2,000 draws:")
print(f"  Exponential(5): {exp_late_jumps:.4f}  (settling)")
print(f"  Cauchy:         {cauchy_late_jumps:.4f}  (still lurching)")
print(
    "  Cauchy has no finite mean, so there is nothing to converge to —\n"
    "  one extreme draw can move the average more at n=20,000 than an\n"
    "  Exponential draw moves it at n=100."
)

# ── Log the run to ExperimentTracker ─────────────────────────────────
# TODO: Log this run — population mean, fare sd, n_for_half_dollar_margin,
# band_at_1000, and both late-jump metrics
# Hint: track_train_run(experiment=..., run_name=..., params={...}, metrics={...})
run_id = track_train_run(
    experiment="mlfp02_ex2_07_lln",
    run_name="running_mean_convergence",
    params={"series": "taxi_fares+exponential+cauchy", "seed": str(DEFAULT_SEED)},
    metrics={
        "population_mean_fare": ____,
        "fare_sd": ____,
        "n_for_half_dollar_margin": ____,
        "band_at_1000": ____,
        "cauchy_late_jump": ____,
        "exponential_late_jump": ____,
    },
)
print(f"Logged training run to ExperimentTracker (run {run_id})")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert n_for_margin > 0, "Required n must be positive"
assert band[-1] < band[0], "The band must narrow as n grows"
assert cauchy_late_jumps > exp_late_jumps, (
    "Cauchy's running mean must keep lurching long after Exponential settles"
)
print("\n✓ Checkpoint 2 passed — convergence quantified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Running means with the σ/√n band; Cauchy contrast
# ════════════════════════════════════════════════════════════════════════

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "LLN holds: running mean of taxi fares",
        "LLN fails: running mean of Cauchy draws",
    ],
)

stride = max(1, n_fares // 4_000)
fig.add_trace(
    go.Scatter(
        x=n_axis[::stride],
        y=running_mean[::stride],
        mode="lines",
        name="Running mean (fares)",
        line={"color": "blue"},
    ),
    row=1,
    col=1,
)
# TODO: Shaded ±2σ/√n band — an x,y polygon that goes forward along the
# upper edge (μ + band) and back along the lower edge (μ - band)
# Hint: x = concat(n_axis[::stride], n_axis[::stride][::-1]),
#       y = concat((true_mean_fare + band)[::stride],
#                  (true_mean_fare - band)[::stride][::-1]),
#       fill="toself", fillcolor="rgba(0,100,255,0.15)", line width 0
fig.add_trace(____, row=1, col=1)
fig.add_hline(
    y=true_mean_fare,
    line_dash="dash",
    line_color="red",
    annotation_text=f"μ = ${true_mean_fare:.2f}",
    row=1,
    col=1,
)

stride_c = max(1, n_synth // 4_000)
fig.add_trace(
    go.Scatter(
        x=np.arange(1, n_synth + 1)[::stride_c],
        y=running_cauchy[::stride_c],
        mode="lines",
        name="Running mean (Cauchy)",
        line={"color": "darkorange"},
    ),
    row=1,
    col=2,
)
# TODO: Add the Exponential running-mean trace on the same panel
fig.add_trace(____, row=1, col=2)
fig.add_hline(
    y=5.0, line_dash="dash", line_color="green", row=1, col=2
)
fig.update_xaxes(title_text="n (draws)", type="log", row=1, col=1)
fig.update_xaxes(title_text="n (draws)", type="log", row=1, col=2)
fig.update_yaxes(title_text="Running mean ($)", row=1, col=1)
fig.update_layout(height=450, title="The Law of Large Numbers — and Its Exception")
save_figure(fig, "lln_convergence.html")
print("Saved: lln_convergence.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert band[0] > band[-1] > 0, "Band ordering must shrink monotonically"
print("\n✓ Checkpoint 3 passed — visualisation saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Sample-Size Planning for a Fare Benchmark
# ════════════════════════════════════════════════════════════════════════
# A transport-research team (anonymised) publishes a monthly "typical
# fare" benchmark. Their rule: the estimate must sit within ±$0.50 of
# the true mean with ~95% confidence. The LLN band inverts directly into
# a sample-size requirement:
#
#   ±2σ/√n ≤ margin   ⟺   n ≥ (2σ / margin)²
#
# The catch: σ is unknown before sampling. Practice: pilot a few hundred
# trips, estimate σ, compute n, then check the pilot was large enough
# that σ̂ itself is stable.

print("=== APPLICATION: Fare Benchmark Sample Size ===")
pilot_n = 400
# TODO: Estimate σ from the first pilot_n shuffled fares
# Hint: shuffled[:pilot_n].std(ddof=1)
pilot_sd = ____
# TODO: Required n from the pilot σ̂
n_from_pilot = ____
print(f"\nTarget margin: ±${target_margin:.2f} (~95% band)")
print(f"Pilot of {pilot_n} trips: σ̂ = ${pilot_sd:.2f} (population σ = ${fare_sd:.2f})")
print(f"Required n from pilot σ̂: {n_from_pilot:,}")
print(f"Required n from true σ:  {n_for_margin:,}")
print(
    f"Available trips: {n_fares:,} — "
    f"{'sufficient' if n_fares >= n_for_margin else 'INSUFFICIENT'}"
)

margins = [1.00, 0.50, 0.25]
print(f"\n{'Margin':>8} {'Required n':>12}")
for m in margins:
    print(f"±${m:>5.2f} {int(np.ceil((2 * fare_sd / m) ** 2)):>12,}")
print(
    "  Halving the margin QUADRUPLES the sample — precision is expensive.\n"
    "  That √n penalty is the LLN's fine print: convergence is guaranteed,\n"
    "  but it is slow, and no estimator escapes it."
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert abs(n_from_pilot - (2.0 * pilot_sd / target_margin) ** 2) < 1.0, (
    "Pilot-based n must follow n = (2σ̂/margin)²"
)
assert n_from_pilot > pilot_n, "A ±$0.50 benchmark must need more than the pilot"
print("\n✓ Checkpoint 4 passed — sample-size application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED (2.7 — Law of Large Numbers)")
print("═" * 70)
print(
    """
  ✓ LLN ≠ CLT: the LLN says ONE running mean settles at E[X]; the CLT
    describes the Normal SHAPE of the mean across repeated samples
  ✓ Convergence speed is σ/√n — quadrupling n halves the wobble
  ✓ The running mean of 50K real fares lands exactly on the population
    mean; the band ±2σ/√n predicts the wobble on the way
  ✓ Cauchy has no finite mean — the LLN simply does not apply, and the
    running mean lurches at n=20,000 as hard as at n=100
  ✓ Sample-size planning inverts the band: n ≥ (2σ/margin)², and the
    √n penalty makes every extra decimal of precision cost 100× the data

  NEXT: In Exercise 3, you'll turn sampling distributions into
  decisions — bootstrap CIs, hypothesis tests, and power analysis.
"""
)

print("\n✓ Exercise 2.7 complete — Law of Large Numbers")
