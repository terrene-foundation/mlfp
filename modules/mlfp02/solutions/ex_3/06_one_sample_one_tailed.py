# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 3.6: One-Sample and One-Tailed Tests
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Run a one-sample t-test: a sample mean against a FIXED benchmark μ₀
#   - Build the benchmark honestly (leave-one-out: never test a town
#     against an island mean that contains the town itself)
#   - Convert between two-tailed and one-tailed p-values, and state when
#     a directional (one-tailed) test is legitimate
#   - See the power bargain: one-tailed tests reject at t > 1.645 instead
#     of |t| > 1.96 — a smaller MDE for the same n
#   - Recognise p-hacking: choosing the tail AFTER seeing the sign
#
# PREREQUISITES: 02_hypothesis_testing.py (two-sample tests, p-values)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — fixed benchmarks and directional hypotheses
#   2. Build — leave-one-out island benchmark; Punggol and Tampines sets
#   3. Train — one-sample t, two-tailed vs one-tailed, critical values
#   4. Visualise — H0 sampling distribution with both rejection regions
#   5. Apply — pre-registered directional question for a housing analyst
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy import stats

from shared.mlfp02.ex_1 import fmt_money, load_hdb_4room
from shared.mlfp02.ex_3 import (
    RANDOM_SEED,
    OUTPUT_DIR,
    print_header,
    track_train_run,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Fixed Benchmarks and Directional Hypotheses
# ════════════════════════════════════════════════════════════════════════
# So far every test compared TWO samples (treatment vs control). A
# ONE-SAMPLE test compares ONE sample against a fixed number μ₀:
#
#   H0: μ = μ₀        T = (x̄ - μ₀) / (s/√n)   ~  t_{n-1} under H0
#
# Benchmarks must be built honestly. "Is Punggol below the island mean?"
# cannot use an island mean that CONTAINS Punggol — the town would be
# part of its own reference, shrinking the gap. Leave-one-out: the
# benchmark for town X is the mean of everything EXCEPT town X.
#
# Tail choice is a second, separate decision:
#
#   TWO-TAILED  H1: μ ≠ μ₀   reject when |T| > 1.96   (α split 2.5%/2.5%)
#   ONE-TAILED  H1: μ > μ₀   reject when T > 1.645    (all 5% in one tail)
#               H1: μ < μ₀   reject when T < -1.645
#
# When the observed sign matches the pre-registered direction,
# p_one = p_two / 2 — the same evidence, half the p-value. That is
# exactly why one-tailed tests are tempting to reach for AFTER seeing
# the data, and exactly why they are only legitimate when the DIRECTION
# was fixed by the question before the data was touched:
#
#   LEGITIMATE:  "The regulator asks whether this estate trades BELOW
#                 the island benchmark" — the policy question is
#                 directional; an upside surprise is not actionable.
#   P-HACKING:   "The data leaned negative, so we tested H1: μ < μ₀" —
#                the tail was chosen by the data; α is doubled in secret.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Leave-one-out benchmark and the two town samples
# ════════════════════════════════════════════════════════════════════════

print_header("MLFP02 Exercise 3.6: One-Sample & One-Tailed Tests")

# Sentinel-price hygiene from M1/ex_8: resale prices are bounded
hdb = load_hdb_4room().filter(
    (pl.col("resale_price") >= 100_000) & (pl.col("resale_price") <= 5_000_000)
)
print(f"\n  4-room transactions (2020+, sentinel prices removed): {hdb.height:,}")


def town_sample(town: str) -> np.ndarray:
    return (
        hdb.filter(pl.col("town") == town)["resale_price"]
        .to_numpy()
        .astype(np.float64)
    )


def loo_benchmark(town: str) -> float:
    """Leave-one-out island mean: every 4-room town EXCEPT ``town``."""
    return float(
        hdb.filter(pl.col("town") != town)["resale_price"]
        .to_numpy()
        .astype(np.float64)
        .mean()
    )


punggol = town_sample("PUNGGOL")
mu0_punggol = loo_benchmark("PUNGGOL")

# Marginal case: a seeded n=220 Bukit Panjang subsample — the tail
# decision will actually flip the verdict here
rng = np.random.default_rng(RANDOM_SEED)
bukitpanjang_full = town_sample("BUKIT PANJANG")
bukitpanjang = rng.choice(bukitpanjang_full, size=220, replace=False)
mu0_bukitpanjang = loo_benchmark("BUKIT PANJANG")

print(f"\n  PUNGGOL  n={len(punggol):,}  mean={fmt_money(float(punggol.mean()))}  "
      f"benchmark(LOO)={fmt_money(mu0_punggol)}")
print(f"  BUKIT PANJANG n=220 (seeded)  mean={fmt_money(float(bukitpanjang.mean()))}  "
      f"benchmark(LOO)={fmt_money(mu0_bukitpanjang)}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(punggol) > 100, "Punggol sample should be the full town set"
assert len(bukitpanjang) == 220, "Bukit Panjang working sample must be n = 220"
assert mu0_punggol > 0 and mu0_bukitpanjang > 0, "Benchmarks must be positive"
print("\n>>> Checkpoint 1 passed — samples and LOO benchmarks built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: One-sample t, two tails vs one tail
# ════════════════════════════════════════════════════════════════════════

print("=== Case A: Punggol (full sample) — is the mean different? ===")
t_p, p_two_p = stats.ttest_1samp(punggol, mu0_punggol)
print(f"H0: μ_Punggol = {fmt_money(mu0_punggol)} (LOO island mean)")
print(f"t = {t_p:.2f}, df = {len(punggol) - 1}, two-tailed p = {p_two_p:.2e}")
print(f"Gap: {fmt_money(float(punggol.mean()) - mu0_punggol)} below the benchmark")
print("Overwhelming evidence either way — the tail choice is irrelevant here.")

print("\n=== Case B: Bukit Panjang (n=220) — where the tail choice decides ===")
t_t, p_two_t = stats.ttest_1samp(bukitpanjang, mu0_bukitpanjang)
# Pre-registered direction: H1 says Bukit Panjang is BELOW the benchmark.
# One-tailed p = half the two-tailed p ONLY when the sign matches H1.
p_one_t = p_two_t / 2 if t_t < 0 else 1 - p_two_t / 2
print(f"H0: μ_BukitPanjang = {fmt_money(mu0_bukitpanjang)} (LOO island mean)")
print(f"t = {t_t:.2f}, df = {len(bukitpanjang) - 1}")
print(f"Two-tailed p = {p_two_t:.4f}  → {'reject' if p_two_t < 0.05 else 'fail to reject'} at α=0.05")
print(f"One-tailed p (H1: below) = {p_one_t:.4f}  → {'reject' if p_one_t < 0.05 else 'fail to reject'} at α=0.05")
print(
    "\nSame data, same t — the ONLY difference is where the 5% rejection\n"
    "region sits. One tail puts all of it on the pre-registered side, so\n"
    "the critical value drops from ≈1.97 to ≈1.65:"
)
crit_two = stats.t.ppf(1 - 0.025, df=len(bukitpanjang) - 1)
crit_one = stats.t.ppf(1 - 0.05, df=len(bukitpanjang) - 1)
print(f"  two-tailed critical |t|: {crit_two:.3f}")
print(f"  one-tailed critical t:  {crit_one:.3f}")
print(f"  observed t:             {t_t:.3f}  ({'between them — tail decides' if crit_one < abs(t_t) < crit_two else 'decisive either way'})")

# The power bargain: for the same n, one-tailed detects a smaller effect
se_t = float(bukitpanjang.std(ddof=1) / np.sqrt(len(bukitpanjang)))
mde_two = crit_two * se_t
mde_one = crit_one * se_t
print(f"\n  SE(x̄) = {fmt_money(se_t)}")
print(f"  Smallest detectable gap at this n: two-tailed {fmt_money(mde_two)}, "
      f"one-tailed {fmt_money(mde_one)}")

# ── Log the tests to ExperimentTracker ───────────────────────────────
run_id = track_train_run(
    experiment="mlfp02_ex3_06_one_sample_one_tailed",
    run_name="punggol_tampines_vs_loo_benchmark",
    params={
        "test": "one-sample t",
        "benchmark": "leave-one-out island mean",
        "preregistered_direction": "bukit_panjang_below",
    },
    metrics={
        "punggol_t": float(t_p),
        "punggol_p_two": float(p_two_p),
        "bukitpanjang_t": float(t_t),
        "bukitpanjang_p_two": float(p_two_t),
        "bukitpanjang_p_one_below": float(p_one_t),
        "bukitpanjang_mde_two": float(mde_two),
        "bukitpanjang_mde_one": float(mde_one),
    },
)
print(f"\nLogged training run to ExperimentTracker (run {run_id})")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert abs(p_one_t - p_two_t / 2) < 1e-12, (
    "Sign matches H1: one-tailed p must equal half the two-tailed p"
)
assert crit_one < crit_two, "One-tailed critical value must be smaller"
assert mde_one < mde_two, "One-tailed MDE must be smaller at the same n"
print("\n>>> Checkpoint 2 passed — tail mechanics verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: H0 sampling distribution, both rejection regions
# ════════════════════════════════════════════════════════════════════════

df_t = len(bukitpanjang) - 1
t_grid = np.linspace(-4, 4, 800)
t_pdf = stats.t.pdf(t_grid, df=df_t)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t_grid, y=t_pdf, name="t under H0 (df=59)",
                         line={"color": "black"}))
# Two-tailed rejection: |t| > critical
mask_two = np.abs(t_grid) >= crit_two
fig.add_trace(
    go.Scatter(
        x=np.concatenate([t_grid[mask_two], t_grid[mask_two][::-1]]),
        y=np.concatenate([t_pdf[mask_two], np.zeros(mask_two.sum())]),
        fill="toself",
        fillcolor="rgba(0,0,255,0.25)",
        line={"width": 0},
        name=f"Two-tailed rejection (|t| > {crit_two:.3f})",
        hoverinfo="skip",
    )
)
# One-tailed rejection: t < -1.645-equivalent
mask_one = t_grid <= -crit_one
fig.add_trace(
    go.Scatter(
        x=np.concatenate([t_grid[mask_one], t_grid[mask_one][::-1]]),
        y=np.concatenate([t_pdf[mask_one], np.zeros(mask_one.sum())]),
        fill="toself",
        fillcolor="rgba(255,0,0,0.35)",
        line={"width": 0},
        name=f"One-tailed rejection (t < {-crit_one:.3f})",
        hoverinfo="skip",
    )
)
fig.add_vline(x=t_t, line_color="darkgreen",
              annotation_text=f"observed t = {t_t:.2f}")
fig.update_layout(
    title=(
        f"Bukit Panjang n=220: two-tailed p={p_two_t:.3f} vs one-tailed p={p_one_t:.3f} — "
        "the observed t lands between the two critical values"
    ),
    xaxis_title="t statistic under H0",
    yaxis_title="Density",
    height=450,
)
fig_path = OUTPUT_DIR / "one_tailed_regions.html"
fig.write_html(str(fig_path))
print(f"Saved: {fig_path}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert fig_path.exists(), "Figure must be written"
print("\n>>> Checkpoint 3 passed — visualisation saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: A Pre-Registered Directional Question
# ════════════════════════════════════════════════════════════════════════
# A housing analyst (anonymised) studies affordability spillover: their
# pre-registered hypothesis is that a large mature-ish heartland estate
# trades BELOW the island benchmark as buyers arbitrage newer towns.
# The question is directional BY CONSTRUCTION — an upside surprise is a
# different research paper, not evidence for this one.
#
# Decision table for the Bukit Panjang n=220 evidence:

print("\n--- Application: pre-registered directional test ---")
print(f"  Pre-registered H1: μ_BukitPanjang < {fmt_money(mu0_bukitpanjang)}")
print(f"  Observed: t = {t_t:.2f}, one-tailed p = {p_one_t:.4f}")
if p_one_t < 0.05:
    print("  Verdict (legitimate one-tailed): reject H0 — Bukit Panjang "
          "trades below the island benchmark at the 5% level.")
else:
    print("  Verdict: fail to reject H0.")
print(f"  Had the same data been tested two-tailed: p = {p_two_t:.4f} — "
      "no rejection.")
print(
    "\n  The same arithmetic in the other direction is the tell of\n"
    "  p-hacking: had t come out POSITIVE, the one-tailed p for 'below'\n"
    f"  would have been {1 - p_two_t / 2:.3f} — no salvage. Choosing the\n"
    "  tail after seeing the sign doubles α in secret: 5% becomes 10%."
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert 0 < p_one_t < 1 and 0 < p_two_t < 1, "p-values must be in (0,1)"
assert (1 - p_two_t / 2) > 0.5, (
    "Wrong-direction one-tailed p must be large — no salvage after a sign flip"
)
print("\n>>> Checkpoint 4 passed — application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED (3.6 — One-Sample & One-Tailed Tests)")
print("=" * 70)
print(
    """
  ✓ One-sample t: T = (x̄ - μ₀)/(s/√n) against a FIXED benchmark
  ✓ Benchmarks are built honestly: leave-one-out, or the town is part
    of its own reference and the gap shrinks
  ✓ Two-tailed vs one-tailed: same data, same t — only the rejection
    region moves (≈1.97 → ≈1.65), so the MDE shrinks for the same n
  ✓ Sign-match rule: p_one = p_two/2 only when the data lands on the
    pre-registered side
  ✓ One-tailed is legitimate when the QUESTION is directional; chosen
    after seeing the sign, it is p-hacking with α secretly doubled

  NEXT: Exercise 4 puts testing to work end-to-end — experiment design,
  SRM detection, Welch's t-test and validity threats on real A/B data.
"""
)

print("\n>>> Exercise 3.6 complete — One-Sample & One-Tailed Tests")
