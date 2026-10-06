# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5.5: The Log-Linear Model — log(price) as Target
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - When to model log(y) instead of y: multiplicative relationships,
#     right-skewed positive targets, errors that scale with the price
#   - Interpret log-linear coefficients as semi-elasticities:
#     ×100 → % change in price per unit of the feature
#   - Back-transform honestly: exp(Xβ) is the MEDIAN prediction; the
#     mean prediction needs Duan's smearing factor, mean(exp(ε̂))
#   - Compare a levels model and a log model on the COMMON $ scale
#
# PREREQUISITES: 01_ols_from_scratch.py (OLS, t-stats, R², F-test)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — why money is often multiplicative, not additive
#   2. Build — log(resale_price) target on the cleaned HDB frame
#   3. Train — fit both models; smearing factor; $-scale R² comparison
#   4. Visualise — price vs log-price distributions; both predictions
#   5. Apply — valuing one flat when the fee is a PERCENTAGE of error
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from shared.mlfp02.ex_5 import (
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    TARGET,
    build_design_matrix,
    fit_ols,
    load_hdb_clean,
    print_coef_table,
    save_actual_vs_predicted,
    track_train_run,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Money Is Often Multiplicative, Not Additive
# ════════════════════════════════════════════════════════════════════════
# The levels model says: +1 sqm of floor area adds $β to the price —
# the SAME dollars for a $400K flat and a $1.2M flat. Market reality is
# closer to multiplicative: +1 sqm adds a PERCENTAGE. Percentage thinking
# is also how the industry talks: "that estate commands a 10% premium".
#
# The log-linear model makes percentages native:
#
#   ln(price) = β₀ + β₁·floor_area + β₂·storey + β₃·lease + ε
#
#   β₁ = 0.009  →  each sqm is associated with ≈ 0.9% higher price
#                  (exact: price multiplies by exp(β₁) = 1.00905)
#
# Side benefits: ln(price) is roughly symmetric when price is mildly
# right-skewed (this dataset: skew +0.39 → -0.56 — the transform pulls a
# touch past zero), so the Normal-flavoured OLS machinery (t-tests,
# F-tests, symmetric CIs) is on safer ground in log space. Errors ε are
# RELATIVE errors — being off by 10% on a $400K flat and a $1.2M flat
# counts the same, which is how a percentage-fee business experiences
# loss.
#
# THE BACK-TRANSFORMATION TRAP: exp(E[ln y]) is the MEDIAN of y, not the
# mean. For the mean you need Duan's smearing factor:
#
#   ŷ_mean = exp(Xβ̂) × mean(exp(ε̂))
#
# Skipping smearing systematically UNDER-predicts the mean — on skewed
# residuals the bias is several percent, larger than many margins.
#
# COMPARING MODELS: R² from the two fits are not comparable (different
# targets, different total variance). The honest comparison computes R²
# on the COMMON $ scale: exp(log-model predictions)×smearing vs actual
# price, against the levels model's predictions.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: The log-price target
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MLFP02 Exercise 5.5: The Log-Linear Model")
print("=" * 70)

hdb_all = load_hdb_clean()
# Sentinel-price hygiene from M1 / ex_8: a handful of records carry
# implausible prices ($10 sales, $9M entries). In LEVELS they drag the
# fit; in LOG space log(10) ≈ 2.3 sits ~11 units below the bulk at 13.6,
# creating a fake left tail that poisons the transform. Remove them as
# the data-cleaning lesson taught.
# TODO: Keep only resale prices in [100_000, 5_000_000]
# Hint: hdb_all.filter((pl.col(TARGET) >= 100_000) & (pl.col(TARGET) <= 5_000_000))
hdb = hdb_all.filter(____)
prices = hdb[TARGET].to_numpy().astype(np.float64)
# TODO: log-transform the target
# Hint: np.log(prices)
log_prices = ____

print(f"\n  Rows: {hdb.height:,} (dropped {hdb_all.height - hdb.height:,} sentinel prices)")
print(
    f"  Price:      mean ${prices.mean():,.0f}, median ${np.median(prices):,.0f}, "
    f"skew {float(((prices - prices.mean()) ** 3).mean() / prices.std() ** 3):.2f}"
)
print(
    f"  log(price): mean {log_prices.mean():.3f}, median {np.median(log_prices):.3f}, "
    f"skew {float(((log_prices - log_prices.mean()) ** 3).mean() / log_prices.std() ** 3):.2f}"
)
# INTERPRETATION: the transform moved skew from +0.39 to -0.56 — past
# zero, slightly overcorrected on this cleaned sample, but comparable in
# magnitude. Either scale is defensible for inference here; the real
# argument for the log model is the multiplicative READING and the
# percentage-error loss, not a dramatic symmetrisation.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert (prices > 0).all(), "log requires strictly positive prices"
assert np.isfinite(log_prices).all(), "log prices must be finite"
print("\n--- Checkpoint 1 passed --- log target built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Both models, smearing factor, common-scale R²
# ════════════════════════════════════════════════════════════════════════

X, y_levels, names = build_design_matrix(hdb)
# TODO: the log target for the SAME design matrix
y_log = ____

# TODO: Fit both models with the shared OLS helper
# Hint: fit_ols(X, y_levels) and fit_ols(X, y_log)
fit_levels = ____
fit_log = ____

print("=== Levels model: price = Xβ + ε ===")
print_coef_table(names, fit_levels)
print(f"R² = {fit_levels['R2']:.4f}   (target: dollars)")

print("\n=== Log model: log(price) = Xβ + ε ===")
print_coef_table(names, fit_log)
print(f"R² = {fit_log['R2']:.4f}   (target: log-dollars — NOT comparable to the levels R²)")

print("\n--- Semi-elasticity reading (log model) ---")
for i, name in enumerate(names[1:], start=1):
    # TODO: exact percentage effect per unit = (exp(β) - 1) × 100
    pct = ____
    print(
        f"  {name:<25} β={fit_log['beta'][i]:>8.5f}  →  {pct:+.2f}% price per unit"
    )

# Duan's smearing factor for honest mean-level predictions
# TODO: smearing = mean of exp(residuals) from the log fit
# Hint: float(np.exp(fit_log["residuals"]).mean())
smearing = ____
print(f"\nDuan's smearing factor: mean(exp(ε̂)) = {smearing:.4f}")
print(
    f"  Naive exp(Xβ̂) under-predicts the mean price by "
    f"{(smearing - 1) * 100:.1f}% if smearing is skipped"
)

# Common-scale comparison: R² measured in DOLLARS for both models
pred_levels = fit_levels["y_hat"]
# TODO: back-transform the log-model predictions WITH smearing
# Hint: np.exp(fit_log["y_hat"]) * smearing
pred_log_smeared = ____
pred_log_naive = np.exp(fit_log["y_hat"])


def r2_on_dollars(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


r2_levels = r2_on_dollars(y_levels, pred_levels)
r2_log_smeared = r2_on_dollars(y_levels, pred_log_smeared)
r2_log_naive = r2_on_dollars(y_levels, pred_log_naive)
mae_levels = float(np.mean(np.abs(y_levels - pred_levels)))
mae_log = float(np.mean(np.abs(y_levels - pred_log_smeared)))

print("\n=== Common $-scale comparison ===")
print(f"{'Model':<28} {'R²($)':>8} {'MAE($)':>12}")
print("-" * 50)
print(f"{'Levels (price)':<28} {r2_levels:>8.4f} {mae_levels:>12,.0f}")
print(f"{'Log + smearing':<28} {r2_log_smeared:>8.4f} {mae_log:>12,.0f}")
print(f"{'Log naive exp(Xβ̂)':<28} {r2_log_naive:>8.4f} "
      f"{float(np.mean(np.abs(y_levels - pred_log_naive))):>12,.0f}")

# ── Log both fits to ExperimentTracker ───────────────────────────────
# TODO: Log both fits — the $-scale R² of each model, the log-target R²,
# the smearing factor, and both MAEs
# Hint: track_train_run(experiment=..., run_name=..., params={...}, metrics={...})
run_id = track_train_run(
    experiment="mlfp02_ex5_05_log_price_model",
    run_name="levels_vs_loglinear_3features",
    params={
        "features": ",".join(NUMERIC_FEATURES),
        "models": "levels,log_linear",
        "back_transform": "duan_smearing",
    },
    metrics={
        "r2_levels_dollars": ____,
        "r2_log_dollars": ____,
        "r2_log_target": ____,
        "smearing_factor": ____,
        "mae_levels": ____,
        "mae_log_smeared": ____,
    },
)
print(f"\nLogged training run to ExperimentTracker (run {run_id})")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 0 < r2_levels < 1 and 0 < r2_log_smeared < 1, "R² values in (0,1)"
assert smearing >= 1.0, "Jensen: mean(exp(ε̂)) ≥ exp(mean(ε̂)) = 1"
assert abs(fit_log["residuals"].mean()) < 1e-6, "OLS residuals sum to ~0"
print("\n--- Checkpoint 2 passed --- both models fitted, smearing computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Distribution shift + both prediction scatters
# ════════════════════════════════════════════════════════════════════════

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["price (right-skewed)", "log(price) (near-symmetric)"],
)
fig.add_trace(go.Histogram(x=prices, nbinsx=80, name="price"), row=1, col=1)
# TODO: Add the log(price) histogram to the second panel
fig.add_trace(____, row=1, col=2)
fig.update_layout(
    title="Why the Log Transform: the Target's Shape", height=380, showlegend=False
)
fig.write_html(str(OUTPUT_DIR / "log_target_shape.html"))
print("Saved: log_target_shape.html")

save_actual_vs_predicted(
    y_levels, pred_levels, "Levels Model: Actual vs Predicted ($)", "avp_levels.html"
)
print("Saved: avp_levels.html")
# TODO: Save the actual-vs-predicted scatter for the smeared log model
# Hint: save_actual_vs_predicted(y_levels, pred_log_smeared, <title>, "avp_log_smeared.html")
____
print("Saved: avp_log_smeared.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (pred_log_smeared > 0).all(), "Back-transformed predictions must be positive"
print("\n--- Checkpoint 3 passed --- visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Valuing One Flat When the Fee Is a Percentage
# ════════════════════════════════════════════════════════════════════════
# An estate agency (anonymised) earns commission as a PERCENTAGE of the
# transacted price, and its internal SLA prices valuation errors the same
# way: a 10% miss costs the same fee-share whether the flat is $400K or
# $1.2M. Which model should the valuation desk use?
#
# Value a reference flat: 100 sqm, mid-storey (8th floor midpoint), 90
# years remaining lease.

print("=== APPLICATION: One Flat, Two Models, Percentage Fees ===")
x_flat = np.array([1.0, 100.0, 8.0, 90.0])
# TODO: levels-model prediction for the reference flat (dot product)
pred_flat_levels = ____
# TODO: log-model prediction for the reference flat, WITH smearing
pred_flat_log = ____
print(f"\nReference flat: 100 sqm, storey mid 8, lease 90y")
print(f"  Levels model:        ${pred_flat_levels:,.0f}")
print(f"  Log model (smeared): ${pred_flat_log:,.0f}")

# Percentage-error behaviour: where does each model err RELATIVELY?
# TODO: absolute percentage errors for both models
# Hint: np.abs(y_levels - pred_levels) / y_levels
rel_err_levels = ____
rel_err_log = ____
print(f"\nMean ABSOLUTE PERCENTAGE error (what a %-fee business feels):")
print(f"  Levels model: {100 * rel_err_levels.mean():.1f}%")
print(f"  Log model:    {100 * rel_err_log.mean():.1f}%")

# Error by price band — the levels model's weakness shows at the extremes
bands = [(0, 450_000), (450_000, 650_000), (650_000, 900_000), (900_000, 10_000_000)]
print(f"\n{'Price band':<22} {'Levels MAPE':>12} {'Log MAPE':>10}")
print("-" * 46)
for lo, hi in bands:
    m = (y_levels >= lo) & (y_levels < hi)
    if m.sum() == 0:
        continue
    print(
        f"${lo / 1e3:>5,.0f}K-${hi / 1e3:>5,.0f}K"
        f" {100 * rel_err_levels[m].mean():>11.1f}% {100 * rel_err_log[m].mean():>9.1f}%"
    )
print(
    "\n  Reading the bands honestly: with only THREE numeric features the\n"
    "  log model wins the $450K-$900K mid-market (where multiplicative\n"
    "  structure holds) and loses the sub-$450K band badly — cheap flats\n"
    "  in far-flung towns obey additive constraints (town floor prices)\n"
    "  that a pure percentage model cannot bend to. The levels model\n"
    "  takes this dataset overall on both $-R² and MAE. The choice is a\n"
    "  BUSINESS decision: match the model to how errors are charged —\n"
    "  and to WHICH segment your desk serves."
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert pred_flat_levels > 0 and pred_flat_log > 0, "Predictions must be positive"
assert 0 < rel_err_log.mean() < 1, "MAPE must be a fraction"
print("\n--- Checkpoint 4 passed --- application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED (5.5 — The Log-Linear Model)")
print("═" * 70)
print(
    """
  ✓ Money is multiplicative: log(price) turns dollars-per-sqm into
    percent-per-sqm — how markets and fees actually talk
  ✓ Semi-elasticities: β from a log-linear fit reads as
    (exp(β) - 1) × 100% price change per unit (here: +1.13% per sqm)
  ✓ The transform moved skew from +0.39 to -0.56 — mild overcorrection,
    not magic; the transform's real value is the READING, not the shape
  ✓ Back-transformation needs Duan's smearing: exp(Xβ̂) alone is the
    MEDIAN prediction and under-predicts the mean by (smearing - 1)
  ✓ Common-scale comparison settled it honestly: on three numeric
    features the LEVELS model wins $-R² (0.831 vs 0.777) and MAE; the
    log model wins only the $450K-$900K mid-market bands

  NEXT: In 06_kfold_cv.py, you'll stop trusting ONE train/test split
  and estimate generalisation error with k-fold cross-validation.
"""
)

print("\n✓ Exercise 5.5 complete — The Log-Linear Model")
