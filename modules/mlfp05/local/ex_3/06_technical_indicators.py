# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 3.6 — Technical Indicators as Features: RSI, MACD,
# Bollinger Bands (polars-native feature engineering)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Explain what RSI, MACD and Bollinger %B each measure — momentum,
#     trend divergence, and volatility position — in plain language
#   - Compute all three in POLARS with backward-looking windows only
#     (no lookahead leakage)
#   - Explain WHY indicators help sequence models: prices are
#     non-stationary, indicators are bounded or mean-reverting transforms
#   - Add indicator features to the windowed dataset and measure the
#     effect on forecast error honestly (it can help OR hurt)
#   - Evaluate a forecaster the way a trading desk does: directional
#     accuracy, not just RMSE
#   - Apply to an index-futures monitoring desk at a regional bank
#
# PREREQUISITES: M5/ex_3/02_lstm.py (LSTMRegressor, windowed datasets,
#   ExperimentTracker logging)
# ESTIMATED TIME: ~30 min
#
# DATASET: STI + APAC/global stocks via yfinance (2010-2024, parquet
#   cache). Indicators are computed from PAST closes only.
#
# PHASES:
#   1. THEORY  — What each indicator measures; stationarity; leakage
#   2. BUILD   — add_technical_indicators in polars; correctness checks
#   3. TRAIN   — LSTM on raw OHLCV vs indicator-augmented features
#   4. VISUALISE — Indicator panels + forecast overlays + horizon RMSE
#   5. APPLY   — Directional accuracy for an index-futures desk
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.mlfp05.ex_3 import (
    EPOCHS,
    FEATURES,
    FORECAST_HORIZON,
    HIDDEN_DIM,
    INDICATOR_FEATURES,
    OUTPUT_DIR,
    add_technical_indicators,
    get_visualizer,
    init_environment,
    load_stock_data,
    plot_horizon_error,
    plot_predictions,
    plot_time_series_overlay,
    plot_training_curves,
    prepare_dataloaders,
    register_best_model,
    setup_engines,
    train_model,
)

device = init_environment()


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why Indicators, and Why They Help Sequence Models
# ════════════════════════════════════════════════════════════════════════
# A price series is NON-STATIONARY: its mean and variance drift. Neural
# networks struggle with inputs whose distribution shifts — the same
# z-score means something different in 2012 and 2022.
#
# Technical indicators are HAND-CRAFTED STATIONARY TRANSFORMS, each
# compressing a different aspect of recent history:
#
#   RSI (Relative Strength Index, 14 days) — MOMENTUM
#     "Are recent gains overpowering recent losses?" Bounded in [0, 100]:
#     average gain vs average loss over two weeks, Wilder-smoothed.
#     >70 is conventionally "overbought", <30 "oversold". For the model:
#     a bounded input whose meaning does not drift with the price level.
#
#   MACD histogram (12/26/9) — TREND DIVERGENCE
#     "Is the fast trend pulling away from the slow trend, or closing?"
#     EMA(12) minus EMA(26), minus its own 9-day signal line. Mean-zero,
#     sign = momentum direction, magnitude = divergence strength.
#
#   Bollinger %B (20 days, 2 sd) — VOLATILITY POSITION
#     "Where is the price inside its own recent envelope?" 0.5 = at the
#     20-day mean; >1 = above the upper band; <0 = below the lower band.
#     Bounded-ish around 0.5 regardless of whether the index is at
#     3,000 or 30,000.
#
# LEAKAGE DISCIPLINE: every indicator at day t must be computable at the
# close of day t. Rolling and exponentially-weighted windows look only
# BACKWARD, so they are safe. (A centred rolling window would peek at
# future days — a classic leakage bug.) Normalisation statistics are
# computed on the train split only (build_dataset already does this).
#
# HONEST EXPECTATION: indicators add information the raw OHLCV window
# already contains in principle. Whether they help a 15-epoch LSTM is an
# empirical question — we measure it and report the sign we get.

print("=" * 70)
print("  PHASE 1 — THEORY: momentum, divergence, volatility position")
print("=" * 70)
print(
    """
  RSI(14)      momentum        bounded [0, 100]   "gains vs losses"
  MACD hist    trend divergence mean-zero         "fast vs slow trend"
  Bollinger %B volatility pos.  centred at 0.5    "where in the envelope"

  All three are BACKWARD-LOOKING transforms of the close — computable at
  the close of day t, safe from lookahead leakage. Prices drift; these
  transforms stay in a fixed range. That is why networks learn them
  faster than raw price levels.
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: indicators in polars, with a hand-check
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: add_technical_indicators (polars)")
print("=" * 70)

stock_data, primary, primary_df = load_stock_data()
# TODO: Append RSI/MACD/%B with the shared helper and drop the warm-up
#   null rows (first ~26 rows have incomplete indicator windows)
df_ind = ____
print(
    f"\n{primary}: {len(primary_df)} rows -> {len(df_ind)} after indicator "
    "warm-up (first 26 rows carry null windows)"
)
print(
    df_ind.select(
        ["Date", "Close", "RSI", "MACD_hist", "BB_pctB"]
    ).tail(5)
)

# ── Checkpoint 1: indicators are correct ──────────────────────────────
# (a) RSI is bounded by construction
rsi = df_ind["RSI"]
assert float(rsi.min()) >= 0.0 and float(rsi.max()) <= 100.0, (
    f"RSI out of bounds: [{float(rsi.min()):.2f}, {float(rsi.max()):.2f}]"
)

# (b) Hand-check RSI at one row with an explicit Wilder recursion in numpy.
# polars alignment: row 0 has a null diff -> gain=loss=0, and
# ewm_mean(alpha=1/14, adjust=False) seeds the recursion with that row:
#   avg_0 = 0.0 ;  avg_r = (1 - alpha) * avg_{r-1} + alpha * x_r
# where x_r (r >= 1) is the gain/loss of closes[r] - closes[r-1].
closes = primary_df["Close"].to_numpy()
deltas = np.diff(closes)
gains = np.maximum(deltas, 0.0)
losses = np.maximum(-deltas, 0.0)
alpha = 1 / 14
avg_gain = avg_loss = 0.0
CHECK_AT = 200  # arbitrary raw-frame row well past the warm-up
for r in range(1, CHECK_AT + 1):
    avg_gain = (1 - alpha) * avg_gain + alpha * gains[r - 1]
    avg_loss = (1 - alpha) * avg_loss + alpha * losses[r - 1]
hand_rsi = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-12))
# polars RSI column is aligned with df_ind rows; row r of df_ind is row
# r + (len(primary_df) - len(df_ind)) of the raw frame.
_offset = len(primary_df) - len(df_ind)
polars_rsi = float(df_ind["RSI"][CHECK_AT - _offset])
assert abs(hand_rsi - polars_rsi) < 0.5, (
    f"RSI hand-check failed: numpy recursion {hand_rsi:.2f} vs polars "
    f"{polars_rsi:.2f} at row {CHECK_AT}"
)

# (c) %B identity: 0.5 means "at the 20-day mean" — the column's own
# median should sit near 0.5 for a real price series.
pctb_median = float(df_ind["BB_pctB"].median())
assert 0.2 < pctb_median < 0.8, f"%B median {pctb_median:.3f} far from 0.5"

print(f"\n  RSI bounds: [{float(rsi.min()):.1f}, {float(rsi.max()):.1f}] (must be [0, 100])")
print(f"  RSI hand-check at row {CHECK_AT}: numpy={hand_rsi:.2f} polars={polars_rsi:.2f}")
print(f"  %B median: {pctb_median:.3f} (0.5 = at the 20-day mean)")
print("\n--- Checkpoint 1 passed --- indicators verified\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: raw OHLCV vs indicator-augmented LSTM
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 3 — TRAIN: same LSTM, same rows, different feature sets")
print("=" * 70)


class LSTMRegressor(nn.Module):
    """Single-layer LSTM forecaster (same architecture for both variants)."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 horizon: int = FORECAST_HORIZON):
        super().__init__()
        # TODO: self.lstm — single-layer nn.LSTM, input_dim -> hidden_dim,
        #   batch-first; self.head — hidden_dim -> horizon linear map
        self.lstm = ____
        self.head = ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Run x through self.lstm; forecast from the LAST timestep's
        #   output, giving shape (batch, horizon)
        out, _ = ____
        return ____


conn, tracker, exp_name, registry, has_registry = setup_engines(
    primary, experiment_suffix="indicators"
)

# SAME trimmed frame for both variants — only the feature set differs.
(
    train_loader_base,
    val_loader_base,
    X_train_t,
    y_train_t,
    X_val_t_base,
    y_val_t_base,
    norm_mean,
    norm_std,
    n_train_w,
    n_features_base,
) = prepare_dataloaders(df_ind, device, feature_cols=FEATURES)

(
    train_loader_ind,
    val_loader_ind,
    _,
    _,
    X_val_t_ind,
    y_val_t_ind,
    _,
    _,
    _,
    n_features_ind,
    # TODO: Same trimmed frame, but the indicator-augmented feature set
    # Hint: pass feature_cols=INDICATOR_FEATURES
) = prepare_dataloaders(____)

print(f"\nBaseline features ({n_features_base}): {FEATURES}")
print(f"Augmented features ({n_features_ind}): {INDICATOR_FEATURES}")

print("\nTraining baseline LSTM (OHLCV only)...")
torch.manual_seed(42)
model_base = LSTMRegressor(n_features_base).to(device)
res_base = train_model(
    model_base, "LSTM_OHLCV", tracker, exp_name,
    train_loader_base, val_loader_base, device, epochs=EPOCHS,
)

print("\nTraining indicator-augmented LSTM...")
torch.manual_seed(42)
# TODO: A fresh LSTMRegressor sized for the augmented feature set, trained
#   under the run name "LSTM_OHLCV_indicators" on the indicator loaders
model_ind = ____
res_ind = ____

# ── Checkpoint 2: both models trained and converging ──────────────────
for name, res in (("baseline", res_base), ("indicators", res_ind)):
    assert res["val_losses"][-1] < res["val_losses"][0], (
        f"{name}: val loss should decrease "
        f"({res['val_losses'][0]:.4f} -> {res['val_losses'][-1]:.4f})"
    )
    assert res["val_losses"][-1] < 0.5, (
        f"{name}: final val loss {res['val_losses'][-1]:.4f} too high — "
        "expected < 0.5 on normalised closes"
    )

delta = res_ind["final_val_loss"] - res_base["final_val_loss"]
print(f"\n{'=' * 62}")
print("  FEATURE-SET COMPARISON (measured, this run)")
print(f"{'=' * 62}")
print(f"  {'Variant':>28} {'Features':>9} {'Final Val MSE':>14}")
print("  " + "-" * 54)
print(f"  {'OHLCV baseline':>28} {n_features_base:>9} {res_base['final_val_loss']:>14.4f}")
print(f"  {'OHLCV + indicators':>28} {n_features_ind:>9} {res_ind['final_val_loss']:>14.4f}")
print(
    f"\n  Indicator delta: {delta:+.4f} MSE "
    f"({'helps' if delta < 0 else 'hurts'} at {EPOCHS} epochs). "
    "Either sign is a real measurement: indicators compress information "
    "the window already contains, so the benefit depends on horizon, "
    "architecture and training budget. What is NOT negotiable is the "
    "leakage discipline behind them."
)
print("\n--- Checkpoint 2 passed --- both variants trained\n")

best_is_ind = delta < 0
best_model = model_ind if best_is_ind else model_base
best_loss = res_ind["final_val_loss"] if best_is_ind else res_base["final_val_loss"]
register_best_model(
    best_model, "lstm_indicators" if best_is_ind else "lstm_ohlcv",
    best_loss, primary, registry, has_registry,
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: indicator panels + forecast quality
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: what the indicators see")
print("=" * 70)

# (a) Indicator panels over the last 500 trading days
panel = df_ind.tail(500)
days = np.arange(len(panel))
fig_ind, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig_ind.suptitle(
    f"{primary} — the three indicator views (last 500 trading days)", fontsize=14
)
axes[0].plot(days, panel["Close"].to_numpy(), color="#2196F3", linewidth=1.2)
axes[0].set_ylabel("Close")
axes[0].set_title("Price (the non-stationary raw series)")
axes[1].plot(days, panel["RSI"].to_numpy(), color="#9C27B0", linewidth=1.2)
axes[1].axhline(70, color="red", linestyle="--", alpha=0.5)
axes[1].axhline(30, color="green", linestyle="--", alpha=0.5)
axes[1].set_ylabel("RSI(14)")
axes[1].set_ylim(0, 100)
axes[1].set_title("Momentum — overbought > 70, oversold < 30")
macd_hist = panel["MACD_hist"].to_numpy()
axes[2].bar(days, macd_hist,
            color=np.where(macd_hist >= 0, "#4CAF50", "#F44336"), width=1.0)
axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].set_ylabel("MACD hist")
axes[2].set_title("Trend divergence — fast EMA vs slow EMA, signal removed")
axes[3].plot(days, panel["BB_pctB"].to_numpy(), color="#FF9800", linewidth=1.2)
axes[3].axhline(1.0, color="red", linestyle="--", alpha=0.5)
axes[3].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
axes[3].axhline(0.0, color="green", linestyle="--", alpha=0.5)
axes[3].set_ylabel("%B(20,2)")
axes[3].set_title("Volatility position — 0.5 = at the 20-day mean")
for ax in axes:
    ax.grid(True, alpha=0.3)
fig_ind.tight_layout()
fig_ind.savefig(str(OUTPUT_DIR / "06_indicator_panels.png"), dpi=150)
plt.close(fig_ind)
print(f"  Saved: {OUTPUT_DIR / '06_indicator_panels.png'}")

# (b) Training curves + forecast quality for both variants
# ModelVisualizer carries a kailash-ml P2 experimental notice (UserWarning
# at construction); the gate runs warnings-as-errors, so mute just that.
import warnings

from kailash_ml._decorators import ExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ExperimentalWarning)
    viz = get_visualizer()
plot_training_curves(viz, res_base, "LSTM_OHLCV", "06_baseline")
plot_training_curves(viz, res_ind, "LSTM_indicators", "06_indicators")

preds_base, actual, _ = plot_predictions(
    viz, model_base, X_val_t_base, y_val_t_base, norm_mean, norm_std,
    "06_baseline",
)
preds_ind, _, _ = plot_predictions(
    viz, model_ind, X_val_t_ind, y_val_t_ind, norm_mean, norm_std,
    "06_indicators",
)
plot_time_series_overlay(
    preds_base, actual, "06_baseline",
    title=f"OHLCV baseline — predicted vs actual close ({primary})",
)
plot_time_series_overlay(
    preds_ind, actual, "06_indicators",
    title=f"Indicator-augmented — predicted vs actual close ({primary})",
)
rmse_base = plot_horizon_error(preds_base, actual, "OHLCV baseline")
rmse_ind = plot_horizon_error(preds_ind, actual, "OHLCV + indicators")

# ── Checkpoint 3: artefacts exist ─────────────────────────────────────
import os

for artefact in (
    OUTPUT_DIR / "06_indicator_panels.png",
    OUTPUT_DIR / "06_baseline_time_series_overlay.png",
    OUTPUT_DIR / "06_indicators_time_series_overlay.png",
):
    assert os.path.exists(artefact), f"Missing artefact: {artefact}"
print("\n--- Checkpoint 3 passed --- visual proof generated\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: directional accuracy for an index-futures desk
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (anonymised, illustrative): the treasury desk of a regional
# bank monitors index futures to hedge overnight exposure. The desk does
# not trade the model's price LEVEL — it acts on the model's DIRECTION
# call for the 5-day horizon: "will the index be higher or lower than
# the last close of the lookback window?"
#
# RMSE punishes magnitude errors; the desk's P&L cares about SIGN errors.
# A forecaster can halve its RMSE and still get the direction wrong on
# every trending day. So we score both variants on the metric the desk
# actually uses: directional hit rate = fraction of validation windows
# where the predicted 5-day move has the same sign as the realised move.

print("=" * 70)
print("  PHASE 5 — APPLY: directional accuracy (the desk's metric)")
print("=" * 70)

close_mean, close_std = norm_mean[0, 0], norm_std[0, 0]
# Last observed close of each validation window (denormalised)
# TODO: Last timestep of each window, feature 0 (Close), de-normalised
# Hint: X_val_t_base[:, -1, 0] is still z-scored — multiply back by
#   close_std and add close_mean
last_close_base = ____
last_close_ind = ____
actual_day5 = actual[:, -1]  # realised close at horizon day 5


def directional_hit_rate(preds: np.ndarray, last_close: np.ndarray) -> float:
    """Fraction of windows where predicted and realised 5-day moves agree in sign."""
    # TODO: pred_move = predicted day-5 close minus last observed close;
    #   real_move = realised day-5 close minus last observed close;
    #   hit rate = mean(sign(pred_move) == sign(real_move))
    pred_move = ____
    real_move = ____
    agree = ____
    return float(np.mean(agree))


hit_base = directional_hit_rate(preds_base, last_close_base)
hit_ind = directional_hit_rate(preds_ind, last_close_ind)

print(
    f"""
  DIRECTIONAL ACCURACY ({len(actual_day5):,} validation windows, day-5 horizon):

    OHLCV baseline:       {hit_base:.1%}  (day-5 RMSE {rmse_base[-1]:.2f})
    OHLCV + indicators:   {hit_ind:.1%}  (day-5 RMSE {rmse_ind[-1]:.2f})
    coin-flip reference:  50.0%

  READING IT:
    A hit rate above 50% means the model carries real directional signal.
    Compare the RMSE ranking with the hit-rate ranking — when they
    disagree, the desk should trust the HIT RATE, because the hedge is a
    binary act (hedge / don't hedge), not a price-level bet.

  STAKEHOLDER-READY OUTPUT:
    "On {len(actual_day5):,} held-out windows the indicator-augmented
    forecaster calls the 5-day direction correctly {hit_ind:.0%} of the
    time vs {hit_base:.0%} for the raw-price baseline. Indicators are
    backward-looking transforms (no lookahead), computed in polars and
    audited against a hand-written RSI recursion."
"""
)

# ── Checkpoint 4: directional metrics computed honestly ───────────────
assert 0.0 <= hit_base <= 1.0 and 0.0 <= hit_ind <= 1.0
assert len(actual_day5) > 100, "Directional metric needs a real sample"
print("--- Checkpoint 4 passed --- desk-metric application demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  THEORY:
  [x] RSI = momentum (bounded [0,100]); MACD hist = trend divergence
      (mean-zero); Bollinger %B = volatility position (centred at 0.5)
  [x] Prices are non-stationary; indicators are stationary transforms —
      that is why networks learn them faster
  [x] Leakage discipline: backward-looking windows only, train-split
      normalisation only

  BUILD + TRAIN:
  [x] add_technical_indicators in pure polars, hand-verified: numpy
      Wilder recursion {hand_rsi:.2f} vs polars {polars_rsi:.2f}
  [x] Same LSTM, same rows, two feature sets ({n_features_base} vs
      {n_features_ind} columns), ExperimentTracker receipts
  [x] Measured effect: indicator delta {delta:+.4f} val MSE at
      {EPOCHS} epochs ({'helped' if delta < 0 else 'hurt'} this run —
      either sign is honest)

  VISUALISE (the proof):
  [x] Four-panel indicator chart: price / RSI / MACD hist / %B
  [x] Forecast overlays and horizon RMSE for both variants
      (day-5: {rmse_base[-1]:.2f} vs {rmse_ind[-1]:.2f})

  APPLY:
  [x] Directional accuracy — the metric a hedging desk is paid on
  [x] Baseline {hit_base:.1%} vs indicators {hit_ind:.1%} vs coin-flip 50%
  [x] When RMSE and hit-rate disagree, the binary desk action follows
      the hit rate

  KEY INSIGHT: Feature engineering is not "more columns = better model".
  It is the craft of turning domain knowledge into STATIONARY,
  LEAKAGE-FREE transforms — and then letting a measured ablation, not
  folklore, decide whether they stay. The audit trail matters as much
  as the delta: every indicator here is defined, hand-checked, and
  tracked in ExperimentTracker.
"""
)
