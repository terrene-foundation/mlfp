# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.6: Temporal Features — Lags, Rolling Windows, and
#                         the Leakage Discipline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Lag, rolling-mean, and calendar features — and what each encodes
#   - The temporal leakage discipline: SHIFT before you roll
#   - Why group_by_dynamic's labelled window runs FORWARD (audit-proven)
#   - Measuring what temporal features buy on a real panel (R² lift)
#
# PREREQUISITES: 01–05 (feature engineering, selection, tracking);
#   ex_5/07 (regression metrics) for the evaluation vocabulary
# ESTIMATED TIME: ~30 min
#
# 5-PHASE STRUCTURE:
#   Theory   — a static model cannot see momentum; temporal features add it
#   Build    — town-month HDB panel, lag/rolling/calendar features
#   Train    — same regressor, with vs without temporal features
#   Visualise — one town's trajectory + the R² lift
#   Apply    — a property portal's monthly market-report model
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score

from shared import MLFPDataLoader
from shared.mlfp03.ex_1 import OUTPUT_DIR

# ── THEORY — momentum is not in the row ──────────────────────────────────
# A flat's price row says WHERE and WHAT, never WHEN. Two identical months
# can precede very different futures: one after a 12-month climb, one after
# a plateau. Temporal features hand a static model that momentum:
#
#   LAG      — the value k periods ago. lag_1 = last month, lag_12 = same
#              month last year (annual seasonality anchor).
#   ROLLING  — mean over the last k periods. Smooths one-month noise into a
#              trend line. MUST be computed on SHIFTED values: rolling over
#              the CURRENT month leaks the target into the feature.
#   CALENDAR — month-of-year (cyclic), quarter. Encodes the CNY-quiet /
#              year-end-rush rhythm without the model relearning it.
#
# THE LEAKAGE DISCIPLINE: every temporal feature for month t must be
# computable from months < t. `shift(1).rolling_mean(3)` is safe;
# `rolling_mean(3)` alone includes t. polars' group_by_dynamic makes this
# worse: its window LABELLED t actually runs FORWARD [t, t+12mo) — the
# audit verified this empirically — so "trailing 12-month mean" written the
# obvious way is really a leading one. We use shift + rolling_* instead.

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet").filter(
    pl.col("resale_price").is_between(50_000, 2_000_000)
)

# ── BUILD — the town-month panel, then temporal features ─────────────────
panel = (
    hdb.with_columns(pl.col("month").str.to_date("%Y-%m").alias("month_date"))
    .group_by("town", "month_date")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.col("resale_price").count().alias("n_sales"),
        pl.col("floor_area_sqm").median().alias("median_sqm"),
    )
    .sort("town", "month_date")
)
print(f"  Panel: {panel.height:,} town-months "
      f"({panel['town'].n_unique()} towns × {panel['month_date'].n_unique()} months)")

def add_temporal_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Lag/rolling/calendar features — every one computable from months < t."""
    out = panel.with_columns(
        # TODO: lag_1 / lag_3 / lag_12 — median_price shifted 1, 3, 12 months per town
        # Hint: pl.col("median_price").shift(k).over("town").alias(...)
        ____.alias("lag_1"),
        ____.alias("lag_3"),
        ____.alias("lag_12"),
        # TODO: rolling 3-month mean on SHIFTED values (shift(1) first — the leak guard)
        # Hint: pl.col("median_price").shift(1).over("town").rolling_mean(3).over("town")
        ____.alias("roll_3"),
        # TODO: rolling 12-month mean, same discipline
        ____.alias("roll_12"),
        # Calendar: cyclic month-of-year + quarter
        pl.col("month_date").dt.month().alias("month_num"),
        pl.col("month_date").dt.quarter().alias("quarter"),
    )
    out = out.with_columns(
        (2 * np.pi * pl.col("month_num") / 12).sin().alias("month_sin"),
        (2 * np.pi * pl.col("month_num") / 12).cos().alias("month_cos"),
        # TODO: momentum = (roll_3 − roll_12) / roll_12
        # Hint: short trend minus long trend, normalised by the long trend
        ____.alias("momentum"),
    )
    return out.drop_nulls(["lag_1", "lag_3", "lag_12", "roll_3", "roll_12"])

feat = add_temporal_features(panel)
print(f"  After warm-up trim: {feat.height:,} town-months with full temporal features")

# ── TRAIN — same regressor, with vs without temporal features ────────────
TARGET = "median_price"
STATIC = ["median_sqm", "n_sales", "month_sin", "month_cos"]
TEMPORAL = ["lag_1", "lag_3", "lag_12", "roll_3", "roll_12", "momentum"]

# Time-ordered split: train on 2015–2022, test on 2023–2024. A RANDOM split
# of a panel would leak — adjacent months share information.
split_date = pl.date(2023, 1, 1)
# TODO: train = rows before split_date; test = rows from split_date onward
# Hint: feat.filter(pl.col("month_date") < split_date)
train = ____
test = ____
print(f"  Train {train.height:,} town-months (<2023) · Test {test.height:,} (2023–24)")

# One-hot town for both arms, so the difference is ONLY the temporal features
import numpy as _np

def with_town(frame: pl.DataFrame, cols: list[str]) -> np.ndarray:
    base = frame.select(cols).to_numpy()
    dummies = frame.select("town").to_dummies().to_numpy()
    return _np.hstack([base, dummies])

def fit_eval_town(cols: list[str], label: str) -> dict:
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              random_state=42, verbose=-1)
    # TODO: fit on the town-augmented train matrix, predict on test
    # Hint: model.fit(with_town(train, cols), train[TARGET].to_numpy())
    ____
    pred = ____
    mae = float(mean_absolute_error(test[TARGET].to_numpy(), pred))
    r2 = float(r2_score(test[TARGET].to_numpy(), pred))
    print(f"    {label:<28} MAE S${mae:>8,.0f}  R² {r2:6.3f}")
    return {"label": label, "mae": mae, "r2": r2, "model": model, "pred": pred}

print("\n  With vs without temporal features (same town one-hot baseline):")
row_static = fit_eval_town(STATIC, "static + town")
row_temporal = fit_eval_town(STATIC + TEMPORAL, "static + temporal + town")

# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert row_temporal["r2"] > row_static["r2"] + 0.02, (
    "temporal features must buy a real R² lift on a genuine panel"
)
assert row_temporal["mae"] < row_static["mae"], "MAE must improve too"
print("[ok] Checkpoint 1 — temporal features buy a measurable lift, honestly split")

# ── VISUALISE — one town's trajectory + the R² lift ──────────────────────
town = "TAMPINES"
town_hist = feat.filter(pl.col("town") == town).sort("month_date")

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f"{town.title()}: actuals vs lag/rolling features",
        "What temporal features buy (held-out 2023–24)",
    ),
)
fig.add_trace(
    go.Scatter(x=town_hist["month_date"].to_list(), y=town_hist["median_price"].to_list(),
               name="median price", line=dict(color="#0F172A")), row=1, col=1)
fig.add_trace(
    go.Scatter(x=town_hist["month_date"].to_list(), y=town_hist["roll_3"].to_list(),
               name="rolling 3mo (shifted)", line=dict(color="#0D9488", dash="dash")), row=1, col=1)
fig.add_trace(
    go.Scatter(x=town_hist["month_date"].to_list(), y=town_hist["roll_12"].to_list(),
               name="rolling 12mo (shifted)", line=dict(color="#6366F1", dash="dot")), row=1, col=1)
fig.add_trace(
    go.Bar(x=[r["label"] for r in (row_static, row_temporal)],
           y=[r["r2"] for r in (row_static, row_temporal)],
           marker_color=["#64748B", "#0D9488"],
           text=[f"R² {r['r2']:.3f}" for r in (row_static, row_temporal)],
           textposition="auto", name="R²"), row=1, col=2)
fig.update_layout(title="Temporal features — momentum made visible", showlegend=True)
fig.write_html(str(OUTPUT_DIR / "ex1_06_temporal_features.html"))
print(f"  Saved: {OUTPUT_DIR / 'ex1_06_temporal_features.html'}")
print("[ok] Checkpoint 2 — trajectory + lift visual saved")

# ── APPLY — a property portal's monthly market-report model ──────────────
print(
    f"\n  APPLY: temporal features moved held-out R² {row_static['r2']:.3f} → "
    f"{row_temporal['r2']:.3f} (MAE S${row_static['mae']:,.0f} → "
    f"S${row_temporal['mae']:,.0f}). A portal's monthly 'market heat' report "
    f"is exactly this: lag-12 anchors the year-ago comparison readers expect, "
    f"the shifted 3-month roll is the publishable trend line, and momentum "
    f"flags towns turning before the annual figures do. The same shift-before-"
    f"roll discipline protects every production feature pipeline that trains "
    f"on history and scores the present."
)

# REFLECTION
print(
    """
  What you've mastered:
    ✓ Lag / rolling / calendar features and what each encodes
    ✓ The leakage discipline: SHIFT before you roll — and why
      group_by_dynamic's labelled window runs FORWARD (audit-proven)
    ✓ Time-ordered splits for panels — random splits leak across months
    ✓ Measuring the lift honestly: same regressor, same split,
      same baseline, only the features differ

  Next: 07_forward_backward_selection.py — wrapper selection beyond RFE:
  SequentialFeatureSelector in both directions.
"""
)
