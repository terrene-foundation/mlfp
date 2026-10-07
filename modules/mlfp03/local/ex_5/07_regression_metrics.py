# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.7: Regression Metrics in the Taxonomy
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - The complete regression metrics taxonomy: MAE, RMSE, MAPE, R²
#   - Why each answers a different business question, and which to pick
#   - Why R² can be negative (a model worse than predicting the mean)
#   - How a residual plot reveals what a single number hides
#
# PREREQUISITES: Exercise 5.1–5.6 (classification metrics, log loss)
# ESTIMATED TIME: ~25 min
#
# 5-PHASE STRUCTURE:
#   Theory   — MAE vs RMSE vs MAPE vs R², and when each lies to you
#   Build    — fit a linear and a boosted regressor on HDB resale
#   Train    — evaluate both with every metric, on held-out rows
#   Visualise — residual scatter + per-metric comparison bars
#   Apply    — a Singapore property platform's price-estimation triage
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from shared import MLFPDataLoader
from shared.kailash_helpers import hdb_storey_range_expr
from shared.mlfp03.ex_5 import OUTPUT_DIR

# ── THEORY — the four regression metrics, and when each lies ────────────
# The classification taxonomy (5.1–5.6) does not transfer to "how far off is
# the price". Regression needs its own four numbers:
#
#   MAE  — mean absolute error. "On average, how many dollars off am I?"
#          Robust to the few extreme misses; the honest everyday error.
#   RMSE — root mean squared error. Squares each miss before averaging, so a
#          single S$500k blunder dominates. Use it when big errors cost
#          disproportionately (loan sizing, safety margins).
#   MAPE — mean absolute percentage error. "Off by what fraction?" Unit-free,
#          so it compares a S$400k flat and a S$2M penthouse fairly — but it
#          blows up when the true value is small (division by near-zero).
#   R²   — 1 − SS_res/SS_tot. "What share of variance do I explain?" 1.0 is
#          perfect; 0 is as good as always predicting the mean; NEGATIVE is
#          worse than the mean — a real warning, not a typo.
#
# MAE and RMSE are in dollars, MAPE is dimensionless, R² is relative. A good
# deployment reads all four plus the residual plot, not one favourite.

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet").filter(
    pl.col("resale_price").is_between(50_000, 2_000_000)
    & pl.col("floor_area_sqm").is_between(20, 300)
)

# Derive model-ready numerics from the raw columns: storey midpoint from the
# "10 TO 12" range string, lease years from "65 years 04 months". The raw file
# has letter-O typos in storey_range ("O4 TO 06", "28 TO 3O") — read a letter O
# next to a digit as zero first (same normalisation as M2 ex_8). One pipeline
# keeps features and target row-aligned.
storey = hdb_storey_range_expr()
hdb_m = (
    hdb.drop_nulls(["storey_range", "remaining_lease"])
    .with_columns(
        # TODO: midpoint of the two numbers in storey_range, as a Float64
        # Hint: storey.str.extract(r"(\d+)", 1) gets the first number; use
        #       r"(\d+)$" for the last; average with .truediv(2)
        ____.alias("storey_midpoint"),
        # TODO: years + months/12 from remaining_lease ("65 years 04 months")
        # Hint: two .str.extract calls, one per r"(\d+)\s*year" / r"(\d+)\s*month",
        #       cast Float64, .fill_null(0), months divided by 12
        ____.alias("remaining_lease_years"),
    )
    .drop_nulls(["storey_midpoint", "remaining_lease_years", "floor_area_sqm", "resale_price"])
)
X_df = hdb_m.select(["floor_area_sqm", "storey_midpoint", "remaining_lease_years"])
y = hdb_m["resale_price"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X_df.to_numpy(), y, test_size=0.2, random_state=42
)


def metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # TODO: MAE with mean_absolute_error
    mae = float(____)
    # TODO: RMSE — root of mean_squared_error
    rmse = float(____)
    # TODO: MAPE — mean of |y_true − y_pred| / y_true, times 100
    mape = float(____)
    # TODO: R² with r2_score
    r2 = float(____)
    print(
        f"  {name:<18} MAE S${mae:>9,.0f}  RMSE S${rmse:>9,.0f}  "
        f"MAPE {mape:5.1f}%  R² {r2:6.3f}"
    )
    return {"name": name, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ── BUILD + TRAIN — two regressors, every metric, held-out rows ─────────
# TODO: fit a LinearRegression on (X_train, y_train)
lin = ____
# TODO: fit an LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
boost = ____

rows = [
    # TODO: the naive baseline predicts the TRAIN mean for every test row
    metrics("Naive mean", y_test, ____),
    metrics("Linear", y_test, lin.predict(X_test)),
    metrics("Boosted trees", y_test, boost.predict(X_test)),
]

print("\n  Regression metrics taxonomy (held-out 20%):")
assert rows[0]["r2"] <= 0.001, "naive-mean R² must be ≈ 0 by definition"
assert rows[2]["mae"] < rows[1]["mae"], "boosted should beat linear on MAE"
assert rows[2]["r2"] > rows[1]["r2"] > 0.5, "both regressors beat the mean (R² > 0.5)"
print("[ok] Checkpoint 1 — every metric computed; boosted beats linear beats the mean")

# ── VISUALISE — residual scatter + metric comparison ────────────────────
pred_lin = lin.predict(X_test)
pred_boost = boost.predict(X_test)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Residuals: boosted trees (a model can be 'good on average' "
        "and still miss whole towns)",
        "The four metrics, side by side",
    ),
)
fig.add_trace(
    go.Scatter(
        x=pred_boost,
        # TODO: residual = y_true − y_pred for the boosted model
        y=____,
        mode="markers",
        marker=dict(size=4, color="#0D9488", opacity=0.5),
        name="residual (y_true − y_pred)",
    ),
    row=1,
    col=1,
)
fig.add_hline(y=0, line_dash="dash", line_color="#DC2626", row=1, col=1)
fig.add_trace(
    go.Bar(
        x=["MAE", "RMSE", "MAPE", "R²"],
        y=[rows[2]["mae"], rows[2]["rmse"], rows[2]["mape"], rows[2]["r2"]],
        marker_color=["#0D9488", "#0D9488", "#F59E0B", "#6366F1"],
        text=[
            f"S${rows[2]['mae']:,.0f}",
            f"S${rows[2]['rmse']:,.0f}",
            f"{rows[2]['mape']:.1f}%",
            f"{rows[2]['r2']:.3f}",
        ],
        textposition="auto",
        name="boosted trees",
    ),
    row=1,
    col=2,
)
fig.update_xaxes(title_text="predicted price (S$)", row=1, col=1)
fig.update_yaxes(title_text="residual (S$)", row=1, col=1)
fig.update_layout(title="Regression metrics — residual view + taxonomy", showlegend=False)
fig.write_html(str(OUTPUT_DIR / "ex5_07_regression_metrics.html"))
print(f"  Saved: {OUTPUT_DIR / 'ex5_07_regression_metrics.html'}")
print("[ok] Checkpoint 2 — residual plot + metric comparison saved")

# ── APPLY — Singapore property platform's price-estimation triage ───────
# TODO: index of the largest absolute residual (np.argmax of np.abs(...))
worst = int(____)
print(
    f"\n  APPLY: worst boosted miss is S${y_test[worst] - pred_boost[worst]:+,.0f} "
    f"on a S${y_test[worst]:,.0f} flat."
)
print(
    "  MAE tells the listing team the everyday error; RMSE flags the one "
    "penthouse-level blunder; MAPE lets you compare a walk-up to a condo; "
    "R² tells you if the model beats 'just use the median'. The residual "
    "scatter shows WHERE the misses cluster — usually at the extremes."
)

# REFLECTION
print(
    """
  What you've mastered:
    ✓ MAE (everyday error) vs RMSE (punishes big misses) vs MAPE (unit-free)
      vs R² (fraction of variance explained, can go negative)
    ✓ Why a naive-mean baseline has R² ≈ 0 and any real model must beat it
    ✓ Reading a residual plot to see what a single number hides
    ✓ Picking the regression metric that matches the business cost structure

  Next: 08_stacking_blending.py — combine several weak models into one
  stronger ensemble with EnsembleEngine.stack and .blend.
"""
)
