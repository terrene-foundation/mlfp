# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 8.2: Point-in-Time Retrieval — Leakage Prevention
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Retrieve features at specific points in time to prevent leakage
#   - Demonstrate how a leaked (target-derived) feature inflates performance
#   - Cross-check FeatureStore PIT retrieval against Polars filtering
#   - Quantify the impact of leakage on out-of-time error
#   - Apply PIT correctness to Singapore property market forecasting
#
# PREREQUISITES: Exercise 8.1 (FeatureSchema v1, feature computation)
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — what data leakage is and why it destroys models
#   2. Build — PIT retrieval via FeatureStore, checked against Polars
#   3. Train — compare leaked vs correct model performance
#   4. Visualise — side-by-side leaked vs correct predictions
#   5. Apply — mortgage valuation model for a Singapore bank
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.mlfp02.ex_8 import (
    OUTPUT_DIR,
    as_of,
    build_schema_v1,
    compute_v1_features,
    create_feature_store,
    fit_ols,
    load_hdb_resale,
    materialize_features,
    validate_v1_features,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — What Data Leakage Is and Why It Destroys Models
# ════════════════════════════════════════════════════════════════════════
# Data leakage occurs when information from the future (or from the test
# set) bleeds into the training data. The model learns patterns that
# will not exist at prediction time, producing artificially high
# validation metrics and catastrophic production failures.
#
# Three common leakage types:
#
#   1. TEMPORAL LEAKAGE — Using 2024 transaction prices to train a model
#      that predicts 2023 prices. The model "knows" the future.
#
#   2. TARGET LEAKAGE — Using a feature derived from the target variable.
#      Example: using "price_per_sqm" (which includes resale_price) to
#      predict resale_price. Circular reasoning.
#
#   3. TRAIN-TEST CONTAMINATION — Same transaction appears in both
#      training and test sets (duplication or random split ignoring time).
#
# Point-in-time (PIT) retrieval prevents temporal leakage by enforcing
# a hard cutoff: at prediction time T, only data from before T is used.
#
# Singapore analogy: official property price indices are published
# quarterly. If you build a Q1-2024 forecast using data up to Q3-2024,
# your "forecast" is just reading the answer sheet. PIT retrieval is the
# discipline of covering the answer sheet during the exam.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Point-in-Time retrieval via FeatureStore and Polars
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exercise 8.2 — Point-in-Time Retrieval: Leakage Prevention")
print("=" * 70)

# --- 2a. Prepare features (validated rows, as in 8.1) ---
hdb = load_hdb_resale()
features_v1, _ = validate_v1_features(compute_v1_features(hdb))
property_schema_v1 = build_schema_v1()

print(f"\n  Features computed: {features_v1.shape[0]:,} rows")
print(
    f"  Date range: {features_v1['transaction_date'].min()} to "
    f"{features_v1['transaction_date'].max()}"
)

# --- 2b. FeatureStore PIT retrieval ---
# get_features(schema, timestamp=T) returns each entity's feature values
# as of T — only rows with event time <= T. Our rule is "strictly before
# the cutoff", so we ask for T = cutoff minus one second.
fs = create_feature_store()
asyncio.run(materialize_features(fs, property_schema_v1, features_v1))  # idempotent

CUTOFF_2023 = datetime(2023, 1, 1)
CUTOFF_2024 = datetime(2024, 1, 1)
ONE_SECOND = timedelta(seconds=1)

# TODO: Retrieve the features as of one second before each cutoff.
# Hint: asyncio.run(fs.get_features(schema, timestamp=...)) — the
#   timestamp is inclusive, so subtract ONE_SECOND from the cutoff.
features_2023 = ____
features_2024 = ____
delta = features_2024.height - features_2023.height
print(f"\n  [FeatureStore PIT]")
print(f"    Features as of 2022-12-31: {features_2023.height:,} rows")
print(f"    Features as of 2023-12-31: {features_2024.height:,} rows")
print(f"    2023 transactions added: {delta:,}")

# Cross-check against a plain Polars filter on the source frame
# TODO: Same cutoff with the Polars helper.
# Hint: as_of(features_v1, CUTOFF_2023) keeps rows strictly before the cutoff.
polars_2023 = ____
print(f"    Polars rows strictly before 2023: {polars_2023.height:,}")

print(f"\n  --- Why Point-in-Time Matters ---")
print(f"  To predict prices at T=2023-01-01, you must ONLY use data before T.")
print(f"  Using 2024 data would leak future info -> over-optimistic evaluation.")


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert features_2023.height > 0, "Task 2: must have pre-2023 features"
assert features_2023.height == polars_2023.height, "Task 2: store PIT must match the Polars cutoff"
assert (
    features_2024.height > features_2023.height
), "Task 2: 2024 cutoff must include more rows than 2023"
print("\n[ok] Checkpoint 1 passed — PIT retrieval demonstrated\n")

# INTERPRETATION: The delta between the 2023 and 2024 cutoffs is an
# entire year of transactions. A model meant to price flats on
# 2023-01-01 must not see any of them — the store enforces that by
# construction, so the training set cannot accidentally include them.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Compare leaked vs PIT-correct model performance
# ════════════════════════════════════════════════════════════════════════
# Both models are trained on the PIT training set (features from the
# store, as of 2022-12-31) and evaluated out-of-time on 2023 sales:
#   - CORRECT: inputs a valuer has BEFORE the sale (area, storey, lease)
#   - LEAKED:  the same inputs PLUS price_per_sqm, which is computed from
#              the sale price itself — target leakage. It is legitimate in
#              the store for market analytics, but it does not exist when
#              you must price a flat that has not sold yet.

print("--- Comparing Leaked vs Correct Models ---")

FEATURE_COLS = [
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
]
LEAKED_COLS = [*FEATURE_COLS, "price_per_sqm"]

# Features come from the store; the label (resale_price) is joined on the
# entity id from the transaction records.
labels = features_v1.select("transaction_id", "transaction_date", "resale_price").with_columns(
    pl.col("transaction_id").cast(pl.Int64)
)
train_set = features_2023.with_columns(pl.col("transaction_id").cast(pl.Int64)).join(
    labels, on="transaction_id", how="inner"
)
# TODO: Filter features_v1 to ONLY 2023 transactions (>= 2023-01-01 and
# < 2024-01-01). This is the out-of-time test set.
# Hint: pl.col("transaction_date") >= pl.lit(CUTOFF_2023.date()), and the
#   matching upper bound, combined with &
test_2023 = ____


def design(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack(
        [np.ones(df.height), df.select(cols).to_numpy().astype(np.float64)]
    )


y_train = train_set["resale_price"].to_numpy().astype(np.float64)
y_test = test_2023["resale_price"].to_numpy().astype(np.float64)
ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))

# Correct model
# TODO: Fit the CORRECT model on the PIT training set.
# Hint: fit_ols(design(train_set, FEATURE_COLS), y_train)
ols_correct = ____
y_pred_correct = design(test_2023, FEATURE_COLS) @ ols_correct["beta"]
resid_correct = y_test - y_pred_correct
rmse_correct = float(np.sqrt(np.mean(resid_correct**2)))
r2_test_correct = 1 - float(np.sum(resid_correct**2)) / ss_tot

# Leaked model (adds the target-derived feature)
# TODO: Fit the LEAKED model — same rows, the leaked column list.
ols_leaked = ____
y_pred_leaked = design(test_2023, LEAKED_COLS) @ ols_leaked["beta"]
resid_leaked = y_test - y_pred_leaked
rmse_leaked = float(np.sqrt(np.mean(resid_leaked**2)))
r2_test_leaked = 1 - float(np.sum(resid_leaked**2)) / ss_tot

print(f"\n  Training rows (as of 2022-12-31): {train_set.height:,}; 2023 test rows: {test_2023.height:,}")
print(f"\n  Correct model (inputs known before the sale):")
print(f"    Training R²: {ols_correct['r2']:.4f}")
print(f"    Test RMSE:   ${rmse_correct:,.0f}")
print(f"    Test R²:     {r2_test_correct:.4f}")

print(f"\n  Leaked model (+ price_per_sqm, derived from the target):")
print(f"    Training R²: {ols_leaked['r2']:.4f}")
print(f"    Test RMSE:   ${rmse_leaked:,.0f}")
print(f"    Test R²:     {r2_test_leaked:.4f}")

# TODO: Signed gap, leaked minus correct (negative = leak flatters).
leakage_gap = ____
print(f"\n  Leakage gap (leaked - correct RMSE): ${leakage_gap:+,.0f}")
print(f"  The leaked backtest understates the real error by "
      f"{-leakage_gap / rmse_correct:.0%} — an improvement that cannot exist")
print(f"  in production, because price_per_sqm is unknown before the sale.")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert ols_correct["r2"] > 0.1, "Task 3: correct model R² should be reasonable"
assert rmse_correct > 0, "Task 3: RMSE must be positive"
assert rmse_leaked < rmse_correct, "Task 3: the leaked feature should flatter the backtest"
print("\n[ok] Checkpoint 2 passed — leaked vs correct comparison complete\n")

# INTERPRETATION: The leaked model looks far better on the 2023 test set
# because one of its inputs is a rescaled copy of the answer. Every
# decision sized from that backtest (error margins, risk buffers) would
# be over-confident by the gap printed above.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Side-by-side leaked vs correct predictions
# ════════════════════════════════════════════════════════════════════════

print("--- Visualising Leakage Impact ---")

rng = np.random.default_rng(42)
n_sample = min(2000, len(y_test))
idx = rng.choice(len(y_test), size=n_sample, replace=False)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        f"CORRECT (PIT) — RMSE ${rmse_correct:,.0f}",
        f"LEAKED — RMSE ${rmse_leaked:,.0f}",
    ],
)

# Correct model
fig.add_trace(
    go.Scatter(
        x=y_test[idx].tolist(),
        y=y_pred_correct[idx].tolist(),
        mode="markers",
        marker={"size": 3, "opacity": 0.4, "color": "steelblue"},
        name="Correct",
    ),
    row=1,
    col=1,
)
# Leaked model
fig.add_trace(
    go.Scatter(
        x=y_test[idx].tolist(),
        y=y_pred_leaked[idx].tolist(),
        mode="markers",
        marker={"size": 3, "opacity": 0.4, "color": "firebrick"},
        name="Leaked",
    ),
    row=1,
    col=2,
)

# Perfect prediction lines
for col in [1, 2]:
    fig.add_trace(
        go.Scatter(
            x=[float(y_test.min()), float(y_test.max())],
            y=[float(y_test.min()), float(y_test.max())],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            showlegend=False,
        ),
        row=1,
        col=col,
    )

fig.update_layout(
    title="Target Leakage Impact — Actual vs Predicted (2023 Test Set)",
    height=500,
    width=1000,
)
fig.update_xaxes(title_text="Actual Price ($)", row=1, col=1)
fig.update_xaxes(title_text="Actual Price ($)", row=1, col=2)
fig.update_yaxes(title_text="Predicted Price ($)", row=1, col=1)

fig.write_html(str(OUTPUT_DIR / "02_leakage_comparison.html"))
print(f"\n  Saved: {OUTPUT_DIR / '02_leakage_comparison.html'}")

# Residual comparison
fig2 = go.Figure()
fig2.add_trace(
    go.Histogram(
        x=resid_correct[idx].tolist(),
        name=f"Correct (RMSE ${rmse_correct:,.0f})",
        opacity=0.6,
        nbinsx=50,
    )
)
fig2.add_trace(
    go.Histogram(
        x=resid_leaked[idx].tolist(),
        name=f"Leaked (RMSE ${rmse_leaked:,.0f})",
        opacity=0.6,
        nbinsx=50,
    )
)
fig2.update_layout(
    title="Residual Distributions — Correct vs Leaked",
    xaxis_title="Residual ($)",
    yaxis_title="Count",
    barmode="overlay",
)
fig2.write_html(str(OUTPUT_DIR / "02_residual_comparison.html"))
print(f"  Saved: {OUTPUT_DIR / '02_residual_comparison.html'}")


# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — leakage visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Mortgage Valuation Model for a Singapore Bank
# ════════════════════════════════════════════════════════════════════════
# Scenario (illustrative figures): a Singapore bank uses an HDB valuation
# model to cap loan amounts. Its risk team sizes a valuation buffer from
# the model's backtest error: the buffer is 1.96 x RMSE, so roughly 95%
# of true prices fall within it (if errors are roughly Normal).
#
# If the backtest used a leaked feature, the buffer is sized from the
# fictional error and is far too thin for real applications, where the
# leaked input does not exist.

print("=== APPLY: Mortgage Valuation — PIT-Correct Backtest ===")
print()
print("  Scenario: a Singapore bank's HDB valuation buffer (illustrative)")
buffer_leaked = 1.96 * rmse_leaked
buffer_correct = 1.96 * rmse_correct
# TODO: Share of real errors (resid_correct) inside the leaked-sized buffer.
# Hint: np.mean(np.abs(...) <= ...)
coverage_leaked = ____
coverage_correct = float(np.mean(np.abs(resid_correct) <= buffer_correct))
applications_per_month = 1_000  # assumed
print()
print(f"  Buffer sized from the LEAKED backtest:  +/- ${buffer_leaked:,.0f}")
print(f"  Buffer sized from the CORRECT backtest: +/- ${buffer_correct:,.0f}")
print()
print("  Share of 2023 sales whose true price falls inside the buffer,")
print("  using the model the bank can actually run (no price_per_sqm):")
print(f"    leaked-sized buffer:  {coverage_leaked:.0%}")
print(f"    correct-sized buffer: {coverage_correct:.0%}")
print(
    f"  At {applications_per_month:,} applications/month, about "
    f"{(1 - coverage_leaked) * applications_per_month:,.0f} would fall outside "
    f"the leaked-sized buffer each month."
)
print()
print(f"  Your PIT model performance on 2023 data:")
print(f"    R² = {r2_test_correct:.4f} (honest, no leakage)")
print(f"    RMSE = ${rmse_correct:,.0f} (true prediction error)")
print(f"    vs leaked RMSE = ${rmse_leaked:,.0f} (fictionally low)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 4 passed — PIT mortgage application demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [ok] Point-in-time retrieval: hard temporal cutoffs for training data
  [ok] Target leakage: a feature computed from the label flatters backtests
  [ok] FeatureStore PIT API: get_features(schema, timestamp=T) returns
       values with event time <= T
  [ok] Polars temporal filtering: as_of() as an independent cross-check
  [ok] Quantified leakage impact: signed RMSE gap and buffer coverage

  KEY INSIGHT: A model that "works great in development" but fails in
  production almost always has a leakage bug. PIT retrieval is the
  structural fix — it makes leakage impossible, not just unlikely.

  Next: In 03_rolling_features.py, you'll extend the feature schema
  with rolling market statistics that capture town-level price trends
  and transaction volumes over time.
"""
)
