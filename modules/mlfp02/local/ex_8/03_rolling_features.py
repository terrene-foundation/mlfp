# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 8.3: Rolling Features — Temporal Market Context
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define FeatureSchema v2 with rolling market-context features
#   - Compute TRAILING rolling statistics with Polars group_by_dynamic
#   - Understand rolling window warm-up periods and null handling
#   - Track schema evolution from v1 to v2 in the FeatureStore
#   - Apply rolling market features to Singapore town-level analytics
#
# PREREQUISITES: Exercise 8.1-8.2 (FeatureSchema v1, PIT retrieval)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — why rolling features capture market momentum
#   2. Build — define schema v2 and compute rolling town statistics
#   3. Train — materialise v2 into the FeatureStore and read back
#   4. Visualise — rolling price trends and transaction volumes by town
#   5. Apply — town-level advisory for a Singapore real-estate agency
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from shared.mlfp02.ex_8 import (
    OUTPUT_DIR,
    build_schema_v1,
    build_schema_v2,
    compute_v1_features,
    compute_v2_features,
    create_feature_store,
    load_hdb_resale,
    materialize_features,
    validate_v1_features,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Rolling Features Capture Market Momentum
# ════════════════════════════════════════════════════════════════════════
# A single transaction price is a noisy signal. It depends on the
# specific flat's condition, the buyer's urgency, the agent's skill,
# and random timing. But the MEDIAN price in a town over the past 6
# months is a stable signal — it captures the local market's direction.
#
# Rolling features aggregate noisy individual observations into smooth
# market-level statistics:
#
#   - town_median_price: "What's the typical price in this town lately?"
#   - town_transaction_volume: "Is this a hot market or a cold one?"
#   - town_price_trend: "Are prices going up, down, or flat?"
#
# These three features transform a model from "predict based on this
# flat's characteristics" to "predict based on this flat in THIS market
# context". The same flat in a booming town sells for more than in a
# stagnant one.
#
# Polars group_by_dynamic is the engine: it buckets transactions into
# monthly windows per town, then rolling_mean/rolling_sum aggregates
# across a 6-month TRAILING window. "Trailing" must mean months m-6 to
# m-1 for a sale in month m: the current month's median contains the
# sale's own price, so including it would leak the target into the
# feature. The series is therefore shifted by one month before rolling.
# The first 6 months per town (7 for the trend) have nulls — the warm-up
# period where the window hasn't filled yet.
#
# Singapore context: HDB towns like Bishan, Tampines, and Woodlands
# have very different price trajectories. A 4-room flat in Bishan
# (mature estate, near MRT) appreciates differently from Woodlands
# (non-mature estate). Rolling features capture this divergence.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Define FeatureSchema v2 and compute rolling features
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exercise 8.3 — Rolling Features: Temporal Market Context")
print("=" * 70)

# --- 2a. Load and compute validated v1 features (baseline, as in 8.1) ---
hdb = load_hdb_resale()
features_v1, _ = validate_v1_features(compute_v1_features(hdb))

# --- 2b. Define FeatureSchema v2 ---
property_schema_v1 = build_schema_v1()
# TODO: Call the shared helper to build the v2 schema.
# Hint: build_schema_v2() returns a FeatureSchema that extends v1
# with three market-context fields: town_median_price,
# town_transaction_volume, town_price_trend.
property_schema_v2 = ____

n_new = len(property_schema_v2.fields) - len(property_schema_v1.fields)
print(f"\n  === FeatureSchema v2 (+{n_new} market features) ===")
for f in property_schema_v2.fields:
    tag = " [NEW]" if f.name not in property_schema_v1.field_names else ""
    print(f"    {f.name}: {f.dtype} (nullable={f.nullable}){tag}")

# --- 2c. Compute v2 features ---
# TODO: Call compute_v2_features(hdb) to compute trailing market features.
# Hint: compute_v2_features shifts each town's monthly series by one
# month, then applies a 6-month rolling window per town.
features_v2 = ____

# TODO: Count how many rows have non-null market context.
# Hint: filter for rows where town_median_price is not null, then .height
n_with_market = ____
pct_with_market = n_with_market / features_v2.height

print(f"\n  Computed v2 features: {features_v2.shape}")
print(f"  Rows with market context: {n_with_market:,} ({pct_with_market:.1%})")
print(f"  (First 6 months per town have nulls — rolling window warm-up)")

# --- 2c-bis. Leakage check: the window must exclude the current month ---
check_town = "BISHAN"
monthly = (
    features_v2.filter(pl.col("town") == check_town)
    .group_by("transaction_date")
    .agg(
        pl.col("resale_price").median().alias("monthly_median"),
        pl.col("town_median_price").first(),
    )
    .sort("transaction_date")
)
row_m = monthly.filter(pl.col("town_median_price").is_not_null()).row(0, named=True)
m_index = monthly["transaction_date"].to_list().index(row_m["transaction_date"])
# TODO: Mean of the 6 monthly medians BEFORE month m (exclude month m).
# Hint: slice monthly["monthly_median"] from m_index - 6 up to (not
#   including) m_index, then .mean()
prior_six = ____
print(f"\n  Leakage check ({check_town}, {row_m['transaction_date']}):")
print(f"    feature value:              ${row_m['town_median_price']:,.0f}")
print(f"    mean of 6 PRIOR month medians: ${prior_six:,.0f}")

# --- 2c-ter. Calendar features (plain Polars) ---
features_v2 = features_v2.with_columns(
    pl.col("transaction_date").dt.month().alias("transaction_date_month"),
    pl.col("transaction_date").dt.quarter().alias("transaction_date_quarter"),
)
print("\n  Calendar features added with Polars .dt accessors: month, quarter")
print(f"  Columns after temporal extraction: {features_v2.shape[1]}")

# INTERPRETATION: Calendar features capture seasonality that rolling
# aggregates alone miss (e.g. Q1 vs Q4 buyer behaviour). The rolling
# market features above are domain-specific; temporal calendar features
# are mechanical and reused across ML pipelines.

# --- 2d. Show sample rolling values for a few towns ---
sample_towns = ["ANG MO KIO", "BISHAN", "TAMPINES", "WOODLANDS"]
for town in sample_towns:
    town_data = features_v2.filter(
        (pl.col("town") == town) & pl.col("town_median_price").is_not_null()
    )
    if town_data.height > 0:
        latest = town_data.sort("transaction_date").tail(1)
        median_p = latest["town_median_price"].item()
        volume = latest["town_transaction_volume"].item()
        trend = latest["town_price_trend"].item()
        print(
            f"    {town:<15}: median=${median_p:>10,.0f}  "
            f"volume={volume:>5}  trend={trend:>+.1f}%"
        )


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert (
    features_v2.height == features_v1.height
), "Task 2: v2 should have same row count as v1"
assert (
    "town_median_price" in features_v2.columns
), "Task 2: town_median_price must be computed"
assert (
    "town_transaction_volume" in features_v2.columns
), "Task 2: town_transaction_volume must be computed"
assert (
    "town_price_trend" in features_v2.columns
), "Task 2: town_price_trend must be computed"
assert (
    pct_with_market > 0.5
), f"Task 2: at least 50% of rows should have market context, got {pct_with_market:.1%}"
assert abs(row_m["town_median_price"] - prior_six) < 1e-6, (
    "Task 2: the rolling feature must use only the 6 months BEFORE the sale"
)
print("\n[ok] Checkpoint 1 passed — v2 features computed with rolling market context\n")

# INTERPRETATION: The v2 schema adds three nullable columns. They're
# nullable because the first months per town can't fill a trailing
# window — that's the warm-up period, not a data quality bug. The
# leakage check confirms the feature for month m uses months m-6..m-1.
# Downstream models must drop_nulls on these columns before training.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Register v2 schema and store versioned features
# ════════════════════════════════════════════════════════════════════════

print("--- FeatureStore Schema Evolution (v1 -> v2) ---")
# In kailash-ml 2.2.x the store keeps one backing table per schema NAME,
# so a version that adds columns is stored under its own name
# (hdb_property_features_v2) while carrying version=2 for lineage. v1
# rows from 8.1 stay untouched — models trained on v1 remain reproducible.

# TODO: Materialise the v2 features and read them back.
# Hint: same calls as 8.1 — create_feature_store(), materialize_features(...),
#   fs.get_features(schema) — each async call wrapped in asyncio.run
fs = ____
materialized_v2 = ____
stored_v2 = ____
print(f"  Materialised {materialized_v2['row_count']:,} v2 rows "
      f"(schema {materialized_v2['group']} v{materialized_v2['version']})")
print(f"  Read back: {stored_v2.height:,} rows, columns {stored_v2.columns}")

print(f"\n  Schema evolution:")
print(f"    v1: {len(property_schema_v1.fields)} features (basic property)")
print(f"    v2: {len(property_schema_v2.fields)} features (+{n_new} market context)")
print(f"    New fields: town_median_price, town_transaction_volume, town_price_trend")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert property_schema_v2.version == 2, "Task 3: v2 schema must be version 2"
assert len(property_schema_v2.fields) == 7, "Task 3: v2 must have 7 features"
assert stored_v2.height == features_v2.height, "Task 3: every v2 row must be stored"
print("\n[ok] Checkpoint 2 passed — v2 features materialised and read back\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Rolling price trends and volumes by town
# ════════════════════════════════════════════════════════════════════════

print("--- Town-Level Rolling Market Trends ---")

# Aggregate monthly medians per town for plotting
town_monthly = (
    features_v2.filter(pl.col("town_median_price").is_not_null())
    .group_by(["town", "transaction_date"])
    .agg(
        pl.col("town_median_price").first(),
        pl.col("town_transaction_volume").first(),
        pl.col("town_price_trend").first(),
    )
    .sort("transaction_date")
)

# Plot rolling median prices for selected towns
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        "Rolling 6-Month Median Price by Town",
        "Rolling 6-Month Transaction Volume by Town",
    ],
    vertical_spacing=0.12,
)

for town in sample_towns:
    town_data = town_monthly.filter(pl.col("town") == town).sort("transaction_date")
    if town_data.height > 0:
        dates = town_data["transaction_date"].to_list()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=town_data["town_median_price"].to_list(),
                name=town,
                mode="lines",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=town_data["town_transaction_volume"].to_list(),
                name=town,
                showlegend=False,
                mode="lines",
            ),
            row=2,
            col=1,
        )

fig.update_layout(
    title="HDB Market Trends by Town (Rolling 6-Month Window)",
    height=700,
    width=1000,
)
fig.update_yaxes(title_text="Median Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Transaction Count", row=2, col=1)

fig.write_html(str(OUTPUT_DIR / "03_town_trends.html"))
print(f"\n  Saved: {OUTPUT_DIR / '03_town_trends.html'}")

# Price trend distribution (how many towns are appreciating vs declining)
trend_data = (
    features_v2.filter(pl.col("town_price_trend").is_not_null())
    .group_by("town")
    .agg(pl.col("town_price_trend").mean().alias("avg_trend"))
)

fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=trend_data.sort("avg_trend")["town"].to_list(),
        y=trend_data.sort("avg_trend")["avg_trend"].to_list(),
        marker_color=[
            "firebrick" if t < 0 else "seagreen"
            for t in trend_data.sort("avg_trend")["avg_trend"].to_list()
        ],
    )
)
fig2.update_layout(
    title="Average 6-Month Price Trend by Town (%)",
    xaxis_title="Town",
    yaxis_title="Average Price Trend (%)",
    xaxis_tickangle=-45,
)
fig2.write_html(str(OUTPUT_DIR / "03_town_price_trends.html"))
print(f"  Saved: {OUTPUT_DIR / '03_town_price_trends.html'}")


# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — rolling market trends visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Town-Level Advisory for a Real-Estate Agency
# ════════════════════════════════════════════════════════════════════════
# Scenario (illustrative figures): advisors at a Singapore real-estate
# agency help HDB upgraders decide WHEN and WHERE to buy. With rolling
# market features they can see which towns are trending up (buy soon)
# and which are flat (negotiate harder).
#
# Without rolling features: advisors rely on gut feel. "Bishan is always
# expensive" — but IS it still appreciating, or has it plateaued?
#
# With rolling features: advisors quote each town's latest 6-month trend,
# computed below from the data, instead of an impression.
#
# Revenue framing (assumed figures): 1,000 agents, one extra deal per
# agent per quarter, S$5,000 commission per deal.

print("=== APPLY: Town-Level Advisory (Singapore real-estate agency) ===")

# Rank towns by their most recent trend value
# TODO: Rank towns by their latest price trend (descending).
# Hint: sort by transaction_date BEFORE group_by so .last() is the most
#   recent month; then sort the result by latest_trend descending.
latest_trends = (
    features_v2.filter(pl.col("town_price_trend").is_not_null())
    .sort(____)
    .group_by("town")
    .agg(
        pl.col("town_price_trend").last().alias("latest_trend"),
        pl.col("town_median_price").last().alias("latest_median"),
        pl.col("town_transaction_volume").last().alias("latest_volume"),
    )
    .sort("latest_trend", descending=True)
)

print()
print("  Top 5 appreciating towns (buy-soon signal):")
for row in latest_trends.head(5).iter_rows(named=True):
    print(
        f"    {row['town']:<15}: trend={row['latest_trend']:>+6.1f}%  "
        f"median=${row['latest_median']:>10,.0f}  volume={row['latest_volume']:>5}"
    )

print()
print("  Bottom 5 towns (negotiate-harder signal):")
for row in latest_trends.tail(5).iter_rows(named=True):
    print(
        f"    {row['town']:<15}: trend={row['latest_trend']:>+6.1f}%  "
        f"median=${row['latest_median']:>10,.0f}  volume={row['latest_volume']:>5}"
    )

agents, extra_deals_per_quarter, commission = 1_000, 1, 5_000  # assumed
print()
print("  Advisory impact (assumed figures):")
print(f"    - {agents:,} agents x {extra_deals_per_quarter} extra deal/quarter x S${commission:,}")
print(f"    - S${agents * extra_deals_per_quarter * 4 * commission:,} additional commission per year")

compare = latest_trends.filter(pl.col("town").is_in(["BISHAN", "TAMPINES"]))
print()
print("  Key insight: 'Bishan is expensive' is qualitative. From the data:")
for row in compare.iter_rows(named=True):
    print(
        f"    {row['town']:<10} median ${row['latest_median']:,.0f}, "
        f"latest 6-month trend {row['latest_trend']:+.1f}%"
    )
print("  — quantitative, current and checkable.")


# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert latest_trends.height > 0, "Task 5: must have town trend data"
print("\n[ok] Checkpoint 4 passed — town-level advisory demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [ok] FeatureSchema v2: extending v1 with rolling market-context fields
  [ok] group_by_dynamic: monthly bucketing of transactions per town
  [ok] Rolling statistics: trailing 6-month median, volume, trend that
       exclude the sale's own month (shift before rolling)
  [ok] Warm-up periods: why the first months per town have nulls
  [ok] Schema versioning: v1 -> v2 evolution with v1 left reproducible

  KEY INSIGHT: Rolling features transform a model from "what is this
  flat worth?" to "what is this flat worth IN THIS MARKET?" The same
  flat in a booming town sells for more than in a stagnant one — and
  rolling features capture that difference quantitatively.

  Next: In 04_modeling_lineage.py, you'll build a full regression model
  on v2 features, apply hypothesis tests and Bayesian posteriors to the
  coefficients, and create a complete audit trail from data to model.
"""
)
