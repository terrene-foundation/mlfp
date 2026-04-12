# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 5: Window Functions and Trends
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Compute rolling averages and YoY changes using window functions
#   - Use .over() to partition window calculations by group
#   - Identify trends and seasonality in time-series data
#   - Understand when lazy evaluation (.lazy() / .collect()) helps performance
#   - Rank and classify towns by growth trajectory
#
# PREREQUISITES: Complete Exercise 4 first (joins, multi-table data).
#
# ESTIMATED TIME: 50-60 minutes
#
# TASKS:
#   1. Build monthly price series per town using group_by + sort
#   2. Apply rolling_mean() with over() for per-group rolling windows
#   3. Compute year-over-year price change with shift() over partitions
#   4. Rank towns by recent price growth using lazy frames and collect()
#   5. Identify trend leaders and laggards across the Singapore market
#
# DATASET: Singapore HDB resale flat transactions (time-series focus)
#   Source: Housing & Development Board (data.gov.sg)
#   Rows: ~500,000 transactions spanning multiple years
#   Key columns: month, town, resale_price, floor_area_sqm
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

# Parse dates and add derived columns used throughout
hdb = hdb.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
)

print("=" * 60)
print("  MLFP01 Exercise 5: Window Functions and Trends")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.height:,} rows, {hdb.width} columns)")
print(f"  Date range: {hdb['transaction_date'].min()} to {hdb['transaction_date'].max()}")
print(f"  You're ready to start!\n")

print("=== HDB Resale Dataset ===")
print(f"Shape: {hdb.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build monthly price series per town
# ══════════════════════════════════════════════════════════════════════

# Before applying window functions we need a monthly aggregate.
# Each row in this DataFrame represents one (town, month) combination.
# Sorting by town then date is critical — rolling windows depend on order.

monthly_prices = (
    hdb.group_by("town", "transaction_date")
    .agg(
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        pl.col("resale_price").median().alias("median_resale_price"),
        pl.len().alias("transaction_count"),
    )
    .sort("town", "transaction_date")
)

print(f"\n=== Monthly Price Series ===")
print(f"Shape: {monthly_prices.shape}  (one row per town per month)")
print(monthly_prices.filter(pl.col("town") == "BISHAN").head(6))
# INTERPRETATION: This monthly aggregate is the foundation for all trend
# analysis. median_price_sqm (not resale_price) is the right metric for
# cross-month comparisons because it normalises for flat size mix — if one
# month happens to have more large flats sold, raw resale_price would be
# inflated even if the underlying market hasn't moved.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert monthly_prices.height > 0, "monthly_prices should have rows"
# Each row should represent a unique (town, transaction_date) pair
n_unique = monthly_prices.select(["town", "transaction_date"]).unique().height
assert monthly_prices.height == n_unique, (
    "monthly_prices should have one row per (town, transaction_date) pair"
)
# Data should be sorted: first date should be <= last date for Bishan
bishan = monthly_prices.filter(pl.col("town") == "BISHAN")
assert bishan["transaction_date"][0] <= bishan["transaction_date"][-1], (
    "Data should be sorted by transaction_date"
)
print("\n✓ Checkpoint 1 passed — monthly price series built correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Rolling average with over() — per-group window functions
# ══════════════════════════════════════════════════════════════════════

# A window function computes a value for each row using a sliding window
# of surrounding rows — without collapsing the DataFrame like group_by does.
#
# rolling_mean(window_size=12) computes a 12-row moving average.
# .over("town") partitions the computation by town, so the window
# never crosses town boundaries. Each town gets its own independent window.
#
# Why 12 months? Seasonal smoothing: Singapore's rainy/dry season cycle
# affects transaction volume, so a 12-month window removes that noise.

monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm")
    .rolling_mean(window_size=12)
    .over("town")
    .alias("rolling_12m_price_sqm"),
)

# A shorter 3-month window for a more reactive signal
monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm")
    .rolling_mean(window_size=3)
    .over("town")
    .alias("rolling_3m_price_sqm"),
)

print(f"\n=== Rolling Averages — Bishan (last 18 months) ===")
bishan = monthly_prices.filter(pl.col("town") == "BISHAN").tail(18)
print(
    bishan.select(
        "transaction_date",
        "median_price_sqm",
        "rolling_3m_price_sqm",
        "rolling_12m_price_sqm",
    )
)
# INTERPRETATION: rolling_3m reacts quickly to price changes — useful for
# detecting early market turns. rolling_12m is much smoother — it shows the
# underlying trend without month-to-month noise. When rolling_3m crosses
# above rolling_12m, it often signals an accelerating market (and vice versa).
# This is the same logic as the "golden cross" used in technical analysis.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert "rolling_12m_price_sqm" in monthly_prices.columns, (
    "rolling_12m_price_sqm column should exist"
)
assert "rolling_3m_price_sqm" in monthly_prices.columns, (
    "rolling_3m_price_sqm column should exist"
)
# First 11 rows per town should be null (not enough data for 12-month window)
bishan_all = monthly_prices.filter(pl.col("town") == "BISHAN").sort("transaction_date")
assert bishan_all["rolling_12m_price_sqm"][0] is None, (
    "First row of rolling_12m should be null (window not full yet)"
)
print("\n✓ Checkpoint 2 passed — rolling averages computed per town with over()\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Year-over-year price change with shift()
# ══════════════════════════════════════════════════════════════════════

# shift(n) moves every value n positions forward, filling the first n
# rows with null. Combined with .over("town"), each town shifts independently.
#
# YoY change = (current - 12_months_ago) / 12_months_ago * 100
#
# shift(12) gives us the value from exactly 12 months earlier for each town.
# The first 12 rows per town will be null — that is correct and expected.

monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm").shift(12).over("town").alias("price_sqm_12m_ago"),
)

monthly_prices = monthly_prices.with_columns(
    (
        (pl.col("median_price_sqm") - pl.col("price_sqm_12m_ago"))
        / pl.col("price_sqm_12m_ago")
        * 100
    ).alias("yoy_price_change_pct"),
)

print(f"\n=== Year-over-Year Price Change — Bishan (last 24 months) ===")
print(
    monthly_prices.filter(pl.col("town") == "BISHAN")
    .tail(24)
    .select(
        "transaction_date",
        "median_price_sqm",
        "price_sqm_12m_ago",
        "yoy_price_change_pct",
    )
)
# INTERPRETATION: yoy_price_change_pct > 0 means prices are higher than
# the same month a year ago — an appreciating market. Negative values
# indicate nominal price declines (rare in Singapore but did occur during
# the 2013–2017 cooling measure period). Large spikes (>10%) often coincide
# with policy changes: the 2022–2023 post-COVID rebound saw 15–20% YoY gains.

# Market-wide YoY: which months had the strongest national price growth?
national_monthly = (
    hdb.group_by("transaction_date")
    .agg(pl.col("price_per_sqm").median().alias("national_median_sqm"))
    .sort("transaction_date")
    .with_columns(
        pl.col("national_median_sqm").shift(12).alias("national_sqm_12m_ago"),
    )
    .with_columns(
        (
            (pl.col("national_median_sqm") - pl.col("national_sqm_12m_ago"))
            / pl.col("national_sqm_12m_ago")
            * 100
        ).alias("national_yoy_pct"),
    )
)

print(f"\n=== National YoY Price Change (top 10 months) ===")
print(
    national_monthly.drop_nulls("national_yoy_pct")
    .sort("national_yoy_pct", descending=True)
    .select("transaction_date", "national_median_sqm", "national_yoy_pct")
    .head(10)
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert "yoy_price_change_pct" in monthly_prices.columns, (
    "yoy_price_change_pct column should exist"
)
assert "national_yoy_pct" in national_monthly.columns, (
    "national_yoy_pct column should exist"
)
# First 12 rows per town should be null (shift of 12)
bishan_sorted = monthly_prices.filter(pl.col("town") == "BISHAN").sort("transaction_date")
assert bishan_sorted["yoy_price_change_pct"][0] is None, (
    "First yoy_price_change_pct per town should be null"
)
print("\n✓ Checkpoint 3 passed — YoY price change computed correctly with shift()\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Lazy frames — defer execution until collect()
# ══════════════════════════════════════════════════════════════════════

# .lazy() converts a DataFrame into a LazyFrame.
# Nothing executes immediately — Polars builds a query plan instead.
# .collect() triggers execution. Polars optimises the plan before running:
#   - Predicate pushdown: filters are applied as early as possible
#   - Projection pushdown: unused columns are dropped before reading
#   - Query fusion: adjacent operations are combined
#
# For large datasets, lazy evaluation is significantly faster because
# Polars avoids materialising intermediate results.

# Find towns with consistently high YoY growth over the last 3 years
# Using lazy evaluation for the multi-step aggregation pipeline

recent_yoy = (
    monthly_prices.lazy()
    # Filter to last 3 years of data — lazy: no data moves yet
    .filter(pl.col("transaction_date") >= pl.date(2021, 1, 1))
    # Drop rows where YoY is null (first 12 months per town)
    .drop_nulls("yoy_price_change_pct")
    # Aggregate per town
    .group_by("town").agg(
        pl.col("yoy_price_change_pct").mean().alias("mean_yoy_pct"),
        pl.col("yoy_price_change_pct").std().alias("std_yoy_pct"),
        pl.col("yoy_price_change_pct").max().alias("peak_yoy_pct"),
        pl.col("yoy_price_change_pct").min().alias("trough_yoy_pct"),
        pl.len().alias("months_of_data"),
    )
    # Sort by mean YoY descending
    .sort("mean_yoy_pct", descending=True)
    # collect() executes the entire plan — this is where computation happens
    .collect()
)

print(f"\n=== Town YoY Growth Rankings (2021–present) ===")
print(f"Towns analysed: {recent_yoy.height}")
print(recent_yoy.head(10))
# INTERPRETATION: mean_yoy_pct is the average annual appreciation rate
# for each town since 2021. A town with mean_yoy=8% has grown at 8% per
# year on average — significantly above Singapore's historical norm of ~3%.
# peak_yoy shows the maximum single-month gain: outliers here may indicate
# a one-off en-bloc effect or policy change rather than sustained growth.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert recent_yoy.height > 0, "recent_yoy should have rows"
assert "mean_yoy_pct" in recent_yoy.columns, "mean_yoy_pct column should exist"
# Sorted descending: first row should have the highest mean YoY
assert recent_yoy["mean_yoy_pct"][0] >= recent_yoy["mean_yoy_pct"][-1], (
    "recent_yoy should be sorted by mean_yoy_pct descending"
)
print("\n✓ Checkpoint 4 passed — lazy evaluation pipeline collected correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Trend leaders and laggards
# ══════════════════════════════════════════════════════════════════════

# A rank window function assigns a position to each row within its partition.
# pl.col("mean_yoy_pct").rank() ranks all towns by growth rate.
# This is different from sort() — rank adds a column without reordering rows.

recent_yoy = recent_yoy.with_columns(
    pl.col("mean_yoy_pct").rank(method="ordinal", descending=True).alias("growth_rank"),
)

# Classify towns as trend leaders, followers, or laggards
mean_growth = recent_yoy["mean_yoy_pct"].mean()
std_growth = recent_yoy["mean_yoy_pct"].std()

recent_yoy = recent_yoy.with_columns(
    pl.when(pl.col("mean_yoy_pct") > mean_growth + std_growth)
    .then(pl.lit("leader"))
    .when(pl.col("mean_yoy_pct") < mean_growth - std_growth)
    .then(pl.lit("laggard"))
    .otherwise(pl.lit("follower"))
    .alias("trend_category"),
)

print(f"\n=== Trend Classification ===")
print(f"Mean YoY growth (all towns): {mean_growth:.2f}%")
print(f"Std dev: {std_growth:.2f}%")

category_counts = (
    recent_yoy.group_by("trend_category")
    .agg(pl.len().alias("count"), pl.col("mean_yoy_pct").mean().alias("avg_yoy"))
    .sort("avg_yoy", descending=True)
)
print(category_counts)
# INTERPRETATION: Leaders are towns growing more than 1 std dev above the
# national average — they're outperforming the market. Laggards are growing
# below 1 std dev from the mean. Under a normal distribution, ~16% of towns
# should be leaders and ~16% laggards. If the counts differ significantly,
# the growth distribution is skewed — some towns are pulling far ahead.

print(f"\n=== Trend Leaders (fastest-growing towns) ===")
print(
    recent_yoy.filter(pl.col("trend_category") == "leader").select(
        "town", "mean_yoy_pct", "peak_yoy_pct", "growth_rank"
    )
)

print(f"\n=== Trend Laggards (slowest-growing towns) ===")
print(
    recent_yoy.filter(pl.col("trend_category") == "laggard").select(
        "town", "mean_yoy_pct", "trough_yoy_pct", "growth_rank"
    )
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert "trend_category" in recent_yoy.columns, "trend_category column should exist"
assert "growth_rank" in recent_yoy.columns, "growth_rank column should exist"
categories = set(recent_yoy["trend_category"].unique().to_list())
assert categories == {"leader", "follower", "laggard"}, (
    f"Expected 3 categories, got: {categories}"
)
print("\n✓ Checkpoint 5 passed — trend classification complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 58)
print("  WHAT YOU'VE MASTERED")
print("═" * 58)
print("""
  ✓ Monthly aggregation: building a time-series base table
  ✓ rolling_mean(): smoothing noisy price data with a sliding window
  ✓ .over("town"): partitioning window functions by group
  ✓ shift(12): comparing each month to the same month a year ago
  ✓ YoY calculation: (current - prior) / prior * 100
  ✓ .lazy() / .collect(): deferring execution for query optimization
  ✓ rank(): adding a ranking column without reordering the DataFrame
  ✓ Trend classification: leader / follower / laggard using std dev bands

  NEXT: In Exercise 6, you'll turn these numbers into interactive
  charts using ModelVisualizer — the Kailash engine wrapping Plotly.
  You'll build histograms, scatter plots, bar charts, heatmaps, and
  line charts. The goal: make the patterns you've found in exercises
  1–5 visible and communicable.
""")
