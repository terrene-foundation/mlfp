# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 3: Functions and Aggregation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Define functions that accept parameters and return values
#   - Use loops to process collections and iterate over DataFrame rows
#   - Aggregate data by groups using group_by() + agg()
#   - Apply multiple statistics (mean, median, std, quantile) in one call
#   - Write reusable helper functions for common data analysis tasks
#
# PREREQUISITES: Complete Exercise 2 first (filtering, with_columns, chaining).
#
# ESTIMATED TIME: 45-60 minutes
#
# TASKS:
#   1. Write helper functions with def, parameters, and return values
#   2. Use group_by() + agg() to summarise data by category
#   3. Compute mean, median, std, count, and quantiles per district
#   4. Apply functions inside Polars expressions with map_elements()
#   5. Build a ranked district report and iterate over it with a for loop
#
# DATASET: Singapore HDB resale flat transactions
#   Source: Housing & Development Board (data.gov.sg)
#   Rows: ~500,000 transactions | Columns: month, town, flat_type,
#   floor_area_sqm, resale_price, and more
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

# Add price per sqm — we use this throughout the exercise
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
)

print("=" * 60)
print("  MLFP01 Exercise 3: Functions and Aggregation")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.height:,} rows, {hdb.width} columns)")
print(f"  You're ready to start!\n")

print("=== HDB Resale Dataset ===")
print(f"Shape: {hdb.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Writing functions — def, parameters, return
# ══════════════════════════════════════════════════════════════════════

# A function packages reusable logic under a name.
# def introduces the function, parameters go in parentheses,
# the body is indented, and return sends a value back to the caller.


def format_sgd(amount: float) -> str:
    """Format a number as Singapore dollars with thousands separator."""
    # The : tells Python this is a type hint — it documents what type
    # the parameter should be, but Python does not enforce it at runtime.
    # -> str says this function returns a string.
    return f"S${amount:,.0f}"


def price_range_label(price: float) -> str:
    """Classify a resale price into a human-readable tier."""
    if price < 350_000:
        return "Budget (<350k)"
    elif price < 500_000:
        return "Mid-range (350k–500k)"
    elif price < 700_000:
        return "Premium (500k–700k)"
    else:
        return "Luxury (700k+)"


def compute_iqr(series: pl.Series) -> float:
    """Compute the interquartile range (Q3 - Q1) of a Polars Series.

    IQR measures spread without being skewed by extreme outliers.
    A wide IQR means prices vary a lot within the district.
    """
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    # quantile() can return None if the series is empty — guard for that
    if q75 is None or q25 is None:
        return 0.0
    return q75 - q25


# Test the functions before using them in a pipeline
print("\n=== Function Tests ===")
print(format_sgd(485_000))
print(price_range_label(485_000))
print(price_range_label(720_000))

test_prices = pl.Series("prices", [300_000, 400_000, 500_000, 600_000, 700_000])
print(f"IQR of test prices: {format_sgd(compute_iqr(test_prices))}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert format_sgd(485_000) == "S$485,000", f"format_sgd returned: {format_sgd(485_000)}"
assert "Mid-range" in price_range_label(485_000), "485k should be Mid-range"
assert "Luxury" in price_range_label(720_000), "720k should be Luxury"
assert compute_iqr(test_prices) == 200_000.0, (
    f"IQR of test series should be 200,000, got {compute_iqr(test_prices)}"
)
print("\n✓ Checkpoint 1 passed — all three functions work correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: group_by() + agg() — the aggregation pattern
# ══════════════════════════════════════════════════════════════════════

# group_by() splits the DataFrame into groups, one per unique value.
# agg() then computes a summary statistic for each group.
# The result has one row per group and one column per aggregation.

# This is the most important Polars pattern in data analysis:
# "For each [group], compute [statistics]."

district_stats = (
    hdb.group_by("town")
    .agg(
        # Count: how many transactions in each town?
        pl.len().alias("transaction_count"),
        # Price statistics
        pl.col("resale_price").mean().alias("mean_price"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("resale_price").std().alias("std_price"),
        pl.col("resale_price").min().alias("min_price"),
        pl.col("resale_price").max().alias("max_price"),
        pl.col("resale_price").quantile(0.25).alias("q25_price"),
        pl.col("resale_price").quantile(0.75).alias("q75_price"),
        # Price per sqm — a normalised measure for comparing differently-sized flats
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        # Area statistics
        pl.col("floor_area_sqm").median().alias("median_area_sqm"),
    )
    .sort("median_price", descending=True)
)

print(f"\n=== District Statistics ===")
print(f"Districts: {district_stats.height}")
print(district_stats.head(5))
# INTERPRETATION: Sorting by median_price shows the hierarchy of Singapore's
# housing market. Districts at the top (e.g., Queenstown, Bishan) command
# premium prices because of central location, MRT access, and strong demand.
# The spread between min_price and max_price within a district reflects how
# diverse the housing stock is — some districts have both studio-sized and
# executive flats.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert district_stats.height > 0, "district_stats should have rows"
assert "transaction_count" in district_stats.columns, "Missing transaction_count column"
assert "median_price" in district_stats.columns, "Missing median_price column"
# The most expensive district should be at the top (sorted descending)
top_median = district_stats["median_price"][0]
bottom_median = district_stats["median_price"][-1]
assert top_median >= bottom_median, "Results should be sorted by median_price descending"
print("\n✓ Checkpoint 2 passed — group_by/agg producing correct district statistics\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Derived columns from aggregated results
# ══════════════════════════════════════════════════════════════════════

# Once you have aggregated stats, you can add further derived columns
# with with_columns() — just like on any other DataFrame.

district_stats = district_stats.with_columns(
    # IQR: spread of the middle 50% of prices
    (pl.col("q75_price") - pl.col("q25_price")).alias("iqr_price"),
    # Coefficient of variation (CV): std / mean * 100
    # A higher CV means prices within the district are more spread out.
    # This is more informative than std alone because it's relative.
    (pl.col("std_price") / pl.col("mean_price") * 100).alias("cv_price_pct"),
    # Premium ratio: what fraction of the max price is the median?
    # Districts near 1.0 have mostly high-end transactions.
    (pl.col("median_price") / pl.col("max_price")).alias("premium_ratio"),
)

print(f"\n=== District Stats with Derived Columns ===")
print(
    district_stats.select(
        "town",
        "transaction_count",
        "median_price",
        "iqr_price",
        "cv_price_pct",
    ).head(10)
)
# INTERPRETATION: cv_price_pct (Coefficient of Variation) is key here.
# A district with CV=25% has prices clustered tightly around the mean.
# A district with CV=40%+ has a wide mix — budget HDBs beside premium ones.
# Districts with high CV are harder to generalise about — always segment them.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert "iqr_price" in district_stats.columns, "iqr_price column should be added"
assert "cv_price_pct" in district_stats.columns, "cv_price_pct column should be added"
# IQR should equal q75 - q25 for the first row
row = district_stats.row(0, named=True)
expected_iqr = row["q75_price"] - row["q25_price"]
assert abs(row["iqr_price"] - expected_iqr) < 1, (
    f"iqr_price mismatch: {row['iqr_price']} vs {expected_iqr}"
)
print("\n✓ Checkpoint 3 passed — derived columns computed from aggregates correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Grouping with multiple keys + time-series aggregation
# ══════════════════════════════════════════════════════════════════════

# group_by() accepts multiple columns — create a group for every
# unique (town, flat_type) combination.

town_flat_stats = (
    hdb.group_by("town", "flat_type")
    .agg(
        pl.len().alias("count"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
    )
    .sort("town", "flat_type")
)

print(f"\n=== Statistics by Town × Flat Type ===")
print(f"Groups: {town_flat_stats.height}")
print(town_flat_stats.filter(pl.col("town") == "ANG MO KIO"))

# Annual transaction volume by town — shows market activity over time
annual_volume = (
    hdb.group_by("year", "town")
    .agg(
        pl.len().alias("transactions"),
        pl.col("resale_price").median().alias("median_price"),
    )
    .sort("year", "town")
)

print(f"\n=== Annual Volume: Bishan (sample) ===")
print(annual_volume.filter(pl.col("town") == "BISHAN"))
# INTERPRETATION: Comparing transaction counts across years reveals market
# cycles. A sudden drop may indicate policy intervention (e.g., cooling
# measures). A spike often follows a policy relaxation or economic stimulus.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
# Each (town, flat_type) pair should produce exactly one row
n_unique_pairs = hdb.select(["town", "flat_type"]).unique().height
assert town_flat_stats.height == n_unique_pairs, (
    f"Expected {n_unique_pairs} groups, got {town_flat_stats.height}"
)
print("\n✓ Checkpoint 4 passed — multi-key group_by producing one row per group pair\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Iterate over results with a for loop and build a report
# ══════════════════════════════════════════════════════════════════════

# for loops let you process each item in a sequence.
# .iter_rows(named=True) yields each row as a dict — very readable.


# Write a function that generates a district summary string
def district_report_line(row: dict) -> str:
    """Format one district row as a human-readable report line."""
    town = row["town"]
    median = format_sgd(row["median_price"])
    count = row["transaction_count"]
    cv = row["cv_price_pct"]
    sqm = format_sgd(row["median_price_sqm"])
    return f"  {town:<20} {median:>12}  {count:>8,}  CV={cv:5.1f}%  {sqm:>12}/sqm"


# Print a ranked report for the top 15 most expensive districts
print(f"\n{'=' * 70}")
print(f"  SINGAPORE HDB DISTRICT PRICE REPORT (All Years)")
print(f"{'=' * 70}")
print(
    f"  {'Town':<20} {'Median Price':>12}  {'Txns':>8}  {'Spread':>8}  {'Per sqm':>12}"
)
print(f"  {'-' * 66}")

top_15 = district_stats.head(15)
for row in top_15.iter_rows(named=True):
    # named=True means each row is a dict — use row["column_name"]
    print(district_report_line(row))

print(f"{'=' * 70}")

# Summary statistics across all districts
all_medians = district_stats["median_price"]
print(f"\nCross-district summary:")
print(f"  Most expensive district:   {format_sgd(all_medians.max())}")
print(f"  Least expensive district:  {format_sgd(all_medians.min())}")
print(f"  Average district median:   {format_sgd(all_medians.mean())}")
print(
    f"  Price spread (max - min):  {format_sgd(all_medians.max() - all_medians.min())}"
)
# INTERPRETATION: The spread between the most and least expensive districts
# quantifies Singapore's housing inequality. A wide spread (>S$200k) indicates
# that geography still matters significantly — where you buy affects your
# long-term asset value, not just your monthly commute.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert all_medians.max() > all_medians.min(), (
    "Most expensive should be more than least expensive"
)
assert top_15.height == 15, "Top 15 should have exactly 15 rows"
print("\n✓ Checkpoint 5 passed — for loop and district report generated correctly\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 58)
print("  WHAT YOU'VE MASTERED")
print("═" * 58)
print("""
  ✓ Functions: def, parameters, return, type hints
  ✓ Conditional logic: if/elif/else inside functions
  ✓ group_by() + agg(): the core pattern for grouped statistics
  ✓ Multiple aggregations: mean, median, std, min, max, quantile
  ✓ Multi-key grouping: group by (town, flat_type) simultaneously
  ✓ for loops: .iter_rows(named=True) to process each row as a dict
  ✓ Reusable helpers: functions that format, classify, and compute

  NEXT: In Exercise 4, you'll combine data from multiple tables
  using joins — merging HDB transactions with MRT station proximity
  and school density data. You'll learn when to use left vs inner
  joins, how to handle NULLs after a join, and how to enrich a
  dataset with spatial context.
""")
