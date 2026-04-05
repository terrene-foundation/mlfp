# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT1 — Exercise 3: Functions and Aggregation
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Write reusable Python functions and use Polars group_by/agg
#   to compute district-level statistics — the foundation of every
#   data summary you will write in this course.
#
# TASKS:
#   1. Write helper functions with def, parameters, and return values
#   2. Use group_by() + agg() to summarise data by category
#   3. Compute mean, median, std, count, and quantiles per district
#   4. Apply functions inside Polars expressions with map_elements()
#   5. Build a ranked district report and iterate over it with a for loop
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = ASCENTDataLoader()
hdb = loader.load("ascent01", "hdb_resale.parquet")

# Add price per sqm — we use this throughout the exercise
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
)

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
    # TODO: Return a formatted string like "S$485,000" using an f-string
    return ____  # Hint: f"S${amount:,.0f}"


def price_range_label(price: float) -> str:
    """Classify a resale price into a human-readable tier."""
    # TODO: Return "Budget (<350k)" when price < 350_000
    if price < ____:  # Hint: 350_000
        return ____  # Hint: "Budget (<350k)"
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
    # TODO: Compute q75 (0.75 quantile) and q25 (0.25 quantile) of series
    q75 = series.quantile(____)  # Hint: 0.75
    q25 = series.quantile(____)  # Hint: 0.25
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


# ══════════════════════════════════════════════════════════════════════
# TASK 2: group_by() + agg() — the aggregation pattern
# ══════════════════════════════════════════════════════════════════════

# group_by() splits the DataFrame into groups, one per unique value.
# agg() then computes a summary statistic for each group.
# The result has one row per group and one column per aggregation.

# This is the most important Polars pattern in data analysis:
# "For each [group], compute [statistics]."

district_stats = (
    # TODO: Group hdb by "town"
    hdb.group_by(____).agg(  # Hint: "town"
        # Count: how many transactions in each town?
        pl.len().alias("transaction_count"),
        # Price statistics
        pl.col("resale_price").mean().alias("mean_price"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("resale_price").std().alias("std_price"),
        pl.col("resale_price").min().alias("min_price"),
        pl.col("resale_price").max().alias("max_price"),
        # TODO: Compute 25th and 75th percentile of resale_price
        pl.col("resale_price").quantile(____).alias("q25_price"),  # Hint: 0.25
        pl.col("resale_price").quantile(____).alias("q75_price"),  # Hint: 0.75
        # Price per sqm — a normalised measure for comparing differently-sized flats
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        # Area statistics
        pl.col("floor_area_sqm").median().alias("median_area_sqm"),
    )
    # TODO: Sort by median_price in descending order
    .sort(____, descending=____)  # Hint: "median_price", True
)

print(f"\n=== District Statistics ===")
print(f"Districts: {district_stats.height}")
print(district_stats.head(5))


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Derived columns from aggregated results
# ══════════════════════════════════════════════════════════════════════

# Once you have aggregated stats, you can add further derived columns
# with with_columns() — just like on any other DataFrame.

district_stats = district_stats.with_columns(
    # IQR: spread of the middle 50% of prices
    # TODO: Compute iqr_price as q75_price minus q25_price
    (pl.col("q75_price") - pl.col(____)).alias("iqr_price"),  # Hint: "q25_price"
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


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Grouping with multiple keys + time-series aggregation
# ══════════════════════════════════════════════════════════════════════

# group_by() accepts multiple columns — create a group for every
# unique (town, flat_type) combination.

town_flat_stats = (
    # TODO: Group by both "town" and "flat_type"
    hdb.group_by(____, ____)  # Hint: "town", "flat_type"
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


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Iterate over results with a for loop and build a report
# ══════════════════════════════════════════════════════════════════════

# for loops let you process each item in a sequence.
# .iter_rows(named=True) yields each row as a dict — very readable.


# Write a function that generates a district summary string
def district_report_line(row: dict) -> str:
    """Format one district row as a human-readable report line."""
    town = row["town"]
    # TODO: Use format_sgd() to format the median price
    median = format_sgd(____)  # Hint: row["median_price"]
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
# TODO: Iterate over top_15 rows using iter_rows(named=True)
for row in top_15.iter_rows(named=____):  # Hint: True
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

print("\n✓ Exercise 3 complete — functions and group_by/agg aggregation")
