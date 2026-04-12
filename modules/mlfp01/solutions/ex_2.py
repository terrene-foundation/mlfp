# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 2: Filtering and Transforming Data
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Filter rows using Boolean logic with comparison operators (>, <, ==, !=)
#   - Select and rename columns to focus your analysis
#   - Create new derived columns with with_columns() and alias()
#   - Apply conditional column logic with pl.when().then().otherwise()
#   - Chain multiple Polars operations together in a readable pipeline
#
# PREREQUISITES: Complete Exercise 1 first (variables, DataFrames, describe()).
#
# ESTIMATED TIME: 40-55 minutes
#
# TASKS:
#   1. Filter HDB resale transactions by town, price, and date
#   2. Select and rename columns to focus your analysis
#   3. Create new derived columns with with_columns()
#   4. Sort results and combine filters with & and |
#   5. Chain operations together to build a clean analysis pipeline
#
# DATASET: Singapore HDB resale flat transactions
#   Source: Housing & Development Board (data.gov.sg)
#   Rows: ~500,000 transactions | Columns: month, town, flat_type, floor_area_sqm,
#   resale_price, and more
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

print("=" * 60)
print("  MLFP01 Exercise 2: Filtering and Transforming Data")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.height:,} rows, {hdb.width} columns)")
print(f"  You're ready to start!\n")

print("=== HDB Resale Dataset ===")
print(f"Shape: {hdb.shape}")
print(f"Columns: {hdb.columns}")
print(hdb.head(3))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Boolean filters — keep only rows that match a condition
# ══════════════════════════════════════════════════════════════════════

# A Boolean is True or False. Polars lets you write conditions using
# pl.col("column_name") and comparison operators: ==, !=, >, <, >=, <=

# Filter for a single town
# pl.col("town") creates a column reference — like pointing at the column
ang_mo_kio = hdb.filter(pl.col("town") == "ANG MO KIO")
print(f"\nAng Mo Kio transactions: {ang_mo_kio.height:,}")

# Filter by price range
# Note: use & for AND (both must be True), | for OR (either can be True)
# You MUST wrap each condition in parentheses when combining with & or |
affordable = hdb.filter(
    (pl.col("resale_price") >= 300_000) & (pl.col("resale_price") <= 500_000)
)
print(f"Transactions S$300k–500k: {affordable.height:,}")

# Filter by flat type
four_room = hdb.filter(pl.col("flat_type") == "4 ROOM")
print(f"4-room flats: {four_room.height:,}")

# Combine multiple conditions — Ang Mo Kio 4-room under S$500k
amk_4room_affordable = hdb.filter(
    (pl.col("town") == "ANG MO KIO")
    & (pl.col("flat_type") == "4 ROOM")
    & (pl.col("resale_price") <= 500_000)
)
print(f"AMK 4-room under S$500k: {amk_4room_affordable.height:,}")

# .is_in() filters for any of several values — cleaner than chaining ==
central_towns = ["BISHAN", "TOA PAYOH", "QUEENSTOWN", "BUKIT MERAH"]
central = hdb.filter(pl.col("town").is_in(central_towns))
print(f"Central towns transactions: {central.height:,}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert ang_mo_kio.height > 0, "AMK filter returned no rows — check column values"
assert affordable.height < hdb.height, "Affordable filter should reduce row count"
assert amk_4room_affordable.height <= ang_mo_kio.height, (
    "Combined filter should be a subset of the single-town filter"
)
assert central.height > 0, "Central towns filter returned no rows"
print("\n✓ Checkpoint 1 passed — Boolean filtering working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Select and rename columns
# ══════════════════════════════════════════════════════════════════════

# .select() keeps only the columns you name — everything else is dropped.
# This is important: always work with the smallest DataFrame that answers
# your question. Extra columns waste memory and clutter your output.

core_cols = hdb.select(
    "month",
    "town",
    "flat_type",
    "floor_area_sqm",
    "resale_price",
)
print(f"\nAfter select: {core_cols.columns}")

# .rename() changes column names — useful when names are awkward or long
# Pass a dict: {"old_name": "new_name"}
renamed = core_cols.rename(
    {
        "month": "sale_month",
        "floor_area_sqm": "area_sqm",
        "resale_price": "price",
    }
)
print(f"After rename: {renamed.columns}")
print(renamed.head(3))

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert core_cols.width == 5, "core_cols should have exactly 5 columns"
assert "resale_price" not in renamed.columns, "resale_price should be renamed to price"
assert "price" in renamed.columns, "renamed DataFrame should have a 'price' column"
print("\n✓ Checkpoint 2 passed — select and rename working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Derived columns with with_columns()
# ══════════════════════════════════════════════════════════════════════

# .with_columns() adds new columns (or replaces existing ones) without
# removing any other columns. This is how you engineer features.
# .alias() gives the new column a name.

hdb = hdb.with_columns(
    # Price per square metre — a normalised measure for fair comparison
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
)

# You can add multiple columns in one call — more efficient than chaining
hdb = hdb.with_columns(
    # Parse the "month" string (e.g. "2023-01") into a proper date
    # str.to_date() converts text → date; "%Y-%m" is the format pattern
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
    # Extract the year as an integer for grouping later
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
)

print(f"\n=== After adding derived columns ===")
print(hdb.select("month", "transaction_date", "year", "price_per_sqm").head(5))
# INTERPRETATION: price_per_sqm normalises for flat size — a 3-room flat in
# Bishan may cost less than a 5-room flat in Jurong, but price_per_sqm
# tells you which neighbourhood is truly more expensive per unit area.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert "price_per_sqm" in hdb.columns, "price_per_sqm column should be added"
assert "transaction_date" in hdb.columns, "transaction_date column should be added"
assert "year" in hdb.columns, "year column should be added"
sample_psm = hdb["price_per_sqm"].drop_nulls()[0]
assert sample_psm > 0, "price_per_sqm should be positive"
print("\n✓ Checkpoint 3 passed — derived columns created correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Conditional columns with pl.when().then().otherwise()
# ══════════════════════════════════════════════════════════════════════

# pl.when() is Polars' if/else for column creation.
# Think of it as: "When this condition, give this value; otherwise, that value."
# You can chain .when().then() as many times as you need.

hdb = hdb.with_columns(
    pl.when(pl.col("resale_price") < 350_000)
    .then(pl.lit("budget"))
    .when(pl.col("resale_price") < 500_000)
    .then(pl.lit("mid_range"))
    .when(pl.col("resale_price") < 700_000)
    .then(pl.lit("premium"))
    .otherwise(pl.lit("luxury"))
    .alias("price_tier")
)

# pl.lit() wraps a constant value into a Polars expression
# Without it, Polars would treat the string as a column name

# Count each tier
tier_counts = (
    hdb.group_by("price_tier")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
print(f"\n=== Price Tier Distribution ===")
print(tier_counts)
# INTERPRETATION: In Singapore's public housing market, "mid_range" flats
# (S$350k–500k) typically dominate. "Luxury" (>S$700k) represents the
# coveted "million-dollar HDB" transactions that make headlines.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert "price_tier" in hdb.columns, "price_tier column should be added"
tier_values = set(hdb["price_tier"].unique().to_list())
expected_tiers = {"budget", "mid_range", "premium", "luxury"}
assert tier_values == expected_tiers, f"Expected tiers {expected_tiers}, got {tier_values}"
print("\n✓ Checkpoint 4 passed — conditional column created with all 4 tiers\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Sort and chain operations — building a pipeline
# ══════════════════════════════════════════════════════════════════════

# .sort() orders rows. descending=True puts the largest values first.
# sort() on multiple columns: first column is the primary sort key.

# Method chaining: instead of saving each intermediate step to a variable,
# you attach the next operation with a dot. Polars is designed for this.
# Reading left-to-right (or top-to-bottom) tells the story of your analysis.

recent_premium = (
    hdb
    # Step 1: Keep only recent years
    .filter(pl.col("year") >= 2020)
    # Step 2: Keep only premium and luxury flats
    .filter(pl.col("price_tier").is_in(["premium", "luxury"]))
    # Step 3: Keep only the columns we care about
    .select(
        "transaction_date",
        "town",
        "flat_type",
        "price_per_sqm",
        "price_tier",
        "resale_price",
    )
    # Step 4: Sort by price descending to see the most expensive first
    .sort("resale_price", descending=True)
)

print(f"\n=== Recent Premium/Luxury Transactions (2020+) ===")
print(f"Count: {recent_premium.height:,}")
print(recent_premium.head(10))

# The town with the most high-value transactions
top_towns = (
    recent_premium.group_by("town")
    .agg(
        pl.len().alias("transaction_count"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
    )
    .sort("transaction_count", descending=True)
)

print(f"\n=== Towns with Most Premium/Luxury Transactions (2020+) ===")
print(top_towns.head(10))
# INTERPRETATION: Towns near MRT interchanges or mature estates (Queenstown,
# Toa Payoh, Bishan) tend to dominate the premium/luxury tier. The table
# above lets you compare both volume and price — some towns have high counts
# at moderate prices; others have fewer but more expensive transactions.

# Polars chaining reads like a sentence:
# "Take the HDB data, filter to recent years, filter to premium tier,
#  select relevant columns, sort by price" — each step is clear.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert recent_premium.height > 0, "recent_premium should have rows"
assert recent_premium.height < hdb.height, "Chained filters should reduce row count"
assert recent_premium["resale_price"][0] >= recent_premium["resale_price"][-1], (
    "DataFrame should be sorted descending by resale_price"
)
print("\n✓ Checkpoint 5 passed — chained pipeline built and sorted correctly\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 58)
print("  WHAT YOU'VE MASTERED")
print("═" * 58)
print("""
  ✓ Boolean filters: pl.col() + comparison operators (==, >, <=)
  ✓ Compound filters: & for AND, | for OR, .is_in() for sets
  ✓ Column selection: .select() to keep only what you need
  ✓ Column renaming: .rename({"old": "new"}) for clearer names
  ✓ Feature engineering: .with_columns() + .alias() for new columns
  ✓ Conditional logic: pl.when().then().otherwise() for categories
  ✓ Method chaining: building readable analysis pipelines step by step

  NEXT: In Exercise 3, you'll write Python functions and use
  group_by() + agg() to compute statistics for every district
  at once — instead of filtering one town at a time. You'll
  also learn for loops to iterate over results and build reports.
""")
