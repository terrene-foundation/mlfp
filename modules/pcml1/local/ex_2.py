# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT1 — Exercise 2: Filtering and Transforming Data
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Master Boolean logic and method chaining in Polars — the
#   core pattern for slicing and reshaping data without writing loops.
#
# TASKS:
#   1. Filter HDB resale transactions by town, price, and date
#   2. Select and rename columns to focus your analysis
#   3. Create new derived columns with with_columns()
#   4. Sort results and combine filters with & and |
#   5. Chain operations together to build a clean analysis pipeline
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = ASCENTDataLoader()
hdb = loader.load("ascent01", "hdb_resale.parquet")

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
# TODO: Filter hdb to rows where town equals "ANG MO KIO"
ang_mo_kio = hdb.filter(pl.col("town") == ____)  # Hint: "ANG MO KIO"
print(f"\nAng Mo Kio transactions: {ang_mo_kio.height:,}")

# Filter by price range
# Note: use & for AND (both must be True), | for OR (either can be True)
# You MUST wrap each condition in parentheses when combining with & or |
# TODO: Filter hdb to rows where resale_price is between 300,000 and 500,000
affordable = hdb.filter(
    (pl.col("resale_price") >= ____)
    & (pl.col("resale_price") <= ____)
    # Hint: 300_000 and 500_000 (underscores are optional readability aids)
)
print(f"Transactions S$300k–500k: {affordable.height:,}")

# Filter by flat type
# TODO: Filter hdb to rows where flat_type equals "4 ROOM"
four_room = hdb.filter(pl.col("flat_type") == ____)  # Hint: "4 ROOM"
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
# TODO: Filter hdb to rows where town is in the central_towns list
central = hdb.filter(pl.col("town").is_in(____))  # Hint: central_towns
print(f"Central towns transactions: {central.height:,}")


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
# TODO: Rename "month" → "sale_month", "floor_area_sqm" → "area_sqm", "resale_price" → "price"
renamed = core_cols.rename(
    {
        ____: "sale_month",  # Hint: "month"
        ____: "area_sqm",  # Hint: "floor_area_sqm"
        ____: "price",  # Hint: "resale_price"
    }
)
print(f"After rename: {renamed.columns}")
print(renamed.head(3))


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Derived columns with with_columns()
# ══════════════════════════════════════════════════════════════════════

# .with_columns() adds new columns (or replaces existing ones) without
# removing any other columns. This is how you engineer features.
# .alias() gives the new column a name.

hdb = hdb.with_columns(
    # Price per square metre — a normalised measure for fair comparison
    # TODO: Divide resale_price by floor_area_sqm and name it "price_per_sqm"
    (pl.col("resale_price") / pl.col(____)).alias(____)
    # Hint: "floor_area_sqm", "price_per_sqm"
)

# You can add multiple columns in one call — more efficient than chaining
hdb = hdb.with_columns(
    # Parse the "month" string (e.g. "2023-01") into a proper date
    # str.to_date() converts text → date; "%Y-%m" is the format pattern
    # TODO: Parse "month" column to date with format "%Y-%m", alias "transaction_date"
    pl.col("month").str.to_date(____).alias(____),
    # Hint: "%Y-%m", "transaction_date"
    # Extract the year as an integer for grouping later
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
)

print(f"\n=== After adding derived columns ===")
print(hdb.select("month", "transaction_date", "year", "price_per_sqm").head(5))


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Conditional columns with pl.when().then().otherwise()
# ══════════════════════════════════════════════════════════════════════

# pl.when() is Polars' if/else for column creation.
# Think of it as: "When this condition, give this value; otherwise, that value."
# You can chain .when().then() as many times as you need.

hdb = hdb.with_columns(
    pl.when(pl.col("resale_price") < 350_000)
    .then(pl.lit("budget"))
    # TODO: Add a .when().then() for mid_range (price < 500_000)
    .when(pl.col("resale_price") < ____)  # Hint: 500_000
    .then(pl.lit(____))  # Hint: "mid_range"
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
    # TODO: Filter to rows where year >= 2020
    .filter(pl.col("year") >= ____)  # Hint: 2020
    # Step 2: Keep only premium and luxury flats
    # TODO: Filter to rows where price_tier is in ["premium", "luxury"]
    .filter(pl.col("price_tier").is_in(____))  # Hint: ["premium", "luxury"]
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
    # TODO: Sort by "resale_price" in descending order
    .sort(____, descending=____)  # Hint: "resale_price", True
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

# Polars chaining reads like a sentence:
# "Take the HDB data, filter to recent years, filter to premium tier,
#  select relevant columns, sort by price" — each step is clear.

print("\n✓ Exercise 2 complete — filtering, transforming, and method chaining")
