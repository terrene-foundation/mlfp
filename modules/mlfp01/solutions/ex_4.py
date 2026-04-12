# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 4: Joins and Multi-Table Data
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Write conditional logic for branching decisions (if/elif/else)
#   - Import and use external packages
#   - Join multiple DataFrames on shared keys using .join()
#   - Reason about which join type to use (left vs inner vs outer)
#   - Handle missing values that arise after a join with fill_null()
#
# PREREQUISITES: Complete Exercise 3 first (functions, group_by/agg).
#
# ESTIMATED TIME: 45-60 minutes
#
# TASKS:
#   1. Inspect HDB, MRT, and school datasets independently
#   2. Enrich HDB transactions with spatial joins (left join on town)
#   3. Build a comprehensive district-level summary with group_by/agg
#
# DATASET: Three Singapore datasets joined together:
#   - HDB resale transactions (Housing & Development Board, data.gov.sg)
#   - MRT station proximity by town (pre-computed from LTA data)
#   - School density by town (pre-computed from MOE data)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()

# Primary dataset: HDB resale transactions
hdb = loader.load("mlfp01", "hdb_resale.parquet")

# Auxiliary datasets for spatial enrichment
# These contain pre-computed proximity data keyed on town name
mrt_stations = loader.load("mlfp_assessment", "mrt_stations.parquet")
schools = loader.load("mlfp_assessment", "schools.parquet")

print("=" * 60)
print("  MLFP01 Exercise 4: Joins and Multi-Table Data")
print("=" * 60)
print(f"\n  Data loaded:")
print(f"    hdb_resale.parquet        ({hdb.height:,} rows)")
print(f"    mrt_stations.parquet      ({mrt_stations.height:,} rows)")
print(f"    schools.parquet           ({schools.height:,} rows)")
print(f"  You're ready to start!\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Inspect each dataset independently
# ══════════════════════════════════════════════════════════════════════

# Before joining, always understand each table on its own:
# - What is the grain? (one row = one what?)
# - What are the key columns used for joining?
# - Are there nulls in the join keys?

print("=== HDB Resale Data ===")
print(f"Shape: {hdb.shape}")
print(f"Columns: {hdb.columns}")
print(hdb.head(5))

print("\n=== MRT Stations ===")
print(f"Shape: {mrt_stations.shape}")
print(f"Columns: {mrt_stations.columns}")
print(mrt_stations.head(5))

print("\n=== Schools ===")
print(f"Shape: {schools.shape}")
print(f"Columns: {schools.columns}")
print(schools.head(5))

# Check join key overlap — how many HDB towns have MRT data?
hdb_towns = set(hdb["town"].unique().to_list())
mrt_towns = set(mrt_stations["town"].unique().to_list())
matched = hdb_towns & mrt_towns
unmatched = hdb_towns - mrt_towns

print(f"\nJoin key check (town):")
print(f"  HDB towns:              {len(hdb_towns)}")
print(f"  MRT towns:              {len(mrt_towns)}")
print(f"  Matched (will join):    {len(matched)}")
print(f"  Unmatched (will be NULL after left join): {len(unmatched)}")
if unmatched:
    print(f"  Unmatched towns: {sorted(unmatched)}")
# INTERPRETATION: "Unmatched" towns will get NULL values for MRT distance
# after a left join. This is not always an error — it may mean those towns
# have no nearby MRT data yet. A left join keeps them; an inner join drops them.
# Always decide consciously: "do I want to keep or drop unmatched rows?"

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert hdb.height > 0, "HDB dataset is empty"
assert mrt_stations.height > 0, "MRT stations dataset is empty"
assert schools.height > 0, "Schools dataset is empty"
assert "town" in hdb.columns, "HDB dataset should have a 'town' column"
assert "town" in mrt_stations.columns, "MRT dataset should have a 'town' column"
print("\n✓ Checkpoint 1 passed — all three datasets loaded and inspected\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Enrich HDB data with spatial joins
# ══════════════════════════════════════════════════════════════════════

# A left join keeps ALL rows from the left table (hdb) and adds
# matching columns from the right table (mrt_stations).
# Rows in hdb with no match in mrt_stations get NULL for the new columns.
#
# how="left"  → keep all HDB rows regardless of match
# on="town"   → match rows where hdb.town == mrt_stations.town
#
# .select() on the right table prevents duplicate columns and limits
# which columns are brought across.

hdb_enriched = hdb.join(
    mrt_stations.select("town", "nearest_mrt", "distance_to_mrt_km"),
    on="town",
    how="left",
)

# Aggregate school data to town level before joining
# (schools table has one row per school; we want a count per town)
school_counts = schools.group_by("town").agg(
    pl.col("school_name").count().alias("school_count")
)

hdb_enriched = hdb_enriched.join(
    school_counts,
    on="town",
    how="left",
)

# After a left join, unmatched rows have NULL school_count — fill with 0
hdb_enriched = hdb_enriched.with_columns(pl.col("school_count").fill_null(0))

# Add a price per sqm column — used in the district summary below
hdb_enriched = hdb_enriched.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm")
)

print(f"\n=== After Enrichment ===")
print(f"Shape: {hdb_enriched.shape}")
print(f"New columns: {[c for c in hdb_enriched.columns if c not in hdb.columns]}")
print(
    hdb_enriched.select(
        "town", "nearest_mrt", "distance_to_mrt_km", "school_count"
    ).head(5)
)

# Sanity check: null counts for the joined columns
for col in ("nearest_mrt", "distance_to_mrt_km", "school_count"):
    nc = hdb_enriched[col].null_count()
    pct = nc / hdb_enriched.height
    print(f"  {col} nulls: {nc:,} ({pct:.1%})")
# INTERPRETATION: After a left join, every HDB row is preserved.
# Null counts in the joined columns reveal how complete the auxiliary
# datasets are. If school_count has 0 nulls, every town was matched.
# If distance_to_mrt_km has nulls, those towns lack MRT coverage data.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
# Left join should preserve all original HDB rows
assert hdb_enriched.height == hdb.height, (
    f"Left join changed row count: {hdb.height} → {hdb_enriched.height}"
)
assert "nearest_mrt" in hdb_enriched.columns, "nearest_mrt column should be added"
assert "school_count" in hdb_enriched.columns, "school_count column should be added"
# school_count should have no nulls (we filled with 0)
assert hdb_enriched["school_count"].null_count() == 0, (
    "school_count should have no nulls after fill_null(0)"
)
print("\n✓ Checkpoint 2 passed — enrichment joins preserved all rows and filled nulls\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: District-level summary with group_by/agg
# ══════════════════════════════════════════════════════════════════════

# Now that each transaction row carries spatial context, we can produce
# a district summary that mixes price statistics with location features.
#
# Key insight: spatial columns like distance_to_mrt_km are the same for
# every transaction in a town (they were joined from a town-level table),
# so .first() is the right aggregation — we just want one copy of the value.

district_summary = (
    hdb_enriched.group_by("town")
    .agg(
        # Volume
        pl.len().alias("total_transactions"),
        # Price statistics — use both mean and median
        # Mean is sensitive to outliers; median is more robust for skewed data
        pl.col("resale_price").median().alias("median_price"),
        pl.col("resale_price").mean().alias("mean_price"),
        pl.col("resale_price").std().alias("std_price"),
        pl.col("resale_price").quantile(0.25).alias("q25_price"),
        pl.col("resale_price").quantile(0.75).alias("q75_price"),
        # Price per sqm — normalised comparison across flat sizes
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        # Flat size
        pl.col("floor_area_sqm").median().alias("median_area_sqm"),
        # Spatial features — same value for every row in a town, so use .first()
        pl.col("nearest_mrt").first().alias("nearest_mrt"),
        pl.col("distance_to_mrt_km").first().alias("distance_to_mrt_km"),
        pl.col("school_count").first().alias("school_count"),
    )
    .sort("median_price", descending=True)
)

# Derived spread metrics
district_summary = district_summary.with_columns(
    (pl.col("q75_price") - pl.col("q25_price")).alias("iqr_price"),
    (pl.col("std_price") / pl.col("mean_price") * 100).alias("cv_price_pct"),
)

print(f"\n=== District Summary (Top 10 by Median Price) ===")
print(
    district_summary.select(
        "town",
        "total_transactions",
        "median_price",
        "median_price_sqm",
        "distance_to_mrt_km",
        "school_count",
    ).head(10)
)

# Which towns are closest to an MRT station?
print(f"\n=== Towns Closest to MRT ===")
print(
    district_summary.sort("distance_to_mrt_km")
    .select("town", "nearest_mrt", "distance_to_mrt_km", "median_price")
    .head(10)
)

# Does MRT proximity correlate with price?
corr_mrt_price = district_summary.select(
    pl.corr("distance_to_mrt_km", "median_price")
).item()
corr_school_price = district_summary.select(
    pl.corr("school_count", "median_price")
).item()
print(f"\nCorrelation: MRT distance ↔ median price: {corr_mrt_price:.3f}")
print(f"Correlation: school count ↔ median price:  {corr_school_price:.3f}")
print("(Positive = more schools / closer MRT → higher price)")
# INTERPRETATION: A negative corr_mrt_price means closer MRT → higher price
# (distance is smaller, price is higher). A positive corr_school_price means
# more schools → higher price. These correlations are not causal — they reflect
# that desirable amenities cluster in the same neighbourhoods. Controlling for
# flat type and size would be needed to isolate the MRT effect.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert district_summary.height > 0, "district_summary should have rows"
# One row per town
assert district_summary.height == hdb_enriched["town"].unique().len(), (
    "district_summary should have one row per town"
)
assert "iqr_price" in district_summary.columns, "iqr_price should be computed"
assert isinstance(corr_mrt_price, float), "Pearson correlation should return a float"
print("\n✓ Checkpoint 3 passed — district summary with spatial features built correctly\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 58)
print("  WHAT YOU'VE MASTERED")
print("═" * 58)
print("""
  ✓ Dataset inspection: checking grain, keys, and null counts before joining
  ✓ Left join: .join(how="left") preserves all rows from the left table
  ✓ Join key overlap: set intersection to predict NULL rates after join
  ✓ Pre-join aggregation: aggregate the right table before joining
  ✓ fill_null(): replacing NULLs introduced by a left join
  ✓ .first() in agg(): extracting a town-level value from transaction rows
  ✓ Pearson correlation: measuring linear relationships between columns

  NEXT: In Exercise 5, you'll move into time-series analysis with
  window functions. You'll compute rolling averages and year-over-year
  price changes — without leaving the DataFrame — using rolling_mean()
  and shift() with .over() partitioning. You'll also see lazy evaluation
  (scan_csv / collect) for the first time as a performance tool.
""")
