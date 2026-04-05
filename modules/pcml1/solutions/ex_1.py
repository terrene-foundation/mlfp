# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT1 — Exercise 1: Polars + Kailash First Contact
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Master Polars fundamentals on real-world data, then use
#   DataExplorer for automated profiling — your first Kailash engine call.
#
# TASKS:
#   1. Load HDB resale data and auxiliary datasets (MRT stations, schools)
#   2. Join datasets to enrich HDB transactions with proximity data
#   3. Use window functions to compute rolling district price trends
#   4. Group and aggregate to build a district-level summary
#   5. Profile the enriched dataset with DataExplorer
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl
from kailash_ml import DataExplorer, AlertConfig

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()

# HDB resale transactions — 15M+ rows, 2000–present
hdb = loader.load("ascent01", "hdb_resale.parquet")

# Auxiliary datasets for spatial enrichment
mrt_stations = loader.load("ascent_assessment", "mrt_stations.parquet")
schools = loader.load("ascent_assessment", "schools.parquet")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Inspect and understand the data
# ══════════════════════════════════════════════════════════════════════

print("=== HDB Resale Data ===")
print(f"Shape: {hdb.shape}")
print(f"Columns: {hdb.columns}")
print(hdb.head(5))

print("\n=== MRT Stations ===")
print(f"Shape: {mrt_stations.shape}")
print(mrt_stations.head(5))

print("\n=== Schools ===")
print(f"Shape: {schools.shape}")
print(schools.head(5))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Enrich HDB data with spatial joins
# ══════════════════════════════════════════════════════════════════════

# Join with nearest MRT station (pre-computed in assessment dataset)
# The MRT dataset has columns: town, nearest_mrt, distance_to_mrt_km
hdb_enriched = hdb.join(
    mrt_stations.select("town", "nearest_mrt", "distance_to_mrt_km"),
    on="town",
    how="left",
)

# Join with school count per town
school_counts = schools.group_by("town").agg(
    pl.col("school_name").count().alias("school_count")
)

hdb_enriched = hdb_enriched.join(
    school_counts,
    on="town",
    how="left",
)

# Fill missing school counts (towns with no schools in dataset)
hdb_enriched = hdb_enriched.with_columns(
    pl.col("school_count").fill_null(0)
)

print(f"\nEnriched dataset shape: {hdb_enriched.shape}")
print(f"New columns: {[c for c in hdb_enriched.columns if c not in hdb.columns]}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Window functions — rolling district price trends
# ══════════════════════════════════════════════════════════════════════

# Parse month column to date for time-series operations
hdb_enriched = hdb_enriched.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date")
)

# Compute price per square metre
hdb_enriched = hdb_enriched.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm")
)

# Monthly median price per town using window functions
# This uses Polars' `over()` for partition-based computation
monthly_town_prices = hdb_enriched.group_by(
    "town", "transaction_date"
).agg(
    pl.col("price_per_sqm").median().alias("median_price_sqm"),
    pl.col("resale_price").median().alias("median_resale_price"),
    pl.col("resale_price").count().alias("transaction_count"),
).sort("town", "transaction_date")

# 12-month rolling average price per town
monthly_town_prices = monthly_town_prices.with_columns(
    pl.col("median_price_sqm")
    .rolling_mean(window_size=12)
    .over("town")
    .alias("rolling_12m_price_sqm")
)

# Year-over-year price change (%)
monthly_town_prices = monthly_town_prices.with_columns(
    (
        (pl.col("median_price_sqm") - pl.col("median_price_sqm").shift(12).over("town"))
        / pl.col("median_price_sqm").shift(12).over("town")
        * 100
    ).alias("yoy_price_change_pct")
)

print("\n=== Monthly Town Prices (sample) ===")
print(
    monthly_town_prices
    .filter(pl.col("town") == "ANG MO KIO")
    .tail(12)
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: District-level aggregation summary
# ══════════════════════════════════════════════════════════════════════

# Summary statistics by town across all time
district_summary = hdb_enriched.group_by("town").agg(
    # Price statistics
    pl.col("resale_price").median().alias("median_price"),
    pl.col("resale_price").mean().alias("mean_price"),
    pl.col("resale_price").std().alias("std_price"),
    pl.col("resale_price").quantile(0.25).alias("q25_price"),
    pl.col("resale_price").quantile(0.75).alias("q75_price"),

    # Size statistics
    pl.col("floor_area_sqm").median().alias("median_area_sqm"),

    # Volume
    pl.col("resale_price").count().alias("total_transactions"),

    # Price per sqm
    pl.col("price_per_sqm").median().alias("median_price_per_sqm"),

    # Proximity features
    pl.col("distance_to_mrt_km").first().alias("distance_to_mrt_km"),
    pl.col("school_count").first().alias("school_count"),
).sort("median_price", descending=True)

# Add IQR and coefficient of variation
district_summary = district_summary.with_columns(
    (pl.col("q75_price") - pl.col("q25_price")).alias("iqr_price"),
    (pl.col("std_price") / pl.col("mean_price") * 100).alias("cv_price_pct"),
)

print("\n=== District Summary (Top 10 by median price) ===")
print(district_summary.head(10))


# ══════════════════════════════════════════════════════════════════════
# TASK 5: DataExplorer profiling — first Kailash engine call
# ══════════════════════════════════════════════════════════════════════

async def profile_with_data_explorer():
    """Use DataExplorer to auto-profile the district summary."""

    # Configure alert thresholds for our use case
    alert_config = AlertConfig(
        high_correlation_threshold=0.85,     # Flag correlated features
        high_null_pct_threshold=0.05,        # Flag >5% missing
        skewness_threshold=2.0,              # Flag highly skewed distributions
        high_cardinality_ratio=0.9,          # Flag near-unique columns
    )

    explorer = DataExplorer(alert_config=alert_config)

    # Profile the district summary
    print("\n=== DataExplorer Profile: District Summary ===")
    profile = await explorer.profile(district_summary)

    print(f"Rows: {profile.n_rows}, Columns: {profile.n_columns}")
    print(f"Duplicates: {profile.duplicate_count} ({profile.duplicate_pct:.1%})")
    print(f"Type summary: {profile.type_summary}")

    # Display alerts
    print(f"\n--- Data Quality Alerts ({len(profile.alerts)}) ---")
    for alert in profile.alerts:
        print(f"  [{alert['severity'].upper()}] {alert['type']}: {alert.get('column', 'N/A')} = {alert.get('value', 'N/A')}")

    # Column-level statistics
    print("\n--- Column Profiles ---")
    for col in profile.columns:
        if col.inferred_type == "numeric":
            print(
                f"  {col.name}: {col.inferred_type} | "
                f"mean={col.mean:.2f}, std={col.std:.2f}, "
                f"nulls={col.null_pct:.1%}, skew={col.skewness:.2f}"
            )
        else:
            print(
                f"  {col.name}: {col.inferred_type} | "
                f"unique={col.unique_count}, nulls={col.null_pct:.1%}"
            )

    # Correlation matrix
    if profile.correlation_matrix:
        print("\n--- Top Correlations (Pearson) ---")
        seen = set()
        correlations = []
        for col_a, row in profile.correlation_matrix.items():
            for col_b, corr in row.items():
                if col_a != col_b and (col_b, col_a) not in seen:
                    seen.add((col_a, col_b))
                    correlations.append((col_a, col_b, corr))

        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        for col_a, col_b, corr in correlations[:10]:
            print(f"  {col_a} <-> {col_b}: {corr:.3f}")

    # Also profile the full enriched dataset (sample for speed)
    print("\n\n=== DataExplorer Profile: Full HDB Data (sampled) ===")
    hdb_sample = hdb_enriched.sample(n=min(100_000, hdb_enriched.height), seed=42)
    profile_full = await explorer.profile(hdb_sample)

    print(f"Rows: {profile_full.n_rows}, Columns: {profile_full.n_columns}")
    print(f"Alerts: {len(profile_full.alerts)}")
    for alert in profile_full.alerts:
        print(f"  [{alert['severity'].upper()}] {alert['type']}: {alert.get('column', 'N/A')}")

    return profile, profile_full


# Run the async profiling
profile_district, profile_full = asyncio.run(profile_with_data_explorer())

print("\n✓ Exercise 1 complete — Polars operations + DataExplorer profiling")
