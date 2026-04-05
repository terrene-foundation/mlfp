# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT1 — Exercise 5 (Challenge): Full EDA Pipeline on Messy Taxi Data
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a complete EDA pipeline on genuinely messy data —
#   profile, clean, transform, visualise, and report. This combines
#   DataExplorer, PreprocessingPipeline, and ModelVisualizer in a
#   realistic data-science workflow.
#
# TASKS:
#   1. Load and inspect messy Singapore taxi trip data
#   2. Profile raw data with DataExplorer to identify quality issues
#   3. Clean the data based on profile alerts (GPS noise, schema drift)
#   4. Engineer features for trip analysis
#   5. Use PreprocessingPipeline to prepare for downstream modelling
#   6. Visualise key patterns with ModelVisualizer
#   7. Preview FeatureSchema (priming for Module 2)
#
# DATA QUALITY ISSUES:
#   - GPS noise (coordinates outside Singapore)
#   - Schema drift across years (column names changed)
#   - Missing pickup/dropoff fields
#   - Fare outliers (negative values, unrealistically high)
#   - Trip duration anomalies (zero-length, multi-day)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from kailash_ml import (
    DataExplorer,
    PreprocessingPipeline,
    ModelVisualizer,
)
from kailash_ml.engines.data_explorer import AlertConfig
from kailash_ml.types import FeatureSchema, FeatureField

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
taxi_raw = loader.load("ascent01", "sg_taxi_trips.parquet")

print("=== Raw Taxi Trip Data ===")
print(f"Shape: {taxi_raw.shape}")
print(f"Columns: {taxi_raw.columns}")
print(f"Dtypes:\n{taxi_raw.dtypes}")
print(taxi_raw.head(5))


# ════════════════════════════════════════════════════════════════════════
# TASK 1: Initial inspection — understand the mess
# ════════════════════════════════════════════════════════════════════════

# Basic statistics
print(f"\n=== Basic Statistics ===")
print(taxi_raw.describe())

# Null counts
print(f"\n=== Null Counts ===")
for col in taxi_raw.columns:
    nc = taxi_raw[col].null_count()
    if nc > 0:
        print(f"  {col}: {nc:,} ({nc / taxi_raw.height:.1%})")

# Value ranges for key columns
numeric_cols = [
    c
    for c, d in zip(taxi_raw.columns, taxi_raw.dtypes)
    if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
]
for col in numeric_cols:
    series = taxi_raw[col].drop_nulls()
    if series.len() > 0:
        print(f"  {col}: [{series.min()}, {series.max()}], mean={series.mean():.2f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 2: Profile raw data with DataExplorer
# ════════════════════════════════════════════════════════════════════════


async def profile_raw_data():
    """Profile the raw taxi data to identify quality issues."""

    alert_config = AlertConfig(
        high_null_pct_threshold=0.02,  # Strict: even 2% missing is flagged
        skewness_threshold=2.0,  # Flag heavy tails in fare/distance
        high_cardinality_ratio=0.8,  # Flag near-unique columns
        zero_pct_threshold=0.1,  # Flag columns with >10% zeros
        high_correlation_threshold=0.9,  # Flag redundant features
    )

    # TODO: Create a DataExplorer with the alert_config above
    # Hint: DataExplorer(alert_config=alert_config)
    explorer = ____  # Hint: DataExplorer(alert_config=alert_config)

    # Profile a sample (full dataset may be very large)
    sample_size = min(200_000, taxi_raw.height)
    taxi_sample = taxi_raw.sample(n=sample_size, seed=42)

    print(f"\n=== DataExplorer Profile (n={sample_size:,}) ===")
    # TODO: Await explorer.profile() on the taxi_sample DataFrame
    # Hint: await explorer.profile(taxi_sample)
    profile = ____  # Hint: await explorer.profile(taxi_sample)

    print(f"Rows: {profile.n_rows}, Columns: {profile.n_columns}")
    print(f"Duplicates: {profile.duplicate_count} ({profile.duplicate_pct:.1%})")

    # Alerts — these guide our cleaning strategy
    print(f"\n--- Data Quality Alerts ({len(profile.alerts)}) ---")
    cleaning_actions = []
    for alert in profile.alerts:
        print(
            f"  [{alert['severity'].upper()}] {alert['type']}: {alert.get('column', 'N/A')}"
        )
        # Map alerts to cleaning actions
        if alert["type"] == "high_nulls":
            cleaning_actions.append(f"Handle missing values in {alert.get('column')}")
        elif alert["type"] == "high_skewness":
            cleaning_actions.append(f"Investigate outliers in {alert.get('column')}")
        elif alert["type"] == "high_zeros":
            cleaning_actions.append(
                f"Check zero values in {alert.get('column')} — real or missing?"
            )

    print(f"\n--- Recommended Cleaning Actions ---")
    for i, action in enumerate(cleaning_actions, 1):
        print(f"  {i}. {action}")

    return profile, cleaning_actions


profile_raw, cleaning_actions = asyncio.run(profile_raw_data())


# ════════════════════════════════════════════════════════════════════════
# TASK 3: Clean the data based on profile alerts
# ════════════════════════════════════════════════════════════════════════

# Singapore bounding box (approximate)
SG_LAT_MIN, SG_LAT_MAX = 1.15, 1.47
SG_LNG_MIN, SG_LNG_MAX = 103.60, 104.05

taxi_clean = taxi_raw.clone()

# Step 3a: Fix GPS coordinates outside Singapore
lat_cols = [c for c in taxi_clean.columns if "lat" in c.lower()]
lng_cols = [c for c in taxi_clean.columns if "lng" in c.lower() or "lon" in c.lower()]

for lat_col in lat_cols:
    before = taxi_clean.height
    # TODO: Filter out rows where lat_col falls outside the Singapore bounding box
    # Keep rows that are null (handle separately) OR within [SG_LAT_MIN, SG_LAT_MAX]
    # Hint: taxi_clean.filter(
    #   (pl.col(lat_col).is_null()) | ((pl.col(lat_col) >= SG_LAT_MIN) & (pl.col(lat_col) <= SG_LAT_MAX))
    # )
    taxi_clean = ____  # Hint: taxi_clean.filter(...)
    removed = before - taxi_clean.height
    if removed > 0:
        print(f"GPS filter ({lat_col}): removed {removed:,} rows outside Singapore")

for lng_col in lng_cols:
    before = taxi_clean.height
    # TODO: Filter out rows where lng_col falls outside the Singapore bounding box
    # Hint: same pattern as lat_col above, using SG_LNG_MIN and SG_LNG_MAX
    taxi_clean = ____  # Hint: taxi_clean.filter(...)
    removed = before - taxi_clean.height
    if removed > 0:
        print(f"GPS filter ({lng_col}): removed {removed:,} rows outside Singapore")

# Step 3b: Fix fare outliers
if "fare" in taxi_clean.columns:
    # Remove negative fares
    before = taxi_clean.height
    # TODO: Filter to keep only rows where fare > 0
    # Hint: taxi_clean.filter(pl.col("fare") > 0)
    taxi_clean = ____  # Hint: taxi_clean.filter(pl.col("fare") > 0)
    print(f"Negative fare filter: removed {before - taxi_clean.height:,} rows")

    # Remove extreme fares (> 99.9th percentile — likely data errors)
    fare_p999 = taxi_clean["fare"].quantile(0.999)
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("fare") <= fare_p999)
    print(
        f"Extreme fare filter (>{fare_p999:.0f}): removed {before - taxi_clean.height:,} rows"
    )

# Step 3c: Fix trip duration anomalies
if "trip_duration_sec" in taxi_clean.columns:
    # Remove zero-length trips
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("trip_duration_sec") > 60)  # Min 1 minute
    print(f"Short trip filter (<60s): removed {before - taxi_clean.height:,} rows")

    # Remove unrealistically long trips (> 3 hours)
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("trip_duration_sec") <= 10_800)
    print(f"Long trip filter (>3h): removed {before - taxi_clean.height:,} rows")

# Step 3d: Drop rows with critical nulls (pickup/dropoff coordinates)
critical_cols = lat_cols + lng_cols
if critical_cols:
    before = taxi_clean.height
    taxi_clean = taxi_clean.drop_nulls(subset=critical_cols)
    print(f"Critical null filter: removed {before - taxi_clean.height:,} rows")

print(f"\n=== After Cleaning ===")
print(
    f"Rows: {taxi_raw.height:,} → {taxi_clean.height:,} ({taxi_clean.height / taxi_raw.height:.1%} retained)"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4: Engineer features for trip analysis
# ════════════════════════════════════════════════════════════════════════

# Parse datetime columns
datetime_cols = [
    c for c in taxi_clean.columns if "time" in c.lower() or "date" in c.lower()
]
for col in datetime_cols:
    if taxi_clean[col].dtype == pl.Utf8:
        taxi_clean = taxi_clean.with_columns(pl.col(col).str.to_datetime().alias(col))

# Extract temporal features from pickup time
pickup_col = next(
    (
        c
        for c in taxi_clean.columns
        if "pickup" in c.lower() and ("time" in c.lower() or "date" in c.lower())
    ),
    None,
)

if pickup_col:
    taxi_clean = taxi_clean.with_columns(
        pl.col(pickup_col).dt.hour().alias("hour_of_day"),
        pl.col(pickup_col).dt.weekday().alias("day_of_week"),  # 0=Monday
        pl.when(pl.col(pickup_col).dt.weekday() >= 5)
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("is_weekend"),
        pl.col(pickup_col).dt.month().alias("month"),
    )

    # Peak hour classification
    taxi_clean = taxi_clean.with_columns(
        pl.when((pl.col("hour_of_day") >= 7) & (pl.col("hour_of_day") <= 9))
        .then(pl.lit("morning_peak"))
        .when((pl.col("hour_of_day") >= 17) & (pl.col("hour_of_day") <= 20))
        .then(pl.lit("evening_peak"))
        .when((pl.col("hour_of_day") >= 22) | (pl.col("hour_of_day") <= 5))
        .then(pl.lit("late_night"))
        .otherwise(pl.lit("off_peak"))
        .alias("time_period")
    )

# Distance-based features (if coordinates available)
if lat_cols and lng_cols and len(lat_cols) >= 2:
    pickup_lat = lat_cols[0]
    dropoff_lat = lat_cols[1] if len(lat_cols) > 1 else lat_cols[0]
    pickup_lng = lng_cols[0]
    dropoff_lng = lng_cols[1] if len(lng_cols) > 1 else lng_cols[0]

    # TODO: Compute the Haversine distance between pickup and dropoff coordinates
    # Formula: 2 * R * arcsin(sqrt(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlng/2)))
    # where R = 6371 km and angles are in radians (multiply degrees by np.pi/180)
    # Hint: taxi_clean.with_columns(
    #   (2 * 6371 * (
    #     (((pl.col(dropoff_lat) - pl.col(pickup_lat)) * np.pi/180/2).sin().pow(2)
    #      + (pl.col(pickup_lat)*np.pi/180).cos()
    #      * (pl.col(dropoff_lat)*np.pi/180).cos()
    #      * ((pl.col(dropoff_lng) - pl.col(pickup_lng))*np.pi/180/2).sin().pow(2)
    #     ).sqrt().arcsin()
    #   )).alias("haversine_km")
    # )
    taxi_clean = taxi_clean.with_columns(
        ____  # Hint: haversine expression → .alias("haversine_km")
    )

    # Speed (km/h)
    if "trip_duration_sec" in taxi_clean.columns:
        taxi_clean = taxi_clean.with_columns(
            (pl.col("haversine_km") / (pl.col("trip_duration_sec") / 3600)).alias(
                "avg_speed_kmh"
            )
        )

        # Filter unrealistic speeds (> 120 km/h in Singapore)
        before = taxi_clean.height
        taxi_clean = taxi_clean.filter(
            (pl.col("avg_speed_kmh") <= 120) & (pl.col("avg_speed_kmh") > 0)
        )
        if before - taxi_clean.height > 0:
            print(
                f"Speed filter: removed {before - taxi_clean.height:,} rows (>120 km/h)"
            )

# Fare per km
if "fare" in taxi_clean.columns and "haversine_km" in taxi_clean.columns:
    taxi_clean = taxi_clean.with_columns(
        (pl.col("fare") / pl.col("haversine_km")).alias("fare_per_km")
    )

print(f"\n=== Engineered Features ===")
new_cols = [c for c in taxi_clean.columns if c not in taxi_raw.columns]
print(f"New columns ({len(new_cols)}): {new_cols}")
print(taxi_clean.select(new_cols).head(5))


# ════════════════════════════════════════════════════════════════════════
# TASK 5: PreprocessingPipeline for downstream modelling
# ════════════════════════════════════════════════════════════════════════

# Prepare features for a fare prediction model
feature_cols = [
    c
    for c in taxi_clean.columns
    if c
    not in (
        "fare",
        "fare_per_km",  # targets / derived from target
        *datetime_cols,  # raw datetime (use extracted features instead)
        *lat_cols,
        *lng_cols,  # raw coordinates (use haversine instead)
    )
    and taxi_clean[c].dtype
    in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Utf8, pl.Boolean, pl.Categorical)
]

# Cast string columns to categorical for PreprocessingPipeline
for col in feature_cols:
    if taxi_clean[col].dtype == pl.Utf8:
        taxi_clean = taxi_clean.with_columns(
            pl.col(col).cast(pl.Categorical).alias(col)
        )

# Sample for pipeline demo (PreprocessingPipeline is synchronous)
taxi_sample = taxi_clean.sample(n=min(50_000, taxi_clean.height), seed=42)

if "fare" in taxi_sample.columns:
    # Select only the columns we want
    pipeline_cols = [c for c in feature_cols if c in taxi_sample.columns] + ["fare"]
    taxi_for_pipeline = taxi_sample.select(pipeline_cols)

    pipeline = PreprocessingPipeline()
    # TODO: Call pipeline.setup() with the correct arguments for fare prediction
    # Hint: pipeline.setup(data=taxi_for_pipeline, target="fare", train_size=0.8,
    #   seed=42, normalize=True, categorical_encoding="onehot",
    #   imputation_strategy="median")
    result = pipeline.setup(
        data=taxi_for_pipeline,
        target=____,  # Hint: "fare"
        train_size=____,  # Hint: 0.8
        seed=42,
        normalize=____,  # Hint: True — standardise numeric features
        categorical_encoding=____,  # Hint: "onehot"
        imputation_strategy=____,  # Hint: "median"
    )

    print(f"\n=== PreprocessingPipeline Result ===")
    print(result.summary)
    print(f"Task type: {result.task_type}")
    print(f"Train shape: {result.train_data.shape}")
    print(f"Test shape: {result.test_data.shape}")
    print(f"Numeric features: {result.numeric_columns}")
    print(f"Categorical features: {result.categorical_columns}")
else:
    print("\nWARNING: 'fare' column not found — skipping PreprocessingPipeline demo")
    result = None


# ════════════════════════════════════════════════════════════════════════
# TASK 6: Visualise key patterns with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Aggregate metrics by time period for comparison
if "time_period" in taxi_clean.columns and "fare" in taxi_clean.columns:
    time_metrics = {}
    for period in ["morning_peak", "evening_peak", "off_peak", "late_night"]:
        subset = taxi_clean.filter(pl.col("time_period") == period)
        if subset.height > 0:
            metrics = {
                "avg_fare": subset["fare"].mean(),
                "avg_distance_km": (
                    subset["haversine_km"].mean()
                    if "haversine_km" in subset.columns
                    else 0.0
                ),
                "avg_speed_kmh": (
                    subset["avg_speed_kmh"].mean()
                    if "avg_speed_kmh" in subset.columns
                    else 0.0
                ),
                "trip_count": float(subset.height),
            }
            time_metrics[period] = metrics

    if time_metrics:
        fig_time = viz.metric_comparison(time_metrics)
        fig_time.update_layout(title="Trip Metrics by Time Period")
        fig_time.write_html("ex5_time_period_comparison.html")
        print("\nSaved: ex5_time_period_comparison.html")

# Hourly trip distribution
if "hour_of_day" in taxi_clean.columns:
    hourly_counts = (
        taxi_clean.group_by("hour_of_day")
        .agg(
            pl.col("hour_of_day").count().alias("trip_count"),
            (
                pl.col("fare").mean().alias("avg_fare")
                if "fare" in taxi_clean.columns
                else pl.lit(0).alias("avg_fare")
            ),
        )
        .sort("hour_of_day")
    )

    hourly_metrics = {
        "Trip Volume": hourly_counts["trip_count"].to_list(),
    }
    fig_hourly = viz.training_history(hourly_metrics, x_label="Hour of Day")
    fig_hourly.update_layout(title="Trip Volume by Hour")
    fig_hourly.write_html("ex5_hourly_distribution.html")
    print("Saved: ex5_hourly_distribution.html")


# ════════════════════════════════════════════════════════════════════════
# TASK 7: FeatureSchema preview (priming for Module 2)
# ════════════════════════════════════════════════════════════════════════

# Define a FeatureSchema describing our cleaned taxi trip features
# This previews the typed feature contracts students will use in M2
# TODO: Define the FeatureSchema for sg_taxi_trip_features
# Hint: FeatureSchema(name="sg_taxi_trip_features", features=[...], ...)
# Each feature uses FeatureField(name=..., dtype=..., nullable=..., description=...)
taxi_schema = FeatureSchema(
    name=____,  # Hint: "sg_taxi_trip_features"
    features=[
        FeatureField(
            name="trip_duration_sec",
            dtype=____,  # Hint: "float64"
            nullable=False,
            description="Trip duration in seconds (60-10800)",
        ),
        FeatureField(
            name="haversine_km",
            dtype="float64",
            nullable=False,
            description="Haversine distance between pickup and dropoff",
        ),
        FeatureField(
            name="hour_of_day",
            dtype="int64",
            nullable=False,
            description="Hour of pickup (0-23)",
        ),
        FeatureField(
            name="day_of_week",
            dtype="int64",
            nullable=False,
            description="Day of week (0=Monday, 6=Sunday)",
        ),
        FeatureField(
            name="is_weekend",
            dtype=____,  # Hint: "bool"
            nullable=False,
            description="Whether trip is on Saturday/Sunday",
        ),
        FeatureField(
            name="time_period",
            dtype="categorical",
            nullable=False,
            description="Peak classification: morning_peak, evening_peak, off_peak, late_night",
        ),
        FeatureField(
            name="avg_speed_kmh",
            dtype="float64",
            nullable=False,
            description="Average trip speed in km/h (0-120)",
        ),
        FeatureField(
            name="fare",
            dtype="float64",
            nullable=False,
            description="Trip fare in SGD (target variable)",
        ),
    ],
    entity_id_column=____,  # Hint: "trip_id"
    timestamp_column=____,  # Hint: "pickup_datetime"
    version=1,
)

print(f"\n=== FeatureSchema Preview ===")
print(f"Schema: {taxi_schema.name} (v{taxi_schema.version})")
print(f"Entity ID: {taxi_schema.entity_id_column}")
print(f"Timestamp: {taxi_schema.timestamp_column}")
print(f"Features ({len(taxi_schema.features)}):")
for f in taxi_schema.features:
    nullable_str = "nullable" if f.nullable else "required"
    print(f"  {f.name}: {f.dtype} ({nullable_str}) — {f.description}")

print("\n--- In Module 2, you'll register this schema with FeatureStore ---")
print("  await feature_store.register_features(taxi_schema)")
print("  await feature_store.store(cleaned_df, taxi_schema)")


# ════════════════════════════════════════════════════════════════════════
# Final: Profile the cleaned data to verify quality improvements
# ════════════════════════════════════════════════════════════════════════


async def profile_cleaned_data():
    """Profile cleaned data and compare with raw profile."""

    explorer = DataExplorer(
        alert_config=AlertConfig(
            high_null_pct_threshold=0.01,  # Strict post-cleaning
            skewness_threshold=2.0,
        )
    )

    sample = taxi_clean.sample(n=min(200_000, taxi_clean.height), seed=42)
    profile_clean = await explorer.profile(sample)

    print(f"\n=== Cleaned Data Profile ===")
    print(f"Alerts before cleaning: {len(profile_raw.alerts)}")
    print(f"Alerts after cleaning:  {len(profile_clean.alerts)}")

    if profile_clean.alerts:
        print("\nRemaining alerts:")
        for alert in profile_clean.alerts:
            print(
                f"  [{alert['severity'].upper()}] {alert['type']}: {alert.get('column', 'N/A')}"
            )
    else:
        print("No alerts — data is clean!")

    # Generate final HTML report
    report_html = await explorer.to_html(
        sample, title="Singapore Taxi Trips — Cleaned Data Profile"
    )
    with open("ex5_taxi_profile_report.html", "w") as f:
        f.write(report_html)
    print("Saved: ex5_taxi_profile_report.html")

    return profile_clean


profile_clean = asyncio.run(profile_cleaned_data())

print("\n✓ Exercise 5 (Challenge) complete — full EDA pipeline on messy taxi data")
print("  Pipeline: Load → Profile → Clean → Engineer → Preprocess → Visualise → Report")
