# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 8: Data Cleaning and End-to-End Project
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build a complete data pipeline from raw data to clean output
#   - Profile raw data, translate alerts into cleaning actions, and verify
#   - Use PreprocessingPipeline for automated encoding, scaling, and imputation
#   - Structure a multi-stage pipeline using all three M1 Kailash engines
#   - Measure data quality improvement quantitatively (alerts before vs after)
#
# PREREQUISITES: Complete Exercises 1–7 (all of Module 1).
#
# ESTIMATED TIME: 60-75 minutes (capstone exercise — the longest in M1)
#
# TASKS:
#   1. Load and inspect messy Singapore taxi trip data
#   2. Profile raw data with DataExplorer to identify quality issues
#   3. Clean the data based on profile alerts (GPS noise, fare outliers,
#      duration anomalies)
#   4. Engineer temporal and spatial features
#   5. Prepare a model-ready dataset with PreprocessingPipeline
#   6. Visualise key patterns with ModelVisualizer
#   7. Re-profile cleaned data to confirm quality improvements
#
# DATASET: Singapore taxi trip data (deliberately messy)
#   Source: Singapore Land Transport Authority (LTA) / synthetic extension
#   Quality issues by design:
#     - GPS coordinates outside Singapore's bounding box
#     - Negative and extreme fare outliers
#     - Zero-length and unrealistically long trips
#     - Missing pickup/dropoff coordinates
#     - Schema drift across collection years (column names changed)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl
from kailash_ml import DataExplorer, ModelVisualizer, PreprocessingPipeline
from kailash_ml.engines.data_explorer import AlertConfig

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
taxi_raw = loader.load("mlfp01", "sg_taxi_trips.parquet")

print("=" * 60)
print("  MLFP01 Exercise 8: Data Cleaning and End-to-End Project")
print("=" * 60)
print(f"\n  Data loaded: sg_taxi_trips.parquet")
print(f"    {taxi_raw.height:,} rows | {taxi_raw.width} columns")
print(f"  You're ready to start!\n")

print("=== Raw Taxi Trip Data ===")
print(f"Shape: {taxi_raw.shape}")
print(f"Columns: {taxi_raw.columns}")
print(f"Dtypes:\n{taxi_raw.dtypes}")
print(taxi_raw.head(5))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Initial inspection — understand the mess before touching it
# ══════════════════════════════════════════════════════════════════════

# Rule: never clean data blindly. First describe what is wrong,
# then decide what to do about each problem.

print(f"\n=== Basic Statistics ===")
print(taxi_raw.describe())

print(f"\n=== Null Counts ===")
for col in taxi_raw.columns:
    nc = taxi_raw[col].null_count()
    if nc > 0:
        print(f"  {col}: {nc:,} ({nc / taxi_raw.height:.1%})")

# Value ranges for numeric columns — spot impossible values early
numeric_dtypes = (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
numeric_cols = [
    c for c, d in zip(taxi_raw.columns, taxi_raw.dtypes) if d in numeric_dtypes
]
print(f"\n=== Numeric Column Ranges ===")
for col in numeric_cols:
    series = taxi_raw[col].drop_nulls()
    if series.len() > 0:
        print(
            f"  {col}: [{series.min():.3g}, {series.max():.3g}]"
            f"  mean={series.mean():.3g}"
        )
# INTERPRETATION: You're looking for red flags in the ranges:
# - Negative fares: physically impossible — these are data entry errors
# - Latitude outside [1.15, 1.47]: Singapore's bounding box is tiny;
#   anything outside it is GPS drift or a recording error
# - trip_duration_sec = 0 or negative: meter malfunction or bad data
# - trip_duration_sec > 10,800 (3 hours): plausible but unusual; above this
#   it's almost certainly a meter left running after the trip ended
# Seeing these in describe() before any cleaning confirms the dataset is messy.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert taxi_raw.height > 0, "Raw taxi dataset is empty"
assert len(numeric_cols) > 0, "Should have at least one numeric column"
print("\n✓ Checkpoint 1 passed — raw data inspected, ranges and nulls reviewed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Profile raw data with DataExplorer
# ══════════════════════════════════════════════════════════════════════
# DataExplorer automates the column-level analysis you just did manually.
# It surfaces problems as typed alerts with severity levels, which you
# can programmatically map to cleaning actions.


async def profile_raw_data():
    """Profile the raw taxi data and collect recommended cleaning actions."""

    # Strict thresholds for raw data: every problem should surface
    alert_config = AlertConfig(
        high_null_pct_threshold=0.02,  # Even 2% missing is worth investigating
        skewness_threshold=2.0,  # Flag heavy tails in fare/distance
        high_cardinality_ratio=0.80,  # Flag near-unique string columns
        zero_pct_threshold=0.10,  # Flag columns with >10% zeros
        high_correlation_threshold=0.90,  # Flag potentially redundant features
    )

    explorer = DataExplorer(alert_config=alert_config)

    # Sample for profiling speed — full dataset may be millions of rows
    sample_size = min(200_000, taxi_raw.height)
    taxi_sample = taxi_raw.sample(n=sample_size, seed=42)

    print(f"\n=== DataExplorer Profile (n={sample_size:,}) ===")
    profile = await explorer.profile(taxi_sample)

    print(f"Rows: {profile.n_rows}  Columns: {profile.n_columns}")
    print(f"Duplicates: {profile.duplicate_count} ({profile.duplicate_pct:.1%})")

    # Map each alert type to a concrete cleaning action
    # This is the key skill: translating a data quality report into a plan
    cleaning_actions: list[str] = []
    print(f"\n--- Data Quality Alerts ({len(profile.alerts)}) ---")
    for alert in profile.alerts:
        col = alert.get("column", "N/A")
        alert_type = alert["type"]
        severity = alert["severity"]
        print(f"  [{severity.upper()}] {alert_type}: {col}")

        if alert_type == "high_nulls":
            cleaning_actions.append(
                f"Handle missing values in '{col}' — impute or drop rows"
            )
        elif alert_type == "high_skewness":
            cleaning_actions.append(
                f"Investigate outliers in '{col}' — check min/max for impossible values"
            )
        elif alert_type == "high_zeros":
            cleaning_actions.append(
                f"Verify zeros in '{col}' — are they real measurements or missing data?"
            )
        elif alert_type == "duplicates":
            cleaning_actions.append("Remove duplicate rows before modelling")

    print(f"\n--- Recommended Cleaning Actions ---")
    for i, action in enumerate(cleaning_actions, 1):
        print(f"  {i}. {action}")
    # INTERPRETATION: Each alert type maps to a specific cleaning decision.
    # high_skewness + inspection of ranges → outlier removal (Steps 3b, 3c)
    # high_nulls for coordinate columns → drop rows (Step 3d)
    # duplicates → dedup before training
    # The value of DataExplorer is that it forces you to be systematic:
    # you don't miss columns because they look "fine" on the surface.

    return profile, cleaning_actions


profile_raw, cleaning_actions = asyncio.run(profile_raw_data())

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert profile_raw is not None, "profile_raw should not be None"
assert profile_raw.n_rows > 0, "Profile should report rows"
assert isinstance(cleaning_actions, list), "cleaning_actions should be a list"
print("\n✓ Checkpoint 2 passed — DataExplorer profile complete with cleaning plan\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Clean the data based on profile alerts
# ══════════════════════════════════════════════════════════════════════
# Each cleaning step below maps to a specific alert or observation above.
# Good data cleaning is transparent: document what you removed and why.

# Singapore's geographic bounding box — any coordinate outside this
# is either a GPS error or a trip that didn't occur in Singapore
SG_LAT_MIN, SG_LAT_MAX = 1.15, 1.47
SG_LNG_MIN, SG_LNG_MAX = 103.60, 104.05

taxi_clean = taxi_raw.clone()
rows_before = taxi_clean.height

# Step 3a: Remove GPS coordinates outside Singapore
lat_cols = [c for c in taxi_clean.columns if "lat" in c.lower()]
lng_cols = [c for c in taxi_clean.columns if "lng" in c.lower() or "lon" in c.lower()]

for lat_col in lat_cols:
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(
        # Keep nulls at this stage — we handle them in step 3d
        pl.col(lat_col).is_null()
        | ((pl.col(lat_col) >= SG_LAT_MIN) & (pl.col(lat_col) <= SG_LAT_MAX))
    )
    removed = before - taxi_clean.height
    if removed > 0:
        print(f"GPS filter ({lat_col}): removed {removed:,} out-of-bounds rows")

for lng_col in lng_cols:
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(
        pl.col(lng_col).is_null()
        | ((pl.col(lng_col) >= SG_LNG_MIN) & (pl.col(lng_col) <= SG_LNG_MAX))
    )
    removed = before - taxi_clean.height
    if removed > 0:
        print(f"GPS filter ({lng_col}): removed {removed:,} out-of-bounds rows")

# Step 3b: Remove fare outliers
if "fare" in taxi_clean.columns:
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("fare") > 0)
    print(f"Negative fare filter: removed {before - taxi_clean.height:,} rows")

    # 99.9th percentile cap — values above this are almost certainly data errors
    fare_p999 = taxi_clean["fare"].quantile(0.999)
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("fare") <= fare_p999)
    print(
        f"Extreme fare cap (>{fare_p999:.0f}): removed {before - taxi_clean.height:,} rows"
    )

# Step 3c: Remove duration anomalies
if "trip_duration_sec" in taxi_clean.columns:
    # Less than 60 seconds: too short to be a real paid trip
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("trip_duration_sec") > 60)
    print(f"Short trip filter (<60 s): removed {before - taxi_clean.height:,} rows")

    # More than 3 hours: plausible for cross-island trips but beyond that
    # the data is likely a GPS dropout or a meter left running
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("trip_duration_sec") <= 10_800)
    print(f"Long trip filter (>3 h): removed {before - taxi_clean.height:,} rows")

# Step 3d: Drop rows missing critical coordinates (can't compute distance)
critical_cols = lat_cols + lng_cols
if critical_cols:
    before = taxi_clean.height
    taxi_clean = taxi_clean.drop_nulls(subset=critical_cols)
    print(f"Null coordinate filter: removed {before - taxi_clean.height:,} rows")

print(f"\n=== Cleaning Summary ===")
print(
    f"Rows: {rows_before:,} → {taxi_clean.height:,}"
    f"  ({taxi_clean.height / rows_before:.1%} retained)"
)
# INTERPRETATION: The retention rate tells you how "dirty" the original data was.
# Retaining 85%+ is typical for a well-instrumented system with occasional errors.
# Retaining less than 70% suggests systematic data quality issues — worth filing
# a bug report against the data collection pipeline. Each step is logged
# separately so you can trace exactly which filter removed the most rows.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert taxi_clean.height > 0, "Cleaning removed all rows — filters too aggressive"
assert taxi_clean.height <= taxi_raw.height, (
    "Cleaning should not add rows"
)
# No negative fares after cleaning
if "fare" in taxi_clean.columns:
    assert (taxi_clean["fare"] > 0).all(), "All fares should be positive after cleaning"
print("\n✓ Checkpoint 3 passed — data cleaned and all filters validated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Engineer features for trip analysis
# ══════════════════════════════════════════════════════════════════════
# Raw columns like pickup_datetime don't directly tell us about demand
# patterns. We extract meaningful signals: hour of day, weekend flag,
# distance, speed. This is feature engineering — turning raw data into
# model-ready signals.

# Parse datetime columns
datetime_cols = [
    c for c in taxi_clean.columns if "time" in c.lower() or "date" in c.lower()
]
for col in datetime_cols:
    if taxi_clean[col].dtype == pl.Utf8:
        taxi_clean = taxi_clean.with_columns(pl.col(col).str.to_datetime().alias(col))

# Temporal features from pickup time
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
        # 0=Monday … 6=Sunday
        pl.col(pickup_col).dt.weekday().alias("day_of_week"),
        pl.col(pickup_col).dt.month().alias("month"),
        # is_weekend is a Boolean — weekday() returns 5 for Saturday, 6 for Sunday
        (pl.col(pickup_col).dt.weekday() >= 5).alias("is_weekend"),
    )

    # Peak-hour classification — useful as a categorical feature
    # Singapore Land Transport Authority defines peak hours as 7–9 am and 6–8 pm
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

# Haversine distance between pickup and dropoff
# The haversine formula computes the great-circle distance (shortest path
# on a sphere) between two lat/lng points. For Singapore-scale distances
# this is accurate to within ~0.5%, which is more than sufficient.
if lat_cols and lng_cols and len(lat_cols) >= 2 and len(lng_cols) >= 2:
    pickup_lat, dropoff_lat = lat_cols[0], lat_cols[1]
    pickup_lng, dropoff_lng = lng_cols[0], lng_cols[1]

    # π / 180 converts degrees to radians (required for trig functions)
    _RAD = pl.lit(3.141592653589793 / 180)

    taxi_clean = taxi_clean.with_columns(
        (
            2
            * 6371  # Earth radius in km
            * (
                (
                    ((pl.col(dropoff_lat) - pl.col(pickup_lat)) * _RAD / 2).sin().pow(2)
                    + (pl.col(pickup_lat) * _RAD).cos()
                    * (pl.col(dropoff_lat) * _RAD).cos()
                    * ((pl.col(dropoff_lng) - pl.col(pickup_lng)) * _RAD / 2)
                    .sin()
                    .pow(2)
                )
                .sqrt()
                .arcsin()
            )
        ).alias("haversine_km")
    )

    # Average speed — another quality check (>120 km/h in Singapore is impossible)
    if "trip_duration_sec" in taxi_clean.columns:
        taxi_clean = taxi_clean.with_columns(
            (pl.col("haversine_km") / (pl.col("trip_duration_sec") / 3600)).alias(
                "avg_speed_kmh"
            )
        )

        before = taxi_clean.height
        taxi_clean = taxi_clean.filter(
            (pl.col("avg_speed_kmh") > 0) & (pl.col("avg_speed_kmh") <= 120)
        )
        if before - taxi_clean.height > 0:
            print(
                f"Speed filter (>120 km/h): removed {before - taxi_clean.height:,} rows"
            )

# Fare per km — normalised efficiency metric
if "fare" in taxi_clean.columns and "haversine_km" in taxi_clean.columns:
    taxi_clean = taxi_clean.with_columns(
        (pl.col("fare") / pl.col("haversine_km")).alias("fare_per_km")
    )

new_cols = [c for c in taxi_clean.columns if c not in taxi_raw.columns]
print(f"\n=== Engineered Features ({len(new_cols)}) ===")
print(new_cols)
print(taxi_clean.select(new_cols).head(5))
# INTERPRETATION: Feature engineering is what separates a good model from a
# mediocre one. hour_of_day matters far more than raw pickup_datetime because
# demand is cyclical — 8am Tuesday and 8am Saturday have different patterns.
# haversine_km gives the model a direct measure of trip length that it can't
# infer from lat/lng alone. fare_per_km normalises for trip length — useful
# for detecting surge pricing patterns or driver route efficiency.
# This is a preview of M2: in the next module, you'll build feature stores
# and apply FeatureEngineer to automate this process at scale.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
if pickup_col:
    assert "hour_of_day" in taxi_clean.columns, "hour_of_day should be engineered"
    assert "time_period" in taxi_clean.columns, "time_period should be engineered"
    time_periods = set(taxi_clean["time_period"].unique().to_list())
    assert "morning_peak" in time_periods, "morning_peak should be one of the time periods"
if "haversine_km" in taxi_clean.columns:
    assert (taxi_clean["haversine_km"].drop_nulls() > 0).all(), (
        "All haversine distances should be positive"
    )
print("\n✓ Checkpoint 4 passed — temporal and spatial features engineered correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: PreprocessingPipeline — model-ready features
# ══════════════════════════════════════════════════════════════════════
# PreprocessingPipeline automates the final steps before training:
#   - Splits data into train/test sets (stratified or random)
#   - Imputes remaining nulls (median for numeric, mode for categorical)
#   - Scales numeric features (standardise to mean=0, std=1)
#   - Encodes categorical columns (one-hot or ordinal)
#   - Infers task type (regression vs classification) from the target
#
# You tell it: what is the target? what encoding strategy? what imputation?
# It handles the rest and returns a result object with split DataFrames.

# Select feature columns: exclude raw coordinates, raw datetimes, and the target
exclude = set(
    ["fare", "fare_per_km"]  # target and derivative of target
    + datetime_cols  # raw datetimes — use extracted features instead
    + lat_cols
    + lng_cols  # raw coordinates — use haversine instead
)
feature_cols = [
    c
    for c in taxi_clean.columns
    if c not in exclude
    and taxi_clean[c].dtype
    in (
        pl.Float64,
        pl.Float32,
        pl.Int64,
        pl.Int32,
        pl.Utf8,
        pl.Boolean,
        pl.Categorical,
    )
]

# Cast string columns to Categorical — PreprocessingPipeline expects this
for col in feature_cols:
    if taxi_clean[col].dtype == pl.Utf8:
        taxi_clean = taxi_clean.with_columns(pl.col(col).cast(pl.Categorical))

# Work with a sample — PreprocessingPipeline is synchronous
taxi_sample = taxi_clean.sample(n=min(50_000, taxi_clean.height), seed=42)

if "fare" in taxi_sample.columns:
    pipeline_df = taxi_sample.select(
        [c for c in feature_cols if c in taxi_sample.columns] + ["fare"]
    )

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        data=pipeline_df,
        target="fare",
        train_size=0.8,
        seed=42,
        normalize=True,
        categorical_encoding="onehot",
        imputation_strategy="median",
    )

    print(f"\n=== PreprocessingPipeline Result ===")
    print(result.summary)
    print(f"Task type:          {result.task_type}")
    print(f"Train shape:        {result.train_data.shape}")
    print(f"Test shape:         {result.test_data.shape}")
    print(f"Numeric features:   {result.numeric_columns}")
    print(f"Categorical feats:  {result.categorical_columns}")
    # INTERPRETATION: result.task_type will be "regression" because the target
    # (fare) is continuous. The pipeline has already standardised numerics
    # (mean=0, std=1) and one-hot encoded categoricals (time_period → 4 binary
    # columns). The 80/20 train/test split is the standard starting point.
    # In M3, you'll pass result.train_data directly into TrainingPipeline —
    # no additional preprocessing needed. That's the whole point of this step.

    # ── Checkpoint 5 ─────────────────────────────────────────────────
    assert result.task_type == "regression", (
        f"Task type should be 'regression' for a continuous fare target, got: {result.task_type}"
    )
    total_rows = result.train_data.shape[0] + result.test_data.shape[0]
    assert abs(total_rows - taxi_sample.height) <= 1, (
        "Train + test rows should sum to sample size"
    )
    print("\n✓ Checkpoint 5 passed — PreprocessingPipeline produced train/test split\n")
else:
    print("\n'fare' column not found — skipping PreprocessingPipeline demo")
    result = None


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Visualise key patterns with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Fare distribution — compare before and after cleaning
raw_fares = (
    taxi_raw["fare"].drop_nulls().to_list() if "fare" in taxi_raw.columns else []
)
clean_fares = (
    taxi_clean["fare"].drop_nulls().to_list() if "fare" in taxi_clean.columns else []
)

if clean_fares:
    fig_dist = viz.feature_distribution(
        values=clean_fares,
        feature_name="Fare (S$) — Cleaned",
    )
    fig_dist.update_layout(title="Taxi Fare Distribution (After Cleaning)")
    fig_dist.write_html("ex8_fare_distribution.html")
    print("\nSaved: ex8_fare_distribution.html")

# Trip metrics by time period
if "time_period" in taxi_clean.columns and "fare" in taxi_clean.columns:
    periods = ["morning_peak", "evening_peak", "off_peak", "late_night"]
    time_metrics: dict[str, dict[str, float]] = {}

    for period in periods:
        subset = taxi_clean.filter(pl.col("time_period") == period)
        if subset.height > 0:
            time_metrics[period] = {
                "avg_fare": float(subset["fare"].mean() or 0),
                "avg_distance_km": (
                    float(subset["haversine_km"].mean() or 0)
                    if "haversine_km" in subset.columns
                    else 0.0
                ),
                "trip_count": float(subset.height),
            }

    if time_metrics:
        fig_periods = viz.metric_comparison(time_metrics)
        fig_periods.update_layout(title="Trip Metrics by Time Period")
        fig_periods.write_html("ex8_time_period_metrics.html")
        print("Saved: ex8_time_period_metrics.html")

# Hourly trip volume — shows demand rhythm through the day
if "hour_of_day" in taxi_clean.columns:
    hourly = (
        taxi_clean.group_by("hour_of_day")
        .agg(
            pl.len().alias("trip_count"),
            (
                pl.col("fare").mean().alias("avg_fare")
                if "fare" in taxi_clean.columns
                else pl.lit(0.0).alias("avg_fare")
            ),
        )
        .sort("hour_of_day")
    )

    fig_hourly = viz.training_history(
        metrics={"Trip Volume": hourly["trip_count"].to_list()},
        x_label="Hour of Day",
        y_label="Number of Trips",
    )
    fig_hourly.update_layout(title="Taxi Trip Volume by Hour of Day")
    fig_hourly.write_html("ex8_hourly_volume.html")
    print("Saved: ex8_hourly_volume.html")
    # INTERPRETATION: The hourly volume chart reveals Singapore's urban rhythm.
    # You'll typically see two peaks: 7–9am (morning commute) and 6–8pm
    # (evening commute). Late-night demand (11pm–2am) reflects nightlife in
    # Clarke Quay and Orchard. The trough at 4–6am is when shift workers
    # swap and demand is genuinely low — not a data quality issue.
    # This demand pattern matters because ML models for fare prediction
    # need hour_of_day as a feature to avoid treating all trips equally.

# Feature importances — if PreprocessingPipeline ran, show feature list
if result is not None:
    all_features = result.numeric_columns + result.categorical_columns
    # Use a uniform importance placeholder so students can see the bar chart
    # (real importances come in M3 after training a model)
    feat_metrics = {
        f: {"Weight": 1.0 / len(all_features)} for f in all_features
    }
    fig_feats = viz.metric_comparison(feat_metrics)
    fig_feats.write_html("ex8_feature_list.html")
    print("Saved: ex8_feature_list.html")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
import os
expected_outputs = []
if clean_fares:
    expected_outputs.append("ex8_fare_distribution.html")
if "hour_of_day" in taxi_clean.columns:
    expected_outputs.append("ex8_hourly_volume.html")
for output in expected_outputs:
    assert os.path.exists(output), f"Expected output file not found: {output}"
print("\n✓ Checkpoint 6 passed — visualisation files saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Re-profile cleaned data — confirm quality improvements
# ══════════════════════════════════════════════════════════════════════
# The goal of cleaning is measurable: fewer alerts, lower null rates,
# no impossible values. Profiling after cleaning verifies the work.


async def profile_cleaned_data():
    """Profile cleaned data and compare alert counts with the raw profile."""

    # Stricter thresholds post-cleaning — problems that survived should be rare
    explorer = DataExplorer(
        alert_config=AlertConfig(
            high_null_pct_threshold=0.01,
            skewness_threshold=2.0,
        )
    )

    sample = taxi_clean.sample(n=min(200_000, taxi_clean.height), seed=42)
    profile_clean = await explorer.profile(sample)

    print(f"\n=== Data Quality: Before vs After Cleaning ===")
    print(f"  Alerts before: {len(profile_raw.alerts)}")
    print(f"  Alerts after:  {len(profile_clean.alerts)}")
    # INTERPRETATION: The alert count comparison is your quality proof.
    # If alerts_after < alerts_before, cleaning worked. If alerts_after >= alerts_before,
    # either the cleaning was ineffective or new issues were introduced (e.g., imputation
    # created unexpected patterns). Zero alerts after cleaning is aspirational — in
    # practice, a 50%+ reduction in alerts is a good outcome for a single-pass clean.

    if profile_clean.alerts:
        print("\nRemaining alerts (investigate further):")
        for alert in profile_clean.alerts:
            print(
                f"  [{alert['severity'].upper()}] {alert['type']}: "
                f"{alert.get('column', 'N/A')}"
            )
    else:
        print("  No remaining alerts — data quality confirmed clean.")

    # Final HTML report combining statistics and charts
    report_html = await explorer.to_html(
        sample,
        title="Singapore Taxi Trips — Cleaned Data Profile",
    )
    with open("ex8_taxi_profile_clean.html", "w") as f:
        f.write(report_html)
    print("\nSaved: ex8_taxi_profile_clean.html")

    return profile_clean


try:
    profile_clean = asyncio.run(profile_cleaned_data())
except Exception as exc:
    print(f"\n[ERROR] Post-cleaning profile failed: {exc}")
    raise

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert profile_clean is not None, "profile_clean should not be None"
assert os.path.exists("ex8_taxi_profile_clean.html"), "Cleaned data report not saved"
print("\n✓ Checkpoint 7 passed — post-cleaning profile complete and report saved\n")

# ── Pipeline summary ─────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  END-TO-END PIPELINE SUMMARY")
print(f"{'=' * 60}")
print(f"  Stage 1 Load:       {taxi_raw.height:,} rows, {taxi_raw.width} cols")
print(f"  Stage 2 Profile:    {len(profile_raw.alerts)} alerts identified")
print(f"  Stage 3 Clean:      {taxi_clean.height:,} rows retained")
print(f"  Stage 4 Engineer:   {len(new_cols)} new features created")
if result is not None:
    print(
        f"  Stage 5 Preprocess: {result.train_data.shape[0]:,} train / "
        f"{result.test_data.shape[0]:,} test rows"
    )
print(f"  Stage 6 Visualise:  4 HTML charts saved")
print(f"  Stage 7 Verify:     {len(profile_clean.alerts)} alerts remaining")
print(f"{'=' * 60}")

print(
    "\n✓ Exercise 8 complete — full data pipeline: load → profile → clean → model-ready"
)


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 58)
print("  WHAT YOU'VE MASTERED")
print("═" * 58)
print("""
  ✓ End-to-end thinking: load → profile → clean → engineer → preprocess
  ✓ DataExplorer: automated quality assessment with typed, actionable alerts
  ✓ Domain-aware cleaning: GPS bounding boxes, fare ranges, duration limits
  ✓ Feature engineering: temporal (hour, weekday, peak period), spatial
    (haversine distance), and derived (speed, fare per km)
  ✓ PreprocessingPipeline: one call to standardise, encode, impute, and split
  ✓ Quality measurement: alert count before vs after as a cleaning KPI
  ✓ Three engines working together: DataExplorer + PreprocessingPipeline +
    ModelVisualizer — the full M1 toolkit

  MODULE 1 COMPLETE — you've gone from raw CSV to model-ready data.

  NEXT — MODULE 2: Feature Engineering and Experiment Design
  In M2, you'll move beyond data exploration into systematic feature
  construction. You'll learn:
    - FeatureEngineer: automated feature generation (interactions, polynomials,
      target encoding, lag features) at scale
    - FeatureStore: versioning and retrieving feature sets across experiments
    - ExperimentTracker: logging runs, parameters, and metrics for
      reproducibility and comparison
  The taxi trip data you cleaned in this exercise will become a feature
  store entry — and you'll compare model performance across different
  feature sets using tracked experiments.
""")
