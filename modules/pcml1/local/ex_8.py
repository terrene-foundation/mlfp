# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT1 — Exercise 8: Data Cleaning and End-to-End Project
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a complete data pipeline from raw messy data to
#   model-ready features using DataExplorer, PreprocessingPipeline,
#   and ModelVisualizer — the three M1 engines working in concert.
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
# DATA QUALITY ISSUES (by design):
#   - GPS coordinates outside Singapore's bounding box
#   - Negative and extreme fare outliers
#   - Zero-length and unrealistically long trips
#   - Missing pickup/dropoff coordinates
#   - Schema drift across collection years (column names changed)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl
from kailash_ml import DataExplorer, ModelVisualizer, PreprocessingPipeline
from kailash_ml.engines.data_explorer import AlertConfig

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = ASCENTDataLoader()
taxi_raw = loader.load("ascent01", "sg_taxi_trips.csv")

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


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Profile raw data with DataExplorer
# ══════════════════════════════════════════════════════════════════════
# DataExplorer automates the column-level analysis you just did manually.
# It surfaces problems as typed alerts with severity levels, which you
# can programmatically map to cleaning actions.


async def profile_raw_data():
    """Profile the raw taxi data and collect recommended cleaning actions."""

    # TODO: create an AlertConfig with strict thresholds for raw data profiling
    alert_config = AlertConfig(
        high_null_pct_threshold=____,  # Hint: 0.02 (even 2% missing is worth investigating)
        skewness_threshold=____,  # Hint: 2.0 (flag heavy tails in fare/distance)
        high_cardinality_ratio=____,  # Hint: 0.80
        zero_pct_threshold=____,  # Hint: 0.10
        high_correlation_threshold=____,  # Hint: 0.90
    )

    explorer = DataExplorer(alert_config=alert_config)

    # Sample for profiling speed — full dataset may be millions of rows
    sample_size = min(200_000, taxi_raw.height)
    taxi_sample = taxi_raw.sample(n=sample_size, seed=42)

    print(f"\n=== DataExplorer Profile (n={sample_size:,}) ===")
    # TODO: profile the taxi sample
    profile = await explorer.profile(____)  # Hint: pass taxi_sample

    print(f"Rows: {profile.n_rows}  Columns: {profile.n_columns}")
    print(f"Duplicates: {profile.duplicate_count} ({profile.duplicate_pct:.1%})")

    # Map each alert type to a concrete cleaning action
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

    return profile, cleaning_actions


profile_raw, cleaning_actions = asyncio.run(profile_raw_data())


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
    # TODO: filter to keep rows where lat_col is null OR within SG bounds
    taxi_clean = taxi_clean.filter(
        pl.col(lat_col).is_null()
        | (
            (pl.col(lat_col) >= ____) & (pl.col(lat_col) <= ____)
        )  # Hint: SG_LAT_MIN, SG_LAT_MAX
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
    # TODO: filter out non-positive fares
    taxi_clean = taxi_clean.filter(pl.col("fare") > ____)  # Hint: 0
    print(f"Negative fare filter: removed {before - taxi_clean.height:,} rows")

    # 99.9th percentile cap
    fare_p999 = taxi_clean["fare"].quantile(0.999)
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("fare") <= fare_p999)
    print(
        f"Extreme fare cap (>{fare_p999:.0f}): removed {before - taxi_clean.height:,} rows"
    )

# Step 3c: Remove duration anomalies
if "trip_duration_sec" in taxi_clean.columns:
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("trip_duration_sec") > 60)
    print(f"Short trip filter (<60 s): removed {before - taxi_clean.height:,} rows")

    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("trip_duration_sec") <= 10_800)
    print(f"Long trip filter (>3 h): removed {before - taxi_clean.height:,} rows")

# Step 3d: Drop rows missing critical coordinates
critical_cols = lat_cols + lng_cols
if critical_cols:
    before = taxi_clean.height
    # TODO: drop rows with nulls in critical coordinate columns
    taxi_clean = taxi_clean.drop_nulls(subset=____)  # Hint: critical_cols
    print(f"Null coordinate filter: removed {before - taxi_clean.height:,} rows")

print(f"\n=== Cleaning Summary ===")
print(
    f"Rows: {rows_before:,} → {taxi_clean.height:,}"
    f"  ({taxi_clean.height / rows_before:.1%} retained)"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Engineer features for trip analysis
# ══════════════════════════════════════════════════════════════════════
# Raw columns like pickup_datetime don't directly tell us about demand
# patterns. We extract meaningful signals: hour of day, weekend flag,
# distance, speed.

# Parse datetime columns
datetime_cols = [
    c for c in taxi_clean.columns if "time" in c.lower() or "date" in c.lower()
]
for col in datetime_cols:
    if taxi_clean[col].dtype == pl.Utf8:
        taxi_clean = taxi_clean.with_columns(pl.col(col).str.to_datetime().alias(col))

pickup_col = next(
    (
        c
        for c in taxi_clean.columns
        if "pickup" in c.lower() and ("time" in c.lower() or "date" in c.lower())
    ),
    None,
)

if pickup_col:
    # TODO: extract hour_of_day, day_of_week, month, is_weekend from pickup_col
    taxi_clean = taxi_clean.with_columns(
        pl.col(pickup_col).dt.hour().alias("hour_of_day"),
        pl.col(pickup_col).dt.weekday().alias(____),  # Hint: "day_of_week"
        pl.col(pickup_col).dt.month().alias("month"),
        (pl.col(pickup_col).dt.weekday() >= ____).alias(
            "is_weekend"
        ),  # Hint: 5 (Saturday/Sunday)
    )

    # Peak-hour classification
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
if lat_cols and lng_cols and len(lat_cols) >= 2 and len(lng_cols) >= 2:
    pickup_lat, dropoff_lat = lat_cols[0], lat_cols[1]
    pickup_lng, dropoff_lng = lng_cols[0], lng_cols[1]

    _RAD = pl.lit(3.141592653589793 / 180)

    taxi_clean = taxi_clean.with_columns(
        (
            2
            * 6371
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

if "fare" in taxi_clean.columns and "haversine_km" in taxi_clean.columns:
    taxi_clean = taxi_clean.with_columns(
        (pl.col("fare") / pl.col("haversine_km")).alias("fare_per_km")
    )

new_cols = [c for c in taxi_clean.columns if c not in taxi_raw.columns]
print(f"\n=== Engineered Features ({len(new_cols)}) ===")
print(new_cols)
print(taxi_clean.select(new_cols).head(5))


# ══════════════════════════════════════════════════════════════════════
# TASK 5: PreprocessingPipeline — model-ready features
# ══════════════════════════════════════════════════════════════════════

exclude = set(["fare", "fare_per_km"] + datetime_cols + lat_cols + lng_cols)
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

for col in feature_cols:
    if taxi_clean[col].dtype == pl.Utf8:
        taxi_clean = taxi_clean.with_columns(pl.col(col).cast(pl.Categorical))

taxi_sample = taxi_clean.sample(n=min(50_000, taxi_clean.height), seed=42)

if "fare" in taxi_sample.columns:
    pipeline_df = taxi_sample.select(
        [c for c in feature_cols if c in taxi_sample.columns] + ["fare"]
    )

    pipeline = PreprocessingPipeline()
    # TODO: configure the pipeline for fare prediction
    result = pipeline.setup(
        data=pipeline_df,
        target=____,  # Hint: "fare"
        train_size=____,  # Hint: 0.8
        seed=42,
        normalize=____,  # Hint: True (required for linear models)
        categorical_encoding=____,  # Hint: "onehot"
        imputation_strategy=____,  # Hint: "median"
    )

    print(f"\n=== PreprocessingPipeline Result ===")
    print(result.summary)
    print(f"Task type:          {result.task_type}")
    print(f"Train shape:        {result.train_data.shape}")
    print(f"Test shape:         {result.test_data.shape}")
    print(f"Numeric features:   {result.numeric_columns}")
    print(f"Categorical feats:  {result.categorical_columns}")
else:
    print("\n'fare' column not found — skipping PreprocessingPipeline demo")
    result = None


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Visualise key patterns with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

raw_fares = (
    taxi_raw["fare"].drop_nulls().to_list() if "fare" in taxi_raw.columns else []
)
clean_fares = (
    taxi_clean["fare"].drop_nulls().to_list() if "fare" in taxi_clean.columns else []
)

if clean_fares:
    # TODO: create a feature distribution chart for cleaned fares
    fig_dist = viz.feature_distribution(
        values=____,  # Hint: clean_fares
        feature_name=____,  # Hint: "Fare (S$) — Cleaned"
    )
    fig_dist.update_layout(title="Taxi Fare Distribution (After Cleaning)")
    fig_dist.write_html("ex8_fare_distribution.html")
    print("\nSaved: ex8_fare_distribution.html")

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
        history={"Trip Volume": hourly["trip_count"].to_list()},
        x_label="Hour of Day",
        y_label="Number of Trips",
    )
    fig_hourly.update_layout(title="Taxi Trip Volume by Hour of Day")
    fig_hourly.write_html("ex8_hourly_volume.html")
    print("Saved: ex8_hourly_volume.html")

if result is not None:
    all_features = result.numeric_columns + result.categorical_columns
    uniform = {f: 1.0 / len(all_features) for f in all_features}
    fig_feats = viz.feature_importance(
        importance_dict=uniform,
        title="Features Prepared by PreprocessingPipeline",
    )
    fig_feats.update_layout(xaxis_title="(Uniform — real importances computed in M3)")
    fig_feats.write_html("ex8_feature_list.html")
    print("Saved: ex8_feature_list.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Re-profile cleaned data — confirm quality improvements
# ══════════════════════════════════════════════════════════════════════


async def profile_cleaned_data():
    """Profile cleaned data and compare alert counts with the raw profile."""

    # Stricter thresholds post-cleaning
    explorer = DataExplorer(
        alert_config=AlertConfig(
            high_null_pct_threshold=____,  # Hint: 0.01
            skewness_threshold=____,  # Hint: 2.0
        )
    )

    sample = taxi_clean.sample(n=min(200_000, taxi_clean.height), seed=42)
    # TODO: profile the cleaned sample
    profile_clean = await explorer.profile(____)  # Hint: pass sample

    print(f"\n=== Data Quality: Before vs After Cleaning ===")
    print(f"  Alerts before: {len(profile_raw.alerts)}")
    print(f"  Alerts after:  {len(profile_clean.alerts)}")

    if profile_clean.alerts:
        print("\nRemaining alerts (investigate further):")
        for alert in profile_clean.alerts:
            print(
                f"  [{alert['severity'].upper()}] {alert['type']}: "
                f"{alert.get('column', 'N/A')}"
            )
    else:
        print("  No remaining alerts — data quality confirmed clean.")

    # TODO: generate an HTML report for the cleaned data
    report_html = await explorer.to_html(
        ____,  # Hint: pass sample
        title=____,  # Hint: "Singapore Taxi Trips — Cleaned Data Profile"
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
