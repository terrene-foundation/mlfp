# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 8: Data Pipelines and End-to-End Project
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build a complete data pipeline from raw data to clean output
#   - Profile raw data, translate alerts into cleaning actions, and verify
#   - Use PreprocessingPipeline for automated encoding, scaling, imputation
#   - Structure a multi-stage pipeline using all three M1 Kailash engines
#   - Measure data quality improvement quantitatively (original vs cleaned)
#
# PREREQUISITES: Complete Exercises 1-7 (all of Module 1).
#
# ESTIMATED TIME: ~150-180 min (capstone exercise — the longest in M1)
#
# TASKS:
#   1.  Load and inspect messy Singapore taxi trip data
#   2.  Manual quality analysis — ranges, nulls, impossible values
#   3.  Profile raw data with DataExplorer
#   4.  Translate alerts into a cleaning action plan
#   5.  Clean the data — GPS, fares, passengers, dates, payment labels, IDs
#   6.  Engineer temporal features (hour, weekday, peak period, duration)
#   7.  Engineer spatial features (distance to the CBD, speed)
#   8.  PreprocessingPipeline — model-ready features
#   9.  Visualise key patterns with ModelVisualizer
#   10. Compare original vs cleaned data and generate a quality report
#
# DATASET: Singapore taxi trip log (sg_taxi_trips.parquet, 50,000 rows)
#   A synthetic trip log generated for this course: the zone names and
#   coordinates are realistic Singapore places, but the trips are not
#   real records. It simulates three merged dispatch systems and is
#   deliberately dirty:
#     - Pickup latitude/longitude swapped in some rows
#     - Negative fares
#     - Passenger counts of 0 or -1
#     - Trips dated in the future
#     - 15 spellings of 4 payment methods ("cash", "CASH", "Cash Payment", ...)
#     - The same trip_id used for two different trips
#     - Missing pickup/dropoff zones and tips
#     - Distances that imply impossible speeds
#   Timestamps are stored as strings ("2024-04-04 03:37:56").
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import os
from datetime import datetime

import polars as pl
from kailash_ml import AlertConfig, ModelVisualizer, PreprocessingPipeline

from shared import MLFPDataLoader, run_compare, run_profile, run_report


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
taxi_raw = loader.load("mlfp01", "sg_taxi_trips.parquet")

print("=" * 60)
print("  MLFP01 Exercise 8: Data Pipelines and End-to-End Project")
print("=" * 60)
print(f"\n  Data loaded: sg_taxi_trips.parquet")
print(f"    {taxi_raw.height:,} rows | {taxi_raw.width} columns")
print(f"  You're ready to start!\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load and inspect — understand the mess before touching it
# ══════════════════════════════════════════════════════════════════════
# Rule: never clean data blindly. First describe what is wrong, then
# decide what to do about each problem.

print("=== Raw Taxi Trip Data ===")
print(f"Shape: {taxi_raw.shape}")
print(f"Columns: {taxi_raw.columns}")
print(f"\nData types:")
for col, dtype in zip(taxi_raw.columns, taxi_raw.dtypes):
    print(f"  {col:>30}: {dtype}")

print(f"\nFirst 5 rows:")
print(taxi_raw.head(5))

# Basic statistics
print(f"\n=== describe() ===")
print(taxi_raw.describe())
# INTERPRETATION: look at the min/max rows before anything else.
# pickup_latitude has a max above 100 and pickup_longitude a min near 1 —
# Singapore sits at roughly latitude 1.3, longitude 103.8, so some rows
# have the two coordinates swapped. passengers has a min below 1, and
# fare_sgd a negative min. The datetime columns are strings (dtype str),
# so their min/max are alphabetical — they must be parsed before any
# date arithmetic.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert taxi_raw.height > 0, "Raw taxi dataset is empty"
assert taxi_raw.width >= 3, "Should have at least 3 columns"
print("\n✓ Checkpoint 1 passed — raw data loaded and inspected\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Manual quality analysis — ranges, nulls, impossible values
# ══════════════════════════════════════════════════════════════════════

# --- 2a: Parse the timestamp strings into real datetimes ---
# Every later check on dates and durations needs Datetime, not str.
# TODO: Parse both timestamp columns with .str.to_datetime(format)
taxi_raw = taxi_raw.with_columns(
    pl.col("pickup_datetime").str.to_datetime(____),  # Hint: "%Y-%m-%d %H:%M:%S"
    pl.col("dropoff_datetime").str.to_datetime(____),  # Hint: same format
)
print("=== Parsed timestamps ===")
print(
    f"  pickup_datetime: {taxi_raw['pickup_datetime'].dtype} "
    f"({taxi_raw['pickup_datetime'].min()} -> {taxi_raw['pickup_datetime'].max()})"
)

# --- 2b: Null counts ---
print("\n=== Null Analysis ===")
null_cols = []
for col in taxi_raw.columns:
    nc = taxi_raw[col].null_count()
    if nc > 0:
        pct = nc / taxi_raw.height
        null_cols.append({"column": col, "nulls": nc, "pct": pct})
        print(f"  {col:>30}: {nc:>8,} nulls ({pct:>6.1%})")

if not null_cols:
    print("  No null values found!")

# --- 2c: Numeric range check ---
numeric_dtypes = (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
numeric_cols = [
    c for c, d in zip(taxi_raw.columns, taxi_raw.dtypes) if d in numeric_dtypes
]

print(f"\n=== Numeric Column Ranges ===")
print(f"  {'Column':>30} {'Min':>12} {'Max':>12} {'Mean':>12} {'Nulls':>8}")
print(f"  {'─' * 76}")
for col in numeric_cols:
    series = taxi_raw[col].drop_nulls()
    if series.len() > 0:
        print(
            f"  {col:>30} {series.min():>12.3g} {series.max():>12.3g} "
            f"{series.mean():>12.3g} {taxi_raw[col].null_count():>8,}"
        )

# --- 2d: Domain rules — count each kind of impossible value ---
# Singapore's bounding box (degrees)
SG_LAT_MIN, SG_LAT_MAX = 1.15, 1.47
SG_LNG_MIN, SG_LNG_MAX = 103.60, 104.05
# The log was extracted at the end of 2024, so no pickup can be later.
DATA_EXTRACT_DATE = datetime(2025, 1, 1)

swapped_gps = taxi_raw.filter(
    pl.col("pickup_latitude").is_between(SG_LNG_MIN, SG_LNG_MAX)
    & pl.col("pickup_longitude").is_between(SG_LAT_MIN, SG_LAT_MAX)
).height
# TODO: Count each kind of impossible value with a filter
neg_fares = taxi_raw.filter(pl.col("fare_sgd") <= ____).height  # Hint: 0
bad_passengers = taxi_raw.filter(pl.col("passengers") < ____).height  # Hint: 1
future_trips = taxi_raw.filter(
    pl.col("pickup_datetime") >= ____  # Hint: DATA_EXTRACT_DATE
).height
payment_spellings = taxi_raw["payment_type"].n_unique()
# Hint: pl.col("trip_id").is_duplicated() is True for every row whose ID repeats
colliding_ids = taxi_raw.filter(pl.col("trip_id").____()).height
full_duplicates = taxi_raw.height - taxi_raw.unique().height

quality_issues = []
if swapped_gps > 0:
    quality_issues.append(f"GPS: {swapped_gps:,} rows with latitude/longitude swapped")
if neg_fares > 0:
    quality_issues.append(f"Fare: {neg_fares:,} zero or negative fares")
if bad_passengers > 0:
    quality_issues.append(f"Passengers: {bad_passengers:,} rows with fewer than 1")
if future_trips > 0:
    quality_issues.append(
        f"Dates: {future_trips:,} pickups on/after {DATA_EXTRACT_DATE.date()}"
    )
if payment_spellings > 4:
    quality_issues.append(
        f"Payment: {payment_spellings} spellings for 4 real payment methods"
    )
if colliding_ids > 0:
    quality_issues.append(
        f"IDs: {colliding_ids:,} rows share a trip_id with another row"
    )

print(f"\n=== Identified Quality Issues ({len(quality_issues)}) ===")
for i, issue in enumerate(quality_issues, 1):
    print(f"  {i}. {issue}")
print(f"\n  Exact duplicate rows (every column equal): {full_duplicates:,}")
print(f"  Payment spellings: {sorted(taxi_raw['payment_type'].unique().to_list())}")
# INTERPRETATION: compare the "IDs" line with the exact-duplicate count.
# If exact duplicates are 0 but trip_ids collide, the colliding rows are
# NOT copies of one trip — they are different trips that were given the
# same ID when the dispatch systems were merged. df.unique() would never
# find them; only a key check on trip_id does.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert taxi_raw["pickup_datetime"].dtype == pl.Datetime, "Parse pickup_datetime"
assert len(numeric_cols) > 0, "Should have numeric columns"
assert len(quality_issues) > 0, "The raw log should show quality issues"
print("\n✓ Checkpoint 2 passed — manual quality analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Profile raw data with DataExplorer
# ══════════════════════════════════════════════════════════════════════
# run_profile() (from shared) runs DataExplorer.profile() for you and
# returns the DataProfile. Each alert is a dict with keys "type",
# "column" (or "columns" for a correlated pair), "value" and "severity".

# TODO: Create an AlertConfig with these thresholds for taxi data:
#   high_null_pct_threshold    = 0.02
#   skewness_threshold         = 2.0
#   high_cardinality_ratio     = 0.80
#   zero_pct_threshold         = 0.10
#   high_correlation_threshold = 0.90
alert_config = AlertConfig(
    high_null_pct_threshold=____,  # Hint: 0.02
    skewness_threshold=____,  # Hint: 2.0
    high_cardinality_ratio=____,  # Hint: 0.80
    zero_pct_threshold=____,  # Hint: 0.10
    high_correlation_threshold=____,  # Hint: 0.90
)

# TODO: Profile the raw data with run_profile(df, alert_config)
profile_raw = run_profile(____, ____)  # Hint: taxi_raw, alert_config

print(f"=== DataExplorer Profile (raw) ===")
print(f"Rows: {profile_raw.n_rows}  Columns: {profile_raw.n_columns}")
print(f"Duplicates: {profile_raw.duplicate_count} ({profile_raw.duplicate_pct:.1%})")


def alert_target(alert: dict) -> str:
    """Return the column (or column pair) an alert refers to."""
    target = alert.get("column", alert.get("columns", "(whole table)"))
    if isinstance(target, (list, tuple)):
        return " & ".join(str(t) for t in target)
    return str(target)


# Categorise alerts
alert_categories: dict[str, list] = {}
for alert in profile_raw.alerts:
    alert_categories.setdefault(alert["type"], []).append(alert)

print(f"\n--- Alert Summary ({len(profile_raw.alerts)} total) ---")
for alert_type, alerts in sorted(alert_categories.items()):
    print(f"  {alert_type}: {len(alerts)} alerts")
    for alert in alerts[:3]:
        print(f"    [{alert['severity'].upper()}] {alert_target(alert)}")
# INTERPRETATION: the profiler sees statistical symptoms (nulls, skew,
# cardinality) — it cannot know that a fare must be positive or that a
# trip cannot happen in 2027. Task 2's domain rules and Task 3's alerts
# are complementary: you need both.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert profile_raw is not None
assert profile_raw.n_rows == taxi_raw.height
print("\n✓ Checkpoint 3 passed — DataExplorer profile complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Translate alerts into a cleaning action plan
# ══════════════════════════════════════════════════════════════════════

cleaning_plan: list[dict] = []

for alert in profile_raw.alerts:
    col = alert_target(alert)
    alert_type = alert["type"]

    if alert_type == "high_nulls":
        cleaning_plan.append(
            {
                "action": f"Handle nulls in '{col}'",
                "method": "Fill with an explicit value if the null has a meaning, else drop",
                "priority": "high",
            }
        )
    elif alert_type == "high_skewness":
        cleaning_plan.append(
            {
                "action": f"Investigate outliers in '{col}'",
                "method": "Check min/max, apply domain filters, consider log transform",
                "priority": "high",
            }
        )
    elif alert_type == "high_zeros":
        cleaning_plan.append(
            {
                "action": f"Verify zeros in '{col}'",
                "method": "Determine if zeros are real measurements or missing data",
                "priority": "medium",
            }
        )
    elif alert_type == "high_cardinality":
        cleaning_plan.append(
            {
                "action": f"Check near-unique column '{col}'",
                "method": "An ID must be a unique key, never a model input; a "
                "continuous value (time, coordinate) is fine as-is",
                "priority": "medium",
            }
        )
    elif alert_type == "duplicates":
        cleaning_plan.append(
            {
                "action": "Remove duplicate rows",
                "method": "df.unique() before modelling",
                "priority": "medium",
            }
        )
    elif alert_type == "high_correlation":
        cleaning_plan.append(
            {
                "action": f"Review collinearity: {col}",
                "method": "Consider dropping one of the highly correlated features",
                "priority": "low",
            }
        )

# Domain-rule findings from Task 2 go into the same plan
for issue in quality_issues:
    cleaning_plan.append(
        {"action": issue, "method": "Domain rule (see Task 5)", "priority": "high"}
    )

print(f"=== Cleaning Action Plan ({len(cleaning_plan)} actions) ===")
for i, plan in enumerate(cleaning_plan, 1):
    print(f"  {i}. [{plan['priority'].upper():>6}] {plan['action']}")
    print(f"             Method: {plan['method']}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert isinstance(cleaning_plan, list)
assert len(cleaning_plan) >= len(quality_issues)
print("\n✓ Checkpoint 4 passed — cleaning action plan created\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Clean the data — one step per problem found above
# ══════════════════════════════════════════════════════════════════════
# Each step logs how many rows it changed or removed, so the pipeline
# is auditable. Repair a value when the fix is unambiguous; drop the row
# when it is not.

taxi_clean = taxi_raw.clone()
rows_before = taxi_clean.height
cleaning_log: list[str] = []


def log_step(message: str) -> None:
    cleaning_log.append(message)
    print(message)


# --- 5a: Repair swapped GPS coordinates ---
# A "latitude" of ~103.8 with a "longitude" of ~1.35 is a Singapore point
# with the two fields swapped — the fix is unambiguous, so swap them back.
is_swapped = pl.col("pickup_latitude").is_between(SG_LNG_MIN, SG_LNG_MAX) & pl.col(
    "pickup_longitude"
).is_between(SG_LAT_MIN, SG_LAT_MAX)
taxi_clean = taxi_clean.with_columns(
    pl.when(is_swapped)
    .then(pl.col("pickup_longitude"))
    .otherwise(pl.col("pickup_latitude"))
    .alias("pickup_latitude"),
    pl.when(is_swapped)
    .then(pl.col("pickup_latitude"))
    .otherwise(pl.col("pickup_longitude"))
    .alias("pickup_longitude"),
)
log_step(f"GPS repair: swapped latitude/longitude back in {swapped_gps:,} rows")

# Anything still outside Singapore cannot be repaired — drop it
before = taxi_clean.height
taxi_clean = taxi_clean.filter(
    pl.col("pickup_latitude").is_between(SG_LAT_MIN, SG_LAT_MAX)
    & pl.col("pickup_longitude").is_between(SG_LNG_MIN, SG_LNG_MAX)
)
log_step(f"GPS filter: removed {before - taxi_clean.height:,} rows outside Singapore")

# --- 5b: Remove non-positive fares ---
before = taxi_clean.height
# TODO: Keep only positive fares
taxi_clean = taxi_clean.filter(pl.col("fare_sgd") > ____)  # Hint: 0
log_step(f"Fare filter (fare_sgd <= 0): removed {before - taxi_clean.height:,} rows")

# --- 5c: Remove impossible passenger counts ---
before = taxi_clean.height
# TODO: Keep only trips with at least one passenger
taxi_clean = taxi_clean.filter(pl.col("passengers") >= ____)  # Hint: 1
log_step(f"Passenger filter (< 1): removed {before - taxi_clean.height:,} rows")

# --- 5d: Remove trips dated after the log was extracted ---
before = taxi_clean.height
# TODO: Keep only pickups before the extract date
taxi_clean = taxi_clean.filter(
    pl.col("pickup_datetime") < ____  # Hint: DATA_EXTRACT_DATE
)
log_step(f"Future-date filter: removed {before - taxi_clean.height:,} rows")

# --- 5e: Normalise payment_type to four canonical labels ---
# Lower-case first, then match on substrings, so "CASH", "Cash Payment"
# and "cash" all become "Cash".
payment_lower = pl.col("payment_type").str.to_lowercase()
taxi_clean = taxi_clean.with_columns(
    pl.when(payment_lower.str.contains("grab"))
    .then(pl.lit("Grab"))
    .when(payment_lower.str.contains("nets"))
    .then(pl.lit("NETS"))
    # TODO: Map anything containing "cash" to "Cash"
    .when(payment_lower.str.contains(____))  # Hint: "cash"
    .then(pl.lit(____))  # Hint: "Cash"
    # TODO: Card payments appear as "card", "visa", "mastercard" or "credit"
    .when(payment_lower.str.contains(____))  # Hint: "card|visa|mastercard|credit"
    .then(pl.lit("Card"))
    .otherwise(pl.lit("Other"))
    .alias("payment_type")
)
log_step(
    f"Payment labels: {payment_spellings} spellings -> "
    f"{taxi_clean['payment_type'].n_unique()} canonical values"
)

# --- 5f: Resolve trip_id collisions ---
# The colliding rows are different trips (different times and places), so
# we cannot tell which one truly owns the ID. Keeping either would attach
# the wrong record to that ID; drop every row whose ID is not unique.
before = taxi_clean.height
# TODO: Keep only rows whose trip_id is NOT duplicated (~ means "not")
taxi_clean = taxi_clean.filter(~pl.col("trip_id").____())  # Hint: is_duplicated
log_step(f"trip_id collisions: removed {before - taxi_clean.height:,} rows")

# --- 5g: Fill nulls whose meaning is known ---
# A missing tip means no tip was given; a missing zone is "Unknown" — both
# are explicit values a model can use, rather than silent gaps.
# TODO: Fill the nulls with fill_null(value)
taxi_clean = taxi_clean.with_columns(
    pl.col("tip_sgd").fill_null(____),  # Hint: 0.0
    pl.col("pickup_zone").fill_null(____),  # Hint: "Unknown"
    pl.col("dropoff_zone").fill_null(____),  # Hint: "Unknown"
)
log_step("Null fill: tip_sgd -> 0.0, pickup_zone/dropoff_zone -> 'Unknown'")

retention_pct = taxi_clean.height / rows_before * 100
print(f"\n=== Cleaning Summary ===")
print(
    f"  Rows: {rows_before:,} -> {taxi_clean.height:,} ({retention_pct:.1f}% retained)"
)
print(f"  Steps applied: {len(cleaning_log)}")
for step in cleaning_log:
    print(f"    - {step}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert taxi_clean.height > 0, "Cleaning removed all rows"
assert taxi_clean.height <= taxi_raw.height, "Cleaning should not add rows"
assert (taxi_clean["fare_sgd"] > 0).all(), "All fares should be positive"
assert (taxi_clean["passengers"] >= 1).all(), "Every trip needs a passenger"
assert taxi_clean["pickup_latitude"].is_between(SG_LAT_MIN, SG_LAT_MAX).all()
assert taxi_clean["pickup_datetime"].max() < DATA_EXTRACT_DATE, "No future trips"
assert sorted(taxi_clean["payment_type"].unique().to_list()) == [
    "Card",
    "Cash",
    "Grab",
    "NETS",
], "payment_type should have exactly 4 canonical values"
assert taxi_clean["trip_id"].n_unique() == taxi_clean.height, "trip_id must be unique"
assert taxi_clean.null_count().sum_horizontal().item() == 0, "No nulls should remain"
print("\n✓ Checkpoint 5 passed — data cleaned and validated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Engineer temporal features
# ══════════════════════════════════════════════════════════════════════

# --- 6a: Extract pickup time features ---
# Polars dt.weekday() uses ISO numbering: Monday = 1 ... Sunday = 7.
# So Friday is 5 and the weekend is 6 (Saturday) and 7 (Sunday).
taxi_clean = taxi_clean.with_columns(
    pl.col("pickup_datetime").dt.hour().alias("hour_of_day"),
    pl.col("pickup_datetime").dt.weekday().alias("day_of_week"),
    pl.col("pickup_datetime").dt.month().alias("month"),
    # TODO: Saturday and Sunday are ISO days 6 and 7
    (pl.col("pickup_datetime").dt.weekday() >= ____).alias("is_weekend"),  # Hint: 6
)

# --- 6b: Peak-hour classification ---
# TODO: Classify each hour into a time_period using pl.when/then/otherwise
# morning_peak: 7-9,  evening_peak: 17-20,  late_night: 22-5,  off_peak: otherwise
taxi_clean = taxi_clean.with_columns(
    pl.when(
        (pl.col("hour_of_day") >= ____) & (pl.col("hour_of_day") <= ____)
    )  # Hint: 7, 9
    .then(pl.lit("morning_peak"))
    .when(
        (pl.col("hour_of_day") >= ____) & (pl.col("hour_of_day") <= ____)
    )  # Hint: 17, 20
    .then(pl.lit("evening_peak"))
    .when(
        (pl.col("hour_of_day") >= ____) | (pl.col("hour_of_day") <= ____)
    )  # Hint: 22, 5
    .then(pl.lit("late_night"))
    .otherwise(pl.lit("off_peak"))
    .alias("time_period")
)

# --- 6c: Day type classification ---
taxi_clean = taxi_clean.with_columns(
    # TODO: weekend = ISO day 6 or 7; Friday = ISO day 5
    pl.when(pl.col("day_of_week") >= ____)  # Hint: 6
    .then(pl.lit("weekend"))
    .when(pl.col("day_of_week") == ____)  # Hint: 5
    .then(pl.lit("friday"))
    .otherwise(pl.lit("weekday"))
    .alias("day_type")
)

# --- 6d: Duration features ---
# Subtracting two Datetime columns gives a Duration; total_seconds()
# turns it into a number.
taxi_clean = taxi_clean.with_columns(
    (
        # TODO: Convert the duration in seconds to minutes
        (pl.col("dropoff_datetime") - pl.col("pickup_datetime")).dt.total_seconds()
        / ____  # Hint: 60
    ).alias("trip_duration_min"),
)
taxi_clean = taxi_clean.with_columns(
    pl.when(pl.col("trip_duration_min") < 15)
    .then(pl.lit("short"))
    .when(pl.col("trip_duration_min") < 40)
    .then(pl.lit("medium"))
    .otherwise(pl.lit("long"))
    .alias("trip_length_category"),
)

temporal_cols = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "time_period",
    "day_type",
    "trip_duration_min",
    "trip_length_category",
]
print(f"=== Temporal Features ({len(temporal_cols)}) ===")
print(taxi_clean.select(temporal_cols).head(5))
print("\nTrips by day_type:")
print(taxi_clean["day_type"].value_counts().sort("day_type"))

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert all(c in taxi_clean.columns for c in temporal_cols)
assert set(taxi_clean["day_of_week"].unique().to_list()) <= set(range(1, 8))
assert (
    taxi_clean.filter(pl.col("is_weekend"))["day_of_week"].min() == 6
), "Weekend must start on Saturday (ISO day 6)"
assert "morning_peak" in set(taxi_clean["time_period"].unique().to_list())
assert (taxi_clean["trip_duration_min"] > 0).all(), "Durations must be positive"
print("\n✓ Checkpoint 6 passed — temporal features engineered\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Engineer spatial features
# ══════════════════════════════════════════════════════════════════════

# --- 7a: Distance from the pickup point to the CBD (haversine) ---
# The haversine formula gives the great-circle distance between two
# latitude/longitude points on a sphere of radius 6,371 km.
CBD_LAT, CBD_LNG = 1.2840, 103.8514  # Raffles Place
_RAD = math.pi / 180

taxi_clean = taxi_clean.with_columns(
    (
        2
        * 6371
        * (
            ((pl.col("pickup_latitude") - CBD_LAT) * _RAD / 2).sin().pow(2)
            + math.cos(CBD_LAT * _RAD)
            * (pl.col("pickup_latitude") * _RAD).cos()
            * ((pl.col("pickup_longitude") - CBD_LNG) * _RAD / 2).sin().pow(2)
        )
        .sqrt()
        .arcsin()
    ).alias("km_from_cbd")
)

# --- 7b: Average speed — a consistency check between distance and time ---
# TODO: Speed = distance / time in hours (minutes / 60)
taxi_clean = taxi_clean.with_columns(
    (pl.col("distance_km") / (pl.col("trip_duration_min") / ____)).alias(  # Hint: 60
        "avg_speed_kmh"
    )
)

# Speeds above 120 km/h ("teleporting" trips) or below 2 km/h (slower than
# walking for the whole trip) mean distance or time was recorded wrongly.
before = taxi_clean.height
# TODO: Keep speeds between 2 and 120 km/h with .is_between(low, high)
taxi_clean = taxi_clean.filter(
    pl.col("avg_speed_kmh").is_between(____, ____)  # Hint: 2, 120
)
log_step(f"Speed filter (outside 2-120 km/h): removed {before - taxi_clean.height:,} rows")

# --- 7c: Distance category ---
taxi_clean = taxi_clean.with_columns(
    pl.when(pl.col("distance_km") < 3)
    .then(pl.lit("short_distance"))
    .when(pl.col("distance_km") < 8)
    .then(pl.lit("medium_distance"))
    .when(pl.col("distance_km") < 15)
    .then(pl.lit("long_distance"))
    .otherwise(pl.lit("cross_island"))
    .alias("distance_category"),
)

# --- 7d: Fare per km ---
taxi_clean = taxi_clean.with_columns(
    (pl.col("fare_sgd") / pl.col("distance_km")).alias("fare_per_km")
)

spatial_cols = ["km_from_cbd", "avg_speed_kmh", "distance_category", "fare_per_km"]
print(f"\n=== Spatial Features ({len(spatial_cols)}) ===")
print(taxi_clean.select(spatial_cols).describe())

new_cols = [c for c in taxi_clean.columns if c not in taxi_raw.columns]
print(f"\nTotal new features: {len(new_cols)}")
retention_pct = taxi_clean.height / rows_before * 100
print(f"Rows after all cleaning: {taxi_clean.height:,} ({retention_pct:.1f}% retained)")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert (taxi_clean["km_from_cbd"] >= 0).all()
assert (
    taxi_clean["km_from_cbd"].max() < 50
), "Every pickup is in Singapore, so within 50 km of the CBD"
assert taxi_clean["avg_speed_kmh"].is_between(2, 120).all()
print("\n✓ Checkpoint 7 passed — spatial features engineered\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: PreprocessingPipeline — model-ready features
# ══════════════════════════════════════════════════════════════════════
# PreprocessingPipeline.setup() learns its imputation values, scaling
# statistics and category lists from EVERY row you hand it — and only
# splits train/test afterwards. Hand it the whole dataset and the test
# rows quietly shape the transformations the model is judged with.
#
# So the order is:
#   1. hold the test rows out FIRST (before anything is fitted),
#   2. fit the pipeline on the TRAINING rows only,
#   3. apply the fitted pipeline to the test rows with transform().

# --- 8a: Select feature columns ---
# Excluded on purpose:
#   trip_id                    — an identifier, not a feature
#   pickup/dropoff_datetime    — already turned into hour/day/duration
#   tip_sgd, fare_per_km       — only known AFTER the fare (target leakage)
#   pickup_latitude/longitude  — summarised by km_from_cbd and the zones
feature_cols = [
    "distance_km",
    "trip_duration_min",
    "avg_speed_kmh",
    "km_from_cbd",
    "passengers",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "time_period",
    "day_type",
    "payment_type",
    "pickup_zone",
    "dropoff_zone",
    "distance_category",
    "trip_length_category",
]
pipeline_df = taxi_clean.select(feature_cols + ["fare_sgd"])

# --- 8b: Hold out the test rows FIRST ---
shuffled = pipeline_df.sample(fraction=1.0, shuffle=True, seed=42)
n_train = int(shuffled.height * 0.8)
# TODO: the first n_train shuffled rows are the training rows; the rest
# are the test rows.
train_raw = ____  # Hint: shuffled.head(n_train)
test_raw = ____  # Hint: shuffled.tail(shuffled.height - n_train)
print(f"Held out before any fitting: {train_raw.height:,} train / {test_raw.height:,} test rows")

# --- 8c: Fit PreprocessingPipeline on the TRAINING rows only ---
# setup() also splits the rows it is given into two parts; here that split
# stays inside the training rows and we do not use it. The honest test
# set is test_raw, which setup() never sees.
pipeline = PreprocessingPipeline()
# TODO: Call pipeline.setup() on the TRAINING rows only:
#   data=train_raw, target="fare_sgd", seed=42
#   normalize=True, categorical_encoding="onehot", imputation_strategy="median"
result = pipeline.setup(
    data=____,  # Hint: train_raw — never the full pipeline_df
    target=____,  # Hint: "fare_sgd"
    seed=42,
    normalize=____,  # Hint: True
    categorical_encoding=____,  # Hint: "onehot"
    imputation_strategy=____,  # Hint: "median"
)
# TODO: Apply the FITTED pipeline to each split.
train_data = ____  # Hint: pipeline.transform(train_raw)
test_data = ____  # Hint: pipeline.transform(test_raw)

print("=== PreprocessingPipeline Result ===")
print(f"  Task type:     {result.task_type}")
print(f"  Train shape:   {train_data.shape}")
print(f"  Test shape:    {test_data.shape}")
print(f"  Numeric feats: {len(result.numeric_columns)}")
print(f"  Cat feats:     {len(result.categorical_columns)}")

# --- 8d: Inspect the processed features ---
print(f"\n  Numeric columns: {result.numeric_columns[:10]}")
print(f"  Categorical columns: {result.categorical_columns[:10]}")
print(f"  Shape before -> after: {train_raw.shape} -> {train_data.shape}")
# INTERPRETATION: the column count grows because one-hot encoding turns
# each categorical column into one 0/1 column per category.

# --- 8e: Prove the test rows did not shape the scaling ---
# The fitted scaler stores the mean it subtracts from each numeric column.
scaler = result.transformers["scaler"]
dist_idx = result.numeric_columns.index("distance_km")
train_mean = train_raw["distance_km"].mean()
all_rows_mean = pipeline_df["distance_km"].mean()
print("\n  distance_km mean:")
print(f"    training rows only:    {train_mean:.4f} km")
print(f"    all rows (incl. test): {all_rows_mean:.4f} km")
print(f"    learned by the scaler: {scaler.mean_[dist_idx]:.4f} km")
# INTERPRETATION: the scaler's mean equals the TRAINING-rows mean, not the
# all-rows mean — the test rows were never seen while fitting. Had we
# passed pipeline_df to setup(), the scaler would hold the all-rows mean.

# --- 8f: Verify train/test split ---
total_rows = train_data.shape[0] + test_data.shape[0]
train_pct = train_data.shape[0] / total_rows * 100
print(f"\n  Train: {train_data.shape[0]:,} ({train_pct:.0f}%)")
print(f"  Test:  {test_data.shape[0]:,} ({100 - train_pct:.0f}%)")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert result is not None, "PreprocessingPipeline.setup() must return a result"
assert result.task_type == "regression"
assert result.target_column == "fare_sgd"
assert total_rows == pipeline_df.height, "Every row is in exactly one split"
assert train_data["fare_sgd"].null_count() == 0
assert abs(scaler.mean_[dist_idx] - train_mean) < 1e-9, \
    "The scaler must be fitted on the training rows only"
assert train_data.columns == test_data.columns, \
    "Both splits must go through the same fitted pipeline"
print("\n✓ Checkpoint 8 passed — PreprocessingPipeline complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Visualise key patterns with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════
# ModelVisualizer returns Plotly figures. Some of its charts were built
# for model results (metric_comparison labels its axes "Model"/"Score",
# training_history plots against "Epoch" 1..N), so when you reuse them
# for data, set the real x values and axis titles — a chart with the
# wrong axis labels is a misleading chart.

viz = ModelVisualizer()
os.makedirs("charts", exist_ok=True)
viz_files: list[str] = []

# --- 9a: Fare distribution (cleaned) ---
# TODO: Use viz.histogram(data, column, bins=..., title=...) on the fare column
fig_fare = viz.histogram(
    taxi_clean, ____, bins=60, title="Taxi Fare Distribution (After Cleaning)"
)  # Hint: "fare_sgd"
fig_fare.update_layout(xaxis_title="Fare (S$)", yaxis_title="Number of trips")
fig_fare.write_html("charts/ex8_fare_distribution.html")
viz_files.append("charts/ex8_fare_distribution.html")
print("Saved: charts/ex8_fare_distribution.html")

# --- 9b: Average fare, distance and speed by time period ---
periods = ["morning_peak", "evening_peak", "off_peak", "late_night"]
period_stats = (
    taxi_clean.group_by("time_period")
    .agg(
        pl.col("fare_sgd").mean().alias("avg_fare_sgd"),
        pl.col("distance_km").mean().alias("avg_distance_km"),
        pl.col("avg_speed_kmh").mean().alias("avg_speed_kmh"),
        pl.len().alias("trips"),
    )
    .sort("time_period")
)
print("\n=== Trips by time period ===")
print(period_stats)
time_metrics = {
    row["time_period"]: {
        "Avg fare (S$)": row["avg_fare_sgd"],
        "Avg distance (km)": row["avg_distance_km"],
        "Avg speed (km/h)": row["avg_speed_kmh"],
    }
    for row in period_stats.iter_rows(named=True)
}
time_metrics = {p: time_metrics[p] for p in periods if p in time_metrics}
period_fare_spread = period_stats["avg_fare_sgd"].max() - period_stats["avg_fare_sgd"].min()
print(f"Spread in average fare across time periods: S${period_fare_spread:.2f}")
# TODO: Use viz.metric_comparison() to compare the periods
fig_periods = viz.metric_comparison(____)  # Hint: time_metrics
fig_periods.update_layout(
    title="Average Fare, Distance and Speed by Time Period",
    xaxis_title="Time period",
    yaxis_title="Average value",
)
fig_periods.write_html("charts/ex8_time_period_metrics.html")
viz_files.append("charts/ex8_time_period_metrics.html")
print("Saved: charts/ex8_time_period_metrics.html")

# --- 9c: Hourly trip volume ---
hourly = (
    taxi_clean.group_by("hour_of_day")
    .agg(pl.len().alias("trip_count"))
    .sort("hour_of_day")
)
fig_hourly = viz.training_history(
    metrics={"Trip Volume": hourly["trip_count"].to_list()},
    x_label=____,  # Hint: "Hour of Day"
    y_label=____,  # Hint: "Number of Trips"
)
# training_history plots against 1..N; put the real hours 0..23 on the x-axis
# TODO: Replace the trace's x values with the real hours
fig_hourly.update_traces(x=____)  # Hint: hourly["hour_of_day"].to_list()
fig_hourly.update_layout(title="Taxi Trip Volume by Hour of Day")
fig_hourly.write_html("charts/ex8_hourly_volume.html")
viz_files.append("charts/ex8_hourly_volume.html")
print("Saved: charts/ex8_hourly_volume.html")

# --- 9d: Distance distribution ---
fig_dist_km = viz.histogram(
    taxi_clean, "distance_km", bins=60, title="Trip Distance Distribution"
)
fig_dist_km.update_layout(xaxis_title="Distance (km)", yaxis_title="Number of trips")
fig_dist_km.write_html("charts/ex8_distance_distribution.html")
viz_files.append("charts/ex8_distance_distribution.html")
print("Saved: charts/ex8_distance_distribution.html")

# --- 9e: Day type comparison ---
day_stats = (
    taxi_clean.group_by("day_type")
    .agg(
        pl.col("fare_sgd").mean().alias("avg_fare_sgd"),
        pl.col("fare_per_km").median().alias("median_fare_per_km"),
    )
    .sort("day_type")
)
print("\n=== Fares by day type ===")
print(day_stats)
day_fare_spread = day_stats["avg_fare_sgd"].max() - day_stats["avg_fare_sgd"].min()
print(f"Spread in average fare across day types: S${day_fare_spread:.2f}")
day_metrics = {
    row["day_type"]: {
        "Avg fare (S$)": row["avg_fare_sgd"],
        "Median fare per km (S$)": row["median_fare_per_km"],
    }
    for row in day_stats.iter_rows(named=True)
}
fig_day = viz.metric_comparison(day_metrics)
fig_day.update_layout(
    title="Fares: Weekday vs Friday vs Weekend",
    xaxis_title="Day type",
    yaxis_title="S$",
)
fig_day.write_html("charts/ex8_day_type_comparison.html")
viz_files.append("charts/ex8_day_type_comparison.html")
print("Saved: charts/ex8_day_type_comparison.html")
# INTERPRETATION: read the printed spreads before reading a story into the
# bars. A spread of a few cents on a ~S$10 fare means time of day and day
# of week barely move the fare in this data — a finding worth reporting
# as such, rather than inventing a "peak-hour premium" the data lacks.

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(viz_files) >= 3, "The report needs at least 3 visualisations"
for f in viz_files:
    assert os.path.exists(f), f"Missing: {f}"
print(f"\n✓ Checkpoint 9 passed — {len(viz_files)} visualisation files saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Compare original vs cleaned data and generate a quality report
# ══════════════════════════════════════════════════════════════════════

# --- 10a: Re-profile the cleaned data with the same thresholds ---
# Profile only the original columns so the alert counts are comparable.
taxi_clean_original_cols = taxi_clean.select(taxi_raw.columns)
# TODO: Re-profile with the SAME alert_config used on the raw data
profile_clean = run_profile(____, alert_config)  # Hint: taxi_clean_original_cols

print(f"=== Data Quality: Before vs After Cleaning ===")
print(f"  Rows before:   {taxi_raw.height:,}")
print(f"  Rows after:    {taxi_clean.height:,}")
print(f"  Retention:     {taxi_clean.height / taxi_raw.height:.1%}")
print(f"  Alerts before: {len(profile_raw.alerts)}")
print(f"  Alerts after:  {len(profile_clean.alerts)}")

alert_reduction = len(profile_raw.alerts) - len(profile_clean.alerts)
if alert_reduction > 0:
    print(f"  Improvement:   {alert_reduction} fewer alerts")
elif alert_reduction == 0:
    print(f"  No change in alert count")
else:
    print(f"  Warning: {abs(alert_reduction)} MORE alerts after cleaning")

if profile_clean.alerts:
    print(f"\n  Remaining alerts ({len(profile_clean.alerts)}):")
    for alert in profile_clean.alerts:
        print(
            f"    [{alert['severity'].upper()}] {alert['type']}: "
            f"{alert_target(alert)}"
        )
else:
    print(f"\n  No remaining alerts.")
# INTERPRETATION: an alert that survives cleaning is not automatically a
# failure. trip_id stays near-unique (high_cardinality) because it is an
# ID; tip_sgd now has many zeros because "no tip" is common. Cleaning can
# even ADD an alert: once the negative fares are gone, the real link
# between distance and fare may show up as high_correlation. Judge each
# remaining alert — do not chase a count of zero.

# --- 10b: Column-by-column comparison of original vs cleaned ---
# TODO: Compare original (first) vs cleaned (second) with run_compare(df_a, df_b)
comparison = run_compare(____, ____)  # Hint: taxi_raw, taxi_clean_original_cols
shape = comparison["shape_comparison"]
print(f"\n=== run_compare(original, cleaned) ===")
print(f"  Rows: {shape['rows_a']:,} -> {shape['rows_b']:,}")
print(f"  {'Column':>20} {'null % before':>14} {'null % after':>13}")
for col in ["fare_sgd", "passengers", "tip_sgd", "pickup_zone", "dropoff_zone"]:
    before_col = comparison["profile_a"].columns
    after_col = comparison["profile_b"].columns
    null_a = next(c.null_pct for c in before_col if c.name == col)
    null_b = next(c.null_pct for c in after_col if c.name == col)
    print(f"  {col:>20} {null_a:>13.1%} {null_b:>13.1%}")
fare_a = next(c for c in comparison["profile_a"].columns if c.name == "fare_sgd")
fare_b = next(c for c in comparison["profile_b"].columns if c.name == "fare_sgd")
print(f"  fare_sgd min: {fare_a.min_val:.2f} -> {fare_b.min_val:.2f}")

# --- 10c: HTML profile report of the cleaned data ---
# TODO: Build the HTML report of the cleaned data with run_report(df, title=...)
report_html = run_report(
    ____,  # Hint: taxi_clean_original_cols
    title="Singapore Taxi Trips — Cleaned Data Profile",
    alert_config=alert_config,
)
with open("ex8_taxi_profile_clean.html", "w") as f:
    f.write(report_html)
print(f"\nSaved: ex8_taxi_profile_clean.html")

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert profile_clean is not None
assert shape["rows_b"] == taxi_clean.height
assert fare_b.min_val > 0, "Cleaned fares should all be positive"
assert os.path.exists("ex8_taxi_profile_clean.html")
print("\n✓ Checkpoint 10 passed — original vs cleaned compared, report saved\n")


# ── Pipeline summary ─────────────────────────────────────────────────
print(f"\n{'═' * 65}")
print(f"  END-TO-END PIPELINE SUMMARY")
print(f"{'═' * 65}")
print(f"  Stage 1  Load:       {taxi_raw.height:,} rows, {taxi_raw.width} cols")
print(f"  Stage 2  Inspect:    {len(quality_issues)} quality issues identified")
print(f"  Stage 3  Profile:    {len(profile_raw.alerts)} alerts from DataExplorer")
print(f"  Stage 4  Plan:       {len(cleaning_plan)} cleaning actions defined")
print(f"  Stage 5  Clean:      {len(cleaning_log)} logged cleaning steps")
print(f"  Stage 6  Temporal:   {len(temporal_cols)} temporal features")
print(f"  Stage 7  Spatial:    {len(spatial_cols)} spatial features")
print(
    f"  Stage 8  Pipeline:   {train_data.shape[0]:,} train / "
    f"{test_data.shape[0]:,} test"
)
print(f"  Stage 9  Visualise:  {len(viz_files)} charts saved")
print(f"  Stage 10 Verify:     {len(profile_clean.alerts)} alerts remaining")
print(f"  Final rows:          {taxi_clean.height:,} ({retention_pct:.1f}% retained)")
print(f"{'═' * 65}")

print(
    "\n✓ Exercise 8 complete — full pipeline: load -> profile -> clean -> "
    "engineer -> preprocess -> visualise -> verify"
)


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(
    """
  ✓ End-to-end thinking: load -> inspect -> profile -> plan -> clean
    -> engineer -> preprocess -> visualise -> verify
  ✓ Manual quality checks: null counts, range validation, domain rules
  ✓ DataExplorer: automated profiling with typed, actionable alerts
  ✓ Action planning: translating alerts into specific cleaning steps
  ✓ Domain-aware cleaning: repair when unambiguous (swapped GPS), drop
    when not (negative fares, future dates, colliding IDs), normalise
    messy labels (15 payment spellings -> 4)
  ✓ Feature engineering:
    - Temporal: hour, ISO weekday, peak period, day type, duration
    - Spatial: haversine distance to the CBD, average speed
    - Derived: fare per km
  ✓ PreprocessingPipeline: hold the test rows out first, fit impute /
    scale / encode on the training rows, transform() the test rows
  ✓ ModelVisualizer: distributions, hourly patterns, segment comparisons
    — with honest axis labels
  ✓ Quality measurement: original vs cleaned with run_compare()
  ✓ Three engines together: DataExplorer + PreprocessingPipeline +
    ModelVisualizer — the full M1 toolkit

  MODULE 1 COMPLETE — you've gone from raw data to model-ready data.

  NEXT — MODULE 2: Statistical Mastery for Machine Learning and
  Artificial Intelligence (AI) Success
  In M2, you'll learn the statistics behind every modelling decision:
    - Probability, Bayesian thinking and statistical inference
    - Bootstrapping, hypothesis testing and A/B experiment design
    - Linear and logistic regression as inference tools
    - ExperimentTracker, FeatureEngineer and FeatureStore
"""
)
