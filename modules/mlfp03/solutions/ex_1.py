# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1: Feature Engineering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Engineer clinical features from multi-table medical data with
#     temporal point-in-time correctness (no leakage)
#   - Aggregate time-series vitals into statistical summaries per admission
#   - Flag clinically significant medication and lab patterns
#   - Compute derived features (abnormal lab ratio, medication intensity)
#   - Validate a feature set against a declared FeatureSchema
#   - Log feature engineering experiments with ExperimentTracker
#
# PREREQUISITES:
#   - MLFP02 complete (statistics, Bayesian thinking, linear regression)
#   - ExperimentTracker introduced in MLFP02 Exercise 7
#
# ESTIMATED TIME: 60-90 minutes
#
# TASKS:
#   1. Load and inspect messy ICU data (irregular vitals, multi-table)
#   2. Create ExperimentTracker experiment (used across all M3 exercises)
#   3. Handle temporal features with point-in-time correctness
#   4. Engineer clinical features (rolling vitals, medication interactions)
#   5. Validate features with FeatureSchema
#   6. Log feature engineering run to ExperimentTracker
#
# DATASET: ICU patient data from MLFP02 (multi-table clinical records)
#   Tables: patients, admissions, vitals, medications, labs
#   Target: in-hospital mortality (binary classification in later exercises)
#   Key challenge: vitals recorded at irregular intervals per patient
#
# DATA QUALITY:
#   - Irregular time-series (vitals recorded at different frequencies)
#   - Multi-table joins (patients, admissions, vitals, medications, labs)
#   - Clinical missing patterns (not MCAR — sicker patients get more tests)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl
from kailash.db import ConnectionManager
from kailash_ml import FeatureEngineer, DataExplorer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.types import FeatureSchema, FeatureField

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()

patients = loader.load("mlfp02", "icu_patients.parquet")
admissions = loader.load("mlfp02", "icu_admissions.parquet")
vitals = loader.load("mlfp02", "icu_vitals.parquet")
medications = loader.load("mlfp02", "icu_medications.parquet")
labs = loader.load("mlfp02", "icu_labs.parquet")

# Cast all timestamp columns to datetime for consistent comparison
_DT_FMT = "%Y-%m-%d %H:%M:%S"
admissions = admissions.with_columns(
    pl.col("admit_time").str.to_datetime(_DT_FMT),
    pl.col("discharge_time").str.to_datetime(_DT_FMT),
)
medications = medications.with_columns(
    pl.col("start_time").str.to_datetime(_DT_FMT),
    pl.col("end_time").str.to_datetime(_DT_FMT),
)
if "timestamp" in labs.columns and labs["timestamp"].dtype == pl.String:
    labs = labs.with_columns(pl.col("timestamp").str.to_datetime(_DT_FMT))

print("=== ICU Dataset ===")
for name, df in [
    ("patients", patients),
    ("admissions", admissions),
    ("vitals", vitals),
    ("medications", medications),
    ("labs", labs),
]:
    print(f"  {name}: {df.shape} — columns: {df.columns}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Inspect the data — understand the mess
# ══════════════════════════════════════════════════════════════════════
# In clinical data, vitals are not recorded on a fixed schedule.
# A patient in crisis may have readings every 5 minutes; a stable
# patient every 2 hours. This irregular sampling is itself a signal
# of patient severity — sicker patients get more monitoring.

# Vitals are recorded at irregular intervals — join with admissions to get patient_id
vitals = vitals.join(
    admissions.select(["admission_id", "patient_id"]),
    on="admission_id",
    how="left",
)

# Cast timestamp string to datetime for temporal operations
vitals = vitals.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S"))

# Melt wide-format vitals to long format (vital_name, value)
vital_cols = ["heart_rate", "systolic_bp", "diastolic_bp", "temperature", "spo2", "respiratory_rate"]
vital_cols_present = [c for c in vital_cols if c in vitals.columns]
if vital_cols_present:
    vitals = vitals.unpivot(vital_cols_present, index=["admission_id", "patient_id", "timestamp"], variable_name="vital_name", value_name="value")

print("\n=== Vital Signs Sample (one patient) ===")
sample_patient = vitals["patient_id"].unique()[0]
patient_vitals = vitals.filter(pl.col("patient_id") == sample_patient).sort(
    "timestamp"
)
print(patient_vitals.head(20))

# Check recording frequency
if patient_vitals.height > 1:
    time_diffs = patient_vitals.with_columns(
        (pl.col("timestamp").diff()).alias("time_gap")
    )
    print(f"\nTime gaps between readings:")
    print(time_diffs.select("vital_name", "time_gap").head(10))

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert patients.height > 0, "patients DataFrame is empty"
assert vitals.height > 0, "vitals DataFrame is empty"
assert "patient_id" in vitals.columns, "vitals must have patient_id"
# INTERPRETATION: ICU data is messy by design — recording frequency
# encodes patient severity. A feature like 'vital_count' captures this
# indirectly: patients with more readings may be more critically ill.
print("\n✓ Checkpoint 1 passed — ICU data loaded and inspected\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Set up ExperimentTracker (persists across all M3 exercises)
# ══════════════════════════════════════════════════════════════════════
# ExperimentTracker records every feature engineering run so we can
# compare approaches, reproduce past results, and audit what data
# was used to build each feature set. This is the foundation of
# reproducible ML — every run is logged, not just the best one.


async def setup_tracking():
    """Initialize ExperimentTracker for Module 3."""
    conn = ConnectionManager("sqlite:///mlfp03_experiments.db")
    await conn.initialize()

    tracker = ExperimentTracker(conn)

    # Create the Module 3 experiment
    experiment_id = await tracker.create_experiment(
        name="mlfp03_healthcare_features",
        description="Feature engineering experiments on ICU data — Module 3",
        tags=["mlfp03", "healthcare", "feature-engineering"],
    )
    print(f"\nExperiment created: {experiment_id}")

    return conn, tracker, experiment_id


conn, tracker, experiment_id = asyncio.run(setup_tracking())

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert conn is not None, "ConnectionManager failed to initialize"
assert tracker is not None, "ExperimentTracker failed to initialize"
assert experiment_id is not None, "Experiment ID should not be None"
# INTERPRETATION: Every ML project needs a tracking system. Without one,
# you cannot reproduce past results or audit which features were used.
print("\n✓ Checkpoint 2 passed — ExperimentTracker initialized\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Temporal features with point-in-time correctness
# ══════════════════════════════════════════════════════════════════════
# CRITICAL: Features must only use data available BEFORE the prediction
# time. Using future data (leakage) inflates validation metrics but
# fails catastrophically in production.
#
# Example of leakage: including discharge diagnosis in features used to
# predict in-hospital mortality. In production, discharge hasn't happened yet.

# Join patients with admissions
patient_admissions = patients.join(admissions, on="patient_id", how="inner")

# For each admission, compute features using ONLY data before discharge
# Prediction target: in-hospital mortality (known at discharge)

# Aggregate vitals PER ADMISSION with temporal correctness
# Only use vitals recorded DURING this admission (between admit and discharge)
vitals_features = (
    vitals.join(
        admissions.select("patient_id", "admission_id", "admit_time", "discharge_time"),
        on="patient_id",
        how="inner",
    )
    # Point-in-time filter: only vitals during THIS admission
    .filter(
        (pl.col("timestamp") >= pl.col("admit_time"))
        & (pl.col("timestamp") <= pl.col("discharge_time"))
    )
)

# Pivot vital signs to columns and compute temporal aggregates
vital_names = vitals_features["vital_name"].unique().to_list()

vital_aggs = []
for vital in vital_names:
    vital_data = vitals_features.filter(pl.col("vital_name") == vital)
    agg = vital_data.group_by("admission_id").agg(
        pl.col("value").mean().alias(f"{vital}_mean"),
        pl.col("value").std().alias(f"{vital}_std"),
        pl.col("value").min().alias(f"{vital}_min"),
        pl.col("value").max().alias(f"{vital}_max"),
        # Trend: last reading minus first reading
        (pl.col("value").last() - pl.col("value").first()).alias(f"{vital}_trend"),
        # Count of readings (proxy for severity — sicker patients get more monitoring)
        pl.col("value").count().alias(f"{vital}_count"),
    )
    vital_aggs.append(agg)

# Join all vital aggregates
features = patient_admissions.clone()
for agg in vital_aggs:
    features = features.join(agg, on="admission_id", how="left")

print(f"\n=== Features after vital aggregation ===")
print(f"Shape: {features.shape}")
print(
    f"New vital columns: {[c for c in features.columns if any(v in c for v in vital_names)][:10]}..."
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
vital_cols = [c for c in features.columns if any(v in c for v in vital_names)]
assert len(vital_cols) > 0, "No vital feature columns were created"
assert features.height > 0, "Feature DataFrame is empty after aggregation"
# INTERPRETATION: The _count suffix columns are particularly valuable.
# A patient with heart_rate_count=120 in a 24h stay (5 readings/hr)
# is being monitored far more intensively than one with count=8.
# This is a form of clinical severity encoding baked into the data.
print("\n✓ Checkpoint 3 passed — temporal vital features created\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Engineer clinical features (medications, labs, interactions)
# ══════════════════════════════════════════════════════════════════════
# Domain knowledge is essential here. A data scientist without clinical
# knowledge would not know that vasopressors indicate hemodynamic
# instability, or that abnormal lab ratios signal systemic illness.
# This is why "data > models" — domain features beat raw data.

# Medication features — count of distinct medications, specific drug flags
med_features = (
    medications.join(
        admissions.select("patient_id", "admission_id", "admit_time", "discharge_time"),
        on="admission_id",
        how="inner",
    )
    .filter(
        (pl.col("start_time") >= pl.col("admit_time"))
        & (pl.col("start_time") <= pl.col("discharge_time"))
    )
    .group_by("admission_id")
    .agg(
        pl.col("drug_name").n_unique().alias("n_unique_medications"),
        pl.col("drug_name").count().alias("n_medication_doses"),
        # Flag high-risk medications (vasopressors indicate hemodynamic instability)
        pl.col("drug_name")
        .str.contains("(?i)vasopressor|norepinephrine|dopamine")
        .any()
        .alias("received_vasopressors"),
        # Antibiotic flag (infection)
        pl.col("drug_name")
        .str.contains("(?i)antibiotic|vancomycin|meropenem")
        .any()
        .alias("received_antibiotics"),
    )
)

features = features.join(med_features, on="admission_id", how="left")

# Lab features — most recent lab values and abnormal counts
lab_features = (
    labs.join(
        admissions.select("admission_id", "admit_time", "discharge_time"),
        on="admission_id",
        how="inner",
    )
    .filter(
        (pl.col("timestamp") >= pl.col("admit_time"))
        & (pl.col("timestamp") <= pl.col("discharge_time"))
    )
    .group_by("admission_id")
    .agg(
        pl.col("test_name").n_unique().alias("n_unique_labs"),
        pl.col("value").count().alias("n_lab_results"),
        # Abnormal results (flag column in source data)
        (pl.col("flag") != "normal").sum().alias("n_abnormal_labs"),
    )
)

features = features.join(lab_features, on="admission_id", how="left")

# Derived features
features = features.with_columns(
    # Abnormal lab ratio
    (pl.col("n_abnormal_labs") / pl.col("n_lab_results").clip(lower_bound=1)).alias(
        "abnormal_lab_ratio"
    ),
    # Medication intensity (doses per day of stay)
    (
        pl.col("n_medication_doses") / pl.col("los_days").clip(lower_bound=1)
    ).alias("medication_intensity"),
)

# Fill nulls for patients with no medications/labs (they exist!)
features = features.with_columns(
    pl.col("n_unique_medications").fill_null(0),
    pl.col("n_medication_doses").fill_null(0),
    pl.col("received_vasopressors").fill_null(False),
    pl.col("received_antibiotics").fill_null(False),
    pl.col("n_unique_labs").fill_null(0),
    pl.col("n_lab_results").fill_null(0),
    pl.col("n_abnormal_labs").fill_null(0),
    pl.col("abnormal_lab_ratio").fill_null(0.0),
    pl.col("medication_intensity").fill_null(0.0),
)

print(f"\n=== Features after medication + lab engineering ===")
print(f"Shape: {features.shape}")
print(f"Total feature columns: {len(features.columns)}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert "n_unique_medications" in features.columns, "Medication features missing"
assert "abnormal_lab_ratio" in features.columns, "Derived lab ratio feature missing"
assert "medication_intensity" in features.columns, "Derived medication intensity missing"
assert features["abnormal_lab_ratio"].null_count() == 0, "Null values in lab ratio"
# INTERPRETATION: abnormal_lab_ratio captures systemic illness severity.
# A ratio of 0.6 means 60% of lab tests came back abnormal — this patient
# is in serious trouble. medication_intensity (doses/day) captures treatment
# burden, which correlates with disease complexity.
print("\n✓ Checkpoint 4 passed — clinical features engineered\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Validate features with FeatureSchema
# ══════════════════════════════════════════════════════════════════════
# FeatureSchema is a contract: it declares what columns are required,
# their types, and whether nulls are allowed. This prevents silent
# failures when features are computed incorrectly or columns are renamed.

# Define the expected schema for our engineered features
icu_schema = FeatureSchema(
    name="icu_clinical_features_v1",
    features=[
        FeatureField(
            name="age",
            dtype="float64",
            nullable=False,
            description="Patient age at admission",
        ),
        FeatureField(
            name="los_days",
            dtype="float64",
            nullable=False,
            description="Length of ICU stay in days",
        ),
        FeatureField(
            name="n_unique_medications",
            dtype="int64",
            nullable=False,
            description="Count of distinct medications administered",
        ),
        FeatureField(
            name="received_vasopressors",
            dtype="bool",
            nullable=False,
            description="Whether patient received vasopressor drugs",
        ),
        FeatureField(
            name="n_abnormal_labs",
            dtype="int64",
            nullable=False,
            description="Count of abnormal lab results",
        ),
        FeatureField(
            name="abnormal_lab_ratio",
            dtype="float64",
            nullable=False,
            description="Proportion of lab results flagged abnormal",
        ),
        FeatureField(
            name="medication_intensity",
            dtype="float64",
            nullable=False,
            description="Medication doses per day of stay",
        ),
    ],
    entity_id_column="patient_id",
    timestamp_column="admit_time",
    version=1,
)

print(f"\n=== FeatureSchema: {icu_schema.name} ===")
print(f"Entity ID: {icu_schema.entity_id_column}")
print(f"Timestamp: {icu_schema.timestamp_column}")
for f in icu_schema.features:
    print(
        f"  {f.name}: {f.dtype} ({'nullable' if f.nullable else 'required'}) — {f.description}"
    )

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert icu_schema.name == "icu_clinical_features_v1", "Schema name mismatch"
assert len(icu_schema.features) == 7, "Schema should declare 7 features"
for field in icu_schema.features:
    assert field.name in features.columns, f"Declared feature '{field.name}' missing from DataFrame"
# INTERPRETATION: A FeatureSchema acts as living documentation AND
# runtime validation. If feature engineering code changes and a column
# disappears, the schema check catches it before a model trains on
# bad data. Think of it as a type system for ML features.
print("\n✓ Checkpoint 5 passed — FeatureSchema validated against features\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Log to ExperimentTracker
# ══════════════════════════════════════════════════════════════════════
# Logging is not optional — it is how you know what worked.
# A run without tracking is a run you cannot reproduce or audit.


async def log_feature_run():
    """Log the feature engineering results to ExperimentTracker."""

    # Quick profile for quality metrics
    explorer = DataExplorer()
    profile = await explorer.profile(features)

    # Log the run using context manager pattern
    async with tracker.run(experiment_id, run_name="icu_clinical_features_v1") as run:
        await run.log_params(
            {
                "source_tables": "patients,admissions,vitals,medications,labs",
                "temporal_filter": "point_in_time",
                "vital_aggregations": "mean,std,min,max,trend,count",
                "medication_flags": "vasopressors,antibiotics",
                "derived_features": "abnormal_lab_ratio,medication_intensity",
            }
        )
        await run.log_metrics(
            {
                "n_features": float(len(features.columns)),
                "n_samples": float(features.height),
                "null_rate": sum(features[c].null_count() for c in features.columns)
                / (features.height * len(features.columns)),
                "n_alerts": float(len(profile.alerts)),
            }
        )
        await run.set_tag("domain", "clinical")
        run_id = run.id if hasattr(run, "id") else "logged"

    print(f"\n=== Experiment Run Logged ===")
    print(f"Run ID: {run_id}")
    print(f"Features: {len(features.columns)}, Samples: {features.height}")

    # List all runs in the experiment
    runs = await tracker.list_runs(experiment_id)
    print(f"Total runs in experiment: {len(runs)}")

    return run_id


run_id = asyncio.run(log_feature_run())

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert run_id is not None, "Run ID should be returned by ExperimentTracker"
print("\n✓ Checkpoint 6 passed — experiment run logged to ExperimentTracker\n")

# Clean up
asyncio.run(conn.close())


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print("""
  ✓ Multi-table joins with temporal correctness (no future leakage)
  ✓ Aggregating irregular time-series into per-admission statistics
  ✓ Domain-driven feature engineering (clinical flags from drug names)
  ✓ Derived features that encode complex relationships (lab ratio, intensity)
  ✓ FeatureSchema: type-safe, self-documenting feature contracts
  ✓ ExperimentTracker: reproducible, auditable feature engineering runs

  KEY INSIGHT: Data quality > model complexity. A clean, well-engineered
  feature set on a linear model outperforms a raw-data deep learning model.
  The features you built here (abnormal_lab_ratio, medication_intensity,
  vital_count) encode years of clinical knowledge in 3 columns.

  NEXT: Exercise 2 explores the bias-variance tradeoff — why adding more
  features or complexity doesn't always improve predictions, and how
  L1/L2 regularisation controls model complexity on the credit scoring data.
""")

print(
    "\n✓ Exercise 1 complete — healthcare feature engineering with temporal correctness"
)
print(
    "  ExperimentTracker is now tracking. Exercises 1 and 2 share this experiment history."
)
