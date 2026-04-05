# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT2 — Exercise 1: Healthcare Feature Engineering
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Engineer clinical features from messy ICU data using
#   kailash-ml FeatureEngineer. Track all work with ExperimentTracker
#   from the first exercise — building cumulative experiment history.
#
# TASKS:
#   1. Load and inspect messy ICU data (irregular vitals, multi-table)
#   2. Create ExperimentTracker experiment (used across all M2 exercises)
#   3. Handle temporal features with point-in-time correctness
#   4. Engineer clinical features (rolling vitals, medication interactions)
#   5. Validate features with FeatureSchema
#   6. Log feature engineering run to ExperimentTracker
#
# DATA QUALITY:
#   - Irregular time-series (vitals recorded at different frequencies)
#   - Multi-table joins (patients, admissions, vitals, medications, labs)
#   - Clinical missing patterns (not MCAR — sicker patients get more tests)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureEngineer, DataExplorer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.types import FeatureSchema, FeatureField

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()

patients = loader.load("ascent02", "icu_patients.parquet")
admissions = loader.load("ascent02", "icu_admissions.parquet")
vitals = loader.load("ascent02", "icu_vitals.parquet")
medications = loader.load("ascent02", "icu_medications.parquet")
labs = loader.load("ascent02", "icu_labs.parquet")

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

# Vitals are recorded at irregular intervals
print("\n=== Vital Signs Sample (one patient) ===")
sample_patient = vitals["patient_id"].unique()[0]
patient_vitals = vitals.filter(pl.col("patient_id") == sample_patient).sort(
    "recorded_at"
)
print(patient_vitals.head(20))

# Check recording frequency
if patient_vitals.height > 1:
    time_diffs = patient_vitals.with_columns(
        (pl.col("recorded_at").diff()).alias("time_gap")
    )
    print(f"\nTime gaps between readings:")
    print(time_diffs.select("vital_name", "time_gap").head(10))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Set up ExperimentTracker (persists across all M2 exercises)
# ══════════════════════════════════════════════════════════════════════


async def setup_tracking():
    """Initialize ExperimentTracker for Module 2."""
    conn = ConnectionManager("sqlite:///ascent02_experiments.db")
    await conn.initialize()

    tracker = ExperimentTracker(conn)
    await tracker.initialize()

    # TODO: create the Module 2 experiment with name, description, and tags
    experiment_id = await tracker.create_experiment(
        name=____,  # Hint: use "ascent02_healthcare_features"
        description=____,  # Hint: brief description of the M2 feature engineering work
        tags=____,  # Hint: list of strings, e.g. ["ascent02", "healthcare", "feature-engineering"]
    )
    print(f"\nExperiment created: {experiment_id}")

    return conn, tracker, experiment_id


conn, tracker, experiment_id = asyncio.run(setup_tracking())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Temporal features with point-in-time correctness
# ══════════════════════════════════════════════════════════════════════
# CRITICAL: Features must only use data available BEFORE the prediction
# time. Using future data (leakage) inflates validation metrics but
# fails catastrophically in production.

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
    # TODO: apply point-in-time filter — keep only vitals during THIS admission
    .filter(
        (pl.col("recorded_at") >= ____)  # Hint: compare with pl.col("admit_time")
        & (pl.col("recorded_at") <= ____)  # Hint: compare with pl.col("discharge_time")
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
        # TODO: compute the trend as the difference between the last and first reading
        (____).alias(
            f"{vital}_trend"
        ),  # Hint: pl.col("value").last() - pl.col("value").first()
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


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Engineer clinical features (medications, labs, interactions)
# ══════════════════════════════════════════════════════════════════════

# Medication features — count of distinct medications, specific drug flags
med_features = (
    medications.join(
        admissions.select("patient_id", "admission_id", "admit_time", "discharge_time"),
        on="patient_id",
        how="inner",
    )
    # TODO: apply point-in-time filter for medications
    .filter(
        (pl.col("administered_at") >= ____)  # Hint: compare with pl.col("admit_time")
        & (
            pl.col("administered_at") <= ____
        )  # Hint: compare with pl.col("discharge_time")
    )
    .group_by("admission_id")
    .agg(
        pl.col("medication_name").n_unique().alias("n_unique_medications"),
        pl.col("medication_name").count().alias("n_medication_doses"),
        # TODO: flag high-risk medications using a regex match (vasopressors indicate hemodynamic instability)
        pl.col("medication_name")
        .str.contains(
            ____
        )  # Hint: regex pattern "(?i)vasopressor|norepinephrine|dopamine"
        .any()
        .alias("received_vasopressors"),
        # TODO: flag antibiotic medications using a regex match
        pl.col("medication_name")
        .str.contains(____)  # Hint: regex pattern "(?i)antibiotic|vancomycin|meropenem"
        .any()
        .alias("received_antibiotics"),
    )
)

features = features.join(med_features, on="admission_id", how="left")

# Lab features — most recent lab values and abnormal counts
lab_features = (
    labs.join(
        admissions.select("patient_id", "admission_id", "admit_time", "discharge_time"),
        on="patient_id",
        how="inner",
    )
    # TODO: apply point-in-time filter for lab results
    .filter(
        (pl.col("collected_at") >= ____)  # Hint: compare with pl.col("admit_time")
        & (
            pl.col("collected_at") <= ____
        )  # Hint: compare with pl.col("discharge_time")
    )
    .group_by("admission_id")
    .agg(
        pl.col("lab_name").n_unique().alias("n_unique_labs"),
        pl.col("value").count().alias("n_lab_results"),
        # Abnormal results (flag=True in source data)
        pl.col("abnormal_flag").sum().alias("n_abnormal_labs"),
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
        pl.col("n_medication_doses") / pl.col("length_of_stay_days").clip(lower_bound=1)
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


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Validate features with FeatureSchema
# ══════════════════════════════════════════════════════════════════════

# Define the expected schema for our engineered features
# TODO: define a FeatureSchema named "icu_clinical_features_v1"
icu_schema = FeatureSchema(
    name=____,  # Hint: "icu_clinical_features_v1"
    features=[
        # TODO: define each FeatureField with name, dtype, nullable, and description
        FeatureField(
            name="age",
            dtype=____,  # Hint: "float64"
            nullable=____,  # Hint: False — this field is required
            description="Patient age at admission",
        ),
        FeatureField(
            name="length_of_stay_days",
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
        # TODO: define the vasopressor flag field
        FeatureField(
            name="received_vasopressors",
            dtype=____,  # Hint: "bool"
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
    # TODO: set the entity ID column and timestamp column for this schema
    entity_id_column=____,  # Hint: "patient_id"
    timestamp_column=____,  # Hint: "admit_time"
    version=1,
)

print(f"\n=== FeatureSchema: {icu_schema.name} ===")
print(f"Entity ID: {icu_schema.entity_id_column}")
print(f"Timestamp: {icu_schema.timestamp_column}")
for f in icu_schema.features:
    print(
        f"  {f.name}: {f.dtype} ({'nullable' if f.nullable else 'required'}) — {f.description}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Log to ExperimentTracker
# ══════════════════════════════════════════════════════════════════════


async def log_feature_run():
    """Log the feature engineering results to ExperimentTracker."""

    # Quick profile for quality metrics
    explorer = DataExplorer()
    profile = await explorer.profile(features)

    # TODO: use the ExperimentTracker context manager to log this run
    # Hint: use `async with tracker.run(experiment_id, run_name=...) as run:`
    async with tracker.run(
        ____, run_name=____
    ) as run:  # Hint: pass experiment_id and "icu_clinical_features_v1"
        # TODO: log the feature engineering parameters
        await run.log_params(
            {
                "source_tables": "patients,admissions,vitals,medications,labs",
                "temporal_filter": "point_in_time",
                "vital_aggregations": "mean,std,min,max,trend,count",
                "medication_flags": "vasopressors,antibiotics",
                "derived_features": "abnormal_lab_ratio,medication_intensity",
            }
        )
        # TODO: log metrics: n_features, n_samples, null_rate, n_alerts
        await run.log_metrics(
            {
                "n_features": ____,  # Hint: float(len(features.columns))
                "n_samples": ____,  # Hint: float(features.height)
                "null_rate": sum(features[c].null_count() for c in features.columns)
                / (features.height * len(features.columns)),
                "n_alerts": ____,  # Hint: float(len(profile.alerts))
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

# Clean up
asyncio.run(conn.close())

print(
    "\n✓ Exercise 1 complete — healthcare feature engineering with temporal correctness"
)
print(
    "  ExperimentTracker is now tracking. All subsequent M2 exercises add to this experiment."
)
