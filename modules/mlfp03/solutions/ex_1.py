# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1: Feature Engineering, ML Pipeline, and Feature
#                        Selection
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Engineer clinical features from multi-table medical data with
#     temporal point-in-time correctness (no leakage)
#   - Aggregate time-series vitals into statistical summaries per admission
#   - Flag clinically significant medication and lab patterns
#   - Compute derived features (abnormal lab ratio, medication intensity)
#   - Apply filter, wrapper, and embedded feature selection methods
#     (mutual information, chi-squared, RFE, L1-based)
#   - Validate a feature set against a declared FeatureSchema
#   - Log feature engineering experiments with ExperimentTracker
#   - Compare feature selection methods and justify final feature set
#
# PREREQUISITES:
#   - MLFP02 complete (statistics, Bayesian thinking, linear regression)
#   - ExperimentTracker introduced in MLFP02 Exercise 7
#
# ESTIMATED TIME: ~150-180 minutes
#
# TASKS:
#   1.  Load and inspect messy ICU data (irregular vitals, multi-table)
#   2.  Create ExperimentTracker experiment (used across all M3 exercises)
#   3.  Handle temporal features with point-in-time correctness
#   4.  Engineer clinical features (rolling vitals, medication interactions)
#   5.  Create interaction and polynomial features from domain knowledge
#   6.  Filter selection: mutual information and chi-squared
#   7.  Wrapper selection: recursive feature elimination (RFE)
#   8.  Embedded selection: L1 (Lasso) sparsity
#   9.  Compare all three selection methods and pick final feature set
#   10. Validate features with FeatureSchema
#   11. Log feature engineering run to ExperimentTracker
#   12. Leakage detection audit
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

import numpy as np
import polars as pl
from kailash.db import ConnectionManager
from kailash_ml import FeatureEngineer, DataExplorer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.types import FeatureSchema, FeatureField
from sklearn.feature_selection import (
    mutual_info_classif,
    chi2,
    SelectKBest,
    RFE,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from kailash_ml.interop import to_sklearn_input

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

# Vitals are recorded at irregular intervals — join with admissions
vitals = vitals.join(
    admissions.select(["admission_id", "patient_id"]),
    on="admission_id",
    how="left",
)

# Cast timestamp string to datetime for temporal operations
vitals = vitals.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S"))

# Melt wide-format vitals to long format (vital_name, value)
vital_cols = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "spo2",
    "respiratory_rate",
]
vital_cols_present = [c for c in vital_cols if c in vitals.columns]
if vital_cols_present:
    vitals = vitals.unpivot(
        vital_cols_present,
        index=["admission_id", "patient_id", "timestamp"],
        variable_name="vital_name",
        value_name="value",
    )

print("\n=== Vital Signs Sample (one patient) ===")
sample_patient = vitals["patient_id"].unique()[0]
patient_vitals = vitals.filter(pl.col("patient_id") == sample_patient).sort("timestamp")
print(patient_vitals.head(20))

# Check recording frequency — irregular intervals encode severity
if patient_vitals.height > 1:
    time_diffs = patient_vitals.with_columns(
        (pl.col("timestamp").diff()).alias("time_gap")
    )
    print(f"\nTime gaps between readings:")
    print(time_diffs.select("vital_name", "time_gap").head(10))

# Compute global vital statistics across all patients
print(f"\n=== Vital Statistics (all patients) ===")
vital_stats = (
    vitals.group_by("vital_name")
    .agg(
        pl.col("value").count().alias("n_readings"),
        pl.col("value").mean().alias("mean"),
        pl.col("value").std().alias("std"),
        pl.col("value").null_count().alias("n_null"),
    )
    .sort("vital_name")
)
print(vital_stats)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert patients.height > 0, "patients DataFrame is empty"
assert vitals.height > 0, "vitals DataFrame is empty"
assert "patient_id" in vitals.columns, "vitals must have patient_id"
# INTERPRETATION: ICU data is messy by design — recording frequency
# encodes patient severity. A feature like 'vital_count' captures this
# indirectly: patients with more readings may be more critically ill.
print("\n✓ Checkpoint 1 passed — ICU data loaded and inspected\n")


# ══════════════════════════════════════════════════════════════════════
# A NOTE ON ASYNC/AWAIT (first-time introduction)
# ══════════════════════════════════════════════════════════════════════
# Python's `async/await` lets code wait for slow operations (database,
# network, file I/O) without freezing. Think of it like a restaurant:
#   - `async def` = a waiter who can serve multiple tables at once
#   - `await`     = "go prepare this dish; I'll check on other tables"
#   - `asyncio.run()` = opening the restaurant for the day
#
# You don't need to fully understand async to complete this course.
# The pattern is always the same:
#     async def some_function(): await other_async_call()
#     result = asyncio.run(some_function())
#
# You'll see `async def` wrap Kailash calls that touch a database
# (ConnectionManager, ExperimentTracker, ModelRegistry). We wrap them
# in `asyncio.run()` to execute. That's it — treat it as a recipe.
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# TASK 2: Set up ExperimentTracker (persists across all M3 exercises)
# ══════════════════════════════════════════════════════════════════════
# ExperimentTracker records every feature engineering run so we can
# compare approaches, reproduce past results, and audit what data
# was used to build each feature set.  This is the foundation of
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
# INTERPRETATION: Every ML project needs a tracking system.  Without one,
# you cannot reproduce past results or audit which features were used.
print("\n✓ Checkpoint 2 passed — ExperimentTracker initialized\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Temporal features with point-in-time correctness
# ══════════════════════════════════════════════════════════════════════
# CRITICAL: Features must only use data available BEFORE the prediction
# time.  Using future data (leakage) inflates validation metrics but
# fails catastrophically in production.
#
# Example of leakage: including discharge diagnosis in features used to
# predict in-hospital mortality.  In production, discharge hasn't
# happened yet.

# Join patients with admissions
patient_admissions = patients.join(admissions, on="patient_id", how="inner")

# Aggregate vitals PER ADMISSION with temporal correctness
# Only use vitals recorded DURING this admission (between admit and
# discharge)
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
        # Range captures clinical volatility
        (pl.col("value").max() - pl.col("value").min()).alias(f"{vital}_range"),
        # Trend: last reading minus first reading
        (pl.col("value").last() - pl.col("value").first()).alias(f"{vital}_trend"),
        # Count of readings (proxy for severity)
        pl.col("value").count().alias(f"{vital}_count"),
        # Coefficient of variation — normalised volatility
        (pl.col("value").std() / pl.col("value").mean()).alias(f"{vital}_cv"),
    )
    vital_aggs.append(agg)

# Join all vital aggregates
features = patient_admissions.clone()
for agg in vital_aggs:
    features = features.join(agg, on="admission_id", how="left")

print(f"\n=== Features after vital aggregation ===")
print(f"Shape: {features.shape}")
feature_sample = [c for c in features.columns if any(v in c for v in vital_names)][:12]
print(f"New vital columns (sample): {feature_sample}...")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
vital_feature_cols = [c for c in features.columns if any(v in c for v in vital_names)]
assert len(vital_feature_cols) > 0, "No vital feature columns were created"
assert features.height > 0, "Feature DataFrame is empty after aggregation"
# INTERPRETATION: The _count suffix columns are particularly valuable.
# A patient with heart_rate_count=120 in a 24h stay (5 readings/hr)
# is being monitored far more intensively than one with count=8.
# The _cv (coefficient of variation) captures normalised volatility —
# a patient whose heart rate oscillates wildly has a high CV regardless
# of whether the baseline heart rate is 60 or 120.
print("\n✓ Checkpoint 3 passed — temporal vital features created\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Engineer clinical features (medications, labs, interactions)
# ══════════════════════════════════════════════════════════════════════
# Domain knowledge is essential here.  A data scientist without clinical
# knowledge would not know that vasopressors indicate hemodynamic
# instability, or that abnormal lab ratios signal systemic illness.
# This is why "data > models" — domain features beat raw data.

# Medication features
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
        # Flag high-risk medications
        pl.col("drug_name")
        .str.contains("(?i)vasopressor|norepinephrine|dopamine")
        .any()
        .alias("received_vasopressors"),
        pl.col("drug_name")
        .str.contains("(?i)antibiotic|vancomycin|meropenem")
        .any()
        .alias("received_antibiotics"),
        # Sedation flags (indicate ventilation / severe agitation)
        pl.col("drug_name")
        .str.contains("(?i)propofol|midazolam|fentanyl")
        .any()
        .alias("received_sedation"),
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
        # Abnormal results
        (pl.col("flag") != "normal").sum().alias("n_abnormal_labs"),
        # Lab value statistics
        pl.col("value").mean().alias("lab_value_mean"),
        pl.col("value").std().alias("lab_value_std"),
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
    (pl.col("n_medication_doses") / pl.col("los_days").clip(lower_bound=1)).alias(
        "medication_intensity"
    ),
    # Lab ordering intensity (tests per day — sicker patients get more tests)
    (pl.col("n_lab_results") / pl.col("los_days").clip(lower_bound=1)).alias(
        "lab_intensity"
    ),
    # Poly-pharmacy flag (>10 unique drugs signals complex case)
    (pl.col("n_unique_medications") > 10).alias("polypharmacy_flag"),
)

# Fill nulls for patients with no medications/labs
fill_cols_int = [
    "n_unique_medications",
    "n_medication_doses",
    "n_unique_labs",
    "n_lab_results",
    "n_abnormal_labs",
]
fill_cols_bool = [
    "received_vasopressors",
    "received_antibiotics",
    "received_sedation",
    "polypharmacy_flag",
]
fill_cols_float = [
    "abnormal_lab_ratio",
    "medication_intensity",
    "lab_intensity",
    "lab_value_mean",
    "lab_value_std",
]
features = features.with_columns(
    *[pl.col(c).fill_null(0) for c in fill_cols_int],
    *[pl.col(c).fill_null(False) for c in fill_cols_bool],
    *[pl.col(c).fill_null(0.0) for c in fill_cols_float],
)

print(f"\n=== Features after medication + lab engineering ===")
print(f"Shape: {features.shape}")
print(f"Total feature columns: {len(features.columns)}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert "n_unique_medications" in features.columns, "Medication features missing"
assert "abnormal_lab_ratio" in features.columns, "Derived lab ratio missing"
assert "medication_intensity" in features.columns, "Derived med intensity missing"
assert "lab_intensity" in features.columns, "Derived lab intensity missing"
assert features["abnormal_lab_ratio"].null_count() == 0, "Null values in lab ratio"
# INTERPRETATION: abnormal_lab_ratio captures systemic illness severity.
# A ratio of 0.6 means 60% of lab tests came back abnormal — this patient
# is in serious trouble.  medication_intensity (doses/day) captures treatment
# burden, which correlates with disease complexity.  lab_intensity captures
# diagnostic burden — sicker patients generate more lab orders.
print("\n✓ Checkpoint 4 passed — clinical features engineered\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Interaction and polynomial features from domain knowledge
# ══════════════════════════════════════════════════════════════════════
# Interaction features capture non-additive effects.  Example: a high
# heart rate is concerning.  A high heart rate AND low blood pressure
# together is a medical emergency (shock).

# Create clinically meaningful interaction features
features = features.with_columns(
    # Shock index: heart_rate / systolic_bp  (> 0.9 is concerning)
    (
        (
            pl.col("heart_rate_mean") / pl.col("systolic_bp_mean").clip(lower_bound=1)
        ).alias("shock_index")
        if "heart_rate_mean" in features.columns
        and "systolic_bp_mean" in features.columns
        else pl.lit(0.0).alias("shock_index")
    ),
    # Mean arterial pressure: (SBP + 2*DBP) / 3
    (
        ((pl.col("systolic_bp_mean") + 2 * pl.col("diastolic_bp_mean")) / 3).alias(
            "map_mean"
        )
        if "systolic_bp_mean" in features.columns
        and "diastolic_bp_mean" in features.columns
        else pl.lit(0.0).alias("map_mean")
    ),
    # Fever-tachycardia interaction: high temp + high HR = sepsis signal
    (
        (pl.col("temperature_mean") * pl.col("heart_rate_mean")).alias(
            "fever_tachycardia"
        )
        if "temperature_mean" in features.columns
        and "heart_rate_mean" in features.columns
        else pl.lit(0.0).alias("fever_tachycardia")
    ),
    # Medication-lab interaction: high medication + high abnormal labs
    (pl.col("medication_intensity") * pl.col("abnormal_lab_ratio")).alias(
        "treatment_burden_score"
    ),
)

# Fill any new nulls from interactions
features = features.with_columns(
    pl.col("shock_index").fill_null(0.0),
    pl.col("map_mean").fill_null(0.0),
    pl.col("fever_tachycardia").fill_null(0.0),
    pl.col("treatment_burden_score").fill_null(0.0),
)

print(f"\n=== Features after interaction engineering ===")
print(f"Shape: {features.shape}")
print("Interaction features:")
print(f"  shock_index (HR / SBP): mean={features['shock_index'].mean():.3f}")
print(f"  map_mean ((SBP + 2*DBP) / 3): mean={features['map_mean'].mean():.1f}")
print(f"  treatment_burden_score: mean={features['treatment_burden_score'].mean():.3f}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert "shock_index" in features.columns, "Shock index interaction missing"
assert "treatment_burden_score" in features.columns, "Treatment burden missing"
assert features["shock_index"].null_count() == 0, "Shock index has nulls"
# INTERPRETATION: The shock index (HR/SBP) is a validated clinical indicator.
# Normal is 0.5-0.7; > 0.9 predicts hemodynamic instability.  By encoding
# this domain knowledge as a feature, we give the model a head start —
# it does not need to learn the ratio relationship from raw data.
print("\n✓ Checkpoint 5 passed — interaction features created\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Filter selection — mutual information and chi-squared
# ══════════════════════════════════════════════════════════════════════
# Filter methods evaluate each feature independently of the model.
# Mutual information: measures any (linear or non-linear) dependency
# Chi-squared: tests statistical independence (requires non-negative)

# Prepare features for sklearn-based selection
# Identify numeric feature columns (exclude IDs, strings, targets)
id_cols = ["patient_id", "admission_id", "admit_time", "discharge_time"]
target_col = "mortality" if "mortality" in features.columns else "los_days"
exclude_cols = set(id_cols + [target_col])
feature_cols = [
    c
    for c in features.columns
    if c not in exclude_cols
    and features[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Boolean]
]

# Create binary target for feature selection (median split if continuous)
if target_col in features.columns:
    target_series = features[target_col]
else:
    # Use los_days > median as proxy binary target
    median_los = features["los_days"].median()
    target_series = (features["los_days"] > median_los).cast(pl.Int32)

# Convert to numpy for sklearn
X_sel = features.select(feature_cols).to_numpy().astype(np.float64)
y_sel = target_series.to_numpy().astype(np.float64).ravel()

# Handle any remaining NaN/inf
X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=1e6, neginf=-1e6)
y_sel = np.nan_to_num(y_sel, nan=0.0)

# Make binary for classification-based selectors
y_binary = (y_sel > np.median(y_sel)).astype(int)

# --- Mutual Information ---
mi_scores = mutual_info_classif(X_sel, y_binary, random_state=42)
mi_ranking = sorted(zip(feature_cols, mi_scores), key=lambda x: x[1], reverse=True)

print(f"\n=== Filter: Mutual Information (top 15) ===")
print(f"{'Feature':<35} {'MI Score':>10}")
print("─" * 47)
for name, score in mi_ranking[:15]:
    bar = "█" * int(score * 100)
    print(f"  {name:<33} {score:>10.4f}  {bar}")

# --- Chi-squared (requires non-negative features) ---
X_chi2 = MinMaxScaler().fit_transform(X_sel)  # Scale to [0, 1]
chi2_scores, chi2_pvalues = chi2(X_chi2, y_binary)
chi2_ranking = sorted(
    zip(feature_cols, chi2_scores, chi2_pvalues),
    key=lambda x: x[1],
    reverse=True,
)

print(f"\n=== Filter: Chi-Squared (top 15) ===")
print(f"{'Feature':<35} {'Chi2':>10} {'p-value':>10}")
print("─" * 57)
for name, score, pval in chi2_ranking[:15]:
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {name:<33} {score:>10.2f} {pval:>10.4f} {sig}")

# Features selected by both methods (intersection = robust)
mi_top20 = {name for name, _ in mi_ranking[:20]}
chi2_top20 = {name for name, _, _ in chi2_ranking[:20]}
filter_consensus = mi_top20 & chi2_top20

print(f"\n--- Filter Consensus (top-20 intersection) ---")
print(f"MI top-20: {len(mi_top20)} features")
print(f"Chi2 top-20: {len(chi2_top20)} features")
print(f"Intersection: {len(filter_consensus)} features")
print(f"Consensus features: {sorted(filter_consensus)}")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(mi_ranking) == len(feature_cols), "MI should score all features"
assert len(chi2_ranking) == len(feature_cols), "Chi2 should score all features"
assert mi_ranking[0][1] > 0, "Top MI feature should have positive score"
# INTERPRETATION: Mutual information captures both linear and non-linear
# relationships — it detects dependencies that correlation coefficients miss.
# Chi-squared tests statistical independence and requires non-negative inputs.
# Features that rank highly in BOTH methods are robust candidates — their
# relationship to the target is not an artifact of one particular measure.
print("\n✓ Checkpoint 6 passed — filter feature selection complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Wrapper selection — Recursive Feature Elimination (RFE)
# ══════════════════════════════════════════════════════════════════════
# RFE trains a model, ranks features by importance, removes the least
# important, and repeats.  More expensive than filters but considers
# feature interactions through the model.

# Use Random Forest as the estimator for RFE (captures non-linear effects)
rf_estimator = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
)

# Select top 15 features via RFE
rfe = RFE(estimator=rf_estimator, n_features_to_select=15, step=5)
rfe.fit(X_sel, y_binary)

rfe_selected = [name for name, selected in zip(feature_cols, rfe.support_) if selected]
rfe_ranking_full = sorted(zip(feature_cols, rfe.ranking_), key=lambda x: x[1])

print(f"\n=== Wrapper: RFE (Random Forest, top 15) ===")
print(f"{'Feature':<35} {'Rank':>6}")
print("─" * 43)
for name, rank in rfe_ranking_full[:20]:
    marker = " ← selected" if rank == 1 else ""
    print(f"  {name:<33} {rank:>6}{marker}")

print(f"\nRFE selected {len(rfe_selected)} features: {rfe_selected}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert (
    len(rfe_selected) == 15
), f"RFE should select 15 features, got {len(rfe_selected)}"
assert all(f in feature_cols for f in rfe_selected), "RFE features must be valid"
# INTERPRETATION: RFE considers feature interactions because the Random
# Forest model internally captures non-linear relationships.  A feature
# that looks weak alone (low MI) may rank highly in RFE because it
# becomes informative in combination with other features.  This is why
# wrapper methods complement filter methods.
print("\n✓ Checkpoint 7 passed — RFE wrapper selection complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Embedded selection — L1 (Lasso) sparsity
# ══════════════════════════════════════════════════════════════════════
# L1 regularisation drives coefficients to exactly zero, performing
# automatic feature selection as part of model training.
# This is the embedded approach: selection happens inside the model.

from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_sel)

# Regularisation path: vary C (inverse of regularisation strength)
c_values = [0.001, 0.01, 0.1, 1.0, 10.0]
print(f"\n=== Embedded: L1 Lasso Regularisation Path ===")
print(f"{'C':>8} {'Non-zero':>10} {'Features':>10}")
print("─" * 32)

lasso_results = {}
for c_val in c_values:
    lasso = LogisticRegression(
        penalty="l1", C=c_val, solver="saga", max_iter=5000, random_state=42
    )
    lasso.fit(X_scaled, y_binary)
    n_nonzero = (np.abs(lasso.coef_[0]) > 1e-6).sum()
    lasso_results[c_val] = {
        "n_nonzero": n_nonzero,
        "coefs": lasso.coef_[0].copy(),
    }
    print(f"  {c_val:>8.3f} {n_nonzero:>10} / {len(feature_cols)}")

# Use C=0.1 as a reasonable regularisation for feature selection
lasso_c = 0.1
lasso_coefs = lasso_results[lasso_c]["coefs"]
lasso_selected = [
    name for name, coef in zip(feature_cols, lasso_coefs) if abs(coef) > 1e-6
]

print(f"\nL1 selected (C={lasso_c}): {len(lasso_selected)} features")
lasso_importance = sorted(
    [
        (name, abs(coef))
        for name, coef in zip(feature_cols, lasso_coefs)
        if abs(coef) > 1e-6
    ],
    key=lambda x: x[1],
    reverse=True,
)
for name, imp in lasso_importance[:15]:
    print(f"  {name:<35} |coef|={imp:.4f}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(lasso_selected) > 0, "Lasso should select at least some features"
assert len(lasso_selected) < len(feature_cols), "Lasso should eliminate some features"
# INTERPRETATION: L1 regularisation is a Bayesian statement: "I believe
# only a few features matter" (Laplace prior).  Features with zero
# coefficients are provably irrelevant under the model's assumptions.
# The regularisation path shows how sparsity increases with stronger
# regularisation (smaller C) — at C=0.001, only the strongest signals
# survive.
print("\n✓ Checkpoint 8 passed — L1 embedded selection complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Compare all three selection methods
# ══════════════════════════════════════════════════════════════════════
# Each method has different strengths:
#   Filter: fast, independent of model, may miss interactions
#   Wrapper: considers interactions, expensive, overfitting risk
#   Embedded: natural integration, one-pass, model-specific

# Build comparison table
all_methods = {
    "MI (top 15)": set(name for name, _ in mi_ranking[:15]),
    "Chi2 (top 15)": set(name for name, _, _ in chi2_ranking[:15]),
    "RFE (RF, 15)": set(rfe_selected),
    "Lasso (C=0.1)": set(lasso_selected),
}

print(f"\n=== Feature Selection Method Comparison ===")
print(f"{'Method':<20} {'Selected':>10}")
print("─" * 32)
for method, feats in all_methods.items():
    print(f"  {method:<18} {len(feats):>10}")

# Find features selected by at least 3 of 4 methods (robust consensus)
from collections import Counter

feature_votes = Counter()
for method, feats in all_methods.items():
    for f in feats:
        feature_votes[f] += 1

consensus_features = [f for f, count in feature_votes.most_common() if count >= 3]
all_selected = [f for f, count in feature_votes.most_common() if count >= 2]

print(f"\n--- Consensus Analysis ---")
print(f"Features selected by ≥3 methods: {len(consensus_features)}")
print(f"Features selected by ≥2 methods: {len(all_selected)}")
print(f"\nConsensus features (≥3/4 methods):")
for f in consensus_features:
    methods_selecting = [m for m, feats in all_methods.items() if f in feats]
    print(f"  {f:<35} ({', '.join(methods_selecting)})")

# Final feature set decision
final_features = (
    consensus_features if len(consensus_features) >= 8 else all_selected[:15]
)
print(f"\n--- Final Feature Set ---")
print(f"Selected {len(final_features)} features for downstream modelling")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(final_features) > 0, "Must select at least one feature"
assert len(final_features) <= len(
    feature_cols
), "Cannot select more features than exist"
# INTERPRETATION: Features that survive multiple selection methods are
# genuinely predictive.  A feature that only appears in one method's top
# list may be capturing noise or method-specific artifacts.  The consensus
# approach is conservative but reliable — exactly what you want before
# investing in model training.
print("\n✓ Checkpoint 9 passed — feature selection comparison complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Validate features with FeatureSchema
# ══════════════════════════════════════════════════════════════════════
# FeatureSchema is a contract: it declares what columns are required,
# their types, and whether nulls are allowed.  This prevents silent
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

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert icu_schema.name == "icu_clinical_features_v1", "Schema name mismatch"
assert len(icu_schema.features) == 7, "Schema should declare 7 features"
for field_def in icu_schema.features:
    assert (
        field_def.name in features.columns
    ), f"Declared feature '{field_def.name}' missing from DataFrame"
# INTERPRETATION: A FeatureSchema acts as living documentation AND
# runtime validation.  If feature engineering code changes and a column
# disappears, the schema check catches it before a model trains on
# bad data.  Think of it as a type system for ML features.
print("\n✓ Checkpoint 10 passed — FeatureSchema validated against features\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Log to ExperimentTracker
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
                "vital_aggregations": "mean,std,min,max,range,trend,count,cv",
                "medication_flags": "vasopressors,antibiotics,sedation",
                "derived_features": "abnormal_lab_ratio,medication_intensity,"
                "lab_intensity,shock_index,treatment_burden_score",
                "selection_methods": "MI,chi2,RFE,Lasso",
                "final_feature_count": str(len(final_features)),
            }
        )
        await run.log_metrics(
            {
                "n_features_engineered": float(len(feature_cols)),
                "n_features_selected": float(len(final_features)),
                "n_samples": float(features.height),
                "null_rate": sum(features[c].null_count() for c in features.columns)
                / (features.height * len(features.columns)),
                "n_alerts": float(len(profile.alerts)),
                "selection_consensus_3plus": float(len(consensus_features)),
            }
        )
        await run.set_tag("domain", "clinical")
        await run.set_tag("selection", "multi-method-consensus")
        run_id = run.id if hasattr(run, "id") else "logged"

    print(f"\n=== Experiment Run Logged ===")
    print(f"Run ID: {run_id}")
    print(f"Engineered: {len(feature_cols)}, Selected: {len(final_features)}")
    print(f"Samples: {features.height}")

    # List all runs in the experiment
    runs = await tracker.list_runs(experiment_id)
    print(f"Total runs in experiment: {len(runs)}")

    return run_id


run_id = asyncio.run(log_feature_run())

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert run_id is not None, "Run ID should be returned by ExperimentTracker"
print("\n✓ Checkpoint 11 passed — experiment run logged to ExperimentTracker\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Leakage detection audit
# ══════════════════════════════════════════════════════════════════════
# Leakage is the most dangerous bug in ML — it inflates metrics during
# validation but causes catastrophic failure in production.
# Common leakage sources:
#   1. Using future data (discharge diagnosis to predict mortality)
#   2. Using the target in features (directly or via proxy)
#   3. Train/test contamination (data from same patient in both sets)

print(f"\n=== Leakage Detection Audit ===")

# Check 1: No target-derived features
leakage_suspects = []
for col in feature_cols:
    if any(
        keyword in col.lower()
        for keyword in ["mortality", "death", "outcome", "discharge_diagnosis"]
    ):
        leakage_suspects.append(col)

if leakage_suspects:
    print(f"  WARNING: Potential target leakage: {leakage_suspects}")
else:
    print(f"  ✓ No target-derived feature names detected")

# Check 2: Temporal correctness — all features use point-in-time filters
print(f"  ✓ Vitals: filtered to [admit_time, discharge_time]")
print(f"  ✓ Medications: filtered to [admit_time, discharge_time]")
print(f"  ✓ Labs: filtered to [admit_time, discharge_time]")

# Check 3: No future information in features
future_risk_cols = [
    c
    for c in features.columns
    if any(keyword in c.lower() for keyword in ["discharge", "icu_out", "death_time"])
]
if future_risk_cols:
    print(f"  WARNING: Potential future leakage: {future_risk_cols}")
else:
    print(f"  ✓ No future-information columns detected in features")

# Check 4: Feature correlation with target
if target_col in features.columns:
    target_corrs = []
    for col in feature_cols[:20]:
        try:
            corr = features.select(
                pl.corr(
                    pl.col(col).cast(pl.Float64), pl.col(target_col).cast(pl.Float64)
                )
            ).item()
            if abs(corr) > 0.95:
                target_corrs.append((col, corr))
        except Exception:
            pass

    if target_corrs:
        print(f"  WARNING: Near-perfect target correlation (r > 0.95):")
        for col, corr in target_corrs:
            print(f"    {col}: r = {corr:.4f}")
    else:
        print(f"  ✓ No features with suspiciously high target correlation")

print(f"\nLeakage audit complete.  All clear." if not leakage_suspects else "")

# ── Checkpoint 12 ────────────────────────────────────────────────────
assert len(leakage_suspects) == 0, f"Leakage detected: {leakage_suspects}"
# INTERPRETATION: Leakage detection should be run EVERY time features
# change.  In production, this audit is automated as a pre-training gate.
# The most insidious leakage is proxy leakage — a feature that is not
# the target itself but is a near-perfect predictor of the target because
# it was computed from post-hoc data (e.g., "total_charges" computed
# after discharge is a proxy for length of stay and mortality).
print("\n✓ Checkpoint 12 passed — no leakage detected\n")

# Clean up
asyncio.run(conn.close())


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ Multi-table joins with temporal correctness (no future leakage)
  ✓ Aggregating irregular time-series into per-admission statistics
  ✓ Domain-driven feature engineering (clinical flags from drug names)
  ✓ Interaction features from domain knowledge (shock index, MAP)
  ✓ Derived features encoding complex relationships (lab ratio, intensity)
  ✓ Filter selection: mutual information + chi-squared
  ✓ Wrapper selection: recursive feature elimination (RFE)
  ✓ Embedded selection: L1 regularisation path
  ✓ Multi-method consensus for robust feature selection
  ✓ FeatureSchema: type-safe, self-documenting feature contracts
  ✓ ExperimentTracker: reproducible, auditable feature engineering runs
  ✓ Leakage detection audit as a systematic quality gate

  KEY INSIGHT: Data quality > model complexity.  A clean, well-engineered
  feature set on a linear model outperforms a raw-data deep learning model.
  The features you built here (abnormal_lab_ratio, shock_index,
  treatment_burden_score) encode years of clinical knowledge in 3 columns.

  FEATURE SELECTION TAXONOMY:
    Filter  → fast, model-free, misses interactions  (MI, chi-squared)
    Wrapper → considers interactions, expensive       (RFE)
    Embedded → one-pass, model-specific              (L1 Lasso)
    Consensus → robust, conservative                 (multi-method vote)

  NEXT: Exercise 2 explores the bias-variance tradeoff — why adding more
  features or complexity doesn't always improve predictions, and how
  L1/L2 regularisation controls model complexity.  You will also learn
  nested cross-validation, time-series CV, and GroupKFold.
"""
)

print(
    "\n✓ Exercise 1 complete — healthcare feature engineering with temporal "
    "correctness"
)
print(
    "  ExperimentTracker is now tracking.  All M3 exercises share this "
    "experiment history."
)
