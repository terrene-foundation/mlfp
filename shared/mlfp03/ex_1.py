# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 1 — Feature Engineering
and Feature Selection on ICU data.

Contains:
    - ICU multi-table loading with temporal casts
    - Point-in-time feature builders (vitals, medications, labs)
    - ExperimentTracker / ConnectionManager setup
    - Shared prep helpers for feature-selection methods
    - Plotting helpers for feature rankings

Technique-specific code (mutual_info, RFE, Lasso paths, schema
validation) does NOT belong here — it lives in the per-technique files.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from dotenv import load_dotenv

from kailash.db import ConnectionManager
from kailash_ml import DataExplorer, ExperimentTracker

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment


# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

setup_environment()
load_dotenv()

OUTPUT_DIR = Path("outputs") / "mlfp03_ex1_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_DB = "sqlite:///mlfp03_experiments.db"
EXPERIMENT_NAME = "mlfp03_healthcare_features"

_DT_FMT = "%Y-%m-%d %H:%M:%S"

# PREDICTION TIME. The model scores each admission PREDICTION_HOURS after
# ICU admission ("will this be a long stay?"). Every event-based feature
# may only use records with admit_time <= event_time <= prediction_cutoff.
# Anything later — and anything that depends on the discharge time — does
# not exist yet at prediction time.
PREDICTION_HOURS = 24

# The target is derived from los_days (known only at discharge). These
# columns are TARGET SOURCES or post-outcome information and must never be
# features. ``audit_feature_list`` enforces this.
ID_COLUMNS: frozenset[str] = frozenset(
    {"patient_id", "admission_id", "admit_time", "prediction_cutoff"}
)
TARGET_SOURCE_COLUMNS: frozenset[str] = frozenset({"los_days", "discharge_time"})
TARGET_NAME = "long_stay"

VITAL_COLS: list[str] = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "spo2",
    "respiratory_rate",
]


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — ICU multi-table
# ════════════════════════════════════════════════════════════════════════


def load_icu_tables() -> dict[str, pl.DataFrame]:
    """Load all five ICU tables and cast timestamp columns to datetime.

    Returns a dict with keys: patients, admissions, vitals (long format),
    medications, labs.

    Vitals are returned in LONG format (columns: admission_id, patient_id,
    timestamp, vital_name, value) — the monolithic exercise unpivots them
    inline, but every technique file wants the long form.
    """
    loader = MLFPDataLoader()
    patients = loader.load("mlfp02", "icu_patients.parquet")
    admissions = loader.load("mlfp02", "icu_admissions.parquet")
    vitals = loader.load("mlfp02", "icu_vitals.parquet")
    medications = loader.load("mlfp02", "icu_medications.parquet")
    labs = loader.load("mlfp02", "icu_labs.parquet")

    # Cast timestamps — polars reads them as strings from the parquet
    admissions = admissions.with_columns(
        pl.col("admit_time").str.to_datetime(_DT_FMT),
        pl.col("discharge_time").str.to_datetime(_DT_FMT),
    ).with_columns(
        (pl.col("admit_time") + pl.duration(hours=PREDICTION_HOURS)).alias(
            "prediction_cutoff"
        )
    )
    medications = medications.with_columns(
        pl.col("start_time").str.to_datetime(_DT_FMT),
        pl.col("end_time").str.to_datetime(_DT_FMT),
    )
    if "timestamp" in labs.columns and labs["timestamp"].dtype == pl.String:
        labs = labs.with_columns(pl.col("timestamp").str.to_datetime(_DT_FMT))

    # Vitals: attach patient_id via admissions join, cast timestamp, unpivot
    vitals = vitals.join(
        admissions.select(["admission_id", "patient_id"]),
        on="admission_id",
        how="left",
    ).with_columns(pl.col("timestamp").str.to_datetime(_DT_FMT))

    present = [c for c in VITAL_COLS if c in vitals.columns]
    if present:
        vitals = vitals.unpivot(
            present,
            index=["admission_id", "patient_id", "timestamp"],
            variable_name="vital_name",
            value_name="value",
        )

    return {
        "patients": patients,
        "admissions": admissions,
        "vitals": vitals,
        "medications": medications,
        "labs": labs,
    }


# ════════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS — point-in-time aggregates
# ════════════════════════════════════════════════════════════════════════


def events_in_prediction_window(
    events: pl.DataFrame, admissions: pl.DataFrame, time_col: str
) -> pl.DataFrame:
    """Keep only events recorded between admit_time and prediction_cutoff.

    This single filter is used by EVERY event-based feature builder, so the
    point-in-time rule lives in one place and can be audited.
    """
    return events.join(
        admissions.select("admission_id", "admit_time", "prediction_cutoff"),
        on="admission_id",
        how="inner",
    ).filter(
        (pl.col(time_col) >= pl.col("admit_time"))
        & (pl.col(time_col) <= pl.col("prediction_cutoff"))
    )


def prediction_window_report(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Audit the event data actually available at prediction time.

    For each event table: how many events / admissions fall inside the
    window, and the latest event offset (hours after admission) used.
    Computed from the same filter the builders use — not asserted.
    """
    admissions = tables["admissions"]
    rows = []
    for name, time_col in (
        ("vitals", "timestamp"),
        ("medications", "start_time"),
        ("labs", "timestamp"),
    ):
        kept = events_in_prediction_window(tables[name], admissions, time_col)
        offsets = (kept[time_col] - kept["admit_time"]).dt.total_seconds() / 3600
        rows.append(
            {
                "table": name,
                "events_total": tables[name].height,
                "events_in_window": kept.height,
                "admissions_with_events": kept["admission_id"].n_unique(),
                "admissions_total": admissions.height,
                "max_offset_hours": float(offsets.max()) if kept.height else 0.0,
            }
        )
    return pl.DataFrame(rows)


def build_vital_features(
    vitals: pl.DataFrame, admissions: pl.DataFrame
) -> pl.DataFrame:
    """Aggregate long-format vitals per admission with temporal correctness.

    Only uses vital readings recorded in the first PREDICTION_HOURS of each
    admission. Returns one row per admission with columns:
        {vital}_{mean,std,min,max,range,trend,count,cv}
    """
    filtered = events_in_prediction_window(vitals, admissions, "timestamp").sort(
        "admission_id", "timestamp"
    )

    names = sorted(filtered["vital_name"].unique().to_list())
    if not names:
        return admissions.select("admission_id")
    aggs: list[pl.DataFrame] = []
    for vital in names:
        agg = (
            filtered.filter(pl.col("vital_name") == vital)
            .group_by("admission_id")
            .agg(
                pl.col("value").mean().alias(f"{vital}_mean"),
                pl.col("value").std().alias(f"{vital}_std"),
                pl.col("value").min().alias(f"{vital}_min"),
                pl.col("value").max().alias(f"{vital}_max"),
                (pl.col("value").max() - pl.col("value").min()).alias(f"{vital}_range"),
                (pl.col("value").last() - pl.col("value").first()).alias(
                    f"{vital}_trend"
                ),
                pl.col("value").count().alias(f"{vital}_count"),
                (pl.col("value").std() / pl.col("value").mean()).alias(f"{vital}_cv"),
            )
        )
        aggs.append(agg)

    # Merge vital aggregates via full-outer join with coalesced key.
    out = aggs[0]
    for a in aggs[1:]:
        out = out.join(a, on="admission_id", how="full", coalesce=True)
    return out


def build_medication_features(
    medications: pl.DataFrame, admissions: pl.DataFrame
) -> pl.DataFrame:
    """Flag high-risk medications and count distinct drugs per admission
    (medications started within the first PREDICTION_HOURS only)."""
    return (
        events_in_prediction_window(medications, admissions, "start_time")
        .group_by("admission_id")
        .agg(
            pl.col("drug_name").n_unique().alias("n_unique_medications"),
            pl.col("drug_name").count().alias("n_medication_doses"),
            pl.col("drug_name")
            .str.contains("(?i)vasopressor|norepinephrine|dopamine")
            .any()
            .alias("received_vasopressors"),
            pl.col("drug_name")
            .str.contains("(?i)antibiotic|vancomycin|meropenem")
            .any()
            .alias("received_antibiotics"),
            pl.col("drug_name")
            .str.contains("(?i)propofol|midazolam|fentanyl")
            .any()
            .alias("received_sedation"),
        )
    )


def build_lab_features(labs: pl.DataFrame, admissions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate lab results per admission with abnormal-flag counts
    (results within the first PREDICTION_HOURS only)."""
    return (
        events_in_prediction_window(labs, admissions, "timestamp")
        .group_by("admission_id")
        .agg(
            pl.col("test_name").n_unique().alias("n_unique_labs"),
            pl.col("value").count().alias("n_lab_results"),
            (pl.col("flag") != "normal").sum().alias("n_abnormal_labs"),
        )
    )


def build_full_feature_frame(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """End-to-end feature matrix used by every technique file.

    Composes patients + admissions with vital / medication / lab / derived /
    interaction features. Every technique file calls this so the starting
    feature matrix is identical — only the SELECTION method differs.
    """
    patients = tables["patients"]
    admissions = tables["admissions"]

    patient_admissions = patients.join(admissions, on="patient_id", how="inner")

    features = patient_admissions.clone()
    vf = build_vital_features(tables["vitals"], admissions)
    features = features.join(vf, on="admission_id", how="left")

    # Vital stat columns (std, trend, cv, ...) are legitimately null
    # when a patient has only 1 reading for a given vital, or when a
    # vital was never sampled. Fill with 0 so downstream selection
    # methods (sklearn) see no nulls.
    vital_stat_cols = [
        c
        for c in features.columns
        if any(
            c.endswith(f"_{s}")
            for s in ("mean", "std", "min", "max", "range", "trend", "count", "cv")
        )
    ]
    features = features.with_columns(
        *[pl.col(c).fill_null(0.0) for c in vital_stat_cols]
    )

    mf = build_medication_features(tables["medications"], admissions)
    features = features.join(mf, on="admission_id", how="left")

    lf = build_lab_features(tables["labs"], admissions)
    features = features.join(lf, on="admission_id", how="left")

    # Derived features. NOTE: nothing here may divide by los_days or use
    # the discharge time — the length of stay is the TARGET and is unknown
    # at prediction time. Rates use the fixed PREDICTION_HOURS window.
    features = features.with_columns(
        (pl.col("n_abnormal_labs") / pl.col("n_lab_results").clip(lower_bound=1)).alias(
            "abnormal_lab_ratio"
        ),
        (pl.col("n_medication_doses") / PREDICTION_HOURS).alias(
            "medication_doses_per_hour"
        ),
        (pl.col("n_unique_medications") > 10).alias("polypharmacy_flag"),
    )

    # Null fills for patients without meds / labs
    fill_int = [
        "n_unique_medications",
        "n_medication_doses",
        "n_unique_labs",
        "n_lab_results",
        "n_abnormal_labs",
    ]
    fill_bool = [
        "received_vasopressors",
        "received_antibiotics",
        "received_sedation",
        "polypharmacy_flag",
    ]
    fill_float = [
        "abnormal_lab_ratio",
        "medication_doses_per_hour",
    ]
    features = features.with_columns(
        *[pl.col(c).fill_null(0) for c in fill_int if c in features.columns],
        *[pl.col(c).fill_null(False) for c in fill_bool if c in features.columns],
        *[pl.col(c).fill_null(0.0) for c in fill_float if c in features.columns],
    )

    # Interactions (clinical domain knowledge)
    cols = features.columns
    exprs: list[pl.Expr] = []
    if "heart_rate_mean" in cols and "systolic_bp_mean" in cols:
        exprs.append(
            (
                pl.col("heart_rate_mean")
                / pl.col("systolic_bp_mean").clip(lower_bound=1)
            ).alias("shock_index")
        )
    if "systolic_bp_mean" in cols and "diastolic_bp_mean" in cols:
        exprs.append(
            ((pl.col("systolic_bp_mean") + 2 * pl.col("diastolic_bp_mean")) / 3).alias(
                "map_mean"
            )
        )
    if "temperature_mean" in cols and "heart_rate_mean" in cols:
        exprs.append(
            (pl.col("temperature_mean") * pl.col("heart_rate_mean")).alias(
                "fever_tachycardia"
            )
        )
    exprs.append(
        (pl.col("medication_doses_per_hour") * pl.col("abnormal_lab_ratio")).alias(
            "treatment_burden_score"
        )
    )
    features = features.with_columns(*exprs)

    # Fill any nulls introduced by the interactions
    for name in (
        "shock_index",
        "map_mean",
        "fever_tachycardia",
        "treatment_burden_score",
    ):
        if name in features.columns:
            features = features.with_columns(pl.col(name).fill_null(0.0))

    return features


# ════════════════════════════════════════════════════════════════════════
# SELECTION INPUT PREP
# ════════════════════════════════════════════════════════════════════════


def audit_feature_list(feature_cols: list[str]) -> None:
    """Raise if any ID, target-source or post-outcome column is a feature.

    A real leakage gate: it FAILS (raises ValueError) instead of printing
    a reassuring message. Called by ``prepare_selection_inputs`` and by the
    validation file before anything is logged.
    """
    forbidden = [
        c
        for c in feature_cols
        if c in ID_COLUMNS
        or c in TARGET_SOURCE_COLUMNS
        or c == TARGET_NAME
        or "discharge" in c.lower()
    ]
    if forbidden:
        raise ValueError(
            f"Leakage: these columns cannot be features at prediction time: {forbidden}"
        )


def build_target(features: pl.DataFrame) -> np.ndarray:
    """``long_stay`` = 1 if the stay is longer than the median length of stay.

    Derived from los_days, which is only known at DISCHARGE — so it is the
    label, never a feature.
    """
    median_los = features["los_days"].median()
    return (features["los_days"] > median_los).cast(pl.Int64).to_numpy().ravel()


def prepare_selection_inputs(
    features: pl.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (feature_cols, X, y) for every feature-selection method.

    - Excludes ID columns, the target and its source columns
      (``audit_feature_list`` raises if any slips through)
    - Keeps bool/int/float columns only (selection methods need numbers)
    - Replaces NaN / inf with bounded numbers
    - y is the binary ``long_stay`` label from ``build_target``
    """
    exclude = ID_COLUMNS | TARGET_SOURCE_COLUMNS | {TARGET_NAME}
    numeric_dtypes = {
        pl.Float64,
        pl.Float32,
        pl.Int64,
        pl.Int32,
        pl.UInt32,
        pl.Boolean,
    }
    feature_cols = [
        c
        for c in features.columns
        if c not in exclude and features[c].dtype in numeric_dtypes
    ]
    audit_feature_list(feature_cols)

    X = features.select(feature_cols).to_numpy().astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = build_target(features)
    return feature_cols, X, y


# ════════════════════════════════════════════════════════════════════════
# EXPERIMENT TRACKING
# ════════════════════════════════════════════════════════════════════════


async def setup_tracking() -> tuple[ConnectionManager, ExperimentTracker, str]:
    """Initialize ConnectionManager + ExperimentTracker (kailash-ml 1.1.1).

    Every technique file in ex_1 logs into the same experiment so selection
    runs are directly comparable. The tracker is constructed via the async
    factory; the experiment is auto-created on first ``tracker.track(...)``.
    """
    tracker = await ExperimentTracker.create(store_url=EXPERIMENT_DB)
    conn = ConnectionManager(EXPERIMENT_DB)
    await conn.initialize()
    return conn, tracker, EXPERIMENT_NAME


def setup_tracking_sync() -> tuple[ConnectionManager, ExperimentTracker, str]:
    """Sync wrapper for setup_tracking — convenience for non-async files."""
    return asyncio.run(setup_tracking())


async def log_selection_run(
    tracker: ExperimentTracker,
    experiment_id: str,
    *,
    run_name: str,
    method: str,
    selected_features: list[str],
    total_features: int,
    extra_params: dict[str, str] | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> str:
    """Log a feature-selection run to ExperimentTracker. Returns run id."""
    params = {
        "method": method,
        "n_features_total": str(total_features),
        "n_features_selected": str(len(selected_features)),
        "selected_features": ",".join(selected_features[:30]),
    }
    if extra_params:
        params.update(extra_params)
    metrics = {
        "n_features_selected": float(len(selected_features)),
        "selection_ratio": float(len(selected_features)) / max(1, total_features),
    }
    if extra_metrics:
        metrics.update(extra_metrics)

    async with tracker.track(experiment=experiment_id, run_name=run_name) as run:
        await run.log_params(params)
        await run.log_metrics(metrics)
        await run.add_tag("domain", "clinical")
        await run.add_tag("selection_family", method)
        run_id = run.run_id
    return run_id


# ════════════════════════════════════════════════════════════════════════
# REPORTING HELPERS
# ════════════════════════════════════════════════════════════════════════


def print_ranking(
    title: str, ranking: list[tuple[str, float]], *, top: int = 15
) -> None:
    """Print a ranked feature list with a simple ASCII bar chart."""
    print(f"\n=== {title} ===")
    print(f"{'Feature':<35} {'Score':>10}")
    print("-" * 48)
    if not ranking:
        print("  (empty ranking)")
        return
    # Guard against NaN/inf scores (e.g. mutual-information ties on duplicate
    # rows can yield NaN) so the ASCII bar never does int(NaN).
    finite = [abs(s) for _, s in ranking[:top] if np.isfinite(s)]
    max_score = (max(finite) if finite else 1.0) or 1.0
    for name, score in ranking[:top]:
        safe = score if np.isfinite(score) else 0.0
        bar_len = int(abs(safe) / max_score * 20)
        bar = "#" * bar_len
        print(f"  {name:<33} {safe:>10.4f}  {bar}")


def save_ranking_csv(
    ranking: list[tuple[str, float]], filename: str, score_col: str = "score"
) -> Path:
    """Persist a ranking as CSV into OUTPUT_DIR. Returns the file path."""
    path = OUTPUT_DIR / filename
    pl.DataFrame(
        {"feature": [n for n, _ in ranking], score_col: [s for _, s in ranking]}
    ).write_csv(path)
    return path
