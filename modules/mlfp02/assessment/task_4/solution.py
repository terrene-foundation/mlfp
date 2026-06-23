# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 4: Feature Engineering & Feature Store
(Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.

Builds an admission-level feature table (entity key = admission_id) by joining
demographics and aggregating the messy vitals / labs / medications event tables.
Every step is deterministic polars; the only non-trivial parts are the string
parsing (lab values with junk like HAEMOLYSED, dose strings like "34.8MG") and
the documented imputation policy.
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader

DT_FMT = "%Y-%m-%d %H:%M:%S"

FEATURE_COLUMNS = [
    "admission_id",
    "feature_timestamp",
    "age",
    "gender",
    "bmi",
    "diagnosis",
    "icu_type",
    "mean_heart_rate",
    "mean_systolic_bp",
    "min_spo2",
    "max_temperature",
    "n_vitals",
    "n_labs",
    "n_abnormal_labs",
    "mean_creatinine",
    "n_distinct_drugs",
    "n_iv_meds",
    "total_dose_mg",
    "los_days",
]
# Columns imputed with the column median (computed over admissions that have them).
_MEDIAN_IMPUTE = [
    "age",
    "bmi",
    "mean_heart_rate",
    "mean_systolic_bp",
    "min_spo2",
    "max_temperature",
    "mean_creatinine",
]
# Count columns imputed with zero (no events recorded).
_ZERO_IMPUTE_INT = ["n_vitals", "n_labs", "n_abnormal_labs", "n_distinct_drugs", "n_iv_meds"]


def solve() -> pl.DataFrame:
    """Assemble the admission-level feature-store table.

    Returns a 19-column Polars DataFrame, one row per admission_id (8,000 rows),
    sorted ascending by ``admission_id``. ``feature_timestamp`` is a Datetime
    point-in-time anchor; ``los_days`` is the outcome label (kept last); every
    other column is a model-ready feature with no nulls.
    """
    loader = MLFPDataLoader()
    adm = loader.load("mlfp02", "icu_admissions.parquet")
    pat = loader.load("mlfp02", "icu_patients.parquet")
    vit = loader.load("mlfp02", "icu_vitals.parquet")
    labs = loader.load("mlfp02", "icu_labs.parquet")
    meds = loader.load("mlfp02", "icu_medications.parquet")

    # --- Base: admissions + point-in-time anchor + patient demographics ---
    base = adm.select(
        "admission_id",
        "patient_id",
        "diagnosis",
        "icu_type",
        "los_days",
        pl.col("admit_time").str.strptime(pl.Datetime, DT_FMT).alias("feature_timestamp"),
    ).join(
        pat.select("patient_id", "age", "gender", "bmi"),
        on="patient_id",
        how="left",
    )

    # --- Vitals aggregation (nulls within a column are ignored by the aggregates) ---
    vag = vit.group_by("admission_id").agg(
        pl.col("heart_rate").mean().alias("mean_heart_rate"),
        pl.col("systolic_bp").mean().alias("mean_systolic_bp"),
        pl.col("spo2").min().alias("min_spo2"),
        pl.col("temperature").max().alias("max_temperature"),
        pl.len().alias("n_vitals"),
    )

    # --- Labs aggregation: parse value (junk -> null), normalise flag casing ---
    labs_parsed = labs.with_columns(
        pl.col("value").cast(pl.Float64, strict=False).alias("val_num"),
        pl.col("flag").str.to_lowercase().alias("flag_l"),
    )
    lag = labs_parsed.group_by("admission_id").agg(
        pl.len().alias("n_labs"),
        (pl.col("flag_l") == "abnormal").sum().alias("n_abnormal_labs"),
        pl.col("val_num")
        .filter(pl.col("test_name") == "Creatinine")
        .mean()
        .alias("mean_creatinine"),
    )

    # --- Medications aggregation: parse the leading numeric dose (mg) ---
    meds_parsed = meds.with_columns(
        pl.col("dose").str.extract(r"([0-9]+\.?[0-9]*)", 1).cast(pl.Float64).alias("dose_mg")
    )
    mag = meds_parsed.group_by("admission_id").agg(
        pl.col("drug_name").n_unique().alias("n_distinct_drugs"),
        (pl.col("route") == "IV").sum().alias("n_iv_meds"),
        pl.col("dose_mg").sum().alias("total_dose_mg"),
    )

    # --- Join all feature blocks onto the admission base ---
    ft = (
        base.join(vag, on="admission_id", how="left")
        .join(lag, on="admission_id", how="left")
        .join(mag, on="admission_id", how="left")
    )

    # --- Imputation policy (all medians computed pre-fill in one pass) ---
    ft = ft.with_columns(
        pl.col("gender").fill_null("Unknown"),
        pl.col("total_dose_mg").fill_null(0.0),
        *[pl.col(col).fill_null(0).cast(pl.Int64) for col in _ZERO_IMPUTE_INT],
        *[
            pl.col(col).cast(pl.Float64).fill_null(pl.col(col).median())
            for col in _MEDIAN_IMPUTE
        ],
    )

    return ft.select(FEATURE_COLUMNS).sort("admission_id")


if __name__ == "__main__":
    out = solve()
    print(out.head())
    print(f"\nShape: {out.shape}")
    nulls = {c: out[c].null_count() for c in out.columns}
    print(f"Total nulls across all columns: {sum(nulls.values())}")
    print(f"Admissions with vitals (n_vitals>0): {out.filter(pl.col('n_vitals') > 0).height}")
    print(f"Admissions with labs    (n_labs>0):  {out.filter(pl.col('n_labs') > 0).height}")
    print(f"feature_timestamp dtype: {out.schema['feature_timestamp']}")
