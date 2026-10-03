# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 8 — FeatureStore + Feature Engineering.

Contains: HDB resale data loading, feature validation, FeatureStore and
ExperimentTracker wiring for the installed kailash-ml 2.x, and
OLS-from-scratch helpers reused across the four R10 technique files:

    01_feature_schema.py    — FeatureSchema v1 + validation + materialisation
    02_point_in_time.py     — Point-in-time retrieval + leakage demonstration
    03_rolling_features.py  — FeatureSchema v2 + trailing rolling windows
    04_modeling_lineage.py  — Regression + hypothesis tests + Bayes + lineage

Technique-specific logic (schema construction, rolling window design,
coefficient interpretation) belongs in the per-technique files. This
module only owns infrastructure and reusable numeric helpers.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

# ════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs") / "mlfp02_ex8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_STORE_URL = "sqlite:///mlfp02_ex8_features.db"
EXPERIMENT_STORE_URL = "sqlite:///mlfp02_experiments.db"
EXPERIMENT_NAME = "mlfp02_ex8_hdb_features"

# Single-tenant course store: kailash-ml's FeatureStore requires a tenant
# scope on every call; "_single" is its documented single-tenant sentinel.
FEATURE_TENANT = "_single"


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — HDB resale flats (data.gov.sg)
# ════════════════════════════════════════════════════════════════════════


def load_hdb_resale() -> pl.DataFrame:
    """Load HDB resale transactions with a parsed transaction_date column.

    The raw file stores `month` as "YYYY-MM"; we convert it to a polars
    Date so every downstream technique can sort, filter, and roll on a
    real temporal axis without string parsing.
    """
    loader = MLFPDataLoader()
    hdb = loader.load("mlfp01", "hdb_resale.parquet")
    hdb = hdb.with_columns(
        pl.col("month").str.to_date("%Y-%m").alias("transaction_date")
    )
    return hdb


# ════════════════════════════════════════════════════════════════════════
# FEATURE STORE + EXPERIMENT TRACKER — kailash-ml wiring
# ════════════════════════════════════════════════════════════════════════


def create_feature_store(url: str = FEATURE_STORE_URL) -> Any:
    """Build a kailash-ml FeatureStore backed by a DataFlow database.

    FeatureStore(dataflow) persists feature tables through DataFlow (no raw
    SQL). Any construction error propagates — there is no silent fallback.
    """
    from dataflow import DataFlow
    from kailash_ml.features import FeatureStore

    return FeatureStore(DataFlow(url), default_tenant_id=FEATURE_TENANT)


async def create_tracker(url: str = EXPERIMENT_STORE_URL) -> Any:
    """Create an ExperimentTracker, independently of the feature store."""
    from kailash_ml import ExperimentTracker

    return await ExperimentTracker.create(store_url=url)


def to_store_frame(df: pl.DataFrame, schema: Any) -> pl.DataFrame:
    """Project ``df`` to the schema's entity, timestamp and field columns.

    The store keys rows by an integer entity id and a datetime event time,
    so ``transaction_id`` is cast to Int64 and ``transaction_date`` (a Date)
    to Datetime.
    """
    casts = {"int64": pl.Int64, "float64": pl.Float64}
    cols = [
        pl.col(schema.entity_id_column).cast(pl.Int64),
        pl.col(schema.timestamp_column).cast(pl.Datetime("us")),
        *[pl.col(f.name).cast(casts.get(f.dtype, pl.Float64)) for f in schema.fields],
    ]
    return df.select(cols)


async def materialize_features(fs: Any, schema: Any, df: pl.DataFrame) -> Any:
    """Write ``df``'s schema columns into the FeatureStore (idempotent upsert).

    Returns kailash-ml's MaterializeResult (row_count, lineage_hash, version...).
    """
    from kailash_ml.features import FeatureGroup

    group = FeatureGroup(schema, dataflow=fs.dataflow)
    return await fs.materialize(group, to_store_frame(df, schema))


# ════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION — v1 (basic property) and v2 (rolling market)
# ════════════════════════════════════════════════════════════════════════


# Plausibility bounds used by validate_v1_features. Prices outside this range
# are sentinel/typo values in the raw file (e.g. $10, $9,000,000).
PRICE_BOUNDS = (100_000, 2_000_000)
MAX_LEASE_YEARS = 99


def validate_v1_features(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Apply value-level contract checks the dtype schema cannot express.

    Rules: remaining lease within (0, 99] years (an HDB lease is 99 years);
    lease cannot commence after the sale; resale price within PRICE_BOUNDS;
    positive floor area. Returns (valid_rows, {rule: violation_count}).
    """
    rules = {
        "remaining_lease_years > 99": pl.col("remaining_lease_years") > MAX_LEASE_YEARS,
        "remaining_lease_years <= 0": pl.col("remaining_lease_years") <= 0,
        "lease commences after sale": pl.col("lease_commence_date")
        > pl.col("transaction_date").dt.year(),
        "resale_price outside bounds": ~pl.col("resale_price").is_between(*PRICE_BOUNDS),
        "floor_area_sqm <= 0": pl.col("floor_area_sqm") <= 0,
    }
    report = {name: int(df.filter(expr).height) for name, expr in rules.items()}
    any_violation = pl.any_horizontal(list(rules.values()))
    return df.filter(~any_violation), report


def compute_v1_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute version-1 HDB property features from raw transactions.

    Produces: storey_midpoint, price_per_sqm, remaining_lease_years,
    transaction_id (row index). These are the base features v2 extends.
    """
    return df.with_columns(
        (
            (
                pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
                + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
            )
            / 2
        ).alias("storey_midpoint"),
        (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
        (99 - (pl.col("transaction_date").dt.year() - pl.col("lease_commence_date")))
        .cast(pl.Float64)
        .alias("remaining_lease_years"),
    ).with_row_index("transaction_id")


def compute_v2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute v2 features = v1 + TRAILING town-level market context.

    Uses polars ``group_by_dynamic`` on ``transaction_date`` bucketed by
    month, then shifts each town's monthly series by one month BEFORE the
    6-month rolling window. A transaction in month m therefore sees only
    months m-6 .. m-1 — never its own month, which contains its own price
    (that would be target leakage). The windows run over the town's months
    that have transactions.

    Only rows that pass ``validate_v1_features`` are kept, so sentinel
    prices never enter the town statistics.

    Warm-up: the first 6 months per town have null median/volume and the
    first 7 have a null trend — callers must ``drop_nulls`` before modelling.
    """
    result, _ = validate_v1_features(compute_v1_features(df))
    result = result.sort("transaction_date")

    town_stats = (
        result.group_by_dynamic("transaction_date", every="1mo", group_by="town")
        .agg(
            pl.col("resale_price").median().alias("monthly_median"),
            pl.col("resale_price").count().alias("monthly_volume"),
        )
        .sort("town", "transaction_date")
    )

    prior_median = pl.col("monthly_median").shift(1).over("town")
    prior_volume = pl.col("monthly_volume").shift(1).over("town")
    town_stats = town_stats.with_columns(
        prior_median.rolling_mean(window_size=6).over("town").alias("town_median_price"),
        prior_volume.rolling_sum(window_size=6)
        .over("town")
        .cast(pl.Int64)
        .alias("town_transaction_volume"),
        (
            (
                pl.col("monthly_median").shift(1).over("town")
                - pl.col("monthly_median").shift(7).over("town")
            )
            / pl.col("monthly_median").shift(7).over("town")
            * 100
        ).alias("town_price_trend"),
    )

    result = result.join(
        town_stats.select(
            "town",
            "transaction_date",
            "town_median_price",
            "town_transaction_volume",
            "town_price_trend",
        ),
        on=["town", "transaction_date"],
        how="left",
    )
    return result


# ════════════════════════════════════════════════════════════════════════
# POINT-IN-TIME RETRIEVAL HELPERS
# ════════════════════════════════════════════════════════════════════════


def as_of(
    df: pl.DataFrame, cutoff: datetime, date_col: str = "transaction_date"
) -> pl.DataFrame:
    """Return rows strictly before ``cutoff`` (Polars point-in-time filter).

    Used to build training sets for a cutoff and to cross-check the
    FeatureStore's ``get_features(schema, timestamp=...)`` result.
    """
    return df.filter(pl.col(date_col) < pl.lit(cutoff.date()))


# ════════════════════════════════════════════════════════════════════════
# FROM-SCRATCH OLS HELPERS — reused across techniques 3 and 4
# ════════════════════════════════════════════════════════════════════════

FEATURE_LIST: list[str] = [
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
    "town_median_price",
    "town_price_trend",
]


def prepare_design_matrix(
    df: pl.DataFrame,
    feature_list: list[str] = FEATURE_LIST,
    target: str = "resale_price",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drop nulls, build ``[1, X]`` design matrix, return ``(X, y, names)``."""
    model_data = df.drop_nulls(subset=[*feature_list, target])
    X_raw = model_data.select(feature_list).to_numpy().astype(np.float64)
    y = model_data[target].to_numpy().astype(np.float64)
    X = np.column_stack([np.ones(len(y)), X_raw])
    names = ["intercept", *feature_list]
    return X, y, names


def fit_ols(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit OLS from scratch and return a dict with betas, SEs, t, p, R²."""
    from scipy import stats as sp_stats  # local import — optional at module load

    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat

    ssr = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ssr / sst
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    rmse = float(np.sqrt(ssr / n))

    sigma_sq = ssr / (n - k)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma_sq * np.diag(xtx_inv))
    t_stat = beta / se
    p_val = 2.0 * (1.0 - sp_stats.t.cdf(np.abs(t_stat), df=n - k))

    sse = float(np.sum((y_hat - y.mean()) ** 2))
    f_stat = (sse / (k - 1)) / (ssr / (n - k))
    f_p = 1.0 - sp_stats.f.cdf(f_stat, dfn=k - 1, dfd=n - k)

    return {
        "n": n,
        "k": k,
        "beta": beta,
        "se": se,
        "t": t_stat,
        "p": p_val,
        "y_hat": y_hat,
        "resid": resid,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": rmse,
        "f_stat": float(f_stat),
        "f_p": float(f_p),
    }


def normal_normal_posterior(
    beta_hat: float,
    se_hat: float,
    mu_prior: float = 0.0,
    sigma_prior: float = 10_000.0,
) -> dict[str, float]:
    """Normal-Normal conjugate posterior for a single OLS coefficient."""
    prec_prior = 1.0 / sigma_prior**2
    prec_data = 1.0 / se_hat**2
    prec_post = prec_prior + prec_data
    mu_post = (mu_prior * prec_prior + beta_hat * prec_data) / prec_post
    sigma_post = float(np.sqrt(1.0 / prec_post))
    return {
        "mu_post": float(mu_post),
        "sigma_post": sigma_post,
        "ci_low": float(mu_post - 1.96 * sigma_post),
        "ci_high": float(mu_post + 1.96 * sigma_post),
    }


# ════════════════════════════════════════════════════════════════════════
# FEATURE SCHEMA BUILDERS — kailash-ml FeatureSchema / FeatureField
# ════════════════════════════════════════════════════════════════════════


def build_schema_v1() -> Any:
    """Return the FeatureSchema v1 definition (basic property features).

    Uses ``kailash_ml.features.FeatureSchema`` — the class FeatureStore
    accepts (the top-level ``kailash_ml.FeatureSchema`` is a different,
    incompatible type in kailash-ml 2.2.x).
    """
    from kailash_ml.features import FeatureField, FeatureSchema

    return FeatureSchema(
        name="hdb_property_features",
        version=1,
        fields=(
            FeatureField("floor_area_sqm", "float64", False, "Floor area in square metres"),
            FeatureField("remaining_lease_years", "float64", False, "Remaining lease in years"),
            FeatureField("storey_midpoint", "float64", False, "Midpoint of storey range"),
            FeatureField("price_per_sqm", "float64", False, "Transaction price per square metre"),
        ),
        entity_id_column="transaction_id",
        timestamp_column="transaction_date",
    )


def build_schema_v2() -> Any:
    """Return FeatureSchema v2 = v1 + three trailing market-context fields.

    In kailash-ml 2.2.x the store's backing table is keyed by schema NAME,
    so a version that adds columns needs its own name (re-using the v1 name
    fails with "table ... has no column named town_median_price").
    """
    from kailash_ml.features import FeatureField, FeatureSchema

    v1 = build_schema_v1()
    return FeatureSchema(
        name="hdb_property_features_v2",
        version=2,
        fields=(
            *v1.fields,
            FeatureField(
                "town_median_price", "float64", True, "Town median price, previous 6 months"
            ),
            FeatureField(
                "town_transaction_volume", "int64", True, "Town transactions, previous 6 months"
            ),
            FeatureField(
                "town_price_trend", "float64", True, "Town median change % (m-7 to m-1)"
            ),
        ),
        entity_id_column="transaction_id",
        timestamp_column="transaction_date",
    )
