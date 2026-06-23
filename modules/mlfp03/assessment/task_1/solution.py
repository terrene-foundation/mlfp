# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 1: Feature Engineering & Leakage-Free Selection
(Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.

Builds six business-meaningful engineered features from the raw Southeast-Asia
e-commerce customer table, then ranks the full candidate pool with the
kailash-ml FeatureEngineer (importance method, fit on the TRAIN split only so
no test-set signal leaks into selection).
"""
from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")  # silence FeatureEngineer P2 ExperimentalWarning

# ── Deterministic contract ────────────────────────────────────────────────
N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
TRAIN_FRACTION = 0.75
TOP_K = 8

# Eight raw, model-ready base features (categoricals encoded to integers).
BASE_FEATURES = [
    "total_revenue",
    "order_count",
    "avg_order_value",
    "days_since_last_order",
    "customer_tenure_days",
    "satisfaction_score",
    "num_returns",
    "loyalty_int",
]
# Six engineered features (exact formulas — see problem.md).
ENGINEERED_FEATURES = [
    "revenue_per_order",
    "returns_per_order",
    "is_satisfied",
    "loyal_and_satisfied",
    "tenure_years",
    "spend_per_tenure_day",
]


def _load_base() -> pl.DataFrame:
    """Load the first N_ROWS (sorted by customer_id) and derive the target.

    The native ``churned`` column is a near-deterministic function of recency,
    so it is unusable for a teaching problem. We derive ``premium_response`` —
    whether a customer accepts a premium-membership upsell — from a documented
    logit over satisfaction, loyalty, spend, returns, and a
    loyalty x high-satisfaction interaction, plus seeded Gaussian noise. The
    ~25%% positive rate gives a realistic 3:1 class imbalance.
    """
    df = MLFPDataLoader().load("mlfp03", "ecommerce_customers.parquet")
    df = df.sort("customer_id").head(N_ROWS)

    rng = np.random.default_rng(SEED)

    def z(col: str) -> np.ndarray:
        a = df[col].to_numpy().astype(float)
        return (a - a.mean()) / (a.std() + 1e-9)

    loyal = df["loyalty_member"].cast(pl.Int64).to_numpy().astype(float)
    sat_high = (df["satisfaction_score"] >= 4).cast(pl.Int64).to_numpy().astype(float)
    logit = (
        1.0 * z("satisfaction_score")
        + 0.9 * loyal
        + 0.8 * z("avg_order_value")
        - 0.7 * z("num_returns")
        + 0.5 * z("order_count")
        + 1.4 * (loyal * sat_high)
        + rng.normal(0.0, 1.3, size=df.height)
    )
    target = (logit > 2.0).astype(np.int64)

    return df.with_columns(
        [
            pl.col("loyalty_member").cast(pl.Int64).alias("loyalty_int"),
            pl.Series(TARGET, target),
        ]
    )


def _engineer(df: pl.DataFrame) -> pl.DataFrame:
    """Add the six engineered features with exact, documented formulas."""
    return df.with_columns(
        [
            (pl.col("total_revenue") / pl.col("order_count")).alias("revenue_per_order"),
            (pl.col("num_returns") / pl.col("order_count")).alias("returns_per_order"),
            (pl.col("satisfaction_score") >= 4).cast(pl.Int64).alias("is_satisfied"),
            (
                pl.col("loyalty_int")
                * (pl.col("satisfaction_score") >= 4).cast(pl.Int64)
            ).alias("loyal_and_satisfied"),
            (pl.col("customer_tenure_days") / 365.0).alias("tenure_years"),
            (pl.col("total_revenue") / pl.col("customer_tenure_days")).alias(
                "spend_per_tenure_day"
            ),
        ]
    )


def solve() -> dict:
    """Engineer features then rank them leakage-free with FeatureEngineer.

    Returns a dict with keys:
      - ``feature_matrix``    : pl.DataFrame of the 14 candidate features plus
                                the target column (``premium_response``), in the
                                original row order (no shuffle). Customer IDs,
                                raw text, and the native ``churned`` label are
                                excluded.
      - ``engineered_columns``: list[str] of the 6 engineered feature names.
      - ``selected_features`` : list[str] of the TOP_K (8) features ranked by
                                kailash-ml FeatureEngineer importance, fit on the
                                training split only.
      - ``target_column``     : ``"premium_response"``.
    """
    from kailash_ml.engines.feature_engineer import (
        FeatureEngineer,
        GeneratedColumn,
        GeneratedFeatures,
    )

    df = _engineer(_load_base())

    candidates = BASE_FEATURES + ENGINEERED_FEATURES
    feature_matrix = df.select(candidates + [TARGET])

    # Leakage-free selection: fit the importance ranker on the TRAIN split only.
    n_train = int(TRAIN_FRACTION * feature_matrix.height)
    train = feature_matrix.head(n_train)

    source_map = {
        "revenue_per_order": ["total_revenue", "order_count"],
        "returns_per_order": ["num_returns", "order_count"],
        "is_satisfied": ["satisfaction_score"],
        "loyal_and_satisfied": ["loyalty_int", "satisfaction_score"],
        "tenure_years": ["customer_tenure_days"],
        "spend_per_tenure_day": ["total_revenue", "customer_tenure_days"],
    }
    generated_cols = [
        GeneratedColumn(
            name=name,
            source_columns=source_map[name],
            strategy="interaction",
            dtype="float64",
        )
        for name in ENGINEERED_FEATURES
    ]
    gen = GeneratedFeatures(
        original_columns=BASE_FEATURES,
        generated_columns=generated_cols,
        total_candidates=len(candidates),
        data=train,
    )

    engineer = FeatureEngineer(max_features=50)
    selected = engineer.select(
        train, gen, target=TARGET, method="importance", top_k=TOP_K
    )

    return {
        "feature_matrix": feature_matrix,
        "engineered_columns": list(ENGINEERED_FEATURES),
        "selected_features": list(selected.selected_columns),
        "target_column": TARGET,
    }


if __name__ == "__main__":
    out = solve()
    fm = out["feature_matrix"]
    print(f"feature_matrix shape : {fm.shape}")
    print(f"candidate columns    : {[c for c in fm.columns if c != TARGET]}")
    print(f"engineered_columns   : {out['engineered_columns']}")
    print(f"selected_features    : {out['selected_features']}")
    print(f"target positive rate : {fm[TARGET].mean():.4f}")
