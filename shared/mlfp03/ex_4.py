# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 4 — Gradient Boosting Deep Dive.

Contains: Singapore credit-scoring data loader, preprocessing pipeline
wrapper, numpy/polars conversion, model-zoo factories (XGBoost/LightGBM/
CatBoost with identical defaults), evaluation metric helpers, and the
output directory used by every technique file.

Technique-specific code (from-scratch boosting loops, split-gain formulas,
hyperparameter sweeps, early-stopping analysis) does NOT belong here — it
lives in the per-technique files under `modules/mlfp03/solutions/ex_4/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from kailash_ml import PreprocessingPipeline
from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# CONFIG — output directory, random seeds, dataset tag
# ════════════════════════════════════════════════════════════════════════

SEED = 42
DATASET_MODULE = "mlfp02"
DATASET_FILE = "sg_credit_scoring.parquet"
TARGET_COLUMN = "default"

# Columns that MUST NOT be model inputs (Lesson 3.1 leakage rule):
#   customer_id              — a row identifier, not a property of the applicant
#   future_default_indicator — recorded AFTER the loan outcome is known; it
#                              agrees with ``default`` on ~99% of rows, so a
#                              model that sees it "predicts" default by
#                              reading the answer.
# ``screen_single_feature_leakage`` below is the EDA check that exposes it.
CREDIT_NON_FEATURE_COLUMNS: tuple[str, ...] = ("customer_id", "future_default_indicator")

OUTPUT_DIR = Path("outputs") / "mlfp03" / "ex_4_boosting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore credit-scoring, shared across all four techniques
# ════════════════════════════════════════════════════════════════════════


def load_credit_data() -> pl.DataFrame:
    """Load the Singapore credit-scoring dataset via MLFPDataLoader."""
    loader = MLFPDataLoader()
    return loader.load(DATASET_MODULE, DATASET_FILE)


def screen_single_feature_leakage(
    df: pl.DataFrame,
    target: str = TARGET_COLUMN,
    *,
    flag_auc: float = 0.95,
) -> pl.DataFrame:
    """EDA leakage screen: how well does EACH numeric column alone rank the target?

    A legitimate credit feature on its own rarely exceeds AUC ~0.75. A column
    that separates defaulters almost perfectly by itself is almost always
    information recorded after the outcome (or the outcome in disguise).

    Returns a polars DataFrame (feature, single_feature_auc, suspicious)
    sorted by AUC, where AUC is orientation-free (max(auc, 1 - auc)).
    """
    y = df[target].to_numpy()
    rows: list[dict[str, Any]] = []
    for col, dtype in df.schema.items():
        if col == target or not dtype.is_numeric():
            continue
        x = df[col].fill_null(df[col].median()).to_numpy().astype(float)
        if np.unique(x).size < 2:
            continue
        auc = float(roc_auc_score(y, x))
        auc = max(auc, 1.0 - auc)
        rows.append(
            {"feature": col, "single_feature_auc": auc, "suspicious": auc >= flag_auc}
        )
    return pl.DataFrame(rows).sort("single_feature_auc", descending=True)


def prepare_credit_split() -> dict[str, Any]:
    """Load credit data and return a train/test split ready for boosting.

    Returns a dict with:
      X_train, y_train, X_test, y_test : numpy arrays
      feature_names                    : list[str]
      default_rate                     : float (positive-class prevalence)

    ``CREDIT_NON_FEATURE_COLUMNS`` (the row ID and the post-outcome leak
    column) are dropped BEFORE preprocessing so no model in this exercise
    can see them.

    Tree models do not need normalisation, so we set ``normalize=False``.
    Categoricals are ordinal-encoded because XGBoost/LightGBM expect numeric
    input; CatBoost would accept raw categoricals but we keep the pipeline
    consistent across all three libraries for a fair comparison.
    """
    credit = load_credit_data().drop(CREDIT_NON_FEATURE_COLUMNS)

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        data=credit,
        target=TARGET_COLUMN,
        train_size=0.8,
        seed=SEED,
        normalize=False,
        categorical_encoding="ordinal",
        imputation_strategy="median",
    )

    feature_cols = [c for c in result.train_data.columns if c != TARGET_COLUMN]
    X_train, y_train, col_info = to_sklearn_input(
        result.train_data,
        feature_columns=feature_cols,
        target_column=TARGET_COLUMN,
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_cols,
        target_column=TARGET_COLUMN,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": col_info["feature_columns"],
        "default_rate": float(credit[TARGET_COLUMN].mean()),
    }


def categorical_feature_indices(feature_names: list[str]) -> list[int]:
    """Column indices (in ``feature_names``) of the dataset's string categoricals.

    ``prepare_credit_split`` ordinal-encodes these (gender, race, region,
    loan_purpose, ...). CatBoost can instead treat them as true categories.
    """
    schema = load_credit_data().schema
    return [
        i for i, name in enumerate(feature_names) if schema.get(name) == pl.String
    ]


def as_catboost_categoricals(X: np.ndarray, cat_idx: list[int]) -> np.ndarray:
    """Return an object array where the categorical columns hold integer codes.

    CatBoost refuses float-valued categorical columns; integer codes (or
    strings) are accepted and treated as unordered categories, so the
    arbitrary ordinal order no longer matters.
    """
    out = X.astype(object)
    for i in cat_idx:
        out[:, i] = X[:, i].astype(int)
    return out


# ════════════════════════════════════════════════════════════════════════
# MODEL FACTORIES — identical defaults for fair comparison
# ════════════════════════════════════════════════════════════════════════


def make_xgboost(
    n_estimators: int = 500,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    **extra: Any,
):
    """Build an XGBoost classifier with course-standard defaults."""
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
        **extra,
    )


def make_lightgbm(
    n_estimators: int = 500,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    **extra: Any,
):
    """Build a LightGBM classifier with course-standard defaults."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=31,
        random_state=SEED,
        verbose=-1,
        **extra,
    )


def make_catboost(
    iterations: int = 500,
    learning_rate: float = 0.1,
    depth: int = 6,
    **extra: Any,
):
    """Build a CatBoost classifier with course-standard defaults."""
    import catboost as cb

    return cb.CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=SEED,
        verbose=0,
        **extra,
    )


# ════════════════════════════════════════════════════════════════════════
# EVALUATION — shared metric helpers for imbalanced binary classification
# ════════════════════════════════════════════════════════════════════════


def evaluate_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the full boosting-eval metric bundle.

    AUC-PR is the primary metric — with a 12% default rate, AUC-ROC rewards
    models that rank negatives correctly even when they miss every default.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def print_metrics(
    name: str, metrics: dict[str, float], train_time: float | None = None
) -> None:
    """One-line headline: AUC-ROC | AUC-PR | Log Loss | F1 | Time."""
    time_str = f" | time={train_time:.2f}s" if train_time is not None else ""
    print(
        f"  {name}: "
        f"AUC-ROC={metrics['auc_roc']:.4f} | "
        f"AUC-PR={metrics['auc_pr']:.4f} | "
        f"log_loss={metrics['log_loss']:.4f} | "
        f"F1={metrics['f1']:.4f}"
        f"{time_str}"
    )


# ════════════════════════════════════════════════════════════════════════
# FROM-SCRATCH GRADIENT BOOSTING (used by 01_boosting_theory.py)
# ════════════════════════════════════════════════════════════════════════


def xgb_split_gain(
    g_left: float,
    h_left: float,
    g_right: float,
    h_right: float,
    lambda_reg: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """XGBoost split-gain formula from 2nd-order Taylor expansion of log-loss.

    Gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
    """
    left_score = g_left**2 / (h_left + lambda_reg)
    right_score = g_right**2 / (h_right + lambda_reg)
    parent_score = (g_left + g_right) ** 2 / (h_left + h_right + lambda_reg)
    return 0.5 * (left_score + right_score - parent_score) - gamma


def make_1d_demo(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 1D logistic-shaped binary classification demo.

    Used by the from-scratch boosting loop to show that residual-fitting
    converges in just a handful of rounds.
    """
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 1, n).reshape(-1, 1)
    true_proba = 1 / (1 + np.exp(-10 * (x.ravel() - 0.5)))
    y = rng.binomial(1, true_proba)
    return x, y
