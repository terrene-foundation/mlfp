# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 8 — Production ML + Drift +
Deployment.

Contains: data loading (Singapore credit scoring via MLFPDataLoader) with
decoded protected-attribute groups, the baseline model (TrainingPipeline +
engine isotonic calibration, registered in this exercise's registry),
split-conformal helpers, per-group fairness measurement, PSI/KS drift
helpers, and the common output directory.

Technique-specific code (conformal quantile logic, dashboard rendering,
readiness checklist) stays in the per-technique files.
"""
from __future__ import annotations

import asyncio
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from kailash_ml import PreprocessingPipeline
from kailash_ml.interop import to_sklearn_input
from kailash_ml.types import FeatureField, FeatureSchema

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

setup_environment()

OUTPUT_DIR = Path("outputs") / "ex8_production"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore credit scoring (from MLFP02)
# ════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42

# Columns that MUST NOT be model inputs (Lesson 3.1 leakage rule):
#   customer_id              — a row identifier, not a property of the applicant
#   future_default_indicator — recorded AFTER the loan outcome is known; it
#                              agrees with ``default`` on ~99% of rows, so a
#                              model that sees it "predicts" default by
#                              reading the answer (Exercise 4 screens for it).
CREDIT_NON_FEATURE_COLUMNS: tuple[str, ...] = ("customer_id", "future_default_indicator")


def _decode_groups(
    X: np.ndarray, feature_names: list[str], mappings: dict[str, dict[str, int]]
) -> dict[str, np.ndarray]:
    """Human-readable protected-attribute labels for each row of ``X``.

    ``race`` and ``gender`` are decoded from the pipeline's ordinal codes;
    ``age`` is banded. These are the groups a fairness audit reports on.
    """
    groups: dict[str, np.ndarray] = {}
    for col in ("race", "gender"):
        inverse = {code: label for label, code in mappings[col].items()}
        codes = X[:, feature_names.index(col)].astype(int)
        groups[col] = np.array([inverse.get(c, "unknown") for c in codes])
    age = X[:, feature_names.index("age")]
    groups["age_band"] = np.select(
        [age < 30, age < 45, age < 60], ["<30", "30-44", "45-59"], default="60+"
    )
    return groups


def load_credit_split() -> dict[str, Any]:
    """Load Singapore credit scoring data, preprocess, and return a split.

    Returns a dict with keys:
        X_train, y_train, X_test, y_test : numpy arrays
        feature_names                    : list[str]
        default_rate                     : float (train positive rate)
        test_groups                      : {"race"|"gender"|"age_band": labels}
    """
    loader = MLFPDataLoader()
    credit = loader.load("mlfp02", "sg_credit_scoring.parquet")

    # Drop the row identifier and the post-outcome leak column BEFORE
    # preprocessing — see CREDIT_NON_FEATURE_COLUMNS above.
    credit = credit.drop(CREDIT_NON_FEATURE_COLUMNS)

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        credit,
        target="default",
        seed=RANDOM_SEED,
        normalize=False,
        categorical_encoding="ordinal",
    )

    feature_cols = [c for c in result.train_data.columns if c != "default"]
    X_train, y_train, col_info = to_sklearn_input(
        result.train_data,
        feature_columns=feature_cols,
        target_column="default",
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_cols,
        target_column="default",
    )
    feature_names = col_info["feature_columns"]
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "default_rate": float(y_train.mean()),
        "test_groups": _decode_groups(
            X_test, feature_names, result.transformers["ordinal_mappings"]
        ),
    }


def to_frame(X: np.ndarray, feature_names: list[str]) -> pl.DataFrame:
    """numpy feature matrix → polars DataFrame with the feature names."""
    return pl.DataFrame(X, schema=feature_names, orient="row")


# ════════════════════════════════════════════════════════════════════════
# BASELINE MODEL — TrainingPipeline (LightGBM) + engine isotonic calibration
# ════════════════════════════════════════════════════════════════════════
# Fixed, reasonable hyperparameters (Exercise 7 showed how to search for
# them) so every technique file trains the identical model. No class
# weighting: Exercise 5 showed it inflates probabilities, and this model's
# probabilities feed conformal sets and a model card.

MODEL_NAME = "credit_default_ex8"
BASELINE_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 40,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}
_DB_ABS_PATH = (OUTPUT_DIR / "ex8_models.db").resolve()
DB_URL: str = os.environ.get("MLFP03_EX8_DB_URL", f"sqlite:///{_DB_ABS_PATH.as_posix()}")


async def open_registry() -> tuple[Any, Any]:
    """Return ``(ModelRegistry, ConnectionManager)``; caller closes the connection."""
    from kailash.db import ConnectionManager
    from kailash_ml import ModelRegistry

    conn = ConnectionManager(DB_URL)
    await conn.initialize()
    return ModelRegistry(conn), conn


async def _train_and_calibrate(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
) -> tuple[Any, int]:
    from kailash_ml import TrainingPipeline
    from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec

    # 80% of the training rows fit the model; the other 20% (stratified)
    # are kept aside to fit the isotonic calibration map.
    idx_fit, idx_cal = train_test_split(
        np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
    )
    fit_frame = to_frame(X_train[idx_fit], feature_names).with_columns(
        pl.Series("default", y_train[idx_fit]),
        pl.int_range(0, len(idx_fit), dtype=pl.Int64).alias("row_id"),
    )
    schema = FeatureSchema(
        name="ex8_credit_input",
        features=[FeatureField(name=f, dtype="float64") for f in feature_names],
        entity_id_column="row_id",
    )
    registry, conn = await open_registry()
    try:
        pipeline = TrainingPipeline(feature_store=None, registry=registry)
        result = await pipeline.train(
            data=fit_frame,
            schema=schema,
            model_spec=ModelSpec(
                model_class="lightgbm.LGBMClassifier",
                framework="lightgbm",
                hyperparameters=BASELINE_PARAMS,
            ),
            eval_spec=EvalSpec(metrics=["auc"], split_strategy="holdout", test_size=0.2),
            experiment_name=MODEL_NAME,
        )
        if result.model_version is None:
            raise RuntimeError("TrainingPipeline did not register the model")
        version = int(result.model_version.version)
        # Unpickling executes code: only load artefacts you trained yourself.
        base = pickle.loads(await registry.load_artifact(MODEL_NAME, version))
        calibrated = await pipeline.calibrate(
            base,
            to_frame(X_train[idx_cal], feature_names),
            pl.Series("default", y_train[idx_cal]),
            method="isotonic",
        )
    finally:
        await conn.close()
    return calibrated, version


def train_calibrated_model(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
) -> Any:
    """Train LightGBM via TrainingPipeline, then isotonic-calibrate it.

    The base model is registered (stage: staging) in this exercise's
    registry as ``MODEL_NAME``; calibration uses TrainingPipeline.calibrate
    on a held-out 20% slice of the training rows.
    """
    calibrated, _ = asyncio.run(_train_and_calibrate(X_train, y_train, feature_names))
    return calibrated


def evaluate_classification(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return the full classification metric bundle used across ex_8."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


# ════════════════════════════════════════════════════════════════════════
# SPLIT CONFORMAL PREDICTION (binary classification)
# ════════════════════════════════════════════════════════════════════════
# Score s = 1 - p(true class | x). With n calibration scores, q̂ is the
# ⌈(n+1)(1-α)⌉-th smallest score — np.quantile at level ⌈(n+1)(1-α)⌉/n
# with method="higher" picks exactly that order statistic (the default
# linear interpolation can land slightly below it and void the guarantee).
# Guarantee: P(Y ∈ C(X)) ≥ 1-α on AVERAGE over future applicants
# (marginal coverage), assuming calibration and future data are
# exchangeable. It is not a per-applicant promise.


def nonconformity_scores(y_true: np.ndarray, p_default: np.ndarray) -> np.ndarray:
    """1 - probability the model gave to the class that actually happened."""
    return np.where(y_true == 1, 1.0 - p_default, p_default)


def conformal_qhat(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected conformal threshold q̂."""
    n = len(cal_scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(cal_scores, level, method="higher"))


def prediction_sets(p_default: np.ndarray, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean membership of class 0 and class 1 in each prediction set.

    A set may be EMPTY (neither class plausible at this α) — that is a
    legitimate conformal outcome, counted as a miss for coverage.
    """
    has_1 = (1.0 - p_default) <= q_hat
    has_0 = p_default <= q_hat
    return has_0, has_1


def conformal_summary(
    y_true: np.ndarray, p_default: np.ndarray, q_hat: float
) -> dict[str, float]:
    """Coverage, average set size and singleton / empty / both-class rates."""
    has_0, has_1 = prediction_sets(p_default, q_hat)
    covered = np.where(y_true == 1, has_1, has_0)
    size = has_0.astype(int) + has_1.astype(int)
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "both_rate": float((size == 2).mean()),
        "empty_rate": float((size == 0).mean()),
    }


# ════════════════════════════════════════════════════════════════════════
# FAIRNESS — measured per protected group (used by the model card + gate)
# ════════════════════════════════════════════════════════════════════════


def fairness_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    groups: dict[str, np.ndarray],
    threshold: float,
) -> pl.DataFrame:
    """Per-group rates for each protected attribute.

    "Flagged" means predicted default (y_proba >= threshold) — the
    adverse outcome for an applicant. Columns: attribute, group, n,
    base_rate, flag_rate, tpr, fpr, mean_pred.
    """
    flagged = y_proba >= threshold
    rows = []
    for attr, labels in groups.items():
        for g in np.unique(labels):
            m = labels == g
            pos, neg = m & (y_true == 1), m & (y_true == 0)
            rows.append(
                {
                    "attribute": attr,
                    "group": str(g),
                    "n": int(m.sum()),
                    "base_rate": float(y_true[m].mean()),
                    "flag_rate": float(flagged[m].mean()),
                    "tpr": float(flagged[pos].mean()) if pos.any() else float("nan"),
                    "fpr": float(flagged[neg].mean()) if neg.any() else float("nan"),
                    "mean_pred": float(y_proba[m].mean()),
                }
            )
    return pl.DataFrame(rows)


def fairness_summary(report: pl.DataFrame, min_group_n: int = 200) -> pl.DataFrame:
    """Per attribute: disparate-impact ratio and equalised-odds gaps.

    Disparate impact here = lowest group flag rate / highest group flag
    rate (1.0 = parity; the four-fifths rule of thumb flags < 0.8).
    Groups smaller than ``min_group_n`` are excluded as too noisy.
    """
    big = report.filter(pl.col("n") >= min_group_n)
    return (
        big.group_by("attribute")
        .agg(
            (pl.col("flag_rate").min() / pl.col("flag_rate").max()).alias("disparate_impact"),
            (pl.col("tpr").max() - pl.col("tpr").min()).alias("tpr_gap"),
            (pl.col("fpr").max() - pl.col("fpr").min()).alias("fpr_gap"),
            (pl.col("mean_pred") - pl.col("base_rate")).abs().max().alias("max_calibration_gap"),
            pl.len().alias("groups"),
        )
        .sort("attribute")
    )


# ════════════════════════════════════════════════════════════════════════
# DRIFT STATISTICS — PSI + KS (the maths DriftMonitor runs for you)
# ════════════════════════════════════════════════════════════════════════
#
# PSI (Population Stability Index):
#     PSI = Σ (p_new - p_ref) * ln(p_new / p_ref)
#     common rule of thumb: < 0.1 no shift, 0.1-0.2 moderate, > 0.2 significant
# KS (Kolmogorov-Smirnov): two-sample test on the empirical CDFs.

# Continuous features used for drift simulations. (The first columns of
# the matrix are ordinal-encoded categoricals such as gender — shifting
# those by "0.5 standard deviations" would be meaningless.)
DRIFT_FEATURES: list[str] = ["income_sgd", "debt_to_income", "credit_utilization"]


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index on 1-D arrays.

    Bin edges come from the reference; the outer edges are opened to ±inf
    so current values outside the reference range still count (dropping
    them would understate exactly the shifts PSI exists to catch).
    """
    _, edges = np.histogram(reference, bins=bins)
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    # Laplace-smooth to avoid log(0)
    ref_props = (ref_counts + 1) / (len(reference) + bins)
    cur_props = (cur_counts + 1) / (len(current) + bins)
    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return (KS statistic, p-value) on 1-D arrays."""
    ks_stat, p_value = stats.ks_2samp(reference, current)
    return float(ks_stat), float(p_value)


def drift_row(
    reference: np.ndarray, current: np.ndarray, psi_threshold: float = 0.1
) -> dict[str, float | str]:
    """Return {psi, ks_stat, ks_pval, drift} for a single feature."""
    psi = compute_psi(reference, current)
    ks_stat, ks_pval = compute_ks(reference, current)
    drift = "YES" if (psi > psi_threshold or ks_pval < 0.05) else "No"
    return {"psi": psi, "ks_stat": ks_stat, "ks_pval": ks_pval, "drift": drift}


def simulate_gradual_drift(
    X_ref: np.ndarray, X_new: np.ndarray, feature_idx: list[int], shift: float = 0.5
) -> np.ndarray:
    """Copy of X_new with the given columns mean-shifted by ``shift`` σ."""
    drifted = X_new.copy()
    for i in feature_idx:
        drifted[:, i] += shift * X_ref[:, i].std()
    return drifted


def simulate_sudden_drift(
    X_ref: np.ndarray,
    X_new: np.ndarray,
    feature_idx: int,
    sigma_shift: float = 3.0,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Replace one column in X_new with a heavily shifted Gaussian."""
    rng = np.random.default_rng(seed)
    drifted = X_new.copy()
    drifted[:, feature_idx] = rng.normal(
        loc=X_ref[:, feature_idx].mean() + sigma_shift * X_ref[:, feature_idx].std(),
        scale=X_ref[:, feature_idx].std(),
        size=drifted.shape[0],
    )
    return drifted
