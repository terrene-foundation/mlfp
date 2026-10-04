# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 2 — Regularisation and
Cross-Validation.

Contains: data loading for the Singapore credit scoring dataset,
feature preparation, synthetic 1D bias-variance problem, alpha grids,
plotting helpers, and a shared OUTPUT_DIR for generated artefacts.

Technique-specific code (Ridge/Lasso model construction, nested CV
loops, learning curves) lives in the per-technique solution files —
this module holds only the helpers those files share.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import preprocess_train_test, split_then_preprocess

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

# Deterministic seed for every random operation in this exercise
SEED = 42

# Alpha sweep used by Ridge and the Ridge regularisation path.
ALPHAS: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Lasso / ElasticNet need their OWN grid: sklearn's Lasso divides the
# squared-error term by 2n, so its α lives on a much smaller scale than
# Ridge's. With a standardised target, α ≥ ~0.5 already zeroes every
# coefficient; the grid below walks the full path from "almost OLS" to
# "intercept only".
LASSO_ALPHAS: list[float] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# Regression target for the credit demos. ``savings_balance`` is only
# partly predictable from the other columns (R² ≈ 0.25) and no single
# feature duplicates it — so regularisation has real work to do. (Contrast
# ``credit_utilization``: it is 0.97-correlated with
# ``avg_balance_utilization``, so every model would score the same and no
# regularisation effect would be visible.)
CREDIT_TARGET = "savings_balance"

# Columns that are never features in this exercise:
#   customer_id              — row identifier
#   default                  — the loan OUTCOME, observed after the fact
#   future_default_indicator — post-outcome copy of ``default`` (a planted
#                              leak; see Lesson 3.1)
CREDIT_NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "default",
    "future_default_indicator",
)

# A SMALL training sample is deliberate: with ~33 correlated features and
# a few hundred rows, OLS visibly overfits (train R² well above test R²),
# which is exactly the regime where Ridge and Lasso earn their keep. The
# large test set keeps every test score stable to ~±0.01.
N_TRAIN_DEFAULT = 300
N_TEST_DEFAULT = 5000

# Output directory for HTML plots, summary CSVs, etc.
OUTPUT_DIR = Path("outputs") / "mlfp03_ex2_regularisation_cv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore credit scoring
# ════════════════════════════════════════════════════════════════════════


def load_credit_data(
    n_train: int = N_TRAIN_DEFAULT,
    n_test: int = N_TEST_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load + preprocess the MLFP02 Singapore credit scoring parquet.

    Returns:
        X_train, y_train, X_test, y_test, feature_names

    Target: ``savings_balance`` (S$), STANDARDISED with the training mean
    and standard deviation — so an MSE of 1.0 means "no better than
    predicting the mean" and MSE = 1 − R² (approximately) on the test set.

    ``n_train`` rows are used for fitting (small on purpose — see
    ``N_TRAIN_DEFAULT``) and ``n_test`` held-out rows for evaluation.
    ID, outcome and leak columns (``CREDIT_NON_FEATURE_COLUMNS``) are
    dropped before preprocessing.

    Uses ``kailash_ml.PreprocessingPipeline`` for normalisation +
    ordinal encoding + median imputation, fitted on the ``n_train``
    training rows only — the test rows are held out FIRST, so none of
    their statistics reach the scaler or the imputer. All regularised
    models REQUIRE normalised features (otherwise the penalty is unevenly
    distributed across the coefficient vector).
    """
    loader = MLFPDataLoader()
    credit = loader.load("mlfp02", "sg_credit_scoring.parquet").drop(
        CREDIT_NON_FEATURE_COLUMNS
    )
    credit = credit.sample(n=n_train + n_test, seed=SEED)

    # Split FIRST (exactly n_test held-out rows), then fit the preprocessing
    # on the training rows only.
    result = split_then_preprocess(
        credit,
        target=CREDIT_TARGET,
        test_size=n_test,
        seed=SEED,
        normalize=True,
        categorical_encoding="ordinal",
        imputation_strategy="median",
    )

    feature_cols = [c for c in result.train_data.columns if c != CREDIT_TARGET]
    X_train, y_train, col_info = to_sklearn_input(
        result.train_data,
        feature_columns=feature_cols,
        target_column=CREDIT_TARGET,
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_cols,
        target_column=CREDIT_TARGET,
    )

    # Standardise the target with TRAINING statistics only (no test leakage).
    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return X_train, y_train, X_test, y_test, col_info["feature_columns"]


# Preprocessing for the credit-default CV demo. It is fitted INSIDE every
# fold (see ``cross_val_auc_split_first``), never once on the whole sample.
CREDIT_DEFAULT_PREPROCESSING: dict[str, Any] = {
    "normalize": True,
    "categorical_encoding": "ordinal",
    "imputation_strategy": "median",
}


def load_credit_default_sample(
    n: int = 600,
) -> tuple[pl.DataFrame, np.ndarray, list[str]]:
    """A small credit sample with the BINARY ``default`` outcome as target.

    Used to show why stratified k-fold matters for an imbalanced target
    (~13% defaults): with plain k-fold the default rate drifts from fold
    to fold. Returns (X_raw, y, feature_names). The ID / post-outcome leak
    columns are dropped, but ``X_raw`` is NOT preprocessed: in
    cross-validation every fold is its own train/test split, so imputation
    and scaling must be fitted inside each fold on that fold's training
    rows — fitting them once on all ``n`` rows would leak every test fold
    into its training fold. Score models with ``cross_val_auc_split_first``.
    """
    loader = MLFPDataLoader()
    credit = (
        loader.load("mlfp02", "sg_credit_scoring.parquet")
        .drop(["customer_id", "future_default_indicator"])
        .sample(n=n, seed=SEED)
    )
    y = credit["default"].to_numpy().astype(int)
    X_raw = credit.drop("default")
    return X_raw, y, X_raw.columns


def cross_val_auc_split_first(
    model: Any, X_raw: pl.DataFrame, y: np.ndarray, *, cv: Any
) -> np.ndarray:
    """ROC-AUC per fold with the preprocessing re-fitted inside every fold.

    Plays the role of ``cross_val_score(model, X, y, cv=cv,
    scoring="roc_auc")`` for RAW features: in each fold the
    ``PreprocessingPipeline`` is fitted on the training rows only and
    applied to the held-out rows, then a fresh clone of ``model`` is fitted
    and scored. Returns one AUC per fold.
    """
    frame = X_raw.with_columns(pl.Series("default", y))
    scores: list[float] = []
    for tr_idx, te_idx in cv.split(np.zeros(len(y)), y):
        prep = preprocess_train_test(
            frame[tr_idx.tolist()],
            frame[te_idx.tolist()],
            "default",
            seed=SEED,
            **CREDIT_DEFAULT_PREPROCESSING,
        )
        features = [c for c in prep.train_data.columns if c != "default"]
        X_tr, y_tr, _ = to_sklearn_input(
            prep.train_data, feature_columns=features, target_column="default"
        )
        X_te, y_te, _ = to_sklearn_input(
            prep.test_data, feature_columns=features, target_column="default"
        )
        fitted = clone(model).fit(X_tr, y_tr.astype(int))
        scores.append(float(roc_auc_score(y_te.astype(int), fitted.predict_proba(X_te)[:, 1])))
    return np.array(scores)


ICU_CV_FEATURES: list[str] = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "bmi",
    "insurance",
    "ethnicity",
    "diagnosis",
    "icu_type",
]


def load_icu_admissions_for_cv() -> dict[str, Any]:
    """ICU admissions with REAL time order and REAL repeated-patient groups.

    One row per admission (8,000 admissions of ~4,000 patients), sorted by
    ``admit_time`` so that TimeSeriesSplit's "earlier rows train, later
    rows test" really means "past predicts future". Features are only
    things known AT admission (patient demographics, diagnosis, ICU type);
    the target is ``los_days`` (length of stay).

    Returns a dict with X, y, groups (patient_id per row), admit_time
    (polars Series), feature_names.
    """
    loader = MLFPDataLoader()
    admissions = loader.load("mlfp02", "icu_admissions.parquet")
    patients = loader.load("mlfp02", "icu_patients.parquet")
    frame = (
        admissions.join(patients, on="patient_id", how="left")
        .with_columns(
            pl.col("admit_time").str.to_datetime("%Y-%m-%d %H:%M:%S")
        )
        .sort("admit_time")
    )
    encoded = frame.with_columns(
        [
            pl.col(c).rank("dense").cast(pl.Float64)
            for c in ICU_CV_FEATURES
            if frame.schema[c] == pl.String
        ]
    ).with_columns(
        [pl.col(c).cast(pl.Float64).fill_null(pl.col(c).median()) for c in ICU_CV_FEATURES]
    )
    return {
        "X": encoded.select(ICU_CV_FEATURES).to_numpy(),
        "y": encoded["los_days"].to_numpy(),
        "groups": encoded["patient_id"].to_numpy(),
        "admit_time": encoded["admit_time"],
        "feature_names": list(ICU_CV_FEATURES),
    }


# ════════════════════════════════════════════════════════════════════════
# SYNTHETIC 1D PROBLEM — for bias/variance and polynomial fits
# ════════════════════════════════════════════════════════════════════════


def sine_truth(x: np.ndarray) -> np.ndarray:
    """The noiseless generating function f(x) = sin(2πx)."""
    return np.sin(2 * np.pi * np.asarray(x).ravel())


def sample_sine_training_set(
    n: int,
    noise_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ONE fresh training set (x ~ U[0,1], y = sin(2πx) + ε).

    The bias-variance decomposition is defined over many independent
    training sets drawn from the same process — this is how we draw them.
    """
    x = rng.uniform(0, 1, n)
    y = sine_truth(x) + rng.normal(0, noise_sigma, n)
    return x.reshape(-1, 1), y


def make_sine_dataset(
    n_train: int = 40,
    n_test: int = 1000,
    noise_sigma: float = 0.2,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a 1D noisy-sine regression problem.

    Returns:
        x_train_2d (n_train, 1), y_train (n_train,),
        x_test_2d (n_test, 1),   y_test  (n_test,),
        noise_variance (float, σ² — the irreducible error floor)

    The true function is ``y = sin(2πx) + ε`` with ε ~ N(0, σ²).
    A small training set (40 points) makes overfitting visible; a large,
    independently drawn test set (1,000 points) makes the test MSE a
    stable estimate rather than the noise of a handful of points.
    The noise variance is returned so callers can use it in the
    bias-variance decomposition (σ² is the "irreducible noise" term).
    """
    rng = np.random.default_rng(seed)
    x_train, y_train = sample_sine_training_set(n_train, noise_sigma, rng)
    x_test, y_test = sample_sine_training_set(n_test, noise_sigma, rng)
    return x_train, y_train, x_test, y_test, noise_sigma**2


def make_poly_pipeline(degree: int) -> Pipeline:
    """Polynomial-features + scaler + linear-regression pipeline."""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]
    )


# ════════════════════════════════════════════════════════════════════════
# REPORTING HELPERS
# ════════════════════════════════════════════════════════════════════════


def print_header(title: str) -> None:
    """Print a banner so each phase of a technique file is easy to spot."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def format_results_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    """Render a small table of dicts as fixed-width text."""
    header = "  ".join(f"{c:>12}" for c in cols)
    sep = "-" * len(header)
    body = "\n".join(
        "  ".join(
            f"{row[c]:>12.4f}" if isinstance(row[c], float) else f"{row[c]:>12}"
            for c in cols
        )
        for row in rows
    )
    return f"{header}\n{sep}\n{body}"


def save_html_plot(fig: Any, name: str) -> Path:
    """Write a plotly figure into OUTPUT_DIR and return the path."""
    path = OUTPUT_DIR / name
    fig.write_html(str(path))
    return path
