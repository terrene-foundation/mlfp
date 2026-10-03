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
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from kailash_ml import PreprocessingPipeline
from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader

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
    ordinal encoding + median imputation. All regularised models
    REQUIRE normalised features (otherwise the penalty is unevenly
    distributed across the coefficient vector).
    """
    loader = MLFPDataLoader()
    credit = loader.load("mlfp02", "sg_credit_scoring.parquet").drop(
        CREDIT_NON_FEATURE_COLUMNS
    )
    credit = credit.sample(n=n_train + n_test, seed=SEED)

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        data=credit,
        target=CREDIT_TARGET,
        train_size=n_train / (n_train + n_test),
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
