# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 3 — The Classical ML Zoo.

Contains: e-commerce churn data loading, preprocessing, CV strategy,
2D PCA projection for decision boundary plots, model comparison helpers,
and a shared ModelVisualizer-backed plot utility.

Technique-specific code (model fitting, parameter sweeps, from-scratch
Gini, OOB convergence, decision guide) does NOT belong here — it lives
in the per-technique files under modules/mlfp03/solutions/ex_3/.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from kailash_ml import ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment, split_then_preprocess

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

setup_environment()
np.random.seed(42)

# Output directory for comparison artifacts (HTML plots, tables)
OUTPUT_DIR = Path("outputs") / "ex3_model_zoo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# E-commerce churn dataset — Singapore APAC retail churn scenario
DATASET_MODULE = "mlfp03"
DATASET_FILE = "ecommerce_customers.parquet"
TARGET_COL = "churned"

# SVM with RBF kernel is O(n²) — cap the training set so every technique
# in the zoo fits in a few seconds on a laptop.
SUBSAMPLE_N = 5000
RANDOM_SEED = 42

# Columns that are not model inputs:
#   customer_id                       — row identifier
#   review_text, product_categories   — free text / multi-valued strings
#   days_since_last_order             — DEFINES the label: the dataset marks
#       a customer churned exactly when days_since_last_order > 180, so
#       keeping it lets a depth-1 tree score 100% (target leakage).
# customer_tenure_days stays in: it is known at prediction time, but note it
# is partly built from recency (tenure >= recency), which is why it ends up
# being the strongest remaining signal.
DROP_COLS = ["customer_id", "review_text", "product_categories", "days_since_last_order"]


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING + PREPROCESSING
# ════════════════════════════════════════════════════════════════════════


def load_ecommerce_churn() -> pl.DataFrame:
    """Load the Singapore e-commerce churn dataset (polars DataFrame).

    Drops ID/text columns and the label-defining recency column, then
    subsamples for SVM tractability.
    """
    loader = MLFPDataLoader()
    df = loader.load(DATASET_MODULE, DATASET_FILE)
    df = df.sample(n=min(SUBSAMPLE_N, df.height), seed=RANDOM_SEED)
    keep = [c for c in df.columns if c not in DROP_COLS]
    return df.select(keep)


def build_train_test_split() -> dict[str, Any]:
    """Return a fully prepared dict: X_train, X_test, y_train, y_test, feature_names, cv.

    Holds out a stratified 20% test split FIRST, then fits kailash_ml's
    PreprocessingPipeline (z-score normalisation, ordinal categorical
    encoding) on the training rows only. Every technique file calls this
    so all models share identical folds and identical preprocessing.

    Also returns the two "do-nothing" reference points every model must be
    judged against, because churners are the MAJORITY class (~74%):
      majority_accuracy — test accuracy of always predicting the majority class
      majority_f1       — churn-class F1 of always predicting "churned"
    A model whose accuracy does not clear ``majority_accuracy`` has learned
    nothing useful, however good its F1 looks.
    """
    df = load_ecommerce_churn()

    # Split FIRST, then fit imputation/encoding on the training rows only:
    # PreprocessingPipeline.setup() on the whole frame would fit them on the
    # test rows too (it splits only after fitting).
    result = split_then_preprocess(
        df,
        target=TARGET_COL,
        test_size=0.2,
        seed=RANDOM_SEED,
        normalize=True,
        normalize_method="zscore",
        categorical_encoding="ordinal",
        imputation_strategy="median",
    )

    feature_cols = [c for c in result.train_data.columns if c != TARGET_COL]
    X_train, y_train, col_info = to_sklearn_input(
        result.train_data,
        feature_columns=feature_cols,
        target_column=TARGET_COL,
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_cols,
        target_column=TARGET_COL,
    )
    feature_names = col_info["feature_columns"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    majority_class = int(np.mean(y_train) >= 0.5)
    always_majority = np.full_like(y_test, majority_class)
    always_churn = np.ones_like(y_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "cv": cv,
        "churn_rate": float(np.mean(y_train)),
        "majority_accuracy": float(accuracy_score(y_test, always_majority)),
        "majority_f1": float(f1_score(y_test, always_churn)),
    }


# ════════════════════════════════════════════════════════════════════════
# 2D PCA PROJECTION — shared so every technique plots on the same axes
# ════════════════════════════════════════════════════════════════════════


def project_2d(X_train: np.ndarray, X_test: np.ndarray) -> dict[str, Any]:
    """Fit PCA(2) on X_train and project both train and test.

    Returns {X_train_2d, X_test_2d, explained_variance, pca}.
    """
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)
    return {
        "X_train_2d": X_train_2d,
        "X_test_2d": X_test_2d,
        "explained_variance": pca.explained_variance_ratio_,
        "pca": pca,
    }


# ════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION HELPER — keep every parameter sweep one line
# ════════════════════════════════════════════════════════════════════════


def cv_scores(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
) -> dict[str, float]:
    """One cross-validation pass, three metrics.

    Returns mean accuracy, mean churn-class F1, mean ROC AUC, the standard
    deviation of ROC AUC across folds, and the mean fit time (seconds).

    Hyperparameters in this exercise are chosen by ROC AUC: it measures how
    well the model RANKS churners above retained customers, so it cannot be
    gamed by predicting the majority class for everyone (which already
    scores ~74% accuracy and ~0.85 F1 on this data).
    """
    r = cross_validate(estimator, X, y, cv=cv, scoring=("accuracy", "f1", "roc_auc"))
    return {
        "accuracy": float(r["test_accuracy"].mean()),
        "f1": float(r["test_f1"].mean()),
        "auc_roc": float(r["test_roc_auc"].mean()),
        "auc_roc_std": float(r["test_roc_auc"].std()),
        "fit_time": float(r["fit_time"].mean()),
    }


# ════════════════════════════════════════════════════════════════════════
# EVALUATION — train on full set, measure timing, return pred/prob/metrics
# ════════════════════════════════════════════════════════════════════════


def fit_and_evaluate(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> dict[str, Any]:
    """Fit, predict, score, and time a single model.

    Returns a dict with keys: name, model, pred, prob, train_time,
    accuracy, f1, auc_roc.
    """
    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        prob = estimator.predict_proba(X_test)[:, 1]
    else:
        # Margin-based models (e.g. SVC without probability=True) expose an
        # uncalibrated decision score; it still ranks, so AUC is valid.
        prob = estimator.decision_function(X_test)

    return {
        "name": name,
        "model": estimator,
        "pred": pred,
        "prob": prob,
        "train_time": float(train_time),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auc_roc": float(roc_auc_score(y_test, prob)),
    }


def print_classification_report(y_test: np.ndarray, pred: np.ndarray) -> None:
    """Print sklearn classification report with churn-friendly target names."""
    print(
        classification_report(
            y_test,
            pred,
            target_names=["Retained", "Churned"],
        )
    )


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Plotly via kailash_ml.ModelVisualizer
# ════════════════════════════════════════════════════════════════════════


def get_visualizer() -> ModelVisualizer:
    """Return a ModelVisualizer instance (polars-native plots)."""
    return ModelVisualizer()


def save_metric_comparison(
    metric_dict: dict[str, dict[str, float]], fname: str
) -> Path:
    """Save a metric_comparison plot to OUTPUT_DIR/fname and return the path."""
    viz = get_visualizer()
    fig = viz.metric_comparison(metric_dict)
    fig.update_layout(title="Classical ML Zoo — Performance Comparison")
    out = OUTPUT_DIR / fname
    fig.write_html(str(out))
    return out


# ════════════════════════════════════════════════════════════════════════
# DECISION BOUNDARY MESH — shared helper so every technique file uses
# the same axes, grid, and figure style.
# ════════════════════════════════════════════════════════════════════════


def decision_boundary_mesh(
    X_2d: np.ndarray,
    step: float = 0.1,
    pad: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xx, yy) meshgrid covering the 2D PCA projection."""
    x_min, x_max = X_2d[:, 0].min() - pad, X_2d[:, 0].max() + pad
    y_min, y_max = X_2d[:, 1].min() - pad, X_2d[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step),
    )
    return xx, yy


def save_decision_boundaries(
    panels: dict[str, np.ndarray],
    xx: np.ndarray,
    yy: np.ndarray,
    X_2d: np.ndarray,
    y: np.ndarray,
    fname: str,
    title: str,
    max_points: int = 1500,
) -> Path:
    """Render one decision-boundary panel per model and save as HTML.

    ``panels`` maps a panel title to the model's predicted class over the
    mesh (``Z`` with shape ``xx.shape``). Each panel shades the predicted
    region (blue = retained, red = churned) and overlays a random sample of
    training customers coloured by their TRUE label, so you can see where
    each model's boundary agrees or disagrees with the data.
    """
    n = len(panels)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(panels))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(y), size=min(max_points, len(y)), replace=False)
    colours = np.where(y[idx] == 1, "#c0392b", "#2471a3")
    for i, (name, Z) in enumerate(panels.items()):
        r, c = i // cols + 1, i % cols + 1
        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                colorscale=[[0, "#aed6f1"], [1, "#f5b7b1"]],
                showscale=False,
                opacity=0.6,
                contours={"start": 0, "end": 1, "size": 0.5},
                name=name,
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=X_2d[idx, 0],
                y=X_2d[idx, 1],
                mode="markers",
                marker={"color": colours, "size": 4, "opacity": 0.6},
                showlegend=False,
                name="customers (red = churned)",
            ),
            row=r,
            col=c,
        )
    fig.update_layout(title=title, height=380 * rows, width=420 * cols)
    fig.update_xaxes(title_text="PC1")
    fig.update_yaxes(title_text="PC2")
    out = OUTPUT_DIR / fname
    fig.write_html(str(out))
    return out


def save_sweep_plot(
    x_values: list[Any],
    series: dict[str, list[float]],
    x_label: str,
    title: str,
    fname: str,
    log_x: bool = False,
) -> Path:
    """Plot hyperparameter-sweep curves against the REAL hyperparameter values.

    (``ModelVisualizer.training_history`` always uses 1, 2, 3, ... on the
    x-axis, which mislabels a sweep over C = 0.01 ... 100 or k = 1 ... 101.)
    """
    fig = go.Figure()
    for label, values in series.items():
        fig.add_trace(go.Scatter(x=x_values, y=values, mode="lines+markers", name=label))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="CV score")
    if log_x:
        fig.update_xaxes(type="log")
    out = OUTPUT_DIR / fname
    fig.write_html(str(out))
    return out


# ════════════════════════════════════════════════════════════════════════
# SINGAPORE E-COMMERCE CHURN — business-impact constants
# ════════════════════════════════════════════════════════════════════════
# Illustrative round numbers for a mid-market regional e-commerce platform,
# used by the "Apply" phases. They are teaching assumptions, not figures
# taken from any company's reports — replace them with your own business's
# numbers when you reuse this analysis.

AVG_CUSTOMER_LIFETIME_VALUE_SGD = 420.0  # avg 12-month CLV per retained SG customer
RETENTION_OFFER_COST_SGD = 18.0  # targeted promo cost per flagged customer
MONTHLY_ACTIVE_CUSTOMERS = 250_000  # typical mid-market SG e-commerce platform
ANNUAL_CHURN_BASELINE = 0.22  # assumed annual churn without intervention


def churn_saved_dollars(true_positives: int) -> float:
    """Dollar value of correctly identified churners (retention offer accepted).

    Assumes a 40% offer-acceptance rate and the retained lifetime value
    net of offer cost (illustrative assumptions — see constants above).
    """
    accept_rate = 0.40
    net_value_per_save = AVG_CUSTOMER_LIFETIME_VALUE_SGD - RETENTION_OFFER_COST_SGD
    return round(true_positives * accept_rate * net_value_per_save, 2)
