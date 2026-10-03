# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 4 — Anomaly Detection and Ensembles.

Contains: data loading (Singapore credit applications + an injected-anomaly
benchmark with an independent label), feature standardisation, score
normalisation helpers, metric reporting, visualisation shortcuts.

Technique-specific code (Z-score thresholding, Isolation Forest fit, LOF
neighbour count, blend weights, EnsembleEngine calls) does NOT belong here —
it lives in the per-technique files in `modules/mlfp04/solutions/ex_4/`.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from kailash_ml import ExperimentTracker
from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
np.random.seed(42)

OUTPUT_DIR = Path("outputs") / "ex4_anomaly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# DATA — Credit applications with an injected-anomaly benchmark
# ════════════════════════════════════════════════════════════════════════
# Why not just threshold a column and call it "fraud"? Because then the
# label is a function of an input feature and every AUC simply measures
# how much a detector looks at that one column (circular evaluation).
#
# Instead we follow the standard benchmark recipe for unsupervised anomaly
# detection (e.g. ADBench, Han et al., NeurIPS 2022): take REAL records as
# the normal population and inject anomalies of known TYPES, so the label
# comes from the injection process, never from a feature threshold:
#
#   global      — a real application with ONE field pushed 5-8 standard
#                 deviations beyond the mean (fat-finger entry, inflated
#                 declared balance). Extreme on a single feature.
#   dependency  — every field copied from a DIFFERENT real application
#                 (a "synthetic identity" stitched from real fragments).
#                 Each value is individually plausible; the COMBINATION is
#                 not (e.g. employment_years vs months_employed disagree).
#   clustered   — a tight group of near-identical applications (a
#                 coordinated application ring), shifted +3 std on three
#                 fields. Rare as a group, but dense locally.
#
# The normal rows are 20,000 real records from the course's Singapore
# credit-scoring dataset (mlfp02/sg_credit_scoring.parquet). Its real
# `default` outcome is kept (not as a feature) for a reality check:
# "statistically unusual" and "will default" are different questions.

N_NORMAL = 20_000
N_GLOBAL = 80
N_DEPENDENCY = 80
N_CLUSTERED = 40
ANOMALY_TYPES = ("global", "dependency", "clustered")
LABEL_COL = "is_anomaly"

# Numeric application fields with no nulls. Excluded on purpose:
#   income_sgd / loan_to_value — 30% / 65% missing
#   cpf_monthly_contribution   — near-constant (one value for most rows)
#   coe_vehicle_owner          — binary flag dominates Euclidean distance
#   future_default_indicator   — leaks the outcome
#   default                    — the outcome itself (reality check only)
FEATURE_COLS = [
    "age",
    "employment_years",
    "months_employed",
    "credit_utilization",
    "avg_balance_utilization",
    "num_credit_lines",
    "credit_age_years",
    "num_hard_inquiries",
    "payment_history_score",
    "num_late_payments",
    "revolving_balance",
    "installment_balance",
    "loan_amount_sgd",
    "monthly_installment",
    "num_dependents",
    "debt_to_income",
    "savings_balance",
    "checking_balance",
    "previous_defaults",
    "property_value_sgd",
]


def load_anomaly_frame(seed: int = 42) -> pl.DataFrame:
    """Return real credit applications plus injected, typed anomalies.

    Columns: FEATURE_COLS, `is_anomaly` (0/1), `anomaly_type`
    ("normal" / "global" / "dependency" / "clustered") and `default`
    (the real outcome for normal rows; null for injected rows). Rows are
    shuffled so injected anomalies are spread across the frame.
    """
    loader = MLFPDataLoader()
    raw = loader.load("mlfp02", "sg_credit_scoring.parquet")
    base = raw.sample(N_NORMAL, seed=seed).select(FEATURE_COLS + ["default"])
    B = base.select(FEATURE_COLS).to_numpy().astype(np.float64)
    mu, sd = B.mean(axis=0), B.std(axis=0)
    rng = np.random.default_rng(seed)
    n_feat = len(FEATURE_COLS)

    # global: one field pushed far into the upper tail
    G = B[rng.choice(len(B), N_GLOBAL, replace=False)].copy()
    for r in range(N_GLOBAL):
        j = int(rng.integers(n_feat))
        G[r, j] = mu[j] + rng.uniform(5.0, 8.0) * sd[j]

    # dependency: each field drawn from a different real application
    D = np.column_stack(
        [B[rng.integers(len(B), size=N_DEPENDENCY), j] for j in range(n_feat)]
    )

    # clustered: tight group around a shifted real application
    centre = B[int(rng.integers(len(B)))].copy()
    shifted = rng.choice(n_feat, 3, replace=False)
    centre[shifted] += 3.0 * sd[shifted]
    C = centre + rng.normal(0.0, 0.05, (N_CLUSTERED, n_feat)) * sd

    injected = np.vstack([G, D, C])
    types = ["global"] * N_GLOBAL + ["dependency"] * N_DEPENDENCY + [
        "clustered"
    ] * N_CLUSTERED

    normal_part = base.with_columns(
        pl.lit(0, dtype=pl.Int64).alias(LABEL_COL),
        pl.lit("normal").alias("anomaly_type"),
    )
    injected_part = pl.from_numpy(injected, schema=FEATURE_COLS).with_columns(
        pl.lit(None, dtype=pl.Int64).alias("default"),
        pl.lit(1, dtype=pl.Int64).alias(LABEL_COL),
        pl.Series("anomaly_type", types),
    )
    frame = pl.concat(
        [normal_part.with_columns(pl.col(FEATURE_COLS).cast(pl.Float64)), injected_part],
        how="vertical",
    )
    return frame.sample(fraction=1.0, shuffle=True, seed=seed)


def build_features(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Standardise FEATURE_COLS and return (X, y, feature_cols).

    Returns standardised X (float64) and the 0/1 `is_anomaly` label. The
    label is used ONLY for evaluation — every detector fits on X alone.
    """
    X, y, _col_info = to_sklearn_input(
        frame,
        feature_columns=FEATURE_COLS,
        target_column=LABEL_COL,
    )
    X_scaled = StandardScaler().fit_transform(X).astype(np.float64)
    return X_scaled, np.asarray(y).astype(int), list(FEATURE_COLS)


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str], pl.DataFrame]:
    """One-call helper: load the frame, build features, return everything."""
    frame = load_anomaly_frame()
    X, y, cols = build_features(frame)
    return X, y, cols, frame


def auc_by_type(frame: pl.DataFrame, scores: np.ndarray) -> dict[str, float]:
    """AUC-ROC of `scores` for each anomaly type vs the normal rows.

    Answers "WHICH kind of anomaly does this detector find?" — 1.0 means
    every anomaly of that type outranks every normal row; 0.5 is chance;
    below 0.5 means the detector ranks that type as MORE normal than the
    real applications.
    """
    types = frame["anomaly_type"].to_numpy()
    scores = np.asarray(scores, dtype=np.float64)
    out: dict[str, float] = {}
    for t in ANOMALY_TYPES:
        mask = (types == t) | (types == "normal")
        out[t] = float(roc_auc_score((types[mask] == t).astype(int), scores[mask]))
    return out


def print_auc_by_type(
    name: str, frame: pl.DataFrame, scores: np.ndarray
) -> dict[str, float]:
    """Compute per-type AUC, print it on one line, and return the dict."""
    by_type = auc_by_type(frame, scores)
    parts = "  ".join(f"{t}={v:.3f}" for t, v in by_type.items())
    print(f"  {name:<24} per-type AUC: {parts}")
    return by_type


def default_reality_check(frame: pl.DataFrame, scores: np.ndarray) -> float:
    """AUC of an anomaly score against the REAL `default` outcome.

    Computed on the real (non-injected) applications only.
    """
    mask = (frame["anomaly_type"] == "normal").to_numpy()
    defaults = frame["default"].to_numpy()[mask].astype(int)
    return float(roc_auc_score(defaults, np.asarray(scores)[mask]))


# ════════════════════════════════════════════════════════════════════════
# SCORE HELPERS
# ════════════════════════════════════════════════════════════════════════
# Anomaly detectors emit scores on wildly different scales. Normalising to
# [0, 1] (or to a rank) is what makes blending across methods possible.


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise an anomaly score array to [0, 1]."""
    scores = np.asarray(scores, dtype=np.float64)
    span = scores.max() - scores.min()
    return (scores - scores.min()) / (span + 1e-10)


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Convert an anomaly score array to percentile ranks in [0, 1]."""
    from scipy.stats import rankdata

    return rankdata(np.asarray(scores, dtype=np.float64)) / len(scores)


def score_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return AUC-ROC and average precision (AUC-PR) for an anomaly score."""
    return {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "avg_precision": float(average_precision_score(y_true, scores)),
    }


def print_metrics(
    name: str, y_true: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Compute metrics, print them on one line, and return the dict."""
    m = score_metrics(y_true, scores)
    print(f"  {name:<24} AUC-ROC={m['auc_roc']:.4f}  " f"AP={m['avg_precision']:.4f}")
    return m


def precision_at_recall(
    y_true: np.ndarray, scores: np.ndarray, target_recall: float
) -> tuple[float, float]:
    """Return (precision, threshold) at the TIGHTEST point where recall >= target.

    sklearn returns precisions/recalls ordered by ascending threshold, so
    recall decreases as threshold increases. We want the highest threshold
    that still meets the recall target — i.e. the last index where recall
    is still >= the target, which gives the maximum precision for that
    recall level.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    # Drop the sentinel last point (precision=1.0, recall=0.0, no threshold)
    ps = precisions[:-1]
    rs = recalls[:-1]
    ts = thresholds
    mask = rs >= target_recall
    if not mask.any():
        return float(ps[0]), float(ts[0])
    # The tightest threshold satisfying the recall target is the largest
    # index where mask is True (thresholds are ascending).
    idx = int(np.where(mask)[0][-1])
    return float(ps[idx]), float(ts[idx])


def split_review_holdout(
    n_rows: int, review_fraction: float = 0.3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into a labelled 'review sample' and a holdout.

    Anything tuned with labels (blend weights, a supervised second stage)
    is fitted on the review sample and evaluated on the holdout, so the
    reported numbers are not scored on the rows that tuned them.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    cut = int(n_rows * review_fraction)
    return np.sort(order[:cut]), np.sort(order[cut:])


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════════


def write_comparison_chart(
    comparison: dict[str, dict[str, float]], filename: str
) -> Path:
    """Render a kailash-ml ModelVisualizer metric_comparison chart to HTML."""
    from kailash_ml import ModelVisualizer

    viz = ModelVisualizer()
    fig = viz.metric_comparison(comparison)
    fig.update_layout(title="Anomaly Detection Method Comparison")
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path


def write_roc_chart(
    y_true: np.ndarray, scores: np.ndarray, name: str, filename: str
) -> Path:
    """Render a ROC curve for a single detector."""
    from kailash_ml import ModelVisualizer

    viz = ModelVisualizer()
    fig = viz.roc_curve(y_true, scores)
    fig.update_layout(title=f"ROC — {name}")
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path


def write_monitoring_chart(anomaly_rates: list[float], filename: str) -> Path:
    """Render an anomaly-rate-over-time chart for production monitoring."""
    from kailash_ml import ModelVisualizer

    viz = ModelVisualizer()
    fig = viz.training_history(
        {"Anomaly Rate %": [r * 100 for r in anomaly_rates]},
        x_label="Time Window",
    )
    fig.update_layout(title="Anomaly Rate Over Time (Production Monitoring)")
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path


# ════════════════════════════════════════════════════════════════════════
# KAILASH-ML EXPERIMENT TRACKER — shared by every anomaly technique
# ════════════════════════════════════════════════════════════════════════
# Every M4 ex_4 lesson logs its sweep + final-fit metrics to a single
# SQLite store so students can compare detectors across the lesson group
# (statistical, isolation forest, LOF, ensemble) after class. Mirrors the
# m4_clustering_zoo / m4_dimreduction_zoo experiments from ex_1 / ex_3.
#
# IMPORTANT: this store is SEPARATE from the clustering and dim-reduction
# stores. Per the SQLite contention trap (see .session-notes), running
# two ExperimentTracker writers against the same SQLite file concurrently
# triggers `disk I/O error`. Each exercise group gets its own DB.

ANOMALY_DB = "sqlite:///mlfp04_ex4_anomaly.db"
ANOMALY_EXPERIMENT_NAME = "m4_anomaly_zoo"


def _finite(x: float) -> float:
    """Tracker rejects NaN/inf via MetricValueError; coerce to 0.0.
    Anomaly metrics like AUC-ROC/AP can NaN on degenerate label arrays
    (e.g., all-positive or all-negative slices in a sweep)."""
    return float(x) if x == x and x not in (float("inf"), float("-inf")) else 0.0


async def _setup_engines_async() -> tuple[ExperimentTracker, str]:
    """Open the anomaly-detection ExperimentTracker."""
    tracker = await ExperimentTracker.create(store_url=ANOMALY_DB)
    return tracker, ANOMALY_EXPERIMENT_NAME


def setup_engines() -> tuple[ExperimentTracker, str]:
    """Sync wrapper. Returns (tracker, experiment_name)."""
    return asyncio.run(_setup_engines_async())


def teardown_engines(tracker: ExperimentTracker) -> None:
    """Drain the aiosqlite worker threads before the script returns.

    kailash's AsyncSQLitePool spawns NON-DAEMON aiosqlite worker threads on
    first pool use. Python 3.13's ``Py_FinalizeEx`` joins non-daemon threads
    BEFORE running ``atexit`` handlers, so an atexit-based close runs too
    late — the interpreter hangs forever in ``wait_for_thread_shutdown``
    waiting on workers stuck in ``queue.get()``.

    Solutions MUST call ``teardown_engines(tracker)`` after the REFLECTION
    block. See ``rules/patterns.md`` § "Async Resource Cleanup".
    """
    asyncio.run(tracker.close())


async def _track_run_async(
    tracker: ExperimentTracker,
    exp_name: str,
    run_name: str,
    params: dict[str, Any],
    scalar_metrics: dict[str, float],
    series_metrics: dict[str, list[float]] | None = None,
) -> None:
    """Log one lesson's run: scalar metrics + optional per-step series."""
    async with tracker.track(experiment=exp_name, run_name=run_name) as run:
        await run.log_params({k: str(v) for k, v in params.items()})
        for name, value in scalar_metrics.items():
            await run.log_metric(name, float(value))
        if series_metrics:
            for name, values in series_metrics.items():
                for step, value in enumerate(values, start=1):
                    await run.log_metric(name, float(value), step=step)


def track_run(
    tracker: ExperimentTracker,
    exp_name: str,
    run_name: str,
    params: dict[str, Any],
    scalar_metrics: dict[str, float],
    series_metrics: dict[str, list[float]] | None = None,
) -> None:
    """Sync wrapper for logging a single technique's run."""
    asyncio.run(
        _track_run_async(
            tracker, exp_name, run_name, params, scalar_metrics, series_metrics
        )
    )
