# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 5 — Linear Regression.

Contains: HDB resale data loading, feature engineering, OLS fitting utilities,
diagnostic helpers, and visualisation helpers. Technique-specific code (the
derivation walk-throughs, the WLS weight construction, the polynomial/dummy
matrices) lives in the per-technique files, not here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.data_loader import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

NUMERIC_FEATURES: list[str] = [
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
]
TARGET: str = "resale_price"
BASE_FLAT_TYPE: str = "3 ROOM"

OUTPUT_DIR = Path("outputs") / "ex5_linear_regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — HDB resale flat transactions
# ════════════════════════════════════════════════════════════════════════


def load_hdb_clean() -> pl.DataFrame:
    """Load HDB resale data, engineer numeric features, drop nulls.

    Returns a polars DataFrame with columns:
      - floor_area_sqm (Float)
      - storey_midpoint (Float, midpoint of '07 TO 09' range)
      - remaining_lease_years (Float, 99 - years_elapsed_since_commence)
      - resale_price (target, SGD)
      - flat_type (categorical)
      - town (categorical)
    """
    loader = MLFPDataLoader()
    hdb = loader.load("mlfp01", "hdb_resale.parquet")

    hdb = hdb.with_columns(
        pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
    )
    hdb_recent = hdb.filter(pl.col("transaction_date") >= pl.date(2020, 1, 1))

    hdb_recent = hdb_recent.with_columns(
        (
            (
                pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
                + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
            )
            / 2
        ).alias("storey_midpoint"),
        (99 - (pl.col("transaction_date").dt.year() - pl.col("lease_commence_date")))
        .cast(pl.Float64)
        .alias("remaining_lease_years"),
    )

    return hdb_recent.drop_nulls(
        subset=[*NUMERIC_FEATURES, TARGET],
    )


def build_design_matrix(
    df: pl.DataFrame, features: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X_with_intercept, y, feature_names) from a cleaned HDB frame."""
    features = features or NUMERIC_FEATURES
    y = df[TARGET].to_numpy().astype(np.float64)
    X_raw = df.select(features).to_numpy().astype(np.float64)
    n_obs = X_raw.shape[0]
    X = np.column_stack([np.ones(n_obs), X_raw])
    feature_names = ["intercept"] + list(features)
    return X, y, feature_names


# ════════════════════════════════════════════════════════════════════════
# OLS CORE — the workhorse every technique reuses
# ════════════════════════════════════════════════════════════════════════


def fit_ols(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit OLS via the normal equation and return core statistics.

    Returns a dict with keys: beta, y_hat, residuals, XtX_inv, SSR, SST, SSE,
    R2, adj_R2, sigma_hat, se_beta, t_stats, p_values, f_stat, f_p_value, n, k.
    """
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y

    y_hat = X @ beta
    residuals = y - y_hat

    SSR = float(np.sum(residuals**2))
    SST = float(np.sum((y - y.mean()) ** 2))
    SSE = float(np.sum((y_hat - y.mean()) ** 2))

    sigma_sq = SSR / (n - k)
    sigma_hat = float(np.sqrt(sigma_sq))
    se_beta = np.sqrt(sigma_sq * np.diag(XtX_inv))
    t_stats = beta / se_beta
    # sf (survival function) keeps precision in the far tail; 1 - cdf
    # rounds to exactly 0 once the tail drops below ~1e-16.
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=n - k)

    r2 = 1 - SSR / SST
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
    f_stat = (SSE / (k - 1)) / (SSR / (n - k))
    f_p = stats.f.sf(f_stat, dfn=k - 1, dfd=n - k)

    return {
        "beta": beta,
        "y_hat": y_hat,
        "residuals": residuals,
        "XtX_inv": XtX_inv,
        "SSR": SSR,
        "SST": SST,
        "SSE": SSE,
        "R2": float(r2),
        "adj_R2": float(adj_r2),
        "sigma_hat": sigma_hat,
        "se_beta": se_beta,
        "t_stats": t_stats,
        "p_values": p_values,
        "f_stat": float(f_stat),
        "f_p_value": float(f_p),
        "n": int(n),
        "k": int(k),
    }


def format_p_value(p: float) -> str:
    """Render a p-value for printing, including the underflow case.

    Returns e.g. "= 3.20e-05", or "< 1e-300" when the tail probability is
    smaller than the smallest representable double and evaluates to 0.
    """
    return f"= {p:.2e}" if p > 0 else "< 1e-300"


# ════════════════════════════════════════════════════════════════════════
# K-FOLD CROSS-VALIDATION — fold assignment, from scratch
# ════════════════════════════════════════════════════════════════════════


def kfold_indices(
    n: int, k: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return k (train_idx, test_idx) pairs from a seeded shuffle.

    Fold sizes differ by at most one row. The shuffle makes folds
    exchangeable — required because the HDB frame is sorted by month and
    an unshuffled split would make the last fold a different time period
    (that is a DIFFERENT technique: out-of-time validation, ex_8).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        pairs.append((train_idx, test_idx))
    return pairs


def ols_r2_on(
    X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> float:
    """Fit OLS on train_idx, return R² computed on test_idx (out-of-sample)."""
    beta = np.linalg.lstsq(X[train_idx], y[train_idx], rcond=None)[0]
    y_true = y[test_idx]
    y_pred = X[test_idx] @ beta
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


# ════════════════════════════════════════════════════════════════════════
# GEO FEATURES — town centroids and haversine distance to the CBD
# ════════════════════════════════════════════════════════════════════════
#
# The resale file has no coordinates, so locations come from a lookup of
# approximate TOWN-CENTRE coordinates (public geographic facts; teaching
# proxy — a production pipeline would geocode block + street via OneMap).
# Distance is to Raffles Place, the centre of the Central Business
# District.

CBD_RAFFLES_PLACE: tuple[float, float] = (1.2844, 103.8510)

TOWN_CENTROIDS: dict[str, tuple[float, float]] = {
    "ANG MO KIO": (1.3691, 103.8454),
    "BEDOK": (1.3236, 103.9273),
    "BISHAN": (1.3526, 103.8352),
    "BOON LAY": (1.3366, 103.7039),
    "BUKIT BATOK": (1.3490, 103.7496),
    "BUKIT MERAH": (1.2819, 103.8239),
    "BUKIT PANJANG": (1.3774, 103.7719),
    "BUKIT TIMAH": (1.3294, 103.8021),
    "CENTRAL AREA": (1.2903, 103.8520),
    "CHOA CHU KANG": (1.3840, 103.7470),
    "CLEMENTI": (1.3162, 103.7649),
    "GEYLANG": (1.3201, 103.8871),
    "HOUGANG": (1.3612, 103.8863),
    "JURONG EAST": (1.3329, 103.7436),
    "JURONG WEST": (1.3404, 103.7090),
    "KALLANG/WHAMPOA": (1.3100, 103.8651),
    "MARINE PARADE": (1.3017, 103.9057),
    "PASIR RIS": (1.3721, 103.9474),
    "PUNGGOL": (1.3984, 103.9072),
    "QUEENSTOWN": (1.2942, 103.7861),
    "SEMBAWANG": (1.4491, 103.8185),
    "SENGKANG": (1.3868, 103.8914),
    "SERANGOON": (1.3554, 103.8679),
    "TAMPINES": (1.3496, 103.9568),
    "TOA PAYOH": (1.3343, 103.8563),
    "WOODLANDS": (1.4382, 103.7890),
    "YISHUN": (1.4304, 103.8354),
}


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float
) -> np.ndarray:
    """Great-circle distance in km (mean Earth radius 6371 km)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def add_geo_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add town centroid lat/lon and haversine distance-to-CBD (km).

    Rows whose town is missing from the centroid lookup become null and
    are dropped by the model frame's drop_nulls downstream.
    """
    towns = pl.DataFrame(
        {
            "town": list(TOWN_CENTROIDS.keys()),
            "town_lat": [c[0] for c in TOWN_CENTROIDS.values()],
            "town_lon": [c[1] for c in TOWN_CENTROIDS.values()],
        }
    )
    out = df.join(towns, on="town", how="left")
    lat = out["town_lat"].to_numpy().astype(np.float64)
    lon = out["town_lon"].to_numpy().astype(np.float64)
    dist = haversine_km(lat, lon, *CBD_RAFFLES_PLACE)
    return out.with_columns(
        pl.Series("dist_to_cbd_km", dist).cast(pl.Float64)
    )


# ════════════════════════════════════════════════════════════════════════
# EXPERIMENT TRACKING — kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════

TRACKER_STORE_URL = (
    f"sqlite:///{(OUTPUT_DIR / 'experiments.db').resolve().as_posix()}"
)


def track_train_run(
    experiment: str,
    run_name: str,
    params: dict[str, str],
    metrics: dict[str, float],
) -> str:
    """Log one Train-phase run to ExperimentTracker (sync wrapper).

    Returns the run_id. The tracker is closed in a finally block — kailash-ml
    holds the store connection open until close() is called.
    """
    import asyncio

    async def _log() -> str:
        from kailash_ml import ExperimentTracker

        tracker = await ExperimentTracker.create(store_url=TRACKER_STORE_URL)
        try:
            async with tracker.track(
                experiment=experiment, run_name=run_name
            ) as run:
                await run.log_params(params)
                await run.log_metrics(metrics)
                return run.run_id
        finally:
            await tracker.close()

    return asyncio.run(_log())


def print_coef_table(names: list[str], fit: dict[str, Any]) -> None:
    """Print coefficient / SE / t / p table for an OLS fit."""
    beta = fit["beta"]
    se = fit["se_beta"]
    t = fit["t_stats"]
    p = fit["p_values"]
    print(f"\n{'Feature':<25} {'β':>14} {'SE(β)':>12} {'t':>8} {'p':>10} {'Sig':>4}")
    print("-" * 78)
    for i, name in enumerate(names):
        if p[i] < 0.001:
            sig = "***"
        elif p[i] < 0.01:
            sig = "**"
        elif p[i] < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(
            f"{name:<25} {beta[i]:>14,.2f} {se[i]:>12,.2f} "
            f"{t[i]:>8.2f} {(f'{p[i]:.2e}' if p[i] > 0 else '<1e-300'):>10} {sig:>4}"
        )


# ════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS — VIF, Breusch-Pagan, residual shape
# ════════════════════════════════════════════════════════════════════════


def compute_vif(X_raw: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Variance Inflation Factor for each feature (no intercept column)."""
    n = X_raw.shape[0]
    results: dict[str, float] = {}
    for j in range(X_raw.shape[1]):
        other = [i for i in range(X_raw.shape[1]) if i != j]
        Xo = np.column_stack([np.ones(n), X_raw[:, other]])
        yj = X_raw[:, j]
        beta_j = np.linalg.lstsq(Xo, yj, rcond=None)[0]
        yhat_j = Xo @ beta_j
        ss_res = np.sum((yj - yhat_j) ** 2)
        ss_tot = np.sum((yj - yj.mean()) ** 2)
        r2_j = 1 - ss_res / ss_tot
        results[feature_names[j]] = (
            float(1.0 / (1.0 - r2_j)) if r2_j < 1 else float("inf")
        )
    return results


def breusch_pagan(residuals: np.ndarray, X_raw: np.ndarray) -> tuple[float, float]:
    """Breusch-Pagan test for heteroscedasticity. Returns (BP statistic, p-value)."""
    n = X_raw.shape[0]
    e_sq = residuals**2
    Xbp = np.column_stack([np.ones(n), X_raw])
    beta = np.linalg.lstsq(Xbp, e_sq, rcond=None)[0]
    pred = Xbp @ beta
    sse = np.sum((e_sq - pred) ** 2)
    sst = np.sum((e_sq - e_sq.mean()) ** 2)
    r2 = 1 - sse / sst
    bp_stat = n * r2
    p = stats.chi2.sf(bp_stat, df=X_raw.shape[1])
    return float(bp_stat), float(p)


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — residual plots + actual-vs-predicted
# ════════════════════════════════════════════════════════════════════════


def save_residual_diagnostics(
    y_hat: np.ndarray,
    residuals: np.ndarray,
    feature_col: np.ndarray,
    feature_label: str,
    filename: str,
) -> Path:
    """Four-panel diagnostic figure: residuals vs fitted, histogram, Q-Q, vs feature."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Residuals vs Fitted",
            "Residual Histogram",
            "Q-Q Plot",
            f"Residuals vs {feature_label}",
        ],
    )
    sample = min(3000, len(residuals))
    fig.add_trace(
        go.Scatter(
            x=y_hat[:sample],
            y=residuals[:sample],
            mode="markers",
            marker={"size": 2, "opacity": 0.3},
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, row=1, col=1, line_dash="dash")
    fig.add_trace(go.Histogram(x=residuals, nbinsx=50, name="Residuals"), row=1, col=2)
    sorted_resid = np.sort(residuals)
    step = max(1, len(sorted_resid) // 2000)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_resid)))
    fig.add_trace(
        go.Scatter(
            x=theoretical[::step],
            y=sorted_resid[::step],
            mode="markers",
            marker={"size": 2},
            name="Q-Q",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=feature_col[:sample],
            y=residuals[:sample],
            mode="markers",
            marker={"size": 2, "opacity": 0.3},
            name=f"vs {feature_label}",
        ),
        row=2,
        col=2,
    )
    fig.update_layout(height=600, title="Residual Diagnostics", showlegend=False)
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path


def save_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str,
) -> Path:
    """Actual-vs-predicted scatter with the perfect-prediction diagonal."""
    sample = min(2000, len(y_true))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true[:sample],
            y=y_pred[:sample],
            mode="markers",
            marker={"size": 3, "opacity": 0.4},
            name="Predictions",
        )
    )
    lo, hi = float(y_true.min()), float(y_true.max())
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="Perfect",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Actual Price (SGD)",
        yaxis_title="Predicted Price (SGD)",
        height=500,
    )
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path
