# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 4: Difference-in-Differences Policy Evaluation
(Reference Solution — withheld from students; verified to pass grader.py)
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

Z = stats.norm.ppf(0.975)


def make_dev_panel(seed: int = 0) -> pl.DataFrame:
    """Development panel (illustrative): 6 pre and 6 post quarters, a measure
    that affects the treated region only, true effect -20,000."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(12):
        for g in (0, 1):
            mean = 450_000 + 100_000 * g + 2_000 * t + (-20_000 if (g and t >= 6) else 0)
            for v in rng.normal(mean, 75_000, size=150):
                rows.append((t, g, int(t >= 6), float(v)))
    return pl.DataFrame(rows, schema=["period", "treated", "post", "y"], orient="row")


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    cov = (resid @ resid / (n - k)) * np.linalg.inv(X.T @ X)
    return beta, cov, n - k


def _did(panel: pl.DataFrame) -> tuple[float, float]:
    def cell(g: int, p: int) -> np.ndarray:
        return panel.filter((pl.col("treated") == g) & (pl.col("post") == p))["y"].to_numpy().astype(np.float64)

    cells = [cell(1, 1), cell(1, 0), cell(0, 1), cell(0, 0)]
    att = (cells[0].mean() - cells[1].mean()) - (cells[2].mean() - cells[3].mean())
    se = float(np.sqrt(sum(c.var(ddof=1) / c.size for c in cells)))
    return float(att), se


def _slope_diff(pre: pl.DataFrame) -> tuple[float, float]:
    t = pre["period"].to_numpy().astype(np.float64)
    g = pre["treated"].to_numpy().astype(np.float64)
    y = pre["y"].to_numpy().astype(np.float64)
    X = np.column_stack([np.ones_like(t), t, g, g * t])
    beta, cov, df = _ols(X, y)
    tstat = beta[3] / np.sqrt(cov[3, 3])
    return float(beta[3]), float(2 * stats.t.sf(abs(tstat), df))


def did_analysis(panel: pl.DataFrame) -> dict:
    att, se = _did(panel)
    pre = panel.filter(pl.col("post") == 0)
    slope, p_trend = _slope_diff(pre)

    pre_periods = sorted(pre["period"].unique().to_list())
    fake_start = pre_periods[len(pre_periods) // 2]
    placebo = pre.with_columns((pl.col("period") >= fake_start).cast(pl.Int64).alias("post"))
    p_att, p_se = _did(placebo)
    placebo_p = float(2 * stats.norm.sf(abs(p_att) / p_se))

    trends_ok = p_trend >= 0.05
    placebo_ok = placebo_p >= 0.05
    return {
        "att": att,
        "se": se,
        "ci_low": att - Z * se,
        "ci_high": att + Z * se,
        "pre_trend_slope_diff": slope,
        "pre_trend_p": p_trend,
        "placebo_att": p_att,
        "placebo_p": placebo_p,
        "credible": bool(trends_ok and placebo_ok),
    }


if __name__ == "__main__":
    print(did_analysis(make_dev_panel()))
