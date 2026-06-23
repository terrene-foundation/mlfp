# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 3: Regression Modelling & Interpretation
(Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.

All regressions are solved in closed form (OLS via the normal equations;
logistic via Newton-Raphson / IRLS to the unique MLE), so the result is fully
deterministic and independently re-derivable. Predictors are standardised
(z-score, population sd) so the design matrix stays well-conditioned even with
the squared / interaction terms.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

OLS_FEATURES = [
    "income_imp",
    "age",
    "employment_years",
    "debt_to_income",
    "credit_age_years",
    "num_dependents",
    "edu_ord",
]
LOGIT_FEATURES = [
    "credit_utilization",
    "num_late_payments",
    "previous_defaults",
    "debt_to_income",
    "num_hard_inquiries",
]
EDU_MAP = {
    "primary": 1.0,
    "secondary": 2.0,
    "diploma": 3.0,
    "degree": 4.0,
    "postgraduate": 5.0,
}
TARGET = "loan_amount_sgd"


def _zscore(mat: np.ndarray) -> np.ndarray:
    """Standardise columns with population sd (ddof=0)."""
    return (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS coefficients via least squares (SVD)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _logistic_irls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Newton-Raphson / IRLS to the unique logistic MLE (convex problem)."""
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1.0 - p)
        grad = X.T @ (y - p)
        hess = (X * w[:, None]).T @ X
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.linalg.norm(step) < 1e-10:
            break
    return beta


def solve() -> dict:
    """Predict loan amount with OLS, test added non-linear terms with a partial
    F-test, then fit a logistic default model for odds-ratio interpretation.

    Returns a dict with these exact keys:
        n_obs (int)                rows used (no rows dropped)
        coefficients (dict)        {feature -> beta} incl. "intercept"
                                   (standardised predictors, raw target)
        t_stats (dict)             {feature -> t} incl. "intercept"
        p_values (dict)            {feature -> two-sided p} incl. "intercept"
        r_squared, adj_r_squared
        f_statistic, f_p_value
        partial_f, partial_f_p_value, delta_r_squared
                                   from adding income_std^2 and age_std*emp_std
        odds_ratios (dict)         {feature -> exp(beta)} incl. "intercept"
                                   for the logistic default model
        strongest_logit_predictor  feature (excl. intercept) with max |beta|
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "sg_credit_scoring.parquet")

    # --- Preprocess: median-impute income, ordinal-encode education ---
    income_median = df["income_sgd"].median()
    d = df.with_columns(
        pl.col("income_sgd").fill_null(income_median).alias("income_imp"),
        pl.col("education").replace_strict(EDU_MAP).cast(pl.Float64).alias("edu_ord"),
    )
    n_obs = d.height
    y = d[TARGET].to_numpy().astype(float)

    # --- 3a/3b: OLS with standardised predictors + full inference ---
    Z = np.column_stack([d[f].to_numpy().astype(float) for f in OLS_FEATURES])
    Zs = _zscore(Z)
    X = np.column_stack([np.ones(n_obs), Zs])
    p = X.shape[1]
    beta = _ols(X, y)
    resid = y - X @ beta
    rss = float((resid**2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - rss / tss
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - p)
    sigma2 = rss / (n_obs - p)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t_vals = beta / se
    p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=n_obs - p)
    f_statistic = (r_squared / (p - 1)) / ((1.0 - r_squared) / (n_obs - p))
    f_p_value = float(stats.f.sf(f_statistic, p - 1, n_obs - p))

    names = ["intercept"] + OLS_FEATURES
    coefficients = {nm: float(beta[i]) for i, nm in enumerate(names)}
    t_stats = {nm: float(t_vals[i]) for i, nm in enumerate(names)}
    p_values = {nm: float(p_vals[i]) for i, nm in enumerate(names)}

    # --- 3c: partial F-test for two added non-linear terms ---
    inc_s = Zs[:, OLS_FEATURES.index("income_imp")]
    age_s = Zs[:, OLS_FEATURES.index("age")]
    emp_s = Zs[:, OLS_FEATURES.index("employment_years")]
    X_full = np.column_stack([X, inc_s**2, age_s * emp_s])
    p_full = X_full.shape[1]
    beta_full = _ols(X_full, y)
    rss_full = float(((y - X_full @ beta_full) ** 2).sum())
    r2_full = 1.0 - rss_full / tss
    q = p_full - p
    partial_f = ((rss - rss_full) / q) / (rss_full / (n_obs - p_full))
    partial_f_p_value = float(stats.f.sf(partial_f, q, n_obs - p_full))
    delta_r_squared = r2_full - r_squared

    # --- 3d: logistic default model, odds ratios on standardised features ---
    Zl = np.column_stack([d[f].to_numpy().astype(float) for f in LOGIT_FEATURES])
    Zls = _zscore(Zl)
    Xl = np.column_stack([np.ones(n_obs), Zls])
    yl = d["default"].to_numpy().astype(float)
    beta_l = _logistic_irls(Xl, yl)
    logit_names = ["intercept"] + LOGIT_FEATURES
    odds_ratios = {nm: float(np.exp(beta_l[i])) for i, nm in enumerate(logit_names)}
    # Strongest effect = largest |beta| among predictors (exclude intercept).
    strongest_idx = 1 + int(np.argmax(np.abs(beta_l[1:])))
    strongest_logit_predictor = logit_names[strongest_idx]

    return {
        "n_obs": int(n_obs),
        "coefficients": coefficients,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": float(r_squared),
        "adj_r_squared": float(adj_r_squared),
        "f_statistic": float(f_statistic),
        "f_p_value": f_p_value,
        "partial_f": float(partial_f),
        "partial_f_p_value": partial_f_p_value,
        "delta_r_squared": float(delta_r_squared),
        "odds_ratios": odds_ratios,
        "strongest_logit_predictor": strongest_logit_predictor,
    }


if __name__ == "__main__":
    r = solve()
    print(f"n_obs = {r['n_obs']}")
    print(f"R^2 = {r['r_squared']:.5f}, adj R^2 = {r['adj_r_squared']:.5f}")
    print(f"F = {r['f_statistic']:.1f} (p = {r['f_p_value']:.2e})")
    print("OLS coefficients (standardised predictors):")
    for nm, b in r["coefficients"].items():
        print(f"  {nm:18s} b={b:14.3f}  t={r['t_stats'][nm]:8.2f}  p={r['p_values'][nm]:.2e}")
    print(
        f"Partial F = {r['partial_f']:.3f} (p = {r['partial_f_p_value']:.2e}), "
        f"delta R^2 = {r['delta_r_squared']:.6f}"
    )
    print("Logistic odds ratios:")
    for nm, o in r["odds_ratios"].items():
        print(f"  {nm:18s} OR={o:.4f}")
    print(f"Strongest logistic predictor: {r['strongest_logit_predictor']}")
