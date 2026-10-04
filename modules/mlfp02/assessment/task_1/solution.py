# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 1: Bayesian Updating & Likelihood Estimation
(Reference Solution — withheld from students; verified to pass grader.py)
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import integrate, optimize, special, stats

from shared import MLFPDataLoader

CONVERSION_THRESHOLD = 50.0  # an order of at least $50 counts as a conversion


def load_orders() -> pl.DataFrame:
    """The experiment log (all arms). Development data only — the grader
    calls your functions on its own frames."""
    return MLFPDataLoader().load("mlfp02", "experiment_data.parquet")


# ── Part A: Beta-Binomial updating ─────────────────────────────────────


def _successes(orders: pl.DataFrame, arm: str) -> tuple[int, int]:
    sub = orders.filter(pl.col("experiment_group") == arm)
    n = sub.height
    if n == 0:
        raise ValueError(f"arm {arm!r} has no rows")
    s = int((sub["metric_value"] >= CONVERSION_THRESHOLD).sum())
    return s, n


def conversion_posterior(
    orders: pl.DataFrame, arm: str, prior: tuple[float, float]
) -> dict[str, float]:
    s, n = _successes(orders, arm)
    a = prior[0] + s
    b = prior[1] + (n - s)
    return {
        "alpha": float(a),
        "beta": float(b),
        "mean": float(a / (a + b)),
        "ci_low": float(stats.beta.ppf(0.025, a, b)),
        "ci_high": float(stats.beta.ppf(0.975, a, b)),
    }


def prob_beats_control(
    orders: pl.DataFrame, treatment: str, prior: tuple[float, float]
) -> float:
    pc = conversion_posterior(orders, "control", prior)
    pt = conversion_posterior(orders, treatment, prior)
    # P(p_t > p_c) = ∫ f_t(x) · F_c(x) dx — exact up to quadrature error.
    lo = min(stats.beta.ppf(1e-12, pt["alpha"], pt["beta"]), stats.beta.ppf(1e-12, pc["alpha"], pc["beta"]))
    hi = max(stats.beta.isf(1e-12, pt["alpha"], pt["beta"]), stats.beta.isf(1e-12, pc["alpha"], pc["beta"]))
    val, _ = integrate.quad(
        lambda x: stats.beta.pdf(x, pt["alpha"], pt["beta"])
        * stats.beta.cdf(x, pc["alpha"], pc["beta"]),
        lo,
        hi,
        limit=200,
        points=[pt["mean"], pc["mean"]],
    )
    return float(min(max(val, 0.0), 1.0))


# ── Part B: maximum likelihood ─────────────────────────────────────────


def _gamma_loglik(x: np.ndarray, shape: float, scale: float) -> float:
    return float(np.sum(stats.gamma.logpdf(x, shape, loc=0, scale=scale)))


def _lognorm_loglik(x: np.ndarray, mu: float, sigma: float) -> float:
    return float(np.sum(stats.lognorm.logpdf(x, sigma, loc=0, scale=np.exp(mu))))


def _gamma_mle(x: np.ndarray) -> tuple[float, float]:
    # Profile likelihood: for fixed shape k the MLE scale is mean/k, and the
    # shape solves log(k) - digamma(k) = log(mean) - mean(log x).
    s = np.log(x.mean()) - np.log(x).mean()
    k = optimize.brentq(lambda k: np.log(k) - special.digamma(k) - s, 1e-6, 1e6)
    return float(k), float(x.mean() / k)


def fit_order_value(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("order values must be strictly positive")
    k, theta = _gamma_mle(x)
    mu = float(np.log(x).mean())
    sigma = float(np.log(x).std(ddof=0))
    ll_g = _gamma_loglik(x, k, theta)
    ll_l = _lognorm_loglik(x, mu, sigma)
    aic_g, aic_l = 4 - 2 * ll_g, 4 - 2 * ll_l
    return {
        "gamma_shape": k,
        "gamma_scale": theta,
        "gamma_loglik": ll_g,
        "lognorm_mu": mu,
        "lognorm_sigma": sigma,
        "lognorm_loglik": ll_l,
        "best_by_aic": "gamma" if aic_g < aic_l else "lognormal",
    }


# ── Part C: MAP ────────────────────────────────────────────────────────

SHAPE_PRIOR = (np.log(2.0), 0.5)   # log(shape) ~ Normal(log 2, 0.5)
SCALE_PRIOR = (np.log(20.0), 1.0)  # log(scale) ~ Normal(log 20, 1.0)


def _log_posterior(x: np.ndarray, shape: float, scale: float) -> float:
    if shape <= 0 or scale <= 0:
        return -np.inf
    return (
        _gamma_loglik(x, shape, scale)
        + stats.lognorm.logpdf(shape, SHAPE_PRIOR[1], scale=np.exp(SHAPE_PRIOR[0]))
        + stats.lognorm.logpdf(scale, SCALE_PRIOR[1], scale=np.exp(SCALE_PRIOR[0]))
    )


def map_gamma(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    k0, t0 = _gamma_mle(x)

    def neg(z: np.ndarray) -> float:
        return -_log_posterior(x, float(np.exp(z[0])), float(np.exp(z[1])))

    res = optimize.minimize(
        neg, x0=np.log([k0, t0]), method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000},
    )
    k, theta = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
    return {"shape": k, "scale": theta, "log_posterior": float(-res.fun)}


if __name__ == "__main__":
    orders = load_orders()
    prior = (2.0, 20.0)
    print(conversion_posterior(orders, "treatment_a", prior))
    print(prob_beats_control(orders, "treatment_a", prior))
    rev = orders.filter(pl.col("metric_value") > 0)["metric_value"].to_numpy()
    print(fit_order_value(rev[:2000]))
    print(map_gamma(rev[:20]))
