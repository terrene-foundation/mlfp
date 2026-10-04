#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP02 Assessment Task 1 — Bayesian Updating & Likelihood
Estimation (instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

Every check calls the submission on data the student has never seen:
secret per-run subsamples of the experiment log with randomised priors, and
synthetic order-value samples drawn from Gamma / LogNormal laws with secret
parameters. References are recomputed here. Optimisation answers are graded by
OPTIMALITY (the student's parameters must reach the maximum log-likelihood /
log-posterior), so any correct optimiser passes and any non-optimal estimate
(method of moments, a free location parameter, MLE in place of MAP) fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import integrate, optimize, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, close, finalize, load_student_module, main  # noqa: E402

from shared import MLFPDataLoader  # noqa: E402

WEIGHT = 20
THRESHOLD = 50.0
SHAPE_PRIOR = (np.log(2.0), 0.5)
SCALE_PRIOR = (np.log(20.0), 1.0)
LL_TOL = 0.01  # absolute log-likelihood slack for "reached the optimum"


# ── references ─────────────────────────────────────────────────────────


def ref_posterior(df: pl.DataFrame, arm: str, prior) -> dict:
    sub = df.filter(pl.col("experiment_group") == arm)
    s = int((sub["metric_value"] >= THRESHOLD).sum())
    a, b = prior[0] + s, prior[1] + sub.height - s
    return {
        "alpha": a, "beta": b, "mean": a / (a + b),
        "ci_low": stats.beta.ppf(0.025, a, b), "ci_high": stats.beta.ppf(0.975, a, b),
    }


def ref_prob_better(df: pl.DataFrame, arm: str, prior) -> float:
    c, t = ref_posterior(df, "control", prior), ref_posterior(df, arm, prior)
    lo = min(stats.beta.ppf(1e-12, t["alpha"], t["beta"]), stats.beta.ppf(1e-12, c["alpha"], c["beta"]))
    hi = max(stats.beta.isf(1e-12, t["alpha"], t["beta"]), stats.beta.isf(1e-12, c["alpha"], c["beta"]))
    v, _ = integrate.quad(
        lambda x: stats.beta.pdf(x, t["alpha"], t["beta"]) * stats.beta.cdf(x, c["alpha"], c["beta"]),
        lo, hi, limit=200, points=[t["mean"], c["mean"]],
    )
    return float(v)


def gamma_ll(x, k, th) -> float:
    if not (k > 0 and th > 0):
        return -np.inf
    return float(np.sum(stats.gamma.logpdf(x, k, scale=th)))


def lognorm_ll(x, mu, sig) -> float:
    if not sig > 0:
        return -np.inf
    return float(np.sum(stats.lognorm.logpdf(x, sig, scale=np.exp(mu))))


def ref_mle(x: np.ndarray) -> dict:
    s = np.log(x.mean()) - np.log(x).mean()
    k = optimize.brentq(lambda k: np.log(k) - special.digamma(k) - s, 1e-6, 1e6)
    th = x.mean() / k
    mu, sig = np.log(x).mean(), np.log(x).std()
    return {"g": gamma_ll(x, k, th), "l": lognorm_ll(x, mu, sig), "k": k, "th": th}


def log_post(x, k, th) -> float:
    if not (k > 0 and th > 0):
        return -np.inf
    return (
        gamma_ll(x, k, th)
        + stats.lognorm.logpdf(k, SHAPE_PRIOR[1], scale=np.exp(SHAPE_PRIOR[0]))
        + stats.lognorm.logpdf(th, SCALE_PRIOR[1], scale=np.exp(SCALE_PRIOR[0]))
    )


def ref_map(x: np.ndarray) -> float:
    m = ref_mle(x)
    best = -np.inf
    for start in ([m["k"], m["th"]], [2.0, 20.0], [1.0, x.mean()]):
        r = optimize.minimize(
            lambda z: -log_post(x, np.exp(z[0]), np.exp(z[1])), np.log(start),
            method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
        )
        best = max(best, -r.fun)
    return float(best)


# ── grader-held data ───────────────────────────────────────────────────


def secret_cohort(df: pl.DataFrame, rng: np.random.Generator, n_per_arm: int) -> pl.DataFrame:
    parts = []
    for arm in ("control", "treatment_a"):
        sub = df.filter(pl.col("experiment_group") == arm)
        idx = rng.choice(sub.height, size=n_per_arm, replace=False)
        parts.append(sub[np.sort(idx).tolist()])
    return pl.concat(parts)


def synthetic_samples(rng: np.random.Generator, real_values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Gamma and LogNormal samples with secret parameters whose AIC verdict is
    unambiguous (|ΔAIC| > 6), plus a secret subsample of real order values."""
    out = []
    while len(out) < 1:
        k, th = rng.uniform(1.2, 6.0), rng.uniform(5.0, 30.0)
        x = rng.gamma(k, th, size=int(rng.integers(400, 900)))
        r = ref_mle(x)
        if 2 * (r["g"] - r["l"]) > 6:
            out.append(("gamma", x))
    while len(out) < 2:
        mu, sig = rng.uniform(2.5, 4.0), rng.uniform(0.6, 1.2)
        x = rng.lognormal(mu, sig, size=int(rng.integers(400, 900)))
        r = ref_mle(x)
        if 2 * (r["l"] - r["g"]) > 6:
            out.append(("lognormal", x))
    real = rng.choice(real_values, size=1500, replace=False)
    out.append(("real", real))
    return out


def map_sample(rng: np.random.Generator) -> np.ndarray:
    """A small sample where the prior matters: MAP log-posterior beats the MLE
    point's log-posterior by a clear margin."""
    while True:
        x = rng.gamma(rng.uniform(3.0, 8.0), rng.uniform(4.0, 12.0), size=int(rng.integers(8, 16)))
        m = ref_mle(x)
        if ref_map(x) - log_post(x, m["k"], m["th"]) > 0.5:
            return x


# ── grading ────────────────────────────────────────────────────────────


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_task1")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    missing = [f for f in ("conversion_posterior", "prob_beats_control", "fit_order_value", "map_gamma")
               if not callable(getattr(st, f, None))]
    if missing:
        return finalize(checks, WEIGHT, seed, f"Missing functions: {missing}")

    rng = np.random.default_rng(seed)
    full = MLFPDataLoader().load("mlfp02", "experiment_data.parquet")

    # A1 — posterior on the full experiment, all four arms, course prior
    def a1():
        prior = (2.0, 20.0)
        ok, note = True, ""
        for arm in ("control", "treatment_a", "treatment_b", "variant_c"):
            got, ref = st.conversion_posterior(full.clone(), arm, prior), ref_posterior(full, arm, prior)
            if not all(close(got[k], ref[k], rtol=1e-6) for k in ("alpha", "beta", "mean")):
                ok, note = False, f"{arm}: got {got}, expected alpha={ref['alpha']}, beta={ref['beta']}"
                break
        return {"posterior_full_data": (ok, note)}

    checks.guarded(["posterior_full_data"], a1)

    # A2 — secret subsample + secret prior: parameters and credible interval
    small = secret_cohort(full, rng, int(rng.integers(150, 400)))
    prior2 = (float(rng.uniform(1, 6)), float(rng.uniform(5, 40)))

    def a2():
        got = st.conversion_posterior(small.clone(), "treatment_a", prior2)
        ref = ref_posterior(small, "treatment_a", prior2)
        params = all(close(got[k], ref[k], rtol=1e-9) for k in ("alpha", "beta", "mean"))
        ci = close(got["ci_low"], ref["ci_low"], rtol=1e-4) and close(got["ci_high"], ref["ci_high"], rtol=1e-4)
        return {
            "posterior_secret_sample": (params, f"got {got}, expected {ref}"),
            "credible_interval_secret_sample": (ci, f"got [{got.get('ci_low')}, {got.get('ci_high')}], "
                                                f"expected [{ref['ci_low']:.6f}, {ref['ci_high']:.6f}]"),
        }

    checks.guarded(["posterior_secret_sample", "credible_interval_secret_sample"], a2)

    # A3 — P(treatment beats control): one check across the full data AND two
    # secret subsamples where the answer is genuinely uncertain (a constant 1.0
    # or a coarse Monte-Carlo estimate cannot pass).
    def a3():
        tiny = secret_cohort(full, rng, int(rng.integers(60, 140)))
        prior3 = (float(rng.uniform(1, 4)), float(rng.uniform(4, 16)))
        cases = [(full, (2.0, 20.0)), (tiny, prior3), (small, prior2)]
        got = [float(st.prob_beats_control(d.clone(), "treatment_a", p)) for d, p in cases]
        ref = [ref_prob_better(d, "treatment_a", p) for d, p in cases]
        ok = all(abs(g - r) <= 0.01 for g, r in zip(got, ref))
        return {"prob_beats_control": (ok, f"got {[round(g, 4) for g in got]}, expected {[round(r, 4) for r in ref]}")}

    checks.guarded(["prob_beats_control"], a3)

    # B — MLE on secret synthetic + real samples, graded by optimality
    pos = full.filter(pl.col("metric_value") > 0)["metric_value"].to_numpy()
    samples = synthetic_samples(rng, pos)

    def b():
        gam_ok = ln_ok = cons_ok = aic_ok = True
        notes = []
        for label, x in samples:
            got = st.fit_order_value(x.copy())
            ref = ref_mle(x)
            llg = gamma_ll(x, float(got["gamma_shape"]), float(got["gamma_scale"]))
            lll = lognorm_ll(x, float(got["lognorm_mu"]), float(got["lognorm_sigma"]))
            if not llg >= ref["g"] - LL_TOL:
                gam_ok = False
                notes.append(f"{label}: gamma loglik at your params {llg:.4f} < max {ref['g']:.4f}")
            if not lll >= ref["l"] - LL_TOL:
                ln_ok = False
                notes.append(f"{label}: lognormal loglik at your params {lll:.4f} < max {ref['l']:.4f}")
            if not (close(got["gamma_loglik"], llg, rtol=1e-7, atol=1e-4)
                    and close(got["lognorm_loglik"], lll, rtol=1e-7, atol=1e-4)):
                cons_ok = False
                notes.append(f"{label}: reported logliks do not equal the loglik of the reported parameters")
            want = "gamma" if ref["g"] > ref["l"] else "lognormal"
            if got["best_by_aic"] != want:
                aic_ok = False
                notes.append(f"{label}: best_by_aic={got['best_by_aic']!r}, expected {want!r}")
        n = "; ".join(notes)
        return {
            "gamma_mle_optimal": (gam_ok, n),
            "lognormal_mle_optimal": (ln_ok, n),
            "reported_loglik_consistent": (cons_ok, n),
            "aic_model_choice": (aic_ok, n),
        }

    checks.guarded(["gamma_mle_optimal", "lognormal_mle_optimal", "reported_loglik_consistent",
                    "aic_model_choice"], b)

    # C — MAP on small secret samples where the prior pulls the estimate
    def c():
        ok_opt = ok_rep = True
        notes = []
        for _ in range(2):
            x = map_sample(rng)
            got = st.map_gamma(x.copy())
            lp = log_post(x, float(got["shape"]), float(got["scale"]))
            best = ref_map(x)
            if not lp >= best - LL_TOL:
                ok_opt = False
                notes.append(f"n={x.size}: log-posterior at your MAP {lp:.4f} < max {best:.4f}")
            if not close(got["log_posterior"], lp, rtol=1e-7, atol=1e-4):
                ok_rep = False
                notes.append("reported log_posterior does not match your (shape, scale)")
        n = "; ".join(notes)
        return {"map_optimal": (ok_opt, n), "map_log_posterior_consistent": (ok_rep, n)}

    checks.guarded(["map_optimal", "map_log_posterior_consistent"], c)
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
