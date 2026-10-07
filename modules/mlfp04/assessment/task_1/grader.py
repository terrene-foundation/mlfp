#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP04 Assessment Task 1 — Customer Segments and Mixture Models
(instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

Segmentation: the grader simulates three customer cohorts the student has
never seen (secret seed): a secret number of planted personas (3-6),
log-normal money and counts, ~1% corporate bulk accounts, an identifier and a
channel column. Recovery is scored against the planted personas, profiles
are recomputed from the student's own labels, and the at-risk call is checked
against the persona with the longest planted recency.

Mixtures: the grader simulates overlapping Gaussian mixtures and checks the
submission's EM output for internal consistency (E-step and log-likelihood
recomputed from the returned parameters), optimality (against the best of ten
scikit-learn fits) and convergence (one more EM step must not improve it).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _cohorts import BEHAVIOUR_COLS, make_cohort, make_mixture  # noqa: E402
from grading_harness import Checks, finalize, load_student_module, main, uses_engine  # noqa: E402

WEIGHT = 20
ARI_FLOOR = 0.90
MIN_SHARE = 0.03
LL_TOL = 0.01  # per-sample log-likelihood slack vs the best reference fit


def _log_joint(X, w, means, covs) -> np.ndarray:
    cols = []
    for j in range(len(w)):
        d = X.shape[1]
        L = np.linalg.cholesky(covs[j])
        sol = np.linalg.solve(L, (X - means[j]).T)
        cols.append(np.log(w[j]) - 0.5 * (sol**2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi))
    return np.column_stack(cols)


def _em_step(X, w, means, covs):
    lj = _log_joint(X, w, means, covs)
    R = np.exp(lj - logsumexp(lj, axis=1)[:, None])
    nk = R.sum(0)
    w2 = nk / len(X)
    m2 = (R.T @ X) / nk[:, None]
    c2 = np.array([((R[:, j, None] * (X - m2[j])).T @ (X - m2[j])) / nk[j] + 1e-6 * np.eye(X.shape[1]) for j in range(len(w))])
    return w2, m2, c2


def _uses_sklearn_mixture(path: Path) -> bool:
    src = path.read_text()
    return bool(
        re.search(r"^\s*(from|import)\s+sklearn\.mixture\b", src, flags=re.M)
        or re.search(r"^\s*from\s+sklearn\s+import\s+[^\n]*\bmixture\b", src, flags=re.M)
        or re.search(r"\bGaussianMixture\b|\bBayesianGaussianMixture\b", src)
    )


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_m4_task1")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    for fn in ("segment_customers", "fit_mixture"):
        if not callable(getattr(st, fn, None)):
            return finalize(checks, WEIGHT, seed, f"Missing function: {fn}")
    rng = np.random.default_rng(seed)

    checks.add("uses_clustering_engine", uses_engine(path, "ClusteringEngine"),
               "segmentation must run through kailash-ml ClusteringEngine")

    # ── segmentation on three secret cohorts ─────────────────────────────
    cohorts = [make_cohort(int(rng.integers(1 << 31))) for _ in range(3)]
    seg_names = ["labels_valid", "persona_count", "persona_recovery", "profiles_consistent", "at_risk_identified"]

    def run_segments():
        ok = {n: True for n in seg_names}
        notes = []
        for frame, persona, k, at_risk, bulk in cohorts:
            out = st.segment_customers(frame.clone())
            lab = np.asarray(out["labels"])
            if lab.shape != (frame.height,) or not np.issubdtype(lab.dtype, np.integer):
                ok = {n: False for n in seg_names}
                notes.append(f"labels must be one int per row (got shape {lab.shape}, dtype {lab.dtype})")
                break
            shares = np.unique(lab, return_counts=True)[1] / frame.height
            n_big = int((shares >= MIN_SHARE).sum())
            ari = adjusted_rand_score(persona[~bulk], lab[~bulk])
            ok["persona_count"] &= n_big == k
            ok["persona_recovery"] &= ari >= ARI_FLOOR
            notes.append(f"planted k={k}, your segments with >=3% of customers={n_big}, ARI={ari:.3f}")

            # profiles recomputed from the student's own labels
            ref = (
                frame.with_columns(pl.Series("segment", lab))
                .group_by("segment")
                .agg(pl.len().alias("n_customers"), *[pl.col(c).cast(pl.Float64).median() for c in BEHAVIOUR_COLS])
                .sort("segment")
            )
            prof = out["profiles"]
            good = isinstance(prof, pl.DataFrame) and set(["segment", "n_customers", *BEHAVIOUR_COLS]) <= set(prof.columns)
            if good:
                prof = prof.select(["segment", "n_customers", *BEHAVIOUR_COLS]).with_columns(pl.col("segment").cast(pl.Int64)).sort("segment")
                good = prof.height == ref.height and prof["segment"].to_list() == ref["segment"].cast(pl.Int64).to_list()
                if good:
                    a = prof.select(["n_customers", *BEHAVIOUR_COLS]).cast(pl.Float64).to_numpy()
                    b = ref.select(["n_customers", *BEHAVIOUR_COLS]).cast(pl.Float64).to_numpy()
                    good = bool(np.allclose(a, b, rtol=1e-6, atol=1e-6))
            ok["profiles_consistent"] &= good

            # at-risk segment: its majority persona must be the planted at-risk persona
            seg = int(out["at_risk_segment"])
            members = persona[lab == seg]
            maj = int(np.bincount(members).argmax()) if members.size else -1
            ok["at_risk_identified"] &= maj == at_risk
        return {n: (ok[n], "; ".join(notes)) for n in seg_names}

    checks.guarded(seg_names, run_segments)

    # ── EM on three secret mixtures ──────────────────────────────────────
    em_names = ["em_from_scratch", "em_parameters_valid", "em_responsibilities_consistent",
                "em_log_likelihood_consistent", "em_reaches_maximum", "em_converged"]
    mixtures = [make_mixture(int(rng.integers(1 << 31))) for _ in range(3)]
    from_scratch = not _uses_sklearn_mixture(path)

    def run_em():
        ok = {n: True for n in em_names}
        ok["em_from_scratch"] = from_scratch
        notes = []
        for X, k in mixtures:
            out = st.fit_mixture(X.copy(), k, int(rng.integers(1 << 16)))
            n, d = X.shape
            w = np.asarray(out["weights"], float)
            m = np.asarray(out["means"], float)
            c = np.asarray(out["covariances"], float)
            R = np.asarray(out["responsibilities"], float)
            if w.shape != (k,) or m.shape != (k, d) or c.shape != (k, d, d) or R.shape != (n, k):
                notes.append(f"shapes: weights {w.shape}, means {m.shape}, covariances {c.shape}, responsibilities {R.shape}")
                return {nm: (False, "; ".join(notes)) for nm in em_names}
            valid = abs(w.sum() - 1) < 1e-6 and (w > 0).all() and np.allclose(c, c.transpose(0, 2, 1), atol=1e-8)
            valid = valid and all(np.linalg.eigvalsh(cj).min() > 0 for cj in c)
            ok["em_parameters_valid"] &= bool(valid)
            if not valid:
                notes.append("weights must be positive and sum to 1; covariances symmetric positive-definite")
                continue
            lj = _log_joint(X, w, m, c)
            ll_rows = logsumexp(lj, axis=1)
            R_ref = np.exp(lj - ll_rows[:, None])
            ok["em_responsibilities_consistent"] &= bool(np.abs(R - R_ref).max() <= 1e-4 and np.allclose(R.sum(1), 1, atol=1e-6))
            ll = float(ll_rows.sum())
            ok["em_log_likelihood_consistent"] &= abs(float(out["log_likelihood"]) - ll) <= 1e-6 * abs(ll) + 1e-6
            ref = GaussianMixture(k, covariance_type="full", n_init=10, random_state=0, tol=1e-6,
                                  max_iter=2000, reg_covar=1e-6).fit(X)
            ref_ll = float(ref.score(X))
            ok["em_reaches_maximum"] &= ll / n >= ref_ll - LL_TOL
            w2, m2, c2 = _em_step(X, w, m, c)
            gain = (float(logsumexp(_log_joint(X, w2, m2, c2), axis=1).sum()) - ll) / n
            ok["em_converged"] &= gain <= 1e-4
            notes.append(f"k={k} d={d}: your mean log-lik {ll / n:.4f} vs best reference {ref_ll:.4f}; one more EM step gains {gain:.2e}")
        return {nm: (ok[nm], "; ".join(notes)) for nm in em_names}

    checks.guarded(em_names, run_em)
    checks.require(["persona_count", "persona_recovery", "at_risk_identified"],
                   ["uses_clustering_engine", "labels_valid", "profiles_consistent"])
    checks.require(["em_reaches_maximum", "em_converged"],
                   ["em_from_scratch", "em_parameters_valid", "em_responsibilities_consistent",
                    "em_log_likelihood_consistent"])
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
