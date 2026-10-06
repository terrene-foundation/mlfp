#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP03 Assessment Task 4 — Release Review
(instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

Three parts, each scored on grader-held truth:

- ``explain``: the grader hands over two of ITS OWN scoring functions (with
  secret coefficients, an interaction, a step, and one field the function
  ignores) and computes the exact interventional Shapley values itself by
  enumerating every coalition against the same background rows.
- ``fairness_report``: the grader builds an audit table from fresh
  applications (decisions from their true risk plus noise, real outcomes) and
  recomputes every rate itself — for race, and for an age band with some
  unrecorded values under a column name chosen per run.
- ``drift_alerts``: the reference and every batch are fresh draws from the
  generating process. Three clean batches must raise nothing; a batch with
  three secretly chosen fields shifted must raise exactly those three. All
  numeric application fields are monitored at once, so per-field
  significance without a multiple-testing correction raises false alarms.
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _solution_heldout import (  # noqa: E402
    ID_COLUMN,
    LEAK_COLUMN,
    TARGET,
    TRUE_PROBABILITY,
    as_submitted,
    fresh_applications,
)
from grading_harness import Checks, finalize, load_student_module, main  # noqa: E402

warnings.filterwarnings("ignore")

WEIGHT = 30
N_BACKGROUND, N_EXPLAIN = 80, 20
SHAPLEY_TOL = 0.01
ADDITIVITY_TOL = 1e-4
DUMMY_TOL = 1e-3
RATE_TOL = 1e-9
REPORT_COLUMNS = [
    "group", "applicants", "approval_rate", "approval_ratio",
    "good_approval_rate", "default_approval_rate", "passes_four_fifths",
]  # fmt: skip
UNRECORDED = "unrecorded"
NAMES = [
    "explanations_add_up",
    "unused_field_gets_nothing",
    "explanations_match_shapley",
    "explanations_follow_the_model",
    "fairness_groups_complete",
    "approval_rates_and_ratios",
    "equalised_odds_rates",
    "four_fifths_flags",
    "no_false_alarms_on_stable_batches",
    "shifted_fields_found",
    "only_shifted_fields_flagged",
]


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _model_a(rng):
    feats = ["credit_utilization", "payment_history_score", "num_late_payments",
             "employment_years", "previous_defaults", "debt_to_income", "age"]  # fmt: skip
    w = rng.uniform(0.7, 1.3, 6)

    def f(X: np.ndarray) -> np.ndarray:  # column 6 (age) is never used
        u, phs, late, emp, prev, dti = (X[:, i] for i in range(6))
        z = (-1.6 + 3.0 * w[0] * u - 0.006 * w[1] * (phs - 650) + 0.3 * w[2] * late
             - 0.06 * w[3] * emp + 0.5 * w[4] * prev + 0.4 * w[5] * np.tanh(dti) + 1.2 * u * (late >= 2))  # fmt: skip
        return _sigmoid(z)

    return feats, 6, f


def _model_b(rng):
    feats = ["num_hard_inquiries", "loan_amount_sgd", "savings_balance",
             "credit_age_years", "months_employed", "num_credit_lines"]  # fmt: skip
    w = rng.uniform(0.7, 1.3, 5)

    def f(X: np.ndarray) -> np.ndarray:  # column 4 (months_employed) is never used
        inq, loan, sav, cage, _, lines = (X[:, i] for i in range(6))
        z = (-2.0 + 0.9 * w[0] * (inq >= 2) + 0.8 * w[1] * np.tanh(loan / 300_000)
             - 0.6 * w[2] * np.tanh(sav / 150_000) - 0.05 * w[3] * cage + 0.15 * w[4] * np.abs(lines - 3.5))  # fmt: skip
        return _sigmoid(z)

    return feats, 4, f


def _exact_shapley(f, bg: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Interventional Shapley values by enumerating all 2^d coalitions."""
    n, d = X.shape
    masks = np.array([[(m >> j) & 1 for j in range(d)] for m in range(1 << d)], dtype=bool)
    v = np.empty((n, 1 << d))
    for m, mask in enumerate(masks):
        Z = np.repeat(bg[None, :, :], n, axis=0).copy()  # n x B x d
        Z[:, :, mask] = X[:, None, mask]
        v[:, m] = f(Z.reshape(-1, d)).reshape(n, -1).mean(axis=1)
    phi = np.zeros((n, d))
    for j in range(d):
        for m in range(1 << d):
            if (m >> j) & 1:
                continue
            s = bin(m).count("1")
            weight = factorial(s) * factorial(d - s - 1) / factorial(d)
            phi[:, j] += weight * (v[:, m | (1 << j)] - v[:, m])
    return phi


def _frame_fn(f, feats):
    def predict_proba(df: pl.DataFrame) -> np.ndarray:
        return f(df.select(feats).cast(pl.Float64).to_numpy())

    return predict_proba


def _attributions(out, apps: pl.DataFrame, feats: list[str]) -> np.ndarray:
    if not isinstance(out, pl.DataFrame):
        raise ValueError(f"explain must return a polars DataFrame, got {type(out).__name__}")
    missing = [c for c in [ID_COLUMN] + feats if c not in out.columns]
    if missing:
        raise ValueError(f"explain output is missing columns {missing}")
    if out.height != apps.height or set(out[ID_COLUMN].to_list()) != set(apps[ID_COLUMN].to_list()):
        raise ValueError("explain must return exactly one row per application")
    order = apps.select(ID_COLUMN).with_row_index("__pos")
    M = order.join(out.select([ID_COLUMN] + feats), on=ID_COLUMN, how="left").sort("__pos")
    M = M.select(feats).cast(pl.Float64).to_numpy()
    if not np.all(np.isfinite(M)):
        raise ValueError("non-finite attributions")
    return M


# --------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------
def _audit_table(rng) -> pl.DataFrame:
    fresh = fresh_applications(20_000, int(rng.integers(1 << 31)))
    noisy = fresh[TRUE_PROBABILITY].to_numpy() * np.exp(rng.normal(0, 0.25, fresh.height))
    cut = float(rng.uniform(0.07, 0.10))
    age = fresh["age"].to_numpy()
    band = np.where(age < 35, "21-34", np.where(age < 50, "35-49", "50+")).astype(object)
    band[rng.random(fresh.height) < 0.03] = None
    return fresh.select(ID_COLUMN, "race", "gender", TARGET).with_columns(
        pl.Series("approved", noisy < cut),
        pl.Series("age_band", band, dtype=pl.Utf8),
    )


def _reference_report(audit: pl.DataFrame, col: str) -> dict:
    g = audit[col].cast(pl.Utf8).fill_null(UNRECORDED).to_numpy()
    a = audit["approved"].to_numpy().astype(float)
    y = audit[TARGET].to_numpy()
    rows = {}
    for k in np.unique(g):
        m = g == k
        rows[k] = {
            "applicants": int(m.sum()),
            "approval_rate": a[m].mean(),
            "good_approval_rate": a[m & (y == 0)].mean(),
            "default_approval_rate": a[m & (y == 1)].mean(),
        }
    best = max(r["approval_rate"] for r in rows.values())
    for r in rows.values():
        r["approval_ratio"] = r["approval_rate"] / best
        r["passes_four_fifths"] = bool(r["approval_ratio"] >= 0.8)
    return rows


def _student_report(out) -> dict:
    if not isinstance(out, pl.DataFrame):
        raise ValueError(f"fairness_report must return a polars DataFrame, got {type(out).__name__}")
    missing = [c for c in REPORT_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"fairness_report output is missing columns {missing}")
    rows = {}
    for r in out.select(REPORT_COLUMNS).iter_rows(named=True):
        key = r["group"] if r["group"] is not None else None
        if key in rows:
            raise ValueError(f"group {key!r} appears twice")
        rows[key] = r
    return rows


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------
def _shift(batch: pl.DataFrame, field: str, rng) -> pl.DataFrame:
    n = batch.height
    if field == "income_sgd":
        return batch.with_columns((pl.col(field).cast(pl.Float64) * 0.85).round(0).cast(batch.schema[field]))
    if field == "credit_utilization":
        return batch.with_columns((pl.col(field) + 0.06).clip(0, 1))
    if field == "savings_balance":
        return batch.with_columns(pl.col(field) * 0.75)
    if field == "num_late_payments":
        bump = (rng.random(n) < 0.25).astype(int)
        return batch.with_columns((pl.col(field) + pl.Series(bump)).cast(batch.schema[field]))
    if field == "loan_amount_sgd":
        return batch.with_columns((pl.col(field) * 0.012).round(0).mul(100).cast(batch.schema[field]))
    if field == "payment_history_score":
        return batch.with_columns((pl.col(field) - 20).cast(batch.schema[field]))
    raise KeyError(field)


SHIFTABLE = ["income_sgd", "credit_utilization", "savings_balance",
             "num_late_payments", "loan_amount_sgd", "payment_history_score"]  # fmt: skip


def _alerts(fn, reference, batch, features) -> set[str]:
    out = fn(reference.clone(), batch.clone(), list(features))
    if not isinstance(out, (list, tuple, set)) or not all(isinstance(c, str) for c in out):
        raise ValueError(f"drift_alerts must return a list of field names, got {out!r}")
    unknown = set(out) - set(features)
    if unknown:
        raise ValueError(f"drift_alerts returned fields that were not monitored: {sorted(unknown)}")
    return set(out)


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_task4")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    missing = [n for n in ("explain", "fairness_report", "drift_alerts") if not callable(getattr(st, n, None))]
    if missing:
        return finalize(checks, WEIGHT, seed, f"Missing function(s): {', '.join(missing)}")

    rng = np.random.default_rng(seed)

    def explanations():
        book = as_submitted(fresh_applications(2_000, int(rng.integers(1 << 31))))
        res = {}
        worst_add, worst_dummy, worst_err = 0.0, 0.0, {}
        for label, make in (("A", _model_a), ("B", _model_b)):
            feats, dummy, f = make(rng)
            frame = book.select([ID_COLUMN] + feats).with_columns([pl.col(c).cast(pl.Float64) for c in feats])
            idx = rng.permutation(frame.height)
            background, apps = frame[idx[:N_BACKGROUND].tolist()], frame[idx[N_BACKGROUND:N_BACKGROUND + N_EXPLAIN].tolist()]
            out = st.explain(_frame_fn(f, feats), background.clone(), apps.clone())
            phi = _attributions(out, apps, feats)
            bg_np, X = background.select(feats).to_numpy(), apps.select(feats).to_numpy()
            gap = f(X) - f(bg_np).mean()
            worst_add = max(worst_add, float(np.max(np.abs(phi.sum(axis=1) - gap))))
            worst_dummy = max(worst_dummy, float(np.max(np.abs(phi[:, dummy]))))
            worst_err[label] = float(np.max(np.abs(phi - _exact_shapley(f, bg_np, X))))
        res["explanations_add_up"] = (worst_add <= ADDITIVITY_TOL, f"attributions miss 'prediction minus average prediction' by up to {worst_add:.3g}")
        res["unused_field_gets_nothing"] = (worst_dummy <= DUMMY_TOL, f"a field the model never reads received attribution up to {worst_dummy:.3g}")
        res["explanations_match_shapley"] = (worst_err["A"] <= SHAPLEY_TOL, f"model A: largest gap to the exact Shapley values {worst_err['A']:.3g} (tolerance {SHAPLEY_TOL})")
        res["explanations_follow_the_model"] = (worst_err["B"] <= SHAPLEY_TOL, f"model B: largest gap to the exact Shapley values {worst_err['B']:.3g} (tolerance {SHAPLEY_TOL})")
        return res

    def fairness():
        audit = _audit_table(rng)
        col_name = f"applicant_age_band_{int(rng.integers(100, 999))}"
        audit = audit.rename({"age_band": col_name})
        shuffled = audit[rng.permutation(audit.height).tolist()]
        ok = {k: True for k in ("fairness_groups_complete", "approval_rates_and_ratios", "equalised_odds_rates", "four_fifths_flags")}
        why = {k: "" for k in ok}
        for col in ("race", col_name):
            ref = _reference_report(audit, col)
            got = _student_report(st.fairness_report(shuffled.clone(), col))
            if set(got) != set(ref) or any(int(got[k]["applicants"]) != ref[k]["applicants"] for k in ref):
                ok["fairness_groups_complete"] = False
                why["fairness_groups_complete"] = f"{col}: groups/counts {sorted((str(k), got[k]['applicants']) for k in got)} vs expected {sorted((k, r['applicants']) for k, r in ref.items())}"
                for k in ("approval_rates_and_ratios", "equalised_odds_rates", "four_fifths_flags"):
                    ok[k], why[k] = False, f"{col}: groups do not match"
                continue

            def gap(names):
                return max(abs(float(got[k][n]) - ref[k][n]) for k in ref for n in names)

            g1 = gap(["approval_rate", "approval_ratio"])
            if g1 > RATE_TOL:
                ok["approval_rates_and_ratios"], why["approval_rates_and_ratios"] = False, f"{col}: off by up to {g1:.3g}"
            g2 = gap(["good_approval_rate", "default_approval_rate"])
            if g2 > RATE_TOL:
                ok["equalised_odds_rates"], why["equalised_odds_rates"] = False, f"{col}: off by up to {g2:.3g}"
            wrong = sorted(k for k in ref if bool(got[k]["passes_four_fifths"]) != ref[k]["passes_four_fifths"])
            if wrong:
                ok["four_fifths_flags"], why["four_fifths_flags"] = False, f"{col}: wrong flag for {wrong}"
        return {k: (ok[k], why[k]) for k in ok}

    def drift():
        reference = as_submitted(fresh_applications(5_000, int(rng.integers(1 << 31))))
        features = [c for c, t in reference.schema.items() if t.is_numeric() and c not in (ID_COLUMN, LEAK_COLUMN, TARGET)]
        alarms = []
        for _ in range(3):
            clean = as_submitted(fresh_applications(3_000, int(rng.integers(1 << 31))))
            alarms += sorted(_alerts(st.drift_alerts, reference, clean, features))
        shifted_fields = sorted(rng.choice(SHIFTABLE, size=3, replace=False).tolist())
        batch = as_submitted(fresh_applications(3_000, int(rng.integers(1 << 31))))
        for fld in shifted_fields:
            batch = _shift(batch, fld, rng)
        got = _alerts(st.drift_alerts, reference, batch, features)
        return {
            "no_false_alarms_on_stable_batches": (not alarms, f"alerts on batches from the same population: {alarms}"),
            "shifted_fields_found": (set(shifted_fields) <= got, f"shifted {shifted_fields}; flagged {sorted(got)}"),
            "only_shifted_fields_flagged": (got <= set(shifted_fields), f"shifted {shifted_fields}; also flagged {sorted(got - set(shifted_fields))}"),
        }

    checks.guarded(NAMES[0:4], explanations)
    checks.guarded(NAMES[4:8], fairness)
    checks.guarded(NAMES[8:11], drift)
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
