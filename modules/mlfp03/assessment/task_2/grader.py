#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP03 Assessment Task 2 — Model Selection You Can Defend
(instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

``select_and_fit`` is called twice per run:

A. a secret sample of 4,000 labelled rows of the credit file;
B. a secret sample of only 500 rows, with 40 extra numeric "bureau" fields
   appended that carry no information about default.

Both fitted models are scored on fresh applications drawn from the dataset's
generating process with a secret seed (the same extra fields are appended for
B). References are the grader's own: an L2 logistic regression on every
legitimate numeric field (A), the same with a cross-validated penalty (B), and
the TRUE default probabilities, whose AUC is the ceiling no honest
out-of-sample estimate can clear.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _solution_heldout import (  # noqa: E402
    ID_COLUMN,
    LEAK_COLUMN,
    TARGET,
    TRUE_PROBABILITY,
    as_submitted,
    auc,
    fresh_applications,
    sample_history,
)
from grading_harness import Checks, finalize, load_student_module, main  # noqa: E402

warnings.filterwarnings("ignore")

WEIGHT = 20
FAMILIES = {"logistic_regression", "svm", "knn", "naive_bayes", "decision_tree", "random_forest", "gradient_boosting"}
REQUIRED = {"logistic_regression", "random_forest", "gradient_boosting"}
N_A, N_B, N_FRESH, N_NOISE = 4_000, 500, 10_000, 40
MARGIN_A, MARGIN_B = 0.015, 0.03
HONEST_TOL = 0.03
GATE = "well_formed_result"
NAMES = [
    GATE,
    "comparison_covers_the_model_zoo",
    "chosen_model_is_best_by_validation",
    "comparison_scores_are_out_of_sample",
    "estimate_matches_unseen_performance",
    "predictions_follow_the_applicant",
    "generalises_on_the_full_sample",
    "generalises_when_data_is_scarce",
]


def _add_noise(df: pl.DataFrame, rng) -> pl.DataFrame:
    cols = {f"bureau_attr_{i:02d}": rng.normal(size=df.height).round(3) for i in range(1, N_NOISE + 1)}
    return df.with_columns([pl.Series(k, v) for k, v in cols.items()])


def _reference_auc(train: pl.DataFrame, fresh: pl.DataFrame, cv: bool) -> float:
    cols = [c for c, t in train.schema.items() if t.is_numeric() and c not in (TARGET, LEAK_COLUMN)]
    med = {c: train[c].median() for c in cols}

    def mat(df):
        return df.select([pl.col(c).cast(pl.Float64).fill_null(med[c]) for c in cols]).to_numpy()

    sc = StandardScaler().fit(mat(train))
    model = (
        LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring="roc_auc", max_iter=3000)
        if cv
        else LogisticRegression(max_iter=3000)
    )
    model.fit(sc.transform(mat(train)), train[TARGET].to_numpy())
    return auc(fresh[TARGET].to_numpy(), model.predict_proba(sc.transform(mat(fresh)))[:, 1])


def _probabilities(fn, apps: pl.DataFrame) -> np.ndarray:
    p = np.asarray(fn(apps.clone()), dtype=float).reshape(-1)
    if p.shape != (apps.height,):
        raise ValueError(f"predict_proba returned {p.shape[0]} values for {apps.height} applications")
    if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        raise ValueError("predict_proba must return finite probabilities in [0, 1]")
    return p


def _valid(out) -> str | None:
    if not isinstance(out, dict) or not {"cv_auc", "chosen", "estimated_auc", "predict_proba"} <= set(out):
        return "return a dict with keys 'cv_auc', 'chosen', 'estimated_auc', 'predict_proba'"
    if not isinstance(out["cv_auc"], dict) or not out["cv_auc"]:
        return "'cv_auc' must be a non-empty dict"
    try:
        vals = [float(v) for v in out["cv_auc"].values()] + [float(out["estimated_auc"])]
    except (TypeError, ValueError):
        return "'cv_auc' values and 'estimated_auc' must be numbers"
    if not all(np.isfinite(vals)):
        return "non-finite scores"
    if not callable(out["predict_proba"]):
        return "'predict_proba' is not callable"
    return None


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_task2")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    if not callable(getattr(st, "select_and_fit", None)):
        return finalize(checks, WEIGHT, seed, "Missing function: select_and_fit")

    rng = np.random.default_rng(seed)
    train_a = sample_history(N_A, int(rng.integers(1 << 31)))
    fresh = fresh_applications(N_FRESH, int(rng.integers(1 << 31)))
    apps = as_submitted(fresh)
    y = fresh[TARGET].to_numpy()
    ceiling = auc(y, fresh[TRUE_PROBABILITY].to_numpy())
    state: dict = {}

    def gate():
        out = st.select_and_fit(train_a.clone())
        why = _valid(out)
        if why:
            return {GATE: (False, why)}
        state["out"] = out
        state["p"] = _probabilities(out["predict_proba"], apps)
        if np.std(state["p"]) == 0:
            return {GATE: (False, "predict_proba returns the same value for every applicant")}
        return {GATE: (True, "")}

    checks.guarded([GATE], gate)
    if not checks.results[GATE]:
        for n in NAMES[1:]:
            checks.add(n, False, "gate failed")
        return finalize(checks, WEIGHT, seed)
    out, p = state["out"], state["p"]
    cv = {str(k): float(v) for k, v in out["cv_auc"].items()}
    got_a = auc(y, p)

    def comparison():
        fams = set(cv) & FAMILIES
        unknown = sorted(set(cv) - FAMILIES)
        ok = len(fams) >= 5 and REQUIRED <= fams and not unknown
        res = {"comparison_covers_the_model_zoo": (ok, f"families {sorted(cv)}; need >= 5 from {sorted(FAMILIES)} incl. {sorted(REQUIRED)}")}
        chosen = out["chosen"]
        best = max(cv.values())
        res["chosen_model_is_best_by_validation"] = (
            chosen in cv and cv[chosen] >= best - 0.005,
            f"chosen {chosen!r} ({cv.get(chosen)}) vs best validated score {best:.4f}",
        )
        top = max(cv.values())
        res["comparison_scores_are_out_of_sample"] = (
            top <= ceiling + HONEST_TOL and min(cv.values()) >= 0.4,
            f"best reported score {top:.4f}, but even the true default probabilities only reach {ceiling:.4f} on unseen applicants",
        )
        est = float(out["estimated_auc"])
        res["estimate_matches_unseen_performance"] = (
            abs(est - got_a) <= HONEST_TOL,
            f"estimated {est:.4f}, measured {got_a:.4f} on unseen applicants (tolerance {HONEST_TOL})",
        )
        return res

    def alignment():
        perm = rng.permutation(apps.height)[:2000]
        q = _probabilities(out["predict_proba"], apps[perm.tolist()])
        moved = float(np.max(np.abs(q - p[perm])))
        return {"predictions_follow_the_applicant": (moved < 1e-9, f"shuffling the applications changed predictions by {moved:.3g}")}

    def full_sample():
        ref = _reference_auc(train_a, fresh, cv=False)
        return {"generalises_on_the_full_sample": (got_a >= ref - MARGIN_A, f"AUC {got_a:.4f} on unseen applicants; reference {ref:.4f}; need >= {ref - MARGIN_A:.4f}")}

    def scarce():
        noise_rng = np.random.default_rng(seed + 99)
        train_b = _add_noise(sample_history(N_B, int(rng.integers(1 << 31))), noise_rng)
        fresh_b = _add_noise(fresh, noise_rng)
        out_b = st.select_and_fit(train_b.clone())
        why = _valid(out_b)
        if why:
            return {"generalises_when_data_is_scarce": (False, why)}
        got = auc(y, _probabilities(out_b["predict_proba"], as_submitted(fresh_b)))
        ref = _reference_auc(train_b, fresh_b, cv=True)
        return {"generalises_when_data_is_scarce": (got >= ref - MARGIN_B, f"AUC {got:.4f} on unseen applicants; reference {ref:.4f}; need >= {ref - MARGIN_B:.4f}")}

    checks.guarded(
        ["comparison_covers_the_model_zoo", "chosen_model_is_best_by_validation",
         "comparison_scores_are_out_of_sample", "estimate_matches_unseen_performance"],
        comparison,
    )  # fmt: skip
    checks.guarded(["predictions_follow_the_applicant"], alignment)
    checks.guarded(["generalises_on_the_full_sample"], full_sample)
    checks.guarded(["generalises_when_data_is_scarce"], scarce)
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
