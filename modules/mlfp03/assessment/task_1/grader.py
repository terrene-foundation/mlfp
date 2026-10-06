#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP03 Assessment Task 1 — Application-Time Model Inputs
(instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

Each run draws a secret labelled history (10,000 rows of the credit file) and a
secret hold-out flag (25% of rows). ``build_model_inputs`` is called on it, and
then the checks are behavioural:

- split-first: the call is repeated with the HOLD-OUT rows altered — their
  outcomes, a planted field that equals the outcome on hold-out rows only, and
  the scale of several money fields. Training rows are untouched. A submission
  that fits selection or preprocessing on all rows changes its selection or
  its training-row inputs; one that learns from training rows only does not;
- the hold-out rows' inputs must be what ``transform`` produces for them;
- identifiers and the post-outcome field are tested by changing them and
  requiring the inputs not to move;
- scoring must not refit: a few applications scored inside a very different
  batch keep their inputs;
- the affordability features are found by rank correlation with the grader's
  own computation on fresh applications;
- signal: the grader fits its own logistic model on the training-row inputs
  and scores fresh applications (drawn from the generating process with a
  secret seed) against their real outcomes.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _solution_heldout import (  # noqa: E402
    ID_COLUMN,
    LEAK_COLUMN,
    TARGET,
    as_submitted,
    auc,
    fresh_applications,
    sample_history,
)
from grading_harness import Checks, finalize, load_student_module, main  # noqa: E402

warnings.filterwarnings("ignore")

WEIGHT = 20
MAX_INPUTS = 12
N_HISTORY = 10_000
N_FRESH = 8_000
HOLDOUT_SHARE = 0.25
AUC_MARGIN = 0.015  # below the grader's all-features logistic reference
GATE = "well_formed_inputs"
NAMES = [
    GATE,
    "identifier_is_not_an_input",
    "post_outcome_field_is_not_an_input",
    "selection_ignores_holdout_rows",
    "preprocessing_learned_from_training_rows_only",
    "holdout_rows_use_the_training_rules",
    "scoring_does_not_refit",
    "instalment_burden_feature",
    "savings_cover_feature",
    "inputs_keep_the_signal",
]


def _matrix(X: pl.DataFrame, ids: pl.Series, cols: list[str]) -> np.ndarray:
    """Rows of ``X`` in the order of ``ids``; columns ``cols`` as float."""
    order = pl.DataFrame({ID_COLUMN: ids}).with_row_index("__pos")
    joined = order.join(X.select([ID_COLUMN] + cols), on=ID_COLUMN, how="left").sort("__pos")
    return joined.select(cols).cast(pl.Float64).to_numpy()


def _check_frame(X, ids: pl.Series, cols: list[str] | None) -> str | None:
    """None when ``X`` is a valid input frame for ``ids``; else the reason."""
    if not isinstance(X, pl.DataFrame):
        return f"expected a polars DataFrame, got {type(X).__name__}"
    if ID_COLUMN not in X.columns:
        return f"no '{ID_COLUMN}' key column"
    feats = [c for c in X.columns if c != ID_COLUMN]
    if cols is not None and feats != cols:
        return f"columns {feats} differ from 'selected' {cols}"
    if X.height != len(ids) or X[ID_COLUMN].n_unique() != X.height or set(X[ID_COLUMN].to_list()) != set(ids.to_list()):
        return "not exactly one row per application"
    bad = [c for c in feats if not (X[c].dtype.is_numeric() or X[c].dtype == pl.Boolean)]
    if bad:
        return f"non-numeric inputs {bad}"
    M = X.select(feats).cast(pl.Float64).to_numpy()
    if not np.all(np.isfinite(M)):
        return "missing or non-finite values in the inputs"
    return None


def _best_rank_corr(M: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    best = 0.0
    for j in range(M.shape[1]):
        x = M[mask, j]
        if np.std(x) == 0:
            continue
        rho = spearmanr(x, ref[mask]).statistic
        if np.isfinite(rho):
            best = max(best, abs(float(rho)))
    return best


def _perturb_holdout(history: pl.DataFrame, is_holdout: pl.Series, rng) -> pl.DataFrame:
    """Alter hold-out rows only: outcomes, a planted outcome-copy, money scales."""
    h = is_holdout.to_numpy()
    y = history[TARGET].to_numpy().copy()
    y_new = y.copy()
    y_new[h] = rng.permutation(y[h])
    decoy = history["coe_vehicle_owner"].to_numpy().copy()
    decoy[h] = y_new[h]
    hold = pl.Series(h)
    return history.with_columns(
        pl.Series(TARGET, y_new, dtype=history.schema[TARGET]),
        pl.Series("coe_vehicle_owner", decoy, dtype=history.schema["coe_vehicle_owner"]),
        pl.when(hold).then(pl.col("income_sgd") * 3).otherwise(pl.col("income_sgd")).alias("income_sgd"),
        pl.when(hold).then(pl.col("savings_balance") * 4).otherwise(pl.col("savings_balance")).alias("savings_balance"),
        pl.when(hold).then(pl.col("monthly_installment") * 2).otherwise(pl.col("monthly_installment")).alias("monthly_installment"),
        pl.when(hold).then(pl.col("credit_utilization") * 0.3 + 0.7).otherwise(pl.col("credit_utilization")).alias("credit_utilization"),
        pl.when(hold).then(pl.col("payment_history_score") - 150).otherwise(pl.col("payment_history_score")).alias("payment_history_score"),
    )


def _reference_auc(history: pl.DataFrame, is_holdout: pl.Series, fresh: pl.DataFrame) -> float:
    """The grader's own baseline: every numeric application field except the
    key and the post-outcome field, median-imputed, logistic regression."""
    cols = [c for c, t in history.schema.items() if t.is_numeric() and c not in (TARGET, LEAK_COLUMN)]
    train = history.filter(~is_holdout)
    med = {c: train[c].median() for c in cols}

    def mat(df):
        return df.select([pl.col(c).cast(pl.Float64).fill_null(med[c]) for c in cols]).to_numpy()

    sc = StandardScaler().fit(mat(train))
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(mat(train)), train[TARGET].to_numpy())
    return auc(fresh[TARGET].to_numpy(), lr.predict_proba(sc.transform(mat(fresh)))[:, 1])


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_task1")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    if not callable(getattr(st, "build_model_inputs", None)):
        return finalize(checks, WEIGHT, seed, "Missing function: build_model_inputs")

    rng = np.random.default_rng(seed)
    history = sample_history(N_HISTORY, int(rng.integers(1 << 31)))
    is_holdout = pl.Series("is_holdout", rng.random(N_HISTORY) < HOLDOUT_SHARE)
    fresh = fresh_applications(N_FRESH, int(rng.integers(1 << 31)))
    apps = as_submitted(fresh)
    ids = history[ID_COLUMN]
    train_ids = ids.filter(~is_holdout)
    hold_ids = ids.filter(is_holdout)

    state: dict = {}

    def gate():
        out = st.build_model_inputs(history.clone(), is_holdout.clone())
        if not isinstance(out, dict) or not {"selected", "inputs", "transform"} <= set(out):
            return {GATE: (False, "return a dict with keys 'selected', 'inputs', 'transform'")}
        sel = out["selected"]
        if not isinstance(sel, (list, tuple)) or not all(isinstance(c, str) for c in sel) or not 1 <= len(sel) <= MAX_INPUTS:
            return {GATE: (False, f"'selected' must list 1–{MAX_INPUTS} column names, got {sel!r}")}
        sel = list(sel)
        why = _check_frame(out["inputs"], ids, sel)
        if why:
            return {GATE: (False, f"'inputs': {why}")}
        if not callable(out["transform"]):
            return {GATE: (False, "'transform' is not callable")}
        Xa = out["transform"](apps.clone())
        why = _check_frame(Xa, apps[ID_COLUMN], sel)
        if why:
            return {GATE: (False, f"transform(applications): {why}")}
        if not np.any(Xa.select(sel).cast(pl.Float64).to_numpy().std(axis=0) > 0):
            return {GATE: (False, "every input is constant")}
        state.update(sel=sel, out=out, Xa=Xa, Ma=_matrix(Xa, apps[ID_COLUMN], sel))
        return {GATE: (True, "")}

    checks.guarded([GATE], gate)
    if not checks.results[GATE]:
        for n in NAMES[1:]:
            checks.add(n, False, "gate failed")
        return finalize(checks, WEIGHT, seed)

    sel, out, Ma = state["sel"], state["out"], state["Ma"]
    transform = out["transform"]
    app_ids = apps[ID_COLUMN]

    def identifiers():
        new_ids = pl.Series(ID_COLUMN, [f"NEW-{i:07d}" for i in rng.permutation(apps.height)])
        X2 = transform(apps.with_columns(new_ids))
        if _check_frame(X2, new_ids, sel):
            return {"identifier_is_not_an_input": (False, "transform failed on renamed applications")}
        moved = float(np.max(np.abs(_matrix(X2, new_ids, sel) - Ma)))
        return {"identifier_is_not_an_input": (moved < 1e-9, f"inputs moved by {moved:.3g} when only the IDs changed")}

    def post_outcome():
        X3 = transform(apps.with_columns(fresh[LEAK_COLUMN]))
        if _check_frame(X3, app_ids, sel):
            return {"post_outcome_field_is_not_an_input": (False, "transform failed once the post-outcome field was filled")}
        moved = float(np.max(np.abs(_matrix(X3, app_ids, sel) - Ma)))
        return {"post_outcome_field_is_not_an_input": (moved < 1e-9, f"inputs moved by {moved:.3g} when the field recorded after the outcome was filled in")}

    def split_first():
        perturbed = _perturb_holdout(history, is_holdout, np.random.default_rng(seed + 1))
        out2 = st.build_model_inputs(perturbed, is_holdout.clone())
        sel2 = list(out2["selected"])
        same_sel = set(sel2) == set(sel)
        res = {"selection_ignores_holdout_rows": (same_sel, f"selection changed when only hold-out rows changed: {sorted(set(sel) ^ set(sel2))}")}
        if not same_sel or _check_frame(out2["inputs"], ids, sel2):
            res["preprocessing_learned_from_training_rows_only"] = (False, "selection or output changed when only hold-out rows changed")
        else:
            moved = float(np.max(np.abs(_matrix(out2["inputs"], train_ids, sel) - _matrix(out["inputs"], train_ids, sel))))
            res["preprocessing_learned_from_training_rows_only"] = (moved < 1e-9, f"training-row inputs moved by {moved:.3g} when only hold-out rows changed")
        return res

    def holdout_rules():
        hold = history.filter(is_holdout)
        Xh = transform(as_submitted(hold))
        if _check_frame(Xh, hold_ids, sel):
            return {"holdout_rows_use_the_training_rules": (False, "transform failed on the hold-out rows")}
        moved = float(np.max(np.abs(_matrix(Xh, hold_ids, sel) - _matrix(out["inputs"], hold_ids, sel))))
        return {"holdout_rows_use_the_training_rules": (moved < 1e-9, f"hold-out rows in 'inputs' differ from transform() by {moved:.3g}")}

    def no_refit():
        other = as_submitted(fresh_applications(600, seed % 100_000 + 7)).with_columns(
            (pl.col("income_sgd") * 3).alias("income_sgd"),
            (pl.col("savings_balance") * 4).alias("savings_balance"),
            (pl.col("credit_utilization") * 0.3 + 0.7).alias("credit_utilization"),
            (pl.col("monthly_installment") * 2).alias("monthly_installment"),
        )
        probe = pl.concat([apps.head(50), other], how="vertical_relaxed")
        X4 = transform(probe)
        if _check_frame(X4, probe[ID_COLUMN], sel):
            return {"scoring_does_not_refit": (False, "transform failed on a mixed batch")}
        moved = float(np.max(np.abs(_matrix(X4, app_ids.head(50), sel) - Ma[:50])))
        return {"scoring_does_not_refit": (moved < 1e-9, f"an application's inputs moved by {moved:.3g} when scored in a different batch")}

    def affordability():
        income = fresh["income_sgd"].cast(pl.Float64).to_numpy()
        inst = fresh["monthly_installment"].to_numpy()
        known = ~np.isnan(income)
        burden = np.where(known, inst / (np.nan_to_num(income) / 12.0 + 1e-12), np.nan)
        cover = fresh["savings_balance"].to_numpy() / inst
        b = _best_rank_corr(Ma, burden, known)
        s = _best_rank_corr(Ma, cover, np.ones(len(cover), bool))
        return {
            "instalment_burden_feature": (b >= 0.98, f"best rank correlation with instalment burden {b:.3f} (need >= 0.98)"),
            "savings_cover_feature": (s >= 0.98, f"best rank correlation with savings cover {s:.3f} (need >= 0.98)"),
        }

    def signal():
        Mt = _matrix(out["inputs"], train_ids, sel)
        y = history.filter(~is_holdout)[TARGET].to_numpy()
        sc = StandardScaler().fit(Mt)
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(Mt), y)
        got = auc(fresh[TARGET].to_numpy(), lr.predict_proba(sc.transform(Ma))[:, 1])
        ref = _reference_auc(history, is_holdout, fresh)
        return {"inputs_keep_the_signal": (got >= ref - AUC_MARGIN, f"fresh-application AUC {got:.4f}; reference {ref:.4f}; need >= {ref - AUC_MARGIN:.4f}")}

    checks.guarded(["identifier_is_not_an_input"], identifiers)
    checks.guarded(["post_outcome_field_is_not_an_input"], post_outcome)
    checks.guarded(["selection_ignores_holdout_rows", "preprocessing_learned_from_training_rows_only"], split_first)
    checks.guarded(["holdout_rows_use_the_training_rules"], holdout_rules)
    checks.guarded(["scoring_does_not_refit"], no_refit)
    checks.guarded(["instalment_burden_feature", "savings_cover_feature"], affordability)
    checks.guarded(["inputs_keep_the_signal"], signal)
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
