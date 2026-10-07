#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP03 Assessment Task 3 — Decisions Priced in Dollars
(instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

``build_decision_model`` is called twice per run on a secret 8,000-row sample
of the credit file, each time with secret costs (a missed default and a
declined good applicant) and a fresh, empty registry directory.

Everything is then measured on fresh applications drawn from the dataset's
generating process with a secret seed. Because the process is known, the
grader holds every fresh applicant's TRUE default probability, so:

- calibration is measured against the true probabilities, not noisy outcomes;
- the expected cost of the decisions is computed exactly
  (approve: p_true x c_missed; decline: (1 - p_true) x c_declined);
- references are the grader's own L2 logistic model (fitted on the same
  sample) with the Bayes-optimal threshold for the same costs.

The registry is opened by the grader after the call: the production version's
artifact must reproduce the submission's probabilities.
"""
from __future__ import annotations

import asyncio
import pickle
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _solution_heldout import (  # noqa: E402
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

WEIGHT = 30
MODULE = "student_task3"
N_HISTORY, N_FRESH = 8_000, 10_000
AUC_MARGIN = 0.015
CAL_LARGE_TOL = 0.01  # |mean predicted - mean true probability|
ECE_TOL = 0.015  # decile-binned |predicted - true probability|
BRIER_MARGIN = 0.0015
COST_SLACK = 0.03  # relative to the reference's expected cost
ESTIMATE_TOL = 0.20  # relative error of the reported expected cost
GATE = "well_formed_result"
NAMES = [
    GATE,
    "ranks_applicants",
    "calibrated_on_average",
    "calibrated_across_the_range",
    "probabilities_score_well",
    "decisions_near_cost_optimal",
    "decisions_follow_the_costs",
    "cost_estimate_is_honest",
    "predictions_follow_the_applicant",
    "production_model_registered",
    "registered_model_reproduces_scores",
]


def _costs(rng, regime: int) -> dict:
    if regime == 0:  # defaults are expensive: threshold around 0.1-0.25
        return {"missed_default": float(rng.uniform(8_000, 12_000)), "declined_good": float(rng.uniform(1_000, 2_500))}
    # defaults are very expensive: threshold around 0.02-0.05
    return {"missed_default": float(rng.uniform(20_000, 30_000)), "declined_good": float(rng.uniform(500, 1_000))}


def _expected_cost(approve: np.ndarray, p_true: np.ndarray, costs: dict) -> float:
    return float(np.mean(np.where(approve, p_true * costs["missed_default"], (1 - p_true) * costs["declined_good"])))


def _ece(p: np.ndarray, p_true: np.ndarray) -> float:
    edges = np.quantile(p, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    return float(sum(abs(p[b == k].mean() - p_true[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any()))


def _expected_brier(p: np.ndarray, p_true: np.ndarray) -> float:
    return float(np.mean(p * p - 2 * p * p_true + p_true))


def _reference(history: pl.DataFrame, fresh: pl.DataFrame) -> np.ndarray:
    cols = [c for c, t in history.schema.items() if t.is_numeric() and c not in (TARGET, LEAK_COLUMN)]
    med = {c: history[c].median() for c in cols}

    def mat(df):
        return df.select([pl.col(c).cast(pl.Float64).fill_null(med[c]) for c in cols]).to_numpy()

    sc = StandardScaler().fit(mat(history))
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(mat(history)), history[TARGET].to_numpy())
    return lr.predict_proba(sc.transform(mat(fresh)))[:, 1]


def _probabilities(fn, apps: pl.DataFrame) -> np.ndarray:
    p = np.asarray(fn(apps.clone()), dtype=float).reshape(-1)
    if p.shape != (apps.height,):
        raise ValueError(f"predict_proba returned {p.shape[0]} values for {apps.height} applications")
    if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        raise ValueError("predict_proba must return finite probabilities in [0, 1]")
    return p


def _decisions(fn, apps: pl.DataFrame) -> np.ndarray:
    a = np.asarray(fn(apps.clone())).reshape(-1)
    if a.shape != (apps.height,):
        raise ValueError(f"decide returned {a.shape[0]} values for {apps.height} applications")
    if a.dtype != bool:
        raise ValueError(f"decide must return booleans (True = approve), got dtype {a.dtype}")
    return a


def _valid(out) -> str | None:
    keys = {"predict_proba", "decide", "expected_cost", "model_name"}
    if not isinstance(out, dict) or not keys <= set(out):
        return f"return a dict with keys {sorted(keys)}"
    if not callable(out["predict_proba"]) or not callable(out["decide"]):
        return "'predict_proba' and 'decide' must be callable"
    try:
        est = float(out["expected_cost"])
    except (TypeError, ValueError):
        return "'expected_cost' must be a number"
    if not np.isfinite(est):
        return "'expected_cost' is not finite"
    if not isinstance(out["model_name"], str) or not out["model_name"]:
        return "'model_name' must be a non-empty string"
    return None


async def _production_artifact(registry_dir: Path, name: str):
    from kailash.db import ConnectionManager
    from kailash_ml import ModelRegistry
    from kailash_ml.engines.model_registry import LocalFileArtifactStore

    db = registry_dir / "registry.db"
    if not db.exists():
        raise FileNotFoundError("no registry.db in the registry directory")
    conn = ConnectionManager(f"sqlite:///{db.as_posix()}")
    await conn.initialize()
    try:
        registry = ModelRegistry(conn, LocalFileArtifactStore(registry_dir / "artifacts"))
        mv = await registry.get_model(name, stage="production")
        blob = await registry.load_artifact(mv.name, mv.version)
    finally:
        await conn.close()
    return mv, blob


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, MODULE)
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    if not callable(getattr(st, "build_decision_model", None)):
        return finalize(checks, WEIGHT, seed, "Missing function: build_decision_model")
    sys.modules[MODULE] = st  # so artefacts pickled from the submission can be loaded

    rng = np.random.default_rng(seed)
    history = sample_history(N_HISTORY, int(rng.integers(1 << 31)))
    fresh = fresh_applications(N_FRESH, int(rng.integers(1 << 31)))
    apps = as_submitted(fresh)
    y = fresh[TARGET].to_numpy()
    p_true = fresh[TRUE_PROBABILITY].to_numpy()
    costs = [_costs(rng, 0), _costs(rng, 1)]
    workdir = Path(tempfile.mkdtemp(prefix="mlfp03_t3_"))
    state: dict = {}

    try:

        def gate():
            out = st.build_decision_model(history.clone(), dict(costs[0]), str(workdir / "registry"))
            why = _valid(out)
            if why:
                return {GATE: (False, why)}
            p = _probabilities(out["predict_proba"], apps)
            if np.std(p) == 0:
                return {GATE: (False, "predict_proba returns the same value for every applicant")}
            state.update(out=out, p=p, a=_decisions(out["decide"], apps))
            return {GATE: (True, "")}

        checks.guarded([GATE], gate)
        if not checks.results[GATE]:
            for n in NAMES[1:]:
                checks.add(n, False, "gate failed")
            return finalize(checks, WEIGHT, seed)
        out, p, a = state["out"], state["p"], state["a"]
        ref = _reference(history, fresh)

        def probabilities():
            got, want = auc(y, p), auc(y, ref)
            gap = abs(float(p.mean()) - float(p_true.mean()))
            ece, ece_ref = _ece(p, p_true), _ece(ref, p_true)
            br, br_ref = _expected_brier(p, p_true), _expected_brier(ref, p_true)
            return {
                "ranks_applicants": (got >= want - AUC_MARGIN, f"AUC {got:.4f} on unseen applicants; reference {want:.4f}; need >= {want - AUC_MARGIN:.4f}"),
                "calibrated_on_average": (gap <= CAL_LARGE_TOL, f"mean predicted default probability {p.mean():.4f} vs true {p_true.mean():.4f}"),
                "calibrated_across_the_range": (ece <= ECE_TOL, f"decile calibration error {ece:.4f} against the true probabilities (reference {ece_ref:.4f}; need <= {ECE_TOL})"),
                "probabilities_score_well": (br <= br_ref + BRIER_MARGIN, f"expected Brier score {br:.5f}; reference {br_ref:.5f}; need <= {br_ref + BRIER_MARGIN:.5f}"),
            }

        def decisions():
            c = costs[0]
            t = c["declined_good"] / (c["declined_good"] + c["missed_default"])
            got, want = _expected_cost(a, p_true, c), _expected_cost(ref < t, p_true, c)
            est = float(out["expected_cost"])
            return {
                "decisions_near_cost_optimal": (got <= want * (1 + COST_SLACK), f"expected cost S${got:,.1f} per application; reference S${want:,.1f}; need <= S${want * (1 + COST_SLACK):,.1f}"),
                "cost_estimate_is_honest": (abs(est - got) <= ESTIMATE_TOL * got, f"reported S${est:,.1f} per application; measured S${got:,.1f} (tolerance {ESTIMATE_TOL:.0%})"),
            }

        def other_costs():
            c = costs[1]
            out2 = st.build_decision_model(history.clone(), dict(c), str(workdir / "registry_2"))
            why = _valid(out2)
            if why:
                return {"decisions_follow_the_costs": (False, why)}
            a2 = _decisions(out2["decide"], apps)
            t = c["declined_good"] / (c["declined_good"] + c["missed_default"])
            got, want = _expected_cost(a2, p_true, c), _expected_cost(ref < t, p_true, c)
            return {"decisions_follow_the_costs": (got <= want * (1 + COST_SLACK), f"with costs {c}: expected cost S${got:,.1f}; reference S${want:,.1f}; need <= S${want * (1 + COST_SLACK):,.1f}")}

        def alignment():
            perm = rng.permutation(apps.height)[:2000]
            q = _probabilities(out["predict_proba"], apps[perm.tolist()])
            moved = float(np.max(np.abs(q - p[perm])))
            return {"predictions_follow_the_applicant": (moved < 1e-9, f"shuffling the applications changed probabilities by {moved:.3g}")}

        def registry():
            mv, blob = asyncio.run(_production_artifact(workdir / "registry", out["model_name"]))
            res = {"production_model_registered": (str(mv.stage) .lower().endswith("production"), f"stage is {mv.stage!r}")}
            art = pickle.loads(blob)
            if not callable(getattr(art, "predict_proba", None)):
                res["registered_model_reproduces_scores"] = (False, "the production artefact has no predict_proba")
                return res
            sub = apps.head(2000)
            q = np.asarray(art.predict_proba(sub.clone()), dtype=float).reshape(-1)
            if q.shape != (sub.height,):
                res["registered_model_reproduces_scores"] = (False, f"production artefact returned shape {q.shape}")
                return res
            moved = float(np.max(np.abs(q - p[: sub.height])))
            res["registered_model_reproduces_scores"] = (moved < 1e-9, f"production artefact differs from predict_proba by {moved:.3g}")
            return res

        checks.guarded(["ranks_applicants", "calibrated_on_average", "calibrated_across_the_range", "probabilities_score_well"], probabilities)
        checks.guarded(["decisions_near_cost_optimal", "cost_estimate_is_honest"], decisions)
        checks.guarded(["decisions_follow_the_costs"], other_costs)
        checks.guarded(["predictions_follow_the_applicant"], alignment)
        checks.guarded(["production_model_registered", "registered_model_reproduces_scores"], registry)
        return finalize(checks, WEIGHT, seed)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        sys.modules.pop(MODULE, None)


if __name__ == "__main__":
    main(grade)
