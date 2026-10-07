#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP04 Assessment Task 2 — Reduction, Embeddings and Anomaly
Screening (instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

Every input is built here with a fresh secret seed from the real
credit-scoring file: random samples of applications with the columns in a
random order and new ids, and two screening batches with injected anomalies
of three known kinds (one field pushed to an extreme, identities stitched
from different applicants, a tight ring of near-identical applications).
PCA references are recomputed here; the embedding is scored by
trustworthiness against the standardised fields; anomaly scores are scored
by ROC-AUC against the injection labels, per kind.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.manifold import trustworthiness
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _applications import TYPES, sample_applications, with_anomalies  # noqa: E402
from grading_harness import Checks, finalize, load_student_module, main, uses_engine  # noqa: E402

WEIGHT = 20
TRUST_FLOOR = 0.90
TYPE_AUC_FLOOR = 0.85
ALL_AUC_FLOOR = 0.90


def _z(frame: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in frame.columns if c != "application_id"]
    X = frame.select(cols).to_numpy().astype(float)
    return (X - X.mean(0)) / X.std(0), cols


def ref_pca(frame: pl.DataFrame) -> dict:
    Z, cols = _z(frame)
    vals, vecs = np.linalg.eigh(np.cov(Z.T))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    evr = vals / vals.sum()
    cum = np.cumsum(evr)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    return {"evr": evr, "cum": cum, "n90": n90, "pc1": dict(zip(cols, vecs[:, 0]))}


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_m4_task2")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    for fn in ("component_profile", "embed_2d", "anomaly_scores"):
        if not callable(getattr(st, fn, None)):
            return finalize(checks, WEIGHT, seed, f"Missing function: {fn}")
    rng = np.random.default_rng(seed)

    checks.add("uses_dim_reduction_engine", uses_engine(path, "DimReductionEngine"),
               "reduction must run through kailash-ml DimReductionEngine")
    checks.add("uses_anomaly_engine", uses_engine(path, "AnomalyDetectionEngine"),
               "at least one detector must run through kailash-ml AnomalyDetectionEngine")

    # ── PCA on two secret samples ─────────────────────────────────────────
    pca_names = ["variance_spectrum", "components_for_90pct", "pc1_loadings"]
    samples = [sample_applications(rng, int(rng.integers(1500, 4000))) for _ in range(2)]

    def run_pca():
        ok = dict.fromkeys(pca_names, True)
        notes = []
        for s in samples:
            out = st.component_profile(s.clone())
            ref = ref_pca(s)
            evr = np.asarray(out["explained_variance_ratio"], float)
            ok["variance_spectrum"] &= evr.shape == ref["evr"].shape and bool(np.allclose(evr, ref["evr"], atol=2e-3))
            n90 = int(out["n_components_90"])
            borderline = abs(ref["cum"][ref["n90"] - 1] - 0.90) < 3e-3 or abs(ref["cum"][max(ref["n90"] - 2, 0)] - 0.90) < 3e-3
            ok["components_for_90pct"] &= n90 == ref["n90"] or (borderline and abs(n90 - ref["n90"]) == 1)
            mine = out["pc1_loadings"]
            a = np.array([float(mine[c]) for c in ref["pc1"]])
            b = np.array(list(ref["pc1"].values()))
            # a direction is defined up to sign
            ok["pc1_loadings"] &= bool(min(np.abs(a - b).max(), np.abs(a + b).max()) <= 0.02)
            notes.append(f"reference: {ref['n90']} components, first EVR {np.round(ref['evr'][:3], 3).tolist()}; "
                         f"yours: {n90}, {np.round(evr[:3], 3).tolist()}")
        return {n: (ok[n], "; ".join(notes)) for n in pca_names}

    checks.guarded(pca_names, run_pca)

    # ── 2-D map ───────────────────────────────────────────────────────────
    emb_frame = sample_applications(rng, 1000)

    def run_emb():
        E = np.asarray(st.embed_2d(emb_frame.clone()), float)
        if E.shape != (emb_frame.height, 2) or not np.isfinite(E).all():
            return {"embedding_preserves_neighbours": (False, f"need a finite ({emb_frame.height}, 2) array, got {E.shape}")}
        Z, _ = _z(emb_frame)
        t = trustworthiness(Z, E, n_neighbors=10)
        return {"embedding_preserves_neighbours": (t >= TRUST_FLOOR, f"trustworthiness(k=10) = {t:.3f}, need >= {TRUST_FLOOR}")}

    checks.guarded(["embedding_preserves_neighbours"], run_emb)

    # ── anomaly screening on two secret batches ──────────────────────────
    an_names = ["global_anomalies_found", "stitched_identities_found", "application_ring_found", "overall_ranking"]
    batches = [with_anomalies(rng) for _ in range(2)]

    def run_an():
        ok = dict.fromkeys(an_names, True)
        notes = []
        for frame, types in batches:
            s = np.asarray(st.anomaly_scores(frame.clone()), float)
            if s.shape != (frame.height,) or not np.isfinite(s).all():
                return {n: (False, f"need one finite score per row, got shape {s.shape}") for n in an_names}
            aucs = {}
            for t in TYPES:
                m = (types == t) | (types == "normal")
                aucs[t] = roc_auc_score((types[m] == t).astype(int), s[m])
            aucs["all"] = roc_auc_score((types != "normal").astype(int), s)
            ok["global_anomalies_found"] &= aucs["global"] >= TYPE_AUC_FLOOR
            ok["stitched_identities_found"] &= aucs["dependency"] >= TYPE_AUC_FLOOR
            ok["application_ring_found"] &= aucs["clustered"] >= TYPE_AUC_FLOOR
            ok["overall_ranking"] &= aucs["all"] >= ALL_AUC_FLOOR
            notes.append("AUC " + ", ".join(f"{k} {v:.3f}" for k, v in aucs.items()))
        return {n: (ok[n], "; ".join(notes)) for n in an_names}

    checks.guarded(an_names, run_an)
    checks.require(["global_anomalies_found", "stitched_identities_found", "application_ring_found", "overall_ranking"],
                   ["uses_anomaly_engine"])
    checks.require(["variance_spectrum", "components_for_90pct", "embedding_preserves_neighbours"],
                   ["uses_dim_reduction_engine"])
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
