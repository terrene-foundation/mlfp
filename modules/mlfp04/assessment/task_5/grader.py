#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP04 Assessment Task 5 — From Discovered Segments to a
Neural Network (instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

Part A: the submission's loss and gradients for a 3-layer network are
compared with an independent reference on random networks and data the
grader draws per run, including a confidently-wrong network (|logit| ~ 60)
where a naive log(sigmoid) overflows.
Part B/C: two secret customer populations from a generator with planted
latent segments whose churn risk no straight line in the raw fields can
separate. Discovered features are scored by how much a plain logistic
regression gains from them; churn predictions are scored on held-out
customers whose labels never reach the submission, against the true
probability ranking and against a logistic regression on the raw fields.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _churn import FIELDS, TARGET, make_customers  # noqa: E402
from grading_harness import Checks, banned_imports, finalize, load_student_module, main, uses_engine  # noqa: E402

WEIGHT = 25
FEATURE_GAIN = 0.10   # CV AUC of logistic on discovered features vs raw fields
RAW_GAIN = 0.12       # held-out AUC vs logistic on raw fields
ORACLE_SLACK = 0.04   # held-out AUC vs the true churn probabilities


def ref_loss_grads(P: dict, X: np.ndarray, y: np.ndarray, l2: float):
    Z1 = X @ P["W1"] + P["b1"]
    H1 = np.maximum(Z1, 0)
    Z2 = H1 @ P["W2"] + P["b2"]
    H2 = np.maximum(Z2, 0)
    z = (H2 @ P["W3"] + P["b3"]).ravel()
    n = len(y)
    loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * l2 * sum((P[k] ** 2).sum() for k in ("W1", "W2", "W3"))
    s = np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))
    d = ((s - y) / n)[:, None]
    g = {"W3": H2.T @ d + l2 * P["W3"], "b3": d.sum(0)}
    d2 = (d @ P["W3"].T) * (Z2 > 0)
    g |= {"W2": H1.T @ d2 + l2 * P["W2"], "b2": d2.sum(0)}
    d1 = (d2 @ P["W2"].T) * (Z1 > 0)
    g |= {"W1": X.T @ d1 + l2 * P["W1"], "b1": d1.sum(0)}
    return float(loss), g


def random_case(rng: np.random.Generator, extreme: bool):
    n, d, h1, h2 = int(rng.integers(20, 60)), int(rng.integers(3, 9)), int(rng.integers(4, 12)), int(rng.integers(3, 9))
    s = 4.0 if extreme else 0.5
    P = {"W1": rng.normal(0, s, (d, h1)), "b1": rng.normal(0, 0.1, h1), "W2": rng.normal(0, s, (h1, h2)),
         "b2": rng.normal(0, 0.1, h2), "W3": rng.normal(0, s, (h2, 1)), "b3": rng.normal(0, 0.1, 1)}
    X = rng.normal(0, 1, (n, d))
    y = rng.integers(0, 2, n).astype(float)
    if extreme:  # confidently wrong: flip labels to disagree with the network
        z = ref_logits(P, X)
        y = (z < 0).astype(float)
    return P, X, y, float(rng.choice([0.0, 0.01, 0.1]))


def ref_logits(P, X):
    H = np.maximum(np.maximum(X @ P["W1"] + P["b1"], 0) @ P["W2"] + P["b2"], 0)
    return (H @ P["W3"] + P["b3"]).ravel()


def _cv_auc(F: np.ndarray, y: np.ndarray) -> float:
    F = StandardScaler().fit_transform(F)
    return float(cross_val_score(LogisticRegression(max_iter=2000), F, y, cv=5, scoring="roc_auc").mean())


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_m4_task5")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    for fn in ("loss_and_gradients", "discover_features", "fit_and_predict"):
        if not callable(getattr(st, fn, None)):
            return finalize(checks, WEIGHT, seed, f"Missing function: {fn}")
    rng = np.random.default_rng(seed)
    src = path.read_text()

    # ── Part A ────────────────────────────────────────────────────────────
    a_names = ["no_autodiff_library", "loss_correct", "gradients_correct", "stable_when_confidently_wrong"]
    cases = [random_case(rng, False) for _ in range(3)] + [random_case(rng, True)]

    def run_a():
        banned = banned_imports(path, ["torch", "jax", "tensorflow", "autograd"])
        ok = {"no_autodiff_library": not banned, "loss_correct": True, "gradients_correct": True,
              "stable_when_confidently_wrong": True}
        notes = [f"banned imports: {banned}"] if banned else []
        for idx, (P, X, y, l2) in enumerate(cases):
            loss, g = st.loss_and_gradients({k: v.copy() for k, v in P.items()}, X.copy(), y.copy(), l2)
            rl, rg = ref_loss_grads(P, X, y, l2)
            l_ok = np.isfinite(loss) and abs(loss - rl) <= 1e-6 * max(1, abs(rl))
            g_ok = all(np.shape(g[k]) == np.shape(rg[k]) and np.allclose(g[k], rg[k], rtol=1e-5, atol=1e-8) for k in rg)
            if idx < 3:
                ok["loss_correct"] &= bool(l_ok)
                ok["gradients_correct"] &= bool(g_ok)
            else:
                ok["stable_when_confidently_wrong"] = bool(l_ok and g_ok)
                notes.append(f"confidently-wrong case: max |logit| {np.abs(ref_logits(P, X)).max():.0f}, "
                             f"reference loss {rl:.4f}, yours {loss}")
        return {n: (ok[n], "; ".join(notes)) for n in a_names}

    checks.guarded(a_names, run_a)

    # ── Parts B and C ────────────────────────────────────────────────────
    bc_names = ["features_are_unsupervised_shape", "discovered_features_reveal_segments",
                "beats_linear_baseline", "close_to_true_risk_ranking"]
    pops = []
    for _ in range(2):
        frame, p = make_customers(int(rng.integers(1 << 31)), n=int(rng.integers(4000, 5000)))
        n_test = frame.height // 3
        pops.append((frame, p, n_test))

    def run_bc():
        ok = dict.fromkeys(bc_names, True)
        notes = []
        for frame, p, n_test in pops:
            train, test = frame.head(frame.height - n_test), frame.tail(n_test)
            y_tr, y_te = train[TARGET].to_numpy(), test[TARGET].to_numpy()
            raw = frame.select(FIELDS).to_numpy()
            base_cv = _cv_auc(raw[: train.height], y_tr)

            feats = st.discover_features(train.drop(TARGET).clone())
            shape_ok = isinstance(feats, pl.DataFrame) and feats.height == train.height and TARGET not in feats.columns
            ok["features_are_unsupervised_shape"] &= shape_ok
            if shape_ok:
                F = feats.select([c for c in feats.columns if feats[c].dtype.is_numeric()]).to_numpy().astype(float)
                F = F[:, F.std(0) > 0]
                feat_cv = _cv_auc(F, y_tr) if F.shape[1] else 0.5
            else:
                feat_cv = 0.5
            ok["discovered_features_reveal_segments"] &= feat_cv >= base_cv + FEATURE_GAIN

            shuffled = test.drop(TARGET).with_row_index("_r").sample(fraction=1.0, shuffle=True, seed=3)
            pred = np.asarray(st.fit_and_predict(train.clone(), shuffled.drop("_r")), float).ravel()
            if pred.shape != (n_test,) or not np.isfinite(pred).all():
                return {n: (False, f"fit_and_predict must return {n_test} finite probabilities, got {pred.shape}") for n in bc_names}
            order = shuffled["_r"].to_numpy()
            pred_in_order = np.empty(n_test)
            pred_in_order[order] = pred
            auc = roc_auc_score(y_te, pred_in_order)
            sc = StandardScaler().fit(raw[: train.height])
            lr = LogisticRegression(max_iter=2000).fit(sc.transform(raw[: train.height]), y_tr)
            lr_auc = roc_auc_score(y_te, lr.predict_proba(sc.transform(raw[train.height:]))[:, 1])
            oracle = roc_auc_score(y_te, p[train.height:])
            ok["beats_linear_baseline"] &= auc >= lr_auc + RAW_GAIN
            ok["close_to_true_risk_ranking"] &= auc >= oracle - ORACLE_SLACK
            notes.append(f"CV AUC logistic: raw {base_cv:.3f}, your features {feat_cv:.3f}; held-out AUC yours {auc:.3f}, "
                         f"raw logistic {lr_auc:.3f}, true-probability ranking {oracle:.3f}")
        return {n: (ok[n], "; ".join(notes)) for n in bc_names}

    checks.guarded(bc_names, run_bc)
    discovers = uses_engine(path, "ClusteringEngine") or uses_engine(path, "DimReductionEngine")
    checks.add("discovery_uses_kailash_engine", discovers,
               "discover_features must use kailash-ml ClusteringEngine or DimReductionEngine")
    neural = uses_engine(path, "SklearnTrainable") or len(re.findall(r"\bloss_and_gradients\s*\(", src)) >= 2
    checks.add("model_is_a_neural_network", neural,
               "train the network through SklearnTrainable or with your own loss_and_gradients")
    checks.require(["loss_correct", "gradients_correct"], ["no_autodiff_library"])
    checks.require(["discovered_features_reveal_segments", "beats_linear_baseline", "close_to_true_risk_ranking"],
                   ["features_are_unsupervised_shape", "discovery_uses_kailash_engine", "model_is_a_neural_network"])
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
