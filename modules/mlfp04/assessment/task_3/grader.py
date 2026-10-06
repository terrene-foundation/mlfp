#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP04 Assessment Task 3 — Baskets and Recommendations
(instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

Rules: two secret till exports (new catalogue subset, planted bundles,
hand-keyed names, double scans) with secret thresholds. The complete rule
set is recomputed here by brute force over every itemset and compared
rule by rule (support, confidence, lift).

Recommender: two secret rating histories from a low-rank taste model. The
grader holds back 25% of every customer's ratings plus every rating of a
dozen brand-new customers; the submission's predictions on those hidden
pairs are scored against the truth and against a bias baseline fitted here.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _retail import make_baskets, make_ratings  # noqa: E402
from grading_harness import Checks, finalize, load_student_module, main  # noqa: E402

WEIGHT = 20
RMSE_RATIO = 0.85   # warm RMSE must be <= 85% of the bias baseline's
NDCG_GAIN = 0.08    # per-customer NDCG@5 must beat the bias baseline by this


def ref_rules(baskets: pl.DataFrame, min_support: float, min_conf: float, max_len: int) -> dict:
    clean = baskets.select(pl.col("basket_id"), pl.col("item").str.strip_chars().str.to_lowercase()).unique()
    ids = clean["basket_id"].unique().sort().to_list()
    pos = {b: k for k, b in enumerate(ids)}
    items = sorted(clean["item"].unique().to_list())
    M = np.zeros((len(ids), len(items)), dtype=bool)
    col = {it: k for k, it in enumerate(items)}
    for b, it in clean.iter_rows():
        M[pos[b], col[it]] = True
    n = len(ids)
    sup = {}
    for size in range(1, max_len + 1):
        for combo in combinations(range(len(items)), size):
            s = M[:, list(combo)].all(1).mean()
            sup[frozenset(items[c] for c in combo)] = s
    out = {}
    for s, v in sup.items():
        if len(s) < 2 or v < min_support:
            continue
        for r in range(1, len(s)):
            for a in combinations(sorted(s), r):
                a = frozenset(a)
                c = s - a
                conf = v / sup[a]
                if conf >= min_conf:
                    out[(tuple(sorted(a)), tuple(sorted(c)))] = (v, conf, conf / sup[c])
    return out


def _bias_baseline(history: pl.DataFrame):
    latest = history.sort("rated_at").unique(["user_id", "item_id"], keep="last")
    mu = latest["rating"].mean()
    bu, bi = {}, {}
    for _ in range(10):
        t = latest.with_columns(pl.col("user_id").replace_strict(bu, default=0.0, return_dtype=pl.Float64).alias("bu"))
        bi = dict(t.group_by("item_id").agg(((pl.col("rating") - mu - pl.col("bu")).sum() / (pl.len() + 5.0)).alias("b")).iter_rows())
        t = latest.with_columns(pl.col("item_id").replace_strict(bi, default=0.0, return_dtype=pl.Float64).alias("bi"))
        bu = dict(t.group_by("user_id").agg(((pl.col("rating") - mu - pl.col("bi")).sum() / (pl.len() + 5.0)).alias("b")).iter_rows())

    def predict(pairs):
        return np.array([mu + bu.get(u, 0.0) + bi.get(i, 0.0) for u, i in pairs.select("user_id", "item_id").iter_rows()])

    return predict, mu


def _ndcg(users: np.ndarray, truth: np.ndarray, pred: np.ndarray, k: int = 5) -> float:
    vals = []
    for u in np.unique(users):
        m = users == u
        if m.sum() < 3:
            continue
        rr, pp = truth[m], pred[m]
        top = np.argsort(-pp, kind="stable")[:k]
        disc = 1 / np.log2(np.arange(2, k + 2))
        dcg = ((2 ** rr[top] - 1) * disc[: len(top)]).sum()
        ideal = ((2 ** np.sort(rr)[::-1][:k] - 1) * disc[: min(k, len(rr))]).sum()
        vals.append(dcg / ideal)
    return float(np.mean(vals))


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_m4_task3")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    for fn in ("mine_rules", "fit_recommender"):
        if not callable(getattr(st, fn, None)):
            return finalize(checks, WEIGHT, seed, f"Missing function: {fn}")
    rng = np.random.default_rng(seed)

    # ── association rules ────────────────────────────────────────────────
    rule_names = ["rules_complete", "no_spurious_rules", "rule_metrics_exact", "strongest_rule"]
    cases = []
    for _ in range(2):
        b = make_baskets(int(rng.integers(1 << 31)))
        cases.append((b, round(float(rng.uniform(0.02, 0.05)), 3), round(float(rng.uniform(0.35, 0.6)), 2), 3))

    def run_rules():
        ok = dict.fromkeys(rule_names, True)
        notes = []
        for b, ms, mc, ml in cases:
            ref = ref_rules(b, ms, mc, ml)
            out = st.mine_rules(b.clone(), ms, mc, ml)
            got = {}
            for row in out.select("antecedent", "consequent", "support", "confidence", "lift").iter_rows():
                got[(tuple(sorted(row[0])), tuple(sorted(row[1])))] = row[2:]
            missing = set(ref) - set(got)
            extra = set(got) - set(ref)
            ok["rules_complete"] &= not missing
            ok["no_spurious_rules"] &= not extra and len(got) == out.height
            shared = set(ref) & set(got)
            ok["rule_metrics_exact"] &= bool(shared) and all(
                np.allclose(got[k], ref[k], rtol=1e-6, atol=1e-9) for k in shared)
            best_ref = max(ref.values(), key=lambda v: v[2])[2] if ref else None
            best_got = float(out["lift"][0]) if out.height else None
            ok["strongest_rule"] &= best_ref is not None and best_got is not None and abs(best_got - best_ref) < 1e-9
            notes.append(f"min_support={ms} min_confidence={mc}: reference {len(ref)} rules, yours {len(got)}; "
                         f"missing e.g. {sorted(missing)[:2]}; extra e.g. {sorted(extra)[:2]}")
        return {n: (ok[n], "; ".join(notes)) for n in rule_names}

    checks.guarded(rule_names, run_rules)

    # ── recommender ──────────────────────────────────────────────────────
    rec_names = ["beats_bias_rmse", "personalised_ranking", "new_customers_handled"]
    datasets = [make_ratings(int(rng.integers(1 << 31))) for _ in range(2)]

    def run_rec():
        ok = dict.fromkeys(rec_names, True)
        notes = []
        for history, holdout, cold in datasets:
            predict = st.fit_recommender(history.clone())
            pairs = holdout.select("user_id", "item_id").sample(fraction=1.0, shuffle=True, seed=1)
            truth = pairs.join(holdout, on=["user_id", "item_id"], how="left")["rating"].to_numpy()
            pred = np.asarray(predict(pairs.clone()), float)
            if pred.shape != truth.shape or not np.isfinite(pred).all():
                return {n: (False, f"predict must return one finite rating per pair, got {pred.shape}") for n in rec_names}
            base, mu = _bias_baseline(history)
            bpred = base(pairs)
            is_cold = np.isin(pairs["user_id"].to_numpy(), cold)
            warm = ~is_cold
            rmse = lambda p, m: float(np.sqrt(np.mean((p[m] - truth[m]) ** 2)))  # noqa: E731
            r_s, r_b = rmse(pred, warm), rmse(bpred, warm)
            users = pairs["user_id"].to_numpy()
            n_s = _ndcg(users[warm], truth[warm], pred[warm])
            n_b = _ndcg(users[warm], truth[warm], bpred[warm])
            c_s, c_g = rmse(pred, is_cold), rmse(np.full(len(pred), mu), is_cold)
            ok["beats_bias_rmse"] &= r_s <= RMSE_RATIO * r_b
            ok["personalised_ranking"] &= n_s >= n_b + NDCG_GAIN
            ok["new_customers_handled"] &= c_s <= c_g
            notes.append(f"warm RMSE yours {r_s:.3f} vs bias {r_b:.3f}; NDCG@5 yours {n_s:.3f} vs bias {n_b:.3f}; "
                         f"new-customer RMSE yours {c_s:.3f} vs global mean {c_g:.3f}")
        return {n: (ok[n], "; ".join(notes)) for n in rec_names}

    checks.guarded(rec_names, run_rec)
    checks.require(["rules_complete", "rule_metrics_exact"], ["no_spurious_rules"])
    checks.require(["beats_bias_rmse", "personalised_ranking"], ["new_customers_handled"])
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
