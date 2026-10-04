#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP02 Assessment Task 3 — Regression, ANOVA & Logistic
Inference (instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

Each run draws a secret training sample and a secret time cut-off. The
student's model is fitted on raw (uncleaned) training rows only, then its
`predict` is scored on later, held-out transactions the student never saw.
Coefficients and inference are compared with an independent reference fit
(statsmodels) on the same rows. ANOVA/Tukey run on a secret subsample and a
secret choice of flat types; the logistic model is checked against a
statsmodels MLE and scored by AUC on held-out borrowers.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, close, finalize, load_student_module, main  # noqa: E402

from shared import MLFPDataLoader  # noqa: E402

WEIGHT = 25
BASE = "3 ROOM"
LOGIT_FEATURES = ["credit_utilization", "num_late_payments", "previous_defaults", "debt_to_income", "num_hard_inquiries"]
MONTHS = [f"{y}-{m:02d}" for y in range(2021, 2024) for m in range(1, 13)]


# ── reference cleaning and fits (written independently of the solution) ──


def ref_clean(raw: pl.DataFrame) -> pl.DataFrame:
    yr = raw["month"].str.slice(0, 4).cast(pl.Int64).to_numpy()
    def mid(s: str):
        s = s.strip().upper()
        if " TO " not in s:
            return None
        lo, hi = s.split(" TO ")
        lo, hi = lo.replace("O", "0"), hi.replace("O", "0")
        if not (lo.isdigit() and hi.isdigit()):
            return None
        return (int(lo) + int(hi)) / 2
    storey = np.array([mid(s) if s is not None else None for s in raw["storey_range"].to_list()], dtype=object)
    df = raw.with_columns(
        pl.Series("remaining_lease_years", (99 - (yr - raw["lease_commence_date"].to_numpy())).astype(float)),
        pl.Series("storey_mid", [None if v is None else float(v) for v in storey], dtype=pl.Float64),
        pl.Series("_yr", yr),
    )
    keep = (
        (df["resale_price"] >= 100_000) & (df["resale_price"] <= 2_000_000)
        & (df["lease_commence_date"] <= df["_yr"]) & df["storey_mid"].is_not_null()
    )
    return df.filter(keep).drop("_yr")


def ref_design(df: pl.DataFrame, levels: list[str], interact: bool = False):
    cols = {"intercept": np.ones(df.height)}
    for c in ("floor_area_sqm", "remaining_lease_years", "storey_mid"):
        cols[c] = df[c].to_numpy().astype(float)
    ft = df["flat_type"].to_numpy()
    for lv in levels:
        cols[f"flat_type={lv}"] = (ft == lv).astype(float)
        if interact:
            cols[f"floor_area_sqm:flat_type={lv}"] = cols[f"flat_type={lv}"] * cols["floor_area_sqm"]
    return np.column_stack(list(cols.values())), list(cols)


def ref_ols(df: pl.DataFrame) -> dict:
    levels = sorted(set(df["flat_type"].to_list()) - {BASE})
    y = df["resale_price"].to_numpy().astype(float)
    X, names = ref_design(df, levels)
    fit = sm.OLS(y, X).fit()
    Xf, _ = ref_design(df, levels, interact=True)
    full = sm.OLS(y, Xf).fit()
    f_int, p_int, _ = full.compare_f_test(fit)
    return {
        "n_used": df.height, "names": names, "levels": levels, "beta": fit.params,
        "coefficients": dict(zip(names, fit.params)), "std_errors": dict(zip(names, fit.bse)),
        "p_values": dict(zip(names, fit.pvalues)), "r_squared": fit.rsquared,
        "adj_r_squared": fit.rsquared_adj, "f_statistic": fit.fvalue,
        "interaction_f": f_int, "interaction_p": p_int,
    }


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def auc(y: np.ndarray, s: np.ndarray) -> float:
    ranks = stats.rankdata(s)
    n1 = y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def dict_close(got: dict, ref: dict, rtol: float, atol: float = 1e-12) -> tuple[bool, str]:
    if set(got) != set(ref):
        return False, f"keys {sorted(got)} != expected {sorted(ref)}"
    bad = [k for k in ref if not close(got[k], ref[k], rtol=rtol, atol=atol)]
    return (not bad), "; ".join(f"{k}: got {got[k]}, expected {ref[k]:.6g}" for k in bad[:4])


# ── grading ────────────────────────────────────────────────────────────


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    import re

    banned = re.findall(r"^\s*(?:from|import)\s+(statsmodels|sklearn)\b", path.read_text(), flags=re.M)
    if banned:
        return finalize(checks, WEIGHT, seed, f"Submission imports {sorted(set(banned))}; implement the fits yourself")
    try:
        st = load_student_module(path, "student_task3")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    missing = [f for f in ("fit_price_model", "anova_flat_types", "fit_default_model") if not callable(getattr(st, f, None))]
    if missing:
        return finalize(checks, WEIGHT, seed, f"Missing functions: {missing}")

    rng = np.random.default_rng(seed)
    loader = MLFPDataLoader()
    hdb = loader.load("mlfp01", "hdb_resale.parquet")
    raw_cols = hdb.columns

    cutoff = str(rng.choice(MONTHS))
    past = hdb.filter(pl.col("month") < cutoff)
    train = past[np.sort(rng.choice(past.height, int(past.height * rng.uniform(0.6, 0.85)), replace=False)).tolist()]
    future_valid = ref_clean(hdb.filter(pl.col("month") >= cutoff))

    # A — linear regression with dummies, inference, interaction test, out-of-time prediction
    a_names = ["ols_rows_used", "ols_coefficients", "ols_inference", "ols_fit_statistics",
               "interaction_test", "out_of_time_prediction"]

    def part_a():
        g = st.fit_price_model(train.clone())
        r = ref_ols(ref_clean(train))
        out = {"ols_rows_used": (int(g["n_used"]) == r["n_used"], f"got {g['n_used']}, expected {r['n_used']}")}
        out["ols_coefficients"] = dict_close(g["coefficients"], r["coefficients"], rtol=1e-6, atol=1e-6)
        se_ok, se_note = dict_close(g["std_errors"], r["std_errors"], rtol=1e-4)
        p_ok, p_note = dict_close(g["p_values"], r["p_values"], rtol=1e-3, atol=1e-10)
        out["ols_inference"] = (se_ok and p_ok, f"{se_note} {p_note}".strip())
        out["ols_fit_statistics"] = (
            all(close(g[k], r[k], rtol=1e-6) for k in ("r_squared", "adj_r_squared", "f_statistic")),
            f"got {[g.get(k) for k in ('r_squared', 'adj_r_squared', 'f_statistic')]}",
        )
        out["interaction_test"] = (
            close(g["interaction_f"], r["interaction_f"], rtol=1e-4) and close(g["interaction_p"], r["interaction_p"], rtol=1e-3, atol=1e-10),
            f"got F={g['interaction_f']}, p={g['interaction_p']}; expected F={r['interaction_f']:.5g}, p={r['interaction_p']:.5g}",
        )
        held = future_valid.select(raw_cols)
        pred = np.asarray(g["predict"](held.clone()), dtype=float).ravel()
        y = future_valid["resale_price"].to_numpy().astype(float)
        Xh, _ = ref_design(future_valid, r["levels"])
        ref_r2 = r2(y, Xh @ r["beta"])
        got_r2 = r2(y, pred) if pred.shape == y.shape else float("nan")
        out["out_of_time_prediction"] = (
            got_r2 >= ref_r2 - 0.01,
            f"held-out R² {got_r2:.4f} (needs >= {ref_r2 - 0.01:.4f}; {len(y)} rows from {cutoff})",
        )
        return out

    checks.guarded(a_names, part_a)

    # B — one-way ANOVA and Tukey HSD on a secret subsample of validated rows
    def part_b():
        valid = ref_clean(hdb).select(raw_cols)
        types = sorted(rng.choice(["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"], int(rng.integers(3, 5)), replace=False).tolist())
        parts = []
        for t in types:
            sub = valid.filter(pl.col("flat_type") == t)
            parts.append(sub[rng.choice(sub.height, int(rng.integers(40, 120)), replace=False).tolist()])
        sample = pl.concat(parts)
        order = [str(t) for t in rng.permutation(types)]
        g = st.anova_flat_types(sample.clone(), order)
        groups = [(sample.filter(pl.col("flat_type") == t)["resale_price"] / sample.filter(pl.col("flat_type") == t)["floor_area_sqm"]).to_numpy() for t in order]
        f, p = stats.f_oneway(*groups)
        allv = np.concatenate(groups)
        eta = sum(x.size * (x.mean() - allv.mean()) ** 2 for x in groups) / ((allv - allv.mean()) ** 2).sum()
        tk = stats.tukey_hsd(*groups)
        ref_t = {}
        for i, j in combinations(range(len(order)), 2):
            a, b = sorted([order[i], order[j]])
            ref_t[f"{a}|{b}"] = float(tk.pvalue[i, j])
        anova_ok = close(g["f_stat"], f, rtol=1e-6) and close(g["p_value"], p, rtol=1e-4, atol=1e-12) and close(g["eta_squared"], eta, rtol=1e-6)
        gt = g["tukey"]
        tk_ok = set(gt) == set(ref_t) and all(
            close(gt[k]["p_adj"], v, rtol=1e-3, atol=2e-3) and (bool(gt[k]["significant"]) == (v < 0.05) or abs(v - 0.05) < 2e-3)
            for k, v in ref_t.items()
        )
        return {
            "anova": (anova_ok, f"got F={g['f_stat']}, p={g['p_value']}, eta²={g['eta_squared']}; expected {f:.5g}, {p:.5g}, {eta:.5g}"),
            "tukey_hsd": (tk_ok, f"got { {k: v.get('p_adj') for k, v in gt.items()} }, expected {ref_t}"),
        }

    checks.guarded(["anova", "tukey_hsd"], part_b)

    # C — logistic regression: odds ratios per SD, Wald test, held-out AUC
    def part_c():
        credit = loader.load("mlfp02", "sg_credit_scoring.parquet")
        perm = rng.permutation(credit.height)
        n_tr = int(rng.integers(15_000, 30_000))
        tr = credit[np.sort(perm[:n_tr]).tolist()]
        ho = credit[np.sort(perm[n_tr:n_tr + 15_000]).tolist()]
        g = st.fit_default_model(tr.clone())
        Z = tr.select(LOGIT_FEATURES).to_numpy().astype(float)
        mu, sd = Z.mean(0), Z.std(0)
        fit = sm.Logit(tr["default"].to_numpy(), sm.add_constant((Z - mu) / sd)).fit(disp=0, tol=1e-12, maxiter=200)
        b, V = fit.params[1:], fit.cov_params()[1:, 1:]
        ref_or = {f: float(np.exp(v)) for f, v in zip(LOGIT_FEATURES, b)}
        ref_se = {f: float(np.sqrt(V[k, k])) for k, f in enumerate(LOGIT_FEATURES)}
        o = np.argsort(-np.abs(b))
        i, j = int(o[0]), int(o[1])
        pdiff = float(2 * stats.norm.sf(abs(b[i] - b[j]) / np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])))
        or_ok, or_note = dict_close(g["odds_ratios"], ref_or, rtol=1e-4)
        se_ok, se_note = dict_close(g["std_errors"], ref_se, rtol=1e-3)
        top_ok = (
            set(g["top_two"]) == {LOGIT_FEATURES[i], LOGIT_FEATURES[j]}
            and abs(float(g["top_two_p"]) - pdiff) <= 0.01
            and (bool(g["top_two_differ"]) == (pdiff < 0.05) or abs(pdiff - 0.05) < 0.01)
        )
        s = np.asarray(g["predict_proba"](ho.clone()), dtype=float).ravel()
        yh = ho["default"].to_numpy()
        Zh = (ho.select(LOGIT_FEATURES).to_numpy().astype(float) - mu) / sd
        ref_auc = auc(yh, Zh @ b)
        got_auc = auc(yh, s) if s.shape == yh.shape else float("nan")
        prob_ok = s.shape == yh.shape and bool(np.all((s >= 0) & (s <= 1)))
        return {
            "odds_ratios": (or_ok, or_note),
            "logit_std_errors": (se_ok, se_note),
            "top_two_comparison": (top_ok, f"got {g['top_two']}, p={g['top_two_p']}, differ={g['top_two_differ']}; expected {[LOGIT_FEATURES[i], LOGIT_FEATURES[j]]}, p={pdiff:.4f}"),
            "held_out_auc": (prob_ok and got_auc >= ref_auc - 0.01, f"AUC {got_auc:.4f} (needs >= {ref_auc - 0.01:.4f}); probabilities in [0,1]: {prob_ok}"),
        }

    checks.guarded(["odds_ratios", "logit_std_errors", "top_two_comparison", "held_out_auc"], part_c)
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
