#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP02 Assessment Task 2 — Experiment Read-out (instructor-side).

    python grader.py submission.py [--seed N]

The submission is run on three experiments the student has never seen, all
built here from the real log with a fresh secret seed:

  A  a random subsample of control + treatment_a           (healthy allocation)
  B  the same kind of sample with a planted logging bug that silently drops
     non-converting treatment users                        (SRM → must not ship)
  C  control users only, randomly relabelled into two arms  (true effect = 0)

plus synthetic allocation counts with a planted faulty arm and random
power-analysis inputs. References are recomputed here; resampling answers are
graded with Monte-Carlo-aware tolerances. The ExperimentTracker run is read
back from the store and compared with the grader's own reference numbers.
"""
from __future__ import annotations

import asyncio
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, close, finalize, load_student_module, main  # noqa: E402

from shared import MLFPDataLoader  # noqa: E402

WEIGHT = 25
THR = 50.0
DESIGN = {"control": 0.40, "treatment_a": 0.35, "treatment_b": 0.15, "variant_c": 0.10}
SRM_ALPHA, ALPHA = 0.01, 0.05
Z = stats.norm.ppf(0.975)


# ── references ─────────────────────────────────────────────────────────


def ref_srm(groups: list[str], design: dict[str, float]) -> dict:
    arms = list(design)
    s = sum(design.values())
    obs = np.array([sum(1 for g in groups if g == a) for a in arms], dtype=float)
    exp = obs.sum() * np.array([design[a] / s for a in arms])
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, len(arms) - 1))
    return {"chi2": chi2, "p_value": p, "srm": p < SRM_ALPHA,
            "worst_arm": arms[int(np.argmax(np.abs((obs - exp) / np.sqrt(exp))))]}


def ref_srm_df(df: pl.DataFrame, design: dict[str, float]) -> dict:
    return ref_srm(df["experiment_group"].to_list(), design)


def ref_n(b: float, mde: float, alpha: float, power: float) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z**2 * (b * (1 - b) + (b + mde) * (1 - b - mde)) / mde**2


def arrays(df: pl.DataFrame, arm: str):
    s = df.filter(pl.col("experiment_group") == arm)
    return (s["metric_value"].to_numpy().astype(float), s["pre_metric_value"].to_numpy().astype(float))


def two_prop_p(c: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """(pooled, unpooled) two-sided z-test p-values for a difference in proportions."""
    pc, pt = c.mean(), t.mean()
    pp = (c.sum() + t.sum()) / (c.size + t.size)
    se_p = math.sqrt(pp * (1 - pp) * (1 / c.size + 1 / t.size))
    se_u = math.sqrt(pc * (1 - pc) / c.size + pt * (1 - pt) / t.size)
    d = pt - pc
    f = lambda se: float(2 * stats.norm.sf(abs(d) / se)) if se > 0 else 1.0  # noqa: E731
    return f(se_p), f(se_u)


def p_ok(got, refs: tuple[float, float]) -> bool:
    try:
        g = float(got)
    except (TypeError, ValueError):
        return False
    return any(abs(g - r) <= 1e-3 + 0.1 * r for r in refs)


def ref_ab(df: pl.DataFrame, rng: np.random.Generator) -> dict:
    yc, xc = arrays(df, "control")
    yt, xt = arrays(df, "treatment_a")
    cc, ct = yc >= THR, yt >= THR
    lift = ct.mean() - cc.mean()
    se_u = math.sqrt(cc.mean() * (1 - cc.mean()) / cc.size + ct.mean() * (1 - ct.mean()) / ct.size)
    pp, pu = two_prop_p(cc, ct)
    diff = yt.mean() - yc.mean()
    se_w = math.sqrt(yt.var(ddof=1) / yt.size + yc.var(ddof=1) / yc.size)
    boot = np.array([rng.choice(yt, yt.size).mean() - rng.choice(yc, yc.size).mean() for _ in range(4000)])
    welch_p = float(stats.ttest_ind(yt, yc, equal_var=False).pvalue)
    y, x = np.concatenate([yt, yc]), np.concatenate([xt, xc])
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    adj = y - theta * (x - x.mean())
    at, ac = adj[: yt.size], adj[yt.size:]
    cdiff = float(at.mean() - ac.mean())
    se_c = math.sqrt(at.var(ddof=1) / at.size + ac.var(ddof=1) / ac.size)
    pair_srm = ref_srm_df(df, {"control": DESIGN["control"], "treatment_a": DESIGN["treatment_a"]})
    ship = (not pair_srm["srm"]) and lift > 0 and pp < ALPHA
    return {
        "pair_srm_p": pair_srm["p_value"], "conv_control": cc.mean(), "conv_treatment": ct.mean(),
        "conv_lift": lift, "conv_ci_low": lift - Z * se_u, "conv_ci_high": lift + Z * se_u,
        "se_u": se_u, "p_pooled": pp, "p_unpooled": pu, "mean_diff": diff, "se_w": se_w,
        "boot_ci_low": np.percentile(boot, 2.5), "boot_ci_high": np.percentile(boot, 97.5),
        "welch_p": welch_p, "cuped_theta": theta,
        "cuped_var_reduction": 1 - adj.var(ddof=1) / y.var(ddof=1),
        "cuped_diff": cdiff, "cuped_ci_low": cdiff - Z * se_c, "cuped_ci_high": cdiff + Z * se_c,
        "se_c": se_c, "decision": "SHIP" if ship else "DO NOT SHIP",
    }


def ref_segment_p(df: pl.DataFrame) -> dict[str, tuple[float, float]]:
    d = df.with_columns((pl.col("metric_value") >= THR).alias("conv"))
    out = {}
    for (seg, plat), g in d.group_by(["segment", "platform"]):
        c = g.filter(pl.col("experiment_group") == "control")["conv"].to_numpy()
        t = g.filter(pl.col("experiment_group") == "treatment_a")["conv"].to_numpy()
        if c.size and t.size:
            out[f"{seg}|{plat}"] = two_prop_p(c, t)
    return out


def bh_set(p: dict[str, float], q: float = ALPHA) -> list[str]:
    items = sorted(p.items(), key=lambda kv: kv[1])
    k = max([i for i, (_, v) in enumerate(items, 1) if v <= q * i / len(items)], default=0)
    return sorted(n for n, _ in items[:k])


# ── grader-held experiments ────────────────────────────────────────────


def build_experiments(full: pl.DataFrame, rng: np.random.Generator):
    pair = full.filter(pl.col("experiment_group").is_in(["control", "treatment_a"]))

    def sample(frame: pl.DataFrame, n: int) -> pl.DataFrame:
        return frame[np.sort(rng.choice(frame.height, n, replace=False)).tolist()]

    a = sample(pair, int(rng.integers(6000, 9000)))
    b0 = sample(pair, int(rng.integers(7000, 9000)))
    drop = (
        (pl.col("experiment_group") == "treatment_a")
        & (pl.col("metric_value") < THR)
        & pl.Series(rng.random(b0.height) < rng.uniform(0.12, 0.2))
    )
    b = b0.filter(~drop)
    ctrl = full.filter(pl.col("experiment_group") == "control")
    c = sample(ctrl, int(rng.integers(6000, 9000)))
    labels = np.where(rng.random(c.height) < 35 / 75, "treatment_a", "control")
    c = c.with_columns(pl.Series("experiment_group", labels))
    # Subgroup sample: redraw until Benjamini-Hochberg and Bonferroni disagree,
    # so the two corrections cannot be confused without failing.
    for _ in range(40):
        seg = sample(pair, int(rng.integers(6000, 25000)))
        p = {k: v[0] for k, v in ref_segment_p(seg).items()}
        bonf = sorted(k for k, v in p.items() if v < ALPHA / len(p))
        if bh_set(p) != bonf:
            break
    return a, b, c, seg


def synthetic_allocation(rng: np.random.Generator) -> tuple[pl.DataFrame, dict[str, float], str]:
    arms = ["control", "treatment_a", "treatment_b", "variant_c"]
    w = rng.dirichlet(np.ones(4) * 4) * 0.8 + 0.05
    design = {a: float(v / w.sum()) for a, v in zip(arms, w)}
    n = int(rng.integers(20000, 40000))
    counts = rng.multinomial(n, list(design.values()))
    bad = arms[int(rng.integers(4))]
    counts[arms.index(bad)] = int(counts[arms.index(bad)] * rng.uniform(1.06, 1.12))
    groups = np.repeat(arms, counts)
    rng.shuffle(groups)
    df = pl.DataFrame({"user_id": [f"U{i}" for i in range(groups.size)], "experiment_group": groups})
    return df, design, bad


# ── grading ────────────────────────────────────────────────────────────


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_task2")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    fns = ("srm_check", "sample_size_per_arm", "analyse_ab", "segment_tests", "log_to_tracker")
    missing = [f for f in fns if not callable(getattr(st, f, None))]
    if missing:
        return finalize(checks, WEIGHT, seed, f"Missing functions: {missing}")

    rng = np.random.default_rng(seed)
    full = MLFPDataLoader().load("mlfp02", "experiment_data.parquet")
    A, B, C, SEG = build_experiments(full, rng)

    # 1 — SRM on the real four-arm log and on a planted faulty allocation
    def srm():
        g = st.srm_check(full.clone(), dict(DESIGN))
        r = ref_srm_df(full, DESIGN)
        ok1 = close(g["chi2"], r["chi2"], rtol=1e-6) and bool(g["srm"]) is True and g["worst_arm"] == r["worst_arm"]
        notes = [] if ok1 else [f"real log: got {g}, expected {r}"]
        ok2 = True
        for _ in range(2):
            df, design, bad = synthetic_allocation(rng)
            g2 = st.srm_check(df, design)
            r2 = ref_srm_df(df, design)
            if not (close(g2["chi2"], r2["chi2"], rtol=1e-6)
                    and close(g2["p_value"], r2["p_value"], rtol=1e-4, atol=1e-12)
                    and bool(g2["srm"]) == r2["srm"] and g2["worst_arm"] == r2["worst_arm"]):
                ok2 = False
                notes.append(f"planted fault in {bad}: got {g2}, expected {r2}")
        return {"srm_check": (ok1 and ok2, "; ".join(notes))}

    checks.guarded(["srm_check"], srm)

    # 2 — power analysis on random planning inputs
    def power():
        bad = []
        for _ in range(5):
            b, alpha, pw = float(rng.uniform(0.05, 0.4)), float(rng.choice([0.01, 0.05, 0.10])), float(rng.choice([0.8, 0.9]))
            mde = float(b * rng.uniform(0.05, 0.3))
            g, r = st.sample_size_per_arm(b, mde, alpha, pw), ref_n(b, mde, alpha, pw)
            if not (isinstance(g, (int, np.integer)) and abs(g - r) <= 0.03 * r + 1):
                bad.append(f"(b={b:.3f}, mde={mde:.4f}, alpha={alpha}, power={pw}): got {g}, expected ≈{math.ceil(r)}")
        return {"sample_size": (not bad, "; ".join(bad))}

    checks.guarded(["sample_size"], power)

    # 3 — the read-out on A (healthy), B (logging bug) and C (null effect)
    names = ["conversion_estimates", "conversion_inference", "bootstrap_ci", "permutation_test",
             "cuped", "pair_srm", "ship_decisions"]

    def readout():
        res, refs = {}, {}
        for label, df in (("A", A), ("B", B), ("C", C)):
            res[label] = st.analyse_ab(df.clone(), "treatment_a", dict(DESIGN))
            refs[label] = ref_ab(df, rng)
        out = {}
        g, r = res["A"], refs["A"]
        out["conversion_estimates"] = (
            all(close(g[k], r[k], rtol=1e-9) for k in ("conv_control", "conv_treatment", "conv_lift")),
            f"A: got {[g.get(k) for k in ('conv_control', 'conv_treatment', 'conv_lift')]}",
        )
        inf_ok, notes = True, []
        for lab in ("A", "C"):
            g, r = res[lab], refs[lab]
            ci = abs(g["conv_ci_low"] - r["conv_ci_low"]) <= 0.1 * r["se_u"] and abs(g["conv_ci_high"] - r["conv_ci_high"]) <= 0.1 * r["se_u"]
            pv = p_ok(g["conv_p"], (r["p_pooled"], r["p_unpooled"]))
            if not (ci and pv):
                inf_ok = False
                notes.append(f"{lab}: CI [{g['conv_ci_low']}, {g['conv_ci_high']}] vs [{r['conv_ci_low']:.5f}, {r['conv_ci_high']:.5f}]; p {g['conv_p']} vs {r['p_pooled']:.4g}")
        out["conversion_inference"] = (inf_ok, "; ".join(notes))
        boot_ok, notes = True, []
        for lab in ("A", "C"):
            g, r = res[lab], refs[lab]
            if not (close(g["mean_diff"], r["mean_diff"], rtol=1e-9, atol=1e-9)
                    and abs(g["boot_ci_low"] - r["boot_ci_low"]) <= 0.25 * r["se_w"]
                    and abs(g["boot_ci_high"] - r["boot_ci_high"]) <= 0.25 * r["se_w"]):
                boot_ok = False
                notes.append(f"{lab}: diff {g['mean_diff']} CI [{g['boot_ci_low']}, {g['boot_ci_high']}] vs {r['mean_diff']:.4f} [{r['boot_ci_low']:.4f}, {r['boot_ci_high']:.4f}]")
        out["bootstrap_ci"] = (boot_ok, "; ".join(notes))
        gA, gC = float(res["A"]["perm_p"]), float(res["C"]["perm_p"])
        out["permutation_test"] = (
            gA <= 0.01 and abs(gC - refs["C"]["welch_p"]) <= 0.04 and 0 < gC <= 1,
            f"A perm_p {gA} (expected <= 0.01); C perm_p {gC} vs ≈{refs['C']['welch_p']:.4f}",
        )
        cup_ok, notes = True, []
        for lab in ("A", "C"):
            g, r = res[lab], refs[lab]
            if not (close(g["cuped_theta"], r["cuped_theta"], rtol=0.02)
                    and abs(g["cuped_var_reduction"] - r["cuped_var_reduction"]) <= 0.01
                    and abs(g["cuped_diff"] - r["cuped_diff"]) <= 0.1 * r["se_c"]
                    and abs(g["cuped_ci_low"] - r["cuped_ci_low"]) <= 0.1 * r["se_c"]
                    and abs(g["cuped_ci_high"] - r["cuped_ci_high"]) <= 0.1 * r["se_c"]):
                cup_ok = False
                notes.append(f"{lab}: theta {g['cuped_theta']} vs {r['cuped_theta']:.4f}; reduction {g['cuped_var_reduction']} vs {r['cuped_var_reduction']:.4f}; diff {g['cuped_diff']} vs {r['cuped_diff']:.4f}")
        out["cuped"] = (cup_ok, "; ".join(notes))
        out["pair_srm"] = (
            all(close(res[l]["pair_srm_p"], refs[l]["pair_srm_p"], rtol=1e-4, atol=1e-12) for l in ("A", "B")),
            f"got {[res[l]['pair_srm_p'] for l in 'AB']}, expected {[refs[l]['pair_srm_p'] for l in 'AB']}",
        )
        out["ship_decisions"] = (
            all(res[l]["decision"] == refs[l]["decision"] for l in "ABC"),
            f"got {[res[l]['decision'] for l in 'ABC']}, expected {[refs[l]['decision'] for l in 'ABC']}",
        )
        readout.cache = (res["A"], refs["A"])
        return out

    checks.guarded(names, readout)

    # 4 — subgroup p-values and both multiple-testing corrections
    def segments():
        g = st.segment_tests(SEG.clone(), "treatment_a")
        r = ref_segment_p(SEG)
        pv = g["p_values"]
        keys_ok = set(pv) == set(r)
        vals_ok = keys_ok and all(p_ok(pv[k], r[k]) for k in r)
        m = len(pv)
        bonf = sorted(k for k, p in pv.items() if p < ALPHA / m)
        corr_ok = keys_ok and sorted(g["bonferroni_significant"]) == bonf and sorted(g["bh_significant"]) == bh_set(pv)
        return {
            "segment_p_values": (vals_ok, f"keys {sorted(pv)} vs {sorted(r)}"),
            "multiple_testing": (corr_ok, f"bonferroni {g['bonferroni_significant']} vs {bonf}; BH {g['bh_significant']} vs {bh_set(pv)}"),
        }

    checks.guarded(["segment_p_values", "multiple_testing"], segments)

    # 5 — the read-out is recorded in a Kailash ExperimentTracker store
    def tracker():
        cached = getattr(readout, "cache", None)
        if cached is None:
            return {"tracker_run": (False, "read-out on dataset A failed, nothing to log")}
        res_a, ref_a = cached
        with tempfile.TemporaryDirectory() as tmp:
            url = f"sqlite:///{Path(tmp, 'grader_tracker.db').as_posix()}"
            run_id = st.log_to_tracker(dict(res_a), url)

            async def read():
                from kailash_ml import ExperimentTracker

                tr = await ExperimentTracker.create(store_url=url)
                try:
                    rec = await tr.get_run(str(run_id))
                    met = await tr.list_metrics(str(run_id))
                    return rec, met
                finally:
                    await tr.close()

            rec, met = asyncio.run(read())
        m = dict(zip(met["key"].to_list(), met["value"].to_list()))
        ok = (rec.status == "FINISHED"
              and close(m.get("conv_lift"), ref_a["conv_lift"], rtol=1e-9)
              and abs(float(m.get("cuped_diff", np.nan)) - ref_a["cuped_diff"]) <= 0.1 * ref_a["se_c"]
              and str(rec.params.get("decision")) == ref_a["decision"])
        return {"tracker_run": (ok, f"status {rec.status}, params {rec.params}, metrics {sorted(m)}")}

    checks.guarded(["tracker_run"], tracker)
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
