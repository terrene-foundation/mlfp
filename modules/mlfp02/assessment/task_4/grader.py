#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP02 Assessment Task 4 — Difference-in-Differences
(instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

The grader simulates fresh policy panels with SECRET true effects, numbers of
periods, policy timing, group sizes (unbalanced across periods) and — for some
panels — a planted pre-existing trend gap. The submission is graded against
the planted truth and an independent reference computed on the same panel, so
a hard-coded or formula-only answer cannot pass and an analysis that skips
the parallel-trends / placebo diagnostics labels the broken panels wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, close, finalize, load_student_module, main  # noqa: E402

WEIGHT = 15


def simulate(rng: np.random.Generator, violate: bool) -> tuple[pl.DataFrame, float]:
    n_pre, n_post = int(rng.integers(5, 10)), int(rng.integers(3, 8))
    effect = float(rng.uniform(-40_000, 40_000))
    base_c, gap = rng.uniform(300_000, 500_000), rng.uniform(-80_000, 120_000)
    trend, sd = rng.uniform(-3_000, 6_000), rng.uniform(40_000, 90_000)
    extra = rng.choice([-1, 1]) * rng.uniform(9_000, 15_000) if violate else 0.0
    frames = []
    for t in range(n_pre + n_post):
        post = int(t >= n_pre)
        for g in (0, 1):
            n = int(rng.integers(80, 260))
            mean = base_c + g * gap + t * trend + g * t * extra + g * post * effect
            frames.append(pl.DataFrame({"period": [t] * n, "treated": [g] * n, "post": [post] * n,
                                        "y": rng.normal(mean, sd, size=n)}))
    return pl.concat(frames).sample(fraction=1.0, shuffle=True, seed=int(rng.integers(1 << 31))), effect


def _fit(y, cols, robust=False):
    X = sm.add_constant(np.column_stack(cols))
    fit = sm.OLS(y, X).fit(cov_type="HC1") if robust else sm.OLS(y, X).fit()
    return fit.params[-1], fit.bse[-1], fit.pvalues[-1]


def reference(panel: pl.DataFrame) -> dict:
    """Independent reference: regression forms of DiD, pre-trend and placebo
    (interaction coefficient = last column)."""
    def arr(df, c):
        return df[c].to_numpy().astype(float)

    y, g, p = arr(panel, "y"), arr(panel, "treated"), arr(panel, "post")
    att, se_r, _ = _fit(y, [g, p, g * p], robust=True)
    pre = panel.filter(pl.col("post") == 0)
    yp, gp, tp = arr(pre, "y"), arr(pre, "treated"), arr(pre, "period")
    slope, _, trend_p = _fit(yp, [tp, gp, gp * tp])
    periods = sorted(pre["period"].unique().to_list())
    fake = (tp >= periods[len(periods) // 2]).astype(float)
    p_att, _, p_p = _fit(yp, [gp, fake, gp * fake], robust=True)
    return {"att": float(att), "se_robust": float(se_r), "slope": float(slope), "trend_p": float(trend_p),
            "placebo_att": float(p_att), "placebo_p": float(p_p)}


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_task4")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    if not callable(getattr(st, "did_analysis", None)):
        return finalize(checks, WEIGHT, seed, "Missing function: did_analysis")
    rng = np.random.default_rng(seed)

    panels = [simulate(rng, False), simulate(rng, False), simulate(rng, True), simulate(rng, True)]
    names = ["att_estimate", "standard_error_and_ci", "ci_covers_truth", "pre_trend_test",
             "placebo_test", "credibility_verdict"]

    def run():
        res = [st.did_analysis(p.clone()) for p, _ in panels]
        refs = [reference(p) for p, _ in panels]
        att_ok = all(close(r["att"], f["att"], rtol=1e-9, atol=1e-6) for r, f in zip(res, refs))
        se_ok = all(
            abs(float(r["se"]) / f["se_robust"] - 1) <= 0.15
            and close(r["ci_low"], r["att"] - 1.959964 * r["se"], rtol=1e-3, atol=1.0)
            and close(r["ci_high"], r["att"] + 1.959964 * r["se"], rtol=1e-3, atol=1.0)
            for r, f in zip(res, refs)
        )
        # truth coverage on the two valid panels (fails for a CI that ignores the data)
        cover = all(r["ci_low"] - f["se_robust"] <= eff <= r["ci_high"] + f["se_robust"]
                    for r, f, (_, eff) in zip(res[:2], refs[:2], panels[:2]))
        trend_ok = all(
            close(r["pre_trend_slope_diff"], f["slope"], rtol=1e-6, atol=1e-6)
            and (abs(float(r["pre_trend_p"]) - f["trend_p"]) <= 0.05 or float(r["pre_trend_p"]) < 1e-4 > f["trend_p"])
            for r, f in zip(res, refs)
        )
        placebo_ok = all(
            close(r["placebo_att"], f["placebo_att"], rtol=1e-9, atol=1e-6)
            and abs(float(r["placebo_p"]) - f["placebo_p"]) <= 0.05
            for r, f in zip(res, refs)
        )
        # verdict: valid panels are credible unless the reference diagnostics fire;
        # violated panels must be flagged
        verdict_ok = True
        for r, f, k in zip(res, refs, range(4)):
            want = f["trend_p"] >= 0.05 and f["placebo_p"] >= 0.05
            borderline = min(abs(f["trend_p"] - 0.05), abs(f["placebo_p"] - 0.05)) < 0.02
            if bool(r["credible"]) != want and not borderline:
                verdict_ok = False
        notes = f"student {[{k: round(float(v), 4) for k, v in r.items()} for r in res]}; reference {refs}; truth {[round(e) for _, e in panels]}"
        return {n: (ok, notes) for n, ok in zip(names, [att_ok, se_ok, cover, trend_ok, placebo_ok, verdict_ok])}

    checks.guarded(names, run)
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
