# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 2: Experiment Read-out — SRM, Power, Inference, CUPED
(Reference Solution — withheld from students; verified to pass grader.py)
"""
from __future__ import annotations

import asyncio
import math

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

CONVERSION_THRESHOLD = 50.0
DESIGN = {"control": 0.40, "treatment_a": 0.35, "treatment_b": 0.15, "variant_c": 0.10}
SRM_ALPHA = 0.01
ALPHA = 0.05
N_RESAMPLES = 4000


def load_orders() -> pl.DataFrame:
    return MLFPDataLoader().load("mlfp02", "experiment_data.parquet")


# ── 1. Allocation health ──────────────────────────────────────────────


def srm_check(orders: pl.DataFrame, design: dict[str, float]) -> dict:
    arms = list(design)
    total_share = sum(design.values())
    counts = orders.group_by("experiment_group").len()
    obs = np.array(
        [counts.filter(pl.col("experiment_group") == a)["len"].sum() for a in arms],
        dtype=np.float64,
    )
    exp = obs.sum() * np.array([design[a] / total_share for a in arms])
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, df=len(arms) - 1))
    resid = (obs - exp) / np.sqrt(exp)
    return {
        "chi2": chi2,
        "p_value": p,
        "srm": bool(p < SRM_ALPHA),
        "worst_arm": arms[int(np.argmax(np.abs(resid)))],
    }


# ── 2. Planning ───────────────────────────────────────────────────────


def sample_size_per_arm(
    baseline: float, mde: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    p1, p2 = baseline, baseline + mde
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return math.ceil(z**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / mde**2)


# ── 3. The read-out ───────────────────────────────────────────────────


def analyse_ab(
    orders: pl.DataFrame, treatment: str, design: dict[str, float], seed: int = 7
) -> dict:
    rng = np.random.default_rng(seed)
    pair = orders.filter(pl.col("experiment_group").is_in(["control", treatment]))
    srm = srm_check(pair, {"control": design["control"], treatment: design[treatment]})

    c = pair.filter(pl.col("experiment_group") == "control")
    t = pair.filter(pl.col("experiment_group") == treatment)
    yc = c["metric_value"].to_numpy().astype(np.float64)
    yt = t["metric_value"].to_numpy().astype(np.float64)
    xc = c["pre_metric_value"].to_numpy().astype(np.float64)
    xt = t["pre_metric_value"].to_numpy().astype(np.float64)

    # conversion (order >= $50)
    cc, ct = (yc >= CONVERSION_THRESHOLD), (yt >= CONVERSION_THRESHOLD)
    pc, pt = cc.mean(), ct.mean()
    lift = pt - pc
    se_unpooled = math.sqrt(pc * (1 - pc) / yc.size + pt * (1 - pt) / yt.size)
    pp = (cc.sum() + ct.sum()) / (yc.size + yt.size)
    se_pooled = math.sqrt(pp * (1 - pp) * (1 / yc.size + 1 / yt.size))
    conv_p = float(2 * stats.norm.sf(abs(lift) / se_pooled))
    z = stats.norm.ppf(0.975)

    # order value: bootstrap CI and permutation test of the mean difference
    diff = yt.mean() - yc.mean()
    boot = np.array(
        [rng.choice(yt, yt.size).mean() - rng.choice(yc, yc.size).mean() for _ in range(N_RESAMPLES)]
    )
    pooled = np.concatenate([yt, yc])
    perm = np.empty(N_RESAMPLES)
    for i in range(N_RESAMPLES):
        rng.shuffle(pooled)
        perm[i] = pooled[: yt.size].mean() - pooled[yt.size :].mean()
    perm_p = float((np.sum(np.abs(perm) >= abs(diff)) + 1) / (N_RESAMPLES + 1))

    # CUPED with the PRE-period covariate, one theta from both arms
    y = np.concatenate([yt, yc])
    x = np.concatenate([xt, xc])
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    adj = y - theta * (x - x.mean())
    at, ac = adj[: yt.size], adj[yt.size :]
    cuped_diff = float(at.mean() - ac.mean())
    se_c = math.sqrt(at.var(ddof=1) / at.size + ac.var(ddof=1) / ac.size)

    significant_up = lift > 0 and conv_p < ALPHA
    decision = "SHIP" if (not srm["srm"] and significant_up) else "DO NOT SHIP"
    return {
        "pair_srm_p": srm["p_value"],
        "conv_control": float(pc),
        "conv_treatment": float(pt),
        "conv_lift": float(lift),
        "conv_ci_low": float(lift - z * se_unpooled),
        "conv_ci_high": float(lift + z * se_unpooled),
        "conv_p": conv_p,
        "mean_diff": float(diff),
        "boot_ci_low": float(np.percentile(boot, 2.5)),
        "boot_ci_high": float(np.percentile(boot, 97.5)),
        "perm_p": perm_p,
        "cuped_theta": theta,
        "cuped_var_reduction": float(1 - adj.var(ddof=1) / y.var(ddof=1)),
        "cuped_diff": cuped_diff,
        "cuped_ci_low": float(cuped_diff - z * se_c),
        "cuped_ci_high": float(cuped_diff + z * se_c),
        "decision": decision,
    }


# ── 4. Subgroup read-out with multiple-testing control ────────────────


def _bh(pvals: dict[str, float], q: float) -> list[str]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    k = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= q * i / m:
            k = i
    return sorted(name for name, _ in items[:k])


def segment_tests(orders: pl.DataFrame, treatment: str) -> dict:
    pair = orders.filter(pl.col("experiment_group").is_in(["control", treatment])).with_columns(
        (pl.col("metric_value") >= CONVERSION_THRESHOLD).alias("conv")
    )
    pvals: dict[str, float] = {}
    for (seg, plat), g in pair.group_by(["segment", "platform"]):
        c = g.filter(pl.col("experiment_group") == "control")["conv"].to_numpy()
        t = g.filter(pl.col("experiment_group") == treatment)["conv"].to_numpy()
        if c.size == 0 or t.size == 0:
            continue
        pp = (c.sum() + t.sum()) / (c.size + t.size)
        se = math.sqrt(pp * (1 - pp) * (1 / c.size + 1 / t.size))
        zz = (t.mean() - c.mean()) / se if se > 0 else 0.0
        pvals[f"{seg}|{plat}"] = float(2 * stats.norm.sf(abs(zz)))
    m = len(pvals)
    return {
        "p_values": pvals,
        "bonferroni_significant": sorted(k for k, p in pvals.items() if p < ALPHA / m),
        "bh_significant": _bh(pvals, ALPHA),
    }


# ── 5. Record the read-out ────────────────────────────────────────────


def log_to_tracker(results: dict, store_url: str) -> str:
    from kailash_ml import ExperimentTracker

    async def _log() -> str:
        tracker = await ExperimentTracker.create(store_url=store_url)
        try:
            run = await tracker.start_run("mlfp02_ab_readout", decision=results["decision"])
            await run.log_metrics(
                {k: float(v) for k, v in results.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            )
            await tracker.end_run(run)
            return run.run_id
        finally:
            await tracker.close()

    return asyncio.run(_log())


if __name__ == "__main__":
    orders = load_orders()
    print(srm_check(orders, DESIGN))
    print(sample_size_per_arm(0.19, 0.01))
    sample = orders.sample(20_000, seed=1)
    print(analyse_ab(sample, "treatment_a", DESIGN))
    print(segment_tests(sample, "treatment_a"))
