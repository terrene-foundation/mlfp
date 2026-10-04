#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP01 Assessment Task 4 — Profile, Clean and Justify.

Usage:
    python grader.py starter.py
    python grader.py solution.py

Nothing the submission reports about itself is trusted:
  - raw alerts are re-profiled here with DataExplorer (via shared.run_profile)
  - the cleaned frame is re-profiled here, and every alert that remains must
    be justified (and only those)
  - imputed values are recomputed from the declared method
  - the submitted AlertConfig is exercised on grader-built frames
  - the function is re-run on an unseen variant of the file
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

import polars as pl

from kailash_ml import AlertConfig
from shared import MLFPDataLoader, run_profile

WEIGHT = 20
GATES = ("returns_contract",)
CHECKS = [
    *GATES,
    "raw_alerts",
    "one_row_per_quarter",
    "tourist_arrivals",
    "untouched_values",
    "imputation",
    "null_alert_config",
    "accepted_alerts",
    "quality_delta",
    "unseen_variant",
]
KEYS = {"raw_alerts", "cleaned", "imputation", "null_alert_config",
        "accepted_alerts", "quality_delta"}  # fmt: skip
COLUMNS = ["period_year", "period_quarter", "gdp_growth_pct", "unemployment_rate",
           "inflation_rate", "trade_balance_sgd_bn", "property_price_index",
           "tourist_arrivals"]  # fmt: skip
IMPUTED = ["inflation_rate", "trade_balance_sgd_bn"]
UNTOUCHED = ["gdp_growth_pct", "unemployment_rate", "property_price_index"]
METHODS = {"median", "mean", "interpolate", "forward_fill"}


def _alert_key(a: dict) -> str:
    if a.get("columns"):
        return f"{a['type']}:{','.join(sorted(a['columns']))}"
    if a.get("column"):
        return f"{a['type']}:{a['column']}"
    return a["type"]


# ── Ground truth ─────────────────────────────────────────────────────────
def _period(text: str) -> tuple[int, int]:
    for pat in (r"Q([1-4])\s+(\d{4})", r"(\d{4})-Q([1-4])", r"(\d{4})-([1-4])"):
        m = re.fullmatch(pat, text.strip())
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return (b, a) if pat.startswith("Q") else (a, b)
    raise ValueError(f"unparseable period {text!r}")


def _deduped(raw: pl.DataFrame) -> list[dict]:
    """Quarterly rows, one per quarter (first occurrence), in time order."""
    seen: dict[tuple[int, int], dict] = {}
    for r in raw.filter(pl.col("period_type") == "quarterly").iter_rows(named=True):
        key = _period(r["period"])
        if key not in seen:
            seen[key] = {
                "period_year": key[0],
                "period_quarter": key[1],
                **{c: r[c] for c in UNTOUCHED + IMPUTED},
                "tourist_arrivals": int(r["tourist_arrivals"].replace(",", "").strip()),
            }
    return [seen[k] for k in sorted(seen)]


def _impute(values: list[float | None], method: str) -> list[float]:
    known = [v for v in values if v is not None]
    if method == "median":
        fill = statistics.median(known)
        return [fill if v is None else v for v in values]
    if method == "mean":
        fill = sum(known) / len(known)
        return [fill if v is None else v for v in values]
    if method == "forward_fill":
        out, last = [], None
        for v in values:
            last = v if v is not None else last
            out.append(last)
        return out
    out = list(values)  # interpolate: linear between nearest known neighbours
    for i, v in enumerate(values):
        if v is None:
            lo = max(j for j in range(i) if values[j] is not None)
            hi = min(j for j in range(i + 1, len(values)) if values[j] is not None)
            out[i] = values[lo] + (values[hi] - values[lo]) * (i - lo) / (hi - lo)
    return out


def _expected(raw: pl.DataFrame, method: str) -> pl.DataFrame:
    rows = _deduped(raw)
    df = pl.DataFrame(rows, infer_schema_length=None)
    return df.with_columns(
        pl.Series(c, _impute(df[c].to_list(), method), dtype=pl.Float64) for c in IMPUTED
    ).select(COLUMNS)


def _variant(raw: pl.DataFrame) -> pl.DataFrame:
    """Shuffled rows, a quarter recorded twice under another spelling, new gaps."""
    rows = raw.to_dicts()[::-1]
    for r in rows:
        if r["period_type"] != "quarterly":
            continue
        y, q = _period(r["period"])
        if (y, q) in {(2006, 2), (2016, 4)}:
            r["inflation_rate"] = None
        if (y, q) == (2012, 1):
            r["trade_balance_sgd_bn"] = None
    extra = next(dict(r) for r in rows if r["period_type"] == "quarterly"
                 and _period(r["period"]) == (2010, 3))  # fmt: skip
    extra["period"] = "Q3 2010" if extra["period"] != "Q3 2010" else "2010-3"
    rows.insert(37, extra)
    return pl.DataFrame(rows, schema=raw.schema)


# ── Comparison ───────────────────────────────────────────────────────────
def _usable(out) -> bool:
    return (
        isinstance(out, dict)
        and KEYS <= set(out)
        and isinstance(out["cleaned"], pl.DataFrame)
        and all(c in out["cleaned"].columns for c in COLUMNS)
    )


def _aligned(cleaned: pl.DataFrame, exp: pl.DataFrame) -> pl.DataFrame | None:
    try:
        s = cleaned.select(
            pl.col("period_year").cast(pl.Int64), pl.col("period_quarter").cast(pl.Int64),
            *[pl.col(c).alias(f"s_{c}") for c in COLUMNS[2:]],
        )
        keys = s.select("period_year", "period_quarter")
        if keys.height != exp.height or keys.unique().height != keys.height:
            return None
        j = exp.join(s, on=["period_year", "period_quarter"], how="inner")
        return j if j.height == exp.height else None
    except Exception:
        return None


def _equal(j: pl.DataFrame, cols: list[str], tol: float = 1e-9) -> bool:
    for c in cols:
        b = j[f"s_{c}"]
        if b.null_count() or ((j[c].cast(pl.Float64) - b.cast(pl.Float64)).abs() > tol).any():
            return False
    return True


def _config_ok(cfg) -> bool:
    if not isinstance(cfg, AlertConfig):
        return False
    one_gap = pl.DataFrame({"x": [float(i) for i in range(4999)] + [None],
                            "y": [float(i % 7) for i in range(5000)]})  # fmt: skip
    complete = one_gap.with_columns(pl.col("x").fill_null(0.0))
    fires = {_alert_key(a) for a in run_profile(one_gap, cfg).alerts}
    quiet = {_alert_key(a) for a in run_profile(complete, cfg).alerts}
    return "high_nulls:x" in fires and not any(k.startswith("high_nulls") for k in quiet)


def _accepted_ok(cleaned: pl.DataFrame, accepted) -> bool:
    remaining = {_alert_key(a) for a in run_profile(cleaned.select(COLUMNS)).alerts}
    if any(k.split(":")[0] in {"high_nulls", "duplicates"} for k in remaining):
        return False
    return (
        isinstance(accepted, dict)
        and set(accepted) == remaining
        and all(isinstance(v, str) and len(v.strip()) >= 20 for v in accepted.values())
    )


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {c: False for c in CHECKS}}
    c = score["checks"]
    try:
        fn = load_student_module(student_path).audit_indicators
    except Exception as e:
        score["error"] = f"Cannot load audit_indicators(): {type(e).__name__}: {e}"
        return _finalize(score)

    raw = MLFPDataLoader().load("mlfp01", "economic_indicators.csv")
    try:
        out = fn(raw.clone())
    except Exception as e:
        score["error"] = f"audit_indicators() raised {type(e).__name__}: {e}"
        return _finalize(score)
    c["returns_contract"] = _usable(out)
    if not c["returns_contract"]:
        score["error"] = "Return value does not follow the dict contract in problem.md"
        return _finalize(score)

    raw_q = raw.filter(pl.col("period_type") == "quarterly")
    truth = {_alert_key(a) for a in run_profile(raw_q).alerts}
    try:
        got = list(out["raw_alerts"])
        c["raw_alerts"] = len(got) == len(set(got)) and set(got) == truth
    except Exception:
        c["raw_alerts"] = False

    method = out["imputation"] if out["imputation"] in METHODS else "median"
    exp = _expected(raw, method)
    j = _aligned(out["cleaned"], exp)
    c["one_row_per_quarter"] = j is not None
    if j is not None:
        c["tourist_arrivals"] = out["cleaned"].schema["tourist_arrivals"] == pl.Int64 and _equal(
            j, ["tourist_arrivals"], tol=0.0
        )
        c["untouched_values"] = _equal(j, UNTOUCHED)
        c["imputation"] = out["imputation"] in METHODS and _equal(j, IMPUTED, tol=1e-6)

    try:
        c["null_alert_config"] = _config_ok(out["null_alert_config"])
    except Exception:
        c["null_alert_config"] = False
    try:
        c["accepted_alerts"] = _accepted_ok(out["cleaned"], out["accepted_alerts"])
    except Exception:
        c["accepted_alerts"] = False

    dedup = pl.DataFrame(_deduped(raw), infer_schema_length=None)
    truth_delta = {
        "rows_removed": raw_q.height - dedup.height,
        "nulls_filled": int(sum(dedup[col].null_count() for col in IMPUTED)),
    }
    try:
        d = out["quality_delta"]
        c["quality_delta"] = all(d.get(k) == v for k, v in truth_delta.items())
    except Exception:
        c["quality_delta"] = False

    try:
        v_raw = _variant(raw)
        v_out = fn(v_raw.clone())
        if _usable(v_out) and v_out["imputation"] in METHODS:
            vj = _aligned(v_out["cleaned"], _expected(v_raw, v_out["imputation"]))
            c["unseen_variant"] = vj is not None and _equal(
                vj, UNTOUCHED + IMPUTED, tol=1e-6
            ) and _equal(vj, ["tourist_arrivals"], tol=0.0)
    except Exception as e:
        score["error"] = f"Unseen variant failed: {type(e).__name__}: {e}"
    score["expected"] = {"raw_alerts": sorted(truth), **truth_delta}
    return _finalize(score)


def _finalize(score: dict) -> dict:
    checks = score["checks"]
    score["total"] = sum(1 for v in checks.values() if v)
    score["max"] = len(CHECKS)
    earned = sum(1 for k, v in checks.items() if v and k not in GATES)
    gates_ok = all(checks[g] for g in GATES)
    score["marks"] = round(WEIGHT * earned / (len(CHECKS) - len(GATES)), 1) if gates_ok else 0.0
    score["passed"] = score["total"] == score["max"]
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
