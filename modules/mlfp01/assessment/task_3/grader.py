#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP01 Assessment Task 3 — Town Price Trends.

Usage:
    python grader.py starter.py
    python grader.py solution.py

The grader computes the expected table with plain-Python calendar
arithmetic (date lookups, not row offsets), matches rows by (town, month),
and also runs the submission on an unseen variant with extra missing months
and new recording errors, so row-offset windows and hard-coded error values
fail.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from datetime import date
from pathlib import Path

import polars as pl

from shared import MLFPDataLoader

WEIGHT = 15
GATES = ("required_columns", "town_month_rows")
CHECKS = [
    *GATES,
    "n_sales",
    "median_price",
    "yoy_pct",
    "rolling_3m_avg",
    "price_rank_in_month",
    "unseen_variant",
]
VALUE_COLS = ["n_sales", "median_price", "yoy_pct", "rolling_3m_avg", "price_rank_in_month"]
REQUIRED = ["town", "month", *VALUE_COLS]


# ── Ground truth ─────────────────────────────────────────────────────────
def _shift(d: date, months: int) -> date:
    k = d.year * 12 + d.month - 1 + months
    return date(k // 12, k % 12 + 1, 1)


def _expected(hdb: pl.DataFrame) -> pl.DataFrame:
    prices: dict[tuple[str, date], list[int]] = {}
    for town, month, price in hdb.select("town", "month", "resale_price").iter_rows():
        # Genuine HDB resale prices in this data lie between ~S$215k and S$1.8M;
        # the recording errors are orders of magnitude outside that.
        if 100_000 <= price <= 2_000_000:
            key = (town, date(int(month[:4]), int(month[5:7]), 1))
            prices.setdefault(key, []).append(price)
    med = {k: float(statistics.median(v)) for k, v in prices.items()}
    by_month: dict[date, list[float]] = {}
    for (_, m), v in med.items():
        by_month.setdefault(m, []).append(v)

    rows = []
    for (town, m), v in med.items():
        prev = med.get((town, _shift(m, -12)))
        window = [med[(town, _shift(m, -k))] for k in range(3) if (town, _shift(m, -k)) in med]
        rows.append(
            {
                "town": town,
                "month": m,
                "n_sales": len(prices[(town, m)]),
                "median_price": v,
                "yoy_pct": None if prev is None else 100.0 * (v - prev) / prev,
                "rolling_3m_avg": sum(window) / len(window),
                "price_rank_in_month": 1 + sum(1 for x in by_month[m] if x > v),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _variant(hdb: pl.DataFrame) -> pl.DataFrame:
    gaps = [("BISHAN", "2019-03"), ("BISHAN", "2019-04"), ("ANG MO KIO", "2022-01"),
            ("PUNGGOL", "2016-07"), ("YISHUN", "2023-12")]  # fmt: skip
    keep = pl.lit(True)
    for town, month in gaps:
        keep = keep & ~((pl.col("town") == town) & (pl.col("month") == month))
    v = hdb.filter(keep)
    errors = v.head(6).with_columns(
        pl.Series("resale_price", [5, 1, 12_500_000, 99, 25_000_000, 3]),
        pl.Series("town", ["TAMPINES", "BEDOK", "TAMPINES", "CLEMENTI", "BEDOK", "YISHUN"]),
        pl.Series("month", ["2018-06", "2020-02", "2018-06", "2021-11", "2015-01", "2024-12"]),
    )
    return pl.concat([v, errors]).reverse()


# ── Comparison ───────────────────────────────────────────────────────────
def _usable(out) -> bool:
    return (
        isinstance(out, pl.DataFrame)
        and all(c in out.columns for c in REQUIRED)
        and out.schema["month"] == pl.Date
    )


def _rows_match(out: pl.DataFrame, exp: pl.DataFrame) -> bool:
    keys = out.select("town", "month")
    return keys.height == keys.unique().height and keys.height == exp.height and (
        keys.join(exp.select("town", "month"), on=["town", "month"], how="anti").height == 0
    )


def _col_ok(out: pl.DataFrame, exp: pl.DataFrame, col: str) -> bool:
    try:
        s = out.select("town", "month", pl.col(col).alias("_s"))
        j = exp.select("town", "month", col).join(s, on=["town", "month"], how="left")
        a, b = j[col], j["_s"]
        if (a.is_null() != b.is_null()).any():
            return False
        diff = (a.cast(pl.Float64) - b.cast(pl.Float64)).abs().fill_null(0.0)
        scale = a.cast(pl.Float64).abs().fill_null(1.0).clip(lower_bound=1.0)
        return bool((diff <= 1e-6 * scale).all())
    except Exception:
        return False


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {c: False for c in CHECKS}}
    c = score["checks"]
    try:
        fn = load_student_module(student_path).town_trends
    except Exception as e:
        score["error"] = f"Cannot load town_trends(): {type(e).__name__}: {e}"
        return _finalize(score)

    hdb = MLFPDataLoader().load("mlfp01", "hdb_resale.parquet")
    try:
        out = fn(hdb.clone())
    except Exception as e:
        score["error"] = f"town_trends() raised {type(e).__name__}: {e}"
        return _finalize(score)
    c["required_columns"] = _usable(out)
    if not c["required_columns"]:
        score["error"] = "Output is not a DataFrame with every required column (month as Date)"
        return _finalize(score)

    exp = _expected(hdb)
    c["town_month_rows"] = _rows_match(out, exp)
    for col in VALUE_COLS:
        c[col] = _col_ok(out, exp, col)

    try:
        v = _variant(hdb)
        v_out = fn(v.clone())
        v_exp = _expected(v)
        c["unseen_variant"] = (
            _usable(v_out)
            and _rows_match(v_out, v_exp)
            and all(_col_ok(v_out, v_exp, col) for col in VALUE_COLS)
        )
    except Exception as e:
        score["error"] = f"Unseen variant failed: {type(e).__name__}: {e}"
    score["expected_rows"] = exp.height
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
