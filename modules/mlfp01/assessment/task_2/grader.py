#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP01 Assessment Task 2 — HDB Feature Table.

Usage:
    python grader.py starter.py
    python grader.py solution.py

The grader computes every expected feature itself, matches the submission's
rows to source records by `row_id` (row order is irrelevant), and also runs
the submission on an unseen variant of the three tables: shuffled records,
planted parsing cases and altered station/school lists. Per-town numbers
remembered from the real tables therefore fail.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import polars as pl

from shared import MLFPDataLoader

WEIGHT = 20
GATES = ("required_columns", "one_row_per_record", "pass_through_unchanged")
CHECKS = [
    *GATES,
    "sale_year_and_price_per_sqm",
    "storey_midpoint",
    "flat_type_rooms",
    "flat_age_years",
    "remaining_lease_years",
    "mrt_station_count",
    "school_count",
    "unseen_variant",
]
FEATURES = [
    "sale_year",
    "storey_midpoint",
    "flat_type_rooms",
    "flat_age_years",
    "remaining_lease_years",
    "price_per_sqm",
    "mrt_station_count",
    "school_count",
]
PASS_THROUGH = ["town", "flat_type", "floor_area_sqm", "resale_price"]
REQUIRED = ["row_id", *PASS_THROUGH, *FEATURES]
ROOMS = {"2 ROOM": 2, "3 ROOM": 3, "4 ROOM": 4, "5 ROOM": 5,
         "EXECUTIVE": 6, "MULTI-GENERATION": 7}  # fmt: skip


# ── Ground truth (row-wise Python, deliberately unlike a Polars pipeline) ──
def _key(town: str) -> str:
    return town.upper().split("/")[0].strip()


def _storey(band: str) -> float:
    lo, hi = band.split(" TO ")
    return (float(lo.replace("O", "0")) + float(hi.replace("O", "0"))) / 2.0


def _lease(text: str | None) -> float | None:
    if text is None:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*years?(?:\s+(\d+)\s*months?)?\s*", text)
    if m:
        return int(m.group(1)) + (int(m.group(2) or 0)) / 12.0
    m = re.fullmatch(r"\s*(\d+)\s*", text)
    return float(m.group(1)) if m else None


def _expected(hdb: pl.DataFrame, mrt: pl.DataFrame, schools: pl.DataFrame):
    stations: dict[str, set] = {}
    for town, name in mrt.select("town", "station_name").iter_rows():
        stations.setdefault(_key(town), set()).add(name)
    n_schools: dict[str, int] = {}
    for (town,) in schools.select("town").iter_rows():
        n_schools[_key(town)] = n_schools.get(_key(town), 0) + 1

    rows = []
    for i, r in enumerate(hdb.iter_rows(named=True)):
        year = int(r["month"][:4])
        age = year - r["lease_commence_date"]
        age = age if age >= 0 else None
        lease = _lease(r["remaining_lease"])
        if lease is None and age is not None:
            lease = 99.0 - age
        rows.append(
            {
                "row_id": i,
                "town": r["town"],
                "flat_type": r["flat_type"],
                "floor_area_sqm": r["floor_area_sqm"],
                "resale_price": r["resale_price"],
                "sale_year": year,
                "storey_midpoint": _storey(r["storey_range"]),
                "flat_type_rooms": ROOMS[r["flat_type"]],
                "flat_age_years": age,
                "remaining_lease_years": lease,
                "price_per_sqm": r["resale_price"] / r["floor_area_sqm"],
                "mrt_station_count": len(stations.get(_key(r["town"]), ())),
                "school_count": n_schools.get(_key(r["town"]), 0),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _variant(hdb, mrt, schools):
    """Unseen variant: 3,000 shuffled records, planted cases, altered lookups."""
    sample = (
        hdb.with_columns(pl.int_range(pl.len()).hash(11).alias("_h"))
        .sort("_h")
        .head(3000)
        .drop("_h")
    )
    planted = sample.head(12).with_columns(
        pl.Series(
            "storey_range",
            ["O4 TO O6", "1O TO 12", "2O TO 22", "O1 TO 03", "37 TO 39", "4O TO 42"]
            * 2,
        ),
        pl.Series(
            "remaining_lease",
            ["65 years 00 months", "70 years 06 months", "88", None, "59 years 01 month",
             None, "47", "92 years 11 months", None, "61", "75 years 03 months", None],
        ),  # fmt: skip
        pl.Series(
            "lease_commence_date",
            [1980, 1990, 2001, 2030, 1999, 2030, 1985, 2010, 1977, 1988, 2002, 1995],
        ),
        pl.Series("month", ["2020-05"] * 12),
    )
    new_hdb = pl.concat([sample.slice(12), planted]).reverse()
    new_mrt = pl.concat(
        [
            mrt.filter(pl.col("town") != "Bishan"),
            mrt.filter(pl.col("town") == "Tampines").with_columns(pl.lit("XYZ").alias("line")),
            pl.DataFrame(
                {"station_name": ["Hougang Central"], "town": ["Hougang"], "line": ["NEL"],
                 "latitude": [1.37], "longitude": [103.89], "nearest_mrt": ["Kovan"],
                 "distance_to_mrt_km": [1.1]}  # fmt: skip
            ).cast(mrt.schema),
        ]
    )
    new_schools = schools.filter(pl.col("town") != "Woodlands").head(200)
    return new_hdb, new_mrt, new_schools


# ── Comparison ───────────────────────────────────────────────────────────
def _usable(out) -> bool:
    return isinstance(out, pl.DataFrame) and all(c in out.columns for c in REQUIRED)


def _one_per_record(out: pl.DataFrame, n: int) -> bool:
    try:
        ids = out["row_id"].cast(pl.Int64)
        return out.height == n and ids.n_unique() == n and ids.min() == 0 and ids.max() == n - 1
    except Exception:
        return False


def _match(out: pl.DataFrame, exp: pl.DataFrame, col: str) -> bool:
    try:
        s = out.select(pl.col("row_id").cast(pl.Int64), pl.col(col).alias("_s"))
        j = exp.select("row_id", col).join(s, on="row_id", how="left")
        a, b = j[col], j["_s"]
        if (a.is_null() != b.is_null()).any():
            return False
        if a.dtype == pl.String:
            return bool((a == b).fill_null(True).all())
        diff = (a.cast(pl.Float64) - b.cast(pl.Float64)).abs().fill_null(0.0)
        return bool((diff <= 1e-6).all())
    except Exception:
        return False


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {c: False for c in CHECKS}}
    c = score["checks"]
    try:
        fn = load_student_module(student_path).engineer_features
    except Exception as e:
        score["error"] = f"Cannot load engineer_features(): {type(e).__name__}: {e}"
        return _finalize(score)

    loader = MLFPDataLoader()
    hdb = loader.load("mlfp01", "hdb_resale.parquet")
    mrt = loader.load("mlfp_assessment", "mrt_stations.parquet")
    schools = loader.load("mlfp_assessment", "schools.parquet")
    try:
        out = fn(hdb.clone(), mrt.clone(), schools.clone())
    except Exception as e:
        score["error"] = f"engineer_features() raised {type(e).__name__}: {e}"
        return _finalize(score)

    c["required_columns"] = _usable(out)
    if not c["required_columns"]:
        score["error"] = "Output is not a DataFrame with every required column"
        return _finalize(score)
    c["one_row_per_record"] = _one_per_record(out, hdb.height)
    if not c["one_row_per_record"]:
        score["error"] = f"Expected {hdb.height} rows, one per record; got {out.height}"
        return _finalize(score)

    exp = _expected(hdb, mrt, schools)
    c["sale_year_and_price_per_sqm"] = _match(out, exp, "sale_year") and _match(
        out, exp, "price_per_sqm"
    )
    for col in FEATURES[1:5] + FEATURES[6:]:
        c[col] = _match(out, exp, col)
    c["pass_through_unchanged"] = all(_match(out, exp, col) for col in PASS_THROUGH)

    try:
        v_hdb, v_mrt, v_schools = _variant(hdb, mrt, schools)
        v_out = fn(v_hdb.clone(), v_mrt.clone(), v_schools.clone())
        v_exp = _expected(v_hdb, v_mrt, v_schools)
        c["unseen_variant"] = (
            _usable(v_out)
            and _one_per_record(v_out, v_hdb.height)
            and all(_match(v_out, v_exp, col) for col in FEATURES + PASS_THROUGH)
        )
    except Exception as e:
        score["error"] = f"Unseen variant failed: {type(e).__name__}: {e}"
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
