#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP02 Assessment Task 5 — Point-in-Time Features with the
Kailash FeatureStore (instructor-side; not distributed to students).

    python grader.py submission.py [--seed N]

Per run the grader picks secret towns and a secret window of months, then:

1. compares the submission's town-month features with an independent
   loop-based reference on that slice;
2. re-runs the feature build on a TRUNCATED history and requires every
   feature row that both runs produce to be identical (a feature that peeks at
   its own or later months changes when the future is removed);
3. opens the FeatureStore the submission published to, with the grader's own
   FeatureStore instance, and checks as-of snapshots at secret timestamps;
4. publishes its OWN secretly perturbed feature values into a fresh store with
   the submission's schema and asks the submission to serve features for
   held-out sales from that store — only code that really reads the store
   returns the perturbed values.
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, close, finalize, load_student_module, main  # noqa: E402

from shared import MLFPDataLoader  # noqa: E402

WEIGHT = 15
TENANT = "_single"


def ref_features(raw: pl.DataFrame) -> dict[tuple[str, int], tuple[float, int]]:
    """{(town, month_index): (median S$/sqm, count)} over sale months M-6..M-1."""
    by_town: dict[str, list[tuple[int, float]]] = {}
    months = []
    for m, town, price, area, lease in raw.select(
        "month", "town", "resale_price", "floor_area_sqm", "lease_commence_date"
    ).iter_rows():
        mi = int(m[:4]) * 12 + int(m[5:7]) - 1
        months.append(mi)
        by_town.setdefault(town, [])
        if 100_000 <= price <= 2_000_000 and lease <= int(m[:4]):
            by_town[town].append((mi, price / area))
    out = {}
    for town, sales in by_town.items():
        for M in range(min(months), max(months) + 1):
            w = [v for mi, v in sales if M - 6 <= mi < M]
            if w:
                out[(town, M)] = (statistics.median(w), len(w))
    return out


def mi_of(ts) -> int:
    return ts.year * 12 + ts.month - 1


def as_of(ref: dict, town: str, t: datetime):
    """Latest feature row for ``town`` with as_of <= t."""
    cands = [M for (tw, M) in ref if tw == town and datetime(M // 12, M % 12 + 1, 1) <= t]
    return ref[(town, max(cands))] if cands else None


def student_dict(df: pl.DataFrame) -> dict:
    return {(t, mi_of(a)): (float(m), int(v)) for t, a, m, v in
            df.select("town", "as_of", "median_psm_6m", "volume_6m").iter_rows()}


def grade(path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(path, "student_task5")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}")
    fns = ("town_month_features", "publish_features", "features_for_sales")
    missing = [f for f in fns if not callable(getattr(st, f, None))]
    if missing:
        return finalize(checks, WEIGHT, seed, f"Missing functions: {missing}")

    from dataflow import DataFlow
    from kailash_ml.features import FeatureGroup, FeatureSchema, FeatureStore

    rng = np.random.default_rng(seed)
    hdb = MLFPDataLoader().load("mlfp01", "hdb_resale.parquet")
    all_towns = sorted(hdb["town"].unique().to_list())
    towns = sorted(rng.choice(all_towns, 3, replace=False).tolist())
    start = int(rng.integers(2015 * 12, 2021 * 12))
    span = int(rng.integers(24, 37))
    lo = f"{start // 12}-{start % 12 + 1:02d}"
    hi = f"{(start + span - 1) // 12}-{(start + span - 1) % 12 + 1:02d}"
    sub = hdb.filter(pl.col("town").is_in(towns) & (pl.col("month") >= lo) & (pl.col("month") <= hi))
    sub = sub.sample(fraction=1.0, shuffle=True, seed=int(rng.integers(1 << 31)))
    codes = {t: int(c) for t, c in zip(all_towns, rng.permutation(len(all_towns)) + 100)}
    ref = ref_features(sub)

    state: dict = {}

    def build():
        f = st.town_month_features(sub.clone())
        state["features"] = f
        got = student_dict(f)
        missing_rows = sorted(set(ref) - set(got))[:3]
        extra_rows = sorted(set(got) - set(ref))[:3]
        vals_ok = not missing_rows and not extra_rows and all(
            close(got[k][0], ref[k][0], rtol=1e-9) and got[k][1] == ref[k][1] for k in ref
        )
        types_ok = (
            f.height == len(ref) > 0
            and isinstance(f.schema["as_of"], pl.Datetime)
            and f.schema["volume_6m"] == pl.Int64
            and f.select("median_psm_6m", "volume_6m").null_count().sum_horizontal().item() == 0
        )
        cut_mi = start + int(rng.integers(8, span - 4))
        cut = f"{cut_mi // 12}-{cut_mi % 12 + 1:02d}"
        trunc = student_dict(st.town_month_features(sub.filter(pl.col("month") < cut).clone()))
        shared_keys = [k for k in trunc if k in got]
        leak_ok = len(shared_keys) > 0 and all(
            close(trunc[k][0], got[k][0], rtol=1e-12) and trunc[k][1] == got[k][1] for k in shared_keys
        ) and set(k for k in ref if k[1] < cut_mi) <= set(trunc)
        return {
            "feature_values": (vals_ok, f"missing {missing_rows}, unexpected {extra_rows}"),
            "feature_types": (types_ok, f"schema {f.schema}"),
            "no_lookahead": (leak_ok, f"features changed when history after {cut} was removed"),
        }

    checks.guarded(["feature_values", "feature_types", "no_lookahead"], build)

    with tempfile.TemporaryDirectory() as tmp:
        url_student = f"sqlite:///{Path(tmp, 'student_store.db').as_posix()}"
        url_grader = f"sqlite:///{Path(tmp, 'grader_store.db').as_posix()}"

        def store():
            if "features" not in state:
                return {"store_contents": (False, "feature build failed")}
            schema = st.publish_features(state["features"].clone(), url_student, dict(codes))
            state["schema"] = schema
            ok_type = isinstance(schema, FeatureSchema) and schema.entity_id_column == "town_id" \
                and schema.timestamp_column == "as_of" \
                and {f.name for f in schema.fields} >= {"median_psm_6m", "volume_6m"}
            if not ok_type:
                return {"store_contents": (False, f"returned schema {schema!r} does not match the contract")}
            inv = {v: k for k, v in codes.items()}
            probes = [datetime((start + 12) // 12, (start + 12) % 12 + 1, 1),
                      datetime((start + span - 3) // 12, (start + span - 3) % 12 + 1, int(rng.integers(2, 28)))]

            async def read():
                fs = FeatureStore(DataFlow(url_student), default_tenant_id=TENANT)
                return [await fs.get_features(schema, timestamp=t) for t in probes]

            snaps = asyncio.run(read())
            notes, ok = [], True
            for t, snap in zip(probes, snaps):
                got = {inv[int(e)]: (float(m), int(v)) for e, m, v in snap.select("town_id", "median_psm_6m", "volume_6m").iter_rows()}
                want = {tw: as_of(ref, tw, t) for tw in towns if as_of(ref, tw, t) is not None}
                if set(got) != set(want) or not all(close(got[k][0], want[k][0], rtol=1e-9) and got[k][1] == want[k][1] for k in want):
                    ok = False
                    notes.append(f"as of {t:%Y-%m-%d}: store has {got}, expected {want}")
            return {"store_contents": (ok, "; ".join(notes))}

        checks.guarded(["store_contents"], store)

        def serve():
            schema = state.get("schema")
            if schema is None:
                return {"serves_from_store": (False, "no schema published")}
            factor = float(rng.uniform(1.3, 1.7))
            bump = int(rng.integers(500, 900))
            rows = [(codes[t], datetime(M // 12, M % 12 + 1, 1), m * factor, v + bump) for (t, M), (m, v) in ref.items()]
            frame = pl.DataFrame(rows, schema={"town_id": pl.Int64, "as_of": pl.Datetime("us"),
                                               "median_psm_6m": pl.Float64, "volume_6m": pl.Int64}, orient="row")

            async def write():
                fs = FeatureStore(DataFlow(url_grader), default_tenant_id=TENANT)
                await fs.materialize(FeatureGroup(schema, dataflow=fs.dataflow), frame)

            asyncio.run(write())
            months = sorted(set(sub["month"].to_list()))
            pick = [months[int(i)] for i in rng.choice(np.arange(1, len(months)), 4, replace=False)]
            sales = sub.filter(pl.col("month").is_in(pick)).head(30).with_row_index("txn_id", offset=int(rng.integers(1000, 9000)))
            sales = sales.with_columns(pl.col("txn_id").cast(pl.Int64))
            got = st.features_for_sales(sales.clone(), url_grader, schema, dict(codes))
            g = {int(i): (m, v) for i, m, v in got.select("txn_id", "median_psm_6m", "volume_6m").iter_rows()}
            bad = []
            for i, town, m in sales.select("txn_id", "town", "month").iter_rows():
                w = as_of(ref, town, datetime(int(m[:4]), int(m[5:7]), 1))
                want = None if w is None else (w[0] * factor, w[1] + bump)
                have = g.get(int(i), "missing")
                if want is None:
                    if have == "missing" or have[0] is not None:
                        bad.append((i, have, want))
                elif have == "missing" or have[0] is None or not (close(have[0], want[0], rtol=1e-9) and int(have[1]) == want[1]):
                    bad.append((i, have, want))
            return {"serves_from_store": (not bad and len(g) == sales.height, f"{len(bad)} wrong rows, e.g. {bad[:2]}")}

        checks.guarded(["serves_from_store"], serve)
    return finalize(checks, WEIGHT, seed)


if __name__ == "__main__":
    main(grade)
