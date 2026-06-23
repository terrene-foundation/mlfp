# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 2: HDB Feature Engineering (Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader

# Ordinal room counts. EXECUTIVE and MULTI-GENERATION sit above 5-room.
_ROOMS = {
    "2 ROOM": 2,
    "3 ROOM": 3,
    "4 ROOM": 4,
    "5 ROOM": 5,
    "EXECUTIVE": 6,
    "MULTI-GENERATION": 7,
}


def solve() -> pl.DataFrame:
    """Engineer model-ready features from the raw HDB resale table.

    Returns a 10-column frame, sorted by ``sale_year`` then ``town``, with the
    messy ``storey_range`` and ``remaining_lease`` strings parsed into clean
    numerics and derived features added.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet")

    # sale_year from the "YYYY-MM" month string.
    df = df.with_columns(
        pl.col("month").str.slice(0, 4).cast(pl.Int64).alias("sale_year")
    )

    # storey_midpoint: split on the " TO " delimiter FIRST (it legitimately
    # contains the letter O), THEN fix OCR "O"->"0" inside the numeric tokens
    # only, and average the bounds.
    low_tok = pl.col("storey_range").str.extract(r"^\s*(\w+)\s+TO\s+", 1)
    high_tok = pl.col("storey_range").str.extract(r"\s+TO\s+(\w+)\s*$", 1)
    low = low_tok.str.replace_all("O", "0").cast(pl.Float64, strict=False)
    high = high_tok.str.replace_all("O", "0").cast(pl.Float64, strict=False)
    df = df.with_columns(((low + high) / 2.0).alias("storey_midpoint"))

    # flat_age and price_per_sqm.
    df = df.with_columns(
        [
            (pl.col("sale_year") - pl.col("lease_commence_date"))
            .cast(pl.Int64)
            .alias("flat_age_years"),
            (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
            pl.col("flat_type")
            .replace_strict(_ROOMS, default=None)
            .alias("flat_type_rooms"),
        ]
    )

    # remaining_lease: dual format ("X years Y months" OR bare "X"); null ->
    # statutory 99-year lease minus current flat age.
    yrs = pl.col("remaining_lease").str.extract(r"(\d+)\s*year", 1).cast(pl.Float64)
    mons = (
        pl.col("remaining_lease")
        .str.extract(r"(\d+)\s*month", 1)
        .cast(pl.Float64)
        .fill_null(0.0)
    )
    plain = pl.col("remaining_lease").str.extract(r"^\s*(\d+)\s*$", 1).cast(pl.Float64)
    parsed_lease = pl.when(yrs.is_not_null()).then(yrs + mons / 12.0).otherwise(plain)
    df = df.with_columns(
        parsed_lease.fill_null(99.0 - pl.col("flat_age_years")).alias(
            "remaining_lease_years"
        )
    )

    cols = [
        "town",
        "flat_type",
        "flat_type_rooms",
        "sale_year",
        "storey_midpoint",
        "floor_area_sqm",
        "flat_age_years",
        "remaining_lease_years",
        "resale_price",
        "price_per_sqm",
    ]
    return df.select(cols).sort(["sale_year", "town"])


if __name__ == "__main__":
    out = solve()
    print(out.head())
    print(f"\nShape: {out.shape}")
    print(f"storey_midpoint nulls: {out['storey_midpoint'].null_count()}")
    print(f"remaining_lease_years nulls: {out['remaining_lease_years'].null_count()}")
    print(f"rooms map: {sorted(out['flat_type_rooms'].unique().to_list())}")
