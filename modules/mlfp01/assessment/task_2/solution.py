# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 2: HDB Feature Table with Town Enrichment
(Reference Solution)

Withheld from students. Verified to pass grader.py.

What inspecting the tables reveals:
  - storey_range: some digits were keyed as the letter "O" ("O1 TO 03",
    "4O TO 42"). The delimiter " TO " also contains an O, so split first and
    repair the number tokens only.
  - remaining_lease: two formats ("71 years 11 months" and a bare "92") plus
    1,474 nulls.
  - 3,298 records have a sale year before the lease commencement year: a
    contradiction, so the flat's age is unknown (94 of them also have no
    recorded lease, so their remaining lease is unknown too).
  - Town names: HDB uses UPPER CASE, the lookups use Title Case, and HDB's
    "KALLANG/WHAMPOA" is "Kallang" in both lookups.
  - The station table has one row per station per LINE: interchanges appear
    several times, so count distinct station names, not rows.
  - Five HDB towns have no station in the table and two have no school: a
    left join keeps their records and the count is 0, not null.
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader

ROOMS = {
    "2 ROOM": 2,
    "3 ROOM": 3,
    "4 ROOM": 4,
    "5 ROOM": 5,
    "EXECUTIVE": 6,
    "MULTI-GENERATION": 7,
}
STATUTORY_LEASE_YEARS = 99


def _town_key(col: str) -> pl.Expr:
    """Join key that survives case and the 'KALLANG/WHAMPOA' style of name."""
    return (
        pl.col(col)
        .str.to_uppercase()
        .str.split("/")
        .list.first()
        .str.strip_chars()
        .alias("town_key")
    )


def _storey_number(token: pl.Expr) -> pl.Expr:
    return token.str.replace_all("O", "0").cast(pl.Float64)


def engineer_features(
    hdb: pl.DataFrame, mrt: pl.DataFrame, schools: pl.DataFrame
) -> pl.DataFrame:
    """Return one feature row per resale record in `hdb`."""
    df = hdb.with_row_index("row_id").with_columns(
        pl.col("row_id").cast(pl.Int64),
        pl.col("month").str.slice(0, 4).cast(pl.Int64).alias("sale_year"),
    )

    bounds = pl.col("storey_range").str.split(" TO ")
    df = df.with_columns(
        (
            (
                _storey_number(bounds.list.get(0))
                + _storey_number(bounds.list.get(1))
            )
            / 2.0
        ).alias("storey_midpoint"),
        pl.col("flat_type").replace_strict(ROOMS, return_dtype=pl.Int64).alias(
            "flat_type_rooms"
        ),
        (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
        pl.when(pl.col("sale_year") >= pl.col("lease_commence_date"))
        .then(pl.col("sale_year") - pl.col("lease_commence_date"))
        .otherwise(None)
        .alias("flat_age_years"),
    )

    years = pl.col("remaining_lease").str.extract(r"(\d+)\s*year", 1).cast(pl.Float64)
    months = (
        pl.col("remaining_lease")
        .str.extract(r"(\d+)\s*month", 1)
        .cast(pl.Float64)
        .fill_null(0.0)
    )
    bare = pl.col("remaining_lease").str.extract(r"^\s*(\d+)\s*$", 1).cast(pl.Float64)
    recorded = pl.when(years.is_not_null()).then(years + months / 12.0).otherwise(bare)
    df = df.with_columns(
        recorded.fill_null(STATUTORY_LEASE_YEARS - pl.col("flat_age_years")).alias(
            "remaining_lease_years"
        )
    )

    stations = (
        mrt.with_columns(_town_key("town"))
        .group_by("town_key")
        .agg(pl.col("station_name").n_unique().cast(pl.Int64).alias("mrt_station_count"))
    )
    school_counts = (
        schools.with_columns(_town_key("town"))
        .group_by("town_key")
        .agg(pl.len().cast(pl.Int64).alias("school_count"))
    )
    df = (
        df.with_columns(_town_key("town"))
        .join(stations, on="town_key", how="left")
        .join(school_counts, on="town_key", how="left")
        .with_columns(
            pl.col("mrt_station_count").fill_null(0),
            pl.col("school_count").fill_null(0),
        )
    )

    return df.select(
        "row_id",
        "town",
        "flat_type",
        "floor_area_sqm",
        "resale_price",
        "sale_year",
        "storey_midpoint",
        "flat_type_rooms",
        "flat_age_years",
        "remaining_lease_years",
        "price_per_sqm",
        "mrt_station_count",
        "school_count",
    ).sort("row_id")


if __name__ == "__main__":
    loader = MLFPDataLoader()
    out = engineer_features(
        loader.load("mlfp01", "hdb_resale.parquet"),
        loader.load("mlfp_assessment", "mrt_stations.parquet"),
        loader.load("mlfp_assessment", "schools.parquet"),
    )
    print(out.head())
    print(f"Shape: {out.shape}")
    print(out.null_count())
