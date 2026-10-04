# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 5: Point-in-Time Features with the Kailash FeatureStore
(Reference Solution — withheld from students; verified to pass grader.py)
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import polars as pl

from shared import MLFPDataLoader

PRICE_BOUNDS = (100_000, 2_000_000)
WINDOW_MONTHS = 6
TENANT = "_single"  # the FeatureStore's single-tenant scope


def load_hdb() -> pl.DataFrame:
    return MLFPDataLoader().load("mlfp01", "hdb_resale.parquet")


def _month_index(col: str = "month") -> pl.Expr:
    return pl.col(col).str.slice(0, 4).cast(pl.Int64) * 12 + pl.col(col).str.slice(5, 2).cast(pl.Int64) - 1


def _index_to_datetime(idx: pl.Expr) -> pl.Expr:
    return pl.datetime(idx // 12, idx % 12 + 1, 1, time_unit="us")


def town_month_features(transactions: pl.DataFrame) -> pl.DataFrame:
    df = transactions.with_columns(_month_index().alias("mi"))
    valid = df.filter(
        pl.col("resale_price").is_between(*PRICE_BOUNDS)
        & (pl.col("lease_commence_date") <= pl.col("month").str.slice(0, 4).cast(pl.Int64))
    ).with_columns((pl.col("resale_price") / pl.col("floor_area_sqm")).alias("psm"))
    first, last = int(df["mi"].min()), int(df["mi"].max())
    grid = pl.DataFrame({"town": sorted(df["town"].unique().to_list())}).join(
        pl.DataFrame({"as_of_mi": list(range(first, last + 1))}), how="cross"
    )
    # Window for as_of month M: sale months M-6 .. M-1 (never month M itself).
    pairs = grid.join(valid.select("town", "mi", "psm"), on="town", how="inner").filter(
        (pl.col("mi") >= pl.col("as_of_mi") - WINDOW_MONTHS) & (pl.col("mi") < pl.col("as_of_mi"))
    )
    out = (
        pairs.group_by("town", "as_of_mi")
        .agg(pl.col("psm").median().alias("median_psm_6m"), pl.len().cast(pl.Int64).alias("volume_6m"))
        .with_columns(_index_to_datetime(pl.col("as_of_mi")).alias("as_of"))
        .select("town", "as_of", "median_psm_6m", "volume_6m")
        .sort("town", "as_of")
    )
    return out


def _schema():
    from kailash_ml.features import FeatureField, FeatureSchema

    return FeatureSchema(
        name="hdb_town_market_6m",
        version=1,
        fields=(
            FeatureField("median_psm_6m", "float64", False, "Median S$/sqm of the town's sales in the previous 6 months"),
            FeatureField("volume_6m", "int64", False, "Number of the town's sales in the previous 6 months"),
        ),
        entity_id_column="town_id",
        timestamp_column="as_of",
    )


def _store(store_url: str):
    from dataflow import DataFlow
    from kailash_ml.features import FeatureStore

    return FeatureStore(DataFlow(store_url), default_tenant_id=TENANT)


def publish_features(features: pl.DataFrame, store_url: str, town_codes: dict[str, int]):
    from kailash_ml.features import FeatureGroup

    schema = _schema()
    frame = features.with_columns(
        pl.col("town").replace_strict(town_codes, return_dtype=pl.Int64).alias("town_id"),
        pl.col("as_of").cast(pl.Datetime("us")),
    ).select("town_id", "as_of", "median_psm_6m", "volume_6m")

    async def _go():
        fs = _store(store_url)
        await fs.materialize(FeatureGroup(schema, dataflow=fs.dataflow), frame)

    asyncio.run(_go())
    return schema


def features_for_sales(
    sales: pl.DataFrame, store_url: str, schema, town_codes: dict[str, int]
) -> pl.DataFrame:
    months = sorted(sales["month"].unique().to_list())

    async def _go() -> pl.DataFrame:
        fs = _store(store_url)
        parts = []
        for m in months:
            t = datetime(int(m[:4]), int(m[5:7]), 1)
            snap = await fs.get_features(schema, timestamp=t)
            parts.append(
                snap.with_columns(pl.col("town_id").cast(pl.Int64), pl.lit(m).alias("month"))
            )
        return pl.concat(parts) if parts else pl.DataFrame()

    snaps = asyncio.run(_go())
    keyed = sales.select("txn_id", "town", "month").with_columns(
        pl.col("town").replace_strict(town_codes, return_dtype=pl.Int64).alias("town_id")
    )
    return (
        keyed.join(snaps.select("town_id", "month", "median_psm_6m", "volume_6m"), on=["town_id", "month"], how="left")
        .select("txn_id", "median_psm_6m", "volume_6m")
        .sort("txn_id")
    )


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    hdb = load_hdb()
    towns = ["BEDOK", "TAMPINES"]
    sub = hdb.filter(pl.col("town").is_in(towns) & pl.col("month").is_between(pl.lit("2019-01"), pl.lit("2020-12")))
    feats = town_month_features(sub)
    print(feats.head(), feats.height)
    codes = {t: i for i, t in enumerate(sorted(hdb["town"].unique().to_list()))}
    with tempfile.TemporaryDirectory() as tmp:
        url = f"sqlite:///{Path(tmp, 'features.db').as_posix()}"
        schema = publish_features(feats, url, codes)
        sales = sub.filter(pl.col("month") >= "2020-10").head(10).with_row_index("txn_id")
        print(features_for_sales(sales, url, schema, codes))
