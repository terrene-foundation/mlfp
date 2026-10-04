# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 5: Point-in-Time Features with the Kailash FeatureStore

Implement the three functions below; problem.md defines what they must
produce and how they are accepted.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from shared import MLFPDataLoader


def load_hdb() -> pl.DataFrame:
    """Raw HDB resale transactions — for development only."""
    return MLFPDataLoader().load("mlfp01", "hdb_resale.parquet")


def town_month_features(transactions: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError


def publish_features(features: pl.DataFrame, store_url: str, town_codes: dict[str, int]):
    raise NotImplementedError


def features_for_sales(
    sales: pl.DataFrame, store_url: str, schema, town_codes: dict[str, int]
) -> pl.DataFrame:
    raise NotImplementedError


if __name__ == "__main__":
    hdb = load_hdb()
    towns = ["BEDOK", "TAMPINES"]
    dev = hdb.filter(
        pl.col("town").is_in(towns) & pl.col("month").is_between(pl.lit("2019-01"), pl.lit("2020-12"))
    )
    codes = {t: i for i, t in enumerate(sorted(hdb["town"].unique().to_list()))}
    store_url = f"sqlite:///{(Path('outputs') / 'task5_features.db').resolve().as_posix()}"
    Path("outputs").mkdir(exist_ok=True)
    print(dev.height, "development rows;", store_url)
