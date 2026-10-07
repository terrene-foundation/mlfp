# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 2: HDB Feature Table with Town Enrichment

Implement `engineer_features()`. problem.md defines every output column and
the acceptance criteria. Inspecting the three tables to find out what makes
them hard to parse and join is part of the task.

The function is graded on the real tables AND on unseen variants of them, so
derive everything from the frames you are given.

    python starter.py               # run your function on the real tables
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def engineer_features(
    hdb: pl.DataFrame, mrt: pl.DataFrame, schools: pl.DataFrame
) -> pl.DataFrame:
    """Return one feature row per resale record in `hdb`.

    Args:
        hdb: resale records (columns as hdb_resale.parquet).
        mrt: station list (columns as mrt_stations.parquet).
        schools: school list (columns as schools.parquet).

    Returns:
        A DataFrame with the columns listed in problem.md, one row per
        record in `hdb`, traceable through `row_id`.
    """
    raise NotImplementedError("Implement engineer_features() — see problem.md")


if __name__ == "__main__":
    loader = MLFPDataLoader()
    hdb = loader.load("mlfp01", "hdb_resale.parquet")
    mrt = loader.load("mlfp_assessment", "mrt_stations.parquet")
    schools = loader.load("mlfp_assessment", "schools.parquet")
    features = engineer_features(hdb, mrt, schools)
    print(f"Input {hdb.shape} -> features {features.shape}")
    print(features.head())
    print(features.null_count())
