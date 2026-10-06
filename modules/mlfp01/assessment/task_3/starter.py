# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 3: Town Price Trends with Correct Time Alignment

Implement `town_trends()`. problem.md defines the output table and what
"correctly aligned" means. Checking whether every town has a sale in every
month, and finding the recording errors in the price column, is part of the
task.

The function is graded on the real data AND on an unseen variant with other
gaps and new recording errors.

    python starter.py               # run your function on the real data
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def town_trends(hdb: pl.DataFrame) -> pl.DataFrame:
    """Return the monthly price-trend table for every town.

    Args:
        hdb: resale records (columns as hdb_resale.parquet).

    Returns:
        One row per (town, month) with at least one genuine sale, with the
        columns listed in problem.md.
    """
    raise NotImplementedError("Implement town_trends() — see problem.md")


if __name__ == "__main__":
    hdb = MLFPDataLoader().load("mlfp01", "hdb_resale.parquet")
    trends = town_trends(hdb)
    print(trends.shape)
    print(trends.sort("town", "month").head(15))
