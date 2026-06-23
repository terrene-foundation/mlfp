# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 4: Profile, Clean & Integrate with DataExplorer
(Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.
"""
from __future__ import annotations

import asyncio

import polars as pl

from kailash_ml import DataExplorer
from shared import MLFPDataLoader


async def _alert_count(df: pl.DataFrame) -> int:
    """Number of data-quality alerts DataExplorer raises for a frame."""
    explorer = DataExplorer()
    profile = await explorer.profile(df)
    return len(profile.alerts)


def _clean(raw: pl.DataFrame) -> pl.DataFrame:
    """Deterministic cleaning of the quarterly economic indicators."""
    q = raw.filter(pl.col("period_type") == "quarterly")

    # period appears in THREE formats: "Q1 2000", "2001-Q1", "2001-2".
    # Year is the 4-digit run; quarter is the digit after "Q" OR the trailing
    # single digit after "-". Coalesce both rules. Drop unparseable.
    q = q.with_columns(
        [
            pl.coalesce(
                [
                    pl.col("period").str.extract(r"Q(\d)", 1),
                    pl.col("period").str.extract(r"-(\d)\s*$", 1),
                ]
            )
            .cast(pl.Int64)
            .alias("period_quarter"),
            pl.col("period")
            .str.extract(r"(\d{4})", 1)
            .cast(pl.Int64)
            .alias("period_year"),
        ]
    ).filter(
        pl.col("period_quarter").is_not_null() & pl.col("period_year").is_not_null()
    )

    # tourist_arrivals: strip thousands separators -> Int64.
    q = q.with_columns(
        pl.col("tourist_arrivals")
        .str.replace_all(",", "")
        .str.strip_chars()
        .cast(pl.Int64)
        .alias("tourist_arrivals")
    )

    # Impute the two sparse numerics with the quarterly median (deterministic).
    q = q.with_columns(
        [
            pl.col("inflation_rate").fill_null(pl.col("inflation_rate").median()),
            pl.col("trade_balance_sgd_bn").fill_null(
                pl.col("trade_balance_sgd_bn").median()
            ),
        ]
    )

    cols = [
        "period_year",
        "period_quarter",
        "gdp_growth_pct",
        "unemployment_rate",
        "inflation_rate",
        "trade_balance_sgd_bn",
        "property_price_index",
        "tourist_arrivals",
    ]
    return q.select(cols).sort(["period_year", "period_quarter"])


def solve() -> dict:
    """Profile the raw indicators, clean them, and confirm quality improved.

    Returns a dict with:
      - ``cleaned``: the cleaned quarterly DataFrame (8 columns)
      - ``raw_alert_count``: DataExplorer alerts on the raw quarterly slice
      - ``clean_alert_count``: DataExplorer alerts on the cleaned frame
    """
    raw = MLFPDataLoader().load("mlfp01", "economic_indicators.csv")
    raw_q = raw.filter(pl.col("period_type") == "quarterly")
    cleaned = _clean(raw)

    raw_alerts, clean_alerts = asyncio.run(_profile_both(raw_q, cleaned))
    return {
        "cleaned": cleaned,
        "raw_alert_count": raw_alerts,
        "clean_alert_count": clean_alerts,
    }


async def _profile_both(raw_q: pl.DataFrame, cleaned: pl.DataFrame):
    return await _alert_count(raw_q), await _alert_count(cleaned)


if __name__ == "__main__":
    out = solve()
    print(out["cleaned"].head())
    print(f"\nCleaned shape: {out['cleaned'].shape}")
    print(f"tourist_arrivals dtype: {out['cleaned']['tourist_arrivals'].dtype}")
    print(
        f"Raw alerts: {out['raw_alert_count']} -> Clean alerts: {out['clean_alert_count']}"
    )
