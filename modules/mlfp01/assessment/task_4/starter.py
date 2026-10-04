# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 4: Profile, Clean and Justify with DataExplorer

Implement `audit_indicators()`. problem.md defines the dict it returns and
the acceptance criteria for the cleaned table. Use DataExplorer to find the
problems; problem.md does not list them.

run_profile / run_compare are synchronous wrappers around DataExplorer that
work in a script, Jupyter and Colab alike.

    python starter.py               # run your function on the real file
"""
from __future__ import annotations

import polars as pl

from kailash_ml import AlertConfig
from shared import MLFPDataLoader, run_compare, run_profile


def audit_indicators(raw: pl.DataFrame) -> dict:
    """Profile, clean and justify the quarterly economic indicators.

    Imputation choice (two or three sentences, read at review):
        ...

    Args:
        raw: the full indicators file (monthly and quarterly rows).

    Returns:
        dict with keys raw_alerts, cleaned, imputation, null_alert_config,
        accepted_alerts and quality_delta (see problem.md).
    """
    raise NotImplementedError("Implement audit_indicators() — see problem.md")


if __name__ == "__main__":
    raw = MLFPDataLoader().load("mlfp01", "economic_indicators.csv")
    result = audit_indicators(raw)
    print("Raw alerts:", result["raw_alerts"])
    print(result["cleaned"].head())
    print("Accepted alerts:", result["accepted_alerts"])
    print("Quality delta:", result["quality_delta"])
    # Optional: before/after comparison while you work
    raw_q = raw.filter(pl.col("period_type") == "quarterly")
    print(run_compare(raw_q, result["cleaned"])["shape_comparison"])
    _ = (AlertConfig, run_profile)  # imported for your use
