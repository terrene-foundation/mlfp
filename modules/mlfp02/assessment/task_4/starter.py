# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 4: Difference-in-Differences Policy Evaluation

Implement did_analysis(); problem.md defines its output and how it is
accepted. The grader runs it on panels it simulates itself.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def make_dev_panel(seed: int = 0) -> pl.DataFrame:
    """Illustrative development panel: 6 pre and 6 post periods, a measure that
    applies to the treated region only. The grader's panels are different."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(12):
        for g in (0, 1):
            mean = 450_000 + 100_000 * g + 2_000 * t + (-20_000 if (g and t >= 6) else 0)
            for v in rng.normal(mean, 75_000, size=150):
                rows.append((t, g, int(t >= 6), float(v)))
    return pl.DataFrame(rows, schema=["period", "treated", "post", "y"], orient="row")


def did_analysis(panel: pl.DataFrame) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    panel = make_dev_panel()
    print(panel.group_by("treated", "post").agg(pl.len(), pl.col("y").mean()).sort("treated", "post"))
