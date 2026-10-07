# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP Module 4 — Unsupervised Machine Learning helpers.

Exercise-specific infrastructure (data loading, preprocessing, metrics,
visualisation helpers) that technique files import. Each exercise gets
its own submodule:

    from shared.mlfp04.ex_1 import load_customers, score_partition

Available after `uv sync` from any directory.
"""

# Re-export the canonical factory (shared/kailash_helpers.py).
from shared.kailash_helpers import create_visualizer

# umap-learn nags "n_jobs value 1 overridden to 1 by setting random_state"
# whenever a seed is set — seeding is the DELIBERATE deterministic-teaching
# choice in ex_3, not an oversight. Message-matched acknowledgement so the
# strict gate keeps everything else fatal.
import warnings as _warnings

_warnings.filterwarnings(
    "ignore", message=r".*n_jobs value 1 overridden.*", category=UserWarning
)

