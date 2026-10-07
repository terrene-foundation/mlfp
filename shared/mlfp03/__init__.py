# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP Module 3 — Supervised ML helpers.

Exercise-specific infrastructure (ICU data loading, feature utilities,
experiment tracker setup) that technique files import. Each exercise
gets its own submodule:

    from shared.mlfp03.ex_1 import load_icu_tables, build_feature_matrix
    ...

Available after `uv sync` from any directory.
"""

# ── LightGBM × sklearn 1.9 spurious-warning filter ───────────────────────
# lightgbm 4.6's sklearn wrapper records auto-generated feature names
# ("Column_0", …) even when fit on a plain numpy ndarray; sklearn 1.9's
# validate_data then warns at predict time because the ndarray "does not
# have valid feature names". Both sides are nameless — the warning is
# spurious for every course call site (all fit AND predict on numpy).
# Suppressing just that message, not the category: genuine name-mismatch
# warnings (fit on a named frame, predict on numpy) still surface.
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBM\w+ was fitted with feature names",
    category=UserWarning,
)

# Re-export the canonical factory (shared/kailash_helpers.py).
from shared.kailash_helpers import create_visualizer
