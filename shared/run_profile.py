# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Sync wrappers for the async DataExplorer engine — used in M1 before async is taught.

``DataExplorer.profile()``, ``.compare()`` and ``.to_html()`` are coroutines.
These helpers run them to completion so a Module 1 student can write::

    from shared import run_profile, run_compare, run_report
    profile = run_profile(df)                     # DataProfile
    profile = run_profile(df, alert_config=cfg)   # custom alert thresholds
    diff = run_compare(df_raw, df_clean)          # dict (see run_compare)
    html = run_report(df, title="My data")        # HTML profile report

They work both in a plain script (no event loop running) and inside
Jupyter / Colab (an event loop is already running there, where a bare
``asyncio.run()`` raises ``RuntimeError``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any, Coroutine, TypeVar

import polars as pl

from kailash_ml import AlertConfig, DataExplorer

if TYPE_CHECKING:
    from kailash_ml.engines.data_explorer import DataProfile

T = TypeVar("T")


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code.

    In a script there is no running loop, so ``asyncio.run`` is used
    directly. In Jupyter/Colab a loop is already running in this thread,
    so the coroutine is run on a fresh loop in a worker thread instead.
    Exceptions raised by the coroutine propagate unchanged in both cases.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def run_profile(
    df: pl.DataFrame,
    alert_config: AlertConfig | None = None,
    *,
    columns: list[str] | None = None,
) -> DataProfile:
    """Profile a DataFrame synchronously with ``DataExplorer.profile()``.

    Args:
        df: Polars DataFrame to profile.
        alert_config: Optional ``AlertConfig`` with custom alert thresholds
            (e.g. ``AlertConfig(high_null_pct_threshold=0.10)``). ``None``
            uses the engine defaults.
        columns: Optional subset of columns to profile.

    Returns:
        ``DataProfile`` — fields include ``n_rows``, ``n_columns``,
        ``columns`` (per-column ``ColumnProfile``), ``correlation_matrix``,
        ``duplicate_count`` and ``alerts`` (a list of dicts with keys
        ``type``, ``column`` or ``columns``, ``value``, ``severity``).
    """
    explorer = DataExplorer(alert_config=alert_config)
    return _run_sync(explorer.profile(df, columns=columns))


def run_compare(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    *,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Compare two DataFrames synchronously with ``DataExplorer.compare()``.

    Typical use is original-vs-cleaned: ``run_compare(df_raw, df_clean)``.

    Args:
        df_a: First DataFrame (e.g. the raw data).
        df_b: Second DataFrame (e.g. the cleaned data).
        columns: Optional subset of columns to compare.

    Returns:
        dict with keys ``profile_a``, ``profile_b``, ``column_deltas``,
        ``shape_comparison`` (``rows_a``, ``rows_b``, ``cols_a``, ``cols_b``),
        ``shared_columns``, ``missing_in_a`` and ``missing_in_b``.
    """
    explorer = DataExplorer()
    return _run_sync(explorer.compare(df_a, df_b, columns=columns))


def run_report(
    df: pl.DataFrame,
    title: str = "Data Profile Report",
    alert_config: AlertConfig | None = None,
) -> str:
    """Build a standalone HTML profile report with ``DataExplorer.to_html()``.

    Args:
        df: Polars DataFrame to report on.
        title: Title shown at the top of the report.
        alert_config: Optional ``AlertConfig`` thresholds; ``None`` uses defaults.

    Returns:
        The report as an HTML string — write it to a ``.html`` file and
        open it in a browser.
    """
    explorer = DataExplorer(alert_config=alert_config)
    return _run_sync(explorer.to_html(df, title=title))


def run_alerts(
    df: pl.DataFrame,
    alert_config: AlertConfig | None = None,
) -> list[dict[str, Any]]:
    """Profile ``df`` and return only the triggered data-quality alerts.

    Equivalent to ``run_profile(df, alert_config).alerts``.

    Args:
        df: Polars DataFrame to check.
        alert_config: Optional ``AlertConfig`` thresholds; ``None`` uses defaults.

    Returns:
        List of alert dicts with keys ``type``, ``column`` (or ``columns``
        for ``high_correlation``), ``value`` and ``severity``.
    """
    return list(run_profile(df, alert_config).alerts)
