# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT1 — Exercise 7: Automated Data Profiling
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use DataExplorer's async profiling engine to automatically
#   diagnose data quality issues in a messy multi-source economic dataset,
#   and learn to configure AlertConfig thresholds for a specific domain.
#
# TASKS:
#   1. Load and inspect Singapore economic indicators (CPI, employment, FX)
#   2. Merge datasets with different reporting frequencies onto a monthly spine
#   3. Configure AlertConfig with domain-appropriate thresholds
#   4. Run async profiling and interpret each alert type
#   5. Compare data quality across pre-COVID and COVID-era periods
#   6. Generate a self-contained HTML profiling report
#
# DATA QUALITY ISSUES (by design):
#   - Mixed granularity: monthly CPI, quarterly employment, daily FX
#   - Missing quarters in employment data after forward-fill
#   - Currency conversion inconsistencies in FX rates
#   - Outlier periods (COVID shock, Global Financial Crisis)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl
from kailash_ml import DataExplorer
from kailash_ml.engines.data_explorer import AlertConfig

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = ASCENTDataLoader()

# Three datasets with deliberately different reporting frequencies
# This mirrors real-world situations where data comes from separate sources
cpi = loader.load("ascent01", "sg_cpi.csv")  # Monthly consumer price index
employment = loader.load("ascent01", "sg_employment.csv")  # Quarterly labour stats
fx_rates = loader.load("ascent01", "sg_fx_rates.csv")  # Daily exchange rates


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Inspect each dataset independently
# ══════════════════════════════════════════════════════════════════════

# Always profile individual tables before merging — that way you know
# which quality issues originate where.

print("=== CPI Data (Monthly) ===")
print(f"Shape: {cpi.shape}")
print(f"Columns: {cpi.columns}")
print(f"Date range: {cpi['date'].min()} to {cpi['date'].max()}")
print("Nulls per column:")
for col in cpi.columns:
    null_count = cpi[col].null_count()
    if null_count > 0:
        print(f"  {col}: {null_count} ({null_count / cpi.height:.1%})")
print(cpi.head(5))

print("\n=== Employment Data (Quarterly) ===")
print(f"Shape: {employment.shape}")
print(f"Columns: {employment.columns}")
print(employment.head(5))

print("\n=== FX Rates Data (Daily) ===")
print(f"Shape: {fx_rates.shape}")
print(f"Columns: {fx_rates.columns}")
print(fx_rates.head(5))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Merge datasets with different granularities
# ══════════════════════════════════════════════════════════════════════
# Strategy: align everything to monthly frequency.
#   - CPI:        already monthly — use as-is
#   - Employment: quarterly → forward-fill each quarter across 3 months
#   - FX rates:   daily → aggregate to monthly mean
#
# Forward-filling is the standard approach for quarterly economic data:
# the Q1 figure applies to Jan, Feb, and Mar; Q2 to Apr, May, Jun, etc.

# Parse all date columns to proper date types
cpi = cpi.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
employment = employment.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
fx_rates = fx_rates.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))

# Truncate all dates to the first of the month for joining
cpi = cpi.with_columns(pl.col("date").dt.truncate("1mo").alias("month_date"))
employment = employment.with_columns(
    pl.col("date").dt.truncate("1mo").alias("month_date")
)

# Build a complete monthly spine from the CPI date range
# This ensures we have a row for every month, even months with no employment data
date_range = pl.date_range(
    cpi["month_date"].min(),
    cpi["month_date"].max(),
    interval="1mo",
    eager=True,
)
monthly_spine = pl.DataFrame({"month_date": date_range})

# Join employment onto spine → creates nulls for non-quarter-start months
# Then forward-fill: each null inherits the previous quarter's value
employment_monthly = (
    monthly_spine.join(
        employment.drop("date"),
        on="month_date",
        how="left",
    )
    .sort("month_date")
    .with_columns(
        [
            pl.col(c).forward_fill()
            for c in employment.columns
            if c not in ("date", "month_date")
        ]
    )
)

# Aggregate FX rates: daily → monthly mean
fx_monthly = (
    fx_rates.with_columns(pl.col("date").dt.truncate("1mo").alias("month_date"))
    .group_by("month_date")
    .agg([pl.col(c).mean() for c in fx_rates.columns if c != "date"])
    .sort("month_date")
)

# Merge all three sources on month_date
# suffix= prevents column name collisions when both tables have similar names
economic = (
    cpi.join(employment_monthly, on="month_date", how="left", suffix="_emp")
    .join(fx_monthly, on="month_date", how="left", suffix="_fx")
    .sort("month_date")
)

print(f"\n=== Merged Economic Dataset ===")
print(f"Shape: {economic.shape}")
print(f"Date range: {economic['month_date'].min()} to {economic['month_date'].max()}")

null_summary = [
    {"column": col, "nulls": nc, "pct": nc / economic.height}
    for col in economic.columns
    if (nc := economic[col].null_count()) > 0
]
if null_summary:
    print("\nNull summary after merge:")
    for ns in null_summary:
        print(f"  {ns['column']}: {ns['nulls']} ({ns['pct']:.1%})")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Configure AlertConfig with domain-appropriate thresholds
# ══════════════════════════════════════════════════════════════════════
# AlertConfig controls when DataExplorer raises a warning.
# The defaults are designed for typical tabular ML datasets — but
# economic time-series data has different characteristics, so we tune each
# threshold to match what we actually expect to see in this dataset.

alert_config = AlertConfig(
    # Correlation: macro indicators (CPI, employment, FX) are structurally
    # correlated — a high correlation is expected, not an error.
    # Raise the threshold so we only flag near-perfect collinearity.
    high_correlation_threshold=0.95,
    # Nulls: quarterly data forward-filled to monthly will have ~0% nulls
    # mid-series but potentially some at the edges. 10% is a safe threshold.
    high_null_pct_threshold=0.10,
    # Constants: a column with only one unique value is definitely a
    # pipeline failure (e.g., a column that never got populated).
    constant_threshold=1,
    # Cardinality: date-derived columns will look "high cardinality" because
    # every month is unique. Relax the threshold to avoid false alarms.
    high_cardinality_ratio=0.95,
    # Skewness: crisis periods (GFC 2008, COVID 2020) create extreme spikes
    # in economic indicators. A threshold of 3.0 flags only severe cases.
    skewness_threshold=3.0,
    # Zeros: some indicators legitimately hit zero (e.g., certain trade flows
    # during supply-chain disruptions). Allow up to 30% zeros.
    zero_pct_threshold=0.30,
    # Imbalance: keep strict — if a categorical column is 98% one value
    # it likely encodes a near-constant feature, which adds no information.
    imbalance_ratio_threshold=0.05,
    # Duplicates: forward-filling creates repeated values but not duplicate rows
    # (because month_date differs). A small threshold catches true duplicates.
    duplicate_pct_threshold=0.05,
)

print(f"\n=== Custom AlertConfig ===")
print(f"Correlation threshold: {alert_config.high_correlation_threshold}")
print(f"Null threshold:        {alert_config.high_null_pct_threshold:.0%}")
print(f"Skewness threshold:    {alert_config.skewness_threshold}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Run async profiling and interpret alerts
# ══════════════════════════════════════════════════════════════════════
# DataExplorer.profile() is an async function — it must run inside an
# async context. asyncio.run() creates an event loop and runs our async
# main() function to completion, then returns the result.
#
# Why async? DataExplorer can profile multiple columns in parallel using
# Python's async I/O model, which is faster for wide DataFrames.


def _interpret_alert(alert_type: str, column: str, value) -> str:
    """Provide domain-specific interpretation of a DataExplorer alert."""
    interpretations = {
        "high_nulls": (
            f"Column '{column}' has {value:.1%} missing values. "
            "For quarterly data forward-filled to monthly, edge nulls are expected. "
            "For monthly CPI/FX columns, investigate the data pipeline."
        ),
        "constant": (
            f"Column '{column}' has <=1 unique value — likely a pipeline failure. "
            "Remove it before modelling; it carries no information."
        ),
        "high_skewness": (
            f"Column '{column}' skewness={value:.2f}. "
            "Crisis-period outliers (GFC 2008, COVID 2020) cause this. "
            "Consider: log-transform, winsorise at 99th percentile, or model separately."
        ),
        "high_zeros": (
            f"Column '{column}' has {value:.1%} zero values. "
            "Check whether zeros are real measurements or missing-data coded as zero."
        ),
        "high_cardinality": (
            f"Column '{column}' cardinality ratio={value:.3f}. "
            "If this is a date column, expected. If categorical, consider binning."
        ),
        "high_correlation": (
            f"Columns {column} have |correlation|={value:.3f}. "
            "Expected for macro indicators — flag for VIF check before regression."
        ),
        "duplicates": (
            f"Dataset has {value:.1%} duplicate rows. "
            "Forward-filling quarterly data should not create true row duplicates. "
            "Check whether month_date is truly unique."
        ),
        "imbalanced": (
            f"Column '{column}' minority class at {value:.1%}. "
            "Rare events (recessions, crises) are naturally rare in economic data."
        ),
    }
    return interpretations.get(
        alert_type, f"Alert type '{alert_type}' — review manually."
    )


async def profile_economic_data():
    """Profile the merged economic dataset with DataExplorer."""
    explorer = DataExplorer(alert_config=alert_config)

    print("\n=== Running DataExplorer Profile ===")
    profile = await explorer.profile(economic)

    # Top-level summary
    print(f"Rows: {profile.n_rows}  Columns: {profile.n_columns}")
    print(f"Duplicates: {profile.duplicate_count} ({profile.duplicate_pct:.1%})")
    print(f"Type summary: {profile.type_summary}")

    # Alerts — the main output to act on
    print(f"\n--- Alerts ({len(profile.alerts)}) ---")
    for alert in profile.alerts:
        alert_type = alert["type"]
        col = alert.get("column", alert.get("columns", "N/A"))
        value = alert.get("value", "N/A")
        severity = alert["severity"]
        interpretation = _interpret_alert(alert_type, col, value)

        print(f"\n  [{severity.upper()}] {alert_type}")
        print(f"    Column: {col}")
        print(f"    Value:  {value}")
        print(f"    Why:    {interpretation}")

    # Column-level statistics
    print("\n--- Column Profiles ---")
    for col in profile.columns:
        if col.inferred_type == "numeric":
            print(
                f"  {col.name}: {col.inferred_type} | "
                f"mean={col.mean:.3g}, std={col.std:.3g}, "
                f"nulls={col.null_pct:.1%}, skew={col.skewness:.2f}"
            )
        else:
            print(
                f"  {col.name}: {col.inferred_type} | "
                f"unique={col.unique_count}, nulls={col.null_pct:.1%}"
            )

    # Spearman correlations — rank-based, better for non-linear economic relationships
    if profile.spearman_matrix:
        print("\n--- Top Spearman Correlations (|r| > 0.8) ---")
        seen: set[tuple[str, str]] = set()
        corrs = []
        for col_a, row in profile.spearman_matrix.items():
            for col_b, corr in row.items():
                if col_a != col_b and (col_b, col_a) not in seen and abs(corr) > 0.8:
                    seen.add((col_a, col_b))
                    corrs.append((col_a, col_b, corr))
        corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        for col_a, col_b, corr in corrs[:10]:
            print(f"  {col_a} <-> {col_b}: {corr:.3f}")

    # Generate HTML visualisations
    print("\n--- Generating Visualisations ---")
    vis_report = await explorer.visualize(economic)
    for name, fig in vis_report.figures.items():
        filename = f"ex7_{name}.html"
        fig.write_html(filename)
        print(f"  Saved: {filename}")

    return profile


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare data quality across time periods
# ══════════════════════════════════════════════════════════════════════


async def compare_periods():
    """Compare pre-COVID vs COVID-era economic data quality.

    DataExplorer.compare() profiles two DataFrames separately and then
    computes column-level deltas — mean shifts, std changes, null changes.
    This is the standard way to detect distribution drift across time.
    """
    explorer = DataExplorer(alert_config=alert_config)

    covid_cutoff = pl.date(2020, 3, 1)
    pre_covid = economic.filter(pl.col("month_date") < covid_cutoff)
    during_covid = economic.filter(pl.col("month_date") >= covid_cutoff)

    print(f"\n=== Period Comparison ===")
    print(f"Pre-COVID months:  {pre_covid.height}")
    print(f"COVID-era months:  {during_covid.height}")

    comparison = await explorer.compare(pre_covid, during_covid)

    print(f"\nShape comparison: {comparison['shape_comparison']}")
    print(f"Shared columns:   {len(comparison['shared_columns'])}")

    # Sort by absolute mean delta to surface the biggest distribution shifts
    print("\n--- Column Deltas (largest mean shifts) ---")
    deltas = sorted(
        comparison["column_deltas"],
        key=lambda d: abs(d.get("mean_delta", 0)),
        reverse=True,
    )
    for delta in deltas[:10]:
        col = delta.get("column", "?")
        mean_delta = delta.get("mean_delta", 0)
        std_delta = delta.get("std_delta", 0)
        print(f"  {col}: mean Δ={mean_delta:+,.3g}  std Δ={std_delta:+,.3g}")

    return comparison


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Generate self-contained HTML report
# ══════════════════════════════════════════════════════════════════════


async def generate_report():
    """Write a single-file HTML profiling report for the economic dataset.

    to_html() embeds all charts and statistics in one HTML file — no server
    needed. Students and stakeholders can open it in any browser.
    """
    explorer = DataExplorer(alert_config=alert_config)

    report_html = await explorer.to_html(
        economic,
        title="Singapore Economic Indicators — Data Profile",
    )

    report_path = "ex7_economic_profile_report.html"
    with open(report_path, "w") as f:
        f.write(report_html)
    print(f"\nSaved: {report_path}")


# ── Run all async tasks ───────────────────────────────────────────────
# We bundle all three async calls into a single main() coroutine and
# run it once with asyncio.run(). This avoids creating multiple event loops.


async def main():
    profile = await profile_economic_data()
    comparison = await compare_periods()
    await generate_report()
    return profile, comparison


# try/except shows students how to handle errors — a natural introduction
# to Python exception handling in a realistic context.
try:
    profile, comparison = asyncio.run(main())
    print("\n✓ Exercise 7 complete — DataExplorer profiling on dirty economic data")
except Exception as exc:
    # In a real pipeline you would log the full traceback and alert on-call.
    # For this exercise, print a readable message and re-raise so the
    # interpreter still shows the stack trace.
    print(f"\n[ERROR] Profiling failed: {exc}")
    raise
