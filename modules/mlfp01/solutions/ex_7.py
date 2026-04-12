# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 7: Automated Data Profiling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Run automated data profiling on any dataset using DataExplorer
#   - Configure AlertConfig thresholds for data quality rules
#   - Compare two datasets and identify distribution differences
#   - Handle errors gracefully with try/except
#   - Use async functions and asyncio.run() in a real pipeline context
#
# PREREQUISITES: Complete Exercise 6 first (all of Exercises 1–6).
#
# ESTIMATED TIME: 50-60 minutes
#
# TASKS:
#   1. Load and inspect Singapore economic indicators (CPI, employment, FX)
#   2. Merge datasets with different reporting frequencies onto a monthly spine
#   3. Configure AlertConfig with domain-appropriate thresholds
#   4. Run async profiling and interpret each alert type
#   5. Compare data quality across pre-COVID and COVID-era periods
#   6. Generate a self-contained HTML profiling report
#
# DATASET: Three Singapore economic time-series datasets (deliberately messy):
#   - sg_cpi.csv:         Monthly Consumer Price Index (data.gov.sg / SingStat)
#   - sg_employment.csv:  Quarterly labour market statistics (MOM)
#   - sg_fx_rates.csv:    Daily SGD exchange rates (MAS)
#   Quality issues by design: mixed granularity, forward-fill gaps,
#   COVID-era outliers, and near-zero values in some trade-flow columns.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl
from kailash_ml import DataExplorer
from kailash_ml.engines.data_explorer import AlertConfig

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()

# Three datasets with deliberately different reporting frequencies
# This mirrors real-world situations where data comes from separate sources
cpi = loader.load("mlfp01", "sg_cpi.csv")  # Monthly consumer price index
employment = loader.load("mlfp01", "sg_employment.csv")  # Quarterly labour stats
fx_rates = loader.load("mlfp01", "sg_fx_rates.csv")  # Daily exchange rates

print("=" * 60)
print("  MLFP01 Exercise 7: Automated Data Profiling")
print("=" * 60)
print(f"\n  Data loaded:")
print(f"    sg_cpi.csv        ({cpi.height:,} rows — monthly)")
print(f"    sg_employment.csv ({employment.height:,} rows — quarterly)")
print(f"    sg_fx_rates.csv   ({fx_rates.height:,} rows — daily)")
print(f"  You're ready to start!\n")


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
# INTERPRETATION: Three datasets, three granularities. Before merging, count
# how many rows you'd expect on a monthly spine and compare with what each
# dataset provides. Employment has ~4x fewer rows than CPI because it's
# quarterly. FX rates have ~22x more rows than CPI because it's daily.
# The merge will resolve all three to monthly frequency — by aggregation (FX)
# and forward-fill (employment).

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert cpi.height > 0, "CPI dataset is empty"
assert employment.height > 0, "Employment dataset is empty"
assert fx_rates.height > 0, "FX rates dataset is empty"
assert "date" in cpi.columns, "CPI should have a 'date' column"
assert employment.height < cpi.height, (
    "Quarterly employment should have fewer rows than monthly CPI"
)
assert fx_rates.height > cpi.height, (
    "Daily FX rates should have more rows than monthly CPI"
)
print("\n✓ Checkpoint 1 passed — all three economic datasets loaded and inspected\n")


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

# Parse date columns — each dataset has a DIFFERENT date format!
# This is real-world messiness: CPI uses mixed MM/YYYY and YYYY-MM,
# employment uses "YYYY QN" quarters, FX rates use YYYY-MM-DD.
# Handling heterogeneous formats is a core data engineering skill.

# CPI: mixed formats ("01/2000", "2000-02") → normalise to first-of-month
# Normalise all CPI dates to YYYY-MM-01 format before parsing.
# The dataset intentionally has THREE different date formats — this is
# real-world messiness that data engineers encounter daily:
#   "01/2000"  → MM/YYYY   (some international sources)
#   "2000-02"  → YYYY-MM   (ISO-like, no day)
#   "201108"   → YYYYMM    (compact, no separator)
cpi = cpi.with_columns(
    pl.col("date")
    .str.replace(r"^(\d{2})/(\d{4})$", "$2-$1-01")    # MM/YYYY → YYYY-MM-01
    .str.replace(r"^(\d{4})(\d{2})$", "$1-$2-01")      # YYYYMM  → YYYY-MM-01
    .str.replace(r"^(\d{4})-(\d{2})$", "$1-$2-01")     # YYYY-MM → YYYY-MM-01
    .str.to_date("%Y-%m-%d")
    .alias("date")
)

# Employment: quarterly ("2000 Q1") → map to quarter start month
def quarter_to_date(q_str: str) -> str:
    """Convert '2000 Q1' to '2000-01-01'."""
    parts = q_str.split()
    year = parts[0]
    q = int(parts[1][1])
    month = {1: "01", 2: "04", 3: "07", 4: "10"}[q]
    return f"{year}-{month}-01"

employment = employment.with_columns(
    pl.col("quarter").map_elements(quarter_to_date, return_dtype=pl.String)
    .str.to_date("%Y-%m-%d")
    .alias("date")
)

# FX rates: YYYY-MM-DD format — Polars may auto-detect this as Date on load,
# so only parse if the column is still a string
if fx_rates["date"].dtype == pl.String:
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
# INTERPRETATION: Nulls after the merge fall into two categories:
# (1) Edge nulls: months at the very start of the series where forward-fill
#     has no prior value to inherit. These are expected and unavoidable.
# (2) Coverage gaps: months not covered by the auxiliary dataset at all.
#     These warrant investigation — check if the source data is complete.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert economic.height > 0, "Merged dataset is empty"
assert economic.height == monthly_spine.height, (
    f"Merged dataset should have {monthly_spine.height} rows (one per month), "
    f"got {economic.height}"
)
assert "month_date" in economic.columns, "Merged dataset should have month_date"
print("\n✓ Checkpoint 2 passed — datasets merged to monthly frequency\n")


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
# INTERPRETATION: Every threshold here is a deliberate choice, not a default.
# Setting high_correlation_threshold=0.95 means CPI and employment will likely
# NOT trigger a correlation alert — their structural relationship is expected.
# Setting skewness_threshold=3.0 means only the most extreme COVID/GFC shocks
# will be flagged. Always document your rationale; a reviewer should understand
# why you chose each number.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert alert_config.high_correlation_threshold == 0.95, (
    "Correlation threshold should be 0.95"
)
assert alert_config.skewness_threshold == 3.0, "Skewness threshold should be 3.0"
assert alert_config.high_null_pct_threshold == 0.10, "Null threshold should be 0.10"
print("\n✓ Checkpoint 3 passed — AlertConfig configured with domain-appropriate thresholds\n")


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
    # INTERPRETATION: Spearman measures rank-order correlation — it captures
    # monotonic relationships that Pearson misses (e.g., CPI may correlate
    # non-linearly with employment). A Spearman r > 0.8 between two predictors
    # signals potential multicollinearity: including both in a regression model
    # inflates coefficient standard errors and makes interpretation unreliable.

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
    # INTERPRETATION: Columns with the largest mean_delta experienced the
    # biggest distributional shift between pre-COVID and COVID-era periods.
    # A positive mean_delta means the COVID-era average is higher (inflation-driven
    # indicators like CPI, or pandemic-era FX volatility). A negative delta
    # signals a post-COVID decline in that variable (e.g., certain employment rates).
    # std_delta shows whether the distribution got wider (more volatile) or
    # narrower during COVID — most economic indicators became more volatile.

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

    # ── Checkpoint 4 ─────────────────────────────────────────────────
    assert profile is not None, "profile should not be None"
    assert profile.n_rows == economic.height, (
        f"Profile n_rows ({profile.n_rows}) should match economic.height ({economic.height})"
    )
    assert comparison is not None, "comparison should not be None"
    assert "column_deltas" in comparison, "comparison should contain column_deltas"
    import os
    assert os.path.exists("ex7_economic_profile_report.html"), (
        "HTML report file not created"
    )
    print("\n✓ Checkpoint 4 passed — DataExplorer profiling, comparison, and report complete\n")
    print("\n✓ Exercise 7 complete — DataExplorer profiling on dirty economic data")
except Exception as exc:
    # In a real pipeline you would log the full traceback and alert on-call.
    # For this exercise, print a readable message and re-raise so the
    # interpreter still shows the stack trace.
    print(f"\n[ERROR] Profiling failed: {exc}")
    raise


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 58)
print("  WHAT YOU'VE MASTERED")
print("═" * 58)
print("""
  ✓ DataExplorer: one call to profile an entire dataset
  ✓ AlertConfig: tuning thresholds for your specific domain
  ✓ Async: async def, await, asyncio.run() — parallel column profiling
  ✓ try/except: handling errors gracefully without crashing the program
  ✓ compare(): detecting distribution drift between two time periods
  ✓ to_html(): generating shareable profiling reports
  ✓ Alert interpretation: mapping each alert type to a cleaning action

  NEXT: In Exercise 8, you'll put all of M1 together in one end-to-end
  pipeline — load messy taxi trip data, profile it with DataExplorer,
  clean it based on the alerts, engineer temporal and spatial features,
  prepare it with PreprocessingPipeline, visualise key patterns, then
  re-profile to confirm quality improvement. This is the capstone of
  Module 1 and the foundation for Module 2 feature engineering.
""")
