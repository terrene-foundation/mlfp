# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT1 — Exercise 3: DataExplorer Profiling on Dirty Economic Data
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use DataExplorer's async profiling to diagnose data quality
#   issues in a messy multi-source economic dataset. Learn to configure
#   AlertConfig and interpret the 8 alert types.
#
# TASKS:
#   1. Load and inspect Singapore economic indicators (CPI, employment, FX)
#   2. Merge datasets with different reporting frequencies
#   3. Configure AlertConfig with domain-appropriate thresholds
#   4. Run async profiling and interpret alerts
#   5. Use DataExplorer.compare() to analyse data quality across time periods
#   6. Generate a profiling report
#
# DATA QUALITY ISSUES (by design):
#   - Mixed granularity (monthly CPI, quarterly employment, daily FX)
#   - Missing quarters in employment data
#   - Currency conversion inconsistencies
#   - Outlier periods (COVID, GFC)
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

# Three datasets with different reporting frequencies
cpi = loader.load("ascent01", "sg_cpi.csv")  # Monthly
employment = loader.load("ascent01", "sg_employment.csv")  # Quarterly
fx_rates = loader.load("ascent01", "sg_fx_rates.csv")  # Daily


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Inspect each dataset independently
# ══════════════════════════════════════════════════════════════════════

print("=== CPI Data ===")
print(f"Shape: {cpi.shape}")
print(f"Columns: {cpi.columns}")
print(f"Date range: {cpi['date'].min()} to {cpi['date'].max()}")
print(f"Nulls per column:")
for col in cpi.columns:
    null_count = cpi[col].null_count()
    if null_count > 0:
        print(f"  {col}: {null_count} ({null_count / cpi.height:.1%})")
print(cpi.head(5))

print("\n=== Employment Data ===")
print(f"Shape: {employment.shape}")
print(f"Columns: {employment.columns}")
print(employment.head(5))

print("\n=== FX Rates Data ===")
print(f"Shape: {fx_rates.shape}")
print(f"Columns: {fx_rates.columns}")
print(fx_rates.head(5))


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Merge datasets with different granularities
# ══════════════════════════════════════════════════════════════════════
# Strategy: Align everything to monthly frequency
#   - CPI: already monthly
#   - Employment: forward-fill quarterly → monthly
#   - FX: aggregate daily → monthly (mean)

# Parse dates
cpi = cpi.with_columns(pl.col("date").str.to_date("%Y-%m-%d").alias("date"))

employment = employment.with_columns(
    pl.col("date").str.to_date("%Y-%m-%d").alias("date")
)

fx_rates = fx_rates.with_columns(pl.col("date").str.to_date("%Y-%m-%d").alias("date"))

# Create a monthly date column for joining
cpi = cpi.with_columns(pl.col("date").dt.truncate("1mo").alias("month_date"))

# Employment: extract month, forward-fill to monthly
employment = employment.with_columns(
    pl.col("date").dt.truncate("1mo").alias("month_date")
)

# Create a complete monthly date range for forward-filling
date_range = pl.date_range(
    cpi["month_date"].min(),
    cpi["month_date"].max(),
    interval="1mo",
    eager=True,
)
monthly_spine = pl.DataFrame({"month_date": date_range})

# Join employment onto monthly spine and forward-fill
employment_monthly = monthly_spine.join(
    employment.drop("date"),
    on="month_date",
    how="left",
).sort("month_date")

# Forward-fill quarterly values
employment_cols = [c for c in employment_monthly.columns if c != "month_date"]
employment_monthly = employment_monthly.with_columns(
    [pl.col(c).forward_fill() for c in employment_cols]
)

# FX rates: aggregate to monthly means
fx_monthly = (
    fx_rates.with_columns(pl.col("date").dt.truncate("1mo").alias("month_date"))
    .group_by("month_date")
    .agg([pl.col(c).mean() for c in fx_rates.columns if c != "date"])
    .sort("month_date")
)

# Merge all three
economic = (
    cpi.join(
        employment_monthly,
        on="month_date",
        how="left",
        suffix="_emp",
    )
    .join(
        fx_monthly,
        on="month_date",
        how="left",
        suffix="_fx",
    )
    .sort("month_date")
)

print(f"\n=== Merged Economic Dataset ===")
print(f"Shape: {economic.shape}")
print(f"Columns: {economic.columns}")
print(f"Date range: {economic['month_date'].min()} to {economic['month_date'].max()}")

# Check null counts after merge
null_summary = []
for col in economic.columns:
    nc = economic[col].null_count()
    if nc > 0:
        null_summary.append({"column": col, "nulls": nc, "pct": nc / economic.height})

if null_summary:
    print("\nNull summary after merge:")
    for ns in null_summary:
        print(f"  {ns['column']}: {ns['nulls']} ({ns['pct']:.1%})")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Configure AlertConfig with domain-appropriate thresholds
# ══════════════════════════════════════════════════════════════════════
# Economic data has specific characteristics:
#   - High correlation between macro indicators is EXPECTED (not alarming)
#   - Missing data from quarterly sources is EXPECTED after forward-fill
#   - Skewness in employment/GDP growth is normal during crisis periods
#   - We care more about constant columns (data pipeline failures)

alert_config = AlertConfig(
    high_correlation_threshold=0.95,  # Relax: macro indicators are correlated
    high_null_pct_threshold=0.10,  # Relax: quarterly data has gaps
    constant_threshold=1,  # Strict: flag constant columns (pipeline issue)
    high_cardinality_ratio=0.95,  # Relax: date-like columns are expected
    skewness_threshold=3.0,  # Relax: crisis periods cause skew
    zero_pct_threshold=0.3,  # Relax: some indicators naturally hit zero
    imbalance_ratio_threshold=0.05,  # Strict: catch extreme class imbalance
    duplicate_pct_threshold=0.05,  # Moderate: some duplication from forward-fill
)

print(f"\n=== Custom AlertConfig ===")
print(f"Correlation threshold: {alert_config.high_correlation_threshold}")
print(f"Null threshold: {alert_config.high_null_pct_threshold}")
print(f"Skewness threshold: {alert_config.skewness_threshold}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Run async profiling and interpret alerts
# ══════════════════════════════════════════════════════════════════════


async def profile_economic_data():
    """Profile the merged economic dataset with DataExplorer."""

    explorer = DataExplorer(alert_config=alert_config)

    print("\n=== Running Async Profile ===")
    profile = await explorer.profile(economic)

    # Summary
    print(f"Rows: {profile.n_rows}, Columns: {profile.n_columns}")
    print(f"Duplicates: {profile.duplicate_count} ({profile.duplicate_pct:.1%})")
    print(f"Type summary: {profile.type_summary}")

    # Detailed alerts with interpretation
    print(f"\n--- Alerts ({len(profile.alerts)}) ---")
    for alert in profile.alerts:
        alert_type = alert["type"]
        col = alert.get("column", alert.get("columns", "N/A"))
        value = alert.get("value", "N/A")
        severity = alert["severity"]

        # Domain-specific interpretation
        interpretation = _interpret_alert(alert_type, col, value)
        print(f"\n  [{severity.upper()}] {alert_type}")
        print(f"    Column: {col}")
        print(f"    Value: {value}")
        print(f"    Interpretation: {interpretation}")

    # Missing patterns — crucial for economic data
    if profile.missing_patterns:
        print(f"\n--- Missing Patterns ---")
        for pattern in profile.missing_patterns[:5]:
            print(f"  {pattern}")

    # Spearman correlations (rank-based, better for non-linear economic relationships)
    if profile.spearman_matrix:
        print(f"\n--- Top Spearman Correlations ---")
        seen = set()
        corrs = []
        for col_a, row in profile.spearman_matrix.items():
            for col_b, corr in row.items():
                if col_a != col_b and (col_b, col_a) not in seen and abs(corr) > 0.8:
                    seen.add((col_a, col_b))
                    corrs.append((col_a, col_b, corr))
        corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        for col_a, col_b, corr in corrs[:10]:
            print(f"  {col_a} <-> {col_b}: {corr:.3f}")

    # Categorical associations (Cramer's V)
    if profile.categorical_associations:
        print(f"\n--- Categorical Associations (Cramer's V) ---")
        for col_a, row in profile.categorical_associations.items():
            for col_b, v in row.items():
                if col_a != col_b and v > 0.3:
                    print(f"  {col_a} <-> {col_b}: V = {v:.3f}")

    # Generate visualisations
    print("\n--- Generating Visualisations ---")
    vis_report = await explorer.visualize(economic)
    for name, fig in vis_report.figures.items():
        filename = f"ex3_{name}.html"
        fig.write_html(filename)
        print(f"  Saved: {filename}")

    return profile


def _interpret_alert(alert_type: str, column: str, value) -> str:
    """Provide domain-specific interpretation of DataExplorer alerts."""
    interpretations = {
        "high_nulls": (
            f"Column '{column}' has {value:.1%} missing values. "
            "For quarterly data forward-filled to monthly, ~25% nulls at edges is expected. "
            "For monthly data, investigate the data pipeline."
        ),
        "constant": (
            f"Column '{column}' has ≤1 unique value — likely a pipeline failure or "
            "a redundant column. Remove it before modelling."
        ),
        "high_skewness": (
            f"Column '{column}' has skewness={value:.2f}. "
            "In economic data, this often indicates crisis-period outliers (GFC, COVID). "
            "Consider: (a) log-transform, (b) winsorize, or (c) model separately."
        ),
        "high_zeros": (
            f"Column '{column}' has {value:.1%} zero values. "
            "Check if zeros represent actual measurements or missing data coded as zero."
        ),
        "high_cardinality": (
            f"Column '{column}' has cardinality ratio={value:.3f}. "
            "If this is a date or ID column, expected. If categorical, "
            "consider binning or encoding strategy."
        ),
        "high_correlation": (
            f"Columns {column} have |correlation|={value:.3f}. "
            "For macro indicators, high correlation is expected. "
            "Flag for VIF check before regression modelling."
        ),
        "duplicates": (
            f"Dataset has {value:.1%} duplicate rows. "
            "Forward-filling quarterly to monthly can create apparent duplicates. "
            "Verify these aren't data pipeline errors."
        ),
        "imbalanced": (
            f"Column '{column}' has minority class at {value:.1%}. "
            "Rare events in economic data (recessions) are naturally imbalanced."
        ),
    }
    return interpretations.get(
        alert_type, f"Alert type '{alert_type}' — review manually."
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare data quality across time periods
# ══════════════════════════════════════════════════════════════════════


async def compare_periods():
    """Compare pre-COVID vs COVID-era economic data quality."""

    explorer = DataExplorer(alert_config=alert_config)

    covid_cutoff = pl.date(2020, 3, 1)

    pre_covid = economic.filter(pl.col("month_date") < covid_cutoff)
    during_covid = economic.filter(pl.col("month_date") >= covid_cutoff)

    print(f"\n=== Period Comparison ===")
    print(f"Pre-COVID: {pre_covid.height} months")
    print(f"COVID-era: {during_covid.height} months")

    comparison = await explorer.compare(pre_covid, during_covid)

    print(f"\nShape comparison: {comparison['shape_comparison']}")
    print(f"Shared columns: {len(comparison['shared_columns'])}")

    # Show the biggest distribution shifts
    print(f"\n--- Column Deltas (biggest shifts) ---")
    deltas = sorted(
        comparison["column_deltas"],
        key=lambda d: abs(d.get("mean_delta", 0)),
        reverse=True,
    )
    for delta in deltas[:10]:
        col = delta.get("column", "?")
        mean_delta = delta.get("mean_delta", 0)
        std_delta = delta.get("std_delta", 0)
        print(f"  {col}: mean Δ={mean_delta:+,.2f}, std Δ={std_delta:+,.2f}")

    return comparison


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Generate HTML report
# ══════════════════════════════════════════════════════════════════════


async def generate_report():
    """Generate a self-contained profiling HTML report."""

    explorer = DataExplorer(alert_config=alert_config)

    report_html = await explorer.to_html(
        economic, title="Singapore Economic Indicators — Data Profile"
    )

    with open("ex3_economic_profile_report.html", "w") as f:
        f.write(report_html)
    print("\nSaved: ex3_economic_profile_report.html")


# ── Run all async tasks ──────────────────────────────────────────────


async def main():
    profile = await profile_economic_data()
    comparison = await compare_periods()
    await generate_report()
    return profile, comparison


profile, comparison = asyncio.run(main())

print("\n✓ Exercise 3 complete — DataExplorer profiling on dirty economic data")
