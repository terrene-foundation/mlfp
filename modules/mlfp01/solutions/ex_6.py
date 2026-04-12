# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 6: Data Visualization
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Select the appropriate chart type for a given data question
#   - Create interactive visualisations with Plotly via ModelVisualizer
#   - Apply chart design principles (Gestalt, visual hierarchy)
#   - Identify and avoid misleading chart designs
#   - Export figures as standalone HTML files for sharing
#
# PREREQUISITES: Complete Exercise 5 first (window functions, trends).
#
# ESTIMATED TIME: 50-60 minutes
#
# TASKS:
#   1. Understand the ModelVisualizer API (histogram, scatter, bar, heatmap, line)
#   2. Visualise the HDB price distribution with a histogram
#   3. Explore price vs area with a scatter plot
#   4. Compare median prices across districts with a bar chart
#   5. Show correlation patterns with a heatmap
#   6. Plot price trends over time with a line chart
#   7. Export all figures as standalone HTML files
#
# DATASET: Singapore HDB resale flat transactions
#   Source: Housing & Development Board (data.gov.sg)
#   Rows: ~500,000 transactions | The data is aggregated differently for
#   each chart type to answer a specific question
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl
from kailash_ml import ModelVisualizer

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

# Prepare derived columns used across all charts
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
)

print("=" * 60)
print("  MLFP01 Exercise 6: Data Visualization")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.height:,} rows, {hdb.width} columns)")
print(f"  You're ready to start!\n")

print("=== HDB Resale Dataset ===")
print(f"Shape: {hdb.shape}")

# Initialise the visualiser — one instance, many chart types
viz = ModelVisualizer()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Understand ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

# ModelVisualizer wraps Plotly to give you a consistent API for the
# most common EDA chart types. Every method returns a Plotly Figure
# object that you can:
#   - Display inline in Jupyter: fig.show()
#   - Save as standalone HTML:   fig.write_html("filename.html")
#   - Further customise:         fig.update_layout(title="...")
#
# The five EDA methods you'll use most:
#   viz.histogram()   — histogram / box plot
#   viz.scatter()           — scatter with optional colour and size
#   viz.feature_importance()     — horizontal bar chart
#   viz.confusion_matrix()       — heatmap (works for correlation too)
#   viz.training_history()       — line chart for time series
#
# Chart data is passed as plain Python dicts and lists — not DataFrames.
# You aggregate with Polars, then hand the results to the visualiser.


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Histogram — price distribution
# ══════════════════════════════════════════════════════════════════════

# Histograms reveal the shape of a distribution:
# - Where is the peak? (modal price)
# - Is it symmetric or skewed? (long right tail = many cheap, few expensive)
# - Are there multiple peaks? (different market segments)

# histogram() takes a DataFrame and column name — it auto-generates a
# distribution chart showing counts per bin.
fig_hist = viz.histogram(
    data=hdb,
    column="resale_price",
    bins=40,
    title="HDB Resale Price Distribution",
)
fig_hist.write_html("ex6_price_histogram.html")
print("Saved: ex6_price_histogram.html")
# INTERPRETATION: The HDB price distribution is right-skewed — most transactions
# cluster in the S$350k–600k range, with a long tail of expensive (>S$800k)
# "million-dollar HDB" transactions. A right-skewed distribution means that
# mean > median: the average is pulled upward by expensive outliers.
# Use median for "typical price" reporting; use mean for total market value.

# Also show price per sqm distribution — normalises for flat size
hdb_clean = hdb.filter(pl.col("price_per_sqm").is_not_null())
fig_sqm = viz.histogram(
    data=hdb_clean,
    column="price_per_sqm",
    bins=40,
    title="HDB Price per sqm Distribution",
)
fig_sqm.write_html("ex6_price_per_sqm_histogram.html")
print("Saved: ex6_price_per_sqm_histogram.html")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
import os
assert os.path.exists("ex6_price_histogram.html"), "Histogram HTML file not created"
assert os.path.exists("ex6_price_per_sqm_histogram.html"), "Price/sqm histogram not created"
print("\n✓ Checkpoint 1 passed — histogram files saved successfully\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Scatter plot — price vs floor area
# ══════════════════════════════════════════════════════════════════════

# Scatter plots reveal relationships between two numeric variables.
# A positive slope here means larger flats cost more — but by how much?
# Outliers stand out as isolated points far from the main cluster.

# Sample for plotting speed — a scatter of 500k points is unreadable
hdb_sample = hdb.sample(n=min(5_000, hdb.height), seed=42)

fig_scatter = viz.scatter(
    data=hdb_sample,
    x="floor_area_sqm",
    y="resale_price",
    title="HDB Resale Price vs Floor Area",
)
fig_scatter.write_html("ex6_price_vs_area_scatter.html")
print("Saved: ex6_price_vs_area_scatter.html")
# INTERPRETATION: You'll see a clear positive relationship but with wide
# vertical spread — at 100 sqm, prices range from S$400k to S$900k+.
# This spread is explained by other factors: town, floor level, remaining
# lease, and proximity to amenities. Floor area alone explains maybe 40–50%
# of price variance. That remaining 50–60% is what the ML models in M3 learn.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert os.path.exists("ex6_price_vs_area_scatter.html"), "Scatter plot HTML not created"
print("\n✓ Checkpoint 2 passed — scatter plot saved successfully\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Bar chart — median price by district
# ══════════════════════════════════════════════════════════════════════

# Bar charts compare a single metric across categories.
# feature_importance() is ModelVisualizer's horizontal bar chart —
# it was designed for ML feature importance but works for any ranked comparison.

# Aggregate: one row per district, sorted by median price
district_prices = (
    hdb.group_by("town")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        pl.len().alias("transaction_count"),
    )
    .sort("median_price", descending=True)
)

# metric_comparison() accepts {model_name: {metric_name: value}} —
# we repurpose it to show town → median price as a comparison chart.
price_by_town = {
    town: {"Median Price (S$)": price}
    for town, price in zip(
        district_prices["town"].to_list(),
        district_prices["median_price"].to_list(),
    )
}

fig_bar = viz.metric_comparison(price_by_town)
fig_bar.write_html("ex6_median_price_by_town.html")
print("Saved: ex6_median_price_by_town.html")
# INTERPRETATION: The bar chart immediately reveals the price hierarchy.
# Central / mature estate towns appear at the top; peripheral new towns
# at the bottom. The Gestalt principle of continuity applies here:
# horizontal bars are easier to compare than vertical ones when labels
# are long — which is why feature_importance() uses horizontal bars.

# Also chart transaction volume — a different story from price
volume_by_town = dict(
    zip(
        district_prices.sort("transaction_count", descending=True)["town"].to_list(),
        district_prices.sort("transaction_count", descending=True)[
            "transaction_count"
        ].to_list(),
    )
)

volume_metrics = {
    town: {"Transactions": float(count)}
    for town, count in volume_by_town.items()
}
fig_volume = viz.metric_comparison(volume_metrics)
fig_volume.write_html("ex6_volume_by_town.html")
print("Saved: ex6_volume_by_town.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert os.path.exists("ex6_median_price_by_town.html"), "Bar chart HTML not created"
assert os.path.exists("ex6_volume_by_town.html"), "Volume bar chart HTML not created"
assert len(price_by_town) == district_prices.height, (
    "price_by_town should have one entry per town"
)
print("\n✓ Checkpoint 3 passed — bar charts saved successfully\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Heatmap — correlation between numeric features
# ══════════════════════════════════════════════════════════════════════

# A correlation heatmap shows whether pairs of variables move together.
# +1.0 = perfectly positively correlated (bigger X → bigger Y)
# -1.0 = perfectly negatively correlated (bigger X → smaller Y)
#  0.0 = no linear relationship
#
# confusion_matrix() is ModelVisualizer's heatmap method — it accepts
# any 2D grid of values, not just ML confusion matrices.

# Build the correlation matrix from numeric columns
numeric_cols = ["resale_price", "floor_area_sqm", "price_per_sqm", "year"]
hdb_numeric = hdb.select(numeric_cols).drop_nulls()

# Compute Pearson correlations using numpy (Polars doesn't have a built-in
# pairwise correlation matrix method — numpy.corrcoef is the standard tool)
import numpy as np

np_data = hdb_numeric.to_numpy()
corr_matrix = np.corrcoef(np_data, rowvar=False)
corr_data = [[round(float(corr_matrix[i, j]), 3) for j in range(len(numeric_cols))]
             for i in range(len(numeric_cols))]

# Plotly heatmap for the correlation matrix (ModelVisualizer's confusion_matrix
# takes y_true/y_pred arrays, so we build the heatmap directly with plotly)
import plotly.graph_objects as go

fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr_data,
    x=numeric_cols,
    y=numeric_cols,
    colorscale="RdBu_r",
    zmin=-1, zmax=1,
    text=[[str(v) for v in row] for row in corr_data],
    texttemplate="%{text}",
))
fig_heatmap.update_layout(
    title="Pearson Correlation Matrix — HDB Features",
    width=600, height=500,
)
fig_heatmap.write_html("ex6_correlation_heatmap.html")
print("Saved: ex6_correlation_heatmap.html")

# Print the correlation values as text too
print("\n=== Pearson Correlations ===")
header = f"{'':>20}" + "".join(f"{c:>16}" for c in numeric_cols)
print(header)
for col_a, row in zip(numeric_cols, corr_data):
    row_str = f"{col_a:>20}" + "".join(f"{v:>16.3f}" for v in row)
    print(row_str)
# INTERPRETATION: The diagonal is always 1.0 (a variable correlates perfectly
# with itself). Off-diagonal values reveal relationships:
# - resale_price vs floor_area_sqm: moderate positive (larger = more expensive)
# - resale_price vs year: positive (prices have risen over time)
# - floor_area_sqm vs price_per_sqm: often near-zero or weakly negative
#   (bigger flats exist in cheaper towns, so sqm price is not driven by size)
# Any correlation above ~0.85 in a model's feature set is a multicollinearity
# warning — including both columns adds no new information.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert os.path.exists("ex6_correlation_heatmap.html"), "Heatmap HTML not created"
# Diagonal should be 1.0
for i in range(len(numeric_cols)):
    assert abs(corr_data[i][i] - 1.0) < 0.001, (
        f"Diagonal element [{i}][{i}] should be 1.0, got {corr_data[i][i]}"
    )
# Matrix should be symmetric
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        assert abs(corr_data[i][j] - corr_data[j][i]) < 0.001, (
            f"Correlation matrix should be symmetric at [{i}][{j}]"
        )
print("\n✓ Checkpoint 4 passed — correlation heatmap with valid symmetric matrix\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Line chart — price trends over time
# ══════════════════════════════════════════════════════════════════════

# Line charts are the natural choice for time series data.
# training_history() is ModelVisualizer's line chart — designed for
# ML training curves but equally useful for any x→y trend.

# Annual median price for the top 5 most-transacted towns
top_5_towns = (
    district_prices.sort("transaction_count", descending=True)["town"].head(5).to_list()
)

annual_prices = (
    hdb.filter(pl.col("town").is_in(top_5_towns))
    .group_by("year", "town")
    .agg(pl.col("resale_price").median().alias("median_price"))
    .sort("year")
)

# training_history() expects a dict of {series_name: [values]}
# and a separate list for the x-axis
years = sorted(annual_prices["year"].unique().to_list())

price_series: dict[str, list[float]] = {}
for town in top_5_towns:
    town_data = annual_prices.filter(pl.col("town") == town).sort("year")
    # Align to full year range — fill missing years with None
    town_prices_by_year = dict(
        zip(town_data["year"].to_list(), town_data["median_price"].to_list())
    )
    price_series[town] = [float(town_prices_by_year.get(y, 0) or 0) for y in years]

fig_line = viz.training_history(
    metrics=price_series,
    x_label="Year",
    y_label="Median Resale Price (S$)",
)
fig_line.update_layout(title="Annual Median HDB Price — Top 5 Towns")
fig_line.write_html("ex6_price_trends.html")
print("Saved: ex6_price_trends.html")
# INTERPRETATION: The line chart reveals divergence between towns over time.
# Towns that started similarly in 2010 may now differ by S$150k+ in median
# price. Look for: (1) which towns consistently lead, (2) which crossed over
# another (catchup towns), (3) which inflected sharply (policy effect or
# en-bloc activity). The Gestalt principle of connection applies: lines make
# temporal patterns visible in a way a bar chart cannot.

# National trend (all towns)
national_annual = (
    hdb.group_by("year")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
    )
    .sort("year")
)

national_series = {
    "National Median Price": [float(v) for v in national_annual["median_price"].to_list()],
}
fig_national = viz.training_history(
    metrics=national_series,
    x_label="Year",
    y_label="Median Resale Price (S$)",
)
fig_national.update_layout(title="Singapore HDB National Price Trend")
fig_national.write_html("ex6_national_price_trend.html")
print("Saved: ex6_national_price_trend.html")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert os.path.exists("ex6_price_trends.html"), "Price trends HTML not created"
assert os.path.exists("ex6_national_price_trend.html"), "National trend HTML not created"
assert len(top_5_towns) == 5, "Should have exactly 5 top towns"
assert len(price_series) == 5, "price_series should have one entry per town"
print("\n✓ Checkpoint 5 passed — line charts saved successfully\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Summary of all outputs
# ══════════════════════════════════════════════════════════════════════

outputs = [
    ("ex6_price_histogram.html", "Price distribution — shape of the market"),
    ("ex6_price_per_sqm_histogram.html", "Price/sqm distribution — normalised"),
    ("ex6_price_vs_area_scatter.html", "Price vs area — does size explain price?"),
    ("ex6_median_price_by_town.html", "Median price by town — where is expensive?"),
    ("ex6_volume_by_town.html", "Transaction volume by town — where is active?"),
    ("ex6_correlation_heatmap.html", "Correlation matrix — feature relationships"),
    ("ex6_price_trends.html", "Price trends — top 5 towns over time"),
    ("ex6_national_price_trend.html", "National price trend — macro view"),
]

print(f"\n{'=' * 60}")
print(f"  VISUALISATION OUTPUTS")
print(f"{'=' * 60}")
for filename, description in outputs:
    print(f"  {filename}")
    print(f"    → {description}")
print(f"{'=' * 60}")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
missing = [f for f, _ in outputs if not os.path.exists(f)]
assert not missing, f"These HTML files were not created: {missing}"
print("\n✓ Checkpoint 6 passed — all 8 visualisation files saved\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 58)
print("  WHAT YOU'VE MASTERED")
print("═" * 58)
print("""
  ✓ ModelVisualizer: one API for histograms, scatter, bar, heatmap, line
  ✓ Chart selection: histogram→distribution, scatter→relationship,
    bar→comparison, heatmap→correlation, line→time series
  ✓ Gestalt principles: continuity (lines), similarity (colors), proximity
  ✓ Plotly interactivity: hover, zoom, pan on every chart
  ✓ HTML export: fig.write_html() for shareable standalone files
  ✓ Sampling for scatter: 5,000 points instead of 500k — still informative
  ✓ Interpretation: what each chart shape means in the HDB market context

  NEXT: In Exercise 7, you'll automate the data quality analysis
  you've been doing manually. DataExplorer profiles an entire messy
  dataset in one call — detecting missing values, outliers, skew,
  high correlation, and duplicates — and returns typed alerts that
  map directly to cleaning actions. You'll also see try/except for
  the first time and learn how async functions work in Python.
""")
