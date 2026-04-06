# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT01 — Exercise 6: Data Visualization
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Produce interactive EDA charts with ModelVisualizer — the
#   Kailash engine that wraps Plotly for consistent, publication-ready
#   figures without writing low-level chart code.
#
# TASKS:
#   1. Understand the ModelVisualizer API (histogram, scatter, bar, heatmap, line)
#   2. Visualise the HDB price distribution with a histogram
#   3. Explore price vs area with a scatter plot
#   4. Compare median prices across districts with a bar chart
#   5. Show correlation patterns with a heatmap
#   6. Plot price trends over time with a line chart
#   7. Export all figures as standalone HTML files
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl
from kailash_ml import ModelVisualizer

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = ASCENTDataLoader()
hdb = loader.load("ascent01", "hdb_resale.parquet")

# Prepare derived columns used across all charts
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
)

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
#   viz.feature_distribution()   — histogram / box plot
#   viz.scatter_plot()           — scatter with optional colour and size
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

# feature_distribution() expects a list of values and a feature name
prices = hdb["resale_price"].to_list()

fig_hist = viz.feature_distribution(
    values=prices,
    feature_name="Resale Price (S$)",
)
fig_hist.update_layout(
    title="HDB Resale Price Distribution",
    xaxis_title="Resale Price (S$)",
    yaxis_title="Number of Transactions",
)
fig_hist.write_html("ex6_price_histogram.html")
print("Saved: ex6_price_histogram.html")

# Also show price per sqm distribution — normalises for flat size
price_sqm_values = hdb["price_per_sqm"].drop_nulls().to_list()
fig_sqm = viz.feature_distribution(
    values=price_sqm_values,
    feature_name="Price per sqm (S$)",
)
fig_sqm.update_layout(title="HDB Price per sqm Distribution")
fig_sqm.write_html("ex6_price_per_sqm_histogram.html")
print("Saved: ex6_price_per_sqm_histogram.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Scatter plot — price vs floor area
# ══════════════════════════════════════════════════════════════════════

# Scatter plots reveal relationships between two numeric variables.
# A positive slope here means larger flats cost more — but by how much?
# Outliers stand out as isolated points far from the main cluster.

# Sample for plotting speed — a scatter of 500k points is unreadable
hdb_sample = hdb.sample(n=min(5_000, hdb.height), seed=42)

x_values = hdb_sample["floor_area_sqm"].to_list()
y_values = hdb_sample["resale_price"].to_list()

fig_scatter = viz.scatter_plot(
    x_values=x_values,
    y_values=y_values,
    x_label="Floor Area (sqm)",
    y_label="Resale Price (S$)",
)
fig_scatter.update_layout(title="HDB Resale Price vs Floor Area")
fig_scatter.write_html("ex6_price_vs_area_scatter.html")
print("Saved: ex6_price_vs_area_scatter.html")


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

# feature_importance() wants a dict: {label: value}
# We build it from the aggregated DataFrame
price_by_town = dict(
    zip(
        district_prices["town"].to_list(),
        district_prices["median_price"].to_list(),
    )
)

fig_bar = viz.feature_importance(
    importance_dict=price_by_town,
    title="Median HDB Resale Price by Town",
)
fig_bar.update_layout(
    xaxis_title="Median Resale Price (S$)",
    yaxis_title="Town",
)
fig_bar.write_html("ex6_median_price_by_town.html")
print("Saved: ex6_median_price_by_town.html")

# Also chart transaction volume — a different story from price
volume_by_town = dict(
    zip(
        district_prices.sort("transaction_count", descending=True)["town"].to_list(),
        district_prices.sort("transaction_count", descending=True)[
            "transaction_count"
        ].to_list(),
    )
)

fig_volume = viz.feature_importance(
    importance_dict=volume_by_town,
    title="HDB Transaction Volume by Town",
)
fig_volume.update_layout(xaxis_title="Number of Transactions")
fig_volume.write_html("ex6_volume_by_town.html")
print("Saved: ex6_volume_by_town.html")


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

# Compute Pearson correlations using Polars
corr_data: list[list[float]] = []
for col_a in numeric_cols:
    row = []
    for col_b in numeric_cols:
        corr = hdb_numeric[col_a].pearson_corr(hdb_numeric[col_b])
        row.append(round(corr, 3))
    corr_data.append(row)

fig_heatmap = viz.confusion_matrix(
    matrix=corr_data,
    labels=numeric_cols,
)
fig_heatmap.update_layout(
    title="Pearson Correlation Matrix — HDB Features",
    coloraxis_colorbar_title="Correlation",
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
    price_series[town] = [town_prices_by_year.get(y) for y in years]

fig_line = viz.training_history(
    history=price_series,
    x_label="Year",
    y_label="Median Resale Price (S$)",
)
fig_line.update_layout(title="Annual Median HDB Price — Top 5 Towns")
fig_line.write_html("ex6_price_trends.html")
print("Saved: ex6_price_trends.html")

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
    "National Median Price": national_annual["median_price"].to_list(),
}
fig_national = viz.training_history(
    history=national_series,
    x_label="Year",
    y_label="Median Resale Price (S$)",
)
fig_national.update_layout(title="Singapore HDB National Price Trend")
fig_national.write_html("ex6_national_price_trend.html")
print("Saved: ex6_national_price_trend.html")


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

print("\n✓ Exercise 6 complete — interactive EDA charts with ModelVisualizer")
