---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 1.6: Data Visualization

### Module 1: Data Pipelines and Visualisation

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Create interactive charts with Plotly Express
- Choose the right chart type for your analytical question
- Use Kailash `ModelVisualizer` for ML-ready visualisations
- Apply visualisation best practices for clear communication

---

## Recap: Lesson 1.5

- `over()` computes per-group values without collapsing rows
- `rolling_mean()` smooths time series to reveal trends
- `shift()` enables period-over-period comparisons
- Lazy frames optimise multi-step pipelines

---

## Why Visualization Matters

```
Raw data:  [480000, 350000, 520000, 410000, 390000, ...]
           → Hard to spot patterns in thousands of numbers

Chart:     Instantly reveals trends, outliers, distributions
           → "Prices are rising in central regions"
```

A good chart answers a question. A bad chart just looks pretty.

---

## Plotly Express: Quick Interactive Charts

```python
import plotly.express as px

# Convert Polars to dict for Plotly
town_avg = (
    df.group_by("town")
    .agg(pl.col("price").mean().alias("avg_price"))
    .sort("avg_price", descending=True)
)

fig = px.bar(
    town_avg.to_pandas(),
    x="town", y="avg_price",
    title="Average HDB Price by Town"
)
fig.show()
```

Note: Plotly requires pandas for input -- `to_pandas()` is used only at the visualisation boundary.

---

## Chart Type Decision Guide

```
What question are you answering?
│
├─ "How much?"        → Bar chart
├─ "How is it spread?"→ Histogram / Box plot
├─ "How does it trend?"→ Line chart
├─ "How do two things relate?" → Scatter plot
├─ "What's the proportion?" → Pie / Stacked bar
└─ "How do categories compare across metrics?"
                       → Heatmap / Grouped bar
```

---

## Histogram: Distribution of Prices

```python
fig = px.histogram(
    df.to_pandas(),
    x="price",
    nbins=50,
    title="Distribution of HDB Resale Prices",
    labels={"price": "Resale Price ($)"},
)
fig.show()
```

Histograms answer: "How are values spread?" and "Are there clusters?"

---

## Box Plot: Comparing Distributions

```python
fig = px.box(
    df.to_pandas(),
    x="flat_type",
    y="price",
    title="Price Distribution by Flat Type",
    color="flat_type",
)
fig.show()
```

Box plots show:

- **Median** (line in box)
- **Interquartile range** (box edges: 25th--75th percentile)
- **Whiskers** (1.5x IQR)
- **Outliers** (dots beyond whiskers)

---

## Scatter Plot: Two Variables

```python
fig = px.scatter(
    df.to_pandas(),
    x="floor_area",
    y="price",
    color="flat_type",
    title="Price vs Floor Area",
    labels={"floor_area": "Floor Area (sqm)", "price": "Price ($)"},
    opacity=0.5,
)
fig.show()
```

Scatter plots reveal relationships: "Bigger flats cost more, but the rate varies by type."

---

## Line Chart: Trends Over Time

```python
monthly = (
    df.group_by("transaction_date")
    .agg(pl.col("price").mean().alias("avg_price"))
    .sort("transaction_date")
)

fig = px.line(
    monthly.to_pandas(),
    x="transaction_date",
    y="avg_price",
    title="HDB Price Trend Over Time",
)
fig.show()
```

---

## Heatmap: Correlation Matrix

```python
# Compute correlations in Polars
numeric_cols = ["price", "floor_area", "lease_years", "storey"]
corr_matrix = df.select(numeric_cols).to_pandas().corr()

fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    title="Feature Correlation Heatmap",
    color_continuous_scale="RdBu_r",
)
fig.show()
```

Heatmaps answer: "Which features are related to each other?"

---

## Kailash ModelVisualizer

```python
from kailash_ml import ModelVisualizer

viz = ModelVisualizer()
viz.configure(theme="professional", interactive=True)

# Distribution analysis
viz.plot_distribution(df, column="price", by="town")

# Correlation analysis
viz.plot_correlation(df, columns=numeric_cols)

# Time series
viz.plot_time_series(df, date_col="transaction_date", value_col="price")
```

`ModelVisualizer` provides ML-ready charts with consistent styling.

---

## ModelVisualizer: Multiple Plot Types

```python
viz = ModelVisualizer()

# Box plot comparison
viz.plot_box(df, x="flat_type", y="price")

# Pair plot (scatter matrix)
viz.plot_pairwise(df, columns=["price", "floor_area", "lease_years"])

# Feature importance (after model training -- preview for later)
viz.plot_feature_importance(model, feature_names=feature_cols)
```

We will use `ModelVisualizer` extensively in Modules 3-6 for model diagnostics.

---

## Visualization Best Practices

| Principle                | Example                                     |
| ------------------------ | ------------------------------------------- |
| Title states the insight | "Prices rose 15% in 2024" not "Price chart" |
| Label axes with units    | "Price ($)" not "price"                     |
| Use colour purposefully  | Colour = category, not decoration           |
| Remove clutter           | No 3D effects, no gridline overload         |
| Choose the right chart   | Bar for comparison, line for trend          |

---

## Avoiding Chart Crimes

```
CRIME: Truncated y-axis (starts at 400k instead of 0)
  → Makes small differences look dramatic

CRIME: Pie chart with 20 categories
  → Human eyes cannot compare many slices

CRIME: Dual y-axes with unrelated scales
  → Implies false correlation

CRIME: Using area/volume to represent single values
  → Our brains misjudge area relationships
```

---

## Saving Charts

```python
# Save as interactive HTML
fig.write_html("price_trend.html")

# Save as static image
fig.write_image("price_trend.png", width=800, height=500)

# In notebooks: inline display (automatic)
fig.show()
```

Local scripts should use `write_html()` for interactivity.

---

## Exercise Preview

**Exercise 1.6: HDB Visual Analysis Dashboard**

You will:

1. Create a histogram, box plot, scatter plot, and line chart
2. Use `ModelVisualizer` for automated distribution analysis
3. Build a correlation heatmap to identify related features
4. Apply best practices: titles, labels, colour coding

Scaffolding level: **Heavy (~70% code provided)**

---

## Common Pitfalls

| Mistake                                       | Fix                                           |
| --------------------------------------------- | --------------------------------------------- |
| Plotly needs pandas input                     | Use `df.to_pandas()` at the viz boundary only |
| Chart title describes data, not insight       | State what the viewer should see              |
| Too many categories in one chart              | Filter to top N or group into "Other"         |
| Forgetting to sort time series                | Always sort by date before line charts        |
| Using `ModelVisualizer` without `configure()` | Configure theme first for consistent styling  |

---

## Summary

- Plotly Express creates interactive charts quickly
- Choose chart type based on the question, not aesthetics
- `ModelVisualizer` provides ML-ready visualisations
- Good charts have clear titles, labelled axes, and purposeful colour
- Save as HTML for interactivity, PNG for reports

---

## Next Lesson

**Lesson 1.7: Automated Data Profiling**

We will learn:

- Using `DataExplorer` for comprehensive data profiling
- Configuring alerts with `AlertConfig`
- Automated data quality assessment
