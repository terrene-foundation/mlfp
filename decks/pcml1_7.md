---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 1.7: Automated Data Profiling

### Module 1: Data Pipelines and Visualisation

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Use `DataExplorer` for comprehensive automated data profiling
- Configure quality alerts with `AlertConfig`
- Interpret profiling reports to identify data issues
- Build a systematic data quality assessment workflow

---

## Recap: Lesson 1.6

- Plotly Express creates interactive charts (bar, histogram, scatter, line)
- Choose chart type based on analytical question
- `ModelVisualizer` provides ML-ready visualisations
- Good charts have clear titles, labels, and purposeful colour

---

## The Data Quality Problem

Before building any model, you must understand your data.

```
Manual profiling:
  df.describe()         → summary stats
  df.null_count()       → missing values
  df.n_unique()         → cardinality per column
  custom plots          → distributions
  ... repeat for every column, every dataset
```

This is tedious and error-prone. `DataExplorer` automates it.

---

## DataExplorer: One Engine, Full Profile

```python
from kailash_ml import DataExplorer

explorer = DataExplorer()
explorer.configure(
    dataset=df,
    target_column="price",
)

report = explorer.run()
```

One call profiles **every column**: types, distributions, missing values, correlations, outliers, and more.

---

## What DataExplorer Produces

| Section            | What It Tells You                                 |
| ------------------ | ------------------------------------------------- |
| **Overview**       | Row count, column count, memory usage             |
| **Type Analysis**  | Data types, mixed-type warnings                   |
| **Missing Values** | Count, percentage, pattern (random vs systematic) |
| **Distributions**  | Histograms, skewness, kurtosis per numeric column |
| **Correlations**   | Pearson, Spearman between all numeric pairs       |
| **Outliers**       | IQR-based detection, z-score flags                |
| **Cardinality**    | Unique values per categorical column              |

---

## Configuring DataExplorer

```python
explorer = DataExplorer()
explorer.configure(
    dataset=df,
    target_column="price",

    # Control what gets computed
    correlations=True,
    outlier_detection=True,
    distribution_analysis=True,

    # Performance tuning for large datasets
    sample_size=10_000,
)
```

---

## Reading the Report: Missing Values

```python
missing = report.missing_values

print(missing)
```

```
┌──────────────┬───────┬────────────┐
│ column       ┆ count ┆ percentage │
│ lease_years  ┆ 45    ┆ 0.36%      │
│ storey_range ┆ 12    ┆ 0.10%      │
│ floor_area   ┆ 0     ┆ 0.00%      │
└──────────────┴───────┴────────────┘
```

Key question: Is missing data **random** or **systematic**?

- Random: safe to impute
- Systematic: investigate the cause first

---

## Reading the Report: Distributions

```python
dist = report.distributions

# Check skewness
for col_name, stats in dist.items():
    if abs(stats["skewness"]) > 1:
        print(f"WARNING: {col_name} is highly skewed ({stats['skewness']:.2f})")
```

Skewed features may need transformation (log, square root) before modelling.

---

## Reading the Report: Correlations

```python
corr = report.correlations

# Find strong correlations with target
target_corr = corr.filter(pl.col("column") == "price")
print(target_corr.sort("pearson", descending=True))
```

```
┌──────────────┬─────────┐
│ column       ┆ pearson │
│ floor_area   ┆ 0.82    │  ← strong positive
│ lease_years  ┆ -0.35   │  ← moderate negative
│ storey       ┆ 0.28    │  ← weak positive
└──────────────┴─────────┘
```

---

## Reading the Report: Outliers

```python
outliers = report.outliers

print(outliers)
```

```
┌──────────────┬───────┬───────────┬───────────┐
│ column       ┆ count ┆ lower_pct ┆ upper_pct │
│ price        ┆ 234   ┆ 0.5%      ┆ 1.4%      │
│ floor_area   ┆ 18    ┆ 0.0%      ┆ 0.1%      │
└──────────────┴───────┴───────────┴───────────┘
```

Outliers are not always errors -- a $1.2M HDB flat is unusual but real.

---

## AlertConfig: Automated Quality Checks

```python
from kailash_ml import DataExplorer, AlertConfig

alerts = AlertConfig(
    missing_threshold=0.05,      # flag columns with >5% missing
    correlation_threshold=0.95,  # flag near-duplicate features
    outlier_threshold=0.02,      # flag >2% outliers
    cardinality_threshold=0.9,   # flag near-unique categoricals
)

explorer = DataExplorer()
explorer.configure(
    dataset=df,
    target_column="price",
    alerts=alerts,
)

report = explorer.run()
```

---

## Interpreting Alerts

```python
for alert in report.alerts:
    print(f"[{alert.severity}] {alert.column}: {alert.message}")
```

```
[WARNING] lease_years: 0.36% missing values detected
[CRITICAL] price_sqm, price: correlation 0.97 (near duplicate)
[INFO] town: 26 unique values (high cardinality categorical)
```

Severity levels:

- **CRITICAL** -- must fix before modelling
- **WARNING** -- investigate, may need action
- **INFO** -- awareness only

---

## Data Quality Workflow

```
1. Load data
   └→ ASCENTDataLoader

2. Profile data
   └→ DataExplorer with AlertConfig

3. Review alerts
   └→ Fix CRITICAL issues
   └→ Investigate WARNING issues

4. Visualise concerns
   └→ ModelVisualizer for distributions, outliers

5. Document decisions
   └→ "Kept outliers because..." / "Imputed because..."
```

---

## Cardinality Analysis

```python
# High cardinality = many unique values
# Important for choosing encoding strategies later

card = report.cardinality

print(card)
```

```
┌──────────────┬──────────┬────────────┐
│ column       ┆ n_unique ┆ pct_unique │
│ town         ┆ 26       ┆ 0.21%      │  ← good categorical
│ block        ┆ 2,341    ┆ 18.9%      │  ← too many for one-hot
│ street_name  ┆ 543      ┆ 4.4%       │  ← consider grouping
└──────────────┴──────────┴────────────┘
```

---

## Full Example

```python
import polars as pl
from shared.data_loader import ASCENTDataLoader
from kailash_ml import DataExplorer, AlertConfig

loader = ASCENTDataLoader()
df = loader.load("ascent01", "hdbprices.csv")

alerts = AlertConfig(
    missing_threshold=0.05,
    correlation_threshold=0.95,
)

explorer = DataExplorer()
explorer.configure(dataset=df, target_column="price", alerts=alerts)
report = explorer.run()

# Print summary
print(f"Dataset: {report.overview.rows:,} rows, {report.overview.columns} columns")
print(f"Alerts: {len(report.alerts)}")
for alert in report.alerts:
    print(f"  [{alert.severity}] {alert.column}: {alert.message}")
```

---

## Exercise Preview

**Exercise 1.7: Automated Data Quality Audit**

You will:

1. Run `DataExplorer` on the HDB dataset with custom `AlertConfig`
2. Interpret missing value patterns and distributions
3. Identify correlated features and outliers
4. Write a structured data quality report with recommended actions

Scaffolding level: **Heavy (~70% code provided)**

---

## Common Pitfalls

| Mistake                                | Fix                                                        |
| -------------------------------------- | ---------------------------------------------------------- |
| Skipping profiling ("data looks fine") | Always profile -- hidden issues break models               |
| Treating all outliers as errors        | Some outliers are valid extreme values                     |
| Ignoring missing value patterns        | Random vs systematic missing requires different strategies |
| Setting alert thresholds too tight     | Start permissive, tighten as you learn the data            |
| Profiling after preprocessing          | Profile raw data first, then again after cleaning          |

---

## Summary

- `DataExplorer` automates comprehensive data profiling
- `AlertConfig` sets quality thresholds for automated checking
- Reports cover missing values, distributions, correlations, outliers, cardinality
- Always profile **before** preprocessing or modelling
- Document your data quality decisions for reproducibility

---

## Next Lesson

**Lesson 1.8: Data Cleaning Project**

We will learn:

- Building a complete cleaning pipeline with `PreprocessingPipeline`
- Handling missing values, outliers, and type conversions
- Putting everything from Module 1 together in a capstone project
