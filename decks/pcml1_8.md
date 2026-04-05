---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 1.8: Data Cleaning Project

### Module 1: Data Pipelines and Visualisation

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Build an end-to-end data cleaning pipeline with `PreprocessingPipeline`
- Handle missing values with appropriate imputation strategies
- Detect and treat outliers systematically
- Combine all Module 1 skills into a capstone project

---

## Recap: Lesson 1.7

- `DataExplorer` automates comprehensive data profiling
- `AlertConfig` sets quality thresholds for automated checks
- Reports cover missing values, distributions, correlations, outliers
- Always profile before preprocessing

---

## Module 1 Journey

```
1.1 Variables, print, Polars basics
1.2 Filtering, selecting, transforming
1.3 Functions, loops, group_by/agg
1.4 Dictionaries, joins, conditionals
1.5 Window functions, rolling, lazy frames
1.6 Visualization with Plotly and ModelVisualizer
1.7 Automated profiling with DataExplorer
1.8 Cleaning pipeline with PreprocessingPipeline  ← YOU ARE HERE
```

Today we tie everything together.

---

## The Cleaning Challenge

Raw data is messy. Common problems:

| Problem                 | Example                                   |
| ----------------------- | ----------------------------------------- |
| Missing values          | `null` in lease_years column              |
| Wrong types             | Price stored as string "480,000"          |
| Outliers                | Floor area of 9999 sqm (data entry error) |
| Inconsistent categories | "TAMPINES", "Tampines", "tampines"        |
| Duplicates              | Same transaction recorded twice           |
| Invalid values          | Negative prices, future dates             |

---

## PreprocessingPipeline Overview

```python
from kailash_ml import PreprocessingPipeline

pipeline = PreprocessingPipeline()
pipeline.configure(
    dataset=df,
    target_column="price",
    steps=[
        "handle_missing",
        "remove_duplicates",
        "handle_outliers",
        "encode_categoricals",
        "scale_numerics",
    ],
)

cleaned_df = pipeline.run()
```

---

## Step 1: Handle Missing Values

```python
pipeline = PreprocessingPipeline()
pipeline.configure(
    dataset=df,
    target_column="price",
    missing_strategy={
        "numeric": "median",       # median imputation for numbers
        "categorical": "mode",     # most frequent for categories
        "lease_years": "forward_fill",  # time-based forward fill
    },
    missing_threshold=0.5,  # drop columns with >50% missing
)
```

---

## Imputation Strategies

| Strategy         | When to Use                                  |
| ---------------- | -------------------------------------------- |
| **median**       | Numeric with outliers (robust to extremes)   |
| **mean**         | Numeric, normally distributed                |
| **mode**         | Categorical (fill with most common value)    |
| **forward_fill** | Time series (carry last known value forward) |
| **constant**     | When a specific default makes domain sense   |
| **drop**         | When missing rows are few and random         |

The right strategy depends on **why** data is missing.

---

## Step 2: Remove Duplicates

```python
# Check for duplicates
dup_count = df.shape[0] - df.unique().shape[0]
print(f"Found {dup_count} duplicate rows")

# PreprocessingPipeline handles this automatically
pipeline.configure(
    dataset=df,
    target_column="price",
    remove_duplicates=True,
    duplicate_subset=["town", "block", "flat_type", "transaction_date"],
)
```

Specify which columns define a "duplicate" -- not always all columns.

---

## Step 3: Handle Outliers

```python
pipeline.configure(
    dataset=df,
    target_column="price",
    outlier_strategy="iqr",       # IQR-based detection
    outlier_action="clip",         # clip to bounds (not remove)
    outlier_columns=["price", "floor_area"],
)
```

| Action     | Effect                                |
| ---------- | ------------------------------------- |
| **clip**   | Cap extreme values at the threshold   |
| **remove** | Delete outlier rows entirely          |
| **flag**   | Add a boolean column marking outliers |

---

## IQR Outlier Detection (Visual)

```
                    IQR
            ┌───────────────┐
  ──────────┤               ├──────────
  Lower     Q1     Q2      Q3     Upper
  Fence                            Fence

  Lower Fence = Q1 - 1.5 * IQR
  Upper Fence = Q3 + 1.5 * IQR

  Values outside fences = outliers
```

Q1 = 25th percentile, Q2 = median, Q3 = 75th percentile

---

## Step 4: Standardise Categories

```python
# Before: inconsistent town names
# "TAMPINES", "Tampines", "tampines", "TAMPINES "

pipeline.configure(
    dataset=df,
    target_column="price",
    categorical_cleaning={
        "uppercase": ["town", "flat_type"],
        "strip_whitespace": True,
        "map_values": {
            "flat_type": {"3RM": "3 ROOM", "4RM": "4 ROOM"},
        },
    },
)
```

---

## Step 5: Encode and Scale

```python
pipeline.configure(
    dataset=df,
    target_column="price",
    encoding={
        "town": "one_hot",           # low cardinality → one-hot
        "flat_type": "ordinal",       # ordered categories → ordinal
    },
    scaling={
        "floor_area": "standard",     # mean=0, std=1
        "lease_years": "minmax",      # scale to [0, 1]
    },
)
```

Encoding and scaling prepare data for ML models (Module 3+).

---

## Complete Pipeline

```python
from kailash_ml import PreprocessingPipeline

pipeline = PreprocessingPipeline()
pipeline.configure(
    dataset=df,
    target_column="price",
    steps=[
        "handle_missing",
        "remove_duplicates",
        "handle_outliers",
        "encode_categoricals",
        "scale_numerics",
    ],
    missing_strategy={"numeric": "median", "categorical": "mode"},
    outlier_strategy="iqr",
    outlier_action="clip",
)

cleaned_df = pipeline.run()
print(f"Before: {df.shape} → After: {cleaned_df.shape}")
```

---

## Validating the Cleaning

Always compare before and after:

```python
from kailash_ml import DataExplorer

# Profile BEFORE
explorer_before = DataExplorer()
explorer_before.configure(dataset=df, target_column="price")
report_before = explorer_before.run()

# Profile AFTER
explorer_after = DataExplorer()
explorer_after.configure(dataset=cleaned_df, target_column="price")
report_after = explorer_after.run()

print(f"Missing before: {report_before.total_missing}")
print(f"Missing after:  {report_after.total_missing}")
print(f"Alerts before:  {len(report_before.alerts)}")
print(f"Alerts after:   {len(report_after.alerts)}")
```

---

## Pipeline Reproducibility

```python
# Save pipeline configuration for reuse
config = pipeline.get_config()

# Apply same cleaning to new data
new_df = loader.load("ascent01", "hdbprices_2025.csv")

pipeline_replay = PreprocessingPipeline()
pipeline_replay.configure(dataset=new_df, **config)
new_cleaned = pipeline_replay.run()
```

The same pipeline should be applied to training and test data identically.

---

## Capstone Project Overview

**Project: End-to-End HDB Data Pipeline**

```
Load raw data (ASCENTDataLoader)
    ↓
Profile (DataExplorer + AlertConfig)
    ↓
Clean (PreprocessingPipeline)
    ↓
Validate (DataExplorer again)
    ↓
Analyse (group_by, window functions, joins)
    ↓
Visualise (Plotly + ModelVisualizer)
    ↓
Report (formatted f-string summary)
```

---

## Exercise Preview

**Exercise 1.8: HDB Data Cleaning Capstone**

You will:

1. Load and profile the raw HDB dataset
2. Configure a `PreprocessingPipeline` with appropriate strategies
3. Clean missing values, duplicates, outliers, and categories
4. Validate the cleaned data with `DataExplorer`
5. Produce a visual comparison (before vs after)

Scaffolding level: **Heavy (~70% code provided)**

---

## Common Pitfalls

| Mistake                               | Fix                                             |
| ------------------------------------- | ----------------------------------------------- |
| Cleaning before profiling             | Always profile raw data first                   |
| Same imputation for all columns       | Different columns need different strategies     |
| Removing all outliers                 | Some outliers are valid -- use domain knowledge |
| Forgetting to validate after cleaning | Run `DataExplorer` on cleaned data too          |
| Different cleaning for train/test     | Save and replay the pipeline config             |
| Not documenting decisions             | Record why you chose each strategy              |

---

## Module 1 Summary

| Lesson | Key Skills                                   |
| ------ | -------------------------------------------- |
| 1.1    | Variables, print, Polars basics              |
| 1.2    | filter, select, with_columns, sort           |
| 1.3    | Functions, for loops, group_by/agg           |
| 1.4    | Dictionaries, joins, conditionals            |
| 1.5    | over(), rolling_mean(), shift(), lazy frames |
| 1.6    | Plotly charts, ModelVisualizer               |
| 1.7    | DataExplorer, AlertConfig                    |
| 1.8    | PreprocessingPipeline, capstone              |

---

## What Comes Next

**Module 2: Statistical Mastery**

- Bayesian thinking and probability distributions
- Hypothesis testing and bootstrap methods
- Experimental design (A/B testing, CUPED)
- Feature engineering with `FeatureEngineer` and `FeatureStore`

You now have the data skills. Next, we add the statistical reasoning.
