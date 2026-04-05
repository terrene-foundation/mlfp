---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 2.5: CUPED and Variance Reduction

### Module 2: Statistical Mastery

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why variance reduction matters for experiments
- Implement CUPED using pre-experiment data
- Apply stratified sampling for balanced experiments
- Design A/B tests with maximum statistical power

---

## Recap: Lesson 2.4

- Bootstrap resamples with replacement to estimate variability
- BCa intervals correct for bias and skewness
- Bootstrap works for any statistic without distributional assumptions
- Permutation tests provide non-parametric hypothesis testing

---

## The Variance Problem in A/B Testing

```
Experiment: Does a new HDB listing format increase clicks?

Control group mean:    245 clicks/day
Treatment group mean:  258 clicks/day

Difference: +13 clicks (+5.3%)

But daily variance is huge:
  Control std:   120 clicks
  Treatment std: 115 clicks
  → p-value = 0.18 (not significant!)
```

High variance drowns out real effects. We need **variance reduction**.

---

## Three Variance Reduction Strategies

```
1. CUPED (Controlled-experiment Using Pre-Experiment Data)
   → Use historical data to remove predictable variation

2. Stratified Randomisation
   → Balance known sources of variation across groups

3. Regression Adjustment
   → Control for covariates in the analysis stage
```

All three reduce noise without increasing sample size.

---

## CUPED: The Core Idea

If you know what a user **would have done** without the treatment, subtract it out.

```
Y_cuped = Y - θ · (X - E[X])

Where:
  Y     = observed outcome during experiment
  X     = pre-experiment outcome (same metric, earlier period)
  θ     = Cov(Y, X) / Var(X)
  E[X]  = mean of pre-experiment values
```

CUPED removes the **predictable** component of variation.

---

## CUPED Visual Intuition

```
Without CUPED:                 With CUPED:
  Outcome                       Adjusted Outcome
    |  · ·  · · ·                 |  · · ·
    | · ·  ·  · ·                 | · · · ·
    |·  · ·  · · ·               |· · · · ·     ← less spread
    |  · ·  ·  · ·                | · · · ·
    └───────────────              └───────────────
    High variance                 Reduced variance
    (hard to detect effect)       (easier to detect effect)
```

Variance reduction = 1 - r^2, where r = correlation(Y, X).

---

## CUPED Implementation

```python
import numpy as np
import polars as pl

def apply_cuped(df, outcome_col, pre_experiment_col):
    y = df[outcome_col].to_numpy().astype(float)
    x = df[pre_experiment_col].to_numpy().astype(float)

    theta = np.cov(y, x)[0, 1] / np.var(x)
    y_cuped = y - theta * (x - np.mean(x))

    return df.with_columns(
        pl.Series(name=f"{outcome_col}_cuped", values=y_cuped)
    )

# Apply CUPED
df_exp = apply_cuped(df_exp, "clicks", "pre_clicks")
```

---

## CUPED Variance Reduction

```python
# How much variance did CUPED remove?
var_original = df_exp["clicks"].var()
var_cuped = df_exp["clicks_cuped"].var()

reduction = 1 - var_cuped / var_original
print(f"Variance reduction: {reduction:.1%}")

# Correlation between pre and post determines reduction
r = np.corrcoef(
    df_exp["clicks"].to_numpy(),
    df_exp["pre_clicks"].to_numpy()
)[0, 1]
print(f"Correlation: {r:.3f}")
print(f"Theoretical reduction: {r**2:.1%}")
```

If pre-experiment data correlates 0.7 with post, CUPED removes ~49% of variance.

---

## When CUPED Works Well

| Scenario                | Correlation | Variance Reduction |
| ----------------------- | ----------- | ------------------ |
| Same metric, last week  | r = 0.8     | ~64%               |
| Same metric, last month | r = 0.6     | ~36%               |
| Related metric          | r = 0.4     | ~16%               |
| Unrelated metric        | r = 0.1     | ~1%                |

CUPED is most powerful when pre-experiment behaviour strongly predicts post-experiment behaviour.

---

## Stratified Randomisation

Ensure treatment/control groups are balanced on known factors.

```python
# Without stratification: random assignment
df_exp = df.with_columns(
    pl.Series("group", np.random.choice(["control", "treatment"], len(df)))
)

# With stratification: balanced within each stratum
def stratified_assign(df, strata_cols):
    assignments = []
    for _, group_df in df.group_by(strata_cols):
        n = len(group_df)
        half = n // 2
        assign = ["control"] * half + ["treatment"] * (n - half)
        np.random.shuffle(assign)
        assignments.extend(assign)
    return df.with_columns(pl.Series("group", assignments))

df_exp = stratified_assign(df, ["town", "flat_type"])
```

---

## Why Stratification Helps

```
Without stratification:
  Control: 60% Tampines, 40% Bedok    (imbalanced!)
  Treatment: 40% Tampines, 60% Bedok
  → Town differences confound the treatment effect

With stratification:
  Control: 50% Tampines, 50% Bedok    (balanced!)
  Treatment: 50% Tampines, 50% Bedok
  → Town effects cancel out
```

Stratify on variables that explain outcome variance.

---

## Regression Adjustment

Control for covariates in the analysis, not the randomisation.

```python
import numpy as np
from scipy import stats

# Simple regression adjustment
# Y_adj = Y - β₁·X₁ - β₂·X₂ + constants

# Using Polars for the adjustment
df_analysis = df_exp.with_columns(
    # De-mean covariates
    (pl.col("floor_area") - pl.col("floor_area").mean()).alias("floor_area_dm"),
    (pl.col("lease_years") - pl.col("lease_years").mean()).alias("lease_dm"),
)

# The treatment effect estimate is the same,
# but the standard error shrinks
```

---

## Combining Strategies

```
Maximum power experiment design:

1. Stratified randomisation
   → Balance groups on known confounders

2. CUPED adjustment
   → Remove predictable variation using pre-experiment data

3. Regression adjustment
   → Control for remaining covariates

Each layer reduces residual variance further.
```

---

## Sample Size with Variance Reduction

```python
def sample_size_with_cuped(effect_size, base_variance, r_squared,
                            alpha=0.05, power=0.80):
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)

    adjusted_variance = base_variance * (1 - r_squared)
    n = 2 * adjusted_variance * ((z_a + z_b) / effect_size) ** 2
    return int(np.ceil(n))

# Without CUPED
n_base = sample_size_with_cuped(10, 10000, r_squared=0)
# With CUPED (r=0.7)
n_cuped = sample_size_with_cuped(10, 10000, r_squared=0.49)

print(f"Without CUPED: {n_base} per group")
print(f"With CUPED:    {n_cuped} per group")  # ~51% fewer
```

---

## A/B Test Design Checklist

```
□ Define primary metric and minimum detectable effect
□ Calculate required sample size (with variance reduction)
□ Choose stratification variables
□ Identify pre-experiment covariate for CUPED
□ Set experiment duration (account for day-of-week effects)
□ Pre-register analysis plan (avoid p-hacking)
□ Define stopping rules (see Lesson 2.6: sequential testing)
```

---

## Full Example: HDB Listing Experiment

```python
import polars as pl
import numpy as np
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent02", "listing_experiment.csv")

# Step 1: CUPED adjustment
df = apply_cuped(df, "views", "pre_views")

# Step 2: Compare groups
control = df.filter(pl.col("group") == "control")
treatment = df.filter(pl.col("group") == "treatment")

# Step 3: Effect estimate with reduced variance
diff_raw = treatment["views"].mean() - control["views"].mean()
diff_cuped = treatment["views_cuped"].mean() - control["views_cuped"].mean()

print(f"Raw difference:   {diff_raw:.1f} (SE: {se_raw:.1f})")
print(f"CUPED difference: {diff_cuped:.1f} (SE: {se_cuped:.1f})")
```

---

## Exercise Preview

**Exercise 2.5: Variance Reduction for A/B Testing**

You will:

1. Implement CUPED from scratch and measure variance reduction
2. Apply stratified randomisation and verify balance
3. Compare standard errors with and without CUPED
4. Design a sample-size-efficient experiment

Scaffolding level: **Moderate+ (~60% code provided)**

---

## Common Pitfalls

| Mistake                              | Fix                                           |
| ------------------------------------ | --------------------------------------------- |
| Using post-experiment data for CUPED | Only pre-experiment data is valid             |
| Stratifying on too many variables    | 2-3 strata max; too many = tiny cells         |
| Forgetting that CUPED preserves ATE  | The point estimate does not change, only SE   |
| Ignoring day-of-week effects         | Run experiments in full-week multiples        |
| Not pre-registering the analysis     | Decide CUPED/covariates before seeing results |

---

## Summary

- High variance makes it hard to detect real experimental effects
- CUPED uses pre-experiment data to remove predictable variation
- Variance reduction = r-squared between pre and post metrics
- Stratified randomisation balances groups on known confounders
- Combining strategies can halve required sample sizes

---

## Next Lesson

**Lesson 2.6: Sequential Testing and Causal Inference**

We will learn:

- Sequential testing: when to stop experiments early
- Difference-in-Differences (DiD)
- Propensity score matching for observational studies
