---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 2.4: Bootstrap and Resampling

### Module 2: Statistical Mastery

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain the bootstrap principle and when it applies
- Implement bootstrap resampling to estimate sampling distributions
- Compute BCa (bias-corrected and accelerated) confidence intervals
- Compare bootstrap intervals to parametric intervals

---

## Recap: Lesson 2.3

- Hypothesis testing: null vs alternative, p-values, decision rules
- Effect size matters as much as statistical significance
- Power analysis determines required sample size
- Multiple testing requires Bonferroni or FDR correction

---

## The Core Problem

We want to know how **uncertain** our estimates are.

```
Sample mean price: $465,000

But how much would this vary if we collected a different sample?
  → We cannot re-run the data collection
  → Parametric methods assume a distribution shape
  → What if we do not know the shape?
```

The bootstrap: **resample from what you have** to estimate variability.

---

## The Bootstrap Principle

```
Original sample (n observations):
  [480k, 350k, 520k, 410k, 390k, 450k, 380k]

Bootstrap sample 1: (sample n WITH replacement)
  [410k, 480k, 480k, 350k, 390k, 520k, 410k]  → mean = 434k

Bootstrap sample 2:
  [350k, 450k, 390k, 390k, 520k, 380k, 480k]  → mean = 423k

Bootstrap sample 3:
  [520k, 480k, 410k, 350k, 450k, 450k, 390k]  → mean = 436k

... repeat 10,000 times → distribution of means
```

---

## Why "With Replacement"?

```
Without replacement: you get the same sample every time
  → No variability, no information

With replacement: some values appear 0 times, some 2+ times
  → Each bootstrap sample is a plausible "alternate universe"
  → The variation across samples estimates sampling uncertainty
```

On average, each bootstrap sample contains ~63.2% of unique original observations.

---

## Bootstrap in Code

```python
import numpy as np
import polars as pl
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent02", "hdbprices.csv")

prices = df.filter(pl.col("town") == "TAMPINES")["price"].to_numpy()

n_bootstrap = 10_000
boot_means = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    sample = np.random.choice(prices, size=len(prices), replace=True)
    boot_means[i] = sample.mean()

print(f"Bootstrap mean: ${boot_means.mean():,.0f}")
print(f"Bootstrap SE:   ${boot_means.std():,.0f}")
```

---

## Percentile Confidence Interval

The simplest bootstrap interval: take percentiles of the bootstrap distribution.

```python
lower = np.percentile(boot_means, 2.5)
upper = np.percentile(boot_means, 97.5)

print(f"95% CI: [${lower:,.0f}, ${upper:,.0f}]")
```

```
Bootstrap distribution of means:

  Count
    |
    |        ╱╲
    |       ╱  ╲
    |      ╱    ╲
    |    ╱        ╲
    |──╱────────────╲──
    └──┤────────────┤──→  Mean price
       2.5%      97.5%
       lower     upper
```

---

## BCa Intervals: Better Than Percentile

BCa (Bias-Corrected and Accelerated) intervals correct for:

1. **Bias**: the bootstrap distribution may not be centred on the estimate
2. **Skewness**: the distribution may be asymmetric

```python
from scipy.stats import bootstrap

# Scipy's bootstrap with BCa
result = bootstrap(
    (prices,),
    statistic=np.mean,
    n_resamples=10_000,
    confidence_level=0.95,
    method='BCa',
)

print(f"BCa 95% CI: [${result.confidence_interval.low:,.0f}, "
      f"${result.confidence_interval.high:,.0f}]")
```

---

## BCa vs Percentile: When It Matters

```
Symmetric data:
  Percentile CI:  [440k, 490k]
  BCa CI:         [441k, 491k]     ← similar

Skewed data (e.g., price distribution):
  Percentile CI:  [440k, 520k]
  BCa CI:         [445k, 530k]     ← adjusted for skew

BCa is always at least as good as percentile.
Use BCa as your default.
```

---

## Bootstrap for Any Statistic

The bootstrap works for **any** function of the data, not just the mean.

```python
# Bootstrap the median
boot_medians = np.array([
    np.median(np.random.choice(prices, size=len(prices), replace=True))
    for _ in range(10_000)
])

# Bootstrap the 90th percentile
boot_p90 = np.array([
    np.percentile(np.random.choice(prices, size=len(prices), replace=True), 90)
    for _ in range(10_000)
])

# Bootstrap the correlation
floor_areas = df.filter(pl.col("town") == "TAMPINES")["floor_area"].to_numpy()
boot_corr = np.array([
    np.corrcoef(
        np.random.choice(len(prices), size=len(prices), replace=True)
        .take(prices), ...  # simplified
    )[0, 1]
    for _ in range(10_000)
])
```

---

## Bootstrap for Difference in Means

```python
tampines = df.filter(pl.col("town") == "TAMPINES")["price"].to_numpy()
bedok = df.filter(pl.col("town") == "BEDOK")["price"].to_numpy()

boot_diffs = np.zeros(10_000)
for i in range(10_000):
    samp_t = np.random.choice(tampines, size=len(tampines), replace=True)
    samp_b = np.random.choice(bedok, size=len(bedok), replace=True)
    boot_diffs[i] = samp_t.mean() - samp_b.mean()

# If 0 is outside the CI, the difference is "significant"
lower, upper = np.percentile(boot_diffs, [2.5, 97.5])
print(f"Difference CI: [${lower:,.0f}, ${upper:,.0f}]")
print(f"Contains zero: {lower <= 0 <= upper}")
```

---

## When Bootstrap Beats Parametric Methods

| Scenario                              | Parametric       | Bootstrap         |
| ------------------------------------- | ---------------- | ----------------- |
| Normal data, large n                  | Exact formulas   | Equivalent        |
| Skewed data                           | Approximate      | More accurate     |
| Small sample (n < 30)                 | Unreliable       | Still works       |
| Complex statistic (ratio, percentile) | Often no formula | Always works      |
| Unknown distribution                  | Must assume      | Distribution-free |

Bootstrap is the **Swiss army knife** of inference.

---

## When Bootstrap Fails

- **Extremely small samples** (n < 5): not enough to resample meaningfully
- **Heavy-tailed distributions**: bootstrap variance can be unstable
- **Dependent data**: standard bootstrap assumes independence
  - Use block bootstrap for time series
- **Extrapolation**: bootstrap cannot go beyond observed data range
- **Population parameters at boundaries**: e.g., maximum of a uniform

---

## Bootstrap Hypothesis Test

```python
# Test: is mean price in Tampines > mean price in Bedok?
observed_diff = tampines.mean() - bedok.mean()

# Combine samples under H₀ (no difference)
combined = np.concatenate([tampines, bedok])
n_t, n_b = len(tampines), len(bedok)

null_diffs = np.zeros(10_000)
for i in range(10_000):
    perm = np.random.permutation(combined)
    null_diffs[i] = perm[:n_t].mean() - perm[n_t:].mean()

# P-value: proportion of null diffs ≥ observed
p_value = (null_diffs >= observed_diff).mean()
print(f"Bootstrap p-value: {p_value:.4f}")
```

This is actually a **permutation test** -- a close relative of bootstrapping.

---

## Visualising Bootstrap Results

```python
from kailash_ml import ModelVisualizer

viz = ModelVisualizer()
viz.configure(theme="professional")

# Histogram of bootstrap distribution
viz.plot_distribution(
    data=boot_means,
    title="Bootstrap Distribution of Mean Price",
    vlines=[lower, upper],  # CI boundaries
    vline_labels=["2.5%", "97.5%"],
)
```

---

## Exercise Preview

**Exercise 2.4: Bootstrap Analysis of HDB Markets**

You will:

1. Bootstrap the mean, median, and price-per-sqm for multiple towns
2. Compute BCa confidence intervals using scipy
3. Test whether price differences between towns are significant
4. Compare bootstrap CIs to parametric CIs and discuss differences

Scaffolding level: **Moderate+ (~60% code provided)**

---

## Common Pitfalls

| Mistake                                 | Fix                                      |
| --------------------------------------- | ---------------------------------------- |
| Sampling without replacement            | Must be WITH replacement                 |
| Too few bootstrap resamples             | Use at least 10,000 (not 100)            |
| Using percentile CI when data is skewed | Use BCa instead                          |
| Bootstrapping dependent data            | Use block bootstrap for time series      |
| Forgetting to set random seed           | `np.random.seed(42)` for reproducibility |

---

## Summary

- Bootstrap resamples **with replacement** to estimate sampling variability
- Works for any statistic: mean, median, correlation, percentiles
- BCa intervals correct for bias and skewness -- use as default
- Bootstrap tests compare observed statistics to null distribution
- No distributional assumptions needed -- truly non-parametric

---

## Next Lesson

**Lesson 2.5: CUPED and Variance Reduction**

We will learn:

- CUPED (Controlled-experiment Using Pre-Experiment Data)
- Variance reduction techniques for A/B testing
- Designing experiments for maximum statistical power
