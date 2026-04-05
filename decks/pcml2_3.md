---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 2.3: Hypothesis Testing

### Module 2: Statistical Mastery

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Formulate null and alternative hypotheses correctly
- Apply the Neyman-Pearson framework for decision making
- Calculate and interpret statistical power
- Correct for multiple comparisons using Bonferroni and FDR methods

---

## Recap: Lesson 2.2

- MLE finds parameters that maximise the likelihood of data
- MAP adds prior knowledge, regularising small-sample estimates
- Full Bayesian inference gives entire posterior distributions
- MAP with Normal/Laplace priors equals L2/L1 regularisation

---

## The Hypothesis Testing Question

"Is this pattern real, or just noise?"

```
Observation: Average price in Tampines rose 8% this quarter.

Is this:
  A) A real market shift?       → take action
  B) Random fluctuation?        → do nothing
```

Hypothesis testing provides a **principled framework** for this decision.

---

## Null and Alternative Hypotheses

| Symbol | Name                   | Meaning                            |
| ------ | ---------------------- | ---------------------------------- |
| H₀     | Null hypothesis        | "Nothing interesting is happening" |
| H₁     | Alternative hypothesis | "Something real is going on"       |

```
H₀: Mean price this quarter = mean price last quarter
H₁: Mean price this quarter ≠ mean price last quarter
```

We assume H₀ is true and ask: "How surprising is the data?"

---

## The Neyman-Pearson Framework

```
                     Reality
                 H₀ True    H₀ False
Decision  ┌──────────────┬──────────────┐
Fail to   │  Correct     │  Type II     │
reject H₀ │  (TN)        │  Error (β)   │
          ├──────────────┼──────────────┤
Reject H₀ │  Type I      │  Correct     │
          │  Error (α)   │  (Power=1-β) │
          └──────────────┴──────────────┘
```

- **Type I error (α)**: False alarm -- seeing a pattern that is not there
- **Type II error (β)**: Missed detection -- missing a real pattern
- **Power (1-β)**: Probability of detecting a real effect

---

## P-Values: What They Are (and Are Not)

**P-value**: The probability of seeing data this extreme **if H₀ were true**.

```
P-value = 0.03
  → "If there were no real price change, there's a 3% chance
     of seeing a change this large by random chance alone."

P-value ≠ "3% chance H₀ is true"    (common misinterpretation!)
P-value ≠ "97% chance H₁ is true"   (also wrong!)
```

The p-value is about the **data**, not the hypothesis.

---

## Computing a T-Test

```python
import polars as pl
from scipy import stats
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent02", "hdbprices.csv")

# Compare prices: Tampines vs Bedok
tampines = df.filter(pl.col("town") == "TAMPINES")["price"].to_numpy()
bedok = df.filter(pl.col("town") == "BEDOK")["price"].to_numpy()

t_stat, p_value = stats.ttest_ind(tampines, bedok)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
```

---

## Decision Rule

```
Choose significance level α (typically 0.05)

If p-value < α:
  → Reject H₀ ("statistically significant")
  → Evidence supports a real difference

If p-value ≥ α:
  → Fail to reject H₀
  → Insufficient evidence (NOT "proven equal")
```

"Fail to reject" is not the same as "accept H₀". Absence of evidence is not evidence of absence.

---

## Effect Size: Practical Significance

Statistical significance does not mean practical importance.

```
Scenario: 100,000 transactions
  Mean difference: $500 (Tampines vs Bedok)
  P-value: 0.001 (highly significant!)

But $500 on a $450,000 flat = 0.1%
  → Statistically significant but practically meaningless
```

Always report **effect size** alongside p-values.

---

## Cohen's d: Standardised Effect Size

```
d = (mean₁ - mean₂) / pooled_std

  d ≈ 0.2  →  Small effect
  d ≈ 0.5  →  Medium effect
  d ≈ 0.8  →  Large effect
```

```python
import numpy as np

pooled_std = np.sqrt((tampines.std()**2 + bedok.std()**2) / 2)
cohens_d = (tampines.mean() - bedok.mean()) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")
```

---

## Statistical Power

**Power** = probability of detecting a real effect when one exists.

```
Power depends on:
  1. Effect size (bigger effect → easier to detect)
  2. Sample size (more data → more power)
  3. Significance level α (higher α → more power, but more false alarms)
  4. Variance (less noise → more power)

Minimum target: power ≥ 0.80 (80%)
```

---

## Sample Size Planning

```python
from scipy.stats import norm
import numpy as np

def required_sample_size(effect_size, alpha=0.05, power=0.80):
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

# How many sales do we need to detect a $10k difference?
effect = 10_000 / 80_000  # $10k difference, $80k std dev
n = required_sample_size(effect)
print(f"Need {n} observations per group")
```

Always plan sample size **before** collecting data.

---

## The Multiple Testing Problem

Testing many hypotheses inflates false positives.

```
Test 1 town pair:   5% false positive rate
Test 10 town pairs: 1 - (0.95)^10 = 40% chance of at least one false positive!
Test 26 towns (all pairs): almost guaranteed false positives
```

If you test enough things, you will find "significant" results by chance.

---

## Bonferroni Correction

The simplest fix: divide α by the number of tests.

```python
n_tests = 10
alpha = 0.05
alpha_corrected = alpha / n_tests  # 0.005

# Apply to each test
for town_pair, p_val in results:
    if p_val < alpha_corrected:
        print(f"{town_pair}: significant (p={p_val:.4f})")
```

**Pros**: Simple, conservative, controls family-wise error rate
**Cons**: Very conservative -- may miss real effects

---

## False Discovery Rate (FDR)

Benjamini-Hochberg procedure: controls the **proportion** of false discoveries.

```python
from scipy.stats import false_discovery_control

p_values = [0.001, 0.008, 0.039, 0.041, 0.15, 0.23, 0.51]

# Returns adjusted p-values
adjusted = false_discovery_control(p_values, method='bh')

for i, (p, adj_p) in enumerate(zip(p_values, adjusted)):
    sig = "***" if adj_p < 0.05 else ""
    print(f"Test {i+1}: p={p:.3f}, adjusted={adj_p:.3f} {sig}")
```

FDR is less conservative than Bonferroni -- preferred in exploratory analysis.

---

## Choosing the Right Test

| Scenario                      | Test                                 |
| ----------------------------- | ------------------------------------ |
| Compare two group means       | t-test (independent samples)         |
| Compare before/after (paired) | Paired t-test                        |
| Compare 3+ group means        | ANOVA (then post-hoc)                |
| Compare proportions           | Chi-squared / Z-test for proportions |
| Non-normal data               | Mann-Whitney U / Wilcoxon            |
| Correlation significance      | Pearson/Spearman correlation test    |

---

## Full Example: Town Price Comparison

```python
import polars as pl
from scipy import stats
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent02", "hdbprices.csv")

towns = df["town"].unique().sort().to_list()[:5]
results = []

for i, town_a in enumerate(towns):
    for town_b in towns[i+1:]:
        a = df.filter(pl.col("town") == town_a)["price"].to_numpy()
        b = df.filter(pl.col("town") == town_b)["price"].to_numpy()
        _, p = stats.ttest_ind(a, b)
        results.append({"pair": f"{town_a} vs {town_b}", "p": p})

# Apply FDR correction
p_vals = [r["p"] for r in results]
adjusted = false_discovery_control(p_vals, method='bh')
```

---

## Exercise Preview

**Exercise 2.3: HDB Market Hypothesis Testing**

You will:

1. Formulate and test hypotheses about town price differences
2. Calculate effect sizes alongside p-values
3. Perform power analysis for sample size planning
4. Apply Bonferroni and FDR corrections to multiple comparisons

Scaffolding level: **Moderate+ (~60% code provided)**

---

## Common Pitfalls

| Mistake                                    | Fix                                            |
| ------------------------------------------ | ---------------------------------------------- |
| "p < 0.05 means the effect is real"        | P-value is about data, not truth               |
| Ignoring effect size                       | Small p + small effect = unimportant finding   |
| Testing many hypotheses without correction | Use Bonferroni or FDR                          |
| "Fail to reject" = "no effect"             | Absence of evidence is not evidence of absence |
| Choosing the test after seeing data        | Pre-register your analysis plan                |

---

## Summary

- Hypothesis testing asks: "Is this pattern real or noise?"
- Neyman-Pearson framework: Type I/II errors, power, significance
- P-values measure data surprise under H₀, not probability of truth
- Always report effect size alongside significance
- Multiple testing requires correction (Bonferroni conservative, FDR balanced)

---

## Next Lesson

**Lesson 2.4: Bootstrap and Resampling**

We will learn:

- Bootstrap resampling for non-parametric inference
- BCa confidence intervals
- When bootstrapping beats parametric methods
