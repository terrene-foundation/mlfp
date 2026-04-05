---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 2.1: Bayesian Thinking

### Module 2: Statistical Mastery

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Distinguish between frequentist and Bayesian approaches
- Describe common probability distributions and when to use them
- Apply Bayes' theorem to update beliefs with evidence
- Explain conjugate priors and compute posterior distributions

---

## Recap: Module 1

- Loaded, filtered, transformed, and joined data with Polars
- Wrote functions, loops, and aggregations
- Profiled data with `DataExplorer` and cleaned it with `PreprocessingPipeline`
- Visualised insights with Plotly and `ModelVisualizer`

Module 2 adds the **statistical reasoning** behind data analysis.

---

## Two Schools of Statistics

| Aspect                             | Frequentist         | Bayesian            |
| ---------------------------------- | ------------------- | ------------------- |
| Probability means                  | Long-run frequency  | Degree of belief    |
| Parameters are                     | Fixed but unknown   | Random variables    |
| Prior knowledge                    | Ignored             | Explicitly included |
| Answer to "what's the mean price?" | Point estimate + CI | Full distribution   |

Neither is "right" -- they answer different questions.

---

## Why Bayesian Thinking Matters for ML

- **Small data**: Bayesian methods shine when you have limited observations
- **Prior knowledge**: Incorporate domain expertise before seeing data
- **Uncertainty quantification**: Get full distributions, not just point estimates
- **Sequential updating**: Update beliefs as new data arrives
- **Regularisation**: Priors act as natural regularisers (Module 3 connection)

---

## Probability Distributions: The Building Blocks

```
Discrete (countable outcomes):
  Bernoulli   → coin flip (0 or 1)
  Binomial    → number of heads in n flips
  Poisson     → count of rare events per interval

Continuous (any value in a range):
  Normal      → bell curve, most natural phenomena
  Exponential → time between events
  Beta        → probability of probability (0 to 1)
  Gamma       → positive continuous values
```

---

## The Normal Distribution

```
         ┌────┐
        ╱│    │╲
       ╱ │    │ ╲
      ╱  │    │  ╲
     ╱   │    │   ╲
   ╱     │    │     ╲
──╱──────┤    ├──────╲──
  -2σ   -1σ   μ   +1σ   +2σ

  68% within ±1σ
  95% within ±2σ
  99.7% within ±3σ
```

Parameters: mean (μ) = centre, std dev (σ) = spread.

---

## The Beta Distribution

The Beta distribution lives on [0, 1] -- perfect for modelling probabilities.

```python
import polars as pl
import numpy as np

# Beta(α, β) — α successes, β failures (loosely)
# Beta(1, 1)  → uniform (no prior knowledge)
# Beta(10, 2) → strong belief probability is high (~0.83)
# Beta(2, 10) → strong belief probability is low (~0.17)
```

```
Beta(1,1):  ─────────────  (flat / uniform)
Beta(2,5):  ╱╲              (skewed left)
Beta(5,2):       ╱╲         (skewed right)
Beta(10,10):   ╱╲           (peaked at 0.5)
```

---

## Bayes' Theorem

$$P(H|D) = \frac{P(D|H) \cdot P(H)}{P(D)}$$

| Term    | Name           | Meaning                                       |
| ------- | -------------- | --------------------------------------------- |
| P(H\|D) | **Posterior**  | Updated belief after seeing data              |
| P(D\|H) | **Likelihood** | How probable is the data given the hypothesis |
| P(H)    | **Prior**      | Belief before seeing data                     |
| P(D)    | **Evidence**   | Total probability of the data                 |

---

## Bayes' Theorem Example

**Question**: A flat sold for over $1M. What's the probability it's in Bukit Timah?

```
Prior:      P(BT) = 0.05        (5% of all flats are in BT)
Likelihood: P(>1M | BT) = 0.30  (30% of BT flats exceed 1M)
Evidence:   P(>1M) = 0.02       (2% of all flats exceed 1M)

Posterior:  P(BT | >1M) = (0.30 × 0.05) / 0.02
                        = 0.015 / 0.02
                        = 0.75
```

After seeing the $1M price, our belief that it is Bukit Timah jumps from 5% to 75%.

---

## Prior, Likelihood, Posterior (Visual)

```
Belief
  |
  |  Prior          Posterior
  |   ╱╲              ╱╲
  |  ╱  ╲    +Data → ╱  ╲     (narrower, shifted)
  | ╱    ╲          ╱    ╲
  |╱      ╲        ╱      ╲
  └────────────────────────→ Parameter value

  More data → narrower posterior → more certainty
  Strong prior → posterior closer to prior
  Weak prior → posterior dominated by data
```

---

## Conjugate Priors

A prior is **conjugate** to a likelihood if the posterior has the same distributional form.

| Likelihood         | Conjugate Prior | Posterior                         |
| ------------------ | --------------- | --------------------------------- |
| Bernoulli/Binomial | Beta(α, β)      | Beta(α + successes, β + failures) |
| Poisson            | Gamma(α, β)     | Gamma(α + Σx, β + n)              |
| Normal (known σ)   | Normal(μ₀, σ₀)  | Normal(updated μ, updated σ)      |

Conjugate priors make computation tractable -- no numerical integration needed.

---

## Beta-Binomial Example

**Question**: What fraction of 4-ROOM flats sell above median price?

```python
# Prior: no strong belief → Beta(1, 1) = uniform
alpha_prior, beta_prior = 1, 1

# Data: observed 73 above median out of 100 sales
successes, failures = 73, 27

# Posterior: Beta(1 + 73, 1 + 27) = Beta(74, 28)
alpha_post = alpha_prior + successes    # 74
beta_post = beta_prior + failures       # 28

# Posterior mean
post_mean = alpha_post / (alpha_post + beta_post)  # 0.725
```

---

## Updating Beliefs Sequentially

```python
# Start with weak prior
alpha, beta = 1, 1

# Month 1: 15 above median out of 20
alpha, beta = alpha + 15, beta + 5       # Beta(16, 6)

# Month 2: 18 above median out of 25
alpha, beta = alpha + 18, beta + 7       # Beta(34, 13)

# Month 3: 12 above median out of 20
alpha, beta = alpha + 12, beta + 8       # Beta(46, 21)

# Each month narrows the posterior
print(f"Current estimate: {alpha/(alpha+beta):.2%}")  # 68.66%
```

Bayesian updating is **incremental** -- you do not need to reprocess all data.

---

## Choosing Priors

| Prior Type               | When to Use                                     |
| ------------------------ | ----------------------------------------------- |
| **Uninformative** (flat) | No domain knowledge; let data speak             |
| **Weakly informative**   | Mild constraints (e.g., "prices are positive")  |
| **Informative**          | Strong domain knowledge (e.g., historical data) |
| **Empirical Bayes**      | Estimate prior from data (pragmatic compromise) |

Rule of thumb: start uninformative, add information only when justified.

---

## Computing with Polars

```python
import polars as pl
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent02", "hdbprices.csv")

# Compute empirical prior from full dataset
overall_median = df["price"].median()

# Per-town Bayesian success rates
town_stats = df.group_by("town").agg(
    (pl.col("price") > overall_median).sum().alias("above"),
    (pl.col("price") <= overall_median).sum().alias("below"),
).with_columns(
    ((pl.col("above") + 1) / (pl.col("above") + pl.col("below") + 2))
        .alias("bayesian_rate")  # Beta posterior mean with uniform prior
)
```

---

## Exercise Preview

**Exercise 2.1: Bayesian HDB Price Analysis**

You will:

1. Compute prior distributions from historical data
2. Update beliefs with new monthly observations
3. Compare Bayesian estimates to raw frequencies (especially for small samples)
4. Visualise prior, likelihood, and posterior distributions

Scaffolding level: **Moderate+ (~60% code provided)**

---

## Common Pitfalls

| Mistake                              | Fix                                              |
| ------------------------------------ | ------------------------------------------------ |
| Confusing prior and posterior        | Prior = before data; posterior = after data      |
| Using overly strong priors           | Let data dominate unless you have real expertise |
| Forgetting to update both α and β    | Successes update α; failures update β            |
| Treating Bayesian estimates as exact | They are distributions, not point values         |
| Ignoring sample size                 | Small samples need stronger priors               |

---

## Summary

- Bayesian thinking updates beliefs with evidence via Bayes' theorem
- Distributions (Normal, Beta, Poisson) model different phenomena
- Conjugate priors make posterior computation tractable
- Sequential updating allows incremental learning from new data
- Prior choice ranges from uninformative to highly informative

---

## Next Lesson

**Lesson 2.2: Estimation and Inference**

We will learn:

- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) estimation
- How MLE and MAP connect to Bayesian thinking
