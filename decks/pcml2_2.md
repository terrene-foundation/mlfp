---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 2.2: Estimation and Inference

### Module 2: Statistical Mastery

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Derive Maximum Likelihood Estimates (MLE) for common distributions
- Compute Maximum A Posteriori (MAP) estimates with priors
- Explain the relationship between MLE, MAP, and Bayesian inference
- Apply estimation methods to real datasets using Polars

---

## Recap: Lesson 2.1

- Bayesian thinking updates beliefs via Bayes' theorem
- Common distributions: Normal, Beta, Poisson, Binomial
- Conjugate priors keep posterior computation tractable
- Sequential updating allows incremental learning

---

## The Estimation Problem

Given observed data, what are the most plausible parameter values?

```
Observed HDB prices: [350k, 420k, 480k, 510k, 390k, ...]

Question: What normal distribution best describes these prices?
  → What is the best μ (mean)?
  → What is the best σ (standard deviation)?
```

Three approaches: MLE, MAP, and full Bayesian.

---

## Maximum Likelihood Estimation (MLE)

**Idea**: Find the parameters that make the observed data most probable.

$$\hat{\theta}_{MLE} = \arg\max_{\theta} P(D | \theta)$$

"Which parameter values maximise the likelihood of seeing this data?"

- No prior knowledge used
- Purely data-driven
- The frequentist default

---

## MLE for the Normal Distribution

For data x_1, x_2, ..., x_n from a Normal distribution:

```
MLE for mean:     μ_hat = (1/n) Σ x_i     (sample mean)
MLE for variance: σ²_hat = (1/n) Σ (x_i - μ_hat)²
```

```python
import polars as pl

prices = df["price"]

mu_mle = prices.mean()
sigma_mle = prices.std()

print(f"MLE mean:  ${mu_mle:,.0f}")
print(f"MLE stdev: ${sigma_mle:,.0f}")
```

The sample mean is the MLE -- the most familiar estimator.

---

## Log-Likelihood (Why Logs?)

Likelihoods are products of probabilities -- they get tiny fast.

```
L(θ) = P(x₁|θ) × P(x₂|θ) × ... × P(xₙ|θ)    (underflows!)

ℓ(θ) = log P(x₁|θ) + log P(x₂|θ) + ... + log P(xₙ|θ)  (stable)
```

Maximising the log-likelihood gives the same answer (log is monotonic) but is numerically stable.

---

## MLE for Bernoulli

**Question**: What fraction of flats sell above $500k?

```
Data: [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]  (1 = above, 0 = below)

MLE: p_hat = number of 1s / total = 6/10 = 0.60
```

```python
above_500k = df.with_columns(
    (pl.col("price") > 500_000).cast(pl.Int32).alias("above")
)
p_mle = above_500k["above"].mean()
print(f"MLE probability above $500k: {p_mle:.3f}")
```

---

## The Problem with MLE: Small Samples

```
Scenario: New BTO town, only 3 sales observed
  Prices: [680k, 720k, 710k]
  MLE mean: $703,333
  MLE std:  $16,997

Is this reliable? With only 3 observations,
the MLE is highly sensitive to each data point.
```

MLE treats all sample sizes equally -- it has no concept of uncertainty about the estimate itself.

---

## Maximum A Posteriori (MAP)

**Idea**: Find the parameters that maximise the posterior (likelihood x prior).

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta | D) = \arg\max_{\theta} P(D | \theta) \cdot P(\theta)$$

MAP = MLE + prior knowledge.

- When prior is flat (uninformative): MAP = MLE
- When prior is strong: MAP is pulled toward the prior

---

## MAP Estimation Visual

```
                    MLE
Parameter value      ↓
       ←─────────────●─────────────→

Prior belief: ──╱╲──
                 ↑
              Prior peak

MAP estimate:    ●    (between MLE and prior)
                 ↑
              Compromise
```

MAP balances the evidence (data) against prior belief.

---

## MAP for the Normal Mean

With a Normal prior on μ and known σ:

```
Prior:      μ ~ Normal(μ₀, σ₀²)
Likelihood: x ~ Normal(μ, σ²)

MAP estimate:
  μ_MAP = (μ₀/σ₀² + n·x̄/σ²) / (1/σ₀² + n/σ²)
```

```python
# Prior: national average HDB price
mu_prior = 450_000
sigma_prior = 100_000

# Data: 3 observations from new town
x_bar = 703_333
n = 3
sigma = 50_000  # assumed known

precision_prior = 1 / sigma_prior**2
precision_data = n / sigma**2

mu_map = (mu_prior * precision_prior + x_bar * precision_data) / (precision_prior + precision_data)
print(f"MAP estimate: ${mu_map:,.0f}")  # pulled toward 450k
```

---

## MLE vs MAP vs Full Bayesian

| Method            | Output            | Prior? | Uncertainty?             |
| ----------------- | ----------------- | ------ | ------------------------ |
| **MLE**           | Single best value | No     | No (need bootstrap)      |
| **MAP**           | Single best value | Yes    | No (point estimate)      |
| **Full Bayesian** | Full distribution | Yes    | Yes (credible intervals) |

```
MLE:        ●                    (just the peak)
MAP:        ●                    (peak shifted by prior)
Bayesian:   ╱╲                   (entire distribution)
            ╱  ╲   with credible interval
```

---

## Connection to Regularisation

MAP estimation with specific priors equals regularised regression:

```
Prior on weights ~ Normal(0, σ²)  →  L2 regularisation (Ridge)
Prior on weights ~ Laplace(0, b)  →  L1 regularisation (Lasso)
```

This is a deep connection we will revisit in Module 3 (Lesson 3.1).

Regularisation = MAP estimation with a shrinkage prior.

---

## Confidence vs Credible Intervals

| Frequentist (MLE)                                                        | Bayesian (MAP/Full)                                            |
| ------------------------------------------------------------------------ | -------------------------------------------------------------- |
| 95% **confidence** interval                                              | 95% **credible** interval                                      |
| "If we repeated sampling, 95% of intervals would contain the true value" | "There is 95% probability the parameter lies in this interval" |
| About the procedure                                                      | About the parameter                                            |

Credible intervals answer the question people actually want to ask.

---

## Computing Credible Intervals

```python
import numpy as np
from scipy import stats

# Beta posterior from Lesson 2.1
alpha_post, beta_post = 74, 28

# 95% credible interval
lower = stats.beta.ppf(0.025, alpha_post, beta_post)
upper = stats.beta.ppf(0.975, alpha_post, beta_post)

print(f"95% credible interval: [{lower:.3f}, {upper:.3f}]")
# [0.637, 0.804]
```

"We are 95% confident the true proportion is between 63.7% and 80.4%."

---

## Estimation Workflow

```
1. Choose a model (Normal, Bernoulli, Poisson, ...)
   └→ Based on data type and domain knowledge

2. Choose an estimation method
   └→ MLE if no prior knowledge + large sample
   └→ MAP if domain knowledge available
   └→ Full Bayesian if uncertainty matters

3. Compute estimates
   └→ Closed-form (conjugate) or numerical optimisation

4. Validate
   └→ Check residuals, goodness of fit, prediction accuracy
```

---

## Full Example with Polars

```python
import polars as pl
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent02", "hdbprices.csv")

# Per-town MLE and MAP estimates
national_mean = df["price"].mean()

town_estimates = df.group_by("town").agg(
    pl.col("price").mean().alias("mle_mean"),
    pl.col("price").count().alias("n"),
).with_columns(
    # MAP: shrink small-sample towns toward national mean
    (
        (pl.col("mle_mean") * pl.col("n") + national_mean * 10)
        / (pl.col("n") + 10)
    ).alias("map_mean")  # prior weight = 10 pseudo-observations
)

print(town_estimates.sort("n"))  # small towns show biggest MAP shift
```

---

## Exercise Preview

**Exercise 2.2: MLE and MAP for HDB Prices**

You will:

1. Compute MLE estimates for price distributions by town
2. Apply MAP estimation with a national-average prior
3. Compare MLE vs MAP for small-sample vs large-sample towns
4. Compute and visualise credible intervals

Scaffolding level: **Moderate+ (~60% code provided)**

---

## Common Pitfalls

| Mistake                                     | Fix                                          |
| ------------------------------------------- | -------------------------------------------- |
| Using MLE with tiny samples                 | MAP is more reliable with n < 30             |
| Confusing confidence and credible intervals | Confidence = procedure; credible = parameter |
| Setting prior variance too small            | Overly informative prior dominates the data  |
| Forgetting MAP = MLE when prior is flat     | They are not always different                |
| Using MLE variance (biased with 1/n)        | Use Bessel correction (1/(n-1)) for unbiased |

---

## Summary

- MLE finds parameters that maximise the likelihood of observed data
- MAP adds prior knowledge, pulling estimates toward prior beliefs
- Full Bayesian gives entire posterior distributions with uncertainty
- MAP with Normal/Laplace priors equals L2/L1 regularisation
- Credible intervals answer "where is the parameter?" directly

---

## Next Lesson

**Lesson 2.3: Hypothesis Testing**

We will learn:

- Neyman-Pearson framework for hypothesis testing
- Statistical power and sample size planning
- Multiple testing corrections (Bonferroni, FDR)
