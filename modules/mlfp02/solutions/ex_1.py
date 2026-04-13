# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 1: Probability and Bayesian Thinking
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Construct truth tables for probability problems and compute joint/
#     conditional probabilities from real HDB transaction data
#   - Apply Bayes' theorem to real-world scenarios (medical tests, property
#     valuation, A/B test results)
#   - Compute MLE for Normal distribution parameters and quantify estimation
#     uncertainty via the Cramér-Rao bound
#   - Implement Normal-Normal and Beta-Binomial conjugate priors, derive
#     posterior distributions analytically, and compare them
#   - Run prior sensitivity analysis — show how the posterior changes as
#     the prior mean and variance are swept through a range
#   - Distinguish Bayesian credible intervals from frequentist confidence
#     intervals with a repeated-sampling simulation
#   - Compute expected value and demonstrate sampling bias (friendship
#     paradox) with a graph simulation
#   - Visualise prior, likelihood, and posterior using ModelVisualizer
#
# PREREQUISITES: Complete M1 — you should be comfortable loading data,
#   computing summary statistics, and reading Polars DataFrames.
#
# ESTIMATED TIME: ~170 minutes
#
# TASKS:
#    1. Load data, compute probability fundamentals (truth tables, joint probs)
#    2. Compute MLE for Normal parameters with Cramér-Rao bound
#    3. Bayes' theorem applied — medical test & HDB valuation scenarios
#    4. Normal-Normal conjugate prior: derive and compute posterior
#    5. Prior sensitivity analysis — sweep prior hyperparameters
#    6. Beta-Binomial conjugate: model HDB transaction success rates
#    7. Credible vs confidence interval — repeated-sampling simulation
#    8. Expected value and sampling bias (friendship paradox simulation)
#    9. Bootstrap confidence intervals (percentile + BCa) for comparison
#   10. Bayesian estimation across flat types — compare data vs prior balance
#   11. Visualise all results with ModelVisualizer
#   12. Business interpretation synthesis
#
# DATASET: HDB resale flat transactions (Singapore)
#   Source: data.gov.sg — public housing resale records
#   Filtered to: 4-ROOM flats, 2020 onwards (primary), all types (Task 10)
#   Key column: resale_price (SGD)
#
# THEORY:
#   Normal-Normal conjugate: prior μ ~ N(μ₀, σ₀²), likelihood x ~ N(μ, σ²)
#   Posterior: μ|x ~ N(μₙ, σₙ²) where:
#     μₙ = (μ₀/σ₀² + n*x̄/σ²) / (1/σ₀² + n/σ²)
#     σₙ² = 1 / (1/σ₀² + n/σ²)
#
#   Beta-Binomial conjugate: prior p ~ Beta(α, β), likelihood x ~ Binomial(n,p)
#   Posterior: p|x ~ Beta(α + k, β + n - k) where k = number of successes
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kailash_ml import ModelVisualizer
from scipy import stats

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

print("=" * 70)
print("  MLFP02 Exercise 1: Probability and Bayesian Thinking")
print("=" * 70)

# Focus on a specific flat type and recent period for clearer analysis
hdb_recent = hdb.filter(
    (pl.col("flat_type") == "4 ROOM")
    & (pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))
)

prices = hdb_recent["resale_price"].to_numpy().astype(np.float64)
print(f"\n  Data loaded: {len(prices):,} 4-room HDB transactions (2020+)")
print(f"  Price range: ${prices.min():,.0f} – ${prices.max():,.0f}")
print(f"  Sample mean: ${prices.mean():,.0f}")
print(f"  Sample std:  ${prices.std():,.0f}\n")

# ══════════════════════════════════════════════════════════════════════
# BAYESIAN THINKING — THE INTUITION (before the math)
# ══════════════════════════════════════════════════════════════════════
# Before diving into formulas, here's the everyday intuition:
#
# Imagine you're estimating the price of a 4-room HDB flat. You have:
#
#   1. A PRIOR belief (what you already think):
#      "I've been watching the market — 4-room flats cost around $500K."
#      This comes from your experience, NOT from this dataset.
#
#   2. NEW DATA (what you just observed):
#      You pull 50 recent transactions and compute the sample mean.
#
#   3. A POSTERIOR belief (your updated guess):
#      A principled combination of your prior + the new data.
#
# The key insight: with LITTLE data, your prior dominates the posterior.
# With LOTS of data, the data dominates and the prior "fades away."
# This is NOT subjective opinion — it's the mathematically OPTIMAL way
# to update beliefs in the face of evidence (Bayes' theorem).
#
# Why this matters for business:
#   - You rarely have "infinite data" — priors help when data is thin.
#   - Frequentist methods (confidence intervals) pretend you have no
#     prior knowledge; Bayesian methods let you USE what you already know.
#   - Every decision you make already uses priors — you just don't
#     usually write them down. Bayesian analysis makes them explicit.
#
# The formulas below (Normal-Normal, Beta-Binomial) are just specific
# recipes for combining specific kinds of priors with specific kinds
# of data. The intuition is always the same: prior + data → posterior.
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Probability Fundamentals — Truth Tables and Joint Probabilities
# ══════════════════════════════════════════════════════════════════════
# Probability begins with counting. We use HDB data to compute
# empirical joint and conditional probabilities from real events.
#
# Key rules:
#   P(A) + P(A') = 1
#   P(A,B) = P(A) × P(B|A)
#   Independent events: P(A,B) = P(A) × P(B)

print("\n" + "=" * 70)
print("TASK 1: Probability Fundamentals")
print("=" * 70)

# Define events on the full HDB data (not just 4-room)
hdb_all = hdb.filter(pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))
total_n = hdb_all.height

# Event A: transaction is a 4-room flat
n_4room = hdb_all.filter(pl.col("flat_type") == "4 ROOM").height
p_4room = n_4room / total_n

# Event B: price above $500K
n_above_500k = hdb_all.filter(pl.col("resale_price") > 500_000).height
p_above_500k = n_above_500k / total_n

# Joint probability: P(4-room AND price > 500K)
n_4room_and_above = hdb_all.filter(
    (pl.col("flat_type") == "4 ROOM") & (pl.col("resale_price") > 500_000)
).height
p_joint = n_4room_and_above / total_n

# Conditional: P(price > 500K | 4-room)
p_above_given_4room = n_4room_and_above / n_4room if n_4room > 0 else 0

# Test independence: P(A,B) vs P(A)×P(B)
p_independent = p_4room * p_above_500k

print(f"\n--- Truth Table (empirical) ---")
print(f"Total transactions (2020+): {total_n:,}")
print(f"P(4-room)           = {p_4room:.4f} ({p_4room:.1%})")
print(f"P(price > $500K)    = {p_above_500k:.4f} ({p_above_500k:.1%})")
print(f"P(4-room AND >$500K)= {p_joint:.4f} ({p_joint:.1%})")
print(f"P(>$500K | 4-room)  = {p_above_given_4room:.4f} ({p_above_given_4room:.1%})")
print(f"\n--- Independence Check ---")
print(f"P(A)×P(B) = {p_independent:.4f}")
print(f"P(A,B)    = {p_joint:.4f}")
print(f"Difference: {abs(p_joint - p_independent):.4f}")
if abs(p_joint - p_independent) < 0.01:
    print("Events are approximately independent")
else:
    print("Events are NOT independent — flat type affects price probability")
# INTERPRETATION: If flat type and price are not independent, knowing
# the flat type tells you something about the price distribution. This
# is the foundation for conditional reasoning in property valuation.

# Compute a full cross-tabulation: flat_type x price_category
price_cats = hdb_all.with_columns(
    pl.when(pl.col("resale_price") <= 400_000)
    .then(pl.lit("≤400K"))
    .when(pl.col("resale_price") <= 600_000)
    .then(pl.lit("400K-600K"))
    .when(pl.col("resale_price") <= 800_000)
    .then(pl.lit("600K-800K"))
    .otherwise(pl.lit(">800K"))
    .alias("price_band")
)
cross_tab = (
    price_cats.group_by("flat_type", "price_band")
    .agg(pl.len().alias("count"))
    .sort("flat_type", "price_band")
)
print(f"\n--- Cross-Tabulation (sample) ---")
print(cross_tab.head(12))

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 < p_4room < 1, "P(4-room) must be a valid probability"
assert 0 < p_above_500k < 1, "P(>500K) must be a valid probability"
assert p_joint <= min(p_4room, p_above_500k), "Joint prob cannot exceed marginals"
assert (
    abs(p_above_given_4room - p_joint / p_4room) < 1e-10
), "Conditional probability identity must hold: P(B|A) = P(A,B)/P(A)"
print("\n✓ Checkpoint 1 passed — probability fundamentals computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Maximum Likelihood Estimation (MLE)
# ══════════════════════════════════════════════════════════════════════
# For X ~ N(μ, σ²): MLE gives μ̂ = x̄, σ̂² = (1/n)Σ(xᵢ - x̄)²
# MLE is asymptotically efficient (achieves Cramér-Rao lower bound).
# The Fisher information quantifies how much data tells us about θ.

n = len(prices)
mle_mean = prices.mean()
mle_var = prices.var(ddof=0)  # MLE uses ddof=0 (biased estimator)
mle_std = np.sqrt(mle_var)

# Bessel's correction: unbiased variance uses ddof=1
unbiased_var = prices.var(ddof=1)
unbiased_std = np.sqrt(unbiased_var)

# Fisher information for Normal: I(μ) = n/σ² → Var(μ̂) ≥ σ²/n
fisher_info = n / mle_var
cramer_rao_bound = 1 / fisher_info  # Minimum variance for any unbiased estimator
mle_se = np.sqrt(cramer_rao_bound)

print(f"\n=== MLE Estimates ===")
print(f"μ̂ = ${mle_mean:,.0f}")
print(f"σ̂ (MLE, ddof=0)     = ${mle_std:,.0f}")
print(f"σ̂ (unbiased, ddof=1) = ${unbiased_std:,.0f}")
print(f"Bias: MLE σ underestimates by ${unbiased_std - mle_std:,.2f}")
print(f"\nFisher information I(μ) = {fisher_info:.4f}")
print(f"Cramér-Rao lower bound: Var(μ̂) ≥ {cramer_rao_bound:.2f}")
print(f"MLE standard error: ${mle_se:,.2f}")
# INTERPRETATION: The standard error tells you the precision of the
# mean estimate. With {n:,} transactions, the SE is tiny relative to
# the mean — our estimate of the average 4-room HDB price is very
# precise. The Cramér-Rao bound guarantees no unbiased estimator can
# do better than this SE for the Normal model.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert n > 0, "No data loaded — check the filter conditions"
assert mle_mean > 0, "MLE mean should be positive (price cannot be zero)"
assert mle_std > 0, "MLE std should be positive"
assert mle_se > 0, "Standard error should be positive"
assert mle_se < mle_std, "SE of mean should be much smaller than std of prices"
assert unbiased_std > mle_std, "Unbiased σ must be > MLE σ (Bessel's correction)"
print("\n✓ Checkpoint 2 passed — MLE estimates and Cramér-Rao bound computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Bayes' Theorem — Real-World Applications
# ══════════════════════════════════════════════════════════════════════
# Bayes' theorem: P(A|B) = P(B|A) × P(A) / P(B)
# Two worked examples: medical test and property valuation.

print(f"\n=== Bayes' Theorem Application 1: Medical Test (COVID ART) ===")

# COVID ART test sensitivity and specificity
sensitivity = 0.85  # P(positive test | infected)
specificity = 0.995  # P(negative test | not infected)
prevalence = 0.02  # P(infected) — base rate in Singapore community

# P(positive test) = P(+|infected)P(infected) + P(+|not infected)P(not infected)
p_positive = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)

# P(infected | positive test)
p_infected_given_positive = (sensitivity * prevalence) / p_positive

# P(not infected | positive test) — false positive rate among positives
p_false_positive = 1 - p_infected_given_positive

print(f"Sensitivity: {sensitivity:.1%} — P(+test | infected)")
print(f"Specificity: {specificity:.1%} — P(-test | not infected)")
print(f"Prevalence:  {prevalence:.1%} — P(infected)")
print(f"")
print(f"P(positive test)           = {p_positive:.4f} ({p_positive:.2%})")
print(
    f"P(infected | positive test) = {p_infected_given_positive:.4f} ({p_infected_given_positive:.1%})"
)
print(f"P(false positive)           = {p_false_positive:.4f} ({p_false_positive:.1%})")
# INTERPRETATION: Even with a 99.5% specificity test, when prevalence is
# only 2%, a positive test means you're truly infected only {p_infected_given_positive:.0%}
# of the time. This is the base rate fallacy — ignoring prevalence leads
# to overconfidence in test results. This is why confirmatory tests exist.

# Sweep prevalence to show how posterior changes
print(f"\n--- Effect of Prevalence on P(infected | +test) ---")
for prev in [0.001, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
    p_pos = sensitivity * prev + (1 - specificity) * (1 - prev)
    p_inf = (sensitivity * prev) / p_pos
    print(f"  Prevalence {prev:>5.1%} → P(infected | +) = {p_inf:.1%}")

print(f"\n=== Bayes' Theorem Application 2: HDB Valuation ===")

# Scenario: a 4-room flat in Bishan is listed at $650K. Is it overpriced?
# Prior: market average for 4-room is $500K ± $100K
# Likelihood: given true value = $650K, listing price ~ N(650K, 30K²)
# We want P(true value > $600K | listing = $650K)

# From our data: what fraction of 4-room Bishan flats sell above $600K?
bishan_flats = hdb_recent.filter(pl.col("town") == "BISHAN")
if bishan_flats.height > 0:
    p_above_600k_bishan = (
        bishan_flats.filter(pl.col("resale_price") > 600_000).height
        / bishan_flats.height
    )
    mean_bishan = bishan_flats["resale_price"].mean()
    print(f"Bishan 4-room data: {bishan_flats.height} transactions")
    print(f"Mean price: ${mean_bishan:,.0f}")
    print(f"P(price > $600K | Bishan 4-room) = {p_above_600k_bishan:.2%}")
    print(f"This empirical probability is the 'data-driven prior' for Bishan.")
else:
    p_above_600k_bishan = 0.5
    print("No Bishan 4-room data found — using uninformative prior")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 0 < p_infected_given_positive < 1, "Posterior probability must be valid"
assert (
    p_infected_given_positive > prevalence
), "Positive test must increase probability of infection above base rate"
print("\n✓ Checkpoint 3 passed — Bayes' theorem applications computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Normal-Normal Conjugate Prior — Posterior Distribution
# ══════════════════════════════════════════════════════════════════════
# Prior belief: Singapore 4-room HDB prices centre around $500K
# with moderate uncertainty (σ₀ = $100K)
#
# Normal-Normal conjugate: prior for μ
#   μ ~ N(μ₀, σ₀²)
# We treat σ² as known (plug in MLE estimate) for analytical tractability.

# Prior hyperparameters
mu_0 = 500_000.0  # Prior mean: $500K
sigma_0 = 100_000.0  # Prior std: moderate uncertainty

# Known variance (plug-in from MLE)
sigma_known = mle_std

# Posterior computation
precision_prior = 1.0 / sigma_0**2
precision_data = n / sigma_known**2
precision_posterior = precision_prior + precision_data
sigma_n_sq = 1.0 / precision_posterior
sigma_n = np.sqrt(sigma_n_sq)

# Posterior mean = precision-weighted combination
mu_n = sigma_n_sq * (mu_0 * precision_prior + n * mle_mean / sigma_known**2)

print(f"\n=== Normal-Normal Conjugate Posterior ===")
print(f"Prior: μ ~ N(μ₀={mu_0:,.0f}, σ₀={sigma_0:,.0f})")
print(f"Likelihood: X|μ ~ N(μ, σ²) with σ={sigma_known:,.0f} (plug-in)")
print(f"\nPosterior: μ|data ~ N(μₙ={mu_n:,.0f}, σₙ={sigma_n:,.2f})")
print(f"Prior precision:     {precision_prior:.2e}")
print(f"Data precision:      {precision_data:.2e}")
print(f"Posterior precision:  {precision_posterior:.2e}")
print(f"Data-to-prior precision ratio: {precision_data / precision_prior:.0f}x")
print(
    f"  → Posterior is dominated by {'data' if precision_data > precision_prior else 'prior'}"
)

# 95% credible interval
ci_95_lower = mu_n - 1.96 * sigma_n
ci_95_upper = mu_n + 1.96 * sigma_n
print(f"\n95% Bayesian credible interval: [${ci_95_lower:,.2f}, ${ci_95_upper:,.2f}]")
# INTERPRETATION: A credible interval has a direct probability statement:
# "Given the data, there is a 95% probability that the true mean price
# lies in this range." This is STRONGER than the frequentist CI, which
# says "95% of intervals constructed this way would contain the true mean."

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert precision_data > 0, "Data precision must be positive"
assert sigma_n < sigma_0, "Posterior std should be narrower than prior (data adds info)"
assert ci_95_lower < mu_n < ci_95_upper, "Posterior mean must be within its own CI"
assert abs(mu_n - mle_mean) < sigma_0, "Posterior should be close to MLE with large n"
print("\n✓ Checkpoint 4 passed — Normal-Normal posterior computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Prior Sensitivity Analysis
# ══════════════════════════════════════════════════════════════════════
# How does the posterior change as we vary the prior?
# Sweep prior mean from $300K to $700K and prior std from $20K to $200K.
# This shows when the prior matters and when data overwhelms it.

print(f"\n=== Prior Sensitivity Analysis ===")

# Sweep prior mean
print(f"\n--- Varying prior mean (σ₀ = ${sigma_0:,.0f} fixed) ---")
print(
    f"{'Prior μ₀':>12} {'Posterior μₙ':>15} {'Shift from MLE':>16} {'Prior Weight':>13}"
)
print("─" * 60)
for mu_sweep in [300_000, 400_000, 500_000, 600_000, 700_000]:
    prec_pr = 1.0 / sigma_0**2
    prec_dt = n / sigma_known**2
    prec_post = prec_pr + prec_dt
    mu_post = (mu_sweep * prec_pr + n * mle_mean / sigma_known**2) / prec_post
    prior_wt = prec_pr / prec_post * 100
    print(
        f"${mu_sweep:>10,.0f}  ${mu_post:>13,.0f}  {mu_post - mle_mean:>+14,.0f}  {prior_wt:>11.4f}%"
    )

# Sweep prior std
print(f"\n--- Varying prior std (μ₀ = ${mu_0:,.0f} fixed) ---")
print(f"{'Prior σ₀':>12} {'Posterior μₙ':>15} {'Prior Weight':>13}")
print("─" * 45)
for sigma_sweep in [20_000, 50_000, 100_000, 200_000, 500_000]:
    prec_pr = 1.0 / sigma_sweep**2
    prec_dt = n / sigma_known**2
    prec_post = prec_pr + prec_dt
    mu_post = (mu_0 * prec_pr + n * mle_mean / sigma_known**2) / prec_post
    prior_wt = prec_pr / prec_post * 100
    print(f"${sigma_sweep:>10,.0f}  ${mu_post:>13,.0f}  {prior_wt:>11.4f}%")
# INTERPRETATION: Even a very opinionated prior (σ₀=$20K) gets overwhelmed
# by the data when n is large. But with small n, the prior choice matters.
# A sensitivity analysis like this should accompany any Bayesian report
# to show stakeholders that conclusions are robust to prior assumptions.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
# Verify that varying the prior mean within ±$200K barely moves posterior
for mu_test in [300_000, 700_000]:
    prec_pr = 1.0 / sigma_0**2
    prec_dt = n / sigma_known**2
    mu_post_test = (mu_test * prec_pr + n * mle_mean / sigma_known**2) / (
        prec_pr + prec_dt
    )
    assert (
        abs(mu_post_test - mle_mean) < 5000
    ), f"With large n, posterior should be near MLE regardless of prior mean"
print("\n✓ Checkpoint 5 passed — prior sensitivity demonstrates data dominance\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Beta-Binomial Conjugate — Transaction Success Rates
# ══════════════════════════════════════════════════════════════════════
# Different conjugate family: Beta prior for a proportion.
# Use case: what fraction of 4-room HDB transactions close above $500K?
#
# Prior: p ~ Beta(α, β) where E[p] = α/(α+β)
# Likelihood: k successes in n trials ~ Binomial(n, p)
# Posterior: p | data ~ Beta(α + k, β + n - k)

print(f"\n=== Beta-Binomial Conjugate ===")

# Define "success" = transaction price > $500K
threshold = 500_000
k_success = int((prices > threshold).sum())
n_trials = len(prices)
empirical_rate = k_success / n_trials

print(f"Threshold: ${threshold:,.0f}")
print(
    f"Successes (price > threshold): {k_success:,} / {n_trials:,} = {empirical_rate:.2%}"
)

# Weakly informative prior: Beta(2, 2) — slight preference for 50%
alpha_prior = 2.0
beta_prior = 2.0
prior_mean = alpha_prior / (alpha_prior + beta_prior)

# Strong prior: Beta(20, 80) — 20% success rate expected
alpha_strong = 20.0
beta_strong = 80.0
strong_prior_mean = alpha_strong / (alpha_strong + beta_strong)

# Posterior with weak prior
alpha_post_weak = alpha_prior + k_success
beta_post_weak = beta_prior + (n_trials - k_success)
post_mean_weak = alpha_post_weak / (alpha_post_weak + beta_post_weak)

# Posterior with strong prior
alpha_post_strong = alpha_strong + k_success
beta_post_strong = beta_strong + (n_trials - k_success)
post_mean_strong = alpha_post_strong / (alpha_post_strong + beta_post_strong)

# 95% credible intervals
ci_weak = stats.beta.ppf([0.025, 0.975], alpha_post_weak, beta_post_weak)
ci_strong = stats.beta.ppf([0.025, 0.975], alpha_post_strong, beta_post_strong)

print(f"\n--- Weak Prior: Beta({alpha_prior}, {beta_prior}), E[p]={prior_mean:.2f} ---")
print(f"Posterior: Beta({alpha_post_weak:.0f}, {beta_post_weak:.0f})")
print(f"Posterior mean: {post_mean_weak:.4f} ({post_mean_weak:.2%})")
print(f"95% CI: [{ci_weak[0]:.4f}, {ci_weak[1]:.4f}]")

print(
    f"\n--- Strong Prior: Beta({alpha_strong}, {beta_strong}), E[p]={strong_prior_mean:.2f} ---"
)
print(f"Posterior: Beta({alpha_post_strong:.0f}, {beta_post_strong:.0f})")
print(f"Posterior mean: {post_mean_strong:.4f} ({post_mean_strong:.2%})")
print(f"95% CI: [{ci_strong[0]:.4f}, {ci_strong[1]:.4f}]")

print(f"\nEmpirical rate: {empirical_rate:.4f}")
print(
    f"Both posteriors converge toward {empirical_rate:.2%} because n={n_trials:,} is large."
)
# INTERPRETATION: The Beta-Binomial is the natural Bayesian model for
# proportions. Even a strong prior (Beta(20,80) suggesting 20%) gets
# overwhelmed by thousands of observations. In practice, use informative
# priors when you have domain knowledge and small samples.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert alpha_post_weak == alpha_prior + k_success, "Posterior alpha update incorrect"
assert beta_post_weak == beta_prior + (
    n_trials - k_success
), "Posterior beta update incorrect"
assert ci_weak[0] < post_mean_weak < ci_weak[1], "Posterior mean must be within CI"
assert (
    abs(post_mean_weak - empirical_rate) < 0.01
), "With weak prior and large n, posterior mean should be near empirical rate"
print("\n✓ Checkpoint 6 passed — Beta-Binomial conjugate computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Credible vs Confidence Interval — Repeated Sampling Simulation
# ══════════════════════════════════════════════════════════════════════
# The critical difference:
#   Frequentist CI: "If I repeated this experiment many times, 95% of
#     the intervals I construct would contain the true μ."
#   Bayesian credible: "Given THIS data, there is a 95% probability
#     that μ lies in this interval."
# We simulate repeated sampling to demonstrate the frequentist coverage.

print(f"\n=== Credible vs Confidence Interval Simulation ===")

rng = np.random.default_rng(seed=42)
true_mu = mle_mean  # Treat our MLE as the "true" population mean
true_sigma = mle_std
n_simulations = 1000
sample_size = 100  # Small sample to show variation

freq_covers = 0
bayes_covers = 0
freq_widths = []
bayes_widths = []

for _ in range(n_simulations):
    sample = rng.normal(true_mu, true_sigma, size=sample_size)
    xbar = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(sample_size)

    # Frequentist 95% CI
    freq_lower = xbar - 1.96 * se
    freq_upper = xbar + 1.96 * se
    if freq_lower <= true_mu <= freq_upper:
        freq_covers += 1
    freq_widths.append(freq_upper - freq_lower)

    # Bayesian 95% credible interval (Normal-Normal with weak prior)
    prec_pr = 1.0 / sigma_0**2
    prec_dt = sample_size / sample.std(ddof=0) ** 2
    prec_post = prec_pr + prec_dt
    sigma_post = np.sqrt(1.0 / prec_post)
    mu_post = (
        mu_0 * prec_pr + sample_size * xbar / sample.std(ddof=0) ** 2
    ) / prec_post
    bayes_lower = mu_post - 1.96 * sigma_post
    bayes_upper = mu_post + 1.96 * sigma_post
    if bayes_lower <= true_mu <= bayes_upper:
        bayes_covers += 1
    bayes_widths.append(bayes_upper - bayes_lower)

freq_coverage = freq_covers / n_simulations
bayes_coverage = bayes_covers / n_simulations

print(f"Simulations: {n_simulations:,}, sample size: {sample_size}")
print(f"Frequentist CI coverage: {freq_coverage:.1%} (target: 95%)")
print(f"Bayesian credible coverage: {bayes_coverage:.1%}")
print(f"Mean freq CI width: ${np.mean(freq_widths):,.0f}")
print(f"Mean Bayes CI width: ${np.mean(bayes_widths):,.0f}")
# INTERPRETATION: Both methods achieve approximately 95% coverage when
# the model is correct. The Bayesian interval can be slightly narrower
# because the prior adds information. The key difference is philosophical:
# the freq CI is about the procedure, the Bayes CI is about THIS interval.

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert (
    0.90 < freq_coverage < 1.0
), f"Frequentist coverage should be near 95%, got {freq_coverage:.1%}"
assert (
    0.90 < bayes_coverage < 1.0
), f"Bayesian coverage should be near 95%, got {bayes_coverage:.1%}"
print("\n✓ Checkpoint 7 passed — coverage simulation validates both methods\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Expected Value and Sampling Bias (Friendship Paradox)
# ══════════════════════════════════════════════════════════════════════
# Expected value: E[X] = Σ pᵢ × xᵢ
# Sampling bias: your friends have more friends than you (on average)
# because popular people appear in more friend lists.

print(f"\n=== Expected Value: HDB Price by Flat Type ===")

# Compute expected price across flat types (weighted by transaction volume)
flat_type_stats = (
    hdb_all.group_by("flat_type")
    .agg(
        pl.col("resale_price").mean().alias("mean_price"),
        pl.len().alias("count"),
    )
    .sort("flat_type")
)

total_transactions = flat_type_stats["count"].sum()
flat_type_stats = flat_type_stats.with_columns(
    (pl.col("count") / total_transactions).alias("probability")
)

print(flat_type_stats)

# E[price] = Σ P(flat_type) × E[price | flat_type]
expected_price = (
    flat_type_stats["probability"].to_numpy() * flat_type_stats["mean_price"].to_numpy()
).sum()
print(f"\nE[price] (transaction-weighted) = ${expected_price:,.0f}")
print(f"Simple average across types     = ${flat_type_stats['mean_price'].mean():,.0f}")
print(f"Difference: ${expected_price - flat_type_stats['mean_price'].mean():,.0f}")
# INTERPRETATION: The transaction-weighted expected price differs from
# the simple average because different flat types have very different
# transaction volumes. 4-room flats dominate the market, pulling the
# weighted average toward their price point.

# Friendship paradox simulation
print(f"\n=== Sampling Bias: Friendship Paradox ===")
n_people = 200
# Create a power-law-like degree distribution (few popular, many unpopular)
degrees = rng.zipf(a=2.0, size=n_people).clip(max=n_people - 1)

# Average degree
avg_degree = degrees.mean()

# "Friend's average degree" — sample a person, then sample one of their
# friends. Popular people are sampled more often.
friend_degrees = []
for person_idx in range(n_people):
    if degrees[person_idx] > 0:
        # A friend is drawn proportional to degree (popularity)
        friend_probs = degrees / degrees.sum()
        friend_idx = rng.choice(n_people, p=friend_probs)
        friend_degrees.append(degrees[friend_idx])

avg_friend_degree = np.mean(friend_degrees)

print(f"People: {n_people}")
print(f"Your average number of friends: {avg_degree:.1f}")
print(f"Your friends' average number of friends: {avg_friend_degree:.1f}")
print(f"Ratio: {avg_friend_degree / avg_degree:.2f}x")
print(
    f"→ Your friends have {avg_friend_degree / avg_degree:.1f}x more friends than you!"
)
# INTERPRETATION: This is sampling bias in action. Popular people appear
# disproportionately in friend lists. In data science, this same bias
# affects product reviews (people who feel strongly review more), survey
# responses (self-selection), and click-through rates (placement bias).

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert avg_friend_degree > avg_degree, "Friends should have more friends (paradox)"
assert expected_price > 0, "Expected price must be positive"
print("\n✓ Checkpoint 8 passed — expected value and sampling bias demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Bootstrap Confidence Intervals for Comparison
# ══════════════════════════════════════════════════════════════════════
# Non-parametric bootstrap: resample with replacement, compute statistic
# BCa (bias-corrected accelerated) is the gold standard.

n_bootstrap = 10_000

# Bootstrap distribution of the sample mean
bootstrap_means = np.array(
    [rng.choice(prices, size=n, replace=True).mean() for _ in range(n_bootstrap)]
)

# Percentile confidence interval
boot_ci_lower = np.percentile(bootstrap_means, 2.5)
boot_ci_upper = np.percentile(bootstrap_means, 97.5)

# BCa confidence interval using scipy
bca_result = stats.bootstrap(
    (prices,),
    statistic=np.mean,
    n_resamples=n_bootstrap,
    confidence_level=0.95,
    method="BCa",
    random_state=42,
)
bca_ci_lower = bca_result.confidence_interval.low
bca_ci_upper = bca_result.confidence_interval.high

# Normal theory CI for reference
normal_ci_lower = mle_mean - 1.96 * mle_se
normal_ci_upper = mle_mean + 1.96 * mle_se

print(f"\n=== Confidence / Credible Intervals Comparison ===")
print(f"{'Method':<25} {'Lower':>14} {'Upper':>14} {'Width':>12}")
print("─" * 70)
print(
    f"{'Normal theory 95% CI':<25} ${normal_ci_lower:>12,.2f} ${normal_ci_upper:>12,.2f} ${normal_ci_upper - normal_ci_lower:>10,.2f}"
)
print(
    f"{'Bootstrap percentile CI':<25} ${boot_ci_lower:>12,.2f} ${boot_ci_upper:>12,.2f} ${boot_ci_upper - boot_ci_lower:>10,.2f}"
)
print(
    f"{'Bootstrap BCa CI':<25} ${bca_ci_lower:>12,.2f} ${bca_ci_upper:>12,.2f} ${bca_ci_upper - bca_ci_lower:>10,.2f}"
)
print(
    f"{'Bayesian 95% credible':<25} ${ci_95_lower:>12,.2f} ${ci_95_upper:>12,.2f} ${ci_95_upper - ci_95_lower:>10,.2f}"
)
# INTERPRETATION: All methods agree because n is large and the distribution
# is approximately Normal. Bootstrap percentile and BCa diverge for small n
# or skewed distributions. BCa is preferred as it corrects for both bias
# and skewness in the bootstrap distribution.

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert boot_ci_lower < boot_ci_upper, "Bootstrap CI lower must be below upper"
assert bca_ci_lower < bca_ci_upper, "BCa CI lower must be below upper"
assert normal_ci_lower < mle_mean < normal_ci_upper, "MLE mean within normal CI"
assert len(bootstrap_means) == n_bootstrap, "Should have n_bootstrap samples"
print("\n✓ Checkpoint 9 passed — bootstrap CIs computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Bayesian Estimation Across Flat Types
# ══════════════════════════════════════════════════════════════════════
# Apply Normal-Normal conjugate to each flat type to see how the
# prior-vs-data balance differs with sample size.

flat_types = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]

results_by_type = {}
for ft in flat_types:
    subset = hdb.filter(
        (pl.col("flat_type") == ft)
        & (pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))
    )
    if subset.height == 0:
        continue

    p = subset["resale_price"].to_numpy().astype(np.float64)
    n_ft = len(p)
    xbar = p.mean()
    s = p.std()

    # Same prior for all types
    prec_prior = 1.0 / sigma_0**2
    prec_data = n_ft / s**2
    prec_post = prec_prior + prec_data
    mu_post = (mu_0 * prec_prior + n_ft * xbar / s**2) / prec_post
    sigma_post = np.sqrt(1.0 / prec_post)

    results_by_type[ft] = {
        "n": n_ft,
        "mle_mean": xbar,
        "posterior_mean": mu_post,
        "posterior_std": sigma_post,
        "prior_weight": (prec_prior / prec_post) * 100,
        "ci_lower": mu_post - 1.96 * sigma_post,
        "ci_upper": mu_post + 1.96 * sigma_post,
    }

print(f"\n=== Bayesian Estimates by Flat Type ===")
print(
    f"{'Type':<12} {'n':>8} {'MLE Mean':>12} {'Post Mean':>12} {'Post σ':>10} {'Prior %':>8}"
)
print("─" * 70)
for ft, r in results_by_type.items():
    print(
        f"{ft:<12} {r['n']:>8,} ${r['mle_mean']:>10,.0f} "
        f"${r['posterior_mean']:>10,.0f} ${r['posterior_std']:>8,.2f} {r['prior_weight']:>7.3f}%"
    )
# INTERPRETATION: Flat types with fewer transactions (e.g. 2 ROOM, EXECUTIVE)
# show a larger prior weight — the posterior is pulled toward $500K more
# than for 4 ROOM which has abundant data. This is Bayesian regularisation
# in action: less data → prior matters more. For a property valuer,
# this means Executive flat estimates are more uncertain and more
# influenced by market assumptions than 4-room flat estimates.

# ── Checkpoint 10 ────────────────────────────────────────────────────
for ft, r in results_by_type.items():
    assert r["posterior_std"] < sigma_0, f"{ft}: posterior std should shrink from prior"
    assert 0 < r["prior_weight"] < 100, f"{ft}: prior weight must be between 0 and 100%"
print("\n✓ Checkpoint 10 passed — flat type posteriors all valid\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Visualise with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# -- Plot 1: Prior vs Posterior distributions (plotly direct) --
x_range = np.linspace(mu_0 - 3 * sigma_0, mu_0 + 3 * sigma_0, 500)
prior_pdf = stats.norm.pdf(x_range, mu_0, sigma_0)

x_posterior = np.linspace(mu_n - 5 * sigma_n, mu_n + 5 * sigma_n, 500)
posterior_pdf = stats.norm.pdf(x_posterior, mu_n, sigma_n)

fig1 = make_subplots(
    rows=1, cols=2, subplot_titles=["Prior Distribution", "Posterior Distribution"]
)
fig1.add_trace(
    go.Scatter(
        x=x_range, y=prior_pdf, name="Prior N(500K, 100K²)", line={"color": "blue"}
    ),
    row=1,
    col=1,
)
fig1.add_trace(
    go.Scatter(
        x=x_posterior,
        y=posterior_pdf,
        name=f"Posterior N({mu_n:,.0f}, {sigma_n:,.0f}²)",
        line={"color": "red"},
    ),
    row=1,
    col=2,
)
fig1.update_layout(title="Prior vs Posterior: 4-Room HDB Mean Price", height=400)
fig1.write_html("ex1_prior_posterior.html")
print("\nSaved: ex1_prior_posterior.html")

# -- Plot 2: Bootstrap distribution histogram --
fig2 = viz.histogram(
    bootstrap_means,
    title="Bootstrap Distribution of Sample Mean",
    x_label="Mean Price ($)",
)
fig2.write_html("ex1_bootstrap_distribution.html")
print("Saved: ex1_bootstrap_distribution.html")

# -- Plot 3: Beta-Binomial prior vs posterior --
x_beta = np.linspace(0, 1, 500)
beta_prior_pdf = stats.beta.pdf(x_beta, alpha_prior, beta_prior)
beta_post_pdf = stats.beta.pdf(x_beta, alpha_post_weak, beta_post_weak)

fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=x_beta,
        y=beta_prior_pdf,
        name="Prior Beta(2,2)",
        line={"color": "blue", "dash": "dash"},
    )
)
fig3.add_trace(
    go.Scatter(x=x_beta, y=beta_post_pdf, name="Posterior", line={"color": "red"})
)
fig3.add_vline(x=empirical_rate, line_dash="dot", annotation_text="Empirical rate")
fig3.update_layout(
    title="Beta-Binomial: P(price > $500K) Prior vs Posterior",
    xaxis_title="Proportion",
    yaxis_title="Density",
)
fig3.write_html("ex1_beta_binomial.html")
print("Saved: ex1_beta_binomial.html")

# -- Plot 4: Flat type comparison scatter --
if results_by_type:
    fig4 = viz.scatter(
        x=[r["mle_mean"] for r in results_by_type.values()],
        y=[r["posterior_mean"] for r in results_by_type.values()],
        title="MLE vs Posterior Mean by Flat Type",
        x_label="MLE Mean ($)",
        y_label="Posterior Mean ($)",
    )
    fig4.write_html("ex1_flat_type_comparison.html")
    print("Saved: ex1_flat_type_comparison.html")

# ── Checkpoint 11 ────────────────────────────────────────────────────
running_means = np.cumsum(bootstrap_means) / np.arange(1, n_bootstrap + 1)
assert (
    abs(running_means[-1] - mle_mean) < mle_se * 3
), "Bootstrap mean should converge to MLE mean"
print("\n✓ Checkpoint 11 passed — visualisations saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Business Interpretation Synthesis
# ══════════════════════════════════════════════════════════════════════
# Synthesize all findings into a stakeholder-ready summary.

print(f"\n=== Business Interpretation: Property Valuation Insights ===")
print(
    f"""
1. MARKET POSITION: The average 4-room HDB resale price is ${mle_mean:,.0f}
   with a standard error of ${mle_se:,.0f}. This estimate is very precise
   because it's based on {n:,} recent transactions.

2. BAYESIAN ESTIMATE: Even starting with a prior belief of $500K,
   the data overwhelms the prior — the posterior mean is ${mu_n:,.0f}.
   This means our data is strong enough that prior assumptions
   barely matter. For a property valuer, this is reassuring.

3. PRICE SEGMENTS: {empirical_rate:.1%} of 4-room transactions close
   above $500K. The Beta-Binomial model gives a precise credible
   interval of [{ci_weak[0]:.2%}, {ci_weak[1]:.2%}] for this rate.

4. FLAT TYPE VARIATION: Prior influence varies dramatically by
   segment. For abundant flat types (4-room), the prior contributes
   <0.01% — pure data-driven. For rare types (2-room, Executive),
   the prior contributes more, meaning market assumptions play a
   larger role in valuation.

5. SAMPLING BIAS WARNING: The friendship paradox applies to property
   markets too — properties that appear in many listings are not
   representative of the market. High-visibility properties skew
   our perception of "normal" prices upward.

6. INTERVAL METHODS: All four interval methods (Normal, Bootstrap,
   BCa, Bayesian) agree closely with n={n:,}. For smaller samples
   or skewed data, BCa bootstrap is recommended.
"""
)

# ── Checkpoint 12 ────────────────────────────────────────────────────
print("✓ Checkpoint 12 passed — business interpretation complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ Probability fundamentals: joint P(A,B), conditional P(A|B),
    independence test, cross-tabulation from real data
  ✓ Bayes' theorem: P(A|B) = P(B|A)P(A)/P(B) — medical test
    and property valuation applications
  ✓ MLE: μ̂ = x̄, σ̂² = (1/n)Σ(xᵢ - x̄)² — most efficient estimator
  ✓ Cramér-Rao bound: MLE achieves minimum possible variance
  ✓ Normal-Normal conjugate: prior + likelihood → posterior analytically
  ✓ Prior sensitivity: sweeping μ₀ and σ₀ shows when prior matters
  ✓ Beta-Binomial conjugate: natural model for proportions
  ✓ Credible vs confidence interval: philosophical and practical
    differences validated with 1,000 simulations
  ✓ Expected value: transaction-weighted vs simple average
  ✓ Sampling bias: friendship paradox shows why naive averages mislead
  ✓ Bootstrap CI: non-parametric alternative, BCa for small/skewed
  ✓ Business interpretation: translating statistics into decisions

  NEXT: In Exercise 2, you'll implement MLE from scratch using
  scipy.optimize.minimize, add a prior to get MAP estimation,
  explore the Central Limit Theorem through simulation, and
  diagnose three cases where MLE fails (small n, multimodal
  data, misspecified likelihood). Singapore GDP growth data.
"""
)

print("\n✓ Exercise 1 complete — Probability and Bayesian Thinking")
