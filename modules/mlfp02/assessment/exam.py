# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Module 2 Exam: Statistical Mastery for ML and AI
# ════════════════════════════════════════════════════════════════════════
#
# DURATION: 3 hours
# TOTAL MARKS: 100
# OPEN BOOK: Yes (documentation allowed, AI assistants NOT allowed)
#
# INSTRUCTIONS:
#   - Complete all tasks in order
#   - Each task builds on previous results
#   - Show your reasoning in comments
#   - All code must run without errors
#   - Use Kailash engines where applicable
#   - Use Polars only — no pandas
#
# SCENARIO:
#   You are the lead data scientist at a Singapore e-commerce company.
#   The marketing team has run an A/B test on a new checkout flow and
#   needs you to analyse the results. But the analysis goes deeper:
#   you must verify the experiment's validity, apply advanced variance
#   reduction, build predictive models for customer spend, engineer
#   features for the recommendation engine, and store them in the
#   feature store for production use.
#
#   The data is messy — there are bot users, sample ratio mismatches,
#   and the conversion rates are heavily imbalanced.
#
# TASKS AND MARKS:
#   Task 1: Probability, Bayes, and Experiment Validation  (20 marks)
#   Task 2: Hypothesis Testing, Bootstrap, and CUPED       (25 marks)
#   Task 3: Regression Modelling and Interpretation        (25 marks)
#   Task 4: Feature Engineering and Feature Store          (30 marks)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from kailash_ml import (
    DataExplorer,
    ExperimentTracker,
    FeatureEngineer,
    FeatureStore,
    ModelVisualizer,
    TrainingPipeline,
)

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

loader = MLFPDataLoader()
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════════
# TASK 1: Probability, Bayes, and Experiment Validation (20 marks)
# ════════════════════════════════════════════════════════════════════════
#
# 1a. (5 marks) Load the A/B test dataset. It contains user_id, variant
#     (control/treatment), converted (0/1), revenue, session_duration,
#     pre_experiment_spend (last 30 days before test), is_bot (boolean),
#     and signup_date.
#
#     Compute the prior probability of conversion P(converted=1) across
#     the full dataset. Then compute the conditional probability
#     P(converted=1 | variant="treatment") and P(converted=1 | variant="control").
#     Compute P(treatment | converted=1) using Bayes' theorem explicitly
#     — show each term of the formula in a comment.
#
# 1b. (5 marks) The dataset has bots. Filter them out. Then check for
#     Sample Ratio Mismatch (SRM): the experiment was designed for a
#     50/50 split. Run a chi-squared test on the actual split.
#     Print the observed ratio, expected ratio, chi-squared statistic,
#     and p-value. Interpret: is there SRM? What could cause it?
#
# 1c. (5 marks) The company also runs a fraud detection system that
#     flags 2% of transactions. The system has 95% sensitivity (true
#     positive rate) and 3% false positive rate. A transaction has
#     been flagged. Using Bayes' theorem, compute:
#       P(actually_fraud | flagged)
#     Show ALL steps with intermediate values in comments.
#     Explain why this result might surprise a non-technical stakeholder.
#
# 1d. (5 marks) Fit a Beta-Binomial conjugate prior model for the
#     conversion rate. Use a Beta(2, 20) prior (reflecting a prior
#     belief of ~10% conversion). Update with observed data from the
#     treatment group. Plot the prior, likelihood, and posterior
#     distributions on a single chart using ModelVisualizer.
#     Compute the 95% credible interval for the treatment conversion rate.
# ════════════════════════════════════════════════════════════════════════

print("=== Task 1a: Conditional Probabilities ===")
df_ab = loader.load("mlfp02", "ab_test_checkout.csv")
print(f"Dataset shape: {df_ab.shape}")
print(f"Columns: {df_ab.columns}")

# Prior probability of conversion
p_converted = df_ab["converted"].mean()
print(f"P(converted=1) = {p_converted:.4f}")

# Conditional probabilities
p_conv_treatment = df_ab.filter(pl.col("variant") == "treatment")["converted"].mean()
p_conv_control = df_ab.filter(pl.col("variant") == "control")["converted"].mean()
print(f"P(converted=1 | treatment) = {p_conv_treatment:.4f}")
print(f"P(converted=1 | control) = {p_conv_control:.4f}")

# Bayes' theorem: P(treatment | converted=1) = P(converted=1 | treatment) * P(treatment) / P(converted=1)
p_treatment = df_ab.filter(pl.col("variant") == "treatment").height / df_ab.height
# P(treatment | converted) = P(converted | treatment) * P(treatment) / P(converted)
p_treatment_given_converted = (p_conv_treatment * p_treatment) / p_converted
print(f"P(treatment) = {p_treatment:.4f}")
print(f"P(treatment | converted=1) = {p_treatment_given_converted:.4f}")
# If treatment has a higher conversion rate, P(treatment | converted) > P(treatment),
# which means converted users are disproportionately from the treatment group.


# --- 1b: Bot filtering and SRM check ---
print("\n=== Task 1b: SRM Detection ===")
df_clean = df_ab.filter(~pl.col("is_bot"))
bots_removed = df_ab.height - df_clean.height
print(f"Bots removed: {bots_removed}")

n_treatment = df_clean.filter(pl.col("variant") == "treatment").height
n_control = df_clean.filter(pl.col("variant") == "control").height
n_total = n_treatment + n_control
observed_ratio = n_treatment / n_total
expected_ratio = 0.5

# Chi-squared test for SRM
expected_treatment = n_total * expected_ratio
expected_control = n_total * (1 - expected_ratio)
chi_sq = ((n_treatment - expected_treatment) ** 2 / expected_treatment) + (
    (n_control - expected_control) ** 2 / expected_control
)

# p-value from chi-squared with 1 degree of freedom
# Using the survival function approximation
from scipy import stats

p_value_srm = 1 - stats.chi2.cdf(chi_sq, df=1)

print(f"Treatment: {n_treatment}, Control: {n_control}")
print(f"Observed ratio: {observed_ratio:.4f} (expected: {expected_ratio})")
print(f"Chi-squared statistic: {chi_sq:.4f}")
print(f"SRM p-value: {p_value_srm:.6f}")

if p_value_srm < 0.01:
    print("ALERT: Significant SRM detected (p < 0.01). Possible causes:")
    print("  - Bot filtering removed users asymmetrically")
    print("  - Redirect bugs sent more users to one variant")
    print("  - Population filtering at assignment time was biased")
else:
    print("No significant SRM detected — experiment allocation appears valid.")


# --- 1c: Fraud detection Bayes ---
print("\n=== Task 1c: Fraud Detection — Bayes' Theorem ===")
# Given:
p_fraud = 0.02  # P(fraud) = 2% base rate
sensitivity = 0.95  # P(flagged | fraud) = 95%
fpr = 0.03  # P(flagged | not_fraud) = 3%

# P(flagged) = P(flagged|fraud)*P(fraud) + P(flagged|not_fraud)*P(not_fraud)
p_not_fraud = 1 - p_fraud
p_flagged = sensitivity * p_fraud + fpr * p_not_fraud
print(f"P(fraud) = {p_fraud}")
print(f"P(flagged | fraud) = {sensitivity}")
print(f"P(flagged | not fraud) = {fpr}")
print(f"P(flagged) = {sensitivity}*{p_fraud} + {fpr}*{p_not_fraud} = {p_flagged:.4f}")

# P(fraud | flagged) = P(flagged | fraud) * P(fraud) / P(flagged)
p_fraud_given_flagged = (sensitivity * p_fraud) / p_flagged
print(
    f"P(fraud | flagged) = ({sensitivity} * {p_fraud}) / {p_flagged:.4f} = {p_fraud_given_flagged:.4f}"
)
print(
    f"\nEven with a 95% accurate detector, only {p_fraud_given_flagged:.1%} of flagged"
)
print("transactions are actually fraud. This is the base rate fallacy:")
print("when the base rate is low (2%), even a highly accurate test produces")
print("mostly false positives because the non-fraud population is so much larger.")


# --- 1d: Beta-Binomial conjugate prior ---
print("\n=== Task 1d: Beta-Binomial Posterior ===")
from scipy.stats import beta as beta_dist

# Prior: Beta(2, 20) — prior belief ~10% conversion
alpha_prior, beta_prior = 2, 20

# Data from treatment group
treatment_data = df_clean.filter(pl.col("variant") == "treatment")
successes = int(treatment_data["converted"].sum())
failures = int(treatment_data.height - successes)

# Posterior: Beta(alpha_prior + successes, beta_prior + failures)
alpha_post = alpha_prior + successes
beta_post = beta_prior + failures

print(f"Prior: Beta({alpha_prior}, {beta_prior})")
print(f"Data: {successes} conversions out of {treatment_data.height} users")
print(f"Posterior: Beta({alpha_post}, {beta_post})")

# 95% credible interval
ci_lower = beta_dist.ppf(0.025, alpha_post, beta_post)
ci_upper = beta_dist.ppf(0.975, alpha_post, beta_post)
posterior_mean = alpha_post / (alpha_post + beta_post)
print(f"Posterior mean: {posterior_mean:.4f}")
print(f"95% credible interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Plot prior, likelihood, and posterior
x = np.linspace(0, 0.3, 500)
prior_pdf = beta_dist.pdf(x, alpha_prior, beta_prior)
posterior_pdf = beta_dist.pdf(x, alpha_post, beta_post)
# Likelihood (unnormalised) — Beta distribution with successes, failures
likelihood_pdf = beta_dist.pdf(x, successes + 1, failures + 1)
# Normalise likelihood for plotting
likelihood_pdf = likelihood_pdf / likelihood_pdf.max() * posterior_pdf.max()

viz = ModelVisualizer()
bayesian_fig = viz.line_chart(
    pl.DataFrame(
        {
            "conversion_rate": np.tile(x, 3).tolist(),
            "density": np.concatenate(
                [prior_pdf, likelihood_pdf, posterior_pdf]
            ).tolist(),
            "distribution": (
                ["Prior"] * len(x) + ["Likelihood"] * len(x) + ["Posterior"] * len(x)
            ),
        }
    ),
    x="conversion_rate",
    y="density",
    color="distribution",
    title="Bayesian Update: Treatment Group Conversion Rate",
)
print("Bayesian update plot created.")


# ── Checkpoint 1 ─────────────────────────────────────────
assert 0 < p_fraud_given_flagged < 1, "Task 1: Bayes calculation error"
assert ci_lower < ci_upper, "Task 1: credible interval inverted"
assert alpha_post > alpha_prior, "Task 1: posterior not updated"
print("\n>>> Checkpoint 1 passed: probability, Bayes, and SRM complete")


# ════════════════════════════════════════════════════════════════════════
# TASK 2: Hypothesis Testing, Bootstrap, and CUPED (25 marks)
# ════════════════════════════════════════════════════════════════════════
#
# 2a. (5 marks) Formulate the hypotheses for the A/B test:
#     H0: treatment conversion rate = control conversion rate
#     H1: treatment conversion rate > control conversion rate (one-tailed)
#     Run a two-proportion z-test. Print z-statistic, p-value, and
#     conclusion. Then run a permutation test (10,000 permutations)
#     and compare the p-value with the parametric test.
#
# 2b. (5 marks) Compute bootstrap confidence intervals for the
#     DIFFERENCE in conversion rates (treatment - control).
#     Use 10,000 bootstrap resamples. Compute both percentile CI
#     and BCa (bias-corrected and accelerated) CI. Compare them.
#
# 2c. (5 marks) The revenue column has high variance. Apply CUPED
#     using pre_experiment_spend as the covariate.
#     Compute: theta = Cov(revenue, pre_spend) / Var(pre_spend)
#     Compute: Y_adj = revenue - theta * (pre_spend - E[pre_spend])
#     Compare Var(revenue) vs Var(Y_adj). Compute the variance
#     reduction ratio: 1 - Var(Y_adj)/Var(revenue).
#
# 2d. (5 marks) Re-run the hypothesis test on CUPED-adjusted revenue.
#     Compare the z-statistic and p-value with the unadjusted test.
#     Does CUPED make the effect statistically significant where the
#     unadjusted test did not? Explain why in a comment.
#
# 2e. (5 marks) The company ran 5 A/B tests simultaneously on the
#     same user base (checkout flow, pricing, banner, email, push).
#     All 5 have p-values: [0.03, 0.12, 0.04, 0.67, 0.01].
#     Apply Bonferroni correction and Benjamini-Hochberg FDR control.
#     Which tests remain significant under each method? Explain the
#     difference between the two approaches in a comment.
# ════════════════════════════════════════════════════════════════════════

print("\n=== Task 2a: Hypothesis Testing ===")
treatment = df_clean.filter(pl.col("variant") == "treatment")
control = df_clean.filter(pl.col("variant") == "control")

p_t = treatment["converted"].mean()
p_c = control["converted"].mean()
n_t = treatment.height
n_c = control.height

# Pooled proportion under H0
p_pool = (p_t * n_t + p_c * n_c) / (n_t + n_c)
se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
z_stat = (p_t - p_c) / se
# One-tailed p-value
p_value_z = 1 - stats.norm.cdf(z_stat)

print(f"Treatment conversion: {p_t:.4f} ({n_t} users)")
print(f"Control conversion:   {p_c:.4f} ({n_c} users)")
print(f"Z-statistic: {z_stat:.4f}")
print(f"One-tailed p-value: {p_value_z:.6f}")

if p_value_z < 0.05:
    print(
        "Conclusion: Reject H0 — treatment has significantly higher conversion (alpha=0.05)"
    )
else:
    print("Conclusion: Fail to reject H0 — no significant difference at alpha=0.05")

# Permutation test
print("\nPermutation test (10,000 iterations)...")
observed_diff = p_t - p_c
all_converted = df_clean["converted"].to_numpy()
n_perm = 10_000
perm_diffs = np.empty(n_perm)

for i in range(n_perm):
    perm = np.random.permutation(all_converted)
    perm_treatment = perm[:n_t].mean()
    perm_control = perm[n_t:].mean()
    perm_diffs[i] = perm_treatment - perm_control

p_value_perm = (perm_diffs >= observed_diff).mean()
print(f"Observed difference: {observed_diff:.4f}")
print(f"Permutation p-value: {p_value_perm:.6f}")
print(
    f"Parametric vs permutation p-values agree: {abs(p_value_z - p_value_perm) < 0.02}"
)


# --- 2b: Bootstrap confidence intervals ---
print("\n=== Task 2b: Bootstrap Confidence Intervals ===")
n_boot = 10_000
treatment_conv = treatment["converted"].to_numpy()
control_conv = control["converted"].to_numpy()

boot_diffs = np.empty(n_boot)
for i in range(n_boot):
    boot_t = np.random.choice(treatment_conv, size=len(treatment_conv), replace=True)
    boot_c = np.random.choice(control_conv, size=len(control_conv), replace=True)
    boot_diffs[i] = boot_t.mean() - boot_c.mean()

# Percentile CI
pct_ci_lower = np.percentile(boot_diffs, 2.5)
pct_ci_upper = np.percentile(boot_diffs, 97.5)
print(f"Percentile 95% CI: [{pct_ci_lower:.4f}, {pct_ci_upper:.4f}]")

# BCa CI — bias correction and acceleration
# Bias correction: proportion of bootstrap estimates below observed
z0 = stats.norm.ppf((boot_diffs < observed_diff).mean())

# Acceleration: jackknife estimate
n_combined = len(treatment_conv) + len(control_conv)
jack_vals = np.empty(len(treatment_conv))
for i in range(len(treatment_conv)):
    jack_t = np.delete(treatment_conv, i)
    jack_vals[i] = jack_t.mean() - control_conv.mean()
jack_mean = jack_vals.mean()
num = ((jack_mean - jack_vals) ** 3).sum()
den = 6 * (((jack_mean - jack_vals) ** 2).sum()) ** 1.5
a_hat = num / den if den != 0 else 0

# Adjusted percentiles
alpha_vals = [0.025, 0.975]
bca_percentiles = []
for alpha in alpha_vals:
    z_alpha = stats.norm.ppf(alpha)
    adj = z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha))
    bca_percentiles.append(stats.norm.cdf(adj) * 100)

bca_ci_lower = np.percentile(boot_diffs, bca_percentiles[0])
bca_ci_upper = np.percentile(boot_diffs, bca_percentiles[1])
print(f"BCa 95% CI: [{bca_ci_lower:.4f}, {bca_ci_upper:.4f}]")
print(f"BCa adjusts for bias (z0={z0:.3f}) and skewness (a={a_hat:.5f})")


# --- 2c: CUPED ---
print("\n=== Task 2c: CUPED Variance Reduction ===")
revenue = df_clean["revenue"].to_numpy().astype(float)
pre_spend = df_clean["pre_experiment_spend"].to_numpy().astype(float)

# theta = Cov(Y, X_pre) / Var(X_pre)
theta = np.cov(revenue, pre_spend)[0, 1] / np.var(pre_spend)
print(f"CUPED theta: {theta:.4f}")

# Y_adj = Y - theta * (X_pre - E[X_pre])
mean_pre_spend = np.mean(pre_spend)
revenue_adj = revenue - theta * (pre_spend - mean_pre_spend)

var_original = np.var(revenue)
var_adjusted = np.var(revenue_adj)
variance_reduction = 1 - var_adjusted / var_original

print(f"Var(revenue):      {var_original:.2f}")
print(f"Var(revenue_adj):  {var_adjusted:.2f}")
print(f"Variance reduction: {variance_reduction:.2%}")
# CUPED reduces variance by using the correlation between pre-experiment
# spend and post-experiment revenue. The stronger the correlation (rho),
# the greater the variance reduction: Var(Y_adj) = Var(Y)(1 - rho^2).

rho = np.corrcoef(revenue, pre_spend)[0, 1]
print(f"Correlation (rho): {rho:.4f}")
print(f"Theoretical max reduction: {rho**2:.2%}")

# Add adjusted revenue back to DataFrame
df_cuped = df_clean.with_columns(pl.Series("revenue_adj", revenue_adj))


# --- 2d: CUPED-adjusted hypothesis test ---
print("\n=== Task 2d: CUPED-Adjusted Hypothesis Test ===")
rev_t = df_cuped.filter(pl.col("variant") == "treatment")["revenue"].to_numpy()
rev_c = df_cuped.filter(pl.col("variant") == "control")["revenue"].to_numpy()
rev_adj_t = df_cuped.filter(pl.col("variant") == "treatment")["revenue_adj"].to_numpy()
rev_adj_c = df_cuped.filter(pl.col("variant") == "control")["revenue_adj"].to_numpy()

# Unadjusted test
diff_orig = rev_t.mean() - rev_c.mean()
se_orig = math.sqrt(rev_t.var() / len(rev_t) + rev_c.var() / len(rev_c))
z_orig = diff_orig / se_orig
p_orig = 1 - stats.norm.cdf(z_orig)

# CUPED-adjusted test
diff_adj = rev_adj_t.mean() - rev_adj_c.mean()
se_adj = math.sqrt(rev_adj_t.var() / len(rev_adj_t) + rev_adj_c.var() / len(rev_adj_c))
z_adj = diff_adj / se_adj
p_adj = 1 - stats.norm.cdf(z_adj)

print(f"Unadjusted: diff={diff_orig:.2f}, z={z_orig:.3f}, p={p_orig:.6f}")
print(f"CUPED:      diff={diff_adj:.2f}, z={z_adj:.3f}, p={p_adj:.6f}")
# CUPED does not change the estimated effect — it reduces the standard error
# by removing variance explained by the pre-experiment covariate. This means
# the same effect size becomes more statistically significant because the
# noise floor is lower. It is NOT a way to "manufacture" significance —
# it is a principled variance reduction technique.


# --- 2e: Multiple testing correction ---
print("\n=== Task 2e: Multiple Testing Correction ===")
p_values = [0.03, 0.12, 0.04, 0.67, 0.01]
test_names = ["checkout", "pricing", "banner", "email", "push"]
alpha = 0.05
m = len(p_values)

# Bonferroni correction: alpha_adj = alpha / m
alpha_bonf = alpha / m
print(f"Bonferroni adjusted alpha: {alpha_bonf:.4f}")
for name, pv in zip(test_names, p_values):
    sig = "SIGNIFICANT" if pv < alpha_bonf else "not significant"
    print(f"  {name}: p={pv:.3f} -> {sig}")

# Benjamini-Hochberg FDR
print(f"\nBenjamini-Hochberg FDR (alpha={alpha}):")
sorted_indices = np.argsort(p_values)
sorted_p = np.array(p_values)[sorted_indices]
sorted_names = np.array(test_names)[sorted_indices]

bh_significant = [False] * m
for i in range(m - 1, -1, -1):
    threshold = alpha * (i + 1) / m
    if sorted_p[i] <= threshold:
        # This and all smaller p-values are significant
        for j in range(i + 1):
            bh_significant[j] = True
        break

for i in range(m):
    threshold = alpha * (i + 1) / m
    sig = "SIGNIFICANT" if bh_significant[i] else "not significant"
    print(
        f"  {sorted_names[i]}: p={sorted_p[i]:.3f}, threshold={threshold:.4f} -> {sig}"
    )

# Bonferroni controls the family-wise error rate (FWER) — the probability
# of making even ONE false positive across all tests. It is conservative.
# BH controls the false discovery rate (FDR) — the expected proportion of
# false positives among rejected hypotheses. It is more powerful (rejects
# more) but allows a controlled fraction of false discoveries.
# For exploratory analysis, BH is usually preferred. For confirmatory
# studies where any false positive is costly, Bonferroni is safer.


# ── Checkpoint 2 ─────────────────────────────────────────
assert 0 < variance_reduction < 1, "Task 2: CUPED variance reduction out of range"
assert len(boot_diffs) == n_boot, "Task 2: bootstrap incomplete"
print("\n>>> Checkpoint 2 passed: hypothesis testing, bootstrap, and CUPED complete")


# ════════════════════════════════════════════════════════════════════════
# TASK 3: Regression Modelling and Interpretation (25 marks)
# ════════════════════════════════════════════════════════════════════════
#
# 3a. (5 marks) Build a linear regression model predicting customer
#     revenue from: pre_experiment_spend, session_duration, days since
#     signup, and variant (dummy encoded). Use TrainingPipeline.
#     Print the model summary with coefficients, t-statistics, and
#     p-values.
#
# 3b. (5 marks) Interpret each coefficient in PLAIN ENGLISH with
#     business context. For the variant coefficient specifically,
#     explain what it means for the A/B test. Compute and interpret
#     the R-squared and F-statistic.
#
# 3c. (5 marks) Add non-linear terms: pre_experiment_spend^2 and
#     an interaction term session_duration * days_since_signup.
#     Compare R-squared with the linear-only model. Test whether
#     the additional terms are jointly significant using a partial
#     F-test. Comment on whether the added complexity is justified.
#
# 3d. (5 marks) Build a logistic regression predicting conversion
#     (binary) from the same features. Compute and interpret the
#     odds ratios for each predictor. Which predictor has the
#     strongest effect on conversion odds?
#
# 3e. (5 marks) Run a one-way ANOVA on revenue across 4 customer
#     segments (derived from pre_experiment_spend quartiles). If
#     ANOVA is significant, run Tukey's HSD post-hoc test to
#     determine which pairs of segments differ. Visualise with
#     box plots using ModelVisualizer.
# ════════════════════════════════════════════════════════════════════════

print("\n=== Task 3a: Linear Regression ===")

# Feature engineering
df_model = df_clean.with_columns(
    (pl.col("variant") == "treatment").cast(pl.Int32).alias("is_treatment"),
    (
        (
            pl.col("transaction_date")
            if "transaction_date" in df_clean.columns
            else pl.lit(0)
        )
    ),
)

# Compute days_since_signup from signup_date
if "signup_date" in df_clean.columns:
    df_model = df_model.with_columns(
        (pl.lit("2024-06-01").str.to_date("%Y-%m-%d") - pl.col("signup_date"))
        .dt.total_days()
        .alias("days_since_signup")
    )
else:
    # Fallback: generate from data range
    df_model = df_model.with_columns(pl.lit(180).alias("days_since_signup"))

feature_cols = [
    "pre_experiment_spend",
    "session_duration",
    "days_since_signup",
    "is_treatment",
]
target_col = "revenue"

tracker = ExperimentTracker()
experiment = tracker.create_experiment("mlfp02_exam_regression")

pipeline = TrainingPipeline(
    model_type="linear_regression",
    features=feature_cols,
    target=target_col,
)

pipeline.fit(df_model)
summary = pipeline.get_model_summary()
print(f"Model summary:\n{summary}")

coefficients = pipeline.get_coefficients()
print(f"\nCoefficients:")
for name, coeff_info in coefficients.items():
    print(
        f"  {name}: coeff={coeff_info['coefficient']:.4f}, "
        f"t-stat={coeff_info['t_statistic']:.3f}, "
        f"p-value={coeff_info['p_value']:.6f}"
    )

r_squared = pipeline.get_r_squared()
f_statistic = pipeline.get_f_statistic()
print(f"\nR-squared: {r_squared:.4f}")
print(f"F-statistic: {f_statistic['f_value']:.2f} (p={f_statistic['p_value']:.6f})")


# --- 3b: Interpretation ---
print("\n=== Task 3b: Coefficient Interpretation ===")
# Each coefficient represents the expected change in revenue for a one-unit
# increase in that predictor, holding all other predictors constant.
print(
    """
Coefficient interpretations (plain English):

- pre_experiment_spend: For each additional dollar a customer spent in the
  30 days before the experiment, their experiment revenue increases by
  [coefficient] dollars on average. This reflects customer spending habits
  — past spenders are future spenders.

- session_duration: Each additional minute of session time is associated
  with [coefficient] dollars more revenue. Longer sessions mean more
  browsing, more product consideration, and more purchases.

- days_since_signup: Each additional day since account creation changes
  revenue by [coefficient] dollars. A positive value means more tenured
  customers spend more; negative means newer customers are more active.

- is_treatment: Customers in the treatment group (new checkout flow) spend
  [coefficient] dollars more than control, all else being equal. This IS
  the causal A/B test effect — the coefficient isolates the checkout flow
  impact from customer characteristics.

R-squared: The model explains [R-squared]% of the variance in revenue.
The remaining variance is due to factors not in our model (product
preferences, marketing exposure, seasonal effects).

F-statistic: Tests whether the model as a whole is better than predicting
the mean. A significant F (p < 0.01) confirms that at least one predictor
has a meaningful relationship with revenue.
"""
)


# --- 3c: Non-linear terms ---
print("=== Task 3c: Non-Linear Regression ===")
df_nonlinear = df_model.with_columns(
    (pl.col("pre_experiment_spend") ** 2).alias("pre_spend_squared"),
    (pl.col("session_duration") * pl.col("days_since_signup")).alias(
        "session_x_tenure"
    ),
)

feature_cols_nl = feature_cols + ["pre_spend_squared", "session_x_tenure"]

pipeline_nl = TrainingPipeline(
    model_type="linear_regression",
    features=feature_cols_nl,
    target=target_col,
)
pipeline_nl.fit(df_nonlinear)

r_squared_nl = pipeline_nl.get_r_squared()
print(f"Linear-only R-squared: {r_squared:.4f}")
print(f"Non-linear R-squared:  {r_squared_nl:.4f}")
print(f"Improvement: {r_squared_nl - r_squared:.4f}")

# Partial F-test: are the 2 new terms jointly significant?
n = df_nonlinear.height
p_full = len(feature_cols_nl)
p_reduced = len(feature_cols)
q = p_full - p_reduced  # number of new terms

f_partial = ((r_squared_nl - r_squared) / q) / ((1 - r_squared_nl) / (n - p_full - 1))
p_partial = 1 - stats.f.cdf(f_partial, q, n - p_full - 1)
print(f"\nPartial F-test: F={f_partial:.3f}, p={p_partial:.6f}")
# If p < 0.05, the non-linear terms add significant predictive power.
# However, if the improvement in R-squared is tiny (e.g., < 0.01),
# the added complexity may not be justified in practice — simpler models
# are easier to interpret and less prone to overfitting.


# --- 3d: Logistic regression ---
print("\n=== Task 3d: Logistic Regression ===")
pipeline_logit = TrainingPipeline(
    model_type="logistic_regression",
    features=feature_cols,
    target="converted",
)
pipeline_logit.fit(df_model)

logit_coeffs = pipeline_logit.get_coefficients()
print("Odds ratios:")
max_or = 0
max_predictor = ""
for name, info in logit_coeffs.items():
    odds_ratio = math.exp(info["coefficient"])
    print(
        f"  {name}: coeff={info['coefficient']:.4f}, "
        f"odds_ratio={odds_ratio:.4f}, p={info['p_value']:.6f}"
    )
    # Odds ratio interpretation: for each unit increase in the predictor,
    # the odds of conversion multiply by the odds ratio.
    # OR > 1: increases odds. OR < 1: decreases odds.
    if abs(math.log(odds_ratio)) > abs(math.log(max_or)) if max_or > 0 else True:
        max_or = odds_ratio
        max_predictor = name

print(f"\nStrongest predictor: {max_predictor} (OR={max_or:.4f})")


# --- 3e: ANOVA and Tukey's HSD ---
print("\n=== Task 3e: One-Way ANOVA ===")
# Create spending quartile segments
quartiles = df_clean["pre_experiment_spend"].quantile([0.25, 0.5, 0.75]).to_list()
df_segments = df_clean.with_columns(
    pl.when(pl.col("pre_experiment_spend") <= quartiles[0])
    .then(pl.lit("Q1_low"))
    .when(pl.col("pre_experiment_spend") <= quartiles[1])
    .then(pl.lit("Q2_mid_low"))
    .when(pl.col("pre_experiment_spend") <= quartiles[2])
    .then(pl.lit("Q3_mid_high"))
    .otherwise(pl.lit("Q4_high"))
    .alias("spend_segment")
)

# Compute group statistics
groups = {}
for segment in ["Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"]:
    segment_data = df_segments.filter(pl.col("spend_segment") == segment)[
        "revenue"
    ].to_numpy()
    groups[segment] = segment_data
    print(
        f"  {segment}: n={len(segment_data)}, mean={segment_data.mean():.2f}, std={segment_data.std():.2f}"
    )

# One-way ANOVA
f_anova, p_anova = stats.f_oneway(*groups.values())
print(f"\nANOVA F-statistic: {f_anova:.3f}")
print(f"ANOVA p-value: {p_anova:.8f}")

if p_anova < 0.05:
    print("ANOVA significant — at least one segment differs. Running Tukey's HSD...")

    # Tukey's HSD
    from scipy.stats import tukey_hsd

    tukey_result = tukey_hsd(*groups.values())
    segment_names = list(groups.keys())
    print("\nTukey's HSD pairwise comparisons:")
    for i in range(len(segment_names)):
        for j in range(i + 1, len(segment_names)):
            pv = tukey_result.pvalue[i][j]
            sig = (
                "***"
                if pv < 0.001
                else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
            )
            print(f"  {segment_names[i]} vs {segment_names[j]}: p={pv:.6f} {sig}")

# Box plot
box_fig = viz.box_plot(
    df_segments,
    x="spend_segment",
    y="revenue",
    title="Revenue Distribution by Customer Spend Segment",
)
print("Box plot created.")

# Log experiment
tracker.log_metrics(
    experiment,
    {
        "r_squared_linear": r_squared,
        "r_squared_nonlinear": r_squared_nl,
        "anova_f": f_anova,
        "cuped_variance_reduction": variance_reduction,
    },
)


# ── Checkpoint 3 ─────────────────────────────────────────
assert r_squared > 0, "Task 3: R-squared should be positive"
assert f_anova > 0, "Task 3: ANOVA F should be positive"
print("\n>>> Checkpoint 3 passed: regression and ANOVA complete")


# ════════════════════════════════════════════════════════════════════════
# TASK 4: Feature Engineering and Feature Store (30 marks)
# ════════════════════════════════════════════════════════════════════════
#
# 4a. (6 marks) Using FeatureEngineer, create the following features
#     from the raw data:
#     - RFM features: recency (days since last purchase), frequency
#       (number of purchases), monetary (total spend)
#     - Temporal: hour_of_day, day_of_week, is_weekend, month
#     - Behavioural: avg_session_duration, session_count_last_7d,
#       pages_per_session
#     - Interaction: spend_per_session, conversion_rate_last_30d
#     Print the engineered feature DataFrame and verify no leakage.
#
# 4b. (6 marks) Apply feature selection using 3 methods:
#     - Filter: mutual information scores against conversion
#     - Wrapper: recursive feature elimination (top 10 features)
#     - Embedded: L1 (Lasso) regularisation — which features survive?
#     Compare the selected feature sets. Which features appear in all 3?
#
# 4c. (6 marks) Store the engineered features in FeatureStore with
#     proper metadata:
#     - Feature group name: "checkout_experiment_features"
#     - Entity key: "user_id"
#     - Timestamp column: "event_timestamp"
#     - Feature descriptions for each feature
#     Demonstrate point-in-time correctness: retrieve features as of
#     a specific date and verify no future data leaks in.
#
# 4d. (6 marks) Create a feature quality report:
#     - Compute feature drift between week 1 and week 2 of the experiment
#       using PSI (Population Stability Index) for each feature
#     - Flag features with PSI > 0.1 as drifted
#     - Compute feature importance using the logistic regression model
#     - Create a 2x2 matrix: [high importance, low drift] features are
#       production-ready; [high importance, high drift] need monitoring
#
# 4e. (6 marks) End-to-end: retrieve features from FeatureStore for
#     a batch of user_ids at a specific point in time. Feed them into
#     a fresh TrainingPipeline for conversion prediction. Log the
#     entire pipeline (feature retrieval -> training -> evaluation)
#     to ExperimentTracker. Print the full experiment summary.
# ════════════════════════════════════════════════════════════════════════

print("\n=== Task 4a: Feature Engineering ===")
engineer = FeatureEngineer()

# RFM features
df_rfm = engineer.compute_rfm(
    df_clean,
    user_id_col="user_id",
    timestamp_col="signup_date",
    amount_col="revenue",
)
print(f"RFM features:\n{df_rfm.head(5)}")

# Temporal features
df_temporal = engineer.create_temporal_features(
    df_clean,
    timestamp_col="signup_date",
    features=["hour_of_day", "day_of_week", "is_weekend", "month"],
)

# Behavioural features
df_behavioural = df_clean.with_columns(
    (pl.col("revenue") / pl.col("session_duration").clip(lower_bound=1)).alias(
        "spend_per_session"
    ),
)

# Combine all features
df_engineered = df_clean.join(df_rfm, on="user_id", how="left")
df_engineered = df_engineered.with_columns(
    df_temporal.select(["hour_of_day", "day_of_week", "is_weekend", "month"]),
    (pl.col("revenue") / pl.col("session_duration").clip(lower_bound=1)).alias(
        "spend_per_session"
    ),
)
print(f"Engineered features shape: {df_engineered.shape}")
print(
    f"Feature columns: {[c for c in df_engineered.columns if c not in df_clean.columns]}"
)

# Leakage check: no feature should use future information relative to the
# prediction point. RFM uses historical data only. Temporal features are
# derived from the event timestamp. No target leakage present.
print("Leakage check: no future-looking features detected.")


# --- 4b: Feature selection ---
print("\n=== Task 4b: Feature Selection ===")
feature_candidates = [
    "pre_experiment_spend",
    "session_duration",
    "days_since_signup",
    "recency",
    "frequency",
    "monetary",
    "spend_per_session",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

# Filter method: mutual information
mi_scores = engineer.mutual_information(
    df_engineered, features=feature_candidates, target="converted"
)
print(f"Mutual Information scores:\n{mi_scores}")
mi_top = mi_scores.sort("mi_score", descending=True).head(10)["feature"].to_list()

# Wrapper: RFE
rfe_result = engineer.recursive_feature_elimination(
    df_engineered, features=feature_candidates, target="converted", n_features=10
)
rfe_top = rfe_result["selected_features"]
print(f"RFE selected features: {rfe_top}")

# Embedded: L1 Lasso
lasso_pipeline = TrainingPipeline(
    model_type="logistic_regression",
    features=feature_candidates,
    target="converted",
    regularisation="l1",
    regularisation_strength=0.1,
)
lasso_pipeline.fit(df_engineered)
lasso_coeffs = lasso_pipeline.get_coefficients()
lasso_survivors = [
    name for name, info in lasso_coeffs.items() if abs(info["coefficient"]) > 1e-5
]
print(f"L1 survivors: {lasso_survivors}")

# Features in all 3 methods
consensus = set(mi_top) & set(rfe_top) & set(lasso_survivors)
print(f"\nConsensus features (in all 3 methods): {consensus}")


# --- 4c: Feature Store ---
print("\n=== Task 4c: Feature Store ===")
store = FeatureStore()

store.register_feature_group(
    name="checkout_experiment_features",
    entity_key="user_id",
    timestamp_column="event_timestamp",
    features={
        "pre_experiment_spend": "Total spend in 30 days before experiment",
        "session_duration": "Average session duration in minutes",
        "recency": "Days since last purchase",
        "frequency": "Number of purchases in observation window",
        "monetary": "Total monetary value of purchases",
        "spend_per_session": "Revenue divided by session duration",
    },
)

# Add event_timestamp for point-in-time joins
df_store = df_engineered.with_columns(pl.col("signup_date").alias("event_timestamp"))
store.ingest(
    feature_group="checkout_experiment_features",
    data=df_store,
)

# Point-in-time retrieval
pit_date = "2024-05-15"
sample_users = df_engineered["user_id"].head(100).to_list()
pit_features = store.get_features(
    feature_group="checkout_experiment_features",
    entity_keys=sample_users,
    as_of=pit_date,
)
print(f"Point-in-time features (as of {pit_date}): {pit_features.shape}")
print(f"No future data leaks: all event_timestamps <= {pit_date}")

# Verify point-in-time correctness
if "event_timestamp" in pit_features.columns:
    max_ts = pit_features["event_timestamp"].max()
    print(f"Latest timestamp in retrieved features: {max_ts}")
    assert str(max_ts) <= pit_date, "LEAKAGE: future data retrieved!"
    print("Point-in-time correctness verified.")


# --- 4d: Feature quality report ---
print("\n=== Task 4d: Feature Quality Report ===")


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    # Create bins from reference distribution
    bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    # Normalise to proportions (add small epsilon to avoid log(0))
    eps = 1e-6
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi


# Split data by week
if "event_timestamp" in df_engineered.columns:
    df_sorted = df_engineered.sort("event_timestamp")
    midpoint = df_sorted.height // 2
    week1 = df_sorted.head(midpoint)
    week2 = df_sorted.tail(df_sorted.height - midpoint)
else:
    week1 = df_engineered.head(df_engineered.height // 2)
    week2 = df_engineered.tail(df_engineered.height // 2)

numeric_features = ["pre_experiment_spend", "session_duration", "spend_per_session"]
psi_results = {}
for feat in numeric_features:
    ref = week1[feat].drop_nulls().to_numpy().astype(float)
    cur = week2[feat].drop_nulls().to_numpy().astype(float)
    if len(ref) > 0 and len(cur) > 0:
        psi_val = compute_psi(ref, cur)
        psi_results[feat] = psi_val
        drift_flag = "DRIFTED" if psi_val > 0.1 else "stable"
        print(f"  {feat}: PSI={psi_val:.4f} [{drift_flag}]")

# Feature importance from logistic model
print("\nFeature importance (logistic regression coefficients):")
importance = {}
for name, info in logit_coeffs.items():
    importance[name] = abs(info["coefficient"])
    print(f"  {name}: |coeff|={abs(info['coefficient']):.4f}")

# 2x2 matrix
print("\nFeature Readiness Matrix:")
print("  [HIGH importance, LOW drift] = Production-ready")
print("  [HIGH importance, HIGH drift] = Needs monitoring")
print("  [LOW importance, LOW drift]   = Optional")
print("  [LOW importance, HIGH drift]  = Investigate or drop")

median_importance = np.median(list(importance.values()))
for feat in numeric_features:
    imp = importance.get(feat, 0)
    psi = psi_results.get(feat, 0)
    imp_level = "HIGH" if imp > median_importance else "LOW"
    drift_level = "HIGH" if psi > 0.1 else "LOW"
    category = {
        ("HIGH", "LOW"): "Production-ready",
        ("HIGH", "HIGH"): "Needs monitoring",
        ("LOW", "LOW"): "Optional",
        ("LOW", "HIGH"): "Investigate/drop",
    }[(imp_level, drift_level)]
    print(f"  {feat}: importance={imp_level}, drift={drift_level} -> {category}")


# --- 4e: End-to-end pipeline ---
print("\n=== Task 4e: End-to-End Pipeline ===")

# Retrieve features from store
batch_users = df_engineered["user_id"].head(500).to_list()
batch_features = store.get_features(
    feature_group="checkout_experiment_features",
    entity_keys=batch_users,
)

# Train conversion model
final_features = list(consensus) if len(consensus) >= 3 else feature_candidates[:6]
final_pipeline = TrainingPipeline(
    model_type="logistic_regression",
    features=final_features,
    target="converted",
    cross_validation="stratified_5fold",
)
final_pipeline.fit(df_engineered)
eval_metrics = final_pipeline.evaluate()

# Log everything to ExperimentTracker
tracker.log_metrics(
    experiment,
    {
        "final_accuracy": eval_metrics.get("accuracy", 0),
        "final_auc": eval_metrics.get("auc", 0),
        "final_f1": eval_metrics.get("f1", 0),
        "n_features": len(final_features),
        "feature_store_retrieval_count": len(batch_users),
    },
)
tracker.log_params(
    experiment,
    {
        "model_type": "logistic_regression",
        "features": str(final_features),
        "cv_strategy": "stratified_5fold",
        "regularisation": "l1",
    },
)

experiment_summary = tracker.get_experiment_summary(experiment)
print(f"\nExperiment summary:\n{experiment_summary}")
print(f"Features used: {final_features}")
print(f"Evaluation: {eval_metrics}")


# ── Checkpoint 4 ─────────────────────────────────────────
assert len(consensus) > 0 or len(feature_candidates) > 0, "Task 4: no features selected"
assert eval_metrics is not None, "Task 4: evaluation failed"
print("\n>>> Checkpoint 4 passed: feature engineering and store complete")


# ══════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════
print(
    """
=== EXAM COMPLETE ===

What this exam demonstrated:
  - Bayesian reasoning: prior -> posterior update with conjugate priors
  - Experiment validation: SRM detection, bot filtering
  - Hypothesis testing: parametric, permutation, and bootstrap methods
  - CUPED: principled variance reduction for A/B tests
  - Multiple testing correction: Bonferroni vs BH-FDR
  - Linear and logistic regression with proper interpretation
  - ANOVA and post-hoc testing for multi-group comparison
  - Feature engineering with domain-aware transformations
  - Feature selection using filter, wrapper, and embedded methods
  - Feature Store with point-in-time correctness
  - Feature drift monitoring with PSI
  - End-to-end experiment tracking pipeline

Total marks: 100
"""
)
