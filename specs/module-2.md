### MODULE 2: Statistical Mastery for Machine Learning and Artificial Intelligence (AI) Success

**Description**: Statistical foundations including regression models. Following R5 Deck 3A: regression and logistic regression are taught as inference tools BEFORE the ML pipeline.

**Module Learning Objectives**: By the end of M2, students can:
- Reason about probability, conditional probability, and Bayes' theorem
- Estimate parameters using MLE and bootstrap methods
- Design and analyse A/B tests with proper hypothesis testing
- Build and interpret linear and logistic regression models
- Perform ANOVA for multi-group comparison
- Implement CUPED for variance reduction in experiments
- Engineer features and manage them in a feature store

**Kailash Engines**: ExperimentTracker, FeatureEngineer, FeatureStore, TrainingPipeline, ModelVisualizer

---

#### Lesson 2.1: Probability and Bayesian Thinking

**Prerequisites**: M1 complete (Python, data manipulation)
**Spectrum Position**: Uncertainty quantification — foundation for all statistical reasoning

**Topics**:
- Probability fundamentals: truth tables, composite events, P(A) + P(A') = 1
- Independent vs dependent events (steps to decide: order possible? one affects other?)
- Conditional probability: P(A|B), P(A,B) = P(A) x P(B|A)
- Joint probability, total probability (supermarket example, COVID ART example from Deck 2C)
- Bayes' theorem: P(T|P) = P(P|T) x P(T) / P(P)
- Distributions: Normal, Beta, Poisson, Exponential, Uniform (from Deck 2A "statistics theme park")
- Conjugate priors: Normal-Normal, Beta-Binomial
- Expected value: E[X] = Sum(p_i x x_i)
- Sampling bias (friendship paradox from Deck 2A: your friends have more friends than you)

**Key Formulas**:
- Bayes' theorem: P(A|B) = P(B|A) x P(A) / P(B)
- Expected value: E[X] = Sum(p_i * x_i)
- Conditional probability: P(A,B) = P(A) x P(B|A)

**Bridge from M1**: "In M1 you explored data and computed summary statistics. Now we ask: how confident are we in those numbers? Statistics gives us the language of uncertainty."

**Learning Objectives**: Students can:
- Construct truth tables for probability problems
- Apply Bayes' theorem to real-world scenarios (medical tests, A/B test results)
- Identify and correct for sampling bias
- Choose appropriate probability distributions for different data types

**Exercise**: Bayesian estimation on HDB prices — compute posterior distribution of mean price given prior beliefs and observed data. Visualise prior, likelihood, and posterior.

**Assessment Criteria**: Correct application of Bayes' theorem. Posterior plot shows update from prior. Interpretation explains what the posterior means for decision-making.

**R5 Source**: Deck 2A (expected value, sampling bias) + Deck 2C (conditional probability, Bayes' theorem, COVID example)

---

#### Lesson 2.2: Parameter Estimation and Inference

**Prerequisites**: 2.1 (probability, distributions)
**Spectrum Position**: Model fitting — connecting distributions to data

**Topics**:
- Population vs sample: parameter (mu, sigma) vs statistic (x-bar, s)
- Sampling distributions: taking many samples, calculating statistics
- Law of Large Numbers (LLN), Central Limit Theorem (CLT)
- Confidence intervals: what they mean (confidence to find population parameter in the range)
- PDF and CDF: histogram with area=1, cumulative probability
- Maximum Likelihood Estimation (MLE): write log-likelihood, optimise, interpret
- MAP estimation: MLE with prior (connects back to 2.1 Bayesian thinking)
- When MLE fails: small n, multimodal, misspecified

**Key Formulas**:
- Population variance: sigma^2 = Sum((x_i - mu)^2) / N
- Sample variance: s^2 = Sum((x_i - x_bar)^2) / (n-1) (Bessel's correction)
- Degrees of freedom: n-1 (one value determined by the sample mean)
- Log-likelihood: L(theta) = Sum(log(P(x_i | theta)))

**Learning Objectives**: Students can:
- Distinguish population parameters from sample statistics
- Compute confidence intervals and explain their meaning correctly
- Implement MLE for a simple distribution (e.g., Normal parameters)
- Explain when MLE fails and MAP helps

**Exercise**: Fit Normal and Exponential distributions to Singapore economic data. Implement MLE from scratch. Compare MLE vs MAP estimates.

**Assessment Criteria**: MLE implementation correct (log-likelihood maximised). CI computed and interpreted correctly (NOT "95% chance the parameter is in this range").

**R5 Source**: Deck 2A (population vs sample, Bessel's correction, degrees of freedom) + Deck 3A slides 5-10 (parameter estimation, optimal parameters)

---

#### Lesson 2.3: Bootstrapping and Hypothesis Testing

**Prerequisites**: 2.2 (sampling distributions, confidence intervals)
**Spectrum Position**: Resampling and decision-making — when theory isn't enough

**Topics**:
- **Bootstrapping**:
  - Problem: can't get more samples, costly, one sample available
  - Solution: resample with replacement (bootstrap sample), compute statistic (bootstrap replicate)
  - CI from bootstrap: percentile method (not t-distribution)
  - BCa intervals (bias-corrected and accelerated)
  - Parametric vs non-parametric bootstrap
- **Hypothesis Testing**:
  - Null (H0) vs Alternative (H1) hypothesis
  - p-value: conditional probability of observing data at least as extreme, given H0 is true
  - p-value is NOT P(H0 is true)
  - Significance levels: alpha = 0.1, 0.05, 0.01 (social science), 0.0005 (hard science)
  - "Reject or fail to reject" — never "accept" the null
  - Common test types: one-sample, two-sample, one-tailed, two-tailed
  - Permutation tests: resample assuming equal distribution, compute difference
  - Power analysis, minimum detectable effect (MDE)
  - Multiple testing: Bonferroni correction, BH-FDR

**Key Formulas**:
- Bootstrap CI: [percentile(alpha/2), percentile(1 - alpha/2)]
- Test statistic: T = (x_bar - mu_0) / (s / sqrt(n))
- Bonferroni correction: alpha_adjusted = alpha / m (m = number of tests)

**Learning Objectives**: Students can:
- Implement bootstrap resampling and compute CIs
- Formulate null and alternative hypotheses for business questions
- Compute and interpret p-values correctly
- Apply multiple testing corrections

**Exercise**: A/B test analysis with multiple corrections. Bootstrap CI for conversion rate difference. Permutation test for statistical significance.

**Assessment Criteria**: Bootstrap CI computed correctly. p-value interpreted correctly (not as "probability H0 is true"). Multiple testing correction applied.

**R5 Source**: Deck 3A slides 11-35 (bootstrapping, hypothesis testing, permutation)

---

#### Lesson 2.4: A/B Testing and Experiment Design

**Prerequisites**: 2.3 (hypothesis testing, p-values)
**Spectrum Position**: Experimental design — structured learning from interventions

**Topics**:
- A/B test design from Deck 3A:
  - Hypothesis formulation (e.g., "BOGO gives higher average spend")
  - Experiment setup: randomisation, equal allocation
  - Issues: overlapping treatments, prior treatment influence, temporal biases
- Data Collection Framework (Deck 3A, very detailed):
  - **Why**: hypotheses being tested, value and impact, performance measures
  - **What**: wish list of ideal information, budget/time constraints
  - **Where**: internal (CRM, POS, apps, finance) vs external (Kaggle, data.gov.sg, APIs)
  - **How**: at start (review, discover, validate), continuous (automated real-time/batch, human augmented)
  - **Frequency**: per transaction vs batch, collection vs analysis frequency, duration for statistical power
- Common problems: silos, red tape, incomplete/inconsistent data, aggregated vs transaction-level
- Clean DataOps architecture: data schema, data transfer, processing logic, connectors
- SRM (Sample Ratio Mismatch) detection

**Key Concepts**: Experiment design, data collection strategy, DataOps, SRM

**Learning Objectives**: Students can:
- Design an A/B test with proper randomisation and power analysis
- Create a data collection plan (what, where, how, frequency)
- Detect SRM in experiment data
- Identify common data collection pitfalls and mitigations

**Exercise**: Design and analyse a complete A/B experiment using ExperimentTracker. Create data collection plan. Detect SRM. Run hypothesis test. Report results with business recommendation.

**Assessment Criteria**: Experiment properly randomised. Power analysis done before experiment. SRM checked. Results interpreted with business context.

**R5 Source**: Deck 3A slides 15-26 (A/B testing hypothesis, data collection framework — very detailed, 12 slides)

---

#### Lesson 2.5: Linear Regression

**Prerequisites**: 2.2 (MLE, parameter estimation), 2.3 (hypothesis testing, p-values)
**Spectrum Position**: First predictive model — quantifying relationships

**Bridge**: "In M2.3 you tested whether a sample statistic differs from zero. Regression coefficients ARE sample statistics — the t-statistic you already know is the same test applied to each coefficient."

**Topics**:
- OLS (Ordinary Least Squares): fit a line, minimise squared residuals
- Regression formula: y = beta_0 + beta_1 * x_1 + ... + beta_n * x_n + epsilon
- Coefficients: direction (sign) and magnitude (size)
- Introducing non-linearity: squared terms (x^2), interaction terms (x1 * x2), loglinear (ln(y))
- **T-statistic**: coefficient / standard error. Tests H0: coefficient = 0.
  - Cutoffs: 90% (t > 1.6), 95% (t > 1.8), 99% (t > 1.97)
- **R-squared**: proportion of variance explained. Adjusted R-squared for multiple predictors.
- **F-statistic**: model vs intercept-only. p < 0.01 = model is better than random.
- Multivariate regression: ceteris paribus interpretation, "stripping out" effect
- Categorical variable encoding: dummy variables, base/benchmark category
  - Example from Deck 3A: salary = age + age^2 + female + transgender (male as base)
- Cross-validation basics: train/test split, k-fold (introduced, detailed in M3.2)

**Key Formulas**:
- OLS: minimise Sum((y_i - y_hat_i)^2)
- T-statistic: t = beta_hat / SE(beta_hat)
- R-squared: R^2 = 1 - SS_res / SS_tot
- F-statistic: F = (SS_reg / k) / (SS_res / (n - k - 1))

**Learning Objectives**: Students can:
- Build and interpret a multivariate linear regression
- Test coefficient significance using t-statistics
- Evaluate model fit using R-squared and F-statistic
- Handle categorical variables with proper dummy encoding
- Introduce non-linearity through interaction and polynomial terms

**Exercise**: Build HDB price prediction model. Engineer lat/lon features (from Deck 4A geocoding example). Interpret coefficients. Test significance. Evaluate fit. Add interaction terms and compare.

**Assessment Criteria**: Coefficients correctly interpreted (direction + magnitude + significance). Categorical encoding correct (base category identified). Model evaluated with R-squared and F-statistic.

**R5 Source**: Deck 3A slides 36-48 (comprehensive: OLS, t-stat, R-squared, F-stat, multivariate, categorical encoding, loglinear)

---

#### Lesson 2.6: Logistic Regression and Classification Foundations

**Prerequisites**: 2.5 (linear regression, coefficients, significance testing)
**Spectrum Position**: Binary prediction — from continuous to categorical outcomes

**Topics**:
- **Logistic Regression** (primary, ~3 hours):
  - When outcome is binary (0/1, yes/no): linear regression predicts outside [0,1]
  - Log-odds transformation: link function maps {0,1} to {-inf, +inf}
  - Sigmoid function: P(y=1|X) = 1 / (1 + exp(-z))
  - Odds interpretation: P(happening) / P(not happening). Example: 80%/20% = 4:1
  - MLE for logistic regression (not OLS — different optimisation)
  - Multiclass: one-vs-one, one-vs-rest, multinomial
  - Evaluation preview: accuracy, confusion matrix (detailed metrics in M3.5)
- **ANOVA** (survey, ~1 hour):
  - Generalises t-test to 3+ groups
  - One-way ANOVA: one explanatory variable with 3+ levels
  - Post-hoc tests: Tukey's HSD, Bonferroni, Scheffe
  - Two-way ANOVA and repeated measures: mention as extensions (reference material)

**Key Formulas**:
- Sigmoid: sigma(z) = 1 / (1 + exp(-z))
- Log-odds: log(P / (1-P)) = beta_0 + beta_1 * x_1 + ...
- Odds ratio: exp(beta_1) = multiplicative change in odds per unit change in x_1
- ANOVA F-statistic: F = MS_between / MS_within

**Design Note**: Logistic regression gets full depth because it's foundational for M3 classification and M4.8 neural networks (sigmoid is the activation function). ANOVA is scoped to one-way only because it's rarely used in the ML pipeline — students who need advanced ANOVA can reference supplementary material.

**Learning Objectives**: Students can:
- Build and interpret a logistic regression model
- Explain the sigmoid function and log-odds transformation
- Compute and interpret odds ratios
- Perform one-way ANOVA and interpret post-hoc tests
- Know when to use ANOVA vs t-test vs regression

**Exercise**: Build logistic regression for employee attrition prediction. Interpret odds ratios. Perform ANOVA on HDB prices across flat types (3-room, 4-room, 5-room). Apply Tukey's HSD.

**Assessment Criteria**: Logistic regression coefficients interpreted as odds ratios. Sigmoid function understood (not just used). ANOVA null hypothesis stated correctly.

**R5 Source**: Deck 3A slides 49-54 (logistic regression, ANOVA)

---

#### Lesson 2.7: CUPED and Causal Inference

**Prerequisites**: 2.4 (A/B testing), 2.5 (linear regression)
**Spectrum Position**: Advanced experiment analysis — beyond basic A/B testing

**Topics**:
- **CUPED** (Control Using Pre-Experiment Data) (~2.5 hours):
  - The single most impactful A/B test technique
  - Derive: Var(Y_adj) = Var(Y)(1 - rho^2)
  - Pre-experiment covariates: use historical data to reduce variance
  - Implementation: adjusted metric = Y - theta * X_pre
  - When CUPED helps most (high rho between pre and post metrics)
- **SRM Detection** (~30 min):
  - Sample Ratio Mismatch: expected 50/50 split, observed 52/48
  - Chi-squared test for SRM
  - Common causes: bot filtering, redirect bugs, population filtering
- **Difference-in-Differences** (~1 hour):
  - When randomisation isn't possible (observational data)
  - ATT (Average Treatment Effect on the Treated) derivation
  - Parallel trends assumption and how to test it
  - Placebo tests

**Key Formulas**:
- CUPED: Var(Y_adj) = Var(Y)(1 - rho^2)
- CUPED estimator: Y_adj = Y - theta * (X_pre - E[X_pre]), where theta = Cov(Y, X_pre) / Var(X_pre)
- DiD: ATT = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)

**Design Note**: Propensity score matching deferred to reference material (complex, less commonly used than DiD in practice). CUPED prioritised because every A/B testing practitioner should know it.

**Learning Objectives**: Students can:
- Implement CUPED to reduce A/B test variance by up to 50%
- Detect SRM in experiment data
- Apply DiD when randomisation is impossible
- Test the parallel trends assumption

**Exercise**: Implement CUPED on experiment data. Compare variance with and without CUPED adjustment. DiD analysis on Singapore cooling measures policy.

**Assessment Criteria**: CUPED reduces variance (quantified). SRM detection correctly identifies problematic experiments. DiD parallel trends tested before applying.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 2.8: Capstone — Statistical Analysis Project

**Prerequisites**: All of M2 (2.1-2.7)
**Spectrum Position**: Integration — applying all statistical tools to a real problem

**Topics**:
- End-to-end statistical analysis: load -> describe -> hypothesise -> test -> model -> interpret -> report
- Feature engineering preview: temporal features, interaction terms (bridges to M3.1)
- FeatureEngineer: generate features, select features
- FeatureStore: point-in-time correctness, data lineage, lifecycle
- Project options (student choice): wine quality analysis (from R5 PCML3-6), economic indicator prediction, experiment design and analysis

**Key Concepts**: Feature engineering as a discipline, feature store lifecycle, point-in-time correctness

**Learning Objectives**: Students can:
- Execute a complete statistical analysis from data to recommendations
- Engineer temporal and interaction features
- Store and retrieve features with point-in-time correctness
- Present statistical findings to a non-technical audience

**Exercise**: Choose one of 3 project options. Full pipeline: load data, explore, engineer features, store in FeatureStore, build regression or logistic model, test hypotheses, present findings.

**Assessment Criteria**: Complete pipeline executed. Features engineered with domain rationale. Model interpreted correctly. Report accessible to non-technical reader.

**R5 Source**: PCML3-6 "Putting It Together" (wine dataset) + ASCENT M2 ex_1/2/5

**End of Module Assessment**: Quiz + mini-project presentation.
