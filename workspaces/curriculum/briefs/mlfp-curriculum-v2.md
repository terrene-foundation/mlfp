# MLFP Curriculum v2 — Authoritative Handbook (6 Modules x 8 Lessons)

**Official name**: ML Foundations for Professionals — Terrene Open Academy
**Version**: v2 (supersedes v1). Incorporates red-team findings and completeness audit.
**Audience**: Working professionals with ZERO Python to production ML engineering
**Structure**: 6 modules x 8 lessons = 48 lessons (~4 hours each, ~192 contact hours)
**Certification**: Foundation Certificate (M1-M4, ~128h) + Advanced Certificate (M5-M6, ~64h)
**Stack**: Kailash SDK exclusively. No PyCaret, no ydata_profiling, no broken dependency chains.

---

## Part I: Design Principles

### 1. Kailash-Only Stack

Every ML operation uses Kailash engines. No PyCaret (broken deps, slow install). No ydata_profiling (conflicts). No raw sklearn/pandas/PyTorch without Kailash wrapper.

| Instead of | Use |
|---|---|
| ydata_profiling / Sweetviz | DataExplorer (8 alert types, profiling, compare) |
| PyCaret / AutoML | AutoMLEngine, TrainingPipeline |
| pandas | polars |
| sklearn Pipeline | PreprocessingPipeline |
| MLflow | ExperimentTracker + ModelRegistry |
| CrewAI / LangChain | Kaizen (Delegate, BaseAgent, Signature) |

### 2. Python Through Data

Every Python concept grounded in data manipulation from Lesson 1.1. No abstract exercises. Students never learn a `for` loop in isolation — they learn it by iterating over data.

### 3. Engines Before Infrastructure

Students use Kailash engines (DataExplorer, ModelVisualizer) in M1 before learning workflow orchestration (WorkflowBuilder, custom nodes) in M3. Motivation before machinery.

### 4. The Feature Engineering Spectrum

The organising spine of the entire curriculum (from R5 Deck 5B "Unsupervised meets Supervised Learning"):

```
M3:     Manual feature engineering — human designs features from domain knowledge
M4.1-6: USML discovers features independently — no labels, no error signal
M4.7:   Collaborative filtering learns embeddings via optimisation — THE PIVOT
M4.8:   DL generalises — hidden layers are automated feature engineering WITH error feedback
        2+ hidden layers can write any non-linear combination (representation learning)
        Node values = embeddings
M5:     Specialised architectures learn domain-specific features (vision, sequence, graph, generative)
M6:     LLMs learn semantic features from language at scale
```

### 5. Statistics Teaches Models, SML Teaches the Pipeline

Following R5 delivery (Deck 3A): regression, logistic regression, and ANOVA are taught as inferential statistics (M2) BEFORE the ML pipeline (M3). M3 does NOT re-teach these models — it builds on them with advanced ensembles and production engineering.

### 6. Engineering, Not Philosophy

Every concept is taught through implementation. Governance is access controls you code (PACT), not frameworks you discuss. Alignment is a training loop you run (DPO/GRPO), not a position paper you read.

### 7. Progressive Scaffolding

| Module | Scaffolding | Code Provided | Audience State |
|---|---|---|---|
| M1 | Heavy | ~70% | Zero Python |
| M2 | Moderate+ | ~60% | Can wrangle data |
| M3 | Moderate | ~50% | Understands statistics |
| M4 | Light+ | ~40% | Can build supervised models |
| M5 | Light | ~30% | Understands USML + DL basics |
| M6 | Minimal | ~20% | Fluent in DL architectures |

### 8. Three Layers Per Concept

Every concept taught at three depths simultaneously:

| Layer | Marker | Audience | How to Use |
|---|---|---|---|
| Intuition | FOUNDATIONS | Zero-background | Plain language, analogies, visualisation |
| Mathematics | THEORY | Intermediate | Derivation, step-by-step formula |
| Research | ADVANCED | Masters+ / PhD | Paper reference, frontier result |

A banker and a PhD sit in the same classroom. Both leave having learned something they didn't know.

---

## Part II: Module Specifications

---

### MODULE 1: Machine Learning Data Pipelines and Visualisation Mastery with Python

**Description**: Zero to productive. Learn Python by exploring real Singapore data. Every Python concept grounded in a data task.

**Module Learning Objectives**: By the end of M1, students can:
- Write Python programs with variables, functions, loops, conditionals, and collections
- Load, filter, join, aggregate, and transform datasets using Polars
- Create interactive visualisations using Plotly via ModelVisualizer
- Profile datasets automatically and detect data quality issues
- Build end-to-end data pipelines: load, profile, clean, visualise, report

**Kailash Engines**: DataExplorer, PreprocessingPipeline, ModelVisualizer

---

#### Lesson 1.1: Your First Data Exploration

**Prerequisites**: None (first lesson)
**Spectrum Position**: Data acquisition — before features exist

**Topics**:
- Python basics: variables, data types (int, float, str, bool), `print()`, f-strings
- Polars: `pl.read_csv()`, `df.shape`, `df.columns`, `df.head()`, `df.describe()`
- First 90 min: Python REPL basics. Second 90 min: load and explore real data.

**Key Concepts**: Variable assignment, data types, string formatting, DataFrame as the fundamental data structure

**Learning Objectives**: Students can:
- Assign variables, perform arithmetic, format output strings
- Load a CSV file into a Polars DataFrame
- Inspect shape, columns, head, and summary statistics

**Exercise**: Load Singapore weather CSV (~1K rows). Print shape, column names, first 5 rows, summary statistics. Answer 3 data questions using only `describe()` output.

**Assessment Criteria**: Code runs without error. All 3 questions answered correctly with evidence from data.

**R5 Source**: Deck 1B (data types, operators, variables) + PCML1-1

---

#### Lesson 1.2: Filtering and Transforming Data

**Prerequisites**: 1.1 (variables, DataFrame basics)
**Spectrum Position**: Data selection — choosing relevant observations

**Topics**:
- Booleans and comparison operators (`>`, `<`, `==`, `!=`)
- Polars expressions: `pl.col()`, `filter()`, `select()`, `sort()`, `with_columns()`
- Method chaining (fluent API style)

**Note**: Students use boolean expressions within Polars (declarative). Python `if/else` is deferred to 1.4. Add forward-reference: "You are writing expressions that evaluate to True/False. Python also has `if/else` for code-level decisions — Lesson 1.4."

**Key Concepts**: Boolean logic, expression-based filtering, method chaining, column transformation

**Learning Objectives**: Students can:
- Filter rows by one or more conditions
- Select specific columns
- Sort data by any column
- Create new computed columns

**Exercise**: Filter HDB resale data by town, price range, and date. Create a new column (price per sqm). Sort by price descending.

**Assessment Criteria**: Correct filters applied. New column computed correctly. Output sorted.

**R5 Source**: Deck 1B (comparison operators) + PCML1-2

---

#### Lesson 1.3: Functions and Aggregation

**Prerequisites**: 1.2 (filtering, expressions)
**Spectrum Position**: Data summarisation — compressing information

**Topics**:
- `def` functions, parameters, `return` statements
- `for` loops, lists, dictionaries
- Polars: `group_by()`, `agg()`, `pl.mean()`, `pl.sum()`, `pl.count()`
- Writing helper functions for reusable analysis

**Key Concepts**: Function abstraction, iteration, collection types, grouped aggregation

**Learning Objectives**: Students can:
- Define functions that accept parameters and return values
- Use loops to process collections
- Aggregate data by groups (mean, sum, count per category)
- Write reusable helper functions for common data tasks

**Exercise**: Write functions to compute district-level statistics (mean price, transaction count, price range) for HDB data. Use `group_by` + `agg` to produce summary table.

**Assessment Criteria**: Functions are reusable (not hardcoded). Aggregation produces correct grouped results.

**R5 Source**: Deck 1B (functions, collections) + PCML1-3

---

#### Lesson 1.4: Joins and Multi-Table Data

**Prerequisites**: 1.3 (functions, collections, aggregation)
**Spectrum Position**: Data integration — combining information sources

**Topics**:
- `if/else/elif` conditional statements
- `import` and packages
- Join concepts: left, inner, outer. When to use each.
- Polars: `join()`, multi-table operations on HDB 15M rows
- Dictionary lookups and mapping

**Key Concepts**: Conditional logic, package imports, relational joins, multi-source data integration

**Learning Objectives**: Students can:
- Write conditional logic for branching decisions
- Import and use external packages
- Join multiple DataFrames on shared keys
- Reason about which join type to use for a given task

**Exercise**: Join HDB resale data with MRT station data and school data. Compute distance-to-amenity features. Handle missing joins with appropriate join type.

**Assessment Criteria**: Correct join type selected. Missing data handled (not silently dropped). Combined dataset has expected row count.

**R5 Source**: Deck 2B (merging: join, merge, concat) + ASCENT M1 ex_1 (joins portion)

---

#### Lesson 1.5: Window Functions and Trends

**Prerequisites**: 1.4 (joins, conditionals)
**Spectrum Position**: Temporal feature creation — extracting time-based patterns

**Topics**:
- Polars window functions: `over()`, `rolling_mean()`, `shift()`
- Rolling aggregations, YoY calculations, moving averages
- Lazy frames (introduced as performance optimisation, not core concept): `scan_csv()`, `collect()`

**Note**: Lazy frames are a "make it faster" add-on, not a prerequisite. Students can complete all exercises with eager evaluation.

**Key Concepts**: Window functions, rolling statistics, temporal trends, lazy evaluation

**Learning Objectives**: Students can:
- Compute rolling averages and YoY changes
- Use window functions for within-group calculations
- Identify trends and seasonality in time-series data
- Understand when lazy evaluation helps performance

**Exercise**: Compute 3-month rolling average of HDB prices per district. Calculate YoY price change. Identify districts with highest/lowest growth trends.

**Assessment Criteria**: Rolling calculations correct. YoY computed with proper time alignment.

**R5 Source**: ASCENT M1 ex_1 (windows portion)

---

#### Lesson 1.6: Data Visualisation

**Prerequisites**: 1.5 (aggregation, trends)
**Spectrum Position**: Data communication — making patterns visible

**Topics**:
- Visualisation principles from Deck 1C:
  - Why visualise (tables vs charts for decision-making)
  - Attributes of good charts: simple, clean, subtle attention, truthful
  - Gestalt principles: proximity, similarity, closure, enclosure, continuity, connection
  - Visual order: Z-pattern reading (left-to-right, top-to-bottom)
  - Charts to avoid: 3D charts, misleading pie charts
- Chart selection by data type:
  - Heatmaps (correlation), line charts (time-series), vertical bars (comparison), horizontal bars (categorical)
  - Stacked bars (composition), 100% stacked (Likert/survey data)
- Plotly Express / ModelVisualizer for interactive charts

**Key Formulas**: None (visual design principles, not mathematical)

**Learning Objectives**: Students can:
- Select the appropriate chart type for a given data question
- Create interactive visualisations with Plotly via ModelVisualizer
- Apply Gestalt principles to improve chart readability
- Identify and avoid misleading chart designs

**Exercise**: Create 6 different chart types from HDB data: heatmap (price correlation), line (price trend), bar (district comparison), scatter (size vs price), histogram (price distribution), stacked bar (flat type composition).

**Assessment Criteria**: Chart type appropriate for each question. No misleading axes. Interactive features (hover, zoom) functional.

**R5 Source**: Deck 1C (30 slides on viz principles, chart types, Plotly)

---

#### Lesson 1.7: Automated Data Profiling

**Prerequisites**: 1.6 (visualisation)
**Spectrum Position**: Automated data assessment — machine-detected quality issues

**Topics**:
- DataExplorer: automated profiling with 8 alert types
- AlertConfig: configure thresholds for missing values, outliers, duplicates, skew, correlation, cardinality, constants, type inference
- DataProfile object: access profiling results programmatically
- `compare()`: compare two datasets (before/after cleaning, train/test distributions)
- Classes as users (not authors): students use DataExplorer, not build it
- `try/except` basics for error handling
- Async hidden behind `shared.run_profile()` sync wrapper

**Key Concepts**: Automated data quality assessment, alert configuration, dataset comparison

**Learning Objectives**: Students can:
- Run automated data profiling on any dataset
- Configure alert thresholds for data quality rules
- Compare two datasets and identify distribution differences
- Handle errors gracefully with try/except

**Exercise**: Profile dirty economic indicators dataset. Identify issues (missing values, outliers, skew). Configure alerts. Compare original vs cleaned version.

**Assessment Criteria**: All data quality issues identified. Alerts configured with sensible thresholds. Comparison shows improvement.

**R5 Source**: ASCENT M1 ex_3

---

#### Lesson 1.8: Data Pipelines and End-to-End Project

**Prerequisites**: All of M1 (1.1-1.7)
**Spectrum Position**: Complete data pipeline — acquisition to report

**Topics**:
- None/null handling: `is_null()`, `fill_null()`, `drop_nulls()`
- ETL concepts: Extract (APIs, files), Transform (clean, encode, scale), Load (output)
- REST APIs: GET, POST, JSON responses, query parameters (OneMap Singapore example from Deck 1C)
- PreprocessingPipeline: auto-detect data types, encode categoricals, scale numerics, impute missing values
- Full pipeline: load -> profile -> clean -> visualise -> report
- Project structure: modules, imports, putting it all together

**Key Concepts**: ETL pipeline, data cleaning automation, API data extraction, preprocessing pipeline

**Learning Objectives**: Students can:
- Build a complete data pipeline from raw data to clean output
- Extract data from REST APIs
- Use PreprocessingPipeline for automated cleaning
- Structure a multi-file Python project

**Exercise**: Build full EDA pipeline for messy taxi trip data: load from API -> profile with DataExplorer -> clean with PreprocessingPipeline -> visualise key patterns -> generate HTML report.

**Assessment Criteria**: Pipeline runs end-to-end. Data quality improved (fewer missing values, outliers handled). Report contains at least 3 visualisations with insights.

**R5 Source**: Deck 1C (APIs, REST) + PCML1-5 (ETL dashboard) + ASCENT M1 ex_5

**End of Module Assessment**: Quiz (AI-resilient, context-specific questions on data types, Polars operations, viz principles, ETL concepts).

---

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

---

### MODULE 3: Supervised Machine Learning for Building and Deploying Models

**Description**: The ML pipeline — from feature engineering to production deployment. Builds on M2's regression foundation. Following R5 Deck 4A: focus on the PIPELINE and advanced models, not re-teaching basic regression.

**Module Learning Objectives**: By the end of M3, students can:
- Engineer features and select the most predictive ones
- Explain bias-variance tradeoff and apply regularisation
- Train and evaluate the complete supervised model zoo (linear, SVM, KNN, Naive Bayes, trees, forests, gradient boosting)
- Handle class imbalance and calibrate probabilistic predictions
- Interpret models using SHAP, LIME, and fairness metrics
- Orchestrate ML workflows with custom nodes
- Deploy production pipelines with model registry, drift monitoring, and DataFlow persistence

**Kailash Engines**: FeatureEngineer, FeatureStore, PreprocessingPipeline, TrainingPipeline, AutoMLEngine, HyperparameterSearch, ModelRegistry, EnsembleEngine, WorkflowBuilder, DataFlow, DriftMonitor, ModelVisualizer

---

#### Lesson 3.1: Feature Engineering, ML Pipeline, and Feature Selection

**Prerequisites**: M2 complete (statistics, regression)
**Spectrum Position**: Manual feature engineering — human designs features from domain knowledge

**Topics**:
- Feature engineering philosophy from Deck 4A: "Data > Models > Hyperparameter Tuning"
  - Geocoding example: address -> lat/lon via OneMap API
  - Domain knowledge drives feature creation
- ML pipeline stages: data ingestion -> preprocessing -> feature engineering -> model selection -> training/eval -> hyperparameters -> deployment
- Statistics vs ML distinction (from Deck 4A): explaining the past vs predicting the future
- Temporal features: lag, rolling mean, day-of-week, month, season
- Interaction terms, polynomial features
- Leakage detection: features that would not be available at prediction time
- **Feature Selection Methods** (new, not in R5):
  - Filter: mutual information, chi-squared, correlation thresholds
  - Wrapper: forward selection, backward elimination, RFE (Recursive Feature Elimination)
  - Embedded: L1 sparsity (Lasso), tree-based importance
  - FeatureEngineer: generate + select

**Key Concepts**: Domain-driven feature engineering, ML pipeline stages, feature selection taxonomy, leakage

**Learning Objectives**: Students can:
- Engineer features from domain knowledge (not just data manipulation)
- Identify and prevent feature leakage
- Apply filter, wrapper, and embedded feature selection methods
- Explain why data quality matters more than model complexity

**Exercise**: Engineer features for HDB price prediction (geocode addresses, create temporal features, interaction terms). Apply 3 feature selection methods and compare selected features.

**Assessment Criteria**: Features have domain rationale. No leakage. Feature selection methods compared with rationale for final selection.

**R5 Source**: Deck 4A (feature engineering, ML pipeline, statistics vs ML)

---

#### Lesson 3.2: Bias-Variance, Regularisation, and Cross-Validation

**Prerequisites**: 2.5 (linear regression), 3.1 (feature engineering)
**Spectrum Position**: Model complexity control — the fundamental ML tradeoff

**Topics**:
- Bias-variance decomposition: E[(y - y_hat)^2] = Bias^2 + Variance + sigma^2
  - Intuition: darts at a target (bias = aim, variance = spread)
  - Underfitting (high bias) vs overfitting (high variance)
- Regularisation:
  - L1 (Lasso): drives coefficients to zero, sparse solutions. Geometry: diamond constraint.
  - L2 (Ridge): shrinks coefficients toward zero. Geometry: circle constraint.
  - Elastic Net: L1 + L2 combination (alpha mixing parameter)
  - Bayesian interpretation: L2 = Gaussian prior on coefficients (connects to M2.1)
- Cross-validation:
  - k-fold: split data into k folds, train on k-1, validate on 1, rotate
  - Stratified k-fold: preserve class proportions
  - Time-series split: walk-forward validation (no future data leakage)
  - Nested CV: outer loop for model selection, inner loop for hyperparameters (connects to M3.7)
  - GroupKFold: when observations are grouped (e.g., same patient, same company)

**Key Formulas**:
- Bias-variance: E[(y - y_hat)^2] = Bias^2(y_hat) + Var(y_hat) + sigma^2
- L1 penalty: lambda * Sum(|beta_i|)
- L2 penalty: lambda * Sum(beta_i^2)
- Elastic Net: alpha * L1 + (1-alpha) * L2

**Learning Objectives**: Students can:
- Derive the bias-variance decomposition for squared loss
- Apply L1, L2, and Elastic Net regularisation
- Explain the Bayesian interpretation of L2
- Select appropriate cross-validation strategy for different data types
- Implement nested CV for unbiased model selection

**Exercise**: Demonstrate bias-variance tradeoff by varying model complexity on HDB data. Compare L1 vs L2 vs ElasticNet on same dataset. Implement nested CV.

**Assessment Criteria**: Bias-variance demonstrated visually (train/test error curves). Regularisation impact on coefficients shown. CV strategy matches data structure.

**R5 Source**: ASCENT (new derivation, not in R5 decks)

---

#### Lesson 3.3: The Complete Supervised Model Zoo

**Prerequisites**: 3.2 (bias-variance, regularisation, cross-validation)
**Spectrum Position**: Model breadth — knowing when to use what

**Topics**:
- **SVM (Support Vector Machines)**: margin maximisation, kernel trick (linear, RBF, polynomial), soft margin (C parameter). When to use: high-dimensional, clear margin of separation.
- **KNN (K-Nearest Neighbors)**: instance-based learning, distance metrics (Euclidean, Manhattan, cosine), curse of dimensionality, k selection. When to use: small data, interpretable boundaries.
- **Naive Bayes**: GaussianNB, MultinomialNB, BernoulliNB. Naive independence assumption. When to use: text classification, fast baseline. Connects to M2.1 Bayesian thinking.
- **Decision Trees**: splitting criteria (Gini impurity, entropy/information gain), recursive partitioning, pruning (pre-pruning: max_depth, min_samples; post-pruning). Overfitting visualised. When to use: interpretable, non-linear boundaries.
- **Random Forests**: bagging (bootstrap aggregating), feature subsampling, out-of-bag (OOB) estimation, feature importance. When to use: robust default, handles missing data.
- **Model comparison framework**: accuracy vs interpretability vs speed vs data size

**Key Formulas**:
- SVM: maximise 2/||w|| subject to y_i(w x x_i + b) >= 1
- Gini impurity: G = 1 - Sum(p_i^2)
- Information gain: IG = H(parent) - Sum(w_i * H(child_i))
- OOB error: ~36.8% of samples not in each bootstrap sample

**Learning Objectives**: Students can:
- Explain the mathematical basis of SVM, KNN, Naive Bayes, decision trees, and random forests
- Select the appropriate algorithm for a given problem based on data characteristics
- Tune key hyperparameters for each algorithm
- Compare models using proper evaluation methodology

**Exercise**: Train all 5 model families on the same dataset (e-commerce customer classification). Compare performance, training time, and interpretability. Produce a model comparison table.

**Assessment Criteria**: All 5 models trained correctly. Comparison uses consistent evaluation (same CV splits). Model selection justified with data evidence, not opinion.

**R5 Source**: Deck 4B (lists 18 models in monitoring slide) + ASCENT. Note: SVM, KNN, Naive Bayes are new additions not in R5 or ASCENT — need new deck content and exercises.

---

#### Lesson 3.4: Gradient Boosting Deep Dive

**Prerequisites**: 3.3 (decision trees, random forests)
**Spectrum Position**: Model depth — mastering the dominant tabular algorithm

**Topics**:
- **Boosting theory**: sequential ensemble, bias reduction (vs bagging's variance reduction)
- **AdaBoost**: as conceptual warmup — reweight misclassified samples
- **XGBoost**:
  - 2nd-order Taylor expansion of loss function
  - Split gain formula: Gain = 1/2 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
  - Regularisation: lambda (L2 on leaf weights), gamma (min split loss)
- **LightGBM**: Gradient-based One-Side Sampling (GOSS), histogram-based split finding, leaf-wise growth
- **CatBoost**: ordered boosting (prevents target leakage), native categorical feature support
- Model comparison across the boosting family

**Key Formulas**:
- XGBoost objective: Sum(L(y_i, y_hat_i)) + Sum(Omega(f_k))
- XGBoost split gain (see above)
- LightGBM GOSS: keep top-a% gradient samples, randomly sample b% of small gradients

**Learning Objectives**: Students can:
- Explain how boosting reduces bias (vs bagging reducing variance)
- Derive the XGBoost split gain formula
- Compare XGBoost, LightGBM, and CatBoost on the same dataset
- Select and tune the appropriate boosting algorithm

**Exercise**: Train XGBoost, LightGBM, CatBoost on credit scoring data. Compare accuracy, training time, feature importance. Tune key hyperparameters. Explain when to choose each.

**Assessment Criteria**: All three trained and compared. Key hyperparameters tuned (not default). Selection justified with evidence.

**R5 Source**: ASCENT M3. Note: XGBoost 2nd-order Taylor derivation is new in ASCENT, not in R5 decks.

---

#### Lesson 3.5: Model Evaluation, Imbalance, and Calibration

**Prerequisites**: 3.3 + 3.4 (full model zoo)
**Spectrum Position**: Model assessment — knowing how good your model really is

**Topics**:
- **Complete Metrics Taxonomy**:
  - Classification: accuracy, precision, recall, F1-score, ROC-AUC, log loss, confusion matrix, precision-recall curve, specificity, sensitivity
  - Regression: R-squared, adjusted R-squared, MAE, MSE, RMSE, MAPE
  - When to use which: imbalanced data (precision-recall, not accuracy), probabilistic output (log loss, not accuracy), business cost (custom cost matrix)
- **Class Imbalance**:
  - Why accuracy fails on imbalanced data
  - SMOTE and its failures (boundary samples, high-dimensional)
  - Cost-sensitive learning: class weights in loss function
  - Focal Loss: down-weight easy examples (gamma parameter)
- **Calibration**:
  - Platt scaling (logistic regression on model output)
  - Isotonic regression (non-parametric calibration)
  - Calibration plots: reliability diagram
  - Proper scoring rules: Brier score
- **Stacking and blending**: combining model predictions (brief, connects to EnsembleEngine in M4)

**Key Formulas**:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1: 2 * (Precision * Recall) / (Precision + Recall)
- AUC: area under ROC curve (TPR vs FPR)
- Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
- Brier Score: BS = 1/N * Sum((p_i - y_i)^2)

**Learning Objectives**: Students can:
- Select appropriate metrics for classification and regression tasks
- Handle class imbalance with cost-sensitive learning (not just SMOTE)
- Calibrate model probabilities using Platt scaling
- Read and interpret calibration plots

**Exercise**: Train model on imbalanced credit scoring data. Compare accuracy vs F1 vs AUC. Apply SMOTE and cost-sensitive approaches. Calibrate with Platt scaling. Generate calibration plot.

**Assessment Criteria**: Metrics correctly chosen for the problem. Imbalance handled (not just SMOTE). Calibration improves Brier score. Calibration plot shows improvement.

**R5 Source**: ASCENT M3 ex_2

---

#### Lesson 3.6: Interpretability and Fairness

**Prerequisites**: 3.3-3.5 (trained models, evaluation)
**Spectrum Position**: Model transparency — explaining predictions and checking for bias

**Topics**:
- **SHAP (SHapley Additive exPlanations)**:
  - Shapley axioms: efficiency, symmetry, dummy, linearity
  - TreeSHAP: efficient computation for tree-based models
  - KernelSHAP: model-agnostic (slower)
  - SHAP plots: summary, dependence, waterfall, force
- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Perturb input, fit local linear model
  - Interpretation: "for THIS prediction, these features mattered most"
- **ALE (Accumulated Local Effects)**: alternative to PDP, handles feature correlation
- **Fairness**:
  - Disparate impact: ratio of selection rates between groups
  - Equalized odds: TPR and FPR equal across groups
  - Calibration parity: predicted probabilities equally reliable across groups
  - **Impossibility theorem**: cannot simultaneously satisfy demographic parity, equalized odds, AND calibration (Chouldechova 2017, Kleinberg et al. 2016)
  - Fairness as engineering: measure it, report it, mitigate where possible

**Key Formulas**:
- Shapley value: phi_i = Sum over S of [|S|!(|F|-|S|-1)!/|F|!] * [f(S u {i}) - f(S)]
- Disparate impact ratio: P(Y=1|G=minority) / P(Y=1|G=majority) (should be > 0.8)

**Learning Objectives**: Students can:
- Compute and interpret SHAP values for individual and global explanations
- Apply LIME for local interpretability
- Measure fairness using disparate impact and equalized odds
- Explain the impossibility theorem and its implications for model deployment

**Exercise**: Compute SHAP values for the credit scoring model. Generate SHAP summary and waterfall plots. Measure disparate impact across demographic groups. Report fairness findings.

**Assessment Criteria**: SHAP values computed and interpreted correctly. Fairness measured quantitatively. Impossibility theorem explained with model-specific example.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 3.7: Workflow Orchestration, Model Registry, and Hyperparameter Search

**Prerequisites**: 3.1-3.6 (complete ML knowledge)
**Spectrum Position**: ML engineering — automating the training pipeline

**Topics**:
- **WorkflowBuilder**: nodes, connections, runtime, `runtime.execute(workflow.build())`
- **Custom Nodes**: `@register_node`, `Node` subclass, `PythonCodeNode`, `ConditionalNode`
- **Logic nodes**: branching, merging, conditional execution
- **HyperparameterSearch**: Bayesian optimisation, SearchSpace, ParamDistribution, SearchConfig
- **ModelRegistry**: model versioning, metadata, staging -> production promotion
- **MetricSpec, ModelSignature**: schema validation for model inputs/outputs
- Model lifecycle: experiment -> register -> stage -> promote -> serve -> retire

**Key Concepts**: Workflow orchestration, node-based pipelines, Bayesian hyperparameter optimisation, model versioning

**Learning Objectives**: Students can:
- Build ML workflows using WorkflowBuilder with custom nodes
- Implement Bayesian hyperparameter search
- Register, version, and promote models through the lifecycle
- Define model signatures for input/output validation

**Exercise**: Build an automated ML pipeline as a workflow: data loading node -> preprocessing node -> training node -> evaluation node -> conditional promotion node. Register best model.

**Assessment Criteria**: Workflow executes end-to-end. Custom nodes correctly defined. Hyperparameter search improves model. Model registered with signature.

**R5 Source**: Deck 4B (MLOps components) + ASCENT M3 ex_4/5

---

#### Lesson 3.8: Production Pipeline — DataFlow, Drift, and Deployment

**Prerequisites**: 3.7 (workflow orchestration, model registry)
**Spectrum Position**: Production ML — from training to serving and monitoring

**Topics**:
- **DataFlow**: `@db.model`, `field()`, `db.express.create/list/get/update/delete`, `ConnectionManager`
  - Schema design for ML results
  - Async/await primer for database operations
- **DriftMonitor**: monitor deployed models for distribution shift
  - PSI (Population Stability Index): compare feature distributions
  - KS test (Kolmogorov-Smirnov): compare CDFs
  - Performance degradation detection
  - Monitoring frequency and alerting
  - DriftSpec configuration
- **Model Card**: document model purpose, performance, limitations, fairness findings (Mitchell et al.)
- **Conformal prediction**: distribution-free prediction intervals
- **Full production pipeline**: train -> persist results to DataFlow -> calibrate -> register -> promote -> monitor for drift -> model card
- **MLOps concepts** from Deck 4B: CI/CD for ML, clean architecture, model versioning and governance

**Key Concepts**: Database persistence, drift monitoring, model documentation, production deployment

**Learning Objectives**: Students can:
- Persist ML results to a database using DataFlow
- Monitor deployed models for drift using PSI and KS tests
- Create model cards documenting performance and limitations
- Build a complete production ML pipeline from training to monitoring

**Exercise**: Deploy the credit scoring model as a full production pipeline. Persist results to DataFlow. Set up DriftMonitor with alerting thresholds. Generate model card. Simulate drift and verify detection.

**Assessment Criteria**: Pipeline end-to-end. DataFlow CRUD operations work. Drift detected when injected. Model card complete.

**R5 Source**: Deck 4B (MLOps) + ASCENT M3 ex_6

**End of Module Assessment**: Quiz + ML pipeline project (full pipeline from raw data to deployed, monitored model).

---

### MODULE 4: Unsupervised Machine Learning and Advanced Techniques for Insights

**Description**: Pattern discovery without labels, then the bridge to neural feature learning. USML = automated feature engineering (from R5 Deck 5A).

**Module Learning Objectives**: By the end of M4, students can:
- Apply clustering algorithms and evaluate cluster quality
- Implement EM algorithm and understand mixture models
- Reduce dimensionality with PCA, t-SNE, and UMAP
- Detect anomalies using statistical and ML methods
- Discover transactional patterns with association rules
- Extract topics from text using TF-IDF, LDA, and BERTopic
- Build recommender systems using collaborative filtering
- Explain how neural network hidden layers are automated feature engineering with error feedback
- Train a basic neural network with proper training practices

**Kailash Engines**: AutoMLEngine, EnsembleEngine, ModelVisualizer, OnnxBridge

---

#### Lesson 4.1: Clustering

**Prerequisites**: M3 complete (supervised ML)
**Spectrum Position**: USML begins — discovering group structure without labels

**Bridge from M3**: "In M3, you predicted outcomes using labelled data. Now: what if there are no labels? USML discovers structure in data without being told what to look for."

**Topics**:
- **K-means**: algorithm, convergence, elbow method, sensitivity to initialisation (k-means++)
- **Hierarchical clustering** (from Deck 5A):
  - Agglomerative (bottom-up) vs divisive (top-down)
  - Linkage methods: single (min distance), complete (max distance), average, Ward's
  - Dendrograms: reading, cutting threshold
  - Pros/cons of each linkage method (Deck 5A covers this in detail)
- **DBSCAN**: epsilon-neighbourhood, minPts, core/border/noise points
- **HDBSCAN**: hierarchical extension of DBSCAN, auto-selects epsilon
- **Spectral clustering**: graph Laplacian, for non-convex clusters
- **Cluster evaluation**:
  - Internal: silhouette score, Davies-Bouldin index, Calinski-Harabasz index
  - External: ARI (Adjusted Rand Index), NMI (Normalised Mutual Information)
  - Gap statistic: compare within-cluster dispersion to null reference
- **Customer segmentation** application (from PCML5-1)

**Key Formulas**:
- K-means objective: minimise Sum_k Sum_{x in C_k} ||x - mu_k||^2
- Silhouette: s(i) = (b(i) - a(i)) / max(a(i), b(i))
- Davies-Bouldin: DB = 1/k * Sum max_{j!=i} (s_i + s_j) / d(c_i, c_j)

**Learning Objectives**: Students can:
- Apply K-means, hierarchical, DBSCAN, and HDBSCAN clustering
- Read and interpret dendrograms
- Evaluate clusters using silhouette, DB index, and gap statistic
- Select clustering algorithm based on data characteristics

**Exercise**: Customer segmentation on retail data. Compare K-means, hierarchical (with dendrogram), and HDBSCAN. Evaluate with silhouette and DB index. Interpret clusters with business meaning.

**Assessment Criteria**: Multiple algorithms compared. Evaluation metrics computed. Clusters interpreted with business rationale (not just "cluster 1, cluster 2").

**R5 Source**: Deck 5A (K-means, hierarchical with 4 linkage methods, dendrograms, t-SNE) + PCML5-1 (customer segmentation)

---

#### Lesson 4.2: EM Algorithm and Gaussian Mixture Models

**Prerequisites**: 4.1 (clustering), 2.1 (Bayesian thinking)
**Spectrum Position**: Soft clustering — probabilistic assignment to groups

**Topics**:
- Soft vs hard clustering: GMM assigns probabilities, K-means assigns labels
- **EM Algorithm**:
  - E-step: compute responsibilities (probability each point belongs to each cluster)
  - M-step: update parameters (means, covariances, mixing coefficients) using responsibilities
  - Convergence: log-likelihood is non-decreasing
  - 20-line implementation from scratch
- **Gaussian Mixture Models**: EM applied to Gaussian components
- EM as a general template: applicable to any latent variable model
- **Mixture of Experts** (brief): modern application of mixture models (e.g., GPT-4 architecture). Gating network selects expert based on input. Connect to M6 LLMs.

**Key Formulas**:
- E-step: r_nk = (pi_k * N(x_n | mu_k, Sigma_k)) / Sum_j(pi_j * N(x_n | mu_j, Sigma_j))
- M-step: mu_k = Sum_n(r_nk * x_n) / Sum_n(r_nk)
- Log-likelihood: L = Sum_n log(Sum_k pi_k * N(x_n | mu_k, Sigma_k))

**Learning Objectives**: Students can:
- Implement the EM algorithm from scratch (20 lines)
- Explain the difference between hard and soft clustering
- Fit GMMs and interpret component probabilities
- Describe how Mixture of Experts extends mixture models

**Exercise**: Implement EM on 2D synthetic data (3 Gaussians). Compare with sklearn GMM on real e-commerce data. Visualise soft assignments.

**Assessment Criteria**: EM implementation converges. Responsibilities sum to 1. Comparison with GMM shows similar results.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 4.3: Dimensionality Reduction

**Prerequisites**: 4.1 (clustering), 2.5 (linear algebra concepts from regression)
**Spectrum Position**: Feature compression — discovering latent axes

**Topics**:
- **PCA** (from Deck 5A, 2-step process):
  - Step 1: Decorrelate — rotate axes to align with data directions (principal components)
  - Step 2: Reduce — keep top-k components by variance explained
  - SVD connection: PCA via eigendecomposition or SVD (X = U * Sigma * V^T)
  - Scree plot: variance explained per component
  - Loadings: interpret what each component represents
  - Reconstruction error: what information is lost
  - PCA as feature extraction (not just visualisation)
- **Kernel PCA**: nonlinear dimensionality reduction via kernel trick (RBF, polynomial)
- **t-SNE** (from Deck 5A): stochastic neighbour embedding, perplexity parameter, good for visualisation but NOT for feature extraction (non-deterministic, no inverse transform)
- **UMAP**: faster than t-SNE, preserves more global structure, deterministic. Can be used for feature extraction.
- **Manifold learning** (brief mention): Isomap (geodesic distances), LLE (Locally Linear Embedding), MDS (Multidimensional Scaling) — reference table for when to use each
- **Intrinsic dimension**: how many components needed to approximate data (from Deck 5A)

**Key Formulas**:
- PCA: maximise Var(w^T X) subject to ||w|| = 1
- SVD: X = U * Sigma * V^T
- Variance explained: lambda_k / Sum(lambda_i)
- Reconstruction error: ||X - X_hat||^2

**Learning Objectives**: Students can:
- Implement PCA and interpret scree plots and loadings
- Explain the SVD connection to PCA
- Apply t-SNE and UMAP for visualisation and compare results
- Select dimensionality reduction method based on use case (visualisation vs feature extraction vs nonlinear)

**Exercise**: Apply PCA to e-commerce data, interpret first 3 components via loadings. Compare t-SNE vs UMAP visualisations (vary hyperparameters). Demonstrate reconstruction error tradeoff.

**Assessment Criteria**: Scree plot shows variance explained. Loadings interpreted with domain meaning. t-SNE/UMAP hyperparameters varied and compared.

**R5 Source**: Deck 5A (PCA 2-step, t-SNE, intrinsic dimension)

---

#### Lesson 4.4: Anomaly Detection and Ensembles

**Prerequisites**: 4.1 (clustering), 3.5 (evaluation metrics)
**Spectrum Position**: Outlier discovery — finding what doesn't belong

**Topics**:
- **Statistical outlier detection** (from Deck 2A): Z-score method (3 sigma rule), IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR), winsorisation
- **Isolation Forest**: random trees isolate anomalies faster (shorter path length = more anomalous)
- **LOF (Local Outlier Factor)**: density-based, compares local density to neighbours
- **Score blending**: combine multiple anomaly detectors for robustness
- **EnsembleEngine**: `blend()`, `stack()`, `bag()`, `boost()` — unified ensemble API
- Anomaly detection as production monitoring (connects to M3.8 drift monitoring)

**Key Formulas**:
- Z-score: z = (x - x_bar) / s. Outlier if |z| > 3.
- IQR method: outlier if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR
- Isolation Forest anomaly score: s(x, n) = 2^(-E(h(x)) / c(n))
- LOF: LOF_k(x) = (Sum_{o in N_k(x)} lrd_k(o) / lrd_k(x)) / |N_k(x)|

**Learning Objectives**: Students can:
- Apply statistical and ML anomaly detection methods
- Combine multiple detectors using score blending
- Use EnsembleEngine for unified ensemble operations
- Explain when to use each anomaly detection method

**Exercise**: Detect anomalies in financial transaction data using Z-score, Isolation Forest, and LOF. Blend scores. Compare results. Identify true anomalies vs false positives.

**Assessment Criteria**: Multiple methods applied and compared. Blended score improves over individual methods. Business interpretation of detected anomalies.

**R5 Source**: Deck 2A (Z-score, IQR, winsorisation) + ASCENT

---

#### Lesson 4.5: Association Rules and Market Basket Analysis

**Prerequisites**: 4.1 (pattern discovery concept)
**Spectrum Position**: Co-occurrence pattern discovery — finding what appears together

**Topics**:
- Association rules: discovering co-occurrence patterns in transactional data
- **Apriori algorithm**: generate frequent itemsets, prune by minimum support
- **FP-Growth**: compressed representation (FP-tree), no candidate generation, faster than Apriori
- **Metrics**: support (frequency), confidence (conditional probability), lift (surprise factor)
- Applications: retail basket analysis, web click patterns, medical co-diagnoses
- **Forward connection**: association rules discover co-occurrence features. These features can be used as inputs to supervised models (M3). Collaborative filtering (M4.7) extends this to learning latent factors.

**Key Formulas**:
- Support: supp(X) = count(X) / total_transactions
- Confidence: conf(X -> Y) = supp(X u Y) / supp(X)
- Lift: lift(X -> Y) = conf(X -> Y) / supp(Y). Lift > 1 = positive association.

**Design Note**: This is NOT a dead-end topic. It connects forward: (1) discovered rules become features for supervised models, (2) the idea of "finding patterns in co-occurrence data" is exactly what collaborative filtering does with embeddings (M4.7).

**Learning Objectives**: Students can:
- Implement Apriori and FP-Growth for frequent itemset mining
- Compute and interpret support, confidence, and lift
- Extract actionable business rules from transaction data
- Use discovered patterns as features for supervised models

**Exercise**: Market basket analysis on Singapore retail transaction data. Find top association rules. Interpret business meaning. Create features from rules for a classification model.

**Assessment Criteria**: Rules discovered with appropriate support threshold. Business interpretation provided. Rules used as features improve supervised model.

**R5 Source**: New (not in R5). Need new deck content and exercise.

---

#### Lesson 4.6: NLP — Text to Topics

**Prerequisites**: 4.3 (dimensionality reduction), 4.1 (clustering)
**Spectrum Position**: Text feature discovery — extracting meaning from unstructured text

**Topics**:
- Text as data: how to represent text for ML (from Deck 5A: text structure, classification vs clustering)
- **TF-IDF** derivation: term frequency * inverse document frequency. Why it works (common words get low weight).
- **BM25**: improved TF-IDF (saturation, document length normalisation)
- **Word embeddings** (tools, not derivation — derivation in M4.8):
  - Word2Vec (CBOW, Skip-gram): "words that appear in similar contexts have similar meanings"
  - GloVe: global co-occurrence statistics
  - FastText: subword embeddings, handles OOV words
  - **Note**: "How does Word2Vec learn these vectors? We will see in M4.8 when we study neural networks."
- **LDA (Latent Dirichlet Allocation)**: generative topic model. Each document = mixture of topics, each topic = distribution over words.
- **NMF (Non-negative Matrix Factorisation)**: matrix factorisation approach to topics (from Deck 5A, used in NLP)
- **BERTopic**: transformer-based topic modelling, UMAP + HDBSCAN + c-TF-IDF
- **Coherence metrics**: NPMI, UMass. How to evaluate topic quality.
- Sentiment analysis (brief): as application of text classification

**Key Formulas**:
- TF-IDF: tfidf(t, d) = tf(t, d) * log(N / df(t))
- LDA: P(word | document) = Sum_k P(word | topic_k) * P(topic_k | document)
- NPMI: NPMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) * P(w_j))) / -log(P(w_i, w_j))

**Learning Objectives**: Students can:
- Derive and implement TF-IDF from scratch
- Apply LDA and BERTopic for topic extraction
- Evaluate topic quality using coherence metrics
- Use word embeddings as features (without understanding the training yet)

**Exercise**: Extract topics from Singapore news articles using TF-IDF + NMF, LDA, and BERTopic. Compare topic quality using NPMI. Classify sentiment of customer reviews.

**Assessment Criteria**: Multiple topic methods compared. Coherence metrics computed. Topics interpreted with human-readable labels.

**R5 Source**: Deck 5A (NLP text structure, TF-IDF, NMF) + PCML5-2

---

#### Lesson 4.7: Recommender Systems and Collaborative Filtering

**Prerequisites**: 4.3 (PCA/SVD), 4.6 (word embeddings concept)
**Spectrum Position**: THE PIVOT — optimisation drives feature discovery

**Topics**:
- **Content-based filtering**: recommend items similar to what user liked (feature similarity)
- **Collaborative filtering**: recommend items that similar users liked
  - User-based CF: find similar users, recommend their items
  - Item-based CF: find similar items, recommend to users who liked similar items
  - Pros/cons: user-based (cold start for new users) vs item-based (more stable)
- **Matrix factorisation**: factorise user-item matrix R into U * V^T
  - U = user embeddings, V = item embeddings
  - Optimise: minimise ||R - U * V^T||^2 (reconstruction error)
  - ALS (Alternating Least Squares): fix U, optimise V; fix V, optimise U
  - SVD++: extends SVD with implicit feedback
  - Connection to PCA (M4.3): PCA factorises X = U * Sigma * V^T. Collaborative filtering factorises R = U * V^T. Same idea: find low-rank structure.
- **Implicit vs explicit feedback**: ratings (explicit) vs clicks/views/purchases (implicit)
- **Hybrid systems**: combine content-based and collaborative filtering
- **THE PIVOT**: "Matrix factorisation learns user and item embeddings by minimising reconstruction error. This is the first time you've seen OPTIMISATION DRIVE FEATURE DISCOVERY. In M4.8, neural networks generalise this — hidden layer activations ARE embeddings, learned by minimising a loss function."

**Key Formulas**:
- Matrix factorisation: minimise Sum_{(u,i) in observed} (r_ui - u_u^T * v_i)^2 + lambda * (||u_u||^2 + ||v_i||^2)
- ALS update for U: U = (V^T * V + lambda * I)^{-1} * V^T * R^T

**Learning Objectives**: Students can:
- Build content-based and collaborative filtering recommenders
- Implement matrix factorisation with ALS
- Explain how matrix factorisation learns embeddings
- Articulate the connection: "optimisation drives feature discovery" → bridge to neural networks

**Exercise**: Build recommender system on Singapore retail data. Implement user-based CF, item-based CF, and matrix factorisation. Compare recommendation quality. Visualise learned embeddings.

**Assessment Criteria**: All three approaches implemented and compared. Embeddings visualised (2D projection shows meaningful clusters). The pivot concept articulated.

**R5 Source**: PCML5-3 (recommenders, adapted)

---

#### Lesson 4.8: DL Foundations — Neural Networks, Backpropagation, and the Training Toolkit

**Prerequisites**: 4.7 (embeddings, optimisation-driven feature discovery), 2.5 (regression)
**Spectrum Position**: The bridge — hidden layers are USML + error feedback

**Bridge (from Deck 5B)**: "In M4.7, matrix factorisation learned embeddings by minimising reconstruction error. A neural network does the same thing: hidden layer activations ARE embeddings, learned by minimising a loss function. The difference: neural networks can learn NON-LINEAR combinations through activation functions."

**Topics**:
- **Neural network architecture** (from Deck 5B):
  - Input layer, hidden layers, output layer
  - Weights, biases, fully connected layers
  - Each node connected to all nodes in previous layer
- **Forward pass**: multiply inputs by weights, sum at each node, predict
- **Error / Loss**: predicted vs actual
- **Linear regression as a neural network** (from Deck 5B slides 22-31):
  - Regression with zero hidden layers = linear regression
  - Add hidden layers → model "writes its own parametric function"
  - Feature interaction through activation functions → non-linearity
- **Gradient descent** (from Deck 5B, step-by-step with HDB example):
  - Cost function: SSE = 1/2 * Sum((y - y_hat)^2)
  - Gradient: dJ/dw = -x * (actual - predicted)
  - Weight update: w_new = w_old - learning_rate * gradient
  - Iterate until convergence (demonstrate with error decreasing)
- **Backpropagation**: chain rule through layers. Compute gradient of loss with respect to each weight.
- **Hidden layers** (from Deck 5B slides 34-41):
  - 2+ hidden layers can represent ANY non-linear function
  - "Automated feature engineering": hidden layers discover features automatically
  - **Representation learning**: DL learns deep representations of data relationships
  - **Embeddings**: hidden node values encode learned knowledge
  - "Unsupervised meets supervised learning": hidden layers perform unsupervised feature discovery while the output layer performs supervised prediction
- **Regression vs classification output**: 1 node for regression, N nodes for N classes
- **DL Training Toolkit** (new, from completeness audit):
  - **Activation functions**: ReLU, Leaky ReLU, PReLU, ELU, GELU, Swish, Sigmoid, Tanh. When to use each. ReLU default for hidden layers. Sigmoid for binary output. Softmax for multiclass.
  - **Dropout**: randomly zero out neurons during training. Prevents co-adaptation. Dropout rate typically 0.1-0.5. Turn off during inference.
  - **Batch normalisation**: normalise layer inputs to zero mean, unit variance. Stabilises training, enables higher learning rates. Layer norm for transformers (M5.4).
  - **Weight initialisation**: Xavier/Glorot (for sigmoid/tanh), Kaiming/He (for ReLU). Why random init is needed (symmetry breaking). Why zero init fails.
  - **Optimisers**: SGD (+ momentum), RMSProp, Adam (adaptive learning rates), AdamW (decoupled weight decay). Comparison table.
  - **Loss functions taxonomy**: MSE (regression), MAE (robust regression), cross-entropy (classification), binary cross-entropy, focal loss (imbalanced), contrastive loss (similarity), triplet loss (metric learning), KL divergence (distribution matching), reconstruction loss (autoencoders).
  - **Learning rate schedules**: step decay, cosine annealing, warmup + cosine, one-cycle policy, ReduceLROnPlateau.
  - **Gradient clipping**: prevent exploding gradients (max norm or value clipping).
  - **Early stopping**: monitor validation loss, stop when it increases for patience epochs.

**Key Formulas**:
- Forward pass: z = W * x + b, a = f(z)
- Gradient descent: w = w - lr * dL/dw
- Backpropagation chain rule: dL/dw_1 = dL/da_2 * da_2/dz_2 * dz_2/da_1 * da_1/dz_1 * dz_1/dw_1
- Batch norm: y = gamma * (x - mu_B) / sqrt(sigma_B^2 + epsilon) + beta
- Adam: m_t = beta_1 * m_{t-1} + (1-beta_1) * g_t; v_t = beta_2 * v_{t-1} + (1-beta_2) * g_t^2

**Design Note**: This is a dense lesson — the most important lesson in the curriculum. It bridges USML to DL and provides the complete training toolkit. Allocate 4.5 hours if possible.

**Learning Objectives**: Students can:
- Build a neural network from scratch (forward pass, loss, backprop, weight update)
- Explain how hidden layers are automated feature engineering with error feedback
- Select appropriate activation function, optimiser, and loss function
- Apply dropout, batch normalisation, and learning rate scheduling
- Explain representation learning and embeddings

**Exercise**: Build a 3-layer neural network from scratch for HDB price prediction. Implement forward pass, backprop, gradient descent. Then: add dropout, batch norm, Adam optimiser, LR schedule. Compare training curves with and without each technique.

**Assessment Criteria**: From-scratch implementation works (loss decreases). Each training technique improves convergence (demonstrated with plots). The "unsupervised meets supervised" concept articulated.

**R5 Source**: Deck 5B (42 slides, comprehensive) + PCML5-4 (DL basics notebook)

**End of Module Assessment**: Quiz + project (unsupervised analysis → DL bridge: cluster data, reduce dimensions, build neural network on discovered features).

---

### MODULE 5: Deep Learning and Machine Learning Mastery in Vision and Transfer Learning

**Description**: Every major DL architecture. One paradigm per lesson. All implemented. Following R5 Deck 6A (comprehensive architecture coverage) + PCML6 notebooks (crown jewel implementations).

**Module Learning Objectives**: By the end of M5, students can:
- Build and train autoencoders (vanilla, denoising, VAE, convolutional)
- Implement and train CNNs with modern enhancements (ResNet, SE blocks, mixed precision)
- Build LSTM and GRU networks with attention mechanisms
- Derive self-attention from scratch and fine-tune transformers
- Implement GANs (DCGAN, WGAN) and understand diffusion model basics
- Apply GNNs to graph-structured data
- Transfer pre-trained models to new tasks (CV and NLP)
- Implement RL algorithms (DQN, DDPG, SAC, A2C, PPO) for business applications

**Kailash Engines**: ModelVisualizer, OnnxBridge, InferenceServer, RLTrainer

---

#### Lesson 5.1: Autoencoders

**Prerequisites**: M4.8 (neural networks, training toolkit)
**Spectrum Position**: Unsupervised DL — learning compressed representations

**DL Toolkit Refresher** (30 min opening): "In M4.8 you learned forward pass, backprop, gradient descent, dropout, batch norm, optimisers. Quick exercise: build a 2-layer classifier. Good — now we modify this architecture for UNSUPERVISED learning. That's an autoencoder."

**Topics**:
- Autoencoder concept: encoder (compress) → latent space → decoder (reconstruct). Minimise reconstruction error.
- **Deep dive** (4 variants with full implementations):
  - Vanilla autoencoder: simplest form, undercomplete
  - Denoising autoencoder (DAE): corrupt input, learn to reconstruct clean version
  - Variational autoencoder (VAE): ELBO, reparameterisation trick, latent space is a probability distribution. Generate NEW data by sampling from latent space.
  - Convolutional autoencoder: use conv layers for image data
- **Survey** (5 additional variants as reference):
  - Sparse autoencoder (L1 penalty on activations)
  - Contractive autoencoder (penalty on Jacobian)
  - Stacked autoencoder (progressively deeper)
  - Recurrent autoencoder (for sequences)
  - CVAE (Contractive + Variational)

**Key Formulas**:
- Reconstruction loss: L = ||x - decoder(encoder(x))||^2
- VAE ELBO: L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
- Reparameterisation: z = mu + sigma * epsilon, epsilon ~ N(0,1)

**Learning Objectives**: Students can:
- Implement vanilla, denoising, VAE, and convolutional autoencoders
- Explain the VAE reparameterisation trick and why it enables gradient flow
- Generate new data by sampling from VAE latent space
- Know when to use each variant

**Exercise**: Implement 4 autoencoder variants on MNIST/Fashion-MNIST. Visualise latent spaces. Generate new images from VAE. Compare reconstruction quality.

**Assessment Criteria**: 4 variants implemented and trained. VAE generates plausible new images. Latent space visualisation shows meaningful structure.

**R5 Source**: Deck 6A (9 variants) + PCML6-1 (10+ variants, implementations)

---

#### Lesson 5.2: CNNs and Computer Vision

**Prerequisites**: 5.1 (autoencoder training experience)
**Spectrum Position**: Spatial feature learning — extracting patterns from grid data

**Topics**:
- **CNN fundamentals** (from Deck 6A):
  - Convolution operation: filters, stride, padding, feature maps
  - Pooling: max pooling, average pooling — reduce spatial dimensions
  - Normalisation layers: batch norm (between conv and pooling, from M4.8 toolkit)
  - Earlier layers → low-level features (edges, textures). Later layers → high-level features (objects, faces).
- **Architecture history** (from Deck 6A):
  - LeNet-5: earliest CNN (handwritten digits)
  - AlexNet: deeper, ReLU, dropout
  - VGGNet: very small (3x3) filters, depth
  - GoogLeNet/Inception: multiple filter sizes in parallel
  - **ResNet**: residual connections (skip connections) solving vanishing gradients
- **Modern training enhancements** (from PCML6-2):
  - SE blocks (Squeeze-and-Excitation): channel recalibration
  - Kaiming initialisation (proper for ReLU)
  - Mixed precision training (FP16/FP32)
  - Mixup augmentation (smooth decision boundaries)
  - Label smoothing (prevent overconfident predictions)
  - Gradient flow analysis
- **Vision Transformers (ViT)**: brief intro — applying transformers to image patches (connects to M5.4)

**Key Formulas**:
- Conv output size: (W - F + 2P) / S + 1
- ResNet: H(x) = F(x) + x (skip connection)
- SE block: s = sigmoid(W2 * ReLU(W1 * GAP(x)))

**Learning Objectives**: Students can:
- Build CNNs with convolution, pooling, and normalisation layers
- Implement ResNet with skip connections
- Apply modern training enhancements (SE blocks, mixed precision, Mixup)
- Explain why ResNet solves the vanishing gradient problem

**Exercise**: Build CNN for image classification (Fashion-MNIST or mask detection). Start simple, add ResBlock, add SE block. Compare training curves. Export to ONNX with OnnxBridge.

**Assessment Criteria**: Architecture progressively improved. Training enhancements measurably help. ONNX export successful.

**R5 Source**: Deck 6A (architecture history) + PCML6-2 (44MB, advanced enhancements)

---

#### Lesson 5.3: RNNs and Sequence Models

**Prerequisites**: 5.2 (training experience, batch norm, gradient concepts)
**Spectrum Position**: Temporal feature learning — extracting patterns from sequences

**Topics**:
- **RNN fundamentals**: directed cycles, hidden state memory, vanishing gradient problem
- **LSTM** (from Deck 6A + PCML6-3):
  - 4 components: cell state, forget gate, input gate, output gate
  - All 6 gate equations
  - Why LSTM solves vanishing gradients (cell state highway)
- **GRU**: simplified LSTM (update gate, reset gate). Fewer parameters, faster.
- **Multi-layer with residual connections** (from PCML6-3)
- **Attention mechanisms** (from PCML6-3):
  - Temporal attention: focus on important time steps
  - Spatial attention: feature relationships via multi-headed attention
  - Connects to M5.4 (Transformers replace RNNs with pure attention)
- **Performance metrics** (from Deck 6A): perplexity, BLEU score, sequence accuracy, cross-entropy loss
- **Applications** (from PCML6-3): financial time series prediction (technical indicators: RSI, MACD, Bollinger Bands), Shakespeare text generation
- **Gradient clipping** (from M4.8 toolkit): essential for RNNs to prevent exploding gradients

**Key Formulas**:
- LSTM forget gate: f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)
- LSTM input gate: i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)
- LSTM cell update: C_t = f_t * C_{t-1} + i_t * tanh(W_C * [h_{t-1}, x_t] + b_C)
- LSTM output gate: o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)
- LSTM hidden state: h_t = o_t * tanh(C_t)
- Perplexity: PP = exp(-1/N * Sum(log P(w_i)))

**Learning Objectives**: Students can:
- Implement LSTM and GRU networks
- Explain all LSTM gate equations and their purpose
- Apply temporal attention to sequence models
- Train RNNs for time-series prediction and text generation

**Exercise**: Build LSTM for Singapore stock price prediction with technical indicators. Add attention layer. Compare LSTM vs GRU. Implement text generation with character-level LSTM.

**Assessment Criteria**: LSTM gates implemented correctly. Attention improves prediction. Text generation produces coherent output.

**R5 Source**: Deck 6A (LSTM/GRU theory, metrics) + PCML6-3 (10MB, attention, financial prediction)

---

#### Lesson 5.4: Transformers

**Prerequisites**: 5.3 (attention mechanisms, sequence models)
**Spectrum Position**: Attention-based feature learning — processing sequences in parallel

**Topics**:
- **Self-attention** (derive from scratch):
  - Query, Key, Value matrices
  - Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
  - Why divide by sqrt(d_k): prevents softmax from saturating (dot products grow with dimension)
  - Multi-head attention: multiple attention heads capture different relationships
- **Positional encoding**: sinusoidal or learned embeddings (transformers have no inherent position sense)
- **Encoder-decoder architecture** (from Deck 6A):
  - Encoder: multi-head self-attention → feed-forward → layer norm + residual
  - Decoder: same + masked self-attention + cross-attention to encoder
- **Transformer variants** (from Deck 6A):
  - BERT: bidirectional, masked language modelling, NLU tasks
  - GPT: autoregressive decoder, next token prediction, generation
  - T5: text-to-text unified framework
  - Transformer-XL: segment-level recurrence for long contexts
  - Reformer/Longformer: efficient long-sequence handling
- **Vision Transformers (ViT)**: split image into patches → treat as sequence → transformer encoder. Dominant for image classification 2024+.
- **Layer normalisation** (vs batch norm from M4.8): why transformers use layer norm (sequence length varies)
- **BERT fine-tuning** (applied exercise from PCML6-4): fine-tune pre-trained BERT for text classification

**Key Formulas**:
- Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
- Multi-head attention: MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
- Positional encoding: PE(pos, 2i) = sin(pos / 10000^{2i/d}), PE(pos, 2i+1) = cos(pos / 10000^{2i/d})

**Consolidation segment** (30 min): After Transformers, students have seen the four major paradigms (AE, CNN, RNN, Transformer). Compare: "Which architecture for which data type?" Table: images→CNN/ViT, sequences→RNN/Transformer, graphs→GNN (M5.6), generation→VAE/GAN (M5.5).

**Learning Objectives**: Students can:
- Derive scaled dot-product attention from scratch
- Explain why dividing by sqrt(d_k) is necessary
- Fine-tune BERT for a downstream NLP task
- Compare transformer variants and select appropriate one
- Explain ViT and how vision tasks use transformers

**Exercise**: Derive self-attention from scratch (pen-and-paper + code). Fine-tune BERT for text classification (from PCML6-4 TREC6 dataset). Compare with LSTM baseline (M5.3).

**Assessment Criteria**: Self-attention implemented correctly. BERT fine-tuning produces good classification. Comparison with LSTM quantified (accuracy, training time).

**R5 Source**: Deck 6A (architecture + model variants) + PCML6-4 (BERT fine-tuning)

---

#### Lesson 5.5: Generative Models — GANs and Diffusion

**Prerequisites**: 5.1 (autoencoders, VAE), 5.2 (CNNs)
**Spectrum Position**: Generative modelling — learning to create new data

**Topics**:
- **GAN fundamentals** (from Deck 6A):
  - Generator vs Discriminator: zero-sum game
  - Adversarial loss: binary cross-entropy
  - Training dynamics: alternating optimisation
- **GAN variants** (from Deck 6A, all covered):
  - **DCGAN**: convolutional generator/discriminator, no max pooling or FC layers, batch norm
  - **Conditional GAN (cGAN)**: condition on class labels for controlled generation
  - **WGAN**: Wasserstein distance instead of JS divergence, gradient penalty, prevents mode collapse
  - **CycleGAN**: unpaired image-to-image translation
  - **StyleGAN**: style-based generation, progressive growing, high-resolution
- **Training challenges**: mode collapse, training instability, evaluation difficulty
- **Evaluation**: FID (Frechet Inception Distance), IS (Inception Score)
- **Diffusion models** (brief, from completeness audit):
  - DDPM (Denoising Diffusion Probabilistic Models): add noise progressively, learn to reverse
  - More stable than GANs, better diversity
  - Stable Diffusion as practical application
  - When to use GANs vs diffusion vs VAE (from Deck 6A generation guide): images→GAN/diffusion, text→transformers, time-series→VAE/LSTM
- **Data generation applications** from Deck 6A: synthetic data for privacy, augmentation, simulation

**Key Formulas**:
- GAN minimax: min_G max_D [E[log D(x)] + E[log(1 - D(G(z)))]]
- WGAN: min_G max_D [E[D(x)] - E[D(G(z))]]
- FID: ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^{1/2})

**Learning Objectives**: Students can:
- Implement DCGAN and WGAN with training loops
- Explain mode collapse and how WGAN addresses it
- Compare GAN, VAE, and diffusion models for different generation tasks
- Evaluate generated data quality using FID

**Exercise**: Implement DCGAN for image generation. Implement WGAN and compare training stability. Evaluate with FID. Discuss when to use GANs vs diffusion.

**Assessment Criteria**: DCGAN generates recognisable images. WGAN more stable (demonstrated). FID computed. Generation model selection guide understood.

**R5 Source**: Deck 6A (6 GAN variants + generation model guide) + PCML6-6 (expanded)

---

#### Lesson 5.6: Graph Neural Networks

**Prerequisites**: 5.4 (attention mechanisms)
**Spectrum Position**: Graph feature learning — patterns in connected data

**Topics**:
- Graph data: nodes, edges, adjacency matrix. Applications: social networks, knowledge graphs, molecular structures.
- **GNN architectures** (from Deck 6A):
  - **GCN (Graph Convolutional Networks)**: spectral methods, message passing, aggregation
  - **GraphSAGE**: sampling + aggregating from local neighbourhood, inductive (handles unseen nodes)
  - **GAT (Graph Attention Networks)**: attention weights on neighbours (connects to M5.4 attention)
  - **GIN (Graph Isomorphism Networks)**: captures graph structure more effectively
- **Tasks**: node classification, graph classification, link prediction
- torch_geometric library: `GCNConv`, `global_mean_pool`, DataLoader for graphs

**Key Formulas**:
- GCN: H^(l+1) = sigma(D^{-1/2} A D^{-1/2} H^(l) W^(l))
- GAT attention: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])

**Learning Objectives**: Students can:
- Build GCNs for node and graph classification
- Explain message passing and neighbourhood aggregation
- Compare GCN, GraphSAGE, GAT for different tasks
- Use torch_geometric for graph ML

**Exercise**: Graph classification on TUDataset using GCN. Compare GCN vs GAT. Visualise learned node embeddings.

**Assessment Criteria**: GCN implemented correctly. GAT comparison shows attention weights. Node embeddings visualised.

**R5 Source**: Deck 6A (4 GNN architectures) + PCML6-5

---

#### Lesson 5.7: Transfer Learning

**Prerequisites**: 5.2 (CNNs), 5.4 (Transformers/BERT)
**Spectrum Position**: Knowledge transfer — leveraging pre-trained representations

**Topics**:
- Transfer learning concept: pre-trained on large dataset, fine-tune on small target dataset
- **CV Transfer Learning** (from PCML6-8):
  - ResNet fine-tuning: freeze early layers, train later layers + new classifier head
  - Data augmentation for small datasets
  - Applications: mask detection (from PCML6-8), MNIST classification
- **NLP Transfer Learning** (from PCML6-9):
  - BERT fine-tuning: HuggingFace Pipeline API, Lightning-based training
  - Adapter modules as a concept (bottleneck layers between transformer layers)
  - Connects to M6.2 (LoRA + Adapters for LLM fine-tuning)
- **ONNX export**: OnnxBridge for portable model deployment
- **InferenceServer**: predict, predict_batch, warm_cache, PredictionResult
- **Architecture Selection Guide** (consolidation):
  | Data Type | Best Architecture | When to Transfer |
  |---|---|---|
  | Images | CNN/ViT | Always (ImageNet pre-trained) |
  | Text | Transformer | Always (BERT/GPT pre-trained) |
  | Sequences | LSTM/Transformer | Sometimes (domain-specific) |
  | Graphs | GNN | Rarely (task-specific) |
  | Tabular | Gradient boosting | Never (train from scratch) |

**Learning Objectives**: Students can:
- Fine-tune pre-trained vision and NLP models for new tasks
- Apply proper transfer learning techniques (freeze/unfreeze layers)
- Export models to ONNX for deployment
- Select the right architecture for a given problem

**Exercise**: Fine-tune ResNet for mask detection (PCML6-8 task). Fine-tune BERT for text classification (PCML6-9 task). Export both to ONNX. Deploy with InferenceServer.

**Assessment Criteria**: Both models fine-tuned and outperform training from scratch. ONNX export successful. InferenceServer serves predictions.

**R5 Source**: Deck 6A + PCML6-8 (CV transfer) + PCML6-9 (NLP transfer)

---

#### Lesson 5.8: Reinforcement Learning

**Prerequisites**: 5.2-5.4 (neural network training experience)
**Spectrum Position**: Learning from interaction — policies learned through environment feedback

**Bridge**: "All DL so far learns from static data (images, text, sequences). RL learns from INTERACTION with an environment. The agent takes actions, receives rewards, and learns a policy that maximises cumulative reward."

**Topics**:
- **RL fundamentals**:
  - Agent, environment, state, action, reward
  - Episode: sequence of (state, action, reward) until termination
  - Policy: mapping from states to actions
  - Value function: expected cumulative reward from a state
- **Bellman equations**: expectation + optimality
  - V(s) = E[R + gamma * V(s')]
  - Q(s,a) = E[R + gamma * max_{a'} Q(s', a')]
- **5 algorithms, 5 business applications** (from PCML6-13):
  - **DQN** (Deep Q-Network): customer churn prevention. Discrete actions.
  - **DDPG** (Deep Deterministic Policy Gradient): manufacturing control. Continuous actions.
  - **SAC** (Soft Actor-Critic): dynamic pricing. Handles uncertainty.
  - **A2C** (Advantage Actor-Critic): resource allocation. Variance reduction via baseline.
  - **PPO** (Proximal Policy Optimization): supply chain optimisation. Clipped objective prevents large updates.
- Custom Gymnasium environments for each use case
- Connection to M6: "RLHF uses PPO to align LLMs with human preferences. DPO achieves the same goal without the reward model."

**Key Formulas**:
- Bellman expectation: V(s) = E[R_{t+1} + gamma * V(S_{t+1}) | S_t = s]
- Bellman optimality: Q*(s,a) = E[R_{t+1} + gamma * max_{a'} Q*(S_{t+1}, a') | S_t = s, A_t = a]
- PPO clipped objective: L^{CLIP} = E[min(r_t * A_t, clip(r_t, 1-epsilon, 1+epsilon) * A_t)]
- DQN loss: L = E[(r + gamma * max_{a'} Q(s', a'; theta^-) - Q(s, a; theta))^2]

**Learning Objectives**: Students can:
- Explain the Bellman equations and what they represent
- Implement DQN for a discrete action problem
- Implement PPO for a continuous action problem
- Create custom Gymnasium environments for business applications
- Explain how RL connects to RLHF for LLM alignment (bridge to M6)

**Exercise**: Implement DQN for customer churn prevention. Implement PPO for supply chain optimisation. Create custom environments. Compare performance.

**Assessment Criteria**: Both algorithms converge. Custom environments correctly implement reward functions. PPO→RLHF connection articulated.

**R5 Source**: PCML6-13 (5 algorithms, advanced implementations). Note: RL needs new DECK content — R5 has notebook only.

**End of Module Assessment**: Quiz + DL architecture project (choose a problem, select architecture, train, evaluate, deploy with ONNX).

---

### MODULE 6: Machine Learning with Language Models and Agentic Workflows

**Description**: Build LLM applications, fine-tune models, deploy governed agents. All engineering, all code. Following R5 Deck 6B (10 fine-tuning techniques, agentic design) — adapted from CrewAI to Kaizen.

**Module Learning Objectives**: By the end of M6, students can:
- Use LLMs effectively with prompt engineering and structured output
- Fine-tune LLMs using LoRA and adapter layers (from scratch implementation)
- Survey all 10+ fine-tuning techniques and know when to use each
- Align models using DPO and GRPO
- Build RAG systems with proper evaluation
- Build ReAct agents with tool use and cost budgets
- Orchestrate multi-agent systems and implement MCP servers
- Implement AI governance with PACT (access controls, operating envelopes)
- Deploy full production platforms with Nexus

**Kailash Engines**: Kaizen (Delegate, BaseAgent, Signature, agents), kailash-align (AlignmentPipeline, AdapterRegistry), kailash-pact (GovernanceEngine, PactGovernedAgent), kailash-nexus (Nexus, auth, middleware), kailash-mcp

---

#### Lesson 6.1: LLM Fundamentals, Prompt Engineering, and Structured Output

**Prerequisites**: M5 complete (DL architectures, transformers)
**Spectrum Position**: Semantic feature learning — language models as feature extractors at scale

**Topics**:
- **LLM Foundation** (from Deck 6B):
  - Transformer architecture recap (from M5.4)
  - Pre-training: next token prediction (GPT), masked language modelling (BERT)
  - Scaling laws: parameters, data, compute
  - Notable models: GPT, Claude, Gemini, Llama, Phi, Mistral, Gemma (from Deck 6B)
  - RLHF overview: how production LLMs are aligned (connects M5.8 RL to alignment)
- **Prompt Engineering** (new, from completeness audit):
  - Zero-shot prompting: task description only
  - Few-shot prompting: provide examples
  - Chain-of-thought (CoT): "Let's think step by step"
  - Zero-shot CoT: append "Let's think step by step" without examples
  - Self-consistency: sample multiple CoT paths, majority vote
  - Structured prompting: output format specification (JSON, tables)
  - Prompt engineering as the most immediately practical LLM skill
- **Kaizen Structured Output**:
  - Signature: InputField, OutputField
  - Delegate: streaming, events, cost tracking
  - Type-safe structured output (not free-form text)
- **Inference considerations** (brief): KV-cache, speculative decoding, continuous batching (from completeness audit)

**Learning Objectives**: Students can:
- Explain how LLMs are pre-trained and aligned
- Apply 5+ prompt engineering techniques effectively
- Use Kaizen Delegate for structured LLM output with cost tracking
- Understand basic inference optimisation concepts

**Exercise**: Build classification system using Kaizen Delegate. Compare zero-shot vs few-shot vs CoT prompting on same task. Measure cost and accuracy for each approach.

**Assessment Criteria**: All prompting techniques demonstrated. Accuracy comparison shows when each technique helps. Cost tracking working.

**R5 Source**: Deck 6B (LLM foundation models) + PCML6-12 (adapted CrewAI → Kaizen)

---

#### Lesson 6.2: LLM Fine-tuning — LoRA, Adapters, and the Technique Landscape

**Prerequisites**: 6.1 (LLM fundamentals), 5.4 (transformer architecture)
**Spectrum Position**: Customising language models — making them domain-specific

**Topics**:
- **LoRA (Deep Dive)** (from Deck 6B + PCML6-11):
  - Theory: reduce weight updates to low-rank matrices A x B
  - Pre-trained weights remain frozen, new weights stored separately
  - FROM-SCRATCH implementation: LoRALayer with reset_parameters
  - Connects to M4.3 SVD: LoRA IS low-rank factorisation
- **Adapter Layers (Deep Dive)** (from Deck 6B + PCML6-10):
  - Theory: bottleneck modules (FC → activation → FC) inserted between transformer layers
  - FROM-SCRATCH implementation: AdapterLayer, AdapterTransformerModel
  - Task-specific fine-tuning without changing original weights
- **LoRA vs Adapter comparison** (from Deck 6B slide 8):
  - Parameter update mechanism
  - Parameter efficiency
  - Implementation complexity
  - Flexibility and modularity
- **Fine-tuning Landscape Survey** (remaining 8 techniques from Deck 6B, 30-min lecture + reference table):
  - Prefix Tuning: task-specific vectors prepended to attention K/V
  - Prompt Tuning: learnable prompt tokens added to input
  - Task-specific Fine-tuning: full backprop with LR schedulers + gradient clipping + mixed precision
  - LLRD (Layer-wise Learning Rate Decay): lower LR for earlier layers
  - Progressive Layer Freezing: top-down unfreezing
  - Knowledge Distillation: teacher-student with soft labels
  - Differential Privacy: DPSGD, gradient noise injection
  - Elastic Weight Consolidation: Fisher Information Matrix, prevent catastrophic forgetting
- **Model Merging** (new, from completeness audit):
  - TIES: trim, elect sign, merge
  - DARE: drop and rescale
  - SLERP: spherical linear interpolation
  - Task arithmetic: add/subtract fine-tuned weights
  - Application: combine LoRA adapters from different tasks
- **Quantisation** (brief, from completeness audit):
  - GPTQ, AWQ, GGUF, bitsandbytes
  - QLoRA: quantise base model + LoRA on top
  - When to quantise: deployment on constrained hardware
- **kailash-align**: AlignmentPipeline, AlignmentConfig, AdapterRegistry

**Learning Objectives**: Students can:
- Implement LoRA from scratch (understand the mathematics)
- Implement adapter layers from scratch
- Compare LoRA vs adapters across 4 dimensions
- Survey the full fine-tuning landscape and select the right technique
- Explain model merging techniques and when to use them

**Exercise**: Implement LoRA from scratch on IMDB sentiment classification (from PCML6-11). Implement adapter layers from scratch (from PCML6-10). Compare performance, parameter count, training time. Merge two LoRA adapters with TIES.

**Assessment Criteria**: Both from-scratch implementations work. Comparison quantified. Merging produces functional combined model.

**R5 Source**: Deck 6B slides 3-8 (10 techniques) + PCML6-10 (adapter from scratch) + PCML6-11 (LoRA from scratch)

---

#### Lesson 6.3: Preference Alignment — DPO and GRPO

**Prerequisites**: 6.2 (fine-tuning), 5.8 (RL: PPO)
**Spectrum Position**: Aligning models with human preferences — engineering the training signal

**Topics**:
- **RLHF overview**: reward model + PPO. Why it's complex (reward model training, PPO instability).
- **DPO (Direct Preference Optimization)**:
  - Derive from RLHF: bypass the reward model entirely
  - Bradley-Terry preference model: P(y_w > y_l | x) = sigma(beta * (log pi(y_w|x) - log pi_ref(y_w|x)) - beta * (log pi(y_l|x) - log pi_ref(y_l|x)))
  - Implementation: training loop with preference pairs (chosen, rejected)
  - Hyperparameter beta controls deviation from reference policy
- **GRPO (Group Relative Policy Optimization)** (new, from completeness audit):
  - Used in DeepSeek-R1 (2025)
  - Sample multiple completions, score relative to group mean
  - No reward model needed (like DPO), but maintains policy gradient framework
  - Comparison with DPO: when to use each
- **LLM-as-Judge Evaluation**:
  - Use one LLM to evaluate another's outputs
  - Known biases: position bias, verbosity bias, self-enhancement bias
  - Mitigation strategies: swap positions, normalize lengths
- **Evaluation Benchmarks** (from completeness audit):
  - MMLU: multi-task language understanding
  - HellaSwag: commonsense reasoning
  - HumanEval: code generation
  - MT-Bench: multi-turn conversation quality
  - lm-eval-harness: unified evaluation framework
- **kailash-align**: AlignmentPipeline (method="dpo"), evaluator

**Key Formulas**:
- DPO loss: L_DPO = -E[log sigma(beta * log(pi(y_w|x)/pi_ref(y_w|x)) - beta * log(pi(y_l|x)/pi_ref(y_l|x)))]
- GRPO: advantage estimated relative to group mean reward

**Learning Objectives**: Students can:
- Derive DPO from the RLHF objective
- Implement DPO training with preference pairs
- Explain GRPO and when to prefer it over DPO
- Evaluate fine-tuned models using LLM-as-judge and standard benchmarks
- Use lm-eval-harness for systematic evaluation

**Exercise**: Fine-tune a model with DPO on preference data. Evaluate using LLM-as-judge (measure position and verbosity bias). Run lm-eval benchmarks before and after alignment.

**Assessment Criteria**: DPO training converges. LLM-as-judge biases measured and mitigated. Benchmarks show alignment impact.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 6.4: RAG Systems

**Prerequisites**: 6.1 (LLM fundamentals, prompt engineering), 4.6 (NLP, embeddings)
**Spectrum Position**: Knowledge-augmented generation — grounding LLMs in facts

**Topics**:
- RAG concept: Retrieval-Augmented Generation. External knowledge injected into LLM context.
- **Chunking strategies**: fixed size, sentence, paragraph, semantic. Overlap. Chunk size tradeoffs.
- **Retrieval**:
  - Dense retrieval: sentence embeddings, vector similarity (cosine, dot product)
  - Sparse retrieval: BM25 (from M4.6)
  - Hybrid retrieval: combine dense + sparse
  - Re-ranking: cross-encoder scoring
- **RAGAS evaluation framework**: faithfulness, answer relevance, context relevance, context recall
- **HyDE (Hypothetical Document Embeddings)**: generate hypothetical answer, use it for retrieval
- **Advanced RAG patterns**: multi-hop retrieval, document summarisation, metadata filtering
- **Kaizen RAG agents**: RAGResearchAgent, MemoryAgent

**Key Concepts**: Chunking, dense/sparse retrieval, hybrid retrieval, RAGAS evaluation, HyDE

**Learning Objectives**: Students can:
- Build a complete RAG pipeline from documents to answers
- Compare dense, sparse, and hybrid retrieval approaches
- Evaluate RAG quality using RAGAS metrics
- Implement HyDE for improved retrieval

**Exercise**: Build RAG system on Singapore policy documents. Compare BM25, dense, and hybrid retrieval. Evaluate with RAGAS. Implement HyDE and measure improvement.

**Assessment Criteria**: RAG pipeline end-to-end. Three retrieval methods compared. RAGAS metrics computed. HyDE measurably improves retrieval.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 6.5: AI Agents — ReAct, Tool Use, and Function Calling

**Prerequisites**: 6.1 (LLM fundamentals, Kaizen Delegate)
**Spectrum Position**: Autonomous ML — agents that reason and act

**Topics**:
- **Agent concept**: reason about a task, take actions, observe results, iterate
- **ReAct** (Reasoning + Acting): thought → action → observation loop
- **Chain-of-Thought agents**: step-by-step reasoning before acting
- **Tool use**:
  - Custom tools wrapping Kailash engines (DataExplorer, TrainingPipeline as agent tools)
  - Tool selection: agent decides which tool to use
  - Cost budget safety: LLMCostTracker prevents runaway spending
- **Function calling protocol** (from completeness audit):
  - Structured tool schemas (JSON schema definitions)
  - tool_choice parameter: auto, required, specific function
  - Parallel function calling: multiple tools invoked simultaneously
- **Mental framework for agent creation** (from Deck 6B):
  - What is our goal?
  - What is our thought process?
  - What kind of specialist would we hire? (Be sharp: "researcher" vs "HR research specialist")
  - What tools do they need? (Versatile, fault-tolerant, caching)
- **Agent design considerations** (from Deck 6B):
  - Iterative refinement: critic agent that recommends improvements
  - Human-in-the-loop: pause workflow for validation
  - Monitoring and logging: real-time tracking of intermediate outputs
- **Kaizen agents**: ReActAgent, ChainOfThoughtAgent, custom agents with BaseAgent

**Learning Objectives**: Students can:
- Build ReAct agents with custom tools
- Implement function calling with structured schemas
- Apply cost budgets to prevent runaway LLM spending
- Design agents using the mental framework from Deck 6B

**Exercise**: Build data analysis agent with ReAct: wraps DataExplorer, TrainingPipeline, and ModelVisualizer as tools. Agent autonomously explores data, selects model, trains, reports results. Cost budget enforced.

**Assessment Criteria**: Agent reasons through steps (not random tool calls). Tools correctly invoked. Cost budget respected. Results interpretable.

**R5 Source**: Deck 6B (agent design, mental framework, task definition) + PCML6-12 (adapted CrewAI → Kaizen)

---

#### Lesson 6.6: Multi-Agent Orchestration and MCP

**Prerequisites**: 6.5 (single agents, tool use)
**Spectrum Position**: Agent coordination — multiple specialists working together

**Topics**:
- **Multi-agent patterns**:
  - Supervisor-worker: one agent delegates to specialists
  - Sequential: output of one agent feeds into next
  - Parallel: multiple agents work simultaneously, results aggregated
  - Handoff: agent transfers to specialist when topic changes
- **A2A (Agent-to-Agent) protocol**: structured communication between agents
- **Agent memory** (from Deck 6B slide 12):
  - Short-term memory: current conversation context
  - Long-term memory: persistent knowledge across sessions
  - Entity memory: structured knowledge about people, places, concepts
- **MCP (Model Context Protocol)**:
  - Protocol for exposing tools to agents at scale
  - Tool registration: define tools with schemas
  - Transport: stdio, HTTP/SSE
  - Build an MCP server: expose Kailash engines as MCP tools
- **Agent design (from Deck 6B)**: architectural considerations (modularity, load balancing, dynamic agent creation), security (prevent data leakage between agents)

**Learning Objectives**: Students can:
- Implement supervisor-worker, sequential, and parallel multi-agent patterns
- Build an MCP server that exposes ML tools
- Configure agent memory (short-term, long-term, entity)
- Apply security considerations to agent architectures

**Exercise**: Build multi-agent ML pipeline: DataScientist agent → FeatureEngineer agent → ModelSelector agent → ReportWriter agent. Build MCP server exposing ML tools. Test cross-agent coordination.

**Assessment Criteria**: Multi-agent pipeline produces correct results. MCP server functional. Agents communicate structured outputs.

**R5 Source**: Deck 6B (agent architecture, memory, security) + ASCENT

---

#### Lesson 6.7: AI Governance Engineering

**Prerequisites**: 6.6 (multi-agent systems)
**Spectrum Position**: Governed AI — engineering safety and accountability into systems

**Topics**:
- **PACT framework** (engineering focus):
  - D/T/R addressing: Domain/Team/Role structure for access control
  - GovernanceEngine: `compile_org()` to create governance structure
  - `Address`: identify who is requesting access
  - `can_access()`, `explain_access()`: check and explain access decisions
  - Operating envelopes: define boundaries for what agents can do
    - Task envelopes: restrict agent to specific task types
    - Role envelopes: restrict based on role in organisation
    - Monotonic tightening: envelopes can only get stricter, never looser
  - Enforcement modes: warn, block, audit
  - Fail-closed: if governance check fails, deny access (not fail-open)
- **CostTracker**: budget allocation and cascading
  - Budget cascading: parent agent allocates budget to children
  - What happens when budget runs out: agent stops gracefully
- **PactGovernedAgent**: agent wrapper that enforces governance
- **Audit trails**: log every access decision for compliance
- **Clearance levels**: graduated access based on trust level
- **Governance testing**: test that governance WORKS (denied access stays denied)

**Design Note**: This is ENGINEERING. Students implement access controls, test them, and verify they work. No philosophical discussion of AI ethics frameworks. The code IS the governance.

**Learning Objectives**: Students can:
- Implement PACT governance with D/T/R addressing
- Define and enforce operating envelopes for agents
- Implement budget cascading across agent hierarchies
- Test governance rules (verify denied access stays denied)
- Create audit trails for compliance

**Exercise**: Build governed multi-agent system. Define D/T/R structure. Set operating envelopes (task and role). Implement budget cascading. Write governance tests that verify access controls. Generate audit trail.

**Assessment Criteria**: Governance correctly denies unauthorised access. Operating envelopes enforce boundaries. Budget cascading works. Tests verify governance.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 6.8: Capstone — Full Production Platform

**Prerequisites**: All of M6 (6.1-6.7)
**Spectrum Position**: Integration — ship a complete governed AI system

**Topics**:
- **Nexus deployment**: multi-channel (API + CLI + MCP simultaneously)
  - One codebase, three interfaces
  - Auth: RBAC (Role-Based Access Control), JWT tokens
  - Middleware: rate limiting, logging, CORS
  - Plugins: extend Nexus with custom functionality
- **Production monitoring**: DriftMonitor integration from M3.8
- **Full platform integration**: Core SDK → DataFlow → ML → Kaizen → PACT → Nexus → Align
  - Train model (TrainingPipeline)
  - Persist to DataFlow
  - Wrap in agent (Kaizen)
  - Govern agent (PACT)
  - Deploy (Nexus)
  - Monitor (DriftMonitor)
- **Debugging traces**: understanding agent reasoning chains
- **Testing agents**: automated testing for agentic systems
- **Inference optimisation** (brief, from completeness audit): KV-cache, flash attention, vLLM for production serving
- **Multimodal LLMs** (brief mention): vision-language models (GPT-4V, LLaVA, Gemini) as awareness

**Scaffolding**: ~40% (capstone tests integration, not from-scratch). Students connect existing components, not build everything from zero.

**Learning Objectives**: Students can:
- Deploy a complete AI system with Nexus (API + CLI + MCP)
- Implement authentication and authorization
- Integrate all Kailash packages into a production pipeline
- Debug agent reasoning chains
- Monitor deployed models for drift

**Exercise**: Deploy the M6 multi-agent system via Nexus. Add RBAC authentication. Integrate DriftMonitor. Test end-to-end: query via API, CLI, and MCP. Verify governance enforces access controls at deployment level.

**Assessment Criteria**: System deployed and accessible via all 3 channels. Auth works (unauthenticated requests rejected). Drift monitoring active. Governance enforced in production.

**R5 Source**: ASCENT (new, not in R5)

**End of Module Assessment**: Capstone project presentation + comprehensive quiz.

---

## Part III: Cross-Cutting References

### Complete SDK Coverage Matrix

| Package | Lessons | Key Classes |
|---|---|---|
| **kailash** (core) | M1, M3.6-7, M6.6 | WorkflowBuilder, LocalRuntime, Node, @register_node, PythonCodeNode, ConditionalNode, ConnectionManager, MCP server |
| **kailash-ml** | M1-M5 | DataExplorer, PreprocessingPipeline, FeatureEngineer, FeatureStore, TrainingPipeline, AutoMLEngine, HyperparameterSearch, EnsembleEngine, ModelRegistry, InferenceServer, DriftMonitor, ExperimentTracker, ModelVisualizer, OnnxBridge, RLTrainer |
| **kailash-dataflow** | M3.8 | @db.model, field(), db.express CRUD |
| **kailash-nexus** | M6.8 | Nexus, auth (RBAC/JWT), middleware, plugins |
| **kailash-kaizen** | M6.1, M6.4-6.6 | Signature, InputField/OutputField, Delegate, BaseAgent, RAGResearchAgent, ReActAgent, ChainOfThoughtAgent, MemoryAgent, coordination patterns |
| **kaizen-agents** | M6.5-6.6 | ML agents (DataScientist, FeatureEngineer, ModelSelector, etc.) |
| **kailash-pact** | M6.7 | GovernanceEngine, PactGovernedAgent, Address, CostTracker, operating envelopes |
| **kailash-align** | M6.2-6.3 | AlignmentPipeline, AlignmentConfig, AdapterRegistry, evaluator |

### ASCENT Program Additions Needed

These topics exist in MLFP but do NOT have corresponding ASCENT exercises or deck content:

| Topic | MLFP Lesson | Priority | Notes |
|---|---|---|---|
| SVM, KNN, Naive Bayes | M3.3 | CRITICAL | Need new deck slides + exercises |
| Association rules / market basket | M4.5 | HIGH | Need new deck + exercise |
| Recommender systems (full lesson) | M4.7 | HIGH | PCML5-3 exists as reference, needs Kailash adaptation |
| Prompt engineering | M6.1 | CRITICAL | Need new deck content |
| GRPO | M6.3 | HIGH | Need new content for 2025-2026 curriculum |
| Model merging (TIES, DARE) | M6.2 | MEDIUM | Need new content |
| LLM evaluation benchmarks | M6.3 | HIGH | Need new content |
| RL deck slides | M5.8 | HIGH | R5 has notebook but no deck |
| Diffusion models (DDPM) | M5.5 | MEDIUM | Brief coverage needed |
| Vision Transformers (ViT) | M5.4 | MEDIUM | Brief coverage in transformers lesson |

### Red-Team Resolutions (v2)

| Issue | Resolution in v2 |
|---|---|
| C1: M3.3 overloaded (5 model families) | Split into M3.3 (model zoo) + M3.4 (gradient boosting) |
| C2: SVM/KNN/NB missing | Added to M3.3 |
| C3: Dropout/batch norm missing | Added to M4.8 (DL Training Toolkit) |
| C4: Prompt engineering missing | Added to M6.1 |
| C5: M6.2 attempts 10 techniques | Restructured: 2 deep (LoRA, Adapters) + 8 survey |
| H1-H4: DL training mechanics scattered | Consolidated in M4.8 Training Toolkit section |
| H5: Feature selection missing | Added to M3.1 |
| H6: GRPO missing | Added to M6.3 alongside DPO |
| H7: Model merging missing | Added to M6.2 |
| H8: LLM eval missing | Added to M6.3 |
| H9: M2.6 logistic + ANOVA too dense | Logistic prioritised, ANOVA scoped to one-way only |
| H10: M5.8 RL no transition | Added explicit bridge segment |
| M4.7→4.8 bridge implicit | Made explicit opening segment in M4.8 |
| M5.1 autoencoders 9 variants | Scoped to 4 deep + 5 survey |
| M5 consolidation missing | Added comparison segment after M5.4 (Transformers) |

---

## Version History

| Version | Date | Changes |
|---|---|---|
| v1 | 2026-04-09 | Initial spec based on user's 6-module structure |
| v2 | 2026-04-09 | Red-team + completeness audit incorporation. Added: learning objectives per lesson, key formulas, assessment criteria, DL training toolkit, prompt engineering, GRPO, model merging, SVM/KNN/NB, feature selection, evaluation benchmarks. Split M3.3. Restructured M6.2. |

## What This Spec Replaces

This document (mlfp-curriculum-v2.md) is the **authoritative curriculum spec** for the MLFP course. It supersedes all prior versions (v1, expanded-curriculum-v2/v3/v4).
