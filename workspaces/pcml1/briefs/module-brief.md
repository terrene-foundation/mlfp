# Module 1: Foundations — Statistics, Probability & Data Fluency

**Duration**: 7 hours (3h lecture + 3h lab + 1h assessment)  
**Kailash**: kailash-ml (DataExplorer, PreprocessingPipeline, ModelVisualizer)  
**Scaffolding**: 70%

## Lecture Topics

### 1A: Statistical Foundations (90 min)
- Probability: exponential family distributions, sufficient statistics (Fisher-Neyman factorization), moment-generating functions, convergence types
- Bayesian thinking: prior specification, conjugate priors (reference table: Normal-Normal, Beta-Binomial, Gamma-Poisson, Dirichlet-Multinomial), posterior computation, credible vs confidence intervals
- **MAP estimation** (P0): derive log-posterior = log-likelihood + log-prior, show Gaussian prior → L2 penalty, Laplace prior → L1 penalty — this is the foundation for Module 3's regularization section
- Bayesian vs frequentist: practical decision guide — when to use each, what each answers, production implications
- MLE: derivation for Gaussian, properties (consistency, asymptotic normality, efficiency), Fisher information, **Cramér-Rao lower bound** (Var(θ̂) ≥ 1/I(θ) — why MLE is asymptotically optimal), forward-reference to EM algorithm (Module 4)
- Hypothesis testing: Neyman-Pearson framework + **likelihood ratio tests** (LRT is uniformly most powerful, chi-squared and F-test as special cases), power analysis, multiple testing (Bonferroni, BH-FDR), effect sizes
- **Permutation tests**: distribution-free alternative, algorithm (permute → recompute → null distribution), connects to Module 3 permutation importance
- Bootstrapping: Efron's theory, parametric vs non-parametric, BCa confidence intervals

### 1B: Data Fluency with Polars + Kailash Environment Setup (45 min)
- Arrow backend, multi-threaded execution, lazy evaluation
- Expression API: `pl.col()`, `pl.when()`, `over()` window functions
- Joins, pivots, melts, rolling aggregations, lazy frames
- **kailash_ml.interop** module: the sole conversion point between polars and external frameworks (to_sklearn_input, to_pandas, polars_to_arrow) — teach this from day 1

### 1C: Exploratory Data Analysis at Scale + Data Governance (45 min)
- **ConnectionManager** setup: `ConnectionManager("sqlite:///ml.db")` — the foundation for every persistent engine. Introduce here, reuse in every subsequent module.
- DataExplorer: async profiling, 8 alert types, correlation matrices (Pearson, Spearman, Cramer's V)
- **AlertConfig**: configure which alerts fire — high cardinality (PII risk?), missing patterns, constant columns
- PreprocessingPipeline: auto-detect task, encode, scale, impute
- ModelVisualizer: plotly interactive charts
- **FeatureSchema + FeatureField** preview: "Here is how Kailash describes a dataset" — typed fields, entity IDs, timestamps. Primes students for Module 2's FeatureStore.
- **Data governance foundations**: data classification levels (Public, Internal, Confidential, Restricted, Secret). "Which columns might contain PII? What classification level?"
- **CARE principles** (1 slide): accountability, transparency — the philosophical foundation for everything that follows

## Lab Exercises (5)

1. **Polars + Kailash first contact**: Load HDB resale (15M+ rows), join with MRT/school data, window functions for rolling district prices. **End with DataExplorer.profile(df)** — first Kailash engine call from Exercise 1.
2. **Bayesian estimation**: Posterior distributions for property price parameters using bootstrap + conjugate priors. Use **ModelVisualizer** for plotting posterior distributions (not raw plotly).
3. **DataExplorer profiling**: Async profile of dirty Singapore economic data (mixed granularity CPI + employment + FX)
4. **Hypothesis testing**: A/B test analysis on e-commerce conversion data with multiple testing correction
5. **Challenge**: Full EDA on messy taxi trip data (schema drift, GPS noise, missing fields) → profile → clean → visualize → report

## Datasets
- **Singapore HDB Resale** (data.gov.sg): 15M+ records, merge with ascent_assessment MRT/school parquets
- **Singapore Economic Indicators**: CPI + employment + FX (different reporting frequencies, missing quarters)
- **Singapore Taxi Trips** (LTA): GPS noise, schema changes across years
- **E-commerce A/B Test**: 500K users, conversion + revenue, SRM issues

## Quiz Topics
- Polars expression API (filter, group_by, with_columns)
- DataExplorer alert interpretation
- Bootstrap CI calculation
- Power analysis: "Given n=5000 and baseline=3%, what MDE can you detect at 80% power?"
- Bayesian posterior interpretation

## Deck Opening Case
**Singapore HDB flash crash analysis (Q4 2023 price anomaly)** — EDA catches what dashboards miss. A DataExplorer alert would have flagged the anomaly before it hit the news.
