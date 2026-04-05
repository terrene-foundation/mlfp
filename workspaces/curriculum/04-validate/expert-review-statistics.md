# Expert Review: Modules 1-2 (Statistics & Feature Engineering / Experiment Design)

**Reviewer**: Statistics professor and A/B testing practitioner  
**Standard**: Stanford CS229 / Georgia Tech OMSCS CS 7641 depth  
**Audience**: Working professionals targeting senior ML engineer roles  
**Date**: 2026-04-05

---

## Overall Assessment

Modules 1 and 2 are strong. The existing coverage hits the core topics that a senior ML engineer needs. The design philosophy -- math derivations for core concepts, messy real-world data, Singapore-specific datasets -- is well calibrated for the audience. What follows is a list of gaps and improvements organized by topic, ranging from "must add" to "nice to have."

I grade each recommendation with a priority tier:

- **P0 (Must add)**: A senior ML engineer who lacks this will make consequential errors in production.
- **P1 (Should add)**: Common interview topic or frequent production need. Omission is a noticeable gap.
- **P2 (Nice to have)**: Deepens understanding for top-quartile students. Can be relegated to supplementary reading.

---

## Module 1: Statistics, Probability & Data Fluency

### 1.1 Probability Theory

**Current coverage**: Exponential family distributions, moment-generating functions, convergence types (in probability, in distribution, almost sure).

#### Recommendation 1.1a: Add Sufficient Statistics (P1)

**What to add**: Definition of sufficient statistics, the Fisher-Neyman factorization theorem, and the connection to exponential family (exponential families always admit a finite-dimensional sufficient statistic).

**Why**: Sufficient statistics explain *why* exponential family distributions are so central to ML. Without this, the exponential family section feels like a catalog of distributions rather than a principled framework. When students later encounter feature compression, information bottleneck, or even the intuition behind autoencoders, they lack the vocabulary of "information preservation under reduction." In interviews for senior roles, being asked "why do we care about the exponential family?" and answering "because of sufficient statistics and conjugate priors" is the expected caliber.

**How deep**: One slide with definition + factorization theorem statement (no proof). One worked example: show that the sample mean is sufficient for the Poisson parameter. Connect to why MLE for exponential families reduces to matching sufficient statistics to their expected values.

**Where**: Section 1A, after the exponential family introduction. Approximately 5 minutes.

#### Recommendation 1.1b: Add Conjugate Prior Reference Table (P2)

**What to add**: A one-slide reference table mapping common likelihoods to their conjugate priors: Normal-Normal, Beta-Binomial, Gamma-Poisson, Dirichlet-Multinomial.

**Why**: Lab 2 asks students to compute posterior distributions using conjugate priors for Singapore property data, but the lecture topics list "conjugate priors" without specifying whether a reference table is provided. Students will waste lab time deriving or looking up conjugate pairs if this is not given upfront. The table also reinforces the sufficient-statistics connection (conjugate priors exist precisely because exponential families have finite-dimensional sufficient statistics).

**How deep**: Reference table only, no derivation. One sentence per row explaining the intuition (e.g., "Beta prior on a Binomial probability: your prior belief is equivalent to having already observed alpha-1 successes and beta-1 failures").

**Where**: Section 1A, within Bayesian thinking, immediately after conjugate priors are introduced. 3 minutes.

#### Recommendation 1.1c: Skip Information Geometry (P2, not recommended)

**What**: Information geometry (Fisher-Rao metric, natural gradient, manifold structure of statistical models).

**Why to skip**: This is beautiful mathematics but not production-relevant for the target audience. Natural gradient descent (Amari) is niche even for senior ML engineers. The 90-minute time budget is already tight. If students are interested, point them to Amari's textbook as supplementary reading. The one exception: if Module 6's RL section covers natural policy gradient (it does not currently), then a brief mention here would seed that connection.

**Verdict**: Omit from lecture. Mention in a "further reading" slide if desired.

---

### 1.2 Bayesian Thinking

**Current coverage**: Prior specification, conjugate priors, posterior computation, credible intervals vs confidence intervals.

#### Recommendation 1.2a: Add MAP Estimation Explicitly (P0)

**What to add**: Maximum A Posteriori (MAP) estimation as the bridge between MLE and full Bayesian inference. Show that MAP = MLE + prior, and that L2 regularization is MAP with a Gaussian prior, L1 regularization is MAP with a Laplace prior.

**Why**: This is a P0 because Module 3 teaches regularization (L1/L2 geometry, "regularization as Bayesian prior") but the Bayesian foundation for that statement is never laid in Module 1. If students first encounter the MLE-MAP-fullBayes spectrum in Module 1, the regularization section in Module 3 becomes a satisfying callback rather than a hand-wave. Every senior ML engineer interview cycle I have participated in has included "explain the Bayesian interpretation of regularization." The answer requires MAP.

**How deep**: Full derivation. Show log-posterior = log-likelihood + log-prior, take derivative, show how Gaussian prior yields the L2 penalty term, Laplace prior yields L1. This is a 5-minute derivation that pays dividends in Module 3.

**Where**: Section 1A, after posterior computation, before credible intervals. 5-7 minutes.

#### Recommendation 1.2b: Add Bayesian vs Frequentist Framing (P1)

**What to add**: A structured comparison of the two paradigms. Not a philosophical debate, but a practical decision guide: when to use each, what each one answers, and why the distinction matters for A/B testing (Module 2) and model selection (Module 3).

**Why**: The course covers frequentist hypothesis testing AND Bayesian posterior computation, but never explicitly addresses when to use which. Module 2's A/B testing section is purely frequentist (Neyman-Pearson + sequential testing). Bayesian A/B testing has become standard at companies like Spotify, Dynamic Yield, and platforms like Statsig and Eppo. Without the framing in Module 1, students cannot evaluate the trade-off in Module 2.

**How deep**: One comparison slide with a 2x4 table (paradigm / what probability means / what it answers / when to use / production examples). No derivation. One worked example: "You ran an A/B test. The frequentist says p=0.04. The Bayesian says P(B > A) = 0.92. What do you report to a product manager, and why?"

**Where**: Section 1A, as a capstone after both MLE and Bayesian sections are complete. 5 minutes.

#### Recommendation 1.2c: Add Bayesian Model Comparison -- WAIC and LOO-CV (P1)

**What to add**: Brief overview of model comparison criteria: BIC (asymptotic approximation to marginal likelihood), WAIC (widely applicable information criterion), PSIS-LOO-CV (Pareto-smoothed importance sampling leave-one-out cross-validation). Emphasize that BIC answers "which model is true?" while WAIC/LOO-CV answer "which model predicts best?" -- these are different questions.

**Why**: Model selection is a recurring theme (Module 3 model comparison, Module 4 clustering evaluation, Module 6 fine-tuning evaluation). Students who only know AIC/BIC will misapply them to Bayesian models. PSIS-LOO-CV (the Stan ecosystem standard) is increasingly the recommended approach. For senior engineers working with probabilistic models (uncertainty quantification, Bayesian neural networks, GP-based optimization), this is essential vocabulary.

**How deep**: Intuition only. State the formulas for BIC, WAIC, LOO-CV without derivation. Focus on the conceptual distinction (in-sample penalty vs. out-of-sample predictive accuracy). One sentence on PSIS diagnostics ("if Pareto k > 0.7, your LOO estimate is unreliable").

**Where**: Section 1A, after Bayesian thinking, as a brief forward-reference. 4 minutes. Alternatively, defer to Module 3 Section 3B (Model Evaluation) where it fits more naturally. I weakly prefer Module 1 so students have the vocabulary early.

---

### 1.3 Maximum Likelihood Estimation

**Current coverage**: Derivation, properties (consistency, asymptotic normality, efficiency), Fisher information.

#### Recommendation 1.3a: Add Cramer-Rao Lower Bound (P1)

**What to add**: The Cramer-Rao inequality: Var(theta_hat) >= 1 / I(theta), where I(theta) is Fisher information. State that MLE achieves this bound asymptotically (asymptotic efficiency).

**Why**: Fisher information is already in the curriculum, but its significance is incomplete without the Cramer-Rao bound. The bound answers "how good can ANY estimator be?" -- which is the reason Fisher information matters. Without the bound, Fisher information is an abstract quantity. With the bound, it becomes a benchmark: "no unbiased estimator can do better than this." This directly connects to why MLE is the preferred estimator (it is asymptotically the best you can do). In production, when students need to evaluate whether a custom estimator is reasonable, the Cramer-Rao bound provides the floor.

**How deep**: State the inequality. One worked example: compute Fisher information for the Bernoulli, derive the Cramer-Rao bound on estimating p, show that the sample proportion achieves it. Skip the proof of the inequality itself (it uses Cauchy-Schwarz and is not production-relevant).

**Where**: Section 1A, immediately after Fisher information. 5 minutes.

#### Recommendation 1.3b: Mention EM Algorithm as MLE Optimization (P1)

**What to add**: A brief forward-reference to the EM algorithm as the standard approach for MLE when the likelihood involves latent variables. Do NOT derive EM here (Module 4 derives it for GMMs). Simply state: "When your model has hidden variables (mixture models, missing data), you cannot just take the derivative and set it to zero. The EM algorithm handles this by alternating between estimating the hidden variables and maximizing the likelihood. We will derive it in Module 4."

**Why**: Module 4 introduces EM for Gaussian mixture models, but Module 1 teaches MLE without mentioning its limitations. Students may assume MLE is always a straightforward optimization. The forward-reference sets up Module 4 and also connects to the missing data discussion in Module 1's EDA section (DataExplorer detects missing patterns -- EM is one principled way to handle them).

**How deep**: Two sentences, no math. Pure forward-reference.

**Where**: Section 1A, at the end of MLE, as a "limitations and what comes next" note. 1-2 minutes.

#### Recommendation 1.3c: Skip Newton-Raphson / Numerical Optimization Details (P2, not recommended)

**What**: Detailed coverage of Newton-Raphson, Fisher scoring, or other numerical optimization methods for MLE.

**Why to skip**: The audience learns gradient-based optimization in Modules 3-4 (gradient boosting internals, deep learning training dynamics). Spending time on Newton-Raphson for MLE specifically would duplicate effort. The Kailash engines abstract the optimization. Mention that "MLE is computed via numerical optimization" in one sentence and move on.

**Verdict**: Omit. One sentence reference is sufficient.

---

### 1.4 Hypothesis Testing

**Current coverage**: Neyman-Pearson framework, power analysis, multiple testing correction (Bonferroni, BH-FDR), effect sizes.

#### Recommendation 1.4a: Add Likelihood Ratio Tests (P1)

**What to add**: The likelihood ratio test (LRT) as the most powerful test for composite hypotheses (Neyman-Pearson lemma for simple hypotheses, then generalization). Show that many common tests (chi-squared test, F-test) are special cases of the LRT.

**Why**: The Neyman-Pearson framework is listed but the Neyman-Pearson lemma's conclusion -- that the likelihood ratio test is the uniformly most powerful test -- is the *point* of the framework. Without the LRT, Neyman-Pearson is just "a framework for thinking about Type I and Type II errors," which undersells it. For senior ML engineers evaluating model comparisons (nested model tests, feature significance), the LRT is the workhorse. It also connects to Module 3's model evaluation: "is adding this feature statistically justified?" can be answered by an LRT.

**How deep**: State the Neyman-Pearson lemma (no proof). One example: LRT for testing whether a coin is fair. Show the connection: chi-squared goodness-of-fit is an asymptotic LRT. Deviance in logistic regression is -2 log(likelihood ratio).

**Where**: Section 1A, within hypothesis testing, right after Neyman-Pearson framework. 5 minutes.

#### Recommendation 1.4b: Add Permutation Tests (P1)

**What to add**: Permutation testing as a distribution-free alternative to parametric tests. Basic algorithm: compute test statistic, permute labels, recompute, build null distribution, compute p-value.

**Why**: Permutation tests are the gold standard when parametric assumptions are questionable, which is the default in production ML. They are computationally straightforward (a senior engineer can implement one in 10 lines of Python), require no distributional assumptions, and handle complex test statistics (e.g., "is the difference in AUC between two models significant?" -- no parametric test exists for this, but a permutation test handles it trivially). Module 2's A/B testing section would benefit from students already knowing permutation tests. In production, permutation tests are also used for feature importance (permutation importance in Module 3).

**How deep**: Algorithm description + one visual (null distribution histogram with observed statistic). No proof of validity. Connect forward to permutation feature importance in Module 3.

**Where**: Section 1A, after bootstrapping (both are resampling methods, natural pairing). 5 minutes.

#### Recommendation 1.4c: Add Equivalence Testing / TOST (P1)

**What to add**: Two One-Sided Tests (TOST) for equivalence testing. The key insight: "failure to reject H0" is NOT evidence of equivalence. TOST explicitly tests whether the effect is within a pre-specified equivalence margin.

**Why**: This is directly production-relevant for A/B testing in Module 2. The most common production question is not "is version B better?" but "can we confirm version B is *not worse*?" (guardrail metrics, non-inferiority testing). Every major experimentation platform (Statsig, Eppo, Optimizely) now supports non-inferiority/equivalence testing alongside superiority testing. A senior ML engineer who cannot distinguish "we found no significant difference" from "we confirmed equivalence" will misinterpret guardrail metrics.

**How deep**: State the TOST procedure (two one-sided t-tests against equivalence margins). One worked example: "The new checkout flow had a -0.2% conversion rate change with 95% CI [-0.5%, +0.1%]. The equivalence margin is +/- 1%. Can you confirm non-inferiority?" Show how to compute the TOST conclusion.

**Where**: Section 1A, within hypothesis testing, after power analysis (TOST requires understanding of alpha and power). 5 minutes. Alternatively, defer to Module 2's A/B testing section where it is more contextual.

#### Recommendation 1.4d: Bayesian Hypothesis Testing -- Brief Mention Only (P2)

**What to add**: A one-slide mention of Bayes factors as an alternative to p-values. State the interpretation scale (Jeffreys' scale: BF > 3 is "substantial evidence"). Note that Bayesian A/B testing platforms use posterior probability of superiority rather than Bayes factors.

**Why**: Bayesian A/B testing is increasingly adopted (Spotify, Dynamic Yield, Kameleoon). However, the mathematical machinery (marginal likelihood computation, prior sensitivity of Bayes factors) is too heavy for Module 1's time budget. A brief mention with forward-reference to Module 2 is sufficient.

**How deep**: Intuition only. One slide. No computation.

**Where**: Section 1A, as a closing note in hypothesis testing. 2 minutes.

---

### 1.5 Bootstrapping

**Current coverage**: Efron's theory, parametric vs non-parametric, bootstrap confidence intervals (percentile, BCa).

#### Recommendation 1.5a: Add Wild Bootstrap -- Brief Mention (P2)

**What to add**: One sentence on the wild bootstrap: designed for regression settings with heteroskedastic errors. It resamples residuals while preserving the original design matrix.

**Why**: Relevant when students do regression in Modules 2-3. The standard non-parametric bootstrap resamples entire (x, y) pairs, which can break the covariate structure. The wild bootstrap preserves x and resamples scaled residuals. This is a subtlety that matters in production when computing bootstrap confidence intervals for regression coefficients on heteroskedastic data.

**How deep**: One sentence definition + one sentence on when to use it ("when your regression residuals have non-constant variance"). No derivation.

**Where**: Section 1A, within bootstrapping, as a "variants" note after BCa. 1 minute.

#### Recommendation 1.5b: Add Block Bootstrap for Time Series -- Brief Mention (P2)

**What to add**: One sentence on the block bootstrap: preserves temporal dependence by resampling contiguous blocks rather than individual observations.

**Why**: Module 1 uses Singapore HDB data spanning 25 years and taxi trip data with temporal structure. Module 2 uses time-series features (lag, rolling stats, Fourier). If students apply standard bootstrap to time series data, they destroy the autocorrelation structure and produce invalid confidence intervals. A one-sentence warning prevents this common error.

**How deep**: One sentence: "For time series data, use the block bootstrap (resample contiguous blocks of length l) to preserve temporal dependence. Standard bootstrap assumes i.i.d., which time series violate."

**Where**: Section 1A, within bootstrapping, alongside the wild bootstrap mention. 1 minute.

#### Recommendation 1.5c: Bootstrap-Bagging Connection (P1)

**What to add**: Explicitly state that bagging (bootstrap aggregating) in ensemble learning is *literally* the bootstrap applied to model training. Each bagged model is trained on a bootstrap sample. The variance reduction of bagging is a direct consequence of bootstrap variance estimation.

**Why**: Module 3 teaches bagging with a variance reduction proof. If the bootstrap-bagging connection is made in Module 1, Module 3's ensemble theory section becomes a callback. This is one of the most satisfying cross-module connections in the curriculum and should be seeded early.

**How deep**: One sentence forward-reference: "Bootstrap is not just for confidence intervals. In Module 3, you will see that the entire random forest algorithm is built on bootstrap resampling -- each tree trains on a different bootstrap sample, and the ensemble reduces variance by the same mechanism that bootstrap confidence intervals work."

**Where**: Section 1A, at the end of bootstrapping. 1 minute.

---

### 1.6 Missing Topics Assessment

#### Recommendation 1.6a: Skip Kernel Density Estimation (P2, not recommended for lecture)

**Why to skip**: KDE is useful but niche for the target audience. DataExplorer's distribution analysis handles the practical need. The theory (bandwidth selection, curse of dimensionality for KDE) is interesting but does not pay forward into later modules. If anything, KDE shows up implicitly in HDBSCAN's density estimation (Module 4), where it can be mentioned briefly.

**Verdict**: Omit from Module 1 lecture. Mention in Module 4 if relevant to HDBSCAN discussion.

#### Recommendation 1.6b: Skip Robust Statistics / M-Estimators (P2, not recommended)

**Why to skip**: M-estimators and breakdown points are valuable in classical statistics but the course handles robustness through practical means (DataExplorer outlier alerts, preprocessing pipeline, Winsorization/clipping in feature engineering). The formal theory of influence functions and breakdown points does not pay forward into the ML modules. A senior ML engineer is more likely to need "clip outliers before training" than "derive the influence function of the Huber estimator."

**Verdict**: Omit. If robustness matters, address it practically in Module 2's feature engineering (outlier handling strategies).

#### Recommendation 1.6c: Skip Order Statistics (P2, not recommended)

**Why to skip**: Order statistics are mathematically elegant but have minimal production relevance for the target audience. The one exception (quantile regression) is not in the curriculum.

**Verdict**: Omit entirely.

---

### Module 1 Time Budget Impact

Current 1A time: 90 minutes. My P0 and P1 recommendations add approximately:

| Addition | Time |
|----------|------|
| Sufficient statistics | 5 min |
| Conjugate prior table | 3 min |
| MAP estimation (P0) | 7 min |
| Bayesian vs frequentist framing | 5 min |
| Bayesian model comparison (WAIC/LOO-CV) | 4 min |
| Cramer-Rao bound | 5 min |
| EM forward-reference | 2 min |
| Likelihood ratio tests | 5 min |
| Permutation tests | 5 min |
| Equivalence testing (TOST) | 5 min |
| Bayesian hypothesis testing mention | 2 min |
| Bootstrap-bagging connection | 1 min |
| Wild + block bootstrap mentions | 2 min |
| **Total additions** | **~51 min** |

This exceeds what can fit in 90 minutes without cuts. My recommendation:

**Strategy A (preferred)**: Extend Section 1A to 110 minutes and compress Section 1C (EDA at Scale) to 25 minutes. The EDA section is primarily a Kailash engine demo -- students will learn it hands-on in the lab. The theory content (alert types, correlation types) can be covered in a pre-reading or embedded in the lab instructions.

**Strategy B**: Keep 90 minutes but defer TOST and permutation tests to Module 2 Section 2B (where they are immediately applied to A/B testing). This saves 10 minutes in Module 1 and enriches Module 2. Defer Bayesian model comparison to Module 3 Section 3B (Model Evaluation). This saves another 4 minutes. Net addition: ~37 minutes, which requires compressing 1C by ~17 minutes (feasible).

**Strategy C**: Keep 90 minutes and move all P2 items to supplementary reading. Net P0+P1 addition: ~44 minutes. Still tight. Combine with slight compression of 1C.

I recommend **Strategy B** as the best balance.

---

## Module 2: Feature Engineering & Experiment Design

### 2.1 Feature Engineering Theory

**Current coverage**: Feature selection (MI, Boruta, RFE, stability selection), collinearity (VIF, condition number, eigenvalue), interactions (polynomial, tree-based), temporal features (lag, rolling, Fourier, point-in-time), target encoding (James-Stein, hierarchical, CV), domain-specific (RFM, finance, healthcare).

#### Recommendation 2.1a: Add Feature Hashing (P1)

**What to add**: The hashing trick for high-cardinality categorical features. Hash function maps category values to a fixed-size vector. Collision handling. Trade-off: memory efficiency vs. information loss.

**Why**: High-cardinality categoricals (user IDs, product SKUs, IP addresses, postal codes) are ubiquitous in production ML. One-hot encoding is infeasible when cardinality exceeds ~10K. Target encoding (already covered) is one solution; feature hashing is the other standard approach and is the default in many production systems (Vowpal Wabbit, Spark MLlib, scikit-learn's HashingVectorizer). A 2025 AAAI paper on Probabilistic Hash Embeddings (Amazon Science) shows this remains an active area. A senior ML engineer should know both target encoding and feature hashing and be able to choose between them.

**How deep**: Algorithm description (hash function -> index in fixed-size vector). One visual showing collision. One comparison slide: feature hashing vs. target encoding vs. one-hot (when to use each). No math derivation needed.

**Where**: Section 2A, after target encoding (both solve the same problem: high-cardinality categoricals). 5 minutes.

#### Recommendation 2.1b: Add Entity Embeddings for Categoricals (P1)

**What to add**: Learning dense vector representations for categorical features via a neural network embedding layer. The key insight: embeddings capture semantic similarity between categories (e.g., nearby postal codes get similar embeddings). Originally from Guo & Berkhahn (2016), now standard in tabular deep learning (TabNet, TabTransformer, FT-Transformer).

**Why**: Entity embeddings are the modern alternative to target encoding for high-cardinality categoricals when you have sufficient data. They are standard in recommendation systems, advertising, and e-commerce (all production ML). Module 4 teaches word embeddings (Word2Vec, GloVe) -- entity embeddings are the exact same idea applied to tabular data. Making this connection explicit deepens understanding of both. The Singapore HDB dataset has district codes, flat types, and MRT station names that are natural candidates for entity embeddings.

**How deep**: Intuition + architecture diagram (embedding layer -> concatenate with numerical features -> feedforward network). One sentence on the connection to Word2Vec. No training details (those come in Module 4).

**Where**: Section 2A, after feature hashing. 4 minutes.

#### Recommendation 2.1c: Add Feature Importance Stability Analysis (P1)

**What to add**: The problem of feature importance instability: SHAP values, permutation importance, and tree-based importance can give different rankings across random seeds, train/test splits, and time periods. Stability selection (already listed) partially addresses this. Add: how to diagnose instability (run importance N times with different seeds, compute rank correlation), and why unstable features should be treated with suspicion.

**Why**: This is a common production failure mode. A feature that ranks #1 in one run and #15 in another is not reliably informative -- it may be picking up noise. Senior ML engineers need to distinguish stable signals from fragile ones. This also connects to Module 3's SHAP interpretability section: "if SHAP says this feature matters, how confident should you be?"

**How deep**: One slide showing a rank instability plot (feature rankings across 10 bootstrap runs). One algorithm: "run importance 10 times, compute Spearman rank correlation, flag features with high rank variance." No math derivation.

**Where**: Section 2A, after feature selection methods (Boruta, RFE, stability selection). This is a natural extension of stability selection. 4 minutes.

#### Recommendation 2.1d: Add Automated Feature Engineering Mention (P2)

**What to add**: Brief mention of automated feature engineering approaches (Deep Feature Synthesis / Featuretools, OpenFE). The idea: given relational tables, automatically generate features by composing primitives (mean, count, max) across entity relationships.

**Why**: Kailash's FeatureEngineer already does automated feature generation (Lab 5 uses it). Mentioning the conceptual framework (DFS) gives students vocabulary. However, this is P2 because the practical implementation is covered by the Kailash engine, and the theory is lightweight.

**How deep**: One sentence: "Automated feature engineering systems like Deep Feature Synthesis recursively compose aggregation and transformation primitives across relational tables. Kailash's FeatureEngineer implements a similar approach." No code, no derivation.

**Where**: Section 2A, at the end, as a forward-reference to the FeatureEngineer lab. 1 minute.

---

### 2.2 Causal Inference

**Current coverage**: Potential outcomes (Rubin), DAGs (Pearl), do-calculus, diff-in-diff, regression discontinuity, instrumental variables, propensity score matching, Double ML.

#### Recommendation 2.2a: Add Synthetic Control Method (P1)

**What to add**: The synthetic control method (Abadie, Diamond, Hainmueller 2010): when you have one treated unit and multiple control units over time, construct a "synthetic" control as a weighted combination of donor units that matches the treated unit's pre-treatment trajectory.

**Why**: The existing coverage has diff-in-diff, which requires parallel trends. Synthetic control relaxes this assumption and is the standard method when you have a single treated unit (one city, one country, one product launch). This is extremely common in tech: "We launched feature X in Singapore. How do we estimate its effect using Malaysia, Thailand, and Indonesia as controls?" Lab 4 does diff-in-diff on Singapore housing cooling measures -- adding synthetic control as an alternative estimator for the same dataset would be an excellent comparison exercise.

**How deep**: Intuition + algorithm (optimize donor weights to match pre-treatment outcomes). One visual: pre-treatment fit, post-treatment divergence. No formal optimization derivation. Mention the connection to diff-in-diff: "DiD assumes parallel trends; synthetic control constructs a control that matches the treated unit's actual trajectory."

**Where**: Section 2B, after diff-in-diff (natural comparison). 5 minutes.

#### Recommendation 2.2b: Add Heterogeneous Treatment Effects / CATE (P0)

**What to add**: Conditional Average Treatment Effect (CATE) estimation. The question shifts from "what is the average effect?" to "for whom is the effect largest?" Cover the meta-learner framework:
- S-learner: single model with treatment indicator as a feature
- T-learner: separate models for treatment and control
- X-learner: uses both, with propensity-weighted crossover estimation
- Causal forests (Wager & Athey): random forests adapted for CATE estimation

**Why**: This is P0 because heterogeneous treatment effects are the bridge between causal inference and personalization -- the single most valuable application of causal ML in industry. Every recommendation system, pricing algorithm, and marketing campaign optimization depends on knowing *who* benefits most from treatment. The 2025 Fisher-Schultz lecture at Econometrica was on "Generic Machine Learning Inference on Heterogeneous Treatment Effects" (Chernozhukov et al.), confirming this is the frontier of applied causal ML. Uber's CausalML library (actively maintained, 5K+ GitHub stars) implements all of these. A senior ML engineer who only knows ATE but not CATE is missing the industry standard.

**How deep**: Intuition for each meta-learner (one slide each, with a diagram). Mathematical formulation of the S-learner and T-learner (straightforward). X-learner formula (state without derivation). Causal forests: explain the key insight (honest splitting -- use one half of data for tree structure, other half for estimation). Connect to Module 3's random forest: "causal forests modify the splitting criterion to maximize treatment effect heterogeneity rather than prediction accuracy."

**Where**: Section 2B, after Double ML (both are ML + causal inference). 10 minutes. This is the biggest single addition in Module 2 and is worth the time.

#### Recommendation 2.2c: Add Uplift Modeling as Application of CATE (P1)

**What to add**: Uplift modeling as the production application of CATE estimation. The use case: given limited marketing budget, target customers with the highest predicted uplift (CATE), not the highest predicted conversion. The four customer segments: persuadables (high CATE), sure things (buy regardless), lost causes (do not buy regardless), sleeping dogs (treatment hurts).

**Why**: This is the most common industry application of CATE and makes the abstract concept concrete. It directly connects to the e-commerce A/B test dataset used in Lab 3. "You ran an A/B test on a marketing campaign. Instead of just reporting the average effect, identify which customer segments had the largest uplift." This could be a powerful extension to Lab 3 or a separate mini-exercise.

**How deep**: Conceptual framework (4 segments diagram). One sentence on each meta-learner's suitability for uplift. No new math beyond CATE.

**Where**: Section 2B, immediately after CATE, as its application. 3 minutes.

#### Recommendation 2.2d: Skip Bayesian Structural Time Series (P2, not recommended)

**What**: BSTS (CausalImpact package): Bayesian state-space model for estimating causal impact of interventions on time series.

**Why to skip**: While useful, BSTS requires substantial Bayesian time series machinery (state space models, Kalman filtering) that is not covered in the curriculum. Adding it would require 15+ minutes of prerequisites. Synthetic control covers the same use case (single-unit intervention) with simpler machinery. If time allows, mention CausalImpact as a one-sentence "this also exists" reference.

**Verdict**: Omit from lecture. Mention in supplementary reading.

---

### 2.3 A/B Testing

**Current coverage**: Power analysis, MDE, multi-armed bandits (Thompson sampling, UCB), sequential testing (always-valid p-values).

#### Recommendation 2.3a: Add Bayesian A/B Testing (P0)

**What to add**: The Bayesian approach to A/B testing: posterior distribution over the conversion rate, probability of being best, expected loss, credible intervals for the treatment effect. Contrast with frequentist: no fixed sample size, natural "peeking" behavior, directly answers "what is the probability that B is better than A?"

**Why**: This is P0 because Bayesian A/B testing is now the default at multiple major platforms (Spotify, Dynamic Yield, Kameleoon, AB Tasty) and is a standard option in all major experimentation platforms (Statsig, Eppo, Amplitude, LaunchDarkly). The industry has moved toward Bayesian methods because they answer the question product managers actually ask ("how likely is it that this variant wins?") rather than the question frequentist methods answer ("if there were no effect, how surprising would these results be?"). A senior ML engineer who can only do frequentist A/B testing is missing the industry standard.

Module 1's Bayesian thinking section (with the MAP addition) provides the foundation. Module 2 should apply it.

**How deep**: Worked example using Beta-Binomial model for conversion rates. Compute posterior, compute P(B > A) via simulation (10 lines of Python). Compare to the frequentist p-value for the same data. Discuss when they disagree and why.

**Where**: Section 2B, within A/B testing, after power analysis and before sequential testing. 8 minutes.

#### Recommendation 2.3b: Add Metric Design Hierarchy (P0)

**What to add**: The three-tier metric hierarchy used by mature experimentation teams:
1. **Primary / success metric**: what you are trying to move (e.g., conversion rate)
2. **Guardrail metrics**: what must NOT degrade (e.g., page load time, revenue, user retention) -- tested via non-inferiority / equivalence testing (connects to TOST from Module 1)
3. **Secondary / driver metrics**: diagnostic metrics that explain WHY the primary moved (e.g., click-through rate, add-to-cart rate)

Also cover: Overall Evaluation Criterion (OEC), metric sensitivity analysis, and the principle that guardrail metrics should have pre-specified non-inferiority margins.

**Why**: This is P0 because every production A/B test at any serious tech company uses this hierarchy, and getting it wrong is the #1 source of experimentation failures. Without guardrail metrics, you optimize conversion at the expense of revenue. Without driver metrics, you cannot diagnose why an experiment succeeded or failed. The e-commerce A/B test lab (Lab 3) already has "multiple metric correction" -- this recommendation structures that into a principled framework.

Netflix monitors sample mismatch ratio, engagement, retention, and conversion. Airbnb monitors revenue, bounce rate, page load speed, and bookings. Bing monitors page load time and page returns. These are not ad hoc -- they follow the guardrail/driver/primary hierarchy.

**How deep**: Framework slide with the three tiers. One example from the lab dataset. Connect guardrail testing to equivalence testing (TOST). Discuss the false positive trade-off: "tracking 10 guardrail metrics means ~40% chance of at least one false alarm at alpha=0.05 -- apply BH-FDR correction."

**Where**: Section 2B, at the start of A/B testing methodology (before power analysis -- you need to know your metrics before you compute power). 6 minutes.

#### Recommendation 2.3c: Add Network Effects / Interference (P1)

**What to add**: The SUTVA violation problem: when treatment assignment of one user affects outcomes of another (marketplace, social network, ride-sharing). Solutions: cluster randomization, switchback experiments, ego-network randomization. Real examples: eBay overestimated treatment effects by 100%+ due to seller interference; Airbnb's guest fee experiment showed 32.6% bias from interference.

**Why**: This is the #1 threat to A/B test validity at marketplace and platform companies, which is where many senior ML engineers work. The course already uses e-commerce data -- in any two-sided marketplace, buyer and seller effects interfere. A senior ML engineer should at minimum be able to diagnose when SUTVA is violated and know the solution vocabulary (cluster randomization, switchback).

**How deep**: State the SUTVA assumption. One example of violation (marketplace). One slide on cluster randomization (diagram). One slide on switchback experiments (time/region grid). Mention that cluster randomization requires 3-10x larger sample sizes. No formal derivation.

**Where**: Section 2B, after the core A/B testing methods, as "when standard A/B tests break." 6 minutes.

#### Recommendation 2.3d: Add Interleaving Experiments -- Brief Mention (P2)

**What to add**: Interleaving as an alternative to A/B testing for ranking systems (search, recommendations). Instead of showing different users different results, show each user a mix of results from both systems and measure preference.

**Why**: Relevant for recommendation/search engineers. More sensitive than A/B testing for ranking quality (Netflix research shows interleaving detects differences with 100x fewer users). However, niche -- only applies to ranking systems.

**How deep**: One sentence definition. Forward-reference to ranking evaluation.

**Where**: Section 2B, as a footnote. 1 minute.

---

### 2.4 CUPED / Variance Reduction

**Current coverage**: CUPED/CUPAC, stratification, pre-experiment covariates.

#### Recommendation 2.4a: Explain CUPED Math Explicitly (P0)

**What to add**: The full regression adjustment derivation:
- Start with the adjusted metric: Y_adj = Y - theta * X, where X is the pre-experiment covariate
- Show that theta = Cov(Y, X) / Var(X) minimizes Var(Y_adj)
- Show that Var(Y_adj) = Var(Y) * (1 - rho^2), where rho is the correlation between Y and X
- Therefore, higher correlation between the metric and its pre-experiment value means more variance reduction
- Connect to ANCOVA: CUPED is regression adjustment with pre-experiment data as the covariate

**Why**: This is P0 because the quiz already asks "By how much would this reduce your CI width?" -- students cannot answer this without the formula Var(Y_adj) = Var(Y)(1 - rho^2). The derivation is 5 lines of algebra and produces one of the most useful formulas in applied experimentation. Every experimentation platform (Statsig, Eppo, Optimizely) implements this. The 2025 KDD paper "Variance Reduction in Online Marketplace A/B Testing" and the 2025 arXiv paper on CUPED with trimmed means show this remains an active research area.

Also mention CUPAC: uses ML predictions (instead of raw pre-experiment metric) as the covariate X. This can achieve higher rho and thus more variance reduction, at the cost of model complexity.

**How deep**: Full derivation (5 lines). Worked example: "Pre-experiment metric correlation is rho=0.6. Variance reduction is 1 - 0.36 = 64%. Your 4-week experiment now needs only 1.5 weeks." State CUPAC extension.

**Where**: Section 2B, within variance reduction. This should be the core of the CUPED coverage, not a supplement. 7 minutes.

#### Recommendation 2.4b: Add Pre-Experiment Covariate Selection Guidance (P1)

**What to add**: Practical guidance on choosing the pre-experiment covariate:
- Use the same metric from 1-2 weeks before the experiment (highest correlation, simplest)
- Longer pre-windows capture more variance but introduce more noise from non-stationarity
- Multiple covariates: use multivariate CUPED (regression on multiple pre-experiment features)
- CUPAC: train a model on pre-experiment features, use predicted Y as the single covariate

**Why**: The CUPED formula tells you that rho matters, but does not tell you how to choose X to maximize rho. In practice, this is where teams get it wrong: they use a pre-experiment window that is too long (noisy) or too short (low correlation), or they use a covariate that is affected by the treatment assignment (invalid).

**How deep**: Practical heuristics, no math. One slide with a decision tree: "same metric 1-2 weeks prior -> multivariate CUPED with multiple pre-metrics -> CUPAC with ML model."

**Where**: Section 2B, immediately after CUPED derivation. 3 minutes.

---

### 2.5 Double ML

**Current coverage**: "Using ML for nuisance parameter estimation in causal models."

#### Recommendation 2.5a: Deepen Double ML Coverage (P1)

**What to add**: Make the two key ideas explicit:
1. **Orthogonalization / Neyman orthogonality**: residualize the treatment and outcome with respect to confounders using ML, then regress residuals on residuals. This removes the regularization bias that arises from directly plugging ML predictions into causal formulas.
2. **Cross-fitting**: split data into K folds; train nuisance models on K-1 folds, predict on the held-out fold. This prevents overfitting bias analogous to in-sample prediction.

Show the algorithm:
- Step 1: Estimate E[Y|X] and E[D|X] using any ML model (via cross-fitting)
- Step 2: Compute residuals: Y_tilde = Y - E[Y|X], D_tilde = D - E[D|X]
- Step 3: Regress Y_tilde on D_tilde. The coefficient is the causal effect.

**Why**: The current one-line description ("ML for nuisance parameter estimation") does not convey why Double ML works or why naive ML fails. The regularization bias issue is the crux: naive ML shrinks coefficients toward zero, biasing the causal estimate. Orthogonalization removes this bias. A senior ML engineer using causal inference in production needs to understand why you cannot just "throw XGBoost at the confounders" without the orthogonalization step.

**How deep**: State the algorithm (3 steps above). One worked example: "Estimate the effect of a marketing campaign on revenue, controlling for 50 confounders." Show that naive regression gives a biased estimate, Double ML corrects it. Skip the semiparametric efficiency theory.

**Where**: Section 2B, expanding the existing Double ML bullet. 7 minutes total (currently ~2 minutes).

#### Recommendation 2.5b: Skip TMLE / Targeted Learning (P2, not recommended)

**What**: Targeted Minimum Loss-Based Estimation (TMLE) -- a competing approach to Double ML for semiparametric causal inference.

**Why to skip**: TMLE requires understanding of influence functions, efficient influence curves, and targeted updates to initial estimators. This is a full lecture in a causal inference course. The target audience will benefit more from deep understanding of one method (Double ML) than surface exposure to two. TMLE and Double ML solve the same problem (semiparametric efficient estimation with ML nuisance parameters). They differ in the updating mechanism (TMLE uses a targeting step; Double ML uses orthogonalization). For a 90-minute section covering all of experiment design and causal inference, one method done well is better than two done shallowly.

**Verdict**: Omit from lecture. Mention in supplementary reading: "TMLE (van der Laan & Rose, 2011) is an alternative to Double ML that uses a targeting step instead of orthogonalization. Both achieve semiparametric efficiency."

#### Recommendation 2.5c: Skip Debiased/Orthogonal Estimation as Separate Topic (P2, not recommended)

**What**: Debiased machine learning as a general framework beyond Double ML.

**Why to skip**: Double ML IS debiased/orthogonal ML. They are the same thing. Chernozhukov et al. (2018) is the canonical reference for both names. There is no additional content to add -- deepening Double ML (Recommendation 2.5a) covers the debiased/orthogonal estimation framework.

**Verdict**: Already covered by Recommendation 2.5a.

---

### Module 2 Time Budget Impact

Current 2B time: 90 minutes. My P0 and P1 recommendations add approximately:

| Addition | Time |
|----------|------|
| Feature hashing | 5 min |
| Entity embeddings | 4 min |
| Feature importance stability | 4 min |
| Synthetic control | 5 min |
| CATE / heterogeneous treatment effects (P0) | 10 min |
| Uplift modeling | 3 min |
| Bayesian A/B testing (P0) | 8 min |
| Metric design hierarchy (P0) | 6 min |
| Network effects / interference | 6 min |
| CUPED math derivation (P0) | 7 min |
| Pre-experiment covariate selection | 3 min |
| Deepen Double ML | 5 min (net +5 over current) |
| **Total additions** | **~66 min** |

This is substantial. Module 2 has two 90-minute lecture blocks (2A: Feature Engineering, 2B: Experiment Design). The additions split as:

- **Section 2A** (Feature Engineering): +13 min (feature hashing, entity embeddings, stability analysis, automated FE mention)
- **Section 2B** (Experiment Design): +53 min (everything else)

Section 2B is overloaded. Recommendations:

**Strategy A (preferred)**: Rebalance sections. Move some feature engineering content to pre-reading (domain-specific engineering is a reference, not a lecture topic -- students can read about RFM and technical indicators). This frees ~10 minutes in 2A. Redistribute: make 2A = 80 min, 2B = 100 min. Still tight; additionally compress the Section 2C engine demo to 20 min (students learn it in the lab).

**Strategy B**: Defer some P1 items. Specifically:
- Move network effects/interference to supplementary reading (saves 6 min) -- it is important but the audience may not all work at marketplace companies
- Move synthetic control to supplementary reading (saves 5 min) -- DiD covers the most common case
- Move interleaving mention to supplementary reading (saves 1 min)
- Net savings: 12 min. Section 2B net addition: +41 min. With 2C compression to 20 min, feasible.

**Strategy C**: Accept a longer Module 2 lecture (3.5h instead of 3h) and reduce lab time by 30 minutes. The labs already have 5 exercises for 3 hours; dropping one exercise (e.g., Lab 5 which is engine-focused) frees time. I do not love this option because lab time is precious.

I recommend **Strategy A** with **Strategy B** as fallback.

---

## Cross-Module Connection Recommendations

These are not new topics but explicit bridges between Modules 1 and 2 that strengthen both.

### Connection 1: Bootstrap (M1) -> Bagging (M3)

Already recommended in 1.5c. One sentence in Module 1 seeds the connection.

### Connection 2: MAP (M1) -> Regularization (M3)

MAP estimation in Module 1 explains why L1/L2 regularization works in Module 3. The Module 3 slide on "regularization as Bayesian prior" should reference back to Module 1.

### Connection 3: TOST (M1 or M2) -> Guardrail Metrics (M2)

Equivalence testing is the statistical foundation for guardrail metrics. If TOST is deferred to Module 2 (Strategy B for Module 1), it should be introduced before the metric hierarchy slide.

### Connection 4: Permutation Tests (M1) -> Permutation Importance (M3)

Permutation tests in Module 1 are the foundation for permutation feature importance in Module 3's SHAP section.

### Connection 5: CATE (M2) -> Causal Forests (M2) -> Random Forests (M3)

Causal forests modify the random forest splitting criterion. If CATE is covered in Module 2, Module 3's ensemble theory section can reference back.

---

## Summary of All Recommendations

### Module 1

| # | Topic | Priority | Time | Recommended Action |
|---|-------|----------|------|--------------------|
| 1.1a | Sufficient statistics | P1 | 5 min | Add to 1A after exponential family |
| 1.1b | Conjugate prior table | P2 | 3 min | Add as reference slide |
| 1.1c | Information geometry | P2 | -- | Skip (supplementary reading) |
| 1.2a | MAP estimation | **P0** | 7 min | Add to 1A after posterior computation |
| 1.2b | Bayesian vs frequentist framing | P1 | 5 min | Add to 1A as capstone |
| 1.2c | Bayesian model comparison (WAIC, LOO-CV) | P1 | 4 min | Add to 1A or defer to M3 |
| 1.3a | Cramer-Rao bound | P1 | 5 min | Add to 1A after Fisher information |
| 1.3b | EM algorithm forward-reference | P1 | 2 min | Add to 1A end of MLE |
| 1.3c | Newton-Raphson / numerical optimization | P2 | -- | Skip |
| 1.4a | Likelihood ratio tests | P1 | 5 min | Add to 1A within hypothesis testing |
| 1.4b | Permutation tests | P1 | 5 min | Add to 1A after bootstrap |
| 1.4c | Equivalence testing (TOST) | P1 | 5 min | Add to 1A or defer to M2 |
| 1.4d | Bayesian hypothesis testing | P2 | 2 min | Brief mention |
| 1.5a | Wild bootstrap | P2 | 1 min | Brief mention |
| 1.5b | Block bootstrap (time series) | P2 | 1 min | Brief mention |
| 1.5c | Bootstrap-bagging connection | P1 | 1 min | Forward-reference to M3 |

### Module 2

| # | Topic | Priority | Time | Recommended Action |
|---|-------|----------|------|--------------------|
| 2.1a | Feature hashing | P1 | 5 min | Add to 2A after target encoding |
| 2.1b | Entity embeddings | P1 | 4 min | Add to 2A after feature hashing |
| 2.1c | Feature importance stability | P1 | 4 min | Add to 2A after feature selection |
| 2.1d | Automated FE (DFS) mention | P2 | 1 min | Brief mention |
| 2.2a | Synthetic control | P1 | 5 min | Add to 2B after DiD |
| 2.2b | CATE / heterogeneous treatment effects | **P0** | 10 min | Add to 2B after Double ML |
| 2.2c | Uplift modeling | P1 | 3 min | Add to 2B after CATE |
| 2.2d | Bayesian structural time series | P2 | -- | Skip (supplementary reading) |
| 2.3a | Bayesian A/B testing | **P0** | 8 min | Add to 2B within A/B testing |
| 2.3b | Metric design hierarchy | **P0** | 6 min | Add to 2B start of A/B testing |
| 2.3c | Network effects / interference | P1 | 6 min | Add to 2B after core A/B methods |
| 2.3d | Interleaving experiments | P2 | 1 min | Brief mention |
| 2.4a | CUPED math derivation | **P0** | 7 min | Expand existing CUPED coverage |
| 2.4b | Pre-experiment covariate selection | P1 | 3 min | Add after CUPED derivation |
| 2.5a | Deepen Double ML (orthogonalization + cross-fitting) | P1 | +5 min | Expand existing coverage |
| 2.5b | TMLE / targeted learning | P2 | -- | Skip (supplementary reading) |
| 2.5c | Debiased ML as separate topic | P2 | -- | Skip (already covered by 2.5a) |

### P0 Summary (must add)

1. **MAP estimation** (Module 1) -- foundation for regularization in Module 3
2. **CATE / heterogeneous treatment effects** (Module 2) -- bridge between causal inference and personalization; industry standard
3. **Bayesian A/B testing** (Module 2) -- now the default at major platforms
4. **Metric design hierarchy** (Module 2) -- guardrail/driver/primary is how every mature experimentation team operates
5. **CUPED math derivation** (Module 2) -- the quiz tests this; students need the formula

---

## Sources

- [CS229: Machine Learning (Stanford)](https://cs229.stanford.edu/)
- [CS229 Syllabus and Course Schedule](https://cs229.stanford.edu/syllabus-new.html)
- [CS 229 - Probabilities and Statistics Refresher (Shervine Amidi)](https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics)
- [CS 7641: Machine Learning (Georgia Tech OMSCS)](https://omscs.gatech.edu/cs-7641-machine-learning)
- [OMSCS ML Specialization](https://omscs.gatech.edu/specialization-machine-learning)
- [Variance Reduction in Online Marketplace A/B Testing (KDD 2025)](https://kdd2025.kdd.org/wp-content/uploads/2025/07/CameraReady-05.pdf)
- [Improving Sensitivity in A/B Tests: Integrating CUPED with Trimmed Mean Techniques (arXiv 2025)](https://arxiv.org/abs/2510.03468)
- [CUPED Explained (Statsig)](https://www.statsig.com/blog/cuped)
- [Understanding CUPED (Matteo Courthoud)](https://matteocourthoud.github.io/post/cuped/)
- [CUPED's Sting: More Power More Underpowered A/B Tests (Conductrics)](https://blog.conductrics.com/cupeds-sting-more-power-more-underpowered-a-b-tests/)
- [Double/Debiased Machine Learning (Chernozhukov et al.)](https://arxiv.org/abs/1608.00060)
- [DoubleML Documentation - Basics](https://docs.doubleml.org/stable/guide/basics.html)
- [TMLE and DML comparison (PubMed)](https://pubmed.ncbi.nlm.nih.gov/31742333/)
- [Fisher-Schultz Lecture: Generic ML Inference on Heterogeneous Treatment Effects (Econometrica 2025)](https://www.econometricsociety.org/publications/econometrica/2025/07/01/FisherSchultz-Lecture-Generic-Machine-Learning-Inference-on-Heterogeneous-Treatment-Effects-in-Randomized-Experiments-With-an-Application-to-Immunization-in-India/file/ecta200800.pdf)
- [Heterogeneous Treatment Effects and Personalization (Causal Inference for the Brave and True)](https://matheusfacure.github.io/python-causality-handbook/18-Heterogeneous-Treatment-Effects-and-Personalization.html)
- [CausalML (Uber) - Methodology](https://causalml.readthedocs.io/en/latest/methodology.html)
- [Bayesian A/B Testing vs Frequentist (Statsig)](https://www.statsig.com/perspectives/bayesian-ab-testing-vs-frequentist)
- [Frequentist vs Bayesian A/B Testing (AB Tasty)](https://www.abtasty.com/blog/bayesian-ab-testing/)
- [Risk-Aware Product Decisions in A/B Tests with Multiple Metrics (Spotify Engineering)](https://engineering.atspotify.com/2024/03/risk-aware-product-decisions-in-a-b-tests-with-multiple-metrics)
- [Guardrail Metrics for A/B Tests (PostHog)](https://posthog.com/product-engineers/guardrail-metrics)
- [What Are Guardrail Metrics (Eppo)](https://www.geteppo.com/blog/what-are-guardrail-metrics-with-examples)
- [How Meta Tests Products with Strong Network Effects](https://medium.com/@AnalyticsAtMeta/how-meta-tests-products-with-strong-network-effects-96003a056c2c)
- [Clustered Switchback Experiments: Near-Optimal Rates Under Spatiotemporal Interference (arXiv)](https://arxiv.org/html/2312.15574v2)
- [Balancing Network Effects in Experiments (DoorDash)](https://careersatdoordash.com/blog/balancing-network-effects-learning-effects-and-power-in-experiments/)
- [The Interplay Between Design and Analysis of Experiments (HDSR, Winter 2026)](https://hdsr.mitpress.mit.edu/pub/ov1ilnnf)
- [Probabilistic Hash Embeddings (AAAI 2025 / Amazon Science)](https://arxiv.org/abs/2511.20893)
- [Feature Hashing (Wikipedia)](https://en.wikipedia.org/wiki/Feature_hashing)
- [Automated Feature Engineering with FeatureTools (Hopsworks)](https://www.hopsworks.ai/post/automated-feature-engineering-with-featuretools)
- [Bayesian Model Comparison - WAIC and LOO-CV (Stan)](http://mc-stan.org/loo/reference/loo-package.html)
- [Understanding Predictive Information Criteria for Bayesian Models (Gelman et al.)](https://sites.stat.columbia.edu/gelman/research/published/waic_understand3.pdf)
- [Cramer-Rao Bound (Stanford Stats 200, Lecture 15)](https://web.stanford.edu/class/stats200/Lecture15.pdf)
- [Exponential Families and Information Inequality (U of Toronto)](https://erdogdu.github.io/csc2532/lectures/lecture02.pdf)
- [Permutation Testing for Dependence in Time Series (J. Time Series Analysis)](https://ideas.repec.org/a/bla/jtsera/v43y2022i5p781-807.html)
- [Robust TOST Procedures (TOSTER)](https://aaroncaldwell.us/TOSTERpkg/articles/robustTOST.html)
