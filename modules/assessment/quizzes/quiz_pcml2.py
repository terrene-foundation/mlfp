# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 2 — AI-Resilient Assessment Questions

Feature Engineering & Experiment Design
Covers: Bayesian inference, MLE, hypothesis testing, CUPED, bootstrap,
        causal inference, FeatureEngineer, FeatureStore, ExperimentTracker
"""

QUIZ = {
    "module": "ASCENT2",
    "title": "Feature Engineering & Experiment Design",
    "questions": [
        # ── Lesson 1: MLE and Bayesian inference ──────────────────────────
        {
            "id": "2.1.1",
            "lesson": "2.1",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 1, MLE on 4-room HDB prices (2020+) gives "
                "mle_mean = S$485,000 and mle_std = S$78,000. "
                "You then compute a Normal-Normal posterior with a prior mean of S$450,000 "
                "and prior std of S$50,000. The posterior mean comes out to S$481,200. "
                "Why is the posterior mean closer to the MLE than to the prior, "
                "and what property of the data causes this?"
            ),
            "options": [
                "A) The prior is wrong; a correct prior would always equal the MLE",
                "B) Normal-Normal conjugacy always places the posterior mean exactly at the MLE",
                "C) With a large sample (n >> 1/σ₀²), the likelihood dominates the prior — the posterior mean is pulled strongly toward the MLE and the prior's influence shrinks proportionally to 1/n",
                "D) The posterior mean is closer to the MLE because the prior std is too small"
            ],
            "answer": "C",
            "explanation": (
                "In Normal-Normal conjugacy, the posterior precision is 1/σ₀² + n/σ². "
                "With many observations, n/σ² dominates and the posterior mean approaches x̄ (the MLE). "
                "The prior contributes weight proportional to 1/σ₀² — with a large sample this weight "
                "becomes negligible. This is why Bayesian and frequentist estimates converge as n grows."
            ),
            "learning_outcome": "Explain why posterior mean converges to MLE with large samples",
        },
        {
            "id": "2.1.2",
            "lesson": "2.1",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 1 computes a 95% Bayesian credible interval [CI_low, CI_high] "
                "and a 95% bootstrap confidence interval. A regulator asks: "
                "'Does this mean there is a 95% probability the true mean lies in [CI_low, CI_high]?' "
                "Answer correctly for the Bayesian interval and the bootstrap interval separately."
            ),
            "options": [
                "A) Bayesian credible interval: yes, given the prior and data, there is 95% posterior probability the mean lies in the interval. Bootstrap confidence interval: no — it means 95% of such intervals constructed from repeated samples would contain the true mean; any single interval either does or does not contain it",
                "B) Yes for both — both intervals have 95% probability of containing the true mean",
                "C) Neither — only a full posterior distribution can make probability statements",
                "D) Bootstrap interval: yes; Bayesian interval: no — the Bayesian interval depends on the prior which may be wrong"
            ],
            "answer": "A",
            "explanation": (
                "The frequentist confidence interval is a statement about the procedure, not a single interval. "
                "The Bayesian credible interval is a statement about the parameter given the observed data and prior. "
                "This distinction matters for regulatory communication: a 95% credible interval gives a direct "
                "probability statement, which is what most stakeholders intuitively want."
            ),
            "learning_outcome": "Correctly interpret Bayesian credible intervals vs frequentist confidence intervals",
        },
        # ── Lesson 3: Hypothesis testing, SRM, power analysis ─────────────
        {
            "id": "2.3.1",
            "lesson": "2.3",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Running the SRM check from Exercise 3 on the e-commerce A/B test data "
                "produces this output:\n\n"
                "  chi2_stat=18.4, p_value=0.000018\n"
                "  Expected: control=5000, treatment=5000\n"
                "  Observed: control=4212, treatment=5788\n\n"
                "The head of product asks: 'Our treatment conversion was 2% higher — "
                "should we ship the feature?' What do you advise and why?"
            ),
            "options": [
                "A) No — the SRM (Sample Ratio Mismatch) indicates randomisation was compromised. The p-value of 0.000018 is far below 0.05, rejecting H₀ that the 1:1 split was achieved. Any observed lift may be a selection artifact, not a causal effect of the feature",
                "B) Yes — a 2% lift is significant and the sample is large",
                "C) Borderline — run the experiment for another week to get a balanced split",
                "D) Yes — chi-square tests are conservative; the actual p-value is larger than reported"
            ],
            "answer": "A",
            "explanation": (
                "An SRM chi-square p-value of 0.000018 is overwhelming evidence that randomisation failed — "
                "the observed 4212/5788 split is extremely unlikely under a true 50/50 assignment. "
                "When randomisation is broken, the groups are not comparable and the experiment is invalid. "
                "The feature cannot be shipped based on this data; the root cause must be diagnosed first "
                "(likely a bucketing bug, bot traffic in one arm, or self-selection)."
            ),
            "learning_outcome": "Interpret SRM chi-square results and decide experiment validity",
        },
        {
            "id": "2.3.2",
            "lesson": "2.3",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "In Exercise 3, you test three metrics simultaneously: conversion rate, "
                "average order value (AOV), and revenue per user. "
                "After Bonferroni correction with α=0.05, the adjusted threshold is 0.0167. "
                "Your p-values are: conversion=0.009, AOV=0.042, revenue=0.018. "
                "Which metrics survive Bonferroni correction, and what conclusion do you draw "
                "about the treatment's effect on overall business value?"
            ),
            "options": [
                "A) All three survive; the treatment improves the full funnel",
                "B) None survive; Bonferroni is too conservative for business decisions",
                "C) Only conversion survives (0.009 < 0.0167); AOV (0.042) and revenue (0.018) fail the corrected threshold. The treatment improves how many people buy but not how much each person spends, so total revenue lift is uncertain",
                "D) AOV and revenue survive; conversion is the primary metric and must not be corrected"
            ],
            "answer": "C",
            "explanation": (
                "Bonferroni correction: α_adj = 0.05/3 = 0.0167. "
                "Only conversion (0.009) is below 0.0167. "
                "AOV (0.042) and revenue (0.018) both fail. "
                "This means the treatment increases the number of purchasers but there is no "
                "statistically reliable evidence it increases spend-per-order or total revenue. "
                "A business case for the feature must rely on the conversion lift alone."
            ),
            "learning_outcome": "Apply Bonferroni correction across multiple metrics and interpret business impact",
        },
        # ── Lesson 4: Bootstrap and resampling ────────────────────────────
        {
            "id": "2.4.1",
            "lesson": "2.4",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student implements bootstrap confidence intervals but every run gives "
                "a different interval even with the same data. What is missing?"
            ),
            "code": (
                "def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):\n"
                "    boot_means = []\n"
                "    for _ in range(n_bootstrap):\n"
                "        sample = np.random.choice(data, size=len(data), replace=True)\n"
                "        boot_means.append(sample.mean())\n"
                "    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)\n"
                "    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)\n"
                "    return lo, hi\n"
                "\n"
                "ci = bootstrap_ci(revenue_data)  # no seed"
            ),
            "options": [
                "A) np.random.choice() should be np.random.sample()",
                "B) percentile boundaries are computed incorrectly for 95% CI",
                "C) replace=True should be replace=False for bootstrap",
                "D) np.random.seed() or a Generator with a fixed seed is missing — without it, the random resampling differs each run, making results non-reproducible"
            ],
            "answer": "D",
            "explanation": (
                "Bootstrap relies on random resampling. Without a fixed seed, results change each run, "
                "making intervals irreproducible. In ExperimentTracker-tracked runs, reproducibility is "
                "mandatory — set np.random.seed(42) before the loop or use "
                "rng = np.random.default_rng(42); sample = rng.choice(data, ...)."
            ),
            "learning_outcome": "Ensure bootstrap reproducibility with a fixed random seed",
        },
        {
            "id": "2.4.2",
            "lesson": "2.4",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "In Exercise 4, you compare three bootstrap methods: percentile, basic (pivotal), "
                "and BCa (bias-corrected and accelerated). "
                "Your bootstrap distribution of treatment lift is skewed right (long tail toward positive). "
                "Which method should you report to regulators and why?"
            ),
            "options": [
                "A) Percentile — simplest to explain and most widely understood",
                "B) Basic (pivotal) — it is the most conservative and protects against false positives",
                "C) BCa — it corrects for both bias in the bootstrap estimate and skewness in the sampling distribution, making it the most accurate interval when the statistic's distribution is asymmetric",
                "D) The mean of all three intervals — averaging reduces variance"
            ],
            "answer": "C",
            "explanation": (
                "When the bootstrap distribution is skewed, the percentile interval is biased "
                "because it assumes symmetry around the estimate. "
                "The BCa interval applies two corrections: the bias correction (z₀) adjusts for "
                "systematic shift in the bootstrap estimate, and the acceleration (a) adjusts for "
                "how the standard error changes with the parameter. "
                "For regulatory reporting where accuracy matters more than simplicity, BCa is preferred."
            ),
            "learning_outcome": "Select the appropriate bootstrap CI method based on distribution shape",
        },
        # ── Lesson 5: CUPED and variance reduction ────────────────────────
        {
            "id": "2.5.1",
            "lesson": "2.5",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "After applying CUPED in Exercise 5, the pre-experiment covariate X (pre-period revenue) "
                "has correlation ρ = 0.68 with the post-period metric Y (revenue). "
                "The unadjusted variance is Var(Y) = 12,400. "
                "What is the variance of the CUPED-adjusted metric Y_adj, "
                "and by approximately what factor does the confidence interval width shrink?"
            ),
            "options": [
                "A) Var(Y_adj) = 12,400 × 0.68 = 8,432; CI width shrinks by 32%",
                "B) CUPED does not reduce variance when ρ > 0.5; additional covariates are needed",
                "C) Var(Y_adj) = 12,400 × (1 - 0.68) = 3,968; CI width shrinks by 68%",
                "D) Var(Y_adj) = 12,400 × (1 - 0.68²) = 12,400 × 0.5376 = 6,667; CI width shrinks by factor √(1 - 0.68²) ≈ 0.733, a 26.7% reduction"
            ],
            "answer": "D",
            "explanation": (
                "CUPED variance formula: Var(Y_adj) = Var(Y)(1 - ρ²). "
                "With ρ = 0.68: 1 - 0.68² = 1 - 0.4624 = 0.5376. "
                "Var(Y_adj) = 12,400 × 0.5376 ≈ 6,667. "
                "CI width is proportional to √Var, so the reduction factor is √0.5376 ≈ 0.733. "
                "This means CIs are ~27% narrower — equivalent to running an experiment 1/(0.5376) ≈ 1.86× longer."
            ),
            "learning_outcome": "Apply the CUPED variance reduction formula and compute CI width change",
        },
        {
            "id": "2.5.2",
            "lesson": "2.5",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "In Exercise 5, you use ExperimentTracker to log the CUPED run. "
                "A colleague suggests logging the unadjusted result instead 'because it is simpler'. "
                "What information do you lose by not logging the CUPED-adjusted result, "
                "and which ExperimentTracker method records the variance reduction metric?"
            ),
            "options": [
                "A) You lose the reproducible record of which pre-experiment covariate was used (θ coefficient), the variance reduction achieved, and the adjusted effect size — tracker.log_metric() records these after tracker.start_run()",
                "B) Nothing is lost; ExperimentTracker automatically applies CUPED when it detects a covariate",
                "C) ExperimentTracker.log_cuped() is the dedicated method; no other methods are needed",
                "D) The adjusted p-value cannot be reproduced without the tracker; log it manually to a CSV"
            ],
            "answer": "A",
            "explanation": (
                "ExperimentTracker.start_run() opens a run, then log_metric() records named scalar metrics. "
                "You would log: theta (CUPED coefficient), variance_reduction_pct, adjusted_lift, "
                "adjusted_pvalue, and the covariate_column name. "
                "Without this, you cannot reproduce the adjusted analysis or audit which covariate "
                "drove the variance reduction — critical for regulatory submissions."
            ),
            "learning_outcome": "Use ExperimentTracker to log CUPED runs with full reproducibility metadata",
        },
        # ── Lesson 6: Causal inference, DiD ──────────────────────────────
        {
            "id": "2.6.1",
            "lesson": "2.6",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "In Exercise 6, you run a DiD analysis on HDB prices around an ABSD cooling measure. "
                "The pre-period parallel trends plot shows treatment and control groups "
                "diverging before the policy date. What does this mean for your DiD estimate, "
                "and what robustness check does Exercise 6 specifically recommend?"
            ),
            "options": [
                "A) Pre-period divergence is expected; DiD adjusts for it automatically",
                "B) Parallel trends only need to hold post-treatment; pre-period divergence is irrelevant",
                "C) Use a longer pre-period window to find parallel trends; DiD is still valid",
                "D) Pre-period divergence violates the parallel trends assumption — the DiD estimate is biased. Exercise 6 recommends a placebo test: apply the DiD design to a fake (pre-period) policy date where the true effect is zero; if DiD detects a significant 'effect' there, the identification is broken"
            ],
            "answer": "D",
            "explanation": (
                "DiD's key identifying assumption is parallel counterfactual trends: "
                "absent the treatment, both groups would have moved together. "
                "Pre-period divergence is direct evidence this fails. "
                "A placebo test moves the policy date to a pre-treatment period where "
                "no true effect should exist — if DiD still finds a 'significant' effect, "
                "the design is detecting pre-existing differences, not causal effects."
            ),
            "learning_outcome": "Test and interpret the parallel trends assumption in DiD designs",
        },
        # ── Lesson 7: FeatureEngineer ─────────────────────────────────────
        {
            "id": "2.7.1",
            "lesson": "2.7",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student tries to generate clinical features using FeatureEngineer "
                "but gets an AttributeError. What is wrong?"
            ),
            "code": (
                "from kailash_ml import FeatureEngineer\n"
                "engineer = FeatureEngineer()\n"
                "# The student tries to 'fit' the engineer as if it were an sklearn transformer\n"
                "features = engineer.fit(vitals_df, target='mortality')\n"
                "selected = engineer.transform(vitals_df)"
            ),
            "options": [
                "A) FeatureEngineer uses generate() not fit(), and select() not transform() — calling fit() raises AttributeError because the sklearn API is not implemented on Kailash engines",
                "B) FeatureEngineer requires a target column, but 'mortality' is a reserved name",
                "C) FeatureEngineer must be called with await because it runs asynchronously",
                "D) vitals_df must be converted to a numpy array before passing to FeatureEngineer"
            ],
            "answer": "A",
            "explanation": (
                "FeatureEngineer is a Kailash engine with its own API, not an sklearn transformer. "
                "The correct pattern is: features = await engineer.generate(vitals_df, schema=schema) "
                "followed by selected = await engineer.select(features, target='mortality', method='mutual_info'). "
                "The fit/transform naming comes from sklearn — using it on Kailash engines causes AttributeError."
            ),
            "learning_outcome": "Use the correct FeatureEngineer API (generate/select, not fit/transform)",
        },
        {
            "id": "2.7.2",
            "lesson": "2.7",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "Exercise 7 loads ICU data and the docstring notes: "
                "'Clinical missing patterns (not MCAR — sicker patients get more tests)'. "
                "After profiling with DataExplorer, you find that lactate_level has 62% nulls "
                "and creatinine has 41% nulls. Why does this missing data pattern matter "
                "for choosing between mean imputation and indicator-variable imputation?"
            ),
            "options": [
                "A) When missingness is informative (MNAR — Missing Not At Random), a null in lactate_level may signal a low-acuity patient. Mean imputation destroys this signal. Adding a binary indicator column (lactate_level_is_null=1/0) preserves the missingness information as a feature for the model",
                "B) The missing data is MCAR so any imputation is equivalent",
                "C) 62% nulls means the column should always be dropped",
                "D) Median imputation is always superior to mean imputation regardless of missing mechanism"
            ],
            "answer": "A",
            "explanation": (
                "In ICU data, tests are ordered because patients are suspected to be ill — "
                "a missing lactate measurement often means the patient was not sick enough to warrant the test. "
                "This is MNAR (not MCAR or MAR). Mean imputation replaces all nulls with the same value, "
                "losing the low-acuity signal. Adding a binary indicator column lets the model learn "
                "'this patient did not get a lactate test' as a feature — often predictive of lower mortality."
            ),
            "learning_outcome": "Distinguish MCAR/MAR/MNAR and choose imputation strategy accordingly",
        },
        # ── Lesson 8: FeatureStore ─────────────────────────────────────────
        {
            "id": "2.8.1",
            "lesson": "2.8",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student sets up FeatureStore but their features are not persisted. "
                "What is missing from this async function?"
            ),
            "code": (
                "async def store_features():\n"
                "    conn = ConnectionManager('sqlite:///ascent02_experiments.db')\n"
                "    # Missing: await conn.initialize()\n"
                "    fs = FeatureStore(conn, table_prefix='kml_feat_')\n"
                "    await fs.initialize()\n"
                "    schema = FeatureSchema(name='hdb_features', version='1.0', fields=[...])\n"
                "    await fs.store(features_df, schema=schema)\n"
                "    print('Features stored')\n"
                "\n"
                "asyncio.run(store_features())"
            ),
            "options": [
                "A) table_prefix must not have underscores",
                "B) asyncio.run() cannot be used with FeatureStore; use a thread pool instead",
                "C) FeatureStore must be imported from kailash_ml.feature_store, not kailash_ml",
                "D) await conn.initialize() is missing after creating the ConnectionManager — without initialising the connection, fs.initialize() and fs.store() will fail because no database tables have been created"
            ],
            "answer": "D",
            "explanation": (
                "ConnectionManager.initialize() creates the database schema and connection pool. "
                "Without calling it, the connection is in an uninitialised state. "
                "FeatureStore.initialize() tries to create its tables using the connection, "
                "but since the pool is not ready, it raises a runtime error. "
                "The pattern is always: create → await initialize → use."
            ),
            "learning_outcome": "Follow the ConnectionManager initialisation lifecycle correctly",
        },
        {
            "id": "2.8.2",
            "lesson": "2.8",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 8 demonstrates point-in-time feature retrieval from FeatureStore. "
                "A data scientist asks: 'Why not just store features as a Parquet file and join "
                "at training time?' What two specific problems does FeatureStore solve that "
                "flat file storage cannot?"
            ),
            "options": [
                "A) FeatureStore is faster to read; Parquet files are slow",
                "B) FeatureStore compresses data better than Parquet",
                "C) (1) Point-in-time correctness: FeatureStore retrieves the feature value that was available at a given timestamp, preventing label leakage from using future data; (2) Data lineage: FeatureStore tracks which features trained which model version, enabling regulatory audit of model provenance",
                "D) Parquet files cannot store schema information; FeatureStore adds column types"
            ],
            "answer": "C",
            "explanation": (
                "Flat files are snapshots — they do not track when each value was computed. "
                "When you join a flat file at training time you can accidentally use features "
                "computed from data that was not available at the prediction time (label leakage). "
                "FeatureStore stores a timestamp with each value and lets you query 'what was the "
                "value of feature X for entity Y at time T'. "
                "The lineage capability (which schema version trained which model) is critical for "
                "regulatory audit, reproducibility, and debugging production failures."
            ),
            "learning_outcome": "Articulate point-in-time correctness and lineage as FeatureStore value propositions",
        },
        {
            "id": "2.8.3",
            "lesson": "2.8",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "After storing version 1.0 of the HDB features in FeatureStore, "
                "you add a new column rolling_12m_price_sqm and store it as version 2.0. "
                "A production model trained on version 1.0 is still running. "
                "What FeatureStore query retrieves only version 1.0 features for the production model, "
                "and why is versioning critical here?"
            ),
            "options": [
                "A) fs.retrieve(schema_name='hdb_features') always returns the latest version automatically",
                "B) Feature versions are for documentation only; retrieval always uses the most recent data regardless of version",
                "C) fs.retrieve(schema_name='hdb_features', version='1.0', as_of=prediction_timestamp) — without version pinning, a schema update could silently add or remove columns, breaking the production model that expects the exact feature set it was trained on",
                "D) fs.get_schema(version='1.0') returns the schema; you must join manually to get values"
            ],
            "answer": "C",
            "explanation": (
                "Production models have a fixed input contract — the exact feature columns and their semantics. "
                "If retrieval automatically used version 2.0, the model would receive an unexpected extra column "
                "or a column with different computation logic (rolling_12m_price_sqm was not seen during training). "
                "Version pinning ensures the production pipeline always requests the same schema the model was trained on. "
                "as_of prevents label leakage by using the feature value at the time of the prediction, not today's value."
            ),
            "learning_outcome": "Use FeatureStore version pinning to protect production model input contracts",
        },
        # ── Additional questions covering lessons 1–8 breadth ─────────────
        {
            "id": "2.2.1",
            "lesson": "2.2",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "When computing MLE for a Bernoulli parameter (e.g., HDB transaction success rate), "
                "you observe 42 successes in 180 trials. The MLE is p̂ = 0.233. "
                "A student argues 'MLE always gives the most likely value so we should use it directly'. "
                "When is this argument wrong, and what does the Fisher Information "
                "tell you about the reliability of this estimate?"
            ),
            "options": [
                "A) MLE is always reliable regardless of sample size; Fisher Information is irrelevant",
                "B) MLE is consistent and efficient asymptotically, but with small samples it can be unreliable. Fisher Information I(p) = n / (p(1-p)) quantifies the precision of the MLE — higher I means narrower Cramer-Rao lower bound. For n=180 and p=0.233, I ≈ 1007, giving SE = 1/√1007 ≈ 0.032. The 95% CI is [0.170, 0.296] — fairly wide, not a single reliable point estimate",
                "C) MLE is biased for Bernoulli parameters; use Bayes estimate instead",
                "D) Fisher Information only applies to continuous distributions",
            ],
            "answer": "B",
            "explanation": (
                "MLE is asymptotically unbiased and efficient, but 'most likely' does not mean precise. "
                "Fisher Information measures how sharply the likelihood peaks around the MLE. "
                "I(p) = n/(p(1-p)) for Bernoulli. The Cramer-Rao bound gives the minimum variance "
                "achievable: Var(p̂) >= 1/I(p). "
                "For n=180, the SE is ~3.2 percentage points — meaningful uncertainty for business decisions."
            ),
            "learning_outcome": "Use Fisher Information to quantify MLE reliability and compute Cramer-Rao bound",
        },
        {
            "id": "2.3.3",
            "lesson": "2.3",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student runs a permutation test in Exercise 3 but gets the same p-value "
                "every run. What is the bug?"
            ),
            "code": (
                "def permutation_test(control, treatment, n_permutations=10000):\n"
                "    observed_diff = treatment.mean() - control.mean()\n"
                "    combined = np.concatenate([control, treatment])\n"
                "    n_control = len(control)\n"
                "    perm_diffs = []\n"
                "    for _ in range(n_permutations):\n"
                "        # Bug: using sorted order, not random shuffle\n"
                "        perm = combined  # no shuffle\n"
                "        perm_diffs.append(perm[:n_control].mean() - perm[n_control:].mean())\n"
                "    return np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))"
            ),
            "options": [
                "A) combined must be sorted before permutation",
                "B) The loop never shuffles combined — it reuses the original array each time, so all 10,000 'permutations' are identical. Fix: np.random.shuffle(perm) after perm = combined.copy()",
                "C) np.concatenate should be np.stack for 1D arrays",
                "D) The p-value calculation should use np.mean(perm_diffs >= observed_diff) without abs()",
            ],
            "answer": "B",
            "explanation": (
                "The permutation test relies on randomly reassigning observations to groups. "
                "Without shuffle, every iteration uses the original ordering — "
                "perm[:n_control] is always the control group and the permutation null distribution "
                "collapses to a single point. "
                "Fix: perm = combined.copy(); np.random.shuffle(perm); then split and compute diff. "
                "Also add a seed for reproducibility: np.random.seed(42) or use a Generator."
            ),
            "learning_outcome": "Implement a correct permutation test with random shuffling on each iteration",
        },
        {
            "id": "2.6.2",
            "lesson": "2.6",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 6 uses ExperimentTracker to log the DiD results alongside the "
                "CUPED results from Exercise 5. A student asks why both analyses are logged "
                "to the same experiment rather than two separate experiments. "
                "What does logging both to one ExperimentTracker experiment enable?"
            ),
            "options": [
                "A) One experiment is simpler to manage; there is no analytical benefit",
                "B) ExperimentTracker only allows one run per experiment; logging both is a bug",
                "C) Logging both to one experiment enables comparative analysis: you can query which variance reduction method (CUPED vs DiD) gave the tighter CI for the same underlying metric, compare convergence dates, and trace which analysis informed which business decision — all within a single reproducible lineage record",
                "D) Separate experiments are required for regulatory submissions"
            ],
            "answer": "C",
            "explanation": (
                "ExperimentTracker organises work as experiments → runs → metrics. "
                "Multiple runs within one experiment are comparable: "
                "you can query 'show me all runs in the HDB-cooling-2024 experiment' and see "
                "CUPED run (variance reduction 27%) alongside DiD run (ATT estimate ± CI). "
                "This consolidated view is what an analyst or regulator needs to follow "
                "the reasoning from raw data to final policy recommendation."
            ),
            "learning_outcome": "Use a single ExperimentTracker experiment to consolidate multiple analytical methods for comparison",
        },
        {
            "id": "2.7.3",
            "lesson": "2.7",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "Exercise 7 loads five ICU tables and notes 'irregular time-series — "
                "vitals recorded at different frequencies'. "
                "After FeatureEngineer.generate() produces rolling vitals features, "
                "you discover that the 1-hour rolling heart rate is null for 31% of rows. "
                "Explain the root cause, what point-in-time correctness issue this introduces, "
                "and how FeatureSchema validation catches it."
            ),
            "options": [
                "A) Null rolling features are a data loading bug; reload the vitals table",
                "B) Vitals are recorded only when a nurse measures them — irregular intervals mean some 1-hour windows contain zero measurements, producing null. The point-in-time issue: if you forward-fill nulls using a future measurement, the feature contains data the model would not have had at prediction time. FeatureSchema validation raises an error if the null_fraction exceeds a configured threshold, forcing you to decide between forward-fill (leakage risk) or keep null (informative absence)",
                "C) Rolling features should use mean imputation automatically; no action needed",
                "D) 31% nulls means the feature should be dropped; rolling features require >80% coverage",
            ],
            "answer": "B",
            "explanation": (
                "ICU vitals are episodic, not continuous. A 1-hour window where no nurse measured "
                "heart rate produces a null — this is informative (the patient may have been stable). "
                "Forward-filling uses a future observation's value for the current prediction time, "
                "which is leakage (the model 'knows' what the heart rate was hours later). "
                "FeatureSchema with max_null_fraction=0.2 would flag the 31% null column, "
                "forcing an explicit engineering decision rather than silent propagation."
            ),
            "learning_outcome": "Diagnose irregular time-series nulls, identify forward-fill leakage, and apply FeatureSchema null validation",
        },
        {
            "id": "2.4.3",
            "lesson": "2.4",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 4, you log three bootstrap runs with different n_bootstrap values "
                "(100, 1000, 10000) to ExperimentTracker. The CI widths are "
                "[47.2, 41.8, 41.3] respectively. "
                "What does this convergence pattern tell you about the choice of n_bootstrap, "
                "and at what value would you stop increasing it?"
            ),
            "options": [
                "A) The CI width stabilises after n_bootstrap=1000 (41.8 vs 41.3 is negligible). The Monte Carlo error of the bootstrap interval itself shrinks as O(1/√B) — doubling B from 1000 to 10000 only reduces MC error by ~68%. At 1000 resamples the CI is already converged; 10000 is computationally wasteful for routine use",
                "B) Always use n_bootstrap=10000; more resamples are always better",
                "C) n_bootstrap should equal the sample size; use n_bootstrap=len(data)",
                "D) The convergence means bootstrap is not applicable to this data distribution"
            ],
            "answer": "A",
            "explanation": (
                "Bootstrap CI precision improves as O(1/√B) where B is n_bootstrap. "
                "Going from 100 to 1000 (10×) reduces MC error by ~68% — worth it. "
                "Going from 1000 to 10000 (10×) reduces by another ~68% but the absolute change "
                "(41.8 → 41.3) is negligible for practical decisions. "
                "In ExperimentTracker, logging all three runs lets you visualise this convergence "
                "and document the chosen B with justification."
            ),
            "learning_outcome": "Use ExperimentTracker CI convergence plots to justify a sufficient n_bootstrap value",
        },
        {
            "id": "2.1.3",
            "lesson": "2.1",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student computes bootstrap confidence intervals in Exercise 1 "
                "but does not load environment variables first. "
                "The script crashes with KeyError when ModelVisualizer tries to connect to a "
                "rendering service. What is the correct fix?"
            ),
            "code": (
                "import numpy as np\n"
                "import polars as pl\n"
                "from kailash_ml import ModelVisualizer\n"
                "# Bug: missing environment setup\n"
                "from shared import ASCENTDataLoader\n"
                "\n"
                "loader = ASCENTDataLoader()\n"
                "hdb = loader.load('ascent01', 'hdb_resale.parquet')\n"
                "viz = ModelVisualizer()  # crashes: KeyError on API key"
            ),
            "options": [
                "A) ModelVisualizer does not require environment variables",
                "B) from dotenv import load_dotenv; load_dotenv() must be called before any kailash_ml or shared imports that access environment variables — add it as the very first non-stdlib import",
                "C) Set environment variables manually with os.environ['KEY'] = 'value' inline",
                "D) Use from shared.kailash_helpers import setup_environment; setup_environment() which handles load_dotenv() and validates required variables",
            ],
            "answer": "D",
            "explanation": (
                "The course uses setup_environment() from shared.kailash_helpers as the standard pattern — "
                "it calls load_dotenv() and performs validation checks. "
                "Both B and D are valid (D is the course convention; B is the underlying mechanism). "
                "D is the preferred answer because it matches the pattern used in Exercises 4–8 "
                "which all start with: from shared.kailash_helpers import setup_environment; setup_environment()"
            ),
            "learning_outcome": "Load environment variables with setup_environment() before any kailash_ml operations",
        },
        {
            "id": "2.5.3",
            "lesson": "2.5",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Running sequential testing in Exercise 5 produces an always-valid p-value "
                "that stays above 0.05 for the first 3 weeks then drops below 0.05 in week 4. "
                "Your manager says: 'Great — we should have peeked earlier and shipped in week 2 "
                "when the p-value was 0.03 using traditional testing.' "
                "Why is this reasoning wrong, and what does 'always-valid' mean for early stopping?"
            ),
            "options": [
                "A) The manager is right; the first sub-0.05 p-value is always the correct stopping point",
                "B) A traditional p-value viewed at week 2 inflates Type I error because it does not account for multiple looks — peeking at week 2 without adjustment gives a false positive rate much higher than 5%. 'Always-valid' means the sequential p-value maintains Type I error control at every peek point simultaneously; stopping at week 4 when the always-valid p-value crosses 0.05 is statistically sound, while week 2 traditional peeking is not",
                "C) Sequential testing p-values are always larger than traditional; week 2 was correct",
                "D) The always-valid p-value is just a lagging indicator; use the traditional p-value for decision timing",
            ],
            "answer": "B",
            "explanation": (
                "Traditional hypothesis testing assumes a single look at the end. "
                "Each additional look at the accumulating data inflates the false positive rate — "
                "with 10 weekly peeks at α=0.05, the actual FWER is ~40%. "
                "Always-valid (anytime-valid) p-values are constructed using e-values or "
                "mixture sequential probability ratio tests (mSPRT), maintaining Type I error "
                "control at every possible stopping time. "
                "The manager's reasoning applies classical tests to a sequential context, which is invalid."
            ),
            "learning_outcome": "Explain why always-valid p-values are required for sequential testing and early stopping",
        },
        {
            "id": "2.8.4",
            "lesson": "2.8",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student stores features in FeatureStore but gets duplicate rows when retrieving. "
                "What is wrong with this store call?"
            ),
            "code": (
                "# Student runs the same script twice without versioning\n"
                "await fs.store(\n"
                "    features_df,\n"
                "    schema=FeatureSchema(name='hdb_features', version='1.0', fields=[...]),\n"
                ")  # Run once — stores 15,000 rows\n"
                "\n"
                "# Later: re-runs same script without incrementing version\n"
                "await fs.store(\n"
                "    features_df,\n"
                "    schema=FeatureSchema(name='hdb_features', version='1.0', fields=[...]),\n"
                ")  # Stores another 15,000 rows — now 30,000 rows total"
            ),
            "options": [
                "A) FeatureStore.store() should be FeatureStore.upsert() for idempotent writes",
                "B) Storing the same version twice appends duplicate rows because version='1.0' does not provide uniqueness by entity/timestamp key. Either increment to version='1.1' for updated features, or use upsert=True if the engine supports idempotent writes, or clear the existing version with fs.delete_version() before re-storing",
                "C) features_df must have a unique index column to prevent duplicates",
                "D) Duplicate rows are expected; filter by max timestamp when retrieving",
            ],
            "answer": "B",
            "explanation": (
                "FeatureStore.store() with an existing version/schema does not automatically deduplicate — "
                "it appends the new rows. The version string is metadata, not a uniqueness constraint. "
                "The correct approach for re-computation with the same version: "
                "delete the old data first (fs.delete_version('hdb_features', '1.0')) then store, "
                "or increment the version to signal intentional update."
            ),
            "learning_outcome": "Prevent FeatureStore duplicate rows by managing version lifecycle correctly",
        },
        {
            "id": "2.3.4",
            "lesson": "2.3",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 3 applies Benjamini-Hochberg (BH-FDR) correction as an alternative "
                "to Bonferroni. For the same three metrics (p-values: 0.009, 0.042, 0.018), "
                "BH-FDR with α=0.05 and m=3 gives thresholds k=1: 0.017, k=2: 0.033, k=3: 0.050. "
                "Rank the p-values and determine which survive BH-FDR. "
                "Does this differ from Bonferroni, and when should you prefer each?"
            ),
            "options": [
                "A) BH-FDR and Bonferroni always give identical results; they differ only in computation",
                "B) BH-FDR requires independent p-values; since conversion, AOV, and revenue are correlated, it cannot be applied",
                "C) BH-FDR is always less conservative; always prefer it over Bonferroni",
                "D) Ranked p-values: p(1)=0.009 ≤ 0.017 ✓, p(2)=0.018 ≤ 0.033 ✓, p(3)=0.042 ≤ 0.050 ✓. All three survive BH-FDR vs only one under Bonferroni. Use Bonferroni when controlling FWER (zero false positives critical, e.g., clinical trials); use BH-FDR when controlling FDR (acceptable to have some false positives among many discoveries, e.g., exploratory metric analysis)"
            ],
            "answer": "D",
            "explanation": (
                "BH-FDR step: rank p-values ascending, threshold k = (k/m)α. "
                "p(1)=0.009: threshold = (1/3)×0.05 = 0.017. 0.009 ≤ 0.017 ✓. "
                "p(2)=0.018: threshold = (2/3)×0.05 = 0.033. 0.018 ≤ 0.033 ✓. "
                "p(3)=0.042: threshold = (3/3)×0.05 = 0.050. 0.042 ≤ 0.050 ✓. "
                "All three survive. BH-FDR controls the expected fraction of false discoveries (FDR), "
                "not the probability of any false discovery (FWER). "
                "For exploratory business metrics, FDR control is often the right trade-off."
            ),
            "learning_outcome": "Apply BH-FDR step-up procedure and choose between FWER and FDR control based on context",
        },
        {
            "id": "2.1.4",
            "lesson": "2.1",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 1, you set a prior for HDB mean price with μ₀=450,000 and σ₀=50,000. "
                "A classmate uses σ₀=500,000 (diffuse prior). "
                "With n=2,000 observations, how does the posterior mean differ, "
                "and what does the choice of σ₀ communicate to a stakeholder?"
            ),
            "options": [
                "A) Diffuse prior always gives posterior mean exactly equal to MLE",
                "B) With n=2,000 and σ₀=500,000 (very diffuse), the prior precision 1/σ₀² is tiny relative to n/σ². Both posteriors converge very close to the MLE — the diffuse prior communicates 'I have almost no prior knowledge; the data alone drives the estimate'. With σ₀=50,000, the prior says 'I believe the mean is near 450k with moderate confidence' and pulls the posterior slightly toward 450k",
                "C) A diffuse prior always produces a wider posterior CI than an informative prior",
                "D) The σ₀ choice does not affect the posterior mean; only σ₀ affects CI width",
            ],
            "answer": "B",
            "explanation": (
                "Normal-Normal posterior precision: τₙ = τ₀ + nτ = 1/σ₀² + n/σ². "
                "With σ₀=500,000: τ₀ = 4×10⁻¹² — negligible against n/σ². "
                "With σ₀=50,000: τ₀ = 4×10⁻¹⁰ — still small but contributes a slight pull. "
                "Both posteriors are close to the MLE with n=2,000. "
                "The practical difference is communication: informative priors document domain expertise; "
                "diffuse priors signal the analysis is data-driven."
            ),
            "learning_outcome": "Explain prior width's effect on posterior in Normal-Normal conjugate with large n",
        },
    ],
}

if __name__ == "__main__":
    for q in QUIZ["questions"]:
        print(f"\n{'=' * 60}")
        print(f"[{q['id']}] ({q['type']}) — Lesson {q['lesson']}  [{q['difficulty']}]")
        print(f"{'=' * 60}")
        print(q["question"])
        if q.get("code"):
            print(f"\n```python\n{q['code']}\n```")
        for opt in q["options"]:
            print(f"  {opt}")
        print(f"\nAnswer: {q['answer']}")
