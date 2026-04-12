# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP 2 — AI-Resilient Assessment Questions

Feature Engineering & Experiment Design
Covers: Bayesian inference, MLE, hypothesis testing, CUPED, bootstrap,
        causal inference, FeatureEngineer, FeatureStore, ExperimentTracker
"""

QUIZ = {
    "module": "MLFP02",
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
                "    conn = ConnectionManager('sqlite:///mlfp02_experiments.db')\n"
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
                "from shared import MLFPDataLoader\n"
                "\n"
                "loader = MLFPDataLoader()\n"
                "hdb = loader.load('mlfp01', 'hdb_resale.parquet')\n"
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

# --- Merged from mlfp02 (Feature Engineering & Experiment Design) ---


Supervised ML — Theory to Production
Covers: bias-variance, regularisation, boosting, class imbalance, SHAP,
        WorkflowBuilder, DataFlow, ModelRegistry, TrainingPipeline
"""

QUIZ = {
    "module": "MLFP02",
    "title": "Supervised ML — Theory to Production",
    "questions": [
        # ── Lesson 1: Bias-variance, regularisation ───────────────────────
        {
            "id": "3.1.1",
            "lesson": "3.1",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 1, you train polynomial models of degree 1 through 12 on "
                "the Singapore credit dataset. The training MSE decreases monotonically "
                "but the 5-fold cross-validation MSE is lowest at degree 4 and rises sharply "
                "from degree 7 onward. What does the rising CV MSE signal, and what "
                "regularisation approach does the exercise recommend to extend the usable "
                "polynomial degree?"
            ),
            "options": [
                "A) Rising CV MSE signals underfitting; you need a more complex model",
                "B) Rising CV MSE means the polynomial features are correlated; use PCA first",
                "C) The CV MSE rise is a numerical artefact of cross-validation; use the test set instead",
                "D) Rising CV MSE from degree 7 signals overfitting — the model has learned training-set noise rather than the true function. The exercise recommends applying Ridge (L2) regularisation, which shrinks polynomial coefficients and lets higher-degree models generalise",
            ],
            "answer": "D",
            "explanation": (
                "When CV MSE rises after a minimum, the model is memorising training examples. "
                "Ridge regularisation adds λ||w||² to the loss, penalising large coefficients. "
                "This effectively constrains the model even at high polynomial degrees, "
                "allowing you to use degree 8 or 10 with proper regularisation "
                "without the CV MSE blowing up."
            ),
            "learning_outcome": "Diagnose overfitting from bias-variance plots and select correct regularisation",
        },
        {
            "id": "3.1.2",
            "lesson": "3.1",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 1 fits Ridge, Lasso, and ElasticNet on the credit dataset and logs "
                "the non-zero coefficient counts. Lasso produces 8 non-zero coefficients from "
                "40 features while Ridge keeps all 40 active. A risk officer needs to submit a "
                "model explanation to MAS (Monetary Authority of Singapore). "
                "Which model family should you prefer, and what property makes it preferable for regulatory submission?"
            ),
            "options": [
                "A) Ridge — keeping all features shows the model considered all available information",
                "B) ElasticNet — the combination of L1 and L2 is always best for regulatory models",
                "C) Lasso — its L1 penalty drives many coefficients to exactly zero, producing a sparse model with only 8 input variables. This sparsity makes the model interpretable: the regulator sees exactly which features drive creditworthiness decisions, satisfying MAS explainability requirements",
                "D) Neither — neural networks are required for credit scoring in Singapore",
            ],
            "answer": "C",
            "explanation": (
                "Lasso's L1 geometry produces corners in the constraint region that coincide with sparse solutions. "
                "A model with 8 features is auditable — a regulator can read and understand the coefficients. "
                "Ridge retains all 40 features with small but non-zero weights, making it harder to explain "
                "why feature X influenced a loan rejection when its coefficient is 0.003. "
                "Sparsity is a practical regulatory advantage in Singapore's financial services context."
            ),
            "learning_outcome": "Select Lasso for sparse interpretable models required in regulatory contexts",
        },
        # ── Lesson 2: Boosting and SHAP ───────────────────────────────────
        {
            "id": "3.2.1",
            "lesson": "3.2",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "You train a LightGBM model on the credit scoring data and compute SHAP values. "
                "For customer A (declined for credit), the SHAP summary shows:\n\n"
                "  debt_to_income_ratio: +0.42\n"
                "  employment_tenure_months: -0.18\n"
                "  credit_utilisation: +0.31\n"
                "  annual_income: -0.09\n\n"
                "(Positive SHAP = pushes toward default prediction)\n\n"
                "Write the one-sentence adverse action notice MAS guidelines require, "
                "citing the top two factors that contributed to the decline."
            ),
            "options": [
                "A) 'Your application was declined primarily because of your high debt-to-income ratio and high credit utilisation, which together indicate elevated default risk based on your credit profile.'",
                "B) 'Your application was declined due to our credit scoring model scoring below threshold.'",
                "C) 'Your application was declined; the SHAP values for your features were 0.42 and 0.31.'",
                "D) 'Your application was declined because of your employment history and income level.'",
            ],
            "answer": "A",
            "explanation": (
                "The two largest positive SHAP contributors (pushing toward default) are "
                "debt_to_income_ratio (+0.42) and credit_utilisation (+0.31). "
                "MAS guidelines require plain-language adverse action notices that name the actual factors, "
                "not model scores or SHAP values. Option B translates SHAP outputs into "
                "customer-facing language while correctly identifying the top two drivers."
            ),
            "learning_outcome": "Translate SHAP values into regulatory-compliant adverse action notices",
        },
        # ── Lesson 3: Class imbalance and calibration ─────────────────────
        {
            "id": "3.3.1",
            "lesson": "3.3",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 3, you compare models trained with no imbalance handling, "
                "SMOTE oversampling, and cost-sensitive learning. "
                "The baseline achieves ROC-AUC=0.84, SMOTE achieves 0.83, "
                "but Average Precision drops from 0.41 (baseline) to 0.38 (SMOTE). "
                "Why does SMOTE hurt Average Precision despite similar ROC-AUC, "
                "and what does this tell you about using ROC-AUC as the sole metric for imbalanced data?"
            ),
            "options": [
                "A) SMOTE hurts because it adds noise; use more bootstrap samples to fix it",
                "B) SMOTE always improves precision; the result suggests a bug in the implementation",
                "C) ROC-AUC is insensitive to class ratio because it measures rank order across all thresholds. SMOTE generates synthetic minority samples that blur the decision boundary — Average Precision penalises this more harshly because it focuses on precision at high-recall operating points where the imbalance is most severe",
                "D) Average Precision is not meaningful for credit scoring; use F1 instead",
            ],
            "answer": "C",
            "explanation": (
                "ROC-AUC integrates over all thresholds and treats false positives and false negatives symmetrically. "
                "On highly imbalanced data, a model can achieve high ROC-AUC by being confident about the majority class. "
                "Average Precision (PR-AUC) focuses on how well the model ranks positive examples at the top — "
                "more sensitive to precision degradation. SMOTE's synthetic points soften the boundary "
                "between real and synthetic minority samples, hurting precision at the high-recall region."
            ),
            "learning_outcome": "Explain why ROC-AUC and Average Precision diverge under class imbalance",
        },
        {
            "id": "3.3.2",
            "lesson": "3.3",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 3 covers post-hoc calibration with Platt scaling and isotonic regression. "
                "Your model's predicted probabilities cluster around 0.3 and 0.8 "
                "(few predictions near 0.5). A reliability diagram shows the model is "
                "overconfident at both extremes. Which calibrator is appropriate and why, "
                "given that you have 50,000 training examples?"
            ),
            "options": [
                "A) Platt scaling — it is always more accurate than isotonic regression",
                "B) Platt scaling — isotonic regression overfits on datasets larger than 10,000 samples",
                "C) Neither — overconfidence requires retraining from scratch with temperature scaling",
                "D) Isotonic regression — with 50,000 samples there is sufficient data to fit a flexible, non-parametric monotonic calibration curve; it can handle the bimodal overconfidence without assuming a parametric form. Platt scaling assumes a sigmoid shape which may not match this distribution",
            ],
            "answer": "D",
            "explanation": (
                "Platt scaling fits a logistic function to map raw scores to probabilities — it works well "
                "when the uncalibrated scores follow a roughly sigmoidal shape. "
                "When the model is overconfident at both extremes (bimodal output), "
                "the sigmoid assumption is violated. "
                "Isotonic regression fits a piecewise constant monotonic function, "
                "which can capture non-sigmoid calibration curves. "
                "With 50k samples, isotonic regression has enough data and avoids overfitting."
            ),
            "learning_outcome": "Select calibration method based on prediction distribution shape and sample size",
        },
        # ── Lesson 5: WorkflowBuilder and Core SDK patterns ──────────────
        {
            "id": "3.5.1",
            "lesson": "3.5",
            "type": "code_debug",
            "difficulty": "foundation",
            "question": (
                "A student's ML workflow runs without error but produces empty results. "
                "What is the critical bug?"
            ),
            "code": (
                "from kailash.workflow.builder import WorkflowBuilder\n"
                "from kailash.runtime import LocalRuntime\n"
                "\n"
                "workflow = WorkflowBuilder('credit_scoring_pipeline')\n"
                "workflow.add_node('DataPreprocessNode', 'preprocess', {...})\n"
                "workflow.add_node('ModelTrainNode', 'train', {...})\n"
                "workflow.add_connection('preprocess', 'train', 'output', 'input')\n"
                "\n"
                "runtime = LocalRuntime()\n"
                "results, run_id = runtime.execute(workflow)  # Bug here"
            ),
            "options": [
                "A) runtime.execute() receives the WorkflowBuilder object directly — it must be called with workflow.build() to compile the graph first: runtime.execute(workflow.build())",
                "B) WorkflowBuilder should be imported from kailash.workflow",
                "C) add_connection() argument order is wrong; it should be ('train', 'preprocess', ...)",
                "D) LocalRuntime should be AsyncLocalRuntime for ML workflows",
            ],
            "answer": "A",
            "explanation": (
                "This is the single most common Kailash SDK mistake. "
                "workflow.build() validates the node graph, checks for disconnected nodes, "
                "and compiles a runtime-executable plan. "
                "runtime.execute(workflow) passes the builder object (not the plan), "
                "causing a cryptic AttributeError deep in the runtime. "
                "Always: runtime.execute(workflow.build())"
            ),
            "learning_outcome": "Apply the mandatory workflow.build() pattern before runtime.execute()",
        },
        {
            "id": "3.5.2",
            "lesson": "3.5",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student builds a DataFlow CreateNode for storing evaluation results "
                "but gets a validation error on every run. What is wrong with the node config?"
            ),
            "code": (
                "workflow.add_node(\n"
                "    'CreateEvalResult',\n"
                "    'store_result',\n"
                "    {\n"
                "        'data': {        # Bug: nested\n"
                "            'model_name': 'lightgbm_v1',\n"
                "            'roc_auc': 0.84,\n"
                "            'run_id': 'abc123',\n"
                "        }\n"
                "    }\n"
                ")"
            ),
            "options": [
                "A) 'CreateEvalResult' is not a valid node type; use 'CreateNode'",
                "B) workflow.add_node() requires a fourth positional argument for connections",
                "C) roc_auc cannot be a float in a DataFlow node; it must be cast to str first",
                "D) DataFlow CreateNode parameters must be flat (not nested under a 'data' key) — the correct form passes field values directly at the top level of the config dict",
            ],
            "answer": "D",
            "explanation": (
                "DataFlow CreateNode expects a flat parameter dict matching the model's field names. "
                "Nesting under 'data' was a pattern from an older API version — it is no longer valid "
                "and causes a validation error. "
                "Correct: {'model_name': 'lightgbm_v1', 'roc_auc': 0.84, 'run_id': 'abc123'}"
            ),
            "learning_outcome": "Use flat parameter dicts for DataFlow CreateNode (not nested under 'data')",
        },
        # ── Lesson 6: DataFlow, async patterns ───────────────────────────
        {
            "id": "3.6.1",
            "lesson": "3.6",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's DataFlow model for ML evaluation results has a bug that will "
                "cause crashes in production. Identify the structural error."
            ),
            "code": (
                "from kailash_dataflow import DataFlow, field\n"
                "\n"
                "db = DataFlow('sqlite:///mlfp02_results.db')\n"
                "\n"
                "@db.model\n"
                "class EvalResult:\n"
                "    primary_key: int = field(primary_key=True)  # Bug: wrong field name\n"
                "    model_name: str\n"
                "    roc_auc: float\n"
                "    created_at: str  # Bug: wrong type for timestamp"
            ),
            "options": [
                "A) DataFlow models must inherit from a base class",
                "B) @db.model decorator must include the table name as a string argument",
                "C) The primary key field must be named 'id', not 'primary_key'; and timestamps should use datetime, not str — DataFlow auto-manages created_at as datetime if the field is omitted entirely",
                "D) float is not a supported DataFlow field type; use Decimal",
            ],
            "answer": "C",
            "explanation": (
                "DataFlow requires the primary key field to be named 'id' — this is a hard constraint. "
                "Using a different name causes DataFlow to not recognise the primary key, "
                "breaking CRUD operations. "
                "For timestamps, the correct pattern is to omit created_at entirely — "
                "DataFlow auto-manages creation and update timestamps as datetime objects. "
                "Storing them as str loses timezone information and prevents date-range queries."
            ),
            "learning_outcome": "Define DataFlow @db.model with correct 'id' primary key naming",
        },
        {
            "id": "3.6.2",
            "lesson": "3.6",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 6 compares db.express (simple CRUD) against WorkflowBuilder for storing "
                "ML evaluation results. For a training pipeline that stores one evaluation result "
                "per model run, which approach is correct and why does using WorkflowBuilder "
                "here violate the framework-first hierarchy?"
            ),
            "options": [
                "A) db.express — single-record CRUD is exactly the use case db.express is designed for. WorkflowBuilder adds ~23x overhead from graph construction and validation for a single write operation, violating the principle of using the highest appropriate abstraction layer",
                "B) WorkflowBuilder — it provides better error handling for single-record operations",
                "C) Both are equivalent; use whichever is more familiar",
                "D) WorkflowBuilder — db.express cannot be used inside async functions",
            ],
            "answer": "A",
            "explanation": (
                "The Kailash framework hierarchy: WorkflowBuilder is for multi-step pipelines "
                "where the graph structure, node connections, and execution plan add value. "
                "For a single db.express.create('EvalResult', {...}), "
                "WorkflowBuilder's overhead is pure waste — graph construction, "
                "topological sort, and validation for a one-node graph. "
                "db.express is the correct primitive: await db.express.create('EvalResult', result_dict)"
            ),
            "learning_outcome": "Distinguish when db.express is correct versus WorkflowBuilder based on operation complexity",
        },
        # ── Lesson 7: TrainingPipeline, ModelRegistry ─────────────────────
        {
            "id": "3.7.1",
            "lesson": "3.7",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "You use TrainingPipeline with EvalSpec to evaluate a credit model. "
                "The spec includes roc_auc and average_precision as metrics. "
                "After running, results['eval']['metrics'] shows:\n\n"
                "  roc_auc: 0.847\n"
                "  average_precision: 0.392\n"
                "  brier_score: 0.089\n\n"
                "The ModelRegistry has a production model with roc_auc=0.831. "
                "Should you promote the new model to production, and what additional check "
                "beyond raw metrics should you perform before promotion?"
            ),
            "options": [
                "A) Yes — any improvement in ROC-AUC justifies promotion",
                "B) No — production models should never be replaced without a minimum 5% improvement",
                "C) The metric improvement (0.847 vs 0.831) looks promising, but before promotion you must run a calibration check (reliability diagram) to ensure predicted probabilities are accurate, and compare the confusion matrix at the operational threshold — a 0.016 AUC improvement could hide a precision/recall trade-off shift that matters for loan approval decisions",
                "D) Yes — brier_score < 0.1 confirms the model is well-calibrated, no further checks needed",
            ],
            "answer": "C",
            "explanation": (
                "Raw AUC comparison is a starting point, not the full gate. "
                "For a credit scoring model, you also need: "
                "(1) calibration check — does P(default | score=0.3) actually equal 30%? "
                "(2) business threshold analysis — at the operational cut-off, did false positive rate change? "
                "(3) fairness checks — did the improvement come at the cost of a demographic group? "
                "ModelRegistry.promote() should only be called after these gates pass."
            ),
            "learning_outcome": "Identify the full model promotion checklist beyond raw metrics",
        },
        {
            "id": "3.7.2",
            "lesson": "3.7",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student registers a model in ModelRegistry but cannot retrieve it by name "
                "in a subsequent script. What is wrong?"
            ),
            "code": (
                "# Script 1 — training\n"
                "registry = ModelRegistry()  # in-memory, no persistence\n"
                "registry.register(model, name='credit_v1', metrics={'roc_auc': 0.84})\n"
                "\n"
                "# Script 2 — serving (separate process)\n"
                "registry2 = ModelRegistry()  # new in-memory instance\n"
                "loaded = registry2.get('credit_v1')  # returns None"
            ),
            "options": [
                "A) get() should be load() in the ModelRegistry API",
                "B) ModelRegistry is session-scoped; both scripts must be in the same Python session",
                "C) register() requires await; the model was never actually stored",
                "D) ModelRegistry() without a connection or file path creates an in-memory store that is destroyed when the process ends — each script creates a fresh empty registry; pass a shared ConnectionManager or file path to persist across processes",
            ],
            "answer": "D",
            "explanation": (
                "An in-memory ModelRegistry exists only for the lifetime of the process. "
                "To share models across scripts or services, pass a persistent backend: "
                "registry = ModelRegistry(conn) where conn is a ConnectionManager pointing "
                "to a shared SQLite or PostgreSQL database. "
                "This pattern mirrors how ExperimentTracker and FeatureStore work in MLFP02."
            ),
            "learning_outcome": "Configure ModelRegistry with persistent storage for cross-process model sharing",
        },
        # ── Lesson 8: End-to-end pipeline ─────────────────────────────────
        {
            "id": "3.8.1",
            "lesson": "3.8",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You are building a production credit scoring pipeline for a Singapore bank. "
                "Which two Kailash packages do you combine to (a) train and version the model "
                "and (b) serve predictions via a REST API with session management? "
                "Name the specific classes used from each package."
            ),
            "options": [
                "A) kailash-ml (TrainingPipeline + ModelRegistry) for training; kailash-nexus (Nexus) for serving",
                "B) kailash-ml (TrainingPipeline) for training; kailash-kaizen (Delegate) for serving — Delegate handles REST automatically",
                "C) kailash (WorkflowBuilder) for training; kailash-dataflow (DataFlow) for serving via database reads",
                "D) kailash-ml (AutoMLEngine) for training; kailash-align (AlignmentPipeline) for serving",
            ],
            "answer": "A",
            "explanation": (
                "kailash-ml provides the training lifecycle: TrainingPipeline runs experiments, "
                "ModelRegistry versions and promotes models. "
                "kailash-nexus provides production serving: Nexus() registers workflows as REST + CLI + MCP endpoints, "
                "with built-in session management, rate limiting, and auth. "
                "InferenceServer (also kailash-ml) wraps the model for the Nexus endpoint. "
                "Together: train → register → deploy via Nexus."
            ),
            "learning_outcome": "Identify the correct Kailash packages for training-to-serving pipelines",
        },
        {
            "id": "3.8.2",
            "lesson": "3.8",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "After completing the end-to-end credit scoring pipeline in Exercise 8, "
                "describe the four steps that happen between running "
                "runtime.execute(workflow.build()) and a prediction being written to the database. "
                "Your answer should reference specific Kailash classes used in Module 3."
            ),
            "options": [
                "A) workflow.build() → compile graph → LocalRuntime schedules nodes → outputs returned as dict",
                "B) workflow.build() → runtime.execute() → model saved to disk → manual database insert",
                "C) workflow.build() compiles and validates the graph; LocalRuntime executes nodes in topological order (preprocess → train → evaluate); TrainingPipeline returns a trained model artifact; db.express.create() persists the EvalResult using the @db.model schema — all four are specific to the Module 3 exercises",
                "D) workflow.build() → LocalRuntime forks a subprocess per node → results merged at end",
            ],
            "answer": "C",
            "explanation": (
                "Step 1: workflow.build() validates node connectivity and produces a compiled plan. "
                "Step 2: LocalRuntime runs nodes in dependency order — preprocess first, then train. "
                "Step 3: TrainingPipeline (inside the train node) fits the model and returns metrics. "
                "Step 4: A persistence node calls db.express.create('EvalResult', metrics_dict) "
                "to write the evaluation record to the DataFlow-managed database. "
                "This four-step sequence is the canonical Module 3 pipeline."
            ),
            "learning_outcome": "Trace the four-stage execution path from workflow.build() to database persistence",
        },
        # ── Additional questions covering lessons 1–8 breadth ─────────────
        {
            "id": "3.2.2",
            "lesson": "3.2",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "You train LightGBM on the credit dataset and plot a SHAP summary plot. "
                "The feature credit_utilisation appears at the top with SHAP values ranging "
                "from -0.8 (low utilisation) to +1.2 (high utilisation). "
                "What does the width of this range tell you about the feature, "
                "and how do you interpret a SHAP value of exactly 0.0 for a customer?"
            ),
            "options": [
                "A) The range [-0.8, +1.2] means credit_utilisation has the largest impact on the model output; a customer with very low utilisation reduces the default log-odds by 0.8, while very high utilisation increases it by 1.2. A SHAP value of 0.0 means this feature contributes nothing to the prediction for that customer; their utilisation is near the average and the model cannot discriminate on it",
                "B) Width = 2.0 means credit_utilisation has 2.0 times more importance than the base rate",
                "C) SHAP value 0.0 means the feature is missing for that customer",
                "D) The asymmetric range (-0.8 to +1.2) indicates the model is biased and should be recalibrated",
            ],
            "answer": "A",
            "explanation": (
                "SHAP values represent the marginal contribution of each feature to the prediction "
                "relative to the expected output (base value). "
                "SHAP = 0 means the feature value is exactly at the reference distribution mean — "
                "it neither increases nor decreases the prediction. "
                "The asymmetric range is expected: extreme high utilisation is more predictive of "
                "default than extreme low utilisation, because lenders rarely worry about customers "
                "who barely use their credit."
            ),
            "learning_outcome": "Interpret SHAP value ranges and zero-value SHAP for individual features",
        },
        {
            "id": "3.4.1",
            "lesson": "3.4",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 covers HyperparameterSearch. You run a grid search over "
                "LightGBM's learning_rate [0.01, 0.05, 0.1], n_estimators [100, 500, 1000], "
                "and max_depth [3, 5, 7] — 27 combinations × 5-fold CV = 135 model fits. "
                "A colleague suggests Bayesian optimisation instead. "
                "At what number of hyperparameter combinations does Bayesian optimisation "
                "typically outperform grid search, and what does kailash-ml provide for this?"
            ),
            "options": [
                "A) Bayesian optimisation is never better; grid search explores all combinations exhaustively",
                "B) Bayesian optimisation outperforms grid search when the search space has >~20 combinations — it models the performance surface probabilistically and focuses evaluations on promising regions. kailash-ml's HyperparameterSearch supports method='bayesian' which uses a surrogate model (Gaussian process or TPE) to select next trials, typically finding better hyperparameters in 30-50% fewer evaluations",
                "C) Both require the same number of evaluations; Bayesian optimisation only differs in implementation",
                "D) Bayesian optimisation requires a minimum of 10,000 trials to build a reliable surrogate",
            ],
            "answer": "B",
            "explanation": (
                "Grid search evaluates all combinations exhaustively — useful for small spaces (<20 configs). "
                "For larger spaces, each grid evaluation is independent (no learning from previous results). "
                "Bayesian optimisation fits a surrogate model (e.g., TPE in Optuna) that predicts "
                "which hyperparameter combination is likely to perform well given past trials. "
                "In practice, Bayesian optimisation finds competitive configurations in 30-50 trials "
                "instead of 135, making it 3-5× more efficient for this search space."
            ),
            "learning_outcome": "Select HyperparameterSearch method based on search space size and compute budget",
        },
        {
            "id": "3.3.3",
            "lesson": "3.3",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "After applying cost-sensitive learning in Exercise 3 (class_weight='balanced'), "
                "your precision on the default class drops from 0.68 to 0.51 while recall "
                "rises from 0.42 to 0.71. A bank risk manager says the lower precision is "
                "unacceptable (too many false alarms). "
                "Walk through how you would use threshold optimisation to find the threshold "
                "that achieves at least 0.65 precision while maximising recall."
            ),
            "options": [
                "A) Retrain with a lower class weight; threshold optimisation cannot increase precision",
                "B) Threshold optimisation only works with uncalibrated models; calibrate first",
                "C) Set threshold=0.65 directly; precision equals the threshold value",
                "D) Generate the precision-recall curve from predict_proba(), find all threshold values where precision >= 0.65, then select the threshold with the highest recall in that region. In code: from sklearn.metrics import precision_recall_curve; precision, recall, thresholds = precision_recall_curve(y_test, y_scores); valid = thresholds[precision[:-1] >= 0.65]; optimal_threshold = valid[np.argmax(recall[np.where(precision[:-1] >= 0.65)])]",
            ],
            "answer": "D",
            "explanation": (
                "precision_recall_curve() returns precision and recall arrays for all possible thresholds. "
                "Filtering to precision >= 0.65 identifies the valid operating region. "
                "Within that region, argmax(recall) finds the threshold that captures the most "
                "true defaults while staying above the precision floor. "
                "This is the business-constrained threshold optimisation covered in Exercise 3."
            ),
            "learning_outcome": "Implement constraint-based threshold optimisation using the precision-recall curve",
        },
        {
            "id": "3.1.3",
            "lesson": "3.1",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student applies Ridge regression using sklearn directly instead of "
                "going through PreprocessingPipeline first. The test RMSE is extremely high. "
                "What data preparation step is missing that PreprocessingPipeline would handle?"
            ),
            "code": (
                "from sklearn.linear_model import Ridge\n"
                "from sklearn.model_selection import train_test_split\n"
                "import numpy as np\n"
                "\n"
                "# credit DataFrame with mixed types\n"
                "X = credit.drop('default').to_numpy()  # Bug: no normalisation\n"
                "y = credit['default'].to_numpy()\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
                "\n"
                "model = Ridge(alpha=1.0)\n"
                "model.fit(X_train, y_train)  # income=50000, age=35 on same scale"
            ),
            "options": [
                "A) Ridge does not accept numpy arrays; use a Polars DataFrame",
                "B) Features are on vastly different scales (income ~50,000 vs age ~35) — Ridge's L2 penalty penalises all coefficients equally, so the income coefficient gets artificially shrunk much more than the age coefficient. PreprocessingPipeline with normalize=True standardises all features to mean=0, std=1 before fitting",
                "C) train_test_split should be replaced with cross_val_score for Ridge",
                "D) credit.drop() returns a LazyFrame; collect() is needed before to_numpy()",
            ],
            "answer": "B",
            "explanation": (
                "Ridge regularisation adds λΣwᵢ² to the loss. If income has raw values ~50,000 "
                "and age ~35, the income weight must be ~1,429× smaller to have the same prediction impact. "
                "The λ penalty treats both weights symmetrically, so it shrinks income more than age "
                "— not because income is less important, but because its scale makes the coefficient small. "
                "Standardisation (PreprocessingPipeline, normalize=True) puts all features on unit scale "
                "so the regularisation operates fairly."
            ),
            "learning_outcome": "Explain why Ridge requires feature normalisation due to scale-dependent regularisation",
        },
        {
            "id": "3.6.3",
            "lesson": "3.6",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "After running Exercise 6 and querying the EvalResult table, "
                "you have 5 model runs with ROC-AUC [0.821, 0.834, 0.847, 0.839, 0.851]. "
                "You call db.express.update() to mark the best model as production candidate. "
                "A colleague says 'just query for max AUC and update that row'. "
                "What additional check should you perform before promoting the model with "
                "AUC=0.851, and why might the 0.847 model be preferable?"
            ),
            "options": [
                "A) Check: (1) calibration — does the 0.851 model's reliability diagram show over/underconfidence that the 0.847 model does not? (2) training variance — if AUC=0.851 came from a single favourable random seed, it may not reproduce; compare std across cross-validation folds. (3) model complexity — a 0.004 AUC difference may not justify a more complex hyperparameter set that is harder to maintain",
                "B) Max AUC is always the correct criterion; no further checks needed",
                "C) Always promote the model with the highest AUC regardless of other metrics",
                "D) The 0.851 model should be promoted; the difference is statistically significant",
            ],
            "answer": "A",
            "explanation": (
                "A 0.004 AUC difference is within typical cross-validation noise for most datasets. "
                "The production decision requires: "
                "(1) Calibration — a model that is better at ranking (AUC) but miscalibrated "
                "gives wrong probabilities for threshold decisions. "
                "(2) Reproducibility — query the cv_std column in EvalResult to check variance. "
                "(3) Complexity trade-off — if 0.851 uses 1,000 trees vs 0.847's 200 trees, "
                "the inference latency and maintenance cost difference must be considered."
            ),
            "learning_outcome": "Apply multi-criteria model selection beyond AUC when promoting via ModelRegistry",
        },
        {
            "id": "3.5.3",
            "lesson": "3.5",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 5 defines a ModelSignature to specify the input/output schema for "
                "the trained credit model. A student asks: 'Why define a signature if sklearn "
                "models already have input arrays?' "
                "Name two runtime benefits ModelSignature provides that sklearn's raw predict() does not."
            ),
            "options": [
                "A) ModelSignature is only for documentation; sklearn's predict() is always sufficient",
                "B) (1) Input validation: ModelSignature checks that incoming features match the expected column names and types before calling predict(), raising a clear error instead of a silent numpy shape mismatch; (2) Schema drift detection: when production features change (e.g., a column is renamed), the signature validation catches the mismatch before the model produces a garbage prediction",
                "C) ModelSignature compiles the model to ONNX automatically",
                "D) ModelSignature is required by TrainingPipeline; there is no runtime benefit",
            ],
            "answer": "B",
            "explanation": (
                "sklearn's predict() accepts any numpy array — if you pass 15 features when the model "
                "expects 14, you get an IndexError deep in the model code, not a useful error message. "
                "ModelSignature checks: are the required feature names present? Are the dtypes correct? "
                "Does the column order match training? "
                "In production where features come from a FeatureStore or API, "
                "these checks prevent silent failures from upstream schema changes."
            ),
            "learning_outcome": "Articulate ModelSignature runtime validation benefits beyond documentation",
        },
        {
            "id": "3.7.3",
            "lesson": "3.7",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student uses TrainingPipeline but all model evaluations show identical "
                "metrics. What is wrong?"
            ),
            "code": (
                "from kailash_ml.engines.training_pipeline import TrainingPipeline, ModelSpec, EvalSpec\n"
                "\n"
                "tp = TrainingPipeline()\n"
                "\n"
                "spec1 = ModelSpec(model_type='lightgbm', params={'n_estimators': 100})\n"
                "spec2 = ModelSpec(model_type='lightgbm', params={'n_estimators': 500})\n"
                "\n"
                "eval_spec = EvalSpec(metrics=['roc_auc', 'average_precision'])\n"
                "\n"
                "# Bug: same data split used for both — no seed set, but worse: same object\n"
                "result1 = tp.run(X_train, y_train, X_test, y_test, spec=spec1, eval_spec=eval_spec)\n"
                "result2 = tp.run(X_train, y_train, X_test, y_test, spec=spec2, eval_spec=eval_spec)\n"
                "# Both return identical metrics — X_test/y_test was replaced by result1's transform"
            ),
            "options": [
                "A) ModelSpec params must be passed as keyword arguments, not a dict",
                "B) TrainingPipeline.run() may mutate the eval data in-place if a preprocessing step is applied — use separate copies: X_test1, X_test2 = X_test.copy(), X_test.copy(); or pass raw DataFrames and let the pipeline handle splitting fresh each time",
                "C) EvalSpec must be created separately for each run",
                "D) tp.run() is not designed to be called twice on the same TrainingPipeline instance",
            ],
            "answer": "B",
            "explanation": (
                "Some preprocessing steps (normalisation, encoding) operate in-place on numpy arrays. "
                "If X_test is mutated by result1's pipeline (e.g., normalised in-place), "
                "then X_test passed to result2 is already transformed — producing identical results. "
                "The safest pattern: pass polars DataFrames to TrainingPipeline and let it "
                "create its own numpy views, or explicitly copy arrays between runs."
            ),
            "learning_outcome": "Prevent in-place data mutation between sequential TrainingPipeline runs",
        },
        {
            "id": "3.4.2",
            "lesson": "3.4",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "HyperparameterSearch with Bayesian optimisation reports this result after 50 trials:\n\n"
                "  Best trial: learning_rate=0.031, n_estimators=847, max_depth=5, AUC=0.863\n"
                "  Trial 38:   learning_rate=0.031, n_estimators=843, AUC=0.862\n"
                "  Trial 45:   learning_rate=0.030, n_estimators=852, AUC=0.861\n\n"
                "A student says 'I should keep running trials to find a better n_estimators'. "
                "What does this convergence pattern actually indicate, and when should you stop?"
            ),
            "options": [
                "A) Convergence: three nearby trials (n_estimators 843-852) all give ~0.862-0.863 AUC, indicating the surrogate model has found a stable optimum. Marginal gain from additional trials is negligible. Stop when the best AUC improvement per trial drops below a threshold (e.g., 0.001 per 10 trials). The real uncertainty is whether this optimum generalises — run a fresh train/val split to validate",
                "B) Keep running — more trials always improve Bayesian optimisation",
                "C) The convergence means the model is overfit; reduce n_estimators to 100",
                "D) AUC differences < 0.005 between trials are always noise; ignore them",
            ],
            "answer": "A",
            "explanation": (
                "When Bayesian optimisation repeatedly samples similar hyperparameter values "
                "with similar scores, the surrogate model has high confidence in the optimum region. "
                "Further exploration is unlikely to find a better configuration. "
                "The marginal improvement per additional trial becomes negligible. "
                "The more important check: the best configuration found uses n_estimators=847 "
                "which may be overfit to the validation set used during search — "
                "always validate on a held-out test set that was never touched during the search."
            ),
            "learning_outcome": "Identify Bayesian optimisation convergence and determine when to stop trials",
        },
        {
            "id": "3.4.3",
            "lesson": "3.4",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "After running HyperparameterSearch on the credit dataset with "
                "method='random' (50 trials), the best trial achieves AUC=0.851. "
                "A student re-runs HyperparameterSearch with method='bayesian' (50 trials). "
                "Bayesian search achieves AUC=0.863. "
                "Both used the same search space and the same train/val split. "
                "The student concludes: 'Bayesian is always better, so I should always use it.' "
                "Under what conditions would random search be preferable?"
            ),
            "options": [
                "A) Random search is never preferable; Bayesian always finds better hyperparameters",
                "B) Random search is preferable when: (1) the search space has many unimportant dimensions (Bayesian's surrogate wastes trials modelling noise); (2) compute is cheap and trials are fast (run 500 random trials instead of 50 Bayesian trials); (3) parallelism is available (random trials are independent; Bayesian trials are sequential by design)",
                "C) Random search is preferable when the dataset is large",
                "D) Bayesian search requires a GPU; use random search on CPU-only machines",
            ],
            "answer": "B",
            "explanation": (
                "Bayesian optimisation builds a surrogate model that requires sequential trial execution — "
                "each trial informs the next. This prevents parallelism. "
                "If you can run 8 trials in parallel on 8 workers, 50 random trials take 7 wall-clock "
                "units vs 50 sequential Bayesian trials taking 50 units. "
                "For search spaces with many irrelevant hyperparameters, the surrogate model "
                "spends capacity modelling noise. Random search uniformly covers all dimensions."
            ),
            "learning_outcome": "Identify conditions where random search outperforms Bayesian optimisation",
        },
        {
            "id": "3.5.4",
            "lesson": "3.5",
            "type": "process_doc",
            "difficulty": "intermediate",
            "question": (
                "Exercise 5 connects two WorkflowBuilder nodes with add_connection(). "
                "A student adds a third node (evaluate) but does not connect it to the train node. "
                "WorkflowBuilder.build() completes without error but the evaluate node never runs. "
                "What does this tell you about WorkflowBuilder's validation, "
                "and how would you detect the disconnected node before running?"
            ),
            "options": [
                "A) WorkflowBuilder raises an error for any disconnected node; this cannot happen",
                "B) add_connection() must be called twice — once for each direction of data flow",
                "C) Disconnected nodes always produce an error at runtime, not at build() time",
                "D) WorkflowBuilder validates the graph structure but may not flag nodes with no incoming connections if they are valid starting nodes. The evaluate node silently becomes an independent root node. Detect it by calling workflow.validate() before build() or by inspecting workflow.graph.nodes to confirm every non-root node has at least one incoming connection",
            ],
            "answer": "D",
            "explanation": (
                "WorkflowBuilder.build() validates that connections are type-compatible and "
                "that the graph has no cycles. However, isolated nodes (no incoming connections) "
                "are valid as root nodes — WorkflowBuilder assumes they are intentional starting points. "
                "An evaluate node with no dependency on the train node will execute with empty input. "
                "Best practice: after wiring all connections, call workflow.validate() which has "
                "richer semantic checks, or inspect the graph to ensure dependency chains are complete."
            ),
            "learning_outcome": "Understand WorkflowBuilder.build() validation limits and use validate() for semantic checks",
        },
        {
            "id": "3.2.3",
            "lesson": "3.2",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "You train an XGBoost model alongside LightGBM on the credit dataset. "
                "Both achieve similar AUC but XGBoost takes 3× longer to train. "
                "For a weekly model retraining pipeline that must complete in under 30 minutes, "
                "and where the AUC difference is 0.002, which would you choose and why?"
            ),
            "options": [
                "A) XGBoost — the 0.002 AUC improvement is always worth the extra training time",
                "B) LightGBM — for a weekly pipeline with a 30-minute SLA, 3× slower training is a practical constraint. A 0.002 AUC difference is below the noise floor of model validation (well within CV variance). LightGBM's histogram-based algorithm was designed for large datasets with fast retraining as a primary goal",
                "C) Neither — use AutoMLEngine to decide automatically each week",
                "D) XGBoost — it handles class imbalance better than LightGBM",
            ],
            "answer": "B",
            "explanation": (
                "0.002 AUC difference is negligible — it is within the confidence interval of typical "
                "cross-validation estimates for credit scoring datasets. "
                "Operational constraints (30-minute SLA) are real hard requirements. "
                "LightGBM uses histogram-based splits that are significantly faster than XGBoost's "
                "exact greedy algorithm on large datasets. "
                "The framework-first principle applies: choose the tool that meets the requirements, "
                "not the one with the marginally highest benchmark number."
            ),
            "learning_outcome": "Balance model performance vs training time in weekly retraining pipeline design",
        },
        {
            "id": "3.6.4",
            "lesson": "3.6",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "A student's async function to query EvalResult records crashes with "
                "'RuntimeError: no running event loop'. What is wrong?"
            ),
            "code": (
                "async def get_best_model():\n"
                "    db = DataFlow('sqlite:///mlfp02_results.db')\n"
                "    # Missing: await db.connect()\n"
                "    results = await db.express.list(\n"
                "        'EvalResult',\n"
                "        filters={'promoted_to_production': True},\n"
                "    )\n"
                "    return results\n"
                "\n"
                "# Student calls it outside async context\n"
                "best = get_best_model()  # not awaited"
            ),
            "options": [
                "A) DataFlow must be imported with 'from kailash_dataflow import DataFlow'",
                "B) db.express.list() does not support filters for boolean columns",
                "C) Two bugs: (1) get_best_model() is an async function but called without await — it returns a coroutine object, not results; (2) DataFlow needs await db.connect() or async with db: before any express operations; fix: asyncio.run(get_best_model()) from sync context, and add async with db: wrapper",
                "D) 'promoted_to_production' must be a string filter: filters={'promoted_to_production': 'true'}",
            ],
            "answer": "C",
            "explanation": (
                "Two separate async bugs: "
                "(1) Calling an async function without await returns a coroutine that is never executed — "
                "in a sync context use asyncio.run(get_best_model()). "
                "(2) DataFlow requires connection setup before express operations — "
                "either await db.connect() or use async with db: as the context manager. "
                "Both are common patterns in the Module 3 exercises."
            ),
            "learning_outcome": "Fix async function calling and DataFlow connection lifecycle in the same code review",
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
