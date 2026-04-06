# Module 2: Statistical Mastery for ML and AI Success — Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: Title

**Time**: ~1 min
**Talking points**:

- Read the provocation aloud: "The feature that killed a clinical trial looked perfect in every metric."
- Let it land. This is the cautionary theme of Module 2 — statistical rigour is not optional.
- If beginners look confused: "Today we learn why getting the right answer from data is harder than it looks, and how to avoid the traps that catch even experienced teams."
- If experts look bored: "We cover CUPED variance reduction, causal forests, Double ML, and BCa bootstrap — graduate-level material applied to Singapore data."
  **Transition**: "Let us start by recapping where we left off."

---

## Slide 2: Recap: Where We Are

**Time**: ~2 min
**Talking points**:

- Quick recap of M1: Polars, DataExplorer, PreprocessingPipeline, ModelVisualizer.
- "In Module 1, you learned to SEE data. Now you will learn to REASON about it."
- Do not linger here — the audience remembers M1. This is orientation, not review.
  **Transition**: "Here is every engine you have used so far."

---

## Slide 3: Module 1 Engines — Cumulative Map

**Time**: ~1 min
**Talking points**:

- Point to the three M1 engines. "Today we add four new ones: FeatureSchema, FeatureStore, FeatureEngineer, ExperimentTracker."
- This cumulative map will grow each module. By M6, students see the full picture.
  **Transition**: "What exactly is new today?"

---

## Slide 4: What Is New Today

**Time**: ~2 min
**Talking points**:

- Left column: statistics and experiments — Bayesian thinking, MLE/MAP, hypothesis testing, bootstrap, CUPED, causal inference.
- Right column: feature engineering — what is a feature, temporal leakage, selection methods, target encoding, feature stores.
- "Two halves of the same coin. Statistics tells you whether your experiment worked. Feature engineering determines whether your model has the right inputs."
  **Transition**: "Here is the lesson roadmap."

---

## Slide 5: Module 2 Roadmap

**Time**: ~2 min
**Talking points**:

- Walk through the 8 lessons briefly. Highlight that each lesson maps to a Kailash engine.
- "Lessons 2.1-2.6 are statistics and experiments. Lessons 2.7-2.8 are feature engineering and the feature store."
  **Transition**: "Let me tell you the story that motivates everything we do today."

---

## Slide 6: Opening Case — The Feature That Killed a Clinical Trial

**Time**: ~4 min
**Talking points**:

- Tell this as a story. "A major hospital built a readmission-prediction model. 0.97 AUC. Leadership approved a clinical trial. Three months later, it was halted. The model was useless."
- Pause after the reveal. Let the audience guess.
- "One of the input features was 'discharge medication list' — information that only exists AFTER the outcome has already happened."
- "The model was predicting the past, not the future."
- If beginners look confused: "Imagine predicting whether a student will fail the exam, but one of your inputs is 'what grade they received.' That information does not exist at prediction time."
- If experts look bored: "This is a textbook case of target leakage through temporal violation. We will formalise the point-in-time correctness rule and show how FeatureStore prevents it architecturally."
  **Transition**: "Let us see exactly what went wrong."

---

## Slide 7: Feature Leakage — The Silent Killer

**Time**: ~3 min
**Talking points**:

- Walk through the timeline: patient admitted, vitals recorded, lab results available, discharge meds decided (DANGER), readmitted or not (TARGET).
- "At prediction time, you can only use information that EXISTED at that moment."
- "This is why Module 2 exists. Statistical rigour and proper feature engineering are non-negotiable in production ML."
  **Transition**: "Here is the lesson from this case."

---

## Slide 8: The Lesson

**Time**: ~2 min
**Talking points**:

- Left: what went wrong — no temporal validation, train/test split ignored time, high AUC masked the problem.
- Right: what we will learn — point-in-time correctness (2.7), FeatureStore enforces it (2.8), experiment design catches it (2.3), causal thinking prevents it (2.6).
- Kailash bridge: "Point-in-time correctness is built into the FeatureStore engine. You cannot accidentally leak future information."
  **Transition**: "Let us build vocabulary before diving into the mathematics."

**[PAUSE FOR QUESTIONS — 2 min]**

---

## Slide 9: Foundations Section Header

**Time**: ~0.5 min
**Talking points**:

- "Everyone follows these slides. We are building vocabulary and intuition."

---

## Slide 10: What Is a Feature?

**Time**: ~3 min
**Talking points**:

- "A feature is a measurable property of your data that you feed to a model. Think of it as a column in a spreadsheet that helps the model make decisions."
- Walk through the HDB example: floor area, town, remaining lease, storey range, month of sale.
- "Target = what you predict. Features = the inputs that help you predict it."
- If beginners look confused: "In the HDB case, the target is resale price. Everything else — area, town, floor level — are features."
  **Transition**: "Why does feature engineering matter so much?"

---

## Slide 11: Why Feature Engineering Matters

**Time**: ~3 min
**Talking points**:

- "A simple model with great features beats a complex model with bad features."
- Cooking analogy: "The algorithm is your recipe. The features are your ingredients. Even the best chef cannot make a great meal from spoiled ingredients."
- Walk through the progressive engineering examples: raw date to month/quarter/days_since_cooling_measure, distance_to_nearest_MRT, price_per_sqm_rolling_6m.
- "Each engineered feature captures domain knowledge that the raw data does not contain."
  **Transition**: "To know whether your features actually help, you need experiments."

---

## Slide 12: What Is an Experiment?

**Time**: ~3 min
**Talking points**:

- "An experiment is a structured test where you change one thing and measure the effect."
- Walk through the everyday example: balcony photo on HDB listings.
- Five key ingredients: hypothesis, control, treatment, random assignment, metric.
- If beginners look confused: "Think of cooking: you make two batches of the same recipe, changing only one ingredient. Then you taste both. That is an experiment."
  **Transition**: "Correlation and causation are not the same thing."

---

## Slide 13: Correlation vs. Causation

**Time**: ~3 min
**Talking points**:

- Ice cream and drowning: correlated but not causal. Both caused by hot weather.
- "ML models learn correlations. They cannot, by themselves, tell you what causes what. That is why causal inference in Lesson 2.6 matters."
- If beginners look confused: "Just because two things happen together does not mean one causes the other."
  **Transition**: "The simplest experiment is an A/B test."

---

## Slide 14: What Is an A/B Test?

**Time**: ~2 min
**Talking points**:

- Two groups see different versions, you measure which performs better.
- "After enough people have seen each version, you use statistics to determine whether the difference is real or just random noise."
- If beginners look confused: "It is like a taste test. Half the people get recipe A, half get recipe B, you count which gets more thumbs up."
  **Transition**: "To understand A/B tests, you need probability."

---

## Slide 15: What Is Probability?

**Time**: ~3 min
**Talking points**:

- Two views: frequentist (long-run frequency) and Bayesian (degree of belief).
- "Both views are useful. Module 2 teaches you both."
- If beginners look confused: "Frequentist: flip a coin many times, count heads. Bayesian: based on what I know, how confident am I?"
  **Transition**: "Distributions describe the shape of uncertainty."

---

## Slide 16: Distributions — The Shape of Uncertainty

**Time**: ~3 min
**Talking points**:

- Normal: most values cluster around the average. Heights, test scores.
- Beta: values between 0 and 1. Conversion rates, probabilities.
- "You will use both today. Normal for prices, Beta for A/B test conversion rates."
  **Transition**: "Bayes' theorem is the formula for updating beliefs."

---

## Slide 17: Bayes' Theorem — Updating Beliefs

**Time**: ~4 min
**Talking points**:

- Rain analogy: "You think 30% chance of rain. You see dark clouds. You update to 70%. Bayes' theorem is the formula for that update."
- Walk through the equation: posterior = likelihood times prior divided by evidence.
- "This is the most important equation in Module 2. Everything else builds on it."
- If beginners look confused: "Prior = what you believed before. Data changes your mind. Posterior = what you believe after."
- If experts look bored: "We will derive the full conjugate update, Fisher information, and Cramer-Rao bound in the theory section."
  **Transition**: "Let us summarise the foundations before diving into the maths."

---

## Slide 18: Foundations Summary

**Time**: ~2 min
**Talking points**:

- Quick review of all seven concepts: feature, feature engineering, experiment, A/B test, correlation vs causation, probability, Bayes' theorem.
- "Now let us see the mathematics behind these ideas."

**[PAUSE FOR QUESTIONS — 3 min]**

**Transition**: "Theory Block A: Bayesian Thinking, MLE/MAP, Hypothesis Testing."

---

## Slide 19: Theory Block A Header

**Time**: ~0.5 min

---

## Slide 20: 2.1 Conjugate Priors — The Shortcut (THEORY)

**Time**: ~4 min
**Talking points**:

- "When the prior and likelihood belong to the same family, the posterior has a closed-form solution."
- Beta-Binomial: add successes to alpha, failures to beta. Use case: conversion rates.
- Normal-Normal: precision-weighted average. Use case: HDB price estimation with prior knowledge.
- If beginners look confused: "The key takeaway is that sometimes the maths works out perfectly, and you get an exact answer without computers."
  **Transition**: "Let us see a step-by-step Bayesian update."

---

## Slide 21: Bayesian Update — Step by Step (THEORY)

**Time**: ~4 min
**Talking points**:

- Walk through all four steps: prior Beta(2,8), data 35/100 clicks, posterior Beta(37,73), 95% credible interval [25%, 42.8%].
- "The posterior is a compromise between prior (20%) and data (35%). With more data, the posterior shifts toward the data."
- If beginners look confused: "Before seeing data, we guessed 20%. After 100 visitors, 35 clicked. Our updated belief is about 34%. The more data we get, the less our initial guess matters."
  **Transition**: "How much does the prior matter?"

---

## Slide 22: Prior Sensitivity (THEORY)

**Time**: ~3 min
**Talking points**:

- Show the table: with n=10, prior matters a lot. With n=1000, all priors converge.
- "Bernstein-von Mises theorem: as n goes to infinity, all priors converge to the same posterior."
- "In practice, with n > 100, moderate priors barely matter."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Now let us find the single best parameter value."

---

## Slide 23: 2.2 Maximum Likelihood Estimation (THEORY)

**Time**: ~4 min
**Talking points**:

- "Find the parameter values that make the observed data most probable."
- Product turns to sum via logarithm. "We take the log because products of small numbers underflow."
- If beginners look confused: "MLE is like trying on different explanations and picking the one that fits the data best."
  **Transition**: "Let us derive the MLE for the Normal distribution."

---

## Slide 24: MLE for the Normal — Derivation (THEORY)

**Time**: ~4 min
**Talking points**:

- Walk through all four steps: log-likelihood, differentiate w.r.t. mu, solve, get sample mean.
- "The MLE for the mean is the sample mean. Intuitive and optimal."
- Variance MLE divides by n (biased), not n-1.
  **Transition**: "How good can an estimator be?"

---

## Slide 25: Fisher Information & Cramer-Rao Bound (THEORY)

**Time**: ~3 min
**Talking points**:

- "Fisher information = expected curvature of the log-likelihood. Sharp curvature = lots of information."
- Cramer-Rao bound: "No unbiased estimator can have variance lower than 1/(n \* I(theta))."
- "If the log-likelihood is flat, the data does not help you pin down theta."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "MAP adds a prior to MLE."

---

## Slide 26: MAP Estimation (THEORY)

**Time**: ~3 min
**Talking points**:

- "MAP = MLE + regularisation from the prior."
- When to use MLE: lots of data, no prior knowledge, simpler. When to use MAP: small datasets, domain knowledge, regularisation needed.
- "Preview: L2 regularisation = Gaussian prior. You will see this in Module 3."
  **Transition**: "When does MLE fail?"

---

## Slide 27: When MLE Fails (THEORY)

**Time**: ~2 min
**Talking points**:

- Small n, multimodal likelihood, model misspecification, infinite likelihood.
- Bayesian model comparison: BIC, WAIC, LOO-CV.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Now let us design experiments properly."

---

## Slide 28: 2.3 Hypothesis Testing — The Framework (THEORY)

**Time**: ~4 min
**Talking points**:

- Neyman-Pearson framework: H0, H1, alpha, beta, power.
- Walk through the decision table. "Type I = false alarm. Type II = missed effect. Power = probability of catching a real effect."
- If beginners look confused: "H0 says 'nothing interesting happened.' H1 says 'something changed.' We collect evidence and decide which is more likely."
  **Transition**: "What does the p-value actually mean?"

---

## Slide 29: P-Value — What It Really Means (THEORY)

**Time**: ~4 min
**Talking points**:

- Read the warning callout carefully. "The p-value is NOT the probability that H0 is true."
- "Correct interpretation: if there were truly no effect, how surprising would our observed data be?"
- This is one of the most misunderstood concepts in all of statistics. Spend time on it.
- If beginners look confused: "A small p-value means 'this result would be very unusual if there were no real effect.' That is all."
  **Transition**: "Before running an experiment, how many observations do you need?"

---

## Slide 30: Power Analysis (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the formula and the Singapore example: testing whether new MRT increases HDB prices by S$10K, need ~393 flats per group.
- "Power analysis is how you avoid wasting time on an experiment that was never going to detect anything."
  **Transition**: "Testing many things at once creates a problem."

---

## Slide 31: Multiple Testing — The Bonferroni Trap (THEORY)

**Time**: ~3 min
**Talking points**:

- "Testing 20 features at alpha=0.05? You expect 1 false positive by pure chance."
- Bonferroni: conservative, many real effects missed. Benjamini-Hochberg FDR: less conservative, preferred in practice.
- Permutation tests: exact p-values when distributional assumptions fail.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "ExperimentTracker automates all of this."

---

## Slide 32: Kailash Bridge — ExperimentTracker (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the code: create experiment, log parameters (alpha, power, MDE), log metrics (p-value, effect size, CI).
- "Everything is logged, versioned, comparable. You can reproduce any experiment months later."

**[SWITCH TO LIVE CODING — 3 min]**

- Demonstrate: create an experiment, log params and metrics.
  **Transition**: "Theory Block B: Bootstrap, CUPED, Causal Inference."

**[PAUSE FOR QUESTIONS — 3 min]**

---

## Slide 33: Theory Block B Header

**Time**: ~0.5 min

---

## Slide 34: 2.4 Bootstrap — Efron's Insight (THEORY)

**Time**: ~4 min
**Talking points**:

- "Cannot collect more data? Resample with replacement from what you have."
- Walk through the four steps. Show the standard error formula.
- If beginners look confused: "Imagine putting all your data on slips of paper in a hat. Draw a slip, note it, put it back, repeat. Each draw gives you a slightly different estimate."
  **Transition**: "The simple bootstrap can be improved."

---

## Slide 35: BCa Bootstrap Confidence Intervals (THEORY)

**Time**: ~3 min
**Talking points**:

- Bias correction z0 and acceleration a-hat.
- "BCa corrects for both bias and skewness in the bootstrap distribution."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "When does bootstrap fail?"

---

## Slide 36: When Bootstrap Fails (THEORY)

**Time**: ~2 min
**Talking points**:

- Extrema: cannot exceed observed range. Heavy tails: variance may not exist. Dependent data: breaks temporal structure. Small n: not enough variation.
- Wild bootstrap for heteroscedastic data.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "CUPED is the single most impactful A/B testing technique."

---

## Slide 37: 2.5 CUPED — Variance Reduction (THEORY)

**Time**: ~5 min
**Talking points**:

- "CUPED uses pre-experiment data to reduce noise in your A/B test. Less noise = detect smaller effects with fewer observations."
- Walk through the formula: adjusted metric = Y - theta \* (X - X_bar).
- "This is the single most impactful technique in modern A/B testing. Microsoft, Netflix, and Airbnb all use it."
- If beginners look confused: "If you know how each user behaved BEFORE the experiment, you can subtract out the predictable part, leaving only the treatment effect."
  **Transition**: "Let us prove that CUPED actually reduces variance."

---

## Slide 38: CUPED Variance Reduction — Full Derivation (THEORY)

**Time**: ~5 min
**Talking points**:

- Walk through all four steps carefully. The key result: Var(Y_adj) = Var(Y)(1 - rho^2).
- "If rho = 0.7, variance drops by 51%. If rho = 0.9, variance drops by 81%."
- "The optimal theta is the OLS regression coefficient of Y on X."
- If beginners look confused: "The bottom line: the more correlated the pre-experiment behaviour is with the post-experiment metric, the more variance we can remove."
  **Transition**: "What does this mean in practice?"

---

## Slide 39: CUPED — Practical Impact (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the table: rho=0.5 gives 1.3x sample multiplier; rho=0.9 gives 5.3x; rho=0.95 gives 10x.
- Singapore example: "Without CUPED: 50,000 users. With CUPED (rho=0.8): 14,000 users. You run the experiment 3.6x faster."
  **Transition**: "Here is how ExperimentTracker integrates CUPED."

---

## Slide 40: Kailash Bridge — ExperimentTracker + CUPED (THEORY)

**Time**: ~2 min
**Talking points**:

- Walk through the code: log covariate, method, rho, variance reduction, adjusted p-value.
- "Compare raw A/B test with CUPED-adjusted version using compare_runs."
  **Transition**: "Now let us go beyond correlation to causation."

---

## Slide 41: 2.6 Causal Inference — Beyond Correlation (THEORY)

**Time**: ~4 min
**Talking points**:

- "ML models predict. Causal inference tells you what would happen if you intervened."
- Rubin's potential outcomes: Y(1) = outcome if treated, Y(0) = outcome if not treated.
- "The Fundamental Problem: you can only observe ONE of the two potential outcomes. The other is the counterfactual."
- If beginners look confused: "You cannot give someone a medicine AND not give them the medicine at the same time. One of those outcomes is always imaginary."
  **Transition**: "Three types of treatment effects."

---

## Slide 42: Treatment Effects — ATE, ATT, CATE (THEORY)

**Time**: ~3 min
**Talking points**:

- ATE: average effect across the entire population.
- ATT: average effect among those actually treated.
- CATE: effect for a specific subgroup.
- Singapore example: "ATE of cooling measures on all HDB prices. ATT on affected flats. CATE for 3-room vs 5-room."
  **Transition**: "Pearl's DAGs encode causal assumptions visually."

---

## Slide 43: Pearl's DAGs (THEORY)

**Time**: ~3 min
**Talking points**:

- Confounder Z causes both treatment T and outcome Y. Must adjust for it.
- d-Separation: "Conditioning on a confounder blocks the backdoor path. Conditioning on a collider opens a spurious path."
- If beginners look confused: "A DAG is a map of what causes what. Drawing it forces you to think about which variables are confounders."
  **Transition**: "Two criteria for identifying causal effects."

---

## Slide 44: Backdoor & Front-Door Criteria (THEORY)

**Time**: ~3 min
**Talking points**:

- Backdoor: block all backdoor paths from T to Y, no descendants of T.
- Front-door: when confounders are unobserved, use a mediator.
- Show both formulas. "The backdoor criterion is used in 90% of applied causal inference."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Difference-in-Differences is the workhorse of policy evaluation."

---

## Slide 45: Difference-in-Differences (THEORY)

**Time**: ~4 min
**Talking points**:

- "Compare the change in a treated group to the change in an untreated group."
- Walk through the table: treatment change minus control change.
- "If both groups were trending similarly before the intervention, the difference in their changes is the causal effect."
- If beginners look confused: "Imagine two towns. Both have rising prices. A new policy affects only one town. Compare how much prices changed in each. The difference is the policy's effect."
  **Transition**: "Let us derive the ATT estimator."

---

## Slide 46: DiD — ATT Estimator Derivation (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through: observed, counterfactual (parallel trends), ATT, regression form.
- "The coefficient delta on the interaction term D_i x T_t is the DiD estimate."
  **Transition**: "The parallel trends assumption is critical."

---

## Slide 47: DiD — Parallel Trends & Placebo Tests (THEORY)

**Time**: ~3 min
**Talking points**:

- "Parallel trends is the critical assumption. If groups were already diverging, DiD is biased."
- Visual inspection: plot both groups over time.
- Placebo tests: fake treatment time (should find no effect), fake treatment group (should find no effect), pre-trend regression.
  **Transition**: "Let us apply DiD to Singapore cooling measures."

---

## Slide 48: DiD — Singapore Cooling Measures (THEORY)

**Time**: ~3 min
**Talking points**:

- ABSD increase December 2021. Treatment: investment purchases. Control: first-time buyers.
- DiD estimate: -S$18,200 (significant at p < 0.01). Cooling measures reduced investment-segment prices.
- If beginners look confused: "The government raised taxes on investment property. Prices fell in that segment. DiD tells us the tax caused the price drop, not something else."
  **Transition**: "For the experts: advanced causal methods."

---

## Slide 49: Double ML / Debiased ML (ADVANCED)

**Time**: ~3 min
**Talking points**:

- "Use ML models to estimate nuisance parameters, then use residuals for causal estimation."
- Two models: predict outcome from confounders, predict treatment from confounders. Causal estimate from residual regression.
- "Key requirement: cross-fitting to avoid overfitting bias."

**[CAN SKIP IF RUNNING SHORT]**

---

## Slide 50: Causal Forests (ADVANCED)

**Time**: ~3 min
**Talking points**:

- "CATE tells you who benefits most. Causal forests estimate heterogeneous treatment effects."
- Singapore application: MRT station openings. Mature estates +S$15K, non-mature +S$42K.
- "Policy insight: target transit investment where it matters most."

**[CAN SKIP IF RUNNING SHORT]**

---

## Slide 51: Bayesian A/B Testing (ADVANCED)

**Time**: ~3 min
**Talking points**:

- "Instead of p-values, compute the posterior probability that B beats A."
- Advantages: intuitive, no fixed sample size, natural early stopping.

**[CAN SKIP IF RUNNING SHORT]**

---

## Slide 52: Interference & Network Effects (ADVANCED)

**Time**: ~2 min
**Talking points**:

- SUTVA violation: one user's treatment affects another's outcome.
- Solutions: cluster randomisation, switchback experiments.

**[CAN SKIP IF RUNNING SHORT]**

---

## Slide 53: Kailash Bridge — ExperimentTracker for Causal Inference (THEORY)

**Time**: ~2 min
**Talking points**:

- Walk through the code: log DiD parameters, ATT estimate, placebo p-value.
- "Every experiment is logged with full methodology."

**[PAUSE FOR QUESTIONS — 3 min]**

**Transition**: "Theory Block C: Feature Engineering, Feature Selection, Feature Store."

---

## Slide 54: Theory Block C Header

**Time**: ~0.5 min

---

## Slide 55: 2.7 Feature Selection — Mutual Information (THEORY)

**Time**: ~4 min
**Talking points**:

- "How much does knowing one variable tell you about another?"
- "Why not just use correlation? Correlation only captures linear relationships. A feature with zero correlation can have high mutual information."
- Walk through the formula. "If X and Y are independent, MI = 0. Higher MI = more useful feature."
  **Transition**: "Boruta uses a different approach."

---

## Slide 56: Feature Selection — Boruta Algorithm (THEORY)

**Time**: ~3 min
**Talking points**:

- "Is this feature more important than random noise?"
- Walk through the five steps: shadow features, Random Forest, compare importance.
- "By comparing against the best random noise, Boruta finds ALL relevant features, not just the top-k."
  **Transition**: "Stability selection checks consistency."

---

## Slide 57: Stability Selection (THEORY)

**Time**: ~2 min
**Talking points**:

- "Run feature selection on many subsamples. Features consistently selected are truly important."
- Meinshausen & Buhlmann (2010): finite-sample error control.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Temporal features require special care."

---

## Slide 58: Temporal Features — Lag, Rolling, Fourier (THEORY)

**Time**: ~3 min
**Talking points**:

- Lag features: value k periods ago. Rolling features: mean over window w. Fourier features: capture cyclical patterns.
- Show the Polars code for each.
- If beginners look confused: "Lag features answer 'what happened last month?' Rolling features answer 'what has the trend been?'"
  **Transition**: "Temporal features are where leakage happens most."

---

## Slide 59: Point-in-Time Correctness — The Timeline (THEORY)

**Time**: ~5 min
**Talking points**:

- Walk through the timeline: t-3 to t-1 are safe, t is prediction time, t+1 and t+2 are LEAKED.
- Common leakage sources: rolling mean includes future data, target encoding uses all data, feature computed after the event, join brings future records.
- Detection: suspiciously high accuracy, performance drops in production, one feature dominates importance.
- "FeatureStore enforces point-in-time automatically."
- If beginners look confused: "The rule is simple: at the moment you need the prediction, would this information actually exist? If not, you cannot use it."
  **Transition**: "Target encoding needs special treatment."

---

## Slide 60: Target Encoding — James-Stein Shrinkage (THEORY)

**Time**: ~4 min
**Talking points**:

- "1,000 postcodes. One-hot creates 1,000 columns. Target encoding replaces each with the mean target — but naive encoding overfits."
- James-Stein shrinkage: small categories shrink toward the global mean.
- "A postcode with 3 observations at S$800K is unreliable. Shrink toward the global mean. A postcode with 3,000 keeps its own mean."
- "This is exactly the Bayesian posterior mean under a Normal-Normal model."
- If beginners look confused: "Categories with few examples get unreliable estimates. Shrinkage pulls them toward the overall average, which is more stable."
  **Transition**: "Interaction and polynomial features capture non-linear effects."

---

## Slide 61: Interaction & Polynomial Features (THEORY)

**Time**: ~3 min
**Talking points**:

- Interaction: floor area matters MORE in central locations. Multiply features together.
- Polynomial: diminishing returns on floor area. Square and log transforms.
- "Be careful: feature explosion is real. p features with degree d creates many columns."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "FeatureEngineer and FeatureSchema automate all of this."

---

## Slide 62: Kailash Bridge — FeatureEngineer & FeatureSchema (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the code: declare FeatureSchema with typed, versioned, auditable fields.
- "Feature contracts ensure everyone uses the same definition. No ambiguity."
- engineer.generate() creates all declared features. engineer.select() keeps the important ones.

**[SWITCH TO LIVE CODING — 3 min]**

- Demonstrate: define a schema, generate features.
  **Transition**: "Now let us store these features properly."

---

## Slide 63: 2.8 Feature Store — Why It Matters (THEORY)

**Time**: ~4 min
**Talking points**:

- "A Feature Store is a centralised warehouse for your features."
- Walk through the problems without one: training-serving skew, no version tracking, no temporal validation, duplicated features.
- "FeatureStore enforces contracts, prevents leakage, and ensures training and production use the same features."
  **Transition**: "Here is the feature store lifecycle."

---

## Slide 64: Feature Store Lifecycle (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through: Define Schema, Ingest Features, Version Track, Serve Online, Monitor Drift.
- Offline store: historical features for training. Point-in-time joins. Time-travel queries.
- Online store: latest feature values for real-time inference. Low-latency.
  **Transition**: "Point-in-time joins are the key mechanism."

---

## Slide 65: Point-in-Time Joins (THEORY)

**Time**: ~4 min
**Talking points**:

- Walk through the table: event at 2024-03-15. Standard join uses March value (LEAKED). Point-in-time join uses Feb value (correct).
- "Kailash FeatureStore handles this automatically. You never write point-in-time join logic manually."
- This is one of the most important slides in the module. Make sure it lands.
- If beginners look confused: "The store always asks: 'what was the most recent value BEFORE this date?' Never the value computed after."
  **Transition**: "Data lineage completes the picture."

---

## Slide 66: Data Lineage & Reproducibility (THEORY)

**Time**: ~2 min
**Talking points**:

- "Lineage = a complete record of where each feature came from, how it was computed, what version is being used."
- Debugging, compliance (MAS, EU AI Act), reproducibility.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Let us see the Kailash engines in detail."

---

## Slide 67: Kailash Engine Deep Dive Header

**Time**: ~0.5 min

---

## Slide 68: FeatureStore — Architecture (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the internal pipeline: Schema Validation, Type Checking, Temporal Indexing, Version Stamp, Lineage Record.
- Key methods: register_schema, ingest, get_features, get_lineage, list_versions.
- Theory-to-engine map: point-in-time = get_features(as_of=), feature contracts = FeatureSchema, etc.
  **Transition**: "Let us see the code."

---

## Slide 69: FeatureStore — Code Walkthrough (THEORY)

**Time**: ~4 min
**Talking points**:

- Walk through all four steps: define schema, create store, ingest features, retrieve with point-in-time.
- "as_of='2024-01-01' means no data after this date. The store enforces it."

**[SWITCH TO LIVE CODING — 5 min]**

- Demonstrate: define schema, ingest, retrieve with as_of.
  **Transition**: "ExperimentTracker ties it all together."

---

## Slide 70: ExperimentTracker — Full API Surface (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the method table: create_experiment, log_param, log_metric, log_artifact, compare_runs, get_best_run.
- "Reproducibility guarantee: every experiment logged with parameters, metrics, artifacts, timestamps."
  **Transition**: "Here is how all the engines connect."

---

## Slide 71: Module 2 Engine Architecture (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the flow: FeatureSchema (contracts) -> FeatureEngineer (generate + select) -> FeatureStore (store + version) -> ExperimentTracker (log + compare).
- Governance integration: schemas are auditable, versions are immutable, experiment logs are compliance evidence.
  **Transition**: "Time for the lab."

**[PAUSE FOR QUESTIONS — 3 min]**

---

## Slide 72: Lab Section Header

**Time**: ~0.5 min

---

## Slide 73: Lab Overview

**Time**: ~3 min
**Talking points**:

- Walk through five exercises: feature engineering pipeline, Feature Store with point-in-time, CUPED implementation, DiD analysis, capstone.
- "Start with ex_1 and work through. Solutions are available if you get stuck."

**[SWITCH TO EXERCISE DEMO — 5 min]**

- Open ex_1.py. Walk through the structure.
  **Transition**: "Here are the datasets you will use."

---

## Slide 74: Lab — Datasets

**Time**: ~2 min
**Talking points**:

- HDB resale: 15M+ rows for feature engineering and leakage detection.
- A/B test simulation: listing page engagement, pre-experiment covariates for CUPED, cooling measure dates for DiD.
- Show the data loader code.
  **Transition**: "Module 2 provides ~60% scaffolding."

---

## Slide 75: Lab — Progressive Scaffolding

**Time**: ~2 min
**Talking points**:

- "All imports, data loading, and structure are given. You fill in engine parameters, some method calls, interpretation of results."
- Show the TODO marker example with FeatureSchema.

**[BEGIN LAB — allocate remaining time]**

---

## Slides 76-78: Discussion Prompts

**Time**: ~10 min total (after lab or interleaved)
**Talking points**:

- **Feature Leakage in Your Domain**: "Think about your own industry. What data might accidentally leak the future?" Finance: end-of-month balance for mid-month prediction? Healthcare: doctor's notes from the same shift?
- **When A/B Tests Fail**: "p-value is 0.08 after 3 weeks. Manager wants to extend until significant." Discuss optional stopping and p-hacking. Could CUPED have helped?
- **Causal Thinking in Policy**: Walk through Singapore policies — ABSD (DiD), CPF grants (regression discontinuity), MRT stations (DiD + CATE), WFH (Double ML). Discuss challenges for each.

**[PAUSE FOR QUESTIONS — 3 min]**

---

## Slide 79: Synthesis Header

**Time**: ~0.5 min

---

## Slide 80: Key Takeaways — By Level

**Time**: ~5 min
**Talking points**:

- FOUNDATIONS: good features beat complex models, never use future information, correlation is not causation, FeatureStore enforces correctness.
- THEORY: CUPED reduces variance by 1-rho^2, DiD identifies causal effects, MLE/MAP estimation, James-Stein shrinkage.
- ADVANCED: Double ML, causal forests, Bayesian A/B testing, SUTVA violations.
  **Transition**: "Here is the updated engine map."

---

## Slide 81: Kailash Cumulative Engine Map

**Time**: ~2 min
**Talking points**:

- M1: DataExplorer, PreprocessingPipeline, ModelVisualizer.
- M2 (new): FeatureSchema, FeatureEngineer, FeatureStore, ExperimentTracker.
- "Next module adds: TrainingPipeline, HyperparameterSearch, ModelRegistry, WorkflowBuilder."
  **Transition**: "What connects M2 to M3?"

---

## Slide 82: Connection to Module 3

**Time**: ~3 min
**Talking points**:

- What you now know: engineer features that do not leak, store and version features, design experiments, make causal claims.
- What comes next: training pipelines, bias-variance tradeoff, gradient boosting, SHAP, production deployment.
- "Module 3 takes the features from your store and turns them into production models."
  **Transition**: "A preview of the assessment."

---

## Slide 83: Assessment Preview

**Time**: ~2 min
**Talking points**:

- Quiz topics: identify feature leakage, choose feature selection method, interpret CUPED output, validate parallel trends, debug FeatureSchema, interpret ExperimentTracker output.
- "AI-resilient: questions require running code and interpreting YOUR specific outputs."
  **Transition**: "References for further reading."

---

## Slide 84: References & Further Reading

**Time**: ~1 min
**Talking points**:

- Point to key references: Deng et al. for CUPED, Chernozhukov for Double ML, Athey & Imbens for causal forests, Pearl for DAGs.
- "These are graduate-level references. Read them if you want to go deeper."

---

## Slide 85: Closing

**Time**: ~1 min
**Talking points**:

- "You now reason about data — features, experiments, causation."
- Read the closing provocation: "Point-in-time correctness is non-negotiable."
- "See you in Module 3: Supervised ML — Theory to Production."

**[FINAL Q&A — 5 min]**

---

## Timing Summary

| Section                                              | Slides    | Time         |
| ---------------------------------------------------- | --------- | ------------ |
| Title + Recap + Roadmap                              | 1-5       | ~8 min       |
| Opening Case (Clinical Trial)                        | 6-8       | ~9 min       |
| Foundations Block                                    | 9-18      | ~24 min      |
| Theory Block A: Bayesian + MLE + Hypothesis Testing  | 19-32     | ~35 min      |
| Theory Block B: Bootstrap + CUPED + Causal Inference | 33-53     | ~40 min      |
| Theory Block C: Feature Engineering + Feature Store  | 54-66     | ~30 min      |
| Kailash Engine Deep Dive                             | 67-71     | ~13 min      |
| Lab Setup + Exercises                                | 72-75     | ~12 min      |
| Discussion Prompts                                   | 76-78     | ~10 min      |
| Synthesis + Closing                                  | 79-85     | ~14 min      |
| Q&A buffers                                          | scattered | ~14 min      |
| **Total**                                            |           | **~180 min** |
