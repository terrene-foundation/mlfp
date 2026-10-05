# Module 3: Supervised Machine Learning for Building and Deploying Models — Speaker Notes

Master deck: 91 slides. Talking points below sum to ~166 minutes of presented material; with room transitions, live Q&A, and the short coding demos embedded in Lessons 3.1, 3.4, and 3.8, plan ~180 minutes for a full-deck walkthrough. Taught lesson-by-lesson, the eight lesson decks total ~560 minutes (~9.5 hours across sessions).
Audience: working professionals; instructors must scaffold for both novices and experienced ML practitioners. Use the three-layer markers on the deck (FOUNDATIONS / THEORY / ADVANCED) to pace the room.

---

## Slide 1: Supervised Machine Learning for Building and Deploying Models

**Time**: ~2 min
**Talking points**:

- Welcome the class. This module takes everything from M1 (data pipelines) and M2 (statistics and regression) and builds the complete supervised ML pipeline.
- By the end of 8 lessons, students will train every major supervised model, evaluate honestly, interpret predictions, and deploy to production with drift monitoring.
- Ask the room: "How many of you have trained an ML model before?" Gauge experience and adjust depth accordingly.
- Three layers apply as always: green for everyone, blue for the math, purple for experts.
- If beginners look confused: "M2 gave you one model — linear regression. Today we give you the whole toolbox and show you how professionals pick the right tool and ship it to production."
- If experts look bored: "We derive the XGBoost split gain from a second-order Taylor expansion, prove the fairness impossibility theorem, and build a nested cross-validation loop. There is depth here for practitioners too."
  **Transition**: "Let us start with what you will be able to do by the end of today."

---

## Slide 2: What You Will Learn

**Time**: ~2 min
**Talking points**:

- Walk through the three layers: FOUNDATIONS (green) — skills everyone leaves with; THEORY (blue) — the math of why it works; ADVANCED (purple) — production engineering.
- Reassure foundations-level students they will not be left behind. Tell advanced students the blue and purple slides contain full mathematical derivations.
- Call out that the lesson markers on every slide tell them whether to engage or rest.
- If beginners look confused: "You do not need to follow every formula to finish the exercises. The foundations track alone lets you train, evaluate, and deploy a real model."
- If experts look bored: "The blue and purple content goes past most masters-level syllabi — the bias-variance derivation, the XGBoost Newton step, and the impossibility result."
  **Transition**: "Here is the roadmap for our 8 lessons."

---

## Slide 3: Your Journey: 8 Lessons

**Time**: ~1 min
**Talking points**:

- Quick overview — do not read every cell of the table.
- The progression: features first (3.1), then the theory of learning (3.2), then all models (3.3-3.4), then honest evaluation (3.5), then interpretation (3.6), then engineering it (3.7-3.8). Each lesson builds on the last.
- "By Lesson 3.8, you will have a credit-scoring model in a database, with drift monitoring, a model card, and a promotion gate — the same stack a regulated lender ships."
  **Transition**: "To do all that, you will meet a lot of Kailash engines today. Here is the cumulative map."

---

## Slide 4: Kailash Engines — Cumulative Map

**Time**: ~1 min
**Talking points**:

- Show the cumulative build-up. M3 is the largest engine introduction in the course: ten new engines in one module — FeatureEngineer, FeatureStore, TrainingPipeline, AutoMLEngine, HyperparameterSearch, ModelRegistry, EnsembleEngine, WorkflowBuilder, DataFlow, DriftMonitor.
- By the end of M3, students have the full ML lifecycle toolkit, from raw features to monitored production models.
- Reassure: we introduce them progressively across the 8 lessons — never all ten at once.
- If beginners look confused: "Think of these as power tools. We teach the concept by hand first, then show the Kailash engine that automates it."
- If experts look bored: "The engines wrap the usual sklearn, XGBoost, SHAP, and MLflow-style stack with polars-native APIs and point-in-time correctness. The abstraction is thin."
  **Transition**: "Before we dive in, let me place M3 on the MLFP map."

---

## Slide 5: Where We Are

**Time**: ~1 min
**Talking points**:

- Quick recap. Do not linger. Students who missed prior modules get oriented.
- The key connection: M2's linear regression is the simplest supervised model. M3 adds the full zoo, proper evaluation, interpretability, and production deployment.
- M4 completes the Foundation Certificate with unsupervised learning; the Advanced Certificate is M5 (deep learning and vision) and M6 (language models and agents).
  **Transition**: "Lesson 3.1. Features and the pipeline that carries them."

---

## Slide 6: Feature Engineering, ML Pipeline, and Feature Selection

**Time**: ~1 min
**Talking points**:

- Feature engineering is where domain expertise meets machine learning. The best model in the world cannot learn from bad features.
- Today we cover the full pipeline and three families of feature selection.
  **Transition**: "The ML pipeline is bigger than most beginners think. Let us map it."

---

## Slide 7: The ML Pipeline

**Time**: ~2 min
**Talking points**:

- Walk through the pipeline stages. Emphasise iteration. In practice, most time is spent on data and features, not model selection.
- The pipeline visual helps students see the whole picture before diving into details.
- The pipeline is a cycle, not a line. Bad evaluation sends you back to features. Production drift sends you back to retraining.
- If beginners look confused: "Imagine a factory. Raw materials come in, get cleaned, get shaped, then assembled, then quality-checked. ML is the same — and sometimes the quality check sends a product back to the start."
- If experts look bored: "The arrows matter. In MLOps language, every edge in this diagram is a place where contracts can break. We formalise those contracts in 3.7 with ModelSignature."
  **Transition**: "Before we engineer features, let me draw a line between what we are doing today and what we did in M2."

---

## Slide 8: Statistics vs Machine Learning

**Time**: ~2 min
**Talking points**:

- This distinction is critical. Many professionals confuse the two.
- Statistics asks "why?" — explain the past, coefficients, p-values, confidence intervals. ML asks "what next?" — predict the future, test error, cross-validation.
- Both use the same math but with different goals. M2 taught regression as a statistical model (explaining coefficients). M3 uses it as a prediction machine (minimising test error).
- Example: a central bank's stress-testing report uses statistical regression — it needs the coefficients. A ride-hailing ETA model uses ML regression — it needs the next prediction to be accurate.
- If beginners look confused: "A doctor explaining why a patient got sick uses statistics. A doctor predicting who will get sick next uses ML."
- If experts look bored: "Breiman's 2001 'Two Cultures' paper is the classic reference. The cultures are closer now than then, but the goals still diverge."
  **Transition**: "With ML framed as prediction, what matters most for prediction accuracy?"

---

## Slide 9: Data > Models > Hyperparameter Tuning

**Time**: ~2 min
**Talking points**:

- This hierarchy guides the entire module. We start with features (highest impact), then models, then tuning. The multipliers are rules of thumb, not measurements.
- Common mistake: spending days tuning XGBoost when a single well-engineered feature would have helped more than all tuning combined.
- The geocoding example is concrete: a street address is useless to a model, but lat/lon enables spatial features like distance to MRT and proximity to schools.
- Exercise 3.1 applies the same principle to ICU data: clinicians already know that heart rate divided by systolic blood pressure (the shock index) flags deterioration, so we compute it instead of hoping a model discovers the ratio.
- If beginners look confused: "Give a student a good textbook and an average teacher, they will learn more than a mediocre textbook and a great teacher."
- If experts look bored: "The data-centric AI movement formalises this — we will revisit it when we talk about label noise in M4."
  **Transition**: "So what does 'better features' actually mean? Let us taxonomise."

---

## Slide 10: Types of Engineered Features

**Time**: ~2 min
**Talking points**:

- Walk through each type with the HDB example. Temporal features are critical for time-dependent data. Interaction terms capture non-additive effects. Domain features require human expertise, which is why feature engineering is an art as much as a science.
- "For HDB, 'distance to nearest MRT' is a domain feature you have to build. No model finds it automatically."
- If beginners look confused: "A feature is just a column. We are making new columns by combining or transforming the old ones."
- If experts look bored: "Polynomial features are a linear model's way of chasing gradient boosting. We will see why trees do this more elegantly in 3.3."
  **Transition**: "Features can also be your worst enemy. Here is the silent killer."

---

## Slide 11: Feature Leakage — The Silent Killer

**Time**: ~2 min
**Talking points**:

- Leakage is the most common ML bug in production systems. A model with leakage looks like a breakthrough in testing and fails spectacularly live.
- Classic example: a loan default model that uses "collections department contacted" as a feature — that field only exists after default, not at application time.
- The prevention rule is simple: time-travel test every feature. At the moment of prediction, would this value be available?
- The leak screen is the mechanical backup: score each column on its own (single-feature ROC AUC). A legitimate credit feature rarely exceeds about 0.75; the planted `future_default_indicator` scores 0.99 because it is recorded after the outcome (93% of rows with the flag defaulted, versus 0.2% without).
- Exercise 3.4 runs this screen before modelling; `customer_id` is dropped too because a row ID is not a feature.
- If beginners look confused: "Imagine predicting tomorrow's weather using tomorrow's actual temperature. Of course it works. But it is cheating."
- If experts look bored: "Point-in-time correctness is what FeatureStore gives you for free. You will see that in two slides."
  **Transition**: "Once you have features, how do you decide which to keep?"

---

## Slide 12: Feature Selection: Three Families

**Time**: ~2 min
**Talking points**:

- Three families, three tradeoffs. Filters are fast but miss interactions. Wrappers find the best subset but cost N model trainings. Embedded methods are the practical default.
- Filter examples: mutual information, chi-squared, correlation thresholds. Wrapper examples: forward selection, backward elimination, RFE. Embedded examples: L1 sparsity in Lasso, tree-based importance.
- In the exercise, students apply all three and compare which features each selects. They will find little overlap, which is the key lesson.
- If beginners look confused: "Filter is like sorting groceries by price. Wrapper is like trying every recipe to see which tastes best. Embedded is a chef who knows the recipe picks the ingredients."
- If experts look bored: "RFE with cross-validation is still a reasonable practical default for mid-sized tabular problems. L1 tends to underperform when features are highly collinear."
  **Transition**: "Kailash packages the feature stage for you. Here is the bridge."

---

## Slide 13: Kailash Bridge: FeatureEngineer & FeatureStore

**Time**: ~2 min
**Talking points**:

- The Kailash bridge shows how the theory maps to the SDK, on the same ICU data as Exercise 3.1 (one row per admission, features computed from the first 24 hours; `long_stay` = stay longer than the median).
- FeatureEngineer has two calls. `generate()` builds candidate columns from the fields in a FeatureSchema (strategies: interactions, polynomial, binning, temporal); `select()` ranks originals plus candidates against the target and keeps `top_k` (default method: tree-based importance, so it is an embedded-style ranking).
- Fit it on training rows only: ranking features on all rows lets the test set vote on which features you keep, which is leakage.
- The engine does not replace the filter, wrapper and embedded comparisons: the exercise runs mutual information and chi-squared, RFE, and Lasso by hand so students see why they disagree.
- FeatureStore persists features through DataFlow; `get_features(schema, timestamp=T)` returns only rows stamped at or before T — that is point-in-time correctness.
- Note the two schema classes: the top-level `kailash_ml.FeatureSchema` (`features=[...]`) is what FeatureEngineer takes; the store needs `kailash_ml.features.FeatureSchema` (`fields=(...)`).
- The timestamp filter only protects you if the timestamps are honest: a feature computed later but stamped earlier still leaks.
  **Transition**: "That is Lesson 3.1. Here is the summary."

---

## Slide 14: Lesson 3.1 Summary

**Time**: ~1 min
**Talking points**:

- Quick recap before moving to Lesson 3.2.
- The exercise uses the ICU tables from Module 2 (patients, admissions, vitals, medications, labs). The prediction is made 24 hours after admission, so every event feature may only use records inside that window; the length of stay and discharge time are the label's source and are never features (a leakage audit raises if they slip in).
- Students build the features by hand in polars, rank them with mutual information and chi-squared, RFE around a random forest, and an L1 path, then validate the matrix against a FeatureSchema and log the consensus set to ExperimentTracker.
- Be honest about the dataset's window finding: only 198 of 416,526 vital-sign readings (4 admissions) fall in the first 24 hours, and medications and labs are just as sparse. The first-24h features carry almost no signal, so the selection methods rank mostly noise (test AUC ≈ 0.50, barely above a shuffled-target baseline). Exercise 3.1 reports this window audit as a data-quality finding — catching it before modelling is the whole point; in a real project you would go back and fix the data feed.
  **Transition**: "Now to the single most important theoretical idea in ML. Lesson 3.2: bias and variance."

---

## Slide 15: Bias-Variance, Regularisation, and Cross-Validation

**Time**: ~1 min
**Talking points**:

- The fundamental tradeoff in all of ML. This lesson gives the theoretical foundation for every model choice in the rest of the module.
- If students understand bias-variance, they understand why we need regularisation, cross-validation, and model comparison.
  **Transition**: "Let us start with the most intuitive explanation ever drawn."

---

## Slide 16: Bias and Variance — The Darts Analogy

**Time**: ~2 min
**Talking points**:

- The darts analogy is the most intuitive explanation. High bias = you are aiming wrong. High variance = you are shaky. The ideal is low both.
- Every model in our zoo has a different bias-variance profile, which is why model selection matters.
- If beginners look confused: "A calm but wrong shooter has high bias. A shaky but well-aimed shooter has high variance. ML asks you to train both your aim and your steadiness."
- If experts look bored: "The analogy extends to ensembles: bagging reduces variance without touching bias; boosting reduces bias while risking variance. 3.3 and 3.4 are the two sides of this coin."
  **Transition**: "Here is the same idea in math."

---

## Slide 17: Bias-Variance Decomposition

**Time**: ~3 min
**Talking points**:

- Full derivation: E[(y − ŷ)²] = Bias² + Variance + σ².
- Walk through the cancellation: cross terms vanish because noise is zero-mean and independent of the model, and the variance term is zero-mean by definition.
- For beginners: "We split total error into three parts: how wrong we are on average (bias), how much predictions wobble (variance), and noise we cannot fix."
- For experts: note this is for squared loss; the decomposition differs for other losses.
  **Transition**: "Let me summarise what each term means."

---

## Slide 18: The Three Terms

**Time**: ~2 min
**Talking points**:

- Bias² — systematic error, caused by the model being too simple. Variance — sensitivity to training data, caused by the model being too flexible. σ² — irreducible noise; no model fixes it.
- This table is the conceptual anchor for the rest of the module. Every model comparison comes back to this: where does it sit on the bias-variance spectrum?
- Regularisation is next because it is the primary tool for controlling this tradeoff.
  **Transition**: "So how do we control complexity? With a budget."

---

## Slide 19: Regularisation — Controlling Complexity

**Time**: ~3 min
**Talking points**:

- The geometry matters: L1's diamond has corners on the axes, which is why coefficients hit exactly zero. L2's circle has no corners, so coefficients shrink but never reach zero.
- Elastic Net is the practical default when you have correlated features — both sparsity and stability.
- If beginners look confused: "Think of regularisation as a budget. The model has a spending limit on coefficient size."
- If experts look bored: "For correlated features, lasso can be unstable — Elastic Net fixes that with the quadratic term. And ridge does not shrink uniformly: it shrinks low-variance principal directions more."
  **Transition**: "L2 has a surprising Bayesian interpretation that connects back to M2."

---

## Slide 20: Bayesian Interpretation of L2

**Time**: ~2 min
**Talking points**:

- This is an advanced slide for students who covered Bayesian thinking in M2.
- The key insight: regularisation is not arbitrary. It encodes a prior belief that most features have small effects. Ridge is a Gaussian prior; Lasso is a Laplace prior; λ = σ²/τ².
- For experts: mention that the full posterior gives uncertainty estimates, connecting to conformal prediction in Lesson 3.8.
- If beginners look confused: "The math here is optional. The takeaway is: regularisation is mathematically principled, not a hack."
  **Transition**: "Whether you use L1, L2, or Elastic Net, how do you know the model actually generalises? You cross-validate."

---

## Slide 21: Cross-Validation

**Time**: ~2 min
**Talking points**:

- Cross-validation is how we estimate real-world performance. The choice of CV strategy must match the data structure.
- k-fold: split into k folds, train on k−1, validate on 1, rotate. Stratified k-fold preserves class proportions — essential for imbalanced data.
- Time-series split is critical for financial data (very relevant for this audience). GroupKFold prevents data leakage from grouped observations (same patient, same company on both sides of a split).
- Nested CV is the gold standard but expensive; we cover it in detail in 3.7.
- "For financial professionals in the room: if you ever use k-fold on time-series data, you are leaking the future into the past. Always use walk-forward."
  **Transition**: "Here is how the Kailash preprocessing pipeline fits into cross-validation."

---

## Slide 22: Kailash Bridge: PreprocessingPipeline + Cross-Validation

**Time**: ~2 min
**Talking points**:

- This is the pattern Exercise 3.2 uses. The helper loads the Singapore credit data, drops the ID, outcome and leak columns, normalises features with PreprocessingPipeline and standardises the `savings_balance` target with training statistics, so an MSE of 1.0 means "no better than predicting the mean".
- Only 300 training rows on purpose: with about 33 correlated features ordinary least squares overfits, which is where Ridge and Lasso earn their keep.
- On this data cross-validation picks alpha = 100 (CV MSE about 0.77); the single test-set check comes out around 0.84, i.e. R² of roughly 0.16 — savings are hard to predict from the other credit fields, and that is an honest result.
- Kailash note: TrainingPipeline (Lessons 3.4 and 3.7) trains once and scores ONE validation split — a holdout, or only the first fold when you ask for kfold or stratified_kfold — and it does not re-fit preprocessing per fold. Full k-fold scores therefore come from sklearn's `cross_val_score`, as here.
- And one warning that applies everywhere in this module: PreprocessingPipeline's `setup()` learns from every row you pass it — hold the test rows out first, then fit on train only.
- If experts look bored: "RidgeCV computes the same leave-one-out choice in closed form."
  **Transition**: "Lesson 3.2 summary, then we jump into the zoo."

---

## Slide 23: Lesson 3.2 Summary

**Time**: ~1 min
**Talking points**:

- Quick recap. The exercise makes the theory tangible.
- A synthetic sine curve is used for the decomposition because only there do we know the true function, so bias and variance can be measured by refitting on many fresh training sets.
- Regularisation runs on a small credit sample (`savings_balance` target) where OLS visibly overfits.
- Cross-validation strategies use the data structure they are meant for: stratified folds for the rare default outcome, TimeSeriesSplit on admissions sorted by admit time, GroupKFold so a repeat patient never sits on both sides.
- Learning curves are read the standard way: a large gap between training and validation error means HIGH VARIANCE (more data or more regularisation helps); both errors converged and high means HIGH BIAS (add capacity).
  **Transition**: "Lesson 3.3. Meet every classical supervised model."

---

## Slide 24: The Complete Supervised Model Zoo

**Time**: ~1 min
**Talking points**:

- The No Free Lunch theorem says no model wins on all problems. This lesson teaches every major supervised algorithm so students can match the algorithm to the problem.
- We cover SVM, KNN, Naive Bayes, Decision Trees, and Random Forests. XGBoost and the boosting family get their own lesson (3.4).
  **Transition**: "Let us start with why the zoo exists at all."

---

## Slide 25: No Free Lunch Theorem

**Time**: ~2 min
**Talking points**:

- No Free Lunch is why this lesson exists. Averaged over all possible problems, every algorithm performs identically — no universal winner.
- Cite it correctly: the learning version is Wolpert (1996), "The Lack of A Priori Distinctions Between Learning Algorithms". The 1997 Wolpert and Macready paper is the separate NFL theorem for optimisation.
- We must know the full zoo to pick the right model for each problem.
- For experts: in practice, real-world data has structure that some algorithms exploit better than others, so some models do tend to win on tabular data (gradient boosting) and others on images (CNNs).
- If beginners look confused: "Different tools for different jobs. You would not use a hammer for a screw, even if both can hit things."
  **Transition**: "Let us meet the zoo. First: SVM."

---

## Slide 26: Support Vector Machines (SVM)

**Time**: ~3 min
**Talking points**:

- SVM is elegant mathematically but scales poorly to large datasets — O(n²) to O(n³) training.
- The kernel trick is the key insight: it lets SVM handle non-linear boundaries by implicitly working in high-dimensional space.
- C is the main hyperparameter to tune. Low C = wide margin, more tolerance. High C = narrow margin, zero tolerance.
- For beginners: "Imagine drawing a line between two groups of points. SVM finds the line with the widest buffer zone."
- If experts look bored: "The dual formulation and KKT conditions give you support vectors as the only points that matter — a strong sparsity property."
  **Transition**: "Next: the simplest algorithm imaginable."

---

## Slide 27: K-Nearest Neighbors (KNN)

**Time**: ~2 min
**Talking points**:

- KNN is the simplest algorithm: no training, just memorise. That is also its weakness: slow at prediction time, and the curse of dimensionality makes it unreliable in high dimensions.
- Good for small datasets and interpretable boundaries.
- For beginners: "Tell me who your friends are, and I will tell you who you are."
- If experts look bored: "KNN is asymptotically optimal as n goes to infinity with k growing properly, but the convergence rate is terrible in high dimensions. Local methods lose to global methods there."
  **Transition**: "Now the classical Bayesian classifier."

---

## Slide 28: Naive Bayes

**Time**: ~2 min
**Talking points**:

- Connects to M2's Bayesian thinking: prior × likelihood / evidence.
- The naive assumption is almost always wrong (features are correlated), but the classifier still works well because it only needs to get the ranking right, not the exact probabilities.
- For text: each word is treated as independent, which is wrong but effective. Good baseline before trying complex models.
- Three variants worth naming: GaussianNB (continuous), MultinomialNB (counts), BernoulliNB (binary).
- If experts look bored: "For calibrated probabilities you need Platt or isotonic on top — we cover that in 3.5."
  **Transition**: "Now the most interpretable model: trees."

---

## Slide 29: Decision Trees

**Time**: ~3 min
**Talking points**:

- Trees are the most interpretable model: you can draw the tree and explain every decision. For regulated industries, this is valuable.
- Gini and entropy give nearly identical splits in practice; Gini is slightly faster.
- Pruning is essential: pre-pruning (max_depth, min_samples) and post-pruning (cost-complexity) both appear in the exercise.
- The weakness: a single tree overfits badly. That is why we need forests (next slide) and boosting (Lesson 3.4).
- If beginners look confused: "A tree is a flowchart. 'Is income above 80k? Yes → is debt ratio below 30%?' and so on."
- If experts look bored: "CART, ID3, C4.5 are historical. In practice, sklearn's implementation covers Gini and entropy. The real innovation today is how trees are combined — bagging and boosting."
  **Transition**: "And here is how we tame the variance of trees: vote them."

---

## Slide 30: Random Forests

**Time**: ~2 min
**Talking points**:

- Random Forest is the "Swiss army knife" of ML. It works well on almost everything with minimal tuning.
- Two sources of randomness decorrelate the trees: bootstrap sampling (bagging) and feature subsampling at each split.
- The OOB trick is elegant: about 36.8% of samples are left out of each bootstrap ((1 − 1/n)ⁿ → 1/e), so each tree has its own test set for free.
- The connection to bias-variance is critical: this is how ensembles overcome the single-tree overfitting problem — bagging reduces variance without touching bias.
- If beginners look confused: "Ask 500 slightly biased experts and take the vote. Their individual errors cancel out."
  **Transition**: "So we have 5 model families. How do we compare them?"

---

## Slide 31: Model Comparison Framework

**Time**: ~2 min
**Talking points**:

- This table is the "cheat sheet" students will reference. Emphasise that the right model depends on the use case, not just accuracy.
- A regulator wants interpretability (tree). A Kaggle competitor wants accuracy (boosting). A spam filter wants speed (Naive Bayes).
- Ask the room: "If you were building the AML model from Lesson 3.5, which model family would you start with, and why?"
  **Transition**: "To compare 5 models fairly, you need identical splits and preprocessing. Kailash automates that."

---

## Slide 32: Kailash Bridge: AutoMLEngine

**Time**: ~2 min
**Talking points**:

- AutoMLEngine does not choose models or preprocess for you: `trial_fn` trains and scores, and the engine supplies the sweep strategy, a time and cost budget, and an audit record of every trial.
- The value is consistency: every family is scored by the same `cv_scores` helper on the same stratified folds of the same e-commerce churn split, so differences are the models, not the data.
- With these default settings on this data, cross-validated ROC AUC lands around 0.74 to 0.79 for all five families, and Naive Bayes is not last. That is No Free Lunch in action.
- Exercise 3.3 does the comparison by hand and goes further: it tunes each family inside nested CV before comparing, so no model is judged with hand-picked settings.
- Always put the do-nothing row in the table: an "always churn" model scores accuracy 0.745 and churn-class F1 0.854 on the test split with ROC AUC 0.5, because churners are the majority (about 74%). Accuracy and F1 flatter every model; ROC AUC, which the baseline cannot game, is the selection metric.
- "Do not use AutoMLEngine as a black box. Use it to build the leaderboard, then dig into the top 2 or 3 manually."
  **Transition**: "Lesson 3.3 summary."

---

## Slide 33: Lesson 3.3 Summary

**Time**: ~1 min
**Talking points**:

- Quick recap. The exercise forces students to use consistent evaluation (same CV splits) and justify their model selection with data evidence, not opinion.
- One data note worth telling the class: the dataset marks a customer churned exactly when `days_since_last_order` exceeds 180, so that column defines the label and is removed from the features. With it, a depth-1 tree scores 100% — which is leakage, not learning.
  **Transition**: "Lesson 3.4. The dominant algorithm for tabular data."

---

## Slide 34: Gradient Boosting Deep Dive

**Time**: ~1 min
**Talking points**:

- Gradient boosting is the most important family for tabular data. XGBoost, LightGBM, and CatBoost dominate Kaggle and production systems.
- This lesson goes deep into the math of how and why they work.
- For professionals: if your tabular prediction problem is not solved by gradient boosting, you should seriously question whether the problem is solvable at all.
  **Transition**: "Start with the conceptual distinction: bagging versus boosting."

---

## Slide 35: Bagging vs Boosting

**Time**: ~2 min
**Talking points**:

- This is the conceptual difference that students must understand. Bagging and boosting are complementary ensemble strategies.
- Bagging (Random Forest): train many independent models in parallel on bootstrap samples, then average. Reduces variance. Boosting (XGBoost): train models sequentially, each correcting the errors of the previous. Reduces bias.
- For beginners: "Bagging asks many experts and takes a vote. Boosting asks one expert, then asks a specialist to fix the mistakes, then another specialist to fix the remaining mistakes."
- If experts look bored: "Bagging's variance-reduction comes from averaging decorrelated estimators. Boosting's bias reduction comes from descending the empirical loss. Different objectives, different guarantees."
  **Transition**: "Historical warmup: AdaBoost."

---

## Slide 36: AdaBoost — Conceptual Warmup

**Time**: ~2 min
**Talking points**:

- AdaBoost (Freund and Schapire, 1996) is the warmup. Reweight misclassified samples so the next model focuses on the hard ones.
- Students understand the "focus on mistakes" idea before we add gradient descent.
- In practice, nobody uses AdaBoost anymore, but the intuition transfers directly to XGBoost.
  **Transition**: "XGBoost formalises the idea with gradients and Newton steps."

---

## Slide 37: XGBoost — The Math

**Time**: ~3 min
**Talking points**:

- Classic gradient boosting (Friedman's original, sklearn's GradientBoosting) is first-order: each tree fits the negative gradient. XGBoost uses the Hessian too — and so do LightGBM and CatBoost.
- The Hessian gives curvature, so each leaf takes a Newton step w* = −G/(H+λ) instead of a fixed-size gradient step.
- The regulariser Ω charges γ per leaf and λ for large leaf weights: Ω(f) = γT + ½λΣw². Substituting w* back gives the best achievable loss for a given tree structure; the split gain on the next slide is that quantity before minus after the split.
- For beginners: "XGBoost uses both the slope and the curvature to find the best correction."
- For experts: note this is why XGBoost supports arbitrary differentiable loss functions — you supply the gradient and Hessian. Exercise 3.5's focal loss does exactly this.
  **Transition**: "The clever consequence is the split gain formula."

---

## Slide 38: XGBoost Split Gain Formula

**Time**: ~3 min
**Talking points**:

- Gain = ½ · [G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] − γ.
- This is the formula students need to understand, not memorise.
- Walk through the formula term by term: left child score + right child score − parent score − complexity penalty.
- The key insight: the gain formula naturally incorporates regularisation. Lambda penalises extreme leaf weights. Gamma penalises splits that do not improve the objective enough. Together they prevent overfitting without a separate pruning pass.
- If beginners look confused: "The math is advanced. The takeaway: XGBoost has regularisation baked into its split decisions. That is why it rarely overfits."
- If experts look bored: "G and H come from the task-specific loss, so the split decision is task-aware. Classification gets different splits than regression even on the same data."
  **Transition**: "Two other boosting flavours you should know."

---

## Slide 39: LightGBM and CatBoost

**Time**: ~3 min
**Talking points**:

- LightGBM is typically fastest: histogram-based split finding, leaf-wise growth, GOSS (Gradient-based One-Side Sampling).
- CatBoost handles categorical features best: ordered boosting prevents target leakage in categorical encoding, and categoricals are native — no manual encoding required.
- Note the categorical story across all three: CatBoost is native; LightGBM is native via `categorical_feature`; XGBoost supports it with `enable_categorical=True` or encoding. The slide's comparison table reflects this.
- GOSS: keep the top-a% gradient samples, then sample b × N rows from the remaining (1 − a) × N small-gradient rows. A stochastic speedup without losing much accuracy.
- In practice: try all three and compare. The exercise has students do exactly this on credit scoring data — and Exercise 3.4 uses CatBoost's native categoricals, the advantage the text describes.
- Example: for a dataset with 'HDB town', 'flat model', 'storey range' all as categoricals, CatBoost often saves significant preprocessing time.
- If beginners look confused: "They are cousins. Same idea, different optimisations. Pick one and be consistent."
- If experts look bored: "GOSS bias-corrects for the subsampling. Ordered boosting prevents the 'prediction shift' problem in target encoding. Real innovations, not just speedups."
  **Transition**: "Side by side comparison."

---

## Slide 40: Boosting Family Comparison

**Time**: ~2 min
**Talking points**:

- This table is the quick reference. In Kaggle competitions, LightGBM and XGBoost alternate as winners. CatBoost shines when categorical features dominate.
- All three are excellent; the differences are marginal on most problems.
- "Your choice is less about accuracy than about infrastructure. Which one does your production team already use?"
  **Transition**: "Kailash gives you a unified interface for all three."

---

## Slide 41: Kailash Bridge: Gradient Boosting

**Time**: ~1 min
**Talking points**:

- TrainingPipeline gives one `train()` call for all three libraries: a ModelSpec names the class, `train()` holds out 20% of the dev frame (a single holdout, not cross-validation), scores it and registers the model.
- On the leak-free credit data the three land within half a point of each other: holdout ROC AUC about 0.79 each. That is the honest number; with the planted `future_default_indicator` column left in, AUC jumps to about 0.99 and XGBoost puts most of its importance on that one column.
- The engine's own evaluation only computes metrics from labels plus ROC AUC; AUC-PR, the primary metric for 13%-positive data, is computed from `predict_proba` on the test frame.
- Exercise 3.4 calls the three libraries directly (it needs early stopping and native CatBoost categoricals) on the same leak-free split.
  **Transition**: "Lesson 3.4 summary."

---

## Slide 42: Lesson 3.4 Summary

**Time**: ~1 min
**Talking points**:

- Recap: bagging vs boosting, AdaBoost warmup, XGBoost math, split gain, LightGBM and CatBoost, TrainingPipeline.
- Transition to evaluation. Now that students can train every model, they need to know how to evaluate honestly. That is Lesson 3.5.
- "We now have a lot of models. The next question: how do we know any of them are actually good?"
  **Transition**: "Lesson 3.5 — where accuracy goes to die."

---

## Slide 43: Model Evaluation, Imbalance, and Calibration

**Time**: ~1 min
**Talking points**:

- This lesson is where theory meets reality. Students learn why accuracy can lie, how to handle imbalanced data, and how to calibrate model confidence.
- The running example is an illustrative anti-money-laundering (AML) model, introduced on the next slide: 99.999% accuracy and no use at all, because the wrong metric was used.
  **Transition**: "Here is the case."

---

## Slide 44: The 99.999% Accurate AML Model

**Time**: ~2 min
**Talking points**:

- Introduce the case; it is a teaching illustration, not a real bank's model.
- Walk the class through the math: if 0.001% of transactions are truly suspicious, a "predict all negative" model is 99.999% accurate. Completely useless. It missed every money laundering case, and the dashboard said it was perfect.
- Students now have the vocabulary (from Lessons 3.2-3.4) to understand why accuracy failed. We need metrics that penalise missing the positives.
- If beginners look confused: "If you predicted 'no rain' every day in Singapore, you would be right most of the time. That does not make you a useful weather forecaster."
- If experts look bored: "This is the imbalanced-class failure mode. We formalise it with precision, recall, and PR-AUC next."
  **Transition**: "The full metrics taxonomy."

---

## Slide 45: Classification Metrics

**Time**: ~3 min
**Talking points**:

- Walk through each metric with the AML example. Precision: of the flagged transactions, how many are really suspicious? TP / (TP + FP). Recall: of all suspicious transactions, how many did we catch? TP / (TP + FN). F1 balances the two.
- ROC-AUC measures overall ranking quality across all thresholds.
- Precision-Recall curve and PR-AUC: more informative than ROC for imbalanced data — the rule of thumb is to prefer PR whenever the positive class is under about 20%.
- Log loss: probability-based metric; penalises confident wrong predictions.
- For the AML case, recall matters most: missing a money laundering case has catastrophic regulatory consequences.
- If beginners look confused: "Precision is how often your alarm is right. Recall is how often the alarm actually goes off when it should."
- If experts look bored: "Log loss is the only metric here that is a proper scoring rule — which we return to under calibration."
  **Transition**: "Regression has its own set of metrics."

---

## Slide 46: Regression Metrics

**Time**: ~2 min
**Talking points**:

- R-squared from M2 now has company. RMSE is the most common. MAE is more robust. MAPE is easiest to explain to non-technical stakeholders ("our predictions are off by 5% on average").
- For HDB: RMSE in dollars, MAPE in percent. RMSE tells you the spread; MAPE tells business what to expect.
- MAPE breaks when actuals are near zero.
- If experts look bored: "MAPE is asymmetric — a 50% over-prediction costs more than a 50% under-prediction. Use sMAPE or MASE if you care."
  **Transition**: "So how do you pick a metric?"

---

## Slide 47: Choosing the Right Metric

**Time**: ~2 min
**Talking points**:

- The key message: there is no default metric. The right metric depends on the business cost of each type of error.
- Ask students: "In the AML case, what is the cost of a false negative? What is the cost of a false positive?" The asymmetry drives the metric choice.
- False negative: a missed laundering case → regulatory fine in the millions. False positive: an investigator opens a case on a clean transaction → a few hundred dollars of staff time. That ratio drives you to weight recall extremely heavily.
- Generalise: every business problem has a cost matrix. Translate it into the metric.
- If experts look bored: "You can go further and train with a custom loss encoding the cost matrix directly. XGBoost supports this via weighted samples."
  **Transition**: "Imbalance is the scenario where metric choice matters most. Let us look at the techniques."

---

## Slide 48: Handling Class Imbalance

**Time**: ~2 min
**Talking points**:

- SMOTE is the most famous approach but has serious limitations. For professionals: class weights are the practical default.
- For the AML case: cost-sensitive learning with high weight on false negatives is the right approach. SMOTE generates fake transactions, which is dangerous in a regulated context — you cannot fabricate money laundering transactions and present them to a regulator.
- Undersampling the majority: cheap, throws away data; use when the majority is so large that sampling is necessary.
- One honest note from the exercise: cost-sensitive weighting improves ranking for the minority class but distorts probabilities — the weighted model over-predicts positives and its Brier score gets worse. That is why calibration (two slides on) follows.
- If beginners look confused: "Imbalance means the model learns to always predict the majority because that is safe. Fix this by telling it: the minority matters more."
- If experts look bored: "SMOTE's boundary-region failure mode is well documented. ADASYN and BorderlineSMOTE try to fix it. Class weights still outperform in most benchmarks."
  **Transition**: "One more tool: focal loss."

---

## Slide 49: Focal Loss

**Time**: ~2 min
**Talking points**:

- Focal loss was invented for object detection (Lin et al. 2017, RetinaNet) where most anchor boxes are background — extreme imbalance.
- Formula: FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t). The (1 − p_t)^γ factor down-weights easy examples. If the model is confident and correct, that example contributes almost nothing to the loss.
- The insight transfers perfectly to tabular imbalance: most examples are easy majority class, and the model wastes capacity on them.
- For beginners: "Focal loss tells the model: stop celebrating getting the easy cases right. Focus on the hard ones."
- For experts: gamma = 2 is standard; alpha handles the class weight. Exercise 3.5 implements focal loss as a real LightGBM custom objective (gradient and Hessian), and the γ = 0 run must reproduce the plain baseline — that is how the code is tested. Note that a constant class weight cannot replicate this: reweighting is per class, focal loss is per example.
  **Transition**: "Even with the right metric and imbalance handled, probabilities need to be trustworthy."

---

## Slide 50: Probability Calibration

**Time**: ~2 min
**Talking points**:

- Calibration is critical for professionals. A lender that uses model probabilities to set prices needs those probabilities to be correct, not just well-ranked.
- The reliability diagram plots predicted probability against observed frequency; a calibrated model follows the diagonal. Points above the diagonal mean the predicted probabilities are too low; below means too high. Over-confidence shows as above the diagonal in low bins and below in high bins.
- Brier score is not a pure calibration measure: a perfectly calibrated model that always predicts the base rate of 0.3 scores 0.21. Brier decomposes into reliability − resolution + uncertainty; use the reliability diagram or ECE for calibration specifically.
- Platt scaling ("sigmoid") and isotonic regression are both fitted on data the model did not train on: Exercise 3.5 holds out 20% and calls `TrainingPipeline.calibrate`. Isotonic is monotone by construction — it corrects any monotone distortion, not just sigmoid-shaped ones.
- The cost threshold t* = c_FP/(c_FP + c_FN) assumes calibrated probabilities. With the course's costs (a missed default, FN, costs S$10,000; a wrongly declined good applicant, FP, costs S$1,500) t* = 1,500 / 11,500 = 0.130.
- The order is: train the class-weighted booster, recalibrate it on a held-out 20% of the training rows, then apply t* to the calibrated probabilities. This is not double counting: class weights inflate the raw scores, and recalibration maps them back to true default rates, removing the weight's shift before the cost ratio is applied. What WOULD double count is applying t* to the raw weighted scores.
- Check the chosen threshold on out-of-fold predictions and report it once on the test set.
- If beginners look confused: "A weather forecaster who says '80% rain' should be right 80% of the time. If they are only right 50% of the time, they are miscalibrated."
- If experts look bored: "Boosted trees are famously uncalibrated out of the box — the classic sigmoid distortion. Always calibrate post-hoc for decision use."
  **Transition**: "One more evaluation topic before the summary: combining models."

---

## Slide 51: Stacking and Blending

**Time**: ~2 min
**Talking points**:

- Spec 3.5 asks for a brief look at combining models; M4 uses EnsembleEngine to blend anomaly scores.
- `EnsembleEngine.blend` and `.stack` take a polars frame, split it 80/20 themselves, fit the ensemble on the 80% and report metrics on the 20%. They also score each base model on that 20% for comparison, which is why the base models must already be fitted, and fitted on DIFFERENT rows (`fit_part`) so the comparison is not on rows they memorised.
- On the leak-free credit data the result is instructive: ROC AUC is about 0.79 for the stack, about 0.79 for the blend, 0.79 for the random forest, 0.79 for LightGBM and 0.73 for Naive Bayes. Stacking matched the best single model rather than beating it, because the strong base models agree.
- Stacking is also slow: it refits every base model once per fold.
- If experts bored: "The meta-model must be trained on out-of-fold predictions; training it on in-sample predictions rewards whichever base model overfits most."
  **Transition**: "Lesson 3.5 summary."

---

## Slide 52: Lesson 3.5 Summary

**Time**: ~1 min
**Talking points**:

- Quick recap. The exercise forces students to confront imbalanced data and discover why accuracy lies.
- Focal loss is implemented as a true custom objective (gradient and Hessian) and the γ = 0 run must reproduce the plain baseline, which is how the code is tested.
- The threshold is tuned on out-of-fold predictions and reported once on the test set.
- Calibration is fitted on a held-out 20% split, and students read the reliability diagram rather than trusting the Brier score alone.
- Cost-sensitive weighting improves ranking for the minority class but distorts probabilities; calibration is what repairs them.
  **Transition**: "Lesson 3.6: not just accurate, but explainable and fair."

---

## Slide 53: Interpretability and Fairness

**Time**: ~1 min
**Talking points**:

- Interpretability is not optional in regulated industries (finance, healthcare). This lesson covers SHAP, LIME, and fairness metrics.
- For professionals: explainability is increasingly expected. The EU AI Act sets binding requirements for high-risk systems; Singapore's FEAT principles are supervisory guidance that calls for fairness and transparency assessment, not a binding rule.
  **Transition**: "Start with the 'why'."

---

## Slide 54: Why Interpretability Matters

**Time**: ~2 min
**Talking points**:

- For a professional audience, the regulatory angle is the strongest motivator. The FEAT principles are directly relevant for Singapore-based professionals; they are principles, not prescriptive rules, so they do not mandate a particular method such as SHAP.
- Three reasons: regulation, debugging, and trust. Debugging: SHAP values have caught data leakage bugs that no other technique revealed — if one feature dominates the explanation unexpectedly, that is a red flag. Trust: a loan officer needs to tell the applicant why their loan was denied.
- If beginners look confused: "If the model cannot explain itself, the model cannot ship in a regulated setting. Period."
- If experts look bored: "Shapley values are provably unique under the Shapley axioms, which matters legally — no competing attribution method has equivalent guarantees."
  **Transition**: "SHAP is the modern gold standard."

---

## Slide 55: SHAP — Shapley Additive Explanations

**Time**: ~3 min
**Talking points**:

- The Shapley value is the only attribution method that satisfies all four axioms simultaneously: efficiency, symmetry, dummy, and linearity. This is a theorem from game theory (Shapley 1953); SHAP (Lundberg and Lee, 2017) applies it to feature attribution.
- The formula looks scary but the intuition is simple: for every possible team of features, measure how much feature i improves the prediction when added. Average over all possible teams, weighted by how many ways each team can be formed.
- For beginners: "Shapley values are like splitting a restaurant bill fairly based on what each person ordered."
- For experts: "The weighted sum is the marginal contribution of feature i averaged over all 2^N feature coalitions. It is the unique solution satisfying the four axioms."
  **Transition**: "Computing Shapley values exactly is exponential. SHAP has clever shortcuts."

---

## Slide 56: SHAP Variants and Plots

**Time**: ~2 min
**Talking points**:

- TreeSHAP is the practical choice for our model zoo (most are tree-based) — exact computation in polynomial time. KernelSHAP when you have a non-tree model — model-agnostic, slower.
- Four standard plots: summary (global importance), dependence (one feature across its range), waterfall (one prediction, one feature at a time), force (interactive waterfall).
- The summary plot is the most useful for global understanding. The waterfall plot is the most useful for explaining individual predictions to stakeholders.
- One practical note for the exercise: for LightGBM the SHAP values are in log-odds, so the additivity check compares base value plus SHAP sum against the raw (log-odds) prediction, not the probability. The waterfall plot is illustrative in log-odds units.
  **Transition**: "SHAP has a cousin: LIME."

---

## Slide 57: LIME — Local Interpretable Explanations

**Time**: ~2 min
**Talking points**:

- LIME (Ribeiro, Singh, Guestrin, 2016) is complementary to SHAP. It answers the same question ("why this prediction?") but with a different approach: perturb the input, fit a local linear model to the perturbed predictions.
- The key difference: SHAP gives globally consistent attributions. LIME can give different explanations for similar instances.
- For regulatory compliance, SHAP is preferred because of its theoretical guarantees. LIME still has a role for quick debugging.
- If beginners look confused: "LIME is like asking: if I wiggle this input a bit, which direction does the model care about most?"
- If experts look bored: "LIME's local-linear assumption can mislead on strongly non-linear models. SHAP does not share this weakness because of its game-theoretic grounding."
  **Transition**: "Interpretability is half the story. The other half is fairness."

---

## Slide 58: Fairness Metrics

**Time**: ~2 min
**Talking points**:

- These metrics quantify fairness. Disparate impact measures whether the model selects groups at similar rates (the four-fifths rule of thumb comes from the US EEOC Uniform Guidelines on employment selection). Equalized odds checks whether error rates are balanced. Calibration parity asks whether predicted probabilities mean the same thing for different groups.
- For the credit scoring example: does the model deny loans to one demographic at a higher rate than another? Is a 70% predicted probability equally accurate across ethnicities?
- In the exercise the audit is computed for race, gender and age bands — the dataset's actual protected attributes — not for synthetic groups.
- If beginners look confused: "Fairness asks: does the model treat every group equally? And there are several ways to measure 'equally'."
- If experts look bored: "Impossibility results tell us these criteria cannot all hold simultaneously except in trivial cases. We hit that next."
  **Transition**: "Here is the uncomfortable theorem."

---

## Slide 59: The Fairness Impossibility Theorem

**Time**: ~2 min
**Talking points**:

- This is one of the most important results in ML fairness (Chouldechova 2017; Kleinberg et al. 2016).
- Do not teach it as "pick two": with unequal base rates the criteria are pairwise incompatible except for perfect prediction. Parity plus equalized odds forces an uninformative classifier; parity plus calibration forces equal base rates; equalized odds plus calibration forces perfect prediction.
- Exercise 3.6 works the theorem with this model's real base rates. It means every deployed model makes a fairness tradeoff, whether the team acknowledges it or not.
- The responsible approach: choose the criterion that matches the use case, measure it, document it, and justify the choice to stakeholders. For credit, calibration parity is usually the priority (the probability must mean the same thing everywhere); for hiring, equalized odds often matters more.
- If beginners look confused: "There is no perfect fairness. You must pick which kind of unfairness you can live with — and justify it."
- If experts look bored: "The proof uses the confusion matrix decomposition. If base rates differ and TPR/FPR are equal across groups, calibration cannot hold. Elegant and troubling."
  **Transition**: "One more interpretability tool for the advanced students."

---

## Slide 60: ALE — Accumulated Local Effects

**Time**: ~2 min
**Talking points**:

- ALE is a more sophisticated alternative to partial dependence plots. For experts only.
- The key insight: when floor area and number of rooms are correlated, PDP evaluates the model at impossible combinations (large area but 1 room). ALE avoids this by using conditional differences — it only evaluates the model on combinations that actually occur in the data.
- Use when features are correlated. Default to PDP when they are not.
- "For HDB data, floor area and number of rooms are obviously correlated. ALE is the right choice."
  **Transition**: "Lesson 3.6 summary."

---

## Slide 61: Lesson 3.6 Summary

**Time**: ~1 min
**Talking points**:

- Quick recap. The exercise forces students to move from "the model is accurate" to "the model is accurate, interpretable, and fair."
- Two details to stress. SHAP values for LightGBM are in log-odds, so the additivity check compares base value plus SHAP sum against the raw (log-odds) output, not the probability.
- The fairness audit's four-fifths result: race (ratio about 0.94) and gender (about 0.96) pass, but the age bands fail badly (about 0.02), because the model's approvals concentrate in some age bands.
- A failing ratio is not a code bug to hide: it is routed to a human risk-committee decision, with the impossibility theorem explaining why no threshold fixes every criterion at once.
  **Transition**: "We now have the full science half. The next two lessons are pure engineering."

---

## Slide 62: Workflow Orchestration, Model Registry, and Hyperparameter Search

**Time**: ~1 min
**Talking points**:

- Now we move from ML science to ML engineering. Lessons 3.1-3.6 taught what to do. Lessons 3.7-3.8 teach how to automate and productionise it.
- WorkflowBuilder orchestrates the entire pipeline. ModelRegistry tracks versions. HyperparameterSearch finds the best configuration.
- For professionals: this is the layer that separates "ML experiments" from "ML systems that ship."
  **Transition**: "Start with the orchestrator."

---

## Slide 63: WorkflowBuilder — Node-Based Pipelines

**Time**: ~2 min
**Talking points**:

- WorkflowBuilder is from the core Kailash SDK (not kailash-ml). `add_node` takes the node type, a node id and its config; `add_connection` wires a named output to a named input.
- LocalRuntime validates the DAG, runs nodes in dependency order and returns the results per node plus a run id for the audit trail; if a node fails, execute raises — there is no silent fallback.
- PythonCodeNode runs a few inline lines: its inputs arrive as variables and whatever you assign to `result` is its output. SwitchNode routes `input_data` to `true_output` or `false_output`; with `skip_branches`, only the taken branch runs.
- On the credit data the default rate is about 13%, so the gate routes to "go".
- Node outputs must be JSON-serialisable, so Exercise 3.7 passes parquet paths and registry name plus version between nodes, not DataFrames or model objects.
- If beginners look confused: "A workflow is a recipe written in boxes and arrows. Each box does one thing. The arrows say what goes where."
- If experts look bored: "Same mental model as Airflow DAGs — nodes, edges, a scheduler, and a run record."
  **Transition**: "Nodes can be built-in or custom."

---

## Slide 64: Custom Nodes

**Time**: ~2 min
**Talking points**:

- A custom node is a class decorated with `@register_node()` so WorkflowBuilder can find it by its class name.
- `get_parameters` declares the inputs the node accepts (name, type, required, default); `run` receives them as keyword arguments and returns a dict whose keys are the outputs other nodes can connect to.
- Raise on bad input: a raised error stops the workflow and surfaces in the run — that is the point.
- Nodes that call async engines (TrainingPipeline, ModelRegistry) subclass AsyncNode and implement `async_run` instead.
- In Exercise 3.7 the DAG is load, preprocess, train (TrainingPipeline), evaluate on the untouched test frame, then a PythonCodeNode quality gate and a SwitchNode that either promotes the model or holds it in staging.
  **Transition**: "One of those nodes is hyperparameter search."

---

## Slide 65: Bayesian Hyperparameter Search

**Time**: ~2 min
**Talking points**:

- HyperparameterSearch drives TrainingPipeline: each trial is one `train()` call on the dev frame, scored on a holdout carved from dev.
- The space is a list of ParamDistribution objects: `int_uniform`, `uniform`, `log_uniform` (use it for learning rates) and `categorical` with choices.
- `strategy="bayesian"` uses Optuna's TPE sampler; the other strategies are grid, random and successive_halving.
- There is no cross-validation inside the search: each trial scores one holdout, so the winner is re-checked once on the untouched test frame.
- Bayesian search usually finds good settings in fewer trials than random search because it concentrates trials where past trials scored well; how much it helps depends on the problem.
- Exercise 3.7 compares 20 Bayesian trials with a 4-point grid on the same dev frame and validation rows.
- Connects to nested CV: the inner loop of nested CV is hyperparameter search.
- If beginners look confused: "Bayesian search is like a chess player who remembers every move they tried and only considers the promising ones next."
  **Transition**: "Once you find the best model, you must track it."

---

## Slide 66: Model Registry — Versioning and Lifecycle

**Time**: ~2 min
**Talking points**:

- ModelRegistry tracks every model version with its signature, metrics, and lifecycle stage.
- `register_model` stores the artefact bytes, MetricSpec rows (name, value, split, higher_is_better) and a ModelSignature: the input FeatureSchema plus the output columns and dtypes — the contract a serving layer can check requests against.
- TrainingPipeline already registers each model it trains with an automatic one-column signature; Exercise 3.7 registers a production entry with the richer contract.
- The registry's stages are staging, shadow, production and archived, and only some moves are allowed: promoting a new version to production archives the previous production version automatically, and an archived version cannot jump straight back to production; it goes archived → staging → production. That route is the rollback Exercise 3.8 rehearses.
- There is no `registry.promote`; the call is `promote_model`.
- "The registry is your audit trail. Every deployed model has a unique version, a signature, and a provenance chain all the way back to the training code. Model-risk reviewers expect exactly this kind of versioning for models used in regulated decisions."
  **Transition**: "The lifecycle is more than technical."

---

## Slide 67: Model Lifecycle

**Time**: ~2 min
**Talking points**:

- The lifecycle is not just technical; it includes governance gates. Promotion from staging to production requires human approval.
- This is where ML engineering meets organisational process. For professionals: this maps to how banks and insurers manage model risk — staging is where validation and independent review happen; production is where the model affects real decisions.
- Retirement matters too: an archived model still exists in the registry with its history — so you can reproduce any past decision for audit.
- If beginners look confused: "It is like publishing a book. Draft, edit, publish, eventually retire. Each stage has its own rules."
- If experts look bored: "The governance gate is the non-optional piece. Kailash's PACT framework in M6 extends this with D/T/R accountability grammar."
  **Transition**: "One last advanced topic: unbiased model selection."

---

## Slide 68: Nested Cross-Validation — Unbiased Model Selection

**Time**: ~2 min
**Talking points**:

- Nested CV is the advanced technique that separates rigorous evaluation from naive evaluation.
- Outer loop: k-fold for performance estimation. Inner loop (inside each outer fold): k-fold for hyperparameter search. If you tune on the same CV you use to estimate performance, you get an optimistically biased estimate.
- Cost accounting, so the class can budget: with 5 outer folds, 5 inner folds and K hyperparameter candidates, total fits = 5 × (5K + 1) — the +1 is the final refit on each outer fold's full training partition.
- Many published papers use non-nested CV, which gives optimistically biased performance estimates. Do not trust a model comparison that was not nested.
- For professionals: when comparing models for production, always use nested CV. Expensive but honest.
- If beginners look confused: "Think of it as having two separate exam rooms. One for practice, one for the real grade. Never use the same exam for both."
- If experts look bored: "The variance of nested CV is still nontrivial. Repeated nested CV reduces it further. Expensive but gold-standard."
  **Transition**: "Lesson 3.7 summary."

---

## Slide 69: Lesson 3.7 Summary

**Time**: ~1 min
**Talking points**:

- Quick recap: WorkflowBuilder with real nodes, Bayesian search, the registry's four-stage lifecycle, nested CV.
- The exercise brings everything together into an automated pipeline. Students experience the engineering side of ML: not just training a model but building a system that trains, evaluates, and registers models automatically.
- "One lesson to go. The final piece: drift, DataFlow, and deployment."
  **Transition**: "Lesson 3.8. Ship it."

---

## Slide 70: Production Pipeline — DataFlow, Drift, and Deployment

**Time**: ~1 min
**Talking points**:

- The final lesson. Everything from 3.1-3.7 was building toward this: a production ML system.
- We cover DataFlow for persistence, DriftMonitor for monitoring, model cards for documentation, and conformal prediction for uncertainty.
- "By the end of this lesson, you will have a complete mental model of what ships when a lender deploys a credit scoring model."
  **Transition**: "First, where do predictions go?"

---

## Slide 71: DataFlow — Database Persistence

**Time**: ~2 min
**Talking points**:

- DataFlow handles the "where do results go?" question. In production, evaluations and predictions must be stored for audit, monitoring, and business use.
- The plain class annotations ARE the schema (there is no `field()` helper); `id: int` becomes the auto-generated primary key, and `created_at` / `updated_at` are added for you.
- `db.express` takes the model name as a string: `create`, `read` (by id), `find_one` (by filter), `list`, `update`, `delete` and `count`.
- Pass in the metrics you actually measured; Exercise 3.7 writes TrainingPipeline's test-set metrics.
- Zero-config: SQLite for local dev, PostgreSQL for production, same API.
- For beginners: "DataFlow is how the model's predictions get saved to a database so other systems can use them. You write the schema once, and the database operations are free."
- For experts: "DataFlow is Kailash's zero-config data layer. It generates the CRUD nodes automatically from the model class. No ORM boilerplate."
  **Transition**: "Database operations involve waiting. Python's async syntax is the clean way to handle waits."

---

## Slide 72: Async/Await — Quick Primer

**Time**: ~2 min
**Talking points**:

- Many students will not have seen async/await before. Keep it simple: `await` means "wait for this but let others proceed."
- The database operations use it because real databases take time — and while you wait, the CPU is free to do other work.
- For the exercise, students just need to know the syntax pattern: `result = await db.express.create(...)` inside an `async def`. They do not need event loops or coroutine theory yet.
- If beginners look confused: "Think of await as 'pause here and come back when ready.' The function does not block everything else in the program."
- If experts look bored: "Kailash runtime is async-first. All DataFlow operations are coroutines."
  **Transition**: "Async is the right abstraction, but connections are finite."

---

## Slide 73: DataFlow — ConnectionManager

**Time**: ~2 min
**Talking points**:

- Connection management is a production concern. In a notebook, leaked connections are annoying. In a production server handling thousands of requests, they cause crashes.
- ConnectionManager comes from `kailash.db` and takes a database URL; it has no async-with support, so the try/finally pattern is the safe default.
- One practical trap from Exercise 3.8: give DriftMonitor its OWN database file. The registry's database already contains a drift-reports table with a different layout, so pointing both at the same file fails.
- A second practical trap: pass DataFlow an absolute SQLite URL — a relative `sqlite:///x.db` is resolved against the filesystem root and fails to open. The course's shared helpers build absolute URLs.
  **Transition**: "Once predictions are stored, we monitor for drift."

---

## Slide 74: DriftMonitor — Detecting Model Decay

**Time**: ~2 min
**Talking points**:

- Drift is why ML is not "deploy and forget." COVID-19 broke many models trained on pre-pandemic data.
- Data drift: feature distributions shift. Concept drift: the relationship between features and labels shifts.
- For financial models: interest rate changes, regulatory changes, and economic cycles all cause drift. A model trained in 2019 is not valid in 2024 without re-evaluation.
- The DriftMonitor detects this automatically by comparing incoming data against the training distribution.
- If beginners look confused: "Your model learned yesterday. Today's data is different. Drift monitoring tells you when yesterday's knowledge is stale."
- If experts look bored: "We test both feature drift (X distribution) and concept drift (Y given X). The latter is harder because it requires ground truth labels."
  **Transition**: "Two statistical tools for the job."

---

## Slide 75: PSI and KS Test

**Time**: ~2 min
**Talking points**:

- PSI is the industry standard for monitoring feature distributions. Formula: PSI = Σ (actual% − expected%) · ln(actual% / expected%).
- Rule of thumb: below 0.1 = no significant change, 0.1–0.25 = moderate shift, above 0.25 = significant drift, investigate.
- KS test is the classical statistical approach: maximum difference between two empirical CDFs, non-parametric.
- Both are complementary: PSI bins the distributions and compares proportions. KS looks at the continuous CDFs. In practice, monitor both and alert when either crosses the threshold.
- If beginners look confused: "PSI gives a number for how much a distribution moved. A score above 0.25 means 'call the modelling team'."
- If experts look bored: "PSI is essentially a symmetric KL divergence between two discretised distributions. KS has distributional guarantees that PSI lacks, but PSI is easier to report."
  **Transition**: "Kailash packages both."

---

## Slide 76: Kailash Bridge: DriftMonitor

**Time**: ~2 min
**Talking points**:

- DriftMonitor takes a ConnectionManager and stores the reference distribution (the training data the model learned from) under the model name; `check_drift` computes PSI and KS (plus Jensen-Shannon) per feature and returns a report with per-feature results, an overall flag and a severity.
- On the credit data the untouched test batch raises no alert, and the batch with incomes scaled by 1.5 is flagged "severe" on `income_sgd` alone.
- The KS p-value threshold is 0.001 rather than 0.05 because 33 features are tested at once: at 0.05, one or two features would be flagged by chance in every clean batch (Bonferroni: 0.05 / 33 is about 0.0015).
- Give the monitor its own database file; sharing the registry's file fails.
- Scheduled checks (`schedule_monitoring` with a DriftSpec) exist for production; Exercise 3.8 runs the checks explicitly and records each one as a DataFlow DriftCheck row, then acknowledges alerts and deletes a smoke-test row.
  **Transition**: "Monitoring is one piece. Documentation is another."

---

## Slide 77: Model Cards — Documenting Your Model

**Time**: ~2 min
**Talking points**:

- Model cards (Mitchell et al., 2019) are the "nutrition label" for ML models: intended use, performance across subgroups, limitations, fairness findings, training data characteristics.
- They document performance, limitations, and fairness for anyone who uses or governs the model — not just engineers. In regulated industries this kind of documentation is increasingly expected by model-risk reviewers.
- kailash-ml has no model-card generator; Exercise 3.8 writes the card itself, and every number in it, including the fairness results by race, gender and age band, is measured, not typed in as prose.
- If beginners look confused: "A model card is a README for your model, written for regulators and users, not for developers."
- If experts look bored: "Google Model Cards, Hugging Face model cards, and Nvidia's Model Cards++ are the three published standards to know. Ours follows Mitchell's nine sections."
  **Transition**: "One more modern technique: conformal prediction."

---

## Slide 78: Conformal Prediction — Uncertainty Quantification

**Time**: ~2 min
**Talking points**:

- Conformal prediction is the modern approach to uncertainty quantification. Unlike Bayesian methods, it makes no distributional assumptions.
- The guarantee is finite-sample, but it is marginal (on average over applicants, not for each applicant) and it needs exchangeable data, which drift violates.
- Exercise 3.8 uses the classification version on credit default: an applicant whose prediction set holds both classes is ambiguous and is routed to a human reviewer.
- The `ceil((n+1)(1−α))/n` quantile (taken with `method="higher"`) is the finite-sample correction; a plain 90th percentile slightly under-covers.
- For professionals: "Instead of saying the HDB price is S$580,000, the model says S$550,000 to S$610,000 with 90% coverage. That is the level of honesty a business can act on."
- If beginners look confused: "Instead of a single number, the model gives a range. You can trust the range more than the number."
- If experts look bored: "Split conformal is the simplest variant. Jackknife+ and CV+ trade compute for tighter intervals. Angelopoulos and Bates 2021 is the accessible reference."
  **Transition**: "Now let us see the full pipeline."

---

## Slide 79: The Full Production Pipeline

**Time**: ~2 min
**Talking points**:

- This is the capstone view: every engine from M3 in its place.
- Walk the diagram: ingestion → FeatureEngineer/FeatureStore → TrainingPipeline → HyperparameterSearch → evaluation → calibration → ModelRegistry → DataFlow persistence → DriftMonitor → model card.
- The full pipeline is what the exercise builds. WorkflowBuilder orchestrates all the other engines into an automated production system.
- "If you can wire this pipeline end-to-end, you can productionise ML models."
  **Transition**: "Let us zoom out and look at the discipline that makes this repeatable."

---

## Slide 80: MLOps — CI/CD for Machine Learning

**Time**: ~2 min
**Talking points**:

- MLOps is the emerging discipline for managing ML in production. The key insight: ML systems have two axes of change (code and data) instead of one. Both must be tracked, tested, and deployed systematically.
- CI/CD adaptations: data validation tests, model performance tests, skew tests (train vs serving), drift monitoring as part of the pipeline.
- For professionals: this is the organisational capability that separates ML experiments from ML products. Most teams fail here — they can train a model but cannot deploy it repeatably.
- Practical components: version control (code, data, models), pipelines (training, evaluation, deployment), monitoring (performance, drift), alerting (on degradation), rollback (to a previous model version).
- If beginners look confused: "DevOps for code is hard. MLOps adds data and models on top. Three moving parts instead of one."
- If experts look bored: "The Google Rules of Machine Learning and the Uber Michelangelo paper are the two foundational references. Kailash's engines cover most of what those papers describe."
  **Transition**: "Lesson 3.8 summary."

---

## Slide 81: Lesson 3.8 Summary

**Time**: ~1 min
**Talking points**:

- Quick recap. This exercise is the capstone.
- The model is trained and calibrated through TrainingPipeline; the decision threshold is 0.130, the cost-optimal t* from Lesson 3.5.
- Drift is checked with DriftMonitor (PSI 0.2, KS p-value 0.001) and every check is persisted and acknowledged through DataFlow.
- Promotion uses `promote_model`, and the rollback is rehearsed by moving the archived version back through staging to production and verifying the restored model is the same one.
- The readiness check computes 11 gates from the real artefacts left by earlier steps; what code cannot judge (for example the age-band fairness failure) goes to a human sign-off, not a hard-coded pass.
  **Transition**: "That is the eight lessons. Let us take stock before the assessment and preview of M4."

---

## Slide 82: Module 3 — Key Formulas Reference

**Time**: ~1 min
**Talking points**:

- Reference slide. Students can photograph this. Every formula from M3 in one place: bias-variance decomposition, L1/L2/Elastic Net penalties, Gini, information gain, XGBoost split gain, precision/recall/F1, focal loss, Brier score, Shapley value, disparate impact ratio, PSI.
- "You do not need to memorise these for the assessment. You need to know when to apply them."
  **Transition**: "And the matching engine reference."

---

## Slide 83: Module 3 — Complete Engine Map

**Time**: ~1 min
**Talking points**:

- Reference slide. Every engine introduced in M3 and where it was covered.
- Students can use this to find the relevant lesson when they need to revisit an engine — during the capstone or in real work.
- "Bookmark this. When you are back at work trying to remember 'which engine does probability calibration,' this table has the answer."
  **Transition**: "Let us see how far you have come."

---

## Slide 84: Module 3 — What You Can Now Do

**Time**: ~2 min
**Talking points**:

- Walk through the outcomes. Students started M3 knowing linear regression from M2. Now they have the complete supervised ML toolkit: science (models, evaluation, interpretation) and engineering (workflows, registry, monitoring).
- This is the foundation for M4 (unsupervised) and beyond.
- "If you can train a model, evaluate it honestly, interpret it, and deploy it with drift monitoring, you are doing what most production ML teams do on any given day."
- If beginners look confused: "That feels like a lot because it is a lot. You do not have to master everything today. The exercises and the notebooks will be there for you to revisit."
- If experts look bored: "The exercises push further than the slides. If you want depth, do exercise 3.7 with nested CV and exercise 3.8 with real conformal intervals."
  **Transition**: "Now, how we assess all this."

---

## Slide 85: Module 3 — Assessment

**Time**: ~2 min
**Talking points**:

- There is no separate quiz and no free-form project in this module's assessment. Four auto-graded coding tasks, 100 marks total (20 / 25 / 25 / 30), 3 hours, open-book, no AI assistants. One e-commerce dataset (50,000 customers); target `premium_response` (~25% positive).
- Task 1 (20): feature engineering and leakage-free selection — six engineered features ranked with FeatureEngineer, fitted on the train split only. Task 2 (25): the model zoo — six algorithms trained and compared through TrainingPipeline. Task 3 (25): evaluation, imbalance and interpretability — baseline vs class weights, per-class recall, SHAP. Task 4 (30): the production pipeline — TrainingPipeline → ModelRegistry (promote) → DriftMonitor.
- Students complete `solve()` in each task's starter.py; an independent reference implementation re-derives the expected values from the data, and a task's marks are awarded only when all of its checks pass. Marked on outcomes (held-out AUC floors, recall lift, promotion, drift caught on the shifted batch and not the clean one), not code style.
- Point out why the target is derived: the dataset's own `churned` column is a near-deterministic function of recency (the same leak removed from Exercise 3.3), so it is useless for modelling practice.
- Task 1 rewards exactly the discipline of Lesson 3.1: rank features on training rows only.
  **Transition**: "Here is the exercise schedule."

---

## Slide 86: Exercise Summary

**Time**: ~1 min
**Talking points**:

- Quick reference for exercises. Notice the progression: ICU data for point-in-time feature engineering, a synthetic sine curve where the true function is known (so bias and variance can be measured), e-commerce churn for the model zoo, then the Singapore credit data as the running case study from 3.4 through 3.8.
- Every credit exercise drops `customer_id` and the planted `future_default_indicator` leak, leaving 33 features.
- "Each exercise has two formats: a local .py in VS Code and a self-contained Colab notebook. Pick one and stick with it."
  **Transition**: "One more reference: traps to avoid."

---

## Slide 87: Common Mistakes to Avoid

**Time**: ~2 min
**Talking points**:

- These are the mistakes professionals actually make in production. Every item on this list is a common real-world failure.
- Typical items: preprocessing leakage, using accuracy on imbalanced data, not calibrating probabilities, k-fold on time series, tuning on the test set, ignoring drift, no model card, no fairness testing.
- No engine prevents them automatically: PreprocessingPipeline's `setup()` learns from every row you pass it, so hold the test rows out first; DriftMonitor only helps if someone is alerted and acts.
- "Print this list. Stick it next to your monitor. Check it before every ML deployment."
- If beginners look confused: "You will not remember all these today. That is fine. Come back to this list before shipping anything."
- If experts look bored: "Add your own to the list as you gain experience. Every team's top 10 mistakes evolve with their domain."
  **Transition**: "And here is the decision framework."

---

## Slide 88: Model Selection Decision Tree

**Time**: ~2 min
**Talking points**:

- This is the practical decision framework students can use in their projects. Not a rigid algorithm, but a starting point.
- Walk the branches the way the slide draws them: interpretability first decides between transparent models (regularised linear, a single tree) and the rest; then size decides between gradient boosting (LightGBM for speed, CatBoost for many categoricals) and Random Forest or a kernel SVM on smaller data.
- The key message: always try multiple models, always use the right metric, always validate with proper CV.
  **Transition**: "One last big picture."

---

## Slide 89: The Complete Supervised Pipeline

**Time**: ~2 min
**Talking points**:

- The complete picture. Features flow through models, evaluation, interpretation, orchestration, and production.
- The dashed feedback loop shows drift detection triggering retraining. This is the ML lifecycle that every professional ML system follows.
- "This diagram is what M3 has been building toward since slide 1. You now understand every box and every arrow."
- If beginners look confused: "Do not try to memorise this. It is a reference. You will look at it again whenever you plan an ML project."
- If experts look bored: "Extend this mentally to M4 (unsupervised) and M5 (deep learning). The pipeline thinking transfers. Only the evaluation and governance pieces change."
  **Transition**: "A preview of what comes next."

---

## Slide 90: Coming Up: Module 4

**Time**: ~1 min
**Talking points**:

- Brief preview. M4 extends from labelled data (supervised) to unlabelled data (unsupervised): clustering on retail customers, EM and mixtures, PCA through UMAP, anomaly detection on financial transactions, market-basket rules, topics from news text, recommender systems, and finally a neural network built from scratch as the bridge to the Advanced Certificate (Modules 5 and 6: deep learning and vision, then language models and agents).
- kailash-ml's clustering, dimensionality-reduction and anomaly-detection engines are used alongside EnsembleEngine.
- Evaluation changes (no labels means different metrics — silhouette, reconstruction error), but the pipeline thinking stays the same. The tools from M3 carry forward: WorkflowBuilder, ModelRegistry, DriftMonitor, DataFlow.
- M4 completes the Foundation Certificate.
  **Transition**: "That wraps up Module 3."

---

## Slide 91: Supervised Machine Learning for Building and Deploying Models

**Time**: ~1 min
**Talking points**:

- Thank the class. Remind them of the assessment (four auto-graded tasks, 100 marks). Point to the exercise schedule and the two formats (local .py, Colab).
- Encourage them to do at least one end-to-end pipeline before the next session.
- If beginners look confused: "You just completed the supervised ML module. Review the formula reference slide and engine map. If you only remember one thing: always cross-validate, always calibrate, always monitor for drift."
- If experts look bored: "The exercises have depth. Push yourself on nested CV in 3.2 and 3.3, and on conformal prediction and the readiness gates in 3.8."
  **Transition**: "See you in Module 4. Go build something."

---

**Timing summary**: Talking points sum to ~166 min across 91 slides, leaving ~15 min of headroom for room transitions, live Q&A pauses, and the short coding demos embedded in Lessons 3.1, 3.4, and 3.8. Full-deck target: ~180 min. Taught lesson-by-lesson, the eight lesson decks total ~560 min (L1 ~75, L2 ~75, L3 ~90, L4 ~75, L5 ~75, L6 ~55, L7 ~55, L8 ~60).
