# Module 3: Supervised ML — Theory to Production — Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: Title

**Time**: ~1 min
**Talking points**:

- Read the provocation aloud: "They were RIGHT but still lost half a billion dollars."
- Let it sit. This is the central paradox of Module 3 — accuracy alone is not enough.
- If beginners look confused: "Today we learn how to build ML models that actually work in the real world, not just on a test set."
- If experts look bored: "We derive the bias-variance decomposition from first principles, the XGBoost objective with second-order Taylor expansion, focal loss, and conformal prediction. Plus production pipeline orchestration."
  **Transition**: "Let us see where we are in the programme."

---

## Slide 2: Where We Are

**Time**: ~2 min
**Talking points**:

- Quick recap of M1 (data fluency) and M2 (experiment design).
- "M1 taught you to see data. M2 taught you to reason about it. M3 teaches you to MODEL it — and deploy it safely."
- Do not linger. The audience knows what they have learned.
  **Transition**: "What is new today?"

---

## Slide 3: What Is New Today

**Time**: ~2 min
**Talking points**:

- Theory: bias-variance trade-off, gradient boosting, SHAP explainability.
- Production: workflow orchestration, model registry, governed deployment.
- New engines: TrainingPipeline, HyperparameterSearch, ModelRegistry, WorkflowBuilder, DataFlow.
- "By the end of today, you will train, evaluate, explain, register, and deploy a supervised ML model through a governed production pipeline."
  **Transition**: "Here is the cumulative engine map."

---

## Slide 4: Kailash Engines — Cumulative Map

**Time**: ~2 min
**Talking points**:

- M1: explore, clean, visualise. M2: features, experiments. M3 (new, highlighted): train, tune, register, orchestrate, persist.
- "Five new engines today. By the end of Module 3, you will have used 11 of the 13 Kailash ML engines."
  **Transition**: "Let me tell you a story about what happens when you skip the production part."

---

## Slides 5-8: Opening Case — Zillow iBuyer (vertical slide stack)

**Time**: ~8 min total

### Slide 5: Zillow iBuyer

**Time**: ~3 min
**Talking points**:

- "$881 million inventory write-down, Q3 2021." Let that number land.
- "Zillow's Zestimate algorithm predicted home prices with impressive accuracy. They trusted it enough to buy 27,000 homes sight-unseen."
- "The model was accurate. The business lost half a billion dollars."
- If beginners look confused: "Imagine a calculator that gives you the right answer most of the time, but occasionally tells you a $300K house is worth $500K. If you buy thousands of houses based on those answers, the occasional wrong answers bankrupt you."

### Slide 6: What Went Wrong?

**Time**: ~3 min
**Talking points**:

- The model WAS good: 2% median error, continuously retrained, best-in-class.
- The SYSTEM was bad: uncalibrated confidence (no uncertainty), no drift monitoring, no governance gate.
- "The model said 'this house is worth $500K' without saying 'but I could be off by $80K.' That missing uncertainty killed them."

### Slide 7: The Lesson

**Time**: ~2 min
**Talking points**:

- "Accuracy alone is not enough. A production ML system needs calibrated uncertainty, drift detection, governance gates, and a model lifecycle."
- "Today we learn ALL of these pieces."
- Source note: SEC 10-Q filing, 2,000 employees laid off, iBuyer division shut down entirely.

**Transition**: "Let us start with the most fundamental concept in ML theory."

**[PAUSE FOR QUESTIONS — 2 min]**

---

## Slide 9: What Is Prediction?

**Time**: ~3 min
**Talking points**:

- Regression: output is a continuous number (HDB price, temperature).
- Classification: output is a category (loan default yes/no, image cat/dog/bird).
- "This module covers both. The theory applies to both. The Kailash engines handle both."
- If beginners look confused: "Regression = guess a number. Classification = pick a label."
  **Transition**: "But how do we know if our prediction is any good?"

---

## Slide 10: Training vs Testing

**Time**: ~3 min
**Talking points**:

- "Why you cannot grade your own homework."
- Training set (80%): the model learns from this. Test set (20%): data the model has NEVER seen.
- Exam analogy: "A student who memorises past papers might score 100% on those papers but fail the actual exam. The test set IS the actual exam."
- If beginners look confused: "Hide some data. Train on the rest. Then check: does the model work on data it never saw?"
  **Transition**: "When a model memorises instead of learns, that is overfitting."

---

## Slide 11: What Is Overfitting?

**Time**: ~3 min
**Talking points**:

- Three-column comparison: underfitting (too simple), just right, overfitting (memorises noise).
- "More complex is not always better. The goal is to learn the signal, not the noise."
- If beginners look confused: "Imagine connecting all the dots on a chart with a wiggly line. It passes through every point perfectly — but it is useless for predicting new points."
  **Transition**: "The darts analogy makes this concrete."

---

## Slide 12: Bias and Variance — The Darts Analogy

**Time**: ~4 min
**Talking points**:

- Walk through the three dartboards: low bias + low variance (ideal), high bias + low variance (consistent but wrong), low bias + high variance (right on average but unreliable).
- "Bias = how far the average prediction is from the truth. Variance = how spread out predictions are across different training sets."
- This is one of the most important conceptual slides in the module. Make sure every student understands the analogy before moving to the maths.
- If beginners look confused: "Think of throwing darts. Bias = are you aiming at the right spot? Variance = how scattered are your throws?"
  **Transition**: "Now let us formalise this mathematically."

---

## Slide 13: Bias-Variance Decomposition — Setup (THEORY)

**Time**: ~3 min
**Talking points**:

- "We want to predict y = f(x) + noise. Our model is trained on a random sample D. What is the expected prediction error?"
- Write out EPE(x) = E[(y - f_hat(x))^2].
- "This is the question we are about to decompose into three independent, irreducible terms."
- If beginners look confused: "We are asking: on average, how wrong will our model be? And can we break that wrongness into separate causes?"
  **Transition**: "Step 1: expand the square."

---

## Slide 14: Derivation — Step 1: Expand (THEORY)

**Time**: ~3 min
**Talking points**:

- Substitute y = f(x) + epsilon. Add and subtract E[f_hat(x)].
- "The trick is adding and subtracting the average prediction. This separates the bias term from the variance term."
- Walk through the three underbraced terms: bias, variance, noise.
  **Transition**: "Step 2: the cross terms vanish."

---

## Slide 15: Derivation — Step 2: Cross Terms Vanish (THEORY)

**Time**: ~3 min
**Talking points**:

- Three reasons cross terms vanish: noise is zero-mean, E[E[f_hat] - f_hat] = 0 by definition, bias term is constant w.r.t. D.
- Show the final result: EPE = Bias^2 + Variance + Irreducible Noise.
- "This is the most important equation in supervised ML. Everything we do from now on is about managing these three terms."
  **Transition**: "What does each term mean in practice?"

---

## Slide 16: The Three Terms (THEORY)

**Time**: ~4 min
**Talking points**:

- Walk through the table: Bias^2 (how far off on average, fix with more complexity), Variance (how much predictions change, fix with simpler model/more data/regularisation), Noise (randomness, cannot reduce).
- "The trade-off: reducing bias typically increases variance. Reducing variance typically increases bias. The art of ML is finding the sweet spot."
- If beginners look confused: "A simple model is consistently wrong in the same direction (high bias, low variance). A complex model is sometimes right but unpredictable (low bias, high variance). You want the middle ground."
  **Transition**: "Modern deep learning breaks this classical picture."

---

## Slide 17: Double Descent (ADVANCED)

**Time**: ~3 min
**Talking points**:

- "Classical wisdom: more parameters = more overfitting. But modern deep learning uses billions of parameters and generalises well."
- Double descent: test error first decreases, then increases at the interpolation threshold, then decreases again.
- "This does NOT mean 'just make everything bigger.' It means the classical bias-variance picture is incomplete."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "For now, let us learn the classical tool for managing overfitting: regularisation."

---

## Slide 18: Regularisation — Why Constrain Your Model?

**Time**: ~3 min
**Talking points**:

- "Regularisation adds a penalty for complexity."
- Essay analogy: "Your teacher says 'explain it in 500 words, not 5000.' The constraint forces clarity."
- Show the equation: Loss_regularised = Loss_data + lambda \* R(w).
- "Lambda controls how strongly we penalise. Too high = underfitting. Too low = overfitting."
- If beginners look confused: "Regularisation is like a budget constraint. The model cannot spend unlimited resources memorising every detail."
  **Transition**: "L1 and L2 regularisation have different geometric properties."

---

## Slide 19: L1 vs L2 — Diamond vs Circle (THEORY)

**Time**: ~4 min
**Talking points**:

- L2 (Ridge): sphere constraint. Shrinks all weights toward zero, rarely TO zero. All features kept.
- L1 (Lasso): diamond constraint. Corners on axes. Loss contours hit corners where some w_j = 0. Produces sparse solutions — automatic feature selection.
- "L1 is useful when you believe many features are irrelevant."
- If beginners look confused: "Ridge keeps all features but makes them smaller. Lasso actually removes features by setting their weights to exactly zero."
  **Transition**: "There is a beautiful Bayesian interpretation."

---

## Slide 20: Bayesian Interpretation (THEORY)

**Time**: ~3 min
**Talking points**:

- "Regularisation is equivalent to placing a prior on the weights and doing MAP estimation."
- L2 = Gaussian prior. L1 = Laplace prior (sharp peak at zero, heavier tails).
- "The Laplace prior's sharp peak at zero is WHY L1 induces sparsity."
- If beginners look confused: "This connects to what you learned in Module 2 about MAP estimation. L2 regularisation IS a Gaussian prior on your model weights."

**[CAN SKIP IF RUNNING SHORT]**

**[PAUSE FOR QUESTIONS — 3 min]**

**Transition**: "Now let us learn the most powerful algorithm for tabular data: gradient boosting."

---

## Slide 21: What Is a Decision Tree?

**Time**: ~3 min
**Talking points**:

- "A flowchart for making predictions."
- Walk through the loan approval example: income > $50K? -> credit score > 700? -> approve/review/deny.
- "At each node, the tree finds the question that best separates the outcomes."
- If beginners look confused: "Think of 20 Questions. Each question narrows down the answer."
  **Transition**: "One tree is unstable. Many trees are powerful."

---

## Slide 22: Trees to Forests to Boosting

**Time**: ~3 min
**Talking points**:

- Single tree: low bias, high variance (unstable). Random Forest: averaging reduces variance. Gradient boosting: sequential correction reduces bias.
- "Bagging reduces variance by averaging. Boosting reduces bias by iteratively fitting residuals. Boosting is the dominant paradigm for tabular data."
- If beginners look confused: "Random Forest: many opinions, take the average. Boosting: each expert fixes the mistakes of the previous expert."
  **Transition**: "How does gradient boosting actually work?"

---

## Slide 23: Gradient Boosting — Intuition

**Time**: ~3 min
**Talking points**:

- Walk through the team analogy: Tree 1 rough prediction, compute mistakes, Tree 2 predicts mistakes, Tree 3 predicts remaining mistakes, sum all contributions.
- Show the equation: y_hat = sum of eta \* h_m(x). "M trees, each weighted by learning rate eta."
- "Small learning rate = small steps. Avoids overshooting."
- If beginners look confused: "Each tree fixes a little bit of what the previous tree got wrong. Together, they get very close to the right answer."
  **Transition**: "XGBoost improves this with second-order information."

---

## Slide 24: XGBoost — The Objective Function (THEORY)

**Time**: ~3 min
**Talking points**:

- "XGBoost uses second-order information — the curvature — to find better splits."
- Walk through the objective: loss at step t plus regularisation.
- Regularisation Omega: gamma penalises number of leaves, lambda penalises large leaf weights.
  **Transition**: "The key innovation is the Taylor expansion."

---

## Slide 25: XGBoost — 2nd-Order Taylor Expansion (THEORY)

**Time**: ~4 min
**Talking points**:

- "Taylor-expand the loss around the current prediction."
- Define g_i (gradient, direction of steepest descent) and h_i (Hessian, curvature).
- "Standard gradient boosting uses only g_i. XGBoost uses BOTH g_i and h_i for curvature-adjusted steps."
- If beginners look confused: "The gradient tells you which direction to go. The Hessian tells you how big a step to take. Using both is faster and more accurate."
  **Transition**: "This leads to optimal leaf weights."

---

## Slide 26: XGBoost — Optimal Leaf Weights (THEORY)

**Time**: ~3 min
**Talking points**:

- Group by leaf j. Substitute f_t(x_i) = w_j for sample i in leaf j.
- Differentiate w.r.t. w_j, set to zero. Optimal weight: w_j\* = -G_j / (H_j + lambda).
- "The optimal leaf weight is the ratio of gradient sum to Hessian sum, with regularisation."
  **Transition**: "This also tells us whether a split is worthwhile."

---

## Slide 27: XGBoost — Split Gain Formula (THEORY)

**Time**: ~3 min
**Talking points**:

- "Gain = left child score + right child score - parent score - penalty for adding a leaf."
- "If the gain is negative, the split is not worth it — regularisation prevents unnecessary complexity."
- "Why second-order matters: standard gradient boosting uses Newton's method with step size 1. XGBoost uses curvature-adjusted steps, giving faster convergence."
  **Transition**: "LightGBM makes gradient boosting scale."

---

## Slide 28: LightGBM — Speed and Scale (THEORY)

**Time**: ~3 min
**Talking points**:

- Histogram-based splitting: bin features into ~256 bins. O(bins) instead of O(n log n).
- GOSS: keep all large-gradient samples, subsample small-gradient ones.
- Leaf-wise growth vs level-by-level: "Leaf-wise can overfit more but converges faster."
- "LightGBM is 10-20x faster than exact XGBoost on large data."
  **Transition**: "For distributional predictions, there is NGBoost."

---

## Slide 29: NGBoost — Uncertainty from Boosting (ADVANCED)

**Time**: ~2 min
**Talking points**:

- "Standard boosting predicts a point. NGBoost predicts a full probability distribution."
- Uses natural gradients (Fisher information matrix) instead of ordinary gradients.
- "Connection to Zillow: distributional predictions could have quantified uncertainty per house."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "TrainingPipeline wraps all of this into one engine."

---

## Slide 30: Kailash Bridge — TrainingPipeline

**Time**: ~4 min
**Talking points**:

- Walk through the code: ModelSpec (algorithm, task, hyperparameters), EvalSpec (metrics, cv_folds, calibration).
- "One config object describes your entire experiment. The engine handles splits, CV, metric computation, and calibration."
- "Notice calibration=True. That is Platt scaling automatically applied. One line prevents the Zillow problem."

**[SWITCH TO LIVE CODING — 5 min]**

- Demonstrate: create a TrainingPipeline, configure ModelSpec and EvalSpec, train on HDB data.

**[PAUSE FOR QUESTIONS — 3 min]**

**Transition**: "What happens when one class vastly outnumbers the other?"

---

## Slide 31: Class Imbalance — The Problem

**Time**: ~3 min
**Talking points**:

- "99.9% accurate. 0% useful."
- "In fraud detection, a model that always says 'not fraud' gets 99.9% accuracy but catches zero fraudsters."
- Walk through better metrics: precision, recall, F1, AUPRC.
- If beginners look confused: "Accuracy counts all correct predictions equally. But in fraud detection, catching a fraudster is worth much more than correctly labelling a normal transaction."
  **Transition**: "SMOTE is the popular solution, but it often fails."

---

## Slide 32: SMOTE — And Why It Often Fails (THEORY)

**Time**: ~3 min
**Talking points**:

- What SMOTE does: create synthetic minority samples by interpolating between neighbours.
- Failure modes: noisy boundaries, high dimensions, small clusters, information leak before split.
- "SMOTE must only be applied to training data, NEVER before the split. Many published papers have made this error."
  **Transition**: "Cost-sensitive learning is a better approach."

---

## Slide 33: Cost-Sensitive Learning (THEORY)

**Time**: ~3 min
**Talking points**:

- "Adjust the loss function to penalise misclassifying the minority class more heavily."
- scale_pos_weight in XGBoost/LightGBM: ratio of negative to positive examples.
- "Advantage over SMOTE: no synthetic data, no leak risk, works in any dimension, integrates with the optimiser."
  **Transition**: "Focal loss is even more sophisticated."

---

## Slide 34: Focal Loss — Derivation (THEORY)

**Time**: ~3 min
**Talking points**:

- "Standard CE treats all examples equally in difficulty. Focal loss down-weights easy examples."
- Start from binary cross-entropy. Add the modulating factor (1 - p_t)^gamma.
- "The modulating factor acts as a curriculum: easy examples contribute almost nothing to the gradient."
  **Transition**: "Let us see why this works."

---

## Slide 35: Focal Loss — Why It Works (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the table: easy correct (p_t=0.95) -> loss reduced 400x. Hard/misclassified (p_t=0.10) -> loss barely changed.
- "gamma controls focusing strength. gamma=0 recovers standard CE. gamma=2 is the default."
- "alpha handles class balance. (1-p_t)^gamma handles example difficulty. Together more effective than either alone."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Even with good predictions, can you trust the probabilities?"

---

## Slide 36: Calibration — Can You Trust the Probabilities?

**Time**: ~4 min
**Talking points**:

- "When your model says 80% chance of default, does it really mean 80% of similar cases actually default?"
- Calibrated vs uncalibrated: the model says 80% but the true rate is 45%. "You cannot use these probabilities for decisions."
- Why it matters: setting thresholds, comparing risks, Zillow's uncalibrated confidence.
- If beginners look confused: "If a weather app says '80% chance of rain,' you expect it to rain about 80% of the time it says that. If it only rains 40% of the time, the app is uncalibrated."
  **Transition**: "Two methods to fix calibration."

---

## Slide 37: Calibration Methods (THEORY)

**Time**: ~3 min
**Talking points**:

- Platt scaling: logistic regression on raw outputs. Works well for monotonic miscalibration.
- Isotonic regression: non-parametric, more flexible, needs more data.
- ECE: expected calibration error — the standard metric.
  **Transition**: "Proper scoring rules tell you whether your metric encourages calibration."

---

## Slide 38: Proper Scoring Rules (ADVANCED)

**Time**: ~3 min
**Talking points**:

- "A proper scoring rule cannot be gamed by reporting anything other than your true belief."
- Brier score: strictly proper, unique minimum at true probability.
- Accuracy: NOT proper. No incentive to calibrate — 0.51 and 0.99 get the same score for a positive example.

**[CAN SKIP IF RUNNING SHORT]**

**[PAUSE FOR QUESTIONS — 3 min]**

**Transition**: "A model predicts 'deny this loan.' The applicant asks: why?"

---

## Slide 39: Why Explain Your Model?

**Time**: ~3 min
**Talking points**:

- Regulatory requirement (EU AI Act, Singapore PDPA), debugging, trust, fairness.
- "If Zillow could have explained WHICH features drove each estimate, they might have caught the bubble reliance."
- If beginners look confused: "People will not trust what they cannot understand. And regulators will not allow what they cannot audit."
  **Transition**: "SHAP gives mathematically rigorous explanations."

---

## Slide 40: Shapley Values — Intuition

**Time**: ~4 min
**Talking points**:

- Group project analogy: three students, joint grade, how much did each contribute?
- "Shapley's answer: try every possible order. Measure each student's marginal contribution in each order. Their fair share is the average."
- "Replace students with features and grade with model prediction."
- If beginners look confused: "Imagine removing each feature one at a time and seeing how much the prediction changes. SHAP does a careful version of this."
  **Transition**: "Why is Shapley the only fair attribution?"

---

## Slide 41: Shapley Axioms (THEORY)

**Time**: ~3 min
**Talking points**:

- Four axioms: efficiency (contributions sum to total), symmetry (equal contributors get equal credit), dummy (irrelevant features get zero), linearity (for combined models, SHAP values add).
- "Shapley proved in 1953 that ONLY ONE attribution method satisfies all four axioms simultaneously."
  **Transition**: "Here is the formula."

---

## Slide 42: The Shapley Value Formula (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the formula. "Sum over all subsets. Weight by probability of this ordering. Bracket is the marginal contribution."
- "Computational cost: 2^p subsets. With 100 features, that is 10^30 evaluations. Exact Shapley is intractable."
  **Transition**: "TreeSHAP exploits tree structure for exact computation in polynomial time."

---

## Slide 43: TreeSHAP (THEORY)

**Time**: ~3 min
**Talking points**:

- "Dynamic programming over tree paths: O(TLD^2) instead of O(2^p)."
- Comparison: exact Shapley O(2^p) intractable, Kernel SHAP O(n\*p) approximate, TreeSHAP O(TLD^2) EXACT.
- "This is why gradient boosting + SHAP is such a powerful combination."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Counterfactual explanations answer a different question."

---

## Slide 44: Counterfactual Explanations (ADVANCED)

**Time**: ~3 min
**Talking points**:

- "SHAP: why was this prediction made? Counterfactual: what would need to change for a different outcome?"
- Loan denial example: "If income were $5K higher OR credit utilisation below 30%, the loan would have been approved."
- Constraints: changes must be actionable and plausible.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Documentation is governance."

---

## Slide 45: Model Cards (Foundations)

**Time**: ~3 min
**Talking points**:

- Mitchell et al. (2019): standardised documentation for every deployed model.
- Walk through the sections: model details, intended use, factors, metrics, evaluation data, training data, ethical considerations, caveats.
- "Every model in production needs a model card. This is not optional."
  **Transition**: "ModelVisualizer integrates SHAP and calibration plotting."

---

## Slide 46: Kailash Bridge — ModelVisualizer for SHAP

**Time**: ~3 min
**Talking points**:

- Walk through the code: feature_importance with method="shap", calibration_curve, precision_recall_curve.
- "Three lines. Global feature importance, calibration check, and imbalance-aware evaluation."

**[SWITCH TO LIVE CODING — 3 min]**

- Demonstrate: run feature importance, show calibration curve.
  **Transition**: "A production ML system is not a single script. It is a pipeline."

---

## Slide 47: Why Workflow Orchestration?

**Time**: ~3 min
**Talking points**:

- Without orchestration: notebooks run manually, "works on my machine," no retry, no audit trail.
- With orchestration: defined order, parameters passed, failures logged, every run reproducible.
- If beginners look confused: "Think of a factory assembly line vs making things by hand. The assembly line is faster, more reliable, and you can trace every step."
  **Transition**: "WorkflowBuilder implements this pattern."

---

## Slide 48: Kailash WorkflowBuilder

**Time**: ~4 min
**Talking points**:

- Walk through the code: add_node, connect, build, execute.
- "The workflow is a directed acyclic graph. Kailash handles ordering, parallelism, and error propagation."
- Key pattern: add_node -> connect -> build -> execute.

**[SWITCH TO LIVE CODING — 5 min]**

- Demonstrate: build a simple 4-node workflow, execute it.
  **Transition**: "You can extend workflows with custom nodes."

---

## Slide 49: Custom Nodes

**Time**: ~3 min
**Talking points**:

- @register_node decorator. execute() receives inputs dict, returns dict.
- "Node contract: receive dict, return dict. Connection names map output keys to input keys. Type safety at build time."
  **Transition**: "Conditional nodes add decision points."

---

## Slide 50: Logic Nodes — Decision Points

**Time**: ~3 min
**Talking points**:

- ConditionalNode: condition_key, true_target, false_target.
- "Governance built into the pipeline: the model is only registered if quality thresholds are met. Failed models trigger alerts."
- If beginners look confused: "It is an if/else in your pipeline. If the model is good enough, register it. Otherwise, send an alert."
  **Transition**: "Results need to be saved somewhere."

---

## Slide 51: Why Persist ML Results?

**Time**: ~2 min
**Talking points**:

- "Without persistence, everything vanishes when you close the notebook."
- With DataFlow: every metric stored, every experiment queryable, compare runs, full audit trail.
  **Transition**: "DataFlow makes persistence zero-config."

---

## Slide 52: DataFlow — Define Your Schema

**Time**: ~3 min
**Talking points**:

- @db.model auto-generates table, migrations, CRUD operations. No raw SQL.
- Walk through ExperimentResult fields: experiment_name, algorithm, hyperparameters, metrics.
  **Transition**: "CRUD operations are one-liners."

---

## Slide 53: DataFlow — CRUD Operations

**Time**: ~3 min
**Talking points**:

- Walk through: create, query with filters, compare results.
- "Zero-config persistence. Schema changes are auto-detected."
  **Transition**: "A quick primer on async before we continue."

---

## Slide 54: Async/Await Primer

**Time**: ~2 min
**Talking points**:

- Restaurant analogy: instead of standing at the counter, you sit down and wait to be called.
- "In Jupyter: top-level await works. In .py scripts: wrap in async def main(), asyncio.run(main())."
- "You do not need to deeply understand async. Just know: await before database operations."

**[PAUSE FOR QUESTIONS — 2 min]**

**Transition**: "Models have lifecycles, not just training."

---

## Slide 55: Model Lifecycle

**Time**: ~3 min
**Talking points**:

- Walk through the stages: Staging -> Shadow -> Production -> Archived.
- "Shadow mode is critical: the model runs alongside production, predictions logged but not served. You compare before committing."
- If beginners look confused: "Think of a new employee. They shadow a senior colleague before taking over."
  **Transition**: "ModelRegistry manages this lifecycle."

---

## Slide 56: Kailash ModelRegistry

**Time**: ~3 min
**Talking points**:

- Walk through the code: register with model, name, version, signature, metrics, tags.
- "The signature declares inputs and output. The metrics are stored for comparison."
  **Transition**: "Promotion is governed."

---

## Slide 57: Model Promotion — Governed Transitions

**Time**: ~3 min
**Talking points**:

- promote staging -> shadow. Compare shadow metrics with production metrics. Promote shadow -> production only if improvement meets threshold.
- "Promotion from shadow to production is NOT automatic. It requires explicit criteria."
- "In M6, PACT governance will enforce who can promote models."
  **Transition**: "Hyperparameters are settings the model cannot learn from data."

---

## Slide 58: Hyperparameter Search — Finding the Sweet Spot

**Time**: ~3 min
**Talking points**:

- Grid search: try every combination. Simple but exhaustive.
- Bayesian optimisation: use past results to predict which combination to try next. "Typically finds the optimum in 30-50 trials."
- If beginners look confused: "Learning rate, tree depth, regularisation strength — these are knobs you turn. Hyperparameter search finds the best setting."
  **Transition**: "Bayesian optimisation is the smart approach."

---

## Slide 59: Bayesian Optimisation — The Theory (THEORY)

**Time**: ~3 min
**Talking points**:

- Surrogate model: Gaussian Process gives mean prediction AND uncertainty.
- Acquisition function: balance exploitation (try where GP predicts well) vs exploration (try where uncertainty is high).
- Expected Improvement formula.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "HyperparameterSearch wraps this."

---

## Slide 60: Kailash HyperparameterSearch

**Time**: ~3 min
**Talking points**:

- Walk through the code: SearchSpace with ParamDistributions, SearchConfig with strategy and n_trials.
- "50 Bayesian trials instead of 3,125 grid search combinations."

**[SWITCH TO LIVE CODING — 3 min]**

- Demonstrate: define search space, run HPO.
  **Transition**: "Let us put all the pieces together."

---

## Slide 61: Production Pipeline — All Pieces Together

**Time**: ~4 min
**Talking points**:

- Walk through the pipeline diagram: Load -> Features -> Train+HPO -> Calibrate -> SHAP+Model Card -> Register.
- Walk through the table mapping each step to its engine and the lesson where it was taught.
- "This is the complete ML lifecycle, from raw data to registered, governed model."
- If beginners look confused: "Every step we learned today connects to the next. This diagram is the summary of Module 3."
  **Transition**: "Conformal prediction goes beyond calibration."

---

## Slide 62: Conformal Prediction — Guaranteed Uncertainty (THEORY)

**Time**: ~3 min
**Talking points**:

- "Calibration gives better probabilities. Conformal prediction provides prediction intervals with guaranteed coverage."
- "The guarantee: the true value falls within the interval at least (1-alpha) percent of the time, regardless of model or distribution."
- "For regression: interval = [y_hat - q_hat, y_hat + q_hat]."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Here is the complete pipeline in Kailash code."

---

## Slide 63: The Complete Pipeline in Kailash

**Time**: ~3 min
**Talking points**:

- Walk through the full WorkflowBuilder code: load, preprocess, features, split, HPO, train, calibrate, explain, quality_gate, register/alert.
- "This is production-ready orchestration. Every step is logged, every failure handled, every model governed."
  **Transition**: "Model cards in practice."

---

## Slide 64: Model Card — Filled Example

**Time**: ~3 min
**Talking points**:

- Walk through the HDB Price Predictor v3.1 model card.
- Highlight: "NOT for lending decisions without additional underwriting" (intended use limits).
- "Race/ethnicity not used as features. Town may correlate with demographics — monitored via DriftMonitor."
- If beginners look confused: "This is a template. Every deployed model gets one. It answers: what is this model, who should use it, what are its limits."
  **Transition**: "Here is the engine architecture diagram."

---

## Slide 65: Kailash ML Engine Architecture (THEORY)

**Time**: ~2 min
**Talking points**:

- Walk through the SVG: TrainingPipeline (ModelSpec, EvalSpec, Calibration, train), HPO Search, ModelRegistry, ModelVisualizer, DataFlow, WorkflowBuilder.
- "WorkflowBuilder orchestrates all the other engines."
  **Transition**: "Theory maps directly to engine methods."

---

## Slide 66: Theory to Engine Mapping

**Time**: ~3 min
**Talking points**:

- Walk through the table: bias-variance = EvalSpec(cv_folds=5), regularisation = hyperparameters, XGBoost = algorithm="lightgbm", SHAP = feature_importance(method="shap"), etc.
- "Every mathematical concept has a one-line engine call."
  **Transition**: "A few more theory details, then advanced topics."

---

## Slide 67: Elastic Net (THEORY)

**Time**: ~2 min
**Talking points**:

- "L1 alone is unstable with correlated features. Elastic Net combines both."
- alpha=1 pure Lasso, alpha=0 pure Ridge, alpha=0.5 common default.

**[CAN SKIP IF RUNNING SHORT]**

---

## Slide 68: XGBoost Worked Example — Squared Loss (THEORY)

**Time**: ~2 min
**Talking points**:

- For squared loss: g_i = -2 \* residual, h_i = 2 (constant).
- "For squared loss, Hessian is constant. For logistic loss, h_i varies per sample — that is where second-order truly helps."

**[CAN SKIP IF RUNNING SHORT]**

---

## Slide 69: CatBoost — Ordered Boosting (THEORY)

**Time**: ~3 min
**Talking points**:

- Prediction shift: residuals computed using all data including the sample being predicted.
- Ordered boosting: for sample i, compute residuals using only samples 1..i-1.
- Comparison table: XGBoost (2nd-order), LightGBM (histogram, GOSS), CatBoost (ordered, native categoricals).

**[CAN SKIP IF RUNNING SHORT]**

---

## Slide 70: Fairness — When Models Discriminate

**Time**: ~4 min
**Talking points**:

- "Even if you remove race and gender, a model can still discriminate through proxy features."
- Three sources: proxy features (postal code = race), historical bias (past discrimination in training data), measurement bias (arrests != crime rates).
- Disparate impact test: 4/5ths rule.
- If beginners look confused: "A model trained on biased history will perpetuate that bias. We need to check for this explicitly."
  **Transition**: "LIME provides a different style of explanation."

---

## Slide 71: LIME — Local Interpretable Explanations (THEORY)

**Time**: ~2 min
**Talking points**:

- Perturb input, get predictions, fit a simple model locally.
- "SHAP vs LIME: SHAP is mathematically grounded but slower. LIME is fast but can be inconsistent."

**[CAN SKIP IF RUNNING SHORT]**

**[PAUSE FOR QUESTIONS — 3 min]**

**Transition**: "Time for the lab."

---

## Slide 72: Lab Exercises — Overview

**Time**: ~3 min
**Talking points**:

- Walk through six exercises: bias-variance + boosting, class imbalance + calibration, SHAP + LIME, workflow orchestration, HPO + model registry, full production pipeline.
- "Scaffolding is ~50%. You write method calls, setup code, and some logic."
- "Start with ex_1 and work through. ex_6 is the capstone — all M3 engines combined."

**[SWITCH TO EXERCISE DEMO — 5 min]**

- Open ex_1.py. Walk through the TODO markers.
  **Transition**: "Here is the dataset."

---

## Slide 73: Dataset — HDB Resale Prices

**Time**: ~2 min
**Talking points**:

- 15M rows, 11 columns. Regression target: resale_price. Classification target: above_median.
- "Same dataset, two tasks. Regression for exercises 1, 4-6. Classification for exercises 2-3."

**[BEGIN LAB — allocate remaining time]**

---

## Slide 74: Discussion — The SHAP Dilemma

**Time**: ~4 min
**Talking points**:

- Read the scenario: model predicts loan default. Top SHAP features: postal code (+0.25), employment length (-0.15), dependents (+0.10). Postal code is a proxy for ethnicity.
- Three questions for the room: deploy? explain to regulator? how to make fairer?
- Let the room debate. There is no single right answer. Key insight: "Removing postal code may reduce accuracy but improve fairness. The trade-off is a business and ethical decision, not a technical one."

---

## Slide 75: Discussion — The 3am DriftMonitor Alert

**Time**: ~3 min
**Talking points**:

- PSI for floor_area_sqm jumped from 0.02 to 0.35. Model performance has not degraded yet.
- "Do you take the model offline? What data do you investigate? When do you retrain?"
- Key insight: "Feature drift precedes performance degradation. Acting early prevents silent failures."
- "In M4, we build the DriftMonitor engine. For now, think about the human process."

---

## Slide 76: Discussion — Calibration in Practice

**Time**: ~3 min
**Talking points**:

- Sales lead scoring: model outputs conversion probability, team calls top 50 each day.
- "Does calibration matter here?" The team only cares about ranking, not probabilities.
- "Platt scaling is monotonic — it does not change the ranking. So calibration does NOT change which leads are in the top 50."
- "But calibration DOES matter when you use probabilities for expected revenue calculations or threshold decisions."

---

## Slide 77: Key Takeaways — Foundations

**Time**: ~3 min
**Talking points**:

- Overfitting: more complex is not always better.
- Accuracy is misleading for imbalanced data — use F1, AUPRC, calibrated probabilities.
- Explainability is not optional — SHAP tells you WHY.
- Models have lifecycles: staging, shadow, production, archived.
- Production pipeline connects everything.
  **Transition**: "For those who followed the maths."

---

## Slide 78: Key Takeaways — Theory

**Time**: ~2 min
**Talking points**:

- EPE = Bias^2 + Variance + Noise.
- Regularisation = prior on weights.
- XGBoost: 2nd-order Taylor.
- Focal loss: (1-p_t)^gamma.
- Shapley: unique attribution satisfying four axioms.
- Calibration and proper scoring rules.
  **Transition**: "For the experts."

---

## Slide 79: Key Takeaways — Advanced

**Time**: ~2 min
**Talking points**:

- Double descent, NGBoost, proper scoring rules, counterfactual explanations, conformal prediction.
  **Transition**: "Here is the updated engine map."

---

## Slide 80: Kailash Engine Map — After Module 3

**Time**: ~2 min
**Talking points**:

- Show the full map: M1 (explore), M2 (experiment), M3 (train, tune, deploy).
- M4 preview: AutoMLEngine, EnsembleEngine, DriftMonitor, InferenceServer.
  **Transition**: "What comes next?"

---

## Slide 81: What Comes Next — Module 4

**Time**: ~3 min
**Talking points**:

- Unsupervised: K-means, spectral clustering, HDBSCAN, EM algorithm, PCA, UMAP.
- Production monitoring: DriftMonitor, EnsembleEngine, deep learning foundations, InferenceServer.
- Opening question for M4: "Credit Suisse's AML model had 99.9% accuracy and was completely useless."
  **Transition**: "A few more advanced slides for those who want to go deeper."

---

## Slides 82-84: Deep Dives (Reliability Diagram, MAS FEAT, Cross-Validation)

**Time**: ~6 min total

### Slide 82: Reliability Diagram

**Talking points**:

- How to read it: x-axis predicted probability, y-axis observed frequency. Diagonal = perfect.
- Common patterns: S-shaped below (overconfident, typical deep learning), S-shaped above (underconfident, typical Random Forests).

### Slide 83: Singapore Context — MAS FEAT Principles

**Talking points**:

- Fairness, Ethics, Accountability, Transparency. Map each to M3 engines.

### Slide 84: Cross-Validation

**Talking points**:

- K-fold: train on K-1 folds, evaluate on held-out fold, report mean and std.
- Time series: use expanding window. Kailash handles with cv_strategy="timeseries".

**[CAN SKIP ALL IF RUNNING SHORT]**

---

## Slides 85-86: SHAP Force Plot + Workflow Patterns

**Time**: ~4 min total
**Talking points**:

- Force plot: red pushes higher, blue pushes lower, wider = larger contribution.
- Workflow patterns: sequential, parallel fan-out, conditional branch, cyclic with convergence.

**[CAN SKIP IF RUNNING SHORT]**

---

## Slides 87-91: DataFlow Schema, Error Analysis, Interop, Ensembles, Common Pitfalls

**Time**: ~8 min total (or skip)
**Talking points**:

- DataFlow ModelVersion schema with governance fields (promoted_by, stage).
- Error analysis: look at worst predictions, find systematic patterns.
- Kailash interop: to_sklearn_input, from_sklearn_output for bridging ecosystems.
- Ensembles preview: blending, stacking, bagging, boosting. EnsembleEngine in M4.
- Common pitfalls table: data leakage, SMOTE before split, no calibration, no SHAP, no model card, no versioning.

**[CAN SKIP ALL IF RUNNING SHORT]**

---

## Slides 92-93: Proper Scoring Rules + Feature Importance Comparison (Advanced)

**Time**: ~4 min total
**Talking points**:

- Brier score decomposition: REL (calibration) - RES (resolution) + UNC (uncertainty).
- Log loss: more sensitive to confident wrong predictions.
- Feature importance comparison: gain (biased), permutation (affected by correlated features), SHAP (consistent), LIME (inconsistent). Recommendation: SHAP as primary tool.

**[CAN SKIP IF RUNNING SHORT]**

---

## Slide 94: Which Algorithm? — Decision Guide

**Time**: ~2 min
**Talking points**:

- Walk through the table: tabular < 100K rows = LightGBM/XGBoost, many categoricals = CatBoost, need interpretability = logistic regression, need uncertainty = NGBoost/conformal, huge data = LightGBM GOSS.
- "ModelSpec(algorithm='auto') benchmarks multiple algorithms and selects the best."

---

## Slide 95: From Model Registry to Full Governance

**Time**: ~2 min
**Talking points**:

- M3 today: ModelRegistry, quality gates, DataFlow, model cards.
- M6 future: PACT D/T/R, GovernanceEngine, operating envelopes, PactGovernedAgent.

---

## Slide 96: Assessment Preview

**Time**: ~2 min
**Talking points**:

- Quiz topics: bias-variance identification, L1 vs L2, critique SMOTE approach, read SHAP force plot, model lifecycle stage, debug WorkflowBuilder.
- "No pure recall. Every question requires applying to a scenario, reading engine output, or debugging code."

---

## Slide 97: Revisiting Zillow

**Time**: ~3 min
**Talking points**:

- Walk through the table: every Zillow gap mapped to today's solution and Kailash engine.
- "At the start, we asked how an accurate model loses $500 million. Now you know the answer — and how to prevent it."

---

## Slide 98: Final Provocation

**Time**: ~2 min
**Talking points**:

- Read aloud: "Your model is 95% accurate. Your business partner asks: 'Should we put it in production?' What questions do you ask before answering?"
- Let the room think. Possible answers: what is the class balance? Is it calibrated? What does SHAP show? Is there drift monitoring? What is the model card? Who approved it?
- "If you can answer that question well, you have understood Module 3."

---

## Slide 99: Closing

**Time**: ~1 min
**Talking points**:

- "Module 3 complete. Next: Module 4 — Advanced ML: Unsupervised Methods and Deep Learning."

**[FINAL Q&A — 5 min]**

---

## Timing Summary

| Section                                             | Slides    | Time                           |
| --------------------------------------------------- | --------- | ------------------------------ |
| Title + Recap + Opening Case                        | 1-8       | ~17 min                        |
| Foundations: Prediction, Overfitting, Bias-Variance | 9-17      | ~26 min                        |
| Regularisation + Gradient Boosting                  | 18-30     | ~30 min (incl. live coding)    |
| Class Imbalance + Calibration                       | 31-38     | ~19 min                        |
| SHAP, Explainability, Fairness                      | 39-46     | ~22 min                        |
| Workflow Orchestration + DataFlow                   | 47-54     | ~20 min (incl. live coding)    |
| Model Registry + HPO + Production Pipeline          | 55-66     | ~22 min (incl. live coding)    |
| Additional Theory (Elastic Net, CatBoost, etc.)     | 67-71     | ~10 min (most skippable)       |
| Lab + Discussions                                   | 72-76     | ~20 min                        |
| Synthesis + Closing                                 | 77-99     | ~15 min (deep dives skippable) |
| Q&A buffers                                         | scattered | ~13 min                        |
| **Total**                                           |           | **~180 min**                   |
