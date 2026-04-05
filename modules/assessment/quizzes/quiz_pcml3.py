# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 3 — AI-Resilient Assessment Questions

Supervised ML — Theory to Production
Covers: bias-variance, regularisation, boosting, class imbalance, SHAP,
        WorkflowBuilder, DataFlow, ModelRegistry, TrainingPipeline
"""

QUIZ = {
    "module": "ASCENT3",
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
                "db = DataFlow('sqlite:///ascent03_results.db')\n"
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
                "This pattern mirrors how ExperimentTracker and FeatureStore work in ASCENT2."
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
                "    db = DataFlow('sqlite:///ascent03_results.db')\n"
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
