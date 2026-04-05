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
                "B) Rising CV MSE from degree 7 signals overfitting — the model has learned training-set noise rather than the true function. The exercise recommends applying Ridge (L2) regularisation, which shrinks polynomial coefficients and lets higher-degree models generalise",
                "C) The CV MSE rise is a numerical artefact of cross-validation; use the test set instead",
                "D) Rising CV MSE means the polynomial features are correlated; use PCA first",
            ],
            "answer": "B",
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
                "B) Lasso — its L1 penalty drives many coefficients to exactly zero, producing a sparse model with only 8 input variables. This sparsity makes the model interpretable: the regulator sees exactly which features drive creditworthiness decisions, satisfying MAS explainability requirements",
                "C) ElasticNet — the combination of L1 and L2 is always best for regulatory models",
                "D) Neither — neural networks are required for credit scoring in Singapore",
            ],
            "answer": "B",
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
                "A) 'Your application was declined due to our credit scoring model scoring below threshold.'",
                "B) 'Your application was declined primarily because of your high debt-to-income ratio and high credit utilisation, which together indicate elevated default risk based on your credit profile.'",
                "C) 'Your application was declined; the SHAP values for your features were 0.42 and 0.31.'",
                "D) 'Your application was declined because of your employment history and income level.'",
            ],
            "answer": "B",
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
                "B) ROC-AUC is insensitive to class ratio because it measures rank order across all thresholds. SMOTE generates synthetic minority samples that blur the decision boundary — Average Precision penalises this more harshly because it focuses on precision at high-recall operating points where the imbalance is most severe",
                "C) SMOTE always improves precision; the result suggests a bug in the implementation",
                "D) Average Precision is not meaningful for credit scoring; use F1 instead",
            ],
            "answer": "B",
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
                "B) Isotonic regression — with 50,000 samples there is sufficient data to fit a flexible, non-parametric monotonic calibration curve; it can handle the bimodal overconfidence without assuming a parametric form. Platt scaling assumes a sigmoid shape which may not match this distribution",
                "C) Neither — overconfidence requires retraining from scratch with temperature scaling",
                "D) Platt scaling — isotonic regression overfits on datasets larger than 10,000 samples",
            ],
            "answer": "B",
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
                "A) WorkflowBuilder should be imported from kailash.workflow",
                "B) runtime.execute() receives the WorkflowBuilder object directly — it must be called with workflow.build() to compile the graph first: runtime.execute(workflow.build())",
                "C) add_connection() argument order is wrong; it should be ('train', 'preprocess', ...)",
                "D) LocalRuntime should be AsyncLocalRuntime for ML workflows",
            ],
            "answer": "B",
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
                "B) DataFlow CreateNode parameters must be flat (not nested under a 'data' key) — the correct form passes field values directly at the top level of the config dict",
                "C) roc_auc cannot be a float in a DataFlow node; it must be cast to str first",
                "D) workflow.add_node() requires a fourth positional argument for connections",
            ],
            "answer": "B",
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
                "B) The primary key field must be named 'id', not 'primary_key'; and timestamps should use datetime, not str — DataFlow auto-manages created_at as datetime if the field is omitted entirely",
                "C) @db.model decorator must include the table name as a string argument",
                "D) float is not a supported DataFlow field type; use Decimal",
            ],
            "answer": "B",
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
                "A) WorkflowBuilder — it provides better error handling for single-record operations",
                "B) db.express — single-record CRUD is exactly the use case db.express is designed for. WorkflowBuilder adds ~23x overhead from graph construction and validation for a single write operation, violating the principle of using the highest appropriate abstraction layer",
                "C) Both are equivalent; use whichever is more familiar",
                "D) WorkflowBuilder — db.express cannot be used inside async functions",
            ],
            "answer": "B",
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
                "B) The metric improvement (0.847 vs 0.831) looks promising, but before promotion you must run a calibration check (reliability diagram) to ensure predicted probabilities are accurate, and compare the confusion matrix at the operational threshold — a 0.016 AUC improvement could hide a precision/recall trade-off shift that matters for loan approval decisions",
                "C) No — production models should never be replaced without a minimum 5% improvement",
                "D) Yes — brier_score < 0.1 confirms the model is well-calibrated, no further checks needed",
            ],
            "answer": "B",
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
                "B) ModelRegistry() without a connection or file path creates an in-memory store that is destroyed when the process ends — each script creates a fresh empty registry; pass a shared ConnectionManager or file path to persist across processes",
                "C) register() requires await; the model was never actually stored",
                "D) ModelRegistry is session-scoped; both scripts must be in the same Python session",
            ],
            "answer": "B",
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
                "B) workflow.build() compiles and validates the graph; LocalRuntime executes nodes in topological order (preprocess → train → evaluate); TrainingPipeline returns a trained model artifact; db.express.create() persists the EvalResult using the @db.model schema — all four are specific to the Module 3 exercises",
                "C) workflow.build() → runtime.execute() → model saved to disk → manual database insert",
                "D) workflow.build() → LocalRuntime forks a subprocess per node → results merged at end",
            ],
            "answer": "B",
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
