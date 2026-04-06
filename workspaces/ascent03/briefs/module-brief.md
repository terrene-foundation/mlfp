# Module 3: Supervised ML — Theory to Production

**Duration**: 7 hours  
**Kailash**: Core SDK (WorkflowBuilder, LocalRuntime), DataFlow, kailash-ml (TrainingPipeline, HyperparameterSearch, ModelRegistry)  
**Scaffolding**: 50%

## Lecture Topics

### 3A: Supervised ML Theory (90 min)
- Bias-variance decomposition: **full derivation for squared loss**, key steps for log loss (KL divergence form, Domingos 2000), why the decomposition differs for classification
- **Double descent** (P0): Belkin et al. 2019 — classical U-curve + interpolation threshold + overparameterized regime. Benign overfitting: in high dimensions, noise subspace is nearly orthogonal to signal. Why modern deep learning works despite interpolating training data.
- **Bias-variance for ensembles**: bagging variance formula (ρσ² + (1-ρ)σ²/B — why decorrelation matters), boosting reduces bias (residual fitting), stacking meta-learner optimizes trade-off
- Regularization: L1/L2 geometry, **Bayesian interpretation** (P0): L2 = MAP with Gaussian prior, L1 = MAP with Laplace prior (derive from Module 1's MAP foundation), elastic net intuition, dropout as approximate Bayesian inference (Gal & Ghahramani 2016), early stopping as implicit L2
- Gradient boosting internals: **XGBoost 2nd-order Taylor full derivation** (P0) — objective expansion, optimal leaf weight w_j* = -Σg_i/(Σh_i + λ), gain formula. LightGBM: histogram splits (O(K) vs O(n log n)), **GOSS** (gradient-based sampling, correction factor). CatBoost: ordered boosting for categoricals. DART: mention only.
- Class imbalance: SMOTE failure taxonomy (P0: Lipschitz violation, noisy minority, high-dimensional collapse — the "SMOTE Paradox": 92% citation rate, 6% production use), cost-sensitive learning (**derive optimal threshold from cost matrix**), **Focal Loss derivation** (from cross-entropy, show γ parameter effect)

### 3B: Model Evaluation & Interpretability (90 min)
- Metrics: precision-recall trade-off (F-beta), **AUC-ROC vs AUC-PR worked through with numbers** (show why AUC-ROC misleads at 0.1% positive rate — Davis & Goadrich 2006), log loss, Brier score
- **Proper scoring rules** (P0): Brier score IS proper (incentivizes calibrated probabilities), accuracy IS NOT (incentivizes overconfident predictions). Brier decomposition: reliability + resolution - uncertainty.
- Calibration: Platt scaling, isotonic regression, reliability diagrams, ECE, **calibration theory** (reliability, resolution, sharpness)
- **Conformal prediction** (P1): distribution-free prediction intervals with guaranteed coverage — explosive 2025-2026 adoption for uncertainty quantification
- SHAP: **derive Shapley axioms** (efficiency, symmetry, dummy, linearity — why these axioms uniquely determine the Shapley value), TreeSHAP (polynomial-time algorithm, key insight), KernelSHAP, interaction values, dependence plots
- LIME (explain kernel and locality), PDP, ALE, ICE plots, **counterfactual explanations** (Wachter et al., DiCE)
- Model cards: Mitchell et al. template with filled example

### 3C: Workflow Orchestration (30 min)
- WorkflowBuilder: nodes, connections, `runtime.execute(workflow.build())`
- DataFlow: @db.model, db.express for CRUD
- Persisting ML artifacts in database

## Lab Exercises (6)

1. **Gradient boosting deep dive**: XGBoost vs LightGBM vs CatBoost on Singapore credit data — learning curves, hyperparameter sensitivity. Use **kailash_ml.interop.to_sklearn_input()** for data conversion.
2. **Class imbalance workshop**: SMOTE vs cost-sensitive vs Focal Loss vs threshold optimization. Calibration before/after.
3. **SHAP interpretability**: Full SHAP analysis — summary, dependence, interaction plots. Use **ModelVisualizer** for SHAP plots. "Which features drive approval/rejection?" — frame as governance question (fairness across protected attributes).
4. **Workflow orchestration**: Kailash workflow: load → preprocess → train → evaluate → persist to DataFlow. Explicitly define **@db.model** for model evaluation results. Use **db.express.create()** to persist. Introduce **ModelSignature** (input/output schema contract for trained models).
5. **HyperparameterSearch + ModelRegistry**: Use **SearchSpace, SearchConfig, ParamDistribution** for Bayesian optimization. Register best model → **promote staging → production** as governance gate: "A model cannot reach production without explicit promotion." EATP reference: model provenance.
6. **End-to-end pipeline**: Complete supervised ML pipeline with workflow, persistence, **model card generation** (Mitchell et al. template with filled example). **Conformal prediction** intervals for uncertainty quantification.

## Datasets
- **Singapore Credit Scoring** (synthetic): 100K apps, 12% default, 45 features, protected attributes, leakage trap, 30% missing income
- **Lending Club**: 300K+ records, 150 features, real-world messiness

## Key Patterns
```python
from kailash.workflow.builder import WorkflowBuilder
from kailash.runtime import LocalRuntime
from kailash_ml.engines.training_pipeline import TrainingPipeline, ModelSpec, EvalSpec

runtime = LocalRuntime()
results, run_id = runtime.execute(workflow.build())  # MUST use .build()

model_spec = ModelSpec(model_class="lightgbm.LGBMClassifier", hyperparameters={...})
eval_spec = EvalSpec(metrics=["accuracy", "f1", "auc_roc"], test_size=0.2)
```

## Quiz Topics
- Bias-variance: "Which model has higher bias? Higher variance?"
- L1 vs L2: "Why does L1 produce sparse solutions?"
- SHAP: "Interpret this SHAP summary plot"
- Calibration: "AUC=0.95 but ECE=0.15. Production-ready?"
- Debug: "What's wrong with `runtime.execute(workflow)`?" (missing .build())

## Deck Opening Case
**Zillow iBuyer $500M write-off** — their models were accurate but uncalibrated. They consistently overestimated purchase prices. Calibration is the difference between a model that scores well and a model that makes money.
