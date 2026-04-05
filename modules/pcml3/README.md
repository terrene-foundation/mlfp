# Module 3: Supervised ML — Theory to Production

**Kailash**: Core SDK, DataFlow, kailash-ml (TrainingPipeline, HyperparameterSearch, ModelRegistry) | **Scaffolding**: 50%

## Lecture (3h)
- **3A** Supervised ML Theory: bias-variance decomposition, regularization (L1/L2 geometry), gradient boosting internals (XGBoost/LightGBM/CatBoost), ensemble theory, class imbalance (SMOTE failures, Focal Loss)
- **3B** Evaluation & Interpretability: precision-recall trade-off, AUC-ROC vs AUC-PR, calibration (Platt/isotonic, ECE), SHAP (TreeSHAP, KernelSHAP), LIME, PDP, ALE, model cards
- **3C** Workflow Orchestration: WorkflowBuilder, LocalRuntime, DataFlow (@db.model, db.express)

## Lab (3h) — 6 Exercises
1. Gradient boosting deep dive: XGBoost vs LightGBM vs CatBoost comparison
2. Class imbalance workshop: SMOTE vs cost-sensitive vs Focal Loss with calibration
3. SHAP interpretability: full analysis with summary, dependence, interaction plots
4. Kailash workflow: load → preprocess → train → evaluate → persist to DataFlow
5. HyperparameterSearch + ModelRegistry: Bayesian optimization → staging → production
6. End-to-end pipeline with model card generation

## Datasets
Singapore Credit Scoring (100K apps, 12% default, protected attributes, leakage trap), Lending Club (300K+)
