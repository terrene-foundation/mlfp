---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 3.8: Production Pipeline Project

### Module 3: Supervised ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Combine all Module 3 components into an end-to-end production pipeline
- Orchestrate data loading, training, evaluation, and persistence as a workflow
- Apply best practices for reproducible ML in production
- Complete the Module 3 capstone project

---

## Recap: Lesson 3.7

- `HyperparameterSearch` automates tuning with Bayesian optimisation
- `ModelRegistry` manages versioned models through lifecycle stages
- Compare candidates on metrics before promoting to production
- HPO-to-registry pipeline ensures reproducible model selection

---

## Module 3 Journey

```
3.1  Bias-variance, L1/L2 regularisation
3.2  Gradient boosting, TrainingPipeline
3.3  Class imbalance, calibration
3.4  SHAP, LIME, fairness
3.5  WorkflowBuilder, node-connection pattern
3.6  DataFlow, @db.model, persistence
3.7  HyperparameterSearch, ModelRegistry
3.8  Production pipeline capstone  ← YOU ARE HERE
```

Today we assemble everything into a production-grade system.

---

## Production Pipeline Architecture

```
┌──────────┐   ┌────────────┐   ┌─────────────┐   ┌────────────┐
│  DataFlow │──→│ Feature    │──→│ Training    │──→│  Model     │
│  (Load)   │   │ Engineer   │   │ Pipeline    │   │  Registry  │
└──────────┘   └────────────┘   └─────────────┘   └────────────┘
                                       │
                                       ▼
┌──────────┐   ┌────────────┐   ┌─────────────┐
│ DataFlow  │◀──│ Calibrate  │◀──│  Evaluate   │
│ (Persist) │   │ + Explain  │   │  + Fairness │
└──────────┘   └────────────┘   └─────────────┘
```

---

## Step 1: Data Pipeline

```python
from kailash import WorkflowBuilder, Runtime
from kailash_ml import DataExplorer, PreprocessingPipeline, FeatureEngineer
from shared.data_loader import ASCENTDataLoader

loader = ASCENTDataLoader()
df = loader.load("ascent03", "hdbprices.csv")

# Profile
explorer = DataExplorer()
explorer.configure(dataset=df, target_column="price")
report = explorer.run()
print(f"Alerts: {len(report.alerts)}")

# Clean
pipeline = PreprocessingPipeline()
pipeline.configure(
    dataset=df, target_column="price",
    missing_strategy={"numeric": "median", "categorical": "mode"},
    outlier_strategy="iqr", outlier_action="clip",
)
df_clean = pipeline.run()
```

---

## Step 2: Feature Engineering

```python
from kailash_ml import FeatureEngineer, FeatureStore

engineer = FeatureEngineer()
engineer.configure(
    dataset=df_clean, target_column="price",
    strategies=["arithmetic", "temporal", "interaction"],
    selection_method="mutual_information",
    n_select=25,
)
df_features = engineer.generate()
df_selected = engineer.select()

# Register in feature store
store = FeatureStore()
store.configure(storage_path="./feature_store")
store.register(
    name="hdb_production_features",
    features=df_selected,
    version="1.0",
)
```

---

## Step 3: Hyperparameter Search

```python
from kailash_ml import HyperparameterSearch

search = HyperparameterSearch()
search.configure(
    dataset=df_selected, target_column="price",
    task="regression", algorithm="lightgbm",
    strategy="bayesian", n_trials=50,
    cv_folds=5, metric="rmse", direction="minimize",
    search_space={
        "n_estimators": {"type": "int", "low": 100, "high": 2000},
        "learning_rate": {"type": "log_float", "low": 0.005, "high": 0.3},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "num_leaves": {"type": "log_int", "low": 8, "high": 256},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    },
)
search_result = search.run()
print(f"Best RMSE: ${search_result.best_score:,.0f}")
```

---

## Step 4: Evaluate and Explain

```python
import shap

# Evaluate on holdout
model = search_result.best_model
y_pred = model.predict(X_test)

# SHAP explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Fairness check
from kailash_ml import TrainingPipeline
# Compute metrics across protected groups
for group in df_test["town"].unique():
    mask = df_test["town"] == group
    group_rmse = ((y_test[mask] - y_pred[mask])**2).mean()**0.5
    print(f"{group}: RMSE=${group_rmse:,.0f}")
```

---

## Step 5: Register and Persist

```python
from kailash_ml import ModelRegistry
from kailash_dataflow import DataFlow

# Register model
registry = ModelRegistry()
registry.configure(storage_path="./model_registry")
registry.register(
    name="hdb_production_model",
    model=search_result.best_model,
    version="1.0.0",
    metrics=search_result.best_metrics,
    params=search_result.best_params,
)

# Persist predictions
db = DataFlow()
db.configure(database="sqlite:///ascent03_production.db")
# ... save predictions using @db.model pattern from 3.6
```

---

## As a Workflow

```python
from kailash import WorkflowBuilder, Runtime

workflow = WorkflowBuilder()

load = workflow.add_node("LoadCSV", params={"path": "data/hdb.csv"})
profile = workflow.add_node("DataExplorer")
clean = workflow.add_node("PreprocessingPipeline", params={
    "missing_strategy": {"numeric": "median"},
})
engineer_node = workflow.add_node("FeatureEngineer", params={
    "strategies": ["arithmetic", "temporal"],
})
search_node = workflow.add_node("HyperparameterSearch", params={
    "algorithm": "lightgbm", "n_trials": 50,
})
register = workflow.add_node("ModelRegistry", params={
    "name": "hdb_production_model",
})

for a, b in zip(
    [load, profile, clean, engineer_node, search_node],
    [profile, clean, engineer_node, search_node, register],
):
    workflow.connect(a, b)

result = Runtime().execute(workflow.build())
```

---

## Production Checklist

```
��� Data profiled with DataExplorer (no silent quality issues)
□ Preprocessing pipeline saved for replay on new data
□ Features registered in FeatureStore with version
□ HPO completed with sufficient trials (50+)
□ Model evaluated on holdout with appropriate metrics
□ SHAP explanations computed and reviewed
□ Fairness checked across relevant groups
□ Model registered in ModelRegistry with metadata
□ Predictions persisted to DataFlow
□ Workflow reproducible via WorkflowBuilder
```

---

## Exercise Preview

**Exercise 3.8: Module 3 Capstone**

You will:

1. Build an end-to-end production ML pipeline
2. Profile, clean, engineer features, tune, train, evaluate
3. Explain with SHAP and check fairness
4. Register in ModelRegistry and persist to DataFlow
5. Orchestrate as a reproducible workflow

Scaffolding level: **Moderate (~50% code provided)**

---

## Common Pitfalls

| Mistake                                | Fix                                                   |
| -------------------------------------- | ----------------------------------------------------- |
| Skipping data profiling in production  | Always profile -- data quality changes                |
| Different preprocessing train vs serve | Save and replay PreprocessingPipeline config          |
| No model versioning                    | Every deployed model needs a version in ModelRegistry |
| Workflow not reproducible              | Use WorkflowBuilder, not scripts                      |
| No fairness evaluation                 | Check metrics across relevant groups before deploying |

---

## Module 3 Summary

| Lesson | Key Skills                          |
| ------ | ----------------------------------- |
| 3.1    | Bias-variance, L1/L2 regularisation |
| 3.2    | Gradient boosting, TrainingPipeline |
| 3.3    | Class imbalance, calibration        |
| 3.4    | SHAP, LIME, fairness                |
| 3.5    | WorkflowBuilder, orchestration      |
| 3.6    | DataFlow, persistence               |
| 3.7    | HyperparameterSearch, ModelRegistry |
| 3.8    | Production pipeline capstone        |

---

## What Comes Next

**Module 4: Advanced ML**

- Unsupervised learning: clustering, dimensionality reduction
- NLP: text processing, topic modelling
- Deep learning: PyTorch, CNNs, ONNX
- Anomaly detection, drift monitoring, inference serving

You can now build production ML systems. Next, we expand the toolkit.
