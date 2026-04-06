# Chapter 7: TrainingPipeline

## Overview

TrainingPipeline orchestrates the full model training workflow: validate data against a FeatureSchema, split into train/test sets, train a model specified by ModelSpec, evaluate with EvalSpec, and optionally register the result in ModelRegistry. It integrates with FeatureStore and ExperimentTracker, making it the central engine for supervised ML.

This chapter covers:

- Defining models with `ModelSpec` (class, hyperparameters, framework)
- Configuring evaluation with `EvalSpec` (metrics, split strategies, thresholds)
- Training with `pipeline.train()` and inspecting `TrainingResult`
- Split strategies: holdout, k-fold, walk-forward
- Minimum threshold gating (reject models that do not meet quality bars)
- Evaluating registered models on new data
- Retraining with fresh data
- Allowlisted model classes for security

## Prerequisites

| Requirement        | Details                               |
| ------------------ | ------------------------------------- |
| Python             | 3.10+                                 |
| kailash-ml         | `pip install kailash-ml`              |
| kailash (core SDK) | For `ConnectionManager`               |
| Prior chapters     | 5-6 (FeatureStore, ExperimentTracker) |
| Level              | Intermediate                          |

## Concepts

### ModelSpec

`ModelSpec` declares what to train. It contains:

- `model_class` -- A fully qualified class name (e.g., `"sklearn.ensemble.RandomForestClassifier"`)
- `hyperparameters` -- A dict of constructor arguments
- `framework` -- The ML framework (e.g., `"sklearn"`)
- `instantiate()` -- Creates a live model instance from the class name

Model classes are restricted to an allowlist of safe prefixes (sklearn, xgboost, lightgbm). Attempting to instantiate `os.system` or other dangerous classes raises `ValueError`.

### EvalSpec

`EvalSpec` declares how to evaluate. It contains:

- `metrics` -- List of metric names (`"accuracy"`, `"f1"`, `"r2"`, `"rmse"`)
- `split_strategy` -- How to split data: `"holdout"`, `"kfold"`, or `"walk_forward"`
- `test_size` -- Fraction for holdout/walk_forward
- `n_splits` -- Number of folds for kfold
- `min_threshold` -- Optional dict of metric -> minimum value. If the model fails to meet any threshold, it is not registered.

### Split Strategies

| Strategy       | Behavior                                   | Best For         |
| -------------- | ------------------------------------------ | ---------------- |
| `holdout`      | Random split into train/test               | Most datasets    |
| `kfold`        | K cross-validation folds, averaged metrics | Small datasets   |
| `walk_forward` | Sequential split (no shuffle)              | Time-series data |

### Threshold Gating

When `min_threshold` is set, the pipeline checks each metric against its threshold after training. If any metric falls below, `threshold_met` is `False` and the model is not registered in ModelRegistry. This prevents deploying underperforming models.

## Key API Reference

| Class / Method                                                          | Purpose                                                                                                  |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `TrainingPipeline(feature_store, registry)`                             | Create pipeline with FS and registry                                                                     |
| `pipeline.train(data, schema, model_spec, eval_spec, experiment_name)`  | Async. Train, evaluate, optionally register                                                              |
| `pipeline.evaluate(model_name, version, data, schema, eval_spec)`       | Async. Evaluate a registered model on new data                                                           |
| `pipeline.retrain(model_name, schema, model_spec, eval_spec, data)`     | Async. Retrain with fresh data                                                                           |
| `ModelSpec(model_class, hyperparameters, framework)`                    | What to train                                                                                            |
| `EvalSpec(metrics, split_strategy, test_size, n_splits, min_threshold)` | How to evaluate                                                                                          |
| `TrainingResult`                                                        | Fields: `metrics`, `training_time_seconds`, `data_shape`, `threshold_met`, `registered`, `model_version` |

## Code Walkthrough

### Step 1: Set Up Infrastructure

```python
from kailash.db.connection import ConnectionManager
from kailash_ml.engines.feature_store import FeatureStore
from kailash_ml.engines.model_registry import LocalFileArtifactStore, ModelRegistry
from kailash_ml.engines.training_pipeline import TrainingPipeline, ModelSpec, EvalSpec

conn = ConnectionManager("sqlite:///:memory:")
await conn.initialize()

artifact_store = LocalFileArtifactStore("/tmp/artifacts")
registry = ModelRegistry(conn, artifact_store)
fs = FeatureStore(conn)
await fs.initialize()

pipeline = TrainingPipeline(feature_store=fs, registry=registry)
```

### Step 2: Define Schema, ModelSpec, and EvalSpec

```python
from kailash_ml import FeatureField, FeatureSchema

schema = FeatureSchema(
    name="binary_classification",
    features=[
        FeatureField(name="feature_a", dtype="float64"),
        FeatureField(name="feature_b", dtype="float64"),
        FeatureField(name="feature_c", dtype="float64"),
    ],
    entity_id_column="entity_id",
)

model_spec = ModelSpec(
    model_class="sklearn.ensemble.RandomForestClassifier",
    hyperparameters={"n_estimators": 20, "random_state": 42},
    framework="sklearn",
)

eval_spec = EvalSpec(
    metrics=["accuracy", "f1"],
    split_strategy="holdout",
    test_size=0.2,
)
```

### Step 3: Train

```python
result = await pipeline.train(
    data=df,
    schema=schema,
    model_spec=model_spec,
    eval_spec=eval_spec,
    experiment_name="tutorial_experiment",
)

assert result.threshold_met is True
assert result.registered is True
assert "accuracy" in result.metrics
assert result.model_version.stage == "staging"
```

### Step 4: Threshold Gating

```python
strict_eval = EvalSpec(
    metrics=["accuracy"],
    split_strategy="holdout",
    test_size=0.2,
    min_threshold={"accuracy": 0.99},  # Very high
)

strict_result = await pipeline.train(
    data=df, schema=schema,
    model_spec=model_spec,
    eval_spec=strict_eval,
    experiment_name="strict_experiment",
)

if not strict_result.threshold_met:
    assert strict_result.registered is False
    assert strict_result.model_version is None
```

### Step 5: K-Fold and Walk-Forward Splits

```python
# K-fold
kfold_result = await pipeline.train(
    data=df, schema=schema,
    model_spec=model_spec,
    eval_spec=EvalSpec(metrics=["accuracy"], split_strategy="kfold", n_splits=5),
    experiment_name="kfold_experiment",
)

# Walk-forward (time-series)
wf_result = await pipeline.train(
    data=df, schema=schema,
    model_spec=model_spec,
    eval_spec=EvalSpec(metrics=["accuracy"], split_strategy="walk_forward", test_size=0.2),
    experiment_name="walk_forward_experiment",
)
```

### Step 6: Evaluate on New Data

```python
eval_metrics = await pipeline.evaluate(
    model_name="tutorial_experiment",
    version=result.model_version.version,
    data=new_data,
    schema=schema,
    eval_spec=EvalSpec(metrics=["accuracy"]),
)
assert "accuracy" in eval_metrics
```

### Step 7: Retrain

```python
retrain_result = await pipeline.retrain(
    model_name="tutorial_experiment",
    schema=schema,
    model_spec=model_spec,
    eval_spec=eval_spec,
    data=new_data,
)
```

## Common Mistakes

| Mistake                                         | What Happens          | Fix                                        |
| ----------------------------------------------- | --------------------- | ------------------------------------------ |
| Missing columns in data                         | `ValueError`          | Ensure all schema features + target exist  |
| Non-allowlisted model class (e.g., `os.system`) | `ValueError`          | Use sklearn, xgboost, or lightgbm classes  |
| Invalid split strategy                          | `ValueError`          | Use `holdout`, `kfold`, or `walk_forward`  |
| Setting threshold too high                      | Model never registers | Set realistic thresholds based on baseline |
| Forgetting `entity_id_column` in data           | Column mismatch       | Include it even though it is not a feature |

## Exercises

1. **Split strategy comparison**: Train the same model with holdout, kfold, and walk_forward. How do the metrics differ? Which gives the most optimistic result?

2. **Threshold tuning**: Start with `min_threshold={"accuracy": 0.5}` and increase until the model fails to register. What is the boundary?

3. **Retrain workflow**: Train a model, evaluate on new data, then retrain with the new data. Does the retrained model improve?

## Key Takeaways

- TrainingPipeline integrates FeatureStore, ModelRegistry, and ExperimentTracker into a single training workflow.
- ModelSpec declares what to train; EvalSpec declares how to evaluate.
- Three split strategies (holdout, kfold, walk_forward) cover different data scenarios.
- Threshold gating prevents underperforming models from being registered.
- Model classes are restricted to safe allowlists for security.
- `evaluate()` and `retrain()` support the full model lifecycle.

## Next Chapter

Chapter 8 covers **HyperparameterSearch** -- the engine that optimizes hyperparameters using grid, random, and Bayesian search strategies.
