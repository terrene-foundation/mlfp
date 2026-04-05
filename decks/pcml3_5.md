---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 3.5: Workflow Orchestration

### Module 3: Supervised ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Build reproducible ML workflows with Kailash `WorkflowBuilder`
- Connect nodes for data loading, preprocessing, training, and evaluation
- Execute workflows with `runtime.execute(workflow.build())`
- Understand the node-connection-workflow pattern

---

## Recap: Lesson 3.4

- SHAP provides global and local model explanations grounded in game theory
- LIME offers quick local explanations for any model type
- Fairness requires checking bias across protected groups
- Proxy features can encode protected attributes even when removed

---

## Why Workflows?

Scripts break. Notebooks are hard to reproduce. Workflows solve both.

```
Script approach:
  load_data()           ← runs once, breaks if data changes
  preprocess()          ← coupled to load_data output
  train_model()         ← coupled to preprocess output
  evaluate()            ← coupled to train_model output
  → Fragile, hard to test, no retry, no parallelism

Workflow approach:
  [Load] → [Preprocess] → [Train] → [Evaluate]
  → Each step is independent, testable, retriable
  → The framework handles execution order and data flow
```

---

## The Kailash Workflow Model

```
WorkflowBuilder creates a DAG (Directed Acyclic Graph):

  ┌──────┐    ┌────────────┐    ┌───────┐    ┌──────────┐
  │ Load │───→│ Preprocess │���──→│ Train │───→│ Evaluate │
  └──────┘    └────────────┘    └───────┘    └──────────┘
     Node        Node              Node          Node
              Connected by edges (data flows along arrows)
```

Three concepts: **Nodes** (steps), **Connections** (data flow), **Workflow** (the graph).

---

## Your First Workflow

```python
from kailash import WorkflowBuilder, Runtime

workflow = WorkflowBuilder()

# Add nodes
load = workflow.add_node("LoadCSV", params={"path": "data/hdb.csv"})
preprocess = workflow.add_node("Transform", params={"operations": ["scale"]})
train = workflow.add_node("TrainModel", params={"algorithm": "xgboost"})

# Connect nodes (data flows from output to input)
workflow.connect(load, preprocess)
workflow.connect(preprocess, train)

# Build and execute
runtime = Runtime()
result = runtime.execute(workflow.build())
```

---

## Node Types

Kailash SDK provides 140+ built-in nodes across categories:

| Category      | Examples                                     |
| ------------- | -------------------------------------------- |
| **Data**      | LoadCSV, LoadJSON, DataFilter, DataJoin      |
| **Transform** | Scale, Encode, Impute, FeatureEngineer       |
| **AI/ML**     | TrainModel, Predict, Evaluate, CrossValidate |
| **Logic**     | Conditional, Loop, Parallel, Aggregate       |
| **API**       | HTTPRequest, WebhookTrigger                  |
| **File**      | ReadFile, WriteFile, ParsePDF                |

You compose them like building blocks.

---

## Parameterising Nodes

```python
# Static parameters (set at build time)
train_node = workflow.add_node("TrainModel", params={
    "algorithm": "lightgbm",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "task": "regression",
})

# Dynamic parameters (passed at runtime)
load_node = workflow.add_node("LoadCSV", params={
    "path": "{{input.data_path}}",   # resolved at execution
})

result = runtime.execute(
    workflow.build(),
    inputs={"data_path": "data/hdb_2025.csv"},
)
```

---

## Branching Workflows

```python
# Parallel branches
workflow = WorkflowBuilder()

load = workflow.add_node("LoadCSV", params={"path": "data/hdb.csv"})
preprocess = workflow.add_node("Transform")

# Branch 1: XGBoost
train_xgb = workflow.add_node("TrainModel", params={"algorithm": "xgboost"})
eval_xgb = workflow.add_node("Evaluate", params={"name": "xgboost"})

# Branch 2: LightGBM
train_lgb = workflow.add_node("TrainModel", params={"algorithm": "lightgbm"})
eval_lgb = workflow.add_node("Evaluate", params={"name": "lightgbm"})

workflow.connect(load, preprocess)
workflow.connect(preprocess, train_xgb)
workflow.connect(preprocess, train_lgb)
workflow.connect(train_xgb, eval_xgb)
workflow.connect(train_lgb, eval_lgb)
```

---

## Branching Visual

```
                    ┌───────────┐    ┌──────────┐
               ┌───→│ Train XGB │───→│ Eval XGB │
┌──────┐  ┌────┤    └───────────┘    └──────────┘
│ Load │─→│Prep│
└──────┘  └────┤    ┌───────────┐    ┌──────────┐
               └───→│ Train LGB │───→│ Eval LGB │
                    └───────────┘    └──────────┘

Both branches execute in parallel automatically.
```

---

## Conditional Execution

```python
workflow = WorkflowBuilder()

load = workflow.add_node("LoadCSV")
check = workflow.add_node("Conditional", params={
    "condition": "{{data.rows}} > 1000",
})

# If large dataset: use LightGBM
train_large = workflow.add_node("TrainModel", params={
    "algorithm": "lightgbm",
})

# If small dataset: use Ridge regression
train_small = workflow.add_node("TrainModel", params={
    "algorithm": "ridge",
})

workflow.connect(load, check)
workflow.connect(check, train_large, condition="true")
workflow.connect(check, train_small, condition="false")
```

---

## Error Handling in Workflows

```python
# Retry on failure
train = workflow.add_node("TrainModel", params={
    "algorithm": "xgboost",
}, config={
    "retry_count": 3,
    "retry_delay": 5,  # seconds
})

# Fallback node
fallback = workflow.add_node("TrainModel", params={
    "algorithm": "ridge",  # simpler model as fallback
})

workflow.connect(train, evaluate)
workflow.connect(train, fallback, on_error=True)
workflow.connect(fallback, evaluate)
```

---

## Workflow Execution Modes

```python
runtime = Runtime()

# Synchronous (wait for result)
result = runtime.execute(workflow.build())

# Async execution
import asyncio

async def run_workflow():
    result = await runtime.execute_async(workflow.build())
    return result

result = asyncio.run(run_workflow())
```

---

## Inspecting Results

```python
result = runtime.execute(workflow.build())

# Overall status
print(f"Status: {result.status}")          # SUCCESS / FAILED
print(f"Duration: {result.duration:.2f}s")

# Per-node results
for node_name, node_result in result.node_results.items():
    print(f"  {node_name}: {node_result.status} ({node_result.duration:.2f}s)")

# Access specific node output
model_output = result.get_output("TrainModel")
print(f"Model R²: {model_output['metrics']['r2']:.3f}")
```

---

## ML Pipeline Workflow Pattern

```python
from kailash import WorkflowBuilder, Runtime

workflow = WorkflowBuilder()

# Standard ML pipeline as a workflow
load = workflow.add_node("LoadCSV", params={"path": "data/hdb.csv"})
explore = workflow.add_node("DataExplorer")
preprocess = workflow.add_node("PreprocessingPipeline", params={
    "missing_strategy": {"numeric": "median"},
})
engineer = workflow.add_node("FeatureEngineer", params={
    "strategies": ["arithmetic", "temporal"],
})
train = workflow.add_node("TrainModel", params={
    "algorithm": "lightgbm", "task": "regression",
})
evaluate = workflow.add_node("Evaluate")

# Chain all nodes
for a, b in zip(
    [load, explore, preprocess, engineer, train],
    [explore, preprocess, engineer, train, evaluate]
):
    workflow.connect(a, b)

result = Runtime().execute(workflow.build())
```

---

## Exercise Preview

**Exercise 3.5: HDB Price Prediction Workflow**

You will:

1. Build a multi-node workflow with `WorkflowBuilder`
2. Add branching for multiple model comparison
3. Handle errors with retry and fallback nodes
4. Execute and inspect per-node results

Scaffolding level: **Moderate (~50% code provided)**

---

## Common Pitfalls

| Mistake                       | Fix                                                |
| ----------------------------- | -------------------------------------------------- |
| Circular connections          | Workflows must be DAGs (no cycles in basic mode)   |
| Forgetting to call `.build()` | `runtime.execute(workflow.build())` not `workflow` |
| Hardcoded paths in nodes      | Use `{{input.variable}}` for runtime parameters    |
| Not checking per-node status  | A workflow can succeed even if a branch failed     |
| Overly complex workflows      | Start simple, add complexity incrementally         |

---

## Summary

- Kailash `WorkflowBuilder` creates reproducible ML pipelines as DAGs
- Nodes are steps; connections define data flow
- Branching enables parallel model comparison
- Conditional nodes route execution based on data characteristics
- `runtime.execute(workflow.build())` runs the entire pipeline

---

## Next Lesson

**Lesson 3.6: DataFlow and Persistence**

We will learn:

- Persisting models and data with Kailash DataFlow
- Database models with `@db.model` decorators
- Quick queries with `db.express`
