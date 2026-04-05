---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 5.6: ML Agent Pipeline

### Module 5: LLMs and Agents

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Describe the six specialised ML agents and their roles
- Orchestrate an agent-driven ML lifecycle
- Combine ML engines with Kaizen agents for automated analysis
- Build an end-to-end ML pipeline powered by agents

---

## Recap: Lesson 5.5

- MCP standardises tool discovery and execution
- Servers register tools (actions) and resources (data)
- Transports: stdio for local, SSE for network
- Agents discover tools at runtime via MCP protocol

---

## The ML Lifecycle as Agent Tasks

```
Traditional ML pipeline (manual):
  Data scientist profiles data → engineers features →
  selects model → tunes hyperparameters → evaluates → deploys

Agent-driven ML pipeline (automated):
  DataProfilerAgent → FeatureAgent → ModelSelectionAgent →
  TuningAgent → EvaluationAgent → DeploymentAgent

Each agent is a specialist with access to Kailash ML engines.
```

---

## The Six ML Agents

```
┌─────────────────┐   ┌──────────────┐   ┌──────────────────┐
│ DataProfiler    │──→│ Feature      │──→│ ModelSelection   │
│ Agent           │   │ Agent        │   │ Agent            │
│                 │   │              │   │                  │
│ Uses:           │   │ Uses:        │   │ Uses:            │
│ DataExplorer    │   │ FeatureEng.  │   │ TrainingPipeline │
│ AlertConfig     │   │ FeatureStore │   │ AutoMLEngine     │
└─────────────────┘   └──────────────┘   └──────────────────┘
         │                   │                    │
         ▼                   ▼                    ▼
┌─────────────────┐   ┌──────────────┐   ┌──────────────────┐
│ Tuning          │──→│ Evaluation   │──→│ Deployment       │
│ Agent           │   │ Agent        │   │ Agent            │
│                 │   │              │   │                  │
│ Uses:           │   │ Uses:        │   │ Uses:            │
│ HPSearch        │   │ SHAP/LIME    │   │ ModelRegistry    │
│                 │   │ DriftMonitor │   │ InferenceServer  │
└─────────────────┘   └──────────────┘   └──────────────────┘
```

---

## Agent 1: DataProfilerAgent

```python
from kailash_kaizen import ReActAgent, Signature, Tool
from kailash_ml import DataExplorer, AlertConfig

@Tool(description="Profile a dataset for quality issues")
def profile_data(dataset_name: str, target: str) -> dict:
    df = loader.load("ascent05", dataset_name)
    explorer = DataExplorer()
    explorer.configure(dataset=df, target_column=target,
                       alerts=AlertConfig(missing_threshold=0.05))
    report = explorer.run()
    return {
        "rows": report.overview.rows,
        "columns": report.overview.columns,
        "alerts": [{"severity": a.severity, "message": a.message}
                   for a in report.alerts],
        "missing_summary": report.missing_values.to_dicts(),
    }

profiler_agent = ReActAgent()
profiler_agent.configure(model="claude-sonnet", tools=[profile_data])
```

---

## Agent 2: FeatureAgent

```python
@Tool(description="Generate and select features")
def engineer_features(dataset_name: str, target: str,
                      strategies: list) -> dict:
    df = loader.load("ascent05", dataset_name)
    engineer = FeatureEngineer()
    engineer.configure(
        dataset=df, target_column=target,
        strategies=strategies, n_select=25,
    )
    df_features = engineer.generate()
    df_selected = engineer.select()
    return {
        "generated": df_features.shape[1],
        "selected": df_selected.shape[1],
        "top_features": engineer.get_feature_rankings()[:10],
    }

feature_agent = ReActAgent()
feature_agent.configure(model="claude-sonnet",
                        tools=[engineer_features])
```

---

## Agent 3: ModelSelectionAgent

```python
@Tool(description="Train and compare multiple algorithms")
def compare_models(dataset_name: str, target: str,
                   algorithms: list) -> dict:
    df = loader.load("ascent05", dataset_name)
    results = {}
    for algo in algorithms:
        pipeline = TrainingPipeline()
        pipeline.configure(
            dataset=df, target_column=target,
            task="regression", algorithm=algo,
            cv_folds=5,
        )
        result = pipeline.run()
        results[algo] = {
            "rmse": result.cv_metrics["rmse_mean"],
            "r2": result.cv_metrics["r2_mean"],
        }
    return results

selection_agent = ReActAgent()
selection_agent.configure(model="claude-sonnet",
                          tools=[compare_models])
```

---

## Agent 4: TuningAgent

```python
@Tool(description="Tune hyperparameters for a model")
def tune_model(dataset_name: str, target: str,
               algorithm: str, n_trials: int = 50) -> dict:
    df = loader.load("ascent05", dataset_name)
    search = HyperparameterSearch()
    search.configure(
        dataset=df, target_column=target,
        algorithm=algorithm, strategy="bayesian",
        n_trials=n_trials, metric="rmse",
    )
    result = search.run()
    return {
        "best_params": result.best_params,
        "best_rmse": result.best_score,
        "trials_run": n_trials,
    }
```

---

## Agent 5: EvaluationAgent

```python
@Tool(description="Evaluate model with explanations and fairness")
def evaluate_model(model_path: str, test_data: str) -> dict:
    # Load model and compute SHAP, fairness, drift
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    top_features = sorted(
        zip(feature_names, abs(shap_values).mean(axis=0)),
        key=lambda x: x[1], reverse=True
    )[:10]

    return {
        "metrics": {"rmse": rmse, "r2": r2},
        "top_shap_features": top_features,
        "fairness_check": fairness_results,
    }
```

---

## Agent 6: DeploymentAgent

```python
@Tool(description="Register and deploy model")
def deploy_model(model_name: str, version: str,
                 metrics: dict) -> dict:
    registry = ModelRegistry()
    registry.configure(storage_path="./model_registry")
    registry.register(
        name=model_name, model=trained_model,
        version=version, metrics=metrics,
    )
    registry.promote(name=model_name, version=version,
                     stage="production")
    return {
        "status": "deployed",
        "version": version,
        "endpoint": f"http://localhost:8080/predict",
    }
```

---

## Orchestrating the Pipeline

```python
# Agents communicate through structured outputs
# Each agent's output feeds the next agent's input

# Step 1: Profile
profile_result = profiler_agent.execute(profile_sig, inputs={
    "question": "Profile hdbprices.csv for price prediction"
})

# Step 2: Engineer features (informed by profiling)
feature_result = feature_agent.execute(feature_sig, inputs={
    "question": f"Engineer features. Data quality: {profile_result.answer}"
})

# Step 3: Select model
model_result = selection_agent.execute(model_sig, inputs={
    "question": "Compare lightgbm, xgboost, ridge for regression"
})

# Steps 4-6: Tune, evaluate, deploy...
```

---

## Agent Pipeline Benefits

```
Manual pipeline:
  - Human decides each step
  - Context lost between steps
  - Inconsistent decisions across runs

Agent pipeline:
  - Each agent reasons about its domain
  - Structured handoffs preserve context
  - Consistent, documented decisions
  - Agents explain WHY they chose each option
```

---

## Exercise Preview

**Exercise 5.6: Agent-Driven ML Pipeline**

You will:

1. Build the six ML agents with Kailash ML engine tools
2. Orchestrate them into a sequential pipeline
3. Review agent reasoning at each step
4. Compare agent decisions to manual decisions

Scaffolding level: **Light (~30% code provided)**

---

## Common Pitfalls

| Mistake                                | Fix                                      |
| -------------------------------------- | ---------------------------------------- |
| Agents calling wrong ML engine         | Tool descriptions must be precise        |
| No context passing between agents      | Structure outputs to feed the next agent |
| Trusting agent model selection blindly | Review reasoning; agents can be wrong    |
| Running all agents sequentially        | Some can run in parallel (Lesson 5.7)    |
| Not logging agent decisions            | Always save trace for audit              |

---

## Summary

- Six specialised agents cover the full ML lifecycle
- Each agent wraps Kailash ML engines as tools
- Agents reason about decisions, not just execute commands
- Structured handoffs pass context between pipeline stages
- Agent pipelines are reproducible, documented, and explainable

---

## Next Lesson

**Lesson 5.7: Multi-Agent Orchestration**

We will learn:

- Supervisor-Worker, Sequential, Parallel, and Handoff patterns
- Coordinating multiple agents for complex tasks
- Choosing the right orchestration pattern
