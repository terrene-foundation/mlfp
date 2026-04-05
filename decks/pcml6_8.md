---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 6.8: Capstone — Full Platform

### Module 6: Alignment and Governance

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Integrate all 8 Kailash packages into a unified platform
- Build an end-to-end system from data to governed agent deployment
- Demonstrate mastery of the complete ML lifecycle
- Complete the final ASCENT capstone project

---

## Recap: Lesson 6.7

- Clearance hierarchies control data access across agent organisations
- Budget cascading delegates cost limits through the hierarchy
- Organisation monitors track governance health across all agents
- Every action traces through the accountability chain

---

## The ASCENT Journey

```
Module 1: Data Pipelines        → Polars, DataExplorer, PreprocessingPipeline
Module 2: Statistical Mastery   → Bayesian, hypothesis testing, FeatureStore
Module 3: Supervised ML         → TrainingPipeline, WorkflowBuilder, DataFlow
Module 4: Advanced ML           → AutoMLEngine, EnsembleEngine, InferenceServer
Module 5: LLMs and Agents       → Kaizen agents, RAG, MCP, Nexus
Module 6: Alignment/Governance  → Align, PACT, governed agents

All 8 packages. All 13 ML engines. Production-ready.
```

---

## The 8 Kailash Packages

```
┌──────────────────────────────────────────────┐
│                   Platform                    │
│                                               │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐ │
│  │Core SDK │  │ DataFlow │  │   Nexus     │ │
│  │Workflows│  │ Database │  │ API+CLI+MCP │ │
│  └─────────┘  └──────────┘  └─────────────┘ │
│                                               │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐ │
│  │   ML    │  │  Kaizen  │  │   Align     │ │
│  │13 Engines│ │  Agents  │  │Fine-tuning  │ │
│  └─────────┘  └──────────┘  └─────────────┘ │
│                                               │
│  ┌─────────────────────────────────────────┐ │
│  │              PACT Governance             │ │
│  │         D/T/R + Operating Envelopes      │ │
│  └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

---

## The 13 ML Engines

| Engine                | Module | Purpose                           |
| --------------------- | ------ | --------------------------------- |
| DataExplorer          | 1      | Automated data profiling          |
| PreprocessingPipeline | 1      | Data cleaning and transformation  |
| ModelVisualizer       | 1      | ML-ready visualisations           |
| FeatureEngineer       | 2      | Automated feature generation      |
| FeatureStore          | 2      | Feature versioning and serving    |
| ExperimentTracker     | 2      | Experiment logging and comparison |
| TrainingPipeline      | 3      | Model training with CV            |
| HyperparameterSearch  | 3      | Automated HPO                     |
| ModelRegistry         | 3      | Model versioning and lifecycle    |
| AutoMLEngine          | 4      | Automated ML (clustering, etc.)   |
| EnsembleEngine        | 4      | Model combination strategies      |
| DriftMonitor          | 4      | Production monitoring             |
| InferenceServer       | 4      | Model serving                     |

---

## Capstone Architecture

```
Data Layer:
  ASCENTDataLoader → DataExplorer → PreprocessingPipeline
                                        ↓
Feature Layer:
  FeatureEngineer → FeatureStore (versioned)
                         ↓
Model Layer:
  TrainingPipeline → HyperparameterSearch → ModelRegistry
  EnsembleEngine → DriftMonitor → InferenceServer
                         ↓
Agent Layer:
  ReActAgent + RAG + MCP Tools
  ChainOfThought + SupervisorWorker
                         ↓
Alignment Layer:
  AlignmentPipeline (SFT + DPO) → Fine-tuned model
                         ↓
Governance Layer:
  PACT GovernanceEngine → PactGovernedAgent
                         ↓
Deployment:
  Nexus (API + CLI + MCP) with auth + monitoring
```

---

## Step 1: Data and Features

```python
from shared.data_loader import ASCENTDataLoader
from kailash_ml import (DataExplorer, PreprocessingPipeline,
                         FeatureEngineer, FeatureStore)

# Load and profile
loader = ASCENTDataLoader()
df = loader.load("ascent06", "hdbprices.csv")

explorer = DataExplorer()
explorer.configure(dataset=df, target_column="price")
report = explorer.run()

# Clean and engineer features
pipeline = PreprocessingPipeline()
pipeline.configure(dataset=df, target_column="price",
                   missing_strategy={"numeric": "median"})
df_clean = pipeline.run()

engineer = FeatureEngineer()
engineer.configure(dataset=df_clean, target_column="price",
                   strategies=["arithmetic", "temporal", "interaction"])
df_features = engineer.generate()

# Register in feature store
store = FeatureStore()
store.register(name="hdb_capstone", features=df_features, version="1.0")
```

---

## Step 2: Train and Register Models

```python
from kailash_ml import (TrainingPipeline, HyperparameterSearch,
                         EnsembleEngine, ModelRegistry)

# HPO
search = HyperparameterSearch()
search.configure(
    dataset=df_features, target_column="price",
    algorithm="lightgbm", strategy="bayesian", n_trials=50,
)
search_result = search.run()

# Ensemble
ensemble = EnsembleEngine()
ensemble.configure(
    dataset=df_features, target_column="price",
    strategy="stacking",
    base_models=[
        {"algorithm": "lightgbm", "params": search_result.best_params},
        {"algorithm": "xgboost"},
        {"algorithm": "ridge"},
    ],
)
ensemble_result = ensemble.run()

# Register best model
registry = ModelRegistry()
registry.register(name="hdb_capstone", model=ensemble_result.model,
                  version="1.0.0", metrics=ensemble_result.metrics)
```

---

## Step 3: Build Agent System

```python
from kailash_kaizen import (ReActAgent, RAGResearchAgent,
                             SupervisorWorker, Signature)
from kailash_mcp import MCPServer

# MCP server for ML tools
server = MCPServer(name="hdb-ml-server")

@server.tool(description="Predict HDB price")
def predict_price(town: str, flat_type: str, floor_area: float,
                  lease_years: int) -> dict:
    model = registry.load("hdb_capstone", stage="production")
    prediction = model.predict(...)
    return {"predicted_price": int(prediction), "confidence": 0.85}

# RAG for policy knowledge
rag_agent = RAGResearchAgent()
rag_agent.configure(model="claude-sonnet", vector_store=policy_store)

# Orchestrate
orchestrator = SupervisorWorker()
orchestrator.configure(
    supervisor_model="claude-sonnet",
    workers={"pricing": pricing_agent, "policy": rag_agent},
)
```

---

## Step 4: Align the LLM

```python
from kailash_align import AlignmentPipeline

# SFT for domain knowledge
sft_pipeline = AlignmentPipeline()
sft_pipeline.configure(
    base_model="meta-llama/Llama-3-8B",
    method="sft",
    training_data=hdb_instruction_data,
    lora_rank=16,
    epochs=3,
)
sft_result = sft_pipeline.run()

# DPO for preference alignment
dpo_pipeline = AlignmentPipeline()
dpo_pipeline.configure(
    base_model="meta-llama/Llama-3-8B",
    adapter_path=sft_result.adapter_path,
    method="dpo",
    preference_data=hdb_preferences,
    beta=0.1,
    epochs=1,
)
dpo_result = dpo_pipeline.run()
```

---

## Step 5: Add Governance

```python
from kailash_pact import GovernanceEngine, PactGovernedAgent

engine = GovernanceEngine()
engine.configure(
    duties=["Log predictions", "Cite sources", "Add disclaimers"],
    triggers=[
        {"condition": "confidence < 0.6", "action": "escalate"},
        {"condition": "budget > 80%", "action": "warn"},
    ],
    rights=["query_database", "predict_price", "search_policies"],
    budget=BudgetPolicy(daily_limit=50.00),
)

governed = PactGovernedAgent(
    agent=orchestrator,
    governance=engine,
    clearance="internal",
)
```

---

## Step 6: Deploy with Nexus

```python
from kailash_nexus import Nexus, auth, middleware

app = Nexus(name="hdb-advisory-platform")

app.use(middleware.RateLimit(requests_per_minute=30))
app.configure(auth=auth.APIKey(keys_env="PLATFORM_API_KEYS"))

@app.register
def ask_advisor(question: str) -> dict:
    """Ask the governed HDB advisory agent."""
    result = governed.execute(sig, inputs={"question": question})
    return {
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "governance_status": result.governance_status,
    }

@app.register
def predict(town: str, flat_type: str, floor_area: float,
            lease_years: int) -> dict:
    """Direct price prediction."""
    return predict_price(town, flat_type, floor_area, lease_years)

app.start(host="0.0.0.0", port=8080)
```

---

## Step 7: Monitor

```python
from kailash_ml import DriftMonitor
from kailash_pact import OrganisationMonitor

# ML monitoring
drift_monitor = DriftMonitor()
drift_monitor.configure(reference_data=df_features, psi_threshold=0.15)

# Governance monitoring
gov_monitor = OrganisationMonitor(engine=engine)

# Combined health check
@app.health_check
def platform_health():
    drift = drift_monitor.check(recent_data)
    gov = gov_monitor.generate_report(period="1h")
    return {
        "model_drift": drift.status,
        "governance_health": gov.status,
        "budget_remaining": gov.budget_remaining,
    }
```

---

## Capstone Deliverables

| Component      | Packages Used                                      |
| -------------- | -------------------------------------------------- |
| Data pipeline  | Core SDK, ML (DataExplorer, PreprocessingPipeline) |
| Feature store  | ML (FeatureEngineer, FeatureStore)                 |
| Model training | ML (TrainingPipeline, HPO, Ensemble, Registry)     |
| Agent system   | Kaizen (ReAct, RAG, SupervisorWorker)              |
| Tool server    | MCP server with ML tools                           |
| Aligned model  | Align (SFT + DPO)                                  |
| Governance     | PACT (GovernanceEngine, PactGovernedAgent)         |
| Deployment     | Nexus (API + CLI + MCP), DataFlow                  |
| Monitoring     | ML (DriftMonitor), PACT (OrganisationMonitor)      |

---

## Exercise Preview

**Exercise 6.8: ASCENT Final Capstone**

You will:

1. Build the complete platform integrating all 8 packages
2. Train, align, and govern an HDB advisory agent
3. Deploy with Nexus and configure monitoring
4. Demonstrate the full lifecycle from data to governed predictions

Scaffolding level: **Minimal (~20% code provided)**

---

## Common Pitfalls

| Mistake                                   | Fix                                      |
| ----------------------------------------- | ---------------------------------------- |
| Skipping governance on the final platform | Every production agent needs governance  |
| No drift monitoring after deployment      | Attach DriftMonitor to InferenceServer   |
| Alignment without evaluation              | Compare base vs SFT vs DPO with win rate |
| Deploying without authentication          | Always configure auth in Nexus           |
| No health check endpoint                  | Monitor model drift + governance health  |

---

## Module 6 Summary

| Lesson | Key Skills                               |
| ------ | ---------------------------------------- |
| 6.1    | SFT fine-tuning, LoRA, AlignmentPipeline |
| 6.2    | DPO, LLM-as-judge, preference alignment  |
| 6.3    | RL, Bellman, PPO, RLTrainer              |
| 6.4    | TIES/DARE merging, adapter stacking      |
| 6.5    | PACT, GovernanceEngine, D/T/R            |
| 6.6    | PactGovernedAgent, GovernanceContext     |
| 6.7    | Clearance hierarchies, budget cascading  |
| 6.8    | Full platform capstone                   |

---

## ASCENT Complete

```
You started with:           You can now:
  print("Hello")             Build production ML systems
  df.head()                  Fine-tune and align LLMs
  if/else                    Orchestrate multi-agent systems
                             Deploy governed AI platforms
                             Monitor and maintain ML in production

From zero Python to masters-level ML engineering.

48 lessons. 8 Kailash packages. 13 ML engines.
One complete professional certificate.
```

Congratulations on completing the Professional Certificate in Machine Learning.
