# ASCENT Curriculum v4 — Final (6 Modules x 8 Lessons)

**Official name**: Professional Certificate in Machine Learning (Python) — Terrene Open Academy
**Audience**: Working professionals with ZERO Python → Masters-level ML engineering
**Structure**: 6 modules x 8 lessons = 48 lessons (~4 hours each, ~192 contact hours)
**Certification structure**: Foundation Certificate (M1-M4) + Advanced Certificate (M5-M6)

## Design Principles (from 3 rounds of red team)

1. **Tangible results from Lesson 1** — `pl.read_csv()` then `print(df.shape)`, not abstract workflow infrastructure
2. **Python through data** — every concept grounded in data manipulation, never abstract exercises
3. **Kailash engines before Kailash infrastructure** — use DataExplorer/ModelVisualizer (M1) before WorkflowBuilder/custom nodes (M3)
4. **Agents first, deployment last (M5)** — motivation hook before infrastructure
5. **Alignment block, then governance block (M6)** — coherent topic grouping
6. **Each lesson: what WORKS / what the weakest student needs** — progressive scaffolding
7. **Capstone = integration, not from-scratch** — scaffolding increases to ~40% for capstones

---

## Module 1: Data Pipelines and Visualisation Mastery with Python

_Zero to productive. Learn Python by exploring real Singapore data._

| #   | Lesson                              | Python Concepts                                                | Kailash / Tools                                                                                | Exercise                                                        |
| --- | ----------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1.1 | **Your First Data Exploration**     | Variables, strings, numbers, `print()`, f-strings, assignment  | `polars`: `pl.read_csv()`, `df.shape`, `df.columns`, `df.head()`, `df.describe()`              | NEW: Load small Singapore weather CSV, inspect, print summaries |
| 1.2 | **Filtering and Transforming Data** | Comparison operators, booleans, method chaining, lists         | `pl.col()`, `filter()`, `select()`, `sort()`, `with_columns()`                                 | NEW: Filter HDB resale by town/price/date                       |
| 1.3 | **Functions and Aggregation**       | `def`, `for`, `if/else`, dicts, return values                  | `group_by()`, `agg()`, `pl.mean()`, `pl.count()`, writing helper functions                     | NEW: Functions for district statistics                          |
| 1.4 | **Joins, Windows, and Real Scale**  | Method chaining, complex expressions, imports, packages        | `join()`, `over()`, `rolling_mean()`, lazy frames, `collect()` — 15M-row HDB data              | Adapted from M1 ex_1: HDB + MRT + schools                       |
| 1.5 | **Data Visualization**              | Plotly API, figure objects, HTML export                        | `ModelVisualizer` (histogram, scatter, bar, heatmap, line — EDA charts only)                   | NEW: Visualize HDB trends interactively                         |
| 1.6 | **Automated Data Profiling**        | Classes as users (not authors), async provided as scaffolding  | `DataExplorer`, `AlertConfig`, 8 alert types, `DataProfile`, correlation matrices, `compare()` | Adapted from M1 ex_3: Profile dirty economic data               |
| 1.7 | **Data Cleaning Pipeline**          | `try/except` basics, None/null, error messages                 | `PreprocessingPipeline` (auto-detect, encode, scale, impute), `SetupResult`                    | Adapted from M1 ex_5: Clean messy taxi data                     |
| 1.8 | **End-to-End Project**              | Project structure, imports, modules, review of all M1 concepts | Full pipeline: load → profile → clean → visualize → report                                     | NEW: Complete EDA on new Singapore dataset                      |

**Datasets**: Singapore weather (small ~1K for 1.1-1.3), HDB Resale (15M for 1.4+), economic indicators, taxi trips
**Scaffolding**: ~80% for 1.1-1.3 (near-complete code with blanks for key values), ~70% for 1.4-1.8
**What moved OUT**: Bayesian estimation (old ex_2) → M2.1. Hypothesis testing (old ex_4) → M2.3.

---

## Module 2: Statistical Mastery for ML and AI Success

_Statistical foundations taught through experiment tracking and feature engineering._

| #   | Lesson                                    | Theory                                                                                                                     | Kailash SDK                                                                            | Exercise                                         |
| --- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 2.1 | **Probability and Bayesian Thinking**     | Concrete distributions (Normal, Beta, Poisson) first, then conjugate priors (Normal-Normal, Beta-Binomial), Bayes' theorem | `ModelVisualizer` for posterior plots                                                  | FROM M1 ex_2: Bayesian estimation on HDB prices  |
| 2.2 | **Estimation and Inference**              | MLE derivation, MAP, Fisher information, Cramér-Rao, when MLE fails                                                        | Workflow: optimization → visualization                                                 | NEW: MLE for Singapore economic parameters       |
| 2.3 | **Hypothesis Testing**                    | Neyman-Pearson, power analysis, MDE, Bonferroni/BH-FDR, permutation tests                                                  | `ExperimentTracker` (create_experiment, context manager, log_param, log_metric)        | FROM M1 ex_4: A/B test with multiple corrections |
| 2.4 | **Bootstrap and Resampling**              | BCa intervals, parametric vs non-parametric, distribution-free methods                                                     | `ExperimentTracker.compare_runs()`, `ModelVisualizer`                                  | NEW: Bootstrap analysis on experiment data       |
| 2.5 | **A/B Testing at Scale**                  | SRM, CUPED (derive Var(Y_adj) = Var(Y)(1-ρ²)), sequential testing, Bayesian A/B                                            | `ExperimentTracker` full lifecycle                                                     | Existing M2 ex_3: CUPED + Bayesian               |
| 2.6 | **Causal Inference**                      | DiD (derive ATT), propensity matching, parallel trends, placebo tests                                                      | `ExperimentTracker` for causal logging                                                 | Existing M2 ex_4: DiD on cooling measures        |
| 2.7 | **Feature Engineering and Feature Store** | Temporal features, point-in-time correctness, data lineage, leakage detection                                              | `FeatureStore`, `FeatureSchema`, `FeatureField`, `FeatureEngineer` (generate + select) | Existing M2 ex_1 + ex_2 combined                 |
| 2.8 | **Experiment-Driven Project**             | Full lifecycle: design → execute → analyze → report (students choose from 3-4 options)                                     | `ExperimentTracker` + `FeatureStore` + `ModelVisualizer`                               | Existing M2 ex_5                                 |

**Scaffolding**: ~65% (arguments + some method calls stripped)
**New exercises**: 2.2, 2.4 (2 new + 2 moved from M1 + 4 existing)

---

## Module 3: Supervised ML for Building and Deploying Models

_From theory to production — workflow orchestration, model registry, governed deployment._

| #   | Lesson                                       | Theory                                                                                                                   | Kailash SDK                                                                                                                 | Exercise                                           |
| --- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 3.1 | **Bias-Variance and Regularization**         | Full decomposition, L1/L2 geometry, Bayesian interpretation (L2=Gaussian prior)                                          | `PreprocessingPipeline`, `kailash_ml.interop` (to_sklearn_input, etc.)                                                      | Existing M3 ex_1                                   |
| 3.2 | **Gradient Boosting Deep Dive**              | XGBoost 2nd-order Taylor, LightGBM GOSS, CatBoost ordered                                                                | `TrainingPipeline`, `ModelSpec`, `EvalSpec`                                                                                 | Existing M3 ex_1 (second half) or split            |
| 3.3 | **Class Imbalance and Calibration**          | SMOTE failures, cost-sensitive, Focal Loss, Platt/isotonic, proper scoring rules                                         | `ModelVisualizer.calibration_curve()`, `precision_recall_curve()`                                                           | Existing M3 ex_2                                   |
| 3.4 | **SHAP, LIME, and Fairness**                 | Shapley axioms, TreeSHAP, LIME, disparate impact testing                                                                 | `ModelVisualizer.feature_importance()`                                                                                      | Existing M3 ex_3 (add LIME)                        |
| 3.5 | **Workflow Orchestration and Custom Nodes**  | WorkflowBuilder, nodes, connections, `runtime.execute(workflow.build())`, `@register_node`, `Node` subclass, logic nodes | `WorkflowBuilder`, `LocalRuntime`, `PythonCodeNode`, `ConditionalNode`, custom nodes                                        | Existing M3 ex_4 (expanded with advanced patterns) |
| 3.6 | **DataFlow and Persistence**                 | @db.model, db.express CRUD, schema design, async/await primer                                                            | `DataFlow`, `field()`, `db.express.create/list/get/update`, `ConnectionManager`                                             | NEW: Persist ML results, query and compare         |
| 3.7 | **Model Registry and Hyperparameter Search** | Bayesian optimization, model versioning, staging→production promotion as governance gate                                 | `HyperparameterSearch`, `SearchSpace`, `SearchConfig`, `ParamDistribution`, `ModelRegistry`, `MetricSpec`, `ModelSignature` | Existing M3 ex_5                                   |
| 3.8 | **Production Pipeline Project**              | Full pipeline with model card (Mitchell et al.), conformal prediction                                                    | Complete: train → calibrate → conformal → register → promote → model card                                                   | Existing M3 ex_6                                   |

**Scaffolding**: ~50% (method calls + setup stripped)
**New exercises**: 3.6 only (1 new)
**Key placement**: WorkflowBuilder + custom nodes HERE (M3.5) — students now know Python classes from 8 weeks of use

---

## Module 4: Unsupervised ML, NLP, and Deep Learning

_Pattern discovery, text analysis, neural networks, and production monitoring._

| #   | Lesson                              | Theory                                                                                                                             | Kailash SDK                                                                                      | Exercise                                                     |
| --- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| 4.1 | **Clustering**                      | K-means, spectral, HDBSCAN, gap statistic, validation metrics                                                                      | `AutoMLEngine`, `AutoMLConfig` (agent double opt-in, `LLMCostTracker`)                           | Existing M4 ex_1                                             |
| 4.2 | **EM Algorithm and GMMs**           | Full E-step/M-step derivation, ELBO convergence, EM as template algorithm                                                          | `AutoMLEngine` comparison                                                                        | NEW: Implement EM from scratch, compare with sklearn         |
| 4.3 | **Dimensionality Reduction**        | PCA (explicit SVD connection, reconstruction error), UMAP, t-SNE                                                                   | `ModelVisualizer` for embeddings                                                                 | NEW: PCA reconstruction + UMAP/t-SNE comparison              |
| 4.4 | **Anomaly Detection and Ensembles** | Isolation Forest, LOF, score blending                                                                                              | `EnsembleEngine` (`blend()`, `stack()`, `bag()`, `boost()`) — use actual SDK calls               | Existing M4 ex_2 (upgraded: use EnsembleEngine.blend())      |
| 4.5 | **NLP: Text to Topics**             | TF-IDF (derive), BM25, Word2Vec, BERTopic, NPMI coherence                                                                          | `ModelVisualizer` for topics                                                                     | Existing M4 ex_3 (add TF-IDF warmup)                         |
| 4.6 | **Drift Monitoring**                | PSI, KS test, performance degradation, monitoring as governance obligation                                                         | `DriftMonitor` (set_reference, check_drift, check_performance, schedule_monitoring), `DriftSpec` | Existing M4 ex_4                                             |
| 4.7 | **Deep Learning Foundations**       | Backprop chain rule, gradient flow, CNN architecture, ResNet skip connections, training dynamics (LR scheduling, BatchNorm, AdamW) | PyTorch + `ModelVisualizer.training_history()` + `OnnxBridge` (export, validate)                 | Existing M4 ex_5 (with structured synthetic data, not noise) |
| 4.8 | **Model Serving**                   | ONNX format, inference patterns, InferenceServer                                                                                   | `InferenceServer` (predict, predict_batch, warm_cache, register_endpoints), `PredictionResult`   | Existing M4 ex_6 (fix import, add ModelSignature validation) |

**Scaffolding**: ~40% (setup + calls + logic stripped)
**New exercises**: 4.2, 4.3 (2 new)
**Key change from v3**: Nexus deployment moved OUT to M5. M4 ends with InferenceServer (single-channel), M5 adds Nexus (multi-channel).

---

## Module 5: LLMs, AI Agents, and Production Deployment

_Build intelligent agents, then deploy them at scale. Agents first, infrastructure after._

| #   | Lesson                               | Theory / Practice                                                                         | Kailash SDK                                                                                                         | Exercise                                                                             |
| --- | ------------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 5.1 | **LLM Fundamentals and Kaizen**      | Tokenization, scaling laws, Signature/InputField/OutputField, structured output           | `Kaizen` core: `Signature`, `InputField`, `OutputField`, `Delegate` (streaming, events, cost tracking)              | Existing M5 ex_1: Delegate + SimpleQAAgent                                           |
| 5.2 | **Chain-of-Thought Reasoning**       | Step-by-step reasoning, reasoning chain quality, CoT vs direct answering                  | `ChainOfThoughtAgent`                                                                                               | Existing M5 ex_2: CoT on clustering results                                          |
| 5.3 | **ReAct Agents with Tools**          | Reasoning + action loops, tool selection, autonomous data exploration, cost budget safety | `ReActAgent`, custom tools wrapping Kailash engines                                                                 | Existing M5 ex_3: ReAct with DataExplorer/polars tools                               |
| 5.4 | **RAG Systems**                      | Chunking, retrieval (dense/sparse/hybrid), RAGAS evaluation, HyDE                         | `RAGResearchAgent`, `MemoryAgent`                                                                                   | Existing M5 ex_4 (upgraded: real document loading)                                   |
| 5.5 | **MCP Servers and Tool Integration** | MCP protocol, tool registration, transports — "how to expose tools to agents at scale"    | `kailash.mcp_server`, `kailash.mcp`, MCP integration                                                                | NEW: Build MCP server for ML tools                                                   |
| 5.6 | **ML Agent Pipeline**                | LLMs augmenting ML lifecycle, double opt-in, agent confidence                             | 6 ML agents: DataScientist, FeatureEngineer, ModelSelector, ExperimentInterpreter, DriftAnalyst, RetrainingDecision | Existing M5 ex_5: Full 6-agent pipeline                                              |
| 5.7 | **Multi-Agent Orchestration**        | A2A protocol, production patterns: supervisor-worker, sequential, parallel, handoff       | `SupervisorWorkerPattern`, `SequentialPattern`, `ParallelPattern`, `HandoffPattern`                                 | Existing M5 ex_6 (rewrite to use formal coordination patterns)                       |
| 5.8 | **Production Deployment with Nexus** | Multi-channel (API+CLI+MCP), auth (RBAC/JWT), middleware, monitoring                      | `Nexus`, `nexus.auth`, middleware, `nexus.plugins` + DriftMonitor integration                                       | NEW: Deploy M4 model via Nexus with auth + agent wrapper (NO governance — that's M6) |

**Scaffolding**: ~30% (most logic stripped, imports + structure given)
**New exercises**: 5.5, 5.8 (2 new, 1 rewrite of ex_6)
**Key change from v3**: Agents FIRST (5.1-5.4), then MCP (5.5) when students understand tools, then deployment (5.8) as capstone. Motivation-driven sequence.

---

## Module 6: Alignment, Governance, and Organisational Transformation

_Fine-tuning, governance, RL, and the capstone. Masters-level content, coherently grouped._

| #   | Lesson                        | Theory                                                                                           | Kailash SDK                                                                                       | Exercise                                                 |
| --- | ----------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 6.1 | **SFT Fine-Tuning**           | LoRA derivation (W = W₀ + BA, connect to M4.3 SVD), QLoRA NF4, adapter lifecycle                 | `AlignmentPipeline`, `AlignmentConfig`, `AdapterRegistry`                                         | Existing M6 ex_1                                         |
| 6.2 | **Preference Alignment**      | DPO from Bradley-Terry, LLM-as-judge evaluation (biases: position, verbosity, self-enhancement)  | `AlignmentPipeline` (method="dpo"), `evaluator`                                                   | Existing M6 ex_2                                         |
| 6.3 | **Reinforcement Learning**    | Bellman equations (expectation + optimality), PPO clipped objective, reward connection to DPO    | `RLTrainer`, `env_registry`, `policy_registry`, custom Gymnasium env                              | Existing M6 ex_5 (renumbered)                            |
| 6.4 | **Advanced Alignment**        | Model merging (TIES/DARE), evaluation strategies, model comparison                               | `kailash-align`: `merge`, `evaluator`, model comparison patterns                                  | NEW: Merge adapters from 6.1+6.2, evaluate which is best |
| 6.5 | **AI Governance with PACT**   | EU AI Act (pre-read), Singapore AI Verify, D/T/R grammar, operating envelopes, enforcement modes | `GovernanceEngine`, `compile_org()`, `Address`, `can_access()`, `explain_access()`, `CostTracker` | Existing M6 ex_3                                         |
| 6.6 | **Governed Agents**           | Monotonic tightening, frozen GovernanceContext, fail-closed, audit chains                        | `PactGovernedAgent`, `GovernanceContext`, `RoleEnvelope`, `TaskEnvelope`, PACT MCP integration    | Existing M6 ex_4                                         |
| 6.7 | **Agent Governance at Scale** | Clearance levels, budget cascading, what happens when agents fail (dereliction)                  | `kaizen_agents.governance` (clearance, budget, cascade), governance testing                       | NEW: Multi-agent governance with budget cascading        |
| 6.8 | **Capstone: Full Platform**   | All 8 packages integrated. Debugging traces, testing agents, production readiness.               | Core SDK → DataFlow → ML → Kaizen → PACT → Nexus → Align                                          | Existing M6 ex_6 (~40% scaffolding, integration-focused) |

**Scaffolding**: ~20% for 6.1-6.7, ~40% for 6.8 (capstone tests integration, not from-scratch)
**New exercises**: 6.4, 6.7 (2 new)
**Key changes from v3**:

- RL moved to 6.3 (after DPO, connecting reward/preference concepts)
- Alignment block (6.1-6.4) then governance block (6.5-6.7) — coherent grouping
- Serving moved from 6.6 to capstone (6.8)
- GRPO demoted to conceptual mention in 6.2 (not a full exercise)

---

## Exercise Inventory

| Module    | Keep as-is | Adapt | Move in   | New    | Rewrite | Total  |
| --------- | ---------- | ----- | --------- | ------ | ------- | ------ |
| M1        | 0          | 3     | 0         | 5      | 0       | 8      |
| M2        | 4          | 0     | 2 from M1 | 2      | 0       | 8      |
| M3        | 5          | 1     | 0         | 1      | 0       | 7+1    |
| M4        | 4          | 2     | 0         | 2      | 0       | 8      |
| M5        | 4          | 1     | 0         | 2      | 1       | 8      |
| M6        | 4          | 1     | 0         | 2      | 0       | 7+1    |
| **Total** | **21**     | **8** | **2**     | **14** | **1**   | **48** |

**Net new to write**: 14 exercises + 1 rewrite = 15 implementation tasks

---

## Complete SDK Coverage

| Package              | Modules        | Key Classes                                                                                                     |
| -------------------- | -------------- | --------------------------------------------------------------------------------------------------------------- |
| **kailash** (core)   | M1, M3.5, M5.5 | WorkflowBuilder, LocalRuntime, Node, @register_node, PythonCodeNode, logic nodes, ConnectionManager, MCP server |
| **kailash-ml**       | M1-M4, M6      | All 13 engines + 6 ML agents + RLTrainer + OnnxBridge + interop + dashboard                                     |
| **kailash-dataflow** | M3.6           | @db.model, field(), db.express CRUD                                                                             |
| **kailash-nexus**    | M5.8           | Nexus, auth (RBAC/JWT), middleware, plugins                                                                     |
| **kailash-kaizen**   | M5.1-5.4       | Signature, InputField/OutputField, BaseAgent, structured output                                                 |
| **kaizen-agents**    | M5.1-5.8       | Delegate, 6+ specialized agents, coordination patterns, ML agents                                               |
| **kailash-pact**     | M6.5-6.7       | GovernanceEngine, PactGovernedAgent, Address, enforcement, costs, agent governance                              |
| **kailash-align**    | M6.1-6.4       | AlignmentPipeline, AlignmentConfig, AdapterRegistry, merge, evaluator                                           |

---

## Progression Summary

```
M1: Zero Python → Data exploration with Polars + Kailash engines
M2: Statistics → Experiment design with ExperimentTracker + FeatureStore
M3: Supervised ML → Production with WorkflowBuilder + ModelRegistry + DataFlow
M4: Unsupervised + NLP + DL → Monitoring with AutoML + DriftMonitor + ONNX
M5: LLM Agents → Deployed at scale with Kaizen + MCP + Nexus
M6: Alignment + Governance → Full platform capstone with Align + PACT

Foundation Certificate: M1-M4 (32 lessons, 128 hours)
Advanced Certificate: M5-M6 (16 lessons, 64 hours)
```
