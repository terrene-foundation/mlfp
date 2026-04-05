# ASCENT Curriculum v3 — 6 Modules x 8 Lessons (Final)

**Official name**: Professional Certificate in Machine Learning (Python) — Terrene Open Academy
**Audience**: Working professionals with ZERO Python experience → Masters-level ML engineering
**Structure**: 6 modules x 8 lessons = 48 lessons (~4 hours each)
**Design principle**: Python taught through immediate, tangible results — not abstract infrastructure

## Key Design Decisions (from red team v2)

1. **Start with Polars + engine calls, not WorkflowBuilder** — beginners need to SEE results immediately. `pl.read_csv()` then `DataExplorer.profile()` on Day 1. Workflow infrastructure comes in M3 when students understand functions and classes.
2. **Existing 34 exercises are solid** — align curriculum to exercises, then add 14 new ones for expanded lessons.
3. **M5 = Agents (matching existing exercises), M6 = Alignment + Governance + RL** — curriculum matches what was built.
4. **Split overloaded M6 lessons** — agents scope reduced, alignment split into 2 sessions.
5. **Statistical content belongs in M2** — move Bayesian/hypothesis exercises from M1 to M2.

---

## Module 1: Data Pipelines and Visualisation Mastery with Python

_Zero to productive. Learn Python by exploring real Singapore data with Polars and Kailash engines._

| #   | Lesson                                | Python Concepts                                                        | Kailash / Tools                                                                   | Exercise                                                  |
| --- | ------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 1.1 | **Your First Data Exploration**       | Variables, strings, numbers, `print()`, assignment, f-strings          | `polars`: `pl.read_csv()`, `df.shape`, `df.columns`, `df.head()`, `df.describe()` | NEW: Load Singapore weather CSV, inspect, print summaries |
| 1.2 | **Filtering, Sorting, and Selecting** | Comparison operators, booleans, method chaining, lists                 | `pl.col()`, `filter()`, `select()`, `sort()`, `with_columns()`                    | NEW: Filter HDB resale data by town/price/date            |
| 1.3 | **Functions, Loops, and Aggregation** | `def`, `for`, `if/else`, dictionaries, return values                   | `group_by()`, `agg()`, `pl.mean()`, `pl.count()`, `pl.sum()`                      | NEW: Write functions to compute district statistics       |
| 1.4 | **Joins, Windows, and Scale**         | Method chaining, complex expressions, imports                          | `join()`, `over()`, `rolling_mean()`, window functions on 15M rows                | Existing ex_1 (adapted): HDB + MRT + schools joins        |
| 1.5 | **Data Visualization**                | Plotly basics, HTML output, figure objects                             | `ModelVisualizer` (histogram, scatter, bar, heatmap, line)                        | NEW: Visualize HDB price trends, district comparisons     |
| 1.6 | **Automated Data Profiling**          | Classes (as users, not authors), async intro (provided as scaffolding) | `DataExplorer`, `AlertConfig`, `DataProfile`, 8 alert types, correlation matrices | Existing ex_3 (adapted): Profile dirty economic data      |
| 1.7 | **Data Cleaning and Preprocessing**   | Error handling basics (`try/except`), None/null concepts               | `PreprocessingPipeline` (auto-detect task, encode, scale, impute), `SetupResult`  | Existing ex_5 (adapted): Clean messy taxi data            |
| 1.8 | **End-to-End Pipeline Project**       | Project structure, imports, modules, putting it all together           | Full pipeline: load → profile → clean → visualize → report                        | NEW: Complete EDA on a new Singapore dataset              |

**What moved**: Bayesian estimation (old ex_2) → M2.1. Hypothesis testing (old ex_4) → M2.3. These are M2 statistical content.

**New exercises needed**: 1.1, 1.2, 1.3, 1.5, 1.8 (5 new, 3 adapted from existing)

**Datasets**: Singapore weather (small, ~1K rows for 1.1-1.3), HDB Resale (15M for 1.4+), economic indicators, taxi trips

---

## Module 2: Statistical Mastery for ML and AI Success

_Statistical foundations taught through experiment tracking and feature engineering._

| #   | Lesson                                    | Theory                                                                         | Kailash SDK                                                                            | Exercise                                                                         |
| --- | ----------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 2.1 | **Probability and Bayesian Thinking**     | Distributions, Bayes' theorem, conjugate priors (Normal-Normal, Beta-Binomial) | `ModelVisualizer` for posteriors                                                       | Moved from M1 ex_2: Bayesian estimation on HDB prices                            |
| 2.2 | **Estimation Theory**                     | MLE, MAP, Fisher information, Cramér-Rao bound                                 | Numerical optimization in polars/numpy                                                 | NEW: MLE for Singapore economic parameters                                       |
| 2.3 | **Hypothesis Testing and Power**          | Neyman-Pearson, power analysis, MDE, FDR corrections, permutation tests        | `ExperimentTracker` (create_experiment, context manager, log_param, log_metric)        | Moved from M1 ex_4: A/B test with multiple testing correction                    |
| 2.4 | **Bootstrap and Resampling**              | BCa intervals, parametric vs non-parametric bootstrap                          | `ExperimentTracker.compare_runs()`, `ModelVisualizer`                                  | NEW: Bootstrap analysis on experiment results                                    |
| 2.5 | **A/B Testing at Scale**                  | SRM, CUPED (derive variance reduction), sequential testing, Bayesian A/B       | `ExperimentTracker` full lifecycle                                                     | Existing M2 ex_3: CUPED + Bayesian + sequential                                  |
| 2.6 | **Causal Inference**                      | DiD, propensity matching, parallel trends, placebo tests                       | `ExperimentTracker` for causal logging                                                 | Existing M2 ex_4: DiD on Singapore cooling measures                              |
| 2.7 | **Feature Engineering and Feature Store** | Temporal features, target encoding, point-in-time correctness, data lineage    | `FeatureStore`, `FeatureSchema`, `FeatureField`, `FeatureEngineer` (generate + select) | Existing M2 ex_1 + ex_2 (combined): Healthcare features + FeatureStore lifecycle |
| 2.8 | **Experiment-Driven Project**             | Full lifecycle: design → execute → analyze → report                            | `ExperimentTracker` + `FeatureStore` + `ModelVisualizer`                               | Existing M2 ex_5: FeatureEngineer + experiment review                            |

**New exercises needed**: 2.2, 2.4 (2 new, 2 moved from M1, 4 existing)

---

## Module 3: Supervised ML for Building and Deploying Models

_From bias-variance theory to production model registry — orchestrated through Kailash workflows._

| #   | Lesson                                       | Theory                                                                     | Kailash SDK                                                                                                                                            | Exercise                                                          |
| --- | -------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| 3.1 | **Bias-Variance and Regularization**         | Full decomposition, L1/L2 geometry, Bayesian interpretation                | `PreprocessingPipeline`, `kailash_ml.interop` (to_sklearn_input, etc.)                                                                                 | Existing M3 ex_1 (adapted): Gradient boosting comparison          |
| 3.2 | **Gradient Boosting Deep Dive**              | XGBoost Taylor, LightGBM GOSS, CatBoost ordered                            | `TrainingPipeline`, `ModelSpec`, `EvalSpec`                                                                                                            | Existing M3 ex_1 (second half) or new split                       |
| 3.3 | **Class Imbalance and Calibration**          | SMOTE failures, cost-sensitive, Focal Loss, Platt/isotonic, proper scoring | `ModelVisualizer.calibration_curve()`                                                                                                                  | Existing M3 ex_2: Class imbalance workshop                        |
| 3.4 | **SHAP and Interpretability**                | Shapley axioms, TreeSHAP, LIME, fairness audit                             | `ModelVisualizer.feature_importance()`, disparate impact testing                                                                                       | Existing M3 ex_3: SHAP + fairness (add LIME)                      |
| 3.5 | **Workflow Orchestration**                   | WorkflowBuilder, nodes, connections, `runtime.execute(workflow.build())`   | `WorkflowBuilder`, `LocalRuntime`, logic nodes, custom nodes (`@register_node`, `Node` subclass)                                                       | Existing M3 ex_4 (expanded): Full workflow with conditional logic |
| 3.6 | **DataFlow and Persistence**                 | @db.model, db.express CRUD, schema design                                  | `DataFlow`, `field()`, `db.express.create/list/get/update`                                                                                             | NEW: Persist ML results to DataFlow, query and compare            |
| 3.7 | **Model Registry and Hyperparameter Search** | Bayesian optimization, model versioning, staging → production promotion    | `HyperparameterSearch` (SearchSpace, SearchConfig, ParamDistribution), `ModelRegistry` (register_model, promote_model), `MetricSpec`, `ModelSignature` | Existing M3 ex_5: HyperparameterSearch + ModelRegistry            |
| 3.8 | **Production Pipeline Project**              | Full pipeline with model card, conformal prediction                        | Complete: train → calibrate → conformal → register → promote → model card                                                                              | Existing M3 ex_6: End-to-end pipeline                             |

**Key change**: Custom nodes and WorkflowBuilder moved HERE (M3.5) from M1 where beginners aren't ready for them. By M3, students know Python classes.

**New exercises needed**: 3.6 (1 new, rest adapted from existing)

---

## Module 4: Unsupervised ML and Advanced Techniques

_Pattern discovery, NLP, and production monitoring. Reduced scope: deployment moved to M5._

| #   | Lesson                              | Theory                                                                   | Kailash SDK                                                                                                     | Exercise                                                                       |
| --- | ----------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 4.1 | **Clustering Foundations**          | K-means, spectral, HDBSCAN, gap statistic                                | `AutoMLEngine`, `AutoMLConfig` (agent double opt-in, LLMCostTracker)                                            | Existing M4 ex_1: Clustering comparison                                        |
| 4.2 | **EM Algorithm and GMMs**           | Full E-step/M-step derivation, ELBO convergence, connection to K-means   | `AutoMLEngine` algorithm comparison                                                                             | NEW: Implement EM from scratch, compare with sklearn GMM                       |
| 4.3 | **Dimensionality Reduction**        | PCA/SVD, UMAP, t-SNE (KL divergence)                                     | `ModelVisualizer` for embeddings                                                                                | NEW: PCA reconstruction + UMAP/t-SNE comparison                                |
| 4.4 | **Anomaly Detection and Ensembles** | Isolation Forest, LOF, autoencoder scoring, ensemble blending            | `EnsembleEngine` (blend, stack, bag, boost)                                                                     | Existing M4 ex_2: UMAP + anomaly detection (use actual EnsembleEngine.blend()) |
| 4.5 | **NLP: Text to Topics**             | TF-IDF, BM25, Word2Vec, BERTopic (UMAP+HDBSCAN+c-TF-IDF), NPMI coherence | `ModelVisualizer` for topics                                                                                    | Existing M4 ex_3: Topic modeling                                               |
| 4.6 | **Production Drift Monitoring**     | PSI, KS test, performance degradation, governance obligations            | `DriftMonitor` (set_reference, check_drift, check_performance, schedule_monitoring), `DriftSpec`, `DriftReport` | Existing M4 ex_4: DriftMonitor scenarios                                       |
| 4.7 | **Deep Learning Foundations**       | Backprop, gradient flow, CNN architecture, ResNet skip connections       | PyTorch + `ModelVisualizer.training_history()`                                                                  | Existing M4 ex_5 (adapted): CNN training with ONNX export                      |
| 4.8 | **ONNX Export and Model Serving**   | ONNX format, model portability, inference patterns                       | `OnnxBridge` (export, validate), `InferenceServer` (predict, predict_batch, warm_cache)                         | Existing M4 ex_6 (adapted): InferenceServer deployment                         |

**Key change**: Deep learning foundations moved FROM M5 to M4.7-4.8. Nexus multi-channel + auth moved TO M5. MCP moved TO M5.

**New exercises needed**: 4.2, 4.3 (2 new, rest adapted)

---

## Module 5: Production Deployment, LLMs and AI Agents

_From Nexus multi-channel deployment to Kaizen agent framework. The production AI engineering module._

| #   | Lesson                                  | Theory / Practice                                                           | Kailash SDK                                                                                                         | Exercise                                                      |
| --- | --------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 5.1 | **Multi-Channel Deployment with Nexus** | API + CLI + MCP from single codebase, deployment patterns                   | `Nexus`, `nexus.auth` (RBAC, JWT), middleware (CSRF, cache), `nexus.plugins`, `nexus.metrics`                       | NEW: Deploy M4 model via Nexus with auth                      |
| 5.2 | **MCP Server Development**              | MCP protocol, tool registration, transports                                 | `kailash.mcp_server`, `kailash.mcp` (contrib tools), MCP integration patterns                                       | NEW: Build custom MCP server for ML tools                     |
| 5.3 | **LLM Fundamentals and Kaizen**         | Tokenization, scaling laws, Signature/InputField/OutputField                | Kaizen core: `Signature`, `InputField`, `OutputField`, `Delegate` (creation, streaming, events, cost tracking)      | Existing M5 ex_1: Delegate + SimpleQAAgent                    |
| 5.4 | **Reasoning Agents**                    | CoT step-by-step reasoning, ReAct (reasoning + action), tool use            | `ChainOfThoughtAgent`, `ReActAgent`, custom tools                                                                   | Existing M5 ex_2 + ex_3 (combined): CoT + ReAct with tools    |
| 5.5 | **RAG Systems**                         | Chunking, retrieval (dense/sparse/hybrid), RAGAS evaluation, HyDE           | `RAGResearchAgent`, `MemoryAgent`                                                                                   | Existing M5 ex_4 (upgraded): RAG with real document loading   |
| 5.6 | **ML Agent Pipeline**                   | LLMs augmenting ML lifecycle, double opt-in, agent confidence               | 6 ML agents: DataScientist, FeatureEngineer, ModelSelector, ExperimentInterpreter, DriftAnalyst, RetrainingDecision | Existing M5 ex_5: Full 6-agent pipeline                       |
| 5.7 | **Multi-Agent Orchestration**           | A2A protocol, production patterns (supervisor-worker, sequential, parallel) | `SupervisorWorkerPattern`, `SequentialPattern`, `ParallelPattern`, `HandoffPattern`, `Supervisor`                   | Existing M5 ex_6 (upgraded): Use actual coordination patterns |
| 5.8 | **Production Agent Project**            | Full stack: model + agent + deployment + monitoring                         | Full: `InferenceServer` + `Nexus` + `Delegate` + `DriftMonitor`                                                     | NEW: Governed agent serving predictions via Nexus             |

**Key change**: M5 now starts with deployment (Nexus, MCP) then progresses to agents. This gives students the deployment infrastructure BEFORE they build agents on it. Matches logical dependency: deploy first, then add intelligence.

**New exercises needed**: 5.1, 5.2, 5.8 (3 new, rest adapted from existing M5 exercises)

---

## Module 6: Alignment, Governance, RL and Organisational Transformation

_Fine-tuning, governance, RL, and the capstone. The masters-level module._

| #   | Lesson                             | Theory                                                                         | Kailash SDK                                                                                                                            | Exercise                                           |
| --- | ---------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 6.1 | **SFT Fine-Tuning**                | LoRA derivation (W = W₀ + BA), QLoRA NF4, adapter lifecycle                    | `AlignmentPipeline`, `AlignmentConfig`, `AdapterRegistry`                                                                              | Existing M6 ex_1: SFT with LoRA                    |
| 6.2 | **Preference Alignment**           | DPO from Bradley-Terry, GRPO (conceptual), LLM-as-judge evaluation             | `AlignmentPipeline` (method="dpo"), `evaluator`, `rewards`                                                                             | Existing M6 ex_2: DPO vs QLoRA                     |
| 6.3 | **AI Governance with PACT**        | EU AI Act, Singapore AI Verify, D/T/R grammar, operating envelopes             | `GovernanceEngine`, `compile_org()`, `Address`, `can_access()`, `explain_access()`, enforcement modes, `CostTracker`                   | Existing M6 ex_3: PACT governance setup            |
| 6.4 | **Governed Agents**                | Monotonic tightening, frozen GovernanceContext, fail-closed, audit chains      | `PactGovernedAgent`, `GovernanceContext`, `RoleEnvelope`, `TaskEnvelope`, PACT MCP integration                                         | Existing M6 ex_4: Governed agents                  |
| 6.5 | **Reinforcement Learning**         | Bellman equations (expectation + optimality), PPO clipped objective derivation | `RLTrainer`, `env_registry`, `policy_registry`, custom Gymnasium environments                                                          | Existing M6 ex_5: RL inventory management          |
| 6.6 | **Advanced Alignment and Serving** | Model merging (TIES/DARE), reward functions, model evaluation, serving         | `kailash-align`: `merge`, `rewards`, `evaluator`, `serving` (vLLM), `gpu_memory`, `onprem`                                             | NEW: Model merging + evaluation + serving pipeline |
| 6.7 | **Agent Governance at Scale**      | Agent lifecycle, clearance levels, budget cascading, dereliction handling      | `kaizen_agents.governance` (clearance, budget, bypass, dereliction, vacancy, accountability, cascade), PACT governance CLI and testing | NEW: Multi-agent governance with budget cascading  |
| 6.8 | **Capstone: Full Platform**        | All 8 Kailash packages integrated in a governed AI system                      | Core SDK → DataFlow → ML → Kaizen → PACT → Nexus → Align                                                                               | Existing M6 ex_6: Capstone governed credit system  |

**Key change**: RL is its own lesson (6.5), not crammed into capstone. M6.6 and M6.7 are new lessons covering advanced alignment and agent governance — the "masters level" content.

**New exercises needed**: 6.6, 6.7 (2 new, rest existing)

---

## Exercise Count Summary

| Module    | Existing (keep) | Existing (adapt)                   | Move to other module | New                     | Total  |
| --------- | --------------- | ---------------------------------- | -------------------- | ----------------------- | ------ |
| M1        | 0               | 3 (ex_1,3,5)                       | 2 (ex_2→M2, ex_4→M2) | 5 (1.1,1.2,1.3,1.5,1.8) | 8      |
| M2        | 4 (ex_1,3,4,5)  | 0                                  | 0                    | 2 (2.2,2.4) + 2 from M1 | 8      |
| M3        | 5 (ex_1-3,5,6)  | 1 (ex_4)                           | 0                    | 1 (3.6)                 | 8      |
| M4        | 4 (ex_1,3,4,5)  | 2 (ex_2→use EnsembleEngine, ex_6)  | 0                    | 2 (4.2,4.3)             | 8      |
| M5        | 3 (ex_1,5,6)    | 2 (ex_4→real docs, ex_2+3→combine) | 0                    | 3 (5.1,5.2,5.8)         | 8      |
| M6        | 5 (ex_1-5)      | 1 (ex_6)                           | 0                    | 2 (6.6,6.7)             | 8      |
| **Total** | **21**          | **9**                              | **2**                | **15+2 moved**          | **48** |

**Net new exercises to write**: 15
**Exercises to adapt**: 9 (minor changes to existing)
**Exercises to move**: 2 (M1 ex_2 → M2.1, M1 ex_4 → M2.3)

---

## Complete Kailash SDK Coverage

| Package                     | Lessons                                | Key Primitives                                                                                                                                                                                                                                                                       |
| --------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **kailash** (core)          | M1.4-1.8, M3.5                         | WorkflowBuilder, LocalRuntime, Node, @register_node, PythonCodeNode, CSVReaderNode, logic nodes, ConnectionManager                                                                                                                                                                   |
| **kailash-ml** (13 engines) | M1.6-1.7, M2.1-2.8, M3.1-3.8, M4.1-4.8 | DataExplorer, AlertConfig, PreprocessingPipeline, ModelVisualizer, FeatureStore, FeatureEngineer, ExperimentTracker, TrainingPipeline, HyperparameterSearch, ModelRegistry, AutoMLEngine, EnsembleEngine, DriftMonitor, InferenceServer, OnnxBridge, interop, 6 ML agents, RLTrainer |
| **kailash-dataflow**        | M3.6                                   | @db.model, field(), db.express CRUD                                                                                                                                                                                                                                                  |
| **kailash-nexus**           | M5.1                                   | Nexus multi-channel, auth (RBAC/JWT), middleware, plugins                                                                                                                                                                                                                            |
| **kailash-kaizen**          | M5.3-5.4                               | Signature, InputField/OutputField, BaseAgent, structured output                                                                                                                                                                                                                      |
| **kaizen-agents**           | M5.3-5.8                               | Delegate, 6+ specialized agents, coordination patterns, ML agents, governance                                                                                                                                                                                                        |
| **kailash-pact**            | M6.3-6.4, M6.7                         | GovernanceEngine, PactGovernedAgent, Address, enforcement, costs, MCP integration                                                                                                                                                                                                    |
| **kailash-align**           | M6.1-6.2, M6.6                         | AlignmentPipeline, AlignmentConfig, AdapterRegistry, merge, rewards, evaluator, serving                                                                                                                                                                                              |

**Total Kailash primitives taught**: 100+ classes across all 8 packages
