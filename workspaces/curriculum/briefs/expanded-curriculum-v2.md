# ASCENT Expanded Curriculum v2 — 6 Modules x 8 Lessons

**Official name**: Professional Certificate in Machine Learning (Python) — Terrene Open Academy
**Philosophy**: Learn Python BY building with Kailash from Lesson 1. Every Python concept is grounded in a Kailash use case.
**Progression**: Zero Python knowledge → Masters-level ML engineering
**Structure**: 6 modules x 8 lessons = 48 lessons. Each lesson is a full session (~4 hours).
**Kailash coverage**: Every important engine, primitive, and pattern across all 8 packages.

---

## Module 1: Data Pipelines and Visualisation Mastery with Python

_Zero to productive: learn Python by building Kailash workflows on real Singapore data._

| #   | Lesson Title                                      | Python Taught Through Kailash                                | Kailash SDK Coverage                                                                                                                                                                    |
| --- | ------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1 | **Your First Kailash Workflow**                   | Variables, strings, dicts, assignment, `print()`             | `WorkflowBuilder`, `PythonCodeNode(code="result = {...}")`, `LocalRuntime`, `runtime.execute()`                                                                                         |
| 1.2 | **Nodes, Connections, and Data Flow**             | Functions, return values, parameters, type hints             | `Node` base class, `workflow.connect()`, parameter passing between nodes, `NodeParameter`, `CSVReaderNode`, `TransformNode`                                                             |
| 1.3 | **Custom Nodes and Python Classes**               | Classes, inheritance, `__init__`, methods, decorators        | `@register_node`, custom `Node` subclass, `NodeMetadata`, node input/output ports                                                                                                       |
| 1.4 | **Polars Inside Workflows**                       | DataFrames, expressions, column types, lazy evaluation       | `pl.read_csv()`, `pl.col()`, `with_columns()`, `filter()`, `select()`, polars inside `PythonCodeNode`                                                                                   |
| 1.5 | **Polars at Scale: Joins, Windows, Aggregations** | Advanced data manipulation, method chaining                  | `group_by()`, `over()`, `rolling_mean()`, `join()`, lazy frames, `collect()` — on 15M-row HDB data                                                                                      |
| 1.6 | **Data Visualization and Dashboards**             | Plotly API, figure objects, HTML export                      | `ModelVisualizer` (all 9 chart types: histogram, scatter, heatmap, line, bar, confusion matrix, ROC, PR, calibration)                                                                   |
| 1.7 | **Automated Data Profiling**                      | async/await, error handling, configuration objects           | `DataExplorer`, `AlertConfig` (8 alert types), `DataProfile`, `ColumnProfile`, correlation matrices (Pearson, Spearman, Cramer's V), `DataExplorer.compare()`, `DataExplorer.to_html()` |
| 1.8 | **End-to-End Data Pipeline**                      | Putting it all together: modules, imports, project structure | `PreprocessingPipeline` (auto-detect task, encode, scale, impute), `SetupResult`, `ConnectionManager`, environment setup with `.env`                                                    |

**SDK Primitives Introduced**: `WorkflowBuilder`, `LocalRuntime`, `Node`, `NodeMetadata`, `NodeParameter`, `@register_node`, `PythonCodeNode`, `CSVReaderNode`, `TransformNode`, `ModelVisualizer`, `DataExplorer`, `AlertConfig`, `DataProfile`, `ColumnProfile`, `PreprocessingPipeline`, `SetupResult`, `ConnectionManager`

**Datasets**: Singapore HDB Resale Prices (15M+), MRT stations, schools
**By end of module**: Build complete data pipelines, write custom nodes, create interactive visualizations, interpret automated data quality reports.

---

## Module 2: Statistical Mastery for ML and AI Success

_Statistical foundations taught through experiment tracking, feature stores, and workflow orchestration._

| #   | Lesson Title                              | Theory + Python                                                               | Kailash SDK Coverage                                                                                                                                                                                              |
| --- | ----------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1 | **Probability and Distributions**         | Exponential family, Bayes' theorem, conjugate priors, `numpy`/`scipy`         | `DataExplorer` distribution analysis, `ModelVisualizer` for posterior plots                                                                                                                                       |
| 2.2 | **Estimation Theory**                     | MLE, MAP, Fisher information, Cramér-Rao bound, numerical optimization        | Workflow: `PythonCodeNode` for MLE → `TransformNode` for results → `ModelVisualizer`                                                                                                                              |
| 2.3 | **Hypothesis Testing and Power**          | Neyman-Pearson, power analysis, effect sizes, FDR corrections                 | `ExperimentTracker` — `create_experiment()`, context manager `async with tracker.run()`, `log_param()`, `log_metric()`                                                                                            |
| 2.4 | **Bootstrap and Permutation Tests**       | BCa intervals, resampling theory, distribution-free methods                   | `ExperimentTracker.compare_runs()`, `ModelVisualizer.metric_comparison()`                                                                                                                                         |
| 2.5 | **A/B Testing at Production Scale**       | SRM, MDE, CUPED (derive variance reduction), sequential testing, Bayesian A/B | `ExperimentTracker` full lifecycle: `start_run()` → `log_metrics()` → `end_run()` → `search_runs()`                                                                                                               |
| 2.6 | **Causal Inference**                      | DiD, propensity matching, potential outcomes (Rubin), DAGs (Pearl)            | Workflows for causal pipelines, `ExperimentTracker` for analysis logging                                                                                                                                          |
| 2.7 | **Feature Engineering and Feature Store** | Temporal features, target encoding, point-in-time correctness, data lineage   | `FeatureStore` (`register_features()`, `store()`, `get_features()`, `get_training_set()`), `FeatureSchema`, `FeatureField`, `FeatureEngineer` (`generate()`, `select()`), `GeneratedFeatures`, `SelectedFeatures` |
| 2.8 | **Experiment-Driven Statistical Project** | Full experiment lifecycle: design → execute → analyze → report                | `ExperimentTracker` + `FeatureStore` + `ModelVisualizer` + workflow orchestration, `ExperimentTracker.get_metric_history()`, `RunComparison`                                                                      |

**SDK Primitives Introduced**: `ExperimentTracker`, `RunContext`, `Experiment`, `Run`, `MetricEntry`, `RunComparison`, `FeatureStore`, `FeatureSchema`, `FeatureField`, `FeatureEngineer`, `GeneratedFeatures`, `SelectedFeatures`, `FeatureRank`

**Datasets**: Singapore economic indicators, e-commerce A/B test (500K), Singapore housing + cooling measures
**By end of module**: Design experiments, compute proper intervals, detect causal effects, manage features with lineage, track all work reproducibly.

---

## Module 3: Supervised ML for Building and Deploying Models

_From bias-variance theory to production model registry — all orchestrated through Kailash._

| #   | Lesson Title                                 | Theory                                                                      | Kailash SDK Coverage                                                                                                                                                                                                                                             |
| --- | -------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.1 | **Bias-Variance and Regularization**         | Full decomposition, L1/L2 geometry, Bayesian interpretation, double descent | `PreprocessingPipeline`, `kailash_ml.interop` (`to_sklearn_input()`, `from_sklearn_output()`, `polars_to_arrow()`, `to_lgb_dataset()`)                                                                                                                           |
| 3.2 | **Gradient Boosting Deep Dive**              | XGBoost 2nd-order Taylor, LightGBM GOSS, CatBoost ordered boosting          | `TrainingPipeline`, `ModelSpec` (model_class, hyperparameters, framework), `EvalSpec` (metrics, split_strategy, n_splits, test_size, min_threshold)                                                                                                              |
| 3.3 | **Class Imbalance and Calibration**          | SMOTE failures, cost-sensitive, Focal Loss, Platt/isotonic, proper scoring  | `TrainingPipeline` with custom objectives, `ModelVisualizer.calibration_curve()`, `ModelVisualizer.precision_recall_curve()`                                                                                                                                     |
| 3.4 | **SHAP, LIME, and Fairness**                 | Shapley axioms, TreeSHAP, counterfactual explanations, disparate impact     | `ModelVisualizer.feature_importance()`, fairness audit patterns                                                                                                                                                                                                  |
| 3.5 | **Workflow Orchestration**                   | Kailash Core SDK workflow patterns, cyclic workflows, convergence           | `WorkflowBuilder` advanced: multi-node pipelines, `workflow.build()`, logic nodes (`ConditionalNode`, `LoopNode`, `MergeNode`), cycle-aware nodes, convergence criteria                                                                                          |
| 3.6 | **DataFlow: Database Persistence**           | ORM patterns, schema design, CRUD operations                                | `DataFlow` (`@db.model`, `field()`, `db.express.create()`, `db.express.list()`, `db.express.get()`, `db.express.update()`), `ConnectionManager` with SQLite and PostgreSQL                                                                                       |
| 3.7 | **Model Registry and Hyperparameter Search** | Bayesian optimization, model versioning, promotion gates                    | `HyperparameterSearch` (`SearchSpace`, `SearchConfig`, `ParamDistribution`, `TrialResult`, `SearchResult`), `ModelRegistry` (`register_model()`, `promote_model()`, `get_model()`, `list_models()`, `compare()`), `MetricSpec`, `ModelSignature`, `ModelVersion` |
| 3.8 | **Production Supervised ML Project**         | Full pipeline with model card, conformal prediction                         | Complete: `PreprocessingPipeline` → `TrainingPipeline` → `ModelRegistry` → `DataFlow` persistence, model card generation                                                                                                                                         |

**SDK Primitives Introduced**: `TrainingPipeline`, `ModelSpec`, `EvalSpec`, `TrainingResult`, `HyperparameterSearch`, `SearchSpace`, `SearchConfig`, `ParamDistribution`, `TrialResult`, `SearchResult`, `ModelRegistry`, `ModelVersion`, `MetricSpec`, `ModelSignature`, `DataFlow`, `@db.model`, `field()`, `db.express`, `kailash_ml.interop` (all functions), logic nodes, cycle-aware patterns

**Datasets**: Singapore credit scoring (100K), Lending Club (300K+)
**By end of module**: Train, evaluate, explain, register, promote, and persist supervised ML models through Kailash workflows with full audit trails.

---

## Module 4: Unsupervised ML and Advanced Techniques

_Pattern discovery, NLP, anomaly detection, and production monitoring — with Kailash automation and deployment._

| #   | Lesson Title                            | Theory                                                                 | Kailash SDK Coverage                                                                                                                                                                                         |
| --- | --------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 4.1 | **Clustering Foundations**              | K-means, spectral, HDBSCAN, gap statistic, validation metrics          | `AutoMLEngine`, `AutoMLConfig` (task_type, metric, direction, candidate_families, search_strategy, agent double opt-in, `max_llm_cost_usd`, `LLMCostTracker`)                                                |
| 4.2 | **EM Algorithm and Gaussian Mixtures**  | Full E-step/M-step derivation, convergence (ELBO), soft assignment     | `AutoMLEngine` algorithm comparison, `CandidateResult`                                                                                                                                                       |
| 4.3 | **Dimensionality Reduction**            | PCA/SVD, UMAP (topological), t-SNE (KL divergence), embeddings         | `ModelVisualizer` for embedding visualization                                                                                                                                                                |
| 4.4 | **Anomaly Detection and Ensembles**     | Isolation Forest, LOF, autoencoder scoring, score blending             | `EnsembleEngine` (`blend()`, `stack()`, `bag()`, `boost()`), `BlendResult`, `StackResult`, `BagResult`, `BoostResult`                                                                                        |
| 4.5 | **NLP: Text to Vectors to Topics**      | TF-IDF, BM25, Word2Vec, BERTopic (UMAP+HDBSCAN+c-TF-IDF), coherence    | Text processing in polars, `ModelVisualizer` for topics                                                                                                                                                      |
| 4.6 | **Production Drift Monitoring**         | PSI, KS test, performance degradation, monitoring obligations          | `DriftMonitor` (`set_reference()`, `check_drift()`, `check_performance()`, `get_drift_history()`, `schedule_monitoring()`), `DriftSpec`, `DriftReport`, `FeatureDriftResult`, `PerformanceDegradationReport` |
| 4.7 | **ONNX Export and Inference Serving**   | Model portability, ONNX format, serving patterns, signature validation | `OnnxBridge` (`check_compatibility()`, `export()`, `validate()`), `InferenceServer` (`predict()`, `predict_batch()`, `warm_cache()`, `register_endpoints()`), `PredictionResult`                             |
| 4.8 | **Multi-Channel Deployment with Nexus** | API + CLI + MCP from single codebase, auth, middleware                 | `Nexus` (register workflows, `create_session()`), Nexus auth (`RBAC`, `JWT`), middleware (`CSRF`, security headers, cache), `nexus.plugins`, `nexus.metrics`                                                 |

**SDK Primitives Introduced**: `AutoMLEngine`, `AutoMLConfig`, `LLMCostTracker`, `CandidateResult`, `AgentRecommendation`, `EnsembleEngine` (4 methods + result types), `DriftMonitor`, `DriftSpec`, `DriftReport`, `FeatureDriftResult`, `PerformanceDegradationReport`, `OnnxBridge`, `OnnxCompatibility`, `OnnxExportResult`, `OnnxValidationResult`, `InferenceServer`, `PredictionResult`, `Nexus`, Nexus auth/middleware/plugins

**Datasets**: E-commerce transactions (200K), credit card fraud (284K), Singapore news (50K), ChestX-ray14 subset
**By end of module**: Discover patterns, detect anomalies, monitor drift, export to ONNX, deploy as multi-channel API with authentication.

---

## Module 5: Deep Learning and Vision

_Neural networks from first principles to production, with MCP server development and advanced deployment._

| #   | Lesson Title                            | Theory                                                                                      | Kailash SDK Coverage                                                                                     |
| --- | --------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 5.1 | **Neural Network Foundations**          | Universal approximation, backprop chain rule, gradient flow, vanishing/exploding            | Manual backprop in `PythonCodeNode`, `ModelVisualizer.training_history()`                                |
| 5.2 | **CNN Architecture**                    | Convolutions, pooling, ResNet skip connections, receptive field theory                      | PyTorch models + `ModelVisualizer` for learning curves and metrics                                       |
| 5.3 | **Training Dynamics**                   | LR scheduling (cosine, OneCycle, warmup), BatchNorm vs LayerNorm, AdamW, gradient clipping  | `ModelVisualizer.training_history()`, `ModelVisualizer.learning_curve()`                                 |
| 5.4 | **Attention from First Principles**     | Q/K/V derivation, scaled dot-product (why sqrt(d_k)), multi-head attention, Flash Attention | Architecture walkthrough using Kailash codebase patterns                                                 |
| 5.5 | **Transformer Architecture**            | Encoder/decoder, positional encoding (RoPE derivation), GQA, KV-cache                       | Kailash codebase as reference (kaizen internals use transformers)                                        |
| 5.6 | **Computer Vision Project**             | Medical image classification, mixed precision, data augmentation                            | PyTorch → `OnnxBridge.export()` → `OnnxBridge.validate()`                                                |
| 5.7 | **MCP Server Development**              | MCP protocol, tool registration, resource management, transports                            | `kailash.mcp_server` (build custom MCP servers), `kailash.mcp` (contrib tools), MCP integration patterns |
| 5.8 | **Production Deep Learning Deployment** | Full stack: train → ONNX → serve → multi-channel → monitor                                  | `InferenceServer` + `Nexus` + `DriftMonitor` + MCP server, kailash `monitoring` module                   |

**SDK Primitives Introduced**: `kailash.mcp_server` (MCP server creation, tool registration), `kailash.mcp` (contrib tools, client), `kailash.monitoring` module, advanced `InferenceServer` + `Nexus` patterns

**Datasets**: ChestX-ray14 subset (10K), synthetic vision data
**By end of module**: Understand attention/transformers, build custom MCP servers, deploy deep learning as governed production services.

---

## Module 6: Language Models and Agentic Workflows for Organisational Transformation

_LLMs, the full Kaizen agent framework, alignment, governance, RL — the complete AI engineering toolkit._

| #   | Lesson Title                           | Theory                                                                                   | Kailash SDK Coverage                                                                                                                                                                                                                                                                                                               |
| --- | -------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6.1 | **LLMs and the Kaizen Framework**      | Tokenization, scaling laws, inference optimization, quantization                         | `Kaizen` core: `Signature`, `InputField`, `OutputField`, `BaseAgent`, `Delegate` (full lifecycle: creation, streaming, events, hooks, cost tracking), `kaizen.core.structured_output`, `kaizen.core.config`                                                                                                                        |
| 6.2 | **Specialized Agents**                 | Agent paradigms: QA, reasoning, tool use, retrieval                                      | `SimpleQAAgent`, `ChainOfThoughtAgent`, `ReActAgent`, `RAGResearchAgent`, `PlanningAgent`, `SelfReflectionAgent`, `HumanApprovalAgent`, `BatchProcessingAgent`, `CodeGenerationAgent`, `StreamingChatAgent`, `TreeOfThoughtsAgent`                                                                                                 |
| 6.3 | **RAG Systems**                        | Chunking, retrieval (dense/sparse/hybrid), RAGAS evaluation, HyDE, CRAG                  | `RAGResearchAgent`, retrieval evaluation, `MemoryAgent`                                                                                                                                                                                                                                                                            |
| 6.4 | **Multi-Agent Orchestration**          | A2A protocol, coordination patterns, agent lifecycle                                     | Multi-agent patterns: `SupervisorWorkerPattern`, `DebatePattern`, `ConsensusPattern`, `EnsemblePattern`, `ParallelPattern`, `SequentialPattern`, `BlackboardPattern`, `HandoffPattern`. `kaizen_agents.coordination`, `kaizen_agents.orchestration` (planner, recovery, protocols), `Supervisor`                                   |
| 6.5 | **ML Agents and Autonomous Workflows** | LLMs augmenting ML lifecycle, agent confidence, double opt-in                            | 6 ML agents: `DataScientistAgent`, `FeatureEngineerAgent`, `ModelSelectorAgent`, `ExperimentInterpreterAgent`, `DriftAnalystAgent`, `RetrainingDecisionAgent`. Journey system: `kaizen_agents.journey` (intent, transitions, behaviors, state)                                                                                     |
| 6.6 | **LLM Fine-Tuning and Alignment**      | LoRA derivation, QLoRA NF4, SFT, DPO from Bradley-Terry, GRPO, model merging (TIES/DARE) | `kailash-align`: `AlignmentPipeline`, `AlignmentConfig`, `AdapterRegistry`, `method_registry`, `merge` (model merging), `rewards` (reward functions), `evaluator` (LLM-as-judge, BERTScore), `serving` (vLLM backend), `gpu_memory`, `onprem`                                                                                      |
| 6.7 | **AI Governance with PACT**            | EU AI Act, Singapore AI Verify, D/T/R grammar, operating envelopes, monotonic tightening | `kailash-pact`: `GovernanceEngine` (`engine.py`), `PactGovernedAgent`, `Address`, `compile_org()`, `load_org_yaml()`, `enforcement` modes, `costs` (CostTracker), `events`, `work` (task envelopes). PACT MCP integration: `pact.mcp.enforcer`, `pact.mcp.audit`. Governance CLI: `pact.governance.cli`, `pact.governance.testing` |
| 6.8 | **Capstone: Governed AI System**       | RL foundations (Bellman, PPO), full platform integration                                 | `kailash_ml.rl` (`RLTrainer`, `env_registry`, `policy_registry`). **Full platform capstone**: Core SDK → DataFlow → ML → Kaizen → PACT → Nexus → Align. Agent governance: `kaizen_agents.governance` (clearance, budget, bypass, dereliction, vacancy, accountability, cascade)                                                    |

**SDK Primitives Introduced**: Full `kaizen` core (`Signature`, `InputField`, `OutputField`, `BaseAgent`, structured output, config), full `kaizen_agents` (14 specialized agents, 9 multi-agent patterns, journey system, orchestration, governance), full `kailash-align` (pipeline, config, adapter registry, method registry, merge, rewards, evaluator, serving, onprem), full `kailash-pact` (engine, enforcement, costs, events, work, MCP integration, CLI, testing), `kailash_ml.rl` (trainer, env registry, policy registry), `kailash_ml.dashboard`

**Datasets**: Domain Q&A pairs, preference pairs, Singapore regulatory corpus, Gymnasium environments
**By end of module**: Build, fine-tune, govern, and deploy complete AI systems using every Kailash package.

---

## Complete SDK Coverage Map

| Package              | Files | Modules Where Taught | Key Classes/Functions Covered                                                                                                                                                                                                                                                      |
| -------------------- | ----- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **kailash** (core)   | 661   | M1-M5                | WorkflowBuilder, LocalRuntime, Node, PythonCodeNode, CSVReaderNode, TransformNode, logic nodes, ConnectionManager, MCP server/client, monitoring                                                                                                                                   |
| **kailash-ml**       | 42    | M1-M4, M6            | DataExplorer, PreprocessingPipeline, ModelVisualizer, FeatureStore, FeatureEngineer, ExperimentTracker, TrainingPipeline, HyperparameterSearch, ModelRegistry, AutoMLEngine, EnsembleEngine, DriftMonitor, InferenceServer, OnnxBridge, interop, 6 ML agents, RLTrainer, dashboard |
| **kailash-dataflow** | 280   | M3                   | @db.model, field(), db.express CRUD, ConnectionManager, migrations basics                                                                                                                                                                                                          |
| **kailash-nexus**    | 76    | M4-M5                | Nexus multi-channel, auth (RBAC/JWT), middleware, plugins, metrics                                                                                                                                                                                                                 |
| **kailash-kaizen**   | 465   | M6                   | Signature, InputField/OutputField, BaseAgent, structured output, config                                                                                                                                                                                                            |
| **kaizen-agents**    | 173   | M6                   | Delegate, 14 specialized agents, 9 multi-agent patterns, journey system, orchestration, governance                                                                                                                                                                                 |
| **kailash-pact**     | 27    | M6                   | GovernanceEngine, enforcement, costs, MCP integration, CLI, testing                                                                                                                                                                                                                |
| **kailash-align**    | 17    | M6                   | AlignmentPipeline, AlignmentConfig, AdapterRegistry, merge, rewards, evaluator, serving                                                                                                                                                                                            |

**Total Kailash primitives taught**: 100+ classes/functions across 8 packages
**Total lessons**: 48 (6 modules x 8 lessons)
**Python concepts covered**: variables → classes → async → decorators → metaclasses (progressive through modules)

---

## Cross-Cutting Principles

1. **Kailash from Lesson 1.1** — first code is WorkflowBuilder + PythonCodeNode
2. **Python taught through SDK** — every concept grounded in a Kailash use case
3. **Real data always** — Singapore datasets, never iris/titanic
4. **Progressive depth** — M1 beginner, M2-M3 intermediate, M4-M5 advanced, M6 masters
5. **Every engine covered** — no important Kailash primitive left unexercised
6. **Governance thread** — data quality (M1) → lineage (M2) → provenance (M3) → monitoring (M4) → cost budgets (M5) → full PACT (M6)
7. **Kailash codebase as material** — SDK source, tests, and examples used as teaching references
8. **8 full lessons per module** — substantial sessions, not compressed lectures
