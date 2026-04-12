# MLFP Curriculum v1 — Authoritative Spec (6 Modules x 8 Lessons)

**Official name**: ML Foundations for Professionals — Terrene Open Academy
**Audience**: Working professionals with ZERO Python to production ML engineering
**Structure**: 6 modules x 8 lessons = 48 lessons (~4 hours each, ~192 contact hours)
**Certification**: Foundation Certificate (M1-M4) + Advanced Certificate (M5-M6)
**Stack**: Kailash SDK exclusively. No PyCaret, no ydata_profiling, no broken dependency chains.

## First Principles

1. **Kailash-only stack** — Every ML operation uses Kailash engines. No PyCaret (broken deps, slow install). No ydata_profiling (conflicts). DataExplorer replaces profiling. AutoMLEngine replaces AutoML. TrainingPipeline replaces sklearn pipelines.
2. **Python through data** — Every Python concept grounded in data manipulation from Lesson 1.1.
3. **Kailash engines before infrastructure** — Use DataExplorer/ModelVisualizer (M1) before WorkflowBuilder/custom nodes (M3).
4. **The Feature Engineering Spectrum** — The organising spine of the entire curriculum (from R5 Deck 5B "Unsupervised meets Supervised Learning"):
   - M3: Manual feature engineering (human designs features)
   - M4.1-4.6: USML discovers features independently (no labels, no error signal)
   - M4.7: Collaborative filtering learns embeddings through optimisation (the pivot)
   - M4.8: DL generalises — hidden layers are automated feature engineering with error feedback. 2+ hidden layers can write any non-linear combination (representation learning). Node values = embeddings.
   - M5: Specialised architectures learn domain-specific features (vision, sequence, graph, generative)
   - M6: LLMs learn semantic features from language at scale
5. **Engineering, not philosophy** — Every concept is taught through implementation. Governance is access controls you code, not frameworks you discuss.
6. **Progressive scaffolding** — M1 ~70%, M2 ~60%, M3 ~50%, M4 ~40%, M5 ~30%, M6 ~20%.
7. **Statistics teaches models, SML teaches the pipeline** — Following R5 delivery: regression/logistic/ANOVA are taught as inferential statistics (M2) before the ML pipeline (M3). M3 does NOT re-teach these models — it builds on them.

## Reference Materials

**Source program**: ASCENT (lyceum/programs/ascent/) — 10-module canonical knowledge base
**R5 delivery reference**: Google Drive PCML_DIS_R5_2601 — 13 decks (275+ slides), 38 notebooks
**Deck inventory from R5**:
- Deck 1A: Introduction (18 slides) — 6-module overview, AI context, terminology, mastery philosophy
- Deck 1B: Python Fundamentals — data types, operators, variables, strings, flow control, collections
- Deck 1C: Visualization & Dashboarding (30 slides) — viz principles, Gestalt, chart types, Plotly, APIs
- Deck 2A: Descriptive Statistics (37 slides) — variable types, 3Ms, variation, outliers, Z-score, IQR, quantiles
- Deck 2B: Data Exploration (19 slides) — EDA, Pandas operations, merging, cleaning, transformation
- Deck 2C: Probability (large) — truth tables, conditional probability, Bayes' theorem, joint probability
- Deck 3A: Inferential Statistics (55 slides) — parameter estimation, bootstrapping, A/B testing, data collection framework, hypothesis testing, linear regression (OLS, t-stat, R-squared, F-stat, multivariate, categorical), logistic regression (log-odds), ANOVA (one-way, two-way, repeated measures, post-hoc)
- Deck 4A: Supervised ML — ML pipeline, feature engineering (geocoding), statistics vs ML, AutoML
- Deck 4B: ML Pipeline & MLOps (11 slides) — training pipeline, MLOps components, clean architecture
- Deck 5A: Unsupervised ML — USML as automated feature engineering, K-means, hierarchical (dendrograms, linkage), t-SNE, PCA (2-step), NLP text classification/clustering
- Deck 5B: Basics of Deep Learning (42 slides) — neural network architecture, forward pass, gradient descent (step-by-step), linear regression as NN, feature interaction, activation functions, hidden layers as automated feature engineering, representation learning, embeddings, "unsupervised meets supervised learning"
- Deck 6A: Specialised DL Models — autoencoders (9 variants), CNNs (LeNet-5 through ResNet), RNNs (LSTM gates, GRU, perplexity/BLEU), Transformers (BERT, GPT, T5, Transformer-XL, Reformer), GNNs (GCN, GraphSAGE, GAT, GIN), GANs (DCGAN, cGAN, WGAN, CycleGAN, StyleGAN), data generation models
- Deck 6B: LLMs & Agentic Workflows (15 slides) — LLM foundation models, 10 fine-tuning techniques (LoRA, Adapters, Prefix Tuning, Prompt Tuning, Task-specific, LLRD, Progressive Layer Freezing, Knowledge Distillation, Differential Privacy/DPSGD, Elastic Weight Consolidation/Fisher), agentic workflow design, agent frameworks

---

## Module 1: Machine Learning Data Pipelines and Visualisation Mastery with Python

_Zero to productive. Learn Python by exploring real Singapore data._

| #   | Lesson                                   | Python Concepts                                                                 | Kailash / Tools                                                                           | R5 Source          |
| --- | ---------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------ |
| 1.1 | **Your First Data Exploration**          | Variables, strings, numbers, `print()`, f-strings. Then: `pl.read_csv()`, `df.shape`, `df.head()` | `polars`: read_csv, shape, columns, head, describe                                        | Deck 1B + PCML1-1 |
| 1.2 | **Filtering and Transforming Data**      | Booleans, comparison operators, method chaining                                 | `pl.col()`, `filter()`, `select()`, `sort()`, `with_columns()`                            | Deck 1B + PCML1-2 |
| 1.3 | **Functions and Aggregation**            | `def`, parameters, `return`, `for` loops, lists, dicts                          | `group_by()`, `agg()`, `pl.mean()`, writing helper functions                              | Deck 1B + PCML1-3 |
| 1.4 | **Joins and Multi-Table Data**           | `if/else`, imports, packages. Join concepts (left, inner)                       | `join()`, multi-table operations on HDB 15M rows                                          | Deck 2B (merging) + ASCENT |
| 1.5 | **Window Functions and Trends**          | `over()`, `rolling_mean()`, `shift()`, lazy frames, `collect()`                 | Window functions, rolling aggregations, YoY calculations                                  | ASCENT M1 |
| 1.6 | **Data Visualisation**                   | Plotly API, figure objects, chart selection, Gestalt principles                  | `ModelVisualizer` (histogram, scatter, bar, heatmap, line)                                | Deck 1C (30 slides on viz principles, chart types) |
| 1.7 | **Automated Data Profiling**             | Classes as users. Async hidden behind sync wrapper. `try/except`                | `DataExplorer`, `AlertConfig`, 8 alert types, `DataProfile`, `compare()`                  | ASCENT M1 ex_3 |
| 1.8 | **Data Pipelines and End-to-End Project** | None/null handling, ETL concepts, APIs, project structure                       | `PreprocessingPipeline` (auto-detect, encode, scale, impute). Full: load -> profile -> clean -> visualise -> report | Deck 1C (APIs) + PCML1-5 (ETL) |

**Scaffolding**: ~80% for 1.1-1.3, ~70% for 1.4-1.8
**Kailash engines**: DataExplorer, PreprocessingPipeline, ModelVisualizer
**R5 fidelity**: Follows Deck 1A→1B→1C flow. Visualization principles from Deck 1C (Gestalt, truthfulness, chart selection). ETL/API from Deck 1C + PCML1-5.

---

## Module 2: Statistical Mastery for Machine Learning and Artificial Intelligence (AI) Success

_Statistical foundations including regression models — taught as inference tools before the ML pipeline._

Following R5 Deck 3A structure: regression and logistic regression are inferential statistics techniques, taught here BEFORE the SML module. Feature engineering moves to M3 where it belongs (following Deck 4A).

| #   | Lesson                                      | Theory                                                                                                                    | Kailash SDK                                                                     | R5 Source          |
| --- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------ |
| 2.1 | **Probability and Bayesian Thinking**       | Distributions (Normal, Beta, Poisson), conditional probability, Bayes' theorem, joint probability, truth tables            | `ModelVisualizer` for posterior plots                                           | Deck 2A (expected value, sampling bias) + Deck 2C (full probability theory) |
| 2.2 | **Parameter Estimation and Inference**      | Sampling distributions, confidence intervals, PDF/CDF, MLE derivation, MAP, optimal parameters                            | `ExperimentTracker`                                                             | Deck 3A slides 5-10 |
| 2.3 | **Bootstrapping and Hypothesis Testing**    | Bootstrapping (resampling, CI from percentiles), permutation tests, null/alternative hypothesis, p-values, significance    | `ExperimentTracker` (create_experiment, log_param, log_metric)                  | Deck 3A slides 11-35 |
| 2.4 | **A/B Testing and Data Collection**         | A/B test design, randomisation, data collection framework (Why/What/Where/How/Frequency), DataOps architecture            | `ExperimentTracker` full lifecycle                                              | Deck 3A slides 15-26 (very detailed) |
| 2.5 | **Linear Regression**                       | OLS, coefficients (direction + magnitude), t-statistic, R-squared, F-statistic, multivariate, categorical encoding, loglinear | `TrainingPipeline`, `ModelVisualizer`                                          | Deck 3A slides 36-45 (comprehensive) |
| 2.6 | **Logistic Regression and ANOVA**           | Logistic: log-odds, link function, odds interpretation. ANOVA: one-way, two-way, repeated measures, post-hoc (Tukey, Bonferroni, Scheffe) | `TrainingPipeline`, `ModelVisualizer`                                          | Deck 3A slides 49-54 |
| 2.7 | **CUPED and Causal Inference**              | CUPED (derive Var(Y_adj) = Var(Y)(1-rho^2)), SRM detection, DiD (derive ATT), propensity matching, parallel trends         | `ExperimentTracker`, `ModelVisualizer`                                          | ASCENT (new, not in R5) |
| 2.8 | **Capstone: Statistical Analysis Project**  | End-to-end: load data → descriptive stats → hypothesis test → regression model → interpret → report                       | All M2 engines                                                                  | R5 PCML3-6 "Putting It Together" (wine dataset) |

**Scaffolding**: ~60%
**Kailash engines**: ExperimentTracker, TrainingPipeline, ModelVisualizer
**Key R5 fidelity**: Deck 3A teaches regression/logistic/ANOVA as STATISTICS before the ML pipeline. M2 follows this. Students learn what a model IS (inference) before learning how to engineer the pipeline (M3).
**Not in R5 (new for MLFP)**: CUPED, causal inference (DiD, propensity matching). These are ASCENT additions.

---

## Module 3: Supervised Machine Learning for Building and Deploying Models

_The ML pipeline — from feature engineering to production deployment. Builds on M2's regression foundation._

Following R5 Deck 4A: this module teaches the ML PIPELINE and advanced models, not basic regression (already taught in M2).

| #   | Lesson                                       | Theory                                                                                                           | Kailash SDK                                                                                                  | R5 Source          |
| --- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------ |
| 3.1 | **Feature Engineering and the ML Pipeline**  | Feature engineering (geocoding, temporal, interaction terms, leakage detection). ML pipeline stages (Deck 4A). Data > Models > Hyperparameters pecking order. | `FeatureEngineer`, `FeatureStore`, `FeatureSchema`                                                           | Deck 4A (feature engineering, pipeline) |
| 3.2 | **Bias-Variance and Regularisation**         | Full decomposition. L1/L2 geometry (builds on M2.5 regression). Bayesian interpretation (L2 = Gaussian prior). Cross-validation (k-fold, stratified, time-series). | `PreprocessingPipeline`, `kailash_ml.interop`                                                                | ASCENT |
| 3.3 | **Gradient Boosting and the Model Zoo**      | Decision trees (splitting, pruning). Random forests (bagging). XGBoost (2nd-order Taylor), LightGBM (GOSS), CatBoost (ordered). Model comparison. | `TrainingPipeline`, `ModelSpec`, `EvalSpec`                                                                  | Deck 4B (model monitoring shows 18 models) + ASCENT |
| 3.4 | **Class Imbalance and Calibration**          | SMOTE failures, cost-sensitive learning, Focal Loss. Calibration: Platt scaling, isotonic regression, proper scoring rules. Evaluation: precision, recall, F1, ROC-AUC, confusion matrix. | `ModelVisualizer.calibration_curve()`, `precision_recall_curve()`                                            | ASCENT |
| 3.5 | **Interpretability and Fairness**            | SHAP axioms (efficiency, symmetry, dummy, linearity), TreeSHAP, LIME, ALE plots. Fairness: disparate impact, impossibility theorem. | `ModelVisualizer.feature_importance()`                                                                       | ASCENT (new, not in R5) |
| 3.6 | **Workflow Orchestration and Custom Nodes**  | WorkflowBuilder, nodes, connections, runtime. @register_node, Node subclass, conditional logic nodes.            | `WorkflowBuilder`, `LocalRuntime`, `PythonCodeNode`, `ConditionalNode`                                       | ASCENT |
| 3.7 | **Model Registry and Hyperparameter Search** | Bayesian optimisation, SearchSpace, model versioning, staging->production promotion.                              | `HyperparameterSearch`, `SearchSpace`, `ModelRegistry`, `MetricSpec`, `ModelSignature`                       | Deck 4B (MLOps: experiment tracking, versioning, governance) + ASCENT |
| 3.8 | **Production Pipeline: DataFlow, Drift, Deployment** | DataFlow persistence (@db.model, CRUD). DriftMonitor (PSI, KS). Full pipeline: train -> persist -> register -> promote -> monitor -> model card. | `DataFlow`, `DriftMonitor`, `DriftSpec`, `ConnectionManager`                                                 | Deck 4B (MLOps: monitoring, CI/CD) + ASCENT |

**Scaffolding**: ~50%
**Kailash engines**: FeatureEngineer, FeatureStore, TrainingPipeline, HyperparameterSearch, ModelRegistry, WorkflowBuilder, DataFlow, DriftMonitor
**R5 fidelity**: Follows Deck 4A (feature engineering + pipeline) → Deck 4B (MLOps). PyCaret replaced by Kailash TrainingPipeline. ydata_profiling replaced by DataExplorer (M1). MLflow concepts mapped to ModelRegistry + ExperimentTracker.
**Not in R5 (new for MLFP)**: SHAP/LIME, bias-variance derivation, Kailash WorkflowBuilder, DataFlow, DriftMonitor.

---

## Module 4: Unsupervised Machine Learning and Advanced Techniques for Insights

_Pattern discovery without labels, then the bridge to neural feature learning._

Following R5 Deck 5A (USML as automated feature engineering) and Deck 5B (DL as "unsupervised meets supervised learning").

| #   | Lesson                                               | Theory                                                                                                                    | Kailash SDK                                                               | R5 Source          |
| --- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------ |
| 4.1 | **Clustering**                                       | K-means, hierarchical (agglomerative, divisive, dendrograms, linkage: single/complete/average/Ward's), spectral, HDBSCAN. Evaluation: silhouette, Davies-Bouldin, Calinski-Harabasz, elbow, gap statistic. Customer segmentation. | `AutoMLEngine`, `AutoMLConfig`                                            | Deck 5A (K-means, hierarchical with full linkage methods, dendrograms) + PCML5-1 |
| 4.2 | **EM Algorithm and GMMs**                            | E-step/M-step derivation (intuition + 20-line implementation), soft assignment, convergence. EM as general template. Mixture of Experts as modern application. | `AutoMLEngine` comparison                                                 | New (not in R5) |
| 4.3 | **Dimensionality Reduction**                         | PCA deep (2-step: decorrelate then reduce, SVD connection, intrinsic dimension, variance explanation, scree, loadings). Kernel PCA. UMAP/t-SNE (use, interpret, tune). PCA as feature extraction. | `ModelVisualizer` for embeddings                                          | Deck 5A (PCA 2-step, t-SNE) |
| 4.4 | **Anomaly Detection and Ensembles**                  | Isolation Forest, LOF, score blending. Statistical outlier detection connects back to M2 (Z-score, IQR from Deck 2A). | `EnsembleEngine` (blend, stack, bag, boost)                               | Deck 2A (outlier detection: Z-score, IQR, winsorization) + ASCENT |
| 4.5 | **Association Rules and Market Basket Analysis**     | Apriori algorithm, FP-Growth, support/confidence/lift, transactional pattern discovery.                                   | Custom implementation or `AutoMLEngine`                                   | New (not in R5) |
| 4.6 | **NLP: Text to Topics**                              | TF-IDF derivation, BM25, Word2Vec (+ GloVe, FastText), LDA, NMF, BERTopic, NPMI coherence. Text classification vs clustering. | `ModelVisualizer` for topics                                              | Deck 5A (NLP text structure, classification vs clustering, TF-IDF, NMF) + PCML5-2 |
| 4.7 | **Recommender Systems and Collaborative Filtering**  | Matrix factorisation, collaborative filtering as embedding learning, content-based, hybrid. The pivot: optimisation drives feature discovery — connects to Deck 5B's representation learning. | `ModelVisualizer`, custom implementation                                  | PCML5-3 (sentiment + recommenders) |
| 4.8 | **DL Foundations: Neural Networks and Backpropagation** | Neural network architecture (input/hidden/output). Forward pass. Gradient descent (step-by-step, from Deck 5B). Linear regression replicated as NN. Feature interaction → activation functions → non-linearity. Hidden layers as automated feature engineering. Representation learning. Embeddings. | PyTorch + `ModelVisualizer.training_history()` + `OnnxBridge`             | Deck 5B (42 slides — the "Unsupervised meets Supervised Learning" deck) |

**Scaffolding**: ~40%
**Kailash engines**: AutoMLEngine, EnsembleEngine, ModelVisualizer, OnnxBridge
**R5 fidelity**: Deck 5A structure (clustering → PCA → NLP) preserved. Deck 5B's explicit "unsupervised meets supervised learning" framing is the bridge. Hierarchical clustering with all 4 linkage methods comes directly from Deck 5A.
**Not in R5 (new for MLFP)**: HDBSCAN, UMAP, EM/GMMs, anomaly detection (beyond Z-score/IQR), market basket analysis, recommender systems as full lesson.

---

## Module 5: Deep Learning and Machine Learning Mastery in Vision and Transfer Learning

_Every major DL architecture. One paradigm per lesson. All implemented._

Following R5 Deck 6A (all architectures) + PCML6 notebooks (crown jewel implementations).

| #   | Lesson                               | Theory / Implementation                                                                                                                      | Kailash SDK / Tools                                                    | R5 Source          |
| --- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------ |
| 5.1 | **Autoencoders**                     | 9+ variants: vanilla, undercomplete, denoising (DAE), sparse, contractive (CAE), variational (VAE), CVAE, convolutional, stacked, recurrent. Full implementations with training loops. | PyTorch + `ModelVisualizer`                                            | Deck 6A (9 variants listed) + PCML6-1 (10+ variants, intermediate-advanced) |
| 5.2 | **CNNs and Computer Vision**         | Conv layers (filters, stride, padding), pooling (max, average), normalisation. Architectures: LeNet-5, AlexNet, VGGNet, GoogLeNet/Inception, ResNet (skip connections). SE blocks, Kaiming init, mixed precision, Mixup, label smoothing. | PyTorch + `ModelVisualizer` + `OnnxBridge`                             | Deck 6A (architecture history) + PCML6-2 (44MB, advanced enhancements) |
| 5.3 | **RNNs and Sequence Models**         | LSTM (4 components: cell state, forget/input/output gates), GRU (update/reset gates). Multi-layer with residual connections. Temporal + spatial attention (multi-headed). Metrics: perplexity, BLEU, sequence accuracy, cross-entropy. Financial time series + text generation. | PyTorch + `ModelVisualizer`                                            | Deck 6A (LSTM/GRU theory, metrics) + PCML6-3 (10MB, attention, financial prediction) |
| 5.4 | **Transformers**                     | Self-attention (derive from scratch: why divide by sqrt d_k), positional encoding, encoder/decoder architecture. BERT, GPT, T5, Transformer-XL, Reformer/Longformer. Deck derives architecture; exercise fine-tunes BERT. | PyTorch + HuggingFace + `ModelVisualizer`                              | Deck 6A (full architecture + model variants) + PCML6-4 (BERT fine-tuning) |
| 5.5 | **GANs and Generative Models**       | Generator/Discriminator architecture, adversarial loss. DCGAN, Conditional GAN (cGAN), WGAN (Wasserstein distance), CycleGAN, StyleGAN. Training dynamics, mode collapse, evaluation (FID, IS). Data generation: when to use GANs vs diffusion vs VAE vs LSTM. | PyTorch + `ModelVisualizer`                                            | Deck 6A (6 GAN variants + generation model selection guide) + PCML6-6 (expanded) |
| 5.6 | **Graph Neural Networks**            | GCN (spectral methods), GraphSAGE (sampling + aggregation), GAT (attention on neighbours), GIN (graph structure distinction). Graph classification, node classification. | PyTorch + torch_geometric + `ModelVisualizer`                          | Deck 6A (4 GNN architectures) + PCML6-5 |
| 5.7 | **Transfer Learning**                | CV: ResNet fine-tuning (mask detection, MNIST). NLP: BERT fine-tuning (text classification). Adapter technique as concept (bottleneck modules). ONNX export for portable deployment. | PyTorch + `OnnxBridge` + `InferenceServer`                             | Deck 6A + PCML6-8 (CV) + PCML6-9 (NLP) |
| 5.8 | **Reinforcement Learning**           | Bellman equations (expectation + optimality). 5 algorithms, 5 business applications: DQN (customer churn), DDPG (manufacturing), SAC (dynamic pricing), A2C (resource allocation), PPO (supply chain). Custom Gymnasium environments. | PyTorch + `RLTrainer` + Gymnasium                                      | PCML6-13 (5 algorithms, advanced) |

**Scaffolding**: ~30%
**Kailash engines**: ModelVisualizer, OnnxBridge, InferenceServer, RLTrainer
**R5 fidelity**: Deck 6A provides the theory for ALL architectures. PCML6 notebooks provide production-grade implementations. GAN coverage already includes DCGAN, cGAN, WGAN, CycleGAN, StyleGAN in the deck — expand exercise beyond basic to match.
**RL deck needed**: R5 has RL in notebook (PCML6-13) but NOT in any deck. New deck slides needed.

---

## Module 6: Machine Learning with Language Models and Agentic Workflows

_Build LLM applications, fine-tune models, deploy governed agents. All engineering, all code._

Following R5 Deck 6B (10 fine-tuning techniques, agentic design) — adapted from CrewAI to Kaizen, with Kailash production engineering added.

| #   | Lesson                               | Implementation                                                                                                                       | Kailash SDK                                                                                | R5 Source          |
| --- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ------------------ |
| 6.1 | **LLM Fundamentals and Structured Output** | Tokenisation, scaling laws, transformer architecture recap, inference APIs. Signature/InputField/OutputField, structured output, streaming, cost tracking. | `Kaizen`: Signature, Delegate (streaming, events, cost tracking)                          | Deck 6B (LLM foundation models) + PCML6-12 (adapted CrewAI → Kaizen) |
| 6.2 | **LLM Fine-tuning: Techniques Survey** | ALL 10 techniques from Deck 6B: LoRA (low-rank matrices A x B, from-scratch implementation), Adapter Layers (bottleneck modules, from-scratch), Prefix Tuning (task-specific vectors on K/V), Prompt Tuning (learnable prompt tokens), Task-specific (backprop + LR schedulers + gradient clipping + mixed precision), LLRD (layer-wise decay), Progressive Layer Freezing, Knowledge Distillation (teacher-student + soft labels), Differential Privacy (DPSGD), Elastic Weight Consolidation (Fisher Information Matrix). LoRA vs Adapter comparison across 4 dimensions. | `kailash-align`: AlignmentPipeline, AlignmentConfig, AdapterRegistry                      | Deck 6B (slides 3-8, all 10 techniques) + PCML6-10 (adapter from scratch) + PCML6-11 (LoRA from scratch) |
| 6.3 | **Preference Alignment: DPO**        | DPO derivation from RLHF (connects M5.8 RL). Bradley-Terry model. Implement the alignment training loop. LLM-as-judge evaluation (biases: position, verbosity, self-enhancement). | `kailash-align`: AlignmentPipeline (method="dpo"), evaluator                               | ASCENT (new, not in R5) |
| 6.4 | **RAG Systems**                      | Chunking strategies, dense/sparse/hybrid retrieval, RAGAS evaluation, HyDE. Build a working RAG pipeline end-to-end.                  | `RAGResearchAgent`, `MemoryAgent`, Kaizen tools                                           | ASCENT (new, not in R5) |
| 6.5 | **AI Agents: ReAct and Tool Use**    | ReAct reasoning + action loops, tool selection, autonomous data exploration, cost budget safety. Custom tools wrapping Kailash engines. Mental framework for agent creation (from Deck 6B). | `ReActAgent`, `ChainOfThoughtAgent`, custom tools                                          | Deck 6B (agent design considerations, mental framework, task definition) + PCML6-12 |
| 6.6 | **Multi-Agent Orchestration and MCP** | A2A protocol, supervisor-worker, sequential, parallel, handoff patterns. MCP protocol, tool registration, build an MCP server. Deck 6B agent design: modularity, context/memory, dynamic agents, load balancing, security. | `SupervisorWorkerPattern`, `SequentialPattern`, `kailash.mcp_server`                      | Deck 6B (architectural considerations, monitoring, security) + ASCENT |
| 6.7 | **AI Governance Engineering**        | PACT GovernanceEngine: implement access controls with D/T/R addressing, operating envelopes (define + enforce), audit trails, clearance levels, budget cascading. Code it. | `GovernanceEngine`, `PactGovernedAgent`, `Address`, `can_access()`, `explain_access()`    | ASCENT (new, not in R5) |
| 6.8 | **Capstone: Full Production Platform** | Deploy with Nexus (API+CLI+MCP simultaneously). Auth (RBAC/JWT). Middleware. Monitoring + drift integration. Ship a governed, deployed ML system using the full Kailash stack. | `Nexus`, `nexus.auth`, middleware, `nexus.plugins` + DriftMonitor                          | ASCENT (new, not in R5) |

**Scaffolding**: ~20% for 6.1-6.7, ~40% for 6.8 (capstone tests integration, not from-scratch)
**Kailash engines**: Kaizen (Delegate, BaseAgent, Signature, agents), kailash-align (AlignmentPipeline, AdapterRegistry), kailash-pact (GovernanceEngine, PactGovernedAgent), kailash-nexus (Nexus, auth, middleware), kailash-mcp
**R5 fidelity**: Deck 6B's 10 fine-tuning techniques ALL included in 6.2 (the deck taught them — the exercise implements LoRA + Adapters from scratch). Agent design from Deck 6B preserved but adapted from CrewAI/LangChain → Kaizen. Agent memory concepts (short-term, long-term, entity) from Deck 6B slide 12.
**Not in R5 (new for MLFP)**: DPO/preference alignment, RAG systems, MCP protocol, PACT governance, Nexus deployment.

---

## Summary: R5 Coverage vs MLFP Additions

| MLFP Lesson | In R5 Decks | In R5 Notebooks | New for MLFP |
|---|---|---|---|
| M1 (8 lessons) | Decks 1A-1C cover Python, viz, APIs | PCML1-1 to 1-5 | Polars (replaces pandas), Kailash engines |
| M2.1-2.6 | Decks 2A-2C + 3A cover all | PCML2-1/2, PCML3-1 to 3-6 | Kailash ExperimentTracker |
| M2.7-2.8 | Not in R5 | Not in R5 | CUPED, causal inference (ASCENT) |
| M3.1 | Deck 4A (feature engineering) | PCML4-1 to 4-3 | Kailash FeatureEngineer/Store |
| M3.2-3.5 | Not in R5 decks | Not in R5 | Bias-variance, SHAP/LIME, fairness (ASCENT) |
| M3.6-3.8 | Deck 4B (MLOps concepts) | Not in R5 | Kailash WorkflowBuilder, DataFlow, DriftMonitor (ASCENT) |
| M4.1, 4.3, 4.6 | Deck 5A (clustering, PCA, NLP) | PCML5-1/2 | Kailash AutoMLEngine |
| M4.2, 4.4, 4.5 | Not in R5 | Not in R5 | EM/GMMs, anomaly detection, market basket (new) |
| M4.7 | Not in deck | PCML5-3 | Full recommender lesson (new) |
| M4.8 | Deck 5B (42 slides!) | PCML5-4 | Kailash OnnxBridge |
| M5.1-5.7 | Deck 6A (all architectures) | PCML6-1 to 6-9 | Kailash ModelVisualizer, OnnxBridge |
| M5.8 | Not in deck | PCML6-13 (5 algorithms) | RL deck needed, Kailash RLTrainer |
| M6.1 | Deck 6B (LLM models) | PCML6-12 | CrewAI → Kaizen |
| M6.2 | Deck 6B (10 techniques) | PCML6-10/11 | Kailash AlignmentPipeline |
| M6.3-6.4 | Not in R5 | Not in R5 | DPO, RAG (ASCENT, new) |
| M6.5-6.6 | Deck 6B (agent design) | PCML6-12 | Kaizen agents, MCP (ASCENT) |
| M6.7-6.8 | Not in R5 | Not in R5 | PACT, Nexus (ASCENT, new) |

## What This Spec Replaces

This document (mlfp-curriculum-v1.md) is the **authoritative curriculum spec** for the MLFP course. It supersedes:
- expanded-curriculum-v4.md (wrong M4-M6 structure, ASCENT-oriented)
- expanded-curriculum-v3.md (historical)
- expanded-curriculum-v2.md (historical)

The v2-v4 documents are retained as historical references but are NOT authoritative.
