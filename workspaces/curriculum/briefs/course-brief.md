# ASCENT Course Brief — Professional ML Engineering Curriculum

## Overview

**Full Title**: Practical Course in Machine Learning  
**Institution**: Terrene Open Academy  
**Audience**: Working professionals targeting senior data scientist / ML engineer roles  
**Platform**: Kailash Python SDK (Terrene Foundation)  
**Modules**: 6 sessions, each 7 hours (lecture + lab + assessment)  
**Delivery**: Local Python, Jupyter, Google Colab  
**Standard**: Georgia Tech OMSCS / Stanford CS229 depth, production-practice reality

## Design Philosophy

This is not an intro course. Students are working professionals who need to operate at **senior ML engineer level** — understanding theory deeply enough to debug production failures, make principled architecture decisions, and lead ML teams.

**Academic rigor**: Every technique taught with mathematical foundations, not just API calls. Bias-variance decomposition, kernel theory, attention mechanism derivation, EM algorithm, Bellman equations — students understand *why*, not just *how*.

**Production reality**: Every concept practiced on messy, large-scale, real-world data. Missing values, temporal leakage, class imbalance, schema drift, multi-table joins, noisy labels. If the dataset is clean, it's not in this course.

**Kailash as production platform**: Students learn theory with math, practice with Kailash engines. The SDK is the bridge between textbook and production — DataExplorer replaces manual EDA, TrainingPipeline replaces sklearn boilerplate, GovernanceEngine replaces "trust me" deployments.

---

## Module 1: Foundations — Statistics, Probability & Data Fluency

**Duration**: 7 hours (3h lecture + 3h lab + 1h assessment)  
**Kailash**: kailash-ml (DataExplorer, PreprocessingPipeline, ModelVisualizer)  
**Scaffolding**: 70%

### Lecture Topics

#### 1A: Statistical Foundations (90 min)
- Probability theory: distributions (exponential family), moment-generating functions, convergence types (in probability, in distribution, almost sure)
- Bayesian thinking: prior specification, conjugate priors, posterior computation, credible intervals vs confidence intervals
- Maximum likelihood estimation: derivation, properties (consistency, asymptotic normality, efficiency), Fisher information
- Hypothesis testing: Neyman-Pearson framework, power analysis, multiple testing correction (Bonferroni, BH-FDR), effect sizes
- Bootstrapping: theory (Efron), parametric vs non-parametric, bootstrap confidence intervals (percentile, BCa)

#### 1B: Data Fluency with Polars (45 min)
- Polars architecture: Apache Arrow backend, multi-threaded execution, lazy evaluation
- Expression API: `pl.col()`, `pl.when()`, `over()` for window functions
- Joins, pivots, melts, rolling aggregations
- Lazy frames: `scan_csv()`, query optimization, collect strategies

#### 1C: Exploratory Data Analysis at Scale (45 min)
- DataExplorer: async profiling, statistical summaries, correlation matrices (Pearson, Spearman, Cramer's V)
- Alert system: 8 alert types (high cardinality, skewness, missing patterns, constant columns, duplicates, outliers, type inference, imbalanced target)
- PreprocessingPipeline: auto-detect task type, encoding strategies, scaling, imputation
- ModelVisualizer: plotly-based interactive charts, distribution analysis

### Lab Exercises (5)
1. **Polars deep dive**: Load Singapore HDB resale data (15M+ records), perform joins with MRT proximity and school data, window functions for rolling prices by district
2. **Bayesian estimation**: Compute posterior distributions for Singapore property price parameters using bootstrap + conjugate priors
3. **DataExplorer profiling**: Async profile of dirty Singapore economic data (merged CPI, employment, exchange rates — missing values, mixed granularity)
4. **Hypothesis testing at scale**: A/B test analysis on real e-commerce conversion data with multiple testing correction
5. **Challenge**: Full EDA pipeline on messy Singapore taxi trip data (schema changes across years, GPS noise, missing fields) — profile → clean → visualize → report

### Datasets
- **Singapore HDB Resale Prices** (data.gov.sg): 15M+ records, 2000-present, merge with ascent_assessment MRT/school parquets
- **Singapore Economic Indicators**: CPI, employment, exchange rates from data.gov.sg + World Bank (different reporting frequencies, missing quarters, currency conversions)
- **Singapore Taxi/Ridehail Trips**: LTA data (messy GPS, schema drift between years)
- **E-commerce A/B Test Data**: Real conversion experiment data with sample ratio mismatch

---

## Module 2: Feature Engineering & Experiment Design

**Duration**: 7 hours  
**Kailash**: kailash-ml (FeatureStore, FeatureEngineer, ExperimentTracker)  
**Scaffolding**: 60%

### Lecture Topics

#### 2A: Feature Engineering Theory (90 min)
- Feature selection: mutual information, Boruta algorithm, recursive feature elimination, stability selection
- Collinearity: VIF, condition number, eigenvalue analysis
- Feature interactions: polynomial features, interaction detection via tree-based importance
- Temporal features: lag features, rolling statistics, Fourier features for seasonality, point-in-time correctness (preventing leakage)
- Target encoding: James-Stein shrinkage, hierarchical target encoding, cross-validation encoding to prevent overfitting
- Domain-specific engineering: RFM (retail), technical indicators (finance), clinical feature extraction (healthcare)

#### 2B: Experiment Design & Causal Inference (90 min)
- A/B testing methodology: power analysis, minimum detectable effect, multi-armed bandits (Thompson sampling, UCB), sequential testing (always-valid p-values)
- Variance reduction: CUPED/CUPAC, stratification, pre-experiment covariates
- Causal inference: potential outcomes framework (Rubin), DAGs (Pearl), do-calculus
- Quasi-experimental methods: difference-in-differences, regression discontinuity, instrumental variables, propensity score matching
- Double machine learning (DoubleML): using ML for nuisance parameter estimation in causal models

#### 2C: Feature Management at Scale (30 min)
- FeatureSchema contracts: typed fields, entity IDs, timestamps
- FeatureStore: persist, version, retrieve with point-in-time correctness
- ExperimentTracker: log experiments, compare approaches, reproducibility

### Lab Exercises (5)
1. **Feature engineering on healthcare data**: Engineer clinical features from messy MIMIC-style ICU data (irregular time series, missing vitals, medication records → temporal features with point-in-time correctness)
2. **FeatureStore lifecycle**: Define FeatureSchema, compute features, version, retrieve at different points in time — demonstrate leakage prevention
3. **A/B test analysis**: Full experiment analysis on real e-commerce data — power analysis, SRM check, CUPED variance reduction, multiple metric correction
4. **Causal inference**: Estimate treatment effect of a policy change using diff-in-diff on Singapore housing data (before/after cooling measures)
5. **FeatureEngineer + ExperimentTracker**: Automated feature generation with multiple strategies, tracked as experiments with comparison

### Datasets
- **Healthcare ICU data** (MIMIC-III inspired synthetic): 60K+ ICU stays, irregular vitals, medications, labs, demographics. Multi-table requiring joins. Missing values are clinical reality.
- **E-commerce experiment data**: Real A/B test with 500K users, conversion + revenue metrics, SRM issues in some test groups
- **Singapore housing + policy data**: HDB prices + cooling measure dates for causal analysis

---

## Module 3: Supervised ML — Theory to Production

**Duration**: 7 hours  
**Kailash**: Core SDK (WorkflowBuilder, LocalRuntime), DataFlow, kailash-ml (TrainingPipeline, HyperparameterSearch, ModelRegistry)  
**Scaffolding**: 50%

### Lecture Topics

#### 3A: Supervised ML Theory (90 min)
- Bias-variance decomposition: formal derivation, bias-variance-noise decomposition, implications for model selection
- Regularization: L1/L2 geometry (why L1 produces sparsity), elastic net path algorithm, regularization as Bayesian prior
- Gradient boosting internals: second-order Taylor expansion (XGBoost), histogram-based split finding (LightGBM), symmetric trees (CatBoost), DART dropout regularization
- Ensemble theory: bagging (variance reduction proof), boosting (bias reduction), stacking (meta-learner theory), blending
- Class imbalance: SMOTE and its failure modes, cost-sensitive learning (class weights, threshold optimization), Focal Loss, calibration after resampling

#### 3B: Model Evaluation & Interpretability (90 min)
- Evaluation metrics: beyond accuracy — precision-recall trade-off (F-beta), AUC-ROC vs AUC-PR (why AUC-ROC misleads on imbalanced data), log loss, Brier score
- Calibration: Platt scaling, isotonic regression, calibration curves, reliability diagrams, ECE (expected calibration error)
- Interpretability: SHAP values (Shapley theory, TreeSHAP algorithm, KernelSHAP), LIME, partial dependence plots, accumulated local effects, ICE plots
- Counterfactual explanations: DiCE, what-if analysis
- Model cards: documentation standard (Mitchell et al.), when and how to create them

#### 3C: Workflow Orchestration (30 min)
- WorkflowBuilder: nodes, connections, parameter injection
- LocalRuntime / AsyncLocalRuntime: execution, result retrieval
- DataFlow: @db.model, db.express for CRUD, query builder
- Persisting ML artifacts: model metadata, evaluation results, SHAP values in database

### Lab Exercises (6)
1. **Gradient boosting deep dive**: Train XGBoost, LightGBM, CatBoost on Singapore credit data — compare histogram-based split finding, learning curves, hyperparameter sensitivity
2. **Class imbalance workshop**: Same credit dataset — SMOTE vs cost-sensitive vs Focal Loss vs threshold optimization. Measure calibration before/after each approach.
3. **SHAP interpretability**: Full SHAP analysis on the best model — summary plots, dependence plots, interaction values. Identify which features drive approval/rejection.
4. **Workflow orchestration**: Build a multi-step Kailash workflow: load → preprocess → train → evaluate → persist results to DataFlow
5. **HyperparameterSearch + ModelRegistry**: Bayesian optimization → register best model → promote staging → production lifecycle
6. **End-to-end pipeline**: Complete supervised ML pipeline with workflow, persisted results, and model card generation

### Datasets
- **Singapore Credit Scoring** (synthetic): 100K applications, 12% default rate, 45 features including protected attributes (age, gender, ethnicity), deliberately constructed temporal leakage trap, missing income for 30% of self-employed applicants
- **Lending Club Loans**: 300K+ records, 150 features, real-world messiness. Cross-reference with credit dataset for transfer learning exercise.

---

## Module 4: Unsupervised ML, NLP & Deep Learning

**Duration**: 7 hours  
**Kailash**: kailash-ml (AutoMLEngine, EnsembleEngine, DriftMonitor, InferenceServer), Nexus  
**Scaffolding**: 40%

### Lecture Topics

#### 4A: Unsupervised ML Beyond K-means (90 min)
- Clustering theory: K-means limitations (spherical assumption, fixed K), initialization (k-means++, kernel trick)
- Spectral clustering: graph Laplacian construction, eigengap heuristic, normalized cuts
- HDBSCAN: density-based hierarchy, automatic cluster count, noise point identification
- Gaussian mixture models: EM algorithm derivation (E-step, M-step, convergence), relationship to K-means (hard vs soft assignment)
- Dimensionality reduction: PCA (eigendecomposition vs SVD, scree plot, explained variance), t-SNE (perplexity tuning, crowding problem, non-convexity), UMAP (topological data analysis foundations, hyperparameter sensitivity)
- Anomaly detection: Isolation Forest (random partitioning theory), Local Outlier Factor, autoencoders for anomaly scoring

#### 4B: NLP Foundations & Topic Modeling (60 min)
- Text representation: TF-IDF theory, BM25, word embeddings (Word2Vec CBOW/Skip-gram, GloVe co-occurrence matrix)
- Topic modeling: LDA (plate notation, Gibbs sampling), NMF (relationship to K-means), BERTopic (UMAP + HDBSCAN + c-TF-IDF pipeline)
- Sentiment analysis: lexicon-based vs ML-based, aspect-level sentiment
- Text preprocessing: tokenization, stopwords, lemmatization, n-grams for domain-specific vocabularies

#### 4C: Deep Learning Foundations (60 min)
- Neural network theory: universal approximation theorem, gradient flow, vanishing/exploding gradients
- Architecture building blocks: residual connections (skip connection theory), normalization (BatchNorm vs LayerNorm vs RMSNorm), attention mechanism (scaled dot-product derivation, multi-head attention)
- Training dynamics: learning rate schedules (cosine annealing, warm restarts, OneCycleLR), optimizer internals (Adam, AdamW weight decay decoupling), gradient clipping
- Convolutional networks: convolution as feature extraction, receptive field theory, modern architectures (ResNet, EfficientNet)
- Recurrent networks: LSTM gates (forget, input, output), GRU simplification, bidirectional encoding

### Lab Exercises (6)
1. **Clustering comparison**: K-means vs spectral vs HDBSCAN vs GMM on Singapore e-commerce customer data — evaluate with silhouette, Calinski-Harabasz, Davies-Bouldin. Business interpretation of segments.
2. **UMAP + anomaly detection**: Dimensionality reduction on fraud transaction data, Isolation Forest anomaly scoring, visual inspection of anomaly clusters
3. **Topic modeling pipeline**: BERTopic on Singapore news corpus — extract topics, track temporal evolution, visualize topic distributions
4. **DriftMonitor**: Deploy a model from Module 3, simulate production data drift, detect with PSI + KS test, trigger retrain alert
5. **Deep learning training**: Train CNN on image classification task with learning rate scheduling, gradient monitoring, mixed precision
6. **InferenceServer + Nexus**: Deploy trained model as API + CLI + MCP using Nexus — load test, monitor latency, serve predictions

### Datasets
- **Singapore E-commerce Transactions**: 200K transactions, 50K customers, product catalog, text reviews (for NLP), behavioral features (for clustering). Synthetic but realistic.
- **Credit Card Fraud**: Kaggle 284K transactions (0.17% fraud rate). Real PCA-transformed features.
- **Singapore News Corpus**: 50K articles (CC-licensed), multi-topic, temporal span for topic evolution
- **ChestX-ray14** (subset): 10K images for CNN exercise (multi-label, class imbalance)

---

## Module 5: LLMs, AI Agents & RAG Systems

**Duration**: 7 hours  
**Kailash**: Kaizen (Delegate, BaseAgent, Signature, specialized agents), kailash-ml (6 ML agents)  
**Scaffolding**: 30%

### Lecture Topics

#### 5A: Transformer Architecture & LLMs (90 min)
- Transformer deep dive: encoder-decoder architecture, self-attention vs cross-attention, positional encodings (sinusoidal, RoPE, ALiBi)
- Tokenization internals: BPE (byte pair encoding), WordPiece, Unigram (SentencePiece), vocabulary size trade-offs, multilingual tokenization
- Pre-training objectives: masked language modeling (BERT), causal language modeling (GPT), span corruption (T5)
- Scaling laws: Chinchilla scaling, compute-optimal training, emergent abilities
- Inference optimization: KV-cache, speculative decoding, quantization (GPTQ, AWQ, GGUF), batched inference

#### 5B: RAG Architecture & Evaluation (60 min)
- RAG pipeline: chunking strategies (fixed, semantic, recursive), embedding models (Sentence-BERT, instructor embeddings), vector stores
- Retrieval: dense retrieval, sparse retrieval (BM25), hybrid retrieval, re-ranking (cross-encoder)
- Evaluation: faithfulness (is the answer grounded?), relevance (is the retrieval useful?), answer correctness, citation accuracy
- Advanced RAG: multi-hop retrieval, graph RAG, agentic RAG (iterative retrieval)

#### 5C: Agent Architecture & Multi-Agent Systems (60 min)
- Agent paradigm: perception-reasoning-action loop, tool use, memory (short-term buffer, long-term retrieval)
- Signature-based programming: InputField/OutputField type contracts, structured LLM output
- Specialized agents: CoT (step-by-step reasoning), ReAct (reasoning + action interleave), RAG (retrieval-augmented)
- Multi-agent coordination: A2A protocol, supervisor-worker pattern, debate pattern, consensus mechanisms
- ML agents: how LLMs augment the ML lifecycle (feature suggestion, model selection, experiment interpretation, drift analysis)
- Agent safety: prompt injection protection, output validation, cost budgets, human-in-the-loop escalation

### Lab Exercises (6)
1. **Delegate + SimpleQAAgent**: High-level agent for data analysis questions over the e-commerce dataset. Signature contract for structured output.
2. **ChainOfThoughtAgent**: Structured reasoning about clustering results from Module 4 — agent explains WHY segments formed, not just WHAT they are
3. **ReActAgent with tools**: Autonomous data exploration — agent can call DataExplorer, FeatureEngineer, ModelVisualizer as tools. Observe reasoning-action trace.
4. **RAGResearchAgent**: Build RAG over Kailash SDK documentation + Singapore regulatory documents. Evaluate retrieval quality with faithfulness and relevance metrics.
5. **ML Agent Pipeline**: DataScientistAgent → FeatureEngineerAgent → ModelSelectorAgent chain. Compare LLM-augmented feature/model choices to manual Module 3 choices.
6. **Multi-agent A2A**: Full orchestration — research agent gathers context, analyst agent interprets data, engineer agent builds pipeline, reviewer agent validates. End-to-end autonomous ML.

### Datasets
- **Same e-commerce + credit datasets** from Modules 3-4 (agents reason over familiar data)
- **Kailash SDK documentation corpus**: For RAG exercise
- **Singapore regulatory corpus**: AI Verify framework, PDPA guidelines (for governance-aware RAG)

---

## Module 6: Alignment, Governance, RL & Production Deployment

**Duration**: 7 hours  
**Kailash**: Align (AlignmentPipeline), PACT (GovernanceEngine, PactGovernedAgent), kailash-ml (RLTrainer), Nexus  
**Scaffolding**: 20%

### Lecture Topics

#### 6A: LLM Fine-Tuning & Alignment (90 min)
- Fine-tuning landscape: full fine-tuning, LoRA (low-rank adaptation theory, rank selection, target modules), QLoRA (4-bit quantization + LoRA), DoRA, prefix tuning, adapter layers
- Alignment methods: RLHF (reward model → PPO training loop), DPO (direct preference optimization — bypass reward model), GRPO, constitutional AI
- Data for alignment: instruction datasets (quality > quantity), preference pair construction, synthetic data generation for alignment
- Evaluation: perplexity, BLEU, ROUGE, BERTScore, human evaluation protocols, LLM-as-judge methodology, contamination detection
- Practical trade-offs: cost, quality, catastrophic forgetting, quantization impact on fine-tuned models

#### 6B: AI Governance & Responsible Deployment (60 min)
- Regulatory landscape: EU AI Act (risk tiers, high-risk obligations, GPAI rules effective Aug 2025), Singapore AI Verify (ISAGO 2.0, self-assessment), MAS AI guidelines for finance
- PACT framework: D/T/R accountability grammar, operating envelopes, knowledge clearance (5 levels), verification gradient
- Bias & fairness: demographic parity, equalized odds, calibration across groups, pre/in/post-processing mitigation
- Algorithmic auditing: model cards, datasheets for datasets, red teaming, disparate impact testing
- Governance in practice: human-in-the-loop design, appeal mechanisms, incident response, governance as competitive advantage

#### 6C: Reinforcement Learning & Advanced Topics (60 min)
- RL foundations: MDPs, Bellman equations, value iteration, policy iteration, temporal difference learning
- Deep RL: DQN (experience replay, target networks), policy gradient (REINFORCE), actor-critic (A2C, PPO), SAC
- Practical RL: dynamic pricing, recommendation systems (exploration-exploitation), inventory optimization, process control
- Emerging: multi-modal AI (vision-language models, CLIP), federated learning, differential privacy (DP-SGD, privacy budgets), synthetic data generation
- Reward hacking and safety: reward shaping, constrained RL, sim-to-real transfer challenges

### Lab Exercises (6)
1. **SFT fine-tuning**: AlignmentPipeline SFT on a small model (e.g., Llama-3.1-8B) with Singapore domain Q&A. Track adapter in AdapterRegistry.
2. **DPO alignment**: Compare DPO vs SFT on preference data — evaluate quality with LLM-as-judge and human evaluation rubric
3. **Governance setup**: Define a realistic organization in YAML (3 departments, 8 roles, 15 agents). Compile, verify access, explain decisions.
4. **Governed agents**: Wrap Module 5's ReActAgent with PactGovernedAgent — enforce cost budgets ($5 max), tool restrictions (no web scraping), data access policies. Test monotonic tightening.
5. **RL for optimization**: RLTrainer with PPO on an inventory management environment. Compare RL policy vs heuristic baseline. Track with ExperimentTracker.
6. **Capstone deployment**: Full governed ML platform — InferenceServer (Module 3 model) + Kaizen agent (Module 5) + PACT governance + Nexus multi-channel. The complete Kailash stack in production configuration.

### Datasets
- **Domain Q&A pairs**: Synthetic instruction dataset for SFT (1000 pairs, Kailash SDK domain)
- **Preference pairs**: For DPO exercise (500 pairs, good/bad response comparisons)
- **Singapore parliamentary Hansard**: Public record for RAG + governance text analysis
- **Gymnasium environments**: CartPole, LunarLander, custom inventory management

---

## Assessment Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| Module Quizzes (6) | 20% | 15 questions each: theory + code + interpretation |
| Individual Portfolio | 35% | Extend one module to full production depth with model card |
| Team Capstone | 35% | Multi-framework production system using 3+ Kailash packages |
| Peer Review | 10% | Code review of another team's capstone (SHAP analysis, governance audit) |

### Individual Portfolio Requirements
- Select any module's dataset and extend to production-ready system
- Must include: EDA report (DataExplorer), feature engineering rationale, model comparison (≥3 approaches), SHAP interpretability, calibration analysis, drift monitoring setup, model card
- Graded on: statistical rigor (25%), Kailash pattern mastery (25%), production readiness (25%), documentation quality (25%)

### Team Capstone Requirements
- Teams of 3-4, domain of choice, must use ≥3 Kailash packages
- Must include: data pipeline (DataFlow), model lifecycle (ModelRegistry), deployment (Nexus), governance (PACT) OR agents (Kaizen)
- 15-minute live demo + 10-minute Q&A
- Deliverables: working system, architecture doc, model cards for all models, governance specification

---

## Presentation Decks

### Format & Tooling
- **Framework**: Reveal.js 5.1.0 (HTML slides with `?print-pdf` for export)
- **Canvas**: 1280×720 pixels
- **Theme**: Teal primary, indigo depth accents, slate/white palette, amber alerts, rose warnings
- **CSS**: `decks/assets/css/theme.css` (shared across all module decks)

### Deck Design Principles (from co-ax)
- **One idea per slide** — split if >3 bullet points
- **Questions over statements** — provoke thinking, not delivery
- **Cases over theory** — illustrate with real data and real failures
- **Active language** — "What would you do?" not "Here is the answer"
- **Math when it matters** — show derivations for core concepts, skip for API usage
- **Code live, not on slides** — slides set up the problem, lab solves it

### Deck Structure per Module
1. **Title slide**: Module theme + provocative question
2. **Opening case**: Real-world failure or success that motivates the module (e.g., Zillow's iBuyer $500M write-down for Module 3 calibration)
3. **Theory slides**: Math + intuition, building from prior modules
4. **Singapore context slides**: Local data, local regulations, local market
5. **Kailash engine slides**: "Here's the theory. Here's how the engine implements it." — side-by-side
6. **Lab setup slides**: Problem statement, dataset description, expected outputs
7. **Discussion prompts**: "Given this SHAP output, would you approve this loan? Why?"
8. **Synthesis slide**: Key takeaways + connection to next module

### Deck Workspace
Each module deck is a workspace task:
```
decks/
├── assets/css/theme.css     # Shared Reveal.js theme
├── assets/img/              # Shared images
├── ascent01/deck.html          # Module 1 deck
├── ascent02/deck.html          # Module 2 deck
├── ...
└── ascent06/deck.html          # Module 6 deck
```

---

## Real-World Dataset Matrix

| Module | Primary Dataset | Records | Why It's Hard |
|--------|----------------|---------|---------------|
| 1 | Singapore HDB Resale + MRT + Schools | 15M+ | Multi-table joins, 25-year span, geographic |
| 1 | Singapore Taxi Trips (LTA) | 500K+ | GPS noise, schema drift, missing fields |
| 2 | Healthcare ICU Data (MIMIC-style) | 60K stays | Irregular time series, multi-table, missing vitals |
| 2 | E-commerce A/B Test Data | 500K users | SRM issues, multiple metrics, sequential testing |
| 3 | Singapore Credit Scoring (synthetic) | 100K apps | 12% default, protected attributes, leakage trap |
| 3 | Lending Club Loans | 300K+ | 150 features, real messiness, temporal patterns |
| 4 | Credit Card Fraud (Kaggle) | 284K txns | 0.17% fraud rate, PCA features, extreme imbalance |
| 4 | Singapore News Corpus | 50K articles | Multi-topic, temporal, multilingual fragments |
| 4 | ChestX-ray14 (subset) | 10K images | Multi-label, class imbalance, radiologist disagreement |
| 5 | Same datasets + SDK docs | — | Agents reason over familiar data |
| 6 | Parliamentary Hansard + Gym envs | — | Governance text, RL environments |

---

## Curriculum-to-SDK Mapping

Every lecture concept maps to a Kailash engine. Students learn theory in lecture, practice with engines in lab.

| Concept | Theory (Lecture) | Practice (Lab) |
|---------|-----------------|----------------|
| EDA | Descriptive stats, distributions | DataExplorer, ModelVisualizer |
| Feature engineering | Mutual information, Boruta | FeatureEngineer, FeatureStore |
| Experiment tracking | Reproducibility, comparison | ExperimentTracker |
| Model training | Bias-variance, regularization | TrainingPipeline, ModelSpec |
| Hyperparameter search | Bayesian optimization | HyperparameterSearch, AutoMLEngine |
| Ensemble methods | Bagging/boosting theory | EnsembleEngine |
| Model lifecycle | Versioning, promotion | ModelRegistry |
| Drift monitoring | PSI, KS test theory | DriftMonitor |
| Model serving | Latency, batching, caching | InferenceServer |
| Workflow orchestration | DAG theory, node composition | WorkflowBuilder, LocalRuntime |
| Data persistence | Schema design, CRUD | DataFlow (@db.model, db.express) |
| Multi-channel deployment | API/CLI/MCP architecture | Nexus |
| AI agents | CoT, ReAct, tool use | Kaizen (Delegate, BaseAgent, Signature) |
| ML augmentation | LLM for feature/model selection | 6 kailash-ml agents |
| Fine-tuning | LoRA theory, DPO derivation | AlignmentPipeline, AdapterRegistry |
| Governance | D/T/R grammar, envelopes | GovernanceEngine, PactGovernedAgent |
| RL | Bellman equations, PPO | RLTrainer |

---

## Success Criteria

A graduate of this course can:
1. Derive the bias-variance decomposition and explain why their model choice is principled
2. Engineer features with point-in-time correctness and detect temporal leakage
3. Build and interpret SHAP explanations for any tree-based or linear model
4. Design and analyze A/B tests with proper power analysis and variance reduction
5. Train, version, deploy, and monitor ML models through a governed lifecycle
6. Build RAG systems with proper evaluation (faithfulness, relevance)
7. Orchestrate multi-agent ML pipelines with typed signatures and cost budgets
8. Fine-tune LLMs with LoRA/DPO and evaluate alignment quality
9. Enforce organizational governance on AI systems using PACT D/T/R grammar
10. Deploy full-stack governed ML platforms using the complete Kailash SDK
