# R5 Source Mapping

Mapping from R5 delivery materials (PCML_DIS_R5_2601) to MLFP lessons. R5 consists of 13 decks (275+ slides) and 38 notebooks.

Source: mlfp-curriculum-v1.md (deck inventory), mlfp-curriculum-v2.md (per-lesson R5 Source annotations).

## R5 Deck Inventory

**Reference**: Google Drive PCML_DIS_R5_2601

| Deck | Title | Slides | Topics |
|---|---|---|---|
| 1A | Introduction | 18 | 6-module overview, AI context, terminology, mastery philosophy |
| 1B | Python Fundamentals | — | Data types, operators, variables, strings, flow control, collections |
| 1C | Visualization & Dashboarding | 30 | Viz principles, Gestalt, chart types, Plotly, APIs |
| 2A | Descriptive Statistics | 37 | Variable types, 3Ms, variation, outliers, Z-score, IQR, quantiles |
| 2B | Data Exploration | 19 | EDA, Pandas operations, merging, cleaning, transformation |
| 2C | Probability | large | Truth tables, conditional probability, Bayes' theorem, joint probability |
| 3A | Inferential Statistics | 55 | Parameter estimation, bootstrapping, A/B testing, data collection framework, hypothesis testing, linear regression (OLS, t-stat, R-squared, F-stat, multivariate, categorical), logistic regression (log-odds), ANOVA (one-way, two-way, repeated measures, post-hoc) |
| 4A | Supervised ML | — | ML pipeline, feature engineering (geocoding), statistics vs ML, AutoML |
| 4B | ML Pipeline & MLOps | 11 | Training pipeline, MLOps components, clean architecture |
| 5A | Unsupervised ML | — | USML as automated feature engineering, K-means, hierarchical (dendrograms, linkage), t-SNE, PCA (2-step), NLP text classification/clustering |
| 5B | Basics of Deep Learning | 42 | Neural network architecture, forward pass, gradient descent (step-by-step), linear regression as NN, feature interaction, activation functions, hidden layers as automated feature engineering, representation learning, embeddings, "unsupervised meets supervised learning" |
| 6A | Specialised DL Models | — | Autoencoders (9 variants), CNNs (LeNet-5 through ResNet), RNNs (LSTM gates, GRU, perplexity/BLEU), Transformers (BERT, GPT, T5, Transformer-XL, Reformer), GNNs (GCN, GraphSAGE, GAT, GIN), GANs (DCGAN, cGAN, WGAN, CycleGAN, StyleGAN), data generation models |
| 6B | LLMs & Agentic Workflows | 15 | LLM foundation models, 10 fine-tuning techniques (LoRA, Adapters, Prefix Tuning, Prompt Tuning, Task-specific, LLRD, Progressive Layer Freezing, Knowledge Distillation, Differential Privacy/DPSGD, Elastic Weight Consolidation/Fisher), agentic workflow design, agent frameworks |

## Per-Lesson R5 Source Mapping

### Module 1: Data Pipelines and Visualisation

| Lesson | R5 Source |
|---|---|
| 1.1: Your First Data Exploration | Deck 1B (data types, operators, variables) + PCML1-1 |
| 1.2: Filtering and Transforming Data | Deck 1B (comparison operators) + PCML1-2 |
| 1.3: Functions and Aggregation | Deck 1B (functions, collections) + PCML1-3 |
| 1.4: Joins and Multi-Table Data | Deck 2B (merging: join, merge, concat) + ASCENT M1 ex_1 (joins portion) |
| 1.5: Window Functions and Trends | ASCENT M1 ex_1 (windows portion) |
| 1.6: Data Visualisation | Deck 1C (30 slides on viz principles, chart types, Plotly) |
| 1.7: Automated Data Profiling | ASCENT M1 ex_3 |
| 1.8: Data Pipelines and End-to-End Project | Deck 1C (APIs, REST) + PCML1-5 (ETL dashboard) + ASCENT M1 ex_5 |

### Module 2: Statistical Mastery

| Lesson | R5 Source |
|---|---|
| 2.1: Probability and Bayesian Thinking | Deck 2A (expected value, sampling bias) + Deck 2C (conditional probability, Bayes' theorem, COVID example) |
| 2.2: Parameter Estimation and Inference | Deck 2A (population vs sample, Bessel's correction, degrees of freedom) + Deck 3A slides 5-10 (parameter estimation, optimal parameters) |
| 2.3: Bootstrapping and Hypothesis Testing | Deck 3A slides 11-35 (bootstrapping, hypothesis testing, permutation) |
| 2.4: A/B Testing and Experiment Design | Deck 3A slides 15-26 (A/B testing hypothesis, data collection framework -- very detailed, 12 slides) |
| 2.5: Linear Regression | Deck 3A slides 36-48 (comprehensive: OLS, t-stat, R-squared, F-stat, multivariate, categorical encoding, loglinear) |
| 2.6: Logistic Regression and Classification Foundations | Deck 3A slides 49-54 (logistic regression, ANOVA) |
| 2.7: CUPED and Causal Inference | ASCENT (new, not in R5) |
| 2.8: Capstone -- Statistical Analysis Project | PCML3-6 "Putting It Together" (wine dataset) + ASCENT M2 ex_1/2/5 |

### Module 3: Supervised ML

| Lesson | R5 Source |
|---|---|
| 3.1: Feature Engineering, ML Pipeline, and Feature Selection | Deck 4A (feature engineering, ML pipeline, statistics vs ML) |
| 3.2: Bias-Variance, Regularisation, and Cross-Validation | ASCENT (new derivation, not in R5 decks) |
| 3.3: The Complete Supervised Model Zoo | Deck 4B (lists 18 models in monitoring slide) + ASCENT. Note: SVM, KNN, Naive Bayes are new additions not in R5 or ASCENT -- need new deck content and exercises. |
| 3.4: Gradient Boosting Deep Dive | ASCENT M3. Note: XGBoost 2nd-order Taylor derivation is new in ASCENT, not in R5 decks. |
| 3.5: Model Evaluation, Imbalance, and Calibration | ASCENT M3 ex_2 |
| 3.6: Interpretability and Fairness | ASCENT (new, not in R5) |
| 3.7: Workflow Orchestration, Model Registry, and Hyperparameter Search | Deck 4B (MLOps components) + ASCENT M3 ex_4/5 |
| 3.8: Production Pipeline -- DataFlow, Drift, and Deployment | Deck 4B (MLOps) + ASCENT M3 ex_6 |

### Module 4: Unsupervised ML and Advanced Techniques

| Lesson | R5 Source |
|---|---|
| 4.1: Clustering | Deck 5A (K-means, hierarchical with 4 linkage methods, dendrograms, t-SNE) + PCML5-1 (customer segmentation) |
| 4.2: EM Algorithm and Gaussian Mixture Models | ASCENT (new, not in R5) |
| 4.3: Dimensionality Reduction | Deck 5A (PCA 2-step, t-SNE, intrinsic dimension) |
| 4.4: Anomaly Detection and Ensembles | Deck 2A (Z-score, IQR, winsorisation) + ASCENT |
| 4.5: Association Rules and Market Basket Analysis | New (not in R5). Need new deck content and exercise. |
| 4.6: NLP -- Text to Topics | Deck 5A (NLP text structure, TF-IDF, NMF) + PCML5-2 |
| 4.7: Recommender Systems and Collaborative Filtering | PCML5-3 (recommenders, adapted) |
| 4.8: DL Foundations -- Neural Networks, Backpropagation, and the Training Toolkit | Deck 5B (42 slides, comprehensive) + PCML5-4 (DL basics notebook) |

### Module 5: Deep Learning

| Lesson | R5 Source |
|---|---|
| 5.1: Autoencoders | Deck 6A (9 variants) + PCML6-1 (10+ variants, implementations) |
| 5.2: CNNs and Computer Vision | Deck 6A (architecture history) + PCML6-2 (44MB, advanced enhancements) |
| 5.3: RNNs and Sequence Models | Deck 6A (LSTM/GRU theory, metrics) + PCML6-3 (10MB, attention, financial prediction) |
| 5.4: Transformers | Deck 6A (architecture + model variants) + PCML6-4 (BERT fine-tuning) |
| 5.5: Generative Models -- GANs and Diffusion | Deck 6A (6 GAN variants + generation model guide) + PCML6-6 (expanded) |
| 5.6: Graph Neural Networks | Deck 6A (4 GNN architectures) + PCML6-5 |
| 5.7: Transfer Learning | Deck 6A + PCML6-8 (CV transfer) + PCML6-9 (NLP transfer) |
| 5.8: Reinforcement Learning | PCML6-13 (5 algorithms, advanced implementations). Note: RL needs new DECK content -- R5 has notebook only. |

### Module 6: LLMs and Agentic Workflows

| Lesson | R5 Source |
|---|---|
| 6.1: LLM Fundamentals, Prompt Engineering, and Structured Output | Deck 6B (LLM foundation models) + PCML6-12 (adapted CrewAI to Kaizen) |
| 6.2: LLM Fine-tuning -- LoRA, Adapters, and the Technique Landscape | Deck 6B slides 3-8 (10 techniques) + PCML6-10 (adapter from scratch) + PCML6-11 (LoRA from scratch) |
| 6.3: Preference Alignment -- DPO and GRPO | ASCENT (new, not in R5) |
| 6.4: RAG Systems | ASCENT (new, not in R5) |
| 6.5: AI Agents -- ReAct, Tool Use, and Function Calling | Deck 6B (agent design, mental framework, task definition) + PCML6-12 (adapted CrewAI to Kaizen) |
| 6.6: Multi-Agent Orchestration and MCP | Deck 6B (agent architecture, memory, security) + ASCENT |
| 6.7: AI Governance Engineering | ASCENT (new, not in R5) |
| 6.8: Capstone -- Full Production Platform | ASCENT (new, not in R5) |

## R5 Coverage vs MLFP Additions

Summary of what comes from R5 decks, R5 notebooks, and what is new for MLFP.

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
| M6.1 | Deck 6B (LLM models) | PCML6-12 | CrewAI to Kaizen |
| M6.2 | Deck 6B (10 techniques) | PCML6-10/11 | Kailash AlignmentPipeline |
| M6.3-6.4 | Not in R5 | Not in R5 | DPO, RAG (ASCENT, new) |
| M6.5-6.6 | Deck 6B (agent design) | PCML6-12 | Kaizen agents, MCP (ASCENT) |
| M6.7-6.8 | Not in R5 | Not in R5 | PACT, Nexus (ASCENT, new) |
