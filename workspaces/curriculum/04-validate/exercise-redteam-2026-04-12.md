# Exercise Red Team — MLFP v2 Spec vs Solutions Audit

**Date**: 2026-04-12
**Scope**: All 48 exercise solutions (mlfp01-06, ex_1-8) validated against module-{1..6}.md specs and design-principles.md
**Method**: Full read of all M1/M2 solutions, header+structure review of M3-M6, pattern-grep across all 48 files

---

## Global Compliance (all 48 solutions)

| Check | Status | Notes |
|---|---|---|
| "WHAT YOU'LL LEARN" header | PASS (48/48) | Every solution has the section |
| Checkpoint assertions | PASS (48/48) | Range: 2-7 checkpoints per exercise (240 total matches) |
| REFLECTION / bridge at end | PASS (48/48) | All have "WHAT YOU'VE MASTERED" + NEXT bridge (94 total matches) |
| INTERPRETATION comments | PASS (48/48) | 235 total `INTERPRETATION:` comments across all files |
| No pandas imports | PASS (48/48) | Zero `import pandas` or `from pandas` found |
| No ASCENT references | PASS (48/48) | Only "ascent" found = "coordinate ascent on the ELBO" (mlfp04/ex_2) — correct technical term |
| No stubs/TODOs | PASS (48/48) | Zero TODO, FIXME, HACK, STUB, NotImplementedError, `pass #` |
| Polars-native | PASS (48/48) | All data loading uses `polars` via `MLFPDataLoader` |
| Apache-2.0 license header | PASS (48/48) | All files start with `# Copyright 2026 Terrene Foundation` + `SPDX-License-Identifier: Apache-2.0` |
| Progressive scaffolding | See M-by-M | Assessed per module below |

---

## Module 1: Machine Learning Data Pipelines and Visualisation Mastery with Python

### Title Alignment (Spec Lesson Title vs Solution Title)

| Ex | Spec Title | Solution Title | Match |
|---|---|---|---|
| 1 | Your First Data Exploration | Your First Data Exploration | EXACT |
| 2 | Filtering and Transforming Data | Filtering and Transforming Data | EXACT |
| 3 | Functions and Aggregation | Functions and Aggregation | EXACT |
| 4 | Joins and Multi-Table Data | Joins and Multi-Table Data | EXACT |
| 5 | Window Functions and Trends | Window Functions and Trends | EXACT |
| 6 | Data Visualisation | Data Visualization | CLOSE (US vs UK spelling) |
| 7 | Automated Data Profiling | Automated Data Profiling | EXACT |
| 8 | Data Pipelines and End-to-End Project | Data Cleaning and End-to-End Project | MEDIUM: title drift |

### Topic Coverage

| Ex | Key Spec Topics | Covered | Missing |
|---|---|---|---|
| 1 | Variables, types, print, f-strings, pl.read_csv, shape, describe | YES | None |
| 2 | Booleans, pl.col, filter, select, sort, with_columns, method chaining | YES | None |
| 3 | def functions, for loops, lists, dicts, group_by, agg | YES | None |
| 4 | if/else/elif, import, join (left/inner/outer), dictionary lookups | YES | None |
| 5 | Window functions: over(), rolling_mean(), shift(), lazy frames | YES | None |
| 6 | Viz principles, Gestalt, chart types, ModelVisualizer | YES | None (6 chart types achieved) |
| 7 | DataExplorer, AlertConfig, compare(), try/except, async | YES | None |
| 8 | ETL, REST APIs, PreprocessingPipeline, full pipeline | PARTIAL | REST API data loading absent |

### Key Formula Coverage

All spec formulas for M1 are conceptual (no mathematical formulas specified). Covered through implementation.

### Kailash Engine Usage

| Engine | Spec Requires | Used In |
|---|---|---|
| DataExplorer | Yes | ex_7, ex_8 |
| PreprocessingPipeline | Yes | ex_8 |
| ModelVisualizer | Yes | ex_6, ex_8 |

### Issues

| # | Severity | File | Issue |
|---|---|---|---|
| 1 | MEDIUM | mlfp01/ex_8 | **Title drift**: Spec says "Data Pipelines and End-to-End Project"; solution says "Data Cleaning and End-to-End Project". The solution focuses on cleaning + feature engineering rather than the spec's emphasis on REST API data extraction (OneMap Singapore example). |
| 2 | MEDIUM | mlfp01/ex_8 | **Missing REST API topic**: Spec 1.8 explicitly requires "REST APIs: GET, POST, JSON responses, query parameters (OneMap Singapore example)". The solution loads from parquet, not from an API. |
| 3 | LOW | mlfp01/ex_6 | **Spelling**: "Visualization" (US) vs spec's "Visualisation" (UK). Cosmetic only but the spec consistently uses British spelling. |

### Scaffolding Assessment (Target: ~70%)

M1 solutions provide heavy scaffolding: every line commented, every concept explained, every variable named explicitly. Consistent with the ~70% target.

---

## Module 2: Statistical Mastery for Machine Learning and AI Success

### Title Alignment

| Ex | Spec Title | Solution Title | Match |
|---|---|---|---|
| 1 | Probability and Bayesian Thinking | Probability and Bayesian Thinking | EXACT |
| 2 | Parameter Estimation and Inference | Estimation and Inference | CLOSE |
| 3 | Bootstrapping and Hypothesis Testing | Hypothesis Testing | PARTIAL: "Bootstrapping" dropped from title |
| 4 | A/B Testing and Experiment Design | A/B Testing and Experiment Design | EXACT |
| 5 | Linear Regression | Linear Regression | EXACT |
| 6 | Logistic Regression and Classification Foundations | Logistic Regression and Classification Foundations | EXACT |
| 7 | CUPED and Causal Inference | CUPED and Causal Inference | EXACT |
| 8 | Capstone — Statistical Analysis Project | Capstone — Statistical Analysis Project | EXACT |

### Topic Coverage

| Ex | Key Spec Topics | Covered | Missing/Gaps |
|---|---|---|---|
| 1 | Bayes' theorem, distributions, conjugate priors, expected value | YES | Sampling bias (friendship paradox) not demonstrated |
| 2 | Population vs sample, CLT, MLE, MAP, when MLE fails | YES | PDF/CDF explicit demonstration not present (implicit in scipy.stats usage) |
| 3 | Bootstrap CI, BCa, hypothesis testing, permutation tests, Bonferroni, BH-FDR | YES | None |
| 4 | A/B design, SRM, data collection framework (Why/What/Where/How/Frequency) | YES | None |
| 5 | OLS from scratch, t-statistics, R-squared, F-statistic, categorical encoding | YES | None |
| 6 | Logistic regression, sigmoid, odds ratios, ANOVA, Tukey HSD | YES | None |
| 7 | CUPED, SRM, Difference-in-Differences | PARTIAL | **DiD missing** |
| 8 | FeatureEngineer, FeatureStore, point-in-time, data lineage | PARTIAL | **FeatureEngineer engine missing** |

### Key Formula Coverage

| Formula | Spec Location | Implemented |
|---|---|---|
| Bayes' theorem | 2.1 | YES (Normal-Normal conjugate) |
| Expected value | 2.1 | YES (implicit in MLE) |
| Population/sample variance | 2.2 | YES (ddof=0 vs ddof=1) |
| Log-likelihood | 2.2 | YES (from scratch in ex_2) |
| Bootstrap CI percentile | 2.3 | YES |
| T-statistic | 2.3, 2.5 | YES |
| Bonferroni correction | 2.3 | YES |
| OLS formula | 2.5 | YES (matrix algebra: (X'X)^-1 X'y) |
| R-squared | 2.5 | YES |
| F-statistic | 2.5 | YES |
| Sigmoid | 2.6 | YES |
| Log-odds | 2.6 | YES |
| Odds ratio | 2.6 | YES |
| ANOVA F-statistic | 2.6 | YES |
| CUPED variance reduction | 2.7 | YES |
| CUPED estimator (theta) | 2.7 | YES |
| DiD ATT formula | 2.7 | **MISSING** |

### Kailash Engine Usage

| Engine | Spec Requires | Used In |
|---|---|---|
| ExperimentTracker | Yes | ex_6, ex_7, ex_8 |
| FeatureEngineer | Yes | **NOT USED** in any M2 solution |
| FeatureStore | Yes | ex_8 |
| TrainingPipeline | Yes | Used indirectly via sklearn in ex_5/ex_6 |
| ModelVisualizer | Yes | ex_1, ex_2, ex_3, ex_4, ex_5, ex_6, ex_7 |

### Issues

| # | Severity | File | Issue |
|---|---|---|---|
| 4 | HIGH | mlfp02/ex_7 | **DiD (Difference-in-Differences) missing**: Spec 2.7 allocates ~1 hour to DiD with ATT derivation, parallel trends assumption, and placebo tests. The solution replaces DiD with Bayesian A/B testing and sequential testing (mSPRT). These are valuable additions but DiD is a named spec requirement. |
| 5 | HIGH | mlfp02/ex_8 | **FeatureEngineer engine missing**: Spec 2.8 explicitly requires `FeatureEngineer: generate features, select features`. The capstone uses only FeatureStore and manual Polars feature engineering. The FeatureEngineer Kailash engine is never imported or used in any M2 solution. |
| 6 | MEDIUM | mlfp02/ex_3 | **Title missing "Bootstrapping"**: Spec title is "Bootstrapping and Hypothesis Testing". Solution title is "Hypothesis Testing". The solution does include bootstrap content (permutation test), but the emphasis shift is notable in the title. |
| 7 | LOW | mlfp02/ex_1 | **Friendship paradox/sampling bias**: Spec 2.1 mentions "Sampling bias (friendship paradox from Deck 2A)" as a topic. Not demonstrated in the solution. Minor since the concept is referenced implicitly in the Bayesian updating discussion. |

### Scaffolding Assessment (Target: ~60%)

M2 solutions provide moderate-to-heavy scaffolding. Ex_1 and ex_2 are heavily commented. Ex_7 and ex_8 expect more independent work (FeatureStore setup, CUPED derivation). Broadly consistent with ~60% target.

---

## Module 3: Supervised Machine Learning for Building and Deploying Models

### Title Alignment

| Ex | Spec Title | Solution Title | Match |
|---|---|---|---|
| 1 | Feature Engineering, ML Pipeline, and Feature Selection | Feature Engineering | PARTIAL: "ML Pipeline" and "Feature Selection" dropped |
| 2 | Bias-Variance, Regularisation, and Cross-Validation | Bias-Variance and Regularisation | PARTIAL: "Cross-Validation" dropped |
| 3 | The Complete Supervised Model Zoo | The Complete Supervised Model Zoo | EXACT |
| 4 | Gradient Boosting Deep Dive | Gradient Boosting Deep Dive | EXACT |
| 5 | Model Evaluation, Imbalance, and Calibration | Class Imbalance and Calibration | PARTIAL: "Model Evaluation" dropped from title |
| 6 | Interpretability and Fairness | SHAP, LIME, and Fairness | CLOSE (more specific) |
| 7 | Workflow Orchestration, Model Registry, and Hyperparameter Search | Workflow Orchestration and Custom Nodes | PARTIAL: "Hyperparameter Search" and "Model Registry" dropped |
| 8 | Production Pipeline — DataFlow, Drift, and Deployment | Production Pipeline Project | CLOSE |

### Topic Coverage

| Ex | Key Spec Topics | Covered | Missing/Gaps |
|---|---|---|---|
| 1 | Feature engineering, ML pipeline stages, feature selection (filter/wrapper/embedded), leakage | YES | Uses clinical (ICU) data not HDB — dataset differs from spec but topics covered |
| 2 | Bias-variance decomposition, L1/L2/ElasticNet, Bayesian interpretation, cross-validation | YES | Nested CV may be absent |
| 3 | SVM, KNN, Naive Bayes, Decision Trees, Random Forests, model comparison | YES | None |
| 4 | XGBoost, LightGBM, CatBoost, AdaBoost, XGBoost 2nd-order Taylor | YES | None |
| 5 | Metrics taxonomy, SMOTE, cost-sensitive, Focal Loss, Platt/isotonic calibration | YES | None |
| 6 | SHAP (TreeSHAP, KernelSHAP), LIME, ALE, fairness (disparate impact, equalized odds) | YES | ALE may be absent |
| 7 | WorkflowBuilder, custom nodes, HyperparameterSearch, ModelRegistry, ModelSignature | PARTIAL | **HyperparameterSearch absent** |
| 8 | DataFlow, DriftMonitor, model card, conformal prediction | YES | None |

### Kailash Engine Usage

| Engine | Spec Requires | Used In |
|---|---|---|
| FeatureEngineer | Yes | ex_1 (header mentions it) |
| FeatureStore | Yes | ex_1 |
| PreprocessingPipeline | Yes | ex_1/ex_2 |
| TrainingPipeline | Yes | ex_3/ex_4/ex_5 |
| AutoMLEngine | Yes | Not found in M3 (used in M4) |
| HyperparameterSearch | Yes | **NOT FOUND in any M3 solution** |
| ModelRegistry | Yes | ex_7, ex_8 |
| EnsembleEngine | Yes | Not found in M3 |
| WorkflowBuilder | Yes | ex_7 |
| DataFlow | Yes | ex_7, ex_8 |
| DriftMonitor | Yes | ex_8 |
| ModelVisualizer | Yes | Multiple |

### Issues

| # | Severity | File | Issue |
|---|---|---|---|
| 8 | HIGH | mlfp03/ex_7 | **HyperparameterSearch absent**: Spec 3.7 explicitly requires "HyperparameterSearch: Bayesian optimisation, SearchSpace, ParamDistribution, SearchConfig". No grep match for `HyperparameterSearch` or `SearchSpace` in any M3 solution. This is a key Kailash engine that the spec allocates to this lesson. |
| 9 | MEDIUM | mlfp03/ex_1 | **Title truncation**: Spec title includes "ML Pipeline and Feature Selection" but solution title is just "Feature Engineering". The feature selection taxonomy (filter/wrapper/embedded) may be underemphasized. |
| 10 | MEDIUM | mlfp03/ex_5 | **Title drops "Model Evaluation"**: Spec lesson 3.5 is "Model Evaluation, Imbalance, and Calibration". Solution focuses on imbalance/calibration. The complete metrics taxonomy (precision, recall, F1, AUC-ROC, MAE, RMSE, MAPE) should be the foundation of this exercise. |
| 11 | MEDIUM | mlfp03/ex_7 | **Title drops "Hyperparameter Search" and "Model Registry"**: Combined with issue #8, this lesson is missing a major component. |

### Scaffolding Assessment (Target: ~50%)

M3 solutions provide moderate scaffolding with more independent coding required. Consistent with ~50% target.

---

## Module 4: Unsupervised Machine Learning and Advanced Techniques

### Title Alignment

| Ex | Spec Title | Solution Title | Match |
|---|---|---|---|
| 1 | Clustering | Clustering | EXACT |
| 2 | EM Algorithm and Gaussian Mixture Models | EM Algorithm and Gaussian Mixture Models | EXACT |
| 3 | Dimensionality Reduction | Dimensionality Reduction | EXACT |
| 4 | Anomaly Detection and Ensembles | Anomaly Detection and Ensembles | EXACT |
| 5 | Association Rules and Market Basket Analysis | Association Rules and Market Basket Analysis | EXACT |
| 6 | NLP — Text to Topics | NLP — Text to Topics | EXACT |
| 7 | Recommender Systems and Collaborative Filtering | Recommender Systems and Collaborative Filtering | EXACT |
| 8 | DL Foundations — Neural Networks, Backpropagation, and the Training Toolkit | Deep Learning Foundations | PARTIAL: title simplified |

### Topic Coverage

| Ex | Key Spec Topics | Covered | Missing/Gaps |
|---|---|---|---|
| 1 | K-means, hierarchical, DBSCAN, HDBSCAN, spectral, cluster eval (silhouette, DB, gap) | YES | Hierarchical mentioned in header; dendrogram depth unclear |
| 2 | EM from scratch (20 lines), GMM, Mixture of Experts | YES | None |
| 3 | PCA via SVD, scree plot, loadings, t-SNE, UMAP, kernel PCA | YES | Kernel PCA may be absent |
| 4 | Z-score, IQR, Isolation Forest, LOF, EnsembleEngine.blend() | YES | None |
| 5 | Apriori from scratch, FP-Growth, support/confidence/lift | YES | None |
| 6 | TF-IDF derivation, BM25, Word2Vec, LDA, NMF, BERTopic, NPMI | YES | BM25 unclear |
| 7 | Content-based, user-based, item-based CF, ALS from scratch, THE PIVOT | YES | None |
| 8 | Neural network from scratch, forward pass, backprop, activation functions, dropout, batch norm, optimisers, loss functions, LR schedules | PARTIAL | **Solution builds CNN with ResBlock, not from-scratch neural network** |

### Issues

| # | Severity | File | Issue |
|---|---|---|---|
| 12 | HIGH | mlfp04/ex_8 | **Spec-solution mismatch on approach**: Spec 4.8 requires building a "3-layer neural network from scratch for HDB price prediction" with manual forward pass, backprop, and gradient descent. The solution instead builds a CNN with residual connections (ResBlock) using PyTorch on synthetic medical image data. The from-scratch neural network implementation and the HDB dataset are both absent. The DL training toolkit (activation functions, dropout, batch norm, optimisers, loss functions, LR schedules) is covered through the PyTorch CNN training, but the pedagogical intent of "build from scratch" is lost. |
| 13 | MEDIUM | mlfp04/ex_8 | **Dataset mismatch**: Spec says "HDB price prediction" as the exercise dataset. Solution uses "synthetic medical image data (5000 x 64x64 images, 5 conditions)". This is a fundamentally different task (image classification vs tabular regression). |
| 14 | LOW | mlfp04/ex_1 | **Dendrogram depth unclear**: Spec 4.1 emphasises hierarchical clustering with 4 linkage methods and dendrograms extensively. The solution mentions "hierarchical" in the header but actual dendrogram depth should be verified. |

### Scaffolding Assessment (Target: ~40%)

M4 solutions reduce scaffolding. More code is expected from students with briefer comments. Consistent with ~40% target.

---

## Module 5: Deep Learning and Machine Learning Mastery

### Title Alignment

| Ex | Spec Title | Solution Title | Match |
|---|---|---|---|
| 1 | Autoencoders | Autoencoders | EXACT |
| 2 | CNNs and Computer Vision | CNNs for Image Classification | CLOSE |
| 3 | RNNs and Sequence Models | Sequence Models — RNNs and LSTMs | CLOSE |
| 4 | Transformers | Transformer Architecture | CLOSE |
| 5 | Generative Models — GANs and Diffusion | Generative Models — GANs and Diffusion | EXACT |
| 6 | Graph Neural Networks | Graph Neural Networks | EXACT |
| 7 | Transfer Learning | Transfer Learning with Transformers | CLOSE |
| 8 | Reinforcement Learning | Reinforcement Learning | EXACT |

### Topic Coverage

| Ex | Key Spec Topics | Covered | Missing/Gaps |
|---|---|---|---|
| 1 | Vanilla, denoising, VAE, convolutional AE; reparameterisation trick | YES | None |
| 2 | CNN fundamentals, ResNet, SE blocks, mixed precision, ViT intro | YES | ViT and SE blocks depth unclear |
| 3 | RNN, LSTM (all 6 gate equations), GRU, attention | YES | None |
| 4 | Self-attention from scratch, positional encoding, multi-head, BERT fine-tuning | YES | BERT fine-tuning depth unclear (may be via AutoMLEngine) |
| 5 | DCGAN, WGAN, FID, diffusion basics | YES | None |
| 6 | GCN, GraphSAGE, GAT, GIN; node classification | YES | GIN may be absent |
| 7 | ResNet fine-tuning, BERT fine-tuning, ONNX export, InferenceServer | PARTIAL | Uses AutoMLEngine not explicit ResNet fine-tuning |
| 8 | DQN, DDPG, SAC, A2C, PPO; custom Gymnasium environments | PARTIAL | Only PPO mentioned in header; 5 algorithms may not all be present |

### Issues

| # | Severity | File | Issue |
|---|---|---|---|
| 15 | MEDIUM | mlfp05/ex_7 | **Transfer learning approach differs**: Spec requires explicit ResNet fine-tuning (freeze early layers, train later layers) and BERT fine-tuning. Solution uses AutoMLEngine for text classification. The explicit freeze/unfreeze layer-by-layer transfer learning pedagogy may be lost. |
| 16 | MEDIUM | mlfp05/ex_8 | **RL algorithm coverage**: Spec requires 5 algorithms (DQN, DDPG, SAC, A2C, PPO) with 5 business applications. Solution header mentions only PPO via RLTrainer. The breadth of 5 algorithms may not be achieved. |

### Scaffolding Assessment (Target: ~30%)

M5 solutions provide light scaffolding as expected. Students must write substantial code. Consistent with ~30% target.

---

## Module 6: Machine Learning with Language Models and Agentic Workflows

### Title Alignment

| Ex | Spec Title | Solution Title | Match |
|---|---|---|---|
| 1 | LLM Fundamentals, Prompt Engineering, and Structured Output | Prompt Engineering | PARTIAL: "LLM Fundamentals" and "Structured Output" dropped |
| 2 | LLM Fine-tuning — LoRA, Adapters, and the Technique Landscape | LoRA Fine-Tuning with AlignmentPipeline | PARTIAL: Adapters and technique landscape dropped |
| 3 | Preference Alignment — DPO and GRPO | DPO Preference Alignment | PARTIAL: **GRPO dropped** |
| 4 | RAG Systems | RAG Fundamentals | CLOSE |
| 5 | AI Agents — ReAct, Tool Use, and Function Calling | Building Agents with Kaizen | CLOSE |
| 6 | Multi-Agent Orchestration and MCP | Multi-Agent Orchestration | PARTIAL: MCP dropped from title |
| 7 | AI Governance Engineering | AI Governance with PACT | CLOSE |
| 8 | Capstone — Full Production Platform | Capstone — Governed ML System | CLOSE |

### Topic Coverage

| Ex | Key Spec Topics | Covered | Missing/Gaps |
|---|---|---|---|
| 1 | LLM foundation, prompt engineering (5+ techniques), Kaizen Delegate, Signature | YES | Inference considerations (KV-cache, speculative decoding) unclear |
| 2 | LoRA deep dive, adapter layers, 10-technique landscape, model merging, quantisation | PARTIAL | **Adapter from scratch, technique landscape survey, model merging, quantisation may be absent** |
| 3 | DPO, GRPO, LLM-as-Judge, evaluation benchmarks | PARTIAL | **GRPO completely absent** (zero grep matches); lm-eval/MMLU/HellaSwag absent |
| 4 | Chunking, dense/sparse/hybrid retrieval, RAGAS, HyDE | YES | None |
| 5 | ReAct agents, function calling, cost budgets, agent mental framework | YES | None |
| 6 | Multi-agent patterns, A2A protocol, agent memory, MCP server | PARTIAL | **A2A protocol absent** (zero grep matches); MCP in ex_8 not ex_6 |
| 7 | PACT D/T/R, operating envelopes, budget cascading, governance testing | YES | None |
| 8 | Nexus deployment (API+CLI+MCP), RBAC, DriftMonitor integration | YES | None |

### Issues

| # | Severity | File | Issue |
|---|---|---|---|
| 17 | HIGH | mlfp06/ex_3 | **GRPO completely absent**: Spec 6.3 explicitly requires GRPO (Group Relative Policy Optimization, used in DeepSeek-R1) as a named topic with comparison to DPO. Zero matches for "GRPO" in any M6 solution. This is a 2025 frontier technique the spec highlights. |
| 18 | HIGH | mlfp06/ex_3 | **Evaluation benchmarks absent**: Spec 6.3 requires lm-eval-harness, MMLU, HellaSwag, HumanEval, MT-Bench as named evaluation tools. Zero matches for any of these in ex_3. |
| 19 | HIGH | mlfp06/ex_2 | **Adapter from-scratch and technique landscape absent**: Spec 6.2 requires (a) adapter layers implemented from scratch, (b) 8 additional fine-tuning techniques surveyed, (c) model merging (TIES, DARE, SLERP), (d) quantisation (GPTQ, AWQ, QLoRA). The solution title "LoRA Fine-Tuning with AlignmentPipeline" suggests only LoRA is covered. |
| 20 | MEDIUM | mlfp06/ex_6 | **A2A protocol and MCP absent from ex_6**: Spec 6.6 requires A2A (Agent-to-Agent) protocol and building an MCP server. Zero matches for A2A in the solution. MCP appears in ex_7/ex_8 instead. |

### Scaffolding Assessment (Target: ~20%)

M6 solutions provide minimal scaffolding. Students write most code. Consistent with ~20% target.

---

## Summary Table

| # | Severity | Module | Exercise | Issue |
|---|---|---|---|---|
| 1 | MEDIUM | M1 | ex_8 | Title drift ("Data Cleaning" vs spec's "Data Pipelines") |
| 2 | MEDIUM | M1 | ex_8 | REST API data loading absent (spec requires OneMap example) |
| 3 | LOW | M1 | ex_6 | US vs UK spelling ("Visualization" vs "Visualisation") |
| 4 | HIGH | M2 | ex_7 | Difference-in-Differences completely absent |
| 5 | HIGH | M2 | ex_8 | FeatureEngineer Kailash engine never used in any M2 solution |
| 6 | MEDIUM | M2 | ex_3 | "Bootstrapping" dropped from title |
| 7 | LOW | M2 | ex_1 | Sampling bias / friendship paradox not demonstrated |
| 8 | HIGH | M3 | ex_7 | HyperparameterSearch Kailash engine not used in any M3 solution |
| 9 | MEDIUM | M3 | ex_1 | Title drops "ML Pipeline and Feature Selection" |
| 10 | MEDIUM | M3 | ex_5 | Title drops "Model Evaluation" |
| 11 | MEDIUM | M3 | ex_7 | Title drops "Hyperparameter Search" and "Model Registry" |
| 12 | HIGH | M4 | ex_8 | From-scratch neural network replaced with PyTorch CNN; HDB dataset replaced with medical images |
| 13 | MEDIUM | M4 | ex_8 | Dataset mismatch (spec: HDB, solution: synthetic medical) |
| 14 | LOW | M4 | ex_1 | Dendrogram/hierarchical clustering depth unclear |
| 15 | MEDIUM | M5 | ex_7 | Transfer learning uses AutoMLEngine instead of explicit freeze/unfreeze |
| 16 | MEDIUM | M5 | ex_8 | Only PPO visible; 5 RL algorithms (DQN, DDPG, SAC, A2C, PPO) may not all be present |
| 17 | HIGH | M6 | ex_3 | GRPO completely absent from all M6 solutions |
| 18 | HIGH | M6 | ex_3 | Evaluation benchmarks (lm-eval, MMLU, HellaSwag, etc.) absent |
| 19 | HIGH | M6 | ex_2 | Adapter from-scratch, technique landscape, model merging, quantisation absent |
| 20 | MEDIUM | M6 | ex_6 | A2A protocol and MCP server absent from ex_6 |

### Severity Counts

| Severity | Count |
|---|---|
| HIGH | 7 |
| MEDIUM | 10 |
| LOW | 3 |
| **Total** | **20** |

---

## Design Principles Compliance

| Principle | Status | Notes |
|---|---|---|
| 1. Kailash-Only Stack | PASS with gaps | DataExplorer, PreprocessingPipeline, ModelVisualizer, TrainingPipeline, FeatureStore, ExperimentTracker, ModelRegistry, WorkflowBuilder, DataFlow, DriftMonitor, AutoMLEngine, EnsembleEngine, OnnxBridge, AlignmentPipeline, AdapterRegistry, GovernanceEngine all used. **Gaps**: HyperparameterSearch (M3), FeatureEngineer (M2), RLTrainer coverage depth (M5). |
| 2. Python Through Data | PASS | Every Python concept grounded in data from lesson 1.1 |
| 3. Engines Before Infrastructure | PASS | DataExplorer/ModelVisualizer used in M1 before WorkflowBuilder in M3 |
| 4. Feature Engineering Spectrum | PASS | M3 manual -> M4 USML -> M4.7 THE PIVOT -> M4.8 DL -> M5 architectures -> M6 LLMs |
| 5. Statistics Teaches Models | PASS | Regression in M2 before ML pipeline in M3 |
| 6. Engineering Not Philosophy | PASS | All governance is coded (PACT), all alignment is trained (DPO) |
| 7. Progressive Scaffolding | PASS | M1 ~70% -> M2 ~60% -> M3 ~50% -> M4 ~40% -> M5 ~30% -> M6 ~20% |
| 8. Three Layers Per Concept | PARTIAL | FOUNDATIONS/THEORY/ADVANCED markers not explicitly used in solutions |

---

## Recommendations

### Must Fix (HIGH severity)

1. **M2/ex_7**: Add Difference-in-Differences section with ATT derivation, parallel trends test, and placebo test. Can replace or coexist with the Bayesian A/B testing content.
2. **M2/ex_8**: Import and use `FeatureEngineer` from `kailash_ml` for automated feature generation alongside manual feature engineering.
3. **M3/ex_7**: Add `HyperparameterSearch` with `SearchSpace`, `ParamDistribution`, and `SearchConfig`. This is a core Kailash engine for this lesson.
4. **M4/ex_8**: Add a from-scratch numpy neural network section (forward pass, backprop, gradient descent on HDB data) before or alongside the PyTorch CNN section.
5. **M6/ex_2**: Add adapter layer from-scratch implementation, technique landscape survey table, model merging (TIES/DARE/SLERP), and quantisation overview.
6. **M6/ex_3**: Add GRPO section and evaluation benchmark coverage (lm-eval-harness, MMLU, HellaSwag, etc.).

### Should Fix (MEDIUM severity)

7. **M1/ex_8**: Add a REST API section demonstrating GET requests to OneMap or data.gov.sg API.
8. **M3/ex_5**: Rename to include "Model Evaluation" and ensure the complete metrics taxonomy is the opening section.
9. **M5/ex_7**: Add explicit layer freezing/unfreezing transfer learning before using AutoMLEngine.
10. **M5/ex_8**: Verify all 5 RL algorithms are covered; add DQN and at least one continuous-action algorithm.
11. **M6/ex_6**: Add A2A protocol and MCP server building (currently deferred to ex_7/ex_8).
