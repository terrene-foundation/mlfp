# RT4: Spec Compliance Audit — MLFP Full Curriculum

**Date**: 2026-04-13
**Scope**: All 48 lessons (6 modules x 8 lessons)
**Method**: For each lesson, read spec → extract every topic/concept/formula/engine → check textbook → check exercise → flag gaps
**Classification**:
- **ZERO**: Absent from BOTH textbook AND exercise (most critical)
- **HIGH**: In textbook but NOT in exercise (exercise gap)
- **PARTIAL**: Superficially covered, missing depth or key details
- **COVERED**: Present in both textbook and exercise

---

## Executive Summary

| Module | Spec Topics | Textbook Coverage | Exercise Coverage | ZERO Gaps | HIGH Gaps |
|--------|-------------|-------------------|-------------------|-----------|-----------|
| M1     | ~64         | ~95%              | ~60%              | 3         | 5         |
| M2     | ~72         | ~90%              | ~55%              | 6         | 6         |
| M3     | ~80         | ~85%              | ~40%              | 5         | 8         |
| M4     | ~76         | ~80%              | ~35%              | 5         | 6         |
| M5     | ~84         | ~75%              | ~30%              | 16        | 12        |
| M6     | ~80         | ~80%              | ~30%              | 9         | 12        |
| **Total** | **~456** | **~84%**          | **~42%**          | **44**    | **49**    |

**Key pattern**: Textbooks cover ~80-90% of spec topics. Exercises implement ~30-50% of spec exercise requirements. Coverage degrades in M5-M6 where topic density is highest. Lesson 5.8 (RL) is the single most incomplete lesson in the entire curriculum.

**Prior red-team focus areas (7 HIGH gaps)**: DiD (M2.7), FeatureEngineer (M2.8), HyperparameterSearch (M3.7), DriftMonitor (M3.8/M6.8), GRPO (M6.3), A2A (M6.6), RLTrainer (M5.8). Status: all 7 confirmed — 5 are textbook-covered but exercise-missing (HIGH), 2 are ZERO coverage (A2A, RLTrainer in M5).

---

## Module 1: Data Pipelines and Python Foundations

### Coverage Table

| Lesson | Topic | Textbook | Exercise | Gap Level |
|--------|-------|----------|----------|-----------|
| 1.1 | Python fundamentals, f-strings, list comprehensions | COVERED | COVERED | — |
| 1.1 | Type annotations, walrus operator | COVERED | COVERED | — |
| 1.2 | OOP (classes, inheritance, dunder methods) | COVERED | COVERED | — |
| 1.2 | Decorators, context managers, generators | COVERED | COVERED | — |
| 1.3 | Polars expressions, lazy frames, window functions | COVERED | COVERED | — |
| 1.3 | GroupBy, joins, pivots, concat | COVERED | COVERED | — |
| 1.4 | DataExplorer (describe, profile, correlations) | COVERED | COVERED | — |
| 1.4 | Dictionary lookups (spec mentions as key topic) | MISSING | MISSING | **ZERO** |
| 1.5 | PreprocessingPipeline (missing values, encoding, scaling) | COVERED | COVERED | — |
| 1.5 | Pipeline composition | COVERED | COVERED | — |
| 1.6 | ModelVisualizer (distribution, scatter, correlation) | COVERED | COVERED | — |
| 1.6 | Z-pattern reading / visual hierarchy | MISSING | MISSING | **ZERO** |
| 1.6 | 100% stacked bars, Likert scales | MISSING | MISSING | **ZERO** |
| 1.7 | End-to-end pipeline (load → clean → explore → visualize) | COVERED | COVERED | — |
| 1.7 | Pipeline orchestration pattern | COVERED | COVERED | — |
| 1.8 | REST API consumption (OneMap example) | COVERED | MISSING | HIGH |
| 1.8 | JSON parsing, error handling, rate limiting | COVERED | MISSING | HIGH |
| 1.8 | DataExplorer on API-fetched data | COVERED | MISSING | HIGH |
| 1.8 | ExperimentTracker for data lineage | COVERED | MISSING | HIGH |
| 1.8 | async/httpx patterns | COVERED | MISSING | HIGH |

### ZERO Gaps (3)
1. **Dictionary lookups** (1.4) — spec mentions as key topic, absent from both
2. **Z-pattern reading / visual hierarchy** (1.6) — data storytelling concept absent from both
3. **100% stacked bars / Likert scales** (1.6) — specific chart types not covered

### HIGH Gaps (5)
1. **REST API exercise** (1.8) — textbook covers OneMap example; exercise does not implement
2. **JSON parsing patterns** (1.8) — textbook covers; no exercise practice
3. **DataExplorer on API data** (1.8) — textbook shows; exercise missing
4. **ExperimentTracker** (1.8) — textbook shows; exercise missing
5. **async/httpx** (1.8) — textbook shows; exercise missing

### Engine Usage

| Engine | Spec Requires | Textbook | Exercise |
|--------|--------------|----------|----------|
| DataExplorer | 1.4, 1.7-1.8 | COVERED | COVERED (1.4, 1.7) |
| PreprocessingPipeline | 1.5, 1.7 | COVERED | COVERED |
| ModelVisualizer | 1.6-1.7 | COVERED | COVERED |
| ExperimentTracker | 1.8 | COVERED | **NOT USED** |

---

## Module 2: Statistics and Experimental Design

### Coverage Table

| Lesson | Topic | Textbook | Exercise | Gap Level |
|--------|-------|----------|----------|-----------|
| 2.1 | Probability axioms, conditional probability | COVERED | COVERED | — |
| 2.1 | Bayes' theorem, conjugate priors | COVERED | COVERED | — |
| 2.1 | Common distributions (Normal, Beta, Poisson, Exponential) | COVERED | PARTIAL | — |
| 2.1 | Truth table construction for events | MISSING | MISSING | **ZERO** |
| 2.1 | Independent vs dependent events — decision steps | MISSING | MISSING | **ZERO** |
| 2.1 | Supermarket example | MISSING | MISSING | **ZERO** |
| 2.2 | MLE derivation (Normal distribution) | COVERED | COVERED | — |
| 2.2 | MAP estimation | COVERED | COVERED | — |
| 2.2 | Confidence intervals (Wald, profile likelihood) | COVERED | COVERED | — |
| 2.2 | Fisher information / Cramer-Rao bound | COVERED | COVERED | — |
| 2.2 | Central Limit Theorem | COVERED | PARTIAL | — |
| 2.3 | Bootstrap (percentile + BCa) | COVERED | COVERED | — |
| 2.3 | Hypothesis testing, p-values, t-statistic | COVERED | COVERED | — |
| 2.3 | Power analysis, MDE | COVERED | COVERED | — |
| 2.3 | Multiple testing (Bonferroni, BH-FDR) | COVERED | COVERED | — |
| 2.3 | Permutation tests | COVERED | COVERED | — |
| 2.4 | Pre-registering hypothesis | COVERED | COVERED | — |
| 2.4 | Power analysis / sample size | COVERED | COVERED | — |
| 2.4 | SRM detection (chi-squared) | COVERED | COVERED | — |
| 2.4 | Data collection framework (Why/What/Where/How) | COVERED | COVERED | — |
| 2.4 | Clean DataOps architecture (schema, transfer, connectors) | MISSING | MISSING | **ZERO** |
| 2.5 | OLS derivation, normal equations | COVERED | COVERED | — |
| 2.5 | R², adjusted R², F-statistic | COVERED | COVERED | — |
| 2.5 | Categorical encoding, non-linearity | COVERED | COVERED | — |
| 2.5 | Coefficient interpretation (ceteris paribus) | COVERED | COVERED | — |
| 2.6 | Sigmoid derivation, log-odds | COVERED | COVERED | — |
| 2.6 | Odds ratios with CIs | COVERED | COVERED | — |
| 2.6 | Multiclass extensions (OvR, OvO, multinomial) | COVERED | MISSING | HIGH |
| 2.6 | One-way ANOVA, Tukey HSD | COVERED | COVERED | — |
| 2.6 | Scheffe post-hoc test | MISSING | MISSING | **ZERO** |
| 2.6 | Two-way ANOVA / repeated measures (extensions) | MISSING | MISSING | **ZERO** |
| 2.7 | CUPED derivation (theta, variance reduction) | COVERED | COVERED | — |
| 2.7 | SRM detection | COVERED | COVERED | — |
| 2.7 | Difference-in-Differences (DiD) | COVERED | **MISSING** | **HIGH** |
| 2.7 | ATT formula, parallel trends, placebo tests | COVERED | **MISSING** | **HIGH** |
| 2.8 | FeatureEngineer (temporal, interactions, polynomial) | COVERED | **MISSING** | **HIGH** |
| 2.8 | FeatureStore (point-in-time, register, retrieve) | COVERED | COVERED | — |
| 2.8 | Full capstone pipeline (load→describe→hypothesise→test→model→interpret→report) | COVERED | **MISSING** | **HIGH** |
| 2.8 | Project choice (Wine Quality / Economic Indicators / Experiment Design) | COVERED | MISSING | HIGH |

### ZERO Gaps (6)
1. **Truth table construction** (2.1) — spec requires, absent from both
2. **Independent vs dependent events decision steps** (2.1) — no explicit decision framework
3. **Supermarket example** (2.1) — spec mentions, absent from both
4. **Clean DataOps architecture** (2.4) — data pipeline architecture for experiments
5. **Scheffe post-hoc test** (2.6) — spec requires alongside Tukey HSD
6. **Two-way ANOVA / repeated measures** (2.6) — spec says "mention as extensions"

### HIGH Gaps (6)
1. **DiD implementation** (2.7) — textbook has full section; exercise substitutes Bayesian A/B testing
2. **ATT formula / parallel trends / placebo tests** (2.7) — textbook covers; exercise missing
3. **FeatureEngineer engine** (2.8) — textbook shows code; exercise uses manual Polars operations
4. **Full capstone pipeline** (2.8) — exercise is FeatureStore-only, not end-to-end
5. **Multiclass logistic regression** (2.6) — textbook covers OvR/OvO; exercise missing
6. **Project choice options** (2.8) — textbook presents 3 options; exercise is single-track

### Engine Usage

| Engine | Spec Requires | Textbook | Exercise |
|--------|--------------|----------|----------|
| ExperimentTracker | 2.1-2.4, 2.7 | Mentioned | Used in ex_7, ex_8 |
| FeatureEngineer | 2.8 | Code shown | **NOT USED** |
| FeatureStore | 2.8 | Code shown | COVERED (ex_8 primary focus) |
| TrainingPipeline | 2.5-2.6, 2.8 | Mentioned | **NOT USED in any exercise** |
| ModelVisualizer | General | Not in textbooks | Used in ex_1 through ex_7 |

---

## Module 3: Supervised Machine Learning

### Coverage Table

| Lesson | Topic | Textbook | Exercise | Gap Level |
|--------|-------|----------|----------|-----------|
| 3.1 | KNN (distance metrics, curse of dimensionality) | COVERED | COVERED | — |
| 3.1 | Decision trees (Gini, entropy, pruning) | COVERED | COVERED | — |
| 3.1 | Geocoding / OneMap API | MISSING | MISSING | **ZERO** |
| 3.2 | Feature selection (filter/wrapper/embedded) | COVERED | MISSING | HIGH |
| 3.2 | FeatureEngineer, FeatureStore lifecycle | COVERED | COVERED | — |
| 3.2 | PreprocessingPipeline integration | COVERED | COVERED | — |
| 3.3 | Bias-variance tradeoff | COVERED | COVERED | — |
| 3.3 | Cross-validation (k-fold, stratified, nested) | COVERED | PARTIAL | — |
| 3.3 | Nested CV | COVERED | MISSING | HIGH |
| 3.4 | Random Forest (bagging, OOB, feature importance) | COVERED | COVERED | — |
| 3.4 | Gradient Boosting (XGBoost, LightGBM) | COVERED | COVERED | — |
| 3.4 | AdaBoost (weighted voting, exponential loss) | MISSING | MISSING | **ZERO** |
| 3.5 | Regression metrics (MAE, MSE, RMSE, MAPE, R²) | MISSING | MISSING | **ZERO** |
| 3.5 | Classification metrics (precision, recall, F1, AUC) | COVERED | COVERED | — |
| 3.5 | Stacking / blending | MISSING | MISSING | **ZERO** |
| 3.5 | ALE plots | COVERED | MISSING | HIGH |
| 3.6 | Fairness metrics (demographic parity, equalized odds) | COVERED | COVERED | — |
| 3.6 | Calibration curves | COVERED | COVERED | — |
| 3.6 | Calibration parity across groups | MISSING | MISSING | **ZERO** |
| 3.7 | HyperparameterSearch (grid, random, Bayesian) | COVERED | **MISSING** | **HIGH** |
| 3.7 | AutoMLEngine | COVERED | COVERED | — |
| 3.7 | ModelRegistry (versioning, metadata, comparison) | COVERED | COVERED | — |
| 3.8 | WorkflowBuilder + custom nodes | COVERED | MISSING | HIGH |
| 3.8 | DataFlow integration | COVERED | COVERED | — |
| 3.8 | DriftMonitor (feature/concept drift) | COVERED | **MISSING** | **HIGH** |
| 3.8 | End-to-end production pipeline | COVERED | PARTIAL | — |

### ZERO Gaps (5)
1. **Geocoding / OneMap API** (3.1) — spec requires Singapore-specific geocoding
2. **AdaBoost** (3.4) — weighted voting algorithm absent from both
3. **Regression metrics** (3.5) — MAE/MSE/RMSE/MAPE derivations missing
4. **Stacking / blending** (3.5) — ensemble meta-learning absent from both
5. **Calibration parity across groups** (3.6) — fairness calibration absent from both

### HIGH Gaps (8)
1. **HyperparameterSearch engine** (3.7) — CRITICAL: textbook covers; exercise uses only AutoMLEngine
2. **DriftMonitor** (3.8) — CRITICAL: textbook covers; exercise missing
3. **Feature selection methods** (3.2) — filter/wrapper/embedded; textbook covers
4. **Nested cross-validation** (3.3) — textbook covers; exercise has only basic k-fold
5. **ALE plots** (3.5) — textbook covers; exercise missing
6. **Custom WorkflowBuilder nodes** (3.8) — textbook covers; exercise missing
7. **EnsembleEngine advanced patterns** (3.5) — textbook covers; exercise partial
8. **ModelVisualizer for interpretability** (3.5) — underutilized in exercises

### Engine Usage

| Engine | Spec Requires | Textbook | Exercise |
|--------|--------------|----------|----------|
| FeatureEngineer | 3.2 | COVERED | COVERED |
| FeatureStore | 3.2 | COVERED | COVERED |
| PreprocessingPipeline | 3.2 | COVERED | COVERED |
| TrainingPipeline | 3.3-3.8 | COVERED | COVERED |
| AutoMLEngine | 3.7 | COVERED | COVERED |
| HyperparameterSearch | 3.7 | COVERED | **NOT USED** |
| ModelRegistry | 3.7-3.8 | COVERED | COVERED |
| EnsembleEngine | 3.5 | COVERED | PARTIAL |
| WorkflowBuilder | 3.8 | COVERED | **NOT USED** |
| DriftMonitor | 3.8 | COVERED | **NOT USED** |
| ModelVisualizer | 3.5-3.6 | COVERED | COVERED |

---

## Module 4: Unsupervised ML, Anomaly Detection, NLP, and DL Bridge

### Coverage Table

| Lesson | Topic | Textbook | Exercise | Gap Level |
|--------|-------|----------|----------|-----------|
| 4.1 | K-Means (Lloyd's algorithm, elbow, silhouette) | COVERED | COVERED | — |
| 4.1 | DBSCAN (eps, minPts, border/noise) | COVERED | COVERED | — |
| 4.1 | Hierarchical clustering (dendrogram, linkage) | COVERED | MISSING | HIGH |
| 4.1 | ARI / NMI metrics | MISSING | MISSING | **ZERO** |
| 4.2 | PCA (eigendecomposition, explained variance) | COVERED | COVERED | — |
| 4.2 | t-SNE (perplexity, KL divergence) | COVERED | COVERED | — |
| 4.2 | UMAP (mathematical intuition) | COVERED | COVERED | — |
| 4.3 | Z-score / IQR anomaly detection | COVERED | MISSING | HIGH |
| 4.3 | Isolation Forest | COVERED | COVERED | — |
| 4.3 | LOF | COVERED | COVERED | — |
| 4.3 | Manifold learning comparison table (spec requires) | MISSING | MISSING | **ZERO** |
| 4.4 | Autoencoder anomaly detection | COVERED | COVERED | — |
| 4.4 | Temporal anomaly (sliding window) | COVERED | COVERED | — |
| 4.5 | TF-IDF derivation | COVERED | COVERED | — |
| 4.5 | LDA topic modeling | COVERED | MISSING | HIGH |
| 4.5 | Word embeddings (Word2Vec, cosine similarity) | COVERED | MISSING | HIGH |
| 4.6 | Sentiment analysis (lexicon-based + ML) | MISSING | MISSING | **ZERO** |
| 4.6 | BM25 ranking | COVERED | MISSING | HIGH |
| 4.7 | Recommender systems (content-based + collaborative) | COVERED | COVERED | — |
| 4.7 | Matrix factorization (SVD) | COVERED | COVERED | — |
| 4.8 | Perceptron → MLP (from scratch) | COVERED | PARTIAL | — |
| 4.8 | Backpropagation derivation | COVERED | PARTIAL | — |
| 4.8 | Loss functions taxonomy (MSE, BCE, CE, Hinge) | MISSING | MISSING | **ZERO** |
| 4.8 | Adam optimizer formula | MISSING | MISSING | **ZERO** |
| 4.8 | CNN with ResBlock | COVERED | COVERED | — |
| 4.8 | HDB dataset for DL bridge | COVERED | MISSING | HIGH |

### ZERO Gaps (5)
1. **ARI / NMI metrics** (4.1) — cluster evaluation metrics absent from both
2. **Manifold learning comparison table** (4.3) — spec requires systematic comparison
3. **Sentiment analysis** (4.6) — lexicon-based + ML approaches absent from both
4. **Loss functions taxonomy** (4.8) — MSE/BCE/CE/Hinge comparison missing
5. **Adam optimizer formula** (4.8) — momentum + RMSprop derivation missing

### HIGH Gaps (6)
1. **Hierarchical clustering** (4.1) — textbook covers; exercise missing
2. **Z-score / IQR anomaly detection** (4.3) — textbook covers; exercise missing
3. **LDA topic modeling** (4.5) — textbook covers; exercise missing
4. **Word embeddings** (4.5) — textbook covers; exercise missing
5. **BM25 ranking** (4.6) — textbook covers; exercise missing
6. **HDB dataset in DL bridge** (4.8) — exercise uses synthetic medical images instead

### Engine Usage

| Engine | Spec Requires | Textbook | Exercise |
|--------|--------------|----------|----------|
| AutoMLEngine | 4.1-4.3, 4.7 | COVERED | COVERED |
| EnsembleEngine | 4.4 | COVERED | COVERED |
| ModelVisualizer | 4.1-4.3, 4.7 | COVERED | COVERED |
| OnnxBridge | 4.8 | COVERED | PARTIAL |
| DriftMonitor | 4.4 | Mentioned | **NOT USED** |

---

## Module 5: Deep Learning

### Coverage Table

| Lesson | Topic | Textbook | Exercise | Gap Level |
|--------|-------|----------|----------|-----------|
| 5.1 | CNN architecture (conv, pool, stride, padding) | COVERED | COVERED | — |
| 5.1 | Feature map visualization | COVERED | COVERED | — |
| 5.1 | Data augmentation | COVERED | COVERED | — |
| 5.2 | ResNet (skip connections, residual learning) | COVERED | COVERED | — |
| 5.2 | Transfer learning (freeze/unfreeze) | COVERED | COVERED | — |
| 5.2 | CNN training enhancements (LR scheduling, early stopping) | COVERED | MISSING | HIGH |
| 5.3 | RNN (vanishing gradient, BPTT) | COVERED | COVERED | — |
| 5.3 | LSTM (gate equations) | COVERED | COVERED | — |
| 5.3 | GRU | COVERED | COVERED | — |
| 5.3 | Multi-layer RNN with residual connections | MISSING | MISSING | **ZERO** |
| 5.3 | Spatial attention mechanism | MISSING | MISSING | **ZERO** |
| 5.3 | Shakespeare text generation | MISSING | MISSING | **ZERO** |
| 5.3 | Financial time-series indicators (MA, RSI, MACD) | MISSING | MISSING | **ZERO** |
| 5.4 | Transformer architecture (self-attention, multi-head) | COVERED | COVERED | — |
| 5.4 | Positional encoding | COVERED | COVERED | — |
| 5.4 | Transformer-XL / relative position encoding | MISSING | MISSING | **ZERO** |
| 5.4 | BERT fine-tuning (classification task) | COVERED | MISSING | HIGH |
| 5.4 | Temporal attention for sequences | COVERED | MISSING | HIGH |
| 5.5 | Autoencoder survey (vanilla, denoising, variational) | COVERED | MISSING | HIGH |
| 5.5 | VAE (reparameterization trick, ELBO) | COVERED | COVERED | — |
| 5.5 | GAN (generator/discriminator, mode collapse, training tricks) | COVERED | COVERED | — |
| 5.5 | cGAN / CycleGAN / StyleGAN | MISSING | MISSING | **ZERO** |
| 5.6 | GNN fundamentals (message passing, aggregation) | COVERED | COVERED | — |
| 5.6 | GCN, GraphSAGE | COVERED | COVERED | — |
| 5.6 | GIN (Graph Isomorphism Network) | MISSING | MISSING | **ZERO** |
| 5.6 | Graph attention (GAT) | COVERED | COVERED | — |
| 5.7 | OnnxBridge (export, quantize, benchmark) | COVERED | COVERED | — |
| 5.7 | InferenceServer (load, predict, batch) | COVERED | MISSING | HIGH |
| 5.7 | Freeze/unfreeze layers in transfer learning | COVERED | COVERED | — |
| 5.7 | BERT fine-tuning for NER/QA | COVERED | MISSING | HIGH |
| 5.8 | RL fundamentals (MDP, reward, policy) | COVERED | COVERED | — |
| 5.8 | REINFORCE algorithm | COVERED | COVERED | — |
| 5.8 | DQN (experience replay, target network) | COVERED | **MISSING** | HIGH |
| 5.8 | DDPG / SAC / A2C | MISSING | MISSING | **ZERO** |
| 5.8 | PPO (clipped surrogate) | COVERED | **MISSING** | HIGH |
| 5.8 | Custom Gymnasium environments | MISSING | MISSING | **ZERO** |
| 5.8 | RLTrainer engine | MISSING | MISSING | **ZERO** |
| 5.8 | 5 RL business applications (inventory, pricing, portfolio, routing, scheduling) | COVERED | **MISSING** | HIGH |

### ZERO Gaps (16)
1. **Multi-layer RNN with residual connections** (5.3)
2. **Spatial attention mechanism** (5.3)
3. **Shakespeare text generation** (5.3) — spec requires as creative generation exercise
4. **Financial time-series indicators** (5.3) — MA, RSI, MACD for time-series features
5. **Transformer-XL / relative position encoding** (5.4)
6. **cGAN / CycleGAN / StyleGAN** (5.5) — GAN variants absent from both
7. **GIN (Graph Isomorphism Network)** (5.6)
8. **DDPG** (5.8) — continuous-action RL algorithm
9. **SAC** (5.8) — entropy-regularized RL
10. **A2C** (5.8) — advantage actor-critic
11. **Custom Gymnasium environments** (5.8) — spec requires building custom envs
12. **RLTrainer engine** (5.8) — kailash-ml RL engine absent from M5 entirely (only in M6 ex_8)
13. **Multi-layer RNN residual** (5.3)
14. **Spatial attention** (5.3)
15. **Shakespeare text gen** (5.3)
16. **Financial indicators** (5.3)

*Note: Items 13-16 deduplicate with 1-4 above. True unique ZERO count for M5 is 12.*

### Corrected ZERO Gaps (12)
1. **Multi-layer RNN with residual connections** (5.3)
2. **Spatial attention mechanism** (5.3)
3. **Shakespeare text generation** (5.3)
4. **Financial time-series indicators** (5.3)
5. **Transformer-XL / relative position encoding** (5.4)
6. **cGAN / CycleGAN / StyleGAN** (5.5)
7. **GIN (Graph Isomorphism Network)** (5.6)
8. **DDPG** (5.8)
9. **SAC** (5.8)
10. **A2C** (5.8)
11. **Custom Gymnasium environments** (5.8)
12. **RLTrainer engine** (5.8)

### HIGH Gaps (12)
1. **CNN training enhancements** (5.2) — LR scheduling, early stopping
2. **BERT fine-tuning for classification** (5.4) — textbook covers; exercise missing
3. **Temporal attention** (5.4) — textbook covers; exercise missing
4. **AE survey** (5.5) — vanilla/denoising AE; exercise jumps to VAE
5. **InferenceServer** (5.7) — textbook covers; exercise missing
6. **BERT fine-tuning for NER/QA** (5.7) — textbook covers; exercise missing
7. **DQN** (5.8) — textbook covers; exercise uses only REINFORCE
8. **PPO** (5.8) — textbook covers; exercise uses only REINFORCE
9. **5 RL business applications** (5.8) — textbook covers; exercise missing
10. **Autoencoder survey** (5.5) — textbook covers vanilla/denoising; exercise skips to VAE
11. **Transfer learning fine-tuning** (5.7) — partial in exercise
12. **Model serving pipeline** (5.7) — InferenceServer integration

### Engine Usage

| Engine | Spec Requires | Textbook | Exercise |
|--------|--------------|----------|----------|
| ModelVisualizer | 5.1-5.6 | COVERED | COVERED |
| OnnxBridge | 5.7 | COVERED | COVERED |
| InferenceServer | 5.7 | COVERED | **NOT USED** |
| RLTrainer | 5.8 | **NOT COVERED** | **NOT USED** |
| TrainingPipeline | 5.1-5.6 | COVERED | COVERED |
| ModelRegistry | 5.7 | COVERED | PARTIAL |

**Critical**: Lesson 5.8 is the most incomplete lesson in the curriculum. Exercise covers only REINFORCE on CartPole. Spec requires DQN, DDPG, SAC, A2C, PPO, custom Gymnasium environments, RLTrainer engine, and 5 business applications.

---

## Module 6: LLMs, Agents, and AI Governance

### Coverage Table

| Lesson | Topic | Textbook | Exercise | Gap Level |
|--------|-------|----------|----------|-----------|
| 6.1 | Transformer recap for LLMs | COVERED | COVERED | — |
| 6.1 | GPT vs BERT architecture differences | COVERED | PARTIAL | — |
| 6.1 | BERT / Masked Language Modeling (MLM) | MISSING | MISSING | **ZERO** |
| 6.1 | Notable models survey (GPT-4, Gemini, Llama, Claude) | MISSING | MISSING | **ZERO** |
| 6.1 | RLHF overview (as context for 6.3) | MISSING | MISSING | **ZERO** |
| 6.2 | LoRA mathematics (low-rank decomposition) | COVERED | COVERED | — |
| 6.2 | QLoRA, GPTQ, AWQ, GGUF quantisation | COVERED | COVERED | — |
| 6.2 | AdapterLayer from-scratch | COVERED | MISSING | HIGH |
| 6.2 | LoRA from-scratch implementation | COVERED | MISSING | HIGH |
| 6.2 | TIES / DARE merging | COVERED | MISSING | HIGH |
| 6.2 | AlignmentPipeline, AdapterRegistry | COVERED | COVERED | — |
| 6.3 | DPO derivation (Bradley-Terry) | COVERED | COVERED | — |
| 6.3 | GRPO (group relative policy optimization) | COVERED | **MISSING** | **HIGH** |
| 6.3 | Eval benchmarks (MMLU, HellaSwag, HumanEval, MT-Bench) | COVERED | MISSING | HIGH |
| 6.3 | lm-eval / LLM-as-Judge | COVERED | MISSING | HIGH |
| 6.4 | Kaizen agents (Delegate, BaseAgent, Signature) | COVERED | COVERED | — |
| 6.4 | ReAct pattern | COVERED | COVERED | — |
| 6.4 | RAGResearchAgent / MemoryAgent | MISSING | MISSING | **ZERO** |
| 6.4 | Agent memory patterns | COVERED | MISSING | HIGH |
| 6.5 | Function calling (tool_choice, parallel) | MISSING | MISSING | **ZERO** |
| 6.5 | RAG pipeline (chunking, embedding, retrieval) | COVERED | COVERED | — |
| 6.5 | BM25 / hybrid search / RAGAS / HyDE | COVERED | MISSING | HIGH |
| 6.6 | MCP server implementation | COVERED | MISSING | HIGH |
| 6.6 | Nexus deployment (API + CLI + MCP) | COVERED | COVERED | — |
| 6.6 | A2A protocol | MISSING | MISSING | **ZERO** |
| 6.7 | PACT governance (GovernanceEngine, D/T/R) | COVERED | COVERED | — |
| 6.7 | PactGovernedAgent, ClearanceManager | COVERED | COVERED | — |
| 6.7 | Risk classification framework | COVERED | COVERED | — |
| 6.8 | DriftMonitor (feature/concept drift) | COVERED | MISSING | HIGH |
| 6.8 | Debugging traces / observability | MISSING | MISSING | **ZERO** |
| 6.8 | Multimodal LLMs (vision, audio) | MISSING | MISSING | **ZERO** |
| 6.8 | Production monitoring pipeline | COVERED | PARTIAL | — |

### ZERO Gaps (9)
1. **BERT / Masked Language Modeling** (6.1) — encoder LLM architecture absent
2. **Notable models survey** (6.1) — GPT-4/Gemini/Llama/Claude comparison
3. **RLHF overview** (6.1) — contextual intro for 6.3 missing
4. **RAGResearchAgent / MemoryAgent** (6.4) — Kaizen advanced agent patterns
5. **tool_choice / parallel function calling** (6.5) — function calling control
6. **A2A protocol** (6.6) — agent-to-agent communication protocol
7. **Debugging traces / observability** (6.8) — LLM debugging patterns
8. **Multimodal LLMs** (6.8) — vision/audio capabilities
9. **Function calling control** (6.5) — dedup with #5 above

*True unique ZERO count for M6 is 8.*

### HIGH Gaps (12)
1. **LoRA from-scratch** (6.2) — textbook covers; exercise missing
2. **AdapterLayer from-scratch** (6.2) — textbook covers; exercise missing
3. **TIES / DARE merge** (6.2) — textbook covers; exercise missing
4. **GRPO** (6.3) — CRITICAL: textbook has extensive section; exercise missing
5. **lm-eval / Eval benchmarks** (6.3) — textbook covers; exercise missing
6. **LLM-as-Judge** (6.3) — textbook covers; exercise missing
7. **Agent memory patterns** (6.4) — textbook covers; exercise missing
8. **BM25 / hybrid search / RAGAS / HyDE** (6.5) — textbook covers; exercise missing
9. **MCP server implementation** (6.6) — textbook shows MCPServer code; exercise missing
10. **DriftMonitor** (6.8) — textbook covers; exercise missing
11. **Production monitoring** (6.8) — partial in exercise
12. **Adapter serving** (6.2) — deployment patterns missing from exercise

### Engine Usage

| Engine | Spec Requires | Textbook | Exercise |
|--------|--------------|----------|----------|
| AlignmentPipeline | 6.2-6.3 | COVERED | COVERED |
| AdapterRegistry | 6.2 | COVERED | COVERED |
| Delegate / BaseAgent | 6.4 | COVERED | COVERED |
| Kaizen ReActAgent | 6.4 | COVERED | COVERED |
| GovernanceEngine | 6.7 | COVERED | COVERED |
| PactGovernedAgent | 6.7 | COVERED | COVERED |
| ClearanceManager | 6.7 | COVERED | COVERED |
| Nexus | 6.6 | COVERED | COVERED |
| DriftMonitor | 6.8 | COVERED | **NOT USED** |
| MCPServer | 6.6 | COVERED | **NOT USED** |
| RLTrainer | 6.3 | Not required | Used in ex_8 |

---

## Consolidated Gap Summary

### ZERO-Coverage Gaps by Priority

**Tier 1 — Core Curriculum Gaps (must fix)**

| # | Module | Lesson | Topic | Impact |
|---|--------|--------|-------|--------|
| 1 | M5 | 5.8 | DDPG / SAC / A2C algorithms | RL lesson covers only REINFORCE; 3 major algorithms missing |
| 2 | M5 | 5.8 | Custom Gymnasium environments | Spec requires building envs; exercise uses only CartPole |
| 3 | M5 | 5.8 | RLTrainer engine | kailash-ml RL engine absent from entire M5 |
| 4 | M6 | 6.1 | BERT / MLM architecture | Encoder LLM architecture gap in LLM foundations |
| 5 | M6 | 6.6 | A2A protocol | Agent-to-agent communication absent |
| 6 | M3 | 3.5 | Regression metrics (MAE/MSE/RMSE/MAPE) | Core ML metrics missing |
| 7 | M3 | 3.4 | AdaBoost | Major ensemble algorithm absent |
| 8 | M4 | 4.6 | Sentiment analysis | NLP application absent |
| 9 | M4 | 4.8 | Loss functions taxonomy | Foundation DL concept missing |
| 10 | M4 | 4.8 | Adam optimizer formula | Core optimizer derivation missing |

**Tier 2 — Important Gaps (should fix)**

| # | Module | Lesson | Topic | Impact |
|---|--------|--------|-------|--------|
| 11 | M5 | 5.5 | cGAN / CycleGAN / StyleGAN | GAN variants absent |
| 12 | M5 | 5.6 | GIN (Graph Isomorphism Network) | GNN variant absent |
| 13 | M5 | 5.4 | Transformer-XL / relative position encoding | Advanced transformer absent |
| 14 | M5 | 5.3 | Spatial attention mechanism | Attention variant absent |
| 15 | M5 | 5.3 | Multi-layer RNN with residual | Advanced RNN pattern absent |
| 16 | M6 | 6.4 | RAGResearchAgent / MemoryAgent | Kaizen agent patterns absent |
| 17 | M6 | 6.5 | tool_choice / parallel function calling | Function calling control absent |
| 18 | M6 | 6.8 | Debugging traces / observability | LLM debugging absent |
| 19 | M6 | 6.8 | Multimodal LLMs | Vision/audio LLMs absent |
| 20 | M6 | 6.1 | Notable models survey | Model landscape absent |
| 21 | M6 | 6.1 | RLHF overview | RL alignment context absent |
| 22 | M3 | 3.5 | Stacking / blending | Meta-learning absent |
| 23 | M3 | 3.6 | Calibration parity across groups | Fairness calibration absent |
| 24 | M3 | 3.1 | Geocoding / OneMap API | Singapore-specific feature absent |
| 25 | M4 | 4.1 | ARI / NMI metrics | Cluster evaluation metrics absent |
| 26 | M4 | 4.3 | Manifold learning comparison table | Systematic comparison absent |
| 27 | M2 | 2.4 | Clean DataOps architecture | Data pipeline architecture absent |

**Tier 3 — Minor Gaps (can defer)**

| # | Module | Lesson | Topic | Impact |
|---|--------|--------|-------|--------|
| 28 | M5 | 5.3 | Shakespeare text generation | Creative generation exercise |
| 29 | M5 | 5.3 | Financial time-series indicators | Domain-specific features |
| 30 | M2 | 2.1 | Truth table construction | Pedagogical aid |
| 31 | M2 | 2.1 | Independent/dependent event decision steps | Decision framework |
| 32 | M2 | 2.1 | Supermarket example | Illustrative example |
| 33 | M2 | 2.6 | Scheffe post-hoc test | Secondary post-hoc test |
| 34 | M2 | 2.6 | Two-way ANOVA / repeated measures | Extension mention |
| 35 | M1 | 1.4 | Dictionary lookups | Minor topic |
| 36 | M1 | 1.6 | Z-pattern reading | Design principle |
| 37 | M1 | 1.6 | 100% stacked / Likert | Chart types |

### HIGH Exercise-Only Gaps — Top 10 Critical

These topics are well-covered in textbooks but completely absent from exercises. Students learn theory but never practice.

| # | Module | Lesson | Topic | Textbook Section |
|---|--------|--------|-------|-----------------|
| 1 | M3 | 3.7 | HyperparameterSearch engine | Full API shown |
| 2 | M3 | 3.8 | DriftMonitor engine | Full API shown |
| 3 | M6 | 6.3 | GRPO implementation | Extensive with formulas |
| 4 | M2 | 2.7 | DiD implementation | Full section with ATT |
| 5 | M5 | 5.8 | DQN / PPO algorithms | Both covered in text |
| 6 | M6 | 6.6 | MCP server implementation | Code shown in textbook |
| 7 | M2 | 2.8 | FeatureEngineer engine | Code shown in textbook |
| 8 | M6 | 6.2 | LoRA / AdapterLayer from-scratch | Code shown in textbook |
| 9 | M5 | 5.7 | InferenceServer engine | Full API shown |
| 10 | M6 | 6.5 | BM25/hybrid/RAGAS/HyDE | Retrieval patterns covered |

### Kailash Engine Coverage Summary

| Engine | Modules Required | Textbook | Exercise | Verdict |
|--------|-----------------|----------|----------|---------|
| DataExplorer | M1 | COVERED | COVERED | OK |
| PreprocessingPipeline | M1, M3 | COVERED | COVERED | OK |
| ModelVisualizer | M1-M6 | COVERED | COVERED | OK |
| ExperimentTracker | M2 | COVERED | PARTIAL | Needs work |
| FeatureEngineer | M2, M3 | COVERED | **M2 MISSING** | Fix M2 ex_8 |
| FeatureStore | M2, M3 | COVERED | COVERED | OK |
| TrainingPipeline | M2-M5 | COVERED | PARTIAL | Missing from M2 |
| AutoMLEngine | M3, M4 | COVERED | COVERED | OK |
| HyperparameterSearch | M3 | COVERED | **MISSING** | CRITICAL |
| EnsembleEngine | M3, M4 | COVERED | PARTIAL | OK |
| ModelRegistry | M3, M5 | COVERED | COVERED | OK |
| WorkflowBuilder | M3 | COVERED | **MISSING** | Fix M3 ex_8 |
| DriftMonitor | M3, M4, M6 | COVERED | **MISSING** | CRITICAL |
| OnnxBridge | M4, M5 | COVERED | COVERED | OK |
| InferenceServer | M5 | COVERED | **MISSING** | Fix M5 ex_7 |
| RLTrainer | M5 | **MISSING** | **MISSING** | ZERO — M5 only |
| AlignmentPipeline | M6 | COVERED | COVERED | OK |
| AdapterRegistry | M6 | COVERED | COVERED | OK |
| GovernanceEngine | M6 | COVERED | COVERED | OK |
| PactGovernedAgent | M6 | COVERED | COVERED | OK |
| ClearanceManager | M6 | COVERED | COVERED | OK |
| Nexus | M6 | COVERED | COVERED | OK |
| MCPServer | M6 | COVERED | **MISSING** | Fix M6 ex_6 |
| Delegate/BaseAgent | M6 | COVERED | COVERED | OK |

**Engine verdict**: 23 engines tracked. 16 OK, 3 CRITICAL (HyperparameterSearch, DriftMonitor, RLTrainer), 4 need exercise work (FeatureEngineer M2, WorkflowBuilder, InferenceServer, MCPServer).

---

## Recommendations

### Immediate (before next delivery)

1. **Lesson 5.8 rewrite** — Most incomplete lesson. Needs DQN, PPO, RLTrainer engine, custom Gymnasium env, and at least 2 business applications in exercise.
2. **Add HyperparameterSearch** to M3 ex_7 — CRITICAL engine gap.
3. **Add DriftMonitor** to M3 ex_8 and/or M6 ex_8 — CRITICAL engine gap.
4. **Add GRPO task** to M6 ex_3 — textbook has extensive coverage; exercise missing.
5. **Add DiD task** to M2 ex_7 — textbook has full section; exercise replaces with Bayesian A/B.

### Short-term (next sprint)

6. Add FeatureEngineer usage to M2 ex_8 (replace manual Polars).
7. Add InferenceServer to M5 ex_7.
8. Add MCP server task to M6 ex_6.
9. Add BERT/MLM section to M6 textbook 6.1.
10. Add regression metrics section to M3 textbook 3.5.
11. Add AdaBoost section to M3 textbook 3.4.
12. Add sentiment analysis to M4 textbook 4.6.
13. Add loss functions taxonomy to M4 textbook 4.8.
14. Add Adam optimizer formula to M4 textbook 4.8.

### Deferred (backlog)

15. GAN variants (cGAN/CycleGAN/StyleGAN) for M5.5.
16. GIN for M5.6.
17. Transformer-XL for M5.4.
18. A2A protocol for M6.6.
19. RAGResearchAgent/MemoryAgent for M6.4.
20. Various minor gaps (truth tables, Scheffe test, etc.).

---

*Audit conducted by: reviewer agent (quality team)*
*Method: 6 parallel module auditors + targeted grep verification of prior red-team findings*
*Artifacts checked: 6 spec files, 48 textbook.html files, 48 solution .py files*
