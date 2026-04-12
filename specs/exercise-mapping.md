# Exercise-to-Lesson Mapping (v2 Curriculum)

Maps every existing exercise to its v2 lesson. Guides the reorganization from the mechanical ASCENT merge into the v2 module structure.

**Legend**: ✓ = in correct module, → = needs moving, NEW = gap requiring net-new exercise

---

## Module 1: Data Pipelines and Visualisation ✓ COMPLETE

| Lesson | v2 Title | Source File | Status |
|--------|----------|-------------|--------|
| 1.1 | Your First Data Exploration | mlfp01/ex_1 | ✓ |
| 1.2 | Filtering and Transforming Data | mlfp01/ex_2 | ✓ |
| 1.3 | Functions and Aggregation | mlfp01/ex_3 | ✓ |
| 1.4 | Joins and Multi-Table Data | mlfp01/ex_4 | ✓ |
| 1.5 | Window Functions and Trends | mlfp01/ex_5 | ✓ |
| 1.6 | Data Visualisation | mlfp01/ex_6 | ✓ |
| 1.7 | Automated Data Profiling | mlfp01/ex_7 | ✓ |
| 1.8 | Data Pipelines and End-to-End Project | mlfp01/ex_8 | ✓ |

---

## Module 2: Statistical Mastery — 3 GAPS

| Lesson | v2 Title | Source File | Status |
|--------|----------|-------------|--------|
| 2.1 | Probability and Bayesian Thinking | mlfp02/ex_1 | ✓ |
| 2.2 | Parameter Estimation and Inference | mlfp02/ex_2 | ✓ |
| 2.3 | Bootstrapping and Hypothesis Testing | mlfp02/ex_3 | ✓ (ex_4 supplementary) |
| 2.4 | A/B Testing and Experiment Design | — | **NEW** |
| 2.5 | Linear Regression | — | **NEW** |
| 2.6 | Logistic Regression and Classification | — | **NEW** |
| 2.7 | CUPED and Causal Inference | mlfp02/ex_5 | ✓ (ex_6 supplementary) |
| 2.8 | Capstone — Statistical Analysis Project | mlfp02/ex_8 | ✓ |

**Exercises leaving M2** (→ M3):
- ex_7 (Feature Engineering) → M3 ex_1
- ex_9 (Bias-Variance and Regularisation) → M3 ex_2
- ex_10 (Gradient Boosting Deep Dive) → M3 ex_4
- ex_11 (Class Imbalance and Calibration) → M3 ex_5
- ex_12 (SHAP, LIME, and Fairness) → M3 ex_6
- ex_13 (Workflow Orchestration and Custom Nodes) → M3 ex_7
- ex_16 (Production Pipeline Project) → M3 ex_8

**Supplementary** (not assigned, available as extra exercises):
- ex_4 (Bootstrap and Resampling) — overlaps 2.3
- ex_6 (Sequential Testing and Causal Inference) — overlaps 2.7
- ex_14 (DataFlow and Persistence) — partially overlaps 3.8
- ex_15 (Model Registry and Hyperparameter Search) — partially overlaps 3.7

---

## Module 3: Supervised ML — 1 GAP

| Lesson | v2 Title | Source File | Status |
|--------|----------|-------------|--------|
| 3.1 | Feature Engineering and Feature Selection | mlfp02/ex_7 | → move |
| 3.2 | Bias-Variance, Regularisation, and CV | mlfp02/ex_9 | → move |
| 3.3 | The Complete Supervised Model Zoo | — | **NEW** (SVM, KNN, NB, Trees, RF) |
| 3.4 | Gradient Boosting Deep Dive | mlfp02/ex_10 | → move |
| 3.5 | Model Evaluation, Imbalance, and Calibration | mlfp02/ex_11 | → move |
| 3.6 | Interpretability and Fairness | mlfp02/ex_12 | → move |
| 3.7 | Workflow Orchestration and Hyperparameter Search | mlfp02/ex_13 | → move |
| 3.8 | Production Pipeline — DataFlow, Drift, Deployment | mlfp02/ex_16 | → move |

---

## Module 4: Unsupervised ML and Advanced Techniques — 2 GAPS

| Lesson | v2 Title | Source File | Status |
|--------|----------|-------------|--------|
| 4.1 | Clustering | mlfp03/ex_1 | → move |
| 4.2 | EM Algorithm and Gaussian Mixture Models | mlfp03/ex_2 | → move |
| 4.3 | Dimensionality Reduction | mlfp03/ex_3 | → move |
| 4.4 | Anomaly Detection and Ensembles | mlfp03/ex_4 | → move |
| 4.5 | Association Rules and Market Basket | — | **NEW** |
| 4.6 | NLP — Text to Topics | mlfp03/ex_5 | → move |
| 4.7 | Recommender Systems and Collaborative Filtering | — | **NEW** |
| 4.8 | DL Foundations — Neural Networks and Training | mlfp03/ex_7 | → move |

**Supplementary from current mlfp03**:
- ex_6 (Drift Monitoring) — overlaps 3.8
- ex_8 (M4 Capstone) — could adapt for 4.8

**Exercises leaving mlfp03** (→ M6):
- ex_9 (LLM Fundamentals) → supplementary
- ex_10-16 (agents, RAG, MCP, Nexus) → duplicated in mlfp06

---

## Module 5: Deep Learning — 3 GAPS

| Lesson | v2 Title | Source File | Status |
|--------|----------|-------------|--------|
| 5.1 | Autoencoders | — | **NEW** |
| 5.2 | CNNs and Computer Vision | mlfp05/ex_7 | ✓ |
| 5.3 | RNNs and Sequence Models | mlfp05/ex_12 | ✓ (renumber) |
| 5.4 | Transformers | mlfp05/ex_14 | ✓ (renumber) |
| 5.5 | Generative Models — GANs and Diffusion | — | **NEW** |
| 5.6 | Graph Neural Networks | — | **NEW** |
| 5.7 | Transfer Learning | mlfp05/ex_15 | ✓ (renumber) |
| 5.8 | Reinforcement Learning | mlfp04/ex_3 | → move from mlfp04 |

**DL Toolkit exercises** (current mlfp05/ex_1-6): These are M4.8 content (DL foundations, backprop, activations, optimizers). They supplement mlfp03/ex_7 which becomes M4 ex_8. Could be merged into M4.8 or kept as supplementary.

**NLP exercises** (current mlfp05/ex_9-11): Text preprocessing, BoW/TF-IDF, word embeddings — supplement M4.6.

**Supplementary**:
- ex_1 (Linear Regression as NN) — M4.8
- ex_2 (Hidden Layers/XOR) — M4.8
- ex_3 (Activation Functions) — M4.8
- ex_4 (Loss Functions) — M4.8
- ex_5 (Backpropagation) — M4.8
- ex_6 (Optimizers) — M4.8
- ex_8 (DL Capstone) — M5 capstone alternative
- ex_9 (Text Preprocessing) — M4.6
- ex_10 (BoW/TF-IDF) — M4.6
- ex_11 (Word Embeddings) — M4.6
- ex_13 (Attention Mechanisms) — M5.4 supplementary
- ex_16 (NLP Capstone) — M5 capstone alternative

---

## Module 6: LLMs and Agentic Workflows — 0 GAPS (dedup needed)

| Lesson | v2 Title | Source File | Status |
|--------|----------|-------------|--------|
| 6.1 | LLM Fundamentals, Prompt Eng, Structured Output | mlfp06/ex_2 | ✓ (renumber) |
| 6.2 | LLM Fine-tuning — LoRA and Adapters | mlfp06/ex_9 | ✓ (renumber) |
| 6.3 | Preference Alignment — DPO and GRPO | mlfp06/ex_10 | ✓ (renumber) |
| 6.4 | RAG Systems | mlfp06/ex_3 | ✓ (renumber) |
| 6.5 | AI Agents — ReAct, Tool Use | mlfp06/ex_5 | ✓ (renumber) |
| 6.6 | Multi-Agent Orchestration and MCP | mlfp06/ex_6 | ✓ (renumber) |
| 6.7 | AI Governance Engineering | mlfp06/ex_14 | ✓ (renumber) |
| 6.8 | Capstone — Full Production Platform | mlfp06/ex_16 | ✓ (renumber) |

**Duplicates to remove** (content covered by selected exercises):
- ex_1 (LLM Architecture) — overlaps 6.1
- ex_4 (Advanced RAG) — supplementary to 6.4
- ex_7 (MCP Integration) — supplementary to 6.6
- ex_8 (Agent Deployment Capstone) — overlaps 6.8
- ex_11 (RL Fundamentals) → 5.8 content, duplicated with mlfp04/ex_3
- ex_12 (PPO Training) → 5.8 content
- ex_13 (Model Merging) — 6.2 supplementary
- ex_15 (Governed Agents) — 6.7 supplementary

**Exercises from mlfp04 that belong in M6** (duplicated — use mlfp06 versions):
- ex_1 (SFT) → 6.2 duplicate
- ex_2 (DPO) → 6.3 duplicate
- ex_4 (Model Merging) → 6.2 duplicate
- ex_5-7 (PACT/Governance) → 6.7 duplicates
- ex_8 (Capstone) → 6.8 duplicate

---

## Summary

| Action | Count |
|--------|-------|
| Exercises already correct (M1) | 8 |
| Exercises to move between modules | ~20 |
| Exercises to renumber in place | ~8 |
| **Net-new exercises to write** | **9** |
| Supplementary exercises (archive) | ~25 |
| Duplicate exercises (remove) | ~15 |

### 9 Net-New Exercises

1. **2.4** — A/B Testing and Experiment Design
2. **2.5** — Linear Regression
3. **2.6** — Logistic Regression and Classification Foundations
4. **3.3** — The Complete Supervised Model Zoo (SVM, KNN, NB, Trees, RF)
5. **4.5** — Association Rules and Market Basket Analysis
6. **4.7** — Recommender Systems and Collaborative Filtering
7. **5.1** — Autoencoders (vanilla, denoising, VAE, convolutional)
8. **5.5** — Generative Models — GANs and Diffusion
9. **5.6** — Graph Neural Networks
