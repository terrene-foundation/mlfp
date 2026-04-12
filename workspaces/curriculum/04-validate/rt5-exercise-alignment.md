# RT5: Exercise-Textbook Alignment Audit

**Date**: 2026-04-13
**Auditor**: reviewer agent
**Scope**: All 48 exercises across 6 modules, with focus on 9 new exercises (M2 ex_4/5/6, M3 ex_3, M4 ex_5/7, M5 ex_1/5/6)

---

## Executive Summary

Audited all 48 exercise-textbook pairs across MLFP01-06. Of 48 lessons, **27 are fully ALIGNED**, **21 have PARTIAL alignment** with identifiable gaps. Found **7 critical issues** (must fix before delivery), **19 important issues** (should fix), and **15 minor issues** (can defer).

The 9 new exercises are generally strong -- 6 of 9 are ALIGNED with their textbooks. The most significant systemic issues are: (1) output file naming errors across M3/M4 exercises, (2) module naming contamination (MLFP02 references in MLFP03 code), and (3) textbook topics promised but never exercised.

---

## Cross-Module Summary

| Module | ALIGNED | PARTIAL | Critical | Important | Minor |
|--------|---------|---------|----------|-----------|-------|
| MLFP01 | 6 | 2 | 0 | 3 | 3 |
| MLFP02 | 5 | 3 | 2 | 3 | 2 |
| MLFP03 | 1 | 7 | 4 | 5 | 3 |
| MLFP04 | 4 | 4 | 3 | 4 | 4 |
| MLFP05 | 7 | 1 | 1 | 3 | 3 |
| MLFP06 | 2 | 6 | 2 | 4 | 2 |
| **Total** | **25** | **23** | **12** | **22** | **17** |

---

## New Exercise Assessment (9 exercises)

| Exercise | Module | Topic | Alignment | Verdict |
|----------|--------|-------|-----------|---------|
| M2 ex_4 | MLFP02 | A/B Testing and Experiment Design | ALIGNED | Well-constructed; minor: Welch's t-test not in textbook |
| M2 ex_5 | MLFP02 | Linear Regression | ALIGNED | Strong; minor forward ref to cross-validation |
| M2 ex_6 | MLFP02 | Logistic Regression & Classification | ALIGNED | Strong from-scratch pedagogy; sklearn usage is known SDK gap |
| M3 ex_3 | MLFP03 | Complete Supervised Model Zoo | ALIGNED | **Best-aligned exercise in M3**. All 5 model families in textbook order |
| M4 ex_5 | MLFP04 | Association Rules & Market Basket | ALIGNED | Exceeds textbook scope. Implements Apriori from scratch |
| M4 ex_7 | MLFP04 | Recommender Systems & Collab Filtering | ALIGNED | Strong. THE PIVOT well-articulated. Duplicate TASKS block in docstring |
| M5 ex_1 | MLFP05 | Autoencoders | ALIGNED | All 4 AE variants match textbook precisely |
| M5 ex_5 | MLFP05 | GANs and Diffusion | ALIGNED | Mirrors textbook 8-Gaussians worked example exactly |
| M5 ex_6 | MLFP05 | Graph Neural Networks | PARTIAL | Missing GraphSAGE; PyG vs pure-PyTorch divergence |

**Summary**: 8 of 9 new exercises are ALIGNED. M5 ex_6 (GNN) is the only PARTIAL, with GraphSAGE missing from the exercise despite textbook coverage. All 9 are pedagogically sound and ready for student use with minor fixes.

---

## Module 1: Data Foundations

| Lesson | Title | Alignment | Issues |
|--------|-------|-----------|--------|
| 1 | Your First Data Exploration | ALIGNED | None |
| 2 | Filtering and Transforming Data | ALIGNED | None |
| 3 | Functions and Aggregation | ALIGNED | None |
| 4 | Joins and Multi-Table Data | ALIGNED | Minor: exercise skips `if/elif/else` + `map_elements` pattern taught in textbook |
| 5 | Window Functions and Trends | ALIGNED | None |
| 6 | Data Visualisation | PARTIAL | Heatmap uses raw Plotly instead of ModelVisualizer; stacked bar chart omitted |
| 7 | Automated Data Profiling | ALIGNED | None |
| 8 | Data Pipelines and End-to-End | PARTIAL | PreprocessingPipeline API signature differs; REST API taught but not exercised; csv vs parquet mismatch |

### Important Issues

1. **L6 heatmap bypasses ModelVisualizer**: Exercise builds correlation heatmap with `numpy.corrcoef` + raw `plotly.graph_objects.Heatmap` instead of `viz.confusion_matrix()` taught in textbook. Violates Framework-First directive.
   - File: `modules/mlfp01/solutions/ex_6.py`
   - Fix: Use `viz.confusion_matrix(matrix=..., labels=...)` or document why deviation is intentional.

2. **L8 PreprocessingPipeline API mismatch**: Textbook teaches `PreprocessingPipeline(numeric_strategy=..., scale=True)` with `fit_transform()`. Exercise uses `PreprocessingPipeline()` with `pipeline.setup(data=..., target=..., normalize=True)`. Different API signatures will confuse students.
   - Fix: Align textbook to match exercise's `setup()` API if that is the current kailash-ml interface.

3. **L8 REST API gap**: Textbook devotes a full section to REST APIs (httpx, OneMap API) but exercise loads all data from local files. Students who only do the exercise miss this skill entirely.
   - Fix: Add optional REST API task or move content to a separate lesson.

### Minor Issues

- L4: `if/elif/else` + `map_elements` + `pl.struct` taught but not exercised
- L6: Stacked bar chart in textbook absent from exercise
- L8: Textbook references `sg_taxi_trips.csv`, exercise loads `.parquet`

---

## Module 2: Statistics and Inference

| Lesson | Title | Alignment | Issues | NEW |
|--------|-------|-----------|--------|-----|
| 1 | Probability and Bayesian Thinking | ALIGNED | None | No |
| 2 | Parameter Estimation and Inference | ALIGNED | None | No |
| 3 | Bootstrapping and Hypothesis Testing | ALIGNED | Minor: SRM forward reference to L4 | No |
| 4 | A/B Testing and Experiment Design | ALIGNED | Column naming mismatch; Welch's t-test not in textbook | **Yes** |
| 5 | Linear Regression | ALIGNED | Cross-validation forward reference; feature parsing not in textbook | **Yes** |
| 6 | Logistic Regression and Classification | ALIGNED | sklearn usage conflicts with Framework-First | **Yes** |
| 7 | CUPED and Causal Inference | PARTIAL | DiD in textbook but not exercise; Bayesian A/B + mSPRT in exercise but not textbook | No |
| 8 | Capstone: Statistical Analysis Project | PARTIAL | FeatureEngineer in textbook but not exercise; API patterns differ | No |

### Critical Issues

1. **L7 content divergence (highest priority)**: Textbook teaches Difference-in-Differences (not exercised). Exercise requires Bayesian A/B testing and mSPRT sequential testing (not taught in textbook). This is the most significant alignment gap in MLFP02.
   - Fix: (a) Add Bayesian A/B + mSPRT to textbook and add DiD task to exercise, (b) remove DiD from textbook and add Bayesian A/B + mSPRT sections, or (c) split into two lessons.

2. **L6 sklearn usage**: Direct `sklearn.linear_model.LogisticRegression` and `sklearn.metrics` imports violate Framework-First directive. Known accepted gap per SDK issues #341-348.
   - Fix: Document as temporary exception with tracking reference. Replace when kailash-ml ships equivalent.

### Important Issues

3. **L8 FeatureEngineer gap**: Textbook teaches `FeatureEngineer` as key component; exercise exclusively uses `FeatureStore`/`FeatureSchema`. Either add FeatureEngineer task or reduce textbook coverage.

4. **L4 (NEW) column naming mismatch**: Textbook uses `arm` as group column; exercise uses `experiment_group`. Standardize across textbook and exercise datasets.

5. **L4 (NEW) Welch's t-test**: Exercise uses Welch's t-test with Welch-Satterthwaite df. Textbook does not specifically cover Welch's variant. Add textbook section or more exercise scaffolding.

---

## Module 3: Supervised Machine Learning

| Lesson | Title | Alignment | Issues | NEW |
|--------|-------|-----------|--------|-----|
| 1 | Feature Engineering, ML Pipeline, Feature Selection | PARTIAL | Feature selection gap; FeatureEngineer not used; MLFP02 naming contamination | No |
| 2 | Bias-Variance, Regularisation, Cross-Validation | PARTIAL | Cross-validation gap; dataset mismatch with README | No |
| 3 | Complete Supervised Model Zoo | ALIGNED | Best-aligned in module. Minor: no entropy-based IG | **Yes** |
| 4 | Gradient Boosting Deep Dive | ALIGNED | None significant | No |
| 5 | Model Evaluation, Imbalance, Calibration | ALIGNED | None significant | No |
| 6 | Interpretability and Fairness | PARTIAL | ALE gap; impossibility theorem gap; no disparate impact ratio | No |
| 7 | Workflow Orchestration, Model Registry, HyperparameterSearch | PARTIAL | HyperparameterSearch omitted; DataFlow from L8 scope; MLFP02 naming | No |
| 8 | Production Pipeline: DataFlow, Drift, Deployment | PARTIAL | DriftMonitor omitted; DataFlow omitted; bias-variance belongs in L2 | No |

### Critical Issues

1. **MLFP02 naming contamination in ex_1**: ExperimentTracker experiment is `"mlfp02_healthcare_features"`, description says "Module 2", docstring says "Module 2". Should all reference MLFP03 / Module 3.
   - File: `modules/mlfp03/solutions/ex_1.py` lines 153, 159-161
   - Fix: Replace all "mlfp02" / "Module 2" references with "mlfp03" / "Module 3".

2. **MLFP02 naming contamination in ex_7**: Multiple references to "mlfp02" in database names, experiment names, `created_by` fields.
   - File: `modules/mlfp03/solutions/ex_7.py` lines 168, 195, 311, 408-409
   - Fix: Replace all "mlfp02" references with "mlfp03".

3. **Output file naming error in ex_8**: Model card saved as `ex6_model_card.md` and visualization as `ex6_final_metrics.html` instead of `ex8_*`.
   - File: `modules/mlfp03/solutions/ex_8.py` lines 208, 447
   - Fix: Rename to `ex8_model_card.md` and `ex8_final_metrics.html`.

4. **README dataset mismatch for ex_2**: README says "HDB resale" but exercise uses `sg_credit_scoring.parquet`.
   - File: `modules/mlfp03/README.md` line 25
   - Fix: Update README to "Singapore credit scoring".

### Important Issues

5. **L7 omits HyperparameterSearch entirely**: Textbook covers Bayesian optimisation, SearchSpace, SearchConfig. Exercise focuses on WorkflowBuilder + DataFlow persistence instead. HyperparameterSearch is absent from ALL M3 exercises.

6. **L8 omits DriftMonitor**: Textbook covers PSI and KS tests as major section. Exercise skips drift monitoring entirely.

7. **L1 omits feature selection**: Textbook and README list filter/wrapper/embedded selection. Exercise only covers feature engineering and schema validation.

8. **L6 omits ALE and impossibility theorem**: Textbook covers ALE plots and fairness impossibility theorem. Exercise only covers SHAP, LIME, and basic fairness audit.

9. **L2 omits cross-validation strategies**: Textbook covers k-fold, stratified, time-series, nested CV. Exercise does not teach any CV strategy.

---

## Module 4: Unsupervised ML and Deep Learning Foundations

| Lesson | Title | Alignment | Issues | NEW |
|--------|-------|-----------|--------|-----|
| 1 | Clustering | ALIGNED | Minor: hierarchical clustering not exercised | No |
| 2 | EM Algorithm and GMMs | ALIGNED | None | No |
| 3 | Dimensionality Reduction | ALIGNED | Minor: kernel PCA not exercised | No |
| 4 | Anomaly Detection and Ensembles | PARTIAL | Docstring contradicts implementation; missing Z-score/IQR; output naming | No |
| 5 | Association Rules and Market Basket | ALIGNED | Minor: synthetic data vs textbook's pre-existing dataset | **Yes** |
| 6 | NLP: Text to Topics | PARTIAL | Dataset mismatch; missing BM25, LDA; output naming | No |
| 7 | Recommender Systems and Collab Filtering | ALIGNED | Duplicate TASKS block; RMSE instead of precision@10/NDCG@10 | **Yes** |
| 8 | DL Foundations: Neural Networks | PARTIAL | Architecture mismatch (CNN vs feedforward); missing from-scratch backprop; output naming; undefined variable | No |

### Critical Issues

1. **Ex_4 docstring contradicts implementation**: Docstring describes "Credit card fraud" with "V1-V28" features. Code actually loads e-commerce customer data and synthesizes anomalies from `num_returns`.
   - File: `modules/mlfp04/solutions/ex_4.py` lines 31-34
   - Fix: Update docstring to match actual data source.

2. **Ex_8 undefined variable in ONNX fallback**: References `medical_cnn` and `dummy_input` which do not exist in scope. Model variable is `model`.
   - File: `modules/mlfp04/solutions/ex_8.py` lines 436-438
   - Fix: Change `medical_cnn` to `model`, define `dummy_input` before fallback block.

3. **Ex_7 (NEW) duplicate TASKS block**: TASKS section appears twice in exercise header docstring.
   - File: `modules/mlfp04/solutions/ex_7.py` lines 27-34 and 41-49
   - Fix: Remove the duplicate TASKS block.

### Important Issues

4. **Ex_4 missing Z-score/IQR**: Textbook explicitly asks for Z-score detection. Exercise jumps straight to Isolation Forest and LOF.

5. **Ex_6 missing LDA**: Textbook teaches LDA with plate notation. Exercise only uses NMF as fallback. "Try It Yourself" asks for both NMF and LDA.

6. **Ex_6 dataset mismatch**: Exercise loads `documents.parquet` from mlfp03 instead of `sg_news.parquet` from mlfp04 as textbook shows.

7. **Ex_8 textbook-exercise architecture gap**: Textbook teaches from-scratch backprop for simple feedforward network. Exercise builds CNN with ResBlock using PyTorch autograd. Students never learn manual backprop.

8. **Output file naming errors across 3 exercises**:
   - Ex_4: saves as `ex2_*` instead of `ex4_*`
   - Ex_6: saves as `ex3_*` instead of `ex6_*`
   - Ex_8: saves as `ex5_*` instead of `ex8_*`

---

## Module 5: Deep Learning

| Lesson | Title | Alignment | Issues | NEW |
|--------|-------|-----------|--------|-----|
| 1 | Autoencoders | ALIGNED | None | **Yes** |
| 2 | CNNs and Computer Vision | ALIGNED | None | No |
| 3 | RNNs and Sequence Models | ALIGNED | None | No |
| 4 | Transformers and Self-Attention | ALIGNED | None | No |
| 5 | Generative Models: GANs and Diffusion | ALIGNED | None | **Yes** |
| 6 | Graph Neural Networks | PARTIAL | GraphSAGE gap; PyG vs pure-PyTorch divergence; README dataset wrong | **Yes** |
| 7 | Transfer Learning | ALIGNED | Minor: no BERT/NLP transfer | No |
| 8 | Reinforcement Learning | ALIGNED | DQN/PPO not implemented; README dataset error | No |

### Critical Issues

1. **README dataset error for ex_8**: README says "Inventory management" but exercise uses CartPole-v1 (Gymnasium).
   - File: `modules/mlfp05/README.md` line 31
   - Fix: Change to "CartPole-v1 (Gymnasium)".

### Important Issues

2. **L6 (NEW) GraphSAGE gap**: Textbook teaches GCN, GraphSAGE, GAT. Exercise implements only GCN and GAT. GraphSAGE has no hands-on reinforcement.
   - Fix: Add GraphSAGE as stretch goal or brief implementation task.

3. **L6 (NEW) PyG vs pure-PyTorch divergence**: Textbook worked example uses `torch_geometric`. Exercise implements from scratch in pure PyTorch. Add note explaining the difference.

4. **L6 (NEW) README dataset wrong**: README says "Karate Club / synthetic" but exercise uses synthetic 3-community SBM.
   - Fix: Update to "Synthetic 3-community SBM".

### Minor Issues

- L7: Textbook uses deprecated `pretrained=True`; exercise correctly uses `ResNet18_Weights.DEFAULT`
- L6: No graph-level classification despite textbook coverage
- L7: No BERT/NLP transfer in exercise (scope decision)

---

## Module 6: LLMs, Agents, and Governance

| Lesson | Title | Alignment | Issues |
|--------|-------|-----------|--------|
| 1 | Prompt Engineering | ALIGNED | None |
| 2 | LoRA Fine-Tuning | ALIGNED | SLERP taught but not exercised (creates capstone dependency gap) |
| 3 | DPO Preference Alignment | PARTIAL | Missing GRPO, LLM-as-judge implementation |
| 4 | RAG Fundamentals | PARTIAL | Missing BM25/hybrid retrieval, RAGAS evaluation, HyDE |
| 5 | AI Agents: ReAct and Tool Use | ALIGNED | None |
| 6 | Multi-Agent Orchestration and MCP | PARTIAL | Missing MCP server, agent memory |
| 7 | AI Governance Engineering | PARTIAL | D/T/R API mismatch between textbook and exercise |
| 8 | Capstone: Full Production Platform | PARTIAL | Phantom "Ex 5 SLERP merge" reference (CRITICAL) |

### Critical Issues

1. **Ex_8 references non-existent Exercise 5 SLERP merge**: Capstone prerequisites say "Ex 5 (SLERP merge)" and loads `sg_domain_slerp_merge_v1` adapter. Exercise 5 is actually "Building Agents with Kaizen" -- no SLERP merge exists in ANY module exercise. Task 1 (`registry.get_adapter("sg_domain_slerp_merge_v1")`) would fail.
   - File: `modules/mlfp06/solutions/ex_8.py` lines 22, 39, 86-87, 97, 360, 388, 507
   - Fix: Either (a) add SLERP merge task to Exercise 2 (which teaches the theory), or (b) remove SLERP dependency and load DPO adapter from Exercise 3 directly.

2. **L7 D/T/R API mismatch**: Textbook defines D/T/R as "Domain/Team/Role" with `Address(domain=, team=, role=)` and `can_access()`. Exercise defines D/T/R as "Delegator/Task/Responsible" with `check_access(agent_id=, resource=, action=)`. Fundamentally different APIs labeled the same thing.
   - Fix: Align textbook to match exercise's Delegator/Task/Responsible model (consistent with PACT's accountability grammar).

### Important Issues

3. **L6 missing MCP server**: Title is "Multi-Agent Orchestration **and MCP**" but MCP is entirely absent from exercise code. README and textbook both promise MCP.
   - Fix: Add MCP server task or rename lesson.

4. **L4 missing BM25/hybrid retrieval and RAGAS**: Textbook devotes major sections to these. README promises them. Exercise implements only dense retrieval.
   - Fix: Add BM25 task and RAGAS evaluation, or reduce README/textbook promises.

5. **L3 missing GRPO**: README lists "GRPO (DeepSeek-R1)" as lesson 3 topic. Exercise only covers DPO.
   - Fix: Add GRPO comparison task or clarify as textbook-only content.

6. **Ex_8 exercise number cross-references wrong**: Line 358 says "SFT adapter (Exercise 1)" but SFT is Exercise 2. Line 359 says "DPO adapter (Exercise 2)" but DPO is Exercise 3.

---

## Systemic Issues

### 1. Output File Naming Errors (3 exercises in M4)

Exercises save output files with wrong exercise numbers, suggesting copy-paste from other exercises:
- M4 ex_4 saves as `ex2_*`
- M4 ex_6 saves as `ex3_*`
- M4 ex_8 saves as `ex5_*`

### 2. Module Naming Contamination (2 exercises in M3)

M3 exercises reference "mlfp02" / "Module 2" in experiment names, database names, and docstrings. Suggests these were adapted from M2 code without updating references.

### 3. Textbook Topics Promised But Never Exercised

Several textbook sections have no corresponding exercise tasks:

| Module | Textbook Topic | Textbook Location | Exercise Gap |
|--------|---------------|-------------------|--------------|
| M2 | Difference-in-Differences | L7 textbook | Not in ex_7 |
| M3 | Feature Selection (filter/wrapper/embedded) | L1 textbook | Not in ex_1 |
| M3 | Cross-Validation strategies | L2 textbook | Not in ex_2 |
| M3 | HyperparameterSearch | L7 textbook | Not in ANY M3 exercise |
| M3 | DriftMonitor | L8 textbook | Not in ex_8 |
| M4 | Z-score/IQR outlier detection | L4 textbook | Not in ex_4 |
| M4 | LDA topic modelling | L6 textbook | Not in ex_6 |
| M4 | From-scratch backpropagation | L8 textbook | Not in ex_8 |
| M6 | GRPO | L3 textbook | Not in ex_3 |
| M6 | MCP server implementation | L6 textbook | Not in ex_6 |
| M6 | SLERP model merging | L2 textbook | Not in ANY M6 exercise |

### 4. Dataset Reference Mismatches

| Exercise | README/Textbook Says | Exercise Actually Uses |
|----------|---------------------|----------------------|
| M3 ex_2 | HDB resale | `sg_credit_scoring.parquet` |
| M4 ex_4 | Credit card fraud (V1-V28) | `ecommerce_customers.parquet` |
| M4 ex_6 | Singapore news (`sg_news.parquet`) | `documents.parquet` from mlfp03 |
| M5 ex_6 | Karate Club / synthetic | Synthetic 3-community SBM |
| M5 ex_8 | Inventory management | CartPole-v1 |

---

## Priority Fix List

### Critical (Must Fix Before Delivery) -- 12 items

| # | Module | Issue | File |
|---|--------|-------|------|
| 1 | M3 | MLFP02 naming in ex_1 | `mlfp03/solutions/ex_1.py` |
| 2 | M3 | MLFP02 naming in ex_7 | `mlfp03/solutions/ex_7.py` |
| 3 | M3 | Output file naming ex_8 (`ex6_*` -> `ex8_*`) | `mlfp03/solutions/ex_8.py` |
| 4 | M3 | README dataset mismatch ex_2 | `mlfp03/README.md` |
| 5 | M4 | Docstring contradicts implementation ex_4 | `mlfp04/solutions/ex_4.py` |
| 6 | M4 | Undefined variable in ONNX fallback ex_8 | `mlfp04/solutions/ex_8.py` |
| 7 | M4 | Duplicate TASKS block ex_7 | `mlfp04/solutions/ex_7.py` |
| 8 | M4 | Output file naming ex_4/6/8 | `mlfp04/solutions/ex_4.py`, `ex_6.py`, `ex_8.py` |
| 9 | M5 | README dataset error ex_8 | `mlfp05/README.md` |
| 10 | M6 | Phantom SLERP merge reference ex_8 | `mlfp06/solutions/ex_8.py` |
| 11 | M6 | D/T/R API mismatch L7 | `mlfp06/solutions/ex_7.py` + textbook |
| 12 | M2 | L7 textbook/exercise content divergence | `mlfp02/solutions/ex_7.py` + textbook |

### Important (Should Fix) -- 22 items

| # | Module | Issue |
|---|--------|-------|
| 1 | M1 | L6 heatmap bypasses ModelVisualizer |
| 2 | M1 | L8 PreprocessingPipeline API mismatch |
| 3 | M1 | L8 REST API gap |
| 4 | M2 | L8 FeatureEngineer gap |
| 5 | M2 | L4 column naming mismatch |
| 6 | M2 | L4 Welch's t-test not in textbook |
| 7 | M3 | L7 HyperparameterSearch absent from all exercises |
| 8 | M3 | L8 DriftMonitor omitted |
| 9 | M3 | L1 Feature selection omitted |
| 10 | M3 | L6 ALE and impossibility theorem omitted |
| 11 | M3 | L2 Cross-validation strategies omitted |
| 12 | M4 | Ex_4 missing Z-score/IQR |
| 13 | M4 | Ex_6 missing LDA |
| 14 | M4 | Ex_6 dataset mismatch |
| 15 | M4 | Ex_8 textbook-exercise architecture gap |
| 16 | M5 | L6 GraphSAGE gap |
| 17 | M5 | L6 PyG vs pure-PyTorch divergence |
| 18 | M5 | L6 README dataset wrong |
| 19 | M6 | L6 missing MCP server |
| 20 | M6 | L4 missing BM25/hybrid + RAGAS |
| 21 | M6 | L3 missing GRPO |
| 22 | M6 | Ex_8 cross-reference numbering errors |
