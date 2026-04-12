### MODULE 3: Supervised Machine Learning for Building and Deploying Models

**Description**: The ML pipeline — from feature engineering to production deployment. Builds on M2's regression foundation. Following R5 Deck 4A: focus on the PIPELINE and advanced models, not re-teaching basic regression.

**Module Learning Objectives**: By the end of M3, students can:
- Engineer features and select the most predictive ones
- Explain bias-variance tradeoff and apply regularisation
- Train and evaluate the complete supervised model zoo (linear, SVM, KNN, Naive Bayes, trees, forests, gradient boosting)
- Handle class imbalance and calibrate probabilistic predictions
- Interpret models using SHAP, LIME, and fairness metrics
- Orchestrate ML workflows with custom nodes
- Deploy production pipelines with model registry, drift monitoring, and DataFlow persistence

**Kailash Engines**: FeatureEngineer, FeatureStore, PreprocessingPipeline, TrainingPipeline, AutoMLEngine, HyperparameterSearch, ModelRegistry, EnsembleEngine, WorkflowBuilder, DataFlow, DriftMonitor, ModelVisualizer

---

#### Lesson 3.1: Feature Engineering, ML Pipeline, and Feature Selection

**Prerequisites**: M2 complete (statistics, regression)
**Spectrum Position**: Manual feature engineering — human designs features from domain knowledge

**Topics**:
- Feature engineering philosophy from Deck 4A: "Data > Models > Hyperparameter Tuning"
  - Geocoding example: address -> lat/lon via OneMap API
  - Domain knowledge drives feature creation
- ML pipeline stages: data ingestion -> preprocessing -> feature engineering -> model selection -> training/eval -> hyperparameters -> deployment
- Statistics vs ML distinction (from Deck 4A): explaining the past vs predicting the future
- Temporal features: lag, rolling mean, day-of-week, month, season
- Interaction terms, polynomial features
- Leakage detection: features that would not be available at prediction time
- **Feature Selection Methods** (new, not in R5):
  - Filter: mutual information, chi-squared, correlation thresholds
  - Wrapper: forward selection, backward elimination, RFE (Recursive Feature Elimination)
  - Embedded: L1 sparsity (Lasso), tree-based importance
  - FeatureEngineer: generate + select

**Key Concepts**: Domain-driven feature engineering, ML pipeline stages, feature selection taxonomy, leakage

**Learning Objectives**: Students can:
- Engineer features from domain knowledge (not just data manipulation)
- Identify and prevent feature leakage
- Apply filter, wrapper, and embedded feature selection methods
- Explain why data quality matters more than model complexity

**Exercise**: Engineer features for HDB price prediction (geocode addresses, create temporal features, interaction terms). Apply 3 feature selection methods and compare selected features.

**Assessment Criteria**: Features have domain rationale. No leakage. Feature selection methods compared with rationale for final selection.

**R5 Source**: Deck 4A (feature engineering, ML pipeline, statistics vs ML)

---

#### Lesson 3.2: Bias-Variance, Regularisation, and Cross-Validation

**Prerequisites**: 2.5 (linear regression), 3.1 (feature engineering)
**Spectrum Position**: Model complexity control — the fundamental ML tradeoff

**Topics**:
- Bias-variance decomposition: E[(y - y_hat)^2] = Bias^2 + Variance + sigma^2
  - Intuition: darts at a target (bias = aim, variance = spread)
  - Underfitting (high bias) vs overfitting (high variance)
- Regularisation:
  - L1 (Lasso): drives coefficients to zero, sparse solutions. Geometry: diamond constraint.
  - L2 (Ridge): shrinks coefficients toward zero. Geometry: circle constraint.
  - Elastic Net: L1 + L2 combination (alpha mixing parameter)
  - Bayesian interpretation: L2 = Gaussian prior on coefficients (connects to M2.1)
- Cross-validation:
  - k-fold: split data into k folds, train on k-1, validate on 1, rotate
  - Stratified k-fold: preserve class proportions
  - Time-series split: walk-forward validation (no future data leakage)
  - Nested CV: outer loop for model selection, inner loop for hyperparameters (connects to M3.7)
  - GroupKFold: when observations are grouped (e.g., same patient, same company)

**Key Formulas**:
- Bias-variance: E[(y - y_hat)^2] = Bias^2(y_hat) + Var(y_hat) + sigma^2
- L1 penalty: lambda * Sum(|beta_i|)
- L2 penalty: lambda * Sum(beta_i^2)
- Elastic Net: alpha * L1 + (1-alpha) * L2

**Learning Objectives**: Students can:
- Derive the bias-variance decomposition for squared loss
- Apply L1, L2, and Elastic Net regularisation
- Explain the Bayesian interpretation of L2
- Select appropriate cross-validation strategy for different data types
- Implement nested CV for unbiased model selection

**Exercise**: Demonstrate bias-variance tradeoff by varying model complexity on HDB data. Compare L1 vs L2 vs ElasticNet on same dataset. Implement nested CV.

**Assessment Criteria**: Bias-variance demonstrated visually (train/test error curves). Regularisation impact on coefficients shown. CV strategy matches data structure.

**R5 Source**: ASCENT (new derivation, not in R5 decks)

---

#### Lesson 3.3: The Complete Supervised Model Zoo

**Prerequisites**: 3.2 (bias-variance, regularisation, cross-validation)
**Spectrum Position**: Model breadth — knowing when to use what

**Topics**:
- **SVM (Support Vector Machines)**: margin maximisation, kernel trick (linear, RBF, polynomial), soft margin (C parameter). When to use: high-dimensional, clear margin of separation.
- **KNN (K-Nearest Neighbors)**: instance-based learning, distance metrics (Euclidean, Manhattan, cosine), curse of dimensionality, k selection. When to use: small data, interpretable boundaries.
- **Naive Bayes**: GaussianNB, MultinomialNB, BernoulliNB. Naive independence assumption. When to use: text classification, fast baseline. Connects to M2.1 Bayesian thinking.
- **Decision Trees**: splitting criteria (Gini impurity, entropy/information gain), recursive partitioning, pruning (pre-pruning: max_depth, min_samples; post-pruning). Overfitting visualised. When to use: interpretable, non-linear boundaries.
- **Random Forests**: bagging (bootstrap aggregating), feature subsampling, out-of-bag (OOB) estimation, feature importance. When to use: robust default, handles missing data.
- **Model comparison framework**: accuracy vs interpretability vs speed vs data size

**Key Formulas**:
- SVM: maximise 2/||w|| subject to y_i(w x x_i + b) >= 1
- Gini impurity: G = 1 - Sum(p_i^2)
- Information gain: IG = H(parent) - Sum(w_i * H(child_i))
- OOB error: ~36.8% of samples not in each bootstrap sample

**Learning Objectives**: Students can:
- Explain the mathematical basis of SVM, KNN, Naive Bayes, decision trees, and random forests
- Select the appropriate algorithm for a given problem based on data characteristics
- Tune key hyperparameters for each algorithm
- Compare models using proper evaluation methodology

**Exercise**: Train all 5 model families on the same dataset (e-commerce customer classification). Compare performance, training time, and interpretability. Produce a model comparison table.

**Assessment Criteria**: All 5 models trained correctly. Comparison uses consistent evaluation (same CV splits). Model selection justified with data evidence, not opinion.

**R5 Source**: Deck 4B (lists 18 models in monitoring slide) + ASCENT. Note: SVM, KNN, Naive Bayes are new additions not in R5 or ASCENT — need new deck content and exercises.

---

#### Lesson 3.4: Gradient Boosting Deep Dive

**Prerequisites**: 3.3 (decision trees, random forests)
**Spectrum Position**: Model depth — mastering the dominant tabular algorithm

**Topics**:
- **Boosting theory**: sequential ensemble, bias reduction (vs bagging's variance reduction)
- **AdaBoost**: as conceptual warmup — reweight misclassified samples
- **XGBoost**:
  - 2nd-order Taylor expansion of loss function
  - Split gain formula: Gain = 1/2 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
  - Regularisation: lambda (L2 on leaf weights), gamma (min split loss)
- **LightGBM**: Gradient-based One-Side Sampling (GOSS), histogram-based split finding, leaf-wise growth
- **CatBoost**: ordered boosting (prevents target leakage), native categorical feature support
- Model comparison across the boosting family

**Key Formulas**:
- XGBoost objective: Sum(L(y_i, y_hat_i)) + Sum(Omega(f_k))
- XGBoost split gain (see above)
- LightGBM GOSS: keep top-a% gradient samples, randomly sample b% of small gradients

**Learning Objectives**: Students can:
- Explain how boosting reduces bias (vs bagging reducing variance)
- Derive the XGBoost split gain formula
- Compare XGBoost, LightGBM, and CatBoost on the same dataset
- Select and tune the appropriate boosting algorithm

**Exercise**: Train XGBoost, LightGBM, CatBoost on credit scoring data. Compare accuracy, training time, feature importance. Tune key hyperparameters. Explain when to choose each.

**Assessment Criteria**: All three trained and compared. Key hyperparameters tuned (not default). Selection justified with evidence.

**R5 Source**: ASCENT M3. Note: XGBoost 2nd-order Taylor derivation is new in ASCENT, not in R5 decks.

---

#### Lesson 3.5: Model Evaluation, Imbalance, and Calibration

**Prerequisites**: 3.3 + 3.4 (full model zoo)
**Spectrum Position**: Model assessment — knowing how good your model really is

**Topics**:
- **Complete Metrics Taxonomy**:
  - Classification: accuracy, precision, recall, F1-score, ROC-AUC, log loss, confusion matrix, precision-recall curve, specificity, sensitivity
  - Regression: R-squared, adjusted R-squared, MAE, MSE, RMSE, MAPE
  - When to use which: imbalanced data (precision-recall, not accuracy), probabilistic output (log loss, not accuracy), business cost (custom cost matrix)
- **Class Imbalance**:
  - Why accuracy fails on imbalanced data
  - SMOTE and its failures (boundary samples, high-dimensional)
  - Cost-sensitive learning: class weights in loss function
  - Focal Loss: down-weight easy examples (gamma parameter)
- **Calibration**:
  - Platt scaling (logistic regression on model output)
  - Isotonic regression (non-parametric calibration)
  - Calibration plots: reliability diagram
  - Proper scoring rules: Brier score
- **Stacking and blending**: combining model predictions (brief, connects to EnsembleEngine in M4)

**Key Formulas**:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1: 2 * (Precision * Recall) / (Precision + Recall)
- AUC: area under ROC curve (TPR vs FPR)
- Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
- Brier Score: BS = 1/N * Sum((p_i - y_i)^2)

**Learning Objectives**: Students can:
- Select appropriate metrics for classification and regression tasks
- Handle class imbalance with cost-sensitive learning (not just SMOTE)
- Calibrate model probabilities using Platt scaling
- Read and interpret calibration plots

**Exercise**: Train model on imbalanced credit scoring data. Compare accuracy vs F1 vs AUC. Apply SMOTE and cost-sensitive approaches. Calibrate with Platt scaling. Generate calibration plot.

**Assessment Criteria**: Metrics correctly chosen for the problem. Imbalance handled (not just SMOTE). Calibration improves Brier score. Calibration plot shows improvement.

**R5 Source**: ASCENT M3 ex_2

---

#### Lesson 3.6: Interpretability and Fairness

**Prerequisites**: 3.3-3.5 (trained models, evaluation)
**Spectrum Position**: Model transparency — explaining predictions and checking for bias

**Topics**:
- **SHAP (SHapley Additive exPlanations)**:
  - Shapley axioms: efficiency, symmetry, dummy, linearity
  - TreeSHAP: efficient computation for tree-based models
  - KernelSHAP: model-agnostic (slower)
  - SHAP plots: summary, dependence, waterfall, force
- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Perturb input, fit local linear model
  - Interpretation: "for THIS prediction, these features mattered most"
- **ALE (Accumulated Local Effects)**: alternative to PDP, handles feature correlation
- **Fairness**:
  - Disparate impact: ratio of selection rates between groups
  - Equalized odds: TPR and FPR equal across groups
  - Calibration parity: predicted probabilities equally reliable across groups
  - **Impossibility theorem**: cannot simultaneously satisfy demographic parity, equalized odds, AND calibration (Chouldechova 2017, Kleinberg et al. 2016)
  - Fairness as engineering: measure it, report it, mitigate where possible

**Key Formulas**:
- Shapley value: phi_i = Sum over S of [|S|!(|F|-|S|-1)!/|F|!] * [f(S u {i}) - f(S)]
- Disparate impact ratio: P(Y=1|G=minority) / P(Y=1|G=majority) (should be > 0.8)

**Learning Objectives**: Students can:
- Compute and interpret SHAP values for individual and global explanations
- Apply LIME for local interpretability
- Measure fairness using disparate impact and equalized odds
- Explain the impossibility theorem and its implications for model deployment

**Exercise**: Compute SHAP values for the credit scoring model. Generate SHAP summary and waterfall plots. Measure disparate impact across demographic groups. Report fairness findings.

**Assessment Criteria**: SHAP values computed and interpreted correctly. Fairness measured quantitatively. Impossibility theorem explained with model-specific example.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 3.7: Workflow Orchestration, Model Registry, and Hyperparameter Search

**Prerequisites**: 3.1-3.6 (complete ML knowledge)
**Spectrum Position**: ML engineering — automating the training pipeline

**Topics**:
- **WorkflowBuilder**: nodes, connections, runtime, `runtime.execute(workflow.build())`
- **Custom Nodes**: `@register_node`, `Node` subclass, `PythonCodeNode`, `ConditionalNode`
- **Logic nodes**: branching, merging, conditional execution
- **HyperparameterSearch**: Bayesian optimisation, SearchSpace, ParamDistribution, SearchConfig
- **ModelRegistry**: model versioning, metadata, staging -> production promotion
- **MetricSpec, ModelSignature**: schema validation for model inputs/outputs
- Model lifecycle: experiment -> register -> stage -> promote -> serve -> retire

**Key Concepts**: Workflow orchestration, node-based pipelines, Bayesian hyperparameter optimisation, model versioning

**Learning Objectives**: Students can:
- Build ML workflows using WorkflowBuilder with custom nodes
- Implement Bayesian hyperparameter search
- Register, version, and promote models through the lifecycle
- Define model signatures for input/output validation

**Exercise**: Build an automated ML pipeline as a workflow: data loading node -> preprocessing node -> training node -> evaluation node -> conditional promotion node. Register best model.

**Assessment Criteria**: Workflow executes end-to-end. Custom nodes correctly defined. Hyperparameter search improves model. Model registered with signature.

**R5 Source**: Deck 4B (MLOps components) + ASCENT M3 ex_4/5

---

#### Lesson 3.8: Production Pipeline — DataFlow, Drift, and Deployment

**Prerequisites**: 3.7 (workflow orchestration, model registry)
**Spectrum Position**: Production ML — from training to serving and monitoring

**Topics**:
- **DataFlow**: `@db.model`, `field()`, `db.express.create/list/get/update/delete`, `ConnectionManager`
  - Schema design for ML results
  - Async/await primer for database operations
- **DriftMonitor**: monitor deployed models for distribution shift
  - PSI (Population Stability Index): compare feature distributions
  - KS test (Kolmogorov-Smirnov): compare CDFs
  - Performance degradation detection
  - Monitoring frequency and alerting
  - DriftSpec configuration
- **Model Card**: document model purpose, performance, limitations, fairness findings (Mitchell et al.)
- **Conformal prediction**: distribution-free prediction intervals
- **Full production pipeline**: train -> persist results to DataFlow -> calibrate -> register -> promote -> monitor for drift -> model card
- **MLOps concepts** from Deck 4B: CI/CD for ML, clean architecture, model versioning and governance

**Key Concepts**: Database persistence, drift monitoring, model documentation, production deployment

**Learning Objectives**: Students can:
- Persist ML results to a database using DataFlow
- Monitor deployed models for drift using PSI and KS tests
- Create model cards documenting performance and limitations
- Build a complete production ML pipeline from training to monitoring

**Exercise**: Deploy the credit scoring model as a full production pipeline. Persist results to DataFlow. Set up DriftMonitor with alerting thresholds. Generate model card. Simulate drift and verify detection.

**Assessment Criteria**: Pipeline end-to-end. DataFlow CRUD operations work. Drift detected when injected. Model card complete.

**R5 Source**: Deck 4B (MLOps) + ASCENT M3 ex_6

**End of Module Assessment**: Quiz + ML pipeline project (full pipeline from raw data to deployed, monitored model).
