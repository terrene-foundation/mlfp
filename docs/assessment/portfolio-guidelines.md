# Individual Portfolio Guidelines

**Weight**: 35% of final grade  
**Format**: Production-ready ML system with documentation  
**Submission**: PDF report + GitHub repository link

---

## Overview

The individual portfolio is your opportunity to demonstrate end-to-end ML engineering competence. You will select a dataset from any ASCENT module, extend it beyond the exercise scope, and build a production-ready system using the Kailash Python SDK.

This is not a research paper. It is a working system with rigorous documentation. Your portfolio should demonstrate the journey from raw data to a monitored, deployable model — the kind of work a senior ML engineer delivers.

---

## Requirements

### Dataset Selection

Choose any dataset from the ASCENT course modules or the integrated Singapore urban planning dataset in `ascent_assessment/`. You may combine multiple datasets if your problem requires it.

Your selected dataset must present genuine complexity: missing values, class imbalance, temporal dependencies, or multi-source joins. If the data is clean and simple, the portfolio cannot demonstrate production readiness.

### Mandatory Components

Every portfolio must include all seven components below. Missing any component caps the maximum grade at Satisfactory.

#### 1. Exploratory Data Analysis Report

Use `DataExplorer` to produce a comprehensive EDA:

- Distribution analysis for all features (skewness, outliers, modality)
- Correlation structure (pairwise and partial correlations)
- Missing data patterns (MCAR/MAR/MNAR analysis with justification)
- Temporal patterns if applicable (stationarity, seasonality)
- At least 5 publication-quality visualizations using `ModelVisualizer`

#### 2. Feature Engineering Rationale

Use `FeatureEngineer` and `FeatureStore` to construct your feature set:

- Document every engineered feature with its business/statistical motivation
- Feature selection methodology (filter, wrapper, or embedded — justify your choice)
- Feature importance analysis (pre-modelling and post-modelling)
- Address multicollinearity, leakage risk, and encoding decisions explicitly

#### 3. Model Comparison

Use `TrainingPipeline` to train and compare at least 3 materially different approaches:

- At least one linear/parametric model (logistic regression, linear SVM)
- At least one ensemble method (gradient boosting, random forest)
- At least one other approach appropriate to your problem (neural network, clustering-augmented, etc.)
- Hyperparameter tuning via `HyperparameterSearch` for each approach
- Statistical comparison of results (confidence intervals, paired tests — not just point estimates)

#### 4. SHAP Interpretability Analysis

- Global SHAP summary for your best model
- At least 3 local SHAP explanations for individual predictions (choose interesting cases: correct predictions, errors, edge cases)
- Feature interaction effects where relevant
- Compare SHAP-derived importance with your pre-modelling feature importance

#### 5. Calibration Analysis

- Calibration curves (reliability diagrams) for all models
- Brier score decomposition (reliability, resolution, uncertainty)
- If poorly calibrated: apply Platt scaling or isotonic regression, show before/after
- Discuss calibration implications for your specific use case (when does miscalibration matter?)

#### 6. Drift Monitoring Setup

Use `DriftMonitor` to configure monitoring for your deployed model:

- Define reference and production data windows
- Configure PSI (Population Stability Index) thresholds for key features
- Configure model performance monitoring metrics
- Demonstrate drift detection on a simulated data shift
- Document your alerting strategy (what triggers retraining?)

#### 7. Model Card

Complete the model card template (see `docs/assessment/model-card-template.md`):

- All sections filled with substantive content (not placeholders)
- Fairness analysis across at least one protected attribute if applicable to your dataset
- Honest limitations section — identify where your model fails and why

---

## Scope and Length Guidance

| Section                       | Approximate Length                       |
| ----------------------------- | ---------------------------------------- |
| EDA Report                    | 4-6 pages (including figures)            |
| Feature Engineering Rationale | 2-3 pages                                |
| Model Comparison              | 3-5 pages (including tables and figures) |
| SHAP Interpretability         | 2-3 pages (including SHAP plots)         |
| Calibration Analysis          | 1-2 pages                                |
| Drift Monitoring Setup        | 1-2 pages (including configuration code) |
| Model Card                    | 2-3 pages                                |
| **Total Report**              | **15-25 pages**                          |

The code repository has no page limit but should be well-organised and documented. Quality over quantity — concise, precise writing scores higher than verbose padding.

---

## Example Topics by Module

These are starting points, not prescriptions. The best portfolios define a clear problem statement and pursue it with rigour.

| Module | Dataset                 | Example Portfolio Direction                                                                                                    |
| ------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| ASCENT1  | HDB Resale Prices       | Price prediction with spatial features, temporal drift analysis, calibrated confidence intervals for valuation                 |
| ASCENT2  | A/B Testing Dataset     | Causal inference pipeline: propensity score matching, CUPED variance reduction, heterogeneous treatment effects                |
| ASCENT3  | Credit Risk / Fraud     | End-to-end credit scoring: regulatory-compliant features, calibrated default probabilities, fairness audit across demographics |
| ASCENT4  | Customer Reviews (NLP)  | BERTopic-based topic modelling with sentiment overlay, UMAP visualisation, topic drift monitoring for brand health             |
| ASCENT5  | Multi-source Urban Data | RAG-augmented property advisor: agent-based retrieval over planning documents, grounded in structured data                     |
| ASCENT6  | Policy / Governance     | RL-based resource allocation with PACT governance constraints, alignment evaluation, deployment with Nexus                     |

---

## Submission Format

### Repository Structure

```
portfolio/
    README.md              # Problem statement, setup instructions, results summary
    pyproject.toml         # Dependencies (kailash-ml, etc.)
    src/
        eda.py             # DataExplorer analysis
        features.py        # FeatureEngineer pipeline
        train.py           # TrainingPipeline with model comparison
        interpret.py       # SHAP analysis
        calibrate.py       # Calibration analysis
        monitor.py         # DriftMonitor configuration
    notebooks/
        portfolio.ipynb    # Integrated narrative (optional but encouraged)
    docs/
        model_card.md      # Completed model card
        report.pdf         # Full portfolio report
    data/
        README.md          # Data source attribution and access instructions
```

### Submission Checklist

- [ ] Repository is public or shared with instructors
- [ ] `README.md` contains setup instructions that work from a clean environment
- [ ] All code runs end-to-end without errors (`uv sync && uv run python src/train.py`)
- [ ] Report PDF is included in `docs/`
- [ ] Model card is complete (no placeholder sections)
- [ ] No hardcoded API keys, credentials, or absolute file paths

### Deadlines

| Milestone                               | Due           |
| --------------------------------------- | ------------- |
| Topic proposal (1 paragraph + dataset)  | End of Week 2 |
| Progress check (EDA + initial features) | End of Week 4 |
| Final submission                        | End of Week 6 |

Late submissions are penalised at 5% per calendar day. Extensions require written approval from the module coordinator before the deadline.

---

## Grading

Portfolios are graded against the rubric in `docs/assessment/portfolio-rubric.md`. The four equally weighted criteria are:

1. **Statistical Rigor** (25%) — Proper methodology, valid assumptions, confidence intervals
2. **Kailash Pattern Mastery** (25%) — Correct engine usage, framework-first approach, async patterns
3. **Production Readiness** (25%) — Drift monitoring, model registry, reproducibility
4. **Documentation Quality** (25%) — Model card completeness, code clarity, visualisation quality

---

## Academic Integrity

Your portfolio must be your own work. You may use Kailash SDK documentation, course materials, and publicly available references. You must cite any external code, datasets, or methodologies you adapt.

Portfolios that share identical feature engineering logic, model configurations, or documentation text will be investigated under the academic integrity policy.
