# Individual Portfolio Rubric

**Weight**: 35% of final grade  
**Criteria**: 4 dimensions, equally weighted at 25% each

---

## Grading Scale

| Level             | Score Range | Meaning                                        |
| ----------------- | ----------- | ---------------------------------------------- |
| Excellent         | 85-100%     | Exceeds expectations; demonstrates mastery     |
| Good              | 70-84%      | Meets all requirements with minor gaps         |
| Satisfactory      | 50-69%      | Meets minimum requirements; notable weaknesses |
| Needs Improvement | 0-49%       | Missing requirements or fundamental errors     |

---

## Criterion 1: Statistical Rigor (25%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | Assumptions explicitly stated and tested (normality, independence, homoscedasticity). Confidence intervals reported for all key estimates. Multiple comparison corrections applied where needed. Proper cross-validation strategy with justification (stratified k-fold, time-series split as appropriate). Effect sizes reported alongside p-values. Calibration analysis demonstrates deep understanding of probabilistic predictions. |
| **Good**              | 70-84%  | Assumptions acknowledged and mostly tested. Confidence intervals present for primary results. Appropriate cross-validation used. Calibration curves included with interpretation. Minor gaps in assumption testing or multiple comparison handling.                                                                                                                                                                                      |
| **Satisfactory**      | 50-69%  | Basic statistical methodology applied correctly (train/test split, standard metrics). Some assumptions mentioned but not tested. Point estimates reported without confidence intervals. Calibration mentioned but analysis is superficial.                                                                                                                                                                                               |
| **Needs Improvement** | 0-49%   | Fundamental statistical errors: data leakage in cross-validation, comparing models without proper statistical tests, no train/test separation, metrics inappropriate for the problem type (accuracy on imbalanced data without discussion). Missing calibration analysis entirely.                                                                                                                                                       |

### What We Look For

- Is the train/test/validation split appropriate and justified?
- Are results reported with uncertainty (confidence intervals, standard errors)?
- Does the model comparison use proper statistical tests, not just "Model A got 0.87 and Model B got 0.85"?
- Are assumptions of chosen methods explicitly verified?
- Does the calibration analysis connect to the real-world use case?

---

## Criterion 2: Kailash Pattern Mastery (25%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| --------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | All ML operations use appropriate Kailash engines (`DataExplorer`, `TrainingPipeline`, `FeatureEngineer`, `DriftMonitor`, etc.). Framework-first approach throughout — no raw sklearn/pandas workarounds. Engines configured with non-default parameters that demonstrate understanding of their capabilities. `FeatureStore` used for feature versioning. `ModelRegistry` used for model versioning. Async patterns used correctly where applicable. Code is idiomatic Kailash, not "Kailash as a wrapper around sklearn". |
| **Good**              | 70-84%  | Primary operations use correct Kailash engines. Framework-first in most places with occasional raw library usage for edge cases (justified). Engine parameters configured appropriately. Feature and model versioning present. Minor pattern deviations.                                                                                                                                                                                                                                                                    |
| **Satisfactory**      | 50-69%  | Kailash engines used for main tasks but with default configurations. Some operations bypass the framework without justification. Missing feature or model versioning. Engines used but not leveraged (e.g., `TrainingPipeline` with a single model, no hyperparameter search).                                                                                                                                                                                                                                              |
| **Needs Improvement** | 0-49%   | Kailash used minimally or incorrectly. Raw sklearn/pandas used for operations that Kailash engines handle. Incorrect engine selection (using `AutoMLEngine` where `TrainingPipeline` is appropriate, or vice versa). Import errors or misconfigured engines. Framework treated as optional rather than primary.                                                                                                                                                                                                             |

### What We Look For

- Does the code use `DataExplorer` for EDA rather than manual pandas/matplotlib?
- Is `TrainingPipeline` used for model training with proper configuration?
- Are features managed through `FeatureStore`, not loose DataFrames?
- Is `DriftMonitor` genuinely configured, not just imported?
- Does the code feel like it was written by someone who understands the SDK, or someone who wrapped sklearn calls in Kailash classes?

---

## Criterion 3: Production Readiness (25%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Excellent**         | 85-100% | Complete drift monitoring with meaningful PSI thresholds (justified, not default). Model registered in `ModelRegistry` with versioning and metadata. Full reproducibility: random seeds, dependency pinning, data versioning. Retraining strategy documented and implementable. Monitoring alerts defined with escalation paths. Pipeline runs end-to-end from raw data to registered model without manual intervention. |
| **Good**              | 70-84%  | Drift monitoring configured with reasonable thresholds. Model registered with basic metadata. Reproducible with pinned dependencies and seeds. Pipeline mostly automated with minor manual steps documented. Retraining triggers defined.                                                                                                                                                                                |
| **Satisfactory**      | 50-69%  | Basic drift monitoring present (PSI computed but thresholds arbitrary). Model saved but not properly registered/versioned. Some reproducibility measures (seeds set but dependencies not pinned). Pipeline requires manual orchestration.                                                                                                                                                                                |
| **Needs Improvement** | 0-49%   | No drift monitoring or monitoring is placeholder-only. Models saved as pickle files without versioning. Not reproducible (missing seeds, unpinned dependencies, hardcoded paths). Pipeline requires significant manual intervention or fails on re-run.                                                                                                                                                                  |

### What We Look For

- Could another engineer clone this repo and reproduce the results?
- Is the drift monitoring setup realistic (thresholds based on analysis, not arbitrary)?
- Would the retraining strategy actually work in production?
- Is model versioning meaningful (metadata, lineage, not just incrementing numbers)?
- Does the pipeline handle failure gracefully?

---

## Criterion 4: Documentation Quality (25%)

| Level                 | Score   | Indicators                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Excellent**         | 85-100% | Model card complete with substantive content in every section (not boilerplate). Fairness analysis present and meaningful. Limitations section honest and specific (not generic disclaimers). Visualisations are publication-quality: clear labels, appropriate scales, colour-blind accessible, informative captions. Code documented with docstrings and inline comments at decision points. Report narrative flows logically from problem to solution. |
| **Good**              | 70-84%  | Model card complete with most sections substantive. Fairness analysis attempted. Limitations identified. Visualisations clear and well-labelled. Code has docstrings on key functions. Report well-structured with minor narrative gaps.                                                                                                                                                                                                                  |
| **Satisfactory**      | 50-69%  | Model card present but some sections thin or generic. Limited fairness analysis. Visualisations functional but lack polish (missing labels, poor scales). Code comments sparse. Report covers required sections but reads as a checklist rather than a narrative.                                                                                                                                                                                         |
| **Needs Improvement** | 0-49%   | Model card incomplete or contains placeholder text. No fairness analysis. Visualisations misleading or unreadable (truncated axes, unlabelled plots). Code undocumented. Report disorganised or missing sections.                                                                                                                                                                                                                                         |

### What We Look For

- Does the model card pass the "would you deploy based on this?" test?
- Are limitations honest? (Everyone's model has limitations — hiding them is worse than acknowledging them.)
- Would the visualisations make sense to a stakeholder who hasn't seen the code?
- Is the code readable by someone other than the author?
- Does the report tell a coherent story, or is it a disconnected collection of outputs?

---

## Overall Grade Calculation

The final portfolio grade is the weighted average of all four criteria:

```
Portfolio Grade = (Statistical Rigor x 0.25) + (Kailash Mastery x 0.25)
               + (Production Readiness x 0.25) + (Documentation x 0.25)
```

### Grade Boundaries

| Grade | Score   | Description                            |
| ----- | ------- | -------------------------------------- |
| A+    | 90-100% | Exceptional across all criteria        |
| A     | 85-89%  | Excellent in most, good in all         |
| A-    | 80-84%  | Good to excellent across the board     |
| B+    | 75-79%  | Consistently good with some excellence |
| B     | 70-74%  | Meets all requirements well            |
| B-    | 65-69%  | Meets requirements with notable gaps   |
| C+    | 60-64%  | Satisfactory with some strengths       |
| C     | 55-59%  | Meets minimum requirements             |
| D     | 50-54%  | Barely meets minimum requirements      |
| F     | 0-49%   | Does not meet minimum requirements     |

---

## Automatic Penalties

| Issue                                           | Penalty                                     |
| ----------------------------------------------- | ------------------------------------------- |
| Code does not run end-to-end                    | -10% from total                             |
| Hardcoded API keys or credentials in repository | -10% from total                             |
| Missing model card                              | -15% from total (caps Documentation at 50%) |
| Fewer than 3 model approaches compared          | -10% from Kailash Pattern Mastery           |
| No drift monitoring setup                       | -15% from Production Readiness              |
| Late submission                                 | -5% per calendar day                        |
| Plagiarism / shared code                        | Referred to academic integrity board        |

---

## Moderation

All portfolios are first-marked by the module tutor and moderated by a second marker for any grade at a boundary (within 2% of a grade threshold) or any failing grade. Moderated grades are final.
