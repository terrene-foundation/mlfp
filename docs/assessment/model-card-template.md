# Model Card Template

_Based on Mitchell et al., 2019 ("Model Cards for Model Reporting"). Complete every section with substantive content specific to your model. Generic or placeholder text will be marked down._

---

## Model Details

| Field             | Value                                       |
| ----------------- | ------------------------------------------- |
| **Model Name**    | _e.g., HDB Resale Price Predictor v2.1_     |
| **Model Version** | _e.g., 2.1.0 (registered in ModelRegistry)_ |
| **Model Type**    | _e.g., Gradient Boosted Trees (LightGBM)_   |
| **Framework**     | _e.g., kailash-ml TrainingPipeline v0.4.0_  |
| **Training Date** | _e.g., 2026-03-15_                          |
| **Developers**    | _Your name(s) and affiliation_              |
| **Contact**       | _Email for questions about this model_      |
| **License**       | _e.g., Apache 2.0 (for the model code)_     |

### Model Architecture

_Describe the model architecture in 2-3 sentences. For ensemble methods, describe the base learners and combination strategy. For neural networks, describe the layer structure. For linear models, describe the feature representation._

### Training Algorithm

_Describe the training procedure: optimisation algorithm, loss function, regularisation applied, convergence criteria._

---

## Intended Use

### Primary Use Case

_Describe the specific task this model is designed to perform. Be precise: "Predict the resale price of HDB flats in Singapore given property attributes and location features" is better than "Price prediction"._

### Intended Users

_Who should use this model? What level of ML expertise is expected? e.g., "Data analysts in the urban planning department who need price estimates for policy analysis."_

### Out-of-Scope Uses

_List uses this model is explicitly NOT designed for. Be specific:_

- _e.g., This model should not be used for individual property valuation for mortgage decisions_
- _e.g., This model is not trained on commercial properties and should not be applied to them_
- _e.g., Predictions beyond 12 months from the training date should be treated with caution due to market drift_

---

## Training Data

### Data Source

| Field                  | Value                                      |
| ---------------------- | ------------------------------------------ |
| **Source**             | _e.g., data.gov.sg HDB Resale Flat Prices_ |
| **Collection Period**  | _e.g., January 2019 - December 2024_       |
| **Number of Records**  | _e.g., 142,857 transactions_               |
| **Number of Features** | _e.g., 23 (12 original + 11 engineered)_   |
| **Geographic Scope**   | _e.g., All HDB towns in Singapore_         |

### Preprocessing

_Describe the preprocessing pipeline applied:_

- _Missing value treatment (method and justification)_
- _Outlier handling (detection method, treatment decision)_
- _Feature encoding (categorical encoding strategy)_
- _Feature scaling (method and which features)_
- _Feature engineering (list engineered features with rationale)_

### Data Splits

| Split          | Size                          | Strategy                                               |
| -------------- | ----------------------------- | ------------------------------------------------------ |
| **Training**   | _e.g., 70% (100,000 records)_ | _e.g., Temporal split: transactions before 2024-01-01_ |
| **Validation** | _e.g., 15% (21,428 records)_  | _e.g., 2024-01-01 to 2024-06-30_                       |
| **Test**       | _e.g., 15% (21,429 records)_  | _e.g., 2024-07-01 to 2024-12-31_                       |

_Justify your splitting strategy. If you used random splits, explain why temporal splits were not necessary. If you used temporal splits, explain the boundary dates._

---

## Evaluation

### Metrics

_Report all evaluation metrics on the test set with confidence intervals._

| Metric            | Test Set Value  | 95% CI                     |
| ----------------- | --------------- | -------------------------- |
| _e.g., RMSE_      | _e.g., $23,450_ | _e.g., [$22,100, $24,800]_ |
| _e.g., MAE_       | _e.g., $17,200_ | _e.g., [$16,500, $17,900]_ |
| _e.g., R-squared_ | _e.g., 0.912_   | _e.g., [0.905, 0.919]_     |
| _e.g., MAPE_      | _e.g., 4.2%_    | _e.g., [3.9%, 4.5%]_       |

_Explain why you chose these metrics for your specific problem. If classification: precision, recall, F1, AUC-ROC at minimum. If regression: RMSE, MAE, R-squared at minimum._

### Performance Across Subgroups

_Break down performance across meaningful subgroups to identify where the model performs well and where it struggles._

| Subgroup                   | Metric | Value     | N        |
| -------------------------- | ------ | --------- | -------- |
| _e.g., 3-room flats_       | _RMSE_ | _$15,200_ | _42,000_ |
| _e.g., 5-room flats_       | _RMSE_ | _$28,100_ | _35,000_ |
| _e.g., Executive flats_    | _RMSE_ | _$41,500_ | _8,500_  |
| _e.g., Mature estates_     | _RMSE_ | _$26,800_ | _65,000_ |
| _e.g., Non-mature estates_ | _RMSE_ | _$19,300_ | _77,000_ |

### Comparison with Baselines

| Model                        | Primary Metric       | Notes                     |
| ---------------------------- | -------------------- | ------------------------- |
| _Baseline (mean prediction)_ | _e.g., RMSE $78,200_ | _Naive baseline_          |
| _Linear Regression_          | _e.g., RMSE $31,400_ | _Regularised (L2)_        |
| _Random Forest_              | _e.g., RMSE $25,100_ | _500 trees, max_depth=12_ |
| _**Selected Model**_         | _e.g., RMSE $23,450_ | _LightGBM, tuned_         |

### Calibration

_For classification models: include calibration curve and Brier score. For regression models: include prediction interval coverage analysis._

_Describe whether the model is well-calibrated. If not, describe what calibration correction was applied._

---

## Interpretability

### Global Feature Importance

_List the top 10 features by SHAP importance. Include a SHAP summary plot as a figure._

| Rank | Feature                       | Mean          | SHAP |     |
| ---- | ----------------------------- | ------------- | ---- | --- |
| 1    | _e.g., floor_area_sqm_        | _e.g., 0.342_ |
| 2    | _e.g., remaining_lease_years_ | _e.g., 0.287_ |
| 3    | _e.g., dist_to_nearest_mrt_   | _e.g., 0.156_ |
| ...  | ...                           | ...           |

### Key Interactions

_Describe the most important feature interactions identified through SHAP interaction values or partial dependence plots. e.g., "Floor area and storey have a positive interaction: the price premium for larger flats increases on higher floors."_

### Local Explanations

_Provide 2-3 example predictions with local SHAP explanations. Choose cases that illustrate the model's reasoning: a typical prediction, an edge case, and an error._

---

## Limitations

### Known Failure Modes

_Be specific and honest. Every model has failure modes._

- _e.g., The model significantly underpredicts prices for flats near newly announced MRT stations (feature lag)_
- _e.g., Performance degrades for flat types with fewer than 500 training examples (executive apartments in non-mature estates)_
- _e.g., The model does not account for renovation quality, which can add $30,000-$80,000 to resale prices_

### Data Limitations

- _e.g., Training data does not include private property transactions, so the model cannot generalise to condominiums_
- _e.g., COVID-19 period (2020-2021) introduces distributional shift in the training data_
- _e.g., No feature captures neighbourhood gentrification trends_

### Technical Limitations

- _e.g., Inference latency is ~50ms per prediction; batch predictions of 10,000+ require async processing_
- _e.g., Model file size is 180MB; too large for edge deployment without compression_

---

## Ethical Considerations

### Fairness

_Analyse model fairness across protected attributes relevant to your domain._

- _e.g., The model uses geographic features (town, district) which correlate with ethnic composition. While these features improve accuracy, they may encode demographic biases in pricing._
- _e.g., We tested for disparate impact across postal districts with different demographic compositions. The error rate differential was X% (threshold: Y%)._

### Privacy

- _e.g., No personally identifiable information is used as features_
- _e.g., Transaction-level data is publicly available via data.gov.sg_
- _e.g., Model outputs cannot be used to reverse-engineer individual seller/buyer information_

### Societal Impact

_Consider the broader impact of your model's deployment._

- _e.g., Automated price predictions could accelerate speculation if used without human oversight_
- _e.g., Biased predictions in underserved areas could perpetuate pricing inequities_

### Mitigation Strategies

_What steps have been taken to address the ethical considerations above?_

- _e.g., Model predictions are presented with confidence intervals, not point estimates, to communicate uncertainty_
- _e.g., Drift monitoring alerts will trigger human review when demographic composition of input data shifts_

---

## Monitoring and Maintenance

### Drift Monitoring

| Monitor             | Metric                  | Threshold              | Action                          |
| ------------------- | ----------------------- | ---------------------- | ------------------------------- |
| _Feature drift_     | _PSI_                   | _e.g., > 0.2_          | _Alert + manual review_         |
| _Prediction drift_  | _KS statistic_          | _e.g., > 0.1_          | _Alert + retraining evaluation_ |
| _Performance drift_ | _RMSE on labelled data_ | _e.g., > 20% increase_ | _Automatic retraining trigger_  |

### Retraining Schedule

_e.g., Monthly retraining on the latest 5 years of data. Manual retraining triggered by drift alerts. All retrained models evaluated against the current production model before promotion._

### Model Versioning

_e.g., All model versions tracked in ModelRegistry with training data hash, hyperparameters, and evaluation metrics. Production model promotion requires automated evaluation gate (new model must match or exceed current model on test set)._

---

## Additional Information

### Citation

_If this model or its methodology should be cited:_

```
[Your preferred citation format]
```

### References

_List key papers, datasets, or resources used in developing this model._

---

_This model card was last updated on [date] and applies to model version [version]._
