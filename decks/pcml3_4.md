---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 3.4: SHAP, LIME, and Fairness

### Module 3: Supervised ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Compute SHAP values for global and local model explanations
- Apply LIME for instance-level interpretability
- Evaluate ML models for fairness across protected groups
- Choose between explanation methods based on the audience

---

## Recap: Lesson 3.3

- Class imbalance makes accuracy misleading -- use F1, AUC-ROC
- Class weights, SMOTE, and threshold tuning address imbalance
- Probability calibration ensures reliable confidence scores
- Brier score measures calibration quality

---

## Why Explainability Matters

```
Model says: "Loan denied."
Applicant asks: "Why?"

"The model said so" → Not acceptable (legally or ethically)
"Your debt-to-income ratio exceeded the threshold,
 and your employment tenure was below average" → Actionable
```

Explainability is required for trust, debugging, compliance, and fairness.

---

## Two Levels of Explanation

```
Global:  "What features matter OVERALL?"
         → Feature importance rankings
         → SHAP summary plots
         → Understanding model behaviour

Local:   "Why THIS specific prediction?"
         → SHAP waterfall for one instance
         → LIME explanation for one instance
         → Explaining decisions to individuals
```

---

## SHAP: Shapley Additive Explanations

Based on game theory: each feature's contribution to moving the prediction from the average.

```
Average prediction: $465,000

This flat's prediction: $580,000 (+$115,000 above average)

SHAP breakdown:
  floor_area (120 sqm):     +$65,000
  town (Bukit Timah):       +$45,000
  lease_years (90):         +$12,000
  storey (high):            +$8,000
  flat_type (4-ROOM):       -$15,000
                            ─────────
  Total SHAP:               +$115,000  ✓
```

SHAP values always sum to the difference from the mean prediction.

---

## Computing SHAP Values

```python
import shap
from kailash_ml import TrainingPipeline

# Train model
pipeline = TrainingPipeline()
pipeline.configure(
    dataset=df, target_column="price",
    task="regression", algorithm="lightgbm",
)
result = pipeline.run()

# Compute SHAP values
explainer = shap.TreeExplainer(result.model)
shap_values = explainer.shap_values(X_test)

# Summary plot (global)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

---

## SHAP Summary Plot (Text Representation)

```
Feature          Impact on prediction →
─────────────────────────────────────────
floor_area    ◄████████████████████►     High values push price UP
town_encoded  ◄███████████████►          Varies by town
lease_years   ◄██████████►              More lease → higher price
storey        ◄████████►                Higher floor → higher price
flat_type     ◄██████►                  Larger types cost more
month         ◄██►                      Slight seasonal effect
```

Each dot = one prediction. Colour = feature value. Position = SHAP impact.

---

## SHAP Waterfall: Single Prediction

```python
# Explain a single prediction
idx = 0  # first test instance
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test[idx],
        feature_names=feature_names,
    )
)
```

```
E[f(x)] = $465,000
  +$65,000  floor_area = 120
  +$45,000  town = Bukit Timah
  +$12,000  lease_years = 90
  +$8,000   storey = 15
  -$15,000  flat_type = 4-ROOM
  ─────────────────────────
f(x) = $580,000
```

---

## SHAP Dependence Plot

```python
# How does floor_area's SHAP value change with its value?
shap.dependence_plot("floor_area", shap_values, X_test,
                     interaction_index="town_encoded")
```

```
SHAP value
  +80k │              · · ·
       │          · · ·
       │      · · ·          Interaction with town:
       │  · · ·               colour shows town effect
  0    │·─────────────
       │
  -40k │
       └──────────────→ floor_area (sqm)
        40    80   120   160
```

---

## LIME: Local Interpretable Model-Agnostic Explanations

LIME explains any model by fitting a **simple local model** around one prediction.

```python
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    mode="regression",
)

# Explain one prediction
explanation = explainer.explain_instance(
    X_test[0],
    result.model.predict,
    num_features=10,
)
explanation.show_in_notebook()
```

---

## SHAP vs LIME

| Aspect          | SHAP                            | LIME                       |
| --------------- | ------------------------------- | -------------------------- |
| **Theory**      | Shapley values (game theory)    | Local linear approximation |
| **Consistency** | Mathematically guaranteed       | Approximate, can vary      |
| **Speed**       | Fast for trees, slow for others | Fast for any model         |
| **Global view** | Yes (summary plots)             | No (local only)            |
| **Additivity**  | Values sum exactly              | Approximate                |
| **Best for**    | Tree models, detailed analysis  | Quick local explanations   |

Use SHAP for thorough analysis; LIME for quick explanations.

---

## Fairness: Why Models Can Be Biased

```
Training data reflects historical patterns:
  → If historical lending discriminated by race,
    a model trained on this data learns that discrimination.

  → If hiring data favoured men for engineering roles,
    a model perpetuates that bias.

Bias enters through:
  1. Historical data (past discrimination encoded)
  2. Proxy features (postcode → race correlation)
  3. Label bias (who gets labelled as "good customer"?)
  4. Sample bias (underrepresented groups)
```

---

## Protected Attributes

Features that should NOT influence predictions unfairly:

```
Protected:           Can be proxied by:
  Race/ethnicity     → Postcode, name, school
  Gender             → First name, job title
  Age                → Graduation year, experience
  Religion           → Name, neighbourhood
  Disability         → Medical history
```

Even if you remove protected attributes, proxy features can encode them.

---

## Fairness Metrics

```python
# Demographic Parity: equal positive rate across groups
dp = positive_rate_group_a / positive_rate_group_b

# Equalised Odds: equal TPR and FPR across groups
eo_tpr = tpr_group_a / tpr_group_b
eo_fpr = fpr_group_a / fpr_group_b

# Calibration: equal precision across groups
cal = precision_group_a / precision_group_b
```

| Metric                 | Question                                           |
| ---------------------- | -------------------------------------------------- |
| **Demographic Parity** | Do groups get positive predictions at equal rates? |
| **Equalised Odds**     | Do groups have equal error rates?                  |
| **Calibration**        | Does "80% confident" mean the same for all groups? |

---

## Detecting Bias with SHAP

```python
import polars as pl

# Check if protected attributes have high SHAP impact
shap_importance = pl.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": abs(shap_values).mean(axis=0),
}).sort("mean_abs_shap", descending=True)

# Flag if proxy features rank high
proxy_features = ["postcode", "school", "first_name"]
for feat in proxy_features:
    rank = shap_importance.filter(
        pl.col("feature") == feat
    ).row(0, named=True)
    print(f"{feat}: rank={rank}, SHAP={rank['mean_abs_shap']:.2f}")
```

---

## Fairness-Aware Training

```python
from kailash_ml import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.configure(
    dataset=df,
    target_column="approved",
    task="classification",
    algorithm="lightgbm",

    # Fairness constraints
    fairness={
        "protected_column": "gender",
        "metric": "equalised_odds",
        "threshold": 0.8,    # 80% rule
    },
)

result = pipeline.run()
print(f"Accuracy: {result.metrics['accuracy']:.3f}")
print(f"Fairness (EO ratio): {result.fairness_metrics['eo_ratio']:.3f}")
```

---

## The Accuracy-Fairness Tradeoff

```
Accuracy
  1.0 │●
      │  ╲
      │    ╲      Pareto frontier
      │      ●
      │        ╲
      │          ●
      │            ╲
  0.8 │              ●
      └──────────────────→
      Unfair            Fair
      (no constraint)   (strict constraint)
```

Increasing fairness constraints typically reduces raw accuracy. The right balance depends on context and values.

---

## Exercise Preview

**Exercise 3.4: Explainable and Fair HDB Price Models**

You will:

1. Compute SHAP values and create summary and waterfall plots
2. Apply LIME to explain individual predictions
3. Audit a model for fairness across protected groups
4. Investigate proxy features using SHAP dependence plots

Scaffolding level: **Moderate (~50% code provided)**

---

## Common Pitfalls

| Mistake                                             | Fix                                                           |
| --------------------------------------------------- | ------------------------------------------------------------- |
| Explaining training data instead of test            | Always explain on held-out data                               |
| Ignoring proxy features                             | Check if postcode or name correlate with protected attributes |
| Using only one fairness metric                      | Different metrics capture different biases                    |
| Removing protected attributes and assuming fairness | Proxies can encode the same information                       |
| SHAP on huge datasets                               | Sample for speed; use TreeExplainer for tree models           |

---

## Summary

- SHAP provides mathematically grounded global and local explanations
- LIME offers quick local explanations for any model
- Fairness requires checking for bias across protected groups
- Proxy features can encode protected attributes even when removed
- The accuracy-fairness tradeoff requires value judgments, not just metrics

---

## Next Lesson

**Lesson 3.5: Workflow Orchestration**

We will learn:

- Building reproducible ML workflows with `WorkflowBuilder`
- Connecting nodes for data loading, preprocessing, training, evaluation
- Executing workflows with `runtime.execute(workflow.build())`
