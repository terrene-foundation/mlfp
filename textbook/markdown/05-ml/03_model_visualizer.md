# Chapter 3: ModelVisualizer

## Overview

ModelVisualizer generates interactive Plotly figures for common ML evaluation charts. It is stateless -- you pass in data (predictions, labels, metrics) and get back a Plotly Figure object that can be displayed inline in Jupyter, saved as HTML, or embedded in dashboards.

This chapter covers:

- Confusion matrix visualization
- ROC curve with AUC display
- Precision-recall curve with Average Precision
- Feature importance (tree-based, coefficient-based, and permutation-based)
- Learning curves (training vs validation score)
- Residual analysis for regression models
- Calibration curves for probability calibration
- Metric comparison across multiple models
- Training history (loss curves)

## Prerequisites

| Requirement | Details                    |
| ----------- | -------------------------- |
| Python      | 3.10+                      |
| kailash-ml  | `pip install kailash-ml`   |
| plotly      | Installed with kailash-ml  |
| sklearn     | For generating demo models |
| Level       | Basic                      |

## Concepts

### Why Visualize?

Numbers alone can be misleading. A model with 95% accuracy might be terrible on the minority class -- the confusion matrix reveals this. An ROC curve shows how the model trades off true positives against false positives at every threshold. Feature importance tells you which inputs the model actually relies on, which matters for interpretability and debugging.

### Plotly Figures

Every method returns a `plotly.graph_objects.Figure`. You can:

- Display inline in Jupyter: just return the figure
- Save to HTML: `fig.write_html("chart.html")`
- Save as image: `fig.write_image("chart.png")`
- Customize: `fig.update_layout(title="Custom Title")`

### Feature Importance Methods

ModelVisualizer supports three sources of feature importance:

1. **Tree-based** -- Uses `model.feature_importances_` (RandomForest, GradientBoosting, XGBoost)
2. **Coefficient-based** -- Uses `abs(model.coef_)` (LogisticRegression, Ridge, Lasso)
3. **Permutation-based** -- When neither attribute exists but `X` and `y` are provided, permutation importance is computed as a fallback

## Key API Reference

| Method                                                                   | Parameters                                                        | Returns                             |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------- |
| `viz.confusion_matrix(y_true, y_pred, labels=None)`                      | True labels, predicted labels, optional class names               | Plotly Figure (heatmap)             |
| `viz.roc_curve(y_true, y_scores, pos_label=1)`                           | True labels, probability scores                                   | Plotly Figure with AUC in title     |
| `viz.precision_recall_curve(y_true, y_scores, pos_label=1)`              | True labels, probability scores                                   | Plotly Figure with AP in title      |
| `viz.feature_importance(model, feature_names, top_n=20, X=None, y=None)` | Fitted model, feature name list, optional limit and fallback data | Plotly Figure (horizontal bar)      |
| `viz.learning_curve(model, X, y, cv=5, train_sizes=None)`                | Unfitted model, data, CV folds, training sizes                    | Plotly Figure (line chart)          |
| `viz.residuals(y_true, y_pred)`                                          | True values, predicted values                                     | Plotly Figure (scatter + histogram) |
| `viz.calibration_curve(y_true, y_proba, n_bins=10)`                      | True labels, predicted probabilities, bin count                   | Plotly Figure                       |
| `viz.metric_comparison(results)`                                         | Dict of model_name -> metric_dict                                 | Plotly Figure (grouped bars)        |
| `viz.training_history(metrics, x_label="Epoch")`                         | Dict of series_name -> values list                                | Plotly Figure (multi-line)          |

## Code Walkthrough

### Step 1: Create a ModelVisualizer

```python
from kailash_ml.engines.model_visualizer import ModelVisualizer

viz = ModelVisualizer()
```

### Step 2: Train a Classifier

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_cls, y_cls = make_classification(n_samples=200, n_features=5, random_state=42)
feature_names = [f"feat_{i}" for i in range(5)]
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.3)

clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
```

### Step 3: Confusion Matrix

```python
fig_cm = viz.confusion_matrix(y_test, y_pred)
assert fig_cm.layout.title.text == "Confusion Matrix"

# With custom labels
fig_cm_labeled = viz.confusion_matrix(y_test, y_pred, labels=["No", "Yes"])
```

### Step 4: ROC Curve

```python
fig_roc = viz.roc_curve(y_test, y_proba)
assert "AUC" in fig_roc.layout.title.text
```

### Step 5: Precision-Recall Curve

```python
fig_pr = viz.precision_recall_curve(y_test, y_proba)
assert "AP" in fig_pr.layout.title.text
```

### Step 6: Feature Importance

```python
fig_fi = viz.feature_importance(clf, feature_names)

# Limit to top 3
fig_fi_top3 = viz.feature_importance(clf, feature_names, top_n=3)

# Works with coefficient-based models too
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42, max_iter=200)
lr.fit(X_train, y_train)
fig_fi_lr = viz.feature_importance(lr, feature_names)
```

### Step 7: Learning Curve

```python
fig_lc = viz.learning_curve(
    RandomForestClassifier(n_estimators=10, random_state=42),
    X_train, y_train,
    cv=3,
    train_sizes=[0.3, 0.6, 1.0],
)
```

### Step 8: Residual Analysis (Regression)

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

X_reg, y_reg = make_regression(n_samples=100, n_features=3, noise=10.0)
reg = RandomForestRegressor(n_estimators=20, random_state=42)
reg.fit(X_reg[:80], y_reg[:80])
y_reg_pred = reg.predict(X_reg[80:])

fig_res = viz.residuals(y_reg[80:], y_reg_pred)
```

### Step 9: Metric Comparison

```python
comparison_results = {
    "RandomForest": {"accuracy": 0.95, "f1": 0.93, "precision": 0.94},
    "LogisticRegression": {"accuracy": 0.88, "f1": 0.85, "precision": 0.87},
    "GradientBoosting": {"accuracy": 0.97, "f1": 0.96, "precision": 0.95},
}

fig_mc = viz.metric_comparison(comparison_results)
assert fig_mc.layout.barmode == "group"
```

### Step 10: Training History

```python
history = {
    "train_loss": [0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08],
    "val_loss": [1.0, 0.8, 0.65, 0.55, 0.5, 0.48, 0.47, 0.46],
}

fig_th = viz.training_history(history)
assert fig_th.layout.xaxis.title.text == "Epoch"
```

## Common Mistakes

| Mistake                                    | What Happens                     | Fix                                                  |
| ------------------------------------------ | -------------------------------- | ---------------------------------------------------- |
| Mismatched feature_names length            | `ValueError`                     | Ensure `len(feature_names) == model.n_features_in_`  |
| Empty `metric_comparison({})`              | `ValueError`                     | Provide at least one model                           |
| Empty `training_history({})`               | `ValueError`                     | Provide at least one metric series                   |
| Model without importance attributes or X/y | `ValueError`                     | Pass `X` and `y` for permutation importance fallback |
| Forgetting to train the model first        | `AttributeError` on `.predict()` | Call `model.fit()` before visualization              |

## Exercises

1. **Confusion matrix analysis**: Train a classifier on imbalanced data (90/10 split). Generate the confusion matrix. What does it reveal that accuracy alone does not?

2. **ROC vs PR curves**: Generate both curves for the same model. Which curve is more informative when the positive class is rare (< 5% of samples)?

3. **Learning curve diagnosis**: Generate a learning curve. If the training score is much higher than the validation score, what does this indicate? What would you try next?

## Key Takeaways

- ModelVisualizer is stateless -- pass in data, get back a Plotly Figure.
- Eight visualization methods cover the most common ML evaluation needs.
- Feature importance supports three backends: tree-based, coefficient-based, and permutation-based.
- `metric_comparison()` provides a side-by-side view of multiple models for quick selection.
- `training_history()` visualizes training curves to diagnose overfitting.
- All methods validate inputs and raise clear errors for edge cases.

## Next Chapter

Chapter 4 covers **FeatureEngineer** -- the engine that generates candidate features (interactions, polynomials, binning) and selects the best subset using importance, correlation, or mutual information.
