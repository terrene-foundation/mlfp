---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 4.6: Drift Monitoring

### Module 4: Advanced ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain data drift, concept drift, and prediction drift
- Compute Population Stability Index (PSI) for distribution shifts
- Build automated monitoring with `DriftMonitor`
- Design alerting thresholds and retraining triggers

---

## Recap: Lesson 4.5

- Text preprocessing: tokenise, remove stopwords, TF-IDF vectorisation
- BERTopic discovers topics using embeddings + UMAP + HDBSCAN
- Topics become categorical features for ML models
- Combining text and tabular features outperforms either alone

---

## Why Models Decay

```
Model trained on 2024 data. Deployed. Working great.

6 months later:
  - New BTO towns launched (unseen categories)
  - Interest rates changed (shifted price distribution)
  - Government cooling measures (new relationship patterns)
  - Model still predicts using 2024 patterns → errors increase

Models do not "break" — the world changes around them.
```

---

## Three Types of Drift

```
DATA DRIFT:      Input distribution changes
  "New data looks different from training data"
  Example: more large flats in recent data

CONCEPT DRIFT:   Relationship between inputs and target changes
  "Same inputs, different outcomes"
  Example: same flat type costs more due to policy change

PREDICTION DRIFT: Model output distribution shifts
  "Model is predicting differently"
  Example: predictions skewing higher over time
```

---

## Population Stability Index (PSI)

PSI measures how much a distribution has shifted.

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

Interpretation:
  PSI < 0.1:   No significant shift
  PSI 0.1-0.2: Moderate shift — investigate
  PSI > 0.2:   Significant shift — retrain

Computed per feature: bin the feature, compare bin proportions
between training (expected) and production (actual) data.
```

---

## PSI Calculation

```python
import numpy as np

def compute_psi(expected, actual, bins=10):
    """Compute PSI between two distributions."""
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[-1] = np.inf
    breakpoints[0] = -np.inf

    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 0.001, None)
    actual_pct = np.clip(actual_pct, 0.001, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi
```

---

## DriftMonitor: Automated Detection

```python
from kailash_ml import DriftMonitor

monitor = DriftMonitor()
monitor.configure(
    reference_data=df_train,
    target_column="price",

    # What to monitor
    feature_drift=True,
    prediction_drift=True,
    performance_drift=True,

    # Thresholds
    psi_threshold=0.15,
    performance_threshold=0.10,  # 10% degradation triggers alert
)

# Check new data
report = monitor.check(new_data=df_production)
```

---

## Reading the Drift Report

```python
print(f"Overall status: {report.status}")  # OK / WARNING / CRITICAL

# Feature-level drift
for feature in report.feature_drift:
    if feature.drifted:
        print(f"DRIFT: {feature.name} "
              f"(PSI={feature.psi:.3f}, "
              f"threshold={feature.threshold:.3f})")

# Prediction drift
print(f"Prediction PSI: {report.prediction_drift.psi:.3f}")

# Performance drift (if labels available)
if report.performance_drift:
    print(f"RMSE change: {report.performance_drift.metric_change:+.1%}")
```

---

## Monitoring Dashboard Pattern

```
┌─────────────────────────────────────────────────┐
│  HDB Price Model — Drift Monitor                │
├─────────────────────────────────────────────────┤
│                                                  │
│  Feature Drift (PSI)          Status             │
│  ────────────────────────────────────            │
│  floor_area:   0.04  ████░░░░░░  OK             │
│  lease_years:  0.08  ████████░░  OK             │
│  town:         0.18  ██████████  ⚠ WARNING      │
│  storey:       0.03  ███░░░░░░░  OK             │
│                                                  │
│  Prediction PSI: 0.12  ⚠ WARNING                │
│  Model RMSE:     +8.3%  ⚠ INVESTIGATE           │
│                                                  │
│  Last checked: 2026-04-06 10:30 SGT             │
└─────────────────────────────────────────────────┘
```

---

## Automated Monitoring Pipeline

```python
from kailash_ml import DriftMonitor, ModelRegistry

monitor = DriftMonitor()
monitor.configure(
    reference_data=df_train,
    target_column="price",
    psi_threshold=0.15,
)

def monitoring_check(new_data):
    report = monitor.check(new_data)

    if report.status == "CRITICAL":
        # Trigger retraining
        print("CRITICAL drift detected — triggering retrain")
        # ... retrain pipeline ...
    elif report.status == "WARNING":
        print("WARNING — investigate drift sources")
        for f in report.feature_drift:
            if f.drifted:
                print(f"  {f.name}: PSI={f.psi:.3f}")
    else:
        print("OK — no significant drift")

    return report
```

---

## Windowed Monitoring

```python
# Monitor drift over time windows
monitor.configure(
    reference_data=df_train,
    target_column="price",
    window_size="30d",          # check in 30-day windows
    min_samples=100,            # need at least 100 samples per window
)

# Check multiple windows
windows = monitor.check_windowed(
    data=df_production,
    date_column="transaction_date",
)

for window in windows:
    print(f"{window.start} to {window.end}: "
          f"PSI={window.psi:.3f} ({window.status})")
```

---

## Drift Response Strategy

```
PSI < 0.1:    No action. Log and continue.

PSI 0.1-0.2:  Investigate.
              → Which features drifted?
              → Is performance actually affected?
              → Document findings.

PSI > 0.2:    Retrain.
              → Collect new labelled data
              → Retrain with updated distribution
              → Validate before promoting to production
              → Update reference data in DriftMonitor
```

---

## Exercise Preview

**Exercise 4.6: Production Drift Monitoring**

You will:

1. Compute PSI manually and verify against `DriftMonitor`
2. Simulate data drift and observe detection
3. Build an automated monitoring pipeline with alerts
4. Design a retraining trigger workflow

Scaffolding level: **Light+ (~40% code provided)**

---

## Common Pitfalls

| Mistake                                           | Fix                                                  |
| ------------------------------------------------- | ---------------------------------------------------- |
| Monitoring only predictions, not features         | Feature drift is an early warning signal             |
| PSI threshold too tight (< 0.05)                  | Start at 0.1-0.2; tighten with experience            |
| No minimum sample size per window                 | Small samples produce noisy PSI estimates            |
| Retraining on every minor drift                   | Investigate first; only retrain for real degradation |
| Forgetting to update reference data after retrain | New model needs new reference distribution           |

---

## Summary

- Models decay as the world changes: data drift, concept drift, prediction drift
- PSI measures distribution shift: < 0.1 OK, 0.1-0.2 investigate, > 0.2 retrain
- `DriftMonitor` automates feature, prediction, and performance monitoring
- Windowed monitoring tracks drift over time
- Design clear response strategies: ignore, investigate, or retrain

---

## Next Lesson

**Lesson 4.7: Deep Learning**

We will learn:

- PyTorch fundamentals: tensors, autograd, nn.Module
- CNNs and ResBlocks for image data
- OnnxBridge for model export and serving
