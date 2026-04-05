---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 4.8: Capstone — InferenceServer

### Module 4: Advanced ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Deploy trained models with `InferenceServer`
- Build an end-to-end pipeline from training to serving
- Monitor deployed models with `DriftMonitor`
- Complete the Module 4 capstone project

---

## Recap: Lesson 4.7

- PyTorch: tensors, autograd, nn.Module for building networks
- CNNs learn spatial patterns; ResBlocks enable deep networks
- OnnxBridge exports models for production serving
- Use deep learning for images/text; gradient boosting for tabular

---

## Module 4 Journey

```
4.1  Clustering: K-means, HDBSCAN, AutoMLEngine
4.2  EM algorithm, GMMs, soft clustering
4.3  Dimensionality reduction: PCA, UMAP, t-SNE
4.4  Anomaly detection, EnsembleEngine
4.5  NLP: TF-IDF, BERTopic
4.6  DriftMonitor, PSI
4.7  Deep learning: PyTorch, CNN, OnnxBridge
4.8  InferenceServer capstone  ← YOU ARE HERE
```

---

## InferenceServer: Serving Predictions

```python
from kailash_ml import InferenceServer, ModelRegistry

# Load production model
registry = ModelRegistry()
registry.configure(storage_path="./model_registry")

server = InferenceServer()
server.configure(
    model=registry.load("hdb_price_predictor", stage="production"),
    host="0.0.0.0",
    port=8080,

    # Input validation
    input_schema={
        "town": "str",
        "flat_type": "str",
        "floor_area": "float",
        "lease_years": "int",
    },
)

server.start()
```

---

## Making Predictions

```python
# HTTP API (auto-generated)
# POST /predict
# {
#   "town": "TAMPINES",
#   "flat_type": "4 ROOM",
#   "floor_area": 92.0,
#   "lease_years": 72
# }

# Python client
from kailash_ml import InferenceClient

client = InferenceClient(url="http://localhost:8080")

result = client.predict({
    "town": "TAMPINES",
    "flat_type": "4 ROOM",
    "floor_area": 92.0,
    "lease_years": 72,
})
print(f"Predicted price: ${result.prediction:,.0f}")
print(f"Confidence: {result.confidence:.2f}")
```

---

## Batch Predictions

```python
# Predict on a DataFrame
import polars as pl

new_listings = pl.DataFrame({
    "town": ["TAMPINES", "BEDOK", "WOODLANDS"],
    "flat_type": ["4 ROOM", "3 ROOM", "5 ROOM"],
    "floor_area": [92.0, 67.0, 110.0],
    "lease_years": [72, 55, 90],
})

results = server.predict_batch(new_listings)
print(results)
```

```
┌───────────┬───────────┬────────────────┬────────────┐
│ town      ┆ flat_type ┆ predicted_price┆ confidence │
│ TAMPINES  ┆ 4 ROOM   ┆ 485,000        ┆ 0.87       │
│ BEDOK     ┆ 3 ROOM   ┆ 320,000        ┆ 0.82       │
│ WOODLANDS ┆ 5 ROOM   ┆ 510,000        ┆ 0.85       │
└───────────┴───────────┴────────────────┴────────────┘
```

---

## Preprocessing in the Server

```python
server.configure(
    model=production_model,

    # Preprocessing applied before prediction
    preprocessor=preprocessing_pipeline,
    feature_engineer=feature_pipeline,

    # Postprocessing applied after prediction
    postprocessor={
        "round_to": 1000,       # round to nearest $1,000
        "add_confidence": True,  # include confidence interval
    },
)
```

The server handles the full pipeline: raw input to final prediction.

---

## Monitoring the Server

```python
from kailash_ml import DriftMonitor

monitor = DriftMonitor()
monitor.configure(
    reference_data=df_train,
    target_column="price",
    psi_threshold=0.15,
)

# Attach monitor to server
server.configure(
    model=production_model,
    drift_monitor=monitor,
    monitoring_interval="1h",  # check every hour
    alert_callback=send_alert,
)
```

---

## Health and Metrics

```python
# Built-in endpoints
# GET /health     → {"status": "healthy", "model_version": "1.0.0"}
# GET /metrics    → request count, latency p50/p95/p99, error rate
# GET /drift      → latest drift report

# Programmatic access
health = server.health()
print(f"Status: {health.status}")
print(f"Requests served: {health.total_requests}")
print(f"Avg latency: {health.avg_latency_ms:.0f}ms")
print(f"Error rate: {health.error_rate:.2%}")
```

---

## Full Production Architecture

```
┌──────────┐    ┌──────────────┐    ┌─────────────────┐
│ Feature  │───→│  Inference   │───→│   DataFlow      │
│ Store    │    │  Server      │    │   (persist)     │
└──────────┘    └──────────────┘    └─────────────────┘
                      │    ↑
                      │    │
                ┌─────▼────┴──────┐
                │  Drift Monitor  │
                │  (hourly check) │
                └────────┬────────┘
                         │ alert
                         ▼
                ┌─────────────────┐
                │  Retrain        │
                │  Pipeline       │
                └─────────────────┘
```

---

## Capstone Project Overview

**Project: End-to-End Advanced ML Platform**

```
1. Unsupervised Analysis (4.1-4.3)
   └→ Cluster HDB markets, reduce dimensions, visualise

2. Quality & Ensembles (4.4-4.5)
   └→ Anomaly detection, NLP topics, ensemble models

3. Monitoring & Serving (4.6-4.8)
   └→ DriftMonitor, InferenceServer, production deployment
```

---

## Capstone Deliverables

| Deliverable         | Components                                        |
| ------------------- | ------------------------------------------------- |
| Market segmentation | K-means + HDBSCAN + GMM comparison                |
| Anomaly detection   | Isolation Forest flagging suspicious transactions |
| NLP features        | BERTopic topics from listing descriptions         |
| Ensemble model      | Stacking ensemble via EnsembleEngine              |
| Deployed server     | InferenceServer with preprocessing                |
| Monitoring          | DriftMonitor with alerting thresholds             |

---

## Exercise Preview

**Exercise 4.8: Module 4 Capstone**

You will:

1. Cluster HDB markets and detect anomalies
2. Extract NLP features and build an ensemble model
3. Deploy with `InferenceServer` and attach `DriftMonitor`
4. Test the full pipeline: request to prediction to persistence

Scaffolding level: **Light+ (~40% code provided)**

---

## Common Pitfalls

| Mistake                                  | Fix                                          |
| ---------------------------------------- | -------------------------------------------- |
| Serving raw model without preprocessing  | Include preprocessor in InferenceServer      |
| No input validation                      | Define input_schema for type checking        |
| Ignoring latency requirements            | Profile with /metrics; optimise if p99 > SLA |
| No drift monitoring on deployed model    | Always attach DriftMonitor                   |
| Different code paths for batch vs single | InferenceServer handles both consistently    |

---

## Module 4 Summary

| Lesson | Key Skills                          |
| ------ | ----------------------------------- |
| 4.1    | K-means, HDBSCAN, AutoMLEngine      |
| 4.2    | EM algorithm, GMMs, soft clustering |
| 4.3    | PCA, UMAP, t-SNE                    |
| 4.4    | Anomaly detection, EnsembleEngine   |
| 4.5    | TF-IDF, BERTopic, NLP features      |
| 4.6    | DriftMonitor, PSI                   |
| 4.7    | PyTorch, CNN, ResBlock, OnnxBridge  |
| 4.8    | InferenceServer, capstone           |

---

## What Comes Next

**Module 5: LLMs and Agents**

- LLM fundamentals with Signature and Delegate
- Chain-of-thought and ReAct agents
- RAG systems and MCP servers
- Multi-agent orchestration and production deployment

You can now build and deploy ML systems. Next, we add intelligence with LLMs.
