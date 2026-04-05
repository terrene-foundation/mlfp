---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 3.6: DataFlow and Persistence

### Module 3: Supervised ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Define database models with `@db.model` decorators
- Perform CRUD operations with Kailash DataFlow
- Use `db.express` for quick queries
- Persist ML predictions and metadata to a database

---

## Recap: Lesson 3.5

- `WorkflowBuilder` creates reproducible ML pipelines as DAGs
- Nodes are steps; connections define data flow between them
- Branching enables parallel model comparison
- `runtime.execute(workflow.build())` runs the complete pipeline

---

## Why Persistence?

```
Without persistence:
  → Results lost when script ends
  → Cannot serve predictions via API
  → No audit trail of model outputs
  → Re-run everything from scratch each time

With DataFlow:
  → Results stored in database automatically
  → API-ready data (connects to Nexus in Lesson 3.8)
  → Full audit trail
  → Query historical predictions
```

---

## DataFlow: Zero-Config Database

```python
from kailash_dataflow import DataFlow

db = DataFlow()
db.configure(
    database="sqlite:///ascent03.db",  # SQLite for development
)
```

DataFlow auto-generates CRUD operations from model definitions. No SQL needed.

---

## Defining Models with `@db.model`

```python
from kailash_dataflow import DataFlow

db = DataFlow()
db.configure(database="sqlite:///ascent03.db")

@db.model
class HDBPrediction:
    town: str
    flat_type: str
    floor_area: float
    lease_years: int
    predicted_price: float
    confidence: float
    model_version: str
```

This creates a database table with all fields, plus auto-generated `id`, `created_at`, `updated_at`.

---

## CRUD Operations: Create

```python
# Single record
prediction = HDBPrediction(
    town="TAMPINES",
    flat_type="4 ROOM",
    floor_area=92.0,
    lease_years=72,
    predicted_price=485_000.0,
    confidence=0.85,
    model_version="v1.0",
)
db.save(prediction)

# Bulk create
predictions = [
    HDBPrediction(town=row["town"], flat_type=row["flat_type"], ...)
    for row in results
]
db.save_many(predictions)
```

---

## CRUD Operations: Read

```python
# Get by ID
pred = db.get(HDBPrediction, id=1)

# Get all
all_preds = db.get_all(HDBPrediction)

# Filter
tampines_preds = db.filter(
    HDBPrediction,
    town="TAMPINES",
    model_version="v1.0",
)

# Complex filter
expensive = db.filter(
    HDBPrediction,
    predicted_price__gt=500_000,
    confidence__gte=0.8,
)
```

---

## CRUD Operations: Update and Delete

```python
# Update
pred = db.get(HDBPrediction, id=1)
pred.confidence = 0.92
db.save(pred)

# Bulk update
db.update_many(
    HDBPrediction,
    filter={"model_version": "v0.9"},
    values={"model_version": "v1.0"},
)

# Delete
db.delete(HDBPrediction, id=1)

# Bulk delete
db.delete_many(HDBPrediction, model_version="v0.8")
```

---

## `db.express`: Quick Queries

```python
# Express queries for ad-hoc analysis
results = db.express(
    HDBPrediction,
    select=["town", "flat_type", "predicted_price"],
    where={"confidence__gte": 0.8},
    order_by="-predicted_price",
    limit=10,
)

# Aggregation
stats = db.express(
    HDBPrediction,
    select=["town"],
    aggregate={
        "avg_price": ("predicted_price", "mean"),
        "count": ("id", "count"),
    },
    group_by="town",
)
```

---

## Model Relationships

```python
@db.model
class MLModel:
    name: str
    version: str
    algorithm: str
    metrics: dict        # stored as JSON
    trained_at: str

@db.model
class HDBPrediction:
    town: str
    flat_type: str
    floor_area: float
    predicted_price: float
    confidence: float
    model: MLModel       # foreign key relationship
```

DataFlow handles joins automatically when querying related models.

---

## Persisting ML Results

```python
from kailash_ml import TrainingPipeline
from kailash_dataflow import DataFlow

# Train model
pipeline = TrainingPipeline()
pipeline.configure(dataset=df, target_column="price", algorithm="lightgbm")
result = pipeline.run()

# Persist model metadata
db = DataFlow()
db.configure(database="sqlite:///ascent03.db")

model_record = MLModel(
    name="hdb_price_predictor",
    version="v1.0",
    algorithm="lightgbm",
    metrics={
        "rmse": result.metrics["rmse"],
        "r2": result.metrics["r2"],
    },
    trained_at="2026-04-06",
)
db.save(model_record)
```

---

## Persisting Predictions

```python
import polars as pl

# Generate predictions
predictions = result.model.predict(X_test)

# Store each prediction
for i in range(len(X_test)):
    pred = HDBPrediction(
        town=test_df["town"][i],
        flat_type=test_df["flat_type"][i],
        floor_area=test_df["floor_area"][i],
        predicted_price=float(predictions[i]),
        confidence=float(confidences[i]),
        model=model_record,
    )
    db.save(pred)

print(f"Saved {len(predictions)} predictions to database")
```

---

## Querying Predictions with Polars

```python
# Load predictions back into Polars for analysis
all_preds = db.get_all(HDBPrediction)

df_preds = pl.DataFrame([
    {
        "town": p.town,
        "flat_type": p.flat_type,
        "predicted_price": p.predicted_price,
        "confidence": p.confidence,
    }
    for p in all_preds
])

# Analyse prediction distribution
print(df_preds.group_by("town").agg(
    pl.col("predicted_price").mean().alias("avg_prediction"),
    pl.col("confidence").mean().alias("avg_confidence"),
).sort("avg_prediction", descending=True))
```

---

## Database Selection

| Database       | When to Use                               |
| -------------- | ----------------------------------------- |
| **SQLite**     | Development, single-user, file-based      |
| **PostgreSQL** | Production, multi-user, concurrent access |

```python
# Development
db.configure(database="sqlite:///ascent03.db")

# Production
db.configure(database="postgresql://user:pass@localhost:5432/ascent")
```

DataFlow generates dialect-portable SQL -- switch databases without code changes.

---

## Exercise Preview

**Exercise 3.6: ML Prediction Database**

You will:

1. Define DataFlow models for ML models and predictions
2. Persist training results and batch predictions
3. Query predictions with `db.express` for analysis
4. Build a prediction audit trail with model versioning

Scaffolding level: **Moderate (~50% code provided)**

---

## Common Pitfalls

| Mistake                          | Fix                                         |
| -------------------------------- | ------------------------------------------- |
| Raw SQL instead of DataFlow      | Use `@db.model` and `db.express`            |
| Hardcoded database credentials   | Use environment variables                   |
| Not versioning model records     | Always store model version with predictions |
| Saving predictions one at a time | Use `db.save_many()` for bulk operations    |
| Forgetting to close connections  | DataFlow handles connection lifecycle       |

---

## Summary

- DataFlow provides zero-config database operations from model definitions
- `@db.model` creates tables; CRUD operations are auto-generated
- `db.express` handles ad-hoc queries with filtering and aggregation
- Persist ML model metadata and predictions for audit and serving
- SQLite for development; PostgreSQL for production

---

## Next Lesson

**Lesson 3.7: Model Registry and HPO**

We will learn:

- `HyperparameterSearch` for automated tuning
- `ModelRegistry` for versioned model management
- Comparing and promoting model candidates
