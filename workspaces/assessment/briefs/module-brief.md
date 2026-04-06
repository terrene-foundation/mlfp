# Assessment Design Brief

## Structure

| Component | Weight | Format |
|-----------|--------|--------|
| Module Quizzes (6) | 20% | 15 questions each: theory + code + interpretation |
| Individual Portfolio | 35% | Extend one module exercise to production depth with model card |
| Team Capstone | 35% | Multi-framework production system using 3+ Kailash packages |
| Peer Review | 10% | Code review of another team's capstone (SHAP analysis, governance audit) |

## Module Quizzes

Each module has a quiz in `modules/ascentNN/quiz/` with 10-15 questions:
- 5-7 multiple choice (pattern recognition, framework selection)
- 3-5 short code (complete the import, debug this code)
- 2-3 open interpretation (what does this output mean?)

Quizzes test Kailash SDK understanding, not generic ML theory. See `rules/domain-integrity.md`.

## Individual Portfolio

Students select any module's dataset and extend to production-ready system:
- Must include: EDA report (DataExplorer), feature engineering rationale, model comparison (≥3 approaches), SHAP interpretability, calibration analysis, drift monitoring setup, model card
- Must demonstrate full Kailash engine usage for the chosen module's scope

Graded on: statistical rigor (25%), Kailash pattern mastery (25%), production readiness (25%), documentation quality (25%).

## Team Capstone

Teams of 3-4, domain of choice:
- Must use ≥3 Kailash packages
- Must include: data pipeline (DataFlow), model lifecycle (ModelRegistry), deployment (Nexus), governance (PACT) OR agents (Kaizen)
- 15-minute live demo + 10-minute Q&A
- Deliverables: working system, architecture doc, model cards, governance specification

Graded on: system integration (25%), SDK pattern mastery (25%), working demo (25%), team presentation (25%).

## Assessment Datasets

The `ascent_assessment/` folder contains an integrated Singapore urban planning dataset:
- `resi_sales_5y.parquet` — 5 years of residential property sales
- `places.parquet` — Points of interest
- `places_reviews.parquet` — Review text data
- `places_images.parquet` — Image metadata
- `mrt_location_data.parquet` — MRT station locations
- `schools_popularity.parquet` — School proximity and demand

Students must combine these tables (requires joins, cleaning, feature engineering across sources) for their individual/group assessments.
