# Module 2: Feature Engineering & Experiment Design

**Kailash**: kailash-ml (FeatureStore, FeatureEngineer, ExperimentTracker) | **Scaffolding**: 60%

## Lecture (3h)
- **2A** Feature Engineering Theory: mutual information, Boruta, collinearity (VIF), temporal features, target encoding (James-Stein), domain-specific engineering
- **2B** Experiment Design & Causal Inference: A/B testing (power, bandits, sequential), CUPED, causal inference (Rubin/Pearl), diff-in-diff, propensity matching, Double ML
- **2C** Feature Management: FeatureSchema contracts, FeatureStore versioning, ExperimentTracker

## Lab (3h) — 5 Exercises
1. Healthcare feature engineering on messy ICU data (irregular vitals, point-in-time correctness)
2. FeatureStore lifecycle: define → compute → version → retrieve with leakage prevention
3. A/B test analysis: power analysis, SRM check, CUPED variance reduction
4. Causal inference: diff-in-diff on Singapore housing (cooling measures effect)
5. FeatureEngineer + ExperimentTracker: automated generation with experiment comparison

## Datasets
Healthcare ICU (MIMIC-style, 60K stays), E-commerce Experiment (500K users), Singapore Housing + Policy
