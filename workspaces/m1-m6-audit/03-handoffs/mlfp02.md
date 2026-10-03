# mlfp02 — handoffs from the exercise shard (S1, merged ec982293)

## Content corrections the teaching material must follow
- SRM: the experiment was designed 40/35/15/10 (control/treatment_a/treatment_b/variant_c) — test SRM against the DESIGNED allocation, never 50/50; variant_c is the mis-allocated arm; analyses use control vs treatment_a (passes its own SRM, p≈0.62). A failed SRM forces "DO NOT SHIP".
- Conversion = an order of at least $50 (≈19.1% control vs 24.8% treatment) — not `metric_value > 0` (~99%).
- Profile likelihood re-fits sigma at each mean value (differs from Wald).
- Tukey HSD = studentized range (cross-checked vs scipy.stats.tukey_hsd), not Bonferroni pairwise t-tests.
- Valuing a single flat needs a PREDICTION interval, not the CI of the mean.
- Peeking: teach the simulated ≈25% false-positive rate, not the "independent peeks" 64%.
- CUPED uses only pre-treatment covariates.
- ex_4/03 cashback tier LOSES ≈$140/customer/year across the CI (old "clearly profitable" conclusion reversed).
- AIC → prediction; BIC → recovering the true model.
- FeatureStore: use `kailash_ml.features.FeatureSchema` (the top-level `kailash_ml.FeatureSchema` is rejected by FeatureStore); new cooling-measures panel helper in shared/mlfp02.
- ex_8 validates data (removes 3,536 sentinel rows; R² 0.22 → 0.85).
- Real companies/regulators with invented figures anonymised; figures illustrative.

## Integration (S7)
- Regenerate all mlfp02 Colab notebooks.
- Execute every solution except ex_1/03. Heavy: ex_3/01, ex_3/04 (10K resamples × 354K rows); ex_8/01–03 (feature store, minutes each).

## Open questions for review
- ex_1/ex_2 loaders keep sentinel prices ($10, $9M) → price SD ≈ $420K; ex_1/01 uses percentile intervals instead. Decide whether ex_1/ex_2 should clean them (as ex_8 does) or keep them deliberately.
- ex_6/02 re-optimised threshold 0.24 vs theoretical break-even 0.017 — printed explanation is "miscalibration"; verify.

## Deferred spec gaps (S6)
- 2.1–2.3: Poisson; Exponential fit + AIC; LLN demo; parametric bootstrap; one-sample & one-tailed tests; likelihood curve (ex_1/03).
- 2.5–2.8: log(price) model; k-fold CV; lat/lon features; multinomial logit; employee-attrition data (ex_6); FeatureEngineer generate/select; DiD placebo test; capstone options + logistic model; real cooling-measure data; TrainingPipeline.
- ExperimentTracker in the Train phase of ex_1–3, 5, 6.

## Upstream (kailash-ml) notes
- Top-level `kailash_ml.FeatureSchema` rejected by FeatureStore; re-registering a schema name as version 2 fails.
