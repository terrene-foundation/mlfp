# mlfp04 — handoffs from the exercise shard (S1, merged 0cbde807)

## Content corrections the teaching material must follow
- ex_1–3: `churned` is an OUTCOME label — excluded from clustering/GMM/reduction features (shared/mlfp04 NON_FEATURE_COLS).
- ex_1.1: gap statistic with the 1-SE rule. ex_1.4: on two moons spectral is right (ARI 1.0) while silhouette prefers K-means. ex_1.5: AutoMLEngine.run actually searches algorithm × K; HDBSCAN required.
- ex_2.1: from-scratch EM matches sklearn GMM (log-lik gap ≈ 9e-9/sample); covariance-type nesting; corrected Mixtral figures.
- ex_3: reducers ranked by TRUSTWORTHINESS (not K-means silhouette in the embedding); corrected Levina–Bickel intrinsic-dimension formula (gives 8.0 on 8-D data); 4 UMAP configs; UMAP inter-cluster distances are NOT meaningful.
- ex_4: real credit-application rows (sg_credit_scoring) with injected, labelled anomalies; `ensemble_detect` + supervised `blend`/`stack` (blend() is a supervised voting ensemble, not anomaly-score blending); LOF n_neighbors=50 and the masking trap (LOF does NOT flag a tight dense fraud cluster).
- ex_5.4: target no longer a function of the baseline features (baseline AUC ≈ 0.69, not 1.0).
- ex_6: deduplicated AG News corpus (mlfp05/ag_news.parquet) + SST-2 human labels from Hugging Face; LDA fit on COUNTS (not TF-IDF), held-out perplexity; real word vectors; ex_6.4 needs env var TOPIC_EMBED_MODEL.
- ex_7: dataset 300×120 with brand-new items held out; baselines + validation-tuned hybrid with measured lift; content-based filtering does NOT solve brand-new USERS.
- ex_8: learnable synthetic shape images (not random noise); ex_8.5 exports torch via OnnxBridge(framework="torch") — OnnxBridge CAN export torch.
- Universal approximation: ONE hidden layer suffices (not "2+"). EM is not the backbone of MF/NN training.
- Invented company statistics anonymised; figures illustrative.

## Integration (S7)
- Regenerate mlfp04 notebooks ex_1–ex_8. `.env.example`: add TOPIC_EMBED_MODEL. Colab: AG News must be in the Drive mlfp05 folder; SST-2 via Hugging Face (network).
- README.md / index.html dataset tables: ex_4 sg_credit_scoring; ex_6 ag_news + SST-2; ex_7 300×120.
- Execute all 37 solutions (ex_1/01 … ex_8/05).

## Open decision (S6)
- local ex_7/05 still contains compact complete copies of the 01–04 algorithms (so it runs standalone) — that reveals answers to 01–04. Options: blank them, or move them into shared helpers students don't see first.

## Deferred spec gaps (S6)
- 4.6 TF-IDF from scratch; UMass coherence. 4.8 from-scratch numpy network with HDB price regression + loss/optimiser taxonomy.

## Upstream
- kailash-ml OnnxBridge prints UserWarning/FutureWarning during torch export.

---
# Additions from the deck shard (S2a, merged ad801ce5)
- Engines taught: ClusteringEngine (sweep_k/fit), AutoMLEngine search (agent=False, as ex_1.5), DimReductionEngine, AnomalyDetectionEngine (detect/ensemble_detect); EnsembleEngine blend/stack are SUPERVISED; OnnxBridge torch export with check_compatibility + onnxruntime parity.
- Deck assessment slide = four 25-mark auto-graded tasks (S5 redesign must update it).
- Deck "Looking Ahead" previews real Module 5 topics.
- Owner question: specs/module-4.md exercise wording (4.4 "financial transactions", 4.6 "Singapore news", 4.8 from-scratch HDB network) and "Quiz + project" describe PLANNED content — reconcile after S5/S6.
- Integration: regenerate deck.pdf + readings/deck.pdf; refresh parity baseline (M4 not in parity set). speaker-notes.md regenerate (D4).

---
# Additions from the lesson-slides shard (S2b, merged be9a6273)
- Lesson notes.html are now further out of step: lessons 01, 03, 04, 06, 08 gained slides with their own notes → S4 regenerate notes from slides. Open notes.html audit errors: stale APIs (03/04/06), "variational inference" (02), Gibbs-vs-VI (06), contamination (04), single-file exercise paths (01/04/05).
- lessons/04 + 07 textbook.html must follow: AnomalyDetectionEngine framing; content-based filtering does not solve new USERS.
- 4.1: ClusteringEngine on real customers K=3, silhouette 0.182. 4.3: 5 components → 90% variance, 6 → 95% (7 features). 4.4: LOF masking (ring AUC 0.22 at k=20 vs 0.89 at k=50); DriftMonitor is PSI/KS (M3.8), not an anomaly detector. 4.6: BERTopic visualize_heatmap; UMAP ~5 dims; embed model from TOPIC_EMBED_MODEL. 4.8: runnable numpy network on HDB (MSE 107 → 26.4).
- ModelVisualizer has NO scree helper — plot cumulative_variance with plotly.
- Note: 4.5 slide + ex_5 call mlxtend via `.to_pandas()` at the call boundary (no `import pandas`) — owner may want to rule on this vs the polars-only mandate.

---
# Additions from the textbook shard (S3a, merged 1af0941d)
- Lesson pages (S3b) align: 4.1 3,000-customer sample, K=3, silhouette 0.182; 4.3 5 comps → 92.2%, 6 → 98.0% (slides' "90%/95%" thresholds consistent); 4.4 credit-application benchmark; 4.6 AG News; 4.7 300×120 ratings, ALS RMSE 0.557.
- Honest results: from-scratch net ≈ linear regression (R² 0.860 both); equal-weight blend < LOF alone; Apriori faster than FP-Growth on this small data.
- BERTopic verified with TOPIC_EMBED_MODEL=all-MiniLM-L6-v2 (40 topics, 25% outliers).
