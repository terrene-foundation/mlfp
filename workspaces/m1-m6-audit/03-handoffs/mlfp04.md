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
