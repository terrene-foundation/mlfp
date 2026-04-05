# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 4 — AI-Resilient Assessment Questions

Unsupervised ML, NLP & Deep Learning
Covers: clustering, EM/GMM, PCA, EnsembleEngine, NLP (TF-IDF/BERTopic),
        DriftMonitor, deep learning, OnnxBridge
"""

QUIZ = {
    "module": "ASCENT4",
    "title": "Unsupervised ML, NLP & Deep Learning",
    "questions": [
        # ── Lesson 1: Clustering ──────────────────────────────────────────
        {
            "id": "4.1.1",
            "lesson": "4.1",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 1, you run K-means (k=5), GMM (k=5), and HDBSCAN on the "
                "e-commerce customer dataset. The silhouette scores are:\n\n"
                "  K-means:  0.41\n"
                "  GMM:      0.38\n"
                "  HDBSCAN:  0.52 (but 18% of points labeled as noise, cluster=-1)\n\n"
                "A marketing team needs 5 well-defined customer segments to run targeted campaigns. "
                "Which algorithm should you recommend and why?"
            ),
            "options": [
                "A) HDBSCAN — highest silhouette score means best clustering",
                "B) K-means — despite lower silhouette, it produces exactly 5 non-overlapping segments for every customer. HDBSCAN's 18% noise means 18% of customers have no segment and cannot be targeted, making it unsuitable for exhaustive campaign segmentation",
                "C) GMM — soft cluster assignments are better for marketing because customers can belong to multiple segments",
                "D) Increase HDBSCAN's min_cluster_size to eliminate noise points",
            ],
            "answer": "B",
            "explanation": (
                "HDBSCAN's silhouette is higher because it excludes hard-to-cluster points as noise. "
                "This is statistically better but operationally problematic — "
                "18% of customers cannot be assigned to any campaign. "
                "For exhaustive customer segmentation, K-means guarantees every customer gets a segment. "
                "The marketing team's requirement ('5 well-defined segments for every customer') "
                "is a hard operational constraint that overrides the silhouette comparison."
            ),
            "learning_outcome": "Select clustering algorithm based on operational constraints, not just silhouette score",
        },
        {
            "id": "4.1.2",
            "lesson": "4.1",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student tries to use AutoMLEngine for clustering but gets an error "
                "about a missing parameter. What is wrong?"
            ),
            "code": (
                "from kailash_ml import AutoMLEngine\n"
                "engine = AutoMLEngine()\n"
                "# Student calls without the required safety opt-in\n"
                "result = await engine.run(\n"
                "    data=customer_features,\n"
                "    task='clustering',\n"
                "    max_trials=20,\n"
                ")"
            ),
            "options": [
                "A) task='clustering' is not supported; use task='unsupervised'",
                "B) AutoMLEngine requires agent=True as an explicit opt-in when using the autonomous search mode — without it, the engine raises a configuration error to prevent accidental unattended LLM-driven trials",
                "C) AutoMLEngine cannot be awaited; use engine.run_sync() instead",
                "D) data must be a numpy array, not a Polars DataFrame",
            ],
            "answer": "B",
            "explanation": (
                "AutoMLEngine's autonomous mode (which uses an LLM to guide trial selection) "
                "requires explicit opt-in via agent=True. "
                "This is a deliberate safety mechanism — running autonomous model searches has cost "
                "and compute implications that must be acknowledged. "
                "Correct: await engine.run(data=customer_features, task='clustering', max_trials=20, agent=True)"
            ),
            "learning_outcome": "Use AutoMLEngine with the required agent=True opt-in for autonomous search",
        },
        # ── Lesson 2: EM and GMM ──────────────────────────────────────────
        {
            "id": "4.2.1",
            "lesson": "4.2",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "You fit a GMM with 3 components to customer revenue data. "
                "The fitted component means are S$180, S$520, and S$1,840 "
                "with mixing weights 0.62, 0.31, and 0.07. "
                "After E-step, customer A has responsibilities [0.71, 0.24, 0.05]. "
                "What does this mean practically, and what is the hard assignment "
                "that K-means would give this customer?"
            ),
            "options": [
                "A) The customer belongs to all three segments equally; K-means cannot handle this",
                "B) Customer A is most likely from the low-value segment (71% probability), with some chance of being a mid-value customer (24%). K-means hard assignment would place them in segment 1 (lowest mean), ignoring the 24% probability of mid-value membership",
                "C) The responsibilities sum to 1.0 which means the GMM has converged; the customer is in segment 3",
                "D) The 0.05 responsibility for the high-value segment means the customer is high-value",
            ],
            "answer": "B",
            "explanation": (
                "GMM responsibilities (posterior probabilities from the E-step) represent soft membership. "
                "Customer A is most likely low-value (0.71) but has meaningful mid-value probability (0.24). "
                "K-means assigns to the nearest centroid — here the S$180 centroid — "
                "discarding all uncertainty. "
                "GMM's soft assignment is useful when the 24% mid-value probability should influence "
                "which campaign tier the customer receives."
            ),
            "learning_outcome": "Interpret GMM responsibility values and contrast with K-means hard assignment",
        },
        # ── Lesson 3: PCA and dimensionality reduction ─────────────────────
        {
            "id": "4.3.1",
            "lesson": "4.3",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Your e-commerce dataset has 85 features after one-hot encoding. "
                "You want to visualise customer clusters in 2D. "
                "A colleague suggests PCA; you propose UMAP. "
                "Given that the data has non-linear manifold structure (confirmed by "
                "a t-SNE plot showing curved cluster boundaries), which is more appropriate "
                "and what is the operational trade-off?"
            ),
            "options": [
                "A) PCA — always use PCA before any visualisation to avoid overfitting",
                "B) UMAP for visualisation of non-linear structure — it preserves local neighbourhood relationships and handles curved manifolds. Trade-off: UMAP embeddings are not linearly interpretable (you cannot say 'UMAP dimension 1 = purchasing frequency'), while PCA components are linear combinations of original features",
                "C) Both are equivalent for 2D projection; use PCA because it is faster",
                "D) UMAP only works if clusters are spherical; use PCA for curved boundaries",
            ],
            "answer": "B",
            "explanation": (
                "PCA finds the directions of maximum variance assuming linear structure. "
                "When clusters lie on a curved manifold, PCA projects them into overlapping 2D blobs "
                "that look un-separated. "
                "UMAP (and t-SNE) optimise for preserving local neighbourhood relationships, "
                "making non-linear structure visible. "
                "The interpretability trade-off is real — stakeholders often want to know "
                "what each axis means, which PCA provides but UMAP does not."
            ),
            "learning_outcome": "Select dimensionality reduction method based on data manifold structure",
        },
        # ── Lesson 4: EnsembleEngine ──────────────────────────────────────
        {
            "id": "4.4.1",
            "lesson": "4.4",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "You run EnsembleEngine with method='stacking' on the credit scoring task. "
                "The base models are LightGBM (AUC=0.847), LogisticRegression (AUC=0.801), "
                "and XGBoost (AUC=0.839). The stacked ensemble achieves AUC=0.861. "
                "A colleague points out that the improvement is small (1.4%). "
                "Under what conditions is stacking NOT worth the added complexity?"
            ),
            "options": [
                "A) Stacking is never worth it — use the best single model",
                "B) Stacking adds minimal benefit when base models are highly correlated (similar errors on the same samples). If LightGBM and XGBoost make mistakes on the same customers, the meta-learner has nothing to correct — the 1.4% gain may not justify the inference latency, increased model complexity, and maintenance of three model artefacts",
                "C) Stacking only helps when all base models have similar AUC scores",
                "D) The 1.4% improvement is always worth it regardless of operational cost",
            ],
            "answer": "B",
            "explanation": (
                "Ensemble diversity is the key predictor of stacking benefit. "
                "LightGBM and XGBoost are both gradient boosting frameworks — "
                "they tend to make correlated errors. "
                "A meta-learner trained on top of correlated base models cannot learn much. "
                "In production, stacking triples inference time (three models instead of one), "
                "requires three model artefacts in the registry, and complicates SHAP explanations. "
                "If the diversity gain is low, a well-tuned single model is preferable."
            ),
            "learning_outcome": "Identify when stacking adds insufficient value due to base model correlation",
        },
        # ── Lesson 5: NLP, TF-IDF, BERTopic ─────────────────────────────
        {
            "id": "4.5.1",
            "lesson": "4.5",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 5, you compute TF-IDF on the Singapore news corpus. "
                "The word 'Singapore' appears in 94% of all documents and has IDF = 0.062. "
                "The word 'monetary' appears in 3% of documents and has IDF = 3.507. "
                "A document about MAS policy contains both words with TF = 0.08 each. "
                "What are the TF-IDF scores for each word in this document, "
                "and which is more discriminative?"
            ),
            "options": [
                "A) Both scores are identical because TF is the same; TF-IDF cannot distinguish them",
                "B) Singapore: 0.08 × 0.062 = 0.005; monetary: 0.08 × 3.507 = 0.281. 'monetary' is far more discriminative — its high IDF reflects that it appears in very few documents, so high TF-IDF signals a document genuinely about monetary policy",
                "C) Singapore: 0.062; monetary: 3.507. TF is not multiplied in — IDF alone determines importance",
                "D) Singapore TF-IDF = 0; words appearing in >90% of documents are stopwords",
            ],
            "answer": "B",
            "explanation": (
                "TF-IDF = TF × IDF. Both words have the same TF (0.08 occurrences per token). "
                "'Singapore' is near-universal so IDF ≈ log(1/0.94) ≈ 0.062 — tiny. "
                "'monetary' is rare: IDF ≈ log(1/0.03) ≈ 3.507 — large. "
                "In this document, 'monetary' has TF-IDF 56× higher than 'Singapore', "
                "correctly marking the document as being about monetary policy rather than "
                "just being a Singapore-origin document."
            ),
            "learning_outcome": "Compute and interpret TF-IDF values to identify discriminative terms",
        },
        {
            "id": "4.5.2",
            "lesson": "4.5",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You have a corpus of 50,000 Singapore news articles and want to discover "
                "topics. Your compute budget allows a single GPU for 2 hours. "
                "Exercise 5 covers both NMF (on TF-IDF) and BERTopic. "
                "Which is more appropriate given the budget, and what does BERTopic "
                "require that NMF does not?"
            ),
            "options": [
                "A) NMF — always faster; BERTopic is only for academic research",
                "B) With a 2-hour GPU budget, BERTopic is feasible — it requires a pretrained sentence transformer (SentenceTransformer) for embeddings, which NMF does not. NMF on TF-IDF runs on CPU in minutes but produces topics as word bags. BERTopic captures semantic similarity and discovers more coherent topics from 50,000 articles",
                "C) BERTopic requires 10+ GPUs; NMF is the only practical choice",
                "D) NMF is better because BERTopic topics are not interpretable",
            ],
            "answer": "B",
            "explanation": (
                "NMF factors a TF-IDF matrix — purely algebraic, runs on CPU, topics are word frequency vectors. "
                "BERTopic uses a sentence transformer to embed documents into a semantic space, "
                "then clusters with UMAP + HDBSCAN, and extracts topics using c-TF-IDF. "
                "The sentence transformer is the key additional requirement — "
                "it needs a pretrained model (e.g., all-MiniLM-L6-v2) downloaded to the local machine. "
                "On a single GPU, embedding 50,000 short news articles takes ~15-30 minutes, "
                "well within the 2-hour budget."
            ),
            "learning_outcome": "Identify BERTopic's sentence transformer requirement vs NMF's TF-IDF input",
        },
        # ── Lesson 6: DriftMonitor ────────────────────────────────────────
        {
            "id": "4.6.1",
            "lesson": "4.6",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Running DriftMonitor in Exercise 6 produces this alert for the credit model:\n\n"
                "  DRIFT ALERT [CRITICAL]\n"
                "  feature: debt_to_income_ratio\n"
                "  PSI: 0.42\n"
                "  KS statistic: 0.31, KS p-value: 0.0001\n"
                "  drift_type: feature_shift\n\n"
                "The model was trained in 2023. It is now mid-2026. "
                "What action should you take immediately, and what is the governance implication "
                "under MAS model risk management guidelines?"
            ),
            "options": [
                "A) No action needed; PSI < 0.5 is within normal range",
                "B) PSI = 0.42 indicates severe drift (>0.25 is the critical threshold). Immediately shadow the production model with a retrained version on 2026 data. Under MAS model risk management, a drift alert at this severity triggers mandatory model revalidation — continued use without revalidation exposes the bank to regulatory sanction",
                "C) Increase the PSI threshold in DriftSpec to 0.5 to suppress the alert",
                "D) PSI measures concept drift only; feature drift does not require retraining",
            ],
            "answer": "B",
            "explanation": (
                "PSI interpretation: < 0.1 = no drift, 0.1–0.25 = moderate (investigate), > 0.25 = severe (retrain). "
                "PSI = 0.42 is well above the critical threshold — debt_to_income_ratio distribution "
                "has shifted significantly since 2023, likely due to rising interest rates. "
                "A model trained on 2023 data may now incorrectly score 2026 borrowers. "
                "MAS MRM guidelines (MAS Notice 644) require models to be revalidated when material "
                "data or environment changes occur — this drift event qualifies."
            ),
            "learning_outcome": "Interpret PSI thresholds and trigger appropriate model revalidation governance",
        },
        {
            "id": "4.6.2",
            "lesson": "4.6",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student sets up DriftMonitor but the async context manager raises TypeError. "
                "What is wrong?"
            ),
            "code": (
                "async def setup_monitoring():\n"
                "    conn = ConnectionManager('sqlite:///ascent04_drift.db')\n"
                "    await conn.initialize()\n"
                "\n"
                "    monitor = DriftMonitor(conn)\n"
                "    spec = DriftSpec(psi_threshold=0.2, ks_alpha=0.05)\n"
                "\n"
                "    # Student tries to use tracker without async context manager\n"
                "    await monitor.set_reference(reference_data, spec=spec)\n"
                "    result = await monitor.check(current_data)\n"
                "    print(result.alerts)"
            ),
            "options": [
                "A) DriftMonitor must be imported from kailash_ml.engines, not kailash_ml",
                "B) DriftMonitor requires 'async with monitor:' context manager before calling set_reference() and check() — without it, the monitor's internal connection is not initialised, causing TypeError on the first method call",
                "C) set_reference() is a synchronous method and should not be awaited",
                "D) DriftSpec must be created inside the async context manager",
            ],
            "answer": "B",
            "explanation": (
                "DriftMonitor follows the async context manager pattern like ExperimentTracker and FeatureStore. "
                "The correct usage is:\n"
                "async with monitor:\n"
                "    await monitor.set_reference(reference_data, spec=spec)\n"
                "    result = await monitor.check(current_data)\n"
                "Without 'async with', the monitor's __aenter__ is never called, "
                "leaving internal state uninitialised."
            ),
            "learning_outcome": "Use DriftMonitor with the required async context manager pattern",
        },
        # ── Lesson 7: Deep learning, OnnxBridge ───────────────────────────
        {
            "id": "4.7.1",
            "lesson": "4.7",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "In Exercise 7, you train a CNN with cosine annealing LR schedule. "
                "The training loss curve shows:\n\n"
                "  Epoch 1-5:   loss drops from 0.82 to 0.41 (fast descent)\n"
                "  Epoch 6-20:  loss oscillates between 0.38 and 0.44 (rising LR phases)\n"
                "  Epoch 21-30: loss stabilises at 0.31 (final cycle)\n\n"
                "A student says the oscillations in epochs 6-20 indicate training instability. "
                "Why are they wrong, and what is the purpose of the LR rises in cosine annealing?"
            ),
            "options": [
                "A) The student is right — oscillating loss means the LR is too high and should be reduced",
                "B) The oscillations are expected: cosine annealing cyclically increases LR from near-zero back to a warm value, helping the model escape local minima. The loss rises when LR rises (exploration) then falls as LR decreases (exploitation). The final cycle stabilises at 0.31, lower than the 0.41 achieved without restarts — confirming the approach worked",
                "C) Oscillations are a numerical precision issue; switch to Adam optimiser",
                "D) Cosine annealing only has one cycle; multiple oscillations indicate a bug",
            ],
            "answer": "B",
            "explanation": (
                "Cosine Annealing with Warm Restarts (SGDR) deliberately raises the LR at restart points. "
                "The high-LR phase allows the model to jump out of a sharp local minimum into a broader basin. "
                "The loss rising during the LR increase is the model 'exploring' — trading short-term performance "
                "for the chance to find a flatter (more generalisable) minimum. "
                "The lower final loss (0.31 vs 0.41) confirms the exploration was beneficial."
            ),
            "learning_outcome": "Interpret cosine annealing loss curves and distinguish exploration vs instability",
        },
        {
            "id": "4.7.2",
            "lesson": "4.7",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "A student exports a model using OnnxBridge but gets mismatched predictions "
                "when comparing ONNX vs PyTorch outputs. What is the likely cause and how "
                "should Exercise 7's validation step detect this?"
            ),
            "code": (
                "from kailash_ml.bridge.onnx_bridge import OnnxBridge\n"
                "\n"
                "bridge = OnnxBridge()\n"
                "bridge.export(\n"
                "    model=cnn_model,\n"
                "    path='credit_cnn.onnx',\n"
                "    input_shape=(1, 4, 32, 32),  # wrong shape for a tabular model\n"
                ")\n"
                "# Validation: only checks that the file was created\n"
                "import os\n"
                "assert os.path.exists('credit_cnn.onnx')"
            ),
            "options": [
                "A) OnnxBridge does not support CNN models; only linear models can be exported",
                "B) input_shape=(1, 4, 32, 32) is a 4D image shape but the tabular credit model takes 1D feature vectors — the wrong shape causes the ONNX graph to reshape/pad incorrectly. The validation must compare actual numerical outputs: run the same batch through both PyTorch and ONNX, then assert np.allclose(pytorch_out, onnx_out, atol=1e-5)",
                "C) The file existence check is sufficient; ONNX format guarantees bit-identical outputs",
                "D) OnnxBridge export requires opset_version=11 explicitly; the default produces wrong outputs",
            ],
            "answer": "B",
            "explanation": (
                "input_shape tells OnnxBridge how to trace the model's computation graph. "
                "An incorrect shape means the traced graph has the wrong tensor reshape operations. "
                "The validation in Exercise 7 explicitly runs a sample batch through both "
                "the PyTorch model (model(x_test)) and the ONNX runtime (ort_session.run(...)) "
                "and asserts np.allclose() with atol=1e-5. "
                "File existence is not a functional validation — a corrupt ONNX file still exists."
            ),
            "learning_outcome": "Validate ONNX export correctness by comparing numerical outputs, not file existence",
        },
        # ── Lesson 8: InferenceServer, deployment ─────────────────────────
        {
            "id": "4.8.1",
            "lesson": "4.8",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You need to serve the ONNX credit scoring model via REST API with "
                "sub-10ms p99 latency and real-time drift monitoring. "
                "Which kailash-ml engine serves inference, and which engine monitors the "
                "incoming prediction requests for drift? Name both engines and one config "
                "parameter unique to each."
            ),
            "options": [
                "A) TrainingPipeline (serves inference) + AutoMLEngine (detects drift); TrainingPipeline has batch_size, AutoMLEngine has max_trials",
                "B) InferenceServer (serves inference, config: cache_predictions=True for repeated inputs) + DriftMonitor (monitors production data, config: DriftSpec(psi_threshold=0.2)); they share the same ConnectionManager for coordinated alerting",
                "C) ModelRegistry (serves inference) + ExperimentTracker (monitors drift)",
                "D) Nexus (serves inference) + FeatureStore (monitors drift via feature retrieval counts)",
            ],
            "answer": "B",
            "explanation": (
                "InferenceServer wraps a trained model (including ONNX models via OnnxBridge) "
                "and exposes it as a callable with optional prediction caching. "
                "DriftMonitor compares incoming production feature distributions against "
                "the reference (training) distribution using PSI and KS tests. "
                "Sharing a ConnectionManager lets DriftMonitor store alert records "
                "alongside inference logs in the same database, enabling correlation analysis."
            ),
            "learning_outcome": "Identify InferenceServer and DriftMonitor as the production serving pair",
        },
    ],
}

if __name__ == "__main__":
    for q in QUIZ["questions"]:
        print(f"\n{'=' * 60}")
        print(f"[{q['id']}] ({q['type']}) — Lesson {q['lesson']}  [{q['difficulty']}]")
        print(f"{'=' * 60}")
        print(q["question"])
        if q.get("code"):
            print(f"\n```python\n{q['code']}\n```")
        for opt in q["options"]:
            print(f"  {opt}")
        print(f"\nAnswer: {q['answer']}")
