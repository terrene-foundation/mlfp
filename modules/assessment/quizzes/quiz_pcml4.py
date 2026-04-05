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
                "B) GMM — soft cluster assignments are better for marketing because customers can belong to multiple segments",
                "C) K-means — despite lower silhouette, it produces exactly 5 non-overlapping segments for every customer. HDBSCAN's 18% noise means 18% of customers have no segment and cannot be targeted, making it unsuitable for exhaustive campaign segmentation",
                "D) Increase HDBSCAN's min_cluster_size to eliminate noise points"
            ],
            "answer": "C",
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
                "B) AutoMLEngine cannot be awaited; use engine.run_sync() instead",
                "C) AutoMLEngine requires agent=True as an explicit opt-in when using the autonomous search mode — without it, the engine raises a configuration error to prevent accidental unattended LLM-driven trials",
                "D) data must be a numpy array, not a Polars DataFrame"
            ],
            "answer": "C",
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
                "A) Customer A is most likely from the low-value segment (71% probability), with some chance of being a mid-value customer (24%). K-means hard assignment would place them in segment 1 (lowest mean), ignoring the 24% probability of mid-value membership",
                "B) The customer belongs to all three segments equally; K-means cannot handle this",
                "C) The responsibilities sum to 1.0 which means the GMM has converged; the customer is in segment 3",
                "D) The 0.05 responsibility for the high-value segment means the customer is high-value"
            ],
            "answer": "A",
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
                "A) UMAP for visualisation of non-linear structure — it preserves local neighbourhood relationships and handles curved manifolds. Trade-off: UMAP embeddings are not linearly interpretable (you cannot say 'UMAP dimension 1 = purchasing frequency'), while PCA components are linear combinations of original features",
                "B) PCA — always use PCA before any visualisation to avoid overfitting",
                "C) Both are equivalent for 2D projection; use PCA because it is faster",
                "D) UMAP only works if clusters are spherical; use PCA for curved boundaries"
            ],
            "answer": "A",
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
                "B) The 1.4% improvement is always worth it regardless of operational cost",
                "C) Stacking only helps when all base models have similar AUC scores",
                "D) Stacking adds minimal benefit when base models are highly correlated (similar errors on the same samples). If LightGBM and XGBoost make mistakes on the same customers, the meta-learner has nothing to correct — the 1.4% gain may not justify the inference latency, increased model complexity, and maintenance of three model artefacts"
            ],
            "answer": "D",
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
                "B) Singapore TF-IDF = 0; words appearing in >90% of documents are stopwords",
                "C) Singapore: 0.062; monetary: 3.507. TF is not multiplied in — IDF alone determines importance",
                "D) Singapore: 0.08 × 0.062 = 0.005; monetary: 0.08 × 3.507 = 0.281. 'monetary' is far more discriminative — its high IDF reflects that it appears in very few documents, so high TF-IDF signals a document genuinely about monetary policy"
            ],
            "answer": "D",
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
                "B) BERTopic requires 10+ GPUs; NMF is the only practical choice",
                "C) With a 2-hour GPU budget, BERTopic is feasible — it requires a pretrained sentence transformer (SentenceTransformer) for embeddings, which NMF does not. NMF on TF-IDF runs on CPU in minutes but produces topics as word bags. BERTopic captures semantic similarity and discovers more coherent topics from 50,000 articles",
                "D) NMF is better because BERTopic topics are not interpretable"
            ],
            "answer": "C",
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
                "B) Increase the PSI threshold in DriftSpec to 0.5 to suppress the alert",
                "C) PSI = 0.42 indicates severe drift (>0.25 is the critical threshold). Immediately shadow the production model with a retrained version on 2026 data. Under MAS model risk management, a drift alert at this severity triggers mandatory model revalidation — continued use without revalidation exposes the bank to regulatory sanction",
                "D) PSI measures concept drift only; feature drift does not require retraining"
            ],
            "answer": "C",
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
                "A) DriftMonitor requires 'async with monitor:' context manager before calling set_reference() and check() — without it, the monitor's internal connection is not initialised, causing TypeError on the first method call",
                "B) DriftMonitor must be imported from kailash_ml.engines, not kailash_ml",
                "C) set_reference() is a synchronous method and should not be awaited",
                "D) DriftSpec must be created inside the async context manager"
            ],
            "answer": "A",
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
                "A) The oscillations are expected: cosine annealing cyclically increases LR from near-zero back to a warm value, helping the model escape local minima. The loss rises when LR rises (exploration) then falls as LR decreases (exploitation). The final cycle stabilises at 0.31, lower than the 0.41 achieved without restarts — confirming the approach worked",
                "B) The student is right — oscillating loss means the LR is too high and should be reduced",
                "C) Oscillations are a numerical precision issue; switch to Adam optimiser",
                "D) Cosine annealing only has one cycle; multiple oscillations indicate a bug"
            ],
            "answer": "A",
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
                "B) OnnxBridge export requires opset_version=11 explicitly; the default produces wrong outputs",
                "C) The file existence check is sufficient; ONNX format guarantees bit-identical outputs",
                "D) input_shape=(1, 4, 32, 32) is a 4D image shape but the tabular credit model takes 1D feature vectors — the wrong shape causes the ONNX graph to reshape/pad incorrectly. The validation must compare actual numerical outputs: run the same batch through both PyTorch and ONNX, then assert np.allclose(pytorch_out, onnx_out, atol=1e-5)"
            ],
            "answer": "D",
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
        # ── Additional questions covering lessons 1–8 breadth ─────────────
        {
            "id": "4.2.2",
            "lesson": "4.2",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student initialises a GMM but the EM algorithm does not converge "
                "and log-likelihood decreases after iteration 3. What is the most likely cause?"
            ),
            "code": (
                "from sklearn.mixture import GaussianMixture\n"
                "\n"
                "# Bug: covariance_type misspecified, bad initialisation\n"
                "gmm = GaussianMixture(\n"
                "    n_components=5,\n"
                "    covariance_type='full',\n"
                "    init_params='random',   # Bug: random init with small dataset\n"
                "    random_state=None,       # Bug: no seed\n"
                "    max_iter=10,             # Bug: too few iterations\n"
                ")"
            ),
            "options": [
                "A) GaussianMixture does not support n_components=5 for 2D data",
                "B) covariance_type='full' requires n_components=1",
                "C) Log-likelihood decrease is expected in EM; it always increases after iteration 5",
                "D) Three bugs combine: random init_params can place centroids in degenerate positions; None random_state makes results irreproducible; max_iter=10 may not reach convergence. Use init_params='kmeans' (more stable), random_state=42 (reproducibility), and max_iter=200 (sufficient for convergence)"
            ],
            "answer": "D",
            "explanation": (
                "EM is guaranteed to increase the log-likelihood at each step only if the initialisation "
                "is non-degenerate. Random initialisation can place a component centroid on a single point, "
                "causing a near-singular covariance matrix (covariance_type='full' is most vulnerable). "
                "KMeans initialisation places components at cluster centroids — far more stable. "
                "max_iter=10 terminates before the algorithm finds a good local optimum."
            ),
            "learning_outcome": "Configure GaussianMixture with stable initialisation and sufficient iterations",
        },
        {
            "id": "4.3.2",
            "lesson": "4.3",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Running PCA on the 85-feature customer dataset, the explained variance "
                "ratios for the first 5 components are [0.31, 0.18, 0.12, 0.08, 0.06]. "
                "The cumulative variance is [0.31, 0.49, 0.61, 0.69, 0.75]. "
                "A colleague wants to retain 90% variance. "
                "How many components are needed, and what does the slow variance accumulation "
                "after component 1 suggest about the data structure?"
            ),
            "options": [
                "A) 5 components suffice for 90% variance; the first component captures most information",
                "B) The first component (31%) captures most information; use only 1 component",
                "C) More than 5 components are needed (cumulative at 5 is only 75%). The slow accumulation after component 1 (from 31% to 18% to 12%...) suggests the data does not have a dominant low-dimensional linear structure — variance is distributed across many dimensions. This is typical of customer behaviour data where no single factor dominates",
                "D) 90% variance always requires exactly half the original dimensions"
            ],
            "answer": "C",
            "explanation": (
                "With cumulative variance at 75% after 5 components, you need more components to reach 90%. "
                "The steeper the 'elbow' in the scree plot, the more dominant the first few components. "
                "Here, variance decreases gradually (31 → 18 → 12 → 8 → 6), indicating no sharp elbow — "
                "the data variance is spread across many dimensions. "
                "This is common for customer data where purchasing patterns, demographics, and "
                "browsing behaviour all contribute independent variance."
            ),
            "learning_outcome": "Interpret PCA explained variance ratios and scree plot shape for component selection",
        },
        {
            "id": "4.4.2",
            "lesson": "4.4",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student configures EnsembleEngine but gets a data leakage warning. "
                "What is wrong with this stacking setup?"
            ),
            "code": (
                "from kailash_ml import EnsembleEngine\n"
                "\n"
                "engine = EnsembleEngine(method='stacking')\n"
                "\n"
                "# Bug: fitting meta-learner on the same data used for base models\n"
                "base_preds_train = []\n"
                "for model in [lgb_model, lr_model, xgb_model]:\n"
                "    model.fit(X_train, y_train)\n"
                "    base_preds_train.append(model.predict_proba(X_train)[:, 1])  # Bug\n"
                "\n"
                "meta_X = np.column_stack(base_preds_train)\n"
                "meta_learner.fit(meta_X, y_train)  # Meta-learner sees in-sample predictions"
            ),
            "options": [
                "A) EnsembleEngine.method must be 'stack', not 'stacking'",
                "B) Base model predictions on X_train are in-sample — the models have already memorised X_train, so their predictions are overfit. The meta-learner learns to exploit this overfit. Correct: generate out-of-fold predictions via cross_val_predict(model, X_train, y_train, cv=5) for each base model — these are predictions on data the model never saw during training",
                "C) All base models must be the same type for stacking to work",
                "D) meta_learner.fit() should use X_test, not meta_X",
            ],
            "answer": "B",
            "explanation": (
                "The critical stacking implementation rule: base model predictions used to train the "
                "meta-learner must be out-of-fold predictions. In-sample predictions are over-optimistic "
                "because the model has memorised the training labels. "
                "Use cross_val_predict() with cv=5 to generate held-out predictions for each fold — "
                "each training row gets a prediction from a model that was not trained on it. "
                "EnsembleEngine handles this correctly internally; the bug is in the manual implementation."
            ),
            "learning_outcome": "Implement stacking with out-of-fold base predictions to prevent meta-learner leakage",
        },
        {
            "id": "4.6.3",
            "lesson": "4.6",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "After DriftMonitor flags debt_to_income_ratio with PSI=0.42, "
                "you retrain the credit model on 2026 data. "
                "Before deploying the new model, you run DriftMonitor with the 2026 training data "
                "as the new reference. PSI drops to 0.04. "
                "What does this tell you and what should be the new DriftSpec.psi_threshold "
                "for monitoring going forward?"
            ),
            "options": [
                "A) PSI=0.04 means no drift is possible with the new model; set threshold to 0.01",
                "B) Keep the old reference distribution; updating it defeats the purpose of monitoring",
                "C) PSI=0.04 means the reference is wrong; it should always be 0.00",
                "D) PSI=0.04 confirms the new reference distribution matches current production data — the 'drift' is now baselined away. Going forward, set psi_threshold to 0.15 (moderate drift) or 0.25 (severe) with the new reference. The threshold choice depends on how quickly you can retrain: monthly retraining supports 0.25, weekly supports 0.15"
            ],
            "answer": "D",
            "explanation": (
                "When you retrain on current data, the new training distribution becomes the reference. "
                "PSI=0.04 vs the new reference confirms the model and data are now aligned. "
                "The threshold choice is a business decision: how much drift is acceptable before retraining? "
                "A lower threshold (0.15) catches drift sooner at the cost of more false alarms. "
                "A higher threshold (0.25) reduces noise at the cost of longer model degradation before detection."
            ),
            "learning_outcome": "Update DriftMonitor reference after retraining and choose threshold based on retraining cadence",
        },
        {
            "id": "4.5.3",
            "lesson": "4.5",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "After running BERTopic on 50,000 Singapore news articles in Exercise 5, "
                "the model produces 47 topics. Topics 31-47 each contain fewer than 20 documents. "
                "Your stakeholder wants exactly 10 topic categories. "
                "Which BERTopic parameter controls the number of topics, "
                "and what is the alternative approach that does not require retraining?"
            ),
            "options": [
                "A) Increase min_topic_size; BERTopic automatically merges small topics",
                "B) Set nr_topics=10 in BERTopic constructor to request approximately 10 topics during training. Alternatively, after training, use topic_model.reduce_topics(docs, nr_topics=10) which merges similar topics using c-TF-IDF cosine similarity — no retraining required",
                "C) BERTopic always produces exactly as many topics as HDBSCAN clusters; change HDBSCAN min_cluster_size",
                "D) Retrain with a smaller sentence transformer to reduce topic granularity",
            ],
            "answer": "B",
            "explanation": (
                "BERTopic's nr_topics parameter requests a target number of topics by merging "
                "the most similar ones during training. "
                "Post-hoc reduction via topic_model.reduce_topics() merges topics using cosine similarity "
                "of their c-TF-IDF representations — topics with similar word distributions are merged first. "
                "This avoids the costly embedding and clustering step, "
                "which is practical when the stakeholder changes the target number after initial exploration."
            ),
            "learning_outcome": "Control BERTopic topic count via nr_topics and post-hoc reduce_topics()",
        },
        {
            "id": "4.7.3",
            "lesson": "4.7",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "In Exercise 7, after exporting the CNN to ONNX, you compare inference latency:\n\n"
                "  PyTorch (CPU): 4.2 ms/sample\n"
                "  ONNX Runtime (CPU): 1.1 ms/sample\n\n"
                "A developer asks: 'Why is ONNX 4× faster if it runs the same computation?' "
                "Explain two optimisations the ONNX Runtime applies that PyTorch's default CPU backend does not."
            ),
            "options": [
                "A) ONNX Runtime skips error checking; PyTorch validates every tensor operation",
                "B) ONNX Runtime uses GPU automatically even when told to use CPU",
                "C) (1) Operator fusion: ONNX Runtime merges sequential operators (Conv + BatchNorm + ReLU) into a single fused kernel, reducing memory round-trips. (2) Graph optimisation: ONNX Runtime applies constant folding and dead code elimination at the graph level before execution, removing redundant operations that PyTorch's eager mode cannot eliminate",
                "D) PyTorch's Python overhead is the only difference; ONNX eliminates Python calls"
            ],
            "answer": "C",
            "explanation": (
                "PyTorch eager mode executes each operation independently — Conv writes to memory, "
                "BatchNorm reads from memory, ReLU reads from memory again. "
                "ONNX Runtime's graph-level optimiser fuses these into a single operation "
                "that never writes intermediate results to main memory. "
                "Constant folding precomputes fixed operations (e.g., normalisation constants) "
                "at load time. Together, these optimisations explain the 4× speedup — "
                "not just Python overhead removal."
            ),
            "learning_outcome": "Explain ONNX Runtime operator fusion and graph optimisation as sources of inference speedup",
        },
        {
            "id": "4.1.3",
            "lesson": "4.1",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 1 evaluates clusters with three metrics: silhouette (higher=better), "
                "Calinski-Harabasz (CH, higher=better), and Davies-Bouldin (DB, lower=better). "
                "K-means with k=5 gives: silhouette=0.41, CH=1240, DB=0.89. "
                "K-means with k=8 gives: silhouette=0.38, CH=1680, DB=0.72. "
                "The metrics disagree on k=5 vs k=8. How do you resolve this contradiction "
                "for the marketing segmentation use case from question 4.1.1?"
            ),
            "options": [
                "A) Metrics measure different properties: silhouette measures cohesion and separation (k=5 wins), CH measures cluster compactness relative to separation (k=8 wins), DB measures average worst-case similarity (k=8 wins). For 5 well-defined marketing segments (business requirement), k=5 is correct — the silhouette advantage and operational simplicity (fewer segments to manage) outweigh DB/CH improvements from k=8",
                "B) CH always wins; use k=8 because CH=1680 > 1240",
                "C) Always use the metric that gives the best score for the chosen k",
                "D) Use the average of all three metrics to produce a single ranking"
            ],
            "answer": "A",
            "explanation": (
                "Cluster evaluation metrics are not perfectly correlated because they measure different things. "
                "When they disagree, the business requirement is the tiebreaker. "
                "CH and DB both favour k=8 because more clusters generally produce tighter, "
                "more separated groups. But the marketing team has a constraint: 5 segments max "
                "(for campaign budget, message design, and analyst comprehension). "
                "Silhouette's advantage for k=5 confirms the 5-cluster solution is still meaningful."
            ),
            "learning_outcome": "Resolve clustering metric disagreement using business constraints as tiebreaker",
        },
        {
            "id": "4.8.2",
            "lesson": "4.8",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student deploys InferenceServer but predictions are always based on "
                "stale model weights even after retraining. What is the architecture issue?"
            ),
            "code": (
                "from kailash_ml.engines.inference_server import InferenceServer\n"
                "\n"
                "# Model loaded at startup — never refreshed\n"
                "model = registry.get('credit_v1')  # fetched once at import time\n"
                "server = InferenceServer(model=model, cache_predictions=True)\n"
                "\n"
                "# Weekly retrain runs and updates registry to 'credit_v2'\n"
                "# but the running InferenceServer still holds credit_v1"
            ),
            "options": [
                "A) InferenceServer automatically polls ModelRegistry for updates",
                "B) Use registry.get('credit_v2') instead of 'credit_v1' to always get the latest",
                "C) cache_predictions=True prevents model updates; set it to False",
                "D) The model is fetched once at startup and held in memory — the server never checks ModelRegistry for newer versions. Fix: pass a model_loader callable instead of a model object: server = InferenceServer(model_loader=lambda: registry.get_latest('credit'), cache_predictions=True, cache_ttl=3600); the server reloads the model on cache miss"
            ],
            "answer": "D",
            "explanation": (
                "Passing a model object binds the server to that specific instance for its lifetime. "
                "The InferenceServer pattern for live model updates uses a model_loader callable — "
                "a function that fetches the current production model from ModelRegistry. "
                "Combined with cache_ttl, the server reloads the model periodically. "
                "When the weekly retrain promotes credit_v2 to production, the loader returns the new model "
                "on the next cache expiry without restarting the server."
            ),
            "learning_outcome": "Configure InferenceServer with a model_loader for live model updates from ModelRegistry",
        },
        {
            "id": "4.2.3",
            "lesson": "4.2",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "After fitting a GMM with 5 components to customer revenue data, "
                "you check convergence by printing gmm.converged_. It is False. "
                "The lower_bound_ (log-likelihood) at the last iteration is -14,203. "
                "What does non-convergence mean practically, and what are two ways to address it?"
            ),
            "options": [
                "A) non-convergence means the EM algorithm hit max_iter before the log-likelihood stopped improving — the found solution may be a suboptimal local optimum. Address: (1) increase max_iter (e.g., from 100 to 500); (2) run multiple restarts with different random seeds (n_init=10) and take the solution with the highest lower_bound_",
                "B) Non-convergence means the data has no cluster structure; use K-means instead",
                "C) Non-convergence is normal; the model is still usable as-is",
                "D) Reduce n_components until converged_ is True"
            ],
            "answer": "A",
            "explanation": (
                "EM is guaranteed to increase log-likelihood monotonically but may plateau slowly. "
                "converged_=False means the algorithm was cut off before plateauing. "
                "Increasing max_iter gives the algorithm more steps to converge. "
                "n_init=10 runs from 10 different initialisations and keeps the best — "
                "the best lower_bound_ across restarts is the global optimum candidate. "
                "Both are standard GMM best practices in sklearn and kailash-ml wrappers."
            ),
            "learning_outcome": "Interpret GaussianMixture converged_ and lower_bound_ and apply remediation",
        },
        {
            "id": "4.3.3",
            "lesson": "4.3",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student applies PCA but gets worse model performance after dimensionality "
                "reduction compared to the original features. "
                "They reduced from 85 to 10 components. "
                "What did they likely forget to do before PCA, and what does this cause?"
            ),
            "code": (
                "from sklearn.decomposition import PCA\n"
                "\n"
                "# Bug: no standardisation before PCA\n"
                "X_pca_train = PCA(n_components=10).fit_transform(X_train)\n"
                "X_pca_test = PCA(n_components=10).fit_transform(X_test)  # Second bug\n"
                "\n"
                "model.fit(X_pca_train, y_train)\n"
                "model.predict(X_pca_test)"
            ),
            "options": [
                "A) PCA requires categorical encoding before application",
                "B) Two bugs: (1) No StandardScaler before PCA — PCA's principal components are dominated by features with large variance (e.g., income ~50,000 overwhelms age ~35); (2) PCA is fit separately on X_test which produces different component directions — the test PCA is not aligned with training. Fix: fit scaler and PCA on X_train, then transform(X_test) with the same fitted objects",
                "C) PCA should be applied after model training, not before",
                "D) n_components=10 is too few; use n_components='mle' for automatic selection",
            ],
            "answer": "B",
            "explanation": (
                "PCA finds directions of maximum variance. Without standardisation, "
                "high-variance features (income) dominate the first principal components — "
                "not because they are important but because they have large absolute values. "
                "The second bug (fitting PCA on test data) is a leakage issue: "
                "the test PCA components are computed from test labels' implicit information "
                "and are in a different orientation than the training components. "
                "Always fit scaler + PCA on train only, then apply to test."
            ),
            "learning_outcome": "Apply PCA correctly with prior standardisation and train-only fitting",
        },
        {
            "id": "4.6.4",
            "lesson": "4.6",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Your DriftMonitor is configured with DriftSpec(psi_threshold=0.25, ks_alpha=0.05). "
                "In production, PSI=0.18 and KS p-value=0.02 for the same feature. "
                "PSI says no severe drift (0.18 < 0.25); KS test says significant drift (p < 0.05). "
                "Which test should you trust more for a continuous feature, and why might they disagree?"
            ),
            "options": [
                "A) PSI is always more reliable; ignore KS when PSI < 0.25",
                "B) For continuous features, the KS test is more powerful — it is sensitive to location shifts, scale changes, and shape differences in the distribution. PSI discretises the distribution into bins before computing, which can miss subtle distributional changes. A PSI=0.18 with KS p=0.02 suggests a moderate but real distributional shift that PSI's binning smoothed over. Investigate the feature's distribution visually before deciding on retraining",
                "C) Use the average of both thresholds to make a decision",
                "D) KS p-value < 0.05 always requires immediate retraining regardless of PSI",
            ],
            "answer": "B",
            "explanation": (
                "PSI bins the continuous distribution and measures relative frequency shifts per bin. "
                "It can miss shifts that don't change which bin data falls into (e.g., the distribution "
                "mean shifts by 5% but the data stays in the same bins). "
                "KS tests the maximum cumulative distribution difference — it is more sensitive "
                "to location shifts and is distribution-agnostic. "
                "When they disagree, visualise the feature's distribution (both reference and current) "
                "to understand the nature of the shift before deciding on action."
            ),
            "learning_outcome": "Compare PSI and KS test sensitivity for continuous feature drift detection",
        },
        {
            "id": "4.5.4",
            "lesson": "4.5",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student builds a TF-IDF vectoriser for Singapore news classification "
                "but all documents get identical TF-IDF vectors. What is wrong?"
            ),
            "code": (
                "from sklearn.feature_extraction.text import TfidfVectorizer\n"
                "\n"
                "# Bug: max_features not set, but more critically:\n"
                "vectorizer = TfidfVectorizer(\n"
                "    stop_words='english',\n"
                "    max_df=1.0,  # Bug: keeps all terms including universal ones\n"
                "    min_df=1,\n"
                ")\n"
                "X = vectorizer.fit_transform(corpus)\n"
                "# Result: IDF is near-zero for all terms"
            ),
            "options": [
                "A) TfidfVectorizer must receive a Polars Series, not a list of strings",
                "B) max_df=1.0 keeps terms that appear in up to 100% of documents — including near-universal terms like 'Singapore' which appear in almost every article. Their IDF ≈ log(1/1.0) ≈ 0, making all document vectors near-zero. Set max_df=0.85 to remove terms appearing in more than 85% of documents",
                "C) stop_words='english' removes all meaningful terms in Singapore news",
                "D) fit_transform must be called separately: fit() then transform()",
            ],
            "answer": "B",
            "explanation": (
                "IDF = log(N / df(t)) where df(t) is the fraction of documents containing term t. "
                "With max_df=1.0, a term appearing in 98% of Singapore news articles has "
                "IDF = log(1/0.98) ≈ 0.02 — nearly zero. "
                "TF-IDF for that term is near-zero for every document. "
                "Setting max_df=0.85 removes terms present in >85% of documents, "
                "ensuring discriminative terms have meaningful IDF values."
            ),
            "learning_outcome": "Set max_df in TfidfVectorizer to remove near-universal terms with near-zero IDF",
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
