# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 4 — AI-Resilient Assessment Questions

Unsupervised ML, NLP & Deep Learning
Covers: clustering, EM/GMM, PCA, EnsembleEngine, NLP (TF-IDF/BERTopic),
        DriftMonitor, deep learning, OnnxBridge
"""

QUIZ = {
    "module": "MLFP03",
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
                "    conn = ConnectionManager('sqlite:///mlfp03_drift.db')\n"
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

# --- Merged from mlfp03 (ML Engineering & Production) ---


LLMs, AI Agents & RAG Systems
Covers: Delegate, CoT, ReActAgent, RAGResearchAgent, MCP, ML agents,
        multi-agent patterns, Nexus deployment
"""

QUIZ = {
    "module": "MLFP03",
    "title": "LLMs, AI Agents & RAG Systems",
    "questions": [
        # ── Lesson 1: LLM fundamentals and Delegate ───────────────────────
        {
            "id": "5.1.1",
            "lesson": "5.1",
            "type": "code_debug",
            "difficulty": "foundation",
            "question": (
                "A student runs Delegate in Exercise 1 but ignores the mandatory cost budget. "
                "The script runs for 20 minutes and incurs a $47 API charge. "
                "What two things are wrong with this setup?"
            ),
            "code": (
                "from kaizen_agents import Delegate\n"
                "import os\n"
                "\n"
                "delegate = Delegate(\n"
                "    model='gpt-4o',  # Bug 1: hardcoded model name\n"
                "    # Bug 2: missing max_llm_cost_usd\n"
                ")\n"
                "async for event in delegate.run('Analyse this entire 500MB dataset in detail'):\n"
                "    print(event)"
            ),
            "options": [
                "A) delegate.run() should be awaited, not iterated; Delegate does not support async for",
                "B) The import path is wrong; Delegate should be imported from kaizen.agents",
                "C) (1) Model name is hardcoded; must use os.environ['DEFAULT_LLM_MODEL']; (2) max_llm_cost_usd is missing; this is mandatory for all Module 5 exercises to prevent runaway spending",
                "D) 500MB datasets cannot be processed by Delegate; use DataExplorer first",
            ],
            "answer": "C",
            "explanation": (
                "Two violations from the Module 5 mandatory setup: "
                "(1) Hardcoded model names are prohibited — use os.environ.get('DEFAULT_LLM_MODEL'). "
                "This is a zero-tolerance rule per env-models.md. "
                "(2) max_llm_cost_usd is the hard budget cap that prevents unbounded spending. "
                "Without it, a long-running Delegate task can consume unlimited API budget. "
                "Correct: Delegate(model=os.environ['DEFAULT_LLM_MODEL'], max_llm_cost_usd=2.0)"
            ),
            "learning_outcome": "Apply mandatory Delegate setup: env-based model name and cost budget",
        },
        {
            "id": "5.1.2",
            "lesson": "5.1",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 1, you compare Delegate (autonomous) vs SimpleQAAgent "
                "(custom Signature). Running both on the same data analysis question, "
                "Delegate takes 8 API calls and returns a comprehensive answer, "
                "while SimpleQAAgent takes 1 API call and returns a structured dict. "
                "For a production system that answers 10,000 customer queries per day about "
                "their account data, which approach is preferable and why?"
            ),
            "options": [
                "A) SimpleQAAgent — 1 API call per query vs 8 means ~8× lower cost at 10,000 queries/day. The structured Signature output (typed InputField/OutputField) also enables downstream processing and validation. Delegate's autonomous multi-step reasoning is valuable for complex one-off analysis, not high-volume standardised queries",
                "B) Delegate — more API calls means more thorough analysis",
                "C) Delegate — structured output is not important for customer queries",
                "D) Both are identical in cost; Delegate just shows its work",
            ],
            "answer": "A",
            "explanation": (
                "Delegate's TAOD loop (Think-Act-Observe-Decide) is designed for autonomous exploration "
                "of open-ended problems — it deliberately uses multiple calls to build context. "
                "For a standardised question with a known answer structure, SimpleQAAgent's "
                "fixed Signature is 8× cheaper and returns a typed, validated response. "
                "At 10,000 queries/day: Delegate ≈ 80,000 API calls vs SimpleQAAgent ≈ 10,000. "
                "This is the core trade-off between Delegate (autonomy) and Signature agents (efficiency)."
            ),
            "learning_outcome": "Choose between Delegate and Signature agents based on query volume and structure",
        },
        # ── Lesson 2: Chain-of-thought and structured reasoning ───────────
        {
            "id": "5.2.1",
            "lesson": "5.2",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "You are building a credit risk explanation agent that must output a "
                "structured JSON with fields: risk_level, primary_factors (list), "
                "recommended_action, and confidence_score. "
                "Should you use a raw Delegate.run() call or build a custom Signature "
                "with InputField/OutputField? Explain the key advantage of the Signature approach "
                "for a regulated financial service."
            ),
            "options": [
                "A) Custom Signature with OutputField(type=RiskExplanation) — the output is schema-validated at runtime so you can guarantee the agent returns exactly the required fields with correct types. For a regulated service, this means you never get malformed JSON causing a downstream crash, and the schema is auditable by the regulator",
                "B) Raw Delegate — more flexible and handles edge cases better",
                "C) Both are equivalent; parse the Delegate output with json.loads()",
                "D) Signature agents cannot produce JSON output; use a post-processing function",
            ],
            "answer": "A",
            "explanation": (
                "Kaizen Signatures define typed InputField and OutputField contracts. "
                "When the agent runs, Kaizen validates the output against the OutputField types — "
                "if the LLM returns an incomplete or malformed response, Kaizen retries or raises a clear error "
                "rather than passing bad data downstream. "
                "For a regulated credit decision, this schema enforcement is non-negotiable: "
                "you cannot allow an unstructured string to flow into a loan approval system."
            ),
            "learning_outcome": "Use Kaizen Signature for schema-validated agent outputs in regulated contexts",
        },
        # ── Lesson 3: ReAct agents with tools ────────────────────────────
        {
            "id": "5.3.1",
            "lesson": "5.3",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "Running ReActAgent in Exercise 3 on the credit dataset produces this trace:\n\n"
                "  Thought: I need to understand the data distribution first\n"
                "  Action: tool_profile_data('sg_credit_scoring')\n"
                "  Observation: 15,000 rows, default_rate=8.2%, 12 features, 3 alerts\n"
                "  Thought: I should check for highly correlated features before recommending engineering\n"
                "  Action: tool_check_correlations(threshold=0.8)\n"
                "  Observation: income x employment_income corr=0.91 (HIGH)\n"
                "  Thought: One of the pair should be removed or combined\n"
                "  Answer: Remove employment_income; it is 91% correlated with income...\n\n"
                "What does this trace demonstrate about ReAct vs a single-call Delegate query, "
                "and when would you choose ReAct over Delegate?"
            ),
            "options": [
                "A) ReAct is always slower; use Delegate for all production scenarios",
                "B) The trace shows ReAct made two unnecessary tool calls; one call would suffice",
                "C) ReAct cannot call DataExplorer tools; only Delegate can access ML tools",
                "D) The trace shows ReAct's Reason-Act-Observe loop: the agent produces an intermediate thought, calls a tool, observes the result, and reasons again before acting. This is preferable to Delegate when you want to see and audit the exact tool calls and intermediate observations — useful for regulated workflows where explainability of the agent's decision process is required",
            ],
            "answer": "D",
            "explanation": (
                "Delegate is a higher-level abstraction — it autonomously decides when to reason and act "
                "but does not expose a step-by-step trace by default. "
                "ReActAgent exposes every thought-action-observation cycle explicitly. "
                "In regulated ML workflows, being able to audit 'the agent decided to check correlations "
                "because it profiled the data first' provides a complete decision audit trail "
                "that a Delegate summary cannot."
            ),
            "learning_outcome": "Contrast ReAct's explicit trace with Delegate's opaque autonomy for auditability",
        },
        {
            "id": "5.3.2",
            "lesson": "5.3",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's ReActAgent never uses the tool_profile_data function. "
                "The agent always responds with generic analysis. What is wrong?"
            ),
            "code": (
                "from kaizen_agents.agents.specialized.react import ReActAgent\n"
                "\n"
                "async def tool_profile_data(dataset_name: str) -> str:\n"
                '    """Profile a dataset using DataExplorer."""\n'
                "    # ... implementation ...\n"
                "    return profile_summary\n"
                "\n"
                "# Bug: tools not registered\n"
                "agent = ReActAgent(\n"
                "    model=os.environ['DEFAULT_LLM_MODEL'],\n"
                "    max_llm_cost_usd=3.0,\n"
                "    # tools parameter is missing\n"
                ")\n"
                "result = await agent.run('Analyse the credit dataset')"
            ),
            "options": [
                "A) tool_profile_data must be decorated with @agent.tool to be registered",
                "B) ReActAgent cannot use async tools; make tool_profile_data synchronous",
                "C) The tools parameter is missing from ReActAgent; pass tools=[tool_profile_data] to register the function; without it, the agent has no tools to call and falls back to pure LLM reasoning",
                "D) The function signature must include 'self' to be registered as an agent tool",
            ],
            "answer": "C",
            "explanation": (
                "ReActAgent discovers available tools from the tools= parameter in its constructor. "
                "Without this list, the agent operates in reasoning-only mode — "
                "it generates text responses without ever calling an external function. "
                "Correct: ReActAgent(model=..., max_llm_cost_usd=3.0, tools=[tool_profile_data, tool_check_correlations])"
            ),
            "learning_outcome": "Register tools with ReActAgent via the tools= constructor parameter",
        },
        # ── Lesson 4: RAG systems ─────────────────────────────────────────
        {
            "id": "5.4.1",
            "lesson": "5.4",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You are building a RAG system to answer questions about Singapore financial "
                "regulations from a 2,000-page PDF corpus. Your colleague suggests "
                "chunking all documents into 512-token chunks with 50-token overlap. "
                "What problem does overlap solve, and what is the risk of chunks that are too small?"
            ),
            "options": [
                "A) Overlap reduces storage requirements; small chunks are always more precise",
                "B) Smaller chunks are always better because retrieval precision increases",
                "C) Overlap is only needed for code documents; regulatory PDFs do not need it",
                "D) Overlap prevents context loss at chunk boundaries — a sentence about 'the penalty' makes no sense if 'the regulation' was in the previous chunk. Chunks that are too small (e.g., 100 tokens) cause the retrieved context to lack enough information for the LLM to answer, leading to hallucination as the model fills in missing context",
            ],
            "answer": "D",
            "explanation": (
                "Chunking with overlap ensures that concepts split across chunk boundaries "
                "appear in at least one complete chunk. "
                "For regulatory text, a provision might reference a definition from the previous paragraph — "
                "without overlap, the retrieval system might return the provision but not its definition. "
                "Very small chunks (< 200 tokens) also reduce the information density per retrieved chunk, "
                "forcing the system to retrieve more chunks and increasing the chance of irrelevant context."
            ),
            "learning_outcome": "Explain chunk overlap trade-offs for RAG over long-form regulatory documents",
        },
        # ── Lesson 5: MCP servers ─────────────────────────────────────────
        {
            "id": "5.5.1",
            "lesson": "5.5",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student creates an MCP server but agents cannot discover the tools. "
                "What is wrong with this setup?"
            ),
            "code": (
                "from kailash.mcp_server import MCPServer, MCPTool, MCPToolResult\n"
                "\n"
                "server = MCPServer(name='ml_tools')\n"
                "\n"
                "# Define tool\n"
                "async def profile_data(dataset_name: str) -> MCPToolResult:\n"
                "    result = await _run_profiler(dataset_name)\n"
                "    return MCPToolResult(content=result)\n"
                "\n"
                "# Bug: tool is defined but never registered\n"
                "# server.register_tool() is missing\n"
                "\n"
                "await server.start(transport=StdioTransport())"
            ),
            "options": [
                "A) server.register_tool() is missing — the function must be wrapped in MCPTool and registered before server.start(); without registration, the server starts but advertises an empty tool list to clients",
                "B) MCPServer must be imported from kailash.mcp_server.server, not kailash.mcp_server",
                "C) StdioTransport cannot be used for multi-agent scenarios; use SSETransport",
                "D) MCPToolResult must include a status field; content alone is insufficient",
            ],
            "answer": "A",
            "explanation": (
                "MCP tool discovery works by clients calling list_tools — the server returns its registry. "
                "If no tools are registered, the server returns an empty list and agents have nothing to call. "
                "The correct pattern from Exercise 5:\n"
                "tool = MCPTool(name='profile_data', description='...', handler=profile_data)\n"
                "server.register_tool(tool)\n"
                "await server.start(transport=StdioTransport())"
            ),
            "learning_outcome": "Register tools with MCPServer before starting to enable agent discovery",
        },
        {
            "id": "5.5.2",
            "lesson": "5.5",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 covers two MCP transport options: StdioTransport and SSETransport. "
                "Your team wants to deploy the ML tools server so that multiple agents "
                "running in different processes can connect to it simultaneously. "
                "Which transport is required and why?"
            ),
            "options": [
                "A) StdioTransport — it supports multiple simultaneous connections via process forking",
                "B) MCPServer cannot serve multiple clients; deploy one server per agent",
                "C) Both transports support multi-client connections equally",
                "D) SSETransport (Server-Sent Events) — it runs as an HTTP server on a fixed port, allowing multiple clients to connect independently over a network. StdioTransport is subprocess-based (one client per pipe) and cannot serve multiple simultaneous clients",
            ],
            "answer": "D",
            "explanation": (
                "StdioTransport uses a subprocess pipe — one MCPServer process per client. "
                "It is appropriate for local development and single-agent testing. "
                "SSETransport starts an HTTP/SSE server on a configurable port. "
                "Multiple agents connect to the same URL (http://host:port) and make independent "
                "tool calls. This is the production deployment pattern for shared ML tool servers."
            ),
            "learning_outcome": "Select SSETransport for multi-client MCP server deployments",
        },
        # ── Lesson 6: ML agents ───────────────────────────────────────────
        {
            "id": "5.6.1",
            "lesson": "5.6",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "You build an ML agent using Delegate that autonomously trains and evaluates "
                "a credit model. The agent's task prompt includes 'choose the best model'. "
                "After running, the agent reports AUC=0.91 on the training set. "
                "What governance violation has occurred and what should the prompt "
                "have specified to prevent it?"
            ),
            "options": [
                "A) The agent evaluated on training data, not a held-out test set — this is a data leakage violation. The prompt should specify: 'evaluate on the held-out test set only; training set performance is not a valid model selection criterion'. Without this constraint, the agent optimises in-sample fit, producing a model that is overfit and misleading",
                "B) No violation — the agent correctly maximised the AUC metric",
                "C) The violation is using AUC instead of F1-score; the prompt should specify F1",
                "D) The violation is using Delegate for model training; only TrainingPipeline may train models",
            ],
            "answer": "A",
            "explanation": (
                "An autonomous agent given a vague goal ('choose the best model') will optimise "
                "whatever metric it can measure most easily. Training set AUC is trivially improvable "
                "by overfitting. Agents need explicit constraints: 'use train/test split from PreprocessingPipeline', "
                "'report test set metrics only', 'do not tune hyperparameters on the test set'. "
                "This is a concrete example of why agent operating envelopes matter — "
                "covered in Module 6's PACT governance exercises."
            ),
            "learning_outcome": "Identify data leakage risk in autonomous ML agent prompts and write corrective constraints",
        },
        # ── Lesson 7: Multi-agent patterns ────────────────────────────────
        {
            "id": "5.7.1",
            "lesson": "5.7",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You are designing a multi-agent ML pipeline with three stages: "
                "feature engineering, model training, and evaluation. "
                "Each stage should run concurrently where possible. "
                "Which Kaizen pattern is appropriate, and what does SupervisorWorkerPattern "
                "provide that three independent Delegate instances do not?"
            ),
            "options": [
                "A) Three independent Delegate instances — simpler and equivalent to SupervisorWorkerPattern",
                "B) SupervisorWorkerPattern only works for sequential pipelines; use asyncio.gather() for concurrent execution",
                "C) SupervisorWorkerPattern — the supervisor orchestrates task decomposition, monitors worker completion, aggregates results, and handles worker failures with retries. Three independent Delegates have no coordination mechanism: if the feature engineering agent fails, the training agent proceeds with stale features",
                "D) SupervisorWorkerPattern requires a shared database; use it only when persisting intermediate results",
            ],
            "answer": "C",
            "explanation": (
                "SupervisorWorkerPattern provides: "
                "(1) Task dependency management — supervisor knows feature engineering must complete before training starts; "
                "(2) Failure propagation — if a worker fails, the supervisor decides to retry or abort (not silently proceed); "
                "(3) Result aggregation — the supervisor collects worker outputs into a coherent final result. "
                "Three independent Delegates are fire-and-forget with no coordination. "
                "asyncio.gather() parallelises execution but has no semantic understanding of task dependencies."
            ),
            "learning_outcome": "Distinguish SupervisorWorkerPattern coordination from independent Delegate execution",
        },
        # ── Lesson 8: Nexus deployment ────────────────────────────────────
        {
            "id": "5.8.1",
            "lesson": "5.8",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student deploys an ML pipeline via Nexus but the /predict endpoint "
                "always returns 404. What is wrong with this setup?"
            ),
            "code": (
                "from kailash_nexus import Nexus\n"
                "\n"
                "app = Nexus()\n"
                "app.start()  # Bug: starting before registering\n"
                "\n"
                "# Student tries to register after start\n"
                "app.register(credit_scoring_workflow)"
            ),
            "options": [
                "A) Nexus must be imported from kailash.nexus, not kailash_nexus",
                "B) app.register() must be called before app.start() — Nexus builds the route table at startup from the registered workflows. Registering after start does not add routes to the already-running server",
                "C) workflow must be built with workflow.build() before registering with Nexus",
                "D) Nexus requires an explicit port number: Nexus(port=8080)",
            ],
            "answer": "B",
            "explanation": (
                "Nexus generates REST routes, CLI handlers, and MCP tool registrations at startup. "
                "The route table is fixed at the moment app.start() is called. "
                "Registering a workflow after start does nothing because the HTTP server is already "
                "running with its original route map. "
                "Correct pattern: register first, then start:\n"
                "app = Nexus()\n"
                "app.register(credit_scoring_workflow)\n"
                "app.start()"
            ),
            "learning_outcome": "Follow the correct Nexus register-before-start pattern",
        },
        {
            "id": "5.8.2",
            "lesson": "5.8",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You need to expose a credit scoring workflow via REST API with state "
                "maintained across requests (e.g., a user's session history). "
                "Nexus provides app.create_session(). "
                "Why is a Nexus session preferable to storing state in a Python dict on "
                "the server, and what does the session provide that a dict cannot?"
            ),
            "options": [
                "A) A Python dict is preferable — simpler and faster than Nexus sessions",
                "B) Sessions and dicts are equivalent; use a dict for simplicity in production",
                "C) Nexus sessions are only needed when using MCP; REST can use a dict",
                "D) Nexus sessions provide cross-channel consistency (REST, CLI, and MCP share the same session state), automatic expiry, and thread-safe state management across concurrent requests. A Python dict is not thread-safe and is lost when the process restarts or when load-balanced across multiple instances",
            ],
            "answer": "D",
            "explanation": (
                "A Python dict is process-local and not thread-safe under concurrent API requests. "
                "Two simultaneous requests can corrupt the same dict key without locking. "
                "Nexus sessions are backed by the ConnectionManager's storage (persistent, thread-safe), "
                "expire automatically after inactivity, and are shared across REST/CLI/MCP channels "
                "so a user who starts on REST and continues on CLI maintains the same context. "
                "This is critical for stateful workflows like multi-turn credit assessments."
            ),
            "learning_outcome": "Justify Nexus sessions over in-memory state for production multi-channel deployments",
        },
        # ── Additional questions covering lessons 1–8 breadth ─────────────
        {
            "id": "5.1.3",
            "lesson": "5.1",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 1, Delegate reports this at the end of the run:\n\n"
                "  LLM calls: 8\n"
                "  Total tokens: 12,847\n"
                "  Estimated cost: $0.19\n"
                "  Budget remaining: $1.81 / $2.00\n\n"
                "The analysis quality is good. A manager asks: 'Can we run this for all "
                "200 customer segments instead of 1?' "
                "What is the estimated total cost and what governance change is needed?"
            ),
            "options": [
                "A) Cost = $0.19 × 200 = $38; no governance change needed",
                "B) Estimated cost: $0.19 × 200 = $38 if each segment requires similar analysis. The current max_llm_cost_usd=2.0 is insufficient — a human operator must explicitly authorise a higher budget (e.g., max_llm_cost_usd=50.0) before scaling to 200 runs. The budget is a governance gate, not just a soft limit",
                "C) Delegate automatically scales; no cost estimation is possible before running",
                "D) Run all 200 segments in one call by passing all data in a single prompt",
            ],
            "answer": "B",
            "explanation": (
                "Linear cost scaling: $0.19 × 200 = $38. "
                "The current budget cap of $2.00 would terminate after ~10 segments ($2.00 / $0.19 ≈ 10). "
                "The max_llm_cost_usd parameter is a hard governance gate — "
                "Delegate stops as soon as the budget is reached, not gracefully scales back. "
                "A human must explicitly set the higher budget: the agent cannot self-authorise."
            ),
            "learning_outcome": "Estimate LLM cost at scale and identify when human budget authorisation is required",
        },
        {
            "id": "5.2.2",
            "lesson": "5.2",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "You build a credit risk Signature agent with an OutputField "
                "of type RiskExplanation (a Pydantic model). "
                "The LLM occasionally returns a response where confidence_score > 1.0. "
                "Your Pydantic model has confidence_score: float. "
                "What change to the OutputField prevents this and why is field-level validation "
                "superior to post-processing the raw LLM output?"
            ),
            "options": [
                "A) Change confidence_score: float to confidence_score: str and parse manually",
                "B) Wrap the agent call in try/except and clamp the value: min(1.0, max(0.0, score))",
                "C) Add a Pydantic validator: confidence_score: float = Field(..., ge=0.0, le=1.0). When the LLM returns 1.2, Pydantic raises ValidationError before the response reaches downstream code — the agent retries rather than propagating an invalid value. Post-processing requires manually handling every edge case after the fact",
                "D) Instruct the LLM in the system prompt to always return values between 0 and 1",
            ],
            "answer": "C",
            "explanation": (
                "Pydantic field constraints (ge=0.0, le=1.0) are enforced at deserialisation time — "
                "if the LLM output fails validation, Kaizen catches the ValidationError and "
                "can retry the LLM call with the error as feedback. "
                "Post-processing via min/max clamping silently accepts the invalid value "
                "and passes it downstream — it hides the LLM's misbehaviour rather than correcting it. "
                "System prompt instructions are probabilistic; Pydantic validation is deterministic."
            ),
            "learning_outcome": "Use Pydantic field constraints in Kaizen OutputField for deterministic output validation",
        },
        {
            "id": "5.4.2",
            "lesson": "5.4",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's RAGResearchAgent returns hallucinated answers for regulatory questions. "
                "The retrieval step finds relevant documents but the LLM ignores them. "
                "What common prompt construction error causes this?"
            ),
            "code": (
                "context_docs = retriever.search(query, top_k=5)\n"
                "context_text = '\\n'.join([doc.content for doc in context_docs])\n"
                "\n"
                "# Bug: context is appended after the question\n"
                'prompt = f"""\n'
                "Answer the following question about Singapore financial regulations:\n"
                "{query}\n"
                "\n"
                "Here are some relevant documents:\n"
                "{context_text}\n"
                '"""'
            ),
            "options": [
                "A) top_k=5 is too many documents; reduce to top_k=1",
                "B) The question appears before the context — LLMs generate the answer token-by-token and may begin answering before reading the context. Place the context before the question: 'Here are relevant documents:\\n{context_text}\\n\\nBased ONLY on the above, answer: {query}'",
                "C) context_text must be formatted as JSON, not plain text",
                "D) RAGResearchAgent handles prompt construction internally; never build prompts manually",
            ],
            "answer": "B",
            "explanation": (
                "Large language models generate autoregressively — the answer tokens are influenced "
                "by what came before them in the prompt. "
                "If the question appears first, the model may begin generating a plausible-sounding "
                "answer from its parametric memory before encountering the retrieved context. "
                "Placing context first and using an explicit grounding instruction "
                "('based ONLY on the above') reduces hallucination by anchoring the model "
                "to the retrieved documents."
            ),
            "learning_outcome": "Structure RAG prompts with context before question to reduce hallucination",
        },
        {
            "id": "5.6.2",
            "lesson": "5.6",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You are building an ML agent that autonomously selects and trains models. "
                "The agent has access to TrainingPipeline, ModelRegistry.promote(), and DataExplorer. "
                "A security review flags: 'The agent can promote its own trained models to production.' "
                "Which MLFP04 concept solves this, and what is the minimum change to the agent's "
                "configuration to prevent self-promotion?"
            ),
            "options": [
                "A) Both A and the PACT approach work, but the PACT approach is more robust: set can_deploy=False in the agent's role permissions. Even if ModelRegistry.promote() is in the tool list, GovernanceEngine.check_permission() blocks calls to that action for agents whose role lacks deploy permission. Removing the tool only prevents the agent from knowing about promotion; PACT prevents the action at the governance layer regardless of tool knowledge",
                "B) Remove ModelRegistry.promote() from the agent's tool list — the agent cannot promote what it cannot call",
                "C) Use DriftMonitor to detect if the agent is promoting models too frequently",
                "D) The agent should be trusted to make promotion decisions autonomously",
            ],
            "answer": "A",
            "explanation": (
                "Both options prevent self-promotion, but they defend at different layers. "
                "Removing the tool is a capability restriction — the agent cannot call the function at all. "
                "PACT's can_deploy=False is an authority restriction — the agent knows about promotion "
                "but is not authorised to execute it. "
                "The PACT approach is more auditable (every blocked attempt is logged to AuditChain) "
                "and more robust (governance cannot be bypassed by finding an alternate tool reference)."
            ),
            "learning_outcome": "Apply PACT permission restriction vs tool removal for agent deployment authority control",
        },
        {
            "id": "5.7.2",
            "lesson": "5.7",
            "type": "process_doc",
            "difficulty": "advanced",
            "question": (
                "In a multi-agent ML pipeline, the Feature Engineering Agent produces a feature "
                "set and the Training Agent consumes it. The training fails because the feature "
                "columns contain null values that the feature agent did not clean. "
                "How should the inter-agent handoff be structured to catch this before "
                "the training agent starts, and which Kailash class enforces the contract?"
            ),
            "options": [
                "A) The training agent should silently skip null columns",
                "B) Use asyncio.gather() to run both agents in parallel; the training agent will detect nulls",
                "C) Pass raw Polars DataFrames between agents; type checking is the training agent's responsibility",
                "D) The Feature Engineering Agent should return a typed result with a FeatureSchema that declares max_null_fraction=0.0. The supervisor uses ModelSignature or a custom Pydantic model to validate the handoff data before passing it to the Training Agent — validation failure triggers a Dereliction escalation rather than a silent training crash",
            ],
            "answer": "D",
            "explanation": (
                "In a Kaizen SupervisorWorkerPattern, the supervisor mediates inter-agent data flow. "
                "Typed handoff validation at the boundary (FeatureSchema or Pydantic model) catches "
                "data quality issues before the downstream agent starts. "
                "This converts a runtime training crash (hours in) into a fast validation failure "
                "at the handoff boundary (seconds in), and creates an AuditChain record "
                "that the feature agent delivered non-compliant output."
            ),
            "learning_outcome": "Design typed inter-agent handoffs with schema validation to prevent silent downstream failures",
        },
        {
            "id": "5.3.3",
            "lesson": "5.3",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Your ReActAgent is given a 15-feature credit dataset and the task "
                "'identify which features to keep'. The agent's trace shows:\n\n"
                "  Action: tool_check_correlations(threshold=0.8)\n"
                "  Observation: income x employment_income corr=0.91\n"
                "  Thought: employment_income is redundant; drop it\n"
                "  Answer: Drop employment_income\n\n"
                "The agent stopped after one tool call without profiling the data first. "
                "Why is this incomplete, and what did the agent miss by not calling "
                "tool_profile_data before tool_check_correlations?"
            ),
            "options": [
                "A) The agent is correct; correlation is the only relevant feature selection criterion",
                "B) The agent should have run more correlation checks with different thresholds",
                "C) Without profiling first, the agent does not know: (1) null rates — a feature with 40% nulls should be dropped regardless of correlation; (2) cardinality — a near-unique identifier column is harmful even if not correlated; (3) DataExplorer alerts that might flag other issues. The complete ReAct trace should profile_data → interpret alerts → check_correlations → synthesise",
                "D) tool_profile_data is optional; the agent chose the more efficient path",
            ],
            "answer": "C",
            "explanation": (
                "Feature selection requires holistic data understanding, not just correlation. "
                "A proper ReAct trace profiles first: DataExplorer identifies nulls, high-cardinality columns, "
                "constant features, and distribution alerts. "
                "Only after understanding the full data quality picture does correlation analysis "
                "make sense. The agent's shortcut might recommend keeping a feature that has "
                "60% nulls (unacceptable for training) because it appeared uncorrelated."
            ),
            "learning_outcome": "Design complete ReAct tool sequences that profile before performing feature selection",
        },
        {
            "id": "5.5.3",
            "lesson": "5.5",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "After deploying the MCP server in Exercise 5, you notice the profile_data tool "
                "is being called 500 times per minute by a runaway agent loop. "
                "The DataExplorer computation is expensive (2 seconds per call). "
                "Which two changes to the MCPServer configuration address this?"
            ),
            "options": [
                "A) Increase the server's max_workers to handle more concurrent calls",
                "B) Restart the MCP server to clear the runaway agent's connection",
                "C) (1) Add rate limiting: server.register_tool(tool, rate_limit=10) to cap calls per minute per client; (2) Add result caching: MCPTool(handler=profile_data, cache_ttl=300) returns cached results for repeated identical inputs within 5 minutes — the DataExplorer result for the same dataset does not change within seconds",
                "D) Replace SSETransport with StdioTransport to limit concurrent connections",
            ],
            "answer": "C",
            "explanation": (
                "Rate limiting (calls/minute per client) prevents a single misbehaving agent from "
                "monopolising the server. The client receives a rate limit error and must back off. "
                "Result caching returns the cached DataExplorer output for repeated calls on the "
                "same dataset within the TTL window — 500 identical calls cost the same as 1. "
                "Both changes are defensive measures: rate limiting addresses abuse, "
                "caching addresses inefficiency."
            ),
            "learning_outcome": "Apply MCPServer rate limiting and result caching to prevent runaway agent tool abuse",
        },
        {
            "id": "5.8.3",
            "lesson": "5.8",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You deploy the credit scoring Kaizen agent via Nexus as a REST endpoint. "
                "During a load test, you observe that agent responses are inconsistent: "
                "the same credit application gets different risk scores on consecutive calls. "
                "Which of the following explains this, and how does Nexus session management "
                "address it?"
            ),
            "options": [
                "A) LLM temperature causes randomness; always set temperature=0.0 for deterministic agents",
                "B) Both A and session history can cause inconsistency: an agent with conversation history in its session context may reason differently on the second call based on prior context. Nexus sessions isolate per-user state — each credit application should use a fresh session (app.create_session()) so agents start from a clean context. Combine with temperature=0.0 for fully deterministic credit decisions",
                "C) Inconsistency only occurs with streaming responses; use blocking calls",
                "D) The Nexus load balancer routes requests to different instances; add sticky sessions",
            ],
            "answer": "B",
            "explanation": (
                "Two sources of non-determinism: "
                "(1) LLM sampling temperature > 0 adds randomness to token selection. "
                "(2) Session history: if the agent's session accumulates prior credit decisions, "
                "the second application is evaluated with the context of the first one. "
                "For credit scoring, each application must be evaluated independently: "
                "fresh session + temperature=0.0. "
                "Nexus create_session() creates an isolated context; the session should be discarded "
                "after each complete credit assessment."
            ),
            "learning_outcome": "Identify session history and LLM temperature as sources of inconsistency in credit scoring agents",
        },
        {
            "id": "5.2.3",
            "lesson": "5.2",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student builds a Kaizen agent with a custom Signature but the agent "
                "always returns the same answer regardless of the input. "
                "What is the structural bug?"
            ),
            "code": (
                "from kaizen import Signature, InputField, OutputField\n"
                "\n"
                "class CreditRiskSignature(Signature):\n"
                '    """Assess credit risk from customer features."""\n'
                "    # Bug: InputField defined but not used in the class body\n"
                "    customer_profile: InputField = InputField(description='Customer features')\n"
                "    risk_level: OutputField = OutputField(description='low/medium/high')\n"
                "    explanation: OutputField = OutputField(description='Reasoning')\n"
                "\n"
                "agent = SimpleQAAgent(signature=CreditRiskSignature, model=model)\n"
                "# Calling with customer_profile ignored:\n"
                "result = await agent.forward()  # No arguments passed"
            ),
            "options": [
                "A) OutputField must come before InputField in the class definition",
                "B) agent.forward() is called with no arguments — the customer_profile InputField is never populated. The correct call passes the input: result = await agent.forward(customer_profile=profile_dict). Without the input, the LLM receives no customer data and generates a generic response",
                "C) SimpleQAAgent requires model to be set via agent.configure(), not the constructor",
                "D) Signature classes must inherit from BaseSignature, not Signature",
            ],
            "answer": "B",
            "explanation": (
                "Kaizen's Signature-based agents receive inputs via the forward() method arguments. "
                "Each InputField in the Signature corresponds to a keyword argument. "
                "Calling forward() without arguments means the LLM prompt contains no customer data — "
                "the agent can only generate a generic answer. "
                "The fix: await agent.forward(customer_profile={'income': 45000, 'debt_ratio': 0.42})"
            ),
            "learning_outcome": "Pass InputField values via agent.forward() keyword arguments",
        },
        {
            "id": "5.4.3",
            "lesson": "5.4",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You are building a RAG system for Singapore MAS regulations. "
                "After embedding and indexing 2,000 regulatory pages, "
                "retrieval returns 5 chunks per query. "
                "A compliance officer asks: 'How do you ensure the answer cites the exact "
                "regulation section, not just a paraphrase?' "
                "Which metadata field must be stored with each chunk, "
                "and how does the agent include it in the response?"
            ),
            "options": [
                "A) Store chunk_text only; regulation sections can be inferred from the content",
                "B) Use a reranker to put the most authoritative chunk first; the first chunk's source is the citation",
                "C) Citations are the compliance officer's responsibility; RAG only needs to answer correctly",
                "D) Store source_url, page_number, and section_heading with each chunk as metadata. Configure the RAG agent's Signature with an OutputField for citations: List[Citation] where Citation includes source, section, and excerpt. Instruct the agent to populate citations from the retrieved chunk metadata, not to generate them from parametric memory",
            ],
            "answer": "D",
            "explanation": (
                "Citations must be grounded in retrieved metadata, not LLM-generated guesses. "
                "If source metadata is not stored with each chunk, the agent cannot cite accurately — "
                "it would hallucinate plausible-sounding section references. "
                "A typed Citation OutputField in the Signature ensures every answer includes "
                "structured, validated citation data. "
                "The agent is instructed: 'cite only from the provided document metadata'."
            ),
            "learning_outcome": "Design RAG chunk metadata schema to enable accurate regulation section citations",
        },
        {
            "id": "5.6.3",
            "lesson": "5.6",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Your ML agent uses DataExplorer as a tool and receives this alert:\n\n"
                "  ALERT [CORRELATION] column_pair=('income', 'employment_income') r=0.91\n"
                "  ALERT [MISSING] column='credit_score' null_fraction=0.34\n"
                "  ALERT [CONSTANT] column='currency' unique_values=1\n\n"
                "The agent must recommend feature engineering actions for each alert. "
                "What is the correct recommendation for each alert type?"
            ),
            "options": [
                "A) Drop all flagged columns; alerts always mean the feature is useless",
                "B) CORRELATION (r=0.91): keep income, drop employment_income (or create ratio feature). MISSING (34%): add binary indicator is_credit_score_null; impute with median. CONSTANT (currency=1): drop immediately — a constant feature has zero information and will cause issues in some models",
                "C) CORRELATION: keep both columns. MISSING: drop the column. CONSTANT: encode as ordinal",
                "D) Only the CONSTANT alert requires action; the others are informational",
            ],
            "answer": "B",
            "explanation": (
                "CORRELATION: keeping both correlated features is redundant and inflates the feature space. "
                "For income/employment_income, keep the more complete/interpretable one or engineer a ratio. "
                "MISSING (34%): dropping would lose all information. A binary indicator preserves "
                "the MNAR signal (missing = no credit history). Median imputation handles the rest. "
                "CONSTANT: a feature with one unique value has zero variance — it contributes nothing "
                "to any model and may cause division-by-zero in normalisation."
            ),
            "learning_outcome": "Map DataExplorer alert types to correct feature engineering actions",
        },
        {
            "id": "5.7.3",
            "lesson": "5.7",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student implements SupervisorWorkerPattern but worker results are never "
                "collected — the supervisor exits immediately. What is the bug?"
            ),
            "code": (
                "from kaizen_agents import Delegate, SupervisorWorkerPattern\n"
                "\n"
                "supervisor = SupervisorWorkerPattern(\n"
                "    model=os.environ['DEFAULT_LLM_MODEL'],\n"
                "    max_llm_cost_usd=10.0,\n"
                ")\n"
                "\n"
                "# Workers dispatched but not awaited\n"
                "supervisor.dispatch(worker_fe, task='engineer features')\n"
                "supervisor.dispatch(worker_train, task='train model')\n"
                "\n"
                "# Bug: no await on results\n"
                "print('Done')  # exits before workers complete"
            ),
            "options": [
                "A) supervisor.dispatch() is non-blocking — it queues the tasks but does not wait. Add await supervisor.gather_results() or async for result in supervisor.results(): after dispatching to wait for and collect worker outputs before printing 'Done'",
                "B) dispatch() should be run(), not dispatch()",
                "C) Workers must be added to the supervisor before it is instantiated",
                "D) SupervisorWorkerPattern does not support multiple dispatch() calls; use a list",
            ],
            "answer": "A",
            "explanation": (
                "dispatch() is asynchronous fire-and-forget — it submits the task but returns immediately. "
                "Without awaiting completion, the main coroutine continues to 'print Done' and exits "
                "while workers are still running. "
                "gather_results() (or equivalent) blocks until all dispatched workers complete "
                "and aggregates their outputs. "
                "This is the standard producer-consumer pattern for async multi-agent workflows."
            ),
            "learning_outcome": "Await SupervisorWorkerPattern.gather_results() to collect worker outputs before proceeding",
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
