# Module 4: Unsupervised ML, NLP & Deep Learning

**Duration**: 7 hours  
**Kailash**: kailash-ml (AutoMLEngine, EnsembleEngine, DriftMonitor, InferenceServer), Nexus  
**Scaffolding**: 40%

## Lecture Topics

### 4A: Unsupervised ML Beyond K-means (90 min)
- K-means limitations: spherical assumption, fixed K, initialization (k-means++)
- Spectral clustering: graph Laplacian construction (**both normalized and unnormalized**), eigengap heuristic, normalized cuts
- HDBSCAN: **mutual reachability distance, condensed tree** construction, automatic cluster count, noise points
- Gaussian mixture models: **EM algorithm full derivation** (P0) — E-step: posterior responsibility computation, M-step: MLE for means/covariances/weights, convergence guarantee (monotonic ELBO increase). Show EM as template algorithm (reused in LDA, missing data, latent variable models). Connect to K-means as hard-assignment EM.
- Dimensionality reduction: PCA (**explicit SVD connection**, reconstruction error), t-SNE (**KL divergence objective**, perplexity tuning, Barnes-Hut), UMAP (topological data analysis intuition, cross-entropy loss)
- Anomaly detection: Isolation Forest (random partitioning theory), LOF, autoencoders for anomaly scoring
- Clustering validation: silhouette, Calinski-Harabasz, Davies-Bouldin, **gap statistic**

### 4B: NLP Foundations & Topic Modeling (60 min)
- Text representation: **TF-IDF derivation** (term frequency × inverse document frequency, why log-IDF), **BM25 derivation** (TF saturation, document length normalization), **Word2Vec skip-gram with negative sampling** (P0: derive objective, show contrastive learning connection — bridge to modern embedding methods), GloVe (co-occurrence matrix factorization), sentence embeddings (SBERT)
- Topic modeling: LDA (plate notation, Gibbs sampling intuition), NMF (relationship to K-means), BERTopic (UMAP + HDBSCAN + c-TF-IDF), **topic coherence metrics** (NPMI, C_v)
- Sentiment analysis: lexicon vs ML (1 slide — compress to make room for NLP depth)

### 4C: Deep Learning Foundations (60 min)
- Universal approximation theorem, gradient flow, vanishing/exploding gradients
- Residual connections (skip connection theory, gradient highway), BatchNorm vs LayerNorm vs RMSNorm (when each works)
- **Attention from first principles** (P0): derive scaled dot-product from information retrieval analogy (Q=query, K=key, V=value), **why scale by √d_k** (prevent softmax saturation — show numerically), **multi-head attention formula** (parallel heads see different relationships)
- Training dynamics: cosine annealing, OneCycleLR, **learning rate warmup theory** (linear warmup stabilizes Adam's variance estimates in early training), AdamW (weight decay decoupling), gradient clipping
- CNN: convolution as learned feature extraction, receptive field theory, ResNet
- RNN/LSTM: gate mechanism (5 min — focus on why attention replaced RNNs)

## Lab Exercises (6)

1. **Clustering comparison**: K-means vs spectral vs HDBSCAN vs GMM on e-commerce customer data — silhouette, CH, DB scores. Use **AutoMLEngine** with **AutoMLConfig** to compare algorithms (demonstrate `agent=True` double opt-in pattern, `max_llm_cost_usd` — governance in action).
2. **UMAP + anomaly detection**: Dimensionality reduction on fraud data, Isolation Forest scoring, visual anomaly inspection. **EnsembleEngine** to combine multiple anomaly detectors (blending scores).
3. **Topic modeling**: BERTopic on Singapore news corpus — topics, temporal evolution, visualization. Use **ModelVisualizer** for topic distribution charts.
4. **DriftMonitor**: Deploy Module 3 model, simulate drift, detect with PSI + KS test using **DriftSpec** (configure thresholds). Frame as **governance obligation**: "In production, you must prove your model still performs. DriftMonitor is how."
5. **Deep learning**: Train CNN on chest X-ray subset with LR scheduling, gradient monitoring, mixed precision. Use **OnnxBridge** to export trained model to ONNX format.
6. **InferenceServer + Nexus**: Deploy ONNX model via InferenceServer, expose through **Nexus** (API + CLI + MCP). Use **ModelSignature** for input validation. Discussion: "Who can access this API?" — primes PACT in M6.

## Datasets
- **E-commerce Transactions**: 200K txns, 50K customers, text reviews, behavioral features (synthetic, Singapore market)
- **Credit Card Fraud** (Kaggle): 284K txns, 0.17% fraud, PCA features
- **Singapore News Corpus**: 50K articles (CC-licensed), multi-topic, temporal
- **ChestX-ray14** (subset): 10K images, multi-label, class imbalance

## Quiz Topics
- EM algorithm: "What happens in E-step vs M-step?"
- UMAP vs t-SNE: "When would you choose one?"
- BERTopic: "What role does HDBSCAN play in the pipeline?"
- DriftMonitor: "PSI = 0.25. What does this mean?"
- Nexus: "Which engine + framework to deploy as REST API?"

## Deck Opening Case
**Credit Suisse AML system** — 99.9% accuracy, 0% useful. With 0.1% fraud rate, predicting "not fraud" for everything gives 99.9% accuracy but catches zero criminals. The false positive flood made the system unusable. Imbalanced data requires principled approaches.
