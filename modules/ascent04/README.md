# Module 4: Unsupervised ML, NLP & Deep Learning

**Kailash**: kailash-ml (AutoMLEngine, EnsembleEngine, DriftMonitor, InferenceServer), Nexus | **Scaffolding**: 40%

## Lecture (3h)
- **4A** Unsupervised ML: spectral clustering, HDBSCAN, GMM (EM algorithm), PCA/t-SNE/UMAP, anomaly detection (Isolation Forest, LOF)
- **4B** NLP: TF-IDF/BM25, Word2Vec/GloVe, LDA/NMF/BERTopic, sentiment analysis
- **4C** Deep Learning: universal approximation, residual connections, attention mechanism, BatchNorm/LayerNorm, CNN, LSTM/GRU, training dynamics

## Lab (3h) — 6 Exercises
1. Clustering comparison: K-means vs spectral vs HDBSCAN vs GMM on customer data
2. UMAP + anomaly detection on fraud transaction data
3. BERTopic on Singapore news corpus with temporal evolution
4. DriftMonitor: deploy model, simulate drift, detect with PSI + KS test
5. Deep learning: CNN on chest X-ray subset with LR scheduling, mixed precision
6. InferenceServer + Nexus: deploy as API + CLI + MCP

## Datasets
E-commerce Transactions (200K), Credit Card Fraud (Kaggle, 284K), Singapore News (50K), ChestX-ray14 (10K subset)
