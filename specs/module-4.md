### MODULE 4: Unsupervised Machine Learning and Advanced Techniques for Insights

**Description**: Pattern discovery without labels, then the bridge to neural feature learning. USML = automated feature engineering (from R5 Deck 5A).

**Module Learning Objectives**: By the end of M4, students can:
- Apply clustering algorithms and evaluate cluster quality
- Implement EM algorithm and understand mixture models
- Reduce dimensionality with PCA, t-SNE, and UMAP
- Detect anomalies using statistical and ML methods
- Discover transactional patterns with association rules
- Extract topics from text using TF-IDF, LDA, and BERTopic
- Build recommender systems using collaborative filtering
- Explain how neural network hidden layers are automated feature engineering with error feedback
- Train a basic neural network with proper training practices

**Kailash Engines**: AutoMLEngine, EnsembleEngine, ModelVisualizer, OnnxBridge

---

#### Lesson 4.1: Clustering

**Prerequisites**: M3 complete (supervised ML)
**Spectrum Position**: USML begins — discovering group structure without labels

**Bridge from M3**: "In M3, you predicted outcomes using labelled data. Now: what if there are no labels? USML discovers structure in data without being told what to look for."

**Topics**:
- **K-means**: algorithm, convergence, elbow method, sensitivity to initialisation (k-means++)
- **Hierarchical clustering** (from Deck 5A):
  - Agglomerative (bottom-up) vs divisive (top-down)
  - Linkage methods: single (min distance), complete (max distance), average, Ward's
  - Dendrograms: reading, cutting threshold
  - Pros/cons of each linkage method (Deck 5A covers this in detail)
- **DBSCAN**: epsilon-neighbourhood, minPts, core/border/noise points
- **HDBSCAN**: hierarchical extension of DBSCAN, auto-selects epsilon
- **Spectral clustering**: graph Laplacian, for non-convex clusters
- **Cluster evaluation**:
  - Internal: silhouette score, Davies-Bouldin index, Calinski-Harabasz index
  - External: ARI (Adjusted Rand Index), NMI (Normalised Mutual Information)
  - Gap statistic: compare within-cluster dispersion to null reference
- **Customer segmentation** application (from PCML5-1)

**Key Formulas**:
- K-means objective: minimise Sum_k Sum_{x in C_k} ||x - mu_k||^2
- Silhouette: s(i) = (b(i) - a(i)) / max(a(i), b(i))
- Davies-Bouldin: DB = 1/k * Sum max_{j!=i} (s_i + s_j) / d(c_i, c_j)

**Learning Objectives**: Students can:
- Apply K-means, hierarchical, DBSCAN, and HDBSCAN clustering
- Read and interpret dendrograms
- Evaluate clusters using silhouette, DB index, and gap statistic
- Select clustering algorithm based on data characteristics

**Exercise**: Customer segmentation on retail data. Compare K-means, hierarchical (with dendrogram), and HDBSCAN. Evaluate with silhouette and DB index. Interpret clusters with business meaning.

**Assessment Criteria**: Multiple algorithms compared. Evaluation metrics computed. Clusters interpreted with business rationale (not just "cluster 1, cluster 2").

**R5 Source**: Deck 5A (K-means, hierarchical with 4 linkage methods, dendrograms, t-SNE) + PCML5-1 (customer segmentation)

---

#### Lesson 4.2: EM Algorithm and Gaussian Mixture Models

**Prerequisites**: 4.1 (clustering), 2.1 (Bayesian thinking)
**Spectrum Position**: Soft clustering — probabilistic assignment to groups

**Topics**:
- Soft vs hard clustering: GMM assigns probabilities, K-means assigns labels
- **EM Algorithm**:
  - E-step: compute responsibilities (probability each point belongs to each cluster)
  - M-step: update parameters (means, covariances, mixing coefficients) using responsibilities
  - Convergence: log-likelihood is non-decreasing
  - 20-line implementation from scratch
- **Gaussian Mixture Models**: EM applied to Gaussian components
- EM as a general template: applicable to any latent variable model
- **Mixture of Experts** (brief): modern application of mixture models (e.g., GPT-4 architecture). Gating network selects expert based on input. Connect to M6 LLMs.

**Key Formulas**:
- E-step: r_nk = (pi_k * N(x_n | mu_k, Sigma_k)) / Sum_j(pi_j * N(x_n | mu_j, Sigma_j))
- M-step: mu_k = Sum_n(r_nk * x_n) / Sum_n(r_nk)
- Log-likelihood: L = Sum_n log(Sum_k pi_k * N(x_n | mu_k, Sigma_k))

**Learning Objectives**: Students can:
- Implement the EM algorithm from scratch (20 lines)
- Explain the difference between hard and soft clustering
- Fit GMMs and interpret component probabilities
- Describe how Mixture of Experts extends mixture models

**Exercise**: Implement EM on 2D synthetic data (3 Gaussians). Compare with sklearn GMM on real e-commerce data. Visualise soft assignments.

**Assessment Criteria**: EM implementation converges. Responsibilities sum to 1. Comparison with GMM shows similar results.

**R5 Source**: ASCENT (new, not in R5)

---

#### Lesson 4.3: Dimensionality Reduction

**Prerequisites**: 4.1 (clustering), 2.5 (linear algebra concepts from regression)
**Spectrum Position**: Feature compression — discovering latent axes

**Topics**:
- **PCA** (from Deck 5A, 2-step process):
  - Step 1: Decorrelate — rotate axes to align with data directions (principal components)
  - Step 2: Reduce — keep top-k components by variance explained
  - SVD connection: PCA via eigendecomposition or SVD (X = U * Sigma * V^T)
  - Scree plot: variance explained per component
  - Loadings: interpret what each component represents
  - Reconstruction error: what information is lost
  - PCA as feature extraction (not just visualisation)
- **Kernel PCA**: nonlinear dimensionality reduction via kernel trick (RBF, polynomial)
- **t-SNE** (from Deck 5A): stochastic neighbour embedding, perplexity parameter, good for visualisation but NOT for feature extraction (non-deterministic, no inverse transform)
- **UMAP**: faster than t-SNE, preserves more global structure, deterministic. Can be used for feature extraction.
- **Manifold learning** (brief mention): Isomap (geodesic distances), LLE (Locally Linear Embedding), MDS (Multidimensional Scaling) — reference table for when to use each
- **Intrinsic dimension**: how many components needed to approximate data (from Deck 5A)

**Key Formulas**:
- PCA: maximise Var(w^T X) subject to ||w|| = 1
- SVD: X = U * Sigma * V^T
- Variance explained: lambda_k / Sum(lambda_i)
- Reconstruction error: ||X - X_hat||^2

**Learning Objectives**: Students can:
- Implement PCA and interpret scree plots and loadings
- Explain the SVD connection to PCA
- Apply t-SNE and UMAP for visualisation and compare results
- Select dimensionality reduction method based on use case (visualisation vs feature extraction vs nonlinear)

**Exercise**: Apply PCA to e-commerce data, interpret first 3 components via loadings. Compare t-SNE vs UMAP visualisations (vary hyperparameters). Demonstrate reconstruction error tradeoff.

**Assessment Criteria**: Scree plot shows variance explained. Loadings interpreted with domain meaning. t-SNE/UMAP hyperparameters varied and compared.

**R5 Source**: Deck 5A (PCA 2-step, t-SNE, intrinsic dimension)

---

#### Lesson 4.4: Anomaly Detection and Ensembles

**Prerequisites**: 4.1 (clustering), 3.5 (evaluation metrics)
**Spectrum Position**: Outlier discovery — finding what doesn't belong

**Topics**:
- **Statistical outlier detection** (from Deck 2A): Z-score method (3 sigma rule), IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR), winsorisation
- **Isolation Forest**: random trees isolate anomalies faster (shorter path length = more anomalous)
- **LOF (Local Outlier Factor)**: density-based, compares local density to neighbours
- **Score blending**: combine multiple anomaly detectors for robustness
- **EnsembleEngine**: `blend()`, `stack()`, `bag()`, `boost()` — unified ensemble API
- Anomaly detection as production monitoring (connects to M3.8 drift monitoring)

**Key Formulas**:
- Z-score: z = (x - x_bar) / s. Outlier if |z| > 3.
- IQR method: outlier if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR
- Isolation Forest anomaly score: s(x, n) = 2^(-E(h(x)) / c(n))
- LOF: LOF_k(x) = (Sum_{o in N_k(x)} lrd_k(o) / lrd_k(x)) / |N_k(x)|

**Learning Objectives**: Students can:
- Apply statistical and ML anomaly detection methods
- Combine multiple detectors using score blending
- Use EnsembleEngine for unified ensemble operations
- Explain when to use each anomaly detection method

**Exercise**: Detect anomalies in financial transaction data using Z-score, Isolation Forest, and LOF. Blend scores. Compare results. Identify true anomalies vs false positives.

**Assessment Criteria**: Multiple methods applied and compared. Blended score improves over individual methods. Business interpretation of detected anomalies.

**R5 Source**: Deck 2A (Z-score, IQR, winsorisation) + ASCENT

---

#### Lesson 4.5: Association Rules and Market Basket Analysis

**Prerequisites**: 4.1 (pattern discovery concept)
**Spectrum Position**: Co-occurrence pattern discovery — finding what appears together

**Topics**:
- Association rules: discovering co-occurrence patterns in transactional data
- **Apriori algorithm**: generate frequent itemsets, prune by minimum support
- **FP-Growth**: compressed representation (FP-tree), no candidate generation, faster than Apriori
- **Metrics**: support (frequency), confidence (conditional probability), lift (surprise factor)
- Applications: retail basket analysis, web click patterns, medical co-diagnoses
- **Forward connection**: association rules discover co-occurrence features. These features can be used as inputs to supervised models (M3). Collaborative filtering (M4.7) extends this to learning latent factors.

**Key Formulas**:
- Support: supp(X) = count(X) / total_transactions
- Confidence: conf(X -> Y) = supp(X u Y) / supp(X)
- Lift: lift(X -> Y) = conf(X -> Y) / supp(Y). Lift > 1 = positive association.

**Design Note**: This is NOT a dead-end topic. It connects forward: (1) discovered rules become features for supervised models, (2) the idea of "finding patterns in co-occurrence data" is exactly what collaborative filtering does with embeddings (M4.7).

**Learning Objectives**: Students can:
- Implement Apriori and FP-Growth for frequent itemset mining
- Compute and interpret support, confidence, and lift
- Extract actionable business rules from transaction data
- Use discovered patterns as features for supervised models

**Exercise**: Market basket analysis on Singapore retail transaction data. Find top association rules. Interpret business meaning. Create features from rules for a classification model.

**Assessment Criteria**: Rules discovered with appropriate support threshold. Business interpretation provided. Rules used as features improve supervised model.

**R5 Source**: New (not in R5). Need new deck content and exercise.

---

#### Lesson 4.6: NLP — Text to Topics

**Prerequisites**: 4.3 (dimensionality reduction), 4.1 (clustering)
**Spectrum Position**: Text feature discovery — extracting meaning from unstructured text

**Topics**:
- Text as data: how to represent text for ML (from Deck 5A: text structure, classification vs clustering)
- **TF-IDF** derivation: term frequency * inverse document frequency. Why it works (common words get low weight).
- **BM25**: improved TF-IDF (saturation, document length normalisation)
- **Word embeddings** (tools, not derivation — derivation in M4.8):
  - Word2Vec (CBOW, Skip-gram): "words that appear in similar contexts have similar meanings"
  - GloVe: global co-occurrence statistics
  - FastText: subword embeddings, handles OOV words
  - **Note**: "How does Word2Vec learn these vectors? We will see in M4.8 when we study neural networks."
- **LDA (Latent Dirichlet Allocation)**: generative topic model. Each document = mixture of topics, each topic = distribution over words.
- **NMF (Non-negative Matrix Factorisation)**: matrix factorisation approach to topics (from Deck 5A, used in NLP)
- **BERTopic**: transformer-based topic modelling, UMAP + HDBSCAN + c-TF-IDF
- **Coherence metrics**: NPMI, UMass. How to evaluate topic quality.
- Sentiment analysis (brief): as application of text classification

**Key Formulas**:
- TF-IDF: tfidf(t, d) = tf(t, d) * log(N / df(t))
- LDA: P(word | document) = Sum_k P(word | topic_k) * P(topic_k | document)
- NPMI: NPMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) * P(w_j))) / -log(P(w_i, w_j))

**Learning Objectives**: Students can:
- Derive and implement TF-IDF from scratch
- Apply LDA and BERTopic for topic extraction
- Evaluate topic quality using coherence metrics
- Use word embeddings as features (without understanding the training yet)

**Exercise**: Extract topics from Singapore news articles using TF-IDF + NMF, LDA, and BERTopic. Compare topic quality using NPMI. Classify sentiment of customer reviews.

**Assessment Criteria**: Multiple topic methods compared. Coherence metrics computed. Topics interpreted with human-readable labels.

**R5 Source**: Deck 5A (NLP text structure, TF-IDF, NMF) + PCML5-2

---

#### Lesson 4.7: Recommender Systems and Collaborative Filtering

**Prerequisites**: 4.3 (PCA/SVD), 4.6 (word embeddings concept)
**Spectrum Position**: THE PIVOT — optimisation drives feature discovery

**Topics**:
- **Content-based filtering**: recommend items similar to what user liked (feature similarity)
- **Collaborative filtering**: recommend items that similar users liked
  - User-based CF: find similar users, recommend their items
  - Item-based CF: find similar items, recommend to users who liked similar items
  - Pros/cons: user-based (cold start for new users) vs item-based (more stable)
- **Matrix factorisation**: factorise user-item matrix R into U * V^T
  - U = user embeddings, V = item embeddings
  - Optimise: minimise ||R - U * V^T||^2 (reconstruction error)
  - ALS (Alternating Least Squares): fix U, optimise V; fix V, optimise U
  - SVD++: extends SVD with implicit feedback
  - Connection to PCA (M4.3): PCA factorises X = U * Sigma * V^T. Collaborative filtering factorises R = U * V^T. Same idea: find low-rank structure.
- **Implicit vs explicit feedback**: ratings (explicit) vs clicks/views/purchases (implicit)
- **Hybrid systems**: combine content-based and collaborative filtering
- **THE PIVOT**: "Matrix factorisation learns user and item embeddings by minimising reconstruction error. This is the first time you've seen OPTIMISATION DRIVE FEATURE DISCOVERY. In M4.8, neural networks generalise this — hidden layer activations ARE embeddings, learned by minimising a loss function."

**Key Formulas**:
- Matrix factorisation: minimise Sum_{(u,i) in observed} (r_ui - u_u^T * v_i)^2 + lambda * (||u_u||^2 + ||v_i||^2)
- ALS update for U: U = (V^T * V + lambda * I)^{-1} * V^T * R^T

**Learning Objectives**: Students can:
- Build content-based and collaborative filtering recommenders
- Implement matrix factorisation with ALS
- Explain how matrix factorisation learns embeddings
- Articulate the connection: "optimisation drives feature discovery" → bridge to neural networks

**Exercise**: Build recommender system on Singapore retail data. Implement user-based CF, item-based CF, and matrix factorisation. Compare recommendation quality. Visualise learned embeddings.

**Assessment Criteria**: All three approaches implemented and compared. Embeddings visualised (2D projection shows meaningful clusters). The pivot concept articulated.

**R5 Source**: PCML5-3 (recommenders, adapted)

---

#### Lesson 4.8: DL Foundations — Neural Networks, Backpropagation, and the Training Toolkit

**Prerequisites**: 4.7 (embeddings, optimisation-driven feature discovery), 2.5 (regression)
**Spectrum Position**: The bridge — hidden layers are USML + error feedback

**Bridge (from Deck 5B)**: "In M4.7, matrix factorisation learned embeddings by minimising reconstruction error. A neural network does the same thing: hidden layer activations ARE embeddings, learned by minimising a loss function. The difference: neural networks can learn NON-LINEAR combinations through activation functions."

**Topics**:
- **Neural network architecture** (from Deck 5B):
  - Input layer, hidden layers, output layer
  - Weights, biases, fully connected layers
  - Each node connected to all nodes in previous layer
- **Forward pass**: multiply inputs by weights, sum at each node, predict
- **Error / Loss**: predicted vs actual
- **Linear regression as a neural network** (from Deck 5B slides 22-31):
  - Regression with zero hidden layers = linear regression
  - Add hidden layers → model "writes its own parametric function"
  - Feature interaction through activation functions → non-linearity
- **Gradient descent** (from Deck 5B, step-by-step with HDB example):
  - Cost function: SSE = 1/2 * Sum((y - y_hat)^2)
  - Gradient: dJ/dw = -x * (actual - predicted)
  - Weight update: w_new = w_old - learning_rate * gradient
  - Iterate until convergence (demonstrate with error decreasing)
- **Backpropagation**: chain rule through layers. Compute gradient of loss with respect to each weight.
- **Hidden layers** (from Deck 5B slides 34-41):
  - 2+ hidden layers can represent ANY non-linear function
  - "Automated feature engineering": hidden layers discover features automatically
  - **Representation learning**: DL learns deep representations of data relationships
  - **Embeddings**: hidden node values encode learned knowledge
  - "Unsupervised meets supervised learning": hidden layers perform unsupervised feature discovery while the output layer performs supervised prediction
- **Regression vs classification output**: 1 node for regression, N nodes for N classes
- **DL Training Toolkit** (new, from completeness audit):
  - **Activation functions**: ReLU, Leaky ReLU, PReLU, ELU, GELU, Swish, Sigmoid, Tanh. When to use each. ReLU default for hidden layers. Sigmoid for binary output. Softmax for multiclass.
  - **Dropout**: randomly zero out neurons during training. Prevents co-adaptation. Dropout rate typically 0.1-0.5. Turn off during inference.
  - **Batch normalisation**: normalise layer inputs to zero mean, unit variance. Stabilises training, enables higher learning rates. Layer norm for transformers (M5.4).
  - **Weight initialisation**: Xavier/Glorot (for sigmoid/tanh), Kaiming/He (for ReLU). Why random init is needed (symmetry breaking). Why zero init fails.
  - **Optimisers**: SGD (+ momentum), RMSProp, Adam (adaptive learning rates), AdamW (decoupled weight decay). Comparison table.
  - **Loss functions taxonomy**: MSE (regression), MAE (robust regression), cross-entropy (classification), binary cross-entropy, focal loss (imbalanced), contrastive loss (similarity), triplet loss (metric learning), KL divergence (distribution matching), reconstruction loss (autoencoders).
  - **Learning rate schedules**: step decay, cosine annealing, warmup + cosine, one-cycle policy, ReduceLROnPlateau.
  - **Gradient clipping**: prevent exploding gradients (max norm or value clipping).
  - **Early stopping**: monitor validation loss, stop when it increases for patience epochs.

**Key Formulas**:
- Forward pass: z = W * x + b, a = f(z)
- Gradient descent: w = w - lr * dL/dw
- Backpropagation chain rule: dL/dw_1 = dL/da_2 * da_2/dz_2 * dz_2/da_1 * da_1/dz_1 * dz_1/dw_1
- Batch norm: y = gamma * (x - mu_B) / sqrt(sigma_B^2 + epsilon) + beta
- Adam: m_t = beta_1 * m_{t-1} + (1-beta_1) * g_t; v_t = beta_2 * v_{t-1} + (1-beta_2) * g_t^2

**Design Note**: This is a dense lesson — the most important lesson in the curriculum. It bridges USML to DL and provides the complete training toolkit. Allocate 4.5 hours if possible.

**Learning Objectives**: Students can:
- Build a neural network from scratch (forward pass, loss, backprop, weight update)
- Explain how hidden layers are automated feature engineering with error feedback
- Select appropriate activation function, optimiser, and loss function
- Apply dropout, batch normalisation, and learning rate scheduling
- Explain representation learning and embeddings

**Exercise**: Build a 3-layer neural network from scratch for HDB price prediction. Implement forward pass, backprop, gradient descent. Then: add dropout, batch norm, Adam optimiser, LR schedule. Compare training curves with and without each technique.

**Assessment Criteria**: From-scratch implementation works (loss decreases). Each training technique improves convergence (demonstrated with plots). The "unsupervised meets supervised" concept articulated.

**R5 Source**: Deck 5B (42 slides, comprehensive) + PCML5-4 (DL basics notebook)

**End of Module Assessment**: Quiz + project (unsupervised analysis → DL bridge: cluster data, reduce dimensions, build neural network on discovered features).
