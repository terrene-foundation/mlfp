# Module 4: Unsupervised Machine Learning and Advanced Techniques for Insights — Speaker Notes

Total time: ~180 minutes (3 hours)
Audience: working professionals; instructors must scaffold for both novices who just finished M3 and ML practitioners who have used sklearn for years.

Module 4 is THE pivot of the MLFP curriculum. The Feature Engineering Spectrum (from the design principles) is the spine of the entire three hours: manual feature engineering (M1–M3) on the left, unsupervised discovery (4.1–4.6) in the middle, the optimisation-driven pivot (4.7), and deep learning (4.8) on the right. Every lesson returns to this spectrum. Instructors should draw it on the whiteboard at the start and point back to it before every new lesson.

The deck has 90 slides. Four lessons have a **Kailash Bridge** slide (4.1 ×2, 4.3, 4.4, 4.8) that maps the theory onto the engine the exercises and the assessment actually use: `ClusteringEngine`, `AutoMLEngine` (search, `agent=False`), `DimReductionEngine`, `AnomalyDetectionEngine`, and `OnnxBridge` (torch export). Teach those bridges carefully — the end-of-module assessment grades through these exact engines.

---

## Slide 1: Module 4 (title)

**Time**: ~2 min
**Talking points**:

- Welcome the class back. Read the provocation aloud: "The algorithm that found $2 billion in hidden fraud had never seen a single labelled example."
- Let it sit. Ask the room: "How is it possible to find something you were never told to look for?"
- Frame the module: "For three modules you have worked with labelled data. Today those labels disappear, and by the end of the day, the features will engineer themselves."
- "If beginners look confused": "Do not worry about the maths yet. We will walk through one algorithm at a time, and each family of techniques has a Kailash engine you can call with a few lines of code."
- "If experts look bored": "This module is not a grab-bag of clustering algorithms. It is a single continuous story that ends with the deep-learning training toolkit — and a clear theoretical bridge from K-means to backpropagation."
  **Transition**: "Let me remind you where we have been so you can see where we are going."

---

## Slide 2: Recap: Your Journey So Far

**Time**: ~2 min
**Talking points**:

- Walk down the table fast: M1 gave you data pipelines (DataExplorer, PreprocessingPipeline), M2 gave you statistical foundations (ExperimentTracker, ModelVisualizer), M3 gave you supervised ML with labelled data (TrainingPipeline, HyperparameterSearch, ModelRegistry).
- Stop on the callout: "Everything so far needed either a human to craft features or a target label. Today those constraints vanish."
- Set the engine expectation honestly: today adds `ClusteringEngine`, `DimReductionEngine`, `AnomalyDetectionEngine` and the `OnnxBridge` — the bridges appear at 4.1, 4.3, 4.4 and 4.8. There is deliberately no engine slide for 4.5 (association rules) or 4.7 (recommenders) — those we build from scratch, because no kailash engine covers them yet.
- "If beginners look confused": "You have already done the hardest part — you built working supervised models in M3. Today is about relaxing assumptions, not learning something new from scratch."
- "If experts look bored": "The M3 → M4 jump mirrors the Hastie / Tibshirani / Friedman ordering: supervised before unsupervised. The pedagogical reason is that the evaluation instinct from M3 transfers directly to cluster quality and topic coherence today."
  **Transition**: "Here is the single slide that organises everything you will learn today."

---

## Slide 3: The Feature Engineering Spectrum

**Time**: ~4 min
**Talking points**:

- Slow down. This is the most important slide in the module.
- Point to the three nodes in order: Manual (M1–M3), USML (today), Deep Learning (4.8 and M5). Read the tagline under each: "Human crafts features", "Algorithm discovers features", "Architecture learns features".
- Make the counts explicit: manual is n → n (you pick the features), clustering and topic models are n → 1 or n → k (the algorithm compresses), deep learning is n → m (the network invents a new feature space whose size you choose).
- Key line to repeat verbatim: "USML is not a separate discipline. It is the bridge between hand-crafted features and deep-learned representations."
- Promise the room: "By lesson 4.7 I will show you the exact moment when optimisation starts discovering features on its own. By lesson 4.8 you will see that hidden layer activations ARE embeddings, just like the ones collaborative filtering learned."
- "If beginners look confused": "Think of the spectrum as three ways to answer the question: who decides what a good feature is? You, the algorithm, or the architecture? Today we move left to right."
- "If experts look bored": "This is Bengio's representation learning argument, compressed. Every lesson today is a point on that spectrum, and the pivot at 4.7 is where matrix factorisation meets gradient descent."
  **Transition**: "Here is the eight-lesson road map that walks us across the spectrum."

---

## Slide 4: Module 4 Road Map

**Time**: ~2 min
**Talking points**:

- Walk the eight lessons quickly. Do not linger on any one — this is the preview.
- Call out the arc: "4.1 through 4.6 all discover features WITHOUT optimisation. 4.7 is the first time optimisation discovers features. 4.8 generalises optimisation-driven feature discovery to any non-linear function."
- Point at 4.7 and 4.8 and say: "These are the two lessons the rest of the module builds towards. Everything before them is setup; everything after them is scale-up."
- "If beginners look confused": "Eight topics sound like a lot. They are not eight disconnected ideas — they are eight chapters in one story, and the story is: how do features come into existence when nobody creates them by hand?"
- "If experts look bored": "Notice the ordering deviates from the usual textbook (clustering → DR → anomaly → rules → NLP → recommenders → DL). That ordering is deliberate — it moves from concrete group discovery to increasingly sophisticated latent-factor models, ending at the pivot."
  **Transition**: "Lesson 4.1. Clustering. The simplest form of unsupervised learning."

---

## Slide 5: 4.1 Clustering (lesson title)

**Time**: ~1 min
**Talking points**:

- Read the subtitle: "Discovering group structure without labels."
- Contrast with M3: "In M3 you predicted outcomes using labelled data. Now: what if there are no labels? Clustering is the answer that asks the data to organise itself."
- Set the lesson tempo: 12 slides plus an exercise, ~25 minutes. Five algorithms and two families of evaluation metrics, then the two Kailash bridges.
  **Transition**: "Start with the workhorse — K-means."

---

## Slide 6: K-Means: The Workhorse

**Time**: ~3 min
**Talking points**:

- Walk the four steps on screen slowly: choose k centroids, assign points to nearest centroid, recompute centroids as cluster means, repeat until stable.
- If the room allows, draw the two-step "assign then update" loop on the whiteboard with three synthetic points and two centroids.
- Read the objective aloud: "Minimise the sum over clusters of the sum over points of the squared distance from each point to its centroid." Emphasise squared — this is why K-means is sensitive to outliers.
- Stop on the warning: "Convergence is guaranteed but only to a LOCAL minimum. Different random starts give different answers. k-means++ spreads the initial centroids apart so bad starts are rare."
- Singapore angle: "Imagine clustering HDB resale transactions by price and floor area. K-means will find roughly four groups: small old flats, small new flats, big old flats, big new flats. You did not define those groups — the algorithm did."
- "If beginners look confused": "Think of a playground with three teachers. Each child goes to the nearest teacher. Then the teachers walk to the middle of their group. Then the children look again. Repeat until nobody moves. That is K-means."
- "If experts look bored": "The Lloyd iteration is coordinate descent on the squared-error objective, which is why it only finds local optima. The k-means++ seeding achieves an O(log k) expected approximation ratio — see Arthur and Vassilvitskii 2007."
  **Transition**: "K-means asks you for k. How do you choose it?"

---

## Slide 7: Choosing k: The Elbow Method

**Time**: ~2 min
**Talking points**:

- Describe the elbow method: plot inertia (the K-means objective J) against k. Look for the kink where adding more clusters stops helping.
- Admit the limitation: "The elbow is subjective. Two analysts can look at the same plot and pick different k. That is why we always combine it with silhouette and the gap statistic."
- Point to the theory callout: "Gap statistic compares your within-cluster dispersion to a null reference distribution. It gives you a principled answer rather than a visual guess." Exercise 4.1 uses the gap statistic with the 1-SE rule — show all three methods in the exercise.
- "If beginners look confused": "It is like buying groceries. At some point, one more item makes almost no difference to how full your bag looks. That point is the elbow."
- "If experts look bored": "In practice, model selection on k should use the gap statistic or BIC on a Gaussian mixture (coming in 4.2) — both are more principled than silhouette."
  **Transition**: "K-means forces every point into a round cluster. What if your clusters are not round?"

---

## Slide 8: Hierarchical Clustering

**Time**: ~3 min
**Talking points**:

- Two directions on this slide. Agglomerative (bottom-up): start with each point as its own cluster, repeatedly merge the two closest clusters, stop when one cluster remains — this is what scipy's `linkage` computes. Divisive (top-down): start with one cluster and split recursively; exhaustive splitting is exponential, so in practice each split is heuristic — bisecting K-means is the usual choice.
- Hit each linkage method with a physical analogy. Single linkage: "nearest neighbours touch" — produces chains. Complete linkage: "worst case dominates" — produces compact spheres. Average: balanced. Ward's: "minimise total variance increase" — usually the best default.
- Emphasise dendrograms: "You do not have to choose k in advance. You run the algorithm once, look at the tree, and cut it at the height that makes business sense."
- Singapore angle: "Imagine merging Singapore planning areas by demographic similarity. The dendrogram tells you: which two areas are most similar, which larger groupings emerge, and where the natural cuts are."
- "If beginners look confused": "A dendrogram is a family tree for your data. You can look at it and decide how far back to group — grandparents, great-grandparents, or all the way back to one ancestor."
- "If experts look bored": "Ward's method greedily minimises the increase in within-cluster sum of squares at each merge — the same quantity K-means minimises — which is why the two usually agree on blob-shaped data."
  **Transition**: "K-means and hierarchical both assume clusters are blobs. DBSCAN drops that assumption."

---

## Slide 9: DBSCAN & HDBSCAN

**Time**: ~3 min
**Talking points**:

- DBSCAN key idea: clusters are dense regions of points, anything sparse is noise. Walk the three categories: core point (has enough neighbours within epsilon), border point (near a core point), noise point (neither).
- Strengths: finds clusters of any shape, identifies noise explicitly, does not require k.
- Pain: choosing epsilon is hard, and DBSCAN fails on varying-density data.
- HDBSCAN: hierarchical extension that auto-selects epsilon, handles varying densities, returns cluster membership probabilities. "In practice, HDBSCAN has replaced DBSCAN almost entirely — it is one of the most reliable clustering algorithms you can use."
- Singapore angle: "If you cluster ride-hailing pickup points across Singapore, the density varies wildly — Orchard is dense, Tuas is sparse. HDBSCAN handles both in one pass; DBSCAN would force you to choose one epsilon and miss one region."
- "If beginners look confused": "Imagine dropping marbles on a map. Wherever they pile up is a cluster. Wherever they are scattered is noise. DBSCAN and HDBSCAN just measure how densely packed your marbles are."
- "If experts look bored": "HDBSCAN constructs a minimum spanning tree of mutual reachability distances, builds a cluster hierarchy, and extracts clusters by maximising stability — see Campello, Moulavi, Sander 2013."
  **Transition**: "DBSCAN still fails on concentric rings. For that we need spectral clustering."

---

## Slide 10: Spectral Clustering

**Time**: ~2 min
**Talking points**:

- Theory layer. Do not derive; explain the intuition. Spectral clustering transforms your data into a space where K-means works, even on non-convex shapes.
- Three steps: build a similarity graph, compute the graph Laplacian L = D − W, take the bottom-k eigenvectors as new features, run K-means in that new space.
- Payoff: handles concentric circles, interleaving spirals, and other geometries that break every other clustering algorithm on this list. Exercise 4.1 runs it on two moons: spectral recovers the moons (ARI 1.0) while Euclidean silhouette prefers the K-means cut.
- Cost: O(n^3) for the eigendecomposition. "Spectral clustering is the right answer when shape matters and n is small to medium. For n > 50,000 it becomes painful."
- "If beginners look confused": "Think of it as changing the view angle. From one angle, the clusters look impossible to separate. Spectral clustering finds the angle where they look separable, then uses K-means there."
- "If experts look bored": "The connection to graph partitioning is through normalised cut (Shi and Malik 2000). The eigenvectors of the normalised Laplacian are a relaxation of the binary cut indicator."
  **Transition**: "You have five algorithms now. How do you know if a clustering is any good?"

---

## Slide 11: Cluster Evaluation: Internal Metrics

**Time**: ~3 min
**Talking points**:

- Internal metrics do not need labels — they measure cluster quality from the geometry alone.
- Silhouette score: for each point, compare its average distance to its own cluster (a) vs its average distance to the nearest OTHER cluster (b). s = (b − a) / max(a, b). Range [−1, 1]. Values near 1 mean a point is well inside its cluster; near 0 means it sits on a border; negative means it probably belongs to a different cluster.
- Davies-Bouldin index: average similarity between each cluster and its most similar neighbour. Lower is better.
- Calinski-Harabasz: ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.
- "If beginners look confused": "Silhouette is the most intuitive. A silhouette near 1 says the point is surrounded by its own cluster. Near 0 says it is on the fence. Negative says it is in the wrong cluster."
- "If experts look bored": "All three internal metrics prefer compact, well-separated convex clusters, which is why they systematically under-rate DBSCAN and HDBSCAN results on non-convex data — the two-moons demo in the exercise makes this concrete. Always pair them with visual inspection."
  **Transition**: "When you do have labels — say, you are validating a clustering against a known ground truth — external metrics are available."

---

## Slide 12: Cluster Evaluation: External Metrics

**Time**: ~2 min
**Talking points**:

- External metrics compare a clustering against a known reference labelling. Use them during validation, never to drive the clustering itself (otherwise you are doing supervised learning).
- ARI (Adjusted Rand Index): agreement between two labellings, corrected for chance. Range [−1, 1]. 1 = identical. 0 = random.
- NMI (Normalised Mutual Information): information-theoretic measure of shared structure. Range [0, 1].
- The slide also lists the gap statistic here — it sits between the two families: it asks "does the structure even exist?" by comparing dispersion to a null reference.
- Use case: "You cluster customers with K-means. Marketing already has a segmentation. ARI tells you how much overlap there is." In Exercise 4.1 the synthetic data has planted labels, so students can check K-means with ARI/NMI before trusting silhouette on real customers.
- "If beginners look confused": "Think of ARI as: of all the pairs of customers in the data, what fraction does my clustering agree about with the reference?"
- "If experts look bored": "NMI is normalised by entropy, so it handles clusters of very different sizes better than ARI. But NMI is biased towards fine-grained clusterings — adjust with AMI (Adjusted Mutual Information) if you compare at different k."
  **Transition**: "With five algorithms and two families of metrics, how do you choose?"

---

## Slide 13: Algorithm Selection Guide

**Time**: ~2 min
**Talking points**:

- Walk the decision rule of thumb: round convex clusters with known k → K-means. Small data and you want the dendrogram → hierarchical. Arbitrary shapes, noise points, variable density → HDBSCAN. Non-convex geometry → spectral. Soft, overlapping, roughly Gaussian groups → GMM (next lesson).
- Remind the room this is a starting rule, not a law. Always try two algorithms and compare.
- "If beginners look confused": "When in doubt, try K-means first because it is fast, then try HDBSCAN because it is flexible. If both agree, you are done. If they disagree, look at the data and decide which one matches reality."
- "If experts look bored": "Empirically, for tabular data, HDBSCAN with default parameters and silhouette-validated K-means cover 90% of real use cases. Spectral and kernel clustering are reserved for image-like or graph-like data."
  **Transition**: "Let us apply this to the canonical unsupervised use case."

---

## Slide 14: Application: Customer Segmentation

**Time**: ~3 min
**Talking points**:

- Customer segmentation is the default real-world clustering application. Telcos, retailers, banks — everyone does it.
- Walk through the typical pipeline: engineer RFM-style features (recency, frequency, monetary — plus tenure, satisfaction, returns), scale them, run K-means, label each cluster with a human-readable persona ("high-value loyalist", "at-risk lapser", "new explorer").
- Stress the leakage trap on the slide: "`churned` is an outcome — kept OUT of the clustering features." If the churn label goes into the clustering, the clusters partly encode the answer, and the downstream churn model that consumes `cluster_id` looks better than it really is. The exercise enforces this exclusion.
- Business interpretation is 80% of the value. A cluster labelled "cluster 3" is worthless; a cluster labelled "high-frequency low-value young families" can drive a campaign. The persona names on the slide are illustrative — the exercise derives its own from cluster profiles.
- "If beginners look confused": "Segmentation is what happens when marketing asks: who are our customers really? Clustering gives them groups they can see, name, and target."
- "If experts look bored": "The honest hard part is feature engineering. Cluster quality depends more on what you feed the algorithm than which algorithm you run — and keeping outcome labels out of the feature set is the cheapest form of leakage control."
  **Transition**: "Kailash wraps this entire pattern behind one engine — two, in fact."

---

## Slide 15: Kailash Bridge: ClusteringEngine

**Time**: ~2 min
**Talking points**:

- This is the first bridge slide: theory → engine. "ClusteringEngine puts four of today's algorithms — kmeans, gmm, dbscan, spectral — behind one `.fit()` call that returns labels, silhouette, Calinski-Harabasz and inertia."
- Walk the code: load `mlfp03/ecommerce_customers.parquet`, select the seven behaviour features (note again: `churned` excluded), drop nulls, sample 3,000 rows, standardise, then `engine.sweep_k(X, k_range=range(2, 9), criterion="silhouette")` and `engine.fit(X, algorithm="kmeans", n_clusters=sweep.optimal_k)`.
- Be honest about scope: Ward/agglomerative and HDBSCAN are NOT in the engine — the exercise uses scipy and the `hdbscan` library for those. `sweep_k` works for kmeans/gmm/spectral; `fit` also takes dbscan.
- Report the real result: on the 3,000-customer sample this picks K=3 with silhouette ≈ 0.18. Real customer data does not form crisp clusters — that is why business profiling matters more than the score. Set that expectation now so nobody is disappointed in the exercise.
- This is the engine the assessment's clustering task uses — say so explicitly.
- "If beginners look confused": "The engine is the same K-means you just learned, plus bookkeeping. `sweep_k` is the elbow loop you would have written; `fit` returns the metrics alongside the labels."
- "If experts look bored": "Check the returned object — labels, per-algorithm metrics and the fitted sklearn estimator are all on it, so you can drop to the sklearn layer whenever the engine surface is not enough."
  **Transition**: "One level up: what if you want the engine to search algorithms AND K for you, with a budget and an audit trail?"

---

## Slide 16: Kailash Bridge: AutoMLEngine Search

**Time**: ~2 min
**Talking points**:

- Continue from the previous slide (same engine, same X). Key line: "AutoMLEngine does not know what clustering is. YOU declare the search space and a trial function that trains one candidate and reports its metric. The engine runs the search, enforces the trial / time / cost budget, and keeps the audit record."
- Walk the code: `AutoMLConfig(task_type="clustering", metric_name="silhouette", direction="maximize", search_strategy="grid", max_trials=12, agent=False, max_llm_cost_usd=1.0)`; a `space` of two `ParamSpec`s (algorithm ∈ {kmeans, gmm}, n_clusters ∈ [3, 8]); a `trial_fn` that calls `engine.fit(...)` and returns a `TrialOutcome` with the silhouette; then `asyncio.run(automl.run(space=space, trial_fn=trial_fn))` and read `result.best_trial`.
- Explain `agent=False`: no LLM is called. `agent=True` would let an LLM propose trials — that costs money and is non-deterministic, so it needs a double opt-in (the flag plus a cost cap).
- Governance note: without a governance engine or database attached, the engine logs that admission checks are skipped and keeps trials in memory — fine for class, not for production.
- This is exactly the search in Exercise 4.1, file 5.
- "If beginners look confused": "AutoMLEngine is a robot that runs the sweep loop for you and writes down everything it tried. You still own the training code — it owns the search and the receipt."
- "If experts look bored": "The pattern is bring-your-own-trainer: the engine is governance + audit around any `trial_fn`, which is why the same engine covers supervised search in M3 and this clustering sweep."
  **Transition**: "Time to get your hands dirty."

---

## Slide 17: Exercise 4.1: Customer Segmentation

**Time**: ~2 min
**Talking points**:

- Describe the task: segment Singapore e-commerce customers on behaviour features — `churned` excluded. Exercise 1 is a DIRECTORY of five files (`solutions/ex_1/01_kmeans.py` … `05_evaluation_profiling.py`), one technique each.
- Walk the five files: 1.1 K-means++ with elbow, silhouette and the gap statistic (1-SE rule); 1.2 hierarchical with four linkages + Ward dendrogram; 1.3 DBSCAN + HDBSCAN; 1.4 spectral on two moons — spectral recovers them (ARI 1.0) while silhouette prefers K-means; 1.5 silhouette, DB, CH + ARI/NMI agreement, then the AutoMLEngine search from the bridge slide.
- Flag the 1.4 lesson explicitly: "Silhouette is a Euclidean, convexity-loving score — it can rank the WRONG answer higher on non-convex shapes. Always look at the clusters too."
- The assessment criterion is not speed — it is interpretation. "I would rather see three well-labelled clusters than twenty unnamed ones."
- "If beginners look confused": "The exercise walks you through step by step. You do not have to write the loop yourself — the scaffolding gives you the structure."
  **Transition**: "Before we break, one more clustering idea. What if a point belongs to multiple clusters at once?"

---

## Slide 18: 4.2 EM Algorithm & Gaussian Mixture Models (lesson title)

**Time**: ~1 min
**Talking points**:

- Read the subtitle: "Soft clustering — probabilistic assignment to groups."
- Motivate the lesson: "K-means gives every point exactly one cluster. That is wrong for customers who shop both as a family and as an individual. What if a customer is 60% loyal and 40% at-risk? GMMs give you that."
- Preview the 20-line EM implementation the exercise builds from scratch.
  **Transition**: "The core distinction: hard vs soft."

---

## Slide 19: Hard vs Soft Clustering

**Time**: ~3 min
**Talking points**:

- Hard clustering (K-means, DBSCAN): every point gets exactly one label.
- Soft clustering (GMM): every point gets a probability distribution over clusters. A customer might be 70% "family shopper" and 30% "convenience shopper".
- Why this matters: soft probabilities give k probability features per point — richer than a single label. You can weight features, compute expected values across segments, and detect uncertain assignments that may need human review.
- Singapore angle: "Classify HDB flats by type — is a jumbo flat a normal flat or a big flat? Under K-means you pick one. Under GMM it gets, say, 60% normal, 40% big, which is more honest."
- "If beginners look confused": "Hard clustering is like yes/no. Soft clustering is like 70%/30%. Sometimes yes/no is fine. Sometimes you need to express uncertainty."
- "If experts look bored": "Hard assignments are the argmax of soft assignments. K-means is the low-variance limit of EM on a spherical Gaussian mixture with equal mixing weights — they are the same algorithm at different temperatures."
  **Transition**: "To learn soft assignments we need the EM algorithm. Two steps that alternate."

---

## Slide 20: The EM Algorithm: E-Step

**Time**: ~3 min
**Talking points**:

- EM = Expectation-Maximisation. Two alternating steps that increase log-likelihood at every iteration.
- E-step (Expectation): for each point, compute the responsibility r_nk = the probability that point n was generated by component k. This is Bayes' rule given current parameters.
- Read the formula aloud slowly: r_nk = (mixing weight times Gaussian likelihood) / (sum over all components of the same thing).
- Intuition: "Given my current guesses for the means and covariances, how likely is this point to have come from each cluster? Normalise those likelihoods so they sum to 1. Those are the responsibilities."
- "If beginners look confused": "Forget the formula. The E-step is: for each customer, ask each cluster how likely it thinks that customer belongs to it, then turn those into percentages that add to 100."
- "If experts look bored": "The E-step computes the exact posterior over the latent assignment — possible in closed form here because the model is a conjugate exponential-family mixture."
  **Transition**: "Once every point has responsibilities, update the parameters."

---

## Slide 21: The EM Algorithm: M-Step

**Time**: ~3 min
**Talking points**:

- M-step (Maximisation): update the parameters using the responsibilities as weights. New mean = weighted average of points. New covariance = weighted covariance. New mixing weight = average responsibility.
- Read the formula: mu_k = sum over n of (r_nk times x_n) divided by sum over n of r_nk.
- Compare to K-means: "If K-means is hard voting, the M-step is soft voting. In K-means, the centroid update is the mean of assigned points. In EM, the centroid update is the WEIGHTED mean of ALL points, weighted by responsibility. Every point contributes to every cluster, proportional to its responsibility."
- "If beginners look confused": "The M-step says: now that I know how much each customer belongs to each cluster, I recompute the clusters using everyone's contribution, weighted by how much they belong."
- "If experts look bored": "The M-step is the closed-form MLE of a weighted Gaussian — no gradient descent needed because the expected complete-data log-likelihood is concave in (mu, Sigma-inverse) given fixed responsibilities."
  **Transition**: "Alternate E and M. When do you stop?"

---

## Slide 22: EM Convergence & Log-Likelihood

**Time**: ~2 min
**Talking points**:

- Log-likelihood is guaranteed non-decreasing at every EM iteration. Plot it — it should rise and plateau.
- Convergence criterion: stop when the improvement between iterations drops below a tolerance (e.g. 1e-4).
- Warning: EM converges to a LOCAL maximum of the log-likelihood. Run multiple starts and keep the best.
- Visual: show the log-likelihood curve. It is monotone non-decreasing — a line that curves into a plateau. If you see it drop, you have a bug.
- "If beginners look confused": "The algorithm has a score it is trying to improve. Every round, it goes up or stays the same. When it stops going up, you stop."
- "If experts look bored": "The monotonicity proof comes from Jensen's inequality applied to the expected complete-data log-likelihood. EM is a lower-bound maximisation, which is why it cannot decrease."
  **Transition**: "Apply this framework to Gaussian components and you get GMM."

---

## Slide 23: GMM: EM Applied to Gaussians

**Time**: ~3 min
**Talking points**:

- GMM: assume the data was generated by a mixture of Gaussian distributions. Each Gaussian has a mean, a covariance, and a mixing weight. EM estimates all three.
- Visual: two overlapping Gaussian blobs. K-means draws a hard boundary; GMM draws contour lines of probability.
- Use GMM when your clusters are elliptical (not round), when you want soft probabilities, or when you want a generative model you can sample from. Model selection: BIC or AIC to pick the number of components.
- Read the "EM as a general template" box carefully — it is the corrected framing: "EM applies to latent-variable models such as GMMs and LDA (4.6, via variational EM). ALS in 4.7 shares the alternating structure but is NOT EM; neural networks (4.8) are trained by gradient descent." Do not let the room leave thinking EM trains neural networks.
- Singapore angle: "HDB resale transactions by price and area form elliptical clusters (prices and areas are correlated within flat types). K-means forces circles; GMM fits the actual shape."
- "If beginners look confused": "GMM is K-means with ellipses instead of circles, and with percentages instead of hard labels."
- "If experts look bored": "Full covariance GMMs have k·d·(d+1)/2 covariance parameters and overfit on small data. Use tied or diagonal covariance as a regulariser when d is large."
  **Transition**: "Mixture models are not just a clustering trick. Modern LLMs use the same structure."

---

## Slide 24: Mixture of Experts: Modern Application

**Time**: ~2 min
**Talking points**:

- Mixture of Experts (MoE): modern architecture where multiple "expert" networks specialise in different input regions. A gating network selects which experts process each input.
- This is the same mathematical structure as GMM — mixing weights, component-specific parameters — but the components are neural networks instead of Gaussians.
- Concrete, on the slide: the open-weight Mixtral 8x7B routes each token to 2 of 8 experts per layer, so only part of the weights is active per token. Use open models with published architectures as the example; the architectures of closed models are not disclosed — do not state them as fact.
- Payoff for today: "The soft-assignment template is not a toy. GMMs assign each point to components probabilistically; MoE assigns each token to experts probabilistically. Same math, different scale."
- "If beginners look confused": "Think of a hospital with specialists. When you walk in, a receptionist looks at your symptoms and routes you to the right doctor. MoE is the same idea — a gate routes each input to the right expert network."
- "If experts look bored": "Sparse MoE is trained with top-k gating and an auxiliary load-balancing loss — see Shazeer et al. 2017 and GShard — but the underlying probabilistic structure is still the mixture model you just derived."
  **Transition**: "Let us make all of this concrete in twenty lines of code."

---

## Slide 25: Exercise 4.2: EM from Scratch

**Time**: ~2 min
**Talking points**:

- Task: implement the EM algorithm in roughly twenty lines on 2D synthetic data with three Gaussians; then verify against sklearn's GMM on the same data, and on real e-commerce customers choose K by BIC/AIC and compare covariance types.
- The three correctness checks: (1) responsibilities sum to 1 for every point, (2) log-likelihood never decreases, (3) the from-scratch implementation matches sklearn's GMM closely (the exercise compares log-likelihood, weights and means after label matching — the gap is about 1e-8 per sample).
- The last file reads soft assignments like a practitioner: boundary customers, confidence, and Mixture-of-Experts gating.
- "If beginners look confused": "The scaffolding gives you the structure. You fill in E-step and M-step. Twenty lines is a promise, not a challenge."
  **Transition**: "Soft clustering discovers groups. Dimensionality reduction discovers axes. Same theme — the algorithm discovers structure — different object."

---

## Slide 26: 4.3 Dimensionality Reduction (lesson title)

**Time**: ~1 min
**Talking points**:

- Subtitle: "Feature compression — discovering latent axes."
- Framing: "Clustering compresses n data points into k groups. Dimensionality reduction compresses n features into k features. Both discover structure. Both are automated feature engineering."
- Preview: PCA, kernel PCA, t-SNE, UMAP, manifold learning, intrinsic dimension — ten slides plus the DimReductionEngine bridge, ~25 minutes.
  **Transition**: "Start with the workhorse of the other side: PCA."

---

## Slide 27: PCA Step 1: Decorrelate

**Time**: ~3 min
**Talking points**:

- PCA is a two-step process. Step 1: decorrelate. Rotate the axes so they align with the directions of greatest variance in the data.
- Geometric intuition: imagine a cloud of points shaped like a tilted ellipse. The original x and y axes are correlated. Rotate so the new axes align with the long and short sides of the ellipse. The new axes are uncorrelated.
- The new axes are the principal components — eigenvectors of the covariance matrix, in decreasing order of eigenvalue. First PC maximises projected variance; subsequent PCs are orthogonal to the previous ones.
- Emphasise: "PCA does not discard features yet — it only rotates."
- "If beginners look confused": "Imagine you take a photo of a banana from the wrong angle. PCA rotates the banana so you are looking along its length. Same banana, better axes."
- "If experts look bored": "PCA is the eigendecomposition of the covariance matrix, or equivalently the SVD of the centred data matrix — the eigenvectors are the principal directions and the eigenvalues are the variances along those directions."
  **Transition**: "Decorrelating is half the job. Reducing is the other half."

---

## Slide 28: PCA Step 2: Reduce

**Time**: ~2 min
**Talking points**:

- Step 2: drop the lowest-variance components. Keep the top k that explain, say, 95% of the variance.
- Scree plot: bar chart of variance explained per component. "The scree plot is to PCA what the elbow plot is to K-means. Look for the bend."
- Loadings: each principal component is a linear combination of the original features. Looking at the loadings tells you what the component "means" in business terms.
- Singapore angle: "PCA on HDB features — floor area, storey, remaining lease, age, flat type — usually finds PC1 = size, PC2 = age/newness, PC3 = location premium. You did not design those axes; PCA discovered them."
- "If beginners look confused": "PCA is like packing for a trip. You have fifty items but only one bag. PCA keeps the most important items (high variance) and leaves out the rest."
  **Transition**: "PCA has a twin: the singular value decomposition."

---

## Slide 29: PCA via SVD

**Time**: ~3 min
**Talking points**:

- X = U Σ Vᵀ, where X is your centred data matrix, U and V are orthogonal matrices, and Σ is diagonal.
- Principal components = columns of UΣ. Loadings = columns of V — that is what you interpret.
- Why SVD instead of eigendecomposition of the covariance matrix? Numerically more stable, and it works when n < d (more features than samples).
- Plant the seed for 4.7: "Remember this factorisation. When we get to recommender systems, you will see R ≈ U Vᵀ — PCA factorises X; collaborative filtering factorises the user-item matrix R. Same algebra, different data."
- "If beginners look confused": "SVD is a way to break any matrix into three simpler pieces. PCA is what you get when you apply SVD to your data and keep only the biggest pieces."
- "If experts look bored": "SVD gives you the PCA basis without ever forming XᵀX, which matters when d is in the millions and the covariance matrix does not fit in memory."
  **Transition**: "How much information did we lose?"

---

## Slide 30: Reconstruction Error

**Time**: ~2 min
**Talking points**:

- Reconstruction error: project the data onto the top-k components, then project back. Compute ‖X − X̂‖²_F.
- Read the slide formula precisely: for centred X with n rows, ‖X − X̂‖²_F = Σ_{i>k} σ_i² = (n−1) Σ_{i>k} λ_i, where σ_i are the singular values and λ_i the covariance eigenvalues. Information lost = the discarded variance.
- Use reconstruction error to decide k when you cannot eyeball a scree plot. Choose k to retain (e.g.) 95% of total variance.
- Connection to 4.7: "Matrix factorisation in 4.7 also minimises a reconstruction error. PCA minimises it under orthogonality; matrix factorisation minimises it under regularisation, over observed entries only."
- "If beginners look confused": "Reconstruction error is: if I throw away some information and then try to rebuild the data, how much am I off by? Less error means I threw away less important stuff."
  **Transition**: "PCA is linear. What if the structure is curved?"

---

## Slide 31: Kernel PCA

**Time**: ~2 min
**Talking points**:

- Linear PCA finds straight-line axes. Kernel PCA uses the kernel trick to find non-linear axes — the same move kernel SVM made over linear SVM in M3.
- Common kernels: RBF (Gaussian), polynomial. Use kernel PCA when your data has clear non-linear structure (circles, spirals) that linear PCA cannot capture.
- Two honest limitations on the slide: no explicit loadings (harder to interpret), and no exact inverse — reconstructing input-space points (the pre-image problem) needs an approximate, learned inverse (sklearn `fit_inverse_transform=True`).
- Cost: O(n^2) memory for the kernel matrix. Not practical beyond tens of thousands of points.
- "If beginners look confused": "Linear PCA only draws straight lines. Kernel PCA can draw curves. If your data is curved, you need kernel PCA."
- "If experts look bored": "Kernel PCA is eigen-analysis of the centred kernel matrix — linear PCA in the feature space induced by the kernel map, which can be infinite-dimensional for RBF. The pre-image problem is ill-posed precisely because that map is not invertible."
  **Transition**: "PCA is linear. Kernel PCA is kernelised. For visualisation specifically, t-SNE goes a different direction."

---

## Slide 32: t-SNE: Visualisation Specialist

**Time**: ~2 min
**Talking points**:

- t-SNE = t-distributed Stochastic Neighbour Embedding. Converts distances to probabilities in high-D, then finds a low-D layout with similar probabilities by minimising KL divergence.
- Key parameter: perplexity — the effective number of neighbours, typical values 5–50. Vary it; the plot can change significantly. The exercise sweeps 5, 15, 30, 50.
- Strengths and weaknesses on the slide: preserves local structure well, distorts global structure, non-deterministic (fix the seed), and CANNOT embed new points — so it is not a feature extractor.
- CRITICAL caveat: "t-SNE is for LOOKING, not for feeding into downstream models. Distances between clusters in t-SNE plots are NOT meaningful — only local neighbourhood structure is preserved."
- "If beginners look confused": "t-SNE is the algorithm that makes those beautiful scatter plots where MNIST digits end up in clean separated blobs. It is for looking, not for modelling."
- "If experts look bored": "t-SNE minimises the KL divergence between Gaussian similarities in input space and Student-t similarities in 2D space. The heavy-tailed Student-t fixes the crowding problem that affects SNE."
  **Transition**: "t-SNE has a modern replacement that fixes most of its weaknesses."

---

## Slide 33: UMAP: The Modern Default

**Time**: ~2 min
**Talking points**:

- UMAP = Uniform Manifold Approximation and Projection. Walk the comparison table: faster than t-SNE, better global layout, scales to ~1M+ points, and `.transform()` embeds new rows — so it IS usable as a feature extractor.
- Two key hyperparameters: n_neighbors (local vs global structure trade-off) and min_dist (how tightly points cluster). The exercise sweeps four (n_neighbors, min_dist) settings.
- Warn precisely: "In a UMAP plot, cluster sizes and the distances BETWEEN clusters are still not quantitatively meaningful — read neighbourhoods, not gaps." The global layout is better than t-SNE's, not metric.
- Complexity note: exact t-SNE is O(n²); sklearn's default Barnes-Hut is O(n log n) but still much slower than UMAP. Both are stochastic — fix random_state to reproduce a plot.
- UMAP appears again inside BERTopic (4.6) as a preprocessing step, and for visualising learned embeddings (4.7).
- "If beginners look confused": "UMAP makes the same kind of beautiful 2D plots as t-SNE, but faster and more reliable. When in doubt, use UMAP."
- "If experts look bored": "UMAP is grounded in Riemannian geometry and algebraic topology — it constructs a fuzzy topological representation of the data and optimises a low-dimensional representation to match it. See McInnes, Healy, Melville 2018."
  **Transition**: "PCA, Kernel PCA, t-SNE, UMAP. A few more specialised methods you should know exist."

---

## Slide 34: Manifold Learning: Reference Table

**Time**: ~1 min
**Talking points**:

- This slide is a reference, not a deep dive. Walk the table: Isomap (geodesic distances, unrolls curved manifolds), LLE (local linear reconstruction — Swiss roll), MDS (preserves pairwise distances — takes a distance matrix, not features).
- Honesty: "PCA and UMAP cover 90% of practical use cases. Know these exist so you are not caught off guard if a paper uses them."
- "If beginners look confused": "Do not memorise this table. Just know it exists. If you ever need to unroll a Swiss roll, come back here."
  **Transition**: "How many dimensions do you actually need?"

---

## Slide 35: Intrinsic Dimensionality

**Time**: ~2 min
**Talking points**:

- Intrinsic dimension: the minimum number of parameters needed to describe the data. Your data might have 100 columns but only 5 degrees of freedom.
- Estimation: scree plot (variance threshold), correlation dimension, nearest-neighbour MLE estimators. Practical heuristic on the slide: choose k components that explain ≥ 95% cumulative variance — but note this is a LINEAR estimate; a curved manifold may need fewer dimensions, which is what the MLE estimators catch.
- Why it matters: "If the intrinsic dimension is low, you are wasting compute on high-dimensional methods. A linear model with 5 well-chosen features often beats a neural network with 100 raw features."
- Exercise preview: the Levina–Bickel nearest-neighbour MLE estimator returns about 8 on 8-dimensional synthetic data — a useful sanity check before trusting it on the customers. (The estimator needs the self-neighbour dropped; the exercise does this correctly.)
- "If beginners look confused": "A photo has three colour channels and millions of pixels. But the faces in the photo can be described by maybe fifty numbers (nose shape, eye colour, jawline). Fifty is the intrinsic dimension. Millions is the measured dimension."
- "If experts look bored": "Intrinsic dimension connects to the manifold hypothesis — natural data lies on low-dimensional manifolds embedded in high-dimensional measurement spaces. This is the theoretical justification for representation learning."
  **Transition**: "One engine covers the four reducers you will actually use."

---

## Slide 36: Kailash Bridge: DimReductionEngine

**Time**: ~2 min
**Talking points**:

- Second bridge slide. "One `reduce()` call, four algorithms: pca, nmf, tsne, umap. The result object carries the embedding, the explained-variance ratio and the reconstruction error you computed by hand from the SVD."
- Walk the code: `DimReductionEngine().reduce(X, algorithm="pca", n_components=5)` → print `explained_variance_ratio` (the scree plot's numbers) and `reconstruction_error` (what the dropped components cost). Then `reduce(X, algorithm="umap", n_components=2, seed=42)` → `emb.transformed` gives 2-D coordinates to plot.
- Report the real numbers: on the 3,000-customer sample the first five PCs explain about 26%, 23%, 15%, 14%, 14% — there is no single dominant axis, so do not expect a sharp scree elbow on real customer data. (Five components ≈ 92%, six ≈ 98% cumulative.)
- Scope honesty: kernel PCA, Isomap and LLE are NOT in the engine — use scikit-learn for those (Exercises 3.2 and 3.5 do).
- Forward link: the same engine with `algorithm="nmf"` does topic extraction in 4.6. The assessment's dimensionality-reduction and topic tasks use this engine — say so.
- "If beginners look confused": "The engine is the scree plot and the projection code you would have written, packaged. You choose the algorithm; it returns the numbers and the coordinates."
- "If experts look bored": "Because the result object carries both the variance ratios and the reconstruction error, you can sanity-check the engine against your own SVD — the exercise does exactly that in 3.1."
  **Transition**: "Let us put the reducers to work."

---

## Slide 37: Exercise 4.3: PCA, t-SNE & UMAP

**Time**: ~2 min
**Talking points**:

- Task: reduce the Singapore e-commerce customer features (`churned` excluded). Five files: 3.1 PCA via SVD — scree, loadings, reconstruction trade-off; 3.2 kernel PCA — linear vs RBF vs polynomial, RBF gamma sweep; 3.3 t-SNE at perplexity 5, 15, 30, 50; 3.4 UMAP at four (n_neighbors, min_dist) settings plus `.transform()` on held-out rows; 3.5 rank every reducer by TRUSTWORTHINESS and estimate intrinsic dimension.
- Explain why trustworthiness and not silhouette: "Clustering an embedding and scoring the clusters rewards methods that tear the data into blobs — t-SNE does this by design. Trustworthiness asks the right question: are a point's neighbours in the embedding also its neighbours in the original space?"
- Assessment: scree plot with variance explained, loadings interpreted in business terms, hyperparameter sensitivity explored.
- "If beginners look confused": "Follow the files in order. The scaffolding makes the plots for you. Your job is to look at them and write what you see."
  **Transition**: "Groups of similar points. Axes of variation. Now: points that belong to no group and no axis. Anomalies."

---

## Slide 38: 4.4 Anomaly Detection (lesson title)

**Time**: ~1 min
**Talking points**:

- Subtitle: "Outlier discovery — finding what doesn't belong."
- Framing: "Clustering finds groups of similar points. Anomaly detection finds points that belong to NO group. Two sides of the same coin."
- Preview: Z-score and IQR (from M2), Isolation Forest, LOF, score blending, the AnomalyDetectionEngine bridge, production — eight slides, ~20 minutes.
  **Transition**: "Start with the statistical methods you already know from M2."

---

## Slide 39: Statistical Outlier Detection

**Time**: ~2 min
**Talking points**:

- Z-score: z = (x − mean) / std. Flag anything with |z| > 3 (three-sigma rule).
- IQR method: outlier if x < Q1 − 1.5·IQR or x > Q3 + 1.5·IQR. More robust because it does not assume normality.
- Winsorisation: cap extreme values at a percentile instead of removing them. Useful when you cannot afford to lose rows.
- Critical limitation: "These are univariate. They look at one column at a time. Real anomalies are often multivariate — a point can be normal in every single feature but anomalous in the combination."
- Singapore angle: "A S$500,000 HDB resale transaction is normal. A 40-year-old flat is normal. A 40-year-old flat selling for S$500,000 might be highly unusual depending on the type. Univariate tests miss this; multivariate methods catch it."
- "If beginners look confused": "Z-score says: is this number far from average? IQR says: is this number far from the middle half? Simple, but only looks at one column at a time."
- "If experts look bored": "Z-score breaks down on heavy-tailed distributions because the variance is inflated. Use the modified Z-score with median absolute deviation (MAD) for robustness: z_mod = 0.6745·(x − median)/MAD."
  **Transition**: "For multivariate anomalies, we need ML methods. Start with the most practical one."

---

## Slide 40: Isolation Forest

**Time**: ~3 min
**Talking points**:

- Idea: build random trees that isolate points by random splits. Anomalies are isolated in fewer splits. Normal points need many splits.
- Intuition: "Anomalies are few and different. Few means they get isolated quickly by random splits. Different means they get separated early in the tree."
- Score: s(x, n) = 2^(−E[h(x)] / c(n)), where h(x) is the path length and c(n) the average path length in a BST of n points. Values near 1 are anomalies. Walk the exponent direction carefully: anomalous points have SHORT average path lengths, so E[h]/c(n) is close to 0, the exponent approaches 0, and s → 1. Normal points have E[h] ≈ c(n) or larger, giving s ≈ 0.5 or less.
- Why it is the most practical anomaly detector: fast (sub-linear training), handles high dimensions, makes no distribution assumptions, scales to millions of points.
- Singapore angle: "Fraudulent credit applications — anomalous income declarations combined with unusual employment histories and unusual contact details. Isolation Forest catches them at screening time."
- "If beginners look confused": "Imagine playing twenty questions. A normal customer takes twenty questions to identify. A fraudster takes three. Isolation Forest measures how quickly it can identify each point."
- "If experts look bored": "The expected path length for a point in a random tree under uniform splits is logarithmic in n. The score normalises against this baseline. Extended Isolation Forest improves on axis-parallel splits with random hyperplanes."
  **Transition**: "Isolation Forest is great for global anomalies. For local anomalies we need a density-based method."

---

## Slide 41: Local Outlier Factor (LOF)

**Time**: ~2 min
**Talking points**:

- Theory layer. LOF compares a point's local reachability density to its neighbours'. LOF ≫ 1 means the point sits in a sparse pocket relative to its neighbourhood.
- When to use: data with regions of very different density, where Isolation Forest may miss locally anomalous points. Cost: O(n²) naive.
- Spend real time on the masking box — it is the lesson's trap: "LOF compares a point with its k nearest neighbours. If a coordinated fraud ring submits many near-identical applications and the ring has at least k members, each ring member's k neighbours are other ring members — equally dense — so its density ratio is about 1 (or below) and LOF scores it as NORMAL." A tight, dense group of anomalies with ≥ k members is invisible to LOF.
- Therefore k is a domain decision: it must exceed the largest coordinated group you want to catch. Exercise 4.3 sweeps n_neighbors on a 40-member injected application ring and settles on k=50 from that ring-size assumption — the ring is masked at k=20 (AUC ≈ 0.22) and caught at k=50 (AUC ≈ 0.89).
- "If beginners look confused": "Imagine a crowded street with one empty bench. LOF measures how empty the space around each person is compared to their neighbours. The person on the empty bench scores high. But if forty people all sit on forty empty benches together, none of them looks unusual to LOF."
- "If experts look bored": "LOF uses reachability distance rather than raw distance to dampen density variation. See Breunig, Kriegel, Ng, Sander 2000 for the original formulation and its sensitivity to k — masking is the sharp end of that sensitivity."
  **Transition**: "No single detector is best. Blend them."

---

## Slide 42: Score Blending: Combining Detectors

**Time**: ~2 min
**Talking points**:

- No single anomaly detector works everywhere. Z-score catches distributional outliers. Isolation Forest catches multivariate anomalies. LOF catches density anomalies. Blend them.
- Recipe: normalise each detector's scores to [0, 1], then combine with a simple average, weighted average, rank average, or max-vote.
- Say the honest version on the slide: "Blends are OFTEN more robust — but never guaranteed. A weak detector can drag the blend below its best member." A blend helps when detectors make DIFFERENT mistakes and each is better than chance on what it sees; averaging in a detector blind to an anomaly type dilutes the ones that catch it. Always compare the blend against its best single member on a labelled review sample — the exercise does, and on the course data the equal-weight blend can underperform LOF alone.
- "If beginners look confused": "If three detectives investigate a crime and all three flag the same suspect, you trust the verdict. Score blending is the same principle — but check that each detective is actually competent first."
- "If experts look bored": "Blending reduces variance when detector errors are weakly correlated and scores are comparably calibrated. Unsupervised stacking is an active research area — LODA (Pevny 2016) builds many weak histogram detectors, AutoOD picks combinations automatically."
  **Transition**: "Kailash wraps the detectors and the blend behind one engine."

---

## Slide 43: Kailash Bridge: AnomalyDetectionEngine

**Time**: ~2 min
**Talking points**:

- Third bridge slide. "AnomalyDetectionEngine fits the detector, flips sklearn's sign convention so that higher always means more anomalous, normalises the scores to [0, 1], and applies the contamination threshold."
- Walk the code: `AnomalyDetectionEngine().detect(X, algorithm="isolation_forest", contamination=0.01)` — algorithms are `isolation_forest | lof | one_class_svm`; `iso.scores` are in [0, 1], higher = more anomalous. Then `ensemble_detect(X, algorithms=["isolation_forest", "lof"], contamination=0.01, voting="score_average")` — that is the normalised-score blend from the previous slide; the default `voting="majority"` votes on labels instead.
- Contamination caveat: "Contamination only moves the cut-off — it does not change the ranking." If your top-10 flagged points are all normal, raising contamination does not fix the detector; it just flags more of the same ranking.
- Draw the boundary clearly: "Do NOT reach for `EnsembleEngine.blend` to blend anomaly scores. `blend()` and `stack()` are SUPERVISED classifier ensembles — they vote fitted classifiers against a target column and do their own train/test split. They become useful once analysts have labelled a review sample; Exercise 4.4 trains a logistic regression and a random forest on the detector scores and blends / stacks them that way."
- This is the engine the assessment's anomaly task uses.
- "If beginners look confused": "The engine is the three detectors plus the blend recipe, with one consistent score meaning: higher = more suspicious, always."
- "If experts look bored": "The sign-convention flip matters more than it looks — raw sklearn gives you `decision_function` where higher = more normal for Isolation Forest and `negative_outlier_factor_` for LOF. The engine normalises both so downstream blending code never has to remember which is which."
  **Transition**: "Anomaly detection is not just a modelling exercise. It is a production monitoring tool."

---

## Slide 44: Anomaly Detection in Production

**Time**: ~2 min
**Talking points**:

- Use cases: fraud detection, network intrusion, manufacturing quality control, data drift monitoring.
- Production considerations: contamination rate (how many anomalies to expect), false positive cost vs false negative cost, online vs batch detection, explainability ("WHY is this an anomaly?").
- Make the M3 connection precisely: "Anomaly detection flags individual unusual ROWS; drift monitoring compares whole DISTRIBUTIONS. kailash-ml's DriftMonitor (covered in M3) uses PSI and KS tests against a reference set — it is not an anomaly detector and does not use Isolation Forest or LOF. A rising anomaly rate on live inputs is a useful early warning that complements it."
- Singapore angle: "A Singapore bank screening card applications cannot tolerate 5% false positives — every false positive is a manual review and an angry applicant. Threshold tuning is a business decision, not just a statistical one."
- "If beginners look confused": "Detecting anomalies in the lab is easy. Deciding what to do about them in production is hard. That is the gap this slide is about."
  **Transition**: "Time for the exercise."

---

## Slide 45: Exercise 4.4: Credit-Application Anomaly Detection

**Time**: ~2 min
**Talking points**:

- Task: screen 20,000 real Singapore credit-application rows with 200 injected, LABELLED anomalies of three known types — global outliers, "synthetic identity" applicants, and a coordinated application ring. Because the anomaly types are known, every detector can be scored honestly (AUC-ROC / AUC-PR per type).
- Walk the four files: 4.1 Z-score and IQR (+ winsorisation); 4.2 Isolation Forest with a contamination sweep; 4.3 LOF with an n_neighbors sweep — see the masking trap live on the ring (masked at k=20, caught at k=50); 4.4 blending — equal, AUC-weighted, rank, `ensemble_detect` — then, once a labelled review sample exists, `EnsembleEngine.blend` / `.stack` as a supervised second stage over the detector scores.
- The analysis questions: which anomaly type does each detector catch? Which flagged points are true anomalies vs false positives? And the honesty check from the blending slide: does any blend actually beat the best single detector on this data?
- Remind students: "'statistically unusual' and 'will default' are different questions."
- "If beginners look confused": "The exercise gives you the detectors. Your job is to compare them and decide which applications you would actually investigate — and to say why."
  **Transition**: "Groups, axes, outliers. Now: co-occurrence patterns. What items appear together?"

---

## Slide 46: 4.5 Market Basket Analysis (lesson title)

**Time**: ~1 min
**Talking points**:

- Subtitle: "Co-occurrence pattern discovery — finding what appears together."
- Framing: "Clustering finds groups of similar POINTS. Association rules find groups of items that appear TOGETHER in transactions. A different kind of structure discovery."
- Preview: metrics, Apriori, FP-Growth, applications — six slides, ~15 minutes. Note: no Kailash engine for this lesson — we implement Apriori from scratch and call mlxtend for FP-Growth.
  **Transition**: "The three metrics that power the entire field."

---

## Slide 47: Association Rules: The Metrics

**Time**: ~3 min
**Talking points**:

- Support: how common is this itemset. supp(X) = count(X) / total transactions. If 20% of baskets contain bread, support of {bread} is 0.20.
- Confidence: given X, probability of Y. conf(X → Y) = supp(X ∪ Y) / supp(X). "If you bought bread, what fraction of those baskets also had butter?"
- Lift: is this more than chance? lift(X → Y) = conf(X → Y) / supp(Y). Lift > 1 = positive association. Lift = 1 = independent. Lift < 1 = negative.
- Classic example: "Customers who buy diapers also buy beer" — the diaper-beer story is folklore, but lift of 2–3 on that rule is plausible and actionable.
- Singapore angle: "A supermarket chain could discover that customers buying rice are 3x more likely to buy cooking oil in the same visit. That is lift > 1, and it drives shelf placement."
- "If beginners look confused": "Support asks: is this common? Confidence asks: if I know one thing, how likely is the other? Lift asks: is this more than coincidence?"
- "If experts look bored": "Lift is symmetric — lift(X → Y) equals lift(Y → X) — which is why confidence matters too. Use lift for interest and confidence for directionality. The lesson deck adds conviction, which IS directional."
  **Transition**: "Two algorithms to mine these rules at scale."

---

## Slide 48: Apriori vs FP-Growth

**Time**: ~2 min
**Talking points**:

- Apriori: generate candidate itemsets, prune by minimum support, iterate — one pass over the data per itemset size. Classic, pedagogically simple, slow on large datasets because of candidate generation.
- FP-Growth: compress the data into a frequent-pattern tree, then mine the tree recursively. TWO passes over the data (count items, build the tree), no candidate generation, much faster on large sparse data.
- In practice: FP-Growth for production, Apriori for teaching. Honest wrinkle from the exercise: on the course's small synthetic baskets (2,500 baskets, ~40 products), the from-scratch Apriori is actually FASTER than FP-Growth — the tree overhead only pays off at scale. Algorithms' constants matter; measure on YOUR data.
- "If beginners look confused": "Apriori is a brute-force search with a smart pruning rule. FP-Growth builds an index first. Use the index in production."
- "If experts look bored": "FP-Growth's divide-and-conquer mining has cost proportional to the number of frequent patterns, not the number of candidates — that is why it wins by orders of magnitude on sparse high-cardinality data, and why you cannot feel the difference on 2,500 baskets."
  **Transition**: "Association rules are not a dead-end topic. They connect forward."

---

## Slide 49: From Rules to Features

**Time**: ~2 min
**Talking points**:

- This is the design-level connection. Association rules discovered today feed forward in two ways.
- First: rules as features. "Bought bread AND butter" becomes a binary column in your M3 supervised model. Unsupervised discoveries become supervised inputs.
- Second — the important one for the module's arc — co-occurrence patterns are exactly what collaborative filtering discovers in 4.7. Association rules find EXPLICIT co-occurrence. Collaborative filtering finds LATENT co-occurrence via matrix factorisation. Neural embeddings generalise both.
- Feature Engineering Spectrum reminder: "Association rules are still hand-crafted in the sense that you pick the support threshold. Collaborative filtering lets optimisation decide. That is the pivot we are walking towards."
- "If beginners look confused": "Rules are yes/no patterns. 4.7 will show you what happens when you replace yes/no with a number that a model learns. Same idea, softer."
- "If experts look bored": "Word2Vec is literally a neural reparametrisation of co-occurrence matrix factorisation (Levy and Goldberg 2014). The thread from association rules to word embeddings to transformer attention is one continuous story."
  **Transition**: "Before we move on, note that this is not only a retail technique."

---

## Slide 50: Applications Beyond Retail

**Time**: ~1 min
**Talking points**:

- Walk the list quickly: web analytics (page click sequences), medicine (co-diagnoses and drug interactions), cybersecurity (attack pattern signatures), bioinformatics (gene co-expression), telecoms (service bundles), education (co-enrolment).
- Point: "Anywhere you have 'transactions' — sets of co-occurring items — association rules apply."
- "If beginners look confused": "The word 'basket' in 'market basket' is metaphorical. A basket of symptoms, a basket of clicks, a basket of genes — same algorithm."
  **Transition**: "Exercise."

---

## Slide 51: Exercise 4.5: Market Basket Analysis

**Time**: ~2 min
**Talking points**:

- Task: analyse 2,500 synthetic Singapore retail baskets with planted product bundles and impulse buys. Implement Apriori from scratch; run FP-Growth (mlxtend) and check both find the same itemsets. Compute support, confidence, lift. Interpret which rules are actionable.
- The feed-forward part: create binary features from the top rules and test whether they improve a classification model. In file 5.4 the prediction target is the shopper's NEXT trip — deliberately NOT a function of the baseline basket features, so the baseline reaches AUC ≈ 0.69 (not a trivial 1.0) and "do rule features add signal?" is a genuine empirical question. "No lift" is a valid finding — say so.
- Assessment: rules discovered with an appropriate support threshold, business interpretation, rule features tested against the baseline honestly.
- "If beginners look confused": "Start with a high support threshold so you get few rules. Lower it if you want more. The files guide you."
  **Transition**: "Items in a basket is one kind of unstructured data. Text is another. Same theme — discover structure."

---

## Slide 52: 4.6 NLP — Text to Topics (lesson title)

**Time**: ~1 min
**Talking points**:

- Subtitle: "Text feature discovery — extracting meaning from unstructured text."
- Framing: "In the last three lessons we found groups, axes, outliers, co-occurrences. Now we do the same for text. Text has no intrinsic columns — you must discover them. That is why NLP lives in Module 4. Instead of clusters, topics; the output is topic proportions — features."
- Preview: representation, TF-IDF, embeddings, LDA, NMF, BERTopic, coherence, sentiment — eight slides, ~25 minutes.
  **Transition**: "Before you can model text, you need to turn it into numbers."

---

## Slide 53: Text as Data: Representation

**Time**: ~2 min
**Talking points**:

- Text is not tabular. You must choose a representation.
- Bag of words: count how often each word appears. Fast, interpretable, loses word order.
- TF-IDF: weighted bag of words. Common words get low weight. Rare informative words get high weight.
- Word embeddings: dense vectors that capture meaning — coming in two slides.
- The progression on the slide: BoW → TF-IDF → Word2Vec → Transformers. Each step captures more meaning with less manual effort.
- Key line: "Once you see text as a matrix, topic modelling is just USML on that matrix."
- "If beginners look confused": "Imagine each document as a bag of Scrabble tiles. Count the tiles, and you have a representation. The algorithms we see today all start from that."
- "If experts look bored": "Bag of words discards word order and therefore throws away syntax. Transformers (M5) recover word order via positional encoding."
  **Transition**: "TF-IDF is the workhorse. Let us derive it."

---

## Slide 54: TF-IDF: Derivation

**Time**: ~3 min
**Talking points**:

- TF = term frequency: how often does term t appear in document d.
- IDF = inverse document frequency: log(N / df(t)), where N is total documents and df(t) is the number of documents containing t. Rare terms get high IDF; common terms (like "the") get IDF near zero.
- TF-IDF = TF × IDF. High when a term appears often in this document but rarely across the corpus. That is the signature of a term that distinguishes this document from others.
- BM25 extension on the slide: adds term-frequency saturation and document-length normalisation; still the standard lexical retrieval baseline in 2026.
- Singapore angle: "Run TF-IDF on local news articles. The word 'Singapore' has high TF but also very high DF, so its TF-IDF is near zero — it is not distinguishing. The word 'dengue' in a health article has high TF-IDF because it appears often there but rarely elsewhere."
- "If beginners look confused": "TF says the word is common HERE. IDF says the word is rare OVERALL. Multiply them: this word is common HERE but not everywhere. That is the definition of a keyword."
- "If experts look bored": "TF-IDF is a crude approximation to the pointwise mutual information between term and document. Note that sklearn's `TfidfVectorizer` uses a smoothed variant (ln((1+N)/(1+df)) + 1) with L2 normalisation — the exercise's from-scratch version implements the formula on the slide exactly, so small numeric differences are expected."
  **Transition**: "TF-IDF uses counts. Modern NLP uses embeddings."

---

## Slide 55: Word Embeddings (Tools, Not Derivation)

**Time**: ~3 min
**Talking points**:

- Word embeddings are dense vectors that capture meaning. Similar words have similar vectors. Classic demonstration: king − man + woman ≈ queen.
- Walk the table: Word2Vec CBOW (predict word from context — fast, good for syntax), Skip-gram (predict context from word — better for rare words), GloVe (global co-occurrence statistics), FastText (subword embeddings — handles out-of-vocabulary words).
- CRITICAL note: "I am telling you WHAT these do today. HOW they learn those vectors is neural network training — 4.8. The exercise (6.5) makes the bridge concrete: word vectors learned from unlabelled text via co-occurrence → PPMI → SVD — the factorisation Word2Vec implicitly performs."
- "If beginners look confused": "Embeddings are the word version of what PCA did to your tabular data — many dimensions compressed into a few meaningful ones. The difference is how they are learned."
- "If experts look bored": "Word2Vec Skip-gram with negative sampling is implicitly factorising a shifted PMI matrix (Levy and Goldberg 2014). Embeddings are matrix factorisation in disguise — which is why they sit in this module right next to 4.7."
  **Transition**: "TF-IDF and embeddings give you features. How do you discover topics?"

---

## Slide 56: LDA: Latent Dirichlet Allocation

**Time**: ~3 min
**Talking points**:

- LDA is a generative probabilistic model. Assumption: each document is a mixture of topics, and each topic is a distribution over words. Read the factorisation on the slide: P(word | doc) = Σ_k P(word | topic_k) · P(topic_k | doc).
- To "generate" a document: pick a topic mix, then for each word pick a topic from the mix and a word from that topic. Given real documents, we invert the story and learn the topic-word and document-topic distributions.
- Output: each topic is a list of top words; each document has a percentage breakdown across topics — those proportions are features.
- Two operational rules on the slide — teach both firmly: (1) "Fit LDA on raw word COUNTS, not TF-IDF." LDA's generative story draws whole word tokens, so it expects integer counts; TF-IDF weights break the likelihood. (2) "Choose K by HELD-OUT perplexity." Perplexity on the training documents always improves with more topics — measure it on held-out documents.
- Inference, precisely: collapsed Gibbs sampling integrates out the topic and word distributions and samples each word's topic assignment; variational inference instead fits factorised approximate posteriors (sklearn uses variational Bayes). Do not conflate the two.
- Connection to 4.2: "LDA is conceptually a mixture model — soft assignment of words to topics, fit by a variational form of EM."
- "If beginners look confused": "A topic is like a theme. LDA says every document is a mix of themes, and every theme has favourite words. Given a pile of articles, LDA finds the themes for you."
- "If experts look bored": "Online LDA (Hoffman, Blei, Bach 2010) made variational inference scalable to web-size corpora; that is the algorithm sklearn's `LatentDirichletAllocation` implements."
  **Transition**: "LDA is the classical answer. Two modern alternatives."

---

## Slide 57: NMF & BERTopic

**Time**: ~3 min
**Talking points**:

- NMF (Non-negative Matrix Factorisation): factorises the TF-IDF matrix V ≈ W·H with non-negativity constraints. W is documents × topics, H is topics × words. The non-negativity forces additive, parts-based, interpretable topics. This is matrix factorisation applied to text — the same family as PCA (4.3) and collaborative filtering (4.7).
- BERTopic: the modern pipeline — transformer sentence embeddings → UMAP → HDBSCAN → class-based TF-IDF (c-TF-IDF). Point out that it reuses two techniques from earlier today: UMAP (4.3) and HDBSCAN (4.1). "Everything connects."
- Three precision points: (1) BERTopic's UMAP step reduces to about 5 dimensions by default — 2-D is only for plotting. (2) HDBSCAN assigns each document to ONE topic or to noise — unlike LDA's mixed membership. (3) The sentence-embedding model is a configuration choice — the exercise reads it from the `TOPIC_EMBED_MODEL` environment variable rather than hardcoding a model name.
- "If beginners look confused": "LDA is the classical method. NMF is the linear algebra method. BERTopic is the modern method that stacks embeddings, UMAP, and HDBSCAN. When in doubt, try BERTopic first."
- "If experts look bored": "c-TF-IDF treats each cluster as a single pseudo-document and computes TF-IDF across clusters, which gives interpretable topic labels for free. That is BERTopic's key engineering insight."
  **Transition**: "How do you know a topic model is any good?"

---

## Slide 58: Topic Coherence: Evaluating Quality

**Time**: ~2 min
**Talking points**:

- Coherence metrics measure how semantically related the top words in a topic are.
- NPMI (Normalised Pointwise Mutual Information): do topic words co-occur more than chance? Range [−1, 1]; chance is 0. Read the formula on the slide.
- UMass: do topic words co-occur within the TRAINING corpus? (NPMI is usually computed against an external/reference windowing; UMass uses the training documents themselves.)
- "NPMI is the silhouette score of topic modelling — it tells you whether the top words in a topic actually hang together." 'bank, money, loan' coheres; 'bank, tree, phone' does not.
- Complement with human evaluation: coherence correlates imperfectly with human interpretability. Human judgement on a sample is still the final check.
- "If beginners look confused": "Coherence asks: do the words in a topic actually belong together? A good topic reads like a theme, not a random word salad."
- "If experts look bored": "See Lau, Newman, Baldwin 2014 for empirical comparisons of coherence metrics against human ratings. The exercise computes NPMI with its own implementation over the corpus."
  **Transition**: "One application-level slide before the exercise."

---

## Slide 59: Sentiment Analysis: Text Classification

**Time**: ~2 min
**Talking points**:

- Sentiment analysis: classify text as positive, negative, or neutral. Technically supervised — shown here because it is the most common downstream consumer of the text features this lesson discovers.
- Walk the USML connection on the slide: word vectors learned from UNLABELLED text (co-occurrence → PPMI → SVD) are averaged into document features for a supervised sentiment classifier. Unsupervised discovery feeds supervised prediction — the spectrum again.
- Describe the exercise design honestly: in 6.5 the unsupervised part learns word vectors from the training sentences' text alone, ignoring labels; the supervised part trains on HUMAN labels (SST-2 movie-review sentences) and is tested on unseen sentences against a word-list lexicon and a TF-IDF baseline. Labels must come from people, not from a lexicon applied to the same text — otherwise the classifier just relearns the lexicon and the "accuracy" is circular.
- "If beginners look confused": "Sentiment is just text classification with labels 'positive' and 'negative'. You already know classification from M3 — the new part is where the features come from."
  **Transition**: "Exercise."

---

## Slide 60: Exercise 4.6: Topic Modelling & Sentiment

**Time**: ~2 min
**Talking points**:

- Task: extract topics from a deduplicated AG News corpus (world, sports, business, sci/tech — real news text). 6.1 TF-IDF and BM25 retrieval; 6.2 NMF on TF-IDF; 6.3 LDA on COUNTS with held-out perplexity; 6.4 BERTopic (embedding model from `TOPIC_EMBED_MODEL`). Compare topic quality with NPMI coherence and against the human section labels; assign human-readable labels to the discovered topics.
- 6.5 sentiment: PPMI + SVD word vectors trained on human-labelled SST-2 sentences, evaluated on held-out sentences.
- Assessment: multiple topic methods compared, coherence computed, topics interpreted with human-readable labels.
- "If beginners look confused": "The files do most of the preprocessing for you. Your job is to run the topic models, look at the top words, and label the topics."
  **Transition**: "Take a five-minute break. When we come back, we reach THE PIVOT of the entire module."

**[BREAK — 5 min]**

---

## Slide 61: 4.7 Recommender Systems (lesson title)

**Time**: ~1 min
**Talking points**:

- Subtitle: "THE PIVOT — optimisation drives feature discovery."
- Frame the stakes: "What you see in this lesson is the single most important concept in the entire curriculum. In 4.1–4.6, algorithms discovered features. From this lesson forward, features are discovered by OPTIMISATION — and that is exactly what neural networks do."
- Preview: content-based vs collaborative filtering, user-based vs item-based, matrix factorisation, ALS, implicit feedback, THE PIVOT slide, embedding visualisation — nine slides, ~25 minutes.
  **Transition**: "Start with the two main families of recommenders."

---

## Slide 62: Content-Based vs Collaborative Filtering

**Time**: ~3 min
**Talking points**:

- Content-based: recommend items similar to what the user already liked, based on item features (genre, price, description). You liked this action movie → here is another action movie.
- Collaborative filtering: recommend items that similar users liked. People who liked this also liked that. No item features required — it discovers latent preferences from interaction data alone.
- Get the cold-start directions exactly right — the slide states them: content-based has NO cold-start problem for new ITEMS (as long as they have features), but it is still cold for brand-new USERS — a new user has no liked items from which to build a profile. Collaborative filtering is cold for both.
- In practice everyone uses a hybrid (slide 66).
- "If beginners look confused": "Content-based says: you liked a horror movie, so here is another horror movie. Collaborative says: people who liked your movies also liked this one, so try it. Both are valid."
- "If experts look bored": "Content-based models often reduce to nearest-neighbour search over item embeddings. Collaborative filtering is where matrix factorisation lives, and it is the interesting bit for today's pivot."
  **Transition**: "Within collaborative filtering there are two subfamilies."

---

## Slide 63: User-Based vs Item-Based CF

**Time**: ~2 min
**Talking points**:

- User-based: find similar users, recommend their items. Item-based: find similar items, recommend to users who liked similar items. Item-based is more stable because item similarity changes slowly; it scales better when users outnumber items' co-rating structure.
- Cold start hits BOTH directions: a new user has no ratings to find neighbours (user-CF) or to score item similarities against (item-CF); a new item has no co-raters, so neither similarity can be computed. Content features fix the new-item case; new users need onboarding questions or popularity fallbacks.
- Amazon popularised item-based CF in the early 2000s because it scaled better than user-based at their catalogue size.
- "If beginners look confused": "User-based asks: who is like me? Item-based asks: what is like what I already bought? Both work, but item-based is more reliable because items do not change mood."
- "If experts look bored": "Item-item CF is memory-based with O(n_items²) similarity computation. Matrix factorisation is the model-based alternative that scales better — coming up next."
  **Transition**: "Memory-based CF has limits. Now the breakthrough."

---

## Slide 64: Matrix Factorisation: The Core Idea

**Time**: ~3 min
**Talking points**:

- Setup: a user-item rating matrix R. Rows are users, columns are items, cells are ratings. Most cells are empty — that is the recommendation problem.
- Read the objective on the slide: minimise over OBSERVED entries only, Σ (r_ui − u_uᵀ v_i)² + λ(‖u_u‖² + ‖v_i‖²). Factorise R ≈ U Vᵀ. U = user embeddings, V = item embeddings, k latent dimensions (say 50 or 100). A rating is the dot product of the two vectors.
- Draw the factorisation on the whiteboard if possible. "This is PCA with a twist — PCA factorised a FULL matrix to decorrelate features. Matrix factorisation factorises a SPARSE matrix to FILL IN the missing ratings — and the sum runs over observed entries only."
- The latent vectors are LEARNED by minimising prediction error — that is the difference from everything before this lesson.
- "If beginners look confused": "Imagine that every user has a personality vector with 50 numbers (likes drama, likes action, likes old films). Every movie has a profile vector with the same 50 numbers. A rating is how well the user's personality matches the movie's profile. Matrix factorisation learns both sets of vectors at once from the ratings alone."
- "If experts look bored": "This is the Netflix Prize architecture (Koren, Bell, Volinsky 2009). The breakthrough was realising that a low-rank factorisation of the ratings matrix outperforms every neighbour-based method by a wide margin — and the factors are interpretable embeddings."
  **Transition**: "How do you actually learn U and V?"

---

## Slide 65: ALS: Alternating Least Squares

**Time**: ~3 min
**Talking points**:

- The joint objective is NOT convex in (U, V). But fix U and it is convex in V; fix V and it is convex in U. That is the ALS trick: alternate two easy, closed-form least-squares problems until convergence.
- Point at the subscripts in the slide's update equations: each user's vector is a small ridge regression on JUST the items that user rated — u_u = (V_Ωuᵀ V_Ωu + λI)⁻¹ V_Ωuᵀ r_u,Ωu. Writing it densely as (VᵀV + λI)⁻¹VᵀRᵀ would treat every missing rating as a zero rating — a different and wrong objective for explicit ratings.
- Connect and correct the EM analogy from 4.2: "ALS has the same RHYTHM as EM — alternate two easy steps — but it is block-coordinate least squares, not EM." Exercise 7.4 also adds global, user and item bias terms.
- Why it matters for the arc: this is the first time in the curriculum that OPTIMISATION discovers features. Not a closed-form eigendecomposition. Not a fixed clustering algorithm. An actual loss function being minimised.
- "If beginners look confused": "ALS is: you have two unknowns. Fix one, solve for the other. Then fix the other, solve for the first. Repeat. Each step is simple, and together they solve a hard problem."
- "If experts look bored": "ALS is block coordinate descent on a bi-convex objective; each block update is ridge regression. The SGD alternative with implicit-feedback weighting is what large-scale production recommenders use."
  **Transition**: "Real recommenders rarely have explicit ratings. They have clicks and views."

---

## Slide 66: Implicit Feedback & Hybrid Systems

**Time**: ~2 min
**Talking points**:

- Explicit feedback: users rate items 1–5. Honest but scarce. Implicit feedback: clicks, views, dwell time, purchases. Abundant but noisy — a click is not necessarily a like.
- Implicit ALS: weight observed interactions by confidence, treat missing values as weak negatives. SVD++ extends SVD with implicit signals.
- Hybrid systems: combine content-based (solves item cold-start, uses features) with collaborative (captures latent preferences for the tail). Almost every production system is a hybrid.
- The discipline line on the slide: "Tune blend weights on a VALIDATION split, never on the test set." The exercise enforces this — hybrid weights are fit on validation and reported on a held-out test.
- "If beginners look confused": "If you scroll past a video, that is almost a 'no'. If you watch the whole thing, that is almost a 'yes'. Implicit feedback is counting those signals and treating them as approximate ratings."
- "If experts look bored": "See Hu, Koren, Volinsky 2008 for the implicit ALS formulation — the confidence-weighted objective c_ui·(p_ui − u_uᵀv_i)² with c_ui increasing in observed interaction strength."
  **Transition**: "Now — the slide I have been pointing at all day."

---

## Slide 67: THE PIVOT: Optimisation Drives Feature Discovery

**Time**: ~4 min
**Talking points**:

- SLOW DOWN. This is the most important slide of the module after the Feature Engineering Spectrum.
- Read the headline aloud verbatim: "Matrix factorisation learns user and item EMBEDDINGS by minimising reconstruction error. This is the first time you have seen OPTIMISATION DRIVE FEATURE DISCOVERY."
- Walk the diagram: 4.7 matrix factorisation (R = U Vᵀ, embeddings via minimisation) on the left; 4.8 neural networks (a = f(Wx + b)) on the right. The arrow says: same idea, generalised.
- Unpack the generalisation: "In 4.7 the embeddings are LINEAR functions of the observed ratings — a sum of products. In 4.8, hidden layer activations a = f(Wx + b) are NON-LINEAR functions of the input. They are still embeddings. They are still learned by minimising a loss. The difference is the activation function."
- Repeat the key line for emphasis: "Hidden layer activations ARE embeddings, learned by minimising a loss function."
- "If beginners look confused": "You have seen this idea once already today and did not know it was the same idea. When you ran K-means, you minimised within-cluster sum of squares. That was optimisation. Matrix factorisation is the same spirit, except the thing you optimise is an entire embedding space for every user and every item. And that is exactly what a neural network hidden layer is."
- "If experts look bored": "This is the theoretical bridge from classical ML to deep learning. The path from Netflix Prize to modern transformers runs through this slide."
  **Transition**: "Let me show you what these embeddings look like."

---

## Slide 68: Visualising Learned Embeddings

**Time**: ~2 min
**Talking points**:

- Take the learned item embeddings and project to 2D — with PCA (as Exercise 7.4 does) or UMAP (from 4.3, reusing the technique).
- Result: similar items cluster together in embedding space; similar users cluster together. On real catalogues this is where comedy films cluster together and action films together — nobody told the algorithm about genres; it discovered them from rating patterns alone. The latent dimensions often correspond to interpretable concepts (genre, price range, quality).
- The exercise data is synthetic with known true taste factors, so students can check how well ALS recovered them — a rare chance to validate embeddings against ground truth.
- Preview 4.8: "Now imagine the same visualisation for a neural network's hidden layer. You would see the same thing — because a hidden layer is doing the same job, non-linearly."
- "If beginners look confused": "The embeddings are coordinates the algorithm invented. When you look at the map, you can SEE what it learned — because similar things end up in the same neighbourhood."
- "If experts look bored": "This is the same visualisation technique the Word2Vec papers used to show king−man+woman≈queen analogies. The embedding geometry IS the feature."
  **Transition**: "Your turn."

---

## Slide 69: Exercise 4.7: Recommender Systems

**Time**: ~2 min
**Talking points**:

- Task: recommend on a synthetic e-commerce rating matrix — 300 users × 120 items, with 10 brand-new items held out to expose cold start. Build content-based, user-based CF, item-based CF, and biased matrix factorisation (ALS). Hybrid with weights tuned on VALIDATION. Compare RMSE, precision@k, MAP against no-skill baselines. Visualise learned embeddings (2D PCA projection). Articulate THE PIVOT in your own words.
- Insist on the baselines: "A recommender that cannot beat the global mean and item mean has learned nothing. Check that FIRST." On the course data the honest result is that biased ALS beats the baselines on RMSE (≈ 0.56) — students should verify, not trust.
- Cold-start findings to elicit: only content-based filtering can score the brand-new items; NONE of the methods can serve a brand-new user.
- The writing requirement is deliberate: "If you cannot explain the pivot in your own words, you have not yet understood what 4.7 is about. The paragraph is the check."
- "If beginners look confused": "The scaffolding gives you ALS. Your job is to run it, look at the embeddings, and explain what you see."
  **Transition**: "The pivot generalises. Time for neural networks."

---

## Slide 70: 4.8 DL Foundations (lesson title)

**Time**: ~1 min
**Talking points**:

- Subtitle: "Neural Networks, Backpropagation and the Training Toolkit."
- Read the bridge line: "In 4.7, matrix factorisation learned embeddings by minimising reconstruction error. A neural network does the same thing — hidden layer activations ARE embeddings, learned by minimising a loss function. The difference is non-linearity."
- Set expectations for density: "This is the biggest lesson in the module. Sixteen slides. Everything you have seen today is a subset of what happens inside a neural network."
  **Transition**: "Visualise the bridge one more time."

---

## Slide 71: The Bridge: From Matrix Factorisation to Neural Networks

**Time**: ~2 min
**Talking points**:

- Show the spectrum one more time. Manual features on the left, USML in the middle, deep learning on the right — highlighted, because we have arrived.
- Key line: "Hidden layers = automated feature engineering WITH gradient-based error feedback."
- Unpack the two halves: "Automated feature engineering — that is what all of 4.1 through 4.7 was doing. Gradient-based error feedback — that is the part we are about to add. Put them together and you get deep learning."
- "If beginners look confused": "You already know the first half (features are discovered). Now you add the second half (the error tells the features how to update)."
- "If experts look bored": "The unifying view: hidden layers compute learned nonlinear features, and the training loop uses gradient descent to tune those features for the downstream task. That is the representation-learning thesis."
  **Transition**: "Start with the architecture."

---

## Slide 72: Neural Network Architecture

**Time**: ~2 min
**Talking points**:

- Input layer: one node per feature. Hidden layers between input and output. Output layer: one node for regression, C nodes for C-class classification.
- Fully connected: each node in a layer connects to every node in the next. Each connection has a weight; each node has a bias. Node value = weighted sum of inputs + bias, then activation.
- Draw a 3-4-1 network on the whiteboard: three inputs, four hidden nodes, one output. "Every line is a weight. Every circle is a computation." Count the parameters: 12 weights + 4 biases + 4 weights + 1 bias = 21.
- "If beginners look confused": "Each layer is a grid of light bulbs. The previous layer sends voltages. Each bulb mixes the voltages with its own knobs (weights) and lights up. The pattern of lights is what the next layer sees."
- "If experts look bored": "Modern architectures replace full connectivity with structured sparsity (convolutions for vision, attention for sequences). But the fully connected network is the theoretical starting point — universal approximation lives here."
  **Transition**: "How does information flow through the network?"

---

## Slide 73: Forward Pass

**Time**: ~3 min
**Talking points**:

- Forward pass: z = Wx + b, then a = f(z). Pass a to the next layer. Read the one-hidden-layer chain on the slide: x → z₁ = W₁x + b₁ → a₁ = f(z₁) → z₂ = W₂a₁ + b₂ → ŷ = z₂.
- Three things to track: inputs x, weighted sums z, activations a. You will need all three for backprop.
- "The forward pass is just matrix multiplication, addition, and a nonlinear function. That is all a neural network does — over and over, layer by layer."
- Walk a tiny concrete example with real numbers if the room needs it: two inputs, three hidden nodes, one output. Plug in numbers; show z and a at each layer.
- "If beginners look confused": "Forward pass is: plug in your inputs, multiply by weights, add up, squish through an activation, pass along. Repeat for each layer. At the end you get a prediction."
- "If experts look bored": "Forward pass is a matrix multiplication followed by an element-wise activation. On modern hardware, the whole forward pass for a million-row batch is a handful of GPU kernels."
  **Transition**: "A neural network with zero hidden layers is something you already know."

---

## Slide 74: Linear Regression as a Neural Network

**Time**: ~2 min
**Talking points**:

- Zero hidden layers: ŷ = wᵀx + b. That IS linear regression from M3. Same MSE loss, same gradient descent. "Everything you learned in M3 is a special case."
- Add one hidden layer with a non-linear activation: ŷ = W₂·f(W₁x + b₁) + b₂. Now the model learns non-linear features — it "writes its own parametric function."
- State the universal approximation theorem EXACTLY as on the slide: "ONE hidden layer with a non-polynomial activation and enough units can approximate any continuous function on a bounded (compact) domain to any accuracy. Depth buys parameter efficiency, not extra expressive power."
- Add the honest caveats (Cybenko 1989, Hornik 1991): one hidden layer is enough IN PRINCIPLE, but it may need an impractically large number of units; the theorem says nothing about whether gradient descent will find those weights, or how well they generalise. That is why we stack layers. Exercise 8.1 shows the smallest case: XOR — no linear model can fit it; one hidden layer solves it.
- Singapore angle: "Predict HDB resale price from area, storey, age. Linear regression assumes a linear combination. One hidden layer can discover 'size premium for high floors' and 'lease-decay penalty' automatically as non-linear features."
- "If beginners look confused": "A neural network with no hidden layer is linear regression. Add one layer, and it can bend. That is the whole difference."
  **Transition**: "How do you train it?"

---

## Slide 75: Loss Functions & Gradient Descent

**Time**: ~3 min
**Talking points**:

- Loss function: measures how wrong the prediction is. Read the slide's formula precisely: SSE = ½ Σ (y_i − ŷ_i)² — the half-SSE (the ½ makes the derivative clean: ∂J/∂w = −x·(y − ŷ)). MSE divides by n instead — same minimiser, different scale; the taxonomy slide later shows MSE. Do not call the ½·Σ form "MSE".
- Gradient descent: compute the gradient of the loss with respect to each weight; update w_new = w_old − η·gradient. Repeat until the loss converges. "The gradient points uphill. Subtracting it moves downhill."
- Walk a one-weight example with HDB prices: start with a random weight, compute error, compute gradient, update. Show the error decreasing. "This is the engine of all deep learning."
- "If beginners look confused": "Gradient descent is: take a small step downhill, look at where you are, take another small step downhill. Repeat until you cannot go any lower. The hill is the loss function."
- "If experts look bored": "Stochastic gradient descent estimates the full gradient with a mini-batch — biased but fast. The modern recipe is SGD + momentum or AdamW with warmup — coming in a few slides."
  **Transition**: "Gradient descent needs gradients. How do you compute them through many layers?"

---

## Slide 76: Backpropagation: Chain Rule Through Layers

**Time**: ~3 min
**Talking points**:

- Backpropagation is the chain rule from calculus, applied layer by layer, from the output back to the input.
- Write the chain on the whiteboard as on the slide: ∂L/∂w₁ = ∂L/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂w₁. One factor per layer boundary you pass through.
- Why it works: each layer's gradient depends only on the previous layer's activation and the next layer's incoming gradient. It is a local computation — that is why it scales to billions of parameters.
- "If beginners look confused": "Backprop is a rumour game in reverse. The output layer knows the error. It whispers to the layer before it: 'you contributed this much.' That layer whispers to the layer before IT, and so on back to the input."
- "If experts look bored": "Backprop is reverse-mode automatic differentiation with a specific topological order. Modern frameworks implement autodiff via computational graphs, but the underlying algorithm is the one on this slide — the exercise's from-scratch network implements it by hand."
  **Transition**: "So what are hidden layers DOING?"

---

## Slide 77: Hidden Layers = Automated Feature Engineering

**Time**: ~4 min
**Talking points**:

- THIS is the lesson's version of the pivot slide. Say it clearly: "Hidden layers are automated feature engineering with error feedback."
- Unpack the claim: hidden node values are LEARNED functions of the input — features the network invented. Those features are updated by backprop to MINIMISE the loss. The optimisation drives feature discovery. And hidden activations are embeddings — exactly like the matrix-factorisation embeddings from 4.7, but non-linear.
- The "unsupervised meets supervised" line, verbatim from the slide: "Hidden layers perform unsupervised feature discovery (like PCA, clustering). The output layer performs supervised prediction (like M3). Backpropagation couples them with error feedback."
- Come back to the Feature Engineering Spectrum one last time. "Left side: human picks features. Middle: algorithm discovers features. Right side: the network learns features shaped by the task. Everything you have learned today lives on this line, and this slide is where it ends."
- "If beginners look confused": "This is the single most important idea in modern AI. A neural network is not magic. It is feature engineering that happens inside the model itself, guided by the error signal. That is why it works so well — the features are exactly the ones the task needs."
- "If experts look bored": "Representation learning (Bengio, Courville, Vincent 2013) formalised this. Deep networks learn hierarchies — low layers learn edges, mid layers learn parts, top layers learn concepts. The feature engineering you would have done by hand is now emergent from gradient descent."
  **Transition**: "The rest of the lesson is the training toolkit — the practical machinery that makes all of this actually work."

---

## Slide 78: Activation Functions

**Time**: ~2 min
**Talking points**:

- Activation functions introduce non-linearity. Without them, stacked linear layers collapse into a single linear layer.
- Walk the table: ReLU (max(0, z)) — the hidden-layer default; can "die" if z stays negative. The ReLU variants — Leaky ReLU, PReLU (learned slope), ELU — all exist to keep a non-zero gradient for negative inputs, which is the fix for the dead units Exercise 8.2 measures.
- GELU (z·Φ(z)) — the transformer default; Swish/SiLU (z·σ(z)) — the smooth modern default in CNNs and LLMs. Exercise 8.2 compares ReLU, GELU, Tanh and SiLU on the same task.
- Sigmoid — binary output only; saturates and causes vanishing gradients in hidden layers. Tanh — zero-centred, older architectures and RNNs; still saturates. Softmax — multi-class output; turns logits into a probability distribution.
- Quick rule: "ReLU or GELU in hidden layers, sigmoid for binary output, softmax for multi-class. That covers 90% of use cases."
- "If beginners look confused": "Activation functions are the 'squish' at each node. Without them a neural network is just fancy linear regression. The squish is what lets it bend."
- "If experts look bored": "GELU is ReLU weighted by the CDF of a Gaussian (Hendrycks and Gimpel 2016); Swish/SiLU are essentially its close relatives — smooth, non-monotonic near zero."
  **Transition**: "Two regularisation techniques you will reach for every time."

---

## Slide 79: Dropout & Batch Normalisation

**Time**: ~2 min
**Talking points**:

- Dropout: randomly zero out a fraction of neurons during training (rate typically 0.1–0.5). Forces the network not to rely on any single neuron; prevents co-adaptation. "Dropout is like training an ensemble of sub-networks." CRITICAL mechanics: modern frameworks use inverted dropout — scale by 1/(1−p) during TRAINING so inference is the identity. Turned off at eval time.
- Batch normalisation: normalise each layer's inputs to zero mean and unit variance per mini-batch, then learn a scale γ and shift β. Stabilises training, enables higher learning rates, reduces sensitivity to initialisation. "Batch norm shifts and scales activations to keep gradients healthy."
- Layer norm for transformers — M5.
- "If beginners look confused": "Dropout is: randomly turn off some lights during training so the rest of the network learns to work without them. Batch norm is: squeeze the numbers at each layer into a standard range so training is less chaotic."
- "If experts look bored": "BatchNorm has known issues with small batch sizes and distributed training. LayerNorm (Ba, Kiros, Hinton 2016) fixes that and is the transformer default."
  **Transition**: "Before you train you have to initialise. Wrong initialisation kills training."

---

## Slide 80: Weight Initialisation

**Time**: ~2 min
**Talking points**:

- Why not zero init: all neurons compute the same gradient → symmetry is never broken → all neurons learn the same thing. Random init breaks symmetry.
- Xavier / Glorot: scale √(2/(n_in + n_out)) — designed for sigmoid/tanh. Kaiming / He: √(2/n_in) — designed for the ReLU family; the modern default.
- "Initialisation matters more than you think. Too large: exploding gradients. Too small: vanishing gradients. Xavier and Kaiming solve this." Exercise 8.2 shows zero vs Xavier vs Kaiming side by side — and pairs Tanh with Xavier, ReLU with Kaiming, deliberately.
- "If beginners look confused": "If you start all the neurons identical, they all learn the same thing. Random init makes them different from the start so they can specialise."
- "If experts look bored": "Kaiming init preserves activation variance through ReLU layers, preventing exploding or vanishing signals in deep networks (He et al. 2015)."
  **Transition**: "Now the optimiser — the thing that actually does the updating."

---

## Slide 81: Optimisers: From SGD to Adam

**Time**: ~3 min
**Talking points**:

- Walk the table. SGD: raw gradient — simple, good generalisation, slow and LR-sensitive. SGD + momentum: exponential moving average of gradients — faster through consistent slopes. RMSProp: adaptive per-parameter LR from a second-moment estimate. Adam: momentum + RMSProp together (read the m_t / v_t updates on the slide). AdamW: Adam + decoupled weight decay — the best default for deep learning.
- Quick rule: "Adam is the default. It adapts the learning rate per parameter. AdamW adds proper weight decay. Use AdamW unless you have a reason not to." Exercise 8.4 races SGD, momentum, Adam and AdamW on identical data and budget.
- "If beginners look confused": "The optimiser is the thing that actually walks downhill. SGD walks carefully. Adam walks confidently. In practice you use Adam."
- "If experts look bored": "Adam's first steps use biased moment estimates — the bias correction terms and warmup schedules exist for exactly that reason (Kingma and Ba 2014; Loshchilov and Hutter 2017 for AdamW)."
  **Transition**: "The loss function tells the optimiser what to minimise. Pick the right one."

---

## Slide 82: Loss Functions Taxonomy

**Time**: ~2 min
**Talking points**:

- Walk the table as a reference. MSE for regression. MAE for robust regression (outliers should not dominate). Cross-entropy for classification — ALWAYS cross-entropy over MSE with sigmoid/softmax outputs. Binary CE for binary. Focal loss for imbalanced classes — the (1−ŷ)^γ factor down-weights easy examples (Lin et al. 2017). Contrastive loss for similarity learning — pull similar, push dissimilar. Triplet loss for metric learning — anchor a, positive p, negative n: d(a,p) must beat d(a,n) by margin m; the basis of face matching and embedding search. KL divergence for distribution matching. Reconstruction loss for autoencoders (M5).
- Reconcile the two regression formulas the room has seen: the gradient-descent slide used SSE = ½Σ(y − ŷ)²; this table's MSE divides by n. Same minimiser, different scale.
- "The loss function encodes what you care about. Get it right and the network learns the right thing. Get it wrong and the network optimises the wrong objective."
- "If beginners look confused": "The loss function is the scoreboard. Pick the wrong scoreboard and the network cheats to win the wrong game. Regression → MSE. Classification → cross-entropy. That is 90% of what you need."
  **Transition**: "One more piece — how the learning rate changes over time."

---

## Slide 83: Learning Rate Schedules & Early Stopping

**Time**: ~2 min
**Talking points**:

- Fixed learning rate is rarely optimal. Schedules make big moves early and small moves late. Walk the list: step decay (halve every N epochs), cosine annealing (smooth decay), warmup + cosine (ramp up, then decay — Exercise 8.4 uses this), one-cycle (up then down), ReduceLROnPlateau (drop when validation stalls).
- Early stopping: stop when validation loss stops improving for p consecutive epochs (patience); keep the best checkpoint. Prevents overfitting and wasted compute.
- Gradient clipping: cap gradient magnitude to prevent explosion — required for RNNs, sometimes transformers; Exercise 8.5 demonstrates it.
- "These are the practical tools that make training work. Without them, networks either diverge or overfit."
- "If beginners look confused": "Learning rate schedules are the training equivalent of slowing down as you approach the destination. Early stopping is knowing when to park."
- "If experts look bored": "One-cycle training (Smith 2018) treats the LR as a super-convergence knob. The toolkit is full of small tricks like this — Exercise 8 turns each on and off to see the effect."
  **Transition**: "Two notes on the output layer."

---

## Slide 84: Regression vs Classification Output

**Time**: ~1 min
**Talking points**:

- Regression: one output node, linear activation, MSE/MAE loss. Binary classification: one node, sigmoid, binary cross-entropy. Multi-class: C nodes, softmax, cross-entropy.
- "Everything else is the same. Same hidden layers, same backprop, same optimiser. Only the output layer and loss change."
- "If beginners look confused": "Continuous number → linear output + MSE. Yes/no → sigmoid + BCE. Pick-one-of-many → softmax + CE. Match these three and you are done."
- "If experts look bored": "Softmax + cross-entropy is the log-likelihood of a categorical distribution; sigmoid + BCE is a Bernoulli. Both are maximum-likelihood training under different output distributions."
  **Transition**: "Kailash wraps the trained network for portable inference."

---

## Slide 85: Kailash Bridge: OnnxBridge for Neural Networks

**Time**: ~2 min
**Talking points**:

- Fourth bridge slide. Set the scope precisely: "OnnxBridge takes a TRAINED network to a portable inference graph. It does NOT train, checkpoint or serve your model — training is your torch loop (Exercise 8.4–8.5), serving is ONNX Runtime, and InferenceServer arrives in M5."
- Walk the code: train `nn.Sequential(nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 1))` in torch (AdamW, LR schedule, early stopping — the exercise's toolkit). Then `bridge.check_compatibility(model, framework="torch")` before exporting; `bridge.export(model, framework="torch", output_path=..., sample_input=torch.randn(1, 7))` — torch export TRACES the forward pass, which is why it needs a sample input. Export reports a result object (`export.success`, `export.onnx_status`) instead of raising, so a pipeline can decide what to do on failure.
- The parity check is the habit to instill: "Always compare ONNX and torch outputs on the same batch." On the slide, onnxruntime runs `model.onnx` and the max absolute difference vs the torch model prints around 1e-7. `OnnxBridge.validate()` does this comparison for sklearn-style models (it calls `.predict`); for torch the check uses onnxruntime directly — exactly as Exercise 8.5 does.
- Note for the room: the installed exporter prints some warnings during torch export; they are harmless here.
- "If beginners look confused": "ONNX is a saved-file format for trained networks. OnnxBridge checks your model can be exported, exports it, and you verify the exported graph computes the same numbers as your torch model."
- "If experts look bored": "ONNX standardises the computation graph and operator set, so a PyTorch-trained model runs on runtimes optimised for different hardware. The tracing requirement is why dynamic control flow needs care at export time."
  **Transition**: "Now the centrepiece exercise."

---

## Slide 86: Exercise 4.8: The Deep-Learning Training Toolkit

**Time**: ~2 min
**Talking points**:

- The most important exercise in the module: five PyTorch files (`solutions/ex_8/01_xor_proof.py` … `05_regularisation_training.py`), each isolating one part of the toolkit.
- Walk the five: 8.1 XOR — a linear model is stuck near 50%; one hidden layer solves it (universal approximation, smallest case). 8.2 ReLU / GELU / Tanh / SiLU, and zero vs Xavier vs Kaiming init — count the dead units. 8.3 a small CNN with batch norm and a residual block on synthetic shape images. 8.4 SGD, momentum, Adam, AdamW; warmup + cosine LR schedule. 8.5 dropout vs batch norm, gradient clipping, early stopping; export with OnnxBridge.
- Two design facts to mention: every comparison uses the same data and budget, so curves are comparable; and the images in 8.3–8.5 are synthetic SHAPES the network can actually learn — so the differences between variants mean something (they are not noise-on-noise).
- The from-scratch forward/backward pass is the one on the backpropagation and gradient-descent slides — work it through on the board with a small HDB price example so the torch autograd in the exercise has a manual counterpart. (On HDB features, a from-scratch one-hidden-layer net lands close to plain linear regression — R² ≈ 0.86 both — which is itself a great discussion: when does non-linearity buy you anything?)
- Assessment: training curves with and without each technique, and the "unsupervised meets supervised" idea articulated — hidden layers are USML with error feedback.
- "If beginners look confused": "Each file changes ONE thing. Run it, look at the curve, say what changed. The scaffolding handles the boilerplate."
  **Transition**: "Let us step back and see the full arc."

---

## Slide 87: Module 4: The Complete Arc

**Time**: ~2 min
**Talking points**:

- Return to the Feature Engineering Spectrum diagram one final time. Manual → USML → DL.
- Walk the table row by row. Each lesson produces features of a specific kind: cluster labels (4.1), soft probabilities (4.2), components (4.3), anomaly scores (4.4), rule features (4.5), topic proportions (4.6), embeddings (4.7 — the pivot), hidden activations (4.8).
- Walk the engine column honestly: 4.1 → ClusteringEngine + AutoMLEngine; 4.2 → ClusteringEngine (gmm); 4.3 → DimReductionEngine; 4.4 → AnomalyDetectionEngine + EnsembleEngine (supervised second stage); 4.5 → no engine (from scratch); 4.6 → DimReductionEngine (nmf); 4.7 → no engine (from scratch); 4.8 → OnnxBridge + SklearnTrainable. ModelVisualizer draws the plots throughout. SklearnTrainable wraps a scikit-learn estimator (e.g. an MLP) so it trains through kailash-ml's train / register path — the assessment's neural-network task uses it.
- Deliver the arc in one sentence: "We started by finding groups of similar points. We ended with networks whose hidden layers discover features for any task. Same spectrum, increasing automation."
- "If beginners look confused": "If you can point to each row in this table and describe what it produces, you have understood Module 4. That is the check."
- "If experts look bored": "Self-supervised learning, contrastive methods, and foundation-model pretraining are all points further right on this spectrum. The thread from K-means to frontier models is literally the line drawn on this slide."
  **Transition**: "How you will be assessed."

---

## Slide 88: Module 4 Assessment

**Time**: ~2 min
**Talking points**:

- Format: FIVE auto-graded coding tasks, 100 marks total, 3 hours, open-book (documentation allowed; AI assistants NOT allowed). No MCQ, no fill-in-the-blank — each task states a business goal, data, constraints and acceptance criteria; the how is up to the student. (The slide still shows the older four-task layout — an update to the five-task layout is in flight; teach the five-task reality below, and point students to `assessment/README.md` for the authoritative version.)
- Task 1 (20 marks): customer segments and mixture models — feature/scaling decisions, choosing K, recovering and profiling planted personas, EM from scratch — through `ClusteringEngine`.
- Task 2 (20 marks): reduction, embeddings and anomaly screening on the real credit-scoring data with injected anomalies — PCA variance and loadings, neighbour-preserving 2-D maps, three anomaly kinds, combining detectors, the LOF masking trap — through `DimReductionEngine` + `AnomalyDetectionEngine`.
- Task 3 (20 marks): baskets and recommendations — itemsets, support/confidence/lift, matrix factorisation vs a bias baseline, personalised ranking, new users.
- Task 4 (15 marks): topics from AG News text — cleaning, TF-IDF, NMF topics, coherent and distinct keywords — through `DimReductionEngine` (NMF).
- Task 5 (25 marks): the capstone — from discovered segments to a neural network: forward/backward pass, stable cross-entropy, and unsupervised features feeding a supervised network (the module's project) — `ClusteringEngine` / `DimReductionEngine` + `SklearnTrainable`.
- Grading model: automated checks against ground truth the brief does not reveal (e.g. ARI vs planted segments, ROC-AUC vs hidden anomaly flags) — so the only way to score is to genuinely recover the structure. Polars only, keep the given seeds, no hardcoded secrets or model names. Every engine on this slide is one the bridge slides taught.
- "If beginners look confused": "Five tasks, each mapped to lessons you have done. The tasks describe WHAT to achieve, not which function to call — that judgment is the assessment."
  **Transition**: "Looking ahead to Module 5."

---

## Slide 89: Looking Ahead: Module 5

**Time**: ~2 min
**Talking points**:

- M5: Deep Learning — Vision & Transfer Learning. Walk the real list on the slide: autoencoders (reconstruction loss as unsupervised feature learning — the spectrum again), CNNs for vision, RNNs for sequences, transformers (attention, positional encoding), GANs, diffusion and graph neural networks, transfer learning and reinforcement learning.
- The thread: "4.8's hidden layers → M5's specialised architectures → M6: LLMs, agents and RAG." Large language models, agents and retrieval-augmented generation come in Module 6, built on the transformer from M5 — do not promise them for M5.
- The conceptual link to today: "Attention can be read as a soft, input-dependent weighting over tokens — the same idea as the GMM responsibilities and Mixture-of-Experts gating from 4.2."
- "If beginners look confused": "M5 takes the toolkit from 4.8 and applies it to images, sequences and beyond. You already know the foundations — same maths, specialised architectures."
- "If experts look bored": "The interesting bridge for M5 is that attention is a learned kernel on token embeddings; multi-head attention is a soft factorisation of token co-occurrence structure. We derive it next module."
  **Transition**: "One final slide."

---

## Slide 90: Module 4 Complete

**Time**: ~1 min
**Talking points**:

- Read the provocation: "The features that matter most are the ones no human thought to create."
- Thank the class. Remind them of the exercise schedule and the five-task end-of-module assessment.
- Summary line: "You crossed the bridge today. You arrived this morning with hand-crafted features. You leave this evening having seen a neural network discover features on its own — and having run every stage of that discovery yourself. That is the bridge from classical ML to modern AI, and you walked across it in three hours."
- Optional closing: "If you want to see the spectrum continue, Module 5 scales what you just built to the architectures behind modern vision, language and beyond. See you next week."
  **Transition**: [End of module]
