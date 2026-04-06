# Module 4: Advanced ML -- Unsupervised Methods and Deep Learning -- Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: Title Slide

**Time**: ~2 min
**Talking points**:

- Read the provocation aloud: "Would YOU trust a system that cries wolf 10,000 times a day?"
- Let it land. Pause for 3 seconds. This provocation sets up the entire module -- the AML case study will answer it.
- Today marks the halfway point of the programme: M1-M3 gave you supervised ML, M4 opens the unsupervised world.
- If beginners look confused: "By the end of today, you will know what to do when nobody tells you the right answer."
- If experts look bored: "We will derive the EM algorithm from scratch and cover Flash Attention -- this is not a surface pass."
  **Transition**: "Let me set the stage with where you have been so far..."

---

## Slide 2: Recap -- Your Journey So Far

**Time**: ~3 min
**Talking points**:

- Quick recap only -- students should already know this. Do not re-teach M1-M3.
- Emphasise the thread: M1 (explore data), M2 (engineer features), M3 (supervised prediction). All required labels.
- The callout is the key message: "M3 gave you labelled data. Today: what happens when nobody tells you the right answer?"
- If beginners look confused: "Think of labelled data as an answer key. Today you have no answer key."
- If experts look bored: "Think about how your M3 models would fail on imbalanced data -- that is where we start."
  **Transition**: "So what is new in Module 4?"

---

## Slide 3: What's New in Module 4

**Time**: ~2 min
**Talking points**:

- Walk through both columns: Part A (unsupervised + monitoring, 6 lessons) and Part B (deep learning, 2 lessons).
- Highlight the five new Kailash engines. Students should note these down.
- Set expectations: Part A is conceptually harder (no right answers). Part B is computationally harder (neural networks).
- If beginners look confused: "We will take each piece slowly with real examples."
- If experts look bored: "We will cover spectral clustering, HDBSCAN theory, and the full EM derivation."
  **Transition**: "Here is where these engines fit in the full Kailash map..."

---

## Slide 4: Cumulative Kailash Engine Map

**Time**: ~2 min
**Talking points**:

- This is a visual anchor. Students should see the progression from M1 to M4.
- Point to the M4 box: it has 5 engines, the most of any single module.
- Mention that by end of M4, they will have 14 engines across 4 modules.
- SKIPPABLE if running short.
  **Transition**: "Now, let me tell you a story about a bank..."

---

## Slide 5: Opening Case -- The 99.9% Accuracy Trap

**Time**: ~5 min
**Talking points**:

- This is the anchor case for the entire module. Spend time here.
- Read the numbers slowly: 99.9% accuracy, 0.1% suspicious, 10,000 false alerts per day.
- Ask the class: "Raise your hand if you would be happy with 99.9% accuracy." Most will raise their hand. Then reveal the trap.
- The model learned to predict "not suspicious" for everything -- a trivial baseline.
- If beginners look confused: "Imagine a fire alarm that rings 10,000 times a day. Would you run every time? No. You would start ignoring it."
- If experts look bored: "This is the class imbalance problem at industrial scale. Think about precision-recall tradeoffs."
- PAUSE for questions here. This case study generates good discussion.
  **Transition**: "So why does this matter for what we learn today?"

---

## Slide 6: Why This Matters for Module 4

**Time**: ~3 min
**Talking points**:

- Left column: the real problems (no labels, clusters not classes, drift, scale).
- Right column: what you need (clustering, dimensionality reduction, anomaly detection, drift monitoring, deep learning).
- Map each problem to the lesson number. Students should see the structure of the day.
- If beginners look confused: "Each lesson solves one piece of this puzzle."
- If experts look bored: "Think about how you would architect a real AML system. We will build towards that."
  **Transition**: "Before we dive in, let us be precise about the key shift from M3..."

---

## Slide 7: Supervised vs Unsupervised -- The Key Shift

**Time**: ~3 min
**Talking points**:

- Walk through the table row by row. The evaluation row is crucial: unsupervised has no "right answer" to check against.
- Emphasise the risk column: supervised risks overfitting to labels, unsupervised risks finding patterns that do not exist (pareidolia in data).
- The callout is the key insight: "Validation requires domain expertise and multiple complementary methods."
- If beginners look confused: "In supervised learning, the answer key tells you if you are right. In unsupervised learning, you need an expert to look at the results and say 'yes, that makes sense.'"
- If experts look bored: "Think about the philosophical implications -- how do you validate structure you did not expect to find?"
  **Transition**: "Let us start with the most intuitive unsupervised method: clustering."

---

## Slide 8: Section Title -- 4.1 Clustering

**Time**: ~1 min
**Talking points**:

- Section transition. Read the subtitle.
- "Grouping similar things without being told what the groups are."
  **Transition**: Advance immediately.

---

## Slide 9: What Is Clustering?

**Time**: ~3 min
**Talking points**:

- Use the customer example: 10,000 customers, no labels, natural groups emerge.
- Walk through the SVG diagram: three clusters with different colours. Point out that the algorithm finds these groups automatically.
- If beginners look confused: "Imagine sorting your wardrobe by colour without anyone telling you which colours go together. You just see the groups."
- If experts look bored: "Think about the mathematical definition of 'similar' -- distance metrics matter enormously."
  **Transition**: "The simplest algorithm for finding these groups is K-means..."

---

## Slide 10: K-Means -- The Simplest Clustering Algorithm

**Time**: ~4 min
**Talking points**:

- Use the "musical chairs" metaphor. Walk through the 4 steps slowly.
- Emphasise: "You choose k. That is the big decision." This sets up the next slide.
- If beginners look confused: Walk through a tiny example on the whiteboard -- 6 points, k=2. Show the first iteration.
- If experts look bored: "What distance metric does k-means use? Euclidean. What happens with Manhattan distance?"
- PAUSE for questions. K-means is foundational and must be understood before proceeding.
  **Transition**: "Let us formalise what k-means actually optimises..."

---

## Slide 11: Lloyd's Algorithm -- K-Means Formally

**Time**: ~4 min (THEORY)
**Talking points**:

- Show the objective function J (WCSS). Explain each symbol.
- E-step: assign each point to the nearest centroid. M-step: recompute centroids as means.
- If beginners look confused: "The E-step is 'sit at the nearest chair.' The M-step is 'move the chair to the middle of its group.'"
- If experts look bored: "Note the connection to EM for GMMs -- k-means is a special case where the covariance is spherical and the responsibilities are hard assignments."
  **Transition**: "K-means always converges, but does it converge to the right answer?"

---

## Slide 12: Convergence and k-means++

**Time**: ~4 min (THEORY)
**Talking points**:

- Left column: convergence guarantee. Stress that it converges to a LOCAL minimum, not necessarily global.
- Right column: k-means++ solves the initialisation problem. Explain the probability-proportional-to-distance sampling.
- If beginners look confused: "Imagine placing your initial chairs near each other by accident. You get bad groups. k-means++ spreads the chairs out."
- If experts look bored: "The O(log k) competitive guarantee is one of the cleanest results in approximation algorithms."
  **Transition**: "Even with good initialisation, we still need to choose k..."

---

## Slide 13: Choosing k -- Elbow and Gap Statistic

**Time**: ~4 min (THEORY)
**Talking points**:

- Elbow method: intuitive but often ambiguous. Draw a quick sketch if needed.
- Gap statistic: more principled. Compare to uniformly distributed reference data.
- Silhouette score in the callout: range [-1, 1], higher is better.
- If beginners look confused: "The elbow method is like looking at where the returns stop being worth it. Gap statistic is more rigorous."
- If experts look bored: "Discuss the limitations of silhouette for non-convex clusters."
  **Transition**: "K-means assumes spherical clusters. What if clusters are not spherical?"

---

## Slide 14: When K-Means Fails -- Non-Convex Clusters

**Time**: ~4 min (THEORY)
**Talking points**:

- Show the three failure modes: elongated, nested, unequal density.
- Walk through spectral clustering steps: similarity graph, Laplacian, eigendecompose, k-means in eigenspace.
- The normalised Laplacian equation is for reference -- do not derive it line by line unless experts ask.
- If beginners look confused: "Spectral clustering transforms the data so that hard-to-separate clusters become easy to separate."
- If experts look bored: "The connection to graph cuts and the normalised cut objective is deep -- this is where ML meets graph theory."
  **Transition**: "How do we know how many clusters to use in spectral clustering?"

---

## Slide 15: Eigengap Heuristic

**Time**: ~3 min (THEORY)
**Talking points**:

- k connected components yield exactly k zero eigenvalues. In practice, look for the largest gap.
- The normalised cuts callout is advanced -- mention it briefly.
- SKIPPABLE if running short. Combine with previous slide.
- If beginners look confused: Skip the Ncut formula. Just say "the eigenvalues tell you how many natural groups there are."
- If experts look bored: "Shi & Malik's normalised cuts paper is foundational in computer vision."
  **Transition**: "What about clusters of different densities?"

---

## Slide 16: HDBSCAN -- Density-Based Clustering

**Time**: ~4 min (THEORY)
**Talking points**:

- Walk through the 5 steps. Emphasise that you do NOT need to specify k.
- The key advantage callout: automatic number of clusters and noise labelling.
- If beginners look confused: "HDBSCAN finds dense regions and calls everything else noise. You do not have to tell it how many groups to find."
- If experts look bored: "The mutual reachability distance and stability-based extraction are elegant -- the condensed cluster tree is worth exploring."
  **Transition**: "For those who want a different theoretical lens on clustering..."

---

## Slide 17: Information-Theoretic Clustering

**Time**: ~3 min (ADVANCED)
**Talking points**:

- This is an advanced slide. Mention the Information Bottleneck concept briefly.
- The connection to deep learning generalisation is provocative -- mention it as a teaser.
- SKIPPABLE if running short. This is the first slide to cut.
- If beginners look confused: Skip entirely or say "There are alternative ways to define 'similar' using information theory instead of geometry."
- If experts look bored: "Tishby's Information Bottleneck and its connection to deep learning representation learning is one of the most debated topics in ML theory."
  **Transition**: "Let us connect all this theory to Kailash..."

---

## Slide 18: Kailash Bridge -- AutoMLEngine for Clustering

**Time**: ~3 min
**Talking points**:

- This is a KAILASH BRIDGE slide. Show how theory maps to engine.
- Walk through the code: AutoMLConfig with task="clustering", algorithms list, metric, n_trials.
- Emphasise: the engine compares all algorithms programmatically. You do not have to implement each one.
- SWITCH TO LIVE CODING if time allows. Show AutoMLEngine with the exercise dataset.
- If beginners look confused: "All that theory we just covered? Kailash does it in 6 lines of code."
- If experts look bored: "Look at the metric parameter -- you can plug in custom metrics."
  **Transition**: "K-means gives hard assignments. What if a point belongs to multiple groups?"

---

## Slide 19: Section Title -- 4.2 EM Algorithm and GMMs

**Time**: ~1 min
**Talking points**:

- Read the subtitle: "Soft clustering -- when points belong to multiple groups with different probabilities."
  **Transition**: Advance immediately.

---

## Slide 20: Hard vs Soft Clustering

**Time**: ~3 min
**Talking points**:

- The luxury buyer / bulk buyer example is excellent. Use it.
- Hard clustering: "This customer IS a luxury buyer." Soft: "This customer is 70% luxury buyer, 25% bulk buyer."
- If beginners look confused: "Think about yourself. Are you 100% one type of customer? Probably not."
- If experts look bored: "Soft clustering gives you a posterior distribution over cluster memberships. This is Bayesian."
  **Transition**: "The mathematical model behind soft clustering is the Gaussian Mixture Model..."

---

## Slide 21: GMM -- The Generative Model

**Time**: ~4 min (THEORY)
**Talking points**:

- Walk through the generative story: pick a component, draw from its Gaussian.
- Explain mixing weights, means, and covariances.
- The key message: "We can write down the likelihood, but we cannot maximise it directly because of the sum inside the log."
- If beginners look confused: "Imagine mixing three colours of paint. Each colour is a bell curve. The mixture is the combined shape."
- If experts look bored: "Note the identifiability issue -- permuting component labels gives the same likelihood."
  **Transition**: "Why can we not just maximise the likelihood directly?"

---

## Slide 22: The Log-Likelihood Problem

**Time**: ~4 min (THEORY)
**Talking points**:

- Show the log-likelihood with the sum inside the log. Explain why this has no closed-form MLE.
- Show the circularity: mu depends on gamma, which depends on mu.
- If beginners look confused: "It is a chicken-and-egg problem. The EM algorithm breaks the cycle."
- If experts look bored: "This is where the ELBO and variational inference enter the picture."
  **Transition**: "The EM algorithm breaks this circularity with two alternating steps..."

---

## Slide 23: EM Algorithm -- E-Step

**Time**: ~4 min (THEORY)
**Talking points**:

- The E-step computes responsibilities using Bayes' rule.
- Walk through the formula slowly: prior times likelihood over evidence.
- The plain English callout is excellent: "Given our current best guess, how likely is it that this point came from bell curve k?"
- If beginners look confused: Use the plain English version only.
- If experts look bored: "The E-step is computing the posterior over the latent variables given the current parameters."
  **Transition**: "Now that we have responsibilities, we can update the parameters..."

---

## Slide 24: EM Algorithm -- M-Step

**Time**: ~4 min (THEORY)
**Talking points**:

- Walk through each update: effective count, updated mean, updated mixing weight, updated covariance.
- Emphasise that each update is a weighted version of the usual formula -- weighted by responsibilities.
- If beginners look confused: "The M-step is just taking weighted averages. The weights come from the E-step."
- If experts look bored: "Show that when responsibilities are 0 or 1 (hard assignments), this reduces to k-means."
  **Transition**: "Does this process always converge?"

---

## Slide 25: EM Convergence

**Time**: ~3 min (THEORY)
**Talking points**:

- Left column: monotonically increases log-likelihood, converges to local maximum. Mention BIC/AIC for model selection.
- Right column: EM as a general template -- HMMs, Factor Analysis, LDA. This shows the generality of the pattern.
- If beginners look confused: "EM always gets better or stays the same. It never gets worse. But it might find a 'local best' rather than the 'global best.'"
- If experts look bored: "The ELBO proof of monotonic improvement is elegant -- consider reading Bishop Chapter 9."
  **Transition**: "Let me show you the complete EM picture..."

---

## Slide 26: EM Summary -- The Complete Picture

**Time**: ~2 min
**Talking points**:

- Walk through the SVG diagram: Initialise, E-step, M-step, Converged?, loop back.
- The Kailash callout connects back to practice: AutoMLEngine compares GMM against other algorithms automatically.
- SKIPPABLE if running short -- this is a visual summary of what was just taught.
  **Transition**: "Now we have clusters, but we still cannot visualise them in high dimensions..."

---

## Slide 27: Section Title -- 4.3 Dimensionality Reduction

**Time**: ~1 min
**Talking points**:

- "Projecting 100 columns to 2 so you can see patterns."
  **Transition**: Advance immediately.

---

## Slide 28: The Curse of Dimensionality

**Time**: ~3 min
**Talking points**:

- The e-commerce example grounds it: 200 features, cannot visualise, cannot compute distances reliably.
- The shadow analogy is excellent: "A shadow is a 2D projection of a 3D object. PCA finds the best angle to cast the shadow."
- If beginners look confused: "Too many columns makes everything harder. We need to summarise."
- If experts look bored: "The curse of dimensionality has formal consequences -- in high dimensions, all points are roughly equidistant."
  **Transition**: "PCA finds the best summary by finding the directions of maximum variance..."

---

## Slide 29: PCA -- Finding the Best Shadow

**Time**: ~5 min (THEORY)
**Talking points**:

- Walk through all 4 derivation steps. This is a key derivation -- take time.
- Step 1: centre the data. Step 2: covariance matrix. Step 3: eigendecompose. Step 4: project.
- If beginners look confused: "PCA finds the most important directions in your data and drops the less important ones."
- If experts look bored: "The Lagrange multiplier derivation on the next slide makes this rigorous."
  **Transition**: "Why do the eigenvalues tell us about variance?"

---

## Slide 30: PCA -- Why Eigenvalues = Variance

**Time**: ~4 min (THEORY)
**Talking points**:

- The Lagrange multiplier derivation is clean and important. Walk through it.
- Left column: the scree plot and explained variance ratio.
- Right column: SVD connection -- more numerically stable.
- If beginners look confused: "The eigenvalue tells you how much information is in each direction. Big eigenvalue = important direction."
- If experts look bored: "The SVD approach avoids forming the covariance matrix explicitly, which matters for numerical stability with ill-conditioned data."
  **Transition**: "How much information do we lose when we keep only k components?"

---

## Slide 31: PCA Reconstruction Error

**Time**: ~3 min (THEORY)
**Talking points**:

- Reconstruction error = sum of discarded eigenvalues.
- Eckart-Young theorem: PCA gives the best rank-k approximation.
- Rule of thumb: keep enough for 90-95% of variance.
- If beginners look confused: "If 5 super-features explain 93% of the information, those 5 are almost as good as the original 200."
- If experts look bored: "The Eckart-Young theorem is a fundamental result in linear algebra with applications far beyond PCA."
  **Transition**: "PCA preserves global structure. What about local structure?"

---

## Slide 32: t-SNE -- Preserving Local Neighbourhoods

**Time**: ~4 min (THEORY)
**Talking points**:

- Walk through Steps 1 and 2: Gaussian similarities in high-D, Student-t in low-D.
- Emphasise the key difference: PCA preserves global variance, t-SNE preserves who is near whom.
- If beginners look confused: "t-SNE keeps neighbours together. If two points are close in the original data, they will be close in the 2D picture."
- If experts look bored: "The choice of Student-t with 1 DOF is specifically to handle the crowding problem."
  **Transition**: "How do we actually optimise the t-SNE embedding?"

---

## Slide 33: t-SNE -- The KL Divergence Objective

**Time**: ~4 min (THEORY)
**Talking points**:

- Step 3: minimise KL divergence.
- Discuss perplexity and its effect.
- The CAUTION callout is critical: "t-SNE is for VISUALISATION ONLY. Do not cluster on t-SNE output."
- If beginners look confused: "t-SNE makes pretty pictures. It is NOT a preprocessing step."
- If experts look bored: "KL divergence is asymmetric -- it penalises pulling apart nearby points more than pushing together distant points."
- PAUSE for questions. Students often misuse t-SNE.
  **Transition**: "UMAP is a newer alternative that addresses some of t-SNE's limitations..."

---

## Slide 34: UMAP -- Topological Data Analysis

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the comparison table: UMAP is faster, preserves global structure better, supports new point projection.
- Do not get deep into fuzzy simplicial sets unless experts ask.
- If beginners look confused: "UMAP is like t-SNE but faster and better at showing the big picture."
- If experts look bored: "The Riemannian geometry foundation of UMAP is genuinely different from t-SNE's information-theoretic approach."
  **Transition**: "For those who want more dimensionality reduction tools..."

---

## Slide 35: Kernel PCA and ICA

**Time**: ~3 min (ADVANCED)
**Talking points**:

- Kernel PCA: nonlinear PCA via the kernel trick. Mention common kernels.
- ICA: finds independent components (cocktail party problem).
- SKIPPABLE if running short. Second slide to cut after Information-Theoretic Clustering.
- If beginners look confused: "These are advanced variants. PCA and UMAP cover most use cases."
- If experts look bored: "ICA's connection to the Central Limit Theorem is a beautiful result."
  **Transition**: "Let us connect dimensionality reduction to Kailash..."

---

## Slide 36: Kailash Bridge -- ModelVisualizer

**Time**: ~3 min
**Talking points**:

- Show the code: pca_plot, tsne_plot, umap_plot. All three in one engine.
- The lab exercise callout previews ex_4.3.
- SWITCH TO LIVE CODING if time allows.
- If beginners look confused: "Three lines of code for each visualisation method."
- If experts look bored: "Look at the parameter options -- you can tune perplexity, n_neighbors, min_dist."
  **Transition**: "Now we can find groups and visualise them. What about finding the unusual points?"

---

## Slide 37: Section Title -- 4.4 Anomaly Detection and Ensembles

**Time**: ~1 min
**Talking points**:

- "Finding the needles in the haystack, then combining detectors."
  **Transition**: Advance immediately.

---

## Slide 38: What Is an Anomaly?

**Time**: ~3 min
**Talking points**:

- Three types of anomalies: point, contextual, collective. Use the examples.
- Connect back to Credit Suisse: the AML system used supervised classification. Anomaly detection takes a fundamentally different approach.
- If beginners look confused: "An anomaly is something that does not look like everything else."
- If experts look bored: "The distinction between contextual and collective anomalies drives very different modelling choices."
  **Transition**: "The most elegant anomaly detection algorithm is Isolation Forest..."

---

## Slide 39: Isolation Forest

**Time**: ~4 min (THEORY)
**Talking points**:

- The key insight is brilliant: anomalies are EASIER to isolate than normal points.
- Walk through the 4 steps and the anomaly score formula.
- Scores close to 1 = anomaly. Scores close to 0.5 = normal.
- If beginners look confused: "If a point is weird, it does not take many random cuts to separate it from everyone else."
- If experts look bored: "The connection between average path length and BST average depth is the normalisation trick."
  **Transition**: "Isolation Forest uses global structure. LOF uses local density..."

---

## Slide 40: Local Outlier Factor

**Time**: ~3 min (THEORY)
**Talking points**:

- LOF compares local density to neighbours' density. LOF around 1 = normal, much greater than 1 = outlier.
- Mention the strength (varying density) and weakness (O(n-squared), sensitive to k).
- If beginners look confused: "LOF asks: is this point in a sparse area compared to its neighbours?"
- If experts look bored: "The reachability distance smoothing is what makes LOF robust to density variations."
  **Transition**: "No single detector catches everything. Ensembles combine them..."

---

## Slide 41: Ensemble Methods -- Combining Detectors

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the table: blending, stacking, bagging, boosting.
- Emphasise that these are the same concepts from M3 supervised learning, applied to anomaly detection.
- If beginners look confused: "Think of it as asking multiple doctors for an opinion, then combining their advice."
- If experts look bored: "Stacking with a meta-learner is particularly powerful when you have some labelled anomalies."
  **Transition**: "Let us see how EnsembleEngine implements this..."

---

## Slide 42: Kailash Bridge -- EnsembleEngine

**Time**: ~3 min
**Talking points**:

- Show the blend() and stack() code patterns.
- Emphasise the simplicity: 3 detectors combined in a few lines.
- SWITCH TO LIVE CODING if time allows.
  **Transition**: "We can find clusters and detect anomalies. Now what about text data?"

---

## Slide 43: Section Title -- 4.5 NLP: Text to Topics

**Time**: ~1 min
**Talking points**:

- "Teaching computers to read -- from bag of words to BERTopic."
  **Transition**: Advance immediately.

---

## Slide 44: What Is NLP?

**Time**: ~3 min
**Talking points**:

- Focus on the question: "What are people talking about?"
- Walk through topic modelling use cases. Pick one relevant to the class (e.g., customer support tickets).
- The evolution: Bag of Words, Word2Vec, BERTopic. We cover all three.
- If beginners look confused: "NLP turns words into numbers so computers can find patterns in text."
- If experts look bored: "We are covering the progression from count-based to neural representations."
  **Transition**: "The simplest way to represent text as numbers is TF-IDF..."

---

## Slide 45: TF-IDF

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the formula: TF (how often) times IDF (how rare).
- The plain English callout: "'The' appears everywhere (low IDF). 'Transformer' only in ML papers (high IDF)."
- If beginners look confused: "Important words are those that appear a lot in one document but rarely in others."
- If experts look bored: "TF-IDF is a bag-of-words model -- it ignores word order entirely."
  **Transition**: "TF-IDF ignores meaning. Word2Vec captures it..."

---

## Slide 46: Word2Vec

**Time**: ~4 min (THEORY)
**Talking points**:

- Skip-gram objective: predict context words from the centre word.
- Mention negative sampling as a computational shortcut.
- The "King - Man + Woman = Queen" result is iconic. Use it.
- If beginners look confused: "Words that appear in similar contexts get similar number representations."
- If experts look bored: "The negative sampling approximation to the softmax is an elegant variance reduction technique."
  **Transition**: "BERTopic combines modern embeddings with the clustering tools we just learned..."

---

## Slide 47: BERTopic -- The Modern Pipeline

**Time**: ~4 min (THEORY)
**Talking points**:

- Walk through the pipeline: Sentence Embeddings, UMAP, HDBSCAN, c-TF-IDF.
- This is where the module ties together: UMAP from 4.3, HDBSCAN from 4.1, TF-IDF from this lesson.
- Explain c-TF-IDF briefly and mention topic coherence (NPMI).
- If beginners look confused: "BERTopic uses the tools we already learned, just applied to text."
- If experts look bored: "The c-TF-IDF representation is a clever way to get interpretable topic labels from dense cluster embeddings."
  **Transition**: "We can now find patterns in data. But what happens when those patterns change over time?"

---

## Slide 48: Section Title -- 4.6 Drift Monitoring

**Time**: ~1 min
**Talking points**:

- "Your model in production -- detecting when the world changes."
  **Transition**: Advance immediately.

---

## Slide 49: Why Models Degrade

**Time**: ~4 min
**Talking points**:

- Two types: data drift (input distribution changed) and concept drift (relationship changed).
- The Singapore examples are local and relatable: property cooling measures, COVID restaurant reversals.
- The callout is critical: "Models do not know they are wrong."
- If beginners look confused: "The world changes but your model does not. It silently gives worse answers."
- If experts look bored: "Think about the difference between covariate shift, prior probability shift, and concept drift."
- PAUSE for questions. This is a production concern many students have not considered.
  **Transition**: "How do we detect drift? Two statistical tests..."

---

## Slide 50: PSI -- Population Stability Index

**Time**: ~4 min (THEORY)
**Talking points**:

- Walk through the formula: bin the feature, compare proportions.
- Walk through the threshold table: less than 0.1 (no shift), 0.1-0.25 (moderate), greater than 0.25 (significant).
- PSI is an industry standard in credit risk. Mention this for credibility.
- If beginners look confused: "PSI gives you a single number that tells you how much a feature has changed."
- If experts look bored: "PSI is related to KL divergence -- it is actually the sum of two directed KL divergences."
  **Transition**: "PSI requires binning. KS does not..."

---

## Slide 51: KS Test for Drift Detection

**Time**: ~3 min (THEORY)
**Talking points**:

- KS test: maximum distance between two CDFs. No binning required.
- Walk through the comparison: PSI for dashboards, KS for automated detection, both for critical models.
- If beginners look confused: "KS is like PSI but does not require you to choose bin sizes."
- If experts look bored: "The KS test is distribution-free, which is a significant advantage for non-parametric data."
  **Transition**: "Data drift is the leading indicator. Performance drift is the confirmation..."

---

## Slide 52: Performance Drift

**Time**: ~3 min (THEORY)
**Talking points**:

- Data drift is leading, performance drift is lagging.
- Monitor: accuracy/F1, prediction distribution shift, calibration drift, business metrics.
- Governance obligation: EU AI Act Art. 9.4 requires post-market monitoring.
- If beginners look confused: "Check if your model's outputs are still good, not just if the inputs have changed."
- If experts look bored: "Expected Calibration Error (ECE) over time is an underused metric."
  **Transition**: "Let us see how DriftMonitor implements all of this..."

---

## Slide 53: Kailash Bridge -- DriftMonitor

**Time**: ~3 min
**Talking points**:

- Walk through the code: DriftSpec, set_reference, check_drift, schedule_monitoring.
- Emphasise: daily automated monitoring with drift alerts.
- SWITCH TO LIVE CODING if time allows.
  **Transition**: "That completes Part A. Now: deep learning."

---

## Slide 54: Section Title -- 4.7 Deep Learning Foundations

**Time**: ~1 min
**Talking points**:

- "From a single neuron to ResNets -- the math behind neural networks."
- If the class needs a break, this is the natural break point (~90 minutes in).
- PAUSE for a 10-minute break here.
  **Transition**: "Let us start with the simplest building block: one neuron."

---

## Slide 55: What Is a Neural Network?

**Time**: ~3 min
**Talking points**:

- One neuron: weighted sum plus activation function. "Like a dimmer switch."
- A network: stack layers. Input, hidden, output.
- If beginners look confused: "A neuron takes numbers in, does simple math, and passes a number out. Stack many neurons and they learn complex things."
- If experts look bored: "The universal approximation theorem formalises why this works."
  **Transition**: "What makes deep learning 'deep'?"

---

## Slide 56: What Is Deep Learning?

**Time**: ~3 min
**Talking points**:

- Deep = many layers. Each layer builds on the previous.
- The image recognition hierarchy: edges, corners, object parts, objects.
- If beginners look confused: "Shallow networks are like having one employee. Deep networks are like having a department where each person adds to the previous person's work."
- If experts look bored: "The depth efficiency result -- exponentially fewer neurons for the same function -- is the theoretical justification."
  **Transition**: "Can a single-layer network learn anything?"

---

## Slide 57: Universal Approximation Theorem

**Time**: ~3 min (THEORY)
**Talking points**:

- State the theorem. Emphasise: "Existence, not construction."
- The warning callout is key: the theorem says approximation EXISTS but not how to find it.
- If beginners look confused: "In theory, even a shallow network can learn anything. In practice, deep networks are much more efficient."
- If experts look bored: "Barron's theorem gives approximation rates, which is more constructive."
  **Transition**: "How do networks learn? Through backpropagation..."

---

## Slide 58: Backpropagation -- How Networks Learn

**Time**: ~3 min (THEORY)
**Talking points**:

- The goal: compute the gradient of the loss with respect to every weight.
- The plain English: "If the prediction is wrong, backprop figures out how much each weight contributed to the error."
- The mathematical tool: chain rule.
- If beginners look confused: "Backprop is like tracing blame backwards through a chain of decisions."
- If experts look bored: "Backprop is just reverse-mode automatic differentiation on a computation graph."
  **Transition**: "Let me walk through the computation graph..."

---

## Slide 59: Backprop Computation Graph Walkthrough

**Time**: ~5 min (THEORY)
**Talking points**:

- This is a key derivation. Walk through forward pass, then backward pass.
- Show how the chain rule multiplies factors from right to left.
- If beginners look confused: "Each step in the forward direction has a matching step in the backward direction."
- If experts look bored: "Note how the Jacobian at each node is a local computation -- this is what makes backprop efficient."
  **Transition**: "Let me show the annotated chain..."

---

## Slide 60: Backprop -- The Annotated Chain

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the SVG: blue forward arrows, red backward arrows.
- The key insight callout: gradient at each layer is the PRODUCT of all gradients downstream. This explains vanishing/exploding gradients.
- If beginners look confused: "Blue arrows go forward (computing predictions). Red arrows go backward (computing blame)."
- If experts look bored: "The vanishing gradient problem is why ResNet's skip connections were revolutionary."
  **Transition**: "Now that we have gradients, how do we update the weights?"

---

## Slide 61: Gradient Descent -- The Update Rule

**Time**: ~4 min (THEORY)
**Talking points**:

- The update rule: W = W - learning_rate \* gradient.
- SGD vs AdamW. AdamW is the practical default.
- The practical recipe callout: "AdamW with lr~1e-3, then cosine schedule."
- If beginners look confused: "We nudge each weight a little bit in the direction that reduces the error."
- If experts look bored: "The weight decay in AdamW (decoupled from the gradient) is what makes it preferred over Adam."
  **Transition**: "For image data, we use convolutional neural networks..."

---

## Slide 62: CNN -- Convolution as Learned Features

**Time**: ~4 min (THEORY)
**Talking points**:

- Convolution: a small filter sliding over the input.
- Three key properties: parameter sharing, translation equivariance, local connectivity.
- Receptive field grows with depth.
- If beginners look confused: "A filter is like a magnifying glass that scans across the image looking for a specific pattern."
- If experts look bored: "The equivariance property is what ViTs learn to approximate -- the inductive bias is powerful for vision."
  **Transition**: "Let us see the full CNN architecture..."

---

## Slide 63: CNN Architecture -- Conv + Pool + FC

**Time**: ~4 min (THEORY)
**Talking points**:

- Walk through the pipeline: Conv+ReLU, MaxPool, Conv+ReLU, MaxPool, Flatten, FC.
- Explain BatchNorm (stabilises training) and Dropout (regularisation).
- If beginners look confused: "Each layer detects more complex patterns. The final layer makes the prediction."
- If experts look bored: "BatchNorm's effectiveness is still debated -- the original 'internal covariate shift' explanation may be incomplete."
  **Transition**: "What happens when we make networks very deep?"

---

## Slide 64: ResNet -- The Skip Connection Revolution

**Time**: ~5 min (THEORY)
**Talking points**:

- The degradation problem: very deep networks get WORSE. This is counterintuitive.
- The residual block: y = F(x) + x. Learn the residual, not the full mapping.
- Why it works: gradient flows through skip connection, each block learns a small correction.
- ResNet-152 surpassed human performance on ImageNet.
- If beginners look confused: "Instead of learning everything from scratch, each layer only learns what to add or change."
- If experts look bored: "The ensemble interpretation (exponentially many shorter paths) connects to stochastic depth training."
  **Transition**: "One critical hyperparameter remains: the learning rate..."

---

## Slide 65: Learning Rate Scheduling

**Time**: ~3 min (THEORY)
**Talking points**:

- Walk through the table: step decay, cosine annealing, warmup+cosine, one-cycle.
- The practical recipe: "Start with AdamW (lr=1e-3), warmup for 5-10%, then cosine decay to 1e-5."
- SKIPPABLE if running short.
- If beginners look confused: "Start with a big learning rate and gradually reduce it."
- If experts look bored: "One-cycle from Leslie Smith is underrated for fast convergence."
  **Transition**: "The attention mechanism changed everything in deep learning..."

---

## Slide 66: Attention -- The Key Innovation

**Time**: ~4 min (THEORY)
**Talking points**:

- Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V.
- Why scale by sqrt(d_k): prevents softmax saturation.
- Multi-head attention: each head attends to different relationships.
- If beginners look confused: "Attention lets every word look at every other word to understand context."
- If experts look bored: "This is the same attention from 5.1 in the next module. We introduce it here at a higher level."
  **Transition**: "For those who want cutting-edge optimisations..."

---

## Slide 67: Flash Attention

**Time**: ~3 min (ADVANCED)
**Talking points**:

- IO-aware algorithm: never materialise the full n-by-n attention matrix.
- 2-4x speedup, 5-20x memory reduction, exact output.
- Now default in PyTorch 2.0+.
- SKIPPABLE if running short.
- If beginners look confused: "A clever trick to make attention much faster without changing the result."
- If experts look bored: "The key insight is IO-awareness, not compute-awareness. The bottleneck is memory bandwidth."
  **Transition**: "And two more advanced topics for the experts..."

---

## Slide 68: SAM and Vision Transformers

**Time**: ~3 min (ADVANCED)
**Talking points**:

- SAM: optimise for the worst-case loss in a neighbourhood. Flat minima generalise better.
- ViT: treats images as sequences of patches. With enough data, attention alone suffices.
- SKIPPABLE if running short.
- If beginners look confused: Skip this slide entirely. Say "There are advanced variants we can discuss offline."
- If experts look bored: "ViT's demonstration that convolution is not a necessary inductive bias is one of the most important results in recent computer vision."
  **Transition**: "Let us connect deep learning to Kailash..."

---

## Slide 69: Kailash Bridge -- OnnxBridge

**Time**: ~3 min
**Talking points**:

- Show the code: export PyTorch model to ONNX, validate, visualise training history.
- ONNX = Open Neural Network Exchange. Train in PyTorch, serve with ONNX Runtime (2-5x speedup).
- SWITCH TO LIVE CODING if time allows.
  **Transition**: "Now let us put everything together in the capstone..."

---

## Slide 70: Section Title -- 4.8 Capstone

**Time**: ~1 min
**Talking points**:

- "Unsupervised preprocessing + Deep Learning + ONNX + InferenceServer."
  **Transition**: Advance immediately.

---

## Slide 71: Capstone Architecture

**Time**: ~3 min
**Talking points**:

- Walk through the pipeline: Raw Data, PCA/UMAP, Train Model (PyTorch CNN), OnnxBridge, InferenceServer.
- DriftMonitor runs underneath for continuous monitoring.
- This ties together the entire module.
- If beginners look confused: "This is everything we learned today in one pipeline."
- If experts look bored: "Think about where each component could fail in production and how you would diagnose it."
  **Transition**: "The last new engine is InferenceServer..."

---

## Slide 72: InferenceServer

**Time**: ~3 min
**Talking points**:

- Show the code: load model, warm cache, predict, predict_batch.
- ONNX Runtime explanation: 2-5x over native PyTorch inference.
- If beginners look confused: "InferenceServer takes your trained model and makes it ready to answer questions in real time."
- If experts look bored: "The caching and batching strategies can make a 10x difference in throughput."
  **Transition**: "Let us see how all five engines fit together..."

---

## Slide 73: Section Title -- Kailash Engine Deep Dive

**Time**: ~1 min
**Transition**: Advance immediately.

---

## Slide 74: Module 4 Engine Architecture

**Time**: ~3 min
**Talking points**:

- Walk through the SVG: all 5 engines plus ModelVisualizer, connected to the production pipeline.
- This is a reference diagram. Students should screenshot or note this.
- SKIPPABLE if running short.
  **Transition**: "Here is the complete theory-to-engine mapping..."

---

## Slide 75: Theory-to-Engine Mapping

**Time**: ~3 min
**Talking points**:

- Walk through the table: each mathematical concept maps to a Kailash engine and method.
- This is the reference table for the lab exercises.
  **Transition**: "Now let us set up for the labs..."

---

## Slide 76: Section Title -- Lab Setup

**Time**: ~1 min
**Transition**: Advance immediately.

---

## Slide 77: Exercise Overview

**Time**: ~3 min
**Talking points**:

- Walk through the 6 exercises (4.1 to 4.6). ~40% scaffolding for each.
- Students start with ex_1.py in modules/ascent04/local/.
- SWITCH TO LIVE CODING: show the file structure and how to open ex_1.py.
  **Transition**: "And two more exercises for deep learning..."

---

## Slide 78: Deep Learning Exercises

**Time**: ~2 min
**Talking points**:

- Exercises 4.7 (CNN + ResBlock) and 4.8 (full capstone pipeline).
- Scaffolding is ~40%. Structure and imports provided, students write setup, method calls, and logic.
  **Transition**: "Before the lab, let us discuss some scenarios..."

---

## Slide 79: Discussion -- Clustering in Production

**Time**: ~8 min
**Talking points**:

- Scenario 1: HDBSCAN returns 40% noise. Business wants every customer segmented. Discuss the tension.
- Scenario 2: DriftMonitor fires at 3am on a fraud detection model. What is the runbook?
- PAUSE for class discussion. Let students debate. Guide but do not answer immediately.
- If beginners look confused: "There is no single right answer. Think about what a business stakeholder would want."
- If experts look bored: "Think about the operational implications: SLAs, alerting chains, rollback procedures."
  **Transition**: "One more discussion before the lab..."

---

## Slide 80: Discussion -- The Right Tool for the Job

**Time**: ~5 min
**Talking points**:

- Walk through the scenarios table. Let students fill in their recommendations.
- Multiple valid answers exist. The key is the reasoning.
- PAUSE for brief discussion on each scenario.
  **Transition**: "Let us wrap up with the key takeaways..."

---

## Slide 81: Section Title -- Synthesis

**Time**: ~1 min
**Transition**: Advance immediately.

---

## Slide 82: What Everyone Should Remember

**Time**: ~3 min
**Talking points**:

- Read through the 6 bullet points. Each one is a core takeaway.
- Reinforce: "Models degrade in production -- drift monitoring is essential, not optional."
  **Transition**: "For those who followed the math..."

---

## Slide 83: If You Followed the Math

**Time**: ~2 min (THEORY)
**Talking points**:

- Read through the 6 mathematical takeaways.
- This is a summary -- do not re-derive. Students can review the theory slides later.
- SKIPPABLE if running short.
  **Transition**: "And for the experts..."

---

## Slide 84: For the Experts

**Time**: ~2 min (ADVANCED)
**Talking points**:

- Read through the 5 advanced takeaways.
- SKIPPABLE if running short.
  **Transition**: "Here is where you stand in the overall programme..."

---

## Slide 85: Cumulative Kailash Engine Map

**Time**: ~2 min
**Talking points**:

- Walk through the full map. M4 is "just completed." M5 and M6 are next.
- The Foundation Certificate checkpoint callout: M1-M4 completes the foundation (13 engines).
  **Transition**: "A quick preview of the assessment..."

---

## Slide 86: Assessment Preview

**Time**: ~3 min
**Talking points**:

- Walk through the quiz topics in the left column.
- Right column: assignment connection and the AI-resilient note.
- Emphasise that quizzes require their own exercise outputs.
  **Transition**: "Open your laptops. Let us start with Exercise 4.1."

---

## Time Budget Summary

| Section                         | Slides | Time         |
| ------------------------------- | ------ | ------------ |
| Title + Recap                   | 1-4    | ~9 min       |
| Opening Case                    | 5-7    | ~11 min      |
| 4.1 Clustering                  | 8-18   | ~38 min      |
| 4.2 EM / GMMs                   | 19-26  | ~22 min      |
| Break                           | --     | ~10 min      |
| 4.3 Dimensionality Reduction    | 27-36  | ~32 min      |
| 4.4 Anomaly Detection           | 37-42  | ~16 min      |
| 4.5 NLP                         | 43-47  | ~14 min      |
| 4.6 Drift Monitoring            | 48-53  | ~17 min      |
| 4.7 Deep Learning               | 54-69  | ~50 min      |
| 4.8 Capstone + Engine Deep Dive | 70-75  | ~13 min      |
| Lab Setup + Discussion          | 76-80  | ~18 min      |
| Synthesis + Assessment          | 81-86  | ~13 min      |
| **Total**                       |        | **~263 min** |

**Note**: This module is content-heavy. To fit 180 minutes:

- Skip the ADVANCED slides (17, 35, 67, 68): saves ~12 min
- Compress theory derivations (read key result, skip derivation steps): saves ~25 min
- Shorten discussion to one scenario instead of two: saves ~5 min
- Reduce break to 5 min: saves ~5 min
- Total savings: ~47 min, bringing it to ~216 min. For strict 180, also skip the deep learning LR scheduling and attention slides (65-66), which are previewed more thoroughly in M5.

**Mark as skippable**: Slides 15 (Eigengap), 17 (Info-Theoretic), 35 (Kernel PCA/ICA), 65 (LR Scheduling), 67 (Flash Attention), 68 (SAM/ViT), 74 (Engine Architecture SVG), 83 (Math Summary), 84 (Expert Summary).
