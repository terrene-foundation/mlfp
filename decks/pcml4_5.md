---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 4.5: NLP — Text to Topics

### Module 4: Advanced ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Preprocess text data: tokenisation, stopword removal, stemming
- Represent text numerically with TF-IDF
- Extract topics from document collections using BERTopic
- Integrate text features into ML pipelines with Polars

---

## Recap: Lesson 4.4

- Isolation Forest isolates anomalies with few random splits
- LOF detects local density anomalies via neighbourhood density
- `EnsembleEngine` combines models via voting, stacking, or blending
- Diversity among base models drives ensemble performance

---

## Why NLP in ML?

```
Structured data:  price, floor_area, lease_years → numbers
Unstructured text: "Spacious 4-room flat near MRT, renovated kitchen"

Text contains rich signal:
  - Sentiment ("excellent condition" vs "needs renovation")
  - Topics (transportation, amenities, renovation)
  - Specifics (MRT station names, school names)

Challenge: models need numbers, not words.
```

---

## Text Preprocessing Pipeline

```
Raw text:   "The spacious 4-room flat is NEAR Tampines MRT!!!"
    ↓ Lowercase
            "the spacious 4-room flat is near tampines mrt!!!"
    ↓ Remove punctuation
            "the spacious 4room flat is near tampines mrt"
    ↓ Tokenise
            ["the", "spacious", "4room", "flat", "is", "near", "tampines", "mrt"]
    ↓ Remove stopwords
            ["spacious", "4room", "flat", "near", "tampines", "mrt"]
    ↓ Ready for vectorisation
```

---

## Preprocessing with Polars

```python
import polars as pl

df_text = df.with_columns(
    # Lowercase
    pl.col("description").str.to_lowercase().alias("text_clean"),
).with_columns(
    # Remove punctuation
    pl.col("text_clean").str.replace_all(r"[^\w\s]", ""),
).with_columns(
    # Remove extra whitespace
    pl.col("text_clean").str.replace_all(r"\s+", " ").str.strip_chars(),
)
```

Polars string operations are vectorised and fast.

---

## TF-IDF: From Words to Numbers

```
TF  (Term Frequency):     How often a word appears in THIS document
IDF (Inverse Doc Freq):   How rare a word is ACROSS documents

TF-IDF = TF × IDF

  "flat" → high TF, low IDF (appears everywhere)  → low score
  "penthouse" → low TF, high IDF (rare)            → high score
```

TF-IDF scores words that are **distinctive** to a document.

---

## TF-IDF in Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Fit TF-IDF on listing descriptions
vectorizer = TfidfVectorizer(
    max_features=1000,      # keep top 1000 terms
    min_df=5,               # ignore terms in < 5 documents
    max_df=0.95,            # ignore terms in > 95% of documents
    ngram_range=(1, 2),     # unigrams and bigrams
    stop_words="english",
)

tfidf_matrix = vectorizer.fit_transform(df_text["text_clean"].to_list())
feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary: {len(feature_names)} terms")
print(f"Matrix shape: {tfidf_matrix.shape}")
```

---

## Combining Text and Tabular Features

```python
import numpy as np
import polars as pl

# Numeric features
X_numeric = df.select(["price", "floor_area", "lease_years"]).to_numpy()

# TF-IDF features (sparse matrix → dense for small data)
X_text = tfidf_matrix.toarray()

# Combine
X_combined = np.hstack([X_numeric, X_text])

print(f"Numeric features: {X_numeric.shape[1]}")
print(f"Text features:    {X_text.shape[1]}")
print(f"Combined:         {X_combined.shape[1]}")
```

---

## Topic Modelling: What Are People Talking About?

```
1,000 property listings → What are the main themes?

Topic 1: "mrt", "bus", "transport", "station"    → Transportation
Topic 2: "school", "park", "playground"           → Family amenities
Topic 3: "renovated", "kitchen", "bathroom"       → Renovation
Topic 4: "high floor", "view", "unblocked"        → View/Location
Topic 5: "investment", "rental", "yield"           → Investment
```

Topic modelling discovers these themes automatically.

---

## BERTopic: Modern Topic Modelling

BERTopic uses transformer embeddings (not just word counts).

```python
from bertopic import BERTopic

# Fit topic model
topic_model = BERTopic(
    nr_topics="auto",
    min_topic_size=20,
)

topics, probs = topic_model.fit_transform(
    df_text["text_clean"].to_list()
)

# View discovered topics
print(topic_model.get_topic_info())
```

---

## BERTopic Pipeline

```
Documents → Embed (sentence transformers)
         → Reduce (UMAP)
         → Cluster (HDBSCAN)
         → Represent (c-TF-IDF per cluster)

Each step uses what we learned:
  - Embeddings: transform text to vectors (neural)
  - UMAP: reduce dimensions (Lesson 4.3)
  - HDBSCAN: find clusters (Lesson 4.1)
  - c-TF-IDF: name the topics (this lesson)
```

---

## Interpreting Topics

```python
# Top words per topic
for topic_id in range(5):
    words = topic_model.get_topic(topic_id)
    word_list = ", ".join([w for w, _ in words[:5]])
    print(f"Topic {topic_id}: {word_list}")

# Assign topics to DataFrame
df_topics = df.with_columns(
    pl.Series("topic", topics),
)

# Analyse: average price per topic
topic_prices = df_topics.group_by("topic").agg(
    pl.col("price").mean().alias("avg_price"),
    pl.col("topic").count().alias("count"),
).sort("avg_price", descending=True)
print(topic_prices)
```

---

## Topics as Features

```python
# Use topic assignments as a feature for prediction
df_enriched = df.with_columns(
    pl.Series("topic", topics),
    pl.Series("topic_prob", [p.max() for p in probs]),
)

# One-hot encode topics
from sklearn.preprocessing import OneHotEncoder
topic_features = OneHotEncoder(sparse_output=False).fit_transform(
    df_enriched["topic"].to_numpy().reshape(-1, 1)
)
```

Topics transform unstructured text into a small number of meaningful categories.

---

## Exercise Preview

**Exercise 4.5: NLP Analysis of Property Listings**

You will:

1. Preprocess listing descriptions with Polars string operations
2. Build TF-IDF features and combine with tabular data
3. Discover topics with BERTopic and interpret them
4. Use topics as features to improve price prediction

Scaffolding level: **Light+ (~40% code provided)**

---

## Common Pitfalls

| Mistake                                  | Fix                                        |
| ---------------------------------------- | ------------------------------------------ |
| Forgetting stopword removal              | Common words dominate TF-IDF without it    |
| Too many TF-IDF features                 | Use `max_features` to limit vocabulary     |
| Interpreting topic numbers as meaningful | Topic IDs are arbitrary; look at the words |
| BERTopic on very small datasets          | Need 100+ documents minimum                |
| Not combining text with tabular features | Text alone rarely outperforms combined     |

---

## Summary

- Text preprocessing: lowercase, remove punctuation, tokenise, remove stopwords
- TF-IDF converts text to numerical features based on word distinctiveness
- BERTopic discovers topics using embeddings + UMAP + HDBSCAN
- Topics become categorical features for downstream ML models
- Combining text and tabular features typically outperforms either alone

---

## Next Lesson

**Lesson 4.6: Drift Monitoring**

We will learn:

- Detecting data and model drift with `DriftMonitor`
- Population Stability Index (PSI) for distribution shifts
- Building automated monitoring for production models
