# MLFP04 — Task 4: Topics from News Text

**Weight**: 15 marks · **Outcomes**: 4.6 (text cleaning, TF-IDF, topic models by matrix factorisation, coherence, human-readable topics)
**Data**: `mlfp05/ag_news.parquet` (real news articles, title + lead), loaded with `shared.MLFPDataLoader`. The grader draws its own secret corpora from the AG News files.

## Scenario

A media-monitoring team receives batches of wire articles and wants each
batch organised into topics automatically, so analysts can skim a topic's
keywords and read only the articles that matter. The articles arrive as raw
text: they contain wire-service bylines such as "(Reuters) Reuters -",
HTML-entity debris such as `#39;` and `quot;`, and the usual function words.

Analysts judge topics in two ways. A topic's keywords must belong together
(they tend to appear in the same articles), and must say something about the
articles filed under that topic and not about the rest. The team also knows
roughly how many broad news sections each batch covers, and expects the
topics to line up with those sections more than with chance.

## What to submit

`discover_topics(docs, n_topics)` in `starter.py` (signature fixed). `docs`
is a list of raw article strings; `n_topics` is the number of topics wanted.
It returns a dict:

| Key          | Meaning                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------- |
| `doc_topics` | one int topic id in `[0, n_topics)` per document, in input order                             |
| `top_words`  | `n_topics` lists of 10 distinct lowercase keywords each, the best description of each topic |

## Acceptance criteria

The grader draws two secret corpora: a random 3 or 4 of the four AG News
sections (world, sports, business, science/technology), about 350 raw articles
each, shuffled, with `n_topics` set to the number of sections. It never passes
the section labels. On **both** corpora:

- normalised mutual information between `doc_topics` and the hidden sections
  is at least 0.20;
- NPMI coherence of every topic's keywords is at least 0.0, and their mean is
  at least 0.15. NPMI is computed by the grader over the same documents, with
  document-level co-occurrence of lowercase alphabetic tokens; a keyword pair
  that never co-occurs scores −1;
- no keyword appears in the top-10 lists of more than two topics;
- for every topic, its keywords occur on average at least twice as often in
  the documents assigned to it as in the other documents.

Format, distinctness and keyword checks earn marks only when the section or
coherence checks pass.

## Rules

- The topic factorisation runs through kailash-ml `DimReductionEngine`.
  Vectorising text with scikit-learn's feature-extraction tools is allowed.
- Polars for data handling (no pandas).
