# Module 8: NLP & Transformers — Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: NLP & Transformers (Title)

**Time**: ~1 min
**Talking points**:

- Frame the arc: "We start by asking how to turn raw text into numbers, and finish by understanding how transformers changed everything."
- Provocation: "Every modern AI assistant you have ever talked to is built on what you learn today."
- If beginners look confused: "We are going to learn how computers understand language — from basic counting to billion-parameter models."
- If experts look bored: "We will derive attention from scratch, compare positional encodings, and discuss Flash Attention and GQA."

**Transition**: "Let us ground ourselves in where we have been..."

---

## Slide 2: Recap: Where We Are

**Time**: ~1 min
**Talking points**:

- Quick table: M1-M7. Do not re-teach. Key message: "Module 7 gave us neural networks. Module 8 applies them to the most important data type: language."
- If beginners look confused: "Everything we built before was for structured tables. Today we handle text."
- If experts look bored: "We move from fixed-size input MLPs and CNNs to variable-length sequence models with dynamic context."

**Transition**: "Here is why this module is historically significant..."

---

## Slide 3: The NLP Revolution Timeline

**Time**: ~2 min
**Talking points**:

- Walk the timeline: 2013 Word2Vec → 2015 Bahdanau attention → 2017 Transformer → 2018 BERT/GPT → 2020 GPT-3 → 2022 ChatGPT.
- "Seven years from Word2Vec to ChatGPT. Pause at 2017: this single paper made everything before it obsolete."
- If beginners look confused: "Each dot built on the last — you will understand all of them today."
- If experts look bored: "2017-2020 architecture innovation; 2020-2023 pure compute scaling. What comes next?"

**Transition**: "Here is our roadmap for today..."

---

## Slide 4: Module 8 Roadmap

**Time**: ~1 min
**Talking points**:

- Walk through sections A through H plus Kailash synthesis. "Each section is a complete conceptual unit."
- If beginners look confused: "Think of it as layers — we start with text as characters and end with models that understand meaning."

**Transition**: "First, the bridge from deep learning to language..."

---

## Slide 5: The Bridge: DL → Language

**Time**: ~2 min
**Talking points**:

- Three new challenges: variable input length, long-range dependencies, order matters.
- "A sentence is not a fixed-size vector — it is a sequence. The order of words is information."
- If beginners look confused: "A table has rows and columns. A sentence has words in order. That order carries meaning."
- If experts look bored: "The fundamental challenge is encoding permutation variance — 'dog bites man' and 'man bites dog' must produce different representations."

**Transition**: "Let me show you a concrete example of why this matters in industry..."

---

## Slide 6: Case Study: BloombergGPT

**Time**: ~3 min
**Talking points**:

- Bloomberg trained a 50B parameter LLM on 363B tokens of financial text. On financial NLP benchmarks it outperforms GPT-NeoX (20B) by a wide margin despite similar size.
- Key lesson: domain-specific pre-training data matters more than raw parameter count at the same scale.
- If beginners look confused: "A bank trained an AI that speaks finance. It outperforms general AI on finance tasks because it learned from finance data."
- If experts look bored: "The FinPile data mix ratio is the interesting decision — 35% financial, 65% general corpus. Too much domain data causes catastrophic forgetting of general reasoning."

**Transition**: "Why do domain-specific models win so consistently?"

---

## Slide 7: Why Domain-Specific Models Win

**Time**: ~2 min
**Talking points**:

- Three reasons: vocabulary coverage (domain terms are OOV in general tokenisers), distribution shift, and task alignment.
- "You do not need to pre-train from scratch. Domain-specific fine-tuning of a general base model is usually sufficient."
- If beginners look confused: "A doctor trained on medical textbooks knows more about medicine than one trained on Wikipedia."
- If experts look bored: "The continual pre-training vs fine-tuning debate is live research — continual pre-training gives better low-data generalisation but is expensive."

**Transition**: "Here is what you will build in this module..."

---

## Slide 8: What You Will Build

**Time**: ~1 min
**Talking points**:

- Three exercises: (1) text preprocessing pipeline with kailash-ml, (2) TF-IDF document classifier, (3) transformer-based sentiment analysis with ModelVisualizer and AutoMLEngine.
- "By the end you have a production-ready NLP pipeline that runs on real Singapore data."
- If experts look bored: "Pay attention to how AutoMLEngine handles the text vectorisation step — it benchmarks TF-IDF vs BERT automatically."

**Transition**: "Let us start at the very beginning: what is text, really?"

---

## Slide 9: A. Text Preprocessing (Section Header)

**Time**: ~1 min
**Talking points**:

- "Garbage in, garbage out. In NLP, the preprocessing step is often more important than the model choice."
- Signal: this section has ~17 slides covering the full preprocessing pipeline.

**Transition**: "Let us look at the core challenge..."

---

## Slide 10: Text as Data: The Challenge

**Time**: ~2 min
**Talking points**:

- Text is unstructured, multilingual, noisy, context-dependent. "Apple in a tech article vs a recipe."
- Singapore data: Singlish, code-switching between English/Mandarin/Malay.
- If beginners look confused: "How different a WhatsApp message looks versus a legal contract — both are text but completely different."
- If experts look bored: "BPE tokenisers partially handle noisy text, but out-of-vocabulary compound words remain a challenge."

**Transition**: "Here is the standard pipeline for handling this..."

---

## Slide 11: The NLP Pipeline

**Time**: ~2 min
**Talking points**:

- Steps: raw text → tokenisation → normalisation → stop word removal → stemming/lemmatisation → feature extraction.
- "Classical NLP uses the full pipeline. Neural NLP often skips to tokenisation → embedding."
- kailash-ml: `PreprocessingPipeline` implements this as composable stages.
- If beginners look confused: "These are cleaning steps before cooking. You do not cook raw, dirty vegetables."

**Transition**: "The first and most fundamental step: tokenisation..."

---

## Slide 12: Tokenisation: Word-Level

**Time**: ~2 min
**Talking points**:

- Split on whitespace and punctuation. Problem: vocabulary explosion, OOV for unseen words.
- "English has 170,000 words. With proper nouns and domain terms, real corpora have millions of unique tokens."
- If beginners look confused: "Word tokenisation treats each word as a separate item in a list."
- If experts look bored: "Vocabulary-quality tradeoff: larger vocab reduces OOV but exponentially increases embedding matrix size. Sweet spot: 30-50K."

**Transition**: "What about going to the other extreme: characters?"

---

## Slide 13: Tokenisation: Character-Level

**Time**: ~1 min
**Talking points**:

- Vocabulary ~100-300 characters. Zero OOV. But sequences 5-10x longer — attention is O(n²).
- If beginners look confused: "Instead of treating each word as a unit, we treat each letter as a unit."
- If experts look bored: "Character models can generate misspellings naturally — useful for data augmentation and adversarial robustness testing."

**Transition**: "The best of both worlds is subword tokenisation..."

---

## Slide 14: Subword Tokenisation: The Middle Ground

**Time**: ~3 min
**Talking points**:

- Common words are full tokens; rare words are split into subword pieces. "unhappiness" → ["un", "happiness"].
- Balances vocabulary size (~30K) with sequence length and OOV handling.
- Three major algorithms: BPE (GPT family), WordPiece (BERT), Unigram (SentencePiece/T5).
- If beginners look confused: "Handling words you do not know by breaking them into parts you do know."
- If experts look bored: "GPT-style BPE handles code better; BERT WordPiece handles morphologically rich languages better."

**Transition**: "Let us dig into BPE, the most widely used algorithm..."

---

## Slide 15: BPE Algorithm: Step by Step

**Time**: ~3 min
**Talking points**:

- BPE = Byte Pair Encoding. Start with characters. Iteratively merge the most frequent adjacent pair until vocabulary target is reached.
- "A greedy compression algorithm. It finds the most efficient encoding of your training corpus."
- Used in: GPT-2, GPT-3, GPT-4, Llama, Falcon, Mistral — essentially all decoder-only models.
- If beginners look confused: "BPE finds common letter combinations and treats them as single units, like 'th' or 'ing'."
- If experts look bored: "Byte-level BPE (as in GPT-2) avoids the Unicode pre-tokenisation problem and handles any text losslessly."

**Transition**: "Let us trace through a concrete merge example..."

---

## Slide 16: BPE: Detailed Merge Example

**Time**: ~2 min
**Talking points**:

- Trace through step by step. "The final merge table is the tokeniser. Given any new text, you apply merges in learned order."
- Key point: the order of merges is fixed at training time — inference just replays the table.
- If beginners look confused: "This is the dictionary the tokeniser uses. It was learned from data, not hand-crafted."

**Transition**: "Two variants you will encounter in practice: WordPiece and Unigram..."

---

## Slide 17: WordPiece and Unigram

**Time**: ~2 min
**Talking points**:

- WordPiece (BERT): merges the pair that maximises language model likelihood, not raw frequency.
- Unigram (SentencePiece/T5): starts large, prunes tokens. Language-agnostic — handles Japanese, Chinese, Arabic without pre-tokenisation.
- If beginners look confused: "Different flavours of the same idea — finding the best way to split text into chunks."
- If experts look bored: "Unigram's probabilistic formulation gives multiple valid tokenisations per string — useful for data augmentation."

**Transition**: "Now let us look at the linguistic preprocessing steps..."

---

## Slide 18: Stop Words

**Time**: ~1 min
**Talking points**:

- High-frequency function words: the, a, is. Remove for BoW/TF-IDF. Keep for neural models — they carry syntactic information.
- CAUTION: keep negations ("not good") in sentiment analysis.
- If experts look bored: "Task-dependent: topic modelling → remove aggressively. NER → always keep."

**Transition**: "Related: stemming and lemmatisation..."

---

## Slide 19: Stemming

**Time**: ~1 min
**Talking points**:

- Heuristic suffix removal. "running" → "run", "happiness" → "happi". Fast but produces non-words.
- Porter Stemmer and Snowball are the two classical algorithms.
- If experts look bored: "'universe' and 'university' both stem to 'univers' — false positives in retrieval."

**Transition**: "The more linguistically correct alternative is lemmatisation..."

---

## Slide 20: Lemmatisation

**Time**: ~1 min
**Talking points**:

- Maps to dictionary form using morphological analysis. "better" → "good", "mice" → "mouse".
- Slower than stemming but produces real words. In kailash-ml: `PreprocessingPipeline(lemmatize=True)`.
- If experts look bored: "Language-specific lemmatisers vary significantly in quality — always benchmark for non-English NLP."

**Transition**: "Now let us look at n-grams, which capture multi-word phrases..."

---

## Slide 21: N-grams

**Time**: ~2 min
**Talking points**:

- Bigrams: "New York", "machine learning". N-grams partially restore word order lost in BoW.
- Trade-off: bigrams square the vocabulary size; trigrams cube it. Use sparingly.
- If beginners look confused: "'New York' as a bigram is a single concept, not two separate words."
- If experts look bored: "The Kneser-Ney smoothed n-gram LM was state of the art until 2013 — understanding it helps you understand what neural LMs improved on."

**Transition**: "Text normalisation is the housekeeping step..."

---

## Slide 22: Text Normalisation

**Time**: ~1 min
**Talking points**:

- Lowercasing, punctuation removal, URL/email stripping, unicode normalisation.
- "Lowercasing is not always right — 'Apple' the company vs 'apple' the fruit."
- If experts look bored: "Always apply NFC normalisation — the same visually identical character can have two different byte representations."

**Transition**: "A key tool for normalisation is regular expressions..."

---

## Slide 23: Regex Patterns for NLP

**Time**: ~2 min
**Talking points**:

- Key patterns: URL removal `https?://\S+`, email removal `\S+@\S+`, number normalisation `\d+` → `NUM`.
- kailash-ml: `PreprocessingPipeline(custom_regex=[(pattern, replacement)])`.
- If beginners look confused: "Regex is a mini-language for describing text patterns — like a very precise search."
- If experts look bored: "Python's `re` vs Rust-backed `fancy-regex` — throughput differences can be 10x at scale."

**Transition**: "Let us consolidate everything into a pipeline summary..."

---

## Slide 24: Preprocessing Pipeline Summary

**Time**: ~2 min
**Talking points**:

- Full pipeline diagram: raw text → tokenise → lowercase → strip noise → remove stops → lemmatise → n-grams → output.
- "For transformer models, you stop after tokenisation. For classical models, you run the full pipeline."
- `PreprocessingPipeline(tokenizer="wordpiece", lowercase=True, remove_stops=True, lemmatize=True).fit_transform(texts)`.

**Transition**: "Let us look at where people go wrong..."

---

## Slide 25: Preprocessing: Common Mistakes

**Time**: ~2 min
**Talking points**:

- Five mistakes: (1) applying full classical pipeline to transformer input, (2) lowercasing named entities, (3) stripping meaningful punctuation, (4) removing numbers in financial text, (5) applying English stop words to multilingual corpus.
- "Every mistake here silently degrades model performance — no error, just worse results."
- If experts look bored: "The multilingual stop word mistake is particularly pernicious — 'not' in French is 'ne...pas', split across the clause."

**Transition**: "Three key takeaways from Section A..."

---

## Slide 26: Section A: Key Takeaways

**Time**: ~1 min
**Talking points**:

- Three points: (1) tokenisation choice determines vocabulary size and OOV handling; (2) subword tokenisation (BPE/WordPiece) is the modern standard; (3) classical preprocessing steps are skipped for neural models.
- Quick check: "What tokeniser does BERT use? GPT?" (WordPiece vs BPE.)

**Transition**: "Now let us look at the classical representation: Bag of Words and TF-IDF..."

---

## Slide 27: B. Bag of Words & TF-IDF (Section Header)

**Time**: ~1 min
**Talking points**:

- "BoW and TF-IDF still power production search engines and spam filters today. Fast, interpretable, surprisingly effective."
- If experts look bored: "Elasticsearch's relevance scoring is BM25 — understanding it is required for search system design."

**Transition**: "The core idea of Bag of Words..."

---

## Slide 28: Bag of Words: The Idea

**Time**: ~2 min
**Talking points**:

- Every document becomes a vector of word counts. "The cat sat on the mat" → count vector indexed by vocabulary.
- "Dog bites man" and "man bites dog" have identical BoW representations — order is lost.
- If beginners look confused: "Count every word in a document and write the counts in a table. That table IS the representation."
- If experts look bored: "BoW cannot distinguish 'the drug prevented cancer' from 'the drug caused cancer'. This is why it fails for sentiment and negation."

**Transition**: "What are the specific limitations?"

---

## Slide 29: BoW: Limitations

**Time**: ~2 min
**Talking points**:

- Four limitations: order lost, high dimensionality (50K-100K), sparsity, no semantic similarity ("car" and "automobile" are unrelated vectors).
- If beginners look confused: "BoW is like judging a book by its index. You can see the topics but not the story."
- If experts look bored: "Sparsity is actually a feature for certain algorithms — sparse linear models (SGD classifier, SVM) are extremely fast and competitive on short text classification."

**Transition**: "TF-IDF addresses the weighting problem..."

---

## Slide 30: TF-IDF: The Formula

**Time**: ~3 min
**Talking points**:

- TF = term frequency in document d. IDF = log(N / df(w)) where N = corpus size, df = document frequency.
- TF-IDF(w, d) = TF(w, d) × IDF(w).
- "A word that appears often in one document but rarely in others is highly informative."
- If beginners look confused: "TF rewards local frequency. IDF penalises globally common words. Multiply to get importance."
- If experts look bored: "Smooth IDF adds 1 to denominator to avoid division by zero. The log is critical — without it, rare words dominate overwhelmingly."

**Transition**: "Let us work through a concrete example..."

---

## Slide 31: TF-IDF: Worked Example

**Time**: ~3 min
**Talking points**:

- Walk through the table step by step: TF for one document, IDF across corpus, then the product.
- "The word 'Singapore' has high TF in a Singapore-specific article but high IDF because it appears in fewer documents."
- Pause and ask: "Which words do you expect to get high TF-IDF scores in a financial document?"
- If experts look bored: "This demonstrates the saturation problem — a word appearing 100 times gets 10x the score of one appearing 10 times. That is what sublinear TF addresses."

**Transition**: "That saturation problem leads us to sublinear TF..."

---

## Slide 32: Sublinear TF

**Time**: ~1 min
**Talking points**:

- Replace raw TF with 1 + log(TF). A word appearing 100 times gets 1 + log(100) = 5.6, not 100.
- In kailash-ml: `AutoMLEngine(text_params={"sublinear_tf": True})`.
- If experts look bored: "Sublinear TF brings TF-IDF closer to BM25 saturation behaviour."

**Transition**: "Speaking of search engines: BM25 is the production standard..."

---

## Slide 33: BM25: The Search Engine Standard

**Time**: ~3 min
**Talking points**:

- BM25 adds two improvements: (1) saturation via k1 parameter; (2) length normalisation via b parameter.
- Used in: Elasticsearch, Solr, Lucene. "Every search engine you have used in the last 20 years uses a variant of BM25."
- Parameters: k1 typically 1.2-2.0, b typically 0.75.
- If beginners look confused: "BM25 is TF-IDF with two fixes: word repetition has diminishing returns, and long documents are not unfairly boosted."
- If experts look bored: "BM25+ (Lü & Callan, 2011) addresses a weakness where a document with zero occurrences can still outrank one with one occurrence due to length normalisation."

**Transition**: "Let us visualise the saturation difference..."

---

## Slide 34: BM25 vs TF-IDF: Saturation

**Time**: ~1 min
**Talking points**:

- TF-IDF grows linearly with term frequency. BM25 plateaus.
- "At TF = 20, BM25 is essentially at its maximum. TF-IDF continues climbing indefinitely."
- If experts look bored: "k1 controls where the plateau occurs. Setting k1 = 0 degrades to pure IDF."

**Transition**: "How do we use BoW and TF-IDF in practice?"

---

## Slide 35: BoW/TF-IDF in Practice

**Time**: ~2 min
**Talking points**:

- `AutoMLEngine(task="text_classification", vectorizer="tfidf")` handles tokenisation, vocabulary building, TF-IDF matrix, model selection.
- "Start with TF-IDF + logistic regression. Strong baseline that trains in seconds on millions of documents."
- If experts look bored: "TF-IDF + SGD classifier on hashing trick vectorisation handles hundreds of millions of documents in memory-efficient streaming mode."

**Transition**: "When should you still use BoW instead of transformers?"

---

## Slide 36: When BoW Still Wins

**Time**: ~2 min
**Talking points**:

- BoW wins when: inference speed matters (TF-IDF is microseconds, transformers 50-500ms), interpretability required, very small training data (<100 examples), or short formulaic text.
- "Do not reach for BERT when TF-IDF is faster, interpretable, and achieves 95% of the performance."
- If experts look bored: "A 100x inference speedup means you can use 100x larger batch sizes with the same infrastructure."

**Transition**: "Key takeaways from Section B..."

---

## Slide 37: Section B: Key Takeaways

**Time**: ~1 min
**Talking points**:

- Three points: (1) TF-IDF rewards rare but locally frequent terms; (2) BM25 adds saturation and length normalisation; (3) BoW/TF-IDF remains competitive for high-speed, interpretable, small-data scenarios.
- Check: "What does a high IDF score mean for a word?" (Rare across documents — carries distinctive information.)

**Transition**: "Section C: from sparse counts to dense meaning — word embeddings..."

---

## Slide 38: C. Word Embeddings (Section Header)

**Time**: ~1 min
**Talking points**:

- "The breakthrough idea: represent words as points in geometric space, where similar words are nearby."
- Section C covers: distributional hypothesis, Word2Vec, GloVe, FastText, contextual embeddings with ELMo.
- If experts look bored: "We will derive the skip-gram objective and analyse negative sampling, plus the connection between PMI and GloVe."

**Transition**: "The foundation is the distributional hypothesis..."

---

## Slide 39: The Distributional Hypothesis

**Time**: ~2 min
**Talking points**:

- Firth (1957): "You shall know a word by the company it keeps."
- "bank" appears near "money", "loan", "deposit" — similar to "lender" and "creditor". Meaning emerges from statistics.
- If beginners look confused: "If two words are always used in the same kinds of sentences, they probably mean something similar."
- If experts look bored: "The distributional hypothesis is the theoretical foundation for both BoW (document-level co-occurrence) and Word2Vec (window-level). The difference is the window size."

**Transition**: "How do we go from sparse to dense representations?"

---

## Slide 40: From Sparse to Dense

**Time**: ~2 min
**Talking points**:

- BoW: 50,000-dimensional, mostly zeros. Word2Vec: 100-300 dimensional, all non-zero, geometrically meaningful.
- "BoW is a passport photo list. Embeddings are a map where similar people are placed near each other."
- If experts look bored: "Levy & Goldberg (2014) showed that Word2Vec implicitly factorises a shifted PMI matrix — BoW, GloVe, and Word2Vec are all variants of the same underlying idea."

**Transition**: "Let us look at the most influential algorithm: Word2Vec skip-gram..."

---

## Slide 41: Word2Vec: Skip-gram

**Time**: ~3 min
**Talking points**:

- Skip-gram: given a centre word, predict surrounding context words within a window.
- Training is self-supervised: window slides over corpus, generating (centre, context) pairs automatically.
- If beginners look confused: "Given 'cat', the model learns to predict that 'the', 'sat', 'on' are likely nearby."
- If experts look bored: "Skip-gram outperforms CBOW on rare words because predicting multiple context outputs creates stronger gradient signal for infrequent centre words."

**Transition**: "Let us look at the objective function formally..."

---

## Slide 42: Skip-gram: The Objective

**Time**: ~2 min
**Talking points**:

- Maximise average log probability of context words given centre word across the corpus.
- The softmax denominator sums over the entire vocabulary — O(V) per step. With 50K vocabulary, this is the scaling bottleneck.
- "The solution is negative sampling."
- If experts look bored: "Hierarchical softmax (Huffman-coded binary tree) gives O(log V) instead of O(V). Negative sampling is conceptually cleaner."

**Transition**: "Negative sampling makes Word2Vec tractable..."

---

## Slide 43: Negative Sampling

**Time**: ~2 min
**Talking points**:

- Binary classification: is this (word, context) pair real or random? Sample k random negatives.
- k = 5-20 for small corpora, 2-5 for large. Converts O(V) to O(k) per step — 1000x speedup for 50K vocabulary.
- If beginners look confused: "Instead of testing all 50,000 words, we test 5 random wrong words vs the 1 correct one."
- If experts look bored: "The negative sampling distribution is raised to the 3/4 power — more samples to medium-frequency words."

**Transition**: "The other Word2Vec variant: CBOW..."

---

## Slide 44: Word2Vec: CBOW

**Time**: ~1 min
**Talking points**:

- CBOW: given context words, predict centre word. Faster to train, better for frequent words.
- Skip-gram preferred for rare word quality; CBOW for fast training at scale.
- If experts look bored: "CBOW's averaging discards positional context information — why skip-gram generalises better to compositional phrases."

**Transition**: "GloVe offers a different approach with global statistics..."

---

## Slide 45: GloVe: Global Vectors

**Time**: ~2 min
**Talking points**:

- GloVe (Pennington et al., 2014): explicitly factorises the global word co-occurrence matrix. Dot product of two word vectors ≈ log of their co-occurrence count.
- Uses weighted least squares — frequent pairs get more weight but not unbounded weight.
- If beginners look confused: "GloVe learns embeddings by looking at how often every pair of words appears together across the whole corpus."
- If experts look bored: "GloVe's weighting function f(x) = (x/x_max)^alpha prevents dominant co-occurrences. Optimal alpha = 0.75 — same value used in Word2Vec negative sampling."

**Transition**: "FastText handles an important limitation: morphology..."

---

## Slide 46: FastText: Subword Embeddings

**Time**: ~2 min
**Talking points**:

- Problem: Word2Vec and GloVe have no representation for OOV words or morphological variants.
- FastText (Bojanowski et al., 2017): word vector = sum of character n-gram vectors. OOV words are representable from n-grams.
- If beginners look confused: "FastText builds word vectors from letter combinations, so it can handle words it has never seen before."
- If experts look bored: "FastText is still the best pre-trained option for low-resource languages with high morphological richness. Facebook pre-trained vectors cover 157 languages."

**Transition**: "The most famous property: vector arithmetic..."

---

## Slide 47: The Magic of Vector Arithmetic

**Time**: ~3 min
**Talking points**:

- vec("king") - vec("man") + vec("woman") ≈ vec("queen"). Not programmed — emerges from text statistics.
- Other analogies: Paris:France :: Berlin:Germany; doctor:hospital :: teacher:school.
- Singapore-relevant: "Can you guess what Orchard:Singapore :: Champs-Elysees:\_\_\_ gives?"
- If beginners look confused: "Word vectors have captured real-world relationships purely by reading text."
- If experts look bored: "Word2Vec learns a 'royalty' direction approximately orthogonal to the 'gender' direction. This decomposability is a central property of linear representation learning."

**Transition**: "How do we measure similarity between word vectors?"

---

## Slide 48: Embedding Similarity: Cosine

**Time**: ~1 min
**Talking points**:

- Cosine similarity = dot product / (|a| × |b|). Direction matters, magnitude does not.
- Used for: nearest-neighbour retrieval, document similarity, semantic search.
- If experts look bored: "For ANN at scale, FAISS with HNSW indexing gives sub-millisecond lookup at billion-vector scale — the infrastructure behind most semantic search systems."

**Transition**: "A critical issue: word embeddings encode societal biases..."

---

## Slide 49: Bias in Word Embeddings

**Time**: ~3 min
**Talking points**:

- Word2Vec on Google News: man:doctor :: woman:nurse, man:programmer :: woman:homemaker.
- "The model is a statistical mirror of the data. If the data is biased, the model is biased."
- Production consequence: a CV screening system using biased embeddings will systematically downrank qualified women.
- If beginners look confused: "The AI learned from human-written text that had biases in it — so it learned those biases."
- If experts look bored: "Bolukbasi et al. (2016) showed gender is encoded on a single principal direction. Hard debiasing projects it out. Gonen & Goldberg (2019) showed male-biased words still cluster together after debiasing."

**Transition**: "What can we do about it?"

---

## Slide 50: Debiasing Approaches

**Time**: ~2 min
**Talking points**:

- Three approaches: hard debiasing (project out bias subspace), data augmentation (gender-swapped examples), counterfactual data substitution.
- "None is a complete solution. Bias mitigation is active research."
- Production guidance: audit embeddings before deployment for hiring, lending, or criminal justice applications.
- If experts look bored: "WEAT measures bias but debiased models can still exhibit bias on downstream tasks without WEAT flagging it."

**Transition**: "The limitations of static embeddings led to ELMo: contextual embeddings..."

---

## Slide 51: ELMo: Contextual Embeddings

**Time**: ~3 min
**Talking points**:

- Static: "bank" has one vector regardless of context. ELMo: representation differs in "river bank" vs "bank account".
- ELMo uses a bidirectional LSTM language model. Embedding = concatenation of hidden states across layers.
- "Context-dependence is the key insight that BERT and all transformers build on."
- If beginners look confused: "ELMo gives each word a different vector depending on the surrounding words."
- If experts look bored: "ELMo's layer weighting is trainable per task — optimal combination of shallow (syntax) vs deep (semantics) layers differs by task. This insight was preserved in BERT's [CLS] design."

**Transition**: "Let us look at what individual dimensions encode..."

---

## Slide 52: Embedding Dimensions: What They Encode

**Time**: ~1 min
**Talking points**:

- Probing classifiers find dimensions encoding: gender, tense, plurality, sentiment polarity, concreteness.
- "The model did not know these categories existed. It discovered them because they are statistically useful."
- If experts look bored: "Linear probes test if a property is linearly decodable from a representation — the primary interpretability tool for embeddings."

**Transition**: "Practical details for training Word2Vec..."

---

## Slide 53: Training Word2Vec: Practical Details

**Time**: ~2 min
**Talking points**:

- Key hyperparameters: window size (5 for syntax, 10+ for semantics), dim (100-300), negative samples (5-20).
- kailash-ml: `AutoMLEngine(task="text_embedding", algo="word2vec", dim=200, window=5).fit(corpus)`.
- If experts look bored: "Subsampling of frequent words effectively increases context window for rare words by removing high-frequency noise."

**Transition**: "A closer look at GloVe's co-occurrence ratio intuition..."

---

## Slide 54: GloVe: Co-occurrence Ratios

**Time**: ~1 min
**Talking points**:

- GloVe is motivated by ratios of co-occurrence probabilities. P("ice" | "solid") / P("steam" | "solid") is high.
- "GloVe encodes the relationship between concepts by looking at how their contextual patterns differ."
- If experts look bored: "The log-bilinear model structure means GloVe embeddings can be added and subtracted linearly — mathematical reason the king-queen analogy works."

**Transition**: "How do we evaluate embedding quality?"

---

## Slide 55: Evaluating Embeddings

**Time**: ~2 min
**Talking points**:

- Intrinsic: word analogy tasks, word similarity (SimLex-999, WordSim-353).
- Extrinsic: plug into NER, sentiment, QA and measure task performance.
- "Intrinsic scores often do not predict extrinsic performance — always evaluate on your target task."
- If experts look bored: "SimLex-999 correlation doesn't generalise across domains — always benchmark on domain-relevant analogical reasoning."

**Transition**: "How to use pre-trained embeddings in practice..."

---

## Slide 56: Pre-trained Embeddings: When and How

**Time**: ~2 min
**Talking points**:

- Three strategies: frozen (feature extraction), fine-tuned (better adaptation), domain-adaptive pre-training.
- Rule of thumb: small labelled dataset → freeze. Large → fine-tune. Domain-specific vocabulary → domain pre-training.
- kailash-ml: `TrainingPipeline(embedding="glove-300d", freeze_embeddings=True)`.
- If experts look bored: "ULMFiT (Howard & Ruder, 2018) introduced discriminative fine-tuning and slanted triangular LR — these techniques directly preceded BERT fine-tuning."

**Transition**: "Section C key takeaways..."

---

## Slide 57: Section C: Key Takeaways

**Time**: ~1 min
**Talking points**:

- Three points: (1) embeddings place similar words nearby — meaning emerges from co-occurrence statistics; (2) Word2Vec/GloVe are static; ELMo/transformers are contextual; (3) embeddings inherit societal biases — audit before deployment.
- Check: "What is the distributional hypothesis?" (Words in similar contexts have similar meanings.)

**Transition**: "Section D: Recurrent Neural Networks — the first serious sequence models..."

---

## Slide 58: D. Recurrent Neural Networks (Section Header)

**Time**: ~1 min
**Talking points**:

- "Before transformers, RNNs were the dominant sequence model. Understanding them is required to appreciate why attention replaced them."
- Section D: vanilla RNN, BPTT, vanishing gradient, LSTM, GRU, bidirectional, seq2seq.
- If experts look bored: "We will derive the LSTM gates from the vanishing gradient problem, not just state the equations."

**Transition**: "Let us start with the basic RNN equation..."

---

## Slide 59: Vanilla RNN: The Equation

**Time**: ~3 min
**Talking points**:

- Hidden state: h*t = tanh(W_h × h*{t-1} + W_x × x_t + b). The hidden state is a running summary of all inputs seen so far.
- "The hidden state is the RNN's memory. Every step it reads the new word and updates its memory."
- Walk through: h_0 = zeros → h_1 after "The" → h_2 after "cat" → h_3 after "sat".
- If beginners look confused: "The RNN reads words one at a time. After each word it updates a memory vector."
- If experts look bored: "tanh bounds activations to [-1, 1] — chosen to handle recurrent multiplication, but it is the root of the vanishing gradient problem."

**Transition**: "RNNs can be configured in different architectures..."

---

## Slide 60: RNN Types

**Time**: ~2 min
**Talking points**:

- Five types: one-to-one, one-to-many (image captioning), many-to-one (sentiment), many-to-many aligned (POS), many-to-many unaligned (translation).
- "The architecture is a design decision based on the task structure."
- If beginners look confused: "Sentiment classification is many-to-one: read many words, output one sentiment score."
- If experts look bored: "Many-to-many unaligned (seq2seq) requires the encoder-decoder split — output can have different length than input."

**Transition**: "How do we train RNNs?"

---

## Slide 61: Backpropagation Through Time (BPTT)

**Time**: ~3 min
**Talking points**:

- BPTT unrolls the RNN across time steps. For a 100-word sentence, gradients flow through 100 multiplication steps.
- "If eigenvalues > 1: gradients explode. If eigenvalues < 1: gradients vanish."
- Truncated BPTT: only backpropagate through k steps. Trades sequence length for stability.
- If beginners look confused: "Training on a long sentence means the gradient has to travel backwards through every word."
- If experts look bored: "BPTT memory requirement is O(seq length × hidden size) — at inference, RNNs are O(1) memory per step. The fundamental tradeoff versus transformers."

**Transition**: "The vanishing gradient is the critical failure mode..."

---

## Slide 62: The Vanishing Gradient Problem

**Time**: ~3 min
**Talking points**:

- tanh derivative is at most 0.25. After 50 steps: 0.25^50 ≈ 10^{-31}. Effectively zero.
- "The RNN cannot learn that 'bank' at position 1 affects the meaning of 'loan' at position 50."
- If beginners look confused: "The RNN 'forgets' words from the beginning of a long sentence because the training signal cannot reach that far back."
- If experts look bored: "Vanishing gradients mean you are optimising in a region with essentially flat loss surface — saddle points, not just local minima."

**Transition**: "LSTM was designed specifically to solve this..."

---

## Slide 63: LSTM: Long Short-Term Memory

**Time**: ~3 min
**Talking points**:

- Hochreiter & Schmidhuber (1997). Key innovation: separate cell state c_t with additive updates, not multiplicative.
- "The cell state is a highway for gradient flow."
- Three gates: forget (erase), input (write), output (read).
- If beginners look confused: "LSTM has two memory systems. The gates decide what to keep, write, and output."
- If experts look bored: "LSTM's gradient stability is guaranteed by the additive cell state update — the same design principle that transformers later adopted with residual connections."

**Transition**: "Let us understand each gate in detail..."

---

## Slide 64: LSTM: Forget Gate

**Time**: ~1 min
**Talking points**:

- f*t = sigmoid(W_f × [h*{t-1}, x_t] + b_f). f_t = 0: forget. f_t = 1: remember.
- Example: when you read a period, the forget gate fires to clear sentence-level state.
- If experts look bored: "The forget gate was added by Gers et al. (2000) — it was not in the original Hochreiter & Schmidhuber LSTM."

**Transition**: "What does the LSTM write?"

---

## Slide 65: LSTM: Input Gate & Candidate

**Time**: ~1 min
**Talking points**:

- Input gate: i_t — what to write. Candidate g_t — what content to write.
- "The input gate asks 'should we write?' The candidate asks 'what should we write?'"
- If experts look bored: "The separation of gating (sigmoid) and content (tanh) is the key modularity — same pattern appears in multiplicative attention."

**Transition**: "Now we combine forget and input to update the cell state..."

---

## Slide 66: LSTM: Cell State Update

**Time**: ~1 min
**Talking points**:

- c*t = f_t ⊙ c*{t-1} + i_t ⊙ g_t. "Updated additively — this is the gradient highway."
- If experts look bored: "The gradient of c\_{t-k} w.r.t. c_t is the product of forget gates, not the full weight matrix. This is the fundamental difference from vanilla RNN."

**Transition**: "Finally, the output..."

---

## Slide 67: LSTM: Output Gate & Hidden State

**Time**: ~1 min
**Talking points**:

- o*t = sigmoid(W_o × [h*{t-1}, x_t] + b_o). h_t = o_t ⊙ tanh(c_t).
- "The LSTM can remember something without outputting it — the output gate controls visibility."
- If experts look bored: "h_t is 'short-term memory' vs c_t 'long-term memory'. Even LSTMs struggle beyond ~500 tokens."

**Transition**: "Let us see all four equations together..."

---

## Slide 68: LSTM: Complete Equations

**Time**: ~2 min
**Talking points**:

- All four equations side by side. Same input structure, different weight matrices. LSTM has 4x the parameters of a vanilla RNN.
- Optimisation: concatenate all four gate computations into one 4x GEMM then split — what every GPU-optimised implementation does.
- If beginners look confused: "The key equation is the cell state update — the rest are controllers for it."

**Transition**: "GRU is a streamlined variant..."

---

## Slide 69: GRU: Gated Recurrent Unit

**Time**: ~2 min
**Talking points**:

- GRU (Cho et al., 2014): merges cell state and hidden state. Two gates: reset and update.
- "GRU is LSTM minus the output gate and cell state split. 33% fewer parameters, similar performance on most tasks."
- If beginners look confused: "GRU is a simplified LSTM that is faster to train."
- If experts look bored: "Chung et al. (2014) and Greff et al. (2017) show LSTM/GRU parity. GRU preferred when training speed matters."

**Transition**: "For sequence labelling tasks, we often need both past and future context..."

---

## Slide 70: Bidirectional RNNs

**Time**: ~1 min
**Talking points**:

- Run two RNNs: left-to-right and right-to-left. Concatenate hidden states.
- "The word 'restoration' to the right of 'bank' tells you it is a river bank. Bidirectional models can use this."
- If experts look bored: "Bidirectional inference is incompatible with autoregressive generation — GPT is unidirectional, BERT is bidirectional."

**Transition**: "We can also stack multiple layers..."

---

## Slide 71: Multi-Layer and Deep RNNs

**Time**: ~1 min
**Talking points**:

- Stack 2-4 layers. Lower layers capture local syntax; higher layers capture semantics.
- If experts look bored: "Beyond 4 layers, gradients in the depth direction also begin to vanish."

**Transition**: "The most important RNN architecture: seq2seq encoder-decoder..."

---

## Slide 72: Seq2Seq: Encoder-Decoder

**Time**: ~3 min
**Talking points**:

- Encoder: process input into a fixed-size context vector. Decoder: autoregressively generate output conditioned on context.
- Application: machine translation. "How are you?" → context vector → "Comment allez-vous?"
- The bottleneck: "Translate a 100-word paragraph into a single 512-number summary — information is lost."
- If beginners look confused: "Seq2seq reads the whole input and encodes it into a 'meaning vector', then the decoder writes the output."
- If experts look bored: "Bahdanau (2015) solved the fixed-size bottleneck with attention — instead of one context vector, the decoder attends to all encoder hidden states dynamically."

**Transition**: "Let us see RNNs in action on sentiment analysis..."

---

## Slide 73: RNNs in Practice: Sentiment Analysis

**Time**: ~2 min
**Talking points**:

- `TrainingPipeline(model_type="lstm", embedding="glove-300d", bidirectional=True, layers=2)`
- Typical performance: 85-88% accuracy on SST-2 with BiLSTM. BERT: 93%+. "You can see why transformers won."
- If experts look bored: "TrainingPipeline handles pack_padded_sequence — variable-length batching that avoids padding waste and gives correct BPTT gradients."

**Transition**: "A training trick that dramatically improves RNN convergence: teacher forcing..."

---

## Slide 74: Teacher Forcing

**Time**: ~1 min
**Talking points**:

- During training: use ground truth token as input at each decoder step, not the model's prediction.
- Faster convergence but exposure bias at inference. Scheduled sampling mixes both.
- If experts look bored: "In practice, most production seq2seq systems use pure teacher forcing despite the theory — improvement from scheduled sampling is task-dependent."

**Transition**: "Gradient clipping prevents the other gradient problem: explosion..."

---

## Slide 75: Gradient Clipping for RNNs

**Time**: ~1 min
**Talking points**:

- If gradient norm > threshold, scale it down. "If norm > 5, clip to 5. Simple but highly effective."
- kailash-ml TrainingPipeline applies it automatically (default max_norm=5.0).
- If experts look bored: "For LSTMs on text, 1.0-5.0 works well; for RNNs on raw waveforms, 0.1-1.0 is typical."

**Transition**: "Let us consolidate the limitations before moving to attention..."

---

## Slide 76: RNN Limitations Summary

**Time**: ~2 min
**Talking points**:

- Four limitations: (1) sequential processing — cannot parallelise; (2) long-range dependencies — LSTM struggles beyond 500 tokens; (3) fixed bottleneck in seq2seq; (4) training instability.
- "All four of these limitations are solved by the transformer. That is why we moved on."
- If experts look bored: "LSTM training on 1M tokens is 10-100x slower than a transformer — every time step depends on the previous one. This bottleneck prevented scaling."

**Transition**: "Section D key takeaways..."

---

## Slide 77: Section D: Key Takeaways

**Time**: ~1 min
**Talking points**:

- Three points: (1) RNNs process sequences with a hidden state but suffer from vanishing gradients; (2) LSTM solves this with gated additive cell state; (3) seq2seq enables variable-length I/O but the fixed bottleneck limits quality.
- Check: "What makes LSTM better than vanilla RNN?" (Additive cell state update — gradients do not vanish.)

**Transition**: "Section E: attention — the breakthrough that made transformers possible..."

---

## Slide 78: E. Attention Mechanisms (Section Header)

**Time**: ~1 min
**Talking points**:

- "Attention is the single most important idea in modern NLP. Everything that follows — BERT, GPT, every LLM — is built on it."
- If experts look bored: "We will derive scaled dot-product attention from the QKV dictionary lookup analogy and prove why scaling by sqrt(d_k) is necessary."

**Transition**: "The original attention paper: Bahdanau et al. (2015)..."

---

## Slide 79: Bahdanau Attention (2015)

**Time**: ~3 min
**Talking points**:

- Let the decoder look back at all encoder hidden states instead of one context vector.
- Compute alignment scores for each encoder position → softmax → weighted sum = context vector.
- "At each step, the decoder asks: which encoder words should I focus on right now?"
- Visualising attention weights revealed interpretable alignment patterns.
- If beginners look confused: "Instead of reading one summary of the input, the decoder can look back at any part of the input at each step."
- If experts look bored: "Bahdanau used an MLP as the alignment function. Luong (2015) simplified to dot product — the foundation of scaled dot-product attention."

**Transition**: "What does attention look like visually?"

---

## Slide 80: Attention Visualised

**Time**: ~1 min
**Talking points**:

- Heatmap: rows = decoder words, columns = encoder words. Bright = high attention.
- "Notice the diagonal for word-by-word translation, off-diagonal for reordering."
- If experts look bored: "Jain & Wallace (2019) showed attention weights are not reliable explanations — high attention does not necessarily mean causal importance."

**Transition**: "Self-attention generalises this: let every word attend to every other word..."

---

## Slide 81: Self-Attention: Q, K, V

**Time**: ~4 min
**Talking points**:

- Q, K, V all come from the same input: Q = X W_Q, K = X W_K, V = X W_V.
- Q = "what am I looking for?", K = "what do I advertise?", V = "what do I share?"
- Attention output = softmax(Q K^T / sqrt(d_k)) V.
- "Every position simultaneously queries every other. Direct long-range dependencies in a single operation."
- If beginners look confused: "Self-attention lets each word look at all other words and decide how much to borrow from each one."
- If experts look bored: "The QKV decomposition is a generalisation of memory-addressing — Q selects location (key), V is memory content. Linear projections allow learning different patterns for different tasks."

**Transition**: "Let us write out the full formula..."

---

## Slide 82: Scaled Dot-Product Attention

**Time**: ~3 min
**Talking points**:

- Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.
- Walk step by step: (1) Q K^T → score matrix; (2) divide by sqrt(d_k); (3) softmax → attention weights; (4) V weighted sum → output.
- O(n² d) for sequence length n and dimension d.
- If beginners look confused: "Step 1: compute scores. Step 2: normalise. Step 3: mix values."
- If experts look bored: "The output is linear interpolation in value space. Because V is also learned, the model chooses what information to store in value space."

**Transition**: "Why scale by sqrt(d_k)?"

---

## Slide 83: Why Scale by sqrt(d_k)?

**Time**: ~2 min
**Talking points**:

- As d_k grows, dot products grow in variance. Without scaling, softmax saturates — all probability mass on one token.
- Hard attention: gradient through softmax is ~0 everywhere except top-1. Model stops learning.
- Dividing by sqrt(d_k) keeps variance constant at 1 regardless of d_k.
- If beginners look confused: "Without scaling, attention becomes winner-takes-all. Dividing by the square root keeps scores in a reasonable range."
- If experts look bored: "Same problem as vanishing gradient in deep networks — softmax saturation is a gradient highway collapse."

**Transition**: "Let us trace through a small worked example..."

---

## Slide 84: Self-Attention: Worked Example

**Time**: ~3 min
**Talking points**:

- Walk through the 3-token example numerically: Q, K, V matrices → Q K^T → scale → softmax → × V.
- "Do not just watch — trace your finger through each step."
- "Each output vector is a unique combination of all input vectors — this is how information mixes."
- If experts look bored: "In practice d_k = 64 per head for BERT-base (12 heads, d_model = 768). Numerically identical."

**Transition**: "Multi-head attention runs several attention operations in parallel..."

---

## Slide 85: Multi-Head Attention

**Time**: ~3 min
**Talking points**:

- h heads in parallel, each with different learned W_Q, W_K, W_V. Concatenate and project.
- "Different heads learn different types of relationships: syntax, coreference, positional proximity."
- BERT-base: 12 heads × 64-dim = 768-dim total. GPT-3: 96 heads × 128-dim = 12288-dim.
- If beginners look confused: "Multi-head is like 12 different attention patterns running simultaneously."
- If experts look bored: "Clark et al. (2019) found specific BERT heads correspond to specific syntactic relations — remarkable emergence without explicit syntactic supervision."

**Transition**: "Why is attention fundamentally better than RNN?"

---

## Slide 86: Attention vs RNN: The Key Advantages

**Time**: ~2 min
**Talking points**:

- Four advantages: parallelism, O(1) path length, no fixed bottleneck, interpretable weights.
- "In an RNN, information from position 1 must travel through 99 multiplications to reach position 100. In attention, it arrives in one step."
- If experts look bored: "The O(n²) complexity of attention is its weakness at long sequences. Linear attention variants (Performer, Reformer) approximate softmax kernel to O(n)."

**Transition**: "Self-attention versus cross-attention..."

---

## Slide 87: Cross-Attention vs Self-Attention

**Time**: ~1 min
**Talking points**:

- Self-attention: Q, K, V from same sequence. Cross-attention: Q from decoder, K/V from encoder.
- "Cross-attention is the modern implementation of Bahdanau attention inside the transformer."
- If experts look bored: "Three attention types in the full transformer: encoder self-attention (bidirectional), decoder self-attention (causal), encoder-decoder cross-attention."

**Transition**: "A taxonomy of attention variants..."

---

## Slide 88: Attention Types: A Taxonomy

**Time**: ~1 min
**Talking points**:

- Walk through: additive (Bahdanau), dot-product (Luong), scaled dot-product (Vaswani), multi-head, sparse (Longformer), local, global.
- "Most tasks: scaled dot-product multi-head. Long documents: sparse or local."
- If experts look bored: "Linear attention rewrites softmax as a kernel product — O(n) complexity at the cost of approximation quality."

**Transition**: "An elegant way to think about attention: soft dictionary lookup..."

---

## Slide 89: Attention as Soft Dictionary Lookup

**Time**: ~2 min
**Talking points**:

- Hard dictionary: exact key match returns exact value. Soft dictionary: partial matches return a weighted average.
- "You can think of it as a differentiable database query: Q is the query, K is the index, V is the stored data."
- If beginners look confused: "Fuzzy search — instead of finding the exact match, you blend multiple partial matches."
- If experts look bored: "Ramsauer et al. (2020) showed modern Hopfield networks converge to the attention update rule — connecting energy-based memory to transformers."

**Transition**: "How does the computational cost scale?"

---

## Slide 90: Attention Complexity Analysis

**Time**: ~2 min
**Talking points**:

- Time: O(n² d). Memory: O(n²) for the attention matrix. For n = 1024: 1M floats × 4 bytes = 4 MB per layer per batch element.
- "This is why Flash Attention (Section F) is so important."
- If experts look bored: "The bottleneck is HBM bandwidth, not compute. Flash Attention's tile-based approach reduces memory from O(n²) to O(n) at the cost of recomputation in the backward pass."

**Transition**: "Section E key takeaways..."

---

## Slide 91: Section E: Key Takeaways

**Time**: ~1 min
**Talking points**:

- Three points: (1) self-attention allows every position to directly attend to every other; (2) multi-head attention runs h parallel patterns; (3) O(1) path length between any two tokens — long-range dependency problem solved.
- Check: "What are Q, K, V?" (Query: what I'm looking for; Key: what I advertise; Value: what I share.)

**Transition**: "Section F: putting it all together — the Transformer architecture..."

---

## Slide 92: F. The Transformer (Section Header)

**Time**: ~1 min
**Talking points**:

- "'Attention Is All You Need' — Vaswani et al., 2017. The most cited ML paper in history."
- If experts look bored: "We cover all modern improvements: RoPE, ALiBi, Flash Attention, GQA, Pre-LayerNorm."

**Transition**: "The full transformer architecture..."

---

## Slide 93: Transformer: High-Level Architecture

**Time**: ~4 min
**Talking points**:

- Walk through: input embeddings + positional encoding → N encoder blocks → N decoder blocks → linear + softmax.
- Encoder block: multi-head self-attention → add & norm → FFN → add & norm.
- Decoder block: masked self-attention → add & norm → cross-attention → add & norm → FFN → add & norm.
- "The encoder reads the entire input. The decoder generates output one token at a time."
- If beginners look confused: "Left side (encoder) reads input language. Right side (decoder) writes output language."
- If experts look bored: "Original: 6+6 layers for translation. Modern: encoder-only (BERT 12/24), decoder-only (GPT 12/96/96), full (T5, BART, mT5)."

**Transition**: "Let us zoom into the encoder block..."

---

## Slide 94: Encoder Block: Detailed

**Time**: ~3 min
**Talking points**:

- Two sublayers: multi-head self-attention and position-wise FFN. Each has residual + layer norm.
- "Self-attention mixes information across positions. FFN processes each position independently."
- Residual: output = sublayer(x) + x. "The original input is always preserved."
- If beginners look confused: "Each encoder block is a refinement step."
- If experts look bored: "No masking in encoder — fully bidirectional. One change: add causal masking and you have a decoder."

**Transition**: "Residual connections deserve special attention..."

---

## Slide 95: Residual Connections

**Time**: ~2 min
**Talking points**:

- From ResNets (He et al., 2016). Without residual: 12-layer gradients must flow through 12 non-linearities. With residual: gradients bypass any subset of layers.
- "Residual connections are why deep transformers train at all. They are gradient highways."
- If experts look bored: "Residual stream interpretation (Elhage et al., 2021): a single information highway all layers read from and write to. This framing predicts layer superposition and induction heads."

**Transition**: "Layer normalisation stabilises the residual stream..."

---

## Slide 96: Layer Normalisation

**Time**: ~2 min
**Talking points**:

- LayerNorm(x) = gamma × (x - mu) / (sigma + epsilon) + beta. Normalises across feature dimension.
- "Why not BatchNorm? Variable-length sequences make batch statistics noisy. LayerNorm normalises each sample independently."
- If experts look bored: "RMSNorm (Llama): drops mean-centering. Slightly faster, identical performance — a free win adopted by all modern models."

**Transition**: "The decoder's special ingredient: masked self-attention..."

---

## Slide 97: Decoder Block: Masked Self-Attention

**Time**: ~2 min
**Talking points**:

- Causal masking: set attention scores to -infinity for positions j > i before softmax. No information from the future.
- "This is the only difference between an encoder and a decoder: the masking pattern."
- If beginners look confused: "Imagine writing a sentence and only being allowed to look at words you have already written."
- If experts look bored: "Causal masking at training time enables parallel computation — all positions computed simultaneously with mask enforcing autoregressive structure."

**Transition**: "Positional encoding: telling the transformer about word order..."

---

## Slide 98: Positional Encoding: Why We Need It

**Time**: ~2 min
**Talking points**:

- Self-attention is permutation-equivariant. "Without positional encoding, 'dog bites man' and 'man bites dog' are identical."
- Three approaches: fixed sinusoidal (original), learned absolute, relative/rotary (modern).
- If experts look bored: "Permutation equivariance is a feature for set transformers. For language, it is a bug."

**Transition**: "Why sinusoidal encoding?"

---

## Slide 99: Positional Encoding: Why Sin/Cos?

**Time**: ~2 min
**Talking points**:

- PE(pos, 2i) = sin(pos / 10000^{2i/d}). Each position gets a unique vector. Deterministic and generalises to longer sequences.
- Key property: PE(pos + k) is a linear function of PE(pos) — enables length extrapolation.
- If beginners look confused: "Each position gets a unique fingerprint based on sin/cos waves at different frequencies."
- If experts look bored: "Learned absolute position embeddings (GPT-2) often outperform sinusoidal on in-distribution lengths, but fail catastrophically on longer sequences."

**Transition**: "Modern models use RoPE for better length generalisation..."

---

## Slide 100: RoPE: Rotary Position Embeddings

**Time**: ~2 min
**Talking points**:

- RoPE (Su et al., 2021): encode position as a rotation in Q/K space. q_m^T k_n depends only on (q, k, m-n) — relative position.
- Used in: LLaMA, Mistral, Falcon, PaLM 2, Gemma. Has replaced sinusoidal in all modern models.
- If beginners look confused: "RoPE encodes where a word is by rotating its representation — nearby words have similar rotations."
- If experts look bored: "Inner product of rotated Q and K gives relative position — critical for context window extension via YaRN scaling."

**Transition**: "ALiBi is an alternative that requires no modification at test time..."

---

## Slide 101: ALiBi: Attention with Linear Biases

**Time**: ~1 min
**Talking points**:

- Subtract a linear penalty from attention scores proportional to distance. Zero new parameters.
- Used in: BLOOM, MPT. Less prevalent since RoPE dominates.
- If experts look bored: "Slope m per head is set as a geometric sequence — different heads specialise in different attention distance ranges."

**Transition**: "The second component of each transformer block: the FFN..."

---

## Slide 102: Feed-Forward Network (FFN)

**Time**: ~2 min
**Talking points**:

- FFN(x) = max(0, x W_1 + b_1) W_2 + b_2. Applied independently to each token position.
- Inner dimension 4x model dimension: 768 → 3072 → 768 for BERT-base.
- "Attention is the communication layer; FFN is the computation layer. FFN accounts for ~2/3 of parameters — where factual knowledge is stored."
- If experts look bored: "SwiGLU/GeGLU (Llama) have replaced ReLU — ~1% improvement at no extra cost with 2/3 intermediate dimension scaling."

**Transition**: "How many parameters does a transformer have?"

---

## Slide 103: Parameter Count

**Time**: ~2 min
**Talking points**:

- BERT-base: 110M. BERT-large: 340M. GPT-2: 1.5B. GPT-3: 175B.
- "BERT-base: attention 28M + FFN 57M + embedding 23M ≈ 110M."
- If experts look bored: "Embedding tying (Press & Wolf, 2017) shares input embedding and output projection — saves vocab × d_model parameters without performance loss."

**Transition**: "How do we train transformers? Learning rate scheduling is critical..."

---

## Slide 104: Training: Warmup + Cosine Decay

**Time**: ~2 min
**Talking points**:

- Linear warmup (1-5% of steps) then cosine decay. "Warmup prevents catastrophic early divergence."
- kailash-ml: `TrainingPipeline(scheduler="cosine_warmup", warmup_steps=400, total_steps=10000)`.
- If experts look bored: "WSD (Warmup-Stable-Decay) is the modern alternative — stable phase enables checkpoint averaging. Used in Mistral, LLaMA training recipes."

**Transition**: "Other key training techniques..."

---

## Slide 105: Training Techniques

**Time**: ~2 min
**Talking points**:

- Walk through: label smoothing, dropout (attention weights + residual sublayers), weight decay, gradient clipping, mixed precision (BF16/FP16).
- "Most are default in kailash-ml's TrainingPipeline. You adjust them when default training diverges."
- If experts look bored: "BF16 strictly preferred over FP16 — wider exponent range prevents overflow. FP16's exponent overflows at values > 65504, which occurs regularly in deep transformers."

**Transition**: "Flash Attention dramatically improves attention efficiency..."

---

## Slide 106: Flash Attention

**Time**: ~3 min
**Talking points**:

- Standard attention materialises full n × n matrix in HBM. For n = 2048: 16MB per layer per head.
- Flash Attention (Dao et al., 2022): tile-based computation keeps running softmax in SRAM, never writes full matrix to HBM.
- Result: 2-4x speedup, 10-20x memory reduction, mathematically exact.
- "Flash Attention is now the default in every serious transformer implementation."
- If experts look bored: "Flash Attention 2 improves warp utilisation from ~25% to ~50%. Backward pass recomputes attention during backward rather than storing it — classic memory-compute tradeoff."

**Transition**: "GQA reduces KV cache memory for inference..."

---

## Slide 107: Grouped-Query Attention (GQA)

**Time**: ~2 min
**Talking points**:

- GQA (Ainslie et al., 2023): share K/V across groups of query heads. Between MHA (each head has its own K/V) and MQA (one shared K/V).
- Used in: LLaMA 2, Mistral 7B, Gemma, Falcon. "KV cache reduced by 4-8x with minimal quality loss."
- If experts look bored: "KV cache for a 70B LLaMA-2 at seq=4096, batch=8 is multiple tens of gigabytes. GQA makes the difference between fitting in one A100 and needing two."

**Transition**: "Sparse attention for very long sequences..."

---

## Slide 108: Sparse Attention

**Time**: ~2 min
**Talking points**:

- Full attention: O(n²). For n = 16384 tokens: attention matrix is 1GB — impractical.
- Patterns: local windowed (Longformer), global+local (BigBird). BigBird proves sparse attention is Turing-complete.
- If experts look bored: "Turing-completeness requires both local and global attention — pure local window attention cannot represent all functions."

**Transition**: "Pre-LayerNorm vs Post-LayerNorm..."

---

## Slide 109: Pre-LayerNorm vs Post-LayerNorm

**Time**: ~1 min
**Talking points**:

- Post-LN: original paper. Pre-LN: modern standard. Pre-LN is more stable, trains without warmup.
- "Almost all models trained after 2019 use Pre-LayerNorm."
- If experts look bored: "Post-LN instability: residual addition causes layer output scale to grow with depth. Pre-LN normalises this."

**Transition**: "Why did the transformer dominate all alternatives?"

---

## Slide 110: The Transformer: Why It Won

**Time**: ~2 min
**Talking points**:

- Four reasons: parallelism, direct path length, scalability (consistent power-law improvement), transfer learning.
- "The transformer won because it was the first architecture that could be scaled efficiently with compute."
- If experts look bored: "The scalability argument is decisive. CNNs and RNNs exhibit diminishing returns with scale. Transformers show consistent power-law improvement — the Chinchilla scaling law in Section G."

**Transition**: "Encoder-only vs decoder-only vs full encoder-decoder..."

---

## Slide 111: Encoder-Only vs Decoder-Only vs Full

**Time**: ~2 min
**Talking points**:

- Encoder-only (BERT): bidirectional, best for classification/NER/QA. Decoder-only (GPT): causal, excellent for generation. Full (T5, BART): translation, summarisation.
- "Task determines architecture: understand → encoder-only. Generate → decoder-only. Transform → encoder-decoder."
- If beginners look confused: "BERT understands text. GPT generates text. T5 transforms text."
- If experts look bored: "The encoder-only vs decoder-only debate is settled for generation — GPT-style decoder-only scales better. Full encoder-decoder remains alive in structured prediction."

**Transition**: "KV caching makes transformer inference practical..."

---

## Slide 112: KV Cache: Why Transformers Are Fast at Inference

**Time**: ~2 min
**Talking points**:

- KV cache: store K and V from past steps, reuse them. Converts O(n²) inference to O(n).
- "KV cache is what makes real-time chatbots possible."
- If experts look bored: "Continuous batching (vLLM's PagedAttention) treats the KV cache as virtual memory pages — increases throughput by 2-4x over naive batching."

**Transition**: "What does training at scale require?"

---

## Slide 113: Transformer Training at Scale

**Time**: ~2 min
**Talking points**:

- Pipeline parallelism, tensor parallelism, data parallelism. "Training GPT-3: 3.14 × 10^23 FLOPs — roughly 355 GPU-years on a single A100."
- Gradient checkpointing: recompute activations during backward. Trade compute for memory.
- If experts look bored: "Megatron-LM combined 3D parallelism. Pipeline bubble is the efficiency bottleneck — GPipe and PipeDream address it with microbatching."

**Transition**: "The original paper: a brief look back..."

---

## Slide 114: The Original Transformer: "Attention Is All You Need"

**Time**: ~2 min
**Talking points**:

- Vaswani et al. (2017). Achieved 28.4 BLEU on WMT English-German — beating all previous models.
- "The provocative title was accurate. RNNs, convolutions — all unnecessary."
- Training time: 3.5 days on 8 P100 GPUs. GPT-3 would take 355 GPU-years.
- If experts look bored: "The paper was initially rejected — reviewers found the ablation insufficient. Full ablation was added in the NIPS version."

**Transition**: "Section F key takeaways..."

---

## Slide 115: Section F: Key Takeaways

**Time**: ~1 min
**Talking points**:

- Three points: (1) transformer = multi-head attention + FFN + residual + layer norm, repeated N times; (2) positional encoding (sinusoidal or RoPE) injects order; (3) modern improvements — Flash Attention, GQA, Pre-LN — are production requirements.
- Check: "What does the decoder's causal mask do?" (Prevents attending to future tokens.)

**Transition**: "Section G: pre-training — how do we get powerful models cheaply..."

---

## Slide 116: G. Pre-training Paradigms (Section Header)

**Time**: ~1 min
**Talking points**:

- "Pre-training is why a single training run produces a model useful for thousands of different tasks."
- If experts look bored: "We cover Chinchilla's derivation of optimal token-to-parameter ratios and the implications for training compute allocation."

**Transition**: "The three paradigms of modern NLP..."

---

## Slide 117: The Three Paradigms

**Time**: ~2 min
**Talking points**:

- Paradigm 1: feature engineering + linear models. Paradigm 2: neural feature learning. Paradigm 3: pre-training + fine-tuning.
- "Each paradigm shift reduced human expertise required and increased the quality ceiling."
- If experts look bored: "The fourth paradigm is emerging: pre-train → instruction-tune → RLHF → the fine-tuning step is being replaced by prompting. We may be in another transition."

**Transition**: "BERT: the first major pre-training breakthrough..."

---

## Slide 118: BERT: Masked Language Model

**Time**: ~3 min
**Talking points**:

- BERT (Devlin et al., 2018): randomly mask 15% of tokens, predict the masked tokens.
- "MLM forces deep bidirectional understanding — context from both left and right."
- NSP later shown to be unhelpful — RoBERTa drops it.
- If beginners look confused: "BERT is a fill-in-the-blank exam on billions of sentences. Getting good at it means understanding language deeply."
- If experts look bored: "The 80/10/10 split (mask/random/original) prevents the model from learning that [MASK] means 'predict here' — at fine-tune time no masks appear."

**Transition**: "How do we use BERT for specific tasks?"

---

## Slide 119: BERT: Fine-tuning

**Time**: ~2 min
**Talking points**:

- Add task-specific head on [CLS] token. Fine-tune all weights with small labelled dataset.
- "BERT set new state of the art on 11 NLP tasks simultaneously. Nothing like it had happened before."
- kailash-ml: `AutoMLEngine(task="text_classification", model="bert-base-uncased").fit(texts, labels)`.
- If experts look bored: "For long documents, mean-pooling over all token representations typically outperforms [CLS] pooling."

**Transition**: "BERT spawned a family of variants..."

---

## Slide 120: BERT Variants

**Time**: ~2 min
**Talking points**:

- RoBERTa: removes NSP, trains longer. Strictly better than BERT.
- DistilBERT: 40% smaller, 60% faster, 97% of performance via knowledge distillation.
- Domain variants: BioBERT, LegalBERT, FinBERT. Multilingual: mBERT (104 languages), XLM-R (100).
- If experts look bored: "DeBERTa (He et al., 2020) — disentangled attention + virtual adversarial training — is the strongest encoder for most benchmarks today."

**Transition**: "GPT takes the opposite approach: causal language modelling..."

---

## Slide 121: GPT: Causal Language Model

**Time**: ~3 min
**Talking points**:

- GPT: predict next token (causal LM). Decoder-only, left-to-right.
- Fine-tune by reformatting as text generation: "Sentiment: [review] → [positive/negative]".
- "BERT understands by filling blanks. GPT learns by predicting the future."
- If beginners look confused: "GPT learns language by reading billions of sentences and practising finishing them."
- If experts look bored: "Decoder-only architecture enables zero-shot task performance via prompting — key advantage over BERT for few-shot applications."

**Transition**: "How did GPT evolve?"

---

## Slide 122: GPT Evolution

**Time**: ~2 min
**Talking points**:

- GPT-1 (2018): 117M params. Showed transfer learning works. GPT-2 (2019): 1.5B. Zero-shot learning. GPT-3 (2020): 175B. In-context learning changed the industry. InstructGPT (2022): RLHF made GPT usable as assistant.
- If experts look bored: "GPT-2 staged release established the precedent for capability evaluations before release — directly influenced current RLHF practices and AI safety norms."

**Transition**: "T5 unifies all NLP as text-to-text..."

---

## Slide 123: T5: Text-to-Text

**Time**: ~2 min
**Talking points**:

- T5 (Raffel et al., 2020): every NLP task as text-to-text. Same format for translation, summarisation, sentiment.
- Pre-training: span corruption on C4 (750GB cleaned web text).
- "T5 showed a single text-to-text format can match task-specific architectures on every NLP benchmark."
- If experts look bored: "C4's aggressive deduplication removed 70% of raw data — data quality vs quantity is the underrated variable."

**Transition**: "How much compute and data do we need? Scaling laws answer this..."

---

## Slide 124: Scaling Laws: Chinchilla

**Time**: ~3 min
**Talking points**:

- Hoffmann et al. (2022): for a given compute budget, train a smaller model on more tokens.
- Chinchilla optimal: tokens = 20 × parameters. GPT-3 (175B) should train on 3.5T tokens, not 300B.
- "GPT-3 was undertrained. Chinchilla (70B, 1.4T tokens) outperformed Gopher (280B) despite being 4x smaller."
- If beginners look confused: "Scaling laws tell you how to spend compute wisely — more data is often better than a bigger model."
- If experts look bored: "LLaMA computed Chinchilla-optimal at inference, not training — train 7B on 1T tokens to minimise inference cost, not training cost."

**Transition**: "Mixture of Experts allows scaling without proportional cost..."

---

## Slide 125: Mixture of Experts (MoE)

**Time**: ~3 min
**Talking points**:

- Replace dense FFN with N expert FFNs. Router selects k of N per token (k=2, N=8 or N=64). Sparse activation.
- "Mistral 8x7B: 47B parameters but activates only 13B per token — inference cost of 13B, quality of 47B."
- Challenge: load balancing. Auxiliary loss prevents all tokens routing to same experts.
- If beginners look confused: "MoE is a team of specialists — only a few do the work for each input, but you have access to all of them."
- If experts look bored: "Expert collapse is the main failure mode. GShard's auxiliary balancing loss penalises imbalanced routing. Switch Transformer scaled to 1.6T parameters."

**Transition**: "In-context learning: GPT-3's revolutionary property..."

---

## Slide 126: In-Context Learning: How GPT-3 Changed Everything

**Time**: ~3 min
**Talking points**:

- Zero-shot: task description only. Few-shot: 2-5 examples in the prompt. No gradient updates.
- "GPT-3 showed that a sufficiently large LM can perform new tasks without any fine-tuning. This changed the economics of NLP."
- If beginners look confused: "You can teach GPT-3 a new task just by showing it examples in the prompt — no retraining needed."
- If experts look bored: "Min et al. (2022) showed the labels in few-shot examples barely matter — what matters is format and input distribution. ICL is format adaptation, not task learning."

**Transition**: "The transfer learning pipeline in production..."

---

## Slide 127: Transfer Learning Pipeline

**Time**: ~2 min
**Talking points**:

- Pipeline: (1) load pre-trained model; (2) add task head; (3) fine-tune with AdamW + cosine schedule; (4) evaluate; (5) register to ModelRegistry.
- `TrainingPipeline(base_model="bert-base-uncased", task="classification", num_labels=3).fit(texts, labels)`
- If experts look bored: "PEFT (LoRA, prefix tuning, adapter layers) makes fine-tuning accessible at 1-2 GPU hours for 7B models. kailash-ml: `use_lora=True`."

**Transition**: "Instruction tuning and RLHF transform a language model into an assistant..."

---

## Slide 128: Instruction Tuning and RLHF

**Time**: ~3 min
**Talking points**:

- Base LM continues text. Instruction-tuned LM follows instructions. Difference: training on (instruction, response) pairs.
- RLHF: preference rankings → reward model → PPO optimisation.
- "InstructGPT showed a 1.3B RLHF model was preferred by humans over a 175B base GPT-3."
- DPO: simpler alternative, no separate reward model needed.
- If beginners look confused: "RLHF teaches the model what 'good answers' look like by getting human ratings."
- If experts look bored: "DPO's key insight: the optimal RLHF policy has a closed-form solution. Training directly on preference pairs without RL. Covered in depth in Module 6 (Align framework)."

**Transition**: "What goes into pre-training data?"

---

## Slide 129: The Pre-training Data Pipeline

**Time**: ~2 min
**Talking points**:

- Web crawl → language detection → deduplication → quality filtering → domain mixing.
- "The data pipeline is as important as the model architecture. Garbage training data produces garbage models."
- Singapore relevance: "For APAC applications, include local-language corpus and Singapore government corpus."
- If experts look bored: "MinHash LSH deduplication at 13T-token scale found extraordinary Web duplication — removing duplicates improves every benchmark tested."

**Transition**: "Section G key takeaways..."

---

## Slide 130: Section G: Key Takeaways

**Time**: ~1 min
**Talking points**:

- Three points: (1) pre-train on large unlabelled corpus, fine-tune on small labelled set — the dominant paradigm; (2) Chinchilla: tokens = 20 × parameters for compute-optimal training; (3) RLHF / instruction tuning converts a language model into an assistant.
- Check: "What pre-training objective does BERT use?" (MLM — predict masked tokens bidirectionally.)

**Transition**: "Section H: what do we actually do with NLP models — tasks and decoding..."

---

## Slide 131: H. NLP Tasks & Decoding (Section Header)

**Time**: ~1 min
**Talking points**:

- "We have the models. Section H covers how to use them for real tasks and how to generate good output."
- If experts look bored: "We cover BLEU in detail plus modern alternatives (ROUGE, BERTScore, BLEURT) and when each is appropriate."

**Transition**: "A map of NLP tasks..."

---

## Slide 132: NLP Task Taxonomy

**Time**: ~2 min
**Talking points**:

- Four categories: classification (sentiment, topic), sequence labelling (NER, POS), structured prediction (parsing), generation (translation, summarisation, QA, dialogue).
- "Encoder-only for classification/labelling; decoder-only for generation; full transformer for structured generation."
- If experts look bored: "The taxonomy is collapsing — instruction-tuned decoder-only models now handle classification via prompting. Architectural specialisation is dissolving."

**Transition**: "Decoding: how do we choose which token to generate next?"

---

## Slide 133: Decoding: Greedy and Beam Search

**Time**: ~3 min
**Talking points**:

- Greedy: always pick highest probability token. Fast but often repetitive.
- Beam search: maintain top-k sequences, expand each step, keep top-k. k=4 is standard for translation.
- "Beam search with k=4 is standard for translation. Greedy is fine for chatbots where diversity is desired."
- If experts look bored: "Length normalisation penalty = log P / T^alpha with alpha ~0.6 partially mitigates short sequence bias. Diverse beam search adds a diversity penalty across beams."

**Transition**: "Sampling strategies allow more creative and diverse outputs..."

---

## Slide 134: Decoding: Sampling Strategies

**Time**: ~2 min
**Talking points**:

- Temperature: divide logits by T. T < 1 sharpens, T > 1 flattens. T ≈ 0 ≈ greedy.
- Production defaults: temperature 0.7-1.0 for creative tasks, 0.0-0.3 for factual.
- If beginners look confused: "Temperature controls creativity — low = focused answers, high = varied but less accurate."
- If experts look bored: "Temperature calibration: if a model is over-confident, increasing temperature improves calibration on the holdout set."

**Transition**: "Top-k and nucleus sampling are the industry standards..."

---

## Slide 135: Top-k and Top-p (Nucleus) Sampling

**Time**: ~2 min
**Talking points**:

- Top-k: sample from k most probable tokens. Problem: fixed k regardless of distribution shape.
- Top-p (nucleus): smallest set of tokens whose cumulative probability exceeds p. p = 0.9 or 0.95 typical.
- "Nucleus sampling adapts — confident model → small nucleus; uncertain → expands."
- Production: temperature + top-p. "temp=0.8, top_p=0.9" is a common combination.
- If experts look bored: "Typical decoding (Meister et al., 2022) selects tokens near average entropy level — theoretically motivated as maximising local typicality."

**Transition**: "How do we measure NLP model quality?"

---

## Slide 136: NLP Metrics

**Time**: ~2 min
**Talking points**:

- Generation: BLEU (n-gram precision), ROUGE (n-gram recall), BERTScore (semantic similarity), BLEURT (learned).
- Perplexity: intrinsic LM metric — lower is better.
- "Human evaluation is the gold standard. Automated metrics are proxies — use multiple."
- If experts look bored: "BERTScore correlates better with human judgement for abstractive summarisation. Requires GPU and is sensitive to reference model choice."

**Transition**: "BLEU score in detail..."

---

## Slide 137: BLEU Score: Detailed

**Time**: ~2 min
**Talking points**:

- BLEU = modified n-gram precision (1-4-gram) × brevity penalty.
- Modified precision: each reference n-gram matches at most one system n-gram. Brevity penalty discourages short outputs.
- "BLEU-4 on WMT EN-DE: human ~25-30. Google Translate ~34. GPT-4 ~40+."
- If experts look bored: "BLEU measures surface overlap, not meaning. SacreBLEU standardises tokenisation to make scores comparable across papers."

**Transition**: "Named Entity Recognition: extracting structure from text..."

---

## Slide 138: NER: Named Entity Recognition

**Time**: ~2 min
**Talking points**:

- Identify and classify: persons (PER), organisations (ORG), locations (LOC), dates (DATE).
- Singapore-specific: HDB, CPF, MAS, EDB, NEA — required for Singapore financial/government NLP.
- kailash-ml: `TrainingPipeline(task="token_classification", model="bert-base", label_scheme="IOB")`.
- If experts look bored: "Nested NER (entities within entities) requires span-based or biaffine attention models — standard IOB cannot represent simultaneous overlapping entities."

**Transition**: "Machine translation: NLP's oldest and biggest commercial application..."

---

## Slide 139: Machine Translation: A Success Story

**Time**: ~3 min
**Talking points**:

- Evolution: rule-based → statistical → phrase-based → neural seq2seq → transformer. Each transition gave measurable BLEU gains.
- Singapore relevance: four official languages (English, Mandarin, Malay, Tamil) and ASEAN multilingual market.
- Low-resource challenge: Malay-Tamil parallel data is scarce. Cross-lingual transfer via mBERT or XLM-R.
- If experts look bored: "Back-translation (Sennrich et al., 2016) bootstraps low-resource: translate target-side monolingual data to source language with a weak model, creating synthetic training pairs."

**Transition**: "Summarisation and generation: the creative end of NLP..."

---

## Slide 140: Summarisation and Generation

**Time**: ~2 min
**Talking points**:

- Extractive: select sentences. Abstractive: generate new sentences. Transformers enable abstractive.
- Singapore use case: Straits Times summarisation, MAS financial report summarisation, legal document summarisation.
- kailash-ml: `AutoMLEngine(task="summarization", model="facebook/bart-large-cnn")`.
- If experts look bored: "PEGASUS's gap sentence generation objective outperforms generic MLM for summarisation — a lesson in task-adaptive pre-training."

**Transition**: "Common generation failure modes..."

---

## Slide 141: Repetition and Length Control

**Time**: ~2 min
**Talking points**:

- Repetition: fix with repetition penalty. Length: fix with penalty in beam search + min/max constraints.
- Hallucination: "The most dangerous failure mode — plausible-sounding but factually wrong."
- If experts look bored: "Hallucination mitigation: retrieval augmentation, contrastive decoding, calibration-based uncertainty. None fully solves it — active research area."

**Transition**: "Section H key takeaways..."

---

## Slide 142: Section H: Key Takeaways

**Time**: ~1 min
**Talking points**:

- Three points: (1) task taxonomy determines architecture choice; (2) greedy for speed, beam for quality, nucleus for diversity; (3) BLEU/ROUGE measure surface overlap — use BERTScore for semantic quality.
- Check: "When would you use beam search vs nucleus sampling?" (Beam: translation/summarisation. Nucleus: open-ended generation, dialogue.)

**Transition**: "Section I: Kailash engines and synthesis..."

---

## Slide 143: I. Kailash & Synthesis (Section Header)

**Time**: ~1 min
**Talking points**:

- "Everything we have covered maps to specific kailash-ml engines."
- If beginners look confused: "We will now see how all the theory connects to actual code you can run."

**Transition**: "The Kailash engine map for NLP..."

---

## Slide 144: Kailash Engines for NLP

**Time**: ~3 min
**Talking points**:

- Mapping: preprocessing → PreprocessingPipeline; BoW/TF-IDF → AutoMLEngine(vectorizer="tfidf"); embeddings → AutoMLEngine(embedding="glove-300d" or "bert"); fine-tuning → TrainingPipeline; evaluation → ExperimentTracker; visualisation → ModelVisualizer.
- "The engine choice determines the approach — you do not mix raw sklearn with Kailash engines."
- If experts look bored: "TrainingPipeline handles the full HuggingFace model zoo — tokenisation, gradient accumulation, mixed precision, evaluation loop."

**Transition**: "ModelVisualizer gives you interpretability for NLP models..."

---

## Slide 145: ModelVisualizer for Text

**Time**: ~3 min
**Talking points**:

- Four visualisation types: (1) attention heatmaps; (2) embedding projector (UMAP/t-SNE); (3) token attribution (SHAP/integrated gradients); (4) confusion matrix for classification.
- `ModelVisualizer(model=trained_bert).plot_attention(text="The bank is near the river bank")` — watch how the two "bank" tokens attend differently.
- If beginners look confused: "ModelVisualizer shows what the model is 'thinking' — which words it focuses on when making a decision."
- If experts look bored: "Integrated gradients is more reliable than raw attention for attribution — measures causal importance, not attention magnitude."

**Transition**: "AutoMLEngine automates the full text classification pipeline..."

---

## Slide 146: AutoMLEngine for Text Classification

**Time**: ~3 min
**Talking points**:

- `AutoMLEngine(task="text_classification")` tries: TF-IDF + LR, TF-IDF + GBM, DistilBERT fine-tuning, BERT fine-tuning.
- Outputs: best model, comparison table, recommended model with justification.
- "AutoMLEngine is the starting point for any new NLP classification problem — run it first, optimise the winner."
- If experts look bored: "`AutoMLEngine(search_space=['tfidf_lr', 'distilbert'], time_budget=3600)` — time budget constraint makes it practical for production use."

**Transition**: "The grand synthesis: everything connects..."

---

## Slide 147: The Grand Synthesis

**Time**: ~4 min
**Talking points**:

- Draw the evolution chain: text → BoW (order lost) → TF-IDF (importance weighted) → Word2Vec (semantic geometry) → LSTM (sequential context) → attention (direct connections) → transformer (parallel, scalable) → BERT/GPT (pre-trained, transferable).
- "Each step addressed the limitation of the previous one. This is how science progresses."
- The unifying theme: richer representation of meaning, from bag of characters to contextual 768-dimensional vectors.
- If experts look bored: "Probing classifiers at each stage show representational richness improving monotonically — from BoW to transformer layer 12."

**Transition**: "Where each Kailash engine sits in the M1-M8 stack..."

---

## Slide 148: Cumulative Kailash Engine Map

**Time**: ~2 min
**Talking points**:

- Full engine map M1-M8: DataExplorer, PreprocessingPipeline, FeatureEngineer/FeatureStore, TrainingPipeline, AutoMLEngine, ModelVisualizer, ExperimentTracker, ModelRegistry, InferenceServer.
- "You have now touched every major engine in the kailash-ml framework. This is your toolkit for production ML."
- If experts look bored: "In production, this entire pipeline is orchestrated by the Kailash Core SDK WorkflowBuilder — everything is a node."

**Transition**: "A preview of what comes next..."

---

## Slide 149: Preview: Module 9 — LLMs, Agents & RAG

**Time**: ~2 min
**Talking points**:

- Module 9 builds directly on today: Kaizen agent framework, RAG with Nexus, advanced prompting (chain-of-thought, ReAct, reflexion), tool use and MCP.
- "Module 8 gave you the theory. Module 9 is about building systems that use these models in production."
- If beginners look confused: "Module 9 is about building AI assistants that answer questions using real documents."
- If experts look bored: "Module 9 covers dense retrieval (DPR), hybrid BM25 + dense, cross-encoder reranking, and the Nexus deployment layer. The attention and transformer concepts from today are load-bearing prerequisites."

**Transition**: "Let us wrap up Module 8..."

---

## Slide 150: NLP & Transformers (Closing Title)

**Time**: ~2 min
**Talking points**:

- Recap the journey: preprocessing → BoW/TF-IDF → word embeddings → RNNs → attention → transformers → pre-training → NLP tasks.
- "The transformer is the general-purpose learning machine for sequences. Everything you see in modern AI builds on it."
- Exercise assignments: (1) build a complete text preprocessing pipeline with PreprocessingPipeline; (2) run AutoMLEngine comparing TF-IDF vs BERT on Singapore data; (3) use ModelVisualizer to inspect attention patterns and write a one-page interpretation report.
- If experts look bored: "Open research questions: efficient long-context attention, mechanistic interpretability, closing the gap between in-context learning and gradient-based learning."

**Transition**: "Any final questions before we close? See you in Module 9."

---
