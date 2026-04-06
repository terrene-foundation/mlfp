# Module 8: NLP & Transformers — Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: NLP & Transformers (Title)

**Time**: ~2 min
**Talking points**:

- Read the title and let the scope land. This module bridges Module 7 (deep learning) to Module 9 (LLMs, agents, RAG).
- Frame the arc: "We start by asking how to turn raw text into numbers, and finish by understanding how transformers changed everything."
- Provocation: "Every modern AI assistant you have ever talked to is built on what you learn today."
- If beginners look confused: "We are going to learn how computers understand language — from basic counting to billion-parameter models."
- If experts look bored: "We will derive attention from scratch, compare positional encodings, and discuss Flash Attention and GQA."

**Transition**: "Let us ground ourselves in where we have been..."

---

## Slide 2: Recap: Where We Are

**Time**: ~2 min
**Talking points**:

- Quick table: M1 Python, M2 statistics, M3 features, M4 supervised ML, M5 production, M6 unsupervised, M7 deep learning.
- Key message: "Module 7 gave us neural networks. Module 8 applies them to the most important data type: language."
- Do not re-teach. This is orientation, not review.
- If beginners look confused: "Everything we built before was for structured tables. Today we handle text."
- If experts look bored: "Architecturally: we move from fixed-size input MLPs and CNNs to variable-length sequence models with dynamic context."

**Transition**: "Here is why this module is historically significant..."

---

## Slide 3: The NLP Revolution Timeline

**Time**: ~3 min
**Talking points**:

- Walk the timeline: 2013 Word2Vec, 2014 Seq2Seq, 2015 Bahdanau attention, 2017 "Attention Is All You Need", 2018 BERT/GPT, 2019 GPT-2, 2020 GPT-3, 2022 ChatGPT, 2023 GPT-4/Llama.
- "Seven years from Word2Vec to ChatGPT. This is the fastest paradigm shift in the history of technology."
- Pause at 2017: "This single paper made everything before it obsolete."
- If beginners look confused: "Each dot on this timeline is a breakthrough that built on the last one — you will understand all of them today."
- If experts look bored: "The scaling inflection is notable: 2017-2020 architecture innovation, 2020-2023 pure compute scaling. What comes next?"

**Transition**: "Here is our roadmap for today..."

---

## Slide 4: Module 8 Roadmap

**Time**: ~2 min
**Talking points**:

- Walk through sections A through H plus Kailash synthesis: preprocessing, BoW/TF-IDF, embeddings, RNNs, attention, transformers, pre-training, NLP tasks, Kailash engines.
- "We are building from primitives to state-of-the-art. Each section is a complete conceptual unit."
- If beginners look confused: "Think of it as layers — we start with text as characters and end with models that understand meaning."
- If experts look bored: "We cover both classical and modern — you need both because production systems blend them constantly."

**Transition**: "First, the bridge from deep learning to language..."

---

## Slide 5: The Bridge: DL → Language

**Time**: ~3 min
**Talking points**:

- Module 7 gave us feedforward networks, CNNs, and the mechanics of backprop. Now we ask: what changes when input is a sequence of variable length?
- Three new challenges: variable input length, long-range dependencies, order matters.
- The key insight: "A sentence is not a fixed-size vector — it is a sequence. The order of words is information."
- If beginners look confused: "A table has rows and columns. A sentence has words in order. That order carries meaning."
- If experts look bored: "The fundamental challenge is encoding permutation variance — we want 'dog bites man' and 'man bites dog' to produce different representations."

**Transition**: "Let me show you a concrete example of why this matters in industry..."

---

## Slide 6: Case Study: BloombergGPT

**Time**: ~4 min
**Talking points**:

- Bloomberg trained a 50B parameter LLM on 363B tokens of financial text — news, filings, earnings calls, analyst reports.
- On financial NLP benchmarks: BloombergGPT outperforms GPT-NeoX (20B) by a wide margin despite being similar size.
- Key lesson: domain-specific pre-training data matters more than raw parameter count at the same scale.
- Production impact: real-time news sentiment, document summarisation, entity extraction from earnings calls.
- If beginners look confused: "A bank trained an AI that speaks finance. It outperforms general AI on finance tasks because it learned from finance data."
- If experts look bored: "The FinPile data mix ratio is the interesting decision — 35% financial, 65% general corpus. Too much domain data causes catastrophic forgetting of general reasoning."

**Transition**: "Why do domain-specific models win so consistently?"

---

## Slide 7: Why Domain-Specific Models Win

**Time**: ~3 min
**Talking points**:

- Three reasons: vocabulary coverage (medical/legal/financial terms are OOV in general tokenisers), distribution shift (domain text has different statistical patterns), and task alignment (fine-tuned instruction format matches actual work).
- Examples: BioBERT, LegalBERT, FinBERT — each outperforms base BERT on domain tasks.
- Practical point: "You do not need to pre-train from scratch. Domain-specific fine-tuning of a general base model is usually sufficient."
- If beginners look confused: "A doctor trained on medical textbooks knows more about medicine than one trained on Wikipedia."
- If experts look bored: "The continual pre-training vs fine-tuning debate is live research — continual pre-training is expensive but gives better low-data generalisation."

**Transition**: "Here is what you will build in this module..."

---

## Slide 8: What You Will Build

**Time**: ~2 min
**Talking points**:

- Preview the three exercises: (1) text preprocessing pipeline with kailash-ml, (2) TF-IDF document classifier, (3) transformer-based sentiment analysis with ModelVisualizer and AutoMLEngine.
- "By the end you have a production-ready NLP pipeline that runs on real Singapore data."
- If beginners look confused: "You will have working code that classifies text — the same kind of thing that powers spam filters and review systems."
- If experts look bored: "The AutoMLEngine exercise tests multiple architectures automatically — pay attention to how it handles the text vectorisation step."

**Transition**: "Let us start at the very beginning: what is text, really?"

---

## Slide 9: A. Text Preprocessing (Section Header)

**Time**: ~1 min
**Talking points**:

- Section A covers everything that happens before a single number touches a model.
- "Garbage in, garbage out. In NLP, the preprocessing step is often more important than the model choice."
- If beginners look confused: "We need to turn messy human text into something a computer can work with."

**Transition**: "Let us look at the core challenge..."

---

## Slide 10: Text as Data: The Challenge

**Time**: ~3 min
**Talking points**:

- Text is unstructured, multilingual, noisy, and context-dependent. "The same word means different things in different contexts. 'Apple' in a tech article vs a recipe."
- Walk through the challenges: spelling variation, abbreviations, emoji, code-switching, sarcasm.
- Singapore data is particularly challenging: Singlish, code-switching between English/Mandarin/Malay, unique abbreviations (lah, lor, sia).
- If beginners look confused: "Think about how different a WhatsApp message looks versus a legal contract — both are text but completely different."
- If experts look bored: "The noisy text problem is still unsolved at scale — BPE tokenisers partially handle it, but out-of-vocabulary compound words remain a challenge."

**Transition**: "Here is the standard pipeline for handling this..."

---

## Slide 11: The NLP Pipeline

**Time**: ~3 min
**Talking points**:

- Walk through the standard steps: raw text → tokenisation → normalisation → stop word removal → stemming/lemmatisation → feature extraction.
- Not every pipeline uses all steps — for transformer models, you often skip the linguistic steps entirely.
- "Classical NLP uses the full pipeline. Neural NLP often skips to tokenisation → embedding."
- If beginners look confused: "These are like cleaning steps before cooking. You do not cook raw, dirty vegetables."
- If experts look bored: "The pipeline is modular by design — you can swap individual steps. The kailash-ml PreprocessingPipeline implements this as composable stages."

**Transition**: "The first and most fundamental step: tokenisation..."

---

## Slide 12: Tokenisation: Word-Level

**Time**: ~3 min
**Talking points**:

- Simplest approach: split on whitespace and punctuation. "The cat sat." → ["The", "cat", "sat", "."]
- Problem: vocabulary explosion. English has 170,000 words. With proper nouns and domain terms, real corpora have millions of unique tokens.
- Large vocabulary = large embedding matrix = memory and generalisation issues.
- Unknown tokens (OOV) for any word not seen at training time.
- If beginners look confused: "Word tokenisation is like treating each word as a separate item in a list."
- If experts look bored: "The vocabulary-quality tradeoff: larger vocab reduces OOV but exponentially increases embedding matrix size. The sweet spot is around 30-50K for most applications."

**Transition**: "What about going to the other extreme: characters?"

---

## Slide 13: Tokenisation: Character-Level

**Time**: ~2 min
**Talking points**:

- Every character is a token. Vocabulary size is tiny (~100-300 characters). Zero OOV problem.
- Problem: sequences become very long. "Transformer" is 11 tokens instead of 1. Attention complexity is O(n²) — this matters.
- Used in: CharRNN, some multilingual models, spell checking.
- If beginners look confused: "Instead of treating each word as a unit, we treat each letter as a unit."
- If experts look bored: "Character models can generate misspellings naturally — useful for data augmentation and adversarial robustness testing."

**Transition**: "The best of both worlds is subword tokenisation..."

---

## Slide 14: Subword Tokenisation: The Middle Ground

**Time**: ~4 min
**Talking points**:

- Key insight: common words are full tokens; rare words are split into subword pieces.
- "unhappiness" → ["un", "happiness"] or ["un", "happy", "##ness"]
- This balances vocabulary size (~30K tokens) with sequence length and OOV handling.
- Three major algorithms: BPE (GPT family), WordPiece (BERT), Unigram (SentencePiece/T5).
- If beginners look confused: "Subword tokenisation is like handling words you do not know by breaking them into parts you do know."
- If experts look bored: "The choice of tokeniser has downstream effects — GPT-style BPE handles code better; BERT WordPiece handles morphologically rich languages better."

**Transition**: "Let us dig into BPE, the most widely used algorithm..."

---

## Slide 15: BPE Algorithm: Step by Step

**Time**: ~4 min
**Talking points**:

- BPE = Byte Pair Encoding. Start with characters. Iteratively merge the most frequent adjacent pair until vocabulary target is reached.
- Walk through the algorithm: count pairs → find most frequent → merge → repeat.
- "It is a greedy compression algorithm. It finds the most efficient encoding of your training corpus."
- Used in: GPT-2, GPT-3, GPT-4, Llama, Falcon, Mistral — essentially all decoder-only models.
- If beginners look confused: "BPE finds common letter combinations and treats them as single units, like 'th' or 'ing'."
- If experts look bored: "The initialisation matters — byte-level BPE (as in GPT-2) avoids the Unicode pre-tokenisation problem and handles any text losslessly."

**Transition**: "Let us trace through a concrete merge example..."

---

## Slide 16: BPE: Detailed Merge Example

**Time**: ~3 min
**Talking points**:

- Trace through the example step by step. Show the merge table building up.
- "Follow the pointer: the final merge table is the tokeniser. Given any new text, you apply merges in learned order."
- Key point: the order of merges is fixed at training time — inference just replays the table.
- If beginners look confused: "This is the dictionary the tokeniser uses. It was learned from data, not hand-crafted."
- If experts look bored: "The determinism of BPE means tokenisation is reproducible but not optimal for every input — this is why character-level fallbacks exist."

**Transition**: "Two variants you will encounter in practice: WordPiece and Unigram..."

---

## Slide 17: WordPiece and Unigram

**Time**: ~3 min
**Talking points**:

- WordPiece (BERT): like BPE but merges the pair that maximises language model likelihood, not raw frequency. Result: slightly better coverage of rare words.
- Unigram (SentencePiece/T5): starts with a large vocabulary and prunes tokens that least reduce a unigram language model score.
- SentencePiece is language-agnostic — handles Japanese, Chinese, Arabic without language-specific pre-tokenisation.
- If beginners look confused: "These are just different flavours of the same idea — finding the best way to split text into chunks."
- If experts look bored: "Unigram's probabilistic formulation gives multiple valid tokenisations per string — useful for data augmentation and regularisation."

**Transition**: "Now let us look at the linguistic preprocessing steps..."

---

## Slide 18: Stop Words

**Time**: ~2 min
**Talking points**:

- Stop words are high-frequency function words: the, a, is, are, in, on, at. They carry little semantic content for classical NLP.
- Removing them reduces vocabulary size and noise in BoW/TF-IDF models.
- CAUTION: for neural models and transformers, stop words carry syntactic and semantic information — do NOT remove them.
- If beginners look confused: "Stop words are like filler words in speech — not much meaning on their own."
- If experts look bored: "The stop word decision is task-dependent. Sentiment analysis: keep negations ('not good'). Topic modelling: remove aggressively. Named entity recognition: always keep."

**Transition**: "Related: stemming and lemmatisation..."

---

## Slide 19: Stemming

**Time**: ~2 min
**Talking points**:

- Stemming chops word endings with heuristic rules. "running" → "run", "happiness" → "happi", "studies" → "studi".
- Fast but crude — produces non-words. Porter Stemmer and Snowball are the two classical algorithms.
- Used in: classical search engines, BoW models where exact word form does not matter.
- If beginners look confused: "Stemming cuts the endings off words so 'run', 'runs', 'running' all become the same root."
- If experts look bored: "Stemming errors cascade into false positives in retrieval — 'universe' and 'university' both stem to 'univers'. BM25 implementations often skip stemming for this reason."

**Transition**: "The more linguistically correct alternative is lemmatisation..."

---

## Slide 20: Lemmatisation

**Time**: ~2 min
**Talking points**:

- Lemmatisation maps words to their dictionary form using morphological analysis. "better" → "good", "running" → "run", "mice" → "mouse".
- Requires a part-of-speech tagger — slower than stemming but produces real words.
- spaCy and NLTK both provide lemmatisers. In kailash-ml, PreprocessingPipeline's `lemmatize=True` flag handles this.
- If beginners look confused: "Lemmatisation is like looking up a word in the dictionary to find its base form."
- If experts look bored: "Language-specific lemmatisers vary significantly in quality — for non-English NLP, always benchmark lemmatisation accuracy before relying on it."

**Transition**: "Now let us look at n-grams, which capture multi-word phrases..."

---

## Slide 21: N-grams

**Time**: ~3 min
**Talking points**:

- An n-gram is a contiguous sequence of n tokens. Unigrams: individual words. Bigrams: pairs ("New York", "machine learning"). Trigrams: triples.
- BoW loses word order. N-grams partially restore it — "not good" is different from "good not".
- Trade-off: bigrams square the vocabulary size; trigrams cube it. Use sparingly.
- If beginners look confused: "N-grams capture phrases. 'New York' as a bigram is a single concept, not two separate words."
- If experts look bored: "The Kneser-Ney smoothed n-gram language model was the state of the art until 2013. Understanding it deeply helps you understand what neural LMs actually improved on."

**Transition**: "Text normalisation is the housekeeping step..."

---

## Slide 22: Text Normalisation

**Time**: ~2 min
**Talking points**:

- Lowercasing, punctuation removal, number handling, URL/email stripping, unicode normalisation (NFC vs NFD matters for accented characters).
- "Lowercasing is not always right — 'Apple' the company vs 'apple' the fruit. Named entity tasks should often preserve case."
- If beginners look confused: "Normalisation is cleaning up inconsistent formatting before analysis."
- If experts look bored: "Unicode normalisation catches subtle bugs — the same visually identical character can have two different byte representations. Always apply NFC normalisation."

**Transition**: "A key tool for normalisation is regular expressions..."

---

## Slide 23: Regex Patterns for NLP

**Time**: ~3 min
**Talking points**:

- Walk through the key patterns: URL removal `https?://\S+`, email removal `\S+@\S+`, number normalisation `\d+` → `NUM`, Twitter handles `@\w+`.
- "Regex is still the most reliable tool for rule-based text cleaning. Do not underestimate it."
- In kailash-ml: `PreprocessingPipeline(custom_regex=[(pattern, replacement)])` accepts custom patterns.
- If beginners look confused: "Regex is a mini-language for describing text patterns — like a very precise search."
- If experts look bored: "For high-throughput preprocessing, benchmark regex engine choice — Python's `re` vs `regex` vs Rust-backed `fancy-regex`. Throughput differences can be 10x."

**Transition**: "Let us consolidate everything into a pipeline summary..."

---

## Slide 24: Preprocessing Pipeline Summary

**Time**: ~3 min
**Talking points**:

- Show the complete pipeline diagram: raw text → tokenise → lowercase → strip noise → remove stops → lemmatise → n-grams → output.
- "For transformer models, you stop after tokenisation. For classical models, you run the full pipeline."
- kailash-ml code: `PreprocessingPipeline(tokenizer="wordpiece", lowercase=True, remove_stops=True, lemmatize=True).fit_transform(texts)`.
- If beginners look confused: "Think of this as a recipe. Transformers use the short recipe; classical models use the long one."
- If experts look bored: "Pipeline ordering matters — lemmatise before stop word removal or you may miss forms that lemmatise to stop words."

**Transition**: "Let us look at where people go wrong..."

---

## Slide 25: Preprocessing: Common Mistakes

**Time**: ~3 min
**Talking points**:

- Walk through the mistakes: (1) applying full classical pipeline to transformer input, (2) lowercasing named entities, (3) removing all punctuation including meaningful ones ("U.S.A."), (4) stripping numbers in financial/scientific text, (5) applying English stop words to multilingual corpus.
- "Every mistake here will silently degrade model performance — no error, just worse results."
- If beginners look confused: "These are the traps that experienced practitioners fall into. Now you know to avoid them."
- If experts look bored: "The multilingual stop word mistake is particularly pernicious — the word 'not' in French is 'ne...pas', split across the clause. Sentence-level stop word logic is required."

**Transition**: "Three key takeaways from Section A..."

---

## Slide 26: Section A: Key Takeaways

**Time**: ~2 min
**Talking points**:

- Three points: (1) tokenisation choice determines vocabulary size and OOV handling; (2) subword tokenisation (BPE/WordPiece) is the modern standard; (3) classical preprocessing steps are skipped for neural models.
- Quick check: "What tokeniser does BERT use? What tokeniser does GPT use?" (WordPiece vs BPE.)
- If beginners look confused: "Remember: for classical models, run the full pipeline. For transformers, the tokeniser does most of the work."

**Transition**: "Now let us look at the classical representation: Bag of Words and TF-IDF..."

---

## Slide 27: B. Bag of Words & TF-IDF (Section Header)

**Time**: ~1 min
**Talking points**:

- Section B: how to represent documents as numbers before neural methods existed.
- "BoW and TF-IDF still power production search engines and spam filters today. They are fast, interpretable, and surprisingly effective."
- If experts look bored: "Elasticsearch's relevance scoring is BM25 — we will get to that. Understanding it is required for search system design."

**Transition**: "The core idea of Bag of Words..."

---

## Slide 28: Bag of Words: The Idea

**Time**: ~3 min
**Talking points**:

- Every document becomes a vector of word counts. Vocabulary is fixed from the training corpus. "The cat sat on the mat" → [0, 1, 0, 1, 1, 1, 0, ...] indexed by vocabulary position.
- Called "bag" because word order is thrown away — "dog bites man" and "man bites dog" have identical BoW representations.
- Intuition: documents with similar word distributions tend to be about similar topics.
- If beginners look confused: "Imagine counting every word in a document and writing the counts in a table. That table IS the representation."
- If experts look bored: "The information loss from discarding order is substantial. BoW cannot distinguish 'the drug prevented cancer' from 'the drug caused cancer'. This is why it fails for sentiment and negation."

**Transition**: "What are the specific limitations?"

---

## Slide 29: BoW: Limitations

**Time**: ~3 min
**Talking points**:

- Walk through: (1) order lost — negation and syntax invisible; (2) high dimensionality — vocabulary size = vector length, typically 50K-100K; (3) sparsity — most documents use <1% of vocabulary; (4) no semantic similarity — "car" and "automobile" are unrelated vectors.
- "These limitations motivated every subsequent representation method we will study."
- If beginners look confused: "BoW is like judging a book by its index. You can see the topics but not the story."
- If experts look bored: "The sparsity problem is actually a feature for certain algorithms — sparse linear models (SGD classifier, SVM with linear kernel) are extremely fast and still competitive on short text classification."

**Transition**: "TF-IDF addresses the weighting problem..."

---

## Slide 30: TF-IDF: The Formula

**Time**: ~4 min
**Talking points**:

- TF = term frequency: how often does word w appear in document d? IDF = inverse document frequency: how rare is word w across the corpus?
- Formula: TF-IDF(w, d) = TF(w, d) × log(N / df(w)) where N = corpus size and df = document frequency.
- Intuition: "A word that appears often in one document but rarely in others is highly informative. Common words across all documents ('the', 'is') get downweighted."
- If beginners look confused: "TF rewards words that appear a lot in this document. IDF penalises words that appear in every document. Multiply them to get an importance score."
- If experts look bored: "The log in IDF is critical — without it, rare words dominate overwhelmingly. The log smooths the distribution. Variants exist: smooth IDF adds 1 to avoid division by zero."

**Transition**: "Let us work through a concrete example..."

---

## Slide 31: TF-IDF: Worked Example

**Time**: ~4 min
**Talking points**:

- Walk through the table step by step. Show TF computation for document 1, then IDF across corpus, then the product.
- "Follow the numbers. The word 'Singapore' has high TF in a Singapore-specific article but high IDF because it appears in fewer documents than 'the'."
- Pause and ask: "Which words do you expect to get high TF-IDF scores in a financial document?"
- If beginners look confused: "Walk through just one word first: pick 'revenue' — count it in one document, count how many documents contain it, then compute the score."
- If experts look bored: "The worked example demonstrates the saturation problem — a word appearing 100 times gets 10x the score of one appearing 10 times, even though the marginal information gain is much less. That is what sublinear TF addresses."

**Transition**: "That saturation problem leads us to sublinear TF..."

---

## Slide 32: Sublinear TF

**Time**: ~2 min
**Talking points**:

- Replace raw TF with 1 + log(TF). A word appearing 100 times gets score 1 + log(100) = 5.6, not 100.
- "The 50th occurrence of a word adds almost no new information. Sublinear TF captures diminishing returns."
- In kailash-ml: `AutoMLEngine(text_params={"sublinear_tf": True})`.
- If beginners look confused: "It is a smoothing trick — the first occurrence of a word matters most."
- If experts look bored: "Sublinear TF brings TF-IDF closer to BM25 saturation. For most corpora, enabling it improves precision without tuning."

**Transition**: "Speaking of search engines: BM25 is the production standard..."

---

## Slide 33: BM25: The Search Engine Standard

**Time**: ~4 min
**Talking points**:

- BM25 adds two improvements over TF-IDF: (1) saturation — term frequency saturates above a threshold k1; (2) length normalisation — longer documents are penalised relative to average document length (controlled by b).
- Used in: Elasticsearch, Solr, Lucene, Whoosh, BM25Okapi. "Every search engine you have used in the last 20 years uses a variant of BM25."
- Parameters: k1 (saturation, typically 1.2-2.0) and b (length normalisation, typically 0.75).
- If beginners look confused: "BM25 is TF-IDF with two fixes: word repetition has diminishing returns, and long documents are not unfairly boosted."
- If experts look bored: "BM25+ (Lü & Callan, 2011) addresses the BM25 weakness where a document with zero occurrences of a term can still outrank one with one occurrence due to length normalisation. Worth understanding before deploying in production."

**Transition**: "Let us visualise the saturation difference..."

---

## Slide 34: BM25 vs TF-IDF: Saturation

**Time**: ~2 min
**Talking points**:

- Show the graph: TF-IDF grows linearly with term frequency. BM25 plateaus.
- "At TF = 20, BM25 is essentially at its maximum score. TF-IDF continues climbing indefinitely."
- Ask the audience: "Which one gives a fairer comparison between a 100-word document and a 10,000-word document?"
- If beginners look confused: "The flat line means BM25 stops giving a document extra credit for repeating a word many times."
- If experts look bored: "The parameter k1 controls where the plateau occurs. Setting k1 = 0 degrades to pure IDF — useful as a baseline for short text classification."

**Transition**: "How do we use BoW and TF-IDF in practice?"

---

## Slide 35: BoW/TF-IDF in Practice

**Time**: ~3 min
**Talking points**:

- Walk through the kailash-ml code pattern: `AutoMLEngine(task="text_classification", vectorizer="tfidf")`.
- The engine handles: tokenisation, vocabulary building, TF-IDF matrix construction, model selection, cross-validation.
- Practical guidance: "Start with TF-IDF + logistic regression. It is a strong baseline that trains in seconds on millions of documents."
- If beginners look confused: "AutoMLEngine does all the plumbing — you give it text and labels, it returns a trained model."
- If experts look bored: "The TF-IDF + SGD classifier combination on hashing trick vectorisation can handle hundreds of millions of documents in memory-efficient streaming mode."

**Transition**: "When should you still use BoW instead of transformers?"

---

## Slide 36: When BoW Still Wins

**Time**: ~3 min
**Talking points**:

- BoW wins when: (1) inference speed matters — TF-IDF is microseconds, transformers are 50-500ms; (2) interpretability is required — you can inspect feature weights; (3) training data is very small (<100 examples) — transformers overfit; (4) text is very short and formulaic (spam, keyword matching).
- "Do not reach for BERT when TF-IDF is faster, interpretable, and achieves 95% of the performance."
- If beginners look confused: "BoW is like a fast, simple tool. You use it when the job does not require the heavy machinery."
- If experts look bored: "The latency argument is often decisive in production — a 100x inference speedup means you can use 100x larger batch sizes with the same infrastructure."

**Transition**: "Key takeaways from Section B..."

---

## Slide 37: Section B: Key Takeaways

**Time**: ~2 min
**Talking points**:

- Three points: (1) TF-IDF = term frequency × inverse document frequency — rewards rare but locally frequent terms; (2) BM25 adds saturation and length normalisation — used in every production search engine; (3) BoW/TF-IDF remains competitive for high-speed, interpretable, small-data scenarios.
- Quick comprehension check: "What does a high IDF score mean for a word?" (The word is rare across documents — carries distinctive information.)
- If beginners look confused: "TF-IDF is about finding the words that make each document unique."

**Transition**: "Section C: from sparse counts to dense meaning — word embeddings..."

---

## Slide 38: C. Word Embeddings (Section Header)

**Time**: ~1 min
**Talking points**:

- "The breakthrough idea: represent words as points in geometric space, where similar words are nearby."
- Section C covers: distributional hypothesis, Word2Vec, GloVe, FastText, contextual embeddings with ELMo.
- If experts look bored: "We will derive the skip-gram objective and analyse negative sampling. Pay attention to the connection between PMI and GloVe."

**Transition**: "The foundation is the distributional hypothesis..."

---

## Slide 39: The Distributional Hypothesis

**Time**: ~3 min
**Talking points**:

- Firth (1957): "You shall know a word by the company it keeps."
- Words that appear in similar contexts have similar meanings. "bank" appears near "money", "loan", "deposit" — similar to "lender" and "creditor".
- This is NOT hand-crafted — the meaning emerges from statistics over large corpora.
- If beginners look confused: "If two words are always used in the same kinds of sentences, they probably mean something similar."
- If experts look bored: "The distributional hypothesis is the theoretical foundation for both BoW (document-level co-occurrence) and Word2Vec (window-level co-occurrence). The difference is the window size."

**Transition**: "How do we go from sparse to dense representations?"

---

## Slide 40: From Sparse to Dense

**Time**: ~3 min
**Talking points**:

- BoW vectors: 50,000-dimensional, mostly zeros. Word2Vec vectors: 100-300 dimensional, all non-zero, geometrically meaningful.
- The transformation: instead of counting co-occurrences explicitly (PMI matrix), we learn a low-rank factorisation of co-occurrence structure.
- Analogy: "BoW is a passport photo list. Embeddings are a map where similar people are placed near each other."
- If beginners look confused: "Dense vectors pack the meaning of a word into about 300 numbers instead of 50,000."
- If experts look bored: "The connection between GloVe and PMI matrix factorisation (Levy & Goldberg, 2014) shows that Word2Vec implicitly factorises a shifted PMI matrix — all these methods are variants of the same underlying idea."

**Transition**: "Let us look at the most influential algorithm: Word2Vec skip-gram..."

---

## Slide 41: Word2Vec: Skip-gram

**Time**: ~4 min
**Talking points**:

- Skip-gram: given a centre word, predict the surrounding context words within a window.
- Architecture: centre word → lookup embedding → linear layer → softmax over vocabulary → predict context word.
- Training data is self-supervised: window slides over the corpus, generating (centre, context) pairs automatically.
- If beginners look confused: "Given the word 'cat', the model learns to predict that 'the', 'sat', 'on' are likely nearby."
- If experts look bored: "Skip-gram outperforms CBOW on rare words because predicting multiple context outputs creates stronger gradient signal for infrequent centre words."

**Transition**: "Let us look at the objective function formally..."

---

## Slide 42: Skip-gram: The Objective

**Time**: ~3 min
**Talking points**:

- Maximise: average log probability of context words given the centre word across the corpus.
- The softmax denominator sums over the entire vocabulary — this is computationally expensive. With 50K vocabulary, each gradient step requires 50K multiplications.
- "This is the scaling bottleneck. The solution is negative sampling."
- If beginners look confused: "We are training the model to say 'yes, these context words should appear near this centre word'."
- If experts look bored: "The original softmax training is hierarchical softmax in practice — Huffman-coded binary tree gives O(log V) instead of O(V). Negative sampling is conceptually cleaner."

**Transition**: "Negative sampling makes Word2Vec tractable..."

---

## Slide 43: Negative Sampling

**Time**: ~3 min
**Talking points**:

- Instead of computing full softmax, frame as binary classification: is this (word, context) pair real or random?
- Sample k random "negative" words (not in the window). Train to score real pairs high, random pairs low.
- k = 5-20 for small corpora, k = 2-5 for large corpora.
- "This converts O(V) per step to O(k) per step — a 1000x speedup for a 50K vocabulary."
- If beginners look confused: "Instead of testing all 50,000 words, we test 5 random wrong words vs the 1 correct one."
- If experts look bored: "The negative sampling distribution is raised to the 3/4 power — this gives more samples to medium-frequency words and prevents rare words from being systematically ignored."

**Transition**: "The other Word2Vec variant: CBOW..."

---

## Slide 44: Word2Vec: CBOW

**Time**: ~2 min
**Talking points**:

- CBOW (Continuous Bag of Words): given context words, predict the centre word. Inverse of skip-gram.
- Faster to train than skip-gram. Better for frequent words and large corpora.
- Skip-gram is generally preferred for rare word quality. CBOW is preferred for fast training at scale.
- If beginners look confused: "CBOW is the opposite of skip-gram — it reads the surrounding words to guess the missing middle word."
- If experts look bored: "The averaging operation in CBOW is its weakness — it discards positional information from context. This is why skip-gram generalises better to compositional phrases."

**Transition**: "GloVe offers a different approach with global statistics..."

---

## Slide 45: GloVe: Global Vectors

**Time**: ~3 min
**Talking points**:

- GloVe (Pennington et al., 2014): instead of training on local windows, explicitly factorises the global word co-occurrence matrix.
- Objective: dot product of two word vectors should approximate log of their co-occurrence count.
- Uses weighted least squares — frequent pairs get more weight but not unbounded weight.
- If beginners look confused: "GloVe learns embeddings by looking at how often every pair of words appears together across the whole corpus."
- If experts look bored: "GloVe's weighting function f(x) = (x/x_max)^alpha prevents the most frequent co-occurrences from dominating. The optimal alpha = 0.75 was found empirically — the same value used in Word2Vec negative sampling distribution."

**Transition**: "FastText handles an important limitation: morphology..."

---

## Slide 46: FastText: Subword Embeddings

**Time**: ~3 min
**Talking points**:

- Problem: Word2Vec and GloVe have no representation for OOV words or morphological variants.
- FastText (Bojanowski et al., 2017, Facebook): character n-grams instead of whole words. Word vector = sum of its character n-gram vectors.
- "running" uses: <run, runn, unni, nnin, ning, ing> plus the full word. OOV word "unnaturally" can be represented using its n-grams.
- If beginners look confused: "FastText builds word vectors from letter combinations, so it can handle words it has never seen before."
- If experts look bored: "FastText is still the best pre-trained option for many low-resource languages where morphological richness means high OOV rates. The Facebook pre-trained vectors cover 157 languages."

**Transition**: "The most famous property: vector arithmetic..."

---

## Slide 47: The Magic of Vector Arithmetic

**Time**: ~4 min
**Talking points**:

- The classic example: vec("king") - vec("man") + vec("woman") ≈ vec("queen").
- This is not programmed — it emerges from the statistics of text.
- Other analogies that hold: Paris:France :: Berlin:Germany; doctor:hospital :: teacher:school.
- Singapore-relevant: "Can you guess what: Orchard:Singapore :: Champs-Elysees:\_\_\_ gives?"
- If beginners look confused: "The fact that word vectors have this property means they have captured real-world relationships purely by reading text."
- If experts look bored: "The king-queen analogy works because Word2Vec learns a 'royalty' direction in embedding space that is approximately orthogonal to the 'gender' direction. This decomposability of meaning into geometric directions is a central property of linear representation learning."

**Transition**: "How do we measure similarity between word vectors?"

---

## Slide 48: Embedding Similarity: Cosine

**Time**: ~2 min
**Talking points**:

- Cosine similarity = dot product / (|a| × |b|). Ranges from -1 to 1. Direction matters, magnitude does not.
- "Two long documents have a big magnitude but might say the same thing — cosine ignores magnitude."
- Used for: nearest-neighbour word retrieval, document similarity, semantic search.
- If beginners look confused: "Cosine similarity measures the angle between two vectors — 0 degrees = identical direction, 90 degrees = unrelated, 180 degrees = opposite."
- If experts look bored: "For approximate nearest neighbour at scale, FAISS with HNSW indexing gives sub-millisecond lookup at billion-vector scale — the infrastructure behind most semantic search systems."

**Transition**: "A critical issue: word embeddings encode societal biases..."

---

## Slide 49: Bias in Word Embeddings

**Time**: ~4 min
**Talking points**:

- Word2Vec trained on Google News encodes: man:doctor :: woman:nurse, man:programmer :: woman:homemaker.
- This is not a model bug — it reflects the biases present in the training text.
- "The model is a statistical mirror of the data. If the data is biased, the model is biased."
- Production consequences: a CV screening system using biased embeddings will systematically downrank qualified women.
- If beginners look confused: "The AI learned from human-written text that had biases in it — so it learned those biases."
- If experts look bored: "The Bolukbasi et al. (2016) debiasing paper showed that gender is encoded on a single principal direction in embedding space. Hard debiasing projects out this direction; soft debiasing reduces its salience. Both are imperfect — Gonen & Goldberg (2019) showed male-biased words still cluster together after debiasing."

**Transition**: "What can we do about it?"

---

## Slide 50: Debiasing Approaches

**Time**: ~3 min
**Talking points**:

- Three approaches: (1) Hard debiasing — identify bias subspace, project it out geometrically; (2) Data augmentation — add gender-swapped training examples; (3) Counterfactual data substitution — replace protected attribute references during training.
- "None of these is a complete solution. Bias mitigation is active research."
- Production guidance: audit embeddings before deployment, especially for hiring, lending, or criminal justice applications.
- If beginners look confused: "Debiasing tries to remove the gender direction from word vectors so 'programmer' is equidistant from 'man' and 'woman'."
- If experts look bored: "The evaluation of debiasing is unsolved — WEAT (Word Embedding Association Test) measures bias but debiased models can still exhibit bias on downstream tasks without WEAT flagging it."

**Transition**: "The limitations of static embeddings led to ELMo: contextual embeddings..."

---

## Slide 51: ELMo: Contextual Embeddings

**Time**: ~4 min
**Talking points**:

- Static embeddings: "bank" has one vector regardless of context. ELMo: the representation of "bank" differs in "river bank" vs "bank account".
- ELMo uses a bidirectional LSTM language model. The embedding is the concatenation of hidden states across all layers.
- "Context-dependence is the key insight that BERT and all transformers build on."
- ELMo gave massive improvements on NLP benchmarks in 2018 — SQuAD reading comprehension, NER, semantic role labelling.
- If beginners look confused: "ELMo gives each word a different vector depending on the surrounding words — 'bank' by a river gets a different representation than 'bank' in a financial sentence."
- If experts look bored: "ELMo's layer weighting is trainable per task — the optimal combination of shallow (syntax) vs deep (semantics) layers differs by task. This insight was preserved in BERT's [CLS] token fine-tuning design."

**Transition**: "Let us look at what individual dimensions encode..."

---

## Slide 52: Embedding Dimensions: What They Encode

**Time**: ~2 min
**Talking points**:

- Through probing classifiers and PCA, researchers have found embedding dimensions encode: gender, tense, plurality, sentiment polarity, concreteness.
- "The model did not know these categories existed. It discovered them because they are statistically useful for predicting context."
- This is interpretability research — understanding what is in the representation before relying on it in production.
- If beginners look confused: "Researchers can look inside word vectors and find that certain numbers track whether a word is positive or negative."
- If experts look bored: "Probing classifiers are the primary interpretability tool for embeddings — a linear probe tests if a property is linearly decodable from a representation. Non-linear probes test for non-linear encoding at the cost of attribution clarity."

**Transition**: "Practical details for training Word2Vec..."

---

## Slide 53: Training Word2Vec: Practical Details

**Time**: ~3 min
**Talking points**:

- Key hyperparameters: window size (5 for syntax, 10+ for semantics), embedding dimension (100-300), negative samples (5-20), subsampling threshold for frequent words.
- Training time: 100M words → ~10 minutes on a modern CPU with gensim. 1B words → a few hours.
- In kailash-ml: `AutoMLEngine(task="text_embedding", algo="word2vec", dim=200, window=5).fit(corpus)`.
- If beginners look confused: "These are the settings you dial when training your own word vectors — start with the defaults."
- If experts look bored: "The subsampling of frequent words is the underrated hyperparameter — it effectively increases context window for rare words by removing high-frequency noise from windows."

**Transition**: "A closer look at GloVe's co-occurrence ratio intuition..."

---

## Slide 54: GloVe: Co-occurrence Ratios

**Time**: ~2 min
**Talking points**:

- GloVe is motivated by ratios of co-occurrence probabilities, not raw counts. P("ice" | "solid") / P("steam" | "solid") is high — "solid" is associated with ice, not steam.
- "GloVe encodes the relationship between concepts by looking at how their contextual patterns differ."
- If beginners look confused: "GloVe finds word meaning by comparing which contexts distinguish between similar words."
- If experts look bored: "The log-bilinear model structure means GloVe embeddings can be added and subtracted linearly for analogical reasoning — this is the mathematical reason the king-queen analogy works better with GloVe than Word2Vec in some benchmarks."

**Transition**: "How do we evaluate embedding quality?"

---

## Slide 55: Evaluating Embeddings

**Time**: ~3 min
**Talking points**:

- Two evaluation types: intrinsic (embedding space geometry) and extrinsic (downstream task performance).
- Intrinsic: word analogy tasks (Google Analogy Dataset, BATS), word similarity tasks (SimLex-999, WordSim-353).
- Extrinsic: plug into NER, sentiment, QA and measure task performance.
- "Intrinsic scores often do not predict extrinsic performance — always evaluate on your target task."
- If beginners look confused: "Test the quality of word vectors both directly (do 'king - man + woman = queen'?) and indirectly (does sentiment classification improve when using them?)."
- If experts look bored: "The SimLex-999 / WordSim-353 correlation doesn't generalise across domains — an embedding that scores well on general analogies may perform poorly on financial or biomedical analogical reasoning."

**Transition**: "How to use pre-trained embeddings in practice..."

---

## Slide 56: Pre-trained Embeddings: When and How

**Time**: ~3 min
**Talking points**:

- Three strategies: (1) frozen pre-trained — use as feature extraction, fast and cheap; (2) fine-tuned — update during training, better adaptation, risk of catastrophic forgetting; (3) domain-adaptive pre-training — continue training on domain corpus.
- Rule of thumb: small labelled dataset → freeze. Large labelled dataset → fine-tune. Domain-specific vocabulary → domain pre-training.
- In kailash-ml: `TrainingPipeline(embedding="glove-300d", freeze_embeddings=True)`.
- If beginners look confused: "Using pre-trained embeddings is like getting a head start — someone already trained the word vectors on billions of words."
- If experts look bored: "The ULMFiT training protocol (Howard & Ruder, 2018) introduced discriminative fine-tuning and slanted triangular learning rates for embedding adaptation — these techniques directly preceded BERT fine-tuning."

**Transition**: "Section C key takeaways..."

---

## Slide 57: Section C: Key Takeaways

**Time**: ~2 min
**Talking points**:

- Three points: (1) word embeddings place similar words nearby in geometric space — meaning emerges from co-occurrence statistics; (2) Word2Vec/GloVe are static; ELMo/transformers are contextual; (3) embeddings inherit and amplify societal biases — audit before deployment.
- Check: "What is the distributional hypothesis?" (Words that appear in similar contexts have similar meanings.)
- If beginners look confused: "Word vectors turn meaning into geometry. Similar meanings → nearby points."

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

**Time**: ~4 min
**Talking points**:

- Hidden state: h*t = tanh(W_h × h*{t-1} + W_x × x_t + b). Output: y_t = W_y × h_t.
- The recurrent connection: h*t depends on h*{t-1}. The hidden state is a running summary of all inputs seen so far.
- "The hidden state is the RNN's memory. Every step it reads the new word and updates its memory."
- Walk through: h_0 = zeros → h_1 after "The" → h_2 after "cat" → h_3 after "sat".
- If beginners look confused: "The RNN reads words one at a time. After each word, it updates a 'memory vector' that summarises everything read so far."
- If experts look bored: "The tanh nonlinearity bounds activations to [-1, 1] — this is specifically chosen to handle the recurrent multiplication, but it is the root of the vanishing gradient problem."

**Transition**: "RNNs can be configured in different architectures..."

---

## Slide 60: RNN Types

**Time**: ~3 min
**Talking points**:

- Walk through the five types with diagrams: one-to-one (MLP, not really RNN), one-to-many (image captioning), many-to-one (sentiment classification), many-to-many aligned (POS tagging), many-to-many unaligned (translation).
- "The input-output architecture is a design decision based on the task structure."
- If beginners look confused: "Sentiment classification is many-to-one: read many words, output one sentiment score."
- If experts look bored: "The many-to-many unaligned case (seq2seq) requires the encoder-decoder split we will see later. The output sequence can have different length than the input."

**Transition**: "How do we train RNNs?"

---

## Slide 61: Backpropagation Through Time (BPTT)

**Time**: ~4 min
**Talking points**:

- BPTT unrolls the RNN across time steps and backpropagates through each step. For a 100-word sentence, gradients must flow through 100 multiplication steps.
- The shared weight matrix W_h is multiplied by itself 100 times during backprop.
- "If the matrix has eigenvalues > 1: gradients explode. If eigenvalues < 1: gradients vanish."
- Truncated BPTT: only backpropagate through k steps. Trades sequence length for training stability.
- If beginners look confused: "Training a RNN on a long sentence means the gradient has to travel backwards through every word. This creates problems."
- If experts look bored: "The BPTT memory requirement is O(sequence length × hidden size) — at inference time, RNNs are O(1) memory per step. This is the fundamental tradeoff versus transformers."

**Transition**: "The vanishing gradient is the critical failure mode..."

---

## Slide 62: The Vanishing Gradient Problem

**Time**: ~4 min
**Talking points**:

- Formally: if |dh*t/dh*{t-1}| < 1 consistently, then |dL/dh_1| → 0 exponentially as sequence length grows.
- Consequence: early words in a sentence have almost no gradient signal. "The RNN cannot learn that 'bank' at position 1 affects the meaning of 'loan' at position 50."
- Show the intuition: tanh derivative is at most 0.25. After 50 steps: 0.25^50 ≈ 10^{-31}. Effectively zero.
- If beginners look confused: "The RNN 'forgets' words from the beginning of a long sentence because the training signal cannot reach that far back."
- If experts look bored: "The gradient signal also provides information about the curvature of the loss landscape. Vanishing gradients mean you are optimising in a region with essentially flat loss surface — saddle points, not just local minima."

**Transition**: "LSTM was designed specifically to solve this..."

---

## Slide 63: LSTM: Long Short-Term Memory

**Time**: ~4 min
**Talking points**:

- Hochreiter & Schmidhuber (1997). The key innovation: a separate cell state c_t that flows through the network with additive updates, not multiplicative.
- Additive updates mean gradients can flow backward without vanishing. "The cell state is a highway for gradient flow."
- Three gates control the cell state: forget gate (erase), input gate (write), output gate (read).
- If beginners look confused: "LSTM has two memory systems: short-term (hidden state) and long-term (cell state). The gates decide what to keep, what to write, and what to output."
- If experts look bored: "The LSTM's gradient stability is guaranteed by the additive cell state update — this is the key design principle that Transformers later adopted with residual connections."

**Transition**: "Let us understand each gate in detail..."

---

## Slide 64: LSTM: Forget Gate

**Time**: ~2 min
**Talking points**:

- f*t = sigmoid(W_f × [h*{t-1}, x_t] + b_f). Output is between 0 and 1 for each cell state dimension.
- f_t = 0: completely forget. f_t = 1: completely remember.
- Example: in machine translation, when you read a period, the forget gate fires to clear sentence-level state.
- If beginners look confused: "The forget gate decides which parts of memory to erase when a new word comes in."
- If experts look bored: "The forget gate was not in the original Hochreiter & Schmidhuber LSTM — it was added by Gers et al. (2000). The original formulation relied purely on constant error carousel."

**Transition**: "What does the LSTM write?"

---

## Slide 65: LSTM: Input Gate & Candidate

**Time**: ~2 min
**Talking points**:

- Input gate: i*t = sigmoid(W_i × [h*{t-1}, x_t] + b_i) — what to write.
- Candidate: g*t = tanh(W_g × [h*{t-1}, x_t] + b_g) — what content to write.
- "The input gate asks 'should we write?' The candidate cell asks 'what should we write?'"
- If beginners look confused: "Two decisions: whether to update memory, and what the new content should be."
- If experts look bored: "The separation of gating (sigmoid) and content generation (tanh) is the key modularity. This pattern — gate × content — appears again in the multiplicative attention mechanism."

**Transition**: "Now we combine forget and input to update the cell state..."

---

## Slide 66: LSTM: Cell State Update

**Time**: ~2 min
**Talking points**:

- c*t = f_t ⊙ c*{t-1} + i_t ⊙ g_t. Element-wise operations.
- "The cell state is updated additively — this is the gradient highway."
- The additive structure means: dL/dc\_{t-k} = dL/dc_t × product of f_j for j from t to t-k. If forget gates stay near 1, gradients flow cleanly.
- If beginners look confused: "The cell state mixes the old memory (scaled by forget) with new content (scaled by input gate)."
- If experts look bored: "The gradient of c\_{t-k} with respect to c_t is the product of forget gates — not of the full weight matrix. This is the fundamental difference from vanilla RNN."

**Transition**: "Finally, the output..."

---

## Slide 67: LSTM: Output Gate & Hidden State

**Time**: ~2 min
**Talking points**:

- Output gate: o*t = sigmoid(W_o × [h*{t-1}, x_t] + b_o) — what to expose.
- Hidden state: h_t = o_t ⊙ tanh(c_t) — the actual output.
- "The output gate controls what the LSTM shows to the outside world. The cell state holds more information than is exposed at each step."
- If beginners look confused: "The LSTM can remember something without outputting it — the output gate controls visibility."
- If experts look bored: "The hidden state h_t is often called 'short-term memory' vs c_t 'long-term memory'. This distinction breaks down for very long sequences — LSTMs still struggle at 1000+ tokens."

**Transition**: "Let us see all four equations together..."

---

## Slide 68: LSTM: Complete Equations

**Time**: ~3 min
**Talking points**:

- Write all four equations side by side. Show that all four gates share the same input structure — they differ only in their weight matrices.
- "Notice: four separate weight matrices. LSTM has 4x the parameters of a vanilla RNN for the same hidden size."
- Common optimisation: concatenate all four gate computations into one matrix multiplication with 4x output, then split. This is what every GPU-optimised LSTM implementation does.
- If beginners look confused: "The key equation is the cell state update — the rest are controllers for it."
- If experts look bored: "cuDNN's LSTM kernel fuses all four gate computations into a single GEMM followed by a split-and-activate. Understanding this is required to benchmark LSTM vs transformer training throughput fairly."

**Transition**: "GRU is a streamlined variant..."

---

## Slide 69: GRU: Gated Recurrent Unit

**Time**: ~3 min
**Talking points**:

- GRU (Cho et al., 2014): merges cell state and hidden state into one. Two gates instead of three: reset gate and update gate.
- Update gate: z*t. Reset gate: r_t. New hidden: h_t = (1-z_t) ⊙ h*{t-1} + z_t ⊙ h~\_t.
- "GRU is LSTM minus the output gate and cell state split. 33% fewer parameters, similar performance on most tasks."
- If beginners look confused: "GRU is a simplified LSTM that is faster to train. In practice, they perform similarly."
- If experts look bored: "The empirical comparison (Chung et al., 2014, Greff et al., 2017) shows LSTM/GRU parity on most tasks. GRU is preferred when training speed matters. Notably, the reset gate in GRU is more interpretable than LSTM's forget gate."

**Transition**: "For sequence labelling tasks, we often need both past and future context..."

---

## Slide 70: Bidirectional RNNs

**Time**: ~2 min
**Talking points**:

- Run two RNNs: one left-to-right, one right-to-left. Concatenate hidden states at each position.
- "The word 'bank' in 'river bank restoration' — the word 'restoration' to the right tells you it is a river bank, not a financial one. A bidirectional model can use this."
- Used heavily in BERT's architecture (though implemented via attention, not recurrence).
- If beginners look confused: "Bidirectional means the model reads the sentence forwards and backwards simultaneously."
- If experts look bored: "Bidirectional inference is incompatible with autoregressive generation — you cannot read the right context for words you have not generated yet. This is why GPT is unidirectional and BERT is bidirectional."

**Transition**: "We can also stack multiple layers..."

---

## Slide 71: Multi-Layer and Deep RNNs

**Time**: ~2 min
**Talking points**:

- Stack multiple RNN layers: output of layer L becomes input of layer L+1. Typically 2-4 layers in practice.
- Lower layers capture local syntax; higher layers capture semantics and long-range dependencies.
- "Deep RNNs are powerful but slow — each layer must be processed sequentially across time."
- If beginners look confused: "Just like deep neural networks in Module 7, we can stack multiple RNN layers."
- If experts look bored: "The depth-vs-width tradeoff for RNNs: wider hidden states vs more layers. Empirically, 2-4 layers with moderate width outperforms 1 deep layer. Beyond 4 layers, gradients in the depth direction also begin to vanish."

**Transition**: "The most important RNN architecture: seq2seq encoder-decoder..."

---

## Slide 72: Seq2Seq: Encoder-Decoder

**Time**: ~4 min
**Talking points**:

- Encoder: process input sequence into a fixed-size context vector (final hidden state).
- Decoder: autoregressively generate output sequence conditioned on context vector.
- Application: machine translation. "How are you?" → context vector → "Comment allez-vous?"
- The bottleneck: the entire input sentence must be compressed into a single vector. "Translate a 100-word paragraph into a single 512-number summary — information is lost."
- If beginners look confused: "Seq2seq reads the whole input sentence and encodes it into a 'meaning vector', then a decoder reads that vector to produce the output sentence."
- If experts look bored: "The fixed-size bottleneck is the fundamental architectural limitation. Bahdanau (2015) solved it with attention — instead of one context vector, the decoder attends to all encoder hidden states dynamically."

**Transition**: "Let us see RNNs in action on sentiment analysis..."

---

## Slide 73: RNNs in Practice: Sentiment Analysis

**Time**: ~3 min
**Talking points**:

- Walk through the kailash-ml code: load pre-trained embeddings, build LSTM model via TrainingPipeline, train on labelled reviews, evaluate.
- `TrainingPipeline(model_type="lstm", embedding="glove-300d", bidirectional=True, layers=2)`
- Typical performance: 85-88% accuracy on SST-2 with BiLSTM. BERT: 93%+. "You can see why transformers won."
- If beginners look confused: "This is what it looks like to use an LSTM in code — most of the work is handled by TrainingPipeline."
- If experts look bored: "The TrainingPipeline handles pack_padded_sequence — variable-length batching that avoids padding waste and gives correct BPTT gradients."

**Transition**: "A training trick that dramatically improves RNN convergence: teacher forcing..."

---

## Slide 74: Teacher Forcing

**Time**: ~2 min
**Talking points**:

- During training: use the ground truth token as input at each decoder step, not the model's previous prediction.
- Benefit: clean gradient signal, faster convergence. Problem: exposure bias — at inference, the model sees its own (potentially wrong) outputs.
- Scheduled sampling: mix teacher forcing and free running during training to reduce exposure bias.
- If beginners look confused: "Teacher forcing is like showing a student the correct answer during practice, then expecting them to answer on their own in the exam."
- If experts look bored: "The exposure bias problem motivated curriculum learning and scheduled sampling (Bengio et al., 2015). In practice, most production seq2seq systems use pure teacher forcing despite the theory — the improvement from scheduled sampling is task-dependent."

**Transition**: "Gradient clipping prevents the other gradient problem: explosion..."

---

## Slide 75: Gradient Clipping for RNNs

**Time**: ~2 min
**Talking points**:

- Vanishing gradient causes forgetting. Exploding gradient causes divergence — parameters jump wildly.
- Gradient clipping: if gradient norm > threshold, scale gradient down to threshold.
- "If gradient norm > 5, clip it to 5. Simple but highly effective."
- Standard in all production RNN training. kailash-ml TrainingPipeline applies it automatically (default max_norm=5.0).
- If beginners look confused: "Gradient clipping is a safety check that prevents training from exploding when it encounters a very steep region of the loss landscape."
- If experts look bored: "The threshold value matters — too small and learning stalls, too large and it doesn't protect. For LSTMs on text, 1.0-5.0 works well; for RNNs on raw waveforms, 0.1-1.0 is typical."

**Transition**: "Let us consolidate the limitations before moving to attention..."

---

## Slide 76: RNN Limitations Summary

**Time**: ~3 min
**Talking points**:

- Walk through the four limitations: (1) sequential processing — cannot parallelise across time steps during training; (2) long-range dependencies — even LSTM struggles beyond 500 tokens; (3) fixed-capacity bottleneck in seq2seq; (4) training instability requiring clipping and careful initialisation.
- "All four of these limitations are solved by the transformer. That is why we moved on."
- If beginners look confused: "RNNs are powerful but have fundamental speed and memory limits that the next architecture fixes."
- If experts look bored: "The parallelism argument is key — LSTM training on 1M tokens is 10-100x slower than a transformer because every time step depends on the previous one. This bottleneck prevented scaling."

**Transition**: "Section D key takeaways..."

---

## Slide 77: Section D: Key Takeaways

**Time**: ~2 min
**Talking points**:

- Three points: (1) RNNs process sequences with a hidden state but suffer from vanishing gradients for long sequences; (2) LSTM solves vanishing gradients with gated cell state — the additive update is the key innovation; (3) seq2seq enables variable-length input-output but the fixed bottleneck limits quality.
- Check: "What makes LSTM better than vanilla RNN for long sequences?" (Additive cell state update allows gradient flow without vanishing.)
- If beginners look confused: "LSTM's key trick: memory updates by adding, not multiplying — this keeps gradients healthy."

**Transition**: "Section E: attention — the breakthrough that made transformers possible..."

---

## Slide 78: E. Attention Mechanisms (Section Header)

**Time**: ~1 min
**Talking points**:

- "Attention is the single most important idea in modern NLP. Everything that follows — BERT, GPT, every LLM — is built on it."
- Section E: Bahdanau attention, self-attention, QKV formulation, multi-head attention.
- If experts look bored: "We will derive scaled dot-product attention from the QKV dictionary lookup analogy and prove why scaling by sqrt(d_k) is necessary."

**Transition**: "The original attention paper: Bahdanau et al. (2015)..."

---

## Slide 79: Bahdanau Attention (2015)

**Time**: ~4 min
**Talking points**:

- Problem: seq2seq compresses entire input to one vector. Bahdanau's solution: let the decoder look back at all encoder hidden states.
- For each decoder step t: compute alignment scores e*{t,i} = a(s*{t-1}, h_i) for each encoder position i. Softmax to get attention weights alpha. Context vector = weighted sum of encoder states.
- "At each step, the decoder asks: which encoder words should I focus on right now?"
- Showed dramatic improvement in long-sentence translation. Visualising attention weights revealed interpretable alignment patterns.
- If beginners look confused: "Instead of reading one summary of the input, the decoder can look back at any part of the input at each step."
- If experts look bored: "The alignment function a() was a single-layer MLP in Bahdanau's paper. Luong (2015) simplified it to dot product, which is the foundation of scaled dot-product attention. The transition from additive to multiplicative attention enabled the transformer."

**Transition**: "What does attention look like visually?"

---

## Slide 80: Attention Visualised

**Time**: ~2 min
**Talking points**:

- Show the heatmap: rows = decoder words (English output), columns = encoder words (French input). Bright cells = high attention.
- "Notice the diagonal pattern for word-by-word translation — and the off-diagonal for reordering (adjective-noun order differs between languages)."
- This visualisation was the first interpretable window into what neural sequence models were doing.
- If beginners look confused: "Each row shows what the decoder is looking at when generating each output word."
- If experts look bored: "Attention visualisation is qualitatively useful but Jain & Wallace (2019) showed attention weights are not reliable explanations — high attention on a token does not necessarily mean it is causally important to the prediction."

**Transition**: "Self-attention generalises this: let every word attend to every other word..."

---

## Slide 81: Self-Attention: Q, K, V

**Time**: ~5 min
**Talking points**:

- In self-attention, the query, key, and value all come from the same input sequence.
- Three linear projections: Q = X W_Q, K = X W_K, V = X W_V.
- Intuition: Q = "what am I looking for?", K = "what do I advertise?", V = "what do I actually share?"
- Attention output = softmax(Q K^T / sqrt(d_k)) V.
- "Every position simultaneously queries every other position. This enables direct long-range dependencies in a single operation."
- If beginners look confused: "Self-attention lets each word look at all other words and decide how much to borrow from each one."
- If experts look bored: "The QKV decomposition is a generalisation of memory-addressing. Q selects the memory location (key), V is the memory content. The linear projections allow the model to learn different attention patterns for different tasks."

**Transition**: "Let us write out the full formula..."

---

## Slide 82: Scaled Dot-Product Attention

**Time**: ~4 min
**Talking points**:

- Full formula: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.
- Walk through step by step: (1) Q K^T gives a score matrix — how much does each query match each key; (2) divide by sqrt(d_k); (3) softmax normalises to attention weights; (4) V weighted sum gives output.
- This operation is O(n² d) for sequence length n and dimension d.
- If beginners look confused: "Step 1: compute scores. Step 2: normalise scores. Step 3: use scores to mix values."
- If experts look bored: "The output is a weighted sum of value vectors — this is linear interpolation in value space. The attention weights determine the mixing coefficients. Because V is also learned, the model can choose what information to store in value space."

**Transition**: "Why scale by square root of d_k?"

---

## Slide 83: Why Scale by sqrt(d_k)?

**Time**: ~3 min
**Talking points**:

- As d_k grows, dot products Q K^T grow in variance. With d_k = 512, dot products have standard deviation ~22.
- Without scaling, softmax saturates in extreme values: all probability mass on one token (hard attention).
- Hard attention: gradient through softmax is ~0 everywhere except the top-1 — model stops learning.
- Dividing by sqrt(d_k) keeps dot product variance constant at 1 regardless of d_k.
- If beginners look confused: "Without scaling, the attention scores get very large and softmax becomes winner-takes-all. Dividing by the square root keeps scores in a reasonable range."
- If experts look bored: "This is the same problem as the vanishing gradient in deep networks — softmax saturation is a gradient highway collapse. Temperature scaling at inference time exploits this: temperature < 1 sharpens, temperature > 1 flattens attention."

**Transition**: "Let us trace through a small worked example..."

---

## Slide 84: Self-Attention: Worked Example

**Time**: ~4 min
**Talking points**:

- Walk through the 3-token example numerically. Show Q, K, V matrices. Compute Q K^T. Apply scaling and softmax. Multiply by V.
- "Do not just watch — trace your finger through each step on the slide."
- After the exercise: "Notice that each output vector is a unique combination of all input vectors — this is how information mixes."
- If beginners look confused: "The numbers in the softmax row tell you: 'this much from word 1, this much from word 2, this much from word 3'."
- If experts look bored: "The worked example is for d_k = 3 — in practice d_k = 64 per head (for BERT-base with 12 heads and d_model = 768). Numerically the operation is identical."

**Transition**: "Multi-head attention runs several attention operations in parallel..."

---

## Slide 85: Multi-Head Attention

**Time**: ~4 min
**Talking points**:

- Run h attention heads in parallel, each with different learned W_Q, W_K, W_V projections.
- Concatenate outputs, then project: MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W_O.
- "Different heads learn different types of relationships: one head for syntax, another for coreference, another for positional proximity."
- BERT-base: 12 heads × 64-dim = 768-dim total. GPT-3: 96 heads × 128-dim = 12288-dim.
- If beginners look confused: "Multi-head is like having 12 different attention patterns running simultaneously — each one looks for something different."
- If experts look bored: "Probing work (Clark et al., 2019) found specific BERT heads correspond to specific syntactic relations — head 10 in layer 2 tracks subject-verb agreement. This is remarkable emergence without explicit syntactic supervision."

**Transition**: "Why is attention fundamentally better than RNN?"

---

## Slide 86: Attention vs RNN: The Key Advantages

**Time**: ~3 min
**Talking points**:

- Four advantages: (1) parallelism — all positions computed simultaneously; (2) path length — any two tokens have direct connection (O(1) vs O(n) for RNN); (3) no fixed bottleneck; (4) interpretable attention weights.
- "In an RNN, information from position 1 must travel through 99 multiplications to reach position 100. In attention, it arrives in one step."
- If beginners look confused: "Attention connects every word to every other word directly. RNNs pass information through a chain."
- If experts look bored: "The O(n²) complexity of attention is its weakness at long sequences. Linear attention variants (Performer, Reformer) approximate the softmax kernel to reduce to O(n) — active research area."

**Transition**: "Self-attention versus cross-attention..."

---

## Slide 87: Cross-Attention vs Self-Attention

**Time**: ~2 min
**Talking points**:

- Self-attention: Q, K, V all from the same sequence. Used in the encoder and in the decoder's self-attention sublayer.
- Cross-attention: Q from one sequence (decoder), K and V from another sequence (encoder). This is how the decoder reads the encoder in transformers.
- "Cross-attention is the modern implementation of Bahdanau attention inside the transformer."
- If beginners look confused: "Self-attention is a sequence talking to itself. Cross-attention is one sequence asking questions of another."
- If experts look bored: "In the full transformer, three types of attention appear: encoder self-attention (bidirectional), decoder self-attention (causal/masked), and encoder-decoder cross-attention. The masking pattern is different for each."

**Transition**: "A taxonomy of attention variants..."

---

## Slide 88: Attention Types: A Taxonomy

**Time**: ~2 min
**Talking points**:

- Walk through: additive (Bahdanau), dot-product (Luong), scaled dot-product (Vaswani), multi-head, sparse (Longformer), local, global.
- "The taxonomy helps you choose: most tasks use scaled dot-product multi-head. Long documents need sparse or local."
- If beginners look confused: "Different situations call for different attention patterns — this map shows your options."
- If experts look bored: "Linear attention (Katharopoulos et al., 2020) rewrites softmax as a kernel product: phi(Q) × (phi(K)^T V) evaluated right-to-left gives O(n) complexity. The trade-off is approximation quality."

**Transition**: "An elegant way to think about attention: soft dictionary lookup..."

---

## Slide 89: Attention as Soft Dictionary Lookup

**Time**: ~3 min
**Talking points**:

- Hard dictionary: look up key, return exact value. Soft dictionary: partial matches return a weighted average of all values.
- Python dict is hard lookup. Attention is the differentiable generalisation.
- "You can also think of it as a differentiable database query: Q is the query, K is the index, V is the stored data."
- If beginners look confused: "Imagine a fuzzy search — instead of finding the exact match, you blend multiple partial matches."
- If experts look bored: "This framing connects attention to the Hopfield network literature. Ramsauer et al. (2020) showed that modern Hopfield networks converge to the attention update rule — connecting energy-based memory to transformers."

**Transition**: "How does the computational cost scale?"

---

## Slide 90: Attention Complexity Analysis

**Time**: ~3 min
**Talking points**:

- Time: O(n² d). Memory: O(n²) for the attention matrix. For n = 1024, the attention matrix is 1M floats × 4 bytes = 4 MB per layer per batch element.
- "GPT-3 with 96 layers, 96 heads, and sequence length 2048 — the attention matrices alone are enormous before you count activations."
- This is why Flash Attention (which we cover in the transformer section) is so important.
- If beginners look confused: "Attention gets expensive very fast with long sequences — the cost grows quadratically."
- If experts look bored: "The memory bottleneck at long sequences is HBM bandwidth, not compute. Flash Attention's tile-based approach avoids materialising the full attention matrix, reducing memory from O(n²) to O(n) at the cost of recomputation in the backward pass."

**Transition**: "Section E key takeaways..."

---

## Slide 91: Section E: Key Takeaways

**Time**: ~2 min
**Talking points**:

- Three points: (1) self-attention computes query-key-value operations allowing every position to directly attend to every other; (2) multi-head attention runs h parallel attention patterns; (3) attention gives O(1) path length between any two tokens, solving the long-range dependency problem.
- Check: "What are Q, K, and V?" (Query: what I'm looking for; Key: what I advertise; Value: what I share.)
- If beginners look confused: "Attention lets words talk to each other directly. No more message passing through a chain."

**Transition**: "Section F: putting it all together — the Transformer architecture..."

---

## Slide 92: F. The Transformer (Section Header)

**Time**: ~1 min
**Talking points**:

- "'Attention Is All You Need' — Vaswani et al., 2017. The most cited ML paper in history."
- Section F: full architecture, encoder/decoder blocks, positional encoding, FFN, training, Flash Attention, GQA.
- If experts look bored: "We will cover all the modern improvements beyond the original paper: RoPE, ALiBi, Flash Attention, GQA, Pre-LayerNorm."

**Transition**: "The full transformer architecture..."

---

## Slide 93: Transformer: High-Level Architecture

**Time**: ~5 min
**Talking points**:

- Walk through the architecture diagram from the original paper: input embeddings + positional encoding → N encoder blocks → N decoder blocks → linear + softmax.
- Encoder block: multi-head self-attention → add & norm → feed-forward → add & norm.
- Decoder block: masked multi-head self-attention → add & norm → cross-attention → add & norm → feed-forward → add & norm.
- "The encoder reads the entire input. The decoder generates output one token at a time, attending to the encoder."
- If beginners look confused: "The left side (encoder) reads the input language. The right side (decoder) writes the output language."
- If experts look bored: "The original transformer used 6 encoder layers and 6 decoder layers for translation. Modern models use encoder-only (BERT: 12/24 layers) or decoder-only (GPT: 12/96/96 layers). The full encoder-decoder is used in T5, BART, mT5."

**Transition**: "Let us zoom into the encoder block..."

---

## Slide 94: Encoder Block: Detailed

**Time**: ~4 min
**Talking points**:

- Two sublayers: (1) multi-head self-attention, (2) position-wise feed-forward network. Each sublayer has a residual connection and layer normalisation.
- The sublayers are independent: "Self-attention mixes information across positions. FFN processes each position independently."
- Residual connection: output = sublayer(x) + x. "The original input is always preserved — transformations are additive."
- If beginners look confused: "Think of each encoder block as a refinement step — it reads the representations from the previous block and produces better ones."
- If experts look bored: "The encoder is fully bidirectional — no masking. In decoder-only models like GPT, the self-attention is masked to be causal. This single change converts an encoder to a generative model."

**Transition**: "Residual connections deserve special attention..."

---

## Slide 95: Residual Connections

**Time**: ~3 min
**Talking points**:

- From ResNets (He et al., 2016): output = F(x) + x. The skip connection creates an identity path for gradients.
- Without residual: gradients in a 12-layer transformer must flow through 12 non-linearities. With residual: gradients can bypass any subset of layers.
- "Residual connections are why deep transformers train at all. They are gradient highways."
- In transformers: applied after every attention sublayer and FFN sublayer.
- If beginners look confused: "The residual connection adds the original input back to the sublayer output — so the model learns what to ADD to the representation, not what the representation should BE."
- If experts look bored: "The residual stream interpretation (Elhage et al., 2021): the residual connection is a single information highway that all layers read from and write to. This framing predicts layer superposition and induction heads — a powerful interpretability lens."

**Transition**: "Layer normalisation stabilises the residual stream..."

---

## Slide 96: Layer Normalisation

**Time**: ~3 min
**Talking points**:

- LayerNorm(x) = gamma × (x - mu) / (sigma + epsilon) + beta. Normalises across the feature dimension (not the batch dimension as in BatchNorm).
- "Why not BatchNorm? RNN/transformer sequences have variable length — batch statistics are noisy. LayerNorm normalises each sample independently."
- Position in transformer: Pre-LN (modern standard) vs Post-LN (original paper). Pre-LN is more stable — we cover this in a later slide.
- If beginners look confused: "LayerNorm ensures the numbers passing through the network stay in a reasonable range, preventing explosions."
- If experts look bored: "RMSNorm (used in Llama) drops the mean-centering step: RMSNorm(x) = x / RMS(x) × gamma. Slightly faster and performs identically — a free win adopted by all modern models."

**Transition**: "The decoder's special ingredient: masked self-attention..."

---

## Slide 97: Decoder Block: Masked Self-Attention

**Time**: ~3 min
**Talking points**:

- When generating, the decoder cannot attend to future tokens — those have not been generated yet.
- Causal masking: set attention scores to -infinity for positions j > i before softmax. After softmax, these become 0 — no information from the future.
- "This is the only difference between an encoder and a decoder: the masking pattern."
- If beginners look confused: "Imagine writing a sentence and only being allowed to look at words you have already written, not what comes next."
- If experts look bored: "Causal masking at training time enables parallel computation — all positions are computed simultaneously with the mask enforcing autoregressive structure. This is the key advantage of teacher forcing in transformer training."

**Transition**: "Positional encoding: telling the transformer about word order..."

---

## Slide 98: Positional Encoding: Why We Need It

**Time**: ~3 min
**Talking points**:

- Self-attention is permutation-equivariant — shuffle the input tokens and the output reshuffles identically. Order information is lost.
- "Without positional encoding, 'dog bites man' and 'man bites dog' are identical to the transformer."
- Solution: add positional information to the token embeddings before attention.
- Three approaches: fixed sinusoidal (original), learned absolute, relative/rotary (modern).
- If beginners look confused: "The transformer has no sense of word order — positional encoding injects that information."
- If experts look bored: "Permutation equivariance is a feature if you want order-invariant representations (e.g., set transformers). For language, it is a bug — we need to break this symmetry."

**Transition**: "Why sinusoidal encoding?"

---

## Slide 99: Positional Encoding: Why Sin/Cos?

**Time**: ~3 min
**Talking points**:

- PE(pos, 2i) = sin(pos / 10000^{2i/d}). PE(pos, 2i+1) = cos(pos / 10000^{2i/d}).
- Each position gets a unique vector. The encoding is deterministic and generalises to sequences longer than training.
- "The relative position between two tokens can be extracted by a linear function of their encodings — this is why sin/cos was chosen."
- Key property: PE(pos + k) is a linear function of PE(pos) — enables length extrapolation.
- If beginners look confused: "Each position in the sentence gets a unique fingerprint based on sin/cos waves at different frequencies."
- If experts look bored: "The linear transformation property is the theoretical motivation. In practice, learned absolute position embeddings (GPT-2) often outperform sinusoidal on in-distribution lengths, but fail catastrophically on longer sequences."

**Transition**: "Modern models use RoPE for better length generalisation..."

---

## Slide 100: RoPE: Rotary Position Embeddings

**Time**: ~3 min
**Talking points**:

- RoPE (Su et al., 2021): encode position as a rotation in query and key space. q_m^T k_n depends only on (q, k, m-n) — relative position.
- Used in: LLaMA, Mistral, Falcon, PaLM 2, Gemma. Has essentially replaced sinusoidal in all modern models.
- Enables length extrapolation: train on 2048 tokens, extend to 4096-16384 via YaRN or other RoPE scaling methods.
- If beginners look confused: "RoPE encodes where a word is by rotating its representation — nearby words have similar rotations."
- If experts look bored: "RoPE's key property: the inner product of rotated queries and keys gives relative position, not absolute. This is critical for context window extension — YaRN scales the rotation frequency to extrapolate to longer contexts."

**Transition**: "ALiBi is an alternative that requires no modification at test time..."

---

## Slide 101: ALiBi: Attention with Linear Biases

**Time**: ~2 min
**Talking points**:

- ALiBi (Press et al., 2021): instead of adding position encodings to embeddings, subtract a linear penalty from attention scores proportional to distance.
- score(i, j) = q_i^T k_j / sqrt(d) - m |i - j| where m is head-specific slope.
- Benefits: zero new parameters, naturally handles extrapolation (distant tokens are penalised more), no need for position embedding at inference.
- Used in: BLOOM, MPT. Less prevalent since RoPE dominates modern models.
- If beginners look confused: "ALiBi makes the model prefer attending to nearby words by adding a penalty for distance."
- If experts look bored: "The slope m per head is set deterministically as a geometric sequence. This means different heads specialise in different attention distance ranges — a free form of multi-scale processing."

**Transition**: "The second component of each transformer block: the FFN..."

---

## Slide 102: Feed-Forward Network (FFN)

**Time**: ~3 min
**Talking points**:

- Position-wise FFN: FFN(x) = max(0, x W_1 + b_1) W_2 + b_2. Applied independently to each token position.
- Inner dimension is typically 4x the model dimension: 768 → 3072 → 768 for BERT-base.
- "If attention is the communication layer (mixing information), the FFN is the computation layer (processing each position's information)."
- The FFN accounts for about 2/3 of transformer parameters. This is where most factual knowledge is stored.
- If beginners look confused: "After attention mixes information between positions, the FFN processes each position independently to refine the representation."
- If experts look bored: "Gated linear units (GLU variants — SwiGLU, GeGLU) have replaced ReLU in modern FFNs (LLaMA uses SwiGLU). They improve performance by ~1% at no extra computational cost when combined with a 2/3 intermediate dimension scaling."

**Transition**: "How many parameters does a transformer have?"

---

## Slide 103: Parameter Count

**Time**: ~3 min
**Talking points**:

- Walk through parameter count: embedding matrix (vocab × d_model), attention projections (4 matrices of d × d per head per layer), FFN (2 matrices of d × 4d per layer).
- BERT-base: 110M. BERT-large: 340M. GPT-2: 1.5B. GPT-3: 175B.
- "For BERT-base with d=768, 12 heads, 12 layers: attention = 4 × 768² × 12 = 28M. FFN = 2 × 768 × 3072 × 12 = 57M. Embedding: 30K × 768 = 23M. Total approx 110M."
- If beginners look confused: "Most of the parameters are in the attention and FFN layers — the more layers, the more parameters."
- If experts look bored: "The embedding tie — sharing the input embedding and output projection weights — is a common trick (Press & Wolf, 2017) that reduces parameter count by vocab × d_model without performance loss."

**Transition**: "How do we train transformers? Learning rate scheduling is critical..."

---

## Slide 104: Training: Warmup + Cosine Decay

**Time**: ~3 min
**Talking points**:

- Original transformer schedule: linear warmup for 4000 steps then inverse square root decay. Modern: linear warmup + cosine decay.
- "Warmup is essential — at step 0, gradients are random and large. Warmup prevents catastrophic early divergence."
- Typical warmup: 1-5% of total steps. Cosine decay runs to 10% of peak LR.
- In kailash-ml: `TrainingPipeline(scheduler="cosine_warmup", warmup_steps=400, total_steps=10000)`.
- If beginners look confused: "The learning rate starts small, ramps up, then gradually decreases. This prevents the model from making too large updates in either the beginning or the end of training."
- If experts look bored: "WSD (Warmup-Stable-Decay) is the modern alternative — a stable phase at peak LR enables checkpoint averaging, and the decay phase can be short. Used in Mistral, LLaMA training recipes."

**Transition**: "Other key training techniques..."

---

## Slide 105: Training Techniques

**Time**: ~3 min
**Talking points**:

- Walk through: (1) label smoothing — smooths one-hot targets, reduces overconfidence; (2) dropout — applied to attention weights and residual sublayers; (3) weight decay — L2 regularisation on non-bias parameters; (4) gradient clipping; (5) mixed precision (FP16/BF16 with FP32 optimizer state).
- "Most of these are default in kailash-ml's TrainingPipeline. You adjust them when default training diverges."
- If beginners look confused: "These are the dials that prevent overfitting and training instability."
- If experts look bored: "BF16 is strictly preferred over FP16 for transformer training — wider exponent range prevents overflow in residual streams. FP16's 5-bit exponent overflows at values > 65504, which occurs regularly in deep transformers."

**Transition**: "Flash Attention dramatically improves attention efficiency..."

---

## Slide 106: Flash Attention

**Time**: ~4 min
**Talking points**:

- Standard attention materialises the full n × n attention matrix in HBM (GPU memory). For n = 2048, this is 16MB per layer per head.
- Flash Attention (Dao et al., 2022): tile-based computation that keeps the running softmax in SRAM (fast on-chip memory) and never writes the full attention matrix to HBM.
- Result: 2-4x speedup, 10-20x memory reduction, mathematically exact (not an approximation).
- "Flash Attention is now the default in every serious transformer implementation — PyTorch, HuggingFace, JAX."
- If beginners look confused: "Flash Attention computes the same result faster by using the fast memory on the GPU chip instead of writing to slower memory."
- If experts look bored: "Flash Attention 2 (Dao, 2023) adds work partitioning across warps to improve utilisation from ~25% to ~50% of theoretical FLOP/s. The backward pass recomputes attention during the backward rather than storing it — a classic memory-compute tradeoff."

**Transition**: "GQA reduces KV cache memory for inference..."

---

## Slide 107: Grouped-Query Attention (GQA)

**Time**: ~3 min
**Talking points**:

- Multi-Query Attention (MQA): all query heads share one key head and one value head. Reduces KV cache size by h×.
- GQA (Ainslie et al., 2023): share across groups of g query heads — a generalisation between MHA (g=1) and MQA (g=h).
- Used in: LLaMA 2 (GQA with 8 groups), Mistral 7B, Gemma, Falcon. The new default for efficient inference.
- "KV cache at inference is the memory bottleneck. GQA reduces it by 4-8x with minimal quality loss."
- If beginners look confused: "Instead of storing separate keys and values for each attention head, GQA groups heads together to share key/value storage."
- If experts look bored: "The KV cache for a 70B LLaMA-2 with sequence length 4096 and batch size 8 is multiple tens of gigabytes. GQA makes the difference between fitting in one A100 and needing two."

**Transition**: "Sparse attention for very long sequences..."

---

## Slide 108: Sparse Attention

**Time**: ~3 min
**Talking points**:

- Full attention is O(n²). For n = 16384 tokens, the attention matrix is 1GB — impractical.
- Sparse patterns: local windowed (Longformer), strided (Sparse Transformer), random, or global+local (BigBird).
- BigBird (Zaheer et al., 2020): combines local window + random + global tokens. Proves sparse attention is Turing-complete.
- "For documents, code, and long contexts: sparse attention or linear attention variants. For standard NLP: full attention."
- If beginners look confused: "Sparse attention lets each token attend to only a subset of other tokens — nearby words and a few key global positions."
- If experts look bored: "The theoretical contribution of BigBird is showing that Turing-completeness requires both local and global attention — pure local window attention cannot represent all functions. This motivates the [CLS]-style global token that many sparse models include."

**Transition**: "Pre-LayerNorm vs Post-LayerNorm..."

---

## Slide 109: Pre-LayerNorm vs Post-LayerNorm

**Time**: ~2 min
**Talking points**:

- Original paper: Post-LN (LayerNorm after residual addition). Modern: Pre-LN (LayerNorm before sublayer).
- Pre-LN is more stable, trains without learning rate warmup, and allows larger learning rates.
- "Almost all models trained after 2019 use Pre-LayerNorm. If you see training instability, check your norm position first."
- If beginners look confused: "Pre-LN normalises before processing; Post-LN normalises after. Pre-LN is the stable version."
- If experts look bored: "The instability of Post-LN is linked to uncontrolled output scale — the residual addition can cause layer output scale to grow with depth. Pre-LN normalises this. The tradeoff: Pre-LN slightly weakens layer information integration at depth."

**Transition**: "Why did the transformer dominate all alternatives?"

---

## Slide 110: The Transformer: Why It Won

**Time**: ~3 min
**Talking points**:

- Four reasons: (1) parallelism — O(n) parallel ops vs O(n) sequential for RNN; (2) direct path length — O(1) vs O(n) for long-range dependencies; (3) scalability — more compute → better model, predictably; (4) transfer learning — pre-train once, fine-tune everywhere.
- "The transformer won because it was the first architecture that could be scaled efficiently with compute."
- If beginners look confused: "The transformer is faster, better at long sequences, and scales predictably. RNNs could not match it on any of these dimensions."
- If experts look bored: "The scalability argument is the decisive one. CNNs and RNNs exhibit diminishing returns with scale. Transformers show consistent power-law improvement — this is the Chinchilla scaling law we will cover in Section G."

**Transition**: "Encoder-only vs decoder-only vs full encoder-decoder..."

---

## Slide 111: Encoder-Only vs Decoder-Only vs Full

**Time**: ~3 min
**Talking points**:

- Encoder-only (BERT): bidirectional, no generation, best for classification/NER/QA. Reads full context.
- Decoder-only (GPT): causal/unidirectional, excellent for generation. Modern LLMs are all decoder-only.
- Full encoder-decoder (T5, BART): translation, summarisation, structured generation.
- "Task determines architecture: understand → encoder-only. Generate → decoder-only. Transform input to output → encoder-decoder."
- If beginners look confused: "BERT understands text. GPT generates text. T5 transforms text."
- If experts look bored: "The encoder-only vs decoder-only debate is settled for generation tasks — GPT-style decoder-only models scale better and are simpler. For understanding-heavy tasks with small labelled sets, encoder-only still competes. The full encoder-decoder remains alive in structured prediction."

**Transition**: "KV caching makes transformer inference practical..."

---

## Slide 112: KV Cache: Why Transformers Are Fast at Inference

**Time**: ~3 min
**Talking points**:

- During autoregressive generation, we compute K and V for each past token at every step — O(n²) total. KV cache: store K and V from all past steps, reuse them.
- With KV cache: each new token only computes Q, K, V for itself and looks up cached past K, V. O(n) per step.
- "KV cache converts O(n²) inference to O(n). This is what makes real-time chatbots possible."
- Memory: KV cache grows linearly with sequence length — this is the memory constraint for long contexts.
- If beginners look confused: "KV cache saves the work done for past words so we do not redo it for each new word."
- If experts look bored: "Continuous batching (vLLM's PagedAttention) treats the KV cache as virtual memory pages — different sequences can share physical memory, increasing inference throughput by 2-4x over naive batching."

**Transition**: "What does training at scale require?"

---

## Slide 113: Transformer Training at Scale

**Time**: ~3 min
**Talking points**:

- Walk through the infrastructure: pipeline parallelism (layers split across GPUs), tensor parallelism (attention heads split across GPUs), data parallelism (batch split across GPUs).
- "Training GPT-3 required 3.14 × 10^23 FLOPs — roughly 355 GPU-years on a single A100."
- Gradient checkpointing: recompute activations during backward instead of storing them. Trade compute for memory.
- If beginners look confused: "Training large transformers requires many GPUs working together — each GPU handles a different part of the model."
- If experts look bored: "Megatron-LM (NVIDIA) combined tensor + pipeline parallelism at 3D parallelism. The efficiency bottleneck is pipeline bubble — idle GPUs waiting for the previous pipeline stage. GPipe and PipeDream address this with microbatching and async scheduling."

**Transition**: "The original paper: a brief look back..."

---

## Slide 114: The Original Transformer: "Attention Is All You Need"

**Time**: ~3 min
**Talking points**:

- Vaswani et al. (2017). Six authors from Google Brain and Google Research.
- Achieved state-of-the-art on WMT English-German translation: 28.4 BLEU — beating all previous models.
- "The provocative title was accurate: attention alone was sufficient. RNNs, convolutions — all unnecessary."
- Training time: 3.5 days on 8 P100 GPUs. GPT-3 would take 355 GPU-years. Scale is the story of the next 7 years.
- If beginners look confused: "This 2017 paper started the current AI era. Every LLM you have heard of descends from this architecture."
- If experts look bored: "Interesting historical note: the paper was submitted to ICLR before NIPS and the ICLR reviewers found the ablation study insufficient. The full ablation was added in the NIPS version."

**Transition**: "Section F key takeaways..."

---

## Slide 115: Section F: Key Takeaways

**Time**: ~2 min
**Talking points**:

- Three points: (1) transformer = multi-head self-attention + FFN + residual + layer norm, repeated N times; (2) positional encoding (sinusoidal or RoPE) injects order information; (3) modern improvements — Flash Attention, GQA, Pre-LN — are production requirements, not research curiosities.
- Check: "What does the decoder's causal mask do?" (Prevents attending to future tokens during generation.)
- If beginners look confused: "The transformer is the architecture behind every modern AI language system."

**Transition**: "Section G: how do we get powerful models cheaply — pre-training..."

---

## Slide 116: G. Pre-training Paradigms (Section Header)

**Time**: ~1 min
**Talking points**:

- "Pre-training is why a single training run can produce a model useful for thousands of different tasks."
- Section G: BERT, GPT, T5, scaling laws, Mixture of Experts, in-context learning, instruction tuning.
- If experts look bored: "We will cover Chinchilla's derivation of optimal token-to-parameter ratios and the implications for training compute allocation."

**Transition**: "The three paradigms of modern NLP..."

---

## Slide 117: The Three Paradigms

**Time**: ~3 min
**Talking points**:

- Paradigm 1 (pre-2013): Feature engineering + linear models. Human designs features; model is shallow.
- Paradigm 2 (2013-2018): Neural feature learning. Human designs architecture; model learns features.
- Paradigm 3 (2018-present): Pre-training + fine-tuning. Model pre-trained on vast unlabelled data, fine-tuned with small labelled set.
- "Each paradigm shift reduced the human expertise required and increased the quality ceiling."
- If beginners look confused: "We have moved from humans writing rules, to models learning from examples, to models learning from vast unlabelled text."
- If experts look bored: "The fourth paradigm is emerging: pre-train → instruction-tune → RLHF. The fine-tuning step is being replaced by prompting and in-context learning. We may be in another transition."

**Transition**: "BERT: the first major pre-training breakthrough..."

---

## Slide 118: BERT: Masked Language Model

**Time**: ~4 min
**Talking points**:

- BERT (Devlin et al., 2018): Bidirectional Encoder Representations from Transformers.
- Pre-training objective: Masked Language Model (MLM) — randomly mask 15% of tokens, predict the masked tokens.
- Second objective: Next Sentence Prediction (NSP) — predict if sentence B follows sentence A. (NSP later shown to be unhelpful — RoBERTa drops it.)
- "MLM requires the model to understand context from both left and right to predict the masked word — this forces deep bidirectional understanding."
- If beginners look confused: "BERT is like a fill-in-the-blank exam on billions of sentences. It gets so good at filling in blanks that it understands language deeply."
- If experts look bored: "The 15% masking with 80/10/10 split (mask/random/original) is a specific design choice — too high and the model loses context, too low and training is inefficient. The 80/10/10 prevents the model from learning that [MASK] means 'predict here', since at fine-tune time no masks appear."

**Transition**: "How do we use BERT for specific tasks?"

---

## Slide 119: BERT: Fine-tuning

**Time**: ~3 min
**Talking points**:

- Add a task-specific head on top of BERT's [CLS] token representation. Fine-tune all weights with a small labelled dataset.
- Classification: linear layer on [CLS]. Token labelling (NER): linear layer on each token. QA: predict start/end position.
- "BERT set new state of the art on 11 NLP tasks simultaneously when released. Nothing like it had happened before."
- kailash-ml: `AutoMLEngine(task="text_classification", model="bert-base-uncased").fit(texts, labels)`.
- If beginners look confused: "Fine-tuning adds a small task-specific layer on top of pre-trained BERT and trains everything together on labelled data."
- If experts look bored: "The [CLS] token is a learned aggregation of the full sentence — it attends to all positions and summarises them. For long documents, mean-pooling over all token representations typically outperforms [CLS] pooling."

**Transition**: "BERT spawned a family of variants..."

---

## Slide 120: BERT Variants

**Time**: ~3 min
**Talking points**:

- RoBERTa: removes NSP, trains longer, larger batches, more data. Strictly better than BERT.
- DistilBERT: 40% smaller, 60% faster, 97% of performance — student model via knowledge distillation.
- Domain variants: BioBERT, LegalBERT, FinBERT — domain-specific pre-training from BERT checkpoint.
- Multilingual: mBERT (104 languages), XLM-R (100 languages, stronger).
- If beginners look confused: "These are all BERT with specific improvements — use the one that matches your domain and speed requirements."
- If experts look bored: "DeBERTa (He et al., 2020) is the strongest encoder for most benchmarks — it uses disentangled attention (separate position and content attention matrices) and virtual adversarial training. It is the default encoder for NLP benchmarks today."

**Transition**: "GPT takes the opposite approach: causal language modelling..."

---

## Slide 121: GPT: Causal Language Model

**Time**: ~4 min
**Talking points**:

- GPT: pre-train on next-token prediction (causal LM). Decoder-only, left-to-right.
- Pre-training objective: P(x*t | x_1, ..., x*{t-1}). Predict the next token given all previous tokens.
- Fine-tune on downstream tasks by reformatting as text generation: "Sentiment: [review] → [positive/negative]".
- "BERT understands by filling blanks. GPT learns by predicting the future. Both lead to rich representations."
- If beginners look confused: "GPT learns language by reading billions of sentences and practising finishing them."
- If experts look bored: "The decoder-only architecture enables zero-shot task performance via prompt engineering — this is the key advantage over encoder-only BERT for few-shot applications. GPT-1 showed this was possible; GPT-2 demonstrated it at scale; GPT-3 proved it commercially viable."

**Transition**: "How did GPT evolve?"

---

## Slide 122: GPT Evolution

**Time**: ~3 min
**Talking points**:

- GPT-1 (2018): 117M params, 5GB text. Showed transfer learning works.
- GPT-2 (2019): 1.5B params, 40GB WebText. OpenAI initially withheld for "safety". Showed zero-shot learning.
- GPT-3 (2020): 175B params, 300B tokens. In-context learning — few-shot without gradient updates. Changed the industry.
- GPT-3.5 / InstructGPT (2022): added RLHF instruction following. Made GPT usable as an assistant.
- If beginners look confused: "Each GPT version is bigger and smarter than the last. GPT-3 was the jump that made AI assistants realistic."
- If experts look bored: "The GPT-2 staged release decision established the precedent for capability evaluations before release. This directly influenced current RLHF practices and AI safety norms."

**Transition**: "T5 unifies all NLP as text-to-text..."

---

## Slide 123: T5: Text-to-Text

**Time**: ~3 min
**Talking points**:

- T5 (Raffel et al., 2020): every NLP task as text-to-text. "Translate English to French: [sentence]", "Summarise: [article]", "Classify sentiment: [review]".
- Pre-training: span corruption (mask contiguous spans, predict spans) on C4 (750GB cleaned web text).
- "T5 showed that a single text-to-text format can match or beat task-specific architectures on every NLP benchmark."
- If beginners look confused: "T5 turns every task into: give the model a text instruction + input, get a text output."
- If experts look bored: "The C4 dataset curation decisions in T5 are instructive — aggressive deduplication removed 70% of raw data. Data quality vs quantity is the underrated variable in pre-training."

**Transition**: "How much compute and data do we need? Scaling laws answer this..."

---

## Slide 124: Scaling Laws: Chinchilla

**Time**: ~4 min
**Talking points**:

- Hoffmann et al. (2022): for a given compute budget, train a smaller model on more tokens.
- Chinchilla optimal: tokens = 20 × parameters. GPT-3 (175B params) should train on 3.5T tokens, not 300B.
- The empirical law: L decreases as power laws of both parameters and data.
- "GPT-3 was undertrained. Chinchilla trained on 1.4T tokens outperformed Gopher (280B) despite being 4x smaller."
- If beginners look confused: "Scaling laws tell you how to spend your compute budget wisely — more data is often better than a bigger model."
- If experts look bored: "LLaMA's contribution was computing Chinchilla-optimal at inference rather than training. They trained a 7B model on 1T tokens — more tokens than Chinchilla suggests for training optimality. The insight: if you are going to serve the model 10M times, minimise inference cost, not training cost."

**Transition**: "Mixture of Experts allows scaling without proportional cost..."

---

## Slide 125: Mixture of Experts (MoE)

**Time**: ~4 min
**Talking points**:

- MoE: replace the dense FFN with a set of expert FFNs. A router selects k of N experts per token (typically k=2, N=8 or N=64).
- Total parameters scale with N but activated parameters per token stay at 2/N of total — "sparse activation".
- "Mistral 8x7B has 47B parameters but activates only 13B per token — the inference cost of a 13B model with the quality of a 47B model."
- Training challenge: load balancing. If all tokens always select the same 2 experts, the others never train. Auxiliary load-balancing loss needed.
- If beginners look confused: "MoE is like having a team of specialists. For each task, only a few specialists do the work, but you have access to all of them."
- If experts look bored: "Expert collapse is the main MoE failure mode — the gating network learns to always route to the same experts. GShard's auxiliary balancing loss penalises imbalanced routing. Switch Transformer (Google, 2021) scaled to 1.6T parameters with MoE."

**Transition**: "In-context learning: GPT-3's revolutionary property..."

---

## Slide 126: In-Context Learning: How GPT-3 Changed Everything

**Time**: ~4 min
**Talking points**:

- Zero-shot: "Translate to French: Hello → "; few-shot: "English: Hello, French: Bonjour. English: Goodbye, French: Au revoir. English: Thank you, French: \_\_\_".
- No gradient updates. The model adapts purely from the prompt context.
- "GPT-3 showed that a sufficiently large LM can perform new tasks without any fine-tuning. This changed the economics of NLP."
- Why it works: theories include pattern matching over pre-training data, implicit Bayesian inference, meta-learning.
- If beginners look confused: "You can teach GPT-3 a new task just by showing it examples in the prompt — no retraining needed."
- If experts look bored: "The theoretical explanation for ICL is contested. Min et al. (2022) showed that the labels in few-shot examples barely matter — what matters is the format and the distribution of inputs. This suggests ICL is format adaptation, not task learning."

**Transition**: "The transfer learning pipeline in production..."

---

## Slide 127: Transfer Learning Pipeline

**Time**: ~3 min
**Talking points**:

- Walk through the kailash-ml pipeline: (1) load pre-trained model from registry; (2) add task-specific head; (3) fine-tune with AdamW and cosine schedule; (4) evaluate on holdout; (5) register to ModelRegistry.
- `TrainingPipeline(base_model="bert-base-uncased", task="classification", num_labels=3).fit(texts, labels)`
- "This pattern handles a vast range of NLP production use cases."
- If beginners look confused: "You start from a model that already knows language, then teach it your specific task."
- If experts look bored: "Parameter-efficient fine-tuning (PEFT) — LoRA, prefix tuning, adapter layers — has made fine-tuning accessible at 1-2 GPU hours even for 7B-parameter models. kailash-ml's AutoMLEngine supports LoRA fine-tuning via `use_lora=True`."

**Transition**: "Instruction tuning and RLHF transform a language model into an assistant..."

---

## Slide 128: Instruction Tuning and RLHF

**Time**: ~4 min
**Talking points**:

- Base LM: continues text. Instruction-tuned LM: follows instructions. The difference is training on (instruction, response) pairs.
- RLHF: collect human preference rankings → train reward model → use PPO to optimise LM for human preference.
- "InstructGPT showed that a 1.3B RLHF model was preferred by humans over a 175B base GPT-3."
- DPO (Direct Preference Optimisation): simpler alternative to RLHF, no separate reward model needed.
- If beginners look confused: "RLHF teaches the model what 'good answers' look like by getting human ratings and optimising for them."
- If experts look bored: "DPO's key insight: the optimal RLHF policy has a closed-form solution in terms of the base LM and the reward model. This allows training the policy directly on preference pairs without RL. We covered DPO in detail in Module 6 (Align framework)."

**Transition**: "What goes into pre-training data?"

---

## Slide 129: The Pre-training Data Pipeline

**Time**: ~3 min
**Talking points**:

- Web crawl → language detection → deduplication → quality filtering (heuristics + classifier) → domain mixing.
- "The data pipeline is as important as the model architecture. Garbage training data produces garbage models."
- Deduplication matters: memorisation increases with repetition. Deduplicated data improves generalisation.
- Singapore relevance: "For APAC applications, consider including corpus in local languages, plus Singapore government corpus."
- If beginners look confused: "Billions of web pages must be cleaned and filtered before they are useful for training. This is the unglamorous but critical work."
- If experts look bored: "MinHash LSH deduplication (Lee et al., 2022) finds near-duplicate documents at 13-trillion-token scale. The finding: the Web contains extraordinary duplication — removing duplicates improves downstream performance on every benchmark tested."

**Transition**: "Section G key takeaways..."

---

## Slide 130: Section G: Key Takeaways

**Time**: ~2 min
**Talking points**:

- Three points: (1) pre-train on large unlabelled corpus, fine-tune on small labelled set — this is the dominant paradigm; (2) Chinchilla scaling law: tokens = 20 × parameters for compute-optimal training; (3) RLHF / instruction tuning converts a language model into an assistant.
- Check: "What pre-training objective does BERT use?" (Masked Language Model — predict masked tokens bidirectionally.)
- If beginners look confused: "Pre-trained models are the foundation. Fine-tuning adapts them to your specific task."

**Transition**: "Section H: what do we actually do with NLP models — tasks and decoding..."

---

## Slide 131: H. NLP Tasks & Decoding (Section Header)

**Time**: ~1 min
**Talking points**:

- "We have the models. Section H covers how to use them for real tasks and how to generate good output."
- Section H: task taxonomy, decoding strategies, metrics, NER, machine translation, summarisation.
- If experts look bored: "We will cover BLEU in detail plus the modern alternatives (ROUGE, BERTScore, BLEURT) and when each is appropriate."

**Transition**: "A map of NLP tasks..."

---

## Slide 132: NLP Task Taxonomy

**Time**: ~3 min
**Talking points**:

- Four categories: (1) classification — sentiment, topic, intent; (2) sequence labelling — NER, POS, chunking; (3) structured prediction — parsing, coreference; (4) generation — translation, summarisation, QA, dialogue.
- "Which model architecture fits which task? Encoder-only for classification/labelling; decoder-only for generation; full transformer for structured generation."
- If beginners look confused: "Different NLP tasks have different output types — one label, one label per word, or a whole new sentence."
- If experts look bored: "The taxonomy is shifting — decoder-only models now handle classification via prompting. Tasks that required encoder-only models in 2019 are now solved with instruction-tuned decoder-only models. The architectural specialisation is collapsing."

**Transition**: "Decoding: how do we choose which token to generate next?"

---

## Slide 133: Decoding: Greedy and Beam Search

**Time**: ~4 min
**Talking points**:

- Greedy: always pick the highest probability token. Fast but often produces repetitive or locally optimal but globally poor text.
- Beam search: maintain top-k partial sequences ("beams"), expand each at every step, keep top-k. More expensive but finds better global solutions.
- "Beam search with k=4 is the standard for translation. Greedy is fine for chatbots where diversity is desired."
- Beam width tradeoff: wider beam = better quality but more memory and compute. k=4 is the sweet spot for most tasks.
- If beginners look confused: "Greedy is like always taking the obvious next word. Beam search explores a few different paths before committing."
- If experts look bored: "Beam search over-generates short sequences and high-frequency patterns. The length normalisation penalty score = log P / T^alpha with alpha ~ 0.6 partially mitigates this. Diverse beam search (Vijayakumar et al.) adds a diversity penalty across beams."

**Transition**: "Sampling strategies allow more creative and diverse outputs..."

---

## Slide 134: Decoding: Sampling Strategies

**Time**: ~3 min
**Talking points**:

- Pure sampling: sample from the full probability distribution. High diversity, but can sample low-quality tokens.
- Temperature scaling: divide logits by T. T < 1 sharpens (more deterministic). T > 1 flattens (more random).
- "Temperature 0 approximately equals greedy. Temperature 1 = unmodified distribution. Temperature 2 = near-uniform."
- Production defaults: temperature 0.7-1.0 for creative tasks, 0.0-0.3 for factual tasks.
- If beginners look confused: "Temperature controls creativity — low temperature gives focused answers, high temperature gives more varied but potentially less accurate answers."
- If experts look bored: "The temperature-calibration connection: if a model is over-confident (miscalibrated), increasing temperature improves calibration on the holdout set. This is a quick diagnostic for model miscalibration."

**Transition**: "Top-k and nucleus sampling are the industry standards..."

---

## Slide 135: Top-k and Top-p (Nucleus) Sampling

**Time**: ~3 min
**Talking points**:

- Top-k: sample only from the k most probable tokens. k=50 is common. Problem: k is fixed regardless of the distribution shape.
- Top-p (nucleus): sample from the smallest set of tokens whose cumulative probability exceeds p. p=0.9 or 0.95 typical.
- "Nucleus sampling adapts to the distribution — when the model is confident, the nucleus is small; when uncertain, it expands."
- Most production systems: temperature + top-p together. "temp=0.8, top_p=0.9" is a common combination.
- If beginners look confused: "Top-p is like saying 'I will choose from the words that together cover 90% of the probability — ignore the long tail of unlikely words'."
- If experts look bored: "Typical decoding (Meister et al., 2022) selects tokens near the average entropy level — it is theoretically motivated as maximising local typicality rather than probability. Performs similarly to nucleus sampling on most benchmarks."

**Transition**: "How do we measure NLP model quality?"

---

## Slide 136: NLP Metrics

**Time**: ~3 min
**Talking points**:

- Classification: accuracy, F1, AUC (same as Module 4).
- Generation: BLEU (n-gram precision), ROUGE (n-gram recall), BERTScore (semantic similarity), BLEURT (learned metric).
- Perplexity: intrinsic LM metric — lower is better. "How surprised is the model by the test text?"
- "Human evaluation is the gold standard but expensive. Automated metrics are proxies — use multiple."
- If beginners look confused: "For generated text, we measure how similar the output is to a reference answer — both word-by-word and by meaning."
- If experts look bored: "BERTScore correlates better with human judgement than BLEU/ROUGE for abstractive summarisation. However, it requires a GPU for computation and is sensitive to the choice of reference model. BLEURT adds a learned calibration that correlates even better."

**Transition**: "BLEU score in detail..."

---

## Slide 137: BLEU Score: Detailed

**Time**: ~3 min
**Talking points**:

- BLEU = modified n-gram precision (1 to 4-gram) × brevity penalty.
- Modified precision: each reference n-gram can only match one system n-gram (prevents repetition gaming).
- Brevity penalty: penalises outputs shorter than the reference (prevents always outputting 1 word).
- "BLEU-4 on WMT English-German: human ~ 25-30. Google Translate ~ 34. GPT-4 ~ 40+."
- If beginners look confused: "BLEU checks how many word sequences from your output also appear in the correct answer."
- If experts look bored: "BLEU has well-known failure modes: it measures surface overlap, not meaning. 'The president approved the bill' has high BLEU against 'The bill was approved by the president' but they are paraphrases. SacreBLEU standardises tokenisation to make BLEU comparable across papers."

**Transition**: "Named Entity Recognition: extracting structure from text..."

---

## Slide 138: NER: Named Entity Recognition

**Time**: ~3 min
**Talking points**:

- NER: identify and classify named entities in text: persons (PER), organisations (ORG), locations (LOC), dates (DATE), etc.
- Singapore-specific entities: HDB, CPF, MAS, EDB, NEA — required for Singapore financial/government NLP.
- Model: token classification with BERT. Output: IOB labels (Inside, Outside, Beginning of entity).
- kailash-ml: `TrainingPipeline(task="token_classification", model="bert-base", label_scheme="IOB")`.
- If beginners look confused: "NER finds the names in a sentence — people, places, companies, dates — and labels what type of name each one is."
- If experts look bored: "Nested NER (entities within entities) requires a different architecture — span-based or biaffine attention models. 'Bank of Singapore' contains both an ORG and a LOC, which standard IOB cannot represent simultaneously."

**Transition**: "Machine translation: NLP's oldest and biggest commercial application..."

---

## Slide 139: Machine Translation: A Success Story

**Time**: ~4 min
**Talking points**:

- Walk through the evolution: rule-based → statistical (IBM models) → phrase-based → neural seq2seq → transformer.
- Each transition: transformer gave +5 BLEU. Scaling gave +5 more. "The gains are real and customer-visible."
- Singapore relevance: Singapore's four official languages (English, Mandarin, Malay, Tamil) and ASEAN multilingual market.
- Low-resource challenge: Malay-Tamil parallel data is scarce. Cross-lingual transfer via mBERT or XLM-R.
- If beginners look confused: "Machine translation went from unusable to human-competitive in 10 years, mostly through the advances we covered today."
- If experts look bored: "Back-translation (Sennrich et al., 2016) bootstraps low-resource translation: translate target-side monolingual data to the source language with a weak model, creating synthetic training pairs. This technique enabled near-parity for many language pairs with <100K parallel sentences."

**Transition**: "Summarisation and generation: the creative end of NLP..."

---

## Slide 140: Summarisation and Generation

**Time**: ~3 min
**Talking points**:

- Extractive summarisation: select existing sentences. Abstractive: generate new sentences. Transformers enable abstractive.
- Models: BART (for abstractive summarisation), PEGASUS (pre-trained specifically for summarisation with gap-sentence generation objective).
- Singapore use case: Straits Times article summarisation, MAS financial report summarisation, legal document summarisation.
- kailash-ml: `AutoMLEngine(task="summarization", model="facebook/bart-large-cnn")`.
- If beginners look confused: "Extractive is like highlighting. Abstractive is like rewriting in your own words — much harder but more useful."
- If experts look bored: "PEGASUS's gap sentence generation objective (mask whole sentences most similar to the remaining text) is a domain-specific pre-training signal that outperforms generic MLM for summarisation — a lesson in task-adaptive pre-training."

**Transition**: "Common generation failure modes..."

---

## Slide 141: Repetition and Length Control

**Time**: ~3 min
**Talking points**:

- Repetition: models repeat phrases and sentences. Fix: repetition penalty (reduce probability of recently generated tokens).
- Length control: models tend toward average training length. Fix: length penalty in beam search, minimum/maximum length constraints.
- Hallucination: models generate plausible-sounding but factually incorrect text. "The most dangerous failure mode."
- If beginners look confused: "Language models sometimes repeat themselves or make up facts. These are known failure modes with partial fixes."
- If experts look bored: "Hallucination mitigation strategies include: retrieval augmentation (ground generation in retrieved documents), contrastive decoding (contrast with a smaller 'amateur' model that hallucinates more), and calibration-based uncertainty estimation. None fully solves hallucination — it is an active research area."

**Transition**: "Section H key takeaways..."

---

## Slide 142: Section H: Key Takeaways

**Time**: ~2 min
**Talking points**:

- Three points: (1) task taxonomy: classification → encoder-only, generation → decoder-only, structured generation → full transformer; (2) decoding: greedy for speed, beam for quality, nucleus sampling for diversity; (3) BLEU/ROUGE measure surface overlap — use BERTScore for semantic quality.
- Check: "When would you use beam search vs nucleus sampling?" (Beam: translation/summarisation where reference exists. Nucleus: open-ended generation, dialogue.)
- If beginners look confused: "Pick the right tool: beam search for precision, sampling for creativity."

**Transition**: "Section I: Kailash engines and synthesis — how to put this all together..."

---

## Slide 143: I. Kailash & Synthesis (Section Header)

**Time**: ~1 min
**Talking points**:

- "Everything we have covered — preprocessing, embeddings, transformers, fine-tuning — maps to specific kailash-ml engines."
- Section I: Kailash engine mapping, ModelVisualizer for NLP, AutoMLEngine for text, synthesis.
- If beginners look confused: "We will now see how all the theory connects to actual code you can run."

**Transition**: "The Kailash engine map for NLP..."

---

## Slide 144: Kailash Engines for NLP

**Time**: ~4 min
**Talking points**:

- Walk through the mapping: preprocessing → PreprocessingPipeline; BoW/TF-IDF → AutoMLEngine(vectorizer="tfidf"); embeddings → AutoMLEngine(embedding="glove-300d" or "bert"); fine-tuning → TrainingPipeline(model="bert-base"); evaluation → ExperimentTracker; visualisation → ModelVisualizer.
- "The engine choice determines the approach — you do not mix raw sklearn with Kailash engines."
- Show how the engines chain together into a full pipeline.
- If beginners look confused: "Think of each engine as a specialist — you plug them together to build your NLP system."
- If experts look bored: "TrainingPipeline supports the full HuggingFace model zoo via the model name string — it handles tokenisation, gradient accumulation, mixed precision, and evaluation loop. The abstraction is thin enough to inspect but thick enough to prevent common mistakes."

**Transition**: "ModelVisualizer gives you interpretability for NLP models..."

---

## Slide 145: ModelVisualizer for Text

**Time**: ~4 min
**Talking points**:

- Four visualisation types for NLP: (1) attention heatmaps — which tokens attend to which; (2) embedding projector — UMAP/t-SNE of token representations; (3) token attribution — SHAP/gradient × input saliency; (4) confusion matrix for classification.
- `ModelVisualizer(model=trained_bert).plot_attention(text="The bank is near the river bank")` — watch how the two "bank" tokens attend differently.
- Practical use: debugging model failures, presenting results to stakeholders, validating that the model uses correct features.
- If beginners look confused: "ModelVisualizer shows you what the model is 'thinking' — which words it focuses on when making a decision."
- If experts look bored: "Gradient × input (integrated gradients) is more reliable than raw attention for attribution — it measures causal importance rather than attention magnitude. ModelVisualizer supports both and allows direct comparison."

**Transition**: "AutoMLEngine automates the full text classification pipeline..."

---

## Slide 146: AutoMLEngine for Text Classification

**Time**: ~4 min
**Talking points**:

- `AutoMLEngine(task="text_classification")` tries: TF-IDF + logistic regression, TF-IDF + gradient boosting, DistilBERT fine-tuning, BERT fine-tuning.
- Outputs: best model, performance comparison table, recommended model with justification.
- "AutoMLEngine is the starting point for any new NLP classification problem — run it first, then optimise the winner."
- Walk through the typical result: BERT wins on quality, TF-IDF wins on speed. AutoMLEngine explains the tradeoff.
- If beginners look confused: "AutoMLEngine tries all the approaches we learned today and tells you which one works best for your data."
- If experts look bored: "AutoMLEngine's search is configurable — `AutoMLEngine(search_space=['tfidf_lr', 'tfidf_xgb', 'distilbert'], time_budget=3600)`. The time budget constraint makes it practical for production use where engineer time has cost."

**Transition**: "The grand synthesis: everything connects..."

---

## Slide 147: The Grand Synthesis

**Time**: ~5 min
**Talking points**:

- Draw the evolution chain on the board: text → tokenise → BoW (order lost) → TF-IDF (importance weighted) → Word2Vec (semantic geometry) → LSTM (sequential context) → attention (direct connections) → transformer (parallel, scalable) → BERT/GPT (pre-trained, transferable).
- "Each step addressed the limitation of the previous one. This is how science progresses."
- The unifying theme: richer representation of language meaning, from bag of characters to contextual 768-dimensional vectors.
- If beginners look confused: "Every technique we learned today made language representation a little bit better and a little bit more powerful."
- If experts look bored: "The representation quality can be measured empirically with probing classifiers at each stage — from BoW (position 0) to transformer layer 12. Watching the probing accuracy improve is a concrete demonstration of representational richness."

**Transition**: "Where each Kailash engine sits in the M1-M8 stack..."

---

## Slide 148: Cumulative Kailash Engine Map

**Time**: ~3 min
**Talking points**:

- Show the complete engine map from M1-M8: DataExplorer (M1), PreprocessingPipeline (M1/M8), FeatureEngineer/FeatureStore (M3), TrainingPipeline (M4/M8), AutoMLEngine (M6/M8), ModelVisualizer (M1-M8), ExperimentTracker (M2-M8), ModelRegistry (M4-M8), InferenceServer (M5/M8).
- "You have now touched every major engine in the kailash-ml framework. This is your toolkit for production ML."
- If beginners look confused: "This map shows how all the tools you have learned connect into one system."
- If experts look bored: "The horizontal arrows on this map represent the data flow. In production, this entire pipeline is orchestrated by the Kailash Core SDK WorkflowBuilder — everything is a node."

**Transition**: "A preview of what comes next..."

---

## Slide 149: Preview: Module 9 — LLMs, Agents & RAG

**Time**: ~3 min
**Talking points**:

- Module 9 builds directly on today's transformers and pre-training: Kaizen agent framework, Retrieval-Augmented Generation with Nexus, advanced prompting (chain-of-thought, ReAct, reflexion), tool use and MCP.
- "Module 8 gave you the theory. Module 9 is about building systems that use these models in production."
- If beginners look confused: "Module 9 is about building AI assistants that can answer questions using real documents — like a very powerful search engine with language understanding."
- If experts look bored: "Module 9 covers RAG architectures in depth: dense retrieval (DPR), hybrid BM25 + dense, cross-encoder reranking, and the Nexus deployment layer. Prerequisite: the attention and transformer concepts from today are load-bearing."

**Transition**: "Let us wrap up Module 8..."

---

## Slide 150: NLP & Transformers (Closing Title)

**Time**: ~3 min
**Talking points**:

- Recap the journey: text preprocessing → BoW/TF-IDF → word embeddings → RNNs → attention → transformers → pre-training → NLP tasks.
- Key quote to leave them with: "The transformer architecture is the general-purpose learning machine for sequences. Everything you see in modern AI builds on it."
- Exercise assignments: (1) build a complete text preprocessing pipeline with PreprocessingPipeline; (2) run AutoMLEngine on a Singapore dataset comparing TF-IDF vs BERT; (3) use ModelVisualizer to inspect attention patterns and write a one-page interpretation report.
- If beginners look confused: "You now understand how every major AI language system works at its core."
- If experts look bored: "The open research questions: efficient long-context attention, mechanistic interpretability of transformers, and closing the gap between in-context learning and gradient-based learning. All three are active at major labs."

**Transition**: "Any final questions before we close? See you in Module 9."

---
