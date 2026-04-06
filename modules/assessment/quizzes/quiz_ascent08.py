# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 8 — AI-Resilient Assessment Questions

NLP & Transformers
Covers: tokenization, BPE, TF-IDF, BM25, Word2Vec, RNN/LSTM,
        attention, transformers, BERT, ModelVisualizer, AutoMLEngine
"""

QUIZ = {
    "module": "ASCENT08",
    "title": "NLP & Transformers",
    "questions": [
        # ── Section A: Text preprocessing ───────────────────────────────
        {
            "id": "8.A.1",
            "lesson": "8.A",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's BPE tokenizer from Exercise 1 produces 12,000 unique tokens "
                "on a Singapore news corpus, but the vocabulary size was set to 5,000. "
                "The merge loop runs but the vocabulary keeps growing. What is wrong?"
            ),
            "code": (
                "# BPE merge step\n"
                "for i in range(num_merges):\n"
                "    pairs = get_pair_frequencies(tokens)\n"
                "    best_pair = max(pairs, key=pairs.get)\n"
                "    new_token = best_pair[0] + best_pair[1]\n"
                "    vocab.add(new_token)  # Bug: num_merges exceeds target vocab\n"
                "    tokens = merge_pair(tokens, best_pair, new_token)\n"
            ),
            "options": [
                "A) BPE should remove the individual tokens after merging — vocab.discard(best_pair[0]); vocab.discard(best_pair[1])",
                "B) The num_merges parameter should be set to 5000 - 256 (target vocab minus base characters). The vocabulary GROWS by design in BPE — each merge ADDS one token. Starting from 256 base characters, 4744 merges give exactly 5000 tokens. The bug is that num_merges is too large.",
                "C) get_pair_frequencies should only count pairs that appear more than 10 times",
                "D) BPE requires sorting the corpus alphabetically before merging",
            ],
            "answer": "B",
            "explanation": (
                "BPE starts with a base vocabulary (typically 256 byte-level tokens) and "
                "iteratively merges the most frequent pair into a new token. Each merge adds "
                "exactly one token to the vocabulary. To reach target_vocab_size, you need "
                "target_vocab_size - base_vocab_size merges. The code never removes old tokens "
                "because BPE doesn't remove them — subword units remain available. "
                "The fix is: num_merges = 5000 - len(base_vocab), not an arbitrary large number."
            ),
            "learning_outcome": "Implement BPE tokenization with correct merge count for target vocabulary",
        },
        {
            "id": "8.A.2",
            "lesson": "8.A",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "You are preprocessing Singapore Parliament speeches for classification. "
                "Should you apply stemming or lemmatization? The corpus contains legal "
                "terms like 'governing', 'governance', 'government', 'governed'."
            ),
            "options": [
                "A) Stemming — faster and simpler; all four words reduce to 'govern', which is sufficient for classification",
                "B) Lemmatization — preserves meaning distinctions. 'Governance' (noun: the act of governing) and 'government' (noun: the governing body) are different concepts in legal text. Stemming collapses them into 'govern', losing the distinction between policy concepts and institutions.",
                "C) Neither — modern tokenizers like BPE handle this automatically",
                "D) Both — apply stemming first, then lemmatization on the stems",
            ],
            "answer": "B",
            "explanation": (
                "In legal/parliamentary text, the distinction between 'governance' (process), "
                "'government' (institution), and 'governed' (past participle) carries meaning. "
                "Porter stemming reduces all to 'govern', losing these distinctions. "
                "Lemmatization maps to dictionary forms while preserving part-of-speech: "
                "governance→governance (noun), governed→govern (verb). "
                "For classification, these meaning differences affect which topic a speech belongs to. "
                "Note: BPE tokenizers in modern transformers largely bypass this choice, but for "
                "traditional NLP pipelines (Exercise 1), lemmatization preserves more signal."
            ),
            "learning_outcome": "Choose between stemming and lemmatization based on domain requirements",
        },
        # ── Section B: BoW / TF-IDF ────────────────────────────────────
        {
            "id": "8.B.1",
            "lesson": "8.B",
            "type": "output_interpretation",
            "difficulty": "intermediate",
            "question": (
                "Exercise 2 computes TF-IDF on Parliament speeches. The word 'Singapore' "
                "has high TF in every document but very low IDF (because it appears in 95% "
                "of documents). Its TF-IDF score is near zero. A colleague says 'Singapore "
                "must be important — it's everywhere!' Why is the low TF-IDF score correct?"
            ),
            "options": [
                "A) It's a bug — words that appear frequently must have high TF-IDF",
                "B) TF-IDF measures DISCRIMINATIVE power, not importance. 'Singapore' in Singapore Parliament speeches is like 'the' — it appears everywhere and distinguishes nothing. IDF = log(N/df) penalizes terms appearing in many documents. High TF × low IDF ≈ 0. Words like 'cryptocurrency' or 'housing' that appear in few documents have high IDF and actually distinguish topics.",
                "C) The IDF formula is wrong — it should not use logarithm",
                "D) 'Singapore' should be added to the stop word list and removed entirely",
            ],
            "answer": "B",
            "explanation": (
                "TF-IDF's purpose is retrieval and classification, which require DISCRIMINATIVE "
                "features. A word that appears in 95% of documents has IDF ≈ log(100/95) ≈ 0.05 — "
                "nearly zero. Multiplied by any TF, the score stays near zero. "
                "This is by design: ubiquitous terms don't help distinguish documents. "
                "'Cryptocurrency' appearing in 3/100 documents has IDF ≈ log(100/3) ≈ 3.5 — "
                "70× higher discriminative power. This is exactly what Exercise 2 demonstrates "
                "when comparing TF-IDF retrieval quality across terms."
            ),
            "learning_outcome": "Interpret TF-IDF scores as discriminative power, not importance",
        },
        {
            "id": "8.B.2",
            "lesson": "8.B",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 2 implements both TF-IDF and BM25 for document retrieval. "
                "On short queries (2-3 words), BM25 returns better results. On long queries "
                "(full sentences), TF-IDF performs similarly. Why does BM25 excel on short queries?"
            ),
            "options": [
                "A) BM25 uses a different tokenizer that handles short text better",
                "B) BM25 adds term frequency saturation: TF contribution approaches an asymptote as frequency increases (controlled by k1). A word appearing 20× vs 10× in a document gets diminishing extra credit. TF-IDF's linear TF scaling over-weights repeated terms. For short queries with few terms, BM25's saturation prevents any single matching term from dominating the score.",
                "C) BM25 uses word embeddings internally while TF-IDF uses only bag-of-words",
                "D) Short queries have too few terms for TF-IDF to work; BM25 has a minimum score floor",
            ],
            "answer": "B",
            "explanation": (
                "BM25's TF component: tf_bm25 = (k1+1)×tf / (k1×(1-b+b×dl/avgdl) + tf). "
                "As tf→∞, this approaches (k1+1), creating saturation. "
                "TF-IDF's TF is linear: a document with 'finance' 20 times scores 2× a document "
                "with it 10 times, even though both are clearly about finance. "
                "BM25 recognizes diminishing returns: 20× vs 10× barely matters. "
                "For short queries (2-3 terms), this prevents a single high-frequency match from "
                "dominating over documents that match ALL query terms moderately. "
                "The b parameter additionally normalizes for document length."
            ),
            "learning_outcome": "Explain BM25's term frequency saturation advantage over linear TF-IDF",
        },
        # ── Section C: Word embeddings ──────────────────────────────────
        {
            "id": "8.C.1",
            "lesson": "8.C",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 3 trains Word2Vec on Singapore news. You test analogies: "
                "vec('MAS') - vec('Singapore') + vec('USA') returns vec('Federal Reserve') "
                "as the closest match. But vec('HDB') - vec('Singapore') + vec('USA') returns "
                "vec('apartment') instead of the expected vec('HUD'). Why?"
            ),
            "options": [
                "A) The embedding dimensions are too small to capture housing concepts",
                "B) HDB is a Singapore-specific acronym that appears primarily in Singapore context. The model learned 'HDB' as associated with 'flat/apartment/housing' rather than as a government agency. MAS appears in international finance contexts alongside Fed/ECB, so its 'role' is better captured. Embeddings reflect co-occurrence patterns, not semantic taxonomy.",
                "C) Word2Vec cannot handle acronyms — only full words",
                "D) The analogy formula is wrong — it should be addition only, not subtraction",
            ],
            "answer": "B",
            "explanation": (
                "Word2Vec's distributional hypothesis: words that appear in similar contexts "
                "get similar vectors. MAS appears alongside 'central bank', 'monetary policy', "
                "'interest rates' — contexts shared with Federal Reserve. "
                "HDB appears alongside 'flat', 'resale', 'BTO', 'housing' — contexts more "
                "similar to 'apartment' than to 'HUD' (which rarely appears in the corpus). "
                "Analogies work when the relational pattern (institution→country) is consistently "
                "represented in training data. Singapore-specific entities may not have enough "
                "cross-national context for perfect analogies."
            ),
            "learning_outcome": "Interpret word embedding analogies as co-occurrence patterns",
        },
        {
            "id": "8.C.2",
            "lesson": "8.C",
            "type": "output_interpretation",
            "difficulty": "intermediate",
            "question": (
                "Exercise 3 visualizes Word2Vec embeddings with ModelVisualizer t-SNE. "
                "You see three clear clusters: one with financial terms, one with legal terms, "
                "one with technology terms. But 'fintech' appears between the financial and "
                "technology clusters. What does this mean?"
            ),
            "options": [
                "A) 'fintech' is an outlier and should be removed from the vocabulary",
                "B) The t-SNE perplexity parameter is set incorrectly — increase it to force 'fintech' into one cluster",
                "C) 'fintech' co-occurs with both financial AND technology terms in the corpus. Its embedding is a blend of both contexts, placing it between the two clusters in vector space. This is correct behavior — the embedding captures that fintech is genuinely at the intersection of finance and technology.",
                "D) t-SNE distorted the original distances — in the original high-dimensional space, 'fintech' is in the finance cluster",
            ],
            "answer": "C",
            "explanation": (
                "Word embeddings represent words as points in continuous space. Words used in "
                "multiple contexts get vectors that blend those contexts. 'Fintech' appears in "
                "sentences about both 'banking regulations' and 'machine learning', so its vector "
                "has components from both semantic neighborhoods. "
                "t-SNE preserves local structure: if 'fintech' is equidistant from 'banking' and "
                "'AI' in 300-dimensional space, it will appear between them in 2D. "
                "This is one of the most useful properties of embeddings — they capture "
                "semantic relationships that discrete categories cannot."
            ),
            "learning_outcome": "Interpret t-SNE embedding visualizations from ModelVisualizer",
        },
        # ── Section D: RNNs / LSTMs ─────────────────────────────────────
        {
            "id": "8.D.1",
            "lesson": "8.D",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 implements a vanilla RNN for sentiment analysis. The model achieves "
                "98% accuracy on short reviews (< 20 words) but only 52% on long reviews "
                "(> 100 words). What architectural issue explains this and what fix does "
                "Exercise 4 demonstrate?"
            ),
            "options": [
                "A) Long reviews have more complex sentiment — add more hidden units",
                "B) The vanilla RNN suffers from vanishing gradients over long sequences. After 100 steps, gradient ≈ (W_hh)^100 which vanishes if max eigenvalue < 1. The model 'forgets' early words. Exercise 4 fixes this with LSTM: the forget/input/output gates and cell state provide a gradient highway that preserves information over long sequences.",
                "C) Long reviews should be truncated to 20 words to match the short review performance",
                "D) The embedding layer needs larger dimensions for longer texts",
            ],
            "answer": "B",
            "explanation": (
                "Vanilla RNN hidden state: h_t = tanh(W_hh × h_{t-1} + W_xh × x_t). "
                "Gradient through time: ∂h_T/∂h_1 = Π(W_hh × diag(tanh')). "
                "If ||W_hh|| < 1, gradients vanish exponentially with sequence length. "
                "After 100 steps, early words have near-zero influence on the final hidden state. "
                "LSTM's cell state c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t provides an additive "
                "gradient path. The forget gate f_t can be close to 1, passing gradients through "
                "unchanged — solving the vanishing gradient problem for sequences. "
                "Exercise 4 demonstrates: LSTM maintains 85%+ accuracy even on 100+ word reviews."
            ),
            "learning_outcome": "Diagnose RNN sequence length limitation and apply LSTM solution",
        },
        {
            "id": "8.D.2",
            "lesson": "8.D",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 builds a bidirectional LSTM. For the review 'The food was terrible "
                "but the service was excellent, so overall I recommend it', the unidirectional "
                "LSTM classifies as negative (attending to 'terrible') while the bidirectional "
                "LSTM classifies as positive. Why does bidirectionality help here?"
            ),
            "options": [
                "A) Bidirectional LSTMs use twice as many parameters, so they are always more accurate",
                "B) The forward LSTM processes left-to-right, heavily influenced by 'terrible' early in the sequence. By the time it reaches 'I recommend it', the signal is diluted. The backward LSTM processes right-to-left, starting from 'recommend it' — capturing the overall positive conclusion. Concatenating both directions captures the full sentiment arc.",
                "C) Bidirectional LSTMs can attend to any word in the sequence like attention mechanisms",
                "D) The unidirectional model is simply undertrained — more epochs would fix it",
            ],
            "answer": "B",
            "explanation": (
                "A forward-only LSTM at position t only knows words 1..t. By the final position, "
                "early strong signals ('terrible') may be diluted by subsequent words. "
                "The backward LSTM at position t knows words t..T, so at position 1 it has seen "
                "the entire review including the concluding 'recommend it'. "
                "Bidirectional concatenation [h_forward; h_backward] at each position gives the "
                "classifier access to both past and future context. "
                "For sentiment analysis, the concluding sentiment ('overall I recommend') often "
                "overrides earlier complaints, which the backward pass captures effectively."
            ),
            "learning_outcome": "Explain bidirectional LSTM advantage for sentiment with mixed signals",
        },
        # ── Section E: Attention ─────────────────────────────────────────
        {
            "id": "8.E.1",
            "lesson": "8.E",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 implements scaled dot-product attention. The attention weights "
                "are all nearly uniform (1/n for n tokens) regardless of input. The model "
                "performs no better than averaging all token embeddings. What is missing?"
            ),
            "code": (
                "def attention(Q, K, V):\n"
                "    scores = matmul(Q, K.T)  # Bug: missing scaling\n"
                "    weights = softmax(scores)\n"
                "    return matmul(weights, V)\n"
            ),
            "options": [
                "A) The softmax temperature is too high — add a temperature parameter of 0.1",
                "B) Missing the scaling factor 1/√d_k. Without it, dot products grow proportionally to d_k (dimension). For d_k=512, scores can reach magnitude ~22 (√512). But softmax(22×[1,1,...]) ≈ softmax([22,22,...]) = uniform. Dividing by √d_k normalizes scores to unit variance, allowing softmax to differentiate: scores / √d_k.",
                "C) Q, K, V should be the same tensor — self-attention requires Q=K=V",
                "D) The attention is missing positional encoding — without positions, all tokens look identical",
            ],
            "answer": "B",
            "explanation": (
                "For random vectors of dimension d_k, the expected dot product has variance d_k. "
                "With d_k=512, dot products have std ≈ √512 ≈ 22.6. When all scores are "
                "uniformly large, softmax saturates to near-uniform distribution (all exp(22) "
                "are similar). Scaling by 1/√d_k gives variance 1, allowing meaningful differences: "
                "softmax([3.1, -0.5, 1.2]) → [0.72, 0.02, 0.26] — actual attention! "
                "This is why the mechanism is called 'Scaled Dot-Product Attention'. "
                "Exercise 5 demonstrates the difference: without scaling, attention degrades to "
                "mean pooling."
            ),
            "learning_outcome": "Implement scaled dot-product attention with correct √d_k scaling",
        },
        {
            "id": "8.E.2",
            "lesson": "8.E",
            "type": "output_interpretation",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 visualizes attention weights. For 'The bank by the river', head 1 "
                "attends 'bank' → 'river' (0.8) while head 2 attends 'bank' → 'The' (0.6). "
                "For 'The bank approved the loan', head 1 attends 'bank' → 'loan' (0.7) while "
                "head 2 attends 'bank' → 'approved' (0.5). What do the heads learn?"
            ),
            "options": [
                "A) Head 1 always attends to the last noun; head 2 always attends to the first word",
                "B) Head 1 learns semantic disambiguation — attending to context words ('river' vs 'loan') that determine the meaning of 'bank'. Head 2 learns syntactic relationships — attending to grammatically related words ('The' as determiner, 'approved' as predicate). Multi-head attention captures different relationship types simultaneously.",
                "C) The two heads are redundant — one should be removed to save computation",
                "D) Head 1 and head 2 randomly attend to different words — there is no learned pattern",
            ],
            "answer": "B",
            "explanation": (
                "Multi-head attention allows the model to attend to different types of "
                "relationships simultaneously. Head 1 consistently attends to words that "
                "disambiguate meaning (semantic role), while head 2 attends to syntactically "
                "related words. This is emergent behavior — heads are not explicitly trained "
                "for specific roles, but the different Q/K/V projections learn complementary "
                "patterns. In transformers, different heads learn: positional, syntactic, "
                "semantic, and coreference relationships. This is why h=8 or h=16 heads "
                "are standard — each captures different aspects of the input."
            ),
            "learning_outcome": "Interpret multi-head attention patterns from ModelVisualizer output",
        },
        # ── Section F: Transformer architecture ─────────────────────────
        {
            "id": "8.F.1",
            "lesson": "8.F",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 6 builds a transformer encoder. The model trains but performance "
                "is worse than the LSTM from Exercise 4. The student notices that shuffling "
                "the word order in input sentences doesn't change the model's predictions. "
                "What component is missing?"
            ),
            "options": [
                "A) The feed-forward network is too small — increase the hidden dimension",
                "B) Missing positional encoding. Self-attention is permutation-invariant — it has no notion of word order. Without positional encoding, 'dog bites man' and 'man bites dog' produce identical representations. Exercise 6 implements sinusoidal PE: PE(pos,2i) = sin(pos/10000^(2i/d)) which gives each position a unique signature.",
                "C) The model needs decoder layers, not just encoder layers",
                "D) LayerNorm should be applied before attention, not after (Pre-LN vs Post-LN)",
            ],
            "answer": "B",
            "explanation": (
                "Self-attention computes: Attention(Q,K,V) = softmax(QK^T/√d)V. "
                "If you permute the input tokens, Q, K, V are permuted identically, and "
                "the output is the same permutation of the original output. Word ORDER is lost. "
                "Positional encoding adds position-dependent vectors to token embeddings: "
                "x_i = embed(token_i) + PE(i). The sinusoidal formula creates unique patterns "
                "for each position and allows the model to learn relative positions "
                "(PE(pos+k) can be expressed as a linear function of PE(pos)). "
                "Without PE, the transformer degrades to a bag-of-words model with attention."
            ),
            "learning_outcome": "Identify missing positional encoding from order-invariant predictions",
        },
        {
            "id": "8.F.2",
            "lesson": "8.F",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 6 builds a transformer with 4 layers, d_model=256, and 4 attention "
                "heads. The model has 2.5M parameters and takes 45 minutes to train on your "
                "corpus of 10,000 documents. Your colleague suggests using 12 layers and "
                "d_model=768 (like BERT-base). Should you?"
            ),
            "options": [
                "A) Yes — larger models always perform better, and the training time increase is linear",
                "B) No — 12 layers with d_model=768 has ~110M parameters (44× more). On 10,000 documents, this will massively overfit. The model capacity should match data size. 4 layers is appropriate for 10K documents. Use transfer learning (Exercise 7 with AutoMLEngine) instead of training from scratch if you need BERT-level performance.",
                "C) Yes — but only if you also increase the number of attention heads to 12",
                "D) No — 12 layers cannot fit in GPU memory regardless of the data size",
            ],
            "answer": "B",
            "explanation": (
                "BERT-base (110M params) was trained on 3.3B words. With only 10K documents "
                "(perhaps 1M words), you have ~100 parameters per word — severe data starvation. "
                "The 4-layer, 2.5M param model has ~2.5 parameters per word, which is reasonable. "
                "If you need BERT-level representations, Exercise 7 demonstrates the right approach: "
                "AutoMLEngine with transfer learning fine-tunes a pre-trained BERT on your small "
                "dataset, leveraging the 3.3B-word pre-training while adapting to your domain. "
                "Training a 110M model from scratch on 10K documents is the wrong approach."
            ),
            "learning_outcome": "Match transformer model capacity to dataset size and choose transfer learning",
        },
        # ── Section G: Transfer learning ────────────────────────────────
        {
            "id": "8.G.1",
            "lesson": "8.G",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 7 uses AutoMLEngine with task='text_classification' on Singapore "
                "product reviews. The leaderboard shows: TF-IDF+LR (F1=0.82), "
                "TF-IDF+SVM (F1=0.84), Transformer+Classifier (F1=0.91). "
                "Why does the transformer model outperform TF-IDF approaches by 7+ points?"
            ),
            "options": [
                "A) Transformers use more parameters so they always achieve higher F1",
                "B) TF-IDF treats words independently — 'not good' has the same features as 'good not'. Transformer embeddings capture contextual meaning: 'good' in 'not good' gets a different representation than 'good' in 'very good'. This contextual understanding is critical for sentiment where negation, sarcasm, and hedging change meaning.",
                "C) AutoMLEngine allocated more training time to the transformer model",
                "D) TF-IDF cannot handle Singapore English (Singlish) vocabulary",
            ],
            "answer": "B",
            "explanation": (
                "TF-IDF is a bag-of-words model: word order and context are lost. "
                "'This movie is not good at all' and 'This movie is good, not bad at all' "
                "produce similar TF-IDF vectors despite opposite sentiment. "
                "Transformer embeddings are contextual: each word's representation depends "
                "on surrounding words. The pre-trained transformer has already learned these "
                "contextual patterns from billions of words, giving it a massive advantage "
                "even on small fine-tuning datasets. "
                "The 7-point F1 gap reflects the value of contextual vs bag-of-words features."
            ),
            "learning_outcome": "Explain transformer advantage over TF-IDF for text classification",
        },
        {
            "id": "8.G.2",
            "lesson": "8.G",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "Exercise 7 registers the best model in ModelRegistry. A teammate asks "
                "whether to use AutoMLEngine or TrainingPipeline for their next text "
                "classification task (5,000 labeled medical reports). What do you recommend?"
            ),
            "options": [
                "A) TrainingPipeline — AutoMLEngine is only for prototyping",
                "B) AutoMLEngine — it searches across model types (TF-IDF, transformer) and hyperparameters automatically. For a new task with unknown optimal approach, AutoML exploration is more efficient than manually configuring TrainingPipeline. Once AutoML identifies the best approach, you can switch to TrainingPipeline for fine-grained control in production.",
                "C) Neither — 5,000 samples is too few for any text classification model",
                "D) TrainingPipeline with model_type='bert' — always use the largest available model",
            ],
            "answer": "B",
            "explanation": (
                "AutoMLEngine's value is exploration: it tries multiple approaches (TF-IDF+LR, "
                "TF-IDF+SVM, transformer-based) with various hyperparameters in a time-bounded "
                "search. For a new task, you don't know if transformer fine-tuning will beat "
                "TF-IDF+SVM (on small datasets, simpler models sometimes win). "
                "AutoML discovers this empirically. Once the best approach is identified, "
                "TrainingPipeline offers more control for production tuning. "
                "5,000 samples is sufficient for text classification — transfer learning from "
                "pre-trained transformers requires as few as 100-1,000 labeled examples."
            ),
            "learning_outcome": "Choose AutoMLEngine for exploration vs TrainingPipeline for production",
        },
    ],
}
