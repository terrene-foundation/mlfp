# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 1: LLM Architecture and Tokenization
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Understand decoder-only transformer internals — tokenization,
#   KV cache, parameter counting — and make your first LLM call via Delegate.
#
# TASKS:
#   1. Implement BPE tokenizer from scratch
#   2. Calculate model parameter count from architecture spec
#   3. Estimate KV cache memory requirements
#   4. Make first Delegate call with cost budget
#   5. Compare tokenizer output vs model's built-in tokenizer
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import re
from collections import Counter

import polars as pl

from kaizen_agents import Delegate

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
print(f"LLM Model: {model}")

# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")

sample_texts = reports.select("text").head(5).to_series().to_list()
sample_text = (
    sample_texts[0] if sample_texts else "Singapore is a global financial hub."
)

print(f"Loaded {reports.height:,} documents")
print(f"Columns: {reports.columns}")
print(f"Sample text (first 200 chars): {sample_text[:200]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement BPE tokenizer from scratch
# ══════════════════════════════════════════════════════════════════════


def get_word_freqs(text: str) -> dict[str, int]:
    """Split text into words and count frequencies."""
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    return Counter(words)


def get_pair_freqs(vocab: dict[str, int]) -> Counter:
    """Count frequencies of adjacent character pairs across the vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        chars = list(word)
        for i in range(len(chars) - 1):
            pairs[(chars[i], chars[i + 1])] += freq
    return pairs


def merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """Merge the most frequent pair in the vocabulary."""
    new_vocab = {}
    bigram = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(pair[0] + pair[1], bigram)
        new_vocab[new_word] = freq
    return new_vocab


def train_bpe(
    text: str, num_merges: int = 50
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Train BPE tokenizer on text for a given number of merges."""
    word_freqs = get_word_freqs(text)
    # Start with character-level vocabulary
    vocab = {}
    for word, freq in word_freqs.items():
        vocab[" ".join(list(word))] = freq

    merges = []
    for i in range(num_merges):
        pair_freqs = get_pair_freqs({w: f for w, f in vocab.items() if " " in w})
        if not pair_freqs:
            break
        best_pair = pair_freqs.most_common(1)[0][0]
        merges.append(best_pair)
        new_vocab = {}
        bigram = " ".join(best_pair)
        replacement = "".join(best_pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        vocab = new_vocab

    return merges, vocab


def tokenize_bpe(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenize text using learned BPE merges."""
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    tokens = []
    for word in words:
        chars = list(word)
        for pair in merges:
            i = 0
            while i < len(chars) - 1:
                if chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    chars[i] = pair[0] + pair[1]
                    del chars[i + 1]
                else:
                    i += 1
        tokens.extend(chars)
    return tokens


# Train our BPE tokenizer
merges, vocab = train_bpe(sample_text, num_merges=50)
bpe_tokens = tokenize_bpe(sample_text[:500], merges)

print(f"\n=== BPE Tokenizer ===")
print(f"Learned {len(merges)} merge rules")
print(f"Top 10 merges: {merges[:10]}")
print(f"Vocab size: {len(vocab)}")
print(f"Sample tokenization ({len(bpe_tokens)} tokens): {bpe_tokens[:30]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Calculate model parameter count from architecture spec
# ══════════════════════════════════════════════════════════════════════


def count_parameters(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
) -> dict[str, int]:
    """Calculate parameter count for a decoder-only transformer.

    Components:
    - Token embedding: vocab_size * d_model
    - Position embedding: max_seq_len * d_model (assume 4096)
    - Per layer:
      - QKV projection: 3 * d_model * d_model
      - Output projection: d_model * d_model
      - FFN up: d_model * d_ff
      - FFN down: d_ff * d_model
      - Layer norms: 2 * d_model (weights) + 2 * d_model (biases)
    - Final layer norm: d_model + d_model
    - LM head: d_model * vocab_size (often tied with embedding)
    """
    max_seq_len = 4096
    token_emb = vocab_size * d_model
    pos_emb = max_seq_len * d_model

    # Attention: Q, K, V projections + output projection
    attn_per_layer = 3 * d_model * d_model + d_model * d_model
    # Feed-forward: up-projection + down-projection
    ffn_per_layer = d_model * d_ff + d_ff * d_model
    # Layer norms (2 per layer: pre-attention + pre-FFN)
    ln_per_layer = 2 * (d_model + d_model)  # weight + bias each

    per_layer = attn_per_layer + ffn_per_layer + ln_per_layer
    all_layers = n_layers * per_layer

    final_ln = d_model + d_model
    lm_head = d_model * vocab_size  # Assume untied

    total = token_emb + pos_emb + all_layers + final_ln + lm_head

    return {
        "token_embedding": token_emb,
        "position_embedding": pos_emb,
        "attention_per_layer": attn_per_layer,
        "ffn_per_layer": ffn_per_layer,
        "layernorm_per_layer": ln_per_layer,
        "total_transformer_layers": all_layers,
        "final_layernorm": final_ln,
        "lm_head": lm_head,
        "total_parameters": total,
    }


# GPT-2 Small equivalent
params = count_parameters(
    vocab_size=50_257,
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_ff=3072,
)

print(f"\n=== Parameter Count (GPT-2 Small equiv) ===")
for component, count in params.items():
    print(f"  {component}: {count:>15,}")
print(f"  Total: {params['total_parameters'] / 1e6:.1f}M parameters")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Estimate KV cache memory requirements
# ══════════════════════════════════════════════════════════════════════


def estimate_kv_cache(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    d_head: int,
    dtype_bytes: int = 2,  # FP16
) -> dict[str, float]:
    """Estimate KV cache memory in bytes.

    KV cache stores key and value tensors for each layer and head.
    Per token: 2 (K+V) * n_layers * n_heads * d_head * dtype_bytes
    Total: batch_size * seq_len * per_token
    """
    per_token = 2 * n_layers * n_heads * d_head * dtype_bytes
    total_bytes = batch_size * seq_len * per_token
    total_gb = total_bytes / (1024**3)

    return {
        "bytes_per_token": per_token,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024**2),
        "total_gb": total_gb,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


# Estimate for a 7B-class model
kv_cache = estimate_kv_cache(
    batch_size=1,
    seq_len=4096,
    n_layers=32,
    n_heads=32,
    d_head=128,
    dtype_bytes=2,
)

print(f"\n=== KV Cache Memory (7B model, seq_len=4096) ===")
print(f"  Bytes per token: {kv_cache['bytes_per_token']:,}")
print(f"  Total memory: {kv_cache['total_mb']:.1f} MB ({kv_cache['total_gb']:.3f} GB)")
print(f"  At seq_len=32768: {kv_cache['total_mb'] * 8:.1f} MB")

# Show how KV cache scales with sequence length
print(f"\n  KV Cache Scaling:")
for seq in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
    kv = estimate_kv_cache(1, seq, 32, 32, 128, 2)
    print(f"    seq_len={seq:>6}: {kv['total_mb']:>8.1f} MB")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Make first Delegate call with cost budget
# ══════════════════════════════════════════════════════════════════════


async def first_delegate_call():
    """Make a simple Delegate call to analyze document text."""

    delegate = Delegate(
        model=model,
        max_llm_cost_usd=1.0,
    )

    prompt = f"""Analyze this Singapore company report excerpt and identify:
1. Key financial metrics mentioned
2. Industry sector
3. Overall sentiment (positive/negative/neutral)

Text: {sample_text[:1000]}"""

    print(f"\n=== First Delegate Call ===")
    print(f"Prompt length: {len(prompt)} characters")

    response_text = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response_text += event.text

    print(f"Response ({len(response_text)} chars):")
    print(response_text[:500])

    return response_text


delegate_response = asyncio.run(first_delegate_call())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare tokenizer output vs model's built-in tokenizer
# ══════════════════════════════════════════════════════════════════════


async def compare_tokenizers():
    """Ask the Delegate to tokenize the same text and compare."""

    delegate = Delegate(
        model=model,
        max_llm_cost_usd=0.5,
    )

    test_sentence = "Singapore's financial sector grew 8.2% in Q3."

    # Our BPE tokenization
    our_tokens = tokenize_bpe(test_sentence, merges)

    # Ask the model about its tokenization
    prompt = f"""How would a modern LLM tokenizer break this sentence into tokens?
Sentence: "{test_sentence}"

List each token on a separate line, wrapped in | delimiters like |token|.
Then explain why some words are split into subwords."""

    response_text = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response_text += event.text

    print(f"\n=== Tokenizer Comparison ===")
    print(f"Test sentence: '{test_sentence}'")
    print(f"\nOur BPE ({len(our_tokens)} tokens): {our_tokens}")
    print(f"\nModel's perspective:")
    print(response_text[:400])

    print(f"\n  Key differences:")
    print(f"  - Our BPE: trained on a single document, small vocab")
    print(f"  - Production tokenizers: trained on billions of tokens")
    print(f"  - BPE principle is the same — merge frequent pairs")
    print(f"  - More training data = better subword boundaries")


asyncio.run(compare_tokenizers())

print(
    "\n✓ Exercise 1 complete — LLM architecture, tokenization, and first Delegate call"
)
