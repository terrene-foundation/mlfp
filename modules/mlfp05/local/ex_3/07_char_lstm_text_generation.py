# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 3.7 — Character-Level Text Generation with an LSTM:
# Perplexity as the Honest Metric
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Frame text generation as next-character CLASSIFICATION over a
#     character vocabulary (the same cross-entropy as any classifier)
#   - Build the shifted-window dataset: input chars[i:i+L], target
#     chars[i+1:i+L+1]
#   - Train a 2-layer character LSTM and report PERPLEXITY =
#     exp(mean cross-entropy) — "the model's effective branching factor"
#   - Sample with temperature and explain the creativity dial honestly
#   - Compare a trained model's perplexity against the uniform-random
#     baseline (vocab size) — never report perplexity without it
#   - Apply to boilerplate drafting assistance at a professional
#     services firm
#
# PREREQUISITES: M5/ex_3/02_lstm.py (LSTM mechanics); 06 for feature
#   engineering mindset. No stock data here — the corpus is text.
# ESTIMATED TIME: ~35 min
#
# DATASET: tiny-shakespeare (~1.1 MB of Shakespeare dialogue, the
#   canonical char-RNN corpus), downloaded once and cached; falls back
#   to this module's own textbook if the lab is offline.
#
# PHASES:
#   1. THEORY  — next-char classification; perplexity; temperature
#   2. BUILD   — vocabulary, encode/decode, CharLSTM
#   3. TRAIN   — 2-layer LSTM, tracked with ExperimentTracker
#   4. VISUALISE — CE/perplexity curves + temperature samples
#   5. APPLY   — drafting assistance; what perplexity means to a partner
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.mlfp05.ex_3 import (
    OUTPUT_DIR,
    init_environment,
    load_text_corpus,
    setup_engines,
)

device = init_environment()


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Generation = Classification, Repeated
# ════════════════════════════════════════════════════════════════════════
# A language model is a classifier whose classes are CHARACTERS. Given
# the last L characters, predict the distribution of the next one.
# Sample from that distribution, append, slide the window, repeat.
#
# WHY CHARACTERS (not words)?
#   No tokeniser, no out-of-vocabulary problem, and the model must learn
#   spelling, punctuation, spaces — even CAPITALISATION conventions —
#   from scratch. That makes the learning VISIBLE: epoch-1 output is
#   gibberish; epoch-3 output has real words and dialogue structure.
#
# PERPLEXITY = exp(mean cross-entropy):
#   Cross-entropy is the average surprisal in nats. Its exponential has
#   a beautiful reading: "on average, the model is as uncertain as a
#   uniform choice over perplexity-many characters".
#     - Uniform random over a 70-char vocabulary: CE = ln(70) = 4.25,
#       perplexity = 70. This is the floor any trained model must beat.
#     - A trained char-LSTM on this corpus: perplexity in the single
#       digits — the model has narrowed 70 candidates to a handful.
#   NEVER report perplexity without the vocabulary-size reference.
#
# TEMPERATURE — the creativity dial:
#   Sampling divides logits by T before the softmax.
#     T < 1: sharpen — the model plays safe (repetitive, grammatical)
#     T = 1: sample the learned distribution as-is
#     T > 1: flatten — more surprise, more errors
#   There is no "best" T; there is a best T FOR THE TASK (see Phase 5).

print("=" * 70)
print("  PHASE 1 — THEORY: next-char classification and perplexity")
print("=" * 70)
print(
    """
  GENERATION = CLASSIFICATION, REPEATED
    window of L characters -> softmax over vocabulary -> sample -> slide

  PERPLEXITY = exp(mean cross-entropy)
    "as uncertain as a uniform choice over P characters"
    uniform baseline over V characters: CE = ln(V), perplexity = V
    trained model: single digits. Report BOTH or report neither.

  TEMPERATURE: logits / T before softmax
    T=0.5 conservative | T=1.0 as-learned | T=1.5 adventurous
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: vocabulary, encode/decode, CharLSTM
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: character vocabulary + CharLSTM")
print("=" * 70)

text, corpus_name = load_text_corpus()
print(f"Corpus: {corpus_name} — {len(text):,} characters")

chars = sorted(set(text))
VOCAB_SIZE = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}


def encode(s: str) -> list[int]:
    """Characters -> vocabulary indices."""
    # TODO: map each character through char_to_idx
    return ____


def decode(indices) -> str:
    """Vocabulary indices -> characters."""
    # TODO: map each index through idx_to_char and join into a string
    # Hint: "".join(...)
    return ____


# Contiguous split: first 90% train, last 10% validation. Text has no
# i.i.d. assumption to honour, but the val text must be CONTIGUOUS and
# NEVER seen in training — shuffling characters would be nonsense.
split_at = int(0.9 * len(text))
train_ids = np.array(encode(text[:split_at]), dtype=np.int64)
val_ids = np.array(encode(text[split_at:]), dtype=np.int64)
print(
    f"Vocabulary: {VOCAB_SIZE} characters | train {len(train_ids):,} chars, "
    f"val {len(val_ids):,} chars (contiguous split)"
)

SEQ_LEN_CHARS = 128
EMB_DIM = 64
HIDDEN_CHARS = 256
N_LAYERS = 2
CHAR_BATCH = 64
CHAR_EPOCHS = 3
STEPS_PER_EPOCH = 400
CHAR_LR = 2e-3


class CharLSTM(nn.Module):
    """2-layer character LSTM: embedding -> LSTM -> per-timestep logits.

    Output shape (batch, seq, vocab): EVERY timestep predicts the next
    character, so one forward pass yields seq_len classification losses.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, emb_dim: int = EMB_DIM,
                 hidden: int = HIDDEN_CHARS, n_layers: int = N_LAYERS):
        super().__init__()
        # TODO: embedding (vocab -> emb_dim), a 2-layer batch-first LSTM
        #   (emb_dim -> hidden), and a linear head (hidden -> vocab)
        self.emb = ____
        self.lstm = ____
        self.head = ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: embed, run the LSTM, map every timestep to vocab logits
        #   (keep the sequence dimension: output (batch, seq, vocab))
        out, _ = ____
        return ____


def sample_batch(ids: np.ndarray, seq_len: int = SEQ_LEN_CHARS,
                 batch_size: int = CHAR_BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows: input chars[i:i+L], target chars[i+1:i+L+1]."""
    # TODO: draw batch_size random start indices, each leaving room for
    #   seq_len + 1 characters (window PLUS the shifted target)
    starts = ____
    x = np.stack([ids[s : s + seq_len] for s in starts])
    # TODO: the target window — the SAME window shifted right by one char
    y = ____
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )


@torch.no_grad()
def mean_ce(ids: np.ndarray, model: nn.Module, n_batches: int = 25) -> float:
    """Mean cross-entropy over fixed windows of a held-out sequence."""
    model.eval()
    losses = []
    for k in range(n_batches):
        # evenly spaced windows (deterministic, covers the val text)
        starts = np.linspace(0, len(ids) - SEQ_LEN_CHARS - 1,
                             num=CHAR_BATCH, dtype=np.int64)
        x = torch.from_numpy(
            np.stack([ids[s : s + SEQ_LEN_CHARS] for s in starts])
        ).to(device)
        y = torch.from_numpy(
            np.stack([ids[s + 1 : s + SEQ_LEN_CHARS + 1] for s in starts])
        ).to(device)
        logits = model(x)
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)
            ).item()
        )
    return float(np.mean(losses))


# ── Checkpoint 1: encode/decode round-trip and model shapes ──────────
probe = text[:1000]
assert decode(encode(probe)) == probe, "encode/decode must round-trip"
_model_probe = CharLSTM().to(device)
_x, _y = sample_batch(train_ids)
_logits = _model_probe(_x)
assert _logits.shape == (CHAR_BATCH, SEQ_LEN_CHARS, VOCAB_SIZE), (
    f"CharLSTM logits should be (batch, seq, vocab) = "
    f"({CHAR_BATCH}, {SEQ_LEN_CHARS}, {VOCAB_SIZE}), got {_logits.shape}"
)
_n_params = sum(p.numel() for p in _model_probe.parameters())
uniform_ce = float(np.log(VOCAB_SIZE))
print(f"\nCharLSTM built: {_n_params:,} parameters")
print(f"  logits shape verified: {tuple(_logits.shape)}")
print(f"  uniform-random baseline: CE = ln({VOCAB_SIZE}) = {uniform_ce:.3f}, "
      f"perplexity = {VOCAB_SIZE}")
print(f"  encode/decode round-trip: OK ({len(probe)} chars)")
print("\n--- Checkpoint 1 passed --- vocabulary and model verified\n")
del _model_probe, _x, _y, _logits


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: 2-layer char LSTM, tracked with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 3 — TRAIN: char LSTM on next-character prediction")
print("=" * 70)

conn, tracker, exp_name, registry, has_registry = setup_engines(
    "text", experiment_suffix="charlstm"
)

torch.manual_seed(42)
np.random.seed(42)
model = CharLSTM().to(device)
opt = torch.optim.Adam(model.parameters(), lr=CHAR_LR)

train_ces: list[float] = []
val_ces: list[float] = []
val_ppls: list[float] = []


async def train_char_lstm() -> None:
    """Train and log per-epoch CE/perplexity to ExperimentTracker."""
    async with tracker.track(experiment=exp_name, run_name="char_lstm") as run:
        await run.log_params(
            {
                "architecture": "CharLSTM",
                "corpus": corpus_name,
                "vocab_size": str(VOCAB_SIZE),
                "seq_len": str(SEQ_LEN_CHARS),
                "hidden": str(HIDDEN_CHARS),
                "layers": str(N_LAYERS),
                "epochs": str(CHAR_EPOCHS),
                "steps_per_epoch": str(STEPS_PER_EPOCH),
                "lr": str(CHAR_LR),
            }
        )
        for epoch in range(CHAR_EPOCHS):
            model.train()
            step_losses = []
            for _ in range(STEPS_PER_EPOCH):
                xb, yb = sample_batch(train_ids)
                logits = model(xb)
                # TODO: cross-entropy over ALL timesteps at once — flatten
                #   (batch, seq, vocab) -> (batch*seq, vocab) and
                #   (batch, seq) -> (batch*seq)
                loss = F.cross_entropy(____)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step_losses.append(loss.item())

            train_ce = float(np.mean(step_losses))
            val_ce = mean_ce(val_ids, model)
            val_ppl = float(np.exp(val_ce))
            train_ces.append(train_ce)
            val_ces.append(val_ce)
            val_ppls.append(val_ppl)
            await run.log_metrics(
                {"train_ce": train_ce, "val_ce": val_ce, "val_perplexity": val_ppl},
                step=epoch + 1,
            )
            print(
                f"  [char_lstm] epoch {epoch+1}/{CHAR_EPOCHS}  "
                f"train CE={train_ce:.4f}  val CE={val_ce:.4f}  "
                f"val perplexity={val_ppl:.2f}"
            )
        await run.log_metrics(
            {"final_val_ce": val_ces[-1], "final_val_perplexity": val_ppls[-1]}
        )


asyncio.run(train_char_lstm())

# ── Checkpoint 2: the model beat the uniform baseline, honestly ──────
assert len(val_ces) == CHAR_EPOCHS
assert val_ces[-1] < val_ces[0] or train_ces[-1] < train_ces[0], (
    "training should reduce cross-entropy"
)
final_ppl = val_ppls[-1]
assert final_ppl < VOCAB_SIZE, (
    f"perplexity {final_ppl:.1f} must beat the uniform baseline {VOCAB_SIZE}"
)
assert final_ppl < 20.0, (
    f"perplexity {final_ppl:.1f} too high — a trained char-LSTM on this "
    "corpus reaches single digits; < 20 is the conservative floor"
)
print(
    f"\n  Final val CE={val_ces[-1]:.4f}, perplexity={final_ppl:.2f} "
    f"(uniform baseline: {VOCAB_SIZE})"
)
print("\n--- Checkpoint 2 passed --- perplexity beat the baseline\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: perplexity curve + temperature samples
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: perplexity and the temperature dial")
print("=" * 70)

fig_ppl, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
fig_ppl.suptitle("Character LSTM — cross-entropy and perplexity", fontsize=13)
epochs_x = range(1, CHAR_EPOCHS + 1)
ax_l.plot(epochs_x, train_ces, "o-", label="train CE", color="#2196F3")
ax_l.plot(epochs_x, val_ces, "s-", label="val CE", color="#FF5722")
ax_l.axhline(uniform_ce, color="gray", linestyle="--",
             label=f"uniform baseline ln({VOCAB_SIZE})={uniform_ce:.2f}")
ax_l.set_xlabel("Epoch")
ax_l.set_ylabel("Cross-entropy (nats)")
ax_l.legend(fontsize=9)
ax_l.grid(True, alpha=0.3)
ax_r.plot(epochs_x, val_ppls, "o-", color="#4CAF50", label="val perplexity")
ax_r.axhline(VOCAB_SIZE, color="gray", linestyle="--",
             label=f"uniform baseline = {VOCAB_SIZE}")
ax_r.set_xlabel("Epoch")
ax_r.set_ylabel("Perplexity = exp(CE)")
ax_r.legend(fontsize=9)
ax_r.grid(True, alpha=0.3)
fig_ppl.tight_layout()
fig_ppl.savefig(str(OUTPUT_DIR / "07_char_lstm_perplexity.png"), dpi=150)
plt.close(fig_ppl)
print(f"  Saved: {OUTPUT_DIR / '07_char_lstm_perplexity.png'}")


@torch.no_grad()
def generate(model: nn.Module, seed_text: str, n_chars: int = 300,
             temperature: float = 1.0) -> str:
    """Autoregressive sampling: slide the window, sample, append."""
    model.eval()
    ids = encode(seed_text)
    for _ in range(n_chars):
        window = torch.tensor(
            ids[-SEQ_LEN_CHARS:], dtype=torch.long, device=device
        ).unsqueeze(0)
        # TODO: last-timestep logits divided by the temperature, softmax,
        #   then SAMPLE (not argmax) one index with torch.multinomial
        logits = ____
        probs = F.softmax(logits, dim=-1)
        ids.append(____)
    return decode(ids)


SEED = "To be or not to be"
samples: dict[float, str] = {}
for temp in (0.5, 1.0, 1.5):
    torch.manual_seed(int(temp * 100))
    samples[temp] = generate(model, SEED, n_chars=300, temperature=temp)
    print(f"\n  ── temperature {temp} ──")
    print("  " + samples[temp].replace("\n", "\n  "))

# ── Checkpoint 3: artefacts and samples exist, temperatures differ ────
import os

assert os.path.exists(OUTPUT_DIR / "07_char_lstm_perplexity.png")
for temp, s in samples.items():
    assert len(s) == len(SEED) + 300, f"t={temp}: wrong sample length"
    assert set(s) <= set(chars), f"t={temp}: sample left the vocabulary"
assert len({samples[0.5], samples[1.0], samples[1.5]}) == 3, (
    "temperatures should produce different samples"
)
print("\n--- Checkpoint 3 passed --- perplexity curve + samples verified\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: drafting assistance at a professional services firm
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (anonymised, illustrative): a regional professional services
# firm drafts hundreds of engagement letters per quarter. Partners write
# the first paragraph; juniors assemble the boilerplate. The firm wants
# an autocomplete that proposes the next span of standard language.
#
# TWO QUESTIONS THE PARTNER WILL ASK — and the honest answers:
#
#   "How good is it?" — Perplexity translates directly:
#     our model's perplexity means that at each character it has
#     effectively narrowed ~70 possibilities down to ~perplexity-many.
#     That is a strong language model for boilerplate; it is NOT a
#     fact-checker. Character models learn FORM, not TRUTH.
#
#   "Can we tune how safe it plays?" — Yes: temperature.
#     T=0.5 for engagement letters (repetitive, safe, formulaic —
#     exactly what boilerplate wants); T=1.5 only for brainstorming
#     workshops where surprise is the point.
#
# WHAT WE WOULD NOT DO: ship raw character output into a client document
# without a human in the loop. The perplexity number tells you the model
# is fluent; it says nothing about whether the content is correct.

print("=" * 70)
print("  PHASE 5 — APPLY: drafting assistance, honestly scoped")
print("=" * 70)

compression = VOCAB_SIZE / final_ppl
print(
    f"""
  DRAFTING ASSISTANT — MODEL CARD (measured, this run):

    Corpus:                 {corpus_name} ({len(text):,} chars)
    Vocabulary:             {VOCAB_SIZE} characters
    Uniform baseline:       perplexity {VOCAB_SIZE} (no learning)
    Trained model:          perplexity {final_ppl:.2f}
    Effective narrowing:    ~{compression:.0f}x fewer live candidates
                            per character than uniform guessing

  OPERATING GUIDANCE:
    Engagement letters / compliance boilerplate:  temperature 0.5
    Internal report drafting:                     temperature 1.0
    Brainstorming / naming workshops:             temperature 1.5

  STAKEHOLDER-READY OUTPUT:
    "The drafting model reads the partner's opening paragraph and
    proposes the next span of standard language. At each character it
    has narrowed the field from {VOCAB_SIZE} possibilities to about
    {final_ppl:.0f} — fluent enough for boilerplate. It learns FORM, not
    TRUTH: every proposal still gets a human read before it leaves the
    building. The creativity dial (temperature) is set per document
    class: 0.5 for engagement letters, 1.5 for brainstorming."
"""
)

# ── Checkpoint 4: the perplexity interpretation is arithmetically sound
assert abs(np.log(final_ppl) - val_ces[-1]) < 1e-6, (
    "perplexity must equal exp(mean CE) by construction"
)
assert compression > 1.0, "trained model must narrow the candidate field"
print("--- Checkpoint 4 passed --- application quantified honestly\n")

asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  THEORY:
  [x] Generation = next-character classification, repeated; the
      vocabulary is the character set ({VOCAB_SIZE} symbols)
  [x] Perplexity = exp(mean CE): "as uncertain as a uniform choice over
      P characters". Uniform baseline = {VOCAB_SIZE}; ours = {final_ppl:.2f}
  [x] Temperature divides the logits: T<1 sharpens, T>1 flattens

  BUILD + TRAIN:
  [x] encode/decode with a sorted character vocabulary (round-trip
      verified on 1000 chars)
  [x] CharLSTM: embedding -> 2-layer LSTM -> per-timestep logits,
      {_n_params:,} parameters
  [x] {CHAR_EPOCHS} epochs x {STEPS_PER_EPOCH} random windows, gradient
      clipping, ExperimentTracker receipts (train CE {train_ces[0]:.3f}
      -> {train_ces[-1]:.3f})

  VISUALISE (the proof):
  [x] CE and perplexity curves against the uniform baseline line
  [x] Generated samples at three temperatures from the same seed
  [x] Vocabulary closure asserted: samples never leave the char set

  APPLY:
  [x] Drafting assistance with per-document-class temperature guidance
  [x] Perplexity translated for a partner: ~{compression:.0f}x narrower
      candidate field than uniform guessing
  [x] Stated limit: character models learn FORM, not TRUTH

  KEY INSIGHT: A language model's quality claim is only as good as its
  reference point. "Perplexity {final_ppl:.1f}" means nothing until you
  say "against a uniform baseline of {VOCAB_SIZE}". The same discipline
  you applied to accuracy baselines in M3 applies to generative text:
  measure, compare to the trivial baseline, and state what the metric
  does NOT cover.
"""
)
