# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4.4: Fine-Tuning Pre-Trained BERT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain the difference between pre-training and fine-tuning
#   - Describe how transfer learning works for NLP (language -> task)
#   - Fine-tune a pre-trained BERT model on AG News classification
#   - Implement layer-wise freezing for efficient fine-tuning
#   - Use BERT's WordPiece tokeniser vs our word-level vocabulary
#   - Track fine-tuning experiments with ExperimentTracker
#   - Recognise when a fine-tuned classifier is the WRONG tool: a topic
#     head run on out-of-distribution bank messages
#
# PREREQUISITES: ex_4/02_transformer_encoder.py
# ESTIMATED TIME: ~30 min
# DATASET: AG News — 120,000 real news headlines, 4 classes.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

from shared.mlfp05.ex_4 import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LR,
    BERT_MAX_LEN,
    BERT_MODEL_NAME,
    CLASS_NAMES,
    DEVICE,
    load_ag_news,
    setup_engines,
)

print(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Pre-Training vs Fine-Tuning
# ════════════════════════════════════════════════════════════════════════
# Training a Transformer from scratch (as we did in 02_) requires large
# datasets to learn both language understanding AND the task. BERT takes
# a different approach: it separates these two stages.
#
# STAGE 1 — PRE-TRAINING (done by researchers, once):
#   BERT is trained on massive text corpora (BookCorpus + Wikipedia,
#   ~3.3 billion words) with two self-supervised tasks:
#     1. Masked Language Modelling (MLM): predict randomly masked words
#        "The [MASK] sat on the mat" -> "cat"
#     2. Next Sentence Prediction (NSP): predict if two sentences follow
#        each other in the original text.
#   This teaches BERT the structure of language: grammar, semantics,
#   common sense, and factual knowledge. Pre-training takes days on
#   hundreds of GPUs and costs hundreds of thousands of dollars.
#
# STAGE 2 — FINE-TUNING (done by practitioners, for each task):
#   We take the pre-trained BERT and add a small classification head.
#   Then we fine-tune the top layers on our specific task (AG News
#   classification). This is dramatically more sample-efficient:
#     - From-scratch Transformer: needs 120K+ examples
#     - Fine-tuned BERT: can work well with just 1,000 examples
#
# WHY THIS MATTERS: Transfer learning is the single biggest lever in
# modern NLP. Instead of learning English from scratch, BERT brings
# pre-trained knowledge of vocabulary, grammar, semantics, and even
# some reasoning ability. Fine-tuning just teaches it the mapping
# from language understanding to your specific classification task.
#
# LAYER-WISE FREEZING: We freeze BERT's lower layers (which capture
# general language patterns) and only fine-tune the top layers (which
# can be adapted to our task). This is faster, requires less memory,
# and prevents "catastrophic forgetting" of the pre-trained knowledge.
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and set up engines
# ════════════════════════════════════════════════════════════════════════
train_df, test_df = load_ag_news()
conn, tracker, exp_name, registry, has_registry, bridge = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build: Load pre-trained BERT + configure layer freezing
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Loading pre-trained {BERT_MODEL_NAME} ==")
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=4
).to(DEVICE)

# Freeze the lower 8 of 12 encoder layers -- only fine-tune the top 4
# layers plus the pooler and classification head. This is faster and
# prevents catastrophic forgetting of the pre-trained representations.
for name, param in bert_model.named_parameters():
    if "bert.encoder.layer" in name:
        layer_num = int(name.split(".")[3])
        if layer_num < 8:
            param.requires_grad = False
    elif "bert.embeddings" in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in bert_model.parameters())
print(
    f"  BERT params: {total_params:,} total, {trainable:,} trainable "
    f"({trainable/total_params:.1%} unfrozen)"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert total_params > 100_000_000, "BERT-base should have ~110M parameters"
assert trainable < total_params, "Should have frozen some layers"
assert trainable / total_params < 0.5, "Should freeze at least half the layers"
# INTERPRETATION: We're fine-tuning only ~30% of BERT's parameters. The
# frozen layers already encode rich language understanding from pre-training.
# We only need to adapt the top layers to our classification task.
print("\n--- Checkpoint 1 passed --- BERT loaded with layer freezing\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build: BERT tokenisation (WordPiece, not word-level)
# ════════════════════════════════════════════════════════════════════════
# BERT uses WordPiece tokenisation, which splits rare words into subword
# units: "unconditional" -> ["un", "##condition", "##al"]. This gives
# BERT a fixed vocabulary (~30K tokens) that can represent any word,
# even ones it never saw during pre-training.
#
# BERT also adds special tokens:
#   [CLS] at the start: its final representation is used for classification
#   [SEP] at the end: marks the end of the input sequence
#   Padding to a fixed length with [PAD] tokens


def tokenise_for_bert(
    texts: list[str], max_len: int = BERT_MAX_LEN
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise text with BERT's WordPiece tokeniser.

    Returns:
        (input_ids, attention_mask) tensors ready for BERT.
    """
    encoding = bert_tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


# Tokenise full train and test sets
print("  Tokenising train + test sets for BERT...")
bert_train_ids, bert_train_mask = tokenise_for_bert(train_df["text"].to_list())
bert_test_ids, bert_test_mask = tokenise_for_bert(test_df["text"].to_list())
bert_train_y = torch.tensor(train_df["label"].to_list(), dtype=torch.long)
bert_test_y = torch.tensor(test_df["label"].to_list(), dtype=torch.long)

bert_train_loader = DataLoader(
    TensorDataset(
        bert_train_ids.to(DEVICE), bert_train_mask.to(DEVICE), bert_train_y.to(DEVICE)
    ),
    batch_size=BERT_BATCH_SIZE,
    shuffle=True, num_workers=0)
bert_test_loader = DataLoader(
    TensorDataset(
        bert_test_ids.to(DEVICE), bert_test_mask.to(DEVICE), bert_test_y.to(DEVICE)
    ),
    batch_size=BERT_BATCH_SIZE, num_workers=0)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert bert_train_ids.shape[0] == len(train_df), "Should tokenise all training samples"
assert bert_train_ids.shape[1] == BERT_MAX_LEN, "Should pad/truncate to BERT_MAX_LEN"
print(f"  Tokenised {len(train_df):,} train + {len(test_df):,} test samples")
print("\n--- Checkpoint 2 passed --- BERT tokenisation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Train: Fine-tune BERT with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
async def train_bert_async(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = BERT_EPOCHS,
    lr: float = BERT_LR,
) -> tuple[list[float], list[float]]:
    """Fine-tune BERT and log to ExperimentTracker."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs
    )
    train_losses: list[float] = []
    test_accs: list[float] = []

    async with tracker.track(experiment=exp_name, run_name="bert_finetune") as run:
        await run.log_params(
            {
                "model_type": "bert_finetune",
                "base_model": BERT_MODEL_NAME,
                "epochs": str(epochs),
                "lr": str(lr),
                "frozen_layers": "0-7",
                "trainable_params": str(trainable),
                "dataset_size": str(len(train_loader.dataset)),
            }
        )

        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for batch_idx, (ids, mask, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())
                if (batch_idx + 1) % 500 == 0:
                    print(
                        f"    batch {batch_idx+1}/{len(train_loader)}  "
                        f"loss={np.mean(batch_losses[-500:]):.4f}"
                    )
            scheduler.step()
            epoch_loss = float(np.mean(batch_losses))
            train_losses.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                correct = 0
                total_count = 0
                for ids, mask, labels in test_loader:
                    logits = model(input_ids=ids, attention_mask=mask).logits
                    preds = logits.argmax(dim=-1)
                    correct += int((preds == labels).sum().item())
                    total_count += int(labels.size(0))
                acc = correct / total_count
                test_accs.append(acc)

            await run.log_metrics(
                {"train_loss": epoch_loss, "test_accuracy": acc}, step=epoch + 1
            )
            print(
                f"  [BERT] epoch {epoch+1}/{epochs}  "
                f"loss={epoch_loss:.4f}  test_acc={acc:.3f}"
            )

        await run.log_metrics(
            {
                "final_test_accuracy": test_accs[-1],
                "final_train_loss": train_losses[-1],
            }
        )

    return train_losses, test_accs


print(f"\n== Fine-tuning {BERT_MODEL_NAME} on AG News ==")
bert_losses, bert_accs = asyncio.run(
    train_bert_async(bert_model, bert_train_loader, bert_test_loader, epochs=BERT_EPOCHS)
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — BERT fine-tuning (HF batch format)
# ══════════════════════════════════════════════════════════════════
# BERT batches are (ids, mask, labels) tuples, so run_diagnostic_checkpoint
# gets a batch_adapter that unpacks them for the loss function. Probes
# come from the training loader.
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _bert_loss(m, ids, mask, labels):
    return m(input_ids=ids, attention_mask=mask, labels=labels).loss


def _bert_adapter(batch):
    # BERT batches are (ids, mask, labels), passed to the loss as three args
    return batch[0], batch[1], batch[2]


print("\n── Diagnostic Report (BERT fine-tune) ──")
diag, findings = run_diagnostic_checkpoint(
    bert_model,
    bert_train_loader,
    _bert_loss,
    title="BERT fine-tuned (AG News)",
    n_batches=4,  # BERT batches are expensive; 4 is enough for the readings
    train_losses=bert_losses,
    batch_adapter=_bert_adapter,
    show=False,
)
print_prescription_pad(findings, "BERT fine-tuned (AG News)")

# ══════ READING THE PRESCRIPTION PAD (key: see ex_1/01_standard_ae.py) ══════
# Layers 0-7 and the embeddings are frozen (requires_grad=False), so
# they receive NO gradient — expect the gradient reading to reflect the
# unfrozen layers 8-11 and the classifier head only; a "zero gradient"
# on frozen layers is by design, not vanishing. BERT's feed-forward
# blocks use GELU, so the dead-ReLU check has little to say here.
# ══════════════════════════════════════════════════════════════════

# BERT is evaluated on the TEST split after every epoch only to watch
# progress — no epoch is selected on it, so the honest number to report
# is the FINAL model's test accuracy.
bert_test_acc = bert_accs[-1]

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(bert_losses) == BERT_EPOCHS, "BERT should train for all epochs"
assert (
    bert_test_acc > 0.85
), f"BERT should reach >85% test accuracy with fine-tuning, got {bert_test_acc:.3f}"
# INTERPRETATION: BERT's pre-trained language understanding gives it a
# massive head start. While our from-scratch models need to learn word
# meanings, syntax, and semantics from 120K headlines, BERT already
# "knows" English from billions of words of pre-training. Fine-tuning
# just teaches it the specific mapping from language to news categories.
print(f"\n  BERT test accuracy (final epoch): {bert_test_acc:.3f}")
print("\n--- Checkpoint 3 passed --- BERT fine-tuned\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Per-class accuracy breakdown
# ════════════════════════════════════════════════════════════════════════
print("\n== BERT Per-Class Accuracy ==")
bert_model.eval()
class_correct: Counter[int] = Counter()
class_total: Counter[int] = Counter()
with torch.no_grad():
    for ids, mask, labels in bert_test_loader:
        logits = bert_model(input_ids=ids, attention_mask=mask).logits
        preds = logits.argmax(dim=-1)
        for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

for i, cls_name in enumerate(CLASS_NAMES):
    acc = class_correct[i] / max(class_total[i], 1)
    print(f"  {cls_name:<10} {acc:.3f} ({class_correct[i]}/{class_total[i]})")

# ── Visualise: per-class accuracy bar chart ─────────────────────────
from shared.mlfp05.ex_4 import get_viz
import plotly.graph_objects as go

viz = get_viz()

per_class_accs = [
    class_correct[i] / max(class_total[i], 1) for i in range(len(CLASS_NAMES))
]
fig_bar = go.Figure(
    data=go.Bar(
        x=CLASS_NAMES,
        y=per_class_accs,
        marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
        text=[f"{a:.1%}" for a in per_class_accs],
        textposition="auto",
    )
)
fig_bar.update_layout(
    title="BERT Fine-Tuned — Per-Class Accuracy on AG News",
    xaxis_title="News Category",
    yaxis_title="Accuracy",
    yaxis=dict(range=[0, 1]),
)
fig_bar.write_html("ex_4_4_bert_per_class_accuracy.html")
print("\n  Per-class accuracy chart saved to ex_4_4_bert_per_class_accuracy.html")

# ── Visualise: BERT training loss curve ─────────────────────────────
fig_loss = viz.training_history(
    metrics={"BERT train_loss": bert_losses},
    x_label="Epoch",
    y_label="Cross-Entropy Loss",
)
fig_loss.write_html("ex_4_4_bert_training_loss.html")
print("  Training loss curve saved to ex_4_4_bert_training_loss.html")

# ── Visualise: before/after fine-tuning comparison ──────────────────
# Evaluate BERT BEFORE fine-tuning by loading a fresh model (no training)
print("\n  Computing before/after fine-tuning comparison...")
bert_before = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=4
).to(DEVICE)
bert_before.eval()
with torch.no_grad():
    before_correct = 0
    before_total = 0
    before_class_correct: Counter[int] = Counter()
    before_class_total: Counter[int] = Counter()
    for ids, mask, labels in bert_test_loader:
        logits = bert_before(input_ids=ids, attention_mask=mask).logits
        preds = logits.argmax(dim=-1)
        before_correct += int((preds == labels).sum().item())
        before_total += int(labels.size(0))
        for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            before_class_total[label] += 1
            if pred == label:
                before_class_correct[label] += 1
del bert_before  # free memory

before_per_class = [
    before_class_correct[i] / max(before_class_total[i], 1)
    for i in range(len(CLASS_NAMES))
]
after_per_class = per_class_accs
before_overall = before_correct / max(before_total, 1)
after_overall = bert_test_acc

fig_compare = go.Figure()
fig_compare.add_trace(
    go.Bar(
        name="Before Fine-Tuning (random head)",
        x=CLASS_NAMES + ["Overall"],
        y=before_per_class + [before_overall],
        marker_color="rgba(99, 110, 250, 0.4)",
        text=[f"{a:.1%}" for a in before_per_class + [before_overall]],
        textposition="auto",
    )
)
fig_compare.add_trace(
    go.Bar(
        name="After Fine-Tuning",
        x=CLASS_NAMES + ["Overall"],
        y=after_per_class + [after_overall],
        marker_color="rgba(99, 110, 250, 1.0)",
        text=[f"{a:.1%}" for a in after_per_class + [after_overall]],
        textposition="auto",
    )
)
fig_compare.update_layout(
    title="BERT — Before vs After Fine-Tuning on AG News",
    xaxis_title="Category",
    yaxis_title="Accuracy",
    yaxis=dict(range=[0, 1]),
    barmode="group",
)
fig_compare.write_html("ex_4_4_bert_before_after_comparison.html")
print("  Before/after comparison saved to ex_4_4_bert_before_after_comparison.html")
print(
    f"  Before fine-tuning: {before_overall:.1%} overall  |  "
    f"After: {after_overall:.1%} overall"
)

# ── Visualise: BERT training history (loss + accuracy) ──────────────
fig_history = viz.training_history(
    metrics={
        "BERT train_loss": bert_losses,
        "BERT val_accuracy": bert_accs,
    },
    x_label="Epoch",
    y_label="Value",
)
fig_history.write_html("ex_4_4_bert_training_curves.html")
print("  BERT training curves saved to ex_4_4_bert_training_curves.html")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert sum(class_total.values()) >= 5000, "Should evaluate on full test set"
# INTERPRETATION: BERT's per-class accuracy reveals which news categories
# are easiest and hardest. Sports is typically the easiest (distinctive
# vocabulary), while World/Business can be confused (both discuss economics,
# politics, and international events). The before/after comparison shows
# the dramatic impact of fine-tuning: BERT with a random classification head
# performs near-chance (~25%), but after just a few epochs of fine-tuning,
# it achieves >85% accuracy by leveraging its pre-trained language understanding.
# This per-class view is critical for production deployment -- if one category
# underperforms, you know where to focus additional training data.
print("\n--- Checkpoint 4 passed --- per-class analysis complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: What a News-Topic Model Does With Bank Messages
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The customer-experience team at a Singapore retail bank wants
# to triage incoming customer messages — complaints vs praise — and asks
# whether "the BERT model you just fine-tuned" can do it.
#
# It cannot, and showing WHY is the lesson. You fine-tuned BERT on AG
# News, whose labels are TOPICS (World, Sports, Business, Sci/Tech). A
# classifier can only answer the question its labels asked: fed bank
# messages, it returns a news topic for each one — never "complaint".
# Its softmax confidence on these out-of-distribution (OOD) messages is
# NOT evidence that it understood them: neural classifiers are often
# confidently wrong off-distribution. Below we compare that confidence
# with the model's confidence on the in-distribution test headlines.
#
# THE RIGHT FIX: fine-tune a sentiment head on labelled customer
# messages — the same recipe as TASK 4, with different labels. Until
# then the business value is zero, and the cost of deploying the wrong
# head is complaints silently filed as "Business news".
print("\n== Application: a topic model meets bank customer messages ==")

bank_messages = [
    "Digital banking app crashes every time I try to transfer funds",
    "Excellent service from the relationship manager at the city branch",
    "Interest rates on savings account lower than competitors",
    "New bill-splitting feature in the app makes paying friends easy",
    "Three weeks waiting for credit card replacement is unacceptable",
]

bert_model.eval()
with torch.no_grad():
    msg_ids, msg_mask = tokenise_for_bert(bank_messages)
    msg_ids = msg_ids.to(DEVICE)
    msg_mask = msg_mask.to(DEVICE)
    msg_logits = bert_model(input_ids=msg_ids, attention_mask=msg_mask).logits
    msg_probs = F.softmax(msg_logits, dim=-1)
    msg_preds = msg_logits.argmax(dim=-1).cpu().tolist()

print(f"\n  Bank messages through the AG News topic head:")
print(f"  {'Message':<55} {'Topic':<12} {'Confidence':>10}")
print("  " + "-" * 79)
for text, pred, probs in zip(bank_messages, msg_preds, msg_probs.cpu().tolist()):
    print(f"  {text[:53]:<55} {CLASS_NAMES[pred]:<12} {max(probs):>10.1%}")

# Confidence on OOD messages vs on the test headlines the head was built for
with torch.no_grad():
    in_dist = []
    for b, (ids, mask, _labels) in enumerate(bert_test_loader):
        logits = bert_model(input_ids=ids, attention_mask=mask).logits
        in_dist.append(F.softmax(logits, dim=-1).max(dim=-1).values.cpu())
        if b == 4:
            break
in_dist_conf = float(torch.cat(in_dist).mean())
ood_conf = float(msg_probs.max(dim=-1).values.mean())
print(f"\n  Mean top-class confidence, AG News test headlines: {in_dist_conf:.1%}")
print(f"  Mean top-class confidence, bank messages (OOD):   {ood_conf:.1%}")
print("  However high the second number is, every answer above is a news")
print("  TOPIC. Confidence measures how peaked the softmax is, not whether")
print("  the question was the right one.")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(msg_preds) == len(bank_messages), "Should classify all messages"
assert all(0 <= p < len(CLASS_NAMES) for p in msg_preds), "Outputs are topic ids"
print("\n--- Checkpoint 5 passed --- out-of-distribution check complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — BERT Fine-Tuning")
print("=" * 70)
print(
    f"""
  [x] Explained pre-training vs fine-tuning (language knowledge -> task)
  [x] Loaded pre-trained BERT and configured layer-wise freezing
  [x] Used BERT's WordPiece tokeniser (subword, not word-level)
  [x] Fine-tuned BERT on AG News, test acc (final epoch): {bert_test_acc:.1%}
  [x] Analysed per-class accuracy for production deployment decisions
  [x] Showed why a topic head cannot do sentiment, and why OOD
      confidence is not evidence

  KEY INSIGHT:
    Pre-training is the single biggest lever in NLP. The Transformer
    architecture enables it, but the pre-trained weights are what make
    BERT dominate. This is why modern NLP is "pre-train then fine-tune"
    -- you get billions of words of language understanding for free.

  Next: In 05_three_way_comparison.py, you'll see all three models
  side by side: LSTM vs Transformer vs BERT. The comparison reveals
  the exact value of attention (LSTM -> Transformer) and pre-training
  (Transformer -> BERT).
"""
)
