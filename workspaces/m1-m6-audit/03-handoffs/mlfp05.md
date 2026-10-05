# mlfp05 — handoffs from the exercise shard (S1, merged 9ea5295f)

## Content corrections the teaching material must follow (deck, lessons, textbook, notes, spec)
- OnnxBridge: `framework="torch"` (NOT "pytorch"); trace with a 2-row `sample_input` (a batch-of-1 trace fixes the batch size); `validate()` needs a model with `predict()` and cannot take int64 token ids.
- InferenceServer: NO `predict_batch`, `warm_cache`, `PredictionResult`, `InferenceServer(model_path=...)`. Flow = register the ONNX artifact in the registry → `await InferenceServer.from_registry(...)` → start → `await predict(...)`.
- RLTrainer lives at `kailash_ml.rl.RLTrainer` (not top-level); the `rl_train` backend (stable_baselines3) is not installed — state the real status.
- `km.diagnose` / `diagnose(kind="dl")` return findings and print nothing — no "one line, same observability" claims; exercises print via `shared.mlfp05.diagnostics.print_prescription_pad`.
- Contractive AE = real per-sample Jacobian penalty (vmap(jacrev)), not weight decay. "CVAE" name clash: deck = Conditional VAE, exercise = Contractive VAE — resolve.
- WGAN-GP critic loss rises toward 0 as quality improves. Non-saturating generator loss is what the code uses.
- GNN model selection on validation; link prediction scored on held-out edges.
- Adapters start as identity (adapter adds `adapted − pooled`); trainable share ≈ 1%.
- ex_4 heads: a news-topic classifier is not sentiment/feedback/regulatory routing.
- Positional encoding: LOW dimensions oscillate FASTEST (deck/notes have it backwards — audit BLOCKING).
- ex_3 ticker: DBS is `D05.SI` (DBS.SI does not exist). ex_3/02 keeps DBS's real public price data (allowed by D2).
- Organisations anonymised elsewhere.

## Other files
- diagnostic-reference/**: regenerate reports after execution, fix README, commit or remove the capture script.
- specs/module-5.md §5.7 (InferenceServer API), §5.1 (contractive penalty).

## Integration (S7)
- Regenerate notebooks (shared/mlfp05 ex_2, ex_3, ex_4 and new diagnostics.py changed).
- Execute all 44 changed solutions; run ex_2/03, ex_4/05, ex_7/05 FIRST (ONNX export/serving only tested on toy models).

## Deferred spec gaps (S6)
- 5.2 Mixup, label smoothing, Kaiming init. 5.3 technical indicators, char-level LSTM generation, perplexity. 5.5 DCGAN, real Inception Score. 5.6 GIN, graph classification (TUDataset). 5.8 DDPG, SAC, A2C.

---
# Additions from the deck shard (S2a, merged 08ffdfb4)
- OnnxBridge.export needs output_path as a pathlib.Path (a str fails) — teach Path everywhere.
- Deck now 129 slides (2 new: stacked LSTMs & spatial attention; GIN with torch_geometric graph classification). Spec gaps shown on slides as "extensions" pending S6.
- Assessment slide describes the 5-question quiz + four graded tasks (S5 must keep consistent).
- speaker-notes.md rebuild from the deck (129 slides). deck.pdf + readings/deck.pdf regenerate + parity --update. index.html/README: RLTrainer, InferenceServer, exercise descriptions.

---
# Additions from the lesson-slides shard (S2b, merged 9c86b626)
- ModelVisualizer: training_history() / scatter() on polars (lesson 5.1).
- BERT: transformers TrainingArguments uses eval_strategy (evaluation_strategy removed).
- 5.3 exercise = next 5 STI closes from real daily bars (no RSI/MACD); 5.6 = node classification on Cora; 5.7 BERT = AG News, 4 labels; 5.5 MLP GANs (no DCGAN yet); 5.8 hand-written DQN/PPO on discrete actions.
- 5.4 worked example uses an order-dependent task (position-free model capped ≈59%, measured on 200k samples). Whisper is seq2seq, not CTC. SE adds ≈10% params. Bellman: expectation vs optimality labelled. Gymnasium `terminated`.
- notes.html regenerate from slides (5.7 and 5.8 each gained one slide).

---
# Additions from the textbook shard (S3a, merged e74bc748)
- BUG: OnnxBridge writes weights to a separate `.onnx.data` file; shared/mlfp05/ex_7.attach_onnx_artifact stores only the
  `.onnx` → ONNX Runtime "external data path does not exist" at serve time (reproduced). Fix: re-save with
  `onnx.save_model(onnx.load(p), p, save_as_external_data=False)` before attaching (or attach both files). Spec §5.7 note.
  Upstream kailash-ml note.
- OnnxBridge puts a wrapper left in train mode back into training mode after export → parity checks break (max diff 2.4).
  Always `.eval()` wrappers (ex_7 does; check others).
- ex_3: STI.parquet Volume = 0 on 29% of days; price-LEVEL targets extrapolate badly (val err 0.66 vs 0.031 persistence)
  → use percentage-change targets + persistence baseline (S6).
- ex_8 ChurnPreventionEnv: random starting tenure + elapsed-time bonus → not Markov; textbook uses fraction-of-month
  elapsed (S6 fix in exercise).
- Lesson pages: 5.2 small residual CNN on CIFAR-10; 5.4 BERT on bundled AG News slice (TF-IDF 0.855, majority 0.274);
  5.5 conv DCGAN on MNIST in [-1, 1]; 5.6 epochs on validation (GCN 0.804 vs 0.540 graph-free).
