# MLFP05 Audit — Deep Learning (modules/mlfp05)

Audit date: 2026-10-03. Repo: `courses/mlfp` on branch `fix/m1-m6-audit`. This audit is read-only. Installed stack used for API checks: kailash-ml 2.2.2, torch 2.12.1, torchvision 0.27.1, transformers 4.57.6. The `stable_baselines3` package is NOT installed.

Out of scope (already known): the quiz, the DLDiagnostics slides 5B2/5D/5F/5H/5L, Colab notebooks, and solution runtime pass/fail.

Paths are relative to `modules/mlfp05/` unless they start with `shared/`, `specs/` or `data/`.

---

## BLOCKING

### [BLOCKING] Assessment graders can be passed with an untrained model and a made-up `y_test` (Tasks 1, 3 and 4)

**File**
- `assessment/task_1/grader.py:32-43, 98-109, 114-126`
- `assessment/task_3/grader.py:105-124`
- `assessment/task_4/grader.py:104-119`
- `assessment/README.md:5-8, 95-96`

**Evidence**

1. **Accuracy and AUC are scored against labels the student returns.** Every grader scores accuracy/AUC/MSE against the `y_test` returned in the student's dict. It never uses the re-derived reference labels, even though Tasks 2–4 compute `y_test_ref` and then use it only for shape checks. For example, `task_4/grader.py:113`: `acc = float((preds == y_test).mean())` where `y_test = r["y_test"]`.

2. **Task 1's "fresh healthy" batch is off-manifold.** `_fresh_eval_batches()` draws a new `basis` from `default_rng(20260624)`. So its "fresh healthy" batch is not on the healthy manifold the AE was trained on. Measured on the reference model:

   | Batch | Reconstruction MSE |
   | --- | --- |
   | In-distribution healthy | 0.058 |
   | Grader's "fresh healthy" | 2.29 |
   | Anomaly | 4.85 |

   The anomaly/healthy ratio is 2.12. An **untrained** random AE gets a ratio of 1.99–2.04 over 5 seeds, which also clears the 1.5× bar. The check therefore cannot tell a trained model from an untrained one.

3. **Exploit run.** I monkey-patched `load_student_module` to return stub `solve()` functions. Nothing was written to the repo.

   | Task | Stub submission | Result |
   | --- | --- | --- |
   | Task 1 | untrained AE, random scores, `y_test = scores > 0.8` | **passed = True, 8/8** |
   | Task 3 | untrained GRU, `y_test = test_pred` | **passed = True, 8/8** |
   | Task 4 | untrained embedding-mean model with an *unused* `nn.MultiheadAttention` attribute, `y_test ≈ preds` | **passed = True, 8/8** |
   | Task 2 | untrained CNN | failed only `heldout_accuracy_at_least_0p88` (that check uses `y_test_ref`) |

4. **The structure checks only look for presence.** The Conv2d, GRU and MultiheadAttention checks use `isinstance` over `model.modules()`. A layer that is declared but never called in `forward()` passes.

**Problem**

The README says: "each grader re-derives its data, re-runs your returned model, and checks the model itself produces the claimed result — a faked output array fails". This is false for 3 of the 4 tasks. The summative assessment can be passed with zero training, which breaks the assessment's integrity and its AI-resilience claim.

**Fix**

1. Score every metric against `y_test_ref` / `naive_ref` from the grader's own re-derivation. Ignore the submitted labels, or require `np.array_equal(r["y_test"], y_test_ref)` as a check.
2. In Task 1, regenerate fresh healthy data on the **same** basis: draw `basis` from `default_rng(7)` exactly as `make_dataset` does, then use a different rng for `z`/noise. Then require a large ratio: the reference gives about 80× on true in-distribution data, so a floor of about 10× is safe.
3. Add a "model actually uses the layer" check. Either register a forward hook on the Conv2d/GRU/attention module and assert it fired during the re-run, or compare against an ablated copy.
4. Optionally, require the re-run model's own score to clear the floor (as Task 2 does) rather than the submitted array's score.

### [BLOCKING] OnnxBridge code shown to students does not match the installed API (wrong kwargs, missing `framework`)

**File**
- `deck.html:2788-2803` (Lesson 5.2 "Kailash Bridge: OnnxBridge")
- `deck.html:5632-5642` (Lesson 5.7 bridge slide)
- `lessons/02/slides.html:371-379`
- `lessons/07/slides.html:166-169`
- `textbook.md:461, 1271`
- `lessons/02/textbook.html:303-307, 440-444`
- `lessons/07/textbook.html:511-518, 662`

**Evidence**

The installed signatures are:

```
export(self, model, framework: str, schema=None, *, output_path=None, n_features=None, sample_input=None) -> OnnxExportResult
validate(self, model, onnx_path, sample_input, *, tolerance=1e-4) -> OnnxValidationResult
```

- `OnnxValidationResult` has the fields `valid, max_diff, mean_diff, n_samples, notes`. It has no `status`.
- The deck calls `bridge.export(model=resnet, input_shape=(1,1,28,28), output_path=...)` and then `bridge.validate("resnet_fmnist.onnx")` and `result.status  # "valid"`. This omits the required `framework`, passes the unknown kwarg `input_shape`, gives `validate` only one of its three required arguments, and reads a non-existent attribute.
- `lessons/07/slides.html:169`, `bridge.export(model, "model.onnx", sample_input=dummy)`, binds the file path to `framework`.
- Other variants use `dummy_input=`, `input_names=`, `output_names=` and `dynamic_axes=`, none of which exist.
- `lessons/02/slides.html:379` and `lessons/02/textbook.html:295-296` claim that export "validates PyTorch↔ONNX parity automatically. If an op is unsupported, it raises". In fact `export` never validates, and on failure it returns `success=False` rather than raising.

**Problem**

Every OnnxBridge snippet in the slides and textbooks raises `TypeError` or `AttributeError`. This is the module's headline Kailash engine (spec 5.2/5.7, "ONNX export successful").

**Fix**

Use one canonical snippet everywhere:

```python
res = OnnxBridge().export(model, "torch", output_path=p, sample_input=x)  # "torch", not "pytorch" (see ex_2/03 finding)
assert res.success
v = OnnxBridge().validate(model, p, x)
print(v.valid, v.max_diff)
```

Remove the "validates automatically / raises" prose.

### [BLOCKING] InferenceServer is taught with a constructor, methods and result object that do not exist

**File**
- `deck.html:319-` (slide 4 "Kailash Engines": "predict, predict_batch, warm_cache")
- `deck.html:5644-5674` (Lesson 5.7 bridge slide + speaker note)
- `lessons/07/slides.html:171-175`
- `textbook.md:1273-1275`
- `lessons/07/textbook.html:500-503, 521-531, 664-669`
- `lessons/07/notes.html:25`
- `speaker-notes.md:58, 61, 1111-1115, 1126`
- `specs/module-5.md:377`

**Evidence**

The installed API is:

```
InferenceServer.__init__(self, config: InferenceServerConfig, *, registry: ModelRegistry, server_id=None)
predict(self, features: Mapping[str, Any], *, tenant_id=None) -> Mapping[str, Any]
```

- `hasattr(InferenceServer, 'predict_batch')` is False, and so is `hasattr(InferenceServer, 'warm_cache')`.
- No `PredictionResult` type exists.
- The deck shows `InferenceServer(model_path="mask_detector.onnx")`, `server.warm_cache()`, `server.predict(image_tensor)`, `result.label`, and `result.confidence`. It also has a method table listing `predict_batch(xs)` and `warm_cache()`, and says "PredictionResult: Returns .label, .confidence, .probabilities, .latency_ms".

**Problem**

Every InferenceServer snippet raises. The speaker notes coach instructors to explain `warm_cache()` and `PredictionResult`, which do not exist.

**Fix**

Teach the registry flow that exists:

1. Register the ONNX artifact in `ModelRegistry`.
2. Call `server = await InferenceServer.from_registry(name, registry=registry, runtime="onnx")`. It is a coroutine function.
3. Call `await server.predict({...})`, which returns a mapping.

Delete `predict_batch`, `warm_cache` and `PredictionResult` from the deck, lesson pages, notes and `specs/module-5.md` §5.7.

### [BLOCKING] RLTrainer code in the deck raises; the RLTrainer prose is false

**File**
- `deck.html:6409-6461` ("Kailash Bridge: RLTrainer")
- `deck.html:6550` ("RLTrainer: train any algorithm with Kailash")
- `deck.html:319-` (slide 4)
- `lessons/08/textbook.html:591-596`
- `speaker-notes.md:6, 58, 1321-1335, 1346, 1372`
- `README.md:3`
- `index.html` (Tooling table: "kailash_ml.RLTrainer — Wrap policy-gradient / DQN training loops — 5.8")
- `solutions/ex_8/01_dqn.py:4-8`

**Evidence**

- `from kailash_ml import RLTrainer` fails: `hasattr(kailash_ml, 'RLTrainer')` is False.
- The real class is `kailash_ml.rl.RLTrainer(env_registry=None, policy_registry=None, *, root_dir, tenant_id)` with `train(env_name, policy_name, config: RLTrainingConfig)`. The functional entry point is `km.rl_train(env, algo='ppo', *, total_timesteps, hyperparameters, ...)`, and its backend `stable_baselines3` is not installed.
- The deck constructs `RLTrainer(algorithm="ppo", env=ChurnEnv(), policy="MlpPolicy", learning_rate=..., n_steps=..., ...)` and calls `trainer.evaluate(n_episodes=100)` → `metrics.mean_reward`. The real `evaluate(model, env_name, n_episodes)` returns a tuple.
- The speaker note at 6461 says "Students use this for the exercise instead of writing raw training loops". The solution file says instead: "RLTrainer is planned but not yet released. This exercise implements DQN from scratch". That note is also stale, because `kailash_ml.rl.RLTrainer` ships in 2.2.2.

**Problem**

The RL "Kailash bridge" slide teaches a non-existent API, and it misdescribes what the exercise does. All five sources disagree with each other and with the installed package.

**Fix**

1. Replace the slide code with `km.rl_train(ChurnEnv, algo="ppo", total_timesteps=100_000, hyperparameters={...})`, or with `kailash_ml.rl.RLTrainer` + `RLTrainingConfig(algorithm="PPO", ...)`. Add the RL extra (`stable-baselines3`) to the environment.
2. Say plainly that the exercises hand-write DQN and PPO, and that RLTrainer is the production path.
3. Remove the stale "not yet released" note in ex_8/01.

### [BLOCKING] ModelVisualizer methods shown in lesson slides and textbooks do not exist

**File**
- `lessons/01/slides.html:356-360` (`viz.plot_training_curve(losses=...)`, `viz.plot_latent_scatter(z_vectors=..., labels=...)`)
- `lessons/01/textbook.html:335-336`
- `lessons/05/textbook.html:493` (`viz.plot_2d_distribution`)
- `lessons/06/textbook.html:675` (`viz.scatter_2d`)
- `textbook.md:657` (`viz.line`)
- `textbook.md:860` (`viz.heatmap`)

**Evidence**

The installed public methods are: `box_plot, calibration_curve, confusion_matrix, feature_importance, histogram, learning_curve, metric_comparison, precision_recall_curve, residuals, roc_curve, scatter, training_history`. None of the called names exist. The real signatures are `scatter(data, x, y, *, color=None, title=None)` and `training_history(metrics: dict[str, list[float]], x_label, y_label)`.

Related false capability claims:
- `lessons/03/textbook.html:313-316`: "forecast intervals, rolling error bands"
- `lessons/04/textbook.html:389-393`: "attention weight heatmaps"

**Problem**

Every one of these calls raises `AttributeError`. Lesson 5.1's "What you can now do" lists "Visualise training curves and latent spaces with ModelVisualizer" using this broken code.

**Fix**

- Use `viz.training_history({"loss": history})` for loss curves.
- Use `viz.scatter(pl.DataFrame({...}), x="z1", y="z2", color="label")` for latent and embedding plots.
- Remove the unsupported capability claims.

### [BLOCKING] Positional-encoding frequencies are taught backwards (low dims are fast, not slow)

**File**
- `deck.html:3870` (SVG aria-label)
- `deck.html:4035-4041` (SVG annotations: "high dim: rapid oscillation (captures local position)"; "low dim: slow oscillation (captures global position)", with the "high dim" arrow pointing at row d=0)
- `deck.html:4055` (speaker note)
- `speaker-notes.md:680`: "Low dimensions change slowly (capture global position). High dimensions change rapidly (capture local position)."

**Evidence**

From the slide's own formula, `PE(pos,2i) = sin(pos / 10000^{2i/d})`, the angular frequency is `10000^{-2i/d}`:

- At i = 0 the frequency is 1 rad/position, which is the fastest.
- At i → d/2 the frequency approaches 1e-4, which is the slowest.

**Problem**

The slide, the diagram and the instructor notes all invert which dimensions carry fine versus coarse position information.

**Fix**

Swap the labels:

- Low dimensions (small i) oscillate rapidly and capture fine, local position.
- High dimensions oscillate slowly and capture coarse, global position.

Fix the SVG aria-label, the SVG annotations, the deck speaker note, and `speaker-notes.md:680`.

### [BLOCKING] Attention-weight hook in the Lesson 5.4 worked example crashes

**File**
- `lessons/04/textbook.html:455-466`
- `lessons/04/textbook.html:526-532` (exercise A depends on it)
- `lessons/04/notes.html:402-405`

**Evidence**

The hook appends `out[1].cpu()` from `model.encoder.layers[0].self_attn`. A quick check confirms that `nn.TransformerEncoderLayer(batch_first=True, norm_first=True)` calls `self_attn(..., need_weights=False)`, so `out[1]` is `None` in both train and eval mode.

**Problem**

`None.cpu()` raises `AttributeError`. The prose ("PyTorch's built-in attention returns the attention matrix if you hook it correctly") is misleading.

**Fix**

Call `layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)` manually on the normed input. Alternatively, use the page's own `scaled_dot_product_attention`, which returns `attn`.

### [BLOCKING] `TrainingArguments(evaluation_strategy=...)` no longer exists

**File**: `lessons/07/textbook.html:412`

**Evidence**: In transformers 4.57.6, `'evaluation_strategy' in signature(TrainingArguments.__init__).parameters` is False; the parameter is now `eval_strategy`.

**Problem**: The BERT fine-tuning snippet raises `TypeError`.

**Fix**: Use `eval_strategy="epoch"`.

### [BLOCKING] Local scaffolds in ex_1–ex_5 contain the solution a second time after REFLECTION

**File**

29 files under `local/`:
- ex_1/01–10
- ex_2/01–04
- ex_3/01–05
- ex_4/01–05
- ex_5/01–02

Only ex_1/11, ex_0, ex_5/03 and ex_6–ex_8 are clean.

**Evidence**

Each solution has 1 REFLECTION block. Each of these scaffolds has 2 or more: `local/ex_4/01_self_attention_from_scratch.py` has REFLECTION at L334, 388, 442 and 496. A difflib comparison shows that the tail after the first REFLECTION matches the solution almost line for line:

| File | Tail starts at | Tail lines | Lines matching the solution |
| --- | --- | --- | --- |
| `local/ex_1/01_standard_ae.py` | L220 | 186 | 185 |
| `local/ex_2/02_resnet_se.py` | L782 | 597 | 596 |
| `local/ex_3/03_gru.py` | L455 | 537 | 532 |
| `local/ex_4/04_bert_finetuning.py` | L457 | 364 | 363 |

Blanks in the scaffold are answered verbatim in its own tail:

- `local/ex_1/01`: the TASK 3 blank at L118–120 is answered by `show_reconstruction(standard_model, X_test_flat, ...)` at L245–303.
- `local/ex_3/01_vanilla_rnn.py`: the TODOs at L201–212 and L274–291 appear complete in the tail.
- `local/ex_5/01_vanilla_gan.py`: the blanks at 376–423 (`z_synthetic = ____`) are answered at 761–763 (`z_synthetic = torch.randn(18000, LATENT_DIM, device=device)`, `X_synthetic = G_gan(z_synthetic)`). A second Phase 4/5/REFLECTION runs 619–885, and `close_engines` is called twice.
- `local/ex_5/02_wgan_gp.py`: the blanks at 515–516 and 637–638 are answered at 1037–1038 and 1157–1158. `_ld` and `_lg` are blank at 387/394 but filled at 912/919.

**Origin:** `git show adc57080:` has one REFLECTION per file. The duplication was introduced by bf8dffe4 ("DL Diagnostics Toolkit — exercise integration"), which appended the diagnostic + Apply + REFLECTION blocks to already-stripped files.

**Problem**

- Students can read the answers below the blanks.
- A completed scaffold trains every model twice and runs every checkpoint and Apply section twice.
- In ex_5, the engines are closed mid-file.
- The diagnostic section students are meant to read comes after the "end" of the exercise.

**Fix**

1. Regenerate every local file from the current solution using the exercise-designer strip. The result should be a single pass: Phases 1–3 → diagnostic → Phases 4–5 → cleanup → one REFLECTION.
2. Add a parity check: exactly one REFLECTION per local file, and no solution-only code line outside a TODO.

### [BLOCKING] Contractive AE scaffold crashes on installed NumPy (`np.trapz` removed)

**File**: `local/ex_1/05_contractive_ae.py:294`

**Evidence**

- The scaffold has `auc = np.trapz(tpr_arr[sorted_idx], fpr_arr[sorted_idx])`.
- On the installed NumPy, `np.__version__` is 2.4.6 and `hasattr(np, 'trapz')` is False.
- The solution was already fixed: `solutions/ex_1/05_contractive_ae.py:432` uses `np.trapezoid(...)  # np.trapz removed in NumPy 2.0+`.

**Problem**: This is pre-filled code, not a blank, so it raises `AttributeError` even after the student completes every TODO correctly.

**Fix**: Use `np.trapezoid`, or regenerate the file as described in the previous finding.

### [BLOCKING] WGAN-GP critic loss is read in the wrong direction

**File**
- `solutions/ex_5/02_wgan_gp.py:275-278, 290, 325-329, 355-360, 409, 781`
- the same text in `local/ex_5/02_wgan_gp.py:295-298, 338, 713, 742, 807`

**Evidence**

- The loss is `loss_d = D(fake).mean() - D(real_batch).mean() + lam * gp` (line 230), i.e. −Ŵ + λ·GP.
- The text says "Lower critic loss = distributions are closer = better generation. The loss should decrease smoothly". Line 356 says "Critic loss ≈ negative Wasserstein distance. It DECREASES as generation quality improves".
- The file's own expected output at line 329 contradicts this: "critic loss ~-2.3 (trend: monotonically toward 0)".

**Problem**

As the generator improves, W shrinks, so the critic loss (≈ −W) *rises* toward 0. A more negative loss means the distributions are further apart. Students are taught to read their main convergence signal backwards.

**Fix**

State: "−(critic loss without GP) estimates W. As quality improves the critic loss rises toward 0." Update the 4C interpretation, line 781 and the expected-output block to match.

### [BLOCKING] Ex 7.5 "InferenceServer" step never serves; raw PyTorch output is printed as InferenceServer predictions

**File**: `solutions/ex_7/05_production_deployment.py:81, 136, 165-177, 218-262` (the local scaffold is the same)

**Evidence**

1. **Name mismatch.** The model is registered as `"production_resnet18_transfer"` (line 136), but the server looks up `InferenceServer.from_registry("cifar10_transfer", registry=registry)` (line 220). That raises; the `except` prints "InferenceServer demo skipped" and sets `server = None`.
2. **Mislabelled output.** Lines 228–240 then run `serving_model(sample_x)` (plain torch) under the header `"=== InferenceServer Predictions ==="`.
3. **Runtime mismatch.** The registered artifact is `pickle.dumps(state_dict)`, but `from_registry` defaults to `runtime='onnx'`.
4. **Wrong export claim.** Line 81 says OnnxBridge handles export, but the code calls raw `torch.onnx.export` (line 168).
5. **Tautological checkpoint.** Checkpoint 3 is `assert n_correct >= 0`.

**Problem**

This is the "fake integration" pattern. The spec 5.7 outcomes "ONNX export with OnnxBridge" and "InferenceServer serves predictions" are never exercised, and the try/except hides the failure.

**Fix**

1. Export with `OnnxBridge().export(model, "torch", output_path=..., sample_input=...)`.
2. Register the ONNX artifact under the same name the server uses.
3. Serve it with `await InferenceServer.from_registry(..., runtime="onnx")` and print `await server.predict(...)` output.
4. Remove the silent fallback, and make the assert check the prediction count and shape.

---

## MAJOR

### [MAJOR] `diagnose(kind="dl")` prints nothing, yet the EXPECTED OUTPUT blocks claim detailed findings (ex_2–ex_4)

**File**

`diagnose(...)` calls:
- `solutions/ex_2/01_simple_cnn.py:253`
- `solutions/ex_2/02_resnet_se.py:296`
- `solutions/ex_2/03_production_pipeline.py:229`
- `solutions/ex_2/04_hyperparameter_study.py:431`
- `solutions/ex_3/01:181`, `ex_3/02:229`, `ex_3/03:206`, `ex_3/04:210`
- `solutions/ex_4/02:282`, `ex_4/03:163ff`, `ex_4/04:286ff`, `ex_4/05:395/398`

Closing sections that say "km.diagnose: 1 line of code → the same observability":
- `ex_1/11:600`, `ex_2/04:833`, `ex_3/05:~664`, `ex_4/05:731`

The local files have the same issue.

**Evidence**

- In kailash-ml 2.2.2, `diagnose(kind="dl")` is just `return DLDiagnostics(subject, tracker=tracker, data=data)`. It attaches no hooks and prints nothing.
- Calling `.report()` explicitly on a small probe gives `gradient_flow: UNKNOWN 'No gradient tracking enabled - call track_gradients()'`, `dead_neurons: UNKNOWN` and `loss_trend: UNKNOWN`.
- The expected output is invented. `ex_2/02:302-305` reports "min RMS = 6.2e-04 at 'layer3.1.conv2.weight' … Spread across 16 Conv layers", but ResNetSE has no `layer3`/`layer1` modules (its children are `stem`, `block1`, `se1`, `block2`) and has only 5 Conv2d layers.
- `ex_4/02:296` claims "no dead GELU units", but `nn.TransformerEncoderLayer` defaults to ReLU.

**Problem**

Students are asked to interpret a report that never appears, and the claim that one SDK line gives the same observability is false.

**Fix**

- Use the pattern ex_1 already uses: `run_diagnostic_checkpoint(model, loader, loss_fn, title=..., train_losses=..., val_losses=..., show=False)`, with a cross-entropy `loss_fn` for classifiers.
- Regenerate every EXPECTED OUTPUT block from a real run.

### [MAJOR] Grad-CAM in ex_2/01 always fails because it uses an undefined variable

**File**: `solutions/ex_2/01_simple_cnn.py:244-279` (the same code is at `local/ex_2/01_simple_cnn.py:647`)

**Evidence**

- L253 is `report = diagnose(...)`.
- L265 calls `cam = diag.grad_cam(...)`, but `diag` is never defined (`grep -n '\bdiag\b'` finds only L265).
- L277 is `except Exception as _exc: print(f"  Grad-CAM skipped ({_exc})")`.
- The L244 comment says "we use `diagnose_classifier`", which the code does not do.
- `diagnostic-reference/ex_2/01_simple_cnn_report.txt:46` shows a different skip error (a device mismatch) from an older version.

**Problem**

The "sixth instrument" demonstration never runs, and the failure is swallowed.

**Fix**

- Call `report.grad_cam(input_tensor, target_class, layer_name)` with the input tensor on the model's device.
- Fix the comment.
- Remove the blanket `except`.

### [MAJOR] Exercise ONNX export never goes through OnnxBridge (`framework="pytorch"` is unsupported)

**File**
- `solutions/ex_2/03_production_pipeline.py:317-350`
- `solutions/ex_4/05_three_way_comparison.py:633-644`
- `local/ex_2/03_production_pipeline.py:241` (hint)

**Evidence**

- The exercises call `bridge.export(model=..., framework="pytorch", output_path=..., n_features=...)`. On a small conv net this returns `success=False, onnx_status='skipped', error 'Export not implemented for framework: pytorch'`.
- With `framework="torch", sample_input=...` the same export returns `success=True`.
- The ex_2/03 L321 comment justifies the fallback with "optimised for tabular models".
- ex_4/05:643 swallows the failure with `except Exception: pass`.

**Problem**

Spec 5.2 asks students to export to ONNX with OnnxBridge. In practice every run silently falls back to raw `torch.onnx.export`, and students learn a wrong reason for the fallback.

**Fix**

- Use `framework="torch"` with `sample_input=torch.randn(1,3,32,32)`.
- Delete the "tabular" comment and the bare `except`.

### [MAJOR] The ex_2/03 InferenceServer section never awaits its call and never serves a prediction

**File**
- `solutions/ex_2/03_production_pipeline.py:105-108, 425-450`
- `local/ex_2/03_production_pipeline.py:346-355`

**Evidence**

- `inspect.iscoroutinefunction(InferenceServer.from_registry)` is True. L442 nonetheless calls `server = InferenceServer.from_registry(...)` without `await`, then prints "bound to resnet_se_cifar10".
- `server.predict` is never called.
- The model is registered as a pickled `state_dict` (`shared/mlfp05/ex_2.py:349`), while `from_registry` defaults to `runtime='onnx'`.
- The theory (L105-108) mentions `predict_batch()`, and the local hint mentions `InferenceServer(registry=..., cache_size=5)` / `warm_cache`. None of these exist.

**Problem**

The section prints a false success message, raises an un-awaited-coroutine warning, teaches removed APIs, and never actually serves.

**Fix**

- Register the ONNX artifact.
- Call `server = await InferenceServer.from_registry(...)`, then `await server.predict({...})`.
- Rewrite the theory paragraph and the hint.

### [MAJOR] "Contractive" autoencoders apply plain L2 weight decay, not a Jacobian penalty

**File**
- `solutions/ex_1/05_contractive_ae.py:110-120, 139-145`
- `solutions/ex_1/10_contractive_vae.py`
- `solutions/ex_1/11_grand_comparison.py:370-376, 397-401`
- `local/ex_1/11_grand_comparison.py:253` (hint: "Frobenius norm of encoder weights")

**Evidence**

The penalty is `sum(torch.sum(p**2) for p in [enc1.weight, enc2.weight, enc3.weight])`, with the docstring "Frobenius norm of encoder weights (Jacobian approximation)". The L141 comment says "Contractive AEs penalise ‖∂z/∂x‖_F".

**Problem**

The squared-weight sum does not depend on the input, so it is ordinary L2 weight decay. The contractive penalty (Rifai et al. 2011) is the per-sample ‖J_f(x)‖²_F, which depends on which ReLUs are active. Spec 5.1's "penalty on Jacobian" is therefore not implemented, even though the materials say it is.

**Fix**

- Compute the real penalty, e.g. `torch.func.vmap(torch.func.jacrev(model.encoder))(xb).pow(2).sum((1,2)).mean()`. With LATENT_DIM=16 this is affordable.
- Or keep the current term but rename it "weight decay".

### [MAJOR] ex_2/02 diagnostic interpretation gives a backwards overfitting prescription and invented facts

**File**: `solutions/ex_2/02_resnet_se.py:309-349`

**Evidence**

- L342-346 say: "Train-val gap 6% means augmentation (flip + crop) is delivering regularisation … If gap >15%, reduce augmentation strength". But `load_cifar10` uses only `ToTensor()`, so there is no augmentation in the exercise.
- L330-334 claim SE can promote dead channels. SE multiplies each channel by a sigmoid weight in (0,1), so a zero ReLU output stays zero.
- L337-338 advise "Change reduction from 16 to 8", but the file already uses `reduction=8` (L184).

**Problem**

The fix it teaches for overfitting is the opposite of the correct one: a large train-val gap calls for *more* regularisation or augmentation.

**Fix**

Rewrite the block from a real run. Give the correct prescription (gap >15% → add augmentation, weight decay or dropout, or stop earlier), and drop the SE-revival claim and the 16→8 advice.

### [MAJOR] ex_4/02 attention heatmaps come from an untrained layer

**File**: `solutions/ex_4/02_transformer_encoder.py:380-397, 477`

**Evidence**

- L381 builds `mha_viz = EducationalMultiHead(d_model=128, n_heads=4)`, a fresh layer with random Q/K/V weights, and runs it on the trained embeddings.
- The trained model itself uses `nn.TransformerEncoderLayer` (L222-232).
- L393-397 then describe head specialisation ("Head 3: entity-to-entity").

**Problem**

The module's main visual proof for multi-head attention is random projections.

**Fix**

Extract attention from the trained `encoder.layers[i].self_attn(x, x, x, need_weights=True, average_attn_weights=False)`, or build the classifier on `EducationalMultiHead`.

### [MAJOR] The ex_4/04 "sentiment analysis" application runs a 4-class news-topic classifier

**File**: `solutions/ex_4/04_bert_finetuning.py:541-600`; also `solutions/ex_4/03_lstm_baseline.py:283-345`

**Evidence**

- The task is headed "Sentiment Analysis for DBS Bank Customer Reviews … (positive/negative/neutral)", but it runs the AG News model (World/Sports/Business/Sci-Tech) on the reviews.
- L596-598 conclude that "high confidence indicates … strong signal … even on domain-shifted text".
- ex_4/03 does the same with airline reviews.

**Problem**

A topic head cannot output sentiment. High softmax confidence on out-of-distribution text is not evidence of signal, so students learn the wrong way to read model outputs.

**Fix**

Either fine-tune a sentiment head on a public review dataset, or reframe the application as topic routing with an explicit out-of-distribution warning.

### [MAJOR] Spec 5.2 training enhancements are missing from ex_2 (Mixup, label smoothing, Kaiming init)

**File**: `solutions/ex_2/*.py`, `shared/mlfp05/ex_2.py`

**Evidence**

- `grep -rniE 'label.?smooth|kaiming'` over these files returns nothing.
- "mixup" appears only in a comment (`ex_2/02:348`).
- Spec 5.2's learning objective is "Apply modern training enhancements (SE blocks, mixed precision, Mixup)". SE and mixed precision are present; Mixup is not.

**Fix**

Add a Mixup + label-smoothing + Kaiming-init ablation to `02_resnet_se` or `04_hyperparameter_study`, tracked with ExperimentTracker.

### [MAJOR] Spec 5.3 content is missing from ex_3 (technical indicators, character-level text generation, perplexity)

**File**: `solutions/ex_3/*.py`, `shared/mlfp05/ex_3.py:52`

**Evidence**

- `grep -niE 'rsi\b|macd|bollinger|technical indicator'` returns nothing.
- The features are `FEATURES = ["Close", "High", "Low", "Volume"]`.
- There is no character-level LSTM or text generation anywhere, and no perplexity calculation.

**Problem**

The learning outcome "Train RNNs for … text generation" is not exercised, and the spec's exercise (stock prediction with technical indicators, plus character-level text generation) is only half implemented.

**Fix**

- Add RSI/MACD/Bollinger features to `build_dataset`.
- Add a character-level LSTM file that reports perplexity = exp(mean CE).


### [MAJOR] Exercise descriptions in the deck, speaker notes, README and index do not match the exercises that exist

**File**
- `deck.html` exercise slides at 2135-2165, 2828-2855, 3451-3480, 4268-4294, 4821-4847, 5255-5281, 5680-5708, 6467-6497, and the "Real-World Applications" intros at 2234, 4884, 6599
- `speaker-notes.md:135, 239-241, 401, 562-563, 577, 743, 1003, 1126, 1346`
- `README.md:20-31`
- `index.html` (lesson summaries and outcomes)

**Evidence**

What the materials say, against what the exercise code actually does (`shared/mlfp05/ex_N.py` loaders and `solutions/ex_N/*`):

| Lesson | Materials say | Exercise actually does |
| --- | --- | --- |
| 5.1 | "vanilla autoencoder on MNIST (latent dim=32)", "Generate new digits" (deck 2141-2144) | Fashion-MNIST (`shared/mlfp05/ex_1.py:54`), `LATENT_DIM = 16` (line 50); 10 variants + a comparison. The deck's own note at 2234 says "you just built on Fashion-MNIST". |
| 5.2 | "simple CNN for Fashion-MNIST", "Apply Mixup augmentation and label smoothing" (deck 2834-2837); snippet `input_shape=(1,1,28,28)`, `"resnet_fmnist.onnx"` | CIFAR-10 (`shared/mlfp05/ex_2.py:78`); deck 2928 itself says "The same CIFAR-10 pipeline you just built". There is no Mixup or label smoothing anywhere in `solutions/ex_2` (only a comment at 02_resnet_se.py:348). |
| 5.3 | "character-level text generation with LSTM" (deck 3460, speaker notes 562-563) | No char-LSTM / Shakespeare code in any `solutions/ex_3` file. |
| 5.4 | "Fine-tune BERT … (TREC-6 dataset)" (deck 4276; speaker notes 743) | AG News; deck 4367 says "AG News in Exercise 4". |
| 5.5 | "Implement DCGAN for MNIST" (deck 4827); "DCGAN, WGAN-GP, and diffusion — the three generators you built in Exercise 5" (4884) | MLP GANs (no convolution); deck 4727 says students do NOT implement diffusion. |
| 5.6 | "GCN for graph classification on TUDataset … with torch_geometric" (deck 5261, 5270; speaker notes 1003) | Dense node classification on Cora. |
| 5.7 | "Fine-tune ResNet for mask detection", "Fine-tune BERT" (deck 5686-5687; speaker notes 1126) | CIFAR-10 ResNet-18 only; no mask dataset in `data/mlfp05/`; BERT lives in ex_4. |
| 5.8 | "The PPO you wrote for CartPole" (deck 6599) vs "PPO for supply chain optimisation (custom environment)" (deck 6474) | PPO runs on CartPole + `RideHailingPricingEnv`. |

**README.md:24-31**

| Exercise | README dataset | Actual dataset |
| --- | --- | --- |
| Ex 1 | "Synthetic / MNIST-like" | Fashion-MNIST |
| Ex 5 | "2D synthetic data" | MNIST |
| Ex 6 | "Karate Club / synthetic" | Cora (Karate is only the fallback) |
| Ex 7 | "Transfer Learning with Transformers — Singapore text" | CIFAR-10 ResNet |
| Ex 8 | "Inventory management" | partly correct |

The README also omits ex_0.

**index.html**
- "REINFORCE on CartPole" and "Implement a policy-gradient RL agent … relate it to PPO and DQN". Nothing in ex_8 implements REINFORCE; the exercises implement DQN and PPO directly.
- "node and graph classification on an SBM benchmark": it is Cora.
- 5.7 "BERT fine-tune": not in ex_7.
- 5.3 "LSTM (six gate equations)", while the deck slide says "All Five Equations".

**Problem**

Instructors brief, and students read, task lists, datasets and assessment criteria for exercises that do not exist.

**Fix**

Rewrite each "Exercise 5.x" slide, speaker-note block, README row and index summary from the actual `solutions/ex_N` contents. Where the spec requires the missing task (Mixup, char-LSTM, DCGAN, TUDataset graph classification), add it to the exercise rather than to the slide (see the coverage findings).

### [MAJOR] Spec coverage gaps in the taught material (deck + textbooks)

**File**: `deck.html`, `textbook.md`, `lessons/0N/textbook.html` vs `specs/module-5.md`

**Evidence**

**5.3**
- `grep -i -w "spatial attention|multi-layer|stacked LSTM|GIN|isomorphism|longformer" deck.html` returns nothing.
- Spec 5.3 requires "Multi-layer with residual connections" and "Spatial attention: feature relationships via multi-headed attention". Neither is in the deck, and spatial attention is in neither textbook.
- Technical indicators are absent from the textbooks.

**5.4**
- The decoder (masked self-attention + cross-attention) is in the deck (4077-4086, 4135-4207) but in neither textbook, although `textbook.md:763` promises it.
- ViT is not covered in the 5.4 textbook, yet it is spec objective 5.4.5. The textbook cross-references are circular: `textbook.md:572` and `speaker-notes.md:361` say 5.4 covers it "fully"; `lessons/04/textbook.html:563` says it was "touched on in 5.2".
- The 5.4 lesson page defers BERT fine-tuning to 5.7 (`lessons/04/notes.html:380, 458-459`), while the spec and `solutions/ex_4/04_bert_finetuning.py` place it in 5.4.

**5.5**: CycleGAN, StyleGAN, Inception Score and the synthetic-data applications are absent from both textbooks. cGAN appears only as a drill.

**5.6**: GIN is absent from the deck and both textbooks, although the spec lists four GNN architectures. GraphSAGE is absent from `textbook.md`.

**5.7**: Adapter modules get only a cross-reference to 6.2. The HuggingFace Pipeline API is not shown.

**5.8**: DDPG, SAC and A2C get a one-line table row in the textbooks. PPO is formula-only, while the spec objective is "Implement PPO for a continuous action problem".

**Problem**

Required techniques and learning outcomes are not taught, or are materially thinner than the spec.

**Fix**

Add the missing sections: a stacked/residual LSTM plus multi-head spatial attention (5.3); decoder/cross-attention and ViT in the 5.4 textbook; CycleGAN/StyleGAN/IS (5.5); GIN (5.6); adapters (5.7); DDPG/SAC/A2C paragraphs (5.8). Make the ViT and BERT lesson placement consistent across all artifacts.

### [MAJOR] Exercise-level spec gaps for Lesson 5.5: no DCGAN (MLP only); the "FID + Inception Score" run computes no IS

**File**
- `shared/mlfp05/ex_5.py:137-185`
- `solutions/ex_5/03_gan_evaluation.py:83-85, 537, 900-906`

**Evidence**

- `Generator` and `Discriminator` are `nn.Linear` MLPs; there is no Conv or ConvTranspose layer anywhere in ex_5.
- The MLP GAN is registered as `"dcgan_generator"` (line 537).
- The diagnostic title promises "FID + Inception Score", but IS is never computed.
- The "NOVELTY: nearest-neighbour distance" metric (83-85) is promised but never implemented.

**Problem**

Spec 5.5 says "Implement DCGAN … DCGAN generates recognisable images". That outcome is not exercised, and the labels misdescribe what actually runs.

**Fix**

- Add a convolutional DCGAN technique file, or rename the MLP GAN honestly.
- Either compute IS and novelty, or remove those claims.

### [MAJOR] Exercise-level spec gap for Lesson 5.6: no GIN, no graph classification, no TUDataset, no torch_geometric layers

**File**: `solutions/ex_6/*.py`

**Evidence**

Grepping for `GIN|Isomorphism|TUDataset|graph classification|global_mean_pool|GCNConv|GATConv|SAGEConv` across `solutions/ex_6` finds nothing relevant. All five files do dense node classification on Cora; `torch_geometric` is used only for `Planetoid`.

**Problem**

The spec requires:
- "Graph classification on TUDataset using GCN"
- GIN
- the learning objective "Use torch_geometric for graph ML"

None of these is exercised.

**Fix**

Add a technique file that does TUDataset (MUTAG/PROTEINS) graph classification with `GCNConv`/`GINConv` + `global_mean_pool` and a PyG `DataLoader`.

### [MAJOR] Exercise-level spec gap for Lesson 5.8: DDPG, SAC and A2C are absent; PPO is never applied to continuous actions; SAC results are fabricated

**File**: `solutions/ex_8/02_ppo.py`; `solutions/ex_8/04_algorithm_comparison.py:734-769`

**Evidence**

- Grepping ex_8 for `ddpg|sac|a2c` finds only the diagnostic title "(DQN / PPO / SAC)" and an "expected output" containing "SAC: reward 298 ± 35", for an algorithm the file never runs.
- PPO is trained only on discrete CartPole and on a Discrete pricing env.

**Problem**

- The spec says "5 algorithms, 5 business applications" and has the objective "Implement PPO for a continuous action problem". Neither is met.
- The expected output invents results.

**Fix**

1. Add a Box-action environment for PPO.
2. Cover DDPG, SAC and A2C, at least via `km.rl_train` once the RL extra is installed.
3. Delete the fabricated SAC rows.

### [MAJOR] JS divergence misdescribed (the WGAN motivation is stated incorrectly)

**File**
- `deck.html:4603` (speaker note): "unlike JS divergence which is zero when distributions don't overlap"
- `lessons/05/textbook.html:128-130`: "Unlike KL, JS is always finite, which is (one) reason GANs can be trained on supports that don't overlap."

**Evidence**

For disjoint supports, JS(P‖Q) = log 2. This is the maximum, and it is constant, so its gradient with respect to the generator parameters is zero. That is precisely the failure the WGAN paper fixes. The same lesson page's own WGAN section (302-306) and `speaker-notes.md:813, 840` contradict both statements.

**Problem**

The deck says JS is "zero" (wrong value). The textbook says finiteness *enables* training on disjoint supports (wrong conclusion).

**Fix**

Use: "When the supports don't overlap, JS saturates at the constant log 2, so G receives no gradient. Wasserstein distance still varies smoothly with how far apart the distributions are."

### [MAJOR] Non-saturating GAN loss is used, but students are told BCE "saturates / gives zero gradient"

**File**
- `solutions/ex_5/01_vanilla_gan.py:141` vs `274, 288, 590-591`
- `solutions/ex_5/02_wgan_gp.py:286, 770`
- `lessons/05/slides.html:264` (comment says "minimise log(1-D(G(z)))" above code computing `-log D(G(z))`)

**Evidence**

- Line 141 says `L_G = -E[log D(G(z))] (generator loss — non-saturating)`, and the code is `loss_g = bce(D(G(z)), ones)`.
- Lines 590-591 say "The BCE loss gives zero gradient when D perfectly separates real from fake".

**Problem**

With the non-saturating loss, the gradient with respect to D's logit tends to −1 when D rejects fakes, so it does not vanish. Only the original minimax `log(1−D(G(z)))` saturates. The WGAN motivation contradicts the code students just ran.

**Fix**

Explain that the minimax G loss saturates; the non-saturating variant used here avoids that, but training stays unstable because JS is constant on disjoint supports. Fix the slide comment to `# Train G (non-saturating): minimise -log D(G(z))`.

### [MAJOR] WGAN-GP gradient penalty on the lesson slide computes the norm over channels only

**File**: `lessons/05/slides.html:278-286`

**Evidence**

The slide code is:

```
alpha = torch.rand(batch_size, 1, 1, 1)
interp = alpha*real + ...
gradients = autograd.grad(...)[0]
gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
```

- For image tensors `(B, C, H, W)`, `norm(dim=1)` gives a `(B, H, W)` per-pixel channel norm, so the penalty forces every pixel's channel-gradient norm toward 1. That is not the 1-Lipschitz constraint on the whole input.
- The course's own solution does it correctly: `grad.reshape(batch, -1).norm(2, dim=1)` (`solutions/ex_5/02_wgan_gp.py:147`), as does `textbook.md:1030`.

**Problem**

Copy-pasting the slide yields a silently wrong penalty.

**Fix**

`gp = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()`.

### [MAJOR] Generative models presented as privacy-preserving, with FID as "audit evidence"

**File**
- `deck.html:2223` ("VAEs used for synthetic healthcare data, where sampling from the latent prior generates privacy-preserving patient records")
- `deck.html:4890` ("SGH cannot share patient CT scans … but it can share a diffusion model trained on them … synthetic scans … containing no actual patient. This is how rare-disease classifiers get enough training data without violating PDPA or HIPAA")
- `deck.html:4909, 4915` ("FID (Exercise 5) is the audit evidence"; "the FID ↔ regulator connection")
- `lessons/05/slides.html:293` ("privacy-safe")
- `lessons/01/slides.html:52` ("Generation — synthetic patient records (VAE)")

**Evidence**

- Diffusion models and GANs can memorise and regenerate training examples; extraction attacks on diffusion models are well documented (Carlini et al., 2023, "Extracting Training Data from Diffusion Models").
- FID measures distributional similarity of feature statistics. It does not measure memorisation or privacy leakage, and a model that copies training images gets an *excellent* FID.

**Problem**

The lesson tells decision-makers in regulated industries that sharing a generative model trained on patient data is privacy-compliant, and that FID proves it. Both claims are false and could lead to real compliance failures.

**Fix**

State that synthetic data is not private by default. It requires formal guarantees (e.g. DP-SGD training) and memorisation/membership-inference testing. FID measures fidelity and diversity, not privacy. Remove "privacy-safe" and the FID↔regulator framing.

### [MAJOR] Named real organisations are credited with specific ML deployments and invented outcome figures

**File**

Deck "Real-World Applications" slides:

| Location | Claim |
| --- | --- |
| `deck.html:2208` | "Credit Card Fraud (DBS, UOB, OCBC)" |
| `deck.html:2212` | "Medical Imaging Anomalies (SGH, NUH)" |
| `deck.html:2222` | "Network Intrusion Detection (Singtel, StarHub)" |
| `deck.html:2937` | "Shelf Monitoring at FairPrice, Cold Storage" |
| `deck.html:2944` | "SE blocks improved recall on a rare defect class from 71% to 89%" |
| `deck.html:2948` | "SMRT platforms use CNN-based crowd counting" |
| `deck.html:3526` | "Quant desks at DBS and UOB run LSTM+attention models on minute-bar SGX data with the exact feature set from Exercise 3" |
| `deck.html:3530` | "SGH's ICU deploys LSTMs …" |
| `deck.html:3540` | "LTA forecasts MRT ridership …" |
| `deck.html:4377` | "saving 15 minutes per patient across 2M admissions a year" |
| `deck.html:4383` | "IRAS Tax Query Routing … Accuracy on the first 20 classes exceeds the human baseline" |
| `deck.html:4890` | SGH |
| `deck.html:4894` | "Shopee and Lazada use CycleGAN-style networks" |
| `deck.html:4900` | "95%+ of the supervised ceiling" |
| `deck.html:5324` | "A*STAR's Bioinformatics Institute uses GNN-based virtual screening …" |
| `deck.html:5328` | "AML teams at DBS and OCBC run GNN classifiers" |
| `deck.html:5334` | "Singtel and StarHub use GNNs to predict which towers will congest" |
| `deck.html:5338` | PSA |
| `deck.html:5766` | Shopee/Lazada |
| `deck.html:6577` | Singtel |
| `deck.html:6581` | "Lazada / Shopee price 100M SKUs in real time. SAC …" |

Speaker note `deck.html:6610`: "the real state/action/reward structure of the production systems at the companies listed".

Other artifacts:
- `lessons/05/slides.html:294-295`: "at NUS", "for LTA planning models"
- `lessons/06/textbook.html:52-55`, `lessons/06/notes.html:19`: DBS fraud graph, NUS researchers
- `lessons/08/textbook.html:57`: "A DBS dynamic pricing engine"
- Exercises:
  - `solutions/ex_6/01_gcn.py:45, 284-291` (NUS/NTU)
  - `solutions/ex_5/02_wgan_gp.py:536-579, 732, 786` (NUH, MOH)
  - `solutions/ex_6/04_link_prediction.py:449-516` (SGH clinical records)
  - `solutions/ex_6/02_gat.py:375` (DBS/OCBC/UOB)
  - `solutions/ex_7/03_data_efficiency.py:342-351`, `ex_8/02_ppo.py:479`, `ex_8/04_algorithm_comparison.py:557-570` (Grab, Gojek, Singtel, Changi, LTA, SP Group, FairPrice)
  - `solutions/ex_8/04_algorithm_comparison.py:761-762` ("robotics at OpenAI/Anthropic")
  - `solutions/ex_4/01:203-222` (Rajah & Tann, "S$300K-800K")
  - `solutions/ex_2/02:35` and `solutions/ex_1/04:237-246` (GlobalFoundries)
  - `solutions/ex_1/05:253-270` (SGH)
  - `solutions/ex_4/04:543-563` (DBS "S$739B", "McKinsey, 2023")
  - `solutions/ex_4/03:283-287` (Singapore Airlines, Skytrax)
  - `solutions/ex_1/03` and `solutions/ex_3/03` (SMRT)
  - `solutions/ex_3/01` (Ya Kun)
  - `solutions/ex_3/04:84` ("Transformers (GPT, BERT, Claude)", which names a commercial product)

**Evidence**

None of these claims is sourced. Several are specific and checkable as fact: named units, figures, "exact feature set from Exercise 3".

**Problem**

These statements assert, as fact, the internal systems and outcome figures of named banks, hospitals, government agencies, universities and companies. They are unverified and in places invented, which exposes the Foundation to misrepresentation risk. `independence.md` and CLAUDE.md Directive 6 also bar institutional and university references, and the R9B "named industry" requirement does not need named organisations. Context: these are NOT partner or funding-body references, and no PCML or prior-course codes were found anywhere in the module.

**Fix**

Replace each with a generic actor ("a Singapore retail bank", "a public hospital", "a telco", "a port operator", "an e-commerce marketplace"). Mark scenario figures as illustrative ("e.g., a hypothetical lift from 71% to 89%"), or delete them.

### [MAJOR] Assessment starters contain the full reference solution as comments

**File**
- `assessment/task_1/starter.py:67-97`
- `assessment/task_2/starter.py` (TODO 1/3/4 blocks)
- `assessment/task_3/starter.py:72-97`
- `assessment/task_4/starter.py:83-124`

**Evidence**

`diff solution.py starter.py` shows that every removed solution line reappears verbatim as a comment directly under the TODO. Examples:

- Task 4: `# self.embed = nn.Embedding(vocab, dim, padding_idx=0)` … `# h = self.encoder(h, src_key_padding_mask=pad_mask)` … `# pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)`, plus the exact `TransformerEncoderLayer(dim, heads, dim_feedforward=dim*2, dropout=0.1, batch_first=True), num_layers=2, enable_nested_tensor=False` recipe.
- Task 2: the exact layer stack, including `Linear(32*2*2 -> 64)`.
- All four tasks: the exact epochs, learning rate and batch size.

**Problem**

The problem.md files label all four tasks "Difficulty: Hard", 3-hour exam conditions. Completing them only requires uncommenting lines, so the assessment measures nothing beyond reading. The same pattern appears in exercise scaffolds; see the ex_5–ex_8 hint finding below.

**Fix**

Reduce the hints to the contract plus a concept-level hint (e.g. "an undercomplete encoder/decoder; train only on healthy rows"). Remove the literal layer stacks and training-loop code.

### [MAJOR] `diagnostic-reference/` presents crashed or skipped runs as the "healthy reference"

**File**
- `diagnostic-reference/README.md:30-44, 80-86, 99-104`
- `diagnostic-reference/ex_3/02_lstm_report.txt:60-69`
- `diagnostic-reference/ex_3/03_gru_report.txt:98, 104`
- `diagnostic-reference/ex_6/01_gcn_report.txt:144` (and 02, 03)
- `diagnostic-reference/ex_2/01_simple_cnn_report.txt:46`

**Evidence**

- `02_lstm_report.txt` ends with `[RUNNER ERROR] Traceback … AssertionError: LSTM should preserve gradients better than RNN`.
- `03_gru_report.txt` has `ModelRegistry registration skipped (AttributeError: 'ModelRegistry' object has no attribute 'register')` and a `[RUNNER ERROR] Traceback`.
- The ex_6 reports print `[diagnostic skipped: name 'features' is not defined]` instead of a Prescription Pad.
- ex_2 has `Grad-CAM skipped (Input type (MPSFloatType) and weight type (torch.FloatTensor) should be the same)`.

The README misdescribes all of this:
- It labels LSTM/GRU as "Timed out during training", when they actually crashed on an assert.
- It tells students to compare their `diag.report()` against these files.
- It says "GAT (ex_6/02): you should see attention head entropy low enough to trigger the warning". The GAT report contains no diagnostic at all.
- It points to `/tmp/save_diag_outputs.py` ("lives in the source repo"), which is not in the repo.

**Problem**

The student-facing reference outputs are stale error logs, and the README's description of them is false.

**Fix**

Regenerate the references after fixing the exercise diagnostics (see the undefined-names finding). Describe partial captures accurately, and commit the capture script under `scripts/` or remove the reference to it.

### [MAJOR] `speaker-notes.md` is stale against the 127-slide master deck

**File**: `speaker-notes.md` (whole file; e.g. 84, 1453)

**Evidence**

The deck has 127 `<section>` slides, and the speaker notes cover 96. Notes are missing for:

- deck slides 6–25 (the DL Diagnostics toolkit and the appendix)
- all eight "Lesson 5.x: Real-World Applications" slides
- "SE Block: Data Flow", "Encoder-Decoder: The Full Picture" and "GCN: Message Passing in One Picture"

The numbering is also offset:
- The notes' "Slide 6: Lesson 5.1" is deck slide 26.
- Line 1453's must-not-cut list ("13 (reparameterisation), 22 (ResNet), 43-44, 87 (PPO)") maps to deck slides 33, 43, 67-68 and 117. Deck slide 13 is "Instrument 2: Gradient Flow".

**Problem**

Instructors following the notes are misaligned by 20+ slides and have no notes for 31 slides.

**Fix**

Renumber against `deck.html` and add the missing notes. Also remove the session-log leak at line 6 ("rewritten this session").

### [MAJOR] Factual errors in the PPO→RLHF bridge (an assessed outcome)

**File**
- `solutions/ex_8/04_algorithm_comparison.py:573, 703, 761-762`
- `lessons/08/textbook.html:710`
- `textbook.md:1456`
- `deck.html:6599` ("action = next token" is correct here, but "Exercise 8 is literally the modern LLM alignment loop" overclaims)

**Evidence**

- ex_8/04:573 lists the LLM action space as "Continuous (token probs)".
- Line 703 says "Clipping prevents catastrophic forgetting of language ability".
- `lessons/08/textbook.html:710` maps "PPO clipping ↔ KL penalty keeping LLM close to original".

**Problem**

- In RLHF the action is a discrete vocabulary token.
- Staying close to the reference model is enforced by a separate KL penalty to the SFT policy. PPO clipping only bounds each update relative to π_old.

This is the spec's "PPO→RLHF connection articulated" assessment criterion, and it is taught incorrectly.

**Fix**

- Set the action space to "Discrete (vocabulary tokens)".
- Present the KL-to-reference penalty and PPO clipping as two separate mechanisms everywhere.
- Remove the "robotics at OpenAI/Anthropic" attribution.

### [MAJOR] ViT described as "decoder-only" in the speaker notes

**File**: `speaker-notes.md:712`

**Evidence**: The notes say "ViT: decoder-only applied to image patches." The deck (2732-2734) and `lessons/04/textbook.html:381` correctly say it is an encoder.

**Fix**: Change to "encoder-only".

### [MAJOR] Textbook "DCGAN" worked example is an MLP and trains on mismatched pixel ranges

**File**
- `textbook.md:1051-1105`
- `lessons/05/textbook.html:42, 157-188`

**Evidence**

- The worked example is titled "DCGAN", but its Generator and Discriminator are `nn.Linear` stacks.
- The Generator ends in `nn.Tanh()`, which outputs [-1, 1]. The real data comes from `transforms.ToTensor()` only (`textbook.md:245`), which gives [0, 1].
- The Discriminator ends in `nn.Sigmoid()` and is then reused as a WGAN critic in drill 2.

**Problem**

- DCGAN is by definition convolutional.
- The range mismatch lets D separate real from fake by value range alone.
- A WGAN critic must not have a sigmoid; the same page says so at lines 512-517.

**Fix**

- Make the generator ConvTranspose2d and the discriminator a strided-conv network.
- Add `Normalize((0.5,), (0.5,))` to the real-data transform.
- Remove the sigmoid from the critic.

### [MAJOR] DQN worked example never explores one of its actions

**File**: `lessons/08/textbook.html:341, 646, 688-690`

**Evidence**: `select_action` returns `random.randint(0, 1)` when exploring. The same agent is then built with `action_dim=3`.

**Problem**: Action 2 ("personal call") is never explored, so the claimed result ("personal calls for high-value customers") cannot be reached.

**Fix**: Use `random.randrange(self.action_dim)`.

### [MAJOR] 14 of the 17 diagnostic checkpoints in ex_5–ex_8 pass undefined names and always print "[diagnostic skipped]"

**File**: `run_diagnostic_checkpoint(...)` calls in these solutions (the local mirrors are the same; local ex_7/02 and ex_7/04 also fail):

- `solutions/ex_5/03:902`
- `solutions/ex_6/01:435`, `ex_6/02:524`, `ex_6/03:537`, `ex_6/04:608`, `ex_6/05:619`
- `solutions/ex_7/01:379`, `ex_7/03:492`, `ex_7/05:607`
- `solutions/ex_8/01:705`, `ex_8/02:832`, `ex_8/03:1030`, `ex_8/04:736`

**Evidence**

An AST check compared each call's first two arguments against every name assigned in its module. These names are undefined:

| File | Undefined names |
| --- | --- |
| ex_5/03 | `generator`, `noise_loader` |
| ex_6/01, 02, 05 | `features`, `labels` (05 also `best_model`) |
| ex_6/03 | `sampled_loader` |
| ex_6/04 | `edge_loader` |
| ex_7/01 | `model` |
| ex_7/03 | `models_by_frac` |
| ex_7/05 | `calibration_loader` |
| ex_8/01 | `q_network`, `replay_loader` |
| ex_8/02 | `actor_critic`, `rollout_loader` |
| ex_8/03 | `agent`, `rollout_loader` |
| ex_8/04 | `best_agent`, `rollout_loader` |

Other symptoms:
- Each call is wrapped in `except Exception as exc: print(f"[diagnostic skipped: {exc}]")`. `diagnostic-reference/ex_6/01_gcn_report.txt:144` shows exactly this output.
- The `_diag_loss` bodies are unedited templates. ex_8/01 says "TD-error loss" but returns `F.cross_entropy` / `F.mse_loss(out, x)`.
- The ex_7/01 comment says "Fashion-MNIST", but the data is CIFAR-10.

**Problem**

Each file has a long "EXPECTED OUTPUT / interpretation guide" for a report that never appears; ex_8/04 even labels its guide "synthesized reference". The broad `except` hides the bug, which violates zero-tolerance Rule 3.

**Fix**

Pass the real objects, with real per-exercise loss closures. For example, use `gcn` with `[(X, y)]` and a loss over `A_norm` in ex_6, `scratch_model` in ex_7/01, and `dqn_model` with a replay-sampled loader in ex_8/01. Remove the bare `except` so failures surface.

### [MAJOR] GNN exercises select the best epoch by test accuracy and compare it with a final-epoch baseline

**File**
- `solutions/ex_6/01_gcn.py:197, 333-334`
- `solutions/ex_6/02_gat.py:223`
- `solutions/ex_6/03_graphsage.py:256, 429-430`
- `solutions/ex_6/05_architecture_comparison.py:266, 284, 391`

**Evidence**

- `best_test = max(gcn_test)`, where test accuracy is logged every epoch (`shared/mlfp05/ex_6.py:306-311`).
- `improvement = best_test - baseline_acc`, where the baseline is the final-epoch test accuracy.

**Problem**

Picking the epoch by test accuracy is test-set selection. It inflates the reported GNN numbers and the "improvement from graph" figure students are taught to report.

**Fix**

Select the epoch by validation accuracy: `best_test = gcn_test[int(np.argmax(gcn_val))]`. Do the same in 02, 03 and 05.

### [MAJOR] Link prediction is trained and scored on the same edges, which are also in the message-passing graph

**File**: `solutions/ex_6/04_link_prediction.py:205-300`

**Evidence**

- The positive examples are all of `edge_index_np`.
- The encoder uses the full `A_norm`.
- The AUC (260-270) is computed on the same `pos_src`/`neg_src` pairs used for training. There is no held-out edge split.

**Problem**

The reported AUC measures memorisation plus leakage of the target edges into the encoder. It does not measure link prediction, and the knowledge-graph-completion story in the file depends on predicting unseen links.

**Fix**

1. Split edges into train/val/test (85/5/10).
2. Build `A_norm` from the training edges only.
3. Sample fresh negatives.
4. Report AUC on the test edges.

### [MAJOR] The adapter "starts as identity" claim is false: AdaptedBlock adds the pooled features a second time

**File**: `solutions/ex_7/04_adapter_modules.py:69, 132, 140-152, 166-180` (same in local)

**Evidence**

- `BottleneckAdapter.forward` returns `x + self.up(...)`, so it already includes the skip connection.
- `AdaptedBlock.forward` returns `out + adapted[..., None, None]`. At initialisation this equals `out + mean(out)`, not `out`.
- The Checkpoint 1 identity test only checks the bare adapter.
- The theory says "~5–10% of params trainable", but Checkpoint 2's own number is ~104K/11.3M ≈ 0.9%.

**Problem**

The pretrained features are perturbed from step 0, which contradicts the "zero-init ⇒ no change" interpretation. The stated trainable-parameter percentage is also off by about an order of magnitude.

**Fix**

- Add only the residual, e.g. `out + (self.adapter(pooled) - pooled)[..., None, None]`.
- Add an identity checkpoint on the full `AdaptedBlock`.
- Correct the percentage.

### [MAJOR] Exercise scaffold hints paste the full answer for nearly every M5 blank

**File** (examples)
- `local/ex_8/02_ppo.py:211-213`: `____  # TODO: TD error = reward + gamma * next_value * nonterminal - V(s)`, then `ratio = ____  # TODO: torch.exp(new_lp - old_lp_t[mb])` and `policy_loss = ____  # TODO: -torch.min(surr1, surr2).mean()`
- `local/ex_5/01_vanilla_gan.py:156-200`: `# Hint: loss_d = bce(D(real_batch), torch.ones(...)) + bce(D(fake), torch.zeros(...))`
- `local/ex_5/02_wgan_gp.py:513-516`
- `local/ex_6/01_gcn.py:138-141`: `# Hint: return a_norm @ self.W(h)`

**Problem**

`exercise-standards.md` calibrates hints by module: explicit code at M1, partial hints by M3, minimal at M6. At M5 (~30% provided), verbatim answers turn the core-logic tasks (GAN losses, GP, GAE, PPO clip, GCN forward) into copy-paste.

**Fix**

Replace them with concept- or API-level hints, e.g. "ratio of new to old policy probabilities, computed in log space" or "use torch.clamp with clip_eps".

---

## MINOR

### [MINOR] ex_1–ex_4 prose and interpretation slips

**File and evidence**

- `solutions/ex_1/01_standard_ae.py:190-191, 199` says "GELU activations (used from variant 3 onward)". Variants 03–10 use only `nn.ReLU`.
- `solutions/ex_2/02_resnet_se.py:72-99, 133-134` attributes degradation to vanishing gradients. He et al. 2015 §4.1 says it is "unlikely to be caused by vanishing gradients". The same slip is on `lessons/02/slides.html:227`; see the CNN slips finding.
- `solutions/ex_4/01_self_attention_from_scratch.py:215-218` claims that with random embeddings, content words attend to content words and pads form a cluster. In fact Q=K=V=random embeddings gives a diagonal-dominated map (self-score ‖e‖²/√d ≈ 5.7 vs off-diagonal ≈ ±1). `padding_idx=0` pads are zero vectors, so they attend uniformly.
- `solutions/ex_2/03_production_pipeline.py:86-87` says "PyTorch models trained on NVIDIA GPUs need NVIDIA GPUs to serve". This is false: `map_location` loads weights on CPU or MPS.
- `solutions/ex_2/04_hyperparameter_study.py:687-688` says "AWS p3.2xlarge … $3.06/hr in ap-southeast-1". That is the us-east-1 rate; Singapore is about $4.23/hr. It also names a vendor.

**Fix**: Correct each statement as indicated.

### [MINOR] ex_4 reports the best accuracy across epochs on the test split

**File**
- `shared/mlfp05/ex_4.py:175` (`val_loader` is built from `test_t`)
- `solutions/ex_4/02:357-364`, `03:239-246`, `04:478`, `05:469-471`

**Evidence**: The files report `max(transformer_accs)`, `max(lstm_accs)` and `max(bert_accs)` as "best acc". In ex_4/04 that maximum is plotted beside per-class accuracy from the final epoch.

**Problem**: Selecting the epoch on the test split inflates the reported accuracy, and the ex_4/04 comparison mixes numbers from different epochs.

**Fix**: Carve a validation split out of train, report test accuracy for the chosen checkpoint, and take the per-class view from that same checkpoint.

### [MINOR] ex_0 formats drift: the solution lacks the scaffold's checkpoints, neither file has WHAT YOU'LL LEARN, and the MLEngine claim is overstated

**File**: `solutions/ex_0/00_destination_first.py:111-122`, `local/ex_0/00_destination_first.py`

**Evidence**

- The local file has 3 checkpoint blocks (4 asserts); the solution has none.
- Neither file has a WHAT YOU'LL LEARN header.
- The solution says "Every M5 lesson … will compose its own `MLEngine()`", but `MLEngine` appears only in ex_0.

**Fix**

- Sync the checkpoints into the solution.
- Add the header.
- Reword the MLEngine claim.

### [MINOR] Small scaffold and helper defects

**File and evidence**

- `local/ex_2/01_simple_cnn.py:600` and `local/ex_2/02_resnet_se.py:756-766` use doubled braces (`{{param_count:,}}`) inside `f"""…"""`, so the REFLECTION prints the literal placeholder text. The solution uses single braces.
- The `shared/mlfp05/__init__.py` docstring says `from shared.mlfp05.ex_2 import load_cifar10, train_cnn`, but the function is `train_model`.

**Fix**: Use single braces, and rename the function in the docstring to `train_model`.


### [MINOR] Task 4 problem statement makes false claims about bag-of-words and the accuracy ceiling; the majority rate is inconsistent

**File**: `assessment/task_4/problem.md:12, 22-24, 61`

**Evidence**

The problem statement claims:
- "clear an accuracy floor that a bag-of-words guess cannot"
- "the 5K-row slice has a real accuracy ceiling (~0.80)"

I measured the opposite on the bundled data. TF-IDF + LogisticRegression trained on `ag_news.parquet` (5,000 rows) scores **0.855** on `ag_news_test.parquet`. That is above both the 0.72 floor and the claimed ~0.80 ceiling.

The majority rate is also inconsistent. The test majority is 0.274 (`bincount = [268 274 205 253]`). Requirement 4 says "~0.30", while the visible check says 0.27.

**Problem**

Students are taught that attention is necessary to clear the floor and that ~0.80 is the ceiling. Both claims are false.

**Fix**

- Drop the BoW and ceiling claims, or state the measured BoW baseline (~0.85).
- Use "~0.27" consistently for the majority rate.

### [MINOR] Task 2's "held-out generalisation" check is a subset of the already-scored test set; the Task 3 random-walk wording is wrong

**File**
- `assessment/task_2/problem.md:55-56`, `assessment/task_2/grader.py:125-131`
- `assessment/task_3/problem.md:18-21`, `assessment/README.md:35-37`

**Evidence**

- In Task 2, `held = model_preds[::3]` is every third row of the same `X_test` that the 0.90 accuracy check already scores. Nothing in it is held out, so the check cannot detect memorisation, contrary to "Generalisation, not memorisation — … a held-out slice".
- Task 3 says "daily equity returns are a near-perfect random walk". Under the random-walk hypothesis it is *prices* that follow a random walk; *returns* are approximately uncorrelated noise. The task's own "tomorrow = today" baseline is a price forecast.
- "Mathematically impossible" also overstates the case. Beating a random walk is impossible *in expectation*, not mathematically impossible on a finite sample.

**Fix**

- Task 2: either carve a real held-out split from training data the student never sees, or rename the check to "accuracy is stable across a subset".
- Task 3: say "prices follow a near-random walk, so tomorrow = today is near-optimal".

### [MINOR] "CVAE" means Conditional VAE in the deck but Contractive VAE in the exercise

**File**
- `deck.html:2119` ("CVAE — Conditional VAE — condition generation on class labels")
- `speaker-notes.md:225` ("CVAE (conditional VAE)")
- vs `specs/module-5.md:110` ("CVAE (Contractive + Variational)") and `solutions/ex_1/10_contractive_vae.py` (`class ContractiveVAE`)

**Problem**: Students meet two different meanings of "CVAE".

**Fix**: The standard expansion of CVAE is "Conditional VAE", so rename the exercise and spec technique to "Contractive VAE" without the "CVAE" acronym. Alternatively, align the deck survey row to name the Contractive VAE that students actually build.

### [MINOR] Permutation "invariant" should be "equivariant"

**File**
- `deck.html:3847`
- `lessons/04/slides.html:286`
- `lessons/04/textbook.html:266-267`
- `lessons/04/notes.html:266-267`
- `speaker-notes.md:677`

**Problem**: Self-attention without positional encoding is permutation-*equivariant*: permuting the inputs permutes the outputs. The texts' own explanation ("the output shuffles the same way") describes equivariance.

**Fix**: Say "permutation-equivariant (order-blind)".

### [MINOR] Lesson 5.4 slide's majority-bit example contradicts its own "forgetting PE" trap; Whisper is mislabelled as CTC

**File**: `lessons/04/slides.html:432-443, 449, 426`

**Evidence**

The example label is `y = (X.sum(dim=1) > L//2)`, which depends only on the multiset of tokens. The model mean-pools the encoder output. Even without positional encoding, mean-pooling the embeddings gives the fraction of ones exactly, so a set model solves this task perfectly.

**Problem**

Two claims on the slides are wrong:
- "Forgetting positional encoding — model becomes a set classifier, stuck at 50%" is false for this example.
- "non-trivial because the model must count across positions" is also wrong.

Separately, Whisper is a seq2seq encoder–decoder, not CTC.

**Fix**

- Use an order-dependent task to illustrate the PE trap (e.g. "is the first bit 1?").
- Set the Whisper objective to "seq2seq (next-token)".

### [MINOR] LSTM equation and gate counts are inconsistent; the speaker-note gradient sentence is reversed

**File**
- `deck.html:3152` ("LSTM: All Five Equations") and `3202`
- `index.html` ("LSTM (six gate equations)")
- `lessons/03/slides.html:317` ("all six")
- `lessons/03/textbook.html:39` vs `165`, and `229`
- `speaker-notes.md:473, 477`
- `deck.html:3146` and `speaker-notes.md:468` ("Even if f_t is near 1, the gradient passes through unchanged")

**Problem**

- The spec says "All 6 gate equations". The artifacts alternate between 5 and 6 equations and between 3 and 4 gates.
- The gradient sentence inverts its own condition: the gradient passes through *because* f_t ≈ 1.

**Fix**

- Use "three gates + candidate; six equations" everywhere.
- Change the sentence to "When f_t ≈ 1 the gradient passes through nearly unchanged".

### [MINOR] CNN slide numeric and terminology slips

**File**
- `deck.html:2337`, `2653`, `2665`
- `lessons/02/slides.html:227`
- `lessons/02/textbook.html:281-282, 478`
- `lessons/02/notes.html:250`

**Evidence**

- `deck.html:2337` says "Same filter weights everywhere → translation invariance". Weight sharing gives translation *equivariance*. `lessons/02/slides.html:46` says equivariance correctly.
- `deck.html:2653` says "SE adds only ~2,500 parameters for a 64-channel layer". With the slide's own `reduction=16`, the count is 64·4 + 4 + 4·64 + 64 = **580**.
- `deck.html:2665` says "W ~ N(0, √(2/n_in))". In N(μ, σ²) notation the variance is 2/n_in (std √(2/n_in)).
- `lessons/02/slides.html:227` says degradation beyond ~20 layers is "Not overfitting — gradient vanishing". He et al. (2015) state the degradation "is unlikely to be caused by vanishing gradients" (BN was used); they attribute it to optimisation difficulty. `specs/module-5.md:150, 171` repeat the vanishing-gradient framing.
- `lessons/02/textbook.html:478` says "Xavier … variance 1/n_in". Glorot is 2/(n_in+n_out); 1/n_in is LeCun.
- `lessons/02/textbook.html:281-282` says "Modern ResNets use SE blocks by default". They do not.
- `lessons/02/notes.html:250` says "reduction ratio 8 is standard". The SE paper uses 16.

**Fix**: Correct each value as stated above.

### [MINOR] Citation slips in the diagnostics and appendix slides

**File**: `deck.html` slides 5G-ii (≈1000-1040), 5K (≈1429-1500), 5M (≈1565-1610)

**Evidence**

- "Zech et al., 2018 (Nature Medicine)". That paper ("Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs") appeared in *PLOS Medicine*. Its confound was hospital-specific tokens/markers, found with CAM-style heatmaps.
- "μP … Tune at 100M, transfer zero-shot to 7B+ (Yang & Hu, 2021)". μTransfer is *Tensor Programs V* (Yang et al., 2022). Yang & Hu 2021 introduced μP without the transfer result.
- The Scaling Laws speaker note says "The Chinchilla rule says you need ~20 tokens per parameter MINIMUM". Chinchilla's ~20 tokens/param is the *compute-optimal* ratio, not a minimum. The slide body itself says "balanced".

**Fix**: Correct the journal name, the citation and the wording.

### [MINOR] The "Course Home" link in index.html is broken

**File**: `index.html` (`href="../index.html"`)

**Evidence**: `modules/index.html` does not exist; `modules/` contains only `assets/`, `decks-readme.md` and `mlfp01..06`. All six module index pages carry the same link.

**Fix**: Add a `modules/index.html` course landing page, or point the link at the actual course home.

### [MINOR] Wrong cross-module references

**File**
- `deck.html:5620` ("connects back to M4 supervised learning")
- `deck.html:6452` ("Same pattern as TrainingPipeline from M4")
- `deck.html:6663` ("Tabular — Gradient boosting (not DL) — M4")
- `speaker-notes.md:1097`
- `lessons/06/textbook.html:785` ("M3 Feature Engineering")
- `lessons/05/textbook.html:585-586` ("In Module 6, we use diffusion … and revisit FID")

**Evidence**

- Per CLAUDE.md, gradient boosting and TrainingPipeline are M3 (MLFP03).
- FeatureEngineer is M2.
- `specs/module-6.md` contains no diffusion or FID.

**Fix**: Correct the module numbers and drop the M6 diffusion promise.

### [MINOR] FID is described as "per-image quality", with Inception-scale thresholds applied to LeNet-64 features

**File**: `solutions/ex_5/03_gan_evaluation.py:75-77, 96-101, 299, 307-308, 855-856`

**Evidence**

- The text says "QUALITY (per-image fidelity) … Metric: FID", "FID < 10: publication-quality" and "Production papers target FID < 10".
- The feature extractor is a 64-d LeNet (`shared/mlfp05/ex_5.py:192-210`), not 2048-d Inception.
- The comment at line 299 cites `scipy.linalg.sqrtm`, but the code uses `np.linalg.eigvals`.

**Problem**

- FID is a distribution-level metric, not a per-image score.
- Inception-based thresholds do not transfer to LeNet-64 features.

**Fix**

- Describe FID as distributional (fidelity + diversity).
- State that the values are comparable only within this extractor.
- Fix the stale scipy comment.

### [MINOR] Graph-learning terminology slips

**File**
- `solutions/ex_6/03_graphsage.py:105, 488, 565-566`
- `solutions/ex_6/05_architecture_comparison.py:555`
- `shared/mlfp05/ex_6.py:138`

**Evidence**

- `ex_6/03:105` calls GAT transductive. GAT is inductive: its paper evaluates on unseen PPI graphs.
- `ex_6/03` ticks "[x] INDUCTIVE learning: generalises to unseen nodes", but the Cora split is transductive, so this is never demonstrated.
- `shared/mlfp05/ex_6.py:138` calls D^{-1/2}AD^{-1/2} "the symmetric Laplacian". It is the symmetric-normalised adjacency; the Laplacian is I − D^{-1/2}ÂD^{-1/2}.

**Fix**

- Remove GAT from the "transductive" line.
- Either hold out nodes so inductive generalisation is actually shown, or soften the reflection claim.
- Rename the matrix in `ex_6.py` to "normalised adjacency".

### [MINOR] PPO theory says actor and critic share a trunk; the class deliberately doesn't

**File**: `solutions/ex_8/02_ppo.py:85` vs `121-129`

**Evidence**: Line 85 says "They share a neural network trunk". The class docstring says "We deliberately do NOT share a trunk".

**Fix**: Make line 85 match the separate-network design.

### [MINOR] Ex 7: failed pretrained-weight downloads silently fall back to random weights; opset claim and dataset name are wrong

**File**
- `solutions/ex_7/03_data_efficiency.py:93-95`
- `solutions/ex_7/04_adapter_modules.py:190-192, 256-258`
- `solutions/ex_7/05_production_deployment.py:105-107, 158`
- `solutions/ex_7/01_from_scratch_baseline.py:363`
- `solutions/ex_7/02_transfer_resnet18.py:689`

**Evidence**

- The fallback is `except Exception: model = resnet18(weights=None)` with no message. If the download fails, the "transfer" results silently come from a random frozen backbone.
- ex_7/05:158 says "opset 17 is latest stable". ONNX now ships opset 21+.
- ex_7/01 and ex_7/02 say "Fashion-MNIST", but the data is CIFAR-10.

**Fix**

- Warn or raise when the pretrained download fails.
- Change the opset wording to "opset 17 (widely supported)".
- Say CIFAR-10.

### [MINOR] The ex_5 helper writes generated outputs into the `shared/mlfp05/` package directory

**File**: `shared/mlfp05/ex_5.py:44`

**Evidence**: `OUTPUT_DIR = _HERE.parent`. 18 `ex_5_*` png/html files currently sit in `shared/mlfp05/`; they are untracked and ignored by `.gitignore:88`. Every other exercise writes to `outputs/`.

**Problem**: Writing into the package fails if `shared` is installed read-only, and it clutters the package.

**Fix**: `OUTPUT_DIR = Path("outputs") / "ex5_gans"`, then `mkdir(parents=True, exist_ok=True)`.

### [MINOR] local/ex_6/01 is out of date relative to its solution

**File**: `local/ex_6/01_gcn.py:219-272` vs `solutions/ex_6/01_gcn.py:219-279`

**Evidence**: The solution has a "Predicted Labels" embedding plot and interpretation (230-241, 273-279). The local copy has neither, and no TODO marks the gap.

**Fix**: Regenerate the scaffold from the solution.

### [MINOR] Lesson 5.8 textbook code: environments are incomplete or violate their declared spaces; DQN loop ignores truncation

**File**
- `lessons/08/textbook.html:615-645`
- `lessons/08/textbook.html:539-540, 572-573`
- `textbook.md:1433`

**Evidence**

- `ChurnPreventionEnv` has no `reset()`, `_get_state()` or `_base_churn_prob()`, and `customer_data` is undefined; step 2 calls `env.reset()`.
- The supply-chain env declares `Box(low=0, ...)`, but inventory is clamped at −20, so observations leave the declared space.
- `next_state, reward, done, _, _ = env.step(action)` drops `truncated`, so episodes that hit a time limit never end the `while not done` loop.

**Fix**

- Complete the class, or label it explicitly as a sketch.
- Set `low=-20` for the inventory dimension.
- Use `done = terminated or truncated`.

### [MINOR] RL theory slips in the textbooks

**File**
- `lessons/08/textbook.html:431`
- `textbook.md:1341-1343`

**Evidence**

- `lessons/08/textbook.html:431` says "Within clip range: full gradient. Outside: gradient clipped to zero." The gradient is zero only when the ratio leaves the range in the direction the advantage favours. For example, with A>0 and r<1−ε the unclipped term is selected and still has gradient.
- `textbook.md:1341-1343` presents `Q(s,a)=E[R+γ max Q(S',a')]` as the generic "value of a state-action pair". It is the Bellman *optimality* equation for Q*, which the spec asks to be distinguished from the expectation equation.

**Fix**

- State the asymmetric clipping rule.
- Write Q* and label it "Bellman optimality".

### [MINOR] Miscellaneous technical slips in the textbooks

**File**
- `textbook.md:803`
- `lessons/05/textbook.html:526-527`
- `lessons/07/textbook.html:120, 580` and `lessons/07/notes.html:19`
- `textbook.md:1049`

**Evidence**

- `textbook.md:803` says each head uses "d_k/h per head". It should be d_model/h; the code at line 810 already uses `d_model // n_heads`.
- `lessons/05/textbook.html:526-527` calls the zero-centred penalty ‖∇‖²→0 "one-sided / WGAN-LP". The one-sided (LP) penalty is max(0, ‖∇‖−1)²; the zero-centred one is R1 (Mescheder et al. 2018).
- `lessons/07` says ResNet was pretrained on 14M ImageNet images. `ResNet18_Weights.DEFAULT` is IMAGENET1K_V1, which is ~1.28M images; the deck at 5402 says 1.28M correctly.
- `textbook.md:1049` says "Stable Diffusion and DALL-E are based on diffusion". DALL-E 1 was an autoregressive transformer over dVAE tokens; only DALL-E 2 and 3 use diffusion.

**Fix**: Correct each statement as above.

### [MINOR] Deprecated or missing APIs and hard-coded CUDA branches in textbook snippets

**File**
- `textbook.md:1283` (`resnet18(pretrained=True)`); also `deck.html:5469` (`resnet50(pretrained=True)`)
- `textbook.md:432`, `lessons/02/textbook.html:465` (`torch.cuda.amp`)
- `textbook.md:1119` (`from pytorch_fid import fid_score`)
- `lessons/01/textbook.html:361`, `lessons/02/textbook.html:333`, `lessons/03/textbook.html:340`, `lessons/04/textbook.html:416` (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`)

**Evidence**

- torchvision 0.27.1 warns that `pretrained` is deprecated.
- `torch.cuda.amp.autocast` emits a FutureWarning and does not apply on MPS.
- `pytorch_fid` is not installed, so its import raises `ImportError`.
- `specs/module-5.md:59` says there is "NO … `if torch.cuda.is_available()` branch in any M5 lesson", and the branch also ignores MPS.

**Fix**

- Use `weights=...ResNet18_Weights.DEFAULT`.
- Use `torch.amp.autocast(device_type)` or the `PRECISION` constant.
- Use `torchmetrics` FID or the course's helper in `shared/mlfp05/ex_5.py`.
- Use `shared.kailash_helpers.get_device()`.

### [MINOR] Lesson worked examples claim to mirror exercises, or load data, that do not exist

**File**
- `lessons/01/textbook.html:347-348`
- `lessons/02/textbook.html:339`
- `lessons/05/textbook.html:435-436`
- `lessons/07/textbook.html:606, 624-625`

**Evidence**

| Location | Claim | Actual |
| --- | --- | --- |
| `lessons/01/textbook.html:347-348` | "You will recognise every line from M5 exercise 1" | The page uses 16×16 synthetic blobs; ex_1 uses Fashion-MNIST. |
| `lessons/02/textbook.html:339` | `make_shapes` (comment: "see exercise 2 solution for the generator") | It returns all-zero images, so the example trains on blanks. No such generator exists in ex_2. |
| `lessons/05/textbook.html:435-436` | "mirrors exercise 5" | The page uses eight 2-D Gaussians; ex_5 uses MNIST. |
| `lessons/07/textbook.html:606, 624-625` | `ImageFolder("data/mask_detection/train")` | No such data exists. |

**Fix**: Make the examples self-contained (implement `make_shapes`), or point them at the real exercise data. Delete the "mirrors exercise" claims.

### [MINOR] Lesson notes pages have drifted from their slides.html (missing notes, wrong slide counts)

**File**: `lessons/02-07/notes.html`

**Evidence**

| Lesson | Notes | Slides | Gap |
| --- | --- | --- | --- |
| 02 | 17 | 18 | "Pooling" has no note, so every later number is off by one |
| 03 | 17 | 19 | "Why RNNs forget" and "The cell state highway" have no notes |
| 04 | 17 | 19 | "Attention: which words matter" has no note |
| 05 | header says "16 slides" | 14 | — |
| 06 | header says "13" | 12 | — |
| 07 | header says "9" | 10 | "Lower layers transfer…" has no note |

Lessons 06–08 notes are compact but complete, not stubs.

**Fix**: Regenerate the notes against the current slides.

---

## Coverage summary

I read the module-5 spec in full, along with `_index.md`, `exercise-standards.md`, `domain-integrity.md` and `independence.md`. I then checked:

- **Master documents.** `deck.html` (127 slides; the already-fixed DLDiagnostics slides 5B2/5D/5F/5H/5L were skipped), `speaker-notes.md`, `textbook.md`, `README.md` and `index.html`.
- **Lesson pages.** All 24 files: `lessons/01–08/{slides,textbook,notes}.html`.
- **Exercises.** All 43 technique files: `solutions/ex_0`–`ex_8` and their `local/` scaffolds. I compared each pair with diff/difflib, counted blanks, checkpoints and REFLECTION blocks, and ran AST name checks.
- **Shared helpers.** `shared/mlfp05/__init__.py` and `ex_1.py`–`ex_8.py`.
- **Assessment.** All 17 files: README plus `task_1`–`task_4` `{problem,starter,solution,grader}`. All four graders pass their reference solutions (25s / 14s / 51s / 7s under load). Exploit probes (untrained models with a fabricated `y_test`) passed Tasks 1, 3 and 4. A TF-IDF baseline was run on the Task 4 data.
- **`diagnostic-reference/`.** The README and the 7 captured reports.
- **Links.** All relative links in `index.html`, `deck.html` and `lessons/*/slides.html`. The only broken one is `../index.html`; the lesson textbook pages were also checked and are clean.
- **Installed APIs.** Checked with `inspect.signature`/`hasattr` and small probes against kailash-ml 2.2.2 (OnnxBridge, InferenceServer and its config/coroutines, ModelVisualizer, `rl`.RLTrainer/`rl_train`, `diagnose`/DLDiagnostics, ModelRegistry, ExperimentTracker), torch 2.12.1, torchvision 0.27.1, transformers 4.57.6 and NumPy 2.4.6. I also checked which optional packages are present: `stable_baselines3` and `pytorch_fid` are absent; `torch_geometric`, `gymnasium` and `onnxruntime` are present.
- **Independence and model names.** No PCML or prior-course codes, no partner or funding-body references, no hardcoded LLM model names and no pandas imports anywhere in the module. The only finding is the named organisations credited with invented deployments, listed above.

In total about 140 files were read, excluding generated Colab notebooks.

Counts: BLOCKING 12 · MAJOR 32 · MINOR 26.
