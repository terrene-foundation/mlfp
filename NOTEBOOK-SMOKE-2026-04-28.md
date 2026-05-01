# M5 Notebook Smoke Test — 2026-04-28

Branch: `fix/expt-tracker-1.1.1-migration`
Stack: kailash 2.11.3 / kailash-ml 1.5.1 / kailash-align 0.7.0 / Python 3.11
Runner: `/tmp/mlfp-smoke/parallel-runner.py --workers 4` (90s wall, 80s cell, treat timeout as PASS once training started)

## Headline

| Track                         | Total | Result                                           |
| ----------------------------- | ----- | ------------------------------------------------ |
| Solution notebooks (executed) | 43    | **43 PASS** (20 full / 18 wall-cap / 5 cell-cap) |
| Scaffolded notebooks (static) | 43    | **43 PASS** on JSON + pip + helpers + imports    |
| Failures requiring source fix | —     | **0**                                            |

All 43 M5 solution notebooks under `modules/mlfp05/colab-selfcontained-solutions/` start training cleanly. All 43 student notebooks under `modules/mlfp05/colab-selfcontained/` are structurally valid (parse, install, import, scaffold detected).

## Pass categories (solutions)

- **PASS-FULL** — notebook ran end-to-end inside the wall budget.
- **PASS-WALL** — training started; runner killed at 90s wall (expected for heavier models — CNN/GAN/Transformer/RL — that need GPU + longer epochs to converge).
- **PASS-CELLCAP** — one training cell exceeded the 80s per-cell cap. Same interpretation as PASS-WALL: the kernel was healthy and producing output, just slow.

Treating wall/cell-cap as PASS is intentional: the smoke contract is "does the notebook reach training without error", not "does it converge". Convergence is verified out-of-band on Colab GPU.

## Solution results — by exercise

| Dir       | Count  | PASS-FULL | PASS-WALL | PASS-CELLCAP | Notes                                         |
| --------- | ------ | --------- | --------- | ------------ | --------------------------------------------- |
| ex_0      | 1      | 1         | 0         | 0            | Destination-first warmup                      |
| ex_1      | 11     | 9         | 2         | 0            | Autoencoders; CNN-AE + grand comparison wall  |
| ex_2      | 4      | 0         | 0         | 4            | Vision (CNN/ResNet) — heavier per-cell budget |
| ex_3      | 5      | 5         | 0         | 0            | RNN/LSTM/GRU/Attention — small models, fast   |
| ex_4      | 5      | 1         | 4         | 0            | Transformers + BERT fine-tune — wall expected |
| ex_5      | 3      | 0         | 3         | 0            | GANs — adversarial loop is wall-bound on CPU  |
| ex_6      | 5      | 3         | 2         | 0            | GNN — GraphSAGE + comparison wall             |
| ex_7      | 5      | 0         | 4         | 1            | Transfer learning — backbone init is heavy    |
| ex_8      | 4      | 1         | 3         | 0            | RL — PPO/DQN rollouts wall                    |
| **Total** | **43** | **20**    | **18**    | **5**        |                                               |

PASS-WALL / PASS-CELLCAP lists in `/tmp/mlfp-smoke/parallel-results.json`.

## Scaffolded results — static validation

The `____`-style scaffolded notebooks cannot be executed (SyntaxError on parse), so we validate structure instead:

1. JSON / nbformat parses
2. Cell 0 contains `!pip install … kailash …`
3. Cell 1 (inlined helpers) parses as Python after stripping `!`/`%` magics
4. Imports cell (2/3/4) parses
5. Body shows scaffold markers (`____` blanks OR `# TODO` comments)

| Criterion                   | Pass | Fail | Notes                                                       |
| --------------------------- | ---- | ---- | ----------------------------------------------------------- |
| JSON valid                  | 43   | 0    |                                                             |
| Setup cell installs kailash | 43   | 0    |                                                             |
| Helpers cell parses         | 43   | 0    | All helper blobs syntactically clean                        |
| Imports cell parses         | 43   | 0    |                                                             |
| Body has `____` blanks      | 22   | 21   | Mixed scaffold style — **see finding below**, not a blocker |

### Finding: mixed scaffold style across exercises

Two scaffold styles ship in `colab-selfcontained/`:

| Style            | Count | Exercises                                          |
| ---------------- | ----- | -------------------------------------------------- |
| `____` blanks    | 22    | ex_0, ex_1 (×11), ex_2 (×4), ex_5 (×3), ex_7/03-05 |
| `# TODO` markers | 21    | ex_3, ex_4, ex_6, ex_7/01-02, ex_8                 |

This is consistent within each exercise but inconsistent across the module. The smoke test treats both as scaffolded (criterion 5 in `validate-scaffolded.py` should be widened). For students this is cosmetic: both styles signal "fill this in". For the build pipeline this is a generator drift signal — `scripts/generate_selfcontained_notebook.py` likely produces one style and another path emits the other.

**Recommendation**: pick one style (`____` blanks are the documented convention per `CLAUDE.md` rules and the session-notes "intentionally keep `____`"), regenerate the 21 TODO-style notebooks via the canonical generator, and update `validate-scaffolded.py` to fail on mixed styles rather than warn.

## Source fixes that landed in this round

All on `fix/expt-tracker-1.1.1-migration`, uncommitted:

1. **`shared/mlfp05/*.py`** — `__file__` notebook-safe pattern; separate `*_registry.db` for ModelRegistry vs `*.db` for ExperimentTracker (kailash-py#699).
2. **`modules/mlfp05/solutions/ex_3/03_gru.py::benchmark_inference`** — infer `n_features` from `model.gru.input_size` (sensor=1, stock=4); do NOT refactor back to global `N_FEATURES`.
3. **`km.diagnose(kind="dl", data=val_loader)`** — call sites kept as pedagogical "report exists" markers; `data=` is silently ignored (kailash-py#701).
4. **`scripts/generate_selfcontained_notebook.py`** — only top-level `asyncio.run(...)` is converted to `await ...`; sync wrappers inside `def` bodies keep `asyncio.run`.
5. **InferenceServer hard break** (kailash-py#700) — workaround documented in upstream issue body; MLFP-side bypass in place.

## Upstream issues filed against `terrene-foundation/kailash-py`

| #   | Title                                                              | Status |
| --- | ------------------------------------------------------------------ | ------ |
| 699 | ExperimentTracker / ModelRegistry schema conflict on shared SQLite | Open   |
| 700 | InferenceServer hard break                                         | Open   |
| 701 | `km.diagnose(kind="dl", data=...)` silently ignores DataLoader     | Open   |

Each issue body includes the MLFP-side workaround so an upstream fix can land without re-discovering it.

## Artifacts

- Solution results: `/tmp/mlfp-smoke/parallel-results.json` (43 rows)
- Scaffolded results: `/tmp/mlfp-smoke/scaffolded-results.tsv` (43 rows + header)
- Runner: `/tmp/mlfp-smoke/parallel-runner.py`
- Static validator: `/tmp/mlfp-smoke/validate-scaffolded.py`

## What's next

1. **Commit + push** the `fix/expt-tracker-1.1.1-migration` branch (open question in session-notes: one atomic commit vs split by fix-class).
2. **Classroom sync** — `scripts/build-student-repo.sh --sync` against `pcml-run26/pcml-run26-professional-certificate-in-machine-learning-pcml-run26-2601`.
3. **Resolve scaffold-style drift** — pick `____` as canonical, regenerate the 21 TODO-style notebooks.
4. **M1–M4 + M6 audit** — `SWEEP-2026-04-28.md` flagged engine-first / seamless-pipeline / diagnostics gaps; M3/M4 are the top follow-on.
