---
paths:
  - "modules/**"
---

# Exercise Standards

## Every Exercise MUST Have

1. **Learning UX header** — `# ════` block with:
   - Title matching the v2 spec lesson
   - WHAT YOU'LL LEARN: 3-5 bullets from spec Learning Objectives
   - PREREQUISITES: which prior exercises/lessons to complete first
   - ESTIMATED TIME: 45-90 minute target
   - Numbered TASK steps with clear sub-steps

2. **`# TODO:` markers** — Each blank has a hint pointing to the correct API (engine name, method name, or parameter)

3. **Checkpoint assertions** — After each TASK section:

   ```python
   # ── Checkpoint N ─────────────────────────────────────────
   assert result is not None, "Task N: result should not be None"
   print("✓ Checkpoint N passed")
   ```

4. **Interpretation prompts** — After major print outputs, explain what the result means for the learner

5. **Reflection section** — At end of every exercise:

   ```python
   # ══════════════════════════════════════════════════════════
   # REFLECTION
   # ══════════════════════════════════════════════════════════
   print("""
   What you've mastered:
     ✓ [Skill 1]
     ✓ [Skill 2]

   Next: In Exercise N+1, you'll apply this to [topic]...
   """)
   ```

6. **Complete solution** — In `modules/mlfpNN/solutions/` that runs end-to-end without errors

7. **Two formats** — local (.py) + Colab (.ipynb). No Jupyter middle format.

8. **Data loading** — Via `MLFPDataLoader`, never hardcoded paths

## R10 Directory Structure — MUST for Multi-Technique Exercises

Exercises with multiple techniques MUST be directories, not single files. See `specs/redlines.md` Redline 10.

### Layout

```
modules/mlfpNN/solutions/ex_N/       # R10 directory per exercise
  01_technique_a.py
  02_technique_b.py
  03_technique_c.py

modules/mlfpNN/local/ex_N/           # student version (scaffolded)
  01_technique_a.py
  02_technique_b.py
  ...

modules/mlfpNN/colab/ex_N/           # notebook version
  01_technique_a.ipynb
  ...

shared/mlfpNN/ex_N.py                # shared utilities (installable package)
```

### 5-Phase Structure Per Technique File (MUST)

Each technique file MUST contain these 5 phases in order:

1. **Theory** — Why the technique exists, non-technical intuition
2. **Build** — Model/algorithm implementation (signatures preserved, bodies scaffolded)
3. **Train** — Training loop with kailash-ml ExperimentTracker
4. **Visualise** — Visual proof of model behaviour (R9A: not just loss curves)
5. **Apply** — Real-world business scenario with named industry + dollar-value impact (R9B)

**Why:** R10 separates the narrative per technique — students can focus on one idea at a time without wading through 2000 lines. Each file is independently runnable.

**BLOCKED:**

- "The exercise is too short to need splitting" — if it has >1 technique, it's a directory
- "Students can just scroll through the monolithic file" — breaks focus and reading flow
- "Putting applications in a separate file is cleaner" — R10 mandates inline per-technique applications

### Import Pattern (MUST)

Technique files MUST import helpers from the shared package, NOT from local `_shared.py` or `helpers.py`:

```python
# DO — in modules/mlfp05/solutions/ex_1/02_undercomplete_ae.py
from shared.mlfp05.ex_1 import (
    INPUT_DIM, LATENT_DIM, EPOCHS, OUTPUT_DIR, device,
    load_fashion_mnist, train_variant, show_reconstruction,
)

# DO NOT — local helpers that break when cwd changes
from _shared import load_fashion_mnist
from helpers import train_variant
```

**Why:** Local imports only work when running from the exercise directory. The shared package works from any cwd after `uv sync`, which is how students (and interactive Python) actually run code.

### Single-Technique Exercises

Exercises that cover exactly one technique MAY remain single files. The test: if the exercise has multiple `class XxxModel` definitions or multiple distinct algorithms, it MUST be a directory.

## Progressive Scaffolding

| Module | Provided | Stripped                      |
| ------ | -------- | ----------------------------- |
| M1     | ~70%     | Only key arguments            |
| M2     | ~60%     | Arguments + some method calls |
| M3     | ~50%     | Method calls + some setup     |
| M4     | ~40%     | Setup + calls + some logic    |
| M5     | ~30%     | Most logic, keep structure    |
| M6     | ~20%     | Imports and comments only     |

## Scaffolding Rules

- Checkpoint assertions are NEVER stripped — students must make them pass
- WHAT YOU'LL LEARN and REFLECTION blocks are preserved verbatim
- TASK headers and sub-step comments are preserved
- Hints must be calibrated to module difficulty level:
  - M1: explicit hints (`# Hint: pl.col("price").mean()`)
  - M3: partial hints (`# Hint: use .fit() then .predict()`)
  - M6: minimal hints (`# Hint: configure the pipeline`)

## MUST NOT

- Leave `____` placeholders in solution files
- Strip import statements from exercises
- Strip data loading code from exercises
- Strip checkpoint assertions from exercises
- Write exercises that require frameworks from later modules
- Include `import pandas` in any file (polars only)
- Hardcode API keys or model names
- End an exercise without a REFLECTION section
- Skip the WHAT YOU'LL LEARN header
