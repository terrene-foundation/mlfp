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

## Progressive Scaffolding

| Module | Provided | Stripped |
|--------|----------|---------|
| M1 | ~70% | Only key arguments |
| M2 | ~60% | Arguments + some method calls |
| M3 | ~50% | Method calls + some setup |
| M4 | ~40% | Setup + calls + some logic |
| M5 | ~30% | Most logic, keep structure |
| M6 | ~20% | Imports and comments only |

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
