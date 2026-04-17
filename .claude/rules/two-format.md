---
paths:
  - "modules/**"
  - "shared/**"
  - "scripts/generate_selfcontained_notebook.py"
---

# Two-Format Consistency

Every exercise ships in exactly two formats: local `.py` and self-contained Colab `.ipynb`. The `.py` is hand-authored; the `.ipynb` is generated. No third parallel format, no Jupyter `%pip`, no git-clone Colab. See also Redline 11.

Origin: Session 2026-04-17 consolidation (commit `8696560`); generator + CI guard chain landed in commits `2c8676d` and `38549f1`.

## Formats

| Format             | Location                                              | Data Loading         | Authoring                              |
| ------------------ | ----------------------------------------------------- | -------------------- | -------------------------------------- |
| Solution           | `modules/mlfpNN/solutions/ex_N/NN_technique.py`       | `shared.data_loader` | Hand-authored, source of truth         |
| Student            | `modules/mlfpNN/local/ex_N/NN_technique.py`           | `shared.data_loader` | Hand-authored from solution (stripped) |
| Colab (instructor) | `modules/mlfpNN/colab-selfcontained-solutions/…ipynb` | Inlined helpers, pip | Generator output, do NOT hand-edit     |
| Colab (student)    | `modules/mlfpNN/colab-selfcontained/…ipynb`           | Inlined helpers, pip | Generator output, do NOT hand-edit     |

## MUST Rules

### 1. Colab Notebooks MUST Be Produced by the Generator

Every `.ipynb` under `modules/mlfpNN/colab-selfcontained{,-solutions}/` MUST be produced by `scripts/generate_selfcontained_notebook.py` from the corresponding `.py` source. Hand-editing a notebook is BLOCKED.

```bash
# DO — regenerate after any change to solutions/ or local/ or shared/
.venv/bin/python scripts/generate_selfcontained_notebook.py \
  --solutions modules/mlfp05/solutions \
  --local modules/mlfp05/local \
  --out-solutions modules/mlfp05/colab-selfcontained-solutions \
  --out-students modules/mlfp05/colab-selfcontained

# DO NOT — open the .ipynb in Colab/Jupyter, fix a typo, save back
# The next generator run will overwrite the edit silently.
```

**BLOCKED responses:**

- "Just a one-character fix, the generator is overkill"
- "I'll regenerate later"
- "The notebook is for students, the `.py` doesn't need updating"
- "Colab saved this automatically, that's not a hand edit"

**Why:** Any hand-edit silently drifts from `local/*.py`. The next generator run overwrites the edit with no signal, and the two formats diverge until a student reports a broken exercise. The generator is the single source of truth that keeps `.py` and `.ipynb` in lockstep.

### 2. Exercise Code MUST Be Identical Across Formats

The `# TODO:` markers, `____` blanks, hints, variable names, Kailash engine calls, and expected outputs MUST be character-identical between `local/*.py` and the generated notebook's exercise cells. Only the preamble (Cell 0 pip install, Cell 1 inlined helpers) may differ.

```python
# DO — single source, generator produces both formats
# local/ex_1/01_standard_ae.py:
model = # TODO: nn.Sequential(nn.Linear(784, 1024), nn.ReLU(), nn.Linear(1024, 784))

# DO NOT — divergent hints between formats
# local version says `# TODO: nn.Linear(784, 1024)`
# notebook says    `# TODO: implement the model` (weaker hint)
```

**Why:** A student who works in VS Code and another who works in Colab must solve the same exercise with the same hints. Divergent hints create two different learning experiences and split the feedback the instructor can act on.

### 3. Generator Output MUST Pass Syntax + Exec Guards Before Commit

Every change to `solutions/`, `local/`, `shared/`, or the generator itself MUST be followed by a successful run of:

```bash
.venv/bin/python scripts/check_notebook_syntax.py modules/
# (and Cell 1 exec smoke-test on at least one solution per module)
```

A failing guard BLOCKS the commit.

```bash
# DO — run both guards before `git commit`
.venv/bin/python scripts/check_notebook_syntax.py modules/  # → ✓ N notebooks parse cleanly

# DO NOT — commit, assume CI catches it
git commit -m "..." && git push
# Prior session shipped 84 broken notebooks because the guards didn't exist yet.
```

**Why:** Commit `6b28127` shipped 84 M5 notebooks with `SyntaxError` in Cell 3 because there was no pre-commit guard. The `check_notebook_syntax.py` and Cell 1 exec patterns are the only defense against this failure class recurring; skipping them invites the same outage.

## MUST NOT

- Hand-author a `.ipynb` file under `modules/mlfpNN/colab-selfcontained{,-solutions}/`

**Why:** Generator overwrites the edit on next run; divergence is silent and only surfaces as broken student exercises.

- Re-introduce `colab/`, `colab-instructor/`, or `notebooks/` directories

**Why:** Consolidation to self-contained is the mandate; any parallel format re-introduces the maintenance burden that prompted this rule.

- Ship a notebook whose Cell 0 is anything other than the generator-emitted pip + GPU check

**Why:** A custom Cell 0 invariably forgets a package, adds a git clone that drops `shared/` into `sys.path` inconsistently, or installs an outdated Kailash version. Trust the generator.

- Use `pd.read_csv()` or `import pandas` anywhere

**Why:** Polars is the mandated data library (see `framework-first.md` and `rules/independence.md`); pandas breaks the Colab inline installs and contradicts `shared.data_loader`'s API.

- Hardcode Google Drive paths in `local/*.py`

**Why:** Local dev must work without a Drive mount; hardcoded paths break VS Code runs and force students to edit code before running.
