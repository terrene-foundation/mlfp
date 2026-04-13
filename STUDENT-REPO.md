# Student Repository Boundary

This document defines what goes into the student-facing repository and what stays instructor-only.

## Build

```bash
./scripts/build-student-repo.sh /path/to/student-repo
```

## What Students Get

| Path                       | Content                                         |
| -------------------------- | ----------------------------------------------- |
| `data/`                    | All datasets                                    |
| `modules/mlfpNN/readings/` | deck.pdf, textbook.pdf, notes.pdf               |
| `modules/mlfpNN/local/`    | Exercise .py files (R10 directories)            |
| `modules/mlfpNN/colab/`    | Exercise .ipynb notebooks                       |
| `shared/`                  | Python utilities (data_loader, kailash_helpers) |
| `pyproject.toml`           | Dependencies                                    |
| `uv.lock`                  | Lockfile                                        |
| `LICENSE`                  | Apache 2.0                                      |
| `.env.example`             | Environment template                            |
| `README.md`                | Student-facing README                           |

## What Students Do NOT Get

| Path                              | Why                                                            |
| --------------------------------- | -------------------------------------------------------------- |
| `modules/mlfpNN/solutions/`       | Complete solutions with `_shared.py`                           |
| `modules/mlfpNN/lessons/`         | Raw HTML per lesson                                            |
| `modules/mlfpNN/deck.html`        | Raw reveal.js deck source                                      |
| `modules/mlfpNN/textbook.md`      | Raw textbook markdown                                          |
| `modules/mlfpNN/speaker-notes.md` | Raw speaker notes                                              |
| `modules/mlfpNN/assessment/`      | Exam + assignment solutions (last lesson only, moved manually) |
| `modules/mlfpNN/index.html`       | Lesson index                                                   |
| `specs/`                          | Curriculum specifications                                      |
| `scripts/`                        | Build and validation scripts                                   |
| `tests/`                          | Test suite                                                     |
| `workspaces/`                     | Workspace state                                                |
| `.claude/`                        | Agent configurations                                           |
| `pdf/`                            | PDF build directory                                            |
| `textbook/`                       | Textbook source (markdown, html, python, rust)                 |
| `docs/`                           | Developer documentation                                        |
| `outputs/`                        | Audit screenshots, artifacts                                   |

## Exercise Import Pattern

Solutions use `from _shared import ...` (internal, never distributed).

Student exercises MUST NOT use `from _shared import`. Instead:

- Common utilities: `from shared.kailash_helpers import get_device, setup_environment`
- Data loading: `from shared.data_loader import MLFPDataLoader`
- Exercise-specific helpers: inlined at top of each technique file

## PDF Generation

PDFs must be generated before building the student repo:

- `deck.pdf` — print deck.html to PDF (or use existing in `pdf/decks/`)
- `textbook.pdf` — render textbook.md to PDF (or use existing in `pdf/textbooks/`)
- `notes.pdf` — render speaker-notes.md to PDF (or use existing in `pdf/notes/`)

Place all PDFs in `modules/mlfpNN/readings/` before running the build script.

## Missing PDFs

| Module | deck.pdf | textbook.pdf | notes.pdf |
| ------ | -------- | ------------ | --------- |
| mlfp01 | Yes      | Missing      | Missing   |
| mlfp02 | Yes      | Missing      | Missing   |
| mlfp03 | Yes      | Missing      | Missing   |
| mlfp04 | Yes      | Missing      | Missing   |
| mlfp05 | Yes      | Yes          | Missing   |
| mlfp06 | Missing  | Missing      | Missing   |
