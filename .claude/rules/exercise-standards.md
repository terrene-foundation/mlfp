# Exercise Standards

## Every Exercise MUST Have

1. **Learning objective** — One sentence stating which Kailash engine/pattern students practice
2. **Exercise header** — `# ════` block with title, description, and numbered TASK steps
3. **`# TODO:` markers** — Each blank has a hint pointing to the correct API (engine name, method name, or parameter)
4. **Complete solution** — In `modules/ascentN/solutions/` that runs end-to-end without errors
5. **Three formats** — local (.py), Jupyter (.ipynb), Colab (.ipynb)
6. **Data loading** — Via `ASCENTDataLoader`, never hardcoded paths

## Progressive Scaffolding

| Module | Provided | Stripped |
|--------|----------|---------|
| M1 | ~70% | Only key arguments |
| M2 | ~60% | Arguments + some method calls |
| M3 | ~50% | Method calls + some setup |
| M4 | ~40% | Setup + calls + some logic |
| M5 | ~30% | Most logic, keep structure |
| M6 | ~20% | Imports and comments only |

## MUST NOT

- Leave `____` placeholders in solution files
- Strip import statements from exercises
- Strip data loading code from exercises
- Write exercises that require frameworks from later modules
- Include `import pandas` in any file (polars only)
- Hardcode API keys or model names
