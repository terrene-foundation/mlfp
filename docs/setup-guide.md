# Setup Guide

Three ways to run the course materials.

---

## Option 1: Local Python (Recommended)

Best performance, full async support, all Kailash features including Nexus deployment.

### Requirements
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- 4GB+ RAM

### Setup

```bash
git clone https://github.com/terrene-foundation/ascent.git
cd ascent

# Create virtual environment and install dependencies
uv venv
uv sync

# For full course (includes deep learning, RL, agents):
uv sync --extra full

# Configure environment
cp .env.example .env
# Edit .env with your API keys (needed for Modules 5-6)

# Verify installation
uv run python -c "
import polars as pl
from kailash_ml import DataExplorer
print('Setup complete!')
print(f'polars {pl.__version__}')
"
```

### Running Exercises

```bash
# Run a local Python exercise
uv run python modules/ascent01/local/01_polars_fundamentals.py

# Run tests to validate your solutions
uv run pytest tests/test_module1.py -v
```

---

## Option 2: Jupyter Notebooks

Same as local, but interactive notebook environment.

### Additional Setup

```bash
# Install notebook extras
uv sync --extra notebooks

# Launch Jupyter
uv run jupyter notebook
```

### Running
Open any `.ipynb` file in `modules/ascent*/notebooks/`.

---

## Option 3: Google Colab

No local setup required. Runs in browser with free GPU access.

### Setup
1. Open any notebook from `modules/ascent*/colab/` in Google Colab
2. The first cell handles installation and Drive mounting:

```python
# This cell runs automatically in Colab notebooks
!pip install -q kailash-ml polars plotly gdown
from google.colab import drive
drive.mount('/content/drive')
```

3. For Modules 5-6 (agents, fine-tuning), add API keys in Colab Secrets:
   - Click the key icon in the left sidebar
   - Add `OPENAI_API_KEY`, `HF_TOKEN` as needed

### Limitations
- No Nexus deployment (Module 4 exercise 6 is local-only)
- Colab free tier: CPU only (GPU exercises are bonus)
- Session timeout after inactivity

---

## Environment Variables

Required for Modules 5-6 only:

| Variable | Module | Purpose |
|----------|--------|---------|
| `OPENAI_API_KEY` | 5, 6 | Kaizen agents, RAG |
| `DEFAULT_LLM_MODEL` | 5, 6 | Model selection |
| `HF_TOKEN` | 6 | Align fine-tuning |
| `GROQ_API_KEY` | 5 (optional) | Alternative LLM provider |

Modules 1-4 require no API keys.

---

## Troubleshooting

**polars import error**: Ensure you're using the virtual environment (`uv run` or `source .venv/bin/activate`).

**Data download fails**: The data loader caches files in `.data_cache/`. Delete this directory and retry. Check your internet connection.

**Colab Drive mount fails**: Ensure the ascent_data shared folder is accessible. You may need to "Add shortcut to My Drive" from the shared link.
