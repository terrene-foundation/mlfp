# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Unified data loading for ASCENT course — supports local, Jupyter, and Colab."""

from __future__ import annotations

import os
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Google Drive shared folder containing all ASCENT datasets
_DRIVE_FOLDER_ID = "16c3RkGmiwMWbjD7cJKbJx-JRZlgmQdws"

# Module subfolders on the shared Drive
_MODULES = {
    "ascent01",
    "ascent02",
    "ascent03",
    "ascent04",
    "ascent05",
    "ascent06",
    "ascent06-dl",
    "ascent_assessment",
}


def _is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _colab_data_root() -> Path:
    """Return the Drive-mounted ascent_data path in Colab."""
    return Path("/content/drive/MyDrive/ascent_data")


def _local_cache_dir() -> Path:
    """Return local cache directory for downloaded files."""
    cache = Path.cwd() / ".data_cache"
    cache.mkdir(exist_ok=True)
    return cache


def _download_from_drive(module: str, filename: str, dest: Path) -> Path:
    """Download a file from the shared Google Drive using gdown."""
    import gdown

    dest_dir = dest / module
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / filename

    if dest_file.exists():
        logger.debug("Using cached file: %s", dest_file)
        return dest_file

    # gdown can download from a folder by file path
    url = f"https://drive.google.com/drive/folders/{_DRIVE_FOLDER_ID}"
    logger.info("Downloading %s/%s from Google Drive...", module, filename)

    # Download the specific file from the shared folder
    gdown.download_folder(
        url=url,
        output=str(dest),
        quiet=True,
        remaining_ok=True,
    )

    if not dest_file.exists():
        # Try direct download if folder download didn't isolate the file
        # Attempt to find in downloaded structure
        for candidate in dest.rglob(filename):
            if candidate.is_file():
                if candidate != dest_file:
                    candidate.rename(dest_file)
                return dest_file

        msg = (
            f"File not found after download: {module}/{filename}. "
            f"Check that it exists in the ascent_data shared Drive."
        )
        raise FileNotFoundError(msg)

    return dest_file


def _read_file(path: Path) -> pl.DataFrame:
    """Read a data file into a polars DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    elif suffix == ".parquet":
        return pl.read_parquet(path)
    elif suffix == ".json":
        return pl.read_json(path)
    elif suffix in (".p", ".pickle", ".pkl"):
        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if isinstance(obj, pl.DataFrame):
            return obj
        raise TypeError(
            f"Cannot convert pickle object of type {type(obj)} to polars DataFrame. "
            f"Convert the pickle to parquet upstream: pl.from_pandas(obj).write_parquet('out.parquet')"
        )
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .csv, .parquet, or .json"
        )


def _repo_data_dir() -> Path | None:
    """Find the repo-local data/ directory by walking up from cwd."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / "data"
        if candidate.is_dir() and (parent / "pyproject.toml").exists():
            return candidate
    return None


class ASCENTDataLoader:
    """Load ASCENT course datasets with automatic source resolution.

    Resolution order:
    1. Colab: Drive mount at /content/drive/MyDrive/ascent_data/
    2. Local repo data/ directory (committed datasets)
    3. Google Drive download via gdown (cached in .data_cache/)

    Usage:
        loader = ASCENTDataLoader()
        df = loader.load("ascent01", "hdbprices.csv")

    Shortcut:
        df = ASCENTDataLoader.ascent01("hdbprices.csv")
    """

    def __init__(self, cache_dir: Path | str | None = None):
        self._colab = _is_colab()
        if self._colab:
            self._root = _colab_data_root()
        else:
            self._local_data = _repo_data_dir()
            self._cache = Path(cache_dir) if cache_dir else _local_cache_dir()

    def load(self, module: str, filename: str) -> pl.DataFrame:
        """Load a dataset file as a polars DataFrame.

        Args:
            module: Module subfolder (e.g., "ascent01", "ascent_assessment")
            filename: File name within the module folder (e.g., "hdbprices.csv")

        Returns:
            polars DataFrame with the loaded data.
        """
        if module not in _MODULES:
            raise ValueError(
                f"Unknown module '{module}'. Available: {sorted(_MODULES)}"
            )

        if self._colab:
            path = self._root / module / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found: {path}. "
                    f"Ensure ascent_data is accessible in your Google Drive."
                )
        else:
            # Check repo-local data/ first, then fall back to Drive download
            if self._local_data:
                local_path = self._local_data / module / filename
                if local_path.exists():
                    path = local_path
                    logger.info(
                        "Loading %s/%s from local data/ (%s)", module, filename, path
                    )
                    return _read_file(path)
            path = _download_from_drive(module, filename, self._cache)

        logger.info("Loading %s/%s (%s)", module, filename, path)
        return _read_file(path)

    def list_files(self, module: str) -> list[str]:
        """List available data files in a module folder."""
        if module not in _MODULES:
            raise ValueError(
                f"Unknown module '{module}'. Available: {sorted(_MODULES)}"
            )

        if self._colab:
            root = self._root / module
        else:
            root = self._cache / module

        if not root.exists():
            return []

        return sorted(f.name for f in root.iterdir() if f.is_file())

    # ── Module shortcuts ──

    @classmethod
    def ascent01(cls, filename: str) -> pl.DataFrame:
        """Load from ascent01 (Python, Polars & Visualization)."""
        return cls().load("ascent01", filename)

    @classmethod
    def ascent02(cls, filename: str) -> pl.DataFrame:
        """Load from ascent02 (Statistics & Feature Engineering)."""
        return cls().load("ascent02", filename)

    @classmethod
    def ascent03(cls, filename: str) -> pl.DataFrame:
        """Load from ascent03 (Inferential Stats & Workflows)."""
        return cls().load("ascent03", filename)

    @classmethod
    def ascent04(cls, filename: str) -> pl.DataFrame:
        """Load from ascent04 (Supervised ML & Production Lifecycle)."""
        return cls().load("ascent04", filename)

    @classmethod
    def ascent05(cls, filename: str) -> pl.DataFrame:
        """Load from ascent05 (Unsupervised ML, Deep Learning & AI Agents)."""
        return cls().load("ascent05", filename)

    @classmethod
    def ascent06(cls, filename: str) -> pl.DataFrame:
        """Load from ascent06 (Deep Learning & Agents)."""
        return cls().load("ascent06", filename)

    @classmethod
    def assessment(cls, filename: str) -> pl.DataFrame:
        """Load from ascent_assessment (capstone datasets)."""
        return cls().load("ascent_assessment", filename)
