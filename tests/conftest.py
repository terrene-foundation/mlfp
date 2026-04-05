# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT test configuration — auto-loads .env for all tests."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
