"""The single source of truth for "where is the project root" inside
`search/`, regardless of how deep a given module sits under this package.
Every module that needs to anchor a path to the repo root (logs/, the
outputs_* comparison directories, ...) imports PROJECT_ROOT from here
instead of computing its own `Path(__file__).resolve().parent...` — that
pattern silently breaks the moment a file moves to a different depth.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
