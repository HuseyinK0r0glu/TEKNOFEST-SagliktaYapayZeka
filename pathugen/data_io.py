"""Loading the raw panel CSVs (training and test)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pathugen.config import (
    AA_COLS, AL_COLS, CAT_COLS, DATA_DIR, EK_COLS, ID_COL, TRAIN_FILES,
)


def load_panel(panel: str) -> pd.DataFrame:
    """Load a TRAINING panel CSV (used only by `train`)."""
    df = pd.read_csv(DATA_DIR / TRAIN_FILES[panel])
    expected = 1 + len(AL_COLS) + len(CAT_COLS) + len(EK_COLS) + len(AA_COLS) + 1
    if df.shape[1] != expected:
        raise ValueError(f"{panel}: expected {expected} training columns, got {df.shape[1]}")
    return df


def load_test_csv(path: Path) -> pd.DataFrame:
    """Load a TEST CSV for inference. Tolerates a missing Label column (test
    data normally has no ground truth) but otherwise validates the schema."""
    df = pd.read_csv(path)
    with_label = 1 + len(AL_COLS) + len(CAT_COLS) + len(EK_COLS) + len(AA_COLS) + 1
    without_label = with_label - 1
    if df.shape[1] not in (with_label, without_label):
        raise ValueError(
            f"{path}: expected {without_label} columns (no Label) or "
            f"{with_label} (with Label), got {df.shape[1]}")
    if ID_COL not in df.columns:
        raise ValueError(f"{path}: missing required '{ID_COL}' column")
    return df
