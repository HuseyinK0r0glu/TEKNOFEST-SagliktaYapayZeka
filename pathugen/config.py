"""Paths, competition identity, and the raw data schema shared by every
`pathugen` module. Nothing here is a model or a result — only the fixed
facts about the competition (team identity, column layout) and where files
live on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

SEED = 42

# Project root — the repo root, i.e. the parent of this `pathugen/` package
# directory, NOT this file's own directory. Every path below is anchored
# here so moving this package around never silently relocates a file.
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Competition identity — VERIFY these against your KYS registration.
# ---------------------------------------------------------------------------
TEAM_NAME = os.environ.get("TEAM_NAME", "PatHUgen")
TEAM_ID = os.environ.get("TEAM_ID", "761197")
APPLICATION_ID = os.environ.get("APPLICATION_ID", "4882800")
COMPETITION_LEVEL = "UNIVERSITE_VE_UZERI"

# ---------------------------------------------------------------------------
# Data schema (identical across all four panels)
# ---------------------------------------------------------------------------
ID_COL, LABEL_COL = "Variant_ID", "Label"
AA_COLS = ["AA_1", "AA_2"]
CAT_COLS = [f"CAT_{i}" for i in range(1, 7)]
EK_COLS = [f"EK_{i}" for i in range(1, 10)]
AL_COLS = [f"AL_{i}" for i in range(1, 335)]
FEATURE_COLS = AL_COLS + CAT_COLS + EK_COLS + AA_COLS
N_FEATURES = len(FEATURE_COLS)
POS = 1
PANELS = ["GENEL", "KANSER", "PAH", "CFTR"]
AUG_SOURCE = "GENEL"

DATA_DIR = Path(os.environ.get(
    "TEKNOFEST_DATA_DIR", "universite-veri-seti/EĞİTİM (TRAIN) SETLERİ"))
TRAIN_FILES = {
    "GENEL": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}
MODEL_DIR = Path(os.environ.get("PATHUGEN_MODEL_DIR", "pathugen_model"))
SEED_ENSEMBLE = 3   # the champion is re-fit with 3 seeds and averaged (locked)
PROB_TOLERANCE = 1e-6
