"""Central configuration: paths, panel definitions, column groups, seed.

Architecture directive (governs the whole project):
  4 fully INDEPENDENT per-panel models (GENEL/KANSER/PAH/CFTR).
  Panels are never pooled for training/testing, and no cross-panel
  dedup / GroupKFold is performed. Metric = binary F1 on the
  Pathogenic (positive) class, with per-panel threshold tuning.
"""
from __future__ import annotations

from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "universite-veri-seti" / "EĞİTİM (TRAIN) SETLERİ"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"

# Per-panel CSV files. Keys are the canonical panel names used everywhere.
PANEL_FILES = {
    "GENEL": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}
PANELS = list(PANEL_FILES.keys())

# Column groups (verified against the shared 353-column header).
ID_COL = "Variant_ID"
LABEL_COL = "Label"
AA_COLS = ["AA_1", "AA_2"]
CAT_COLS = [f"CAT_{i}" for i in range(1, 7)]
EK_COLS = [f"EK_{i}" for i in range(1, 10)]
AL_COLS = [f"AL_{i}" for i in range(1, 335)]

# 351 model-eligible feature columns (everything except ID + Label).
FEATURE_COLS = AL_COLS + CAT_COLS + EK_COLS + AA_COLS
N_FEATURES = len(FEATURE_COLS)  # 351

# Pathogenic is the positive class; F1 is scored on it.
POSITIVE_CLASS = 1

# Leakage-audit threshold: a CAT category that is >= this fraction one class
# is flagged as a possible proxy label.
CAT_PURITY_FLAG = 0.95
