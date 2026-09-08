"""Knobs and run settings for the reference pipeline (tune speed vs rigor).
QUICK ≈ 15-30 min; full ≈ 60-120 min.

The raw data schema (column names, panel list, ID_COL/LABEL_COL) is shared
with `pathugen.config` since it never differs between the two — only the
run-tuning knobs below are specific to this search pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

from pathugen.config import (
    AA_COLS, AL_COLS, AUG_SOURCE, CAT_COLS, DATA_DIR, EK_COLS, FEATURE_COLS,
    ID_COL, LABEL_COL, N_FEATURES, PANELS, POS, SEED, TRAIN_FILES as PANEL_FILES,
)

import numpy as np

np.random.seed(SEED)

OUTPUT_DIR = Path(os.environ.get("TEKNOFEST_OUT_DIR", "outputs_colab"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUICK = os.environ.get("QUICK", "1").strip().lower() in ("1", "true", "yes")
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "15" if QUICK else "45"))
OPTUNA_TIMEOUT = int(os.environ.get("OPTUNA_TIMEOUT", "120" if QUICK else "600"))
CV_REPEATS = int(os.environ.get("CV_REPEATS", "2" if QUICK else "3"))
NESTED_REPEATS = int(os.environ.get("NESTED_REPEATS", "1" if QUICK else "2"))
NESTED_SPLITS = int(os.environ.get("NESTED_SPLITS", "5"))
RUN_NESTED = os.environ.get("RUN_NESTED", "1").strip().lower() in ("1", "true", "yes")
# Optuna objective + champion-selection metric: AUC is balance-independent, so it
# does not overfit the (unknown) competition class prior the way F1 would.
OBJECTIVE = os.environ.get("OPTUNA_OBJECTIVE", "auc").strip().lower()
SELECT_METRIC = os.environ.get("SELECT_METRIC", "auc").strip().lower()
SEED_JITTER = not QUICK            # per-trial fold seed → winners must generalise
SPW_MULT_HI = float(os.environ.get("SPW_MULT_HI", "6.0"))

# GENEL→panel augmentation: PAH+CFTR gain from borrowed GENEL rows; KANSER loses
# (cancer signal too specific), GENEL is the source. Held-out-validated choice.
AUGMENT_PANELS = tuple(
    p.strip().upper() for p in os.environ.get("AUGMENT_PANELS", "PAH,CFTR").split(",")
    if p.strip()
)
# GPU for the boosters (off by default — CPU is faster on tabular data this size).
USE_GPU = os.environ.get("USE_GPU", "0").strip().lower() in ("1", "true", "yes")

print(f"DATA_DIR = {DATA_DIR}")
print(f"QUICK={QUICK}  OPTUNA_TRIALS={OPTUNA_TRIALS}  TIMEOUT={OPTUNA_TIMEOUT}s  "
      f"CV_REPEATS={CV_REPEATS}  NESTED={RUN_NESTED}(x{NESTED_REPEATS})")
print(f"OBJECTIVE={OBJECTIVE}  SELECT_METRIC={SELECT_METRIC}  "
      f"AUGMENT_PANELS={AUGMENT_PANELS}  USE_GPU={USE_GPU}")
