"""
config.py
=========
Central configuration for the TEKNOFEST 2026 Variant Pathogenicity Classifier.

Everything that is "global state" about the project lives here:
  * reproducibility seed
  * relative paths (Colab-safe: all relative to the repo working directory)
  * raw-file -> source-name mapping
  * column-group conventions (AL_* / EK_* / CAT_* / AA_* / ID / Label)
  * missingness-tier thresholds
  * categorical-encoding strategy decisions (informed by the Phase-1 EDA audit)

This module performs NO heavy computation and has NO third-party dependencies
beyond the standard library, so it is safe to import from any cell/script.

Design notes
------------
* All paths are *relative* (e.g. ``./data/raw``) so the code runs unchanged in
  the Colab working directory.
* ``ensure_dirs()`` lazily creates every output sub-folder, which satisfies the
  "you may create folders under the main ones" constraint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

# --------------------------------------------------------------------------- #
# 0. Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """Seed every RNG we might touch in Phases 0-3 (numpy / random / PYTHONHASHSEED).

    Tree-library seeds (xgboost/lightgbm/catboost) are set at model-construction
    time in later phases; here we only need the data-prep RNGs to be deterministic.
    """
    import random
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# 1. Paths (all relative -> Colab-safe)
# --------------------------------------------------------------------------- #
DATA_DIR = "./data"
RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"

OUTPUTS_DIR = "./outputs"
FOLDS_DIR = "./outputs/folds"
ENCODERS_DIR = "./outputs/encoders"
TE_MAPPINGS_DIR = "./outputs/encoders/target_encoding_mappings"
MODELS_DIR = "./outputs/models"
OOF_DIR = "./outputs/oof_predictions"
# Phase 4-6 artefacts
FEATURE_SELECTION_DIR = "./outputs/feature_selection"   # null-importance / SHAP / final list
CALIBRATORS_DIR = "./outputs/models/calibrators"        # per-base-model Isotonic calibrators
STACKING_DIR = "./outputs/models/stacking"              # meta-learner + stacked OOF
# Phase 7-10 artefacts
PLOTS_DIR = "./outputs/plots"                           # SHAP + calibration report visuals (Phase 8)
FINAL_PIPELINE_DIR = "./outputs/models/final_pipeline"  # packaged, inference-ready bundle (Phase 10)

# Concrete artefact paths -------------------------------------------------- #
CHECKSUMS_PATH = os.path.join(OUTPUTS_DIR, "checksums.txt")
MERGED_PATH = os.path.join(PROCESSED_DIR, "merged_full.parquet")
DEVELOPMENT_PATH = os.path.join(PROCESSED_DIR, "development_set.parquet")
CANARY_PATH = os.path.join(PROCESSED_DIR, "canary_set.parquet")
FOLD_INDICES_PATH = os.path.join(FOLDS_DIR, "fold_indices.json")

# Materialised (Phase-3 output) feature matrices consumed by Phase 4-6.
DEV_FEATURES_PATH = os.path.join(PROCESSED_DIR, "development_features.parquet")
CANARY_FEATURES_PATH = os.path.join(PROCESSED_DIR, "canary_features.parquet")

FEATURE_SCHEMA_PATH = os.path.join(ENCODERS_DIR, "feature_schema.json")
EK_DIRECTION_PATH = os.path.join(ENCODERS_DIR, "ek_direction_map.json")
EK_RANK_THRESHOLDS_PATH = os.path.join(ENCODERS_DIR, "ek_rank_thresholds.json")
FINAL_FEATURE_LIST_PATH = os.path.join(ENCODERS_DIR, "final_feature_list.json")
SELECTED_FEATURE_LIST_PATH = os.path.join(FEATURE_SELECTION_DIR, "selected_features.json")

# Phase 7-10 concrete artefact paths.
THRESHOLDS_PATH = os.path.join(MODELS_DIR, "thresholds.json")        # global + per-source (Phase 7)
EXPERIMENT_LOG_PATH = os.path.join(OUTPUTS_DIR, "experiment_log.csv")  # structured tracking (Phase 10)
PIPELINE_MANIFEST_PATH = os.path.join(FINAL_PIPELINE_DIR, "pipeline_manifest.json")

_ALL_DIRS = [
    DATA_DIR, RAW_DIR, PROCESSED_DIR,
    OUTPUTS_DIR, FOLDS_DIR, ENCODERS_DIR, TE_MAPPINGS_DIR, MODELS_DIR, OOF_DIR,
    FEATURE_SELECTION_DIR, CALIBRATORS_DIR, STACKING_DIR,
    PLOTS_DIR, FINAL_PIPELINE_DIR,
]


def ensure_dirs() -> None:
    """Create every project output directory if it does not already exist."""
    for d in _ALL_DIRS:
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# 2. Raw files & their source tags
# --------------------------------------------------------------------------- #
# The four competition CSVs. The `source` tag is appended as a column on load
# and is used both for stratification (label + source) and for per-dataset
# analysis/threshold tuning in later phases.
RAW_FILES: Dict[str, str] = {
    "master": "YARISMA_TRAIN_MASTER.csv",   # "General" dataset
    "kanser": "YARISMA_TRAIN_KANSER.csv",   # Cancer panel
    "pah":    "YARISMA_TRAIN_PAH.csv",      # Pulmonary Arterial Hypertension panel
    "cftr":   "YARISMA_TRAIN_CFTR.csv",     # CFTR panel
}
SOURCES: List[str] = list(RAW_FILES.keys())

# --------------------------------------------------------------------------- #
# 3. Column conventions
# --------------------------------------------------------------------------- #
ID_COL = "Variant_ID"
LABEL_COL = "Label"
SOURCE_COL = "source"
FINGERPRINT_COL = "variant_fingerprint"
DUP_FLAG_COL = "is_duplicated_across_source"

# Helper / bookkeeping columns that are NEVER fed to a model as features.
NON_FEATURE_COLS = [ID_COL, LABEL_COL, SOURCE_COL, FINGERPRINT_COL, DUP_FLAG_COL]

# Total raw feature columns (verified in EDA): 334 AL + 9 EK + 6 CAT + 2 AA = 351
N_RAW_FEATURES = 351

# Population allele-frequency block.
# EDA confirmed AL_1..AL_26 are continuous, right-skewed allele frequencies in
# [~1e-55, 0.93] (never binary), i.e. per-population AF columns. The remaining
# AL columns are a mix of continuous in-silico scores and binary "damaging" flags.
AF_COLS: List[str] = [f"AL_{i}" for i in range(1, 27)]

# --------------------------------------------------------------------------- #
# 4. Cross-validation / canary configuration
# --------------------------------------------------------------------------- #
N_FOLDS = 5
CANARY_FRACTION = 0.10  # 10% held-out, never touched until final evaluation

# --------------------------------------------------------------------------- #
# 5. Missingness tiers (Phase 2.1 / 2.7)
# --------------------------------------------------------------------------- #
# 3-way ordinal tier used for the `missingness_tier` feature.
MISSINGNESS_TIER_BINS = {
    "dense":   (0, 100),      # 0-100 missing  -> well annotated
    "partial": (101, 300),    # 101-300 missing
    "sparse":  (301, 10_000),  # >=301 missing  -> very sparse
}
MISSINGNESS_TIER_ORDER = {"dense": 0, "partial": 1, "sparse": 2}

# Binary tier flags for the dedicated sub-model heads (Phase 2.7 / 4.6).
SKELETON_MIN_MISSING = 337  # >= 337 missing -> only EK + AA carry signal
DENSE_MAX_MISSING = 20      # <= 20 missing  -> fully annotated

# --------------------------------------------------------------------------- #
# 6. Categorical encoding strategy (Phase 1.3 audit -> Phase 3.2 decisions)
# --------------------------------------------------------------------------- #
# Semantic findings from the Phase-1 EDA (documented per column):
#   CAT_1  : gnomAD-exome population of the max allele frequency (30 categories,
#            e.g. gnomADe_NFE / gnomADg_NFE / gnomADe_SAS ...). -> POPULATION
#            membership => SAFE to target-encode (NOT a pathogenicity proxy).
#   CAT_2  : All-of-Us population (7 categories, AllofUs_EUR/AFR/...). POPULATION
#            membership => SAFE to target-encode.
#   CAT_3  : genotype call (G/G, C/C, T/T, A/A, ./.) -> small cardinality => ONE-HOT.
#   CAT_4  : genotype call (same alphabet as CAT_3)  -> ONE-HOT.
#   CAT_5  : genotype call (same alphabet as CAT_3)  -> ONE-HOT.
#            (The plan *guessed* CAT_5 resembled CAT_1; the data shows it is a
#             genotype, so one-hot is the principled choice. Documented deviation.)
#   CAT_6  : genomic-region mask (segdup / decoy&segdup / lcr) -> 98% missing.
#            This is a *region* flag, NOT a clinical pathogenicity call, but it is
#            sparse and low-cardinality => presence flag + ONE-HOT (kept with care).
#
# Encoding strategy tags consumed by src/encoders.py:
#   "target"  -> leak-proof K-fold target encoding with adaptive noise
#   "onehot"  -> one-hot (small cardinality, safe labels)
#   "drop"    -> exclude entirely
CAT_ENCODING_STRATEGY: Dict[str, str] = {
    "CAT_1": "target",
    "CAT_2": "target",
    "CAT_3": "onehot",
    "CAT_4": "onehot",
    "CAT_5": "onehot",
    "CAT_6": "onehot",
}

# High-cardinality engineered categorical that is also target-encoded (Phase 2.2/3.2).
AA_PAIR_COL = "aa_pair"


# --------------------------------------------------------------------------- #
# 6b. GPU acceleration (forward reference for Phase 4 models)
# --------------------------------------------------------------------------- #
# Phases 0-3 are pure CPU data-prep (no model fitting), but per the project
# constraint every compatible model MUST use GPU. These dicts are the single
# source of truth the Phase-4 training code should spread into each estimator.
def detect_gpu() -> bool:
    """Best-effort CUDA detection (works on Colab GPU runtimes)."""
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def model_gpu_params(use_gpu: bool | None = None) -> Dict[str, dict]:
    """Return GPU-enabled keyword args for XGBoost / LightGBM / CatBoost.

    Falls back to CPU-equivalent settings when no GPU is present, so the same
    call site works on a CPU Colab runtime too.
    """
    use_gpu = detect_gpu() if use_gpu is None else use_gpu
    if use_gpu:
        return {
            # XGBoost >= 2.0 API: hist tree method on CUDA device.
            "xgboost":  {"tree_method": "hist", "device": "cuda", "random_state": SEED},
            "lightgbm": {"device_type": "gpu", "max_bin": 255, "random_state": SEED},
            "catboost": {"task_type": "GPU", "devices": "0", "random_seed": SEED},
        }
    return {
        "xgboost":  {"tree_method": "hist", "device": "cpu", "random_state": SEED},
        "lightgbm": {"device_type": "cpu", "random_state": SEED},
        "catboost": {"task_type": "CPU", "random_seed": SEED},
    }

# Columns handed to CatBoost *raw* (native categorical handling, Phase 4.5).
# Kept here so the modelling phase has a single source of truth.
CATBOOST_RAW_CAT_COLS = list(CAT_ENCODING_STRATEGY.keys()) + ["AA_1", "AA_2", AA_PAIR_COL, "tool_coverage_profile"]


# --------------------------------------------------------------------------- #
# 7. Column-group container
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSchema:
    """Structural description of the raw column space.

    `binary_al` vs `continuous_al` is a *structural* property (does the column
    only ever contain {0,1}?), so computing it from the full development set is
    NOT label leakage. We persist it so val/test reuse the exact same lists.
    """

    al_cols: List[str] = field(default_factory=list)
    ek_cols: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    aa_cols: List[str] = field(default_factory=list)
    af_cols: List[str] = field(default_factory=list)
    binary_al_cols: List[str] = field(default_factory=list)
    continuous_al_cols: List[str] = field(default_factory=list)

    @property
    def feature_cols(self) -> List[str]:
        """Every raw feature column (the 351-column feature space)."""
        return self.al_cols + self.ek_cols + self.cat_cols + self.aa_cols

    def to_dict(self) -> dict:
        return {
            "al_cols": self.al_cols,
            "ek_cols": self.ek_cols,
            "cat_cols": self.cat_cols,
            "aa_cols": self.aa_cols,
            "af_cols": self.af_cols,
            "binary_al_cols": self.binary_al_cols,
            "continuous_al_cols": self.continuous_al_cols,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSchema":
        return cls(**d)
