#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PatHUgen — TEKNOFEST 2026 Health AI Competition, Final Delivery System.

Team: PatHUgen  |  Task: missense variant pathogenicity classification
Panels (always independent): GENEL, KANSER, PAH, CFTR

WHAT THIS FILE IS
------------------
This is the FINAL, LOCKED pipeline. For each of the four panels, an
automated search compared many combinations of feature set, model family,
and decision-threshold strategy, validated every candidate with leak-free
nested cross-validation, and screened out any candidate that showed signs
of overfitting (a large gap between its optimistic and honest scores, an
unstable choice of "best" model across folds, or failure to beat a naive
baseline). The configuration that survived that process for each panel is
hard-coded below in PANEL_CONFIG.

This file hard-codes THAT answer. There is no hyperparameter search here, no
model selection, no sweeping — those decisions are closed. What is left is
exactly what a competition-day delivery needs:

    1. TRAIN  — fit the locked configuration from the raw training CSVs
                (run this once, ahead of time, not during the timed test).
    2. DELIVER — given the organization's test CSVs, produce predictions,
                validate them against the official JSON schema, and package
                a complete, reproducible delivery folder (code + model +
                requirements + README) — this is the command run on
                competition day, once per official/practice submission.

Every panel's winning configuration turned out to be a single tuned XGBoost
model — no averaging or stacking of multiple models beat a single well-tuned
one on any panel once overfitting was accounted for — which is why this file
only needs XGBoost, not a larger zoo of model families.

JSON OUTPUT FORMAT
-------------------
Per `TEKNOFEST_2026_Saglikta_Yapay_Zeka_Final_Uygulama_ve_Sonuc_Teslim_Kilavuzu.pdf`,
section 3.2 ("Üniversite ve Üzeri Seviyesi JSON Formatı"):

    {
      "team_name": "...", "team_id": "...", "application_id": "...",
      "competition_level": "UNIVERSITE_VE_UZERI",
      "predictions": [
        {"id": "VAR_000001", "panel": "CFTR",
         "predicted_class": "1", "predicted_prob": 0.87}
      ]
    }

predicted_class is the string "0" or "1" (the team's final decision, using
the panel's audited threshold). predicted_prob is the model's probability of
the Pathogenic (1) class, used by the organizers for AUROC-style metrics.
This file's `predicted_class` and `predicted_prob` fields follow section 3.2's
worked example exactly; if the official rules are amended before the
competition, re-check this against the latest guide.

IMPORTANT — VERIFY BEFORE THE COMPETITION
-------------------------------------------
TEAM_NAME / TEAM_ID / APPLICATION_ID below are filled from this project's
existing paperwork (PSR report cover). Confirm they match your KYS
(başvuru sistemi) registration exactly before your first real submission —
override with environment variables if anything changed:

    TEAM_NAME=... TEAM_ID=... APPLICATION_ID=... python PatHUgen.py deliver ...

Usage
-----
    # One-time, ahead of the competition (needs the training CSVs):
    python PatHUgen.py train

    # Every time you have test data to score (competition day):
    python PatHUgen.py deliver --input-dir /path/to/test/csvs

    # See HOW_TO_RUN.md for the full walkthrough.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent

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

# ---------------------------------------------------------------------------
# LOCKED CONFIGURATION — the audited best configuration found for each panel.
#
# Every panel's champion is a single tuned XGBoost model. `features` is the
# exact feature list the champion was trained on (order does not matter —
# XGBClassifier is fit on a DataFrame and reads columns by name). `xgb_params`
# are the tuned hyperparameters for that panel, found by automated search and
# validated with nested cross-validation. `threshold` is the decision
# threshold chosen under the `threshold_policy` that survived the overfitting
# audit — this is not the naive "F1-at-training-prior" threshold everywhere;
# GENEL/CFTR use `balanced_f1`, KANSER uses `f1`, PAH uses `mcc`, each chosen
# because it was the one that passed the audit with the best
# robustness score for that specific panel).
# ---------------------------------------------------------------------------
PANEL_CONFIG = {
    "GENEL": {
        "selection_method": "hybrid",       # informational: how features were chosen
        "aug_frac": 0.0,                    # GENEL is the augmentation SOURCE, never a target
        "threshold": 0.670,
        "threshold_policy": "balanced_f1",
        "xgb_params": {
            "max_depth": 8, "learning_rate": 0.057661695328242606,
            "subsample": 0.7457265639467195, "colsample_bytree": 0.48637982372057026,
            "min_child_weight": 4, "gamma": 3.147417096423517,
            "reg_alpha": 0.00036390356638230445, "reg_lambda": 0.010569715896517848,
            "spw_mult": 2.900406586338892,
        },
        "features": [
            "missingtier_x_ekcount", "AL_1", "CAT_6", "EK_7", "AL_16", "n_missing_al",
            "AL_287", "group_change", "AL_238", "AL_59", "AL_68", "weight_1", "AL_215",
            "blosum62", "ek_mean", "AL_328", "blosum80", "turn_prop_1", "ek_max", "AL_48",
            "n_missing_blocks", "pct_missing", "hydropathy_1", "AL_54", "AL_331",
            "n_tools_available", "absd_charge", "AL_315", "absd_pi", "AL_319",
            "al_cont_count", "AL_214", "grantham", "AA_1_code", "AA_2_code", "weight_2",
            "al_log_std", "al_n_ultrarare",
        ],
    },
    "KANSER": {
        "selection_method": "null",
        "aug_frac": 0.0,                    # augmentation measurably HURTS this panel — do not use
        "threshold": 0.705,
        "threshold_policy": "f1",
        "xgb_params": {
            "max_depth": 3, "learning_rate": 0.08391138248483593,
            "subsample": 0.7162938728905041, "colsample_bytree": 0.8998265715443905,
            "min_child_weight": 2, "gamma": 1.733526351356526,
            "reg_alpha": 0.00023288337152878503, "reg_lambda": 1.3535629313364773e-05,
            "spw_mult": 5.176510952209122,
        },
        "features": [
            "AL_215", "missingtier_x_ekcount", "AL_184", "AL_16", "ek_mean", "AL_3",
            "AL_224", "AL_201", "AL_24", "AL_68", "AL_13", "AL_322", "AL_19",
            "damaging_vote_count", "AL_315", "AL_302", "AL_34", "n_missing_total",
            "ek_max", "charge_1", "pct_missing", "helix_prop_1", "AL_26", "AL_9", "AL_79",
            "ek_max_x_grantham", "AL_313", "AL_6", "polarity_1", "ek_std", "AL_10",
            "AL_329", "AL_304", "AL_32", "AL_33",
        ],
    },
    "PAH": {
        "selection_method": "null",
        "aug_frac": 1.0,                    # full GENEL augmentation — lifts this panel
        "threshold": 0.595,
        "threshold_policy": "mcc",
        "xgb_params": {
            "max_depth": 3, "learning_rate": 0.07855195505958872,
            "subsample": 0.5692243649063353, "colsample_bytree": 0.5660656657501308,
            "min_child_weight": 1, "gamma": 1.1094698623215902,
            "reg_alpha": 0.1755020126159336, "reg_lambda": 1.1749644517752007,
            "spw_mult": 5.797385180525024,
        },
        "features": [
            "AL_246", "votes_x_grantham", "AL_171", "AL_49", "AL_72", "AL_328", "AL_283",
            "al_log_mean", "al_log_std", "AL_322", "AL_40", "EK_7", "AL_7", "AL_87",
            "AL_26", "AL_241", "AL_96", "AL_300", "AL_323", "blosum80", "AL_23", "AL_58",
            "AL_277", "turn_prop_2", "AL_48", "AL_331", "al_cont_mean", "AL_43", "AL_145",
            "AL_172", "AL_326", "al_n_ultrarare", "AL_8", "volume_2", "AL_53", "AL_89",
            "AL_286",
        ],
    },
    "CFTR": {
        "selection_method": "null",
        "aug_frac": 1.0,                    # full GENEL augmentation — lifts this panel most
        "threshold": 0.780,
        "threshold_policy": "balanced_f1",
        "xgb_params": {
            "max_depth": 3, "learning_rate": 0.03799624962086076,
            "subsample": 0.7028592783833295, "colsample_bytree": 0.3631562272733966,
            "min_child_weight": 9, "gamma": 4.105252984074675,
            "reg_alpha": 0.00047643128521631495, "reg_lambda": 3.025390723188331e-07,
            "spw_mult": 4.307433607890479,
        },
        "features": [
            "al_n_ultrarare", "AL_7", "ek_max", "hydropathy_2", "al_cont_mean", "EK_2",
            "AL_327", "AL_298", "ek_std", "absd_pi", "AL_326", "AL_126", "AL_282",
        ],
    },
}

# ===========================================================================
# Amino-acid lookup tables (BLOSUM / Grantham / physicochemical scales).
# Fixed, published reference tables (Henikoff & Henikoff 1992; Grantham 1974)
# used to score how biochemically severe each variant's amino-acid change is.
# Hardcoded so this file has no external data-file dependency at inference
# time (the competition environment has no internet access).
# ===========================================================================
"""Amino-acid lookup tables (BLOSUM, Grantham, physicochemical scales)."""
_B62_ORDER = list("ARNDCQEGHILKMFPSTWYV")
_B62_ROWS = [
    [4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
    [-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
    [-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
    [0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
    [-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
    [0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
    [-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
    [-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
    [1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
    [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
    [0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
]
BLOSUM62 = {a: {b: _B62_ROWS[i][j] for j, b in enumerate(_B62_ORDER)}
            for i, a in enumerate(_B62_ORDER)}


def _load_blosum(name, fallback):
    try:
        from Bio.Align import substitution_matrices
        m = substitution_matrices.load(name)
        return {a: {b: float(m[a, b]) for b in _B62_ORDER} for a in _B62_ORDER}
    except Exception:
        return fallback


BLOSUM45 = _load_blosum("BLOSUM45", BLOSUM62)
BLOSUM80 = _load_blosum("BLOSUM80", BLOSUM62)

_GRANTHAM_PROPS = {
    "S":(1.42,9.2,32),"R":(0.65,10.5,124),"L":(0.0,4.9,111),"P":(0.39,8.0,32.5),
    "T":(0.71,8.6,61),"A":(0.0,8.1,31),"V":(0.0,5.9,84),"G":(0.74,9.0,3),
    "I":(0.0,5.2,111),"F":(0.0,5.2,132),"Y":(0.20,6.2,136),"C":(2.75,5.5,55),
    "H":(0.58,10.4,96),"Q":(0.89,10.5,85),"N":(1.33,11.6,56),"K":(0.33,11.3,119),
    "D":(1.38,13.0,54),"E":(0.92,12.3,83),"M":(0.0,5.7,105),"W":(0.13,5.4,170),
}
_GA, _GB, _GG, _GRHO = 1.833, 0.1018, 0.000399, 50.723

PHYS = {
    "hydropathy": {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
        "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,
        "S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2},
    "weight": {"A":89,"R":174,"N":132,"D":133,"C":121,"Q":146,"E":147,"G":75,
        "H":155,"I":131,"L":131,"K":146,"M":149,"F":165,"P":115,"S":105,"T":119,
        "W":204,"Y":181,"V":117},
    "volume": {"A":31,"R":124,"N":56,"D":54,"C":55,"Q":85,"E":83,"G":3,"H":96,
        "I":111,"L":111,"K":119,"M":105,"F":132,"P":32.5,"S":32,"T":61,"W":170,
        "Y":136,"V":84},
    "polarity": {"A":8.1,"R":10.5,"N":11.6,"D":13.0,"C":5.5,"Q":10.5,"E":12.3,
        "G":9.0,"H":10.4,"I":5.2,"L":4.9,"K":11.3,"M":5.7,"F":5.2,"P":8.0,"S":9.2,
        "T":8.6,"W":5.4,"Y":6.2,"V":5.9},
    "pi": {"A":6.0,"R":10.76,"N":5.41,"D":2.77,"C":5.07,"Q":5.65,"E":3.22,"G":5.97,
        "H":7.59,"I":6.02,"L":5.98,"K":9.74,"M":5.74,"F":5.48,"P":6.30,"S":5.68,
        "T":5.60,"W":5.89,"Y":5.66,"V":5.96},
    "charge": {"D":-1.0,"E":-1.0,"K":1.0,"R":1.0,"H":0.1,"A":0.0,"N":0.0,"C":0.0,
        "Q":0.0,"G":0.0,"I":0.0,"L":0.0,"M":0.0,"F":0.0,"P":0.0,"S":0.0,"T":0.0,
        "W":0.0,"Y":0.0,"V":0.0},
    "helix_prop": {"A":1.42,"R":0.98,"N":0.67,"D":1.01,"C":0.70,"Q":1.11,"E":1.51,
        "G":0.57,"H":1.00,"I":1.08,"L":1.21,"K":1.16,"M":1.45,"F":1.13,"P":0.57,
        "S":0.77,"T":0.83,"W":1.08,"Y":0.69,"V":1.06},
    "sheet_prop": {"A":0.83,"R":0.93,"N":0.89,"D":0.54,"C":1.19,"Q":1.10,"E":0.37,
        "G":0.75,"H":0.87,"I":1.60,"L":1.30,"K":0.74,"M":1.05,"F":1.38,"P":0.55,
        "S":0.75,"T":1.19,"W":1.37,"Y":1.47,"V":1.70},
    "turn_prop": {"A":0.66,"R":0.95,"N":1.56,"D":1.46,"C":1.19,"Q":0.98,"E":0.74,
        "G":1.56,"H":0.95,"I":0.47,"L":0.59,"K":1.01,"M":0.60,"F":0.60,"P":1.52,
        "S":1.43,"T":0.96,"W":0.96,"Y":1.14,"V":0.50},
}
GROUP = {"A":"nonpolar","V":"nonpolar","L":"nonpolar","I":"nonpolar","M":"nonpolar",
    "P":"nonpolar","G":"nonpolar","F":"aromatic","W":"aromatic","Y":"aromatic",
    "C":"polar","S":"polar","T":"polar","N":"polar","Q":"polar","H":"positive",
    "K":"positive","R":"positive","D":"negative","E":"negative"}


def _blosum_lookup(matrix, a, b):
    if isinstance(a, str) and isinstance(b, str) and a in matrix and b in matrix[a]:
        return float(matrix[a][b])
    return np.nan


def blosum62(a, b): return _blosum_lookup(BLOSUM62, a, b)
def blosum45(a, b): return _blosum_lookup(BLOSUM45, a, b)
def blosum80(a, b): return _blosum_lookup(BLOSUM80, a, b)


def grantham(a, b):
    if a not in _GRANTHAM_PROPS or b not in _GRANTHAM_PROPS:
        return np.nan
    ca, pa, va = _GRANTHAM_PROPS[a]
    cb, pb, vb = _GRANTHAM_PROPS[b]
    d = _GA*(ca-cb)**2 + _GB*(pa-pb)**2 + _GG*(va-vb)**2
    return float(_GRHO * np.sqrt(d))


def group_change(a, b):
    ga, gb = GROUP.get(a), GROUP.get(b)
    return 0 if ga is None or gb is None else int(ga != gb)
# ===========================================================================
# Data loading
# ===========================================================================
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

# ===========================================================================
# Feature engineering (build_features) — stateless, label-independent.
# Runs identically at train and predict time, so there is no risk of the two
# ever drifting out of sync.
# ===========================================================================
STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _binary_al_columns(df):
    out = []
    for col in AL_COLS:
        vals = df[col].dropna().to_numpy()
        if vals.size and set(np.unique(vals)).issubset({0.0, 1.0}):
            out.append(col)
    return out


def _encode_categorical(series, mapping=None):
    if mapping is None:
        cat = series.astype("category")
        mapping = {v: i + 1 for i, v in enumerate(cat.cat.categories)}
        return (cat.cat.codes + 1).astype(int), mapping
    return series.map(mapping).fillna(0).astype(int), mapping


def get_cat_maps(df):
    maps = {}
    for col in CAT_COLS:
        _, maps[col] = _encode_categorical(df[col])
    _, maps["AA_1_code"] = _encode_categorical(df["AA_1"].astype("string"))
    _, maps["AA_2_code"] = _encode_categorical(df["AA_2"].astype("string"))
    return maps


def build_features(df, cat_maps=None):
    """Stateless, label-independent per-row transform → (X, y, cat_cols)."""
    y = df[LABEL_COL].astype(int) if LABEL_COL in df.columns else None
    a1 = df["AA_1"].astype("string")
    a2 = df["AA_2"].astype("string")
    feats = {}

    n_missing_total = df[FEATURE_COLS].isnull().sum(axis=1)
    feats["n_missing_total"] = n_missing_total
    feats["pct_missing"] = n_missing_total / N_FEATURES
    feats["n_missing_al"] = df[AL_COLS].isnull().sum(axis=1)
    feats["n_missing_ek"] = df[EK_COLS].isnull().sum(axis=1)
    feats["n_missing_cat"] = df[CAT_COLS].isnull().sum(axis=1)
    feats["is_skeleton"] = (n_missing_total >= 301).astype(int)
    feats["is_dense"] = (n_missing_total <= 20).astype(int)
    feats["missingness_tier"] = pd.cut(
        n_missing_total, bins=[-1, 20, 100, 200, 300, 351], labels=[0,1,2,3,4]).astype(int)
    feats["all_ek_missing"] = df[EK_COLS].isnull().all(axis=1).astype(int)
    feats["population_block_missing"] = df[["CAT_1","CAT_2"]].isnull().all(axis=1).astype(int)
    feats["genotype_block_missing"] = df[["CAT_3","CAT_4","CAT_5"]].isnull().all(axis=1).astype(int)
    feats["n_missing_blocks"] = (feats["all_ek_missing"] + feats["population_block_missing"]
                                 + feats["genotype_block_missing"])

    feats["blosum62"] = np.array([blosum62(x, z) for x, z in zip(a1, a2)], dtype=float)
    feats["blosum45"] = np.array([blosum45(x, z) for x, z in zip(a1, a2)], dtype=float)
    feats["blosum80"] = np.array([blosum80(x, z) for x, z in zip(a1, a2)], dtype=float)
    feats["grantham"] = np.array([grantham(x, z) for x, z in zip(a1, a2)], dtype=float)
    for name, table in PHYS.items():
        v1 = a1.map(lambda x: table.get(x, np.nan)).astype(float)
        v2 = a2.map(lambda x: table.get(x, np.nan)).astype(float)
        feats[f"{name}_1"], feats[f"{name}_2"] = v1, v2
        feats[f"d_{name}"], feats[f"absd_{name}"] = v2 - v1, (v2 - v1).abs()
    feats["is_stop_gain"] = (a2 == "*").fillna(False).astype(int)
    feats["is_stop_loss"] = (a1 == "*").fillna(False).astype(int)
    feats["aa_missing"] = (a1.isna() | a2.isna()).astype(int)
    feats["aa1_nonstd"] = ((~a1.isin(STD_AA)) & a1.notna()).astype(int)
    feats["aa2_nonstd"] = ((~a2.isin(STD_AA | {"*"})) & a2.notna()).astype(int)
    feats["group_change"] = np.array([group_change(x, z) for x, z in zip(a1, a2)], dtype=int)

    bin_cols = _binary_al_columns(df)
    if bin_cols:
        bdf = df[bin_cols]
        n_avail = bdf.notna().sum(axis=1)
        dmg = (bdf == 1).sum(axis=1)
        feats["n_tools_available"] = n_avail
        feats["damaging_vote_count"] = dmg
        feats["damaging_vote_ratio"] = dmg / n_avail.replace(0, np.nan)
        feats["benign_vote_count"] = (bdf == 0).sum(axis=1)
        feats["unanimous_damaging"] = ((n_avail > 0) & (dmg == n_avail)).astype(int)
        feats["unanimous_benign"] = ((n_avail > 0) & (dmg == 0)).astype(int)
        feats["conflicting_votes"] = ((dmg > 0) & ((n_avail - dmg) > 0)).astype(int)

    cont_cols = [c for c in AL_COLS if c not in set(bin_cols)]
    cdf = df[cont_cols]
    feats["al_cont_count"] = cdf.notna().sum(axis=1)
    feats["al_cont_mean"] = cdf.mean(axis=1)
    feats["al_cont_max"] = cdf.max(axis=1)
    feats["al_cont_min"] = cdf.min(axis=1)
    feats["al_cont_std"] = cdf.std(axis=1)
    feats["al_n_rare"] = ((cdf > 0) & (cdf < 1e-3)).sum(axis=1)
    log_cdf = np.log1p(cdf.clip(lower=0))
    feats["al_log_mean"] = log_cdf.mean(axis=1)
    feats["al_log_max"] = log_cdf.max(axis=1)
    feats["al_log_min"] = log_cdf.min(axis=1)
    feats["al_log_std"] = log_cdf.std(axis=1)
    min_pos = cdf.where(cdf > 0).min(axis=1)
    feats["al_neglog10_min_pos"] = -np.log10(min_pos.clip(lower=1e-12))
    feats["al_n_ultrarare"] = ((cdf > 0) & (cdf < 1e-4)).sum(axis=1)
    n_cont = cdf.notna().sum(axis=1)
    feats["al_log_range"] = log_cdf.max(axis=1) - log_cdf.min(axis=1)
    feats["al_log_iqr"] = (log_cdf.quantile(0.75, axis=1) - log_cdf.quantile(0.25, axis=1))
    row_logmed = log_cdf.median(axis=1)
    feats["al_frac_above_logmed"] = (log_cdf.gt(row_logmed, axis=0).sum(axis=1)
                                     / n_cont.replace(0, np.nan))

    edf = df[EK_COLS]
    feats["ek_count"] = edf.notna().sum(axis=1)
    feats["ek_mean"] = edf.mean(axis=1)
    feats["ek_max"] = edf.max(axis=1)
    feats["ek_min"] = edf.min(axis=1)
    feats["ek_std"] = edf.std(axis=1)

    eng = pd.DataFrame(feats, index=df.index)
    eng["ek_max_x_grantham"] = eng["ek_max"] * eng["grantham"]
    eng["ek_mean_x_completeness"] = eng["ek_mean"] * (1 - eng["pct_missing"])
    eng["al_logmax_x_ek_max"] = eng["al_log_max"] * eng["ek_max"]
    eng["missingtier_x_ekcount"] = eng["missingness_tier"] * eng["ek_count"]
    if "damaging_vote_ratio" in eng.columns:
        dvr = eng["damaging_vote_ratio"].fillna(0)
        eng["votes_x_grantham"] = dvr * eng["grantham"]
        eng["votes_x_ek_mean"] = dvr * eng["ek_mean"]

    maps = cat_maps or {}
    cat_data, cat_cols = {}, []
    for col in CAT_COLS:
        cat_data[col], _ = _encode_categorical(df[col], maps.get(col))
        cat_cols.append(col)
    cat_data["AA_1_code"], _ = _encode_categorical(a1, maps.get("AA_1_code"))
    cat_data["AA_2_code"], _ = _encode_categorical(a2, maps.get("AA_2_code"))
    cat_cols += ["AA_1_code", "AA_2_code"]
    cat_df = pd.DataFrame(cat_data, index=df.index)

    raw = df[AL_COLS + EK_COLS].copy()
    X = pd.concat([eng, cat_df, raw], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X, y, cat_cols

# ==============================================================================
# GENEL -> PAH/CFTR training-only augmentation pool (leak-free: rows NOT in the
# target panel, used only in training folds; validation/test always stays on
# the target panel's own rows).
# ==============================================================================
def build_aug_pool(panel: str, columns=None):
    """GENEL rows NOT present in `panel`, engineered with `panel`'s cat_maps."""
    cfg = PANEL_CONFIG[panel]
    if cfg["aug_frac"] <= 0.0 or panel == AUG_SOURCE:
        return None, None
    panel_df = load_panel(panel)
    source_df = load_panel(AUG_SOURCE)
    pool = source_df[~source_df[ID_COL].isin(set(panel_df[ID_COL]))].copy()
    if pool.empty:
        return None, None
    cat_maps = get_cat_maps(panel_df)
    X_aug, y_aug, _ = build_features(pool, cat_maps=cat_maps)
    if columns is not None:
        X_aug = X_aug.reindex(columns=list(columns))
    return X_aug.reset_index(drop=True), pd.Series(np.asarray(y_aug)).reset_index(drop=True)


# ==============================================================================
# XGBoost — the only model family needed (every panel's audited champion).
# ==============================================================================
def scale_pos_weight(y_tr) -> float:
    y_tr = np.asarray(y_tr)
    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    return n_neg / n_pos if n_pos else 1.0


def make_xgb(params: dict, y_tr, seed: int) -> XGBClassifier:
    p = dict(params)
    spw = p.pop("spw_mult", 1.0)
    return XGBClassifier(
        n_estimators=p.pop("n_estimators", 400), tree_method="hist",
        eval_metric="logloss", scale_pos_weight=scale_pos_weight(y_tr) * spw,
        random_state=seed, n_jobs=-1, **p)


def fit_seed_ensemble(X, y, params: dict, n_seeds: int = SEED_ENSEMBLE):
    """Fit the same tuned XGBoost with `n_seeds` different random seeds and
    average their predictions. This is part of the locked, audited
    configuration for every panel — averaging a few seeds costs almost
    nothing and reduces variance from a single lucky/unlucky tree-building
    draw, which matters most on the smallest panels (CFTR: 111 rows).
    """
    models = []
    for i in range(n_seeds):
        m = make_xgb(params, y, seed=SEED + i)
        m.fit(X, np.asarray(y))
        models.append(m)
    return models


def predict_proba_ensemble(models, X) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


# ==============================================================================
# TRAIN — reproduce the locked champion for every panel from raw CSVs.
# ==============================================================================
# ==============================================================================
# Model versioning — every `train` run creates a new numbered snapshot
# (v1, v2, v3, ...) instead of overwriting the previous one, exactly like
# `deliver` never overwrites a previous delivery{N}/ folder. Nothing you have
# already trained is ever destroyed by training again.
# ==============================================================================
_VERSION_RE = re.compile(r"v(\d+)")


def _panel_versions(panel: str, model_dir: Path) -> list[int]:
    panel_dir = model_dir / panel
    if not panel_dir.exists():
        return []
    return sorted(int(m.group(1)) for d in panel_dir.iterdir() if d.is_dir()
                  for m in [_VERSION_RE.fullmatch(d.name)] if m)


def next_model_version(panel: str, model_dir: Path = MODEL_DIR) -> int:
    versions = _panel_versions(panel, model_dir)
    return (max(versions) + 1) if versions else 1


def resolve_model_version_dir(panel: str, model_dir: Path = MODEL_DIR,
                              version: int | None = None) -> tuple[Path, int]:
    """Return (path, version_number) for the requested — or, if omitted, the
    most recently trained — snapshot of `panel`."""
    versions = _panel_versions(panel, model_dir)
    if not versions:
        raise FileNotFoundError(
            f"No trained model found for {panel} under {model_dir}. "
            f"Run `python PatHUgen.py train` first.")
    if version is None:
        version = max(versions)
    elif version not in versions:
        raise FileNotFoundError(
            f"{panel}: no such trained version v{version} under {model_dir} "
            f"(available: {['v' + str(v) for v in versions]}).")
    return model_dir / panel / f"v{version}", version


def list_model_versions(model_dir: Path = MODEL_DIR) -> None:
    """Print every trained snapshot per panel, oldest to newest, marking the
    one `deliver` will use by default (the latest)."""
    print(f"Model dir: {model_dir.resolve()}\n")
    for panel in PANELS:
        versions = _panel_versions(panel, model_dir)
        if not versions:
            print(f"  {panel:<7s} (no trained versions — run `train` first)")
            continue
        latest = max(versions)
        print(f"  {panel}:")
        for v in versions:
            cfg_path = model_dir / panel / f"v{v}" / "config.json"
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            mark = "  <- latest, used by `deliver` unless --model-version is given" if v == latest else ""
            print(f"    v{v}: trained {cfg.get('trained_at', '?')}  "
                  f"({cfg.get('n_train_rows', '?')} rows, threshold "
                  f"{cfg.get('threshold', '?')}){mark}")


# ==============================================================================
# TRAIN — reproduce the locked champion for every panel from raw CSVs.
# Every call creates a NEW version (v1, v2, ...); previous versions are kept.
# ==============================================================================
def train(panels=None, model_dir: Path = MODEL_DIR) -> None:
    panels = panels or PANELS
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training PatHUgen — locked configuration, {SEED_ENSEMBLE} seeds/panel")
    print(f"Data dir  : {DATA_DIR.resolve()}")
    print(f"Model dir : {model_dir.resolve()}\n")

    for panel in panels:
        cfg = PANEL_CONFIG[panel]
        t0 = datetime.now()
        df = load_panel(panel)
        cat_maps = get_cat_maps(df)
        X, y, _ = build_features(df, cat_maps=cat_maps)
        y = pd.Series(np.asarray(y))
        train_prior = float(y.mean())

        missing = [f for f in cfg["features"] if f not in X.columns]
        if missing:
            raise RuntimeError(
                f"{panel}: locked feature list references columns not "
                f"produced by build_features(): {missing}")
        Xs = X[cfg["features"]]

        aug_X, aug_y = build_aug_pool(panel, columns=Xs.columns)
        if aug_X is not None:
            Xs_full = pd.concat([Xs, aug_X], axis=0, ignore_index=True)
            y_full = pd.concat([y, aug_y], axis=0, ignore_index=True)
            print(f"  {panel}: {len(X)} rows + {len(aug_X)} augmented GENEL rows "
                  f"= {len(Xs_full)} training rows, {len(cfg['features'])} features")
        else:
            Xs_full, y_full = Xs, y
            print(f"  {panel}: {len(X)} rows (no augmentation), "
                  f"{len(cfg['features'])} features")

        models = fit_seed_ensemble(Xs_full, y_full, cfg["xgb_params"])

        version = next_model_version(panel, model_dir)
        panel_dir = model_dir / panel / f"v{version}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(models):
            joblib.dump(m, panel_dir / f"seed{i}.joblib")
        joblib.dump(cat_maps, panel_dir / "cat_maps.joblib")
        (panel_dir / "selected_features.json").write_text(
            json.dumps(cfg["features"], indent=2))
        (panel_dir / "config.json").write_text(json.dumps({
            "panel": panel, "version": version, "n_train_rows": int(len(X)),
            "n_augmented_rows": int(len(aug_X)) if aug_X is not None else 0,
            "n_features": len(cfg["features"]), "seed_ensemble": SEED_ENSEMBLE,
            "threshold": cfg["threshold"], "threshold_policy": cfg["threshold_policy"],
            "train_prior": train_prior, "trained_at": t0.isoformat(),
        }, indent=2))
        prior_versions = [v for v in _panel_versions(panel, model_dir) if v != version]
        kept = f" (previous kept: {', '.join('v' + str(v) for v in prior_versions)})" \
               if prior_versions else ""
        print(f"    -> {panel_dir}{kept}  ({(datetime.now()-t0).total_seconds():.0f}s)")

    print(f"\nDone. Model bundle ready at {model_dir.resolve()}")
    print("Every panel now has a new version; nothing was overwritten. "
          "`deliver` uses the newest version of each panel automatically — "
          "run `python PatHUgen.py versions` to see everything you have "
          "trained, or pin an older one with `deliver --model-version N`.")
    print("The resolved version directory is what Section 6 of the delivery "
          "guide calls the 'model paketi' — it ships as-is inside every "
          "delivery folder.")


# ==============================================================================
# PREDICT — load a trained panel bundle and score a test CSV.
# ==============================================================================
def load_bundle(panel: str, model_dir: Path = MODEL_DIR, version: int | None = None):
    """Load the requested (or, by default, most recently trained) snapshot
    for `panel`. Returns (models, cat_maps, features, config, panel_dir)."""
    panel_dir, resolved_version = resolve_model_version_dir(panel, model_dir, version)
    models = [joblib.load(panel_dir / f"seed{i}.joblib") for i in range(SEED_ENSEMBLE)]
    cat_maps = joblib.load(panel_dir / "cat_maps.joblib")
    features = json.loads((panel_dir / "selected_features.json").read_text())
    config = json.loads((panel_dir / "config.json").read_text())
    return models, cat_maps, features, config, panel_dir


def prior_shift_threshold(t_ref: float, prior_train: float, prior_target: float) -> float:
    """Move a threshold fit under `prior_train` to what it should be under
    `prior_target`, using calibrated-probability Bayes rescaling. Used only
    when --prior-shift is passed; the panel's model is NOT retrained."""
    r = prior_target / prior_train
    s = (1.0 - prior_target) / (1.0 - prior_train)
    return float(t_ref * s / (t_ref * s + (1.0 - t_ref) * r))


def predict_panel(panel: str, csv_path: Path, model_dir: Path = MODEL_DIR,
                   prior_shift: float | None = None,
                   model_version: int | None = None) -> tuple[list[dict], Path]:
    """Return (prediction records, model snapshot dir actually used)."""
    models, cat_maps, features, config, panel_dir = load_bundle(
        panel, model_dir, version=model_version)
    df = load_test_csv(csv_path)
    X, _, _ = build_features(df, cat_maps=cat_maps)

    missing = [f for f in features if f not in X.columns]
    if missing:
        raise RuntimeError(f"{panel}: test data is missing engineered columns "
                           f"{missing} — check the input CSV matches the "
                           f"expected 351-feature schema.")
    Xs = X[features]

    proba = predict_proba_ensemble(models, Xs)
    proba = np.clip(proba, 0.0, 1.0)   # defensive: guide forbids out-of-range probabilities

    threshold = config["threshold"]
    if prior_shift is not None:
        threshold = prior_shift_threshold(threshold, config["train_prior"], prior_shift)

    pred = (proba >= threshold).astype(int)
    ids = df[ID_COL].astype(str).tolist()
    if len(set(ids)) != len(ids):
        dupes = pd.Series(ids)[pd.Series(ids).duplicated()].unique().tolist()[:5]
        raise RuntimeError(f"{panel}: duplicate {ID_COL} values in the input "
                           f"file (e.g. {dupes}) — cannot produce one "
                           f"prediction per id as the delivery guide requires.")

    records = [
        {"id": vid, "panel": panel, "predicted_class": str(int(p)),
         "predicted_prob": round(float(prob), 6)}
        for vid, p, prob in zip(ids, pred, proba)
    ]
    return records, panel_dir


# ==============================================================================
# JSON assembly + validation against the delivery guide's rules (section 3)
# ==============================================================================
def build_result_json(predictions_by_panel: dict[str, list[dict]]) -> dict:
    predictions = []
    for panel in PANELS:
        predictions.extend(predictions_by_panel.get(panel, []))
    return {
        "team_name": TEAM_NAME, "team_id": TEAM_ID, "application_id": APPLICATION_ID,
        "competition_level": COMPETITION_LEVEL, "predictions": predictions,
    }


def validate_result_json(result: dict, expected_counts: dict[str, int]) -> list[str]:
    """Structural checks mirroring section 3 of the delivery guide. Returns a
    list of problems (empty = valid). This does NOT touch the network or the
    upload system — it is a local pre-flight check only."""
    problems = []
    for field in ("team_name", "team_id", "application_id", "competition_level"):
        if not result.get(field):
            problems.append(f"missing or empty top-level field: {field}")
    if result.get("competition_level") != COMPETITION_LEVEL:
        problems.append(f"competition_level must be '{COMPETITION_LEVEL}'")

    preds = result.get("predictions")
    if not isinstance(preds, list) or not preds:
        problems.append("predictions must be a non-empty array")
        return problems

    seen = {}
    for i, rec in enumerate(preds):
        where = f"predictions[{i}] (id={rec.get('id')!r}, panel={rec.get('panel')!r})"
        panel = rec.get("panel")
        if panel not in PANELS:
            problems.append(f"{where}: invalid panel")
        rid = rec.get("id")
        if not rid:
            problems.append(f"{where}: empty id")
        key = (panel, rid)
        if key in seen:
            problems.append(f"{where}: duplicate id within panel {panel}")
        seen[key] = True
        if rec.get("predicted_class") not in ("0", "1"):
            problems.append(f"{where}: predicted_class must be the string '0' or '1', "
                            f"got {rec.get('predicted_class')!r}")
        prob = rec.get("predicted_prob")
        if not isinstance(prob, (int, float)) or isinstance(prob, bool) \
                or not (0.0 - PROB_TOLERANCE <= prob <= 1.0 + PROB_TOLERANCE) \
                or prob != prob:                      # NaN check
            problems.append(f"{where}: predicted_prob must be a finite number in "
                            f"[0, 1], got {prob!r}")

    for panel, expected_n in expected_counts.items():
        got = sum(1 for p, _ in seen if p == panel)
        if got != expected_n:
            problems.append(f"panel {panel}: expected exactly {expected_n} predictions "
                            f"(one per input row), got {got}")

    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        problems.append(f"result is not valid strict JSON (NaN/Infinity present?): {exc}")
    return problems


# ==============================================================================
# Input discovery — map an input directory's CSV files to panels.
# ==============================================================================
_PANEL_KEYWORDS = {
    "GENEL": ("master", "genel"),
    "KANSER": ("kanser",),
    "PAH": ("pah",),
    "CFTR": ("cftr",),
}


def discover_panel_files(input_dir: Path) -> dict[str, Path]:
    """Match every CSV in `input_dir` to a panel by filename keyword.

    Robust to whatever exact filenames the organization hands out on
    competition day (training used YARISMA_TRAIN_<PANEL>.csv; test files may
    follow a different convention) — matching is by keyword, not by exact
    name. If this fails, pass explicit --genel/--kanser/--pah/--cftr paths.
    """
    csvs = sorted(Path(input_dir).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv files found in {input_dir}")
    found: dict[str, Path] = {}
    for csv in csvs:
        name = csv.name.lower()
        for panel, keywords in _PANEL_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                if panel in found:
                    raise RuntimeError(
                        f"Ambiguous input: both {found[panel].name!r} and "
                        f"{csv.name!r} match panel {panel}. Use explicit "
                        f"--{panel.lower()} PATH flags instead.")
                found[panel] = csv
    missing = [p for p in PANELS if p not in found]
    if missing:
        raise FileNotFoundError(
            f"Could not auto-detect a test file for panel(s) {missing} in "
            f"{input_dir} (files seen: {[c.name for c in csvs]}). "
            f"Pass explicit paths: --genel PATH --kanser PATH --pah PATH --cftr PATH")
    return found


# ==============================================================================
# Delivery packaging — Section 6 of the delivery guide, fully automated.
# ==============================================================================
def next_delivery_dir(root: Path = ROOT) -> tuple[Path, int]:
    existing = [int(m.group(1)) for d in root.iterdir() if d.is_dir()
                for m in [re.fullmatch(r"delivery(\d+)", d.name)] if m]
    n = (max(existing) + 1) if existing else 1
    return root / f"delivery{n}", n


def freeze_requirements(dest: Path) -> None:
    """Snapshot exact installed package versions for reproducibility.
    Falls back to copying the repo's requirements.txt if `pip freeze` fails
    (e.g. running outside the intended virtualenv)."""
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=60, check=True)
        dest.write_text(out.stdout)
        return
    except Exception as exc:
        print(f"  [warning] pip freeze failed ({exc}); copying requirements.txt instead")
    src = ROOT / "requirements.txt"
    if src.exists():
        shutil.copy2(src, dest)
    else:
        dest.write_text("# pip freeze unavailable and no requirements.txt found\n")


# Explicit allowlist of source locations that belong to the PatHUgen pipeline.
# Deliberately NOT a blind `root.rglob("*.py")`: this project's root directory
# can contain unrelated scratch projects (e.g. a stray `Teknofest2026/`
# experiment folder was found sitting here during testing, with its own
# notebooks and outputs, unrelated to this codebase) — a blanket recursive
# copy would silently ship someone else's unrelated code as part of a
# competition delivery. Add a new root-level *.py file or a new subpackage
# directory to this list if the project structure changes.
# PatHUgen.py is deliberately self-contained (no imports from any other file
# in this project — verified: it only imports the standard library and
# third-party packages). Its only local file dependencies are the two files
# listed here: the delivery README template it reads at render time, and
# requirements.txt as a fallback if `pip freeze` is unavailable (handled
# separately by freeze_requirements). A delivery package should contain
# exactly what running PatHUgen.py needs and nothing from the rest of the
# repository's development history (the search scripts, exploratory
# notebooks, report generators, etc. play no part in producing this result).
DEPENDENCY_FILES = ("PatHUgen.py", "Delivery_Readme.md")


def copy_source_code(dest: Path, root: Path = ROOT) -> list[str]:
    """Copy PatHUgen.py and the files it directly reads at runtime into
    dest/ — nothing else. Raises if a required file is missing."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in DEPENDENCY_FILES:
        src = root / name
        if not src.exists():
            raise FileNotFoundError(
                f"Delivery packaging expected {name} at {src} but it is "
                f"missing — the delivery would not be self-reproducible.")
        shutil.copy2(src, dest / name)
        copied.append(name)
    return copied


def render_delivery_readme(template_path: Path, *, delivery_n: int, run_command: str,
                           input_dir: Path, output_json: Path,
                           panel_files: dict[str, Path], counts: dict[str, int],
                           version_dirs: dict[str, Path] | None = None) -> str:
    text = template_path.read_text(encoding="utf-8")
    panel_files_md = "\n".join(f"- `{p}` -> `{f}`" for p, f in panel_files.items())
    version_dirs = version_dirs or {}
    counts_md = "\n".join(
        f"- {p}: {counts.get(p, 0)}"
        + (f"  (model {version_dirs[p].name})" if p in version_dirs else "")
        for p in PANELS)
    try:
        lib_versions = subprocess.run(
            [sys.executable, "-c",
             "import numpy,pandas,sklearn,xgboost,joblib;"
             "print(f'numpy {numpy.__version__}, pandas {pandas.__version__}, "
             "scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}, "
             "joblib {joblib.__version__}')"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except Exception:
        lib_versions = "(unavailable — see requirements.txt)"

    replacements = {
        "{{DELIVERY_NUMBER}}": str(delivery_n),
        "{{TEAM_NAME}}": TEAM_NAME, "{{TEAM_ID}}": TEAM_ID,
        "{{APPLICATION_ID}}": APPLICATION_ID,
        "{{GENERATED_AT}}": datetime.now().isoformat(timespec="seconds"),
        "{{RUN_COMMAND}}": run_command,
        "{{INPUT_DIR}}": str(input_dir.resolve()),
        "{{PANEL_FILES}}": panel_files_md,
        "{{PREDICTION_COUNTS}}": counts_md,
        "{{OUTPUT_JSON}}": f"delivery{delivery_n}/{output_json.name}",
        "{{OUTPUT_JSON_NAME}}": output_json.name,
        "{{PYTHON_VERSION}}": sys.version.split()[0],
        "{{LIBRARY_VERSIONS}}": lib_versions,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


# ==============================================================================
# DELIVER — predict on all four panels, assemble + validate the result JSON,
# and build a complete, reproducible delivery folder. This is the command run
# on competition day, once per (practice or official) submission attempt.
# ==============================================================================
def deliver(input_dir: Path, *, panel_files: dict[str, Path] | None = None,
            model_dir: Path = MODEL_DIR, prior_shift: float | None = None,
            model_version: int | None = None, skip_copy: bool = False,
            run_command: str = "") -> Path:
    input_dir = Path(input_dir)
    files = panel_files or discover_panel_files(input_dir)
    print("Panel -> input file mapping:")
    for p in PANELS:
        print(f"  {p:<7s} {files[p]}")

    predictions_by_panel, counts, version_dirs = {}, {}, {}
    for panel in PANELS:
        recs, panel_dir = predict_panel(panel, files[panel], model_dir=model_dir,
                                        prior_shift=prior_shift,
                                        model_version=model_version)
        predictions_by_panel[panel] = recs
        counts[panel] = len(recs)
        version_dirs[panel] = panel_dir
        n_pos = sum(1 for r in recs if r["predicted_class"] == "1")
        print(f"  {panel:<7s} {len(recs):5d} predictions  "
              f"(predicted pathogenic rate {n_pos/len(recs):.1%})  "
              f"[model {panel_dir.name}]")

    result = build_result_json(predictions_by_panel)
    problems = validate_result_json(result, counts)
    if problems:
        print("\nVALIDATION FAILED — result JSON was NOT written:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nValidation passed: schema, ids, classes, and probabilities all OK.")

    delivery_dir, n = next_delivery_dir()
    delivery_dir.mkdir(parents=True)
    numbered_name = f"TEAM_{TEAM_ID}_FINAL{n}.json"
    numbered_path = delivery_dir / numbered_name
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    numbered_path.write_text(payload, encoding="utf-8")
    print(f"\nResult JSON written:")
    print(f"  {numbered_path}   <- upload this file")

    if not skip_copy:
        req_path = delivery_dir / "requirements.txt"
        freeze_requirements(req_path)
        copied_files = copy_source_code(delivery_dir / "code")
        # Copy only the exact model snapshot used for THIS delivery's
        # predictions (not the whole version history) — the delivery guide
        # wants the package sufficient to reproduce this result, and shipping
        # every old snapshot would only obscure which one actually ran.
        # The version number is dropped from the path (model/<PANEL>/, not
        # model/<PANEL>/v3/) so the guide's expected flat layout is
        # unaffected; the exact version used is recorded in README.md.
        model_dest = delivery_dir / "model"
        for panel, panel_dir in version_dirs.items():
            shutil.copytree(panel_dir, model_dest / panel)
        readme_text = render_delivery_readme(
            ROOT / "Delivery_Readme.md", delivery_n=n,
            run_command=run_command or f"python PatHUgen.py deliver --input-dir {input_dir}",
            input_dir=input_dir, output_json=numbered_path,
            panel_files=files, counts=counts, version_dirs=version_dirs)
        (delivery_dir / "README.md").write_text(readme_text, encoding="utf-8")
        print(f"\nPackaged: {', '.join(copied_files)}, model bundle, "
              f"requirements.txt, README.md -> {delivery_dir}")
    else:
        print("\n--skip-copy set: only the JSON file was written (no code/model/"
              "README packaging). Use this for fast iteration; do a full "
              "(non-skip) run before the final upload.")

    print(f"\nDelivery folder ready: {delivery_dir}")
    return delivery_dir


# ==============================================================================
# CLI
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="PatHUgen.py",
        description="TEKNOFEST 2026 Health AI — PatHUgen final delivery system.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the locked model configuration "
                             "from the official training CSVs. Every run creates a "
                             "NEW numbered snapshot (v1, v2, ...) — previous ones "
                             "are kept, never overwritten.")
    p_train.add_argument("panels", nargs="*", choices=PANELS,
                         default=None, help="Panels to train (default: all four).")
    p_train.add_argument("--model-dir", type=Path, default=MODEL_DIR)

    p_versions = sub.add_parser("versions", help="List every trained snapshot "
                                "per panel (which one `deliver` will use by "
                                "default is marked).")
    p_versions.add_argument("--model-dir", type=Path, default=MODEL_DIR)

    p_deliver = sub.add_parser("deliver", help="Run inference on test data, "
                               "validate the result JSON, and build a full, "
                               "numbered delivery folder (competition day).")
    p_deliver.add_argument("--input-dir", type=Path, required=True,
                           help="Folder containing the four test CSVs.")
    p_deliver.add_argument("--genel", type=Path, default=None)
    p_deliver.add_argument("--kanser", type=Path, default=None)
    p_deliver.add_argument("--pah", type=Path, default=None)
    p_deliver.add_argument("--cftr", type=Path, default=None)
    p_deliver.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p_deliver.add_argument("--model-version", type=int, default=None,
                           help="Use a specific trained snapshot number "
                           "(e.g. 2 for v2) instead of the most recently "
                           "trained one. Applies to all four panels. See "
                           "`python PatHUgen.py versions` for what's available. "
                           "Omit to always use the latest.")
    p_deliver.add_argument("--prior-shift", type=float, default=None,
                           help="Optional: recompute the decision threshold as "
                           "if the test set's positive rate were this value "
                           "(e.g. 0.5 for a balanced test set), using the "
                           "saved calibration. Does not retrain anything. "
                           "Omit to use the audited default threshold.")
    p_deliver.add_argument("--skip-copy", action="store_true",
                           help="Write only the JSON (fast). Skips copying "
                           "code/model/README into the delivery folder — use "
                           "for quick iteration, not for the final submission.")

    args = parser.parse_args()
    run_command = "python " + " ".join([Path(sys.argv[0]).name] + sys.argv[1:])

    try:
        if args.command == "train":
            train(panels=args.panels or None, model_dir=args.model_dir)
        elif args.command == "versions":
            list_model_versions(model_dir=args.model_dir)
        elif args.command == "deliver":
            explicit = {"GENEL": args.genel, "KANSER": args.kanser,
                       "PAH": args.pah, "CFTR": args.cftr}
            explicit = {k: v for k, v in explicit.items() if v is not None}
            panel_files = None
            if explicit:
                missing = [p for p in PANELS if p not in explicit]
                if missing:
                    raise SystemExit(f"--genel/--kanser/--pah/--cftr must all be given "
                                     f"together if any is used; missing: {missing}")
                panel_files = explicit
            deliver(args.input_dir, panel_files=panel_files, model_dir=args.model_dir,
                   prior_shift=args.prior_shift, model_version=args.model_version,
                   skip_copy=args.skip_copy, run_command=run_command)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # Expected, actionable failures (bad input path, schema mismatch,
        # missing model, duplicate ids, ...) — print a clean message instead
        # of a stack trace. This runs live during the timed competition test;
        # a scary traceback wastes time a jury-facing error message would not.
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
