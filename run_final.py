"""TEKNOFEST 2026 — final pipeline, local runner.

AUTO-GENERATED from `TEKNOFEST_pipeline_colab.ipynb` (code cells
7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 — verbatim, in order).
This is the exact pipeline that produced `outputs_colab/results_summary.csv`
and the numbers in the PDR; only the Colab/Drive plumbing was replaced by a CLI.

Run (full report-grade budget, all four panels):

    QUICK=0 USE_GPU=0 RUN_NESTED=1 AUGMENT_PANELS=PAH,CFTR \
    SELECTION_METHOD=null OPTUNA_TRIALS=80 OPTUNA_TIMEOUT=600 \
    TEKNOFEST_OUT_DIR=outputs_final \
    .venv/bin/python run_final.py 2>&1 | tee run_final.log

Her koşu logs/run_final_<zaman>.txt dosyasına da yazılır (logs/run_final_latest.txt).

Single panel:  .venv/bin/python run_final.py CFTR
Regenerate:    python scripts/gen_runner.py   (see repo notes)
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))  # runlog import edilebilsin


# ==========================================================================
# notebook cell [7]
# ==========================================================================
"""TEKNOFEST 2026 — Variant Pathogenicity Classifier (self-contained pipeline).

Binary F1 (Pathogenic=1) across four INDEPENDENT panels: GENEL, KANSER, PAH, CFTR.
Everything below — feature engineering, selection, tuning, calibrated stacking,
GENEL→PAH/CFTR augmentation, honest nested-CV estimate — runs from this one file.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ---- Data location: folder holding the four YARISMA_TRAIN_*.csv files. --------
DATA_DIR = Path(os.environ.get(
    "TEKNOFEST_DATA_DIR",
    "universite-veri-seti/EĞİTİM (TRAIN) SETLERİ",
))
OUTPUT_DIR = Path(os.environ.get("TEKNOFEST_OUT_DIR", "outputs_colab"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILES = {
    "GENEL": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}
PANELS = list(PANEL_FILES)

ID_COL, LABEL_COL = "Variant_ID", "Label"
AA_COLS = ["AA_1", "AA_2"]
CAT_COLS = [f"CAT_{i}" for i in range(1, 7)]
EK_COLS = [f"EK_{i}" for i in range(1, 10)]
AL_COLS = [f"AL_{i}" for i in range(1, 335)]
FEATURE_COLS = AL_COLS + CAT_COLS + EK_COLS + AA_COLS
N_FEATURES = len(FEATURE_COLS)
POS = 1

# ---- Run settings (tune speed vs rigor). QUICK ≈ 15-30 min; full ≈ 60-120 min.
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
AUG_SOURCE = "GENEL"
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

# ==========================================================================
# notebook cell [9]
# ==========================================================================
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

# ==========================================================================
# notebook cell [11]
# ==========================================================================
def load_panel(panel: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / PANEL_FILES[panel])
    expected = 1 + len(AL_COLS) + len(CAT_COLS) + len(EK_COLS) + len(AA_COLS) + 1
    if df.shape[1] != expected:
        raise ValueError(f"{panel}: expected {expected} cols, got {df.shape[1]}")
    return df

# ==========================================================================
# notebook cell [13]
# ==========================================================================
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

# ==========================================================================
# notebook cell [15]
# ==========================================================================
def augment_enabled(panel):
    return panel.upper() in AUGMENT_PANELS and panel.upper() != AUG_SOURCE


def build_aug_pool(panel, columns=None):
    """GENEL rows NOT in `panel` (leak-free extra training rows), or (None,None)."""
    if not augment_enabled(panel):
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
    return X_aug.reset_index(drop=True), pd.Series(y_aug).reset_index(drop=True)

# ==========================================================================
# notebook cell [17]
# ==========================================================================
"""Feature selection: null-importance (gain vs shuffled-label null) with an
optional ReliefF guard (the hybrid selector). Default ships `null`."""
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif


def _impute_for_filters(X):
    Xi = X.apply(pd.to_numeric, errors="coerce")
    Xi = Xi.fillna(Xi.median(numeric_only=True))
    return Xi.fillna(0.0)


def information_gain(X, y, seed=SEED):
    Xi = _impute_for_filters(X)
    mi = mutual_info_classif(Xi.to_numpy(), np.asarray(y), discrete_features=False,
                             random_state=seed)
    return pd.Series(mi, index=X.columns)


def relieff(X, y, n_neighbors=10, sample_size=400, seed=SEED):
    Xi = _impute_for_filters(X).to_numpy(dtype=float)
    y = np.asarray(y)
    n, n_feat = Xi.shape
    mu, sd = Xi.mean(0), Xi.std(0)
    sd[sd == 0] = 1.0
    Xz = (Xi - mu) / sd
    rng = np.random.default_rng(seed)
    m = min(sample_size, n)
    sample_idx = rng.choice(n, size=m, replace=False)
    _, counts = np.unique(y, return_counts=True)
    k = max(1, min(n_neighbors, int(counts.min()) - 1))
    weights = np.zeros(n_feat)
    used = 0
    for i in sample_idx:
        diff = np.abs(Xz - Xz[i])
        dist = np.sqrt((diff ** 2).sum(1))
        dist[i] = np.inf
        same = y == y[i]
        same[i] = False
        other = ~same
        if same.sum() < k or other.sum() < k:
            continue
        hit = np.flatnonzero(same)[np.argsort(dist[same])[:k]]
        miss = np.flatnonzero(other)[np.argsort(dist[other])[:k]]
        weights += diff[miss].mean(0) - diff[hit].mean(0)
        used += 1
    if used:
        weights /= used
    return pd.Series(weights, index=X.columns)


def _imp_estimator(y, seed):
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, min_child_samples=20, reg_alpha=0.1,
        reg_lambda=1.0, scale_pos_weight=n_neg / max(n_pos, 1), random_state=seed,
        n_jobs=1, verbosity=-1, importance_type="gain")


def _fit_gain(X, yv, cat_cols, seed):
    m = _imp_estimator(pd.Series(yv), seed)
    m.fit(X, yv, categorical_feature=list(cat_cols))
    return m.booster_.feature_importance(importance_type="gain")


def null_importance_selection(X, y, cat_cols, n_actual=5, n_null=40, seed=SEED,
                              percentile=95, k_div=12, k_cap=50, k_min=10,
                              method="null", ig_protect_k=15):
    """Per-feature DataFrame with a `keep` flag.

    method='null'   : keep = gate(real>null_p95) OR top-K by gain_score.
    method='hybrid' : AND (ReliefF>=0 OR top-ig_protect_k Info Gain OR categorical).
    """
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    null_seeds = rng.integers(0, 2**31 - 1, size=n_null)
    actual = np.mean([_fit_gain(X, y.to_numpy(), cat_cols, seed + i)
                      for i in range(n_actual)], axis=0)

    def _null_run(i):
        ys = y.sample(frac=1.0, random_state=int(null_seeds[i])).to_numpy()
        return _fit_gain(X, ys, cat_cols, 1000 + i)

    nulls = np.vstack(Parallel(n_jobs=-1)(delayed(_null_run)(i) for i in range(n_null)))
    null_pct = np.percentile(nulls, percentile, axis=0)
    null_p75 = np.percentile(nulls, 75, axis=0)
    gain_score = np.log((1.0 + actual) / (1.0 + null_p75))
    gate = actual > null_pct
    k = int(min(k_cap, max(k_min, len(X) // k_div)))
    topk = np.zeros(len(gain_score), dtype=bool)
    topk[np.argsort(gain_score)[::-1][:k]] = True
    keep = gate | topk
    cols = list(X.columns)

    if method == "hybrid":
        rf = relieff(X, y, seed=seed).reindex(cols).fillna(0.0).to_numpy()
        ig = information_gain(X, y, seed=seed).reindex(cols).fillna(-np.inf)
        ig_protected = (ig.rank(ascending=False, method="min") <= ig_protect_k).to_numpy()
        cat_like = set(cat_cols or [])
        cat_protected = np.array([c in cat_like for c in cols])
        keep = keep & ((rf >= 0.0) | ig_protected | cat_protected)

    table = pd.DataFrame({"feature": cols, "gain_score": gain_score, "keep": keep})
    return table.sort_values("gain_score", ascending=False).reset_index(drop=True)

# ==========================================================================
# notebook cell [19]
# ==========================================================================
"""Model factories (GPU-aware, wide class-weight range) + F1-tuned CV harness
with optional GENEL augmentation appended to TRAINING folds only."""
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from xgboost import XGBClassifier

_XGB_GPU = {"device": "cuda"} if USE_GPU else {}
_CAT_GPU = {"task_type": "GPU"} if USE_GPU else {}


def scale_pos_weight(y_tr):
    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    return n_neg / n_pos if n_pos else 1.0


def _split_cols(feature_cols, cat_cols):
    cat_set = set(cat_cols or [])
    return ([c for c in feature_cols if c in cat_set],
            [c for c in feature_cols if c not in cat_set])


def _preprocessor(feature_cols, cat_cols, scale):
    cat, num = _split_cols(feature_cols, cat_cols)
    tr = []
    if cat:
        steps = [("te", TargetEncoder(target_type="binary", smooth="auto", cv=5))]
        if scale:
            steps.append(("sc", StandardScaler()))
        tr.append(("cat", Pipeline(steps), cat))
    if num:
        steps = [("imp", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("sc", StandardScaler()))
        tr.append(("num", Pipeline(steps), num))
    return ColumnTransformer(tr, remainder="drop")


def make_xgb(params, y_tr, feature_cols=None, cat_cols=None):
    p = dict(params)
    spw = p.pop("spw_mult", 1.0)
    return XGBClassifier(n_estimators=p.pop("n_estimators", 400), tree_method="hist",
        eval_metric="logloss", scale_pos_weight=scale_pos_weight(y_tr) * spw,
        random_state=SEED, n_jobs=-1, **_XGB_GPU, **p)


def make_lgbm(params, y_tr, feature_cols=None, cat_cols=None):
    p = dict(params)
    spw = p.pop("spw_mult", 1.0)
    return LGBMClassifier(n_estimators=p.pop("n_estimators", 400), subsample_freq=1,
        scale_pos_weight=scale_pos_weight(y_tr) * spw, random_state=SEED, n_jobs=-1,
        verbosity=-1, **p)


def make_cat(params, y_tr, feature_cols=None, cat_cols=None):
    p = dict(params)
    return CatBoostClassifier(iterations=p.pop("iterations", 500),
        auto_class_weights="Balanced", random_seed=SEED, verbose=False,
        allow_writing_files=False, **_CAT_GPU, **p)


def make_extratrees(params, y_tr, feature_cols, cat_cols):
    p = dict(params)
    pre = _preprocessor(feature_cols, cat_cols, scale=False)
    clf = ExtraTreesClassifier(n_estimators=p.pop("n_estimators", 400),
        max_depth=p.pop("max_depth", None), min_samples_leaf=p.pop("min_samples_leaf", 1),
        max_features=p.pop("max_features", "sqrt"),
        class_weight=p.pop("class_weight", "balanced_subsample"),
        random_state=SEED, n_jobs=-1, **p)
    return Pipeline([("pre", pre), ("clf", clf)])


def make_logreg_l1(params, y_tr, feature_cols, cat_cols):
    p = dict(params)
    pre = _preprocessor(feature_cols, cat_cols, scale=True)
    clf = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=p.pop("C", 1.0),
        class_weight=p.pop("class_weight", "balanced"), max_iter=2000,
        random_state=SEED, **p)
    return Pipeline([("pre", pre), ("clf", clf)])


MAKERS = {"xgboost": make_xgb, "lightgbm": make_lgbm, "catboost": make_cat,
          "extratrees": make_extratrees, "logreg_l1": make_logreg_l1}
CAT_KW = {"lightgbm": "categorical_feature", "catboost": "cat_features"}
KINDS = tuple(MAKERS)
TREE_KINDS = ("xgboost", "lightgbm", "catboost")


def tune_threshold(y, proba, lo=0.05, hi=0.95, step=0.005):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        f1 = f1_score(y, (proba >= t).astype(int), pos_label=POS, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def _append_aug(X_tr, y_tr, aug):
    if aug is None or aug[0] is None:
        return X_tr, y_tr
    Xa, ya = aug
    Xc = pd.concat([X_tr, Xa[X_tr.columns]], axis=0, ignore_index=True)
    yc = pd.concat([pd.Series(np.asarray(y_tr)), pd.Series(np.asarray(ya))],
                   ignore_index=True)
    return Xc, yc


def run_cv(make_model, X, y, cat_cols=None, n_splits=5, seed=SEED, fit_kwargs_fn=None,
           n_repeats=None, aug=None):
    """StratifiedKFold OOF + F1 metrics. aug rows go to TRAIN folds only."""
    n_repeats = CV_REPEATS if n_repeats is None else n_repeats
    y = pd.Series(np.asarray(y), index=X.index)
    fold_thresholds = []

    def _fit(tr_idx):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_fit, y_fit = _append_aug(X_tr, y_tr, aug)
        m = make_model(y_fit)
        fk = fit_kwargs_fn(X_fit, y_fit) if fit_kwargs_fn else {}
        m.fit(X_fit, y_fit, **fk)
        return m

    if n_repeats > 1:
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                           random_state=seed)
        oof_sum, oof_cnt = np.zeros(len(y)), np.zeros(len(y))
        for tr, va in splitter.split(X, y):
            p = _fit(tr).predict_proba(X.iloc[va])[:, 1]
            np.add.at(oof_sum, va, p)
            np.add.at(oof_cnt, va, 1)
            ft, _ = tune_threshold(y.iloc[va].to_numpy(), p)
            fold_thresholds.append(ft)
        oof = oof_sum / np.maximum(oof_cnt, 1)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, va in skf.split(X, y):
            p = _fit(tr).predict_proba(X.iloc[va])[:, 1]
            oof[va] = p
            ft, _ = tune_threshold(y.iloc[va].to_numpy(), p)
            fold_thresholds.append(ft)

    pooled_t, pooled_f1 = tune_threshold(y.to_numpy(), oof)
    # fold-median threshold: robust to one lucky fold on the small panels.
    threshold = float(np.median(fold_thresholds)) if fold_thresholds else pooled_t
    f1 = f1_score(y, (oof >= threshold).astype(int), pos_label=POS, zero_division=0)
    return {"oof": oof, "threshold": threshold, "f1": float(f1),
            "f1_pooled": pooled_f1, "auc": float(roc_auc_score(y, oof)),
            "ap": float(average_precision_score(y, oof))}

# ==========================================================================
# notebook cell [21]
# ==========================================================================
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _small(n): return n is not None and n < 400


def _space(kind, trial, n_rows):
    s = _small(n_rows)
    if kind == "xgboost":
        return dict(max_depth=trial.suggest_int("max_depth", 2, 5 if s else 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            spw_mult=trial.suggest_float("spw_mult", 0.5, SPW_MULT_HI, log=True))
    if kind == "lightgbm":
        return dict(num_leaves=trial.suggest_int("num_leaves", 15, 63 if s else 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            spw_mult=trial.suggest_float("spw_mult", 0.5, SPW_MULT_HI, log=True))
    if kind == "catboost":
        return dict(depth=trial.suggest_int("depth", 3, 6 if s else 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0))
    if kind == "extratrees":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 12 if s else 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            class_weight=trial.suggest_categorical("class_weight",
                ["balanced_subsample", "balanced", None]))
    return dict(C=trial.suggest_float("C", 1e-3, 20.0, log=True),
        class_weight=trial.suggest_categorical("class_weight", ["balanced", None]))


DEFAULTS = {
    "xgboost": dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
        min_child_weight=1, gamma=0.0, reg_alpha=0.1, reg_lambda=1.0, spw_mult=1.0),
    "lightgbm": dict(num_leaves=31, learning_rate=0.05, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
        min_split_gain=0.0, spw_mult=1.0),
    "catboost": dict(depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
        bagging_temperature=1.0, random_strength=1.0),
    "extratrees": dict(n_estimators=400, max_depth=12, min_samples_leaf=1,
        max_features=0.5, class_weight="balanced_subsample"),
    "logreg_l1": dict(C=1.0, class_weight="balanced"),
}


def _fit_fn(kind, cat_cols):
    kw = CAT_KW.get(kind)
    if kw and cat_cols:
        return lambda X_tr, y_tr: {kw: list(cat_cols)}
    return None


def tune_model(kind, X, y, cat_cols, n_trials, timeout, seed=SEED, aug=None):
    feature_cols = list(X.columns)
    fit_fn = _fit_fn(kind, cat_cols)
    if aug is not None and aug[0] is not None:
        aug = (aug[0].reindex(columns=feature_cols), aug[1])
    n_rows = len(y)

    def objective(trial):
        params = _space(kind, trial, n_rows)
        tseed = seed + trial.number if SEED_JITTER else seed
        res = run_cv(lambda y_tr: MAKERS[kind](params, y_tr, feature_cols, cat_cols),
                     X, y, cat_cols=cat_cols, fit_kwargs_fn=fit_fn, seed=tseed,
                     aug=aug, n_repeats=max(1, CV_REPEATS - 1))
        return res.get(OBJECTIVE, res["f1"])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(DEFAULTS[kind])
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params, float(study.best_value)

# ==========================================================================
# notebook cell [23]
# ==========================================================================
"""Calibrated stacking + per-panel champion selection (one-SE rule on AUC).

Base OOF → per-fold isotonic calibration + logistic meta. Champion = simplest
candidate within 1 SE of the best AUC (single models usually win; a stack must
beat them by more than bootstrap noise).

Single/mean candidates use RAW base probabilities end-to-end (threshold tuned on
raw OOF, prediction on raw proba) — calibration is monotonic, so it would not
change F1 at the tuned threshold, and keeping it raw avoids any train/predict
mismatch. The stack does its OWN isotonic calibration internally (leak-free)."""


def generate_base_oof(X, y, cat_cols, params_by_kind, aug=None, n_repeats=None):
    """(base_oof DataFrame, singles dict, full_models dict). aug → train folds +
    full refit only; OOF stays panel-only honest."""
    base_oof, singles, full_models = {}, {}, {}
    feature_cols = list(X.columns)
    aug_X, aug_y = (aug if aug is not None else (None, None))
    if aug_X is not None:
        aug_X = aug_X.reindex(columns=feature_cols)
        aug = (aug_X, aug_y)
    for kind in KINDS:
        params = params_by_kind.get(kind) or {}
        factory = lambda y_tr, k=kind, p=params: MAKERS[k](p, y_tr, feature_cols, cat_cols)
        fit_fn = _fit_fn(kind, cat_cols)
        res = run_cv(factory, X, y, cat_cols=cat_cols, fit_kwargs_fn=fit_fn,
                     aug=aug, n_repeats=n_repeats)
        base_oof[kind] = res["oof"]
        singles[kind] = {"f1": res["f1"], "threshold": res["threshold"], "auc": res["auc"]}
        if aug_X is not None:
            X_full = pd.concat([X, aug_X], axis=0, ignore_index=True)
            y_full = np.concatenate([np.asarray(y), np.asarray(aug_y)])
        else:
            X_full, y_full = X, np.asarray(y)
        model = factory(pd.Series(y_full))
        fk = fit_fn(X_full, y_full) if fit_fn else {}
        model.fit(X_full, y_full, **fk)
        full_models[kind] = model
    return pd.DataFrame(base_oof, index=X.index), singles, full_models


def evaluate_stack(base_oof, y, cols=None, n_splits=5, seed=SEED, n_repeats=None):
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    bo = base_oof.reset_index(drop=True)
    if cols is not None:
        bo = bo[list(cols)]
    n_repeats = CV_REPEATS if n_repeats is None else n_repeats
    if n_repeats > 1:
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                           random_state=seed)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    meta_sum, meta_cnt, fold_t = np.zeros(len(y)), np.zeros(len(y)), []
    for tr, va in splitter.split(bo, y):
        z_tr = np.zeros((len(tr), bo.shape[1]))
        z_va = np.zeros((len(va), bo.shape[1]))
        for j, col in enumerate(bo.columns):
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                bo[col].iloc[tr].to_numpy(), y.iloc[tr].to_numpy())
            z_tr[:, j] = iso.predict(bo[col].iloc[tr].to_numpy())
            z_va[:, j] = iso.predict(bo[col].iloc[va].to_numpy())
        lr = LogisticRegression(max_iter=1000).fit(z_tr, y.iloc[tr].to_numpy())
        p = lr.predict_proba(z_va)[:, 1]
        meta_sum[va] += p
        meta_cnt[va] += 1
        ft, _ = tune_threshold(y.iloc[va].to_numpy(), p)
        fold_t.append(ft)
    meta_oof = meta_sum / np.maximum(meta_cnt, 1)
    threshold = float(np.median(fold_t)) if fold_t else tune_threshold(y.to_numpy(), meta_oof)[0]
    f1 = float(f1_score(y.to_numpy(), (meta_oof >= threshold).astype(int),
                        pos_label=POS, zero_division=0))
    return meta_oof, threshold, f1


def fit_final_stack(base_oof, y, cols=None):
    y = np.asarray(y)
    bo = base_oof if cols is None else base_oof[list(cols)]
    calibrators, z = {}, np.zeros((len(y), bo.shape[1]))
    for j, col in enumerate(bo.columns):
        iso = IsotonicRegression(out_of_bounds="clip").fit(bo[col].to_numpy(), y)
        calibrators[col] = iso
        z[:, j] = iso.predict(bo[col].to_numpy())
    meta = LogisticRegression(max_iter=1000).fit(z, y)
    return calibrators, meta, list(bo.columns)

# ==========================================================================
# notebook cell [25]
# ==========================================================================
COMPLEXITY = {"single": 0, "mean": 1, "stacked_trees": 2, "stacked_all": 3}
STACK_VARIANTS = {"stacked_all": KINDS, "stacked_trees": TREE_KINDS}
N_BOOT = 200


def _trivial_f1(y):
    p = float(np.mean(np.asarray(y) == POS))
    return 2 * p / (1 + p) if (1 + p) else 0.0


def _boot_std(y, proba, fn, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    m, vals = len(y), np.empty(N_BOOT)
    for i in range(n):
        idx = rng.integers(0, m, m)
        try:
            vals[i] = fn(y[idx], proba[idx])
        except ValueError:
            vals[i] = np.nan
    return float(np.nanstd(vals))


def score_candidates(y, probas, strategies, thresholds=None):
    thresholds = thresholds or {}
    y = np.asarray(y)
    out = {}
    for key, proba in probas.items():
        strat = strategies[key]
        t = thresholds.get(key)
        if t is None:
            t, _ = tune_threshold(y, proba)
        f1m = lambda yy, pp, tt=t: f1_score(yy, (pp >= tt).astype(int), pos_label=POS,
                                            zero_division=0)
        out[key] = {"strategy": strat,
            "model": key[len("single_"):] if strat == "single" else None,
            "proba": proba, "threshold": float(t),
            "f1": float(f1_score(y, (proba >= t).astype(int), pos_label=POS, zero_division=0)),
            "f1_std": _boot_std(y, proba, f1m),
            "auc": float(roc_auc_score(y, proba)),
            "auc_std": _boot_std(y, proba, roc_auc_score),
            "complexity": COMPLEXITY[strat]}
    return out


def pick_champion(candidates, metric=SELECT_METRIC):
    std_key = f"{metric}_std"
    best = max(candidates, key=lambda k: candidates[k][metric])
    floor = candidates[best][metric] - candidates[best][std_key]
    eligible = [k for k, c in candidates.items() if c[metric] >= floor]
    champ = min(eligible, key=lambda k: (candidates[k]["complexity"], -candidates[k][metric]))
    return champ, best


def select_champion(base_oof, singles, y):
    """Build candidate table (singles + mean + stacks) and apply the one-SE rule."""
    probas, strategies, saved_t = {}, {}, {}
    for kind in base_oof.columns:
        probas[f"single_{kind}"] = base_oof[kind].to_numpy()
        strategies[f"single_{kind}"] = "single"
        saved_t[f"single_{kind}"] = singles[kind]["threshold"]
    probas["mean"] = base_oof.mean(axis=1).to_numpy()
    strategies["mean"] = "mean"
    variant_cols = {}
    for vname, vkinds in STACK_VARIANTS.items():
        cols = [c for c in vkinds if c in base_oof.columns]
        m_oof, m_t, _ = evaluate_stack(base_oof, y, cols=cols)
        probas[vname] = m_oof
        strategies[vname] = vname
        saved_t[vname] = m_t
        variant_cols[vname] = cols
    cand = score_candidates(y, probas, strategies, saved_t)
    champ_key, best_key = pick_champion(cand)
    c = cand[champ_key]
    return {"strategy": c["strategy"], "model": c["model"], "threshold": c["threshold"],
            "oof_f1": c["f1"], "f1_std": c["f1_std"], "auc": c["auc"], "auc_std": c["auc_std"],
            "cols": variant_cols.get(c["strategy"]), "candidates": cand,
            "selected_over_best": champ_key != best_key}

# ==========================================================================
# notebook cell [27]
# ==========================================================================
def _balanced_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    n_pos, n_neg = int((y_true == POS).sum()), int((y_true != POS).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    w = np.where(y_true == POS, 0.5 / n_pos, 0.5 / n_neg)
    return float(f1_score(y_true, y_pred, pos_label=POS, sample_weight=w, zero_division=0))


def _predict_test(strategy, model, cols, full_models, base_oof_tr, y_tr, X_te):
    base_te = {k: m.predict_proba(X_te)[:, 1] for k, m in full_models.items()}
    if strategy == "single":
        return base_te[model]
    if strategy == "mean":
        return np.mean([base_te[k] for k in full_models], axis=0)
    calibrators, meta, order = fit_final_stack(base_oof_tr, y_tr, cols=cols)
    z = np.column_stack([calibrators[k].predict(base_te[k]) for k in order])
    return meta.predict_proba(z)[:, 1]


def nested_eval(panel, X, y, cat_cols, params_by_kind, selection_method="null"):
    """Leak-free honest F1/AUC: per outer fold, select + champion on train only,
    score once on untouched test. aug (non-panel GENEL) added to outer-train only."""
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    aug_X_full, aug_y_full = build_aug_pool(panel, columns=list(X.columns))
    if NESTED_REPEATS > 1:
        splitter = RepeatedStratifiedKFold(n_splits=NESTED_SPLITS,
            n_repeats=NESTED_REPEATS, random_state=SEED)
    else:
        splitter = StratifiedKFold(n_splits=NESTED_SPLITS, shuffle=True, random_state=SEED)
    f1s, aucs, balf1s = [], [], []
    for tr, te in splitter.split(X, y):
        X_tr_f, y_tr = X.iloc[tr], y.iloc[tr]
        X_te_f, y_te = X.iloc[te], y.iloc[te]
        tbl = null_importance_selection(X_tr_f, y_tr, cat_cols, method=selection_method,
                                        n_null=25)  # lighter per-fold null run
        sel = tbl.loc[tbl["keep"], "feature"].tolist()
        sc = [c for c in cat_cols if c in set(sel)]
        X_tr, X_te = X_tr_f[sel], X_te_f[sel]
        aug = ((aug_X_full[sel], aug_y_full) if aug_X_full is not None else None)
        base_oof_tr, singles_tr, full_models = generate_base_oof(
            X_tr, y_tr, sc, params_by_kind, aug=aug, n_repeats=1)
        ch = select_champion(base_oof_tr, singles_tr, y_tr)
        proba_te = _predict_test(ch["strategy"], ch["model"], ch["cols"],
                                 full_models, base_oof_tr, y_tr, X_te)
        pred = (proba_te >= ch["threshold"]).astype(int)
        f1s.append(f1_score(y_te, pred, pos_label=POS, zero_division=0))
        aucs.append(roc_auc_score(y_te, proba_te))
        balf1s.append(_balanced_f1(y_te.to_numpy(), pred))
    return {"nested_f1": float(np.mean(f1s)), "nested_f1_std": float(np.std(f1s)),
            "nested_auc": float(np.mean(aucs)),
            "nested_balanced_f1": float(np.nanmean(balf1s))}

# ==========================================================================
# notebook cell [29]
# ==========================================================================
"""Per-panel pipeline: select → tune → OOF champion → honest nested estimate →
fit + persist the full-data champion bundle."""
import time

import joblib

SELECTION_METHOD = os.environ.get("SELECTION_METHOD", "null").strip().lower()


def run_panel(panel):
    t0 = time.time()
    print("\n" + "=" * 70)
    print(f"  PANEL {panel}")
    print("=" * 70, flush=True)
    df = load_panel(panel)
    X, y, cat_cols = build_features(df)
    y = pd.Series(np.asarray(y))
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"  rows={len(y)}  features={X.shape[1]}  "
          f"pathogenic={n_pos} benign={n_neg} (pos rate {y.mean():.1%})")

    # 1) Feature selection (full panel) ---------------------------------------
    tbl = null_importance_selection(X, y, cat_cols, method=SELECTION_METHOD)
    selected = tbl.loc[tbl["keep"], "feature"].tolist()
    Xs = X[selected]
    sel_cat = [c for c in cat_cols if c in set(selected)]
    print(f"  [select:{SELECTION_METHOD}] kept {len(selected)}/{X.shape[1]} features "
          f"({len(sel_cat)} categorical)")

    # 2) GENEL augmentation pool (PAH/CFTR) -----------------------------------
    aug_X, aug_y = build_aug_pool(panel, columns=list(Xs.columns))
    aug = (aug_X, aug_y) if aug_X is not None else None
    if aug is not None:
        print(f"  [augment] += {len(aug_X)} GENEL rows in training folds "
              f"({int((aug_y == 1).sum())} pathogenic / {int((aug_y == 0).sum())} benign)")

    # 3) Optuna tuning per base model -----------------------------------------
    params_by_kind = {}
    for kind in KINDS:
        best, score = tune_model(kind, Xs, y, sel_cat, OPTUNA_TRIALS, OPTUNA_TIMEOUT, aug=aug)
        params_by_kind[kind] = best
        print(f"    tuned {kind:11s} {OBJECTIVE.upper()}={score:.4f}", flush=True)

    # 4) Base OOF + champion selection (mildly optimistic OOF) ----------------
    base_oof, singles, full_models = generate_base_oof(Xs, y, sel_cat, params_by_kind, aug=aug)
    ch = select_champion(base_oof, singles, y)
    label = ch["model"] if ch["strategy"] == "single" else ch["strategy"]
    trivial = _trivial_f1(y)
    print(f"  [OOF champion] {label}  F1={ch['oof_f1']:.4f}±{ch['f1_std']:.4f}  "
          f"AUC={ch['auc']:.4f}±{ch['auc_std']:.4f}  @ t={ch['threshold']:.3f}  "
          f"(trivial F1={trivial:.4f}, lift {ch['oof_f1']-trivial:+.4f})")

    # 5) Honest nested-CV estimate (the number to trust) ----------------------
    nested = {}
    if RUN_NESTED:
        nested = nested_eval(panel, X, y, cat_cols, params_by_kind, SELECTION_METHOD)
        print(f"  [HONEST nested] F1={nested['nested_f1']:.4f}±{nested['nested_f1_std']:.4f}  "
              f"AUC={nested['nested_auc']:.4f}  balF1={nested['nested_balanced_f1']:.4f}")

    # 6) Persist the full-data champion bundle --------------------------------
    bundle = OUTPUT_DIR / "models" / panel
    bundle.mkdir(parents=True, exist_ok=True)
    for kind, model in full_models.items():
        joblib.dump(model, bundle / f"base_{kind}.joblib")
    for vname, vkinds in STACK_VARIANTS.items():
        cols = [c for c in vkinds if c in base_oof.columns]
        calibrators, meta, order = fit_final_stack(base_oof, y, cols=cols)
        joblib.dump(calibrators, bundle / f"calibrators_{vname}.joblib")
        joblib.dump(meta, bundle / f"meta_{vname}.joblib")
        (bundle / f"{vname}_order.json").write_text(json.dumps(order))
    base_order = ch["cols"] if ch["strategy"].startswith("stacked") else list(KINDS)
    champion = {"panel": panel, "strategy": ch["strategy"], "model": ch["model"],
                "threshold": ch["threshold"], "base_order": base_order,
                "augmented": aug is not None, "selection_method": SELECTION_METHOD}
    (bundle / "champion.json").write_text(json.dumps(champion, indent=2))
    (bundle / "selected_features.json").write_text(json.dumps(list(Xs.columns)))

    print(f"  [done in {int(time.time()-t0)}s]  bundle → {bundle}")
    row = {"panel": panel, "n": len(y), "champion": label, "strategy": ch["strategy"],
           "n_features": len(selected), "augmented": aug is not None,
           "threshold": round(ch["threshold"], 3),
           "oof_f1": round(ch["oof_f1"], 4), "oof_auc": round(ch["auc"], 4),
           "trivial_f1": round(trivial, 4)}
    if nested:
        row.update({"honest_f1": round(nested["nested_f1"], 4),
                    "honest_auc": round(nested["nested_auc"], 4),
                    "honest_balanced_f1": round(nested["nested_balanced_f1"], 4),
                    "optimism_gap": round(ch["oof_f1"] - nested["nested_f1"], 4)})
    return row


def run_all(panels=None):
    panels = panels or PANELS
    t0 = time.time()
    rows = [run_panel(p) for p in panels]
    summary = pd.DataFrame(rows)
    OUTPUT_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
    print("\n" + "=" * 70)
    print(f"  ALL PANELS COMPLETE — {int(time.time()-t0)//60}m {int(time.time()-t0)%60}s")
    print("=" * 70)
    return summary

# ==========================================================================
# notebook cell [31]
# ==========================================================================
def predict_panel(panel, df):
    """Label rows in `df` with the saved champion bundle for `panel`."""
    bundle = OUTPUT_DIR / "models" / panel
    champion = json.loads((bundle / "champion.json").read_text())
    selected = json.loads((bundle / "selected_features.json").read_text())
    # Encode categoricals with the TRAIN-time vocabulary (rebuilt from the panel
    # CSV) so codes match those the bundle was trained on.
    cat_maps = get_cat_maps(load_panel(panel))
    X, _, _ = build_features(df, cat_maps=cat_maps)
    X = X[selected]
    strategy, order = champion["strategy"], champion["base_order"]

    def base_raw(kind):
        return joblib.load(bundle / f"base_{kind}.joblib").predict_proba(X)[:, 1]

    if strategy == "single":
        proba = base_raw(champion["model"])
    elif strategy == "mean":
        proba = np.mean([base_raw(k) for k in order], axis=0)
    else:
        calibrators = joblib.load(bundle / f"calibrators_{strategy}.joblib")
        meta = joblib.load(bundle / f"meta_{strategy}.joblib")
        z = np.column_stack([calibrators[k].predict(base_raw(k)) for k in order])
        proba = meta.predict_proba(z)[:, 1]
    pred = (proba >= champion["threshold"]).astype(int)
    return pd.DataFrame({ID_COL: df[ID_COL].to_numpy(), "proba": proba, "prediction": pred})



# ---------------------------------------------------------------------------
# CLI entry point (replaces notebook Steps 4-7)
# ---------------------------------------------------------------------------
def _print_summary(res: "pd.DataFrame") -> None:
    cols = [c for c in ["panel", "n", "champion", "strategy", "augmented",
                        "n_features", "threshold", "oof_f1", "oof_auc", "honest_f1",
                        "honest_auc", "honest_balanced_f1", "optimism_gap",
                        "trivial_f1"] if c in res.columns]
    print("\n=== Per-panel results ===")
    print(res[cols].to_string(index=False))
    if "honest_f1" in res.columns:
        print(f"\nMean honest (nested) F1 across panels: {res['honest_f1'].mean():.4f}")
    print(f"Mean OOF F1 across panels:             {res['oof_f1'].mean():.4f}")


def _inference_check(panels) -> None:
    """Notebook Step 6 — verify each saved bundle reloads and scores."""
    print("\n=== Inference wiring check (in-sample, saved bundles) ===")
    for p in panels:
        df = load_panel(p)
        out = predict_panel(p, df)
        f1 = f1_score(df[LABEL_COL], out["prediction"], zero_division=0)
        print(f"  {p:7s} in-sample F1={f1:.4f}  "
              f"predicted pos rate={out['prediction'].mean():.3f}")


def main() -> None:
    import runlog
    log_path = runlog.start("run_final",
                            note="Referans (baseline) pipeline — PDR'deki koşunun aynısı")

    panels = [a.upper() for a in sys.argv[1:]] or PANELS
    unknown = [p for p in panels if p not in PANEL_FILES]
    if unknown:
        raise SystemExit(f"Unknown panel(s): {unknown}. Choose from {PANELS}.")
    print(f"Panels   : {panels}")
    print(f"Output   : {OUTPUT_DIR.resolve()}")
    summary = run_all(panels)
    _print_summary(summary)
    _inference_check(panels)
    print(f"\nWrote {OUTPUT_DIR / 'results_summary.csv'} and "
          f"per-panel bundles under {OUTPUT_DIR / 'models'}/")
    print(f"Log      : {log_path}")


if __name__ == "__main__":
    main()
