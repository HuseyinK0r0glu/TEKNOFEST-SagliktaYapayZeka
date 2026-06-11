"""Per-panel feature engineering.

`build_features` is a STATELESS, LABEL-INDEPENDENT transform: every feature is
either row-local or derived from feature-column values only (never from the
label). That means it is safe to build once on a full panel and then split by
fold index without target leakage. Anything that must be fold-aware (target
encoding, statistic-based imputation) is intentionally NOT done here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import aa_constants as aac
from src import config

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _binary_al_columns(df: pd.DataFrame) -> list[str]:
    """AL columns whose observed (non-null) values are a subset of {0, 1}."""
    out = []
    for col in config.AL_COLS:
        vals = df[col].dropna().to_numpy()
        if vals.size and set(np.unique(vals)).issubset({0.0, 1.0}):
            out.append(col)
    return out


def _encode_categorical(series, mapping=None):
    """Integer-code a categorical series (+1 so missing/unseen -> 0).

    mapping is None at fit time (codes follow pandas' sorted category order,
    identical to the original astype('category').cat.codes + 1) and the derived
    {value: code} map is returned for persistence. At transform time a saved
    mapping is supplied so test data is encoded with the train-time vocabulary.
    """
    if mapping is None:
        cat = series.astype("category")
        mapping = {value: i + 1 for i, value in enumerate(cat.cat.categories)}
        return (cat.cat.codes + 1).astype(int), mapping
    return series.map(mapping).fillna(0).astype(int), mapping


def get_cat_maps(df: pd.DataFrame) -> dict:
    """Train-time {value: code} maps for the categorical columns (for predict)."""
    maps = {}
    for col in config.CAT_COLS:
        _, maps[col] = _encode_categorical(df[col])
    _, maps["AA_1_code"] = _encode_categorical(df["AA_1"].astype("string"))
    _, maps["AA_2_code"] = _encode_categorical(df["AA_2"].astype("string"))
    return maps


def build_features(df: pd.DataFrame, cat_maps=None):
    """Return (X, y, cat_cols).

    X: numeric feature matrix (engineered + categorical codes + raw AL/EK).
    y: int label Series (or None if absent).
    cat_cols: names of integer-coded categorical columns (for LGBM/CatBoost).
    cat_maps: optional {col: {value: code}} from get_cat_maps(train_df); when
        given (prediction), categoricals use the train-time vocabulary.
    """
    y = df[config.LABEL_COL].astype(int) if config.LABEL_COL in df.columns else None
    a1 = df["AA_1"].astype("string")
    a2 = df["AA_2"].astype("string")
    feats: dict[str, object] = {}

    # --- Missingness meta-features ---
    n_missing_total = df[config.FEATURE_COLS].isnull().sum(axis=1)
    feats["n_missing_total"] = n_missing_total
    feats["pct_missing"] = n_missing_total / config.N_FEATURES
    feats["n_missing_al"] = df[config.AL_COLS].isnull().sum(axis=1)
    feats["n_missing_ek"] = df[config.EK_COLS].isnull().sum(axis=1)
    feats["n_missing_cat"] = df[config.CAT_COLS].isnull().sum(axis=1)
    feats["is_skeleton"] = (n_missing_total >= 301).astype(int)
    feats["is_dense"] = (n_missing_total <= 20).astype(int)
    feats["missingness_tier"] = pd.cut(
        n_missing_total, bins=[-1, 20, 100, 200, 300, 351], labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # --- Amino-acid substitution features ---
    feats["blosum62"] = np.array([aac.blosum62(x, z) for x, z in zip(a1, a2)], dtype=float)
    feats["grantham"] = np.array([aac.grantham(x, z) for x, z in zip(a1, a2)], dtype=float)
    for name, table in aac.PHYS.items():
        v1 = a1.map(lambda x: table.get(x, np.nan)).astype(float)
        v2 = a2.map(lambda x: table.get(x, np.nan)).astype(float)
        feats[f"{name}_1"] = v1
        feats[f"{name}_2"] = v2
        feats[f"d_{name}"] = v2 - v1
        feats[f"absd_{name}"] = (v2 - v1).abs()
    feats["is_stop_gain"] = (a2 == "*").fillna(False).astype(int)
    feats["is_stop_loss"] = (a1 == "*").fillna(False).astype(int)
    feats["aa_missing"] = (a1.isna() | a2.isna()).astype(int)
    feats["aa1_nonstd"] = ((~a1.isin(STD_AA)) & a1.notna()).astype(int)
    feats["aa2_nonstd"] = ((~a2.isin(STD_AA | {"*"})) & a2.notna()).astype(int)
    feats["group_change"] = np.array([aac.group_change(x, z) for x, z in zip(a1, a2)], dtype=int)

    # --- In-silico binary vote consensus (over panel-detected binary AL flags) ---
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

    # --- Continuous AL aggregations ---
    cont_cols = [c for c in config.AL_COLS if c not in set(bin_cols)]
    cdf = df[cont_cols]
    feats["al_cont_count"] = cdf.notna().sum(axis=1)
    feats["al_cont_mean"] = cdf.mean(axis=1)
    feats["al_cont_max"] = cdf.max(axis=1)
    feats["al_cont_min"] = cdf.min(axis=1)
    feats["al_cont_std"] = cdf.std(axis=1)
    feats["al_n_rare"] = ((cdf > 0) & (cdf < 1e-3)).sum(axis=1)  # allele-freq-like rarity

    # --- EK aggregations (all EK_* correlate positively with pathogenicity) ---
    edf = df[config.EK_COLS]
    feats["ek_count"] = edf.notna().sum(axis=1)
    feats["ek_mean"] = edf.mean(axis=1)
    feats["ek_max"] = edf.max(axis=1)
    feats["ek_min"] = edf.min(axis=1)
    feats["ek_std"] = edf.std(axis=1)

    eng = pd.DataFrame(feats, index=df.index)
    # A couple of cheap interactions (trees find the rest).
    eng["ek_max_x_grantham"] = eng["ek_max"] * eng["grantham"]
    eng["ek_mean_x_completeness"] = eng["ek_mean"] * (1 - eng["pct_missing"])

    # --- Categorical codes (label-independent; +1 so missing/unseen = 0) ---
    maps = cat_maps or {}
    cat_data = {}
    cat_cols = []
    for col in config.CAT_COLS:
        cat_data[col], _ = _encode_categorical(df[col], maps.get(col))
        cat_cols.append(col)
    cat_data["AA_1_code"], _ = _encode_categorical(a1, maps.get("AA_1_code"))
    cat_data["AA_2_code"], _ = _encode_categorical(a2, maps.get("AA_2_code"))
    cat_cols += ["AA_1_code", "AA_2_code"]
    cat_df = pd.DataFrame(cat_data, index=df.index)

    # --- Raw passthrough (trees handle NaN + high dimensionality natively) ---
    raw = df[config.AL_COLS + config.EK_COLS].copy()

    X = pd.concat([eng, cat_df, raw], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X, y, cat_cols
