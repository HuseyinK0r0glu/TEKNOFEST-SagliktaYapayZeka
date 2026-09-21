"""Feature engineering (`build_features`) — stateless, label-independent.

Runs identically at train and predict time, so there is no risk of the two
ever drifting out of sync.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathugen.aa_tables import PHYS, blosum45, blosum62, blosum80, grantham, group_change
from pathugen.config import AA_COLS, AL_COLS, CAT_COLS, EK_COLS, FEATURE_COLS, LABEL_COL, N_FEATURES

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def binary_al_columns(df):
    """AL_ columns that hold only 0/1 votes in `df`, as opposed to a
    continuous score.

    This is a property of the TRAINING data, not of whatever batch happens
    to be passed in: `build_features` used to call this fresh on every
    invocation, including at predict time, so a test batch where one of
    these columns happened to be entirely missing (or, in principle,
    genuinely non-binary in that batch) silently produced a different set
    of engineered columns than training did — a train/predict skew that
    surfaces as a confusing "missing engineered columns" error, or worse,
    silently wrong values. Callers that need it to stay fixed across calls
    (train vs. predict) must compute it once from the training data and
    pass it into `build_features` as `bin_cols`, the same way `cat_maps`
    already works.
    """
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


def build_features(df, cat_maps=None, bin_cols=None):
    """Stateless, label-independent per-row transform → (X, y, cat_cols).

    `bin_cols` should be the fixed list from `binary_al_columns()` run on
    the TRAINING data (persisted in the model bundle); when omitted it is
    (re)computed from `df` itself, which is only correct when `df` IS the
    training data — see `binary_al_columns()`.
    """
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

    bin_cols = binary_al_columns(df) if bin_cols is None else [c for c in bin_cols if c in df.columns]
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
