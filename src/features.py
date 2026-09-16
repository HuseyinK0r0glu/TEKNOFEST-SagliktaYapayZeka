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

    # Missingness-block features: which annotation groups are absent together.
    # CAT_1/2 = population labels, CAT_3/4/5 = genotypes (per data profile). A
    # whole block being absent is an annotation-source signature, not random NA.
    feats["all_ek_missing"] = df[config.EK_COLS].isnull().all(axis=1).astype(int)
    feats["population_block_missing"] = df[["CAT_1", "CAT_2"]].isnull().all(axis=1).astype(int)
    feats["genotype_block_missing"] = df[["CAT_3", "CAT_4", "CAT_5"]].isnull().all(axis=1).astype(int)
    feats["n_missing_blocks"] = (
        feats["all_ek_missing"] + feats["population_block_missing"]
        + feats["genotype_block_missing"]
    )

    # --- Amino-acid substitution features ---
    feats["blosum62"] = np.array([aac.blosum62(x, z) for x, z in zip(a1, a2)], dtype=float)
    feats["blosum45"] = np.array([aac.blosum45(x, z) for x, z in zip(a1, a2)], dtype=float)
    feats["blosum80"] = np.array([aac.blosum80(x, z) for x, z in zip(a1, a2)], dtype=float)
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

    # log1p-scaled aggregations: continuous AL values span allele-frequency-like
    # magnitudes (<1e-3) up to ~2.6e5, so the raw mean/max/std above are dominated
    # by the large-magnitude columns. log1p compresses each value before pooling.
    log_cdf = np.log1p(cdf.clip(lower=0))
    feats["al_log_mean"] = log_cdf.mean(axis=1)
    feats["al_log_max"] = log_cdf.max(axis=1)
    feats["al_log_min"] = log_cdf.min(axis=1)
    feats["al_log_std"] = log_cdf.std(axis=1)
    # rare-allele rarity: -log10 of the smallest positive value (larger = rarer).
    min_pos = cdf.where(cdf > 0).min(axis=1)
    feats["al_neglog10_min_pos"] = -np.log10(min_pos.clip(lower=1e-12))
    feats["al_n_ultrarare"] = ((cdf > 0) & (cdf < 1e-4)).sum(axis=1)

    # Continuous-predictor profile (within-row, log scale, stateless): the binary
    # consensus above only covers the binary AL flags, but PAH's signal lives in
    # the *continuous* in-silico predictor columns. These describe the SHAPE of a
    # variant's predictor profile (spread + how many annotations are elevated for
    # this variant) without any cross-row statistic, so they stay leak-free. The
    # per-panel null-importance selection keeps only the ones that actually help.
    n_cont = cdf.notna().sum(axis=1)
    feats["al_log_range"] = log_cdf.max(axis=1) - log_cdf.min(axis=1)
    feats["al_log_iqr"] = (log_cdf.quantile(0.75, axis=1)
                           - log_cdf.quantile(0.25, axis=1))
    row_logmed = log_cdf.median(axis=1)
    feats["al_frac_above_logmed"] = (
        log_cdf.gt(row_logmed, axis=0).sum(axis=1) / n_cont.replace(0, np.nan)
    )

    # --- EK aggregations (all EK_* correlate positively with pathogenicity) ---
    edf = df[config.EK_COLS]
    feats["ek_count"] = edf.notna().sum(axis=1)
    feats["ek_mean"] = edf.mean(axis=1)
    feats["ek_max"] = edf.max(axis=1)
    feats["ek_min"] = edf.min(axis=1)
    feats["ek_std"] = edf.std(axis=1)

    eng = pd.DataFrame(feats, index=df.index)
    # A few cheap interactions (trees find the rest). The vote×severity pairs link
    # in-silico consensus to substitution/evidence strength — aimed at the panels
    # whose signal is concentrated in the AL predictors (PAH); null-importance
    # prunes them elsewhere. Guarded because the vote features only exist when the
    # panel has binary AL flags.
    eng["ek_max_x_grantham"] = eng["ek_max"] * eng["grantham"]
    eng["ek_mean_x_completeness"] = eng["ek_mean"] * (1 - eng["pct_missing"])
    eng["al_logmax_x_ek_max"] = eng["al_log_max"] * eng["ek_max"]
    # Annotation-coverage × evidence-strength: Orange's Rank widget flagged the
    # missingness tier among PAH/GENEL's top features, and t-SNE branches grouped
    # variants by which tools covered them. This interaction lets the trees route
    # on "how complete is this variant's annotation AND how strong is its EK
    # evidence" in one split; null-importance prunes it where it doesn't help.
    eng["missingtier_x_ekcount"] = eng["missingness_tier"] * eng["ek_count"]
    if "damaging_vote_ratio" in eng.columns:
        dvr = eng["damaging_vote_ratio"].fillna(0)
        eng["votes_x_grantham"] = dvr * eng["grantham"]
        eng["votes_x_ek_mean"] = dvr * eng["ek_mean"]

    # --- Categorical codes (label-independent; +1 so missing/unseen = 0) ---
    # An all-null categorical (e.g. CAT_6 in PAH) collapses to the constant code 0
    # here. We keep it rather than drop it so the train/predict column schema stays
    # identical; null-importance selection then prunes the constant column, so it
    # never reaches the final models or wastes a categorical slot.
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
