"""
feature_engineering.py
======================
Phase 2 of the plan. Turns the raw 351-column variant table into the engineered
feature matrix.

Two kinds of transforms, as mandated by the plan:

  * **Stateless / row-level** -- depend only on a single row, need no training
    statistics and no labels. Safe to apply to the whole table at any time.
    (missingness meta-features, AA physicochemical suite, population-frequency
    log aggregations, in-silico consensus, tier flags, source flags.)

  * **Fit/transform** -- require *training-fold* statistics or labels, so the
    `.fit()` must see ONLY the training rows of the current fold to stay
    leakage-free:
      - ``EKRankNormalizer``         : EK directionality + rank-normalisation
      - ``MissingnessPatternEncoder``: cosine similarity to per-class missingness
                                       centroids

Recommended per-fold call order (see ``fit_fold`` / ``transform_split``)::

    schema = build_feature_schema(dev_df)
    stateless = engineer_features_stateless(df, schema)         # row-level
    fitted    = fit_fold(train_stateless, y_train, schema)      # fit on TRAIN only
    train_X   = transform_split(train_stateless, schema, fitted)
    val_X     = transform_split(val_stateless,   schema, fitted)

The EK aggregate + interaction features are computed inside ``transform_split``
because they depend on the (fit-dependent) rank-normalised EK columns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import aa_tables as aat
from .config import (
    AA_PAIR_COL,
    AF_COLS,
    EK_DIRECTION_PATH,
    EK_RANK_THRESHOLDS_PATH,
    FEATURE_SCHEMA_PATH,
    FeatureSchema,
    LABEL_COL,
    MISSINGNESS_TIER_BINS,
    MISSINGNESS_TIER_ORDER,
    SKELETON_MIN_MISSING,
    DENSE_MAX_MISSING,
    SOURCES,
    SOURCE_COL,
)

EPS = 1e-10  # additive constant for log of allele frequencies


# =========================================================================== #
# Schema construction
# =========================================================================== #
def build_feature_schema(df: pd.DataFrame) -> FeatureSchema:
    """Classify the raw columns into AL/EK/CAT/AA groups and split AL into
    binary-flag vs continuous.

    Binary vs continuous is a *structural* property (the column only ever
    contains {0,1}), NOT a label-dependent statistic, so deriving it from the
    full development set is not leakage. We persist it so val/test reuse the
    identical column lists.
    """
    al_cols = sorted([c for c in df.columns if c.startswith("AL_")],
                     key=lambda c: int(c.split("_")[1]))
    ek_cols = sorted([c for c in df.columns if c.startswith("EK_")],
                     key=lambda c: int(c.split("_")[1]))
    cat_cols = sorted([c for c in df.columns if c.startswith("CAT_")],
                      key=lambda c: int(c.split("_")[1]))
    aa_cols = [c for c in ("AA_1", "AA_2") if c in df.columns]

    binary_al, continuous_al = [], []
    for c in al_cols:
        uniq = set(pd.unique(df[c].dropna()))
        if uniq and uniq.issubset({0.0, 1.0}):
            binary_al.append(c)
        else:
            continuous_al.append(c)

    af_cols = [c for c in AF_COLS if c in df.columns]
    return FeatureSchema(
        al_cols=al_cols, ek_cols=ek_cols, cat_cols=cat_cols, aa_cols=aa_cols,
        af_cols=af_cols, binary_al_cols=binary_al, continuous_al_cols=continuous_al,
    )


def save_feature_schema(schema: FeatureSchema, path: str = FEATURE_SCHEMA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema.to_dict(), f, indent=2)


def load_feature_schema(path: str = FEATURE_SCHEMA_PATH) -> FeatureSchema:
    with open(path) as f:
        return FeatureSchema.from_dict(json.load(f))


def _missingness_blocks(schema: FeatureSchema) -> Dict[str, List[str]]:
    """Structural "tool groups" used for block-missingness features (Phase 2.1).

    We use the column-group structure discovered in EDA as the blocks. Each
    block corresponds to a family of annotations that tends to be present or
    absent together. (Phase-1 hierarchical clustering of null patterns can later
    refine these, but the structural grouping is a strong, deterministic proxy.)
    """
    cat = set(schema.cat_cols)
    cont_non_af = [c for c in schema.continuous_al_cols if c not in set(schema.af_cols)]
    blocks = {
        "af": list(schema.af_cols),                       # population allele freqs
        "ek": list(schema.ek_cols),                       # composite/conservation
        "binflag": list(schema.binary_al_cols),           # per-tool damaging flags
        "contscore": cont_non_af,                         # continuous in-silico scores
        "cat_pop": [c for c in ("CAT_1", "CAT_2") if c in cat],
        "cat_geno": [c for c in ("CAT_3", "CAT_4", "CAT_5") if c in cat],
        "cat_region": [c for c in ("CAT_6",) if c in cat],
        "aa": list(schema.aa_cols),                       # amino-acid change
    }
    return {k: v for k, v in blocks.items() if v}  # drop empty blocks


# =========================================================================== #
# Phase 2.1 -- Missingness meta-features
# =========================================================================== #
def add_missingness_features(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Counts, ratios, tier, presence flags, and block-missingness features."""
    out = df.copy()
    feat_cols = schema.feature_cols
    al, ek = schema.al_cols, schema.ek_cols

    # --- counts / ratios -------------------------------------------------- #
    out["n_missing_total"] = df[feat_cols].isna().sum(axis=1)
    out["n_missing_al"] = df[al].isna().sum(axis=1)
    out["n_missing_ek"] = df[ek].isna().sum(axis=1)
    out["pct_missing"] = out["n_missing_total"] / float(len(feat_cols))

    # --- 3-way ordinal tier ---------------------------------------------- #
    def _tier(n: int) -> str:
        for name, (lo, hi) in MISSINGNESS_TIER_BINS.items():
            if lo <= n <= hi:
                return name
        return "sparse"

    out["missingness_tier"] = out["n_missing_total"].apply(_tier).astype("object")
    out["missingness_tier_encoded"] = out["missingness_tier"].map(MISSINGNESS_TIER_ORDER).astype("int8")

    # --- presence flags (verified CAT semantics from Phase 1) ------------- #
    out["has_population_data"] = df["CAT_1"].notna().astype("int8") if "CAT_1" in df else 0
    out["has_allofus_data"] = df["CAT_2"].notna().astype("int8") if "CAT_2" in df else 0
    out["has_genotype"] = df["CAT_3"].notna().astype("int8") if "CAT_3" in df else 0

    # --- block-missingness features (Phase 2.1) --------------------------- #
    blocks = _missingness_blocks(schema)
    profile_parts = []
    for name, cols in blocks.items():
        present = df[cols].notna().any(axis=1).astype("int8")
        out[f"block_{name}_present"] = present
        profile_parts.append(present.astype(str))
    out["n_blocks_present"] = out[[f"block_{n}_present" for n in blocks]].sum(axis=1).astype("int8")
    # String code of which blocks are present (e.g. "10110101") for CatBoost.
    out["tool_coverage_profile"] = profile_parts[0].str.cat([p for p in profile_parts[1:]], sep="")
    return out


# =========================================================================== #
# Phase 2.7 -- Missingness tier sub-model flags
# =========================================================================== #
def add_tier_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for the skeleton / dense tier model heads (Phase 4.6)."""
    out = df.copy()
    assert "n_missing_total" in out, "Run add_missingness_features() first."
    out["is_skeleton_row"] = (out["n_missing_total"] >= SKELETON_MIN_MISSING).astype("int8")
    out["is_dense_row"] = (out["n_missing_total"] <= DENSE_MAX_MISSING).astype("int8")
    return out


# =========================================================================== #
# Phase 2.2 -- Amino-acid physicochemical features
# =========================================================================== #
def _map_property(aa_series: pd.Series, key: str) -> pd.Series:
    """Vectorised lookup of one physicochemical property; non-standard -> NaN."""
    lut = {a: aat.AA_PROPERTIES[a][key] for a in aat.STANDARD_AA}
    return aa_series.map(lut).astype("float32")


def _pair_lookup(aa1: pd.Series, aa2: pd.Series, matrix: Dict[str, Dict[str, float]]) -> pd.Series:
    """Vectorised (AA1, AA2) matrix lookup (Grantham / BLOSUM); non-standard -> NaN."""
    vals = [
        matrix.get(a, {}).get(b, np.nan) if (isinstance(a, str) and isinstance(b, str)) else np.nan
        for a, b in zip(aa1.values, aa2.values)
    ]
    return pd.Series(vals, index=aa1.index, dtype="float32")


def add_amino_acid_features(df: pd.DataFrame) -> pd.DataFrame:
    """BLOSUM45/62/80, Grantham, full physicochemical suite, chemical-group
    membership, signed/absolute deltas, and the special variant-type flags."""
    out = df.copy()
    a1 = df["AA_1"].astype("object")
    a2 = df["AA_2"].astype("object")
    std = set(aat.STANDARD_AA)

    # --- substitution matrices ------------------------------------------- #
    out["blosum45_score"] = _pair_lookup(a1, a2, aat.BLOSUM45)
    out["blosum62_score"] = _pair_lookup(a1, a2, aat.BLOSUM62)
    out["blosum80_score"] = _pair_lookup(a1, a2, aat.BLOSUM80)
    out["blosum_mean"] = out[["blosum45_score", "blosum62_score", "blosum80_score"]].mean(axis=1)
    out["blosum_min"] = out[["blosum45_score", "blosum62_score", "blosum80_score"]].min(axis=1)

    # --- Grantham distance ----------------------------------------------- #
    out["grantham_score"] = _pair_lookup(a1, a2, aat.GRANTHAM)
    out["grantham_category"] = pd.cut(
        out["grantham_score"], bins=[-0.1, 50, 100, 1e9],
        labels=["conservative", "moderate", "radical"],
    ).astype("object")

    # --- per-residue properties + deltas --------------------------------- #
    for key in aat.AA_PROPERTY_KEYS:
        p1 = _map_property(a1, key)
        p2 = _map_property(a2, key)
        out[f"aa1_{key}"] = p1
        out[f"aa2_{key}"] = p2
        delta = p2 - p1                       # signed (AA2 - AA1)
        out[f"delta_{key}"] = delta
        out[f"abs_delta_{key}"] = delta.abs()
    # Worst-case physicochemical disruption across all properties.
    abs_delta_cols = [f"abs_delta_{k}" for k in aat.AA_PROPERTY_KEYS]
    out["max_abs_delta"] = out[abs_delta_cols].max(axis=1)

    # --- chemical-group membership flags --------------------------------- #
    for flag, members in aat.CHEM_GROUP_MEMBERSHIP.items():
        out[f"aa1_{flag}"] = a1.isin(members).astype("int8")
        out[f"aa2_{flag}"] = a2.isin(members).astype("int8")

    # --- group change / charge direction --------------------------------- #
    g1 = a1.map(aat.AA_PRIMARY_GROUP)
    g2 = a2.map(aat.AA_PRIMARY_GROUP)
    both_std = a1.isin(std) & a2.isin(std)
    out["group_change"] = np.where(both_std, (g1 != g2).astype("int8"), np.nan)
    out["is_same_group"] = np.where(both_std, (g1 == g2).astype("int8"), np.nan)
    # charge sign flip (e.g. + -> -, or neutral -> charged)
    c1 = _map_property(a1, "charge")
    c2 = _map_property(a2, "charge")
    out["charge_direction_change"] = (np.sign(c1) != np.sign(c2)).astype("float32")
    out.loc[c1.isna() | c2.isna(), "charge_direction_change"] = np.nan

    # --- special variant-type flags -------------------------------------- #
    out["is_stop_gain"] = (a2 == aat.STOP_CODON).astype("int8")        # nonsense
    out["is_stop_loss"] = (a1 == aat.STOP_CODON).astype("int8")        # invalid for missense
    out["is_synonymous"] = (a1 == a2).astype("int8")                   # expected benign
    out["aa1_is_nonstandard"] = (~a1.isin(std) & a1.notna()).astype("int8")
    out["aa2_is_nonstandard"] = (~a2.isin(std) & a2.notna() & (a2 != aat.STOP_CODON)).astype("int8")

    # transition vs transversion proxy (only defined for standard pairs)
    def _trans(a, b):
        if a in std and b in std:
            return int(aat.TRANSITION_COMPATIBLE[a][b])
        return np.nan
    out["transition_compatible"] = [
        _trans(a, b) for a, b in zip(a1.values, a2.values)
    ]
    out["transition_compatible"] = out["transition_compatible"].astype("float32")

    # categorical AA pair (target-encoded for trees, native for CatBoost)
    out[AA_PAIR_COL] = a1.fillna("?").astype(str) + "_" + a2.fillna("?").astype(str)
    return out


# =========================================================================== #
# Phase 2.3 -- Population-frequency aggregations (log-scale)
# =========================================================================== #
def add_population_frequency_features(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Allele frequencies are extremely right-skewed -> aggregate on log10 scale."""
    out = df.copy()
    af = schema.af_cols
    raw = df[af]
    log_af = np.log10(raw + EPS)  # NaNs propagate (ignored by nan-aware reducers)

    out["af_log_max"] = log_af.max(axis=1)
    out["af_log_mean"] = log_af.mean(axis=1)         # mean over non-null populations
    out["af_log_std"] = log_af.std(axis=1)
    out["af_log_min"] = log_af.min(axis=1)
    out["af_max_raw"] = raw.max(axis=1)
    out["af_count_nonzero"] = (raw > 0).sum(axis=1).astype("int16")
    out["af_population_spread"] = out["af_log_max"] - out["af_log_min"]
    out["af_missing_fraction"] = raw.isna().mean(axis=1).astype("float32")

    # rarity flags -- driven by the *observed* max raw AF
    amax = out["af_max_raw"]
    out["is_rare"] = (amax < 0.001).astype("float32")
    out["is_very_rare"] = (amax < 0.0001).astype("float32")
    out["is_common"] = (amax > 0.01).astype("float32")
    # When AF is entirely absent, rarity is undefined rather than "common".
    no_af = amax.isna()
    for c in ("is_rare", "is_very_rare", "is_common"):
        out.loc[no_af, c] = np.nan
    return out


# =========================================================================== #
# Phase 2.4 -- In-silico score consensus (binary damaging flags)
# =========================================================================== #
def add_insilico_consensus_features(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Vote aggregation over the binary "tool says damaging" flag columns."""
    out = df.copy()
    flags = schema.binary_al_cols
    F = df[flags]

    n_avail = F.notna().sum(axis=1)
    dmg = (F == 1).sum(axis=1)
    ben = (F == 0).sum(axis=1)

    out["n_tools_available"] = n_avail.astype("int16")
    out["damaging_vote_count"] = dmg.astype("int16")
    out["benign_vote_count"] = ben.astype("int16")
    with np.errstate(invalid="ignore", divide="ignore"):
        out["damaging_vote_ratio"] = np.where(n_avail > 0, dmg / n_avail, np.nan).astype("float32")
    out["unanimous_damaging"] = ((n_avail > 0) & (dmg == n_avail)).astype("int8")
    out["unanimous_benign"] = ((n_avail > 0) & (ben == n_avail)).astype("int8")
    out["conflicting_predictions"] = ((dmg > 0) & (ben > 0)).astype("int8")
    out["tools_available_ratio"] = (n_avail / max(len(flags), 1)).astype("float32")
    return out


# =========================================================================== #
# Phase 2.8 -- Dataset-source features
# =========================================================================== #
def add_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binary one-hot flags for the source dataset (always-safe baseline).

    The target-encoded variant (`source_target_encoded`) is produced in the
    encoding phase (src/encoders.py), since it requires fold-wise label stats.
    """
    out = df.copy()
    if SOURCE_COL in df:
        for s in SOURCES:
            out[f"is_{s}"] = (df[SOURCE_COL] == s).astype("int8")
    return out


# =========================================================================== #
# Stateless orchestrator
# =========================================================================== #
def engineer_features_stateless(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Apply every row-level / label-free transform (Phases 2.1-2.4, 2.7, 2.8).

    Output still contains the raw columns (kept for CatBoost native handling and
    for the EK fit/transform step that runs next).
    """
    out = add_missingness_features(df, schema)
    out = add_tier_flags(out)
    out = add_amino_acid_features(out)
    out = add_population_frequency_features(out, schema)
    out = add_insilico_consensus_features(out, schema)
    out = add_source_features(out)
    return out


# =========================================================================== #
# Phase 2.5 -- EK directionality + rank-normalisation (FIT ON TRAIN ONLY)
# =========================================================================== #
@dataclass
class EKRankNormalizer:
    """Direction-correct then percentile-rank-normalise each EK column.

    Directionality is auto-detected from the *training fold* using the
    point-biserial correlation between the raw EK value and the label:
      corr >= 0  -> higher = more pathogenic -> direction = +1
      corr <  0  -> SIFT-like inverted score -> direction = -1 (values negated)
    Correction happens BEFORE rank-normalisation, exactly as the plan requires.

    The fitted state (direction per column + sorted direction-adjusted training
    values) is everything needed to map any new value to a training-set
    percentile without leakage.
    """

    directions: Dict[str, int] = None        # col -> +1 / -1
    sorted_vals: Dict[str, np.ndarray] = None  # col -> sorted adjusted train values
    ek_cols: List[str] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, schema: FeatureSchema,
            manual_directions: Optional[Dict[str, int]] = None) -> "EKRankNormalizer":
        """Detect direction (+1/-1) per EK column then store sorted adjusted values.

        ``manual_directions`` lets a clinician override the auto-detected sign for
        a column known a priori to be SIFT-like inverted (Phase 1.4 investigation);
        any column not listed falls back to correlation-based auto-detection.
        """
        self.ek_cols = list(schema.ek_cols)
        self.directions, self.sorted_vals = {}, {}
        manual_directions = manual_directions or {}
        y = np.asarray(y_train).astype(float)
        for col in self.ek_cols:
            vals = X_train[col].to_numpy(dtype=float)
            mask = ~np.isnan(vals)
            if col in manual_directions:                       # clinician override
                direction = int(manual_directions[col])
            else:                                              # data-driven detection
                direction = 1
                if mask.sum() > 2 and len(np.unique(vals[mask])) > 1 and len(np.unique(y[mask])) > 1:
                    corr = np.corrcoef(vals[mask], y[mask])[0, 1]
                    if np.isfinite(corr) and corr < 0:
                        direction = -1
            adj = np.sort(vals[mask] * direction)
            self.directions[col] = int(direction)
            self.sorted_vals[col] = adj
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.directions is not None, "Call fit() before transform()."
        out = X.copy()
        for col in self.ek_cols:
            sorted_adj = self.sorted_vals[col]
            n = len(sorted_adj)
            vals = X[col].to_numpy(dtype=float) * self.directions[col]
            if n == 0:
                out[f"{col}_ranked"] = np.nan
                continue
            # percentile of each value within the training distribution
            ranks = np.searchsorted(sorted_adj, vals, side="right") / n
            ranks[np.isnan(X[col].to_numpy(dtype=float))] = np.nan
            out[f"{col}_ranked"] = ranks.astype("float32")
        return out

    # --- persistence ----------------------------------------------------- #
    def save(self, direction_path: str = EK_DIRECTION_PATH,
             thresholds_path: str = EK_RANK_THRESHOLDS_PATH) -> None:
        os.makedirs(os.path.dirname(direction_path), exist_ok=True)
        with open(direction_path, "w") as f:
            json.dump(self.directions, f, indent=2)
        payload = {c: {"direction": self.directions[c],
                       "sorted_vals": self.sorted_vals[c].tolist()}
                   for c in self.ek_cols}
        with open(thresholds_path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, thresholds_path: str = EK_RANK_THRESHOLDS_PATH) -> "EKRankNormalizer":
        with open(thresholds_path) as f:
            payload = json.load(f)
        obj = cls(directions={}, sorted_vals={}, ek_cols=list(payload.keys()))
        for c, d in payload.items():
            obj.directions[c] = int(d["direction"])
            obj.sorted_vals[c] = np.asarray(d["sorted_vals"], dtype=float)
        return obj


def add_ek_aggregate_features(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Aggregate the direction-corrected, rank-normalised EK columns.

    Requires the ``{EK}_ranked`` columns produced by ``EKRankNormalizer.transform``.
    """
    out = df.copy()
    ranked = [f"{c}_ranked" for c in schema.ek_cols if f"{c}_ranked" in df.columns]
    R = df[ranked]

    out["ek_count"] = R.notna().sum(axis=1).astype("int8")
    out["ek_mean_ranked"] = R.mean(axis=1)
    out["ek_max_ranked"] = R.max(axis=1)
    out["ek_min_ranked"] = R.min(axis=1)
    out["ek_range_ranked"] = out["ek_max_ranked"] - out["ek_min_ranked"]
    out["ek_std_ranked"] = R.std(axis=1)
    out["ek_unanimity"] = (R > 0.5).sum(axis=1).astype("int8")          # above-median consensus
    out["ek_strong_damaging"] = (R > 0.75).sum(axis=1).astype("int8")   # strong damaging
    above = out["ek_unanimity"]
    out["ek_conflicting"] = ((above > 0) & ((out["ek_count"] - above) > 0)).astype("int8")
    return out


# =========================================================================== #
# Phase 2.1 (extra) -- Missingness pattern similarity (FIT ON TRAIN ONLY)
# =========================================================================== #
@dataclass
class MissingnessPatternEncoder:
    """Cosine similarity of each row's AL-missingness vector to the per-class
    missingness centroid learned on the training fold (medium-impact, Phase 2.1).

    Pathogenic and benign variants have systematically different missingness
    footprints (curation status); this encodes "how pathogenic-shaped is this
    row's missingness pattern" without leaking individual labels.
    """

    al_cols: List[str] = None
    centroid_pos: np.ndarray = None
    centroid_neg: np.ndarray = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, schema: FeatureSchema) -> "MissingnessPatternEncoder":
        self.al_cols = list(schema.al_cols)
        miss = X_train[self.al_cols].isna().astype(float).to_numpy()
        y = np.asarray(y_train).astype(int)
        self.centroid_pos = miss[y == 1].mean(axis=0) if (y == 1).any() else np.zeros(miss.shape[1])
        self.centroid_neg = miss[y == 0].mean(axis=0) if (y == 0).any() else np.zeros(miss.shape[1])
        return self

    @staticmethod
    def _cosine(M: np.ndarray, c: np.ndarray) -> np.ndarray:
        cn = np.linalg.norm(c)
        mn = np.linalg.norm(M, axis=1)
        denom = mn * cn
        sim = (M @ c) / np.where(denom == 0, np.nan, denom)
        return sim

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.centroid_pos is not None, "Call fit() before transform()."
        out = X.copy()
        miss = X[self.al_cols].isna().astype(float).to_numpy()
        sim_pos = self._cosine(miss, self.centroid_pos)
        sim_neg = self._cosine(miss, self.centroid_neg)
        out["misspat_sim_pathogenic"] = sim_pos.astype("float32")
        out["misspat_sim_benign"] = sim_neg.astype("float32")
        out["misspat_sim_diff"] = (sim_pos - sim_neg).astype("float32")
        return out


# =========================================================================== #
# Phase 2.6 -- Interaction features (need stateless feats + EK aggregates)
# =========================================================================== #
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-terms combining pathogenicity scores, AA severity, frequency and
    annotation completeness. Computed AFTER EK aggregates exist."""
    out = df.copy()

    def g(col):  # safe getter -> NaN-filled series if a dependency is absent
        return out[col] if col in out else pd.Series(np.nan, index=out.index)

    ek_max = g("ek_max_ranked")
    ek_mean = g("ek_mean_ranked")
    grantham = g("grantham_score")
    af_log_max = g("af_log_max")
    af_max_raw = g("af_max_raw")
    dmg_ratio = g("damaging_vote_ratio")
    pct_missing = g("pct_missing")
    is_rare = g("is_rare")
    is_stop_gain = g("is_stop_gain")
    n_tools = g("n_tools_available")
    tier_enc = g("missingness_tier_encoded")

    out["int_ek_max_x_grantham"] = ek_max * grantham / aat.GRANTHAM_MAX
    out["int_aflog_x_dmgratio"] = af_log_max * dmg_ratio
    out["int_ekmax_over_af"] = ek_max / (af_max_raw + EPS)
    out["int_grantham_x_dmgratio"] = grantham * dmg_ratio
    out["int_ekmean_x_completeness"] = ek_mean * (1.0 - pct_missing)
    out["int_rare_x_ekmax"] = is_rare * ek_max
    out["int_stopgain_x_ekmax"] = is_stop_gain * ek_max
    out["int_ntools_x_dmgratio"] = n_tools * dmg_ratio
    out["int_tier_x_ekmean"] = tier_enc * ek_mean

    # stop-gain NMD-escape proxy: rank of annotation completeness *within*
    # stop-gains (well-annotated stop-gains may be NMD-escaped / less severe)
    sg = out["is_stop_gain"] == 1
    out["stop_gain_severity_proxy"] = np.nan
    if sg.any():
        out.loc[sg, "stop_gain_severity_proxy"] = (
            out.loc[sg, "n_missing_total"].rank(pct=True).astype("float32")
        )
    return out


# =========================================================================== #
# Per-fold fit / transform orchestrators
# =========================================================================== #
@dataclass
class FittedFoldState:
    """Bundle of every object fitted on a single training fold."""
    ek_normalizer: EKRankNormalizer
    miss_pattern: MissingnessPatternEncoder
    schema: FeatureSchema


def fit_fold(train_stateless: pd.DataFrame, y_train: pd.Series, schema: FeatureSchema) -> FittedFoldState:
    """Fit all train-only transforms (EK directionality/rank, missingness pattern)
    on the training rows of one fold. Input must already have stateless features."""
    ek = EKRankNormalizer().fit(train_stateless, y_train, schema)
    mp = MissingnessPatternEncoder().fit(train_stateless, y_train, schema)
    return FittedFoldState(ek_normalizer=ek, miss_pattern=mp, schema=schema)


def transform_split(stateless_df: pd.DataFrame, fitted: FittedFoldState) -> pd.DataFrame:
    """Apply the fitted fold transforms + EK aggregates + interactions to any
    split (train / val / test). No fitting happens here -> leakage-free."""
    out = fitted.ek_normalizer.transform(stateless_df)
    out = add_ek_aggregate_features(out, fitted.schema)
    out = fitted.miss_pattern.transform(out)
    out = add_interaction_features(out)
    return out
