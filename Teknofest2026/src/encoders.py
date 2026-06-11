"""
encoders.py
===========
Phase 3 of the plan: encoding & missing-value strategy. All encoders follow a
strict fit-on-training-fold / transform-on-everything-else protocol so they are
leakage-free.

What lives here
---------------
* **Native-NaN tree strategy (3.1-A, primary):** ``assemble_tree_matrix`` returns
  the numeric feature matrix with NaN *preserved* (XGBoost ``tree_method='hist'``
  and LightGBM learn the optimal default split direction for missing values).
* **Sentinel / fold-median strategies (3.1-B, secondary experiments):**
  ``sentinel_impute`` and ``fold_median_impute`` (+ ``{col}_was_missing`` flags).
* **K-fold target encoding with adaptive noise (3.3):** ``target_encode_fold``
  (the plan's reference function, verbatim behaviour) and the multi-column,
  persistable ``KFoldTargetEncoder`` class.
* **One-hot encoding (3.2):** ``OneHotEncoderWrapper`` for the genotype / region
  CAT columns (small, safe cardinality).
* **Stateless AA label encoding (3.2):** ``label_encode_amino_acids``.
* **Probability calibration (3.4):** Isotonic / Platt helpers (fitted later, in
  Phase 6, but defined here as the plan places calibration in Phase 3).
* **Fold orchestrator:** ``encode_fold`` ties the above together for one CV fold.

Leakage warning (carried over from the plan, Phases 1.3 / 3.2)
--------------------------------------------------------------
Target-encoding a CAT column that secretly encodes prior expert pathogenicity
assessment will inflate CV and collapse on test. The Phase-1 audit found CAT_1 /
CAT_2 are *population* labels (gnomAD / All-of-Us ancestry) -> safe to target
encode. CAT_3/4/5 are genotypes -> one-hot. CAT_6 is a genomic-region mask
-> one-hot. Re-run the crosstab audit if the data is ever refreshed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import aa_tables as aat
from .config import (
    AA_PAIR_COL,
    CAT_ENCODING_STRATEGY,
    LABEL_COL,
    NON_FEATURE_COLS,
    SEED,
    SOURCE_COL,
    TE_MAPPINGS_DIR,
)

NA_TOKEN = "__NA__"  # explicit category for missing categorical values

# Non-numeric / helper columns that must never enter the tree feature matrix.
_STRING_HELPERS = [
    "missingness_tier", "grantham_category", "tool_coverage_profile",
    "AA_1", "AA_2", AA_PAIR_COL,
]


# =========================================================================== #
# 3.1-A  Native-NaN strategy for tree models (PRIMARY)
# =========================================================================== #
def get_tree_feature_columns(df: pd.DataFrame, extra_drop: Optional[List[str]] = None) -> List[str]:
    """Return the numeric feature columns for XGBoost/LightGBM.

    We keep every numeric column (raw AL/EK scores, engineered features, encoded
    categoricals, one-hot dummies) and drop bookkeeping/string/raw-categorical
    columns. NaNs are deliberately preserved -- no imputation.
    """
    drop = set(NON_FEATURE_COLS) | set(_STRING_HELPERS) | set(CAT_ENCODING_STRATEGY.keys())
    if extra_drop:
        drop |= set(extra_drop)
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def assemble_tree_matrix(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Numeric feature matrix with NaN preserved (ready for hist/GPU tree models)."""
    cols = feature_cols if feature_cols is not None else get_tree_feature_columns(df)
    return df[cols].astype("float32")


# =========================================================================== #
# 3.1-B  Secondary missing-value experiments (for Phase 9.4 ablation)
# =========================================================================== #
def sentinel_impute(X: pd.DataFrame, sentinel: float = -999.0) -> pd.DataFrame:
    """Replace NaN with an out-of-distribution sentinel (tree models split on it)."""
    return X.fillna(sentinel)


@dataclass
class FoldMedianImputer:
    """Median imputation fitted on the training fold ONLY, with explicit
    ``{col}_was_missing`` indicator columns (never use a global median)."""

    medians: Dict[str, float] = None
    add_indicators: bool = True

    def fit(self, X_train: pd.DataFrame) -> "FoldMedianImputer":
        self.medians = X_train.median(numeric_only=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.add_indicators:
            # build all indicator columns at once (avoids DataFrame fragmentation)
            ind = X.isna().astype("int8")
            ind.columns = [f"{c}_was_missing" for c in X.columns]
            out = pd.concat([out, ind], axis=1)
        out = out.fillna(self.medians)
        return out


# =========================================================================== #
# 3.3  K-fold target encoding -- plan reference function (adaptive noise)
# =========================================================================== #
def target_encode_fold(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    col: str,
    y_train: pd.Series,
    n_folds: int = 5,
    smoothing: float = 10.0,
    noise_sigma: Optional[float] = None,
    seed: int = SEED,
):
    """Leak-proof K-fold target encoding of one column (plan section 3.3).

    Returns ``(encoded_train, encoded_val, mapping_full)``:
      * ``encoded_train`` -- OOF-encoded training column (inner K-fold + noise)
      * ``encoded_val``   -- val/test column encoded with the full-train mapping
      * ``mapping_full``  -- {category: smoothed_mean} (save for test-time use)

    Smoothing shrinks rare categories toward the global mean. Noise is adaptive
    (sigma = 0.01 / sqrt(count)) so rare categories get more regularisation,
    common ones almost none -- the ML-Engineer fix over a uniform sigma.
    """
    rng = np.random.RandomState(seed)
    Xtr = X_train[col].reset_index(drop=True)
    ytr = pd.Series(np.asarray(y_train), dtype=float).reset_index(drop=True)
    global_mean = ytr.mean()

    inner_kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    encoded_train = np.full(len(Xtr), global_mean, dtype=float)

    for inner_tr, inner_va in inner_kf.split(Xtr, ytr):
        stats = ytr.iloc[inner_tr].groupby(Xtr.iloc[inner_tr]).agg(["sum", "count"])
        stats["mean"] = stats["sum"] / stats["count"]
        stats["smoothed"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)

        mapping = stats["smoothed"].to_dict()
        count_map = stats["count"].to_dict()
        vals = Xtr.iloc[inner_va].map(mapping).fillna(global_mean).to_numpy(dtype=float)

        if noise_sigma is None:  # adaptive noise ~ 1/sqrt(count)
            counts = Xtr.iloc[inner_va].map(count_map).fillna(1.0).to_numpy(dtype=float)
            sigma = 0.01 / np.sqrt(np.maximum(counts, 1.0))
            vals = vals + rng.normal(0.0, 1.0, size=len(vals)) * sigma
        elif noise_sigma > 0:
            vals = vals + rng.normal(0.0, noise_sigma, size=len(vals))
        encoded_train[inner_va] = vals

    # full-train mapping for val/test (smoothed, no noise)
    stats_full = ytr.groupby(Xtr).agg(["sum", "count"])
    stats_full["mean"] = stats_full["sum"] / stats_full["count"]
    stats_full["smoothed"] = (stats_full["count"] * stats_full["mean"] + smoothing * global_mean) / (stats_full["count"] + smoothing)
    mapping_full = stats_full["smoothed"].to_dict()

    encoded_val = X_val[col].map(mapping_full).fillna(global_mean).to_numpy(dtype=float)
    return encoded_train, encoded_val, mapping_full


# =========================================================================== #
# 3.3  Multi-column, persistable target encoder
# =========================================================================== #
@dataclass
class KFoldTargetEncoder:
    """Target-encode several columns with one fit, with save/load of mappings.

    NaN handling: missing categories are encoded as their own ``__NA__`` group
    (population-column missingness is informative and this stays leak-safe),
    rather than being forced to the global mean.
    """

    cols: List[str] = field(default_factory=list)
    n_inner_folds: int = 5
    smoothing: float = 10.0
    noise_sigma: Optional[float] = None        # None -> adaptive 0.01/sqrt(count)
    na_as_category: bool = True
    seed: int = SEED
    suffix: str = "_te"

    mappings_: Dict[str, Dict] = field(default_factory=dict)
    global_mean_: float = None

    def _prep(self, s: pd.Series) -> pd.Series:
        return s.fillna(NA_TOKEN).astype(str) if self.na_as_category else s

    def fit_transform_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Return the OOF-encoded (noisy) training columns and store full mappings."""
        out = X_train.copy()
        y = pd.Series(np.asarray(y_train), dtype=float)
        self.global_mean_ = float(y.mean())
        for col in self.cols:
            Xt = self._prep(X_train[col])
            enc_tr, _, mapping = target_encode_fold(
                pd.DataFrame({col: Xt}), pd.DataFrame({col: Xt}), col, y,
                n_folds=self.n_inner_folds, smoothing=self.smoothing,
                noise_sigma=self.noise_sigma, seed=self.seed,
            )
            out[f"{col}{self.suffix}"] = enc_tr.astype("float32")
            self.mappings_[col] = mapping
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode val/test with the stored full-train mapping (no noise)."""
        assert self.mappings_, "Call fit_transform_train() first."
        out = X.copy()
        for col in self.cols:
            Xc = self._prep(X[col])
            out[f"{col}{self.suffix}"] = (
                Xc.map(self.mappings_[col]).fillna(self.global_mean_).astype("float32")
            )
        return out

    def save(self, fold_id: int, out_dir: str = TE_MAPPINGS_DIR) -> None:
        os.makedirs(out_dir, exist_ok=True)
        for col, mapping in self.mappings_.items():
            payload = {"global_mean": self.global_mean_,
                       "smoothing": self.smoothing, "mapping": mapping}
            with open(os.path.join(out_dir, f"fold_{fold_id}_{col}.json"), "w") as f:
                json.dump(payload, f)


# =========================================================================== #
# 3.2  One-hot encoding for genotype / region CAT columns
# =========================================================================== #
@dataclass
class OneHotEncoderWrapper:
    """One-hot encode small-cardinality CAT columns. Categories are learned on
    the training fold; unseen categories at transform time map to all-zeros.
    NaN becomes an explicit ``__NA__`` column."""

    cols: List[str] = field(default_factory=list)
    categories_: Dict[str, List[str]] = field(default_factory=dict)

    def fit(self, X_train: pd.DataFrame) -> "OneHotEncoderWrapper":
        for col in self.cols:
            cats = pd.Index(X_train[col].fillna(NA_TOKEN).astype(str).unique()).tolist()
            self.categories_[col] = sorted(cats)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.cols:
            s = X[col].fillna(NA_TOKEN).astype(str)
            for cat in self.categories_[col]:
                safe = str(cat).replace("/", "_").replace(" ", "_").replace("&", "_")
                out[f"{col}_oh_{safe}"] = (s == cat).astype("int8")
        return out


# =========================================================================== #
# 3.2  Stateless amino-acid label encoding
# =========================================================================== #
# Fixed, deterministic label map (standard AAs, then stop, then NA). Stateless,
# so it needs no fitting and is identical across folds/test.
_AA_LABEL_MAP = {a: i + 1 for i, a in enumerate(aat.STANDARD_AA)}
_AA_LABEL_MAP[aat.STOP_CODON] = len(_AA_LABEL_MAP) + 1


def label_encode_amino_acids(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer label encodings for AA_1 / AA_2 (0 = missing/non-standard).

    The physicochemical suite carries the real signal; these labels just give the
    tree models a compact categorical handle (plan 3.2)."""
    out = df.copy()
    for col in ("AA_1", "AA_2"):
        if col in df.columns:
            out[f"{col.lower()}_label"] = df[col].map(_AA_LABEL_MAP).fillna(0).astype("int16")
    return out


# =========================================================================== #
# 3.4  Probability calibration utilities (applied in Phase 6)
# =========================================================================== #
def fit_calibrator(oof_proba: np.ndarray, y_true: np.ndarray, method: str = "isotonic"):
    """Fit a 1-D probability calibrator on OOF predictions.

    method='isotonic' (non-parametric, better for large n) or 'platt' (sigmoid,
    better for small n). Returns a fitted object exposing ``.predict(p)``.
    """
    oof = np.asarray(oof_proba, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(oof, y)
        return cal
    elif method == "platt":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(oof.reshape(-1, 1), y)

        class _Platt:
            def predict(self, p):
                return lr.predict_proba(np.asarray(p, dtype=float).reshape(-1, 1))[:, 1]

        return _Platt()
    raise ValueError(f"Unknown calibration method: {method}")


def calibration_curve_points(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10):
    """Reliability-diagram points (fraction_of_positives, mean_predicted_value)."""
    from sklearn.calibration import calibration_curve
    return calibration_curve(np.asarray(y_true), np.asarray(proba), n_bins=n_bins)


# =========================================================================== #
# Fold-level encoding orchestrator
# =========================================================================== #
@dataclass
class FoldEncoders:
    target_encoder: KFoldTargetEncoder
    onehot_encoder: OneHotEncoderWrapper


def encode_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train: pd.Series,
    cat_strategy: Optional[Dict[str, str]] = None,
    target_encode_source: bool = True,
    target_encode_aa_pair: bool = True,
    smoothing: float = 10.0,
    fold_id: Optional[int] = None,
):
    """Encode all categoricals for ONE fold and return ``(train_enc, val_enc, encoders)``.

    * target-encode CAT columns tagged "target" + ``aa_pair`` (+ optionally ``source``)
    * one-hot CAT columns tagged "onehot"
    * stateless AA label encoding
    All fits use the training fold only. NaN is preserved on numeric columns
    (native-NaN tree strategy) -- only categoricals are encoded here.
    """
    cat_strategy = cat_strategy or CAT_ENCODING_STRATEGY
    target_cols = [c for c, s in cat_strategy.items() if s == "target" and c in train_df.columns]
    onehot_cols = [c for c, s in cat_strategy.items() if s == "onehot" and c in train_df.columns]
    if target_encode_aa_pair and AA_PAIR_COL in train_df.columns:
        target_cols.append(AA_PAIR_COL)
    if target_encode_source and SOURCE_COL in train_df.columns:
        target_cols.append(SOURCE_COL)

    te = KFoldTargetEncoder(cols=target_cols, smoothing=smoothing)
    train_enc = te.fit_transform_train(train_df, y_train)
    val_enc = te.transform(val_df)

    ohe = OneHotEncoderWrapper(cols=onehot_cols).fit(train_df)
    train_enc = ohe.transform(train_enc)
    val_enc = ohe.transform(val_enc)

    train_enc = label_encode_amino_acids(train_enc)
    val_enc = label_encode_amino_acids(val_enc)

    if fold_id is not None:
        te.save(fold_id)

    return train_enc, val_enc, FoldEncoders(target_encoder=te, onehot_encoder=ohe)
