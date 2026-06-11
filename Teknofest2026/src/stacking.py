"""
stacking.py
===========
Phase 6 -- the stacking ensemble.

Pipeline (exactly the plan's order)
-----------------------------------
1. **Collect OOF** (6.1) -- assemble the leakage-free OOF probability columns from
   every base model (``XGB_tuned``, ``LGBM_tuned``, ``CatBoost_tuned``,
   ``Skeleton_head``, ``Dense_head``, ...). Each is the full development length.
2. **Calibrate** (6.2) -- fit an ``IsotonicRegression(out_of_bounds='clip')`` per
   base model *on its own OOF probabilities*, then apply it. Calibrators are
   persisted to ``outputs/models/calibrators/`` for test-time reuse.
3. **Diversity check** (6.3) -- pairwise Pearson correlation of calibrated OOF.
4. **Rank-normalise** (6.3) -- ``rankdata / n`` per model to erase scale differences.
5. **Meta-features** (6.4) -- concatenate the calibrated+ranked OOF columns with
   the raw meta-features: source flags (``is_master/kanser/pah/cftr``) and the
   ordinal ``missingness_tier_encoded``.
6. **Level-1 meta-learner** (6.4) -- Logistic Regression with its own nested CV
   (C tuned), producing the honest stacked OOF and a final model for test time.

A fitted :class:`StackingModel` bundles the calibrators, base-model order, meta
columns, scaler and meta-learner, so test/canary scoring is a single call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CALIBRATORS_DIR,
    LABEL_COL,
    SEED,
    SOURCES,
    SOURCE_COL,
    STACKING_DIR,
)

# Raw meta-features appended to the OOF columns (Phase 6.4).
META_SOURCE_FLAGS = [f"is_{s}" for s in SOURCES]
META_TIER_COL = "missingness_tier_encoded"
META_RAW_COLS = META_SOURCE_FLAGS + [META_TIER_COL]


# =========================================================================== #
# 6.1  OOF collection
# =========================================================================== #
def collect_oof(oof_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Stack base-model OOF arrays into a ``(n, n_models)`` matrix + fixed name order.

    Verifies equal length and absence of NaN (every fold must have filled its slice).
    """
    names = list(oof_dict.keys())
    lengths = {len(oof_dict[n]) for n in names}
    if len(lengths) != 1:
        raise ValueError(f"OOF arrays differ in length: {lengths}")
    M = np.column_stack([np.asarray(oof_dict[n], dtype=float) for n in names])
    if np.isnan(M).any():
        bad = [n for n in names if np.isnan(oof_dict[n]).any()]
        raise ValueError(f"NaN in OOF for models: {bad} (folds did not fully cover dev).")
    return M, names


# =========================================================================== #
# 6.2  Isotonic calibration of OOF probabilities
# =========================================================================== #
@dataclass
class IsotonicCalibratorBundle:
    """One ``IsotonicRegression`` per base model, fitted on that model's OOF."""

    calibrators: Dict[str, object] = field(default_factory=dict)

    def fit(self, oof_dict: Dict[str, np.ndarray], y: Sequence[int]) -> "IsotonicCalibratorBundle":
        from sklearn.isotonic import IsotonicRegression
        y = np.asarray(y).astype(float)
        for name, p in oof_dict.items():
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(np.asarray(p, dtype=float), y)
            self.calibrators[name] = iso
        return self

    def transform(self, pred_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply each model's calibrator (used for OOF, canary and test alike)."""
        out = {}
        for name, p in pred_dict.items():
            iso = self.calibrators[name]
            out[name] = iso.predict(np.asarray(p, dtype=float))
        return out

    def save(self, out_dir: str = CALIBRATORS_DIR) -> None:
        import joblib
        os.makedirs(out_dir, exist_ok=True)
        for name, iso in self.calibrators.items():
            joblib.dump(iso, os.path.join(out_dir, f"{name}_isotonic.pkl"))

    @classmethod
    def load(cls, names: Sequence[str], out_dir: str = CALIBRATORS_DIR) -> "IsotonicCalibratorBundle":
        import joblib
        bundle = cls()
        for name in names:
            bundle.calibrators[name] = joblib.load(os.path.join(out_dir, f"{name}_isotonic.pkl"))
        return bundle


def calibrate_oof(oof_dict: Dict[str, np.ndarray], y: Sequence[int],
                  save: bool = True) -> Tuple[Dict[str, np.ndarray], IsotonicCalibratorBundle]:
    """Fit + apply isotonic calibration on the OOF probabilities (Phase 6.2)."""
    bundle = IsotonicCalibratorBundle().fit(oof_dict, y)
    calibrated = bundle.transform(oof_dict)
    if save:
        bundle.save()
    return calibrated, bundle


def calibration_report(oof_dict: Dict[str, np.ndarray], y: Sequence[int],
                       n_bins: int = 10) -> pd.DataFrame:
    """Expected Calibration Error per model (pre/post sanity; lower = better)."""
    from sklearn.calibration import calibration_curve
    y = np.asarray(y).astype(int)
    rows = []
    for name, p in oof_dict.items():
        p = np.asarray(p, dtype=float)
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        rows.append({"model": name, "ece": ece})
    return pd.DataFrame(rows).sort_values("ece").reset_index(drop=True)


# =========================================================================== #
# 6.3  Diversity check + rank-normalisation
# =========================================================================== #
def diversity_report(oof_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Pearson correlation of OOF predictions (want < 0.95)."""
    df = pd.DataFrame(oof_dict)
    return df.corr()


def rank_normalize(arr: np.ndarray) -> np.ndarray:
    """``rankdata / n`` -> values in (0, 1], scale-free (Phase 6.3)."""
    arr = np.asarray(arr, dtype=float)
    return rankdata(arr) / len(arr)


def rank_normalize_dict(pred_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Rank-normalise every column independently (ranked within the given set)."""
    return {name: rank_normalize(p) for name, p in pred_dict.items()}


# =========================================================================== #
# 6.4  Meta-feature assembly
# =========================================================================== #
def extract_meta_frame(X: pd.DataFrame, meta_cols: Sequence[str] = META_RAW_COLS) -> pd.DataFrame:
    """Pull the raw meta-features (source flags + missingness tier) from a feature
    matrix; any column absent is filled with 0 so the stacker shape stays stable."""
    out = pd.DataFrame(index=X.index)
    for c in meta_cols:
        out[c] = X[c].to_numpy() if c in X.columns else 0.0
    return out.astype("float32")


def build_meta_matrix(
    base_dict_ranked: Dict[str, np.ndarray],
    meta_frame: pd.DataFrame,
    base_order: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Concatenate calibrated+ranked base OOF columns (in ``base_order``) with the
    raw meta-features -> ``(matrix, column_names)`` (Phase 6.4)."""
    base = np.column_stack([base_dict_ranked[n] for n in base_order])
    meta = meta_frame.to_numpy(dtype=float)
    M = np.hstack([base, meta])
    names = list(base_order) + list(meta_frame.columns)
    return M, names


# =========================================================================== #
# 6.4  Level-1 meta-learner (Logistic Regression, nested CV on C)
# =========================================================================== #
def _make_logreg(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                  solver="lbfgs", random_state=SEED)),
    ])


def _inner_splits(y: np.ndarray, folds: Optional[List[dict]], seed: int):
    """Yield (train_idx, val_idx). Reuse the group-aware outer folds when provided
    (true nested CV); else a fresh StratifiedKFold on the meta matrix."""
    if folds is not None:
        for f in folds:
            yield np.asarray(f["train"], int), np.asarray(f["val"], int)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in skf.split(np.zeros(len(y)), y):
            yield tr, va


def train_meta_learner(
    meta_X: np.ndarray,
    y: Sequence[int],
    *,
    folds: Optional[List[dict]] = None,
    C_grid: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    seed: int = SEED,
    verbose: bool = True,
) -> Tuple[Pipeline, np.ndarray, Dict]:
    """Tune ``C`` by nested-CV AUC, return ``(final_model, stacked_oof, info)``.

    ``stacked_oof`` is the honest out-of-fold prediction of the meta-learner at the
    best ``C``; ``final_model`` is refit on all rows for test-time scoring.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    best = {"C": None, "auc": -np.inf, "oof": None}
    for C in C_grid:
        oof = np.full(n, np.nan, dtype=float)
        for tr, va in _inner_splits(y, folds, seed):
            mdl = _make_logreg(C).fit(meta_X[tr], y[tr])
            oof[va] = mdl.predict_proba(meta_X[va])[:, 1]
        auc = roc_auc_score(y, oof)
        if verbose:
            print(f"[meta] C={C:<6} nested-CV AUC={auc:.4f}")
        if auc > best["auc"]:
            best = {"C": C, "auc": auc, "oof": oof}

    final_model = _make_logreg(best["C"]).fit(meta_X, y)
    info = {"best_C": best["C"], "nested_cv_auc": best["auc"],
            "C_grid": list(C_grid), "n_meta_features": meta_X.shape[1]}
    if verbose:
        print(f"[meta] selected C={best['C']} | nested-CV AUC={best['auc']:.4f}")
    return final_model, best["oof"], info


def rank_average_baseline(base_dict_ranked: Dict[str, np.ndarray]) -> np.ndarray:
    """Parameter-free rank-average of the base models (a stacking sanity baseline)."""
    M = np.column_stack(list(base_dict_ranked.values()))
    return M.mean(axis=1)


# =========================================================================== #
# Fitted bundle for test-time scoring + end-to-end orchestrator
# =========================================================================== #
@dataclass
class StackingModel:
    """Everything needed to score new rows: calibrators -> rank-norm -> meta-learner."""

    base_order: List[str]
    meta_cols: List[str]
    calibrators: IsotonicCalibratorBundle
    meta_learner: Pipeline

    def predict_proba(self, base_pred_dict: Dict[str, np.ndarray],
                      X_meta: pd.DataFrame) -> np.ndarray:
        """Score an arbitrary set (test/canary). Ranks are computed *within* the
        provided set, matching the rank-averaging stacking convention."""
        cal = self.calibrators.transform(base_pred_dict)
        ranked = rank_normalize_dict(cal)
        meta_frame = extract_meta_frame(X_meta, self.meta_cols[len(self.base_order):]) \
            if len(self.meta_cols) > len(self.base_order) else pd.DataFrame(index=X_meta.index)
        M, _ = build_meta_matrix(ranked, meta_frame, self.base_order)
        return self.meta_learner.predict_proba(M)[:, 1]

    def save(self, out_dir: str = STACKING_DIR) -> None:
        import joblib
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.meta_learner, os.path.join(out_dir, "meta_learner.pkl"))
        self.calibrators.save()
        with open(os.path.join(out_dir, "stacking_config.json"), "w") as f:
            json.dump({"base_order": self.base_order, "meta_cols": self.meta_cols}, f, indent=2)


@dataclass
class StackingResult:
    model: StackingModel
    stacked_oof: np.ndarray
    info: Dict
    diversity: pd.DataFrame
    calibration: pd.DataFrame
    extra_pred: Optional[np.ndarray] = None

    def save(self, out_dir: str = STACKING_DIR) -> None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "stacked_oof.npy"), self.stacked_oof)
        if self.extra_pred is not None:
            np.save(os.path.join(out_dir, "stacked_extra.npy"), self.extra_pred)
        self.diversity.to_csv(os.path.join(out_dir, "oof_diversity_corr.csv"))
        self.calibration.to_csv(os.path.join(out_dir, "calibration_ece.csv"), index=False)
        with open(os.path.join(out_dir, "stacking_metrics.json"), "w") as f:
            json.dump(self.info, f, indent=2)
        self.model.save(out_dir)
        print(f"[stacking] artefacts -> {out_dir}")


def run_stacking(
    oof_dict: Dict[str, np.ndarray],
    y: Sequence[int],
    X_dev: pd.DataFrame,
    *,
    source: Optional[np.ndarray] = None,
    folds: Optional[List[dict]] = None,
    extra_pred_dict: Optional[Dict[str, np.ndarray]] = None,
    X_extra: Optional[pd.DataFrame] = None,
    meta_raw_cols: Sequence[str] = META_RAW_COLS,
    save: bool = True,
    verbose: bool = True,
) -> StackingResult:
    """End-to-end Phase 6: calibrate -> diversity -> rank-norm -> meta-features
    (+ source flags + tier) -> nested-CV Logistic-Regression meta-learner.

    If ``extra_pred_dict`` / ``X_extra`` are given (canary or test base preds),
    the fitted stack also scores them.
    """
    y = np.asarray(y).astype(int)
    _, base_order = collect_oof(oof_dict)                       # 6.1 validate + order

    calibrated, bundle = calibrate_oof(oof_dict, y, save=save)  # 6.2
    cal_report = calibration_report(oof_dict, y)
    div = diversity_report(calibrated)                          # 6.3
    if verbose:
        mx = div.where(~np.eye(len(div), dtype=bool)).abs().max().max()
        print(f"[stacking] base models: {base_order}")
        print(f"[stacking] max pairwise |corr| (calibrated OOF) = {mx:.3f} (want < 0.95)")

    ranked = rank_normalize_dict(calibrated)                    # 6.3
    meta_frame = extract_meta_frame(X_dev, meta_raw_cols)       # 6.4
    meta_X, meta_cols = build_meta_matrix(ranked, meta_frame, base_order)

    meta_learner, stacked_oof, info = train_meta_learner(       # 6.4
        meta_X, y, folds=folds, seed=SEED, verbose=verbose)

    # honest stacked-OOF diagnostics (global + per source)
    info["stacked_oof_auc"] = float(roc_auc_score(y, stacked_oof))
    info["base_order"] = base_order
    info["rank_average_baseline_auc"] = float(
        roc_auc_score(y, rank_average_baseline(ranked)))
    if source is not None:
        src = np.asarray(source).astype(str)
        info["stacked_oof_auc_per_source"] = {
            s: (float(roc_auc_score(y[src == s], stacked_oof[src == s]))
                if len(np.unique(y[src == s])) > 1 else float("nan"))
            for s in SOURCES if (src == s).any()
        }
    if verbose:
        print(f"[stacking] stacked OOF AUC = {info['stacked_oof_auc']:.4f} "
              f"(rank-avg baseline {info['rank_average_baseline_auc']:.4f})")
        if "stacked_oof_auc_per_source" in info:
            print(f"[stacking] per-source stacked AUC = "
                  f"{ {k: round(v, 4) for k, v in info['stacked_oof_auc_per_source'].items()} }")

    model = StackingModel(base_order=base_order, meta_cols=meta_cols,
                          calibrators=bundle, meta_learner=meta_learner)

    extra_pred = None
    if extra_pred_dict is not None and X_extra is not None:
        extra_pred = model.predict_proba(extra_pred_dict, X_extra)

    result = StackingResult(model=model, stacked_oof=stacked_oof, info=info,
                            diversity=div, calibration=cal_report, extra_pred=extra_pred)
    if save:
        result.save()
    return result
