"""
cv_runner.py
============
Phase 4 (Model Training) + the CV plumbing Phase 5 (Feature Selection) reuses.

Design goals
------------
* **One fold file, ever.** Every routine loads the *exact* positional indices from
  ``outputs/folds/fold_indices.json`` (Phase 0.4). Folds are never regenerated.
* **GPU + native-NaN.** Every learner spreads the centralized GPU kwargs from
  ``config.model_gpu_params()`` and receives the feature matrix with ``NaN``
  preserved -- XGBoost (``tree_method='hist'``), LightGBM and CatBoost all learn
  the optimal default split direction for missing values. **No imputation.**
* **Uniform learner protocol.** ``XGBLearner`` / ``LGBMLearner`` / ``CatBoostLearner``
  (and a CPU ``HistGBLearner`` fallback) all expose the same
  ``fit(X_tr, y_tr, X_va, y_va) -> self`` / ``predict_proba(X) -> P(pathogenic)``
  surface, so ``run_cv`` is model-agnostic.
* **Stacking-ready outputs.** ``run_cv`` returns leakage-free OOF probabilities
  (full development length), per-fold + fold-averaged predictions for any extra
  matrix (canary/test), and per-source AUC/F1/MCC.
* **Missingness heads (Phase 4.6).** ``run_cv`` accepts a ``train_row_mask`` so a
  base model can be *trained* only on skeleton (or dense) rows while still
  predicting OOF for **every** validation row -- giving the meta-learner a dense
  tier-specialised column.
* **Optuna (Phase 4b/5).** ``tune_model`` runs a ``MedianPruner`` study on the
  *selected* feature subset only, maximising mean CV AUC.

Heavy third-party imports (xgboost/lightgbm/catboost/optuna) are done lazily
inside the relevant call sites so this module imports cleanly on a CPU-only box.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    CANARY_FEATURES_PATH,
    DEV_FEATURES_PATH,
    FOLD_INDICES_PATH,
    LABEL_COL,
    MODELS_DIR,
    OOF_DIR,
    SEED,
    SOURCE_COL,
    SOURCES,
    model_gpu_params,
)

# A model-agnostic factory is a zero-arg callable returning a *fresh* learner.
LearnerFactory = Callable[[], "BaseLearner"]


# =========================================================================== #
# Data loading -- consume the Phase-3 materialised matrices + the fold file
# =========================================================================== #
def load_feature_matrix(path: str) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load a materialised feature parquet -> ``(X, y, source)``.

    ``X`` keeps every feature column with ``NaN`` preserved (float32, no impute).
    ``Label`` and ``source`` are split off and never enter ``X``.
    """
    df = pd.read_parquet(path)
    y = df[LABEL_COL].astype(int).reset_index(drop=True)
    source = df[SOURCE_COL].astype(str).to_numpy() if SOURCE_COL in df.columns else \
        np.array(["all"] * len(df))
    drop = [c for c in (LABEL_COL, SOURCE_COL) if c in df.columns]
    X = df.drop(columns=drop).reset_index(drop=True).astype("float32")
    return X, y, source


def load_development_matrix(path: str = DEV_FEATURES_PATH):
    """Development feature matrix (the only data every Phase-4 experiment trains on)."""
    return load_feature_matrix(path)


def load_canary_matrix(path: str = CANARY_FEATURES_PATH):
    """Canary feature matrix -- touched exactly once, at final evaluation."""
    return load_feature_matrix(path)


def load_folds(path: str = FOLD_INDICES_PATH) -> List[dict]:
    """Load the persisted StratifiedGroupKFold indices (positional into development)."""
    with open(path) as f:
        folds = json.load(f)
    return sorted(folds, key=lambda d: d["fold"])


def feature_columns(X: pd.DataFrame) -> List[str]:
    """Every column of an already-split feature matrix is a model feature."""
    return list(X.columns)


# =========================================================================== #
# Class imbalance (Phase 4.2) -- verify direction at runtime, never hardcode
# =========================================================================== #
def compute_scale_pos_weight(y: Sequence[int]) -> float:
    """``count_negative / count_positive``.

    Pathogenic (1) is the *majority* (~73%), so this is ~0.37 and DOWN-weights the
    majority class -- matching the plan's corrected direction.
    """
    y = np.asarray(y).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg) / float(max(pos, 1))


# =========================================================================== #
# Metrics (Phase 4.1) -- global + per-source
# =========================================================================== #
def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """ROC-AUC guarded against single-class slices (returns NaN there)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def binary_metrics(y_true, proba, threshold: float = 0.5) -> Dict[str, float]:
    """AUC (threshold-free) + F1/MCC/precision/recall at a fixed threshold."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    return {
        "auc": _safe_auc(y_true, proba),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def per_source_auc(y_true, proba, source: np.ndarray,
                   sources: Sequence[str] = SOURCES) -> Dict[str, float]:
    """AUC computed independently for each sub-dataset (master/kanser/pah/cftr)."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    source = np.asarray(source).astype(str)
    out: Dict[str, float] = {}
    for s in sources:
        m = source == s
        if m.sum() > 0:
            out[s] = _safe_auc(y_true[m], proba[m])
    return out


def summarize_metric(fold_metrics: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    """``(mean, std)`` of one metric across folds, ignoring NaN slices."""
    vals = np.array([fm[key] for fm in fold_metrics if key in fm], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


# =========================================================================== #
# Learner protocol + library wrappers (uniform fit/predict surface)
# =========================================================================== #
class BaseLearner:
    """Minimal interface ``run_cv`` relies on."""

    def fit(self, X_tr, y_tr, X_va, y_va) -> "BaseLearner":  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


# --- plan strong-default hyper-parameters (Phase 4.3 / 4.4 / 4.5) ---------- #
XGB_DEFAULT_PARAMS: Dict = dict(
    objective="binary:logistic", eval_metric="auc",
    max_depth=6, learning_rate=0.05, n_estimators=2000,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0,
)
LGBM_DEFAULT_PARAMS: Dict = dict(
    objective="binary", metric="auc", boosting_type="gbdt",
    num_leaves=63, learning_rate=0.05, n_estimators=2000,
    subsample=0.8, colsample_bytree=0.6, min_child_samples=20,
    is_unbalance=True, verbosity=-1,
)
CATBOOST_DEFAULT_PARAMS: Dict = dict(
    loss_function="Logloss", eval_metric="AUC",
    depth=6, learning_rate=0.05, iterations=2000, l2_leaf_reg=3,
    auto_class_weights="Balanced",
)
EARLY_STOPPING_ROUNDS = 100


class XGBLearner(BaseLearner):
    """XGBoost ``hist``/CUDA wrapper. NaN native; early stopping on the val fold."""

    def __init__(self, params: Dict, early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
        self.params = dict(params)
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_tr, y_tr, X_va, y_va):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            **self.params, early_stopping_rounds=self.early_stopping_rounds)
        self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class LGBMLearner(BaseLearner):
    """LightGBM wrapper (gbdt or dart). NaN native; early stopping via callbacks."""

    def __init__(self, params: Dict, early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
        self.params = dict(params)
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_tr, y_tr, X_va, y_va):
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        self.model = LGBMClassifier(**self.params)
        callbacks = [log_evaluation(0)]
        # DART does not support early stopping on the best iteration -> skip it.
        if self.params.get("boosting_type", "gbdt") != "dart":
            callbacks.insert(0, early_stopping(self.early_stopping_rounds, verbose=False))
        self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                       eval_metric="auc", callbacks=callbacks)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class CatBoostLearner(BaseLearner):
    """CatBoost wrapper on the numeric encoded matrix. NaN native; early stopping."""

    def __init__(self, params: Dict, early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
        self.params = dict(params)
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_tr, y_tr, X_va, y_va):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(
            **self.params, early_stopping_rounds=self.early_stopping_rounds, verbose=0)
        self.model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class HistGBLearner(BaseLearner):
    """sklearn ``HistGradientBoostingClassifier`` -- a CPU, dependency-light learner.

    Handles NaN natively too. Doubles as (a) a diverse non-XGBoost ensemble member
    and (b) the stand-in that lets the whole CV/stacking framework be smoke-tested
    without the GPU boosters installed.
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = dict(params or {})
        self.model = None

    def fit(self, X_tr, y_tr, X_va, y_va):
        from sklearn.ensemble import HistGradientBoostingClassifier
        kwargs = dict(random_state=SEED, early_stopping=True, validation_fraction=0.1)
        kwargs.update(self.params)
        try:
            self.model = HistGradientBoostingClassifier(class_weight="balanced", **kwargs)
            self.model.fit(X_tr, y_tr)
        except TypeError:  # older sklearn without class_weight
            self.model = HistGradientBoostingClassifier(**kwargs)
            self.model.fit(X_tr, y_tr)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# --- factories: bake in GPU kwargs + imbalance defaults, return zero-arg ---- #
def _merge_gpu(params: Dict, lib: str, use_gpu: Optional[bool]) -> Dict:
    out = dict(params)
    out.update(model_gpu_params(use_gpu)[lib])
    return out


def make_xgb_learner(params: Optional[Dict] = None, y: Optional[Sequence[int]] = None,
                     use_gpu: Optional[bool] = None,
                     early_stopping_rounds: int = EARLY_STOPPING_ROUNDS) -> LearnerFactory:
    base = dict(XGB_DEFAULT_PARAMS)
    if y is not None:
        base["scale_pos_weight"] = compute_scale_pos_weight(y)
    if params:
        base.update(params)
    base = _merge_gpu(base, "xgboost", use_gpu)
    return lambda: XGBLearner(base, early_stopping_rounds)


def make_lgbm_learner(params: Optional[Dict] = None, y: Optional[Sequence[int]] = None,
                      use_gpu: Optional[bool] = None,
                      early_stopping_rounds: int = EARLY_STOPPING_ROUNDS) -> LearnerFactory:
    base = dict(LGBM_DEFAULT_PARAMS)
    if params:
        base.update(params)
    base = _merge_gpu(base, "lightgbm", use_gpu)
    return lambda: LGBMLearner(base, early_stopping_rounds)


def make_catboost_learner(params: Optional[Dict] = None, y: Optional[Sequence[int]] = None,
                          use_gpu: Optional[bool] = None,
                          early_stopping_rounds: int = EARLY_STOPPING_ROUNDS) -> LearnerFactory:
    base = dict(CATBOOST_DEFAULT_PARAMS)
    if params:
        base.update(params)
    base = _merge_gpu(base, "catboost", use_gpu)
    return lambda: CatBoostLearner(base, early_stopping_rounds)


def make_histgb_learner(params: Optional[Dict] = None) -> LearnerFactory:
    return lambda: HistGBLearner(params)


# =========================================================================== #
# Phase 4.6 -- tier-specific feature subsets for the missingness heads
# =========================================================================== #
# Prefixes/exact names that survive on a "skeleton" row (>=337 missing -> only
# EK + AA carry signal). Everything else (raw AL flags, allele-freq aggregates,
# in-silico vote counts, genotype one-hots) is structurally absent there and is
# dropped so the skeleton head is not distracted by all-NaN columns.
_SKELETON_PREFIXES = (
    "EK_", "ek_",                       # conservation/composite scores (+ranked aggs)
    "aa1_", "aa2_", "delta_", "abs_delta_",  # physicochemical suite + deltas
    "blosum", "grantham",               # substitution matrices
    "n_missing", "pct_missing", "missingness_tier",  # missingness meta
    "block_", "n_blocks", "misspat_",   # block + pattern missingness
    "is_skeleton", "is_dense",          # tier flags
    "is_master", "is_kanser", "is_pah", "is_cftr",  # safe source flags
)
_SKELETON_EXACT = {
    "group_change", "is_same_group", "charge_direction_change", "max_abs_delta",
    "is_stop_gain", "is_stop_loss", "is_synonymous",
    "aa1_is_nonstandard", "aa2_is_nonstandard", "transition_compatible",
    "aa_1_label", "aa_2_label", "aa_pair_te", "source_te",
    "stop_gain_severity_proxy",
}


def tier_feature_cols(all_cols: Sequence[str], tier: str) -> List[str]:
    """Feature subset for a missingness-tier head.

    * ``tier='skeleton'`` -> EK + AA + missingness-meta features only (Phase 4.6).
    * ``tier='dense'``    -> the full feature space (dense rows are well annotated).
    """
    if tier == "dense":
        return list(all_cols)
    if tier != "skeleton":
        raise ValueError("tier must be 'skeleton' or 'dense'")
    keep = [c for c in all_cols
            if c in _SKELETON_EXACT or any(c.startswith(p) for p in _SKELETON_PREFIXES)]
    return keep


# =========================================================================== #
# Core CV loop (Phase 4.1) -- leakage-free OOF + per-source metrics
# =========================================================================== #
@dataclass
class CVResult:
    name: str
    oof: np.ndarray                       # P(pathogenic) for every development row
    fold_metrics: List[Dict[str, float]]  # per-fold global metrics (+ _src_<s> AUCs)
    oof_metrics: Dict[str, float]         # metrics on the concatenated OOF
    per_source_oof_auc: Dict[str, float]  # OOF AUC by sub-dataset
    feature_cols: List[str]
    models: List[BaseLearner] = field(default_factory=list)
    extra_pred: Optional[np.ndarray] = None  # fold-averaged preds on an extra matrix
    extra_pred_folds: Optional[np.ndarray] = None  # (n_folds, n_extra) raw per-fold

    def summary(self) -> Dict[str, float]:
        a_m, a_s = summarize_metric(self.fold_metrics, "auc")
        f_m, _ = summarize_metric(self.fold_metrics, "f1")
        m_m, _ = summarize_metric(self.fold_metrics, "mcc")
        return {"cv_auc_mean": a_m, "cv_auc_std": a_s,
                "cv_f1_mean": f_m, "cv_mcc_mean": m_m,
                "oof_auc": self.oof_metrics.get("auc", float("nan"))}

    def save(self, oof_dir: str = OOF_DIR, models_dir: str = MODELS_DIR,
             save_models: bool = False) -> None:
        os.makedirs(oof_dir, exist_ok=True)
        np.save(os.path.join(oof_dir, f"{self.name}_oof.npy"), self.oof)
        if self.extra_pred is not None:
            np.save(os.path.join(oof_dir, f"{self.name}_extra.npy"), self.extra_pred)
        mdir = os.path.join(models_dir, self.name)
        os.makedirs(mdir, exist_ok=True)
        payload = {"name": self.name, "feature_cols": self.feature_cols,
                   "fold_metrics": self.fold_metrics, "oof_metrics": self.oof_metrics,
                   "per_source_oof_auc": self.per_source_oof_auc, "summary": self.summary()}
        with open(os.path.join(mdir, "metrics.json"), "w") as f:
            json.dump(payload, f, indent=2)
        if save_models and self.models:
            import joblib
            for k, m in enumerate(self.models):
                joblib.dump(m, os.path.join(mdir, f"fold_{k}.joblib"))


def run_cv(
    factory: LearnerFactory,
    X: pd.DataFrame,
    y: pd.Series,
    source: np.ndarray,
    folds: List[dict],
    *,
    name: str = "model",
    feature_cols: Optional[Sequence[str]] = None,
    train_row_mask: Optional[np.ndarray] = None,
    X_extra: Optional[pd.DataFrame] = None,
    threshold: float = 0.5,
    keep_models: bool = True,
    verbose: bool = True,
) -> CVResult:
    """Train ``factory()`` across the persisted folds and collect leakage-free OOF.

    Parameters
    ----------
    factory        : zero-arg callable returning a fresh learner (one per fold).
    X, y, source   : development matrix / labels / sub-dataset tags (positional).
    folds          : list of ``{'fold', 'train', 'val'}`` positional index lists.
    feature_cols   : restrict to these columns (e.g. the selected subset). Default = all.
    train_row_mask : boolean mask over development rows; when given, a fold trains
                     ONLY on its train rows that are also ``True`` here (missingness
                     heads). OOF is still predicted for *all* validation rows.
    X_extra        : optional extra matrix (canary/test) predicted every fold and
                     averaged -> ``extra_pred`` (Phase 6.1 test-time averaging).
    """
    cols = list(feature_cols) if feature_cols is not None else feature_columns(X)
    Xf = X[cols]
    y = pd.Series(np.asarray(y).astype(int))
    source = np.asarray(source).astype(str)
    n = len(Xf)

    oof = np.full(n, np.nan, dtype=float)
    fold_metrics: List[Dict[str, float]] = []
    models: List[BaseLearner] = []
    extra_folds: List[np.ndarray] = [] if X_extra is not None else None
    Xe = X_extra[cols] if X_extra is not None else None

    for fold in folds:
        k = fold["fold"]
        tr = np.asarray(fold["train"], dtype=int)
        va = np.asarray(fold["val"], dtype=int)
        if train_row_mask is not None:
            tr = tr[train_row_mask[tr]]  # tier head: train only on in-tier rows

        model = factory().fit(Xf.iloc[tr], y.iloc[tr], Xf.iloc[va], y.iloc[va])
        proba_va = model.predict_proba(Xf.iloc[va])
        oof[va] = proba_va

        fm = binary_metrics(y.iloc[va], proba_va, threshold)
        for s, a in per_source_auc(y.iloc[va], proba_va, source[va]).items():
            fm[f"src_{s}_auc"] = a
        fold_metrics.append(fm)
        if keep_models:
            models.append(model)
        if Xe is not None:
            extra_folds.append(model.predict_proba(Xe))
        if verbose:
            n_tr = len(tr)
            print(f"[cv:{name}] fold {k}: n_train={n_tr} val_auc={fm['auc']:.4f} "
                  f"f1={fm['f1']:.4f} mcc={fm['mcc']:.4f}")

    oof_metrics = binary_metrics(y.to_numpy(), oof, threshold)
    ps = per_source_auc(y.to_numpy(), oof, source)
    extra_pred = np.mean(extra_folds, axis=0) if extra_folds else None
    extra_arr = np.asarray(extra_folds) if extra_folds else None

    if verbose:
        a_m, a_s = summarize_metric(fold_metrics, "auc")
        print(f"[cv:{name}] OOF AUC={oof_metrics['auc']:.4f} | folds {a_m:.4f}+/-{a_s:.4f} "
              f"| per-source { {k: round(v, 4) for k, v in ps.items()} }")

    return CVResult(
        name=name, oof=oof, fold_metrics=fold_metrics, oof_metrics=oof_metrics,
        per_source_oof_auc=ps, feature_cols=cols, models=models,
        extra_pred=extra_pred, extra_pred_folds=extra_arr,
    )


def cv_fold_aucs(
    factory: LearnerFactory,
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[dict],
    feature_cols: Optional[Sequence[str]] = None,
    report_cb: Optional[Callable[[int, float], bool]] = None,
) -> List[float]:
    """Lightweight per-fold val AUC list (used by Optuna; supports pruning).

    ``report_cb(step, running_mean_auc) -> should_prune``: if it returns ``True``
    the loop stops early and returns the AUCs gathered so far.
    """
    cols = list(feature_cols) if feature_cols is not None else feature_columns(X)
    Xf = X[cols]
    y = pd.Series(np.asarray(y).astype(int))
    aucs: List[float] = []
    for step, fold in enumerate(folds):
        tr = np.asarray(fold["train"], dtype=int)
        va = np.asarray(fold["val"], dtype=int)
        model = factory().fit(Xf.iloc[tr], y.iloc[tr], Xf.iloc[va], y.iloc[va])
        aucs.append(_safe_auc(y.iloc[va], model.predict_proba(Xf.iloc[va])))
        if report_cb is not None and report_cb(step, float(np.nanmean(aucs))):
            break
    return aucs


# =========================================================================== #
# Phase 4b / 5 -- Optuna tuning on the SELECTED feature subset (MedianPruner)
# =========================================================================== #
def _suggest_params(trial, kind: str) -> Dict:
    """Plan search spaces (Phase 4.3 / 4.4 / 4.5)."""
    if kind == "xgb":
        return dict(
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            max_delta_step=trial.suggest_float("max_delta_step", 0.0, 10.0),
        )
    if kind == "lgbm":
        return dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            path_smooth=trial.suggest_float("path_smooth", 0.0, 10.0),
        )
    if kind == "catboost":
        return dict(
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 10.0),
            border_count=trial.suggest_int("border_count", 32, 255),
        )
    raise ValueError(f"Unknown kind: {kind}")


def _make_factory(kind: str, params: Dict, y, use_gpu, early_stopping_rounds: int) -> LearnerFactory:
    if kind == "xgb":
        return make_xgb_learner(params, y=y, use_gpu=use_gpu,
                                early_stopping_rounds=early_stopping_rounds)
    if kind == "lgbm":
        return make_lgbm_learner(params, y=y, use_gpu=use_gpu,
                                 early_stopping_rounds=early_stopping_rounds)
    if kind == "catboost":
        return make_catboost_learner(params, y=y, use_gpu=use_gpu,
                                     early_stopping_rounds=early_stopping_rounds)
    raise ValueError(f"Unknown kind: {kind}")


def tune_model(
    kind: str,
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[dict],
    feature_cols: Sequence[str],
    *,
    n_trials: int = 300,
    timeout: Optional[int] = None,
    tuning_n_estimators: int = 1200,
    use_gpu: Optional[bool] = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    n_startup_trials: int = 20,
    n_warmup_steps: int = 10,
    seed: int = SEED,
    verbose: bool = True,
) -> Tuple[Dict, "object"]:
    """Optuna study (TPE + ``MedianPruner``) maximising mean CV AUC on the
    *selected* feature subset only (Phase 5 -> 4b ordering). Returns
    ``(best_params, study)``; ``best_params`` already merged onto the plan defaults.

    The per-trial estimator count is capped (``tuning_n_estimators``) with early
    stopping so 300 trials stay affordable; refit the winner at full
    ``n_estimators`` in ``run_cv`` afterwards.
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cols = list(feature_cols)
    cap = {"xgb": {"n_estimators": tuning_n_estimators},
           "lgbm": {"n_estimators": tuning_n_estimators},
           "catboost": {"iterations": tuning_n_estimators}}[kind]

    def objective(trial):
        params = _suggest_params(trial, kind)
        params.update(cap)
        factory = _make_factory(kind, params, y, use_gpu, early_stopping_rounds)

        def report_cb(step, running_mean):
            trial.report(running_mean, step=step)
            return trial.should_prune()

        aucs = cv_fold_aucs(factory, X, y, folds, cols, report_cb=report_cb)
        if len(aucs) < len(folds):  # pruned mid-way
            raise optuna.TrialPruned()
        return float(np.nanmean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = dict(study.best_params)
    if verbose:
        print(f"[tune:{kind}] best CV AUC={study.best_value:.4f} "
              f"over {len(study.trials)} trials -> {best}")
    return best, study
