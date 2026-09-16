"""Per-panel calibration + stacking ensemble.

Pipeline (leak-free):
  1. Base OOF: each tuned base model produces 5-fold out-of-fold probabilities
     (every row predicted by a model that did not train on it).
  2. Calibration + meta: isotonic calibration of each base column is fit INSIDE
     a meta cross-validation fold, then a logistic-regression meta-learner is
     trained on the calibrated columns. The resulting meta OOF is fully honest.
  3. A single threshold is tuned on the meta OOF (one scalar — negligible
     optimism) to report binary F1 on the Pathogenic class.

`fit_final_stack` refits calibrators + meta on the full OOF so the bundle can be
persisted for test-time prediction.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from src import config, cv, tuning

POS = config.POSITIVE_CLASS

KINDS = tuning.MODEL_KINDS

# The boosted-tree bases are the validated core; the two diverse bases
# (extratrees, logreg_l1) are optional. A diverse base helps the stack on some
# panels (CFTR) but hurts it on others (PAH), so we evaluate two stack variants
# and let champion selection keep whichever wins on honest OOF — guaranteeing we
# never regress below the trees-only stack.
TREE_KINDS = ("xgboost", "lightgbm", "catboost")
STACK_VARIANTS = {"stacked_all": tuple(KINDS), "stacked_trees": TREE_KINDS}


def _tuned_params(panel: str, kind: str) -> dict:
    path = config.OUTPUT_DIR / "optuna" / f"{panel}_{kind}_best.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# Small panels: a panel below this row count uses Platt (sigmoid) calibration,
# because isotonic regression overfits the calibration curve at small n.
_SMALL_PANEL = 400


class _SigmoidCalibrator:
    """Platt scaling: a logistic fit of a 1-D probability onto the label.

    Mirrors IsotonicRegression's .predict(x) -> prob interface so the single-model
    and stack prediction paths share code. Module-level (not a closure) so joblib
    can pickle the persisted calibrator.
    """

    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000)

    def fit(self, x, y):
        self.lr.fit(np.asarray(x, dtype=float).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, x):
        return self.lr.predict_proba(np.asarray(x, dtype=float).reshape(-1, 1))[:, 1]


def fit_single_calibrator(oof_col, y, n_rows=None):
    """Calibrate one model's OOF probabilities.

    Isotonic on large panels, Platt/sigmoid on small ones (n < _SMALL_PANEL),
    where isotonic overfits. Calibration is monotonic, so the model's AUC and its
    best-threshold F1 are unchanged — the gain is a *meaningful* probability and a
    threshold that is stable if the test-set class balance differs from train.
    Returns a fitted object exposing .predict(proba) -> calibrated_proba.
    """
    x = np.asarray(oof_col, dtype=float)
    y = np.asarray(y)
    if n_rows is not None and n_rows < _SMALL_PANEL:
        return _SigmoidCalibrator().fit(x, y)
    return IsotonicRegression(out_of_bounds="clip").fit(x, y)


def generate_base_oof(panel, X, y, cat_cols, seed=config.SEED, params_override=None,
                      n_repeats=None, aug=None):
    """Return (base_oof DataFrame, per-model single results, full-data models).

    params_override: optional {kind: param dict} used instead of reading the
        saved Optuna JSON (e.g. the nested-eval harness passes per-outer-fold
        params). When None, the on-disk tuned params for `panel` are used.
    n_repeats: inner-CV repeats (default: env CV_REPEATS). The nested harness
        passes 1 to keep its inner loop cheap (outer folds give the variance).
    aug: optional (X_aug, y_aug) external training rows (e.g. non-PAH GENEL rows).
        Added to every training fold AND to the full-data refit, never validated
        on, so the OOF stays an honest panel-only estimate.
    """
    base_oof = {}
    singles = {}
    full_models = {}
    feature_cols = list(X.columns)
    aug_X, aug_y = aug if aug is not None else (None, None)
    # Align the augmentation matrix to this (possibly feature-selected) X schema.
    if aug_X is not None:
        aug_X = aug_X.reindex(columns=feature_cols)
        aug = (aug_X, aug_y)
    for kind in KINDS:
        if params_override is not None:
            params = params_override.get(kind) or {}
        else:
            params = _tuned_params(panel, kind)
        factory = (lambda y_tr, k=kind, p=params: tuning.build_estimator(
            k, p, y_tr, feature_cols, cat_cols))
        cat_kw = tuning.cat_kwarg(kind)
        use_cat = bool(cat_kw and cat_cols)
        fit_fn = (lambda X_tr, y_tr, ck=cat_kw: {ck: list(cat_cols)}) if use_cat else None
        eval_fn = tuning.eval_kwargs_fn(kind)

        res = cv.run_cv(factory, X, y, cat_cols=cat_cols, fit_kwargs_fn=fit_fn,
                        eval_kwargs_fn=eval_fn, seed=seed, n_repeats=n_repeats,
                        aug=aug)
        base_oof[kind] = res["oof"]
        singles[kind] = {"f1": res["f1"], "threshold": res["threshold"], "auc": res["auc"]}

        # Full-data refit (for prediction): panel rows + the whole aug pool.
        if aug_X is not None:
            X_full = pd.concat([X, aug_X], axis=0, ignore_index=True)
            y_full = np.concatenate([np.asarray(y), np.asarray(aug_y)])
        else:
            X_full, y_full = X, np.asarray(y)
        model = factory(pd.Series(y_full))
        base_kwargs = fit_fn(X_full, y_full) if fit_fn else {}
        tuning.fit_full(model, X_full, y_full, kind, fit_kwargs=base_kwargs, seed=seed)
        full_models[kind] = model

    return pd.DataFrame(base_oof, index=X.index), singles, full_models


def evaluate_stack(base_oof, y, n_splits=5, seed=config.SEED, cols=None,
                   n_repeats=None):
    """Honest meta OOF via per-fold isotonic calibration + logistic meta-learner.

    cols: optional subset of base columns to stack over (defaults to all).
    n_repeats: meta-CV repeats (default: env CV_REPEATS). The nested harness
        passes 1 to keep its inner loop cheap.
    """
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    bo = base_oof.reset_index(drop=True)
    if cols is not None:
        bo = bo[list(cols)]
    if n_repeats is None:
        n_repeats = cv.cv_repeats()
    if n_repeats > 1:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    meta_sum = np.zeros(len(y))
    meta_cnt = np.zeros(len(y))
    fold_thresholds = []

    for tr, va in splitter.split(bo, y):
        z_tr = np.zeros((len(tr), bo.shape[1]))
        z_va = np.zeros((len(va), bo.shape[1]))
        for j, col in enumerate(bo.columns):
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                bo[col].iloc[tr].to_numpy(), y.iloc[tr].to_numpy()
            )
            z_tr[:, j] = iso.predict(bo[col].iloc[tr].to_numpy())
            z_va[:, j] = iso.predict(bo[col].iloc[va].to_numpy())
        lr = LogisticRegression(max_iter=1000).fit(z_tr, y.iloc[tr].to_numpy())
        p_va = lr.predict_proba(z_va)[:, 1]
        meta_sum[va] += p_va
        meta_cnt[va] += 1
        ft, _ = cv.tune_threshold(y.iloc[va].to_numpy(), p_va)
        fold_thresholds.append(ft)

    meta_oof = meta_sum / np.maximum(meta_cnt, 1)
    pooled_t, pooled_f1 = cv.tune_threshold(y.to_numpy(), meta_oof)
    if cv.threshold_agg() == "fold_median" and fold_thresholds:
        threshold = float(np.median(fold_thresholds))
        f1 = float(f1_score(y.to_numpy(), (meta_oof >= threshold).astype(int),
                            pos_label=POS, zero_division=0))
    else:
        threshold, f1 = pooled_t, pooled_f1
    return meta_oof, threshold, f1


def fit_final_stack(base_oof, y, cols=None):
    """Refit calibrators + meta-learner on the full OOF (for prediction).

    cols: optional subset of base columns to stack over (defaults to all).
    Returns (calibrators, meta, order) where order is the column list the
    meta-learner expects at predict time.
    """
    y = np.asarray(y)
    bo = base_oof if cols is None else base_oof[list(cols)]
    calibrators = {}
    z = np.zeros((len(y), bo.shape[1]))
    for j, col in enumerate(bo.columns):
        iso = IsotonicRegression(out_of_bounds="clip").fit(bo[col].to_numpy(), y)
        calibrators[col] = iso
        z[:, j] = iso.predict(bo[col].to_numpy())
    meta = LogisticRegression(max_iter=1000).fit(z, y)
    return calibrators, meta, list(bo.columns)
