"""Calibrated stacking + per-panel champion selection support.

Base OOF → per-fold isotonic calibration + logistic meta. Champion = simplest
candidate within 1 SE of the best AUC (single models usually win; a stack must
beat them by more than bootstrap noise).

Single/mean candidates use RAW base probabilities end-to-end (threshold tuned on
raw OOF, prediction on raw proba) — calibration is monotonic, so it would not
change F1 at the tuned threshold, and keeping it raw avoids any train/predict
mismatch. The stack does its OWN isotonic calibration internally (leak-free).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from search.reference.config import CV_REPEATS, POS, SEED
from search.reference.modeling import KINDS, MAKERS, run_cv, tune_threshold
from search.reference.tuning import _fit_fn


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
