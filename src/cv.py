"""Per-panel cross-validation harness optimised for binary F1.

The competition metric is binary F1 on the Pathogenic (positive) class with
per-panel threshold tuning. CV uses label-only StratifiedKFold (panels are
modelled independently). ROC-AUC and MCC are reported as diagnostics only.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from src import config

POS = config.POSITIVE_CLASS


def cv_repeats() -> int:
    """Number of CV repeats (env CV_REPEATS, default 1).

    >1 uses RepeatedStratifiedKFold and averages each row's out-of-fold
    prediction over the repeats, giving lower-variance F1 + threshold estimates
    on the small panels (anti-overfitting). 1 is the original single-split path.
    """
    return max(1, int(os.environ.get("CV_REPEATS", "1")))


def threshold_agg() -> str:
    """How the reported threshold is derived (env THRESHOLD_AGG).

    "pooled" (default): a single argmax over 181 thresholds on the pooled OOF —
        the original behaviour, slightly optimistic on small panels.
    "fold_median": the median of each fold's own optimal threshold. More robust
        to one lucky fold (fixes the CFTR t=0.135 pathology); F1 is still
        reported at that threshold on the pooled OOF.
    """
    mode = os.environ.get("THRESHOLD_AGG", "pooled").strip().lower()
    return mode if mode in ("pooled", "fold_median") else "pooled"


def tune_threshold(y, proba, lo=0.05, hi=0.95, step=0.005):
    """Return (best_threshold, best_f1) maximising F1 on the positive class."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        f1 = f1_score(y, (proba >= t).astype(int), pos_label=POS, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def _append_aug(X_tr, y_tr, aug):
    """Concatenate an external augmentation set onto a training fold.

    aug is (X_aug, y_aug) already restricted to X_tr's columns; it is appended to
    the TRAINING rows only (callers never put it in a validation fold), so the OOF
    estimate stays an honest panel-only number. Returns the combined (X, y).
    """
    if aug is None:
        return X_tr, y_tr
    X_aug, y_aug = aug
    X_cat = pd.concat([X_tr, X_aug[X_tr.columns]], axis=0, ignore_index=True)
    y_cat = pd.concat([pd.Series(np.asarray(y_tr)), pd.Series(np.asarray(y_aug))],
                      ignore_index=True)
    return X_cat, y_cat


def _fit_fold(make_model, X_tr, y_tr, fit_kwargs_fn, eval_kwargs_fn,
              early_stop_frac, seed, aug=None):
    """Fit one fold's model, optionally with an internal early-stopping split.

    When eval_kwargs_fn is given, a stratified slice of the TRAINING fold is held
    out as the early-stopping eval set (never the validation rows, so the OOF
    estimate stays honest). eval_kwargs_fn(X_es, y_es) returns the model-specific
    eval-set fit kwargs (e.g. {"eval_set": [(X_es, y_es)]}).

    aug: optional (X_aug, y_aug) external rows appended to the training fold only
    (e.g. non-PAH GENEL rows used to augment the small PAH panel).
    """
    if eval_kwargs_fn is None:
        X_fit, y_fit = _append_aug(X_tr, y_tr, aug)
        model = make_model(y_fit)
        fit_kwargs = fit_kwargs_fn(X_fit, y_fit) if fit_kwargs_fn else {}
        model.fit(X_fit, y_fit, **fit_kwargs)
        return model
    # Early-stopping slice comes from the PANEL training rows (in-distribution
    # stopping signal); the augmentation rows are added to the fit part only.
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_tr, y_tr, test_size=early_stop_frac, stratify=y_tr, random_state=seed
    )
    X_fit, y_fit = _append_aug(X_fit, y_fit, aug)
    model = make_model(y_fit)
    fit_kwargs = fit_kwargs_fn(X_fit, y_fit) if fit_kwargs_fn else {}
    fit_kwargs.update(eval_kwargs_fn(X_es, y_es))
    model.fit(X_fit, y_fit, **fit_kwargs)
    return model


def run_cv(make_model, X, y, cat_cols=None, n_splits=5, seed=config.SEED,
           fit_kwargs_fn=None, n_repeats=None, eval_kwargs_fn=None,
           early_stop_frac=0.15, aug=None):
    """Run StratifiedKFold CV, returning OOF probabilities and F1-centric metrics.

    make_model(y_train) -> fresh estimator (lets the factory set class weights
        from the fold's own training labels).
    fit_kwargs_fn(X_tr, y_tr) -> dict of extra fit kwargs (e.g. categorical_feature).
    eval_kwargs_fn(X_es, y_es) -> dict of early-stopping fit kwargs (eval_set,
        callbacks). When given, each fold carves a stratified internal validation
        slice for early stopping. None disables early stopping (default).
    n_repeats: CV repeats (default: env CV_REPEATS). >1 averages each row's OOF
        probability across repeats for a lower-variance estimate.
    aug: optional (X_aug, y_aug) external rows appended to EVERY training fold but
        never validated on (e.g. non-PAH GENEL rows). Folds/threshold/metrics are
        all computed on the panel rows in X/y only, so the OOF stays honest.
    """
    if n_repeats is None:
        n_repeats = cv_repeats()
    y = pd.Series(np.asarray(y), index=X.index)
    fold_auc = []
    fold_thresholds = []

    def _run_fold(tr_idx, va_idx, oof_writer):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = _fit_fold(make_model, X_tr, y_tr, fit_kwargs_fn,
                          eval_kwargs_fn, early_stop_frac, seed, aug=aug)
        proba = model.predict_proba(X_va)[:, 1]
        oof_writer(va_idx, proba)
        fold_auc.append(roc_auc_score(y_va, proba))
        ft, _ = tune_threshold(y_va.to_numpy(), proba)
        fold_thresholds.append(ft)

    if n_repeats > 1:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        oof_sum = np.zeros(len(y))
        oof_cnt = np.zeros(len(y))
        for tr_idx, va_idx in splitter.split(X, y):
            _run_fold(tr_idx, va_idx,
                      lambda idx, p: (np.add.at(oof_sum, idx, p),
                                      np.add.at(oof_cnt, idx, 1)))
        oof = oof_sum / np.maximum(oof_cnt, 1)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))

        def _write(idx, p):
            oof[idx] = p

        for tr_idx, va_idx in skf.split(X, y):
            _run_fold(tr_idx, va_idx, _write)

    pooled_t, pooled_f1 = tune_threshold(y.to_numpy(), oof)
    if threshold_agg() == "fold_median" and fold_thresholds:
        threshold = float(np.median(fold_thresholds))
        f1 = f1_score(y, (oof >= threshold).astype(int), pos_label=POS,
                      zero_division=0)
    else:
        threshold, f1 = pooled_t, pooled_f1
    y_pred = (oof >= threshold).astype(int)
    return {
        "oof": oof,
        "threshold": threshold,
        "threshold_pooled": pooled_t,
        "fold_thresholds": fold_thresholds,
        "f1": f1,
        "f1_pooled": pooled_f1,
        "auc": float(roc_auc_score(y, oof)),
        "ap": float(average_precision_score(y, oof)),
        "auc_fold_mean": float(np.mean(fold_auc)),
        "auc_fold_std": float(np.std(fold_auc)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "precision": float(precision_score(y, y_pred, pos_label=POS, zero_division=0)),
        "recall": float(recall_score(y, y_pred, pos_label=POS, zero_division=0)),
    }
