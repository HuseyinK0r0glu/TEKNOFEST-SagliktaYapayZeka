"""Per-panel cross-validation harness optimised for binary F1.

The competition metric is binary F1 on the Pathogenic (positive) class with
per-panel threshold tuning. CV uses label-only StratifiedKFold (panels are
modelled independently). ROC-AUC and MCC are reported as diagnostics only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src import config

POS = config.POSITIVE_CLASS


def tune_threshold(y, proba, lo=0.05, hi=0.95, step=0.005):
    """Return (best_threshold, best_f1) maximising F1 on the positive class."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        f1 = f1_score(y, (proba >= t).astype(int), pos_label=POS, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def run_cv(make_model, X, y, cat_cols=None, n_splits=5, seed=config.SEED,
           fit_kwargs_fn=None):
    """Run StratifiedKFold CV, returning OOF probabilities and F1-centric metrics.

    make_model(y_train) -> fresh estimator (lets the factory set class weights
        from the fold's own training labels).
    fit_kwargs_fn(X_tr, y_tr) -> dict of extra fit kwargs (e.g. categorical_feature).
    """
    y = pd.Series(np.asarray(y), index=X.index)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    fold_auc = []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = make_model(y_tr)
        fit_kwargs = fit_kwargs_fn(X_tr, y_tr) if fit_kwargs_fn else {}
        model.fit(X_tr, y_tr, **fit_kwargs)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        fold_auc.append(roc_auc_score(y_va, oof[va_idx]))

    threshold, f1 = tune_threshold(y.to_numpy(), oof)
    y_pred = (oof >= threshold).astype(int)
    return {
        "oof": oof,
        "threshold": threshold,
        "f1": f1,
        "auc": float(roc_auc_score(y, oof)),
        "auc_fold_mean": float(np.mean(fold_auc)),
        "auc_fold_std": float(np.std(fold_auc)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "precision": float(precision_score(y, y_pred, pos_label=POS, zero_division=0)),
        "recall": float(recall_score(y, y_pred, pos_label=POS, zero_division=0)),
    }
