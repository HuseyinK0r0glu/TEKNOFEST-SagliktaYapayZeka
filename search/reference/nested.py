"""Leak-free honest F1/AUC: per outer fold, select + champion on train only,
score once on untouched test. aug (non-panel GENEL) added to outer-train only.

See `WARNING.md` for a caveat about this estimate: hyperparameters
themselves are tuned once on the full panel, not re-tuned inside each outer
fold, so the honest numbers below carry a mild optimistic bias from that one
step (feature selection, champion choice, and the threshold ARE genuinely
redone per fold).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from search.reference.augment import build_aug_pool
from search.reference.champion import select_champion
from search.reference.config import NESTED_REPEATS, NESTED_SPLITS, POS, SEED
from search.reference.selection import null_importance_selection
from search.reference.stacking import fit_final_stack, generate_base_oof


def _balanced_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    n_pos, n_neg = int((y_true == POS).sum()), int((y_true != POS).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    w = np.where(y_true == POS, 0.5 / n_pos, 0.5 / n_neg)
    return float(f1_score(y_true, y_pred, pos_label=POS, sample_weight=w, zero_division=0))


def _predict_test(strategy, model, cols, full_models, base_oof_tr, y_tr, X_te):
    base_te = {k: m.predict_proba(X_te)[:, 1] for k, m in full_models.items()}
    if strategy == "single":
        return base_te[model]
    if strategy == "mean":
        return np.mean([base_te[k] for k in full_models], axis=0)
    calibrators, meta, order = fit_final_stack(base_oof_tr, y_tr, cols=cols)
    z = np.column_stack([calibrators[k].predict(base_te[k]) for k in order])
    return meta.predict_proba(z)[:, 1]


def nested_eval(panel, X, y, cat_cols, params_by_kind, selection_method="null"):
    """Leak-free honest F1/AUC: per outer fold, select + champion on train only,
    score once on untouched test. aug (non-panel GENEL) added to outer-train only."""
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    aug_X_full, aug_y_full = build_aug_pool(panel, columns=list(X.columns))
    if NESTED_REPEATS > 1:
        splitter = RepeatedStratifiedKFold(n_splits=NESTED_SPLITS,
            n_repeats=NESTED_REPEATS, random_state=SEED)
    else:
        splitter = StratifiedKFold(n_splits=NESTED_SPLITS, shuffle=True, random_state=SEED)
    f1s, aucs, balf1s = [], [], []
    for tr, te in splitter.split(X, y):
        X_tr_f, y_tr = X.iloc[tr], y.iloc[tr]
        X_te_f, y_te = X.iloc[te], y.iloc[te]
        tbl = null_importance_selection(X_tr_f, y_tr, cat_cols, method=selection_method,
                                        n_null=25)  # lighter per-fold null run
        sel = tbl.loc[tbl["keep"], "feature"].tolist()
        sc = [c for c in cat_cols if c in set(sel)]
        X_tr, X_te = X_tr_f[sel], X_te_f[sel]
        aug = ((aug_X_full[sel], aug_y_full) if aug_X_full is not None else None)
        base_oof_tr, singles_tr, full_models = generate_base_oof(
            X_tr, y_tr, sc, params_by_kind, aug=aug, n_repeats=1)
        ch = select_champion(base_oof_tr, singles_tr, y_tr)
        proba_te = _predict_test(ch["strategy"], ch["model"], ch["cols"],
                                 full_models, base_oof_tr, y_tr, X_te)
        pred = (proba_te >= ch["threshold"]).astype(int)
        f1s.append(f1_score(y_te, pred, pos_label=POS, zero_division=0))
        aucs.append(roc_auc_score(y_te, proba_te))
        balf1s.append(_balanced_f1(y_te.to_numpy(), pred))
    return {"nested_f1": float(np.mean(f1s)), "nested_f1_std": float(np.std(f1s)),
            "nested_auc": float(np.mean(aucs)),
            "nested_balanced_f1": float(np.nanmean(balf1s))}
