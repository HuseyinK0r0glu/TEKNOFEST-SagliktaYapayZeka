"""Honest (nested) evaluation — every improvement is repeated inside each
outer fold. See `WARNING.md` for the same hyperparameter-reuse caveat that
applies to `search.reference.nested`."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from pathugen.config import SEED
from search.improved.config import DIVERSE_SPECS, PAIRED_BOOTSTRAP, STD_SPECS
from search.improved.paired_selection import select_champion_plus
from search.improved.threshold import scan
from search.improved.views import generate_base_oof_plus, predict_with, relief_pruned_view
from search.reference.config import NESTED_REPEATS, NESTED_SPLITS
from search.reference.nested import _balanced_f1
from search.reference.selection import null_importance_selection


def nested_eval_plus(panel, X, y, cat_cols, params_by_kind, selection_method,
                     diverse, policy, aug_full):
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    aug_X_full, aug_y_full = aug_full if aug_full is not None else (None, None)

    if NESTED_REPEATS > 1:
        splitter = RepeatedStratifiedKFold(n_splits=NESTED_SPLITS,
                                           n_repeats=NESTED_REPEATS,
                                           random_state=SEED)
    else:
        splitter = StratifiedKFold(n_splits=NESTED_SPLITS, shuffle=True,
                                   random_state=SEED)

    f1s, aucs, balf1s, chosen = [], [], [], []
    n_boot = max(100, PAIRED_BOOTSTRAP // 4)          # cheaper per outer fold
    for tr, te in splitter.split(X, y):
        X_tr_f, y_tr = X.iloc[tr], y.iloc[tr]
        X_te_f, y_te = X.iloc[te], y.iloc[te]

        tbl = null_importance_selection(X_tr_f, y_tr, cat_cols,
                                        method=selection_method, n_null=25)
        sel = tbl.loc[tbl["keep"], "feature"].tolist()
        sc = [c for c in cat_cols if c in set(sel)]
        X_tr, X_te = X_tr_f[sel], X_te_f[sel]

        views, specs = {"full": sel}, list(STD_SPECS)
        if diverse:
            rp = relief_pruned_view(X_tr, y_tr, sc)
            if 3 <= len(rp) < len(sel):
                views["rp"] = rp
                specs += list(DIVERSE_SPECS)

        aug = (aug_X_full[sel], aug_y_full) if aug_X_full is not None else None
        bo, singles, full_models, model_view = generate_base_oof_plus(
            X_tr, y_tr, sc, params_by_kind, views, specs, aug=aug, n_repeats=1)
        ch = select_champion_plus(bo, singles, y_tr, specs, policy,
                                  n_boot=n_boot, n_repeats=1)
        proba_te = predict_with(ch["strategy"], ch["col"], ch["cols"],
                                full_models, model_view, bo, y_tr, X_te)
        pred = (proba_te >= ch["threshold"]).astype(int)
        f1s.append(scan(y_te.to_numpy(), proba_te, np.array([ch["threshold"]]))[0][0])
        aucs.append(roc_auc_score(y_te, proba_te))
        balf1s.append(_balanced_f1(y_te.to_numpy(), pred))
        chosen.append(ch["key"])

    return {"nested_f1": float(np.mean(f1s)), "nested_f1_std": float(np.std(f1s)),
            "nested_auc": float(np.mean(aucs)),
            "nested_balanced_f1": float(np.nanmean(balf1s)),
            "strategies": sorted(set(chosen))}
