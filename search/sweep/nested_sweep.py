"""Nested sweep — ALL decision configurations are measured on the same
outer folds."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from pathugen.config import SEED
from search.improved.config import DIVERSE_SPECS, STD_SPECS
from search.improved.pipeline import subsample_aug
from search.improved.threshold import pick_threshold
from search.improved.views import generate_base_oof_plus, relief_pruned_view
from search.reference.config import NESTED_REPEATS, NESTED_SPLITS
from search.reference.selection import null_importance_selection
from search.sweep.augpool import full_aug_pool
from search.sweep.candidates import bootstrap_aucs, build_candidates, pick_champion, score_at
from search.sweep.config import DECISION_GRID, THR_BOOT, dc_key


def nested_sweep(panel, X, y, cat_cols, params, selection, aug_frac, diverse):
    """Every outer fold rebuilds selection+models fresh; all 16 decision
    configurations are scored on those same folds. That way, differences
    between configurations come only from the decision layer, not from
    which fold happened to be drawn."""
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    pool_X, pool_y = full_aug_pool(panel)
    if pool_X is not None and aug_frac > 0:
        pool_X, pool_y = subsample_aug(pool_X, pool_y, aug_frac)
    else:
        pool_X = None

    if NESTED_REPEATS > 1:
        splitter = RepeatedStratifiedKFold(n_splits=NESTED_SPLITS,
                                           n_repeats=NESTED_REPEATS,
                                           random_state=SEED)
    else:
        splitter = StratifiedKFold(n_splits=NESTED_SPLITS, shuffle=True,
                                   random_state=SEED)

    acc = {dc_key(dc): {"f1": [], "balanced_f1": [], "auc": [], "champs": []}
           for dc in DECISION_GRID}

    for tr, te in splitter.split(X, y):
        X_tr_f, y_tr = X.iloc[tr], y.iloc[tr]
        X_te_f, y_te = X.iloc[te], y.iloc[te]

        tbl = null_importance_selection(X_tr_f, y_tr, cat_cols,
                                        method=selection, n_null=25)
        sel = tbl.loc[tbl["keep"], "feature"].tolist()
        sc = [c for c in cat_cols if c in set(sel)]
        X_tr, X_te = X_tr_f[sel], X_te_f[sel]

        from search.reference.modeling import KINDS
        from search.improved.config import DIVERSE_SPECS, STD_SPECS
        views, specs = {"full": sel}, list(STD_SPECS)
        if diverse:
            rp = relief_pruned_view(X_tr, y_tr, sc)
            if 3 <= len(rp) < len(sel):
                views["rp"] = rp
                specs += list(DIVERSE_SPECS)

        aug = (pool_X[sel], pool_y) if pool_X is not None else None
        bo, _, full_models, model_view = generate_base_oof_plus(
            X_tr, y_tr, sc, params, views, specs, aug=aug, n_repeats=1)
        cand_oof, cand_te, strategies, _ = build_candidates(
            bo, y_tr, full_models, model_view, specs, X_te=X_te, n_repeats=1)

        aucs, boot = bootstrap_aucs(y_tr, cand_oof)
        thr_cache: dict[tuple, float] = {}
        for dc in DECISION_GRID:
            champ, _ = pick_champion(aucs, boot, strategies, dc["rule"], dc["pool"])
            ck = (champ, dc["policy"])
            if ck not in thr_cache:
                thr_cache[ck] = pick_threshold(y_tr.to_numpy(), cand_oof[champ],
                                               dc["policy"], n_boot=THR_BOOT)
            s = score_at(y_te.to_numpy(), cand_te[champ], thr_cache[ck])
            a = acc[dc_key(dc)]
            a["f1"].append(s["f1"])
            a["balanced_f1"].append(s["balanced_f1"])
            a["auc"].append(roc_auc_score(y_te, cand_te[champ]))
            a["champs"].append(champ)

    out = {}
    for k, a in acc.items():
        counts = Counter(a["champs"])
        top, top_n = counts.most_common(1)[0]
        out[k] = {"honest_f1": float(np.mean(a["f1"])),
                  "honest_f1_std": float(np.std(a["f1"])),
                  "honest_balanced_f1": float(np.mean(a["balanced_f1"])),
                  "honest_auc": float(np.mean(a["auc"])),
                  "stability": top_n / len(a["champs"]),
                  "modal_champion": top,
                  "n_distinct_champions": len(counts)}
    return out
