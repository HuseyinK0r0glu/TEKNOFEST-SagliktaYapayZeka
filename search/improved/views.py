"""Base-model OOF generation where each column can have its own feature
view (the diverse-stacking improvement)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from search.reference.modeling import MAKERS, TREE_KINDS, run_cv
from search.reference.selection import information_gain, relieff
from search.reference.stacking import fit_final_stack
from search.reference.tuning import _fit_fn


def relief_pruned_view(X, y, cat_cols, ig_protect_k=None):
    """A second view that drops negative-ReliefF-weighted features.

    Features in the top `ig_protect_k` by information gain, plus
    categoricals, are protected — so a globally strong feature doesn't
    silently get dropped because of a local metric. The protection count
    scales with the set size: a fixed 15 on a 13-feature panel like CFTR
    would protect the entire set and make pruning meaningless.
    """
    cols = list(X.columns)
    if ig_protect_k is None:
        ig_protect_k = min(15, max(3, len(cols) // 3))
    try:
        rfw = relieff(X, y).reindex(cols).fillna(0.0)
        ig = information_gain(X, y).reindex(cols).fillna(-np.inf)
    except Exception as exc:                       # fall back to the full view if the filters fail
        print(f"    [warning] could not compute the ReliefF view ({exc}); using the full view")
        return cols
    protected = set(ig.sort_values(ascending=False).head(ig_protect_k).index) | set(cat_cols or [])
    keep = [c for c in cols if rfw[c] >= 0.0 or c in protected]
    return keep


def generate_base_oof_plus(X, y, cat_cols, params_by_kind, views, specs,
                           aug=None, n_repeats=None):
    """A view-aware version of the reference pipeline's generate_base_oof.

    Returns: (base_oof DataFrame, singles, full-data models, column->view).
    Augmentation rows only enter the training folds and the final refit.
    """
    base_oof, singles, full_models, model_view = {}, {}, {}, {}
    aug_X, aug_y = aug if aug is not None else (None, None)

    for col, kind, vkey in specs:
        feats = [c for c in views[vkey] if c in X.columns]
        if not feats:
            continue
        Xv = X[feats]
        cv_cat = [c for c in (cat_cols or []) if c in set(feats)]
        fit_fn = _fit_fn(kind, cv_cat)
        params = params_by_kind.get(kind) or {}

        def factory(y_tr, _k=kind, _p=params, _f=feats, _c=tuple(cv_cat)):
            return MAKERS[_k](_p, y_tr, list(_f), list(_c))

        augv = (aug_X[feats], aug_y) if aug_X is not None else None
        res = run_cv(factory, Xv, y, cat_cols=cv_cat, fit_kwargs_fn=fit_fn,
                     aug=augv, n_repeats=n_repeats)
        base_oof[col] = res["oof"]
        singles[col] = {"f1": res["f1"], "threshold": res["threshold"], "auc": res["auc"]}

        if aug_X is not None:
            X_full = pd.concat([Xv, aug_X[feats]], axis=0, ignore_index=True)
            y_full = np.concatenate([np.asarray(y), np.asarray(aug_y)])
        else:
            X_full, y_full = Xv, np.asarray(y)
        model = factory(pd.Series(y_full))
        model.fit(X_full, y_full, **(fit_fn(X_full, y_full) if fit_fn else {}))
        full_models[col] = model
        model_view[col] = feats

    return pd.DataFrame(base_oof, index=X.index), singles, full_models, model_view


def stack_variants(specs):
    cols = [c for c, _, _ in specs]
    trees = [c for c, k, _ in specs if k in TREE_KINDS and c in cols]
    out = {"stacked_all": [c for c, _, v in specs if v == "full"],
           "stacked_trees": [c for c in trees if c in
                             {s[0] for s in specs if s[2] == "full"}]}
    if any(v != "full" for _, _, v in specs):
        out["stacked_diverse"] = cols
    return out


def predict_with(strategy, col, cols, full_models, model_view, base_oof_tr, y_tr, X_te):
    """Scores test rows with the champion strategy (each column uses its own view)."""
    base_te = {c: m.predict_proba(X_te[model_view[c]])[:, 1] for c, m in full_models.items()}
    if strategy == "single":
        return base_te[col]
    if strategy == "mean":
        return np.mean([base_te[c] for c in full_models], axis=0)
    calibrators, meta, order = fit_final_stack(base_oof_tr, y_tr, cols=cols)
    z = np.column_stack([calibrators[c].predict(base_te[c]) for c in order])
    return meta.predict_proba(z)[:, 1]
