"""Shortcut (annotation completeness) audit: measures the standalone ROC-AUC
of missingness alone, and re-scores the champion with those features
removed — a number for how fragile the model is to a distribution shift."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from search.improved.config import MISS_KEYS
from search.reference.modeling import MAKERS, run_cv
from search.reference.tuning import _fit_fn


def is_missingness(feat):
    return any(feat == k or feat.startswith(k) for k in MISS_KEYS)


def shortcut_audit(panel, X_all, y, selected, cat_cols, champ_kind, params, aug):
    """Standalone strength of the missingness signal + an ablation without it."""
    y_arr = np.asarray(y)
    print("  [shortcut audit] standalone ROC-AUC per feature (if it were the only column used):")
    solo = {}
    for feat in ("pct_missing", "n_missing_total", "n_tools_available", "ek_mean"):
        if feat not in X_all.columns:
            continue
        v = pd.to_numeric(X_all[feat], errors="coerce")
        v = v.fillna(v.median())
        try:
            auc = float(roc_auc_score(y_arr, v))
        except ValueError:
            continue
        solo[feat] = auc
        flip = " (inverted direction)" if auc < 0.5 else ""
        print(f"      {feat:<20s} AUC={auc:.3f}{flip}   -> direction-independent {max(auc, 1-auc):.3f}")

    miss_sel = [f for f in selected if is_missingness(f)]
    print(f"  [shortcut audit] {len(miss_sel)} of the {len(selected)} selected features are "
          f"missingness-based: {miss_sel if miss_sel else '-'}")

    ablation = {}
    kept = [f for f in selected if not is_missingness(f)]
    if miss_sel and len(kept) >= 3:
        kind = champ_kind if champ_kind in MAKERS else "xgboost"
        cv_cat = [c for c in cat_cols if c in set(kept)]
        fit_fn = _fit_fn(kind, cv_cat)
        p = params.get(kind) or {}
        augk = (aug[0][kept], aug[1]) if aug is not None else None
        res = run_cv(lambda y_tr: MAKERS[kind](p, y_tr, kept, cv_cat),
                     X_all[kept], y, cat_cols=cv_cat, fit_kwargs_fn=fit_fn, aug=augk)
        ablation = {"kind": kind, "n_features": len(kept),
                    "f1": res["f1"], "auc": res["auc"]}
        print(f"  [shortcut ablation] {kind}, missingness features removed "
              f"({len(selected)}->{len(kept)}): F1={res['f1']:.4f}  AUC={res['auc']:.4f}")
    return {"solo_auc": solo, "missingness_selected": miss_sel, "ablation": ablation}
