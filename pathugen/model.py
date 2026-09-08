"""XGBoost — the only model family needed (every panel's audited champion)."""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from pathugen.config import SEED, SEED_ENSEMBLE


def scale_pos_weight(y_tr) -> float:
    y_tr = np.asarray(y_tr)
    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    return n_neg / n_pos if n_pos else 1.0


def make_xgb(params: dict, y_tr, seed: int) -> XGBClassifier:
    p = dict(params)
    spw = p.pop("spw_mult", 1.0)
    return XGBClassifier(
        n_estimators=p.pop("n_estimators", 400), tree_method="hist",
        eval_metric="logloss", scale_pos_weight=scale_pos_weight(y_tr) * spw,
        random_state=seed, n_jobs=-1, **p)


def fit_seed_ensemble(X, y, params: dict, n_seeds: int = SEED_ENSEMBLE):
    """Fit the same tuned XGBoost with `n_seeds` different random seeds and
    average their predictions. This is part of the locked, audited
    configuration for every panel — averaging a few seeds costs almost
    nothing and reduces variance from a single lucky/unlucky tree-building
    draw, which matters most on the smallest panels (CFTR: 111 rows).
    """
    models = []
    for i in range(n_seeds):
        m = make_xgb(params, y, seed=SEED + i)
        m.fit(X, np.asarray(y))
        models.append(m)
    return models


def predict_proba_ensemble(models, X) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
