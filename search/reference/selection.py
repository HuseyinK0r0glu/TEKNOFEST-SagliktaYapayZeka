"""Feature selection: null-importance (gain vs shuffled-label null) with an
optional ReliefF guard (the hybrid selector). Default ships `null`."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif

from search.reference.config import SEED


def _impute_for_filters(X):
    Xi = X.apply(pd.to_numeric, errors="coerce")
    Xi = Xi.fillna(Xi.median(numeric_only=True))
    return Xi.fillna(0.0)


def information_gain(X, y, seed=SEED):
    Xi = _impute_for_filters(X)
    mi = mutual_info_classif(Xi.to_numpy(), np.asarray(y), discrete_features=False,
                             random_state=seed)
    return pd.Series(mi, index=X.columns)


def relieff(X, y, n_neighbors=10, sample_size=400, seed=SEED):
    Xi = _impute_for_filters(X).to_numpy(dtype=float)
    y = np.asarray(y)
    n, n_feat = Xi.shape
    mu, sd = Xi.mean(0), Xi.std(0)
    sd[sd == 0] = 1.0
    Xz = (Xi - mu) / sd
    rng = np.random.default_rng(seed)
    m = min(sample_size, n)
    sample_idx = rng.choice(n, size=m, replace=False)
    _, counts = np.unique(y, return_counts=True)
    k = max(1, min(n_neighbors, int(counts.min()) - 1))
    weights = np.zeros(n_feat)
    used = 0
    for i in sample_idx:
        diff = np.abs(Xz - Xz[i])
        dist = np.sqrt((diff ** 2).sum(1))
        dist[i] = np.inf
        same = y == y[i]
        same[i] = False
        other = ~same
        if same.sum() < k or other.sum() < k:
            continue
        hit = np.flatnonzero(same)[np.argsort(dist[same])[:k]]
        miss = np.flatnonzero(other)[np.argsort(dist[other])[:k]]
        weights += diff[miss].mean(0) - diff[hit].mean(0)
        used += 1
    if used:
        weights /= used
    return pd.Series(weights, index=X.columns)


def _imp_estimator(y, seed):
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, min_child_samples=20, reg_alpha=0.1,
        reg_lambda=1.0, scale_pos_weight=n_neg / max(n_pos, 1), random_state=seed,
        n_jobs=1, verbosity=-1, importance_type="gain")


def _fit_gain(X, yv, cat_cols, seed):
    m = _imp_estimator(pd.Series(yv), seed)
    m.fit(X, yv, categorical_feature=list(cat_cols))
    return m.booster_.feature_importance(importance_type="gain")


def null_importance_selection(X, y, cat_cols, n_actual=5, n_null=40, seed=SEED,
                              percentile=95, k_div=12, k_cap=50, k_min=10,
                              method="null", ig_protect_k=15):
    """Per-feature DataFrame with a `keep` flag.

    method='null'   : keep = gate(real>null_p95) OR top-K by gain_score.
    method='hybrid' : AND (ReliefF>=0 OR top-ig_protect_k Info Gain OR categorical).
    """
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    null_seeds = rng.integers(0, 2**31 - 1, size=n_null)
    actual = np.mean([_fit_gain(X, y.to_numpy(), cat_cols, seed + i)
                      for i in range(n_actual)], axis=0)

    def _null_run(i):
        ys = y.sample(frac=1.0, random_state=int(null_seeds[i])).to_numpy()
        return _fit_gain(X, ys, cat_cols, 1000 + i)

    nulls = np.vstack(Parallel(n_jobs=-1)(delayed(_null_run)(i) for i in range(n_null)))
    null_pct = np.percentile(nulls, percentile, axis=0)
    null_p75 = np.percentile(nulls, 75, axis=0)
    gain_score = np.log((1.0 + actual) / (1.0 + null_p75))
    gate = actual > null_pct
    k = int(min(k_cap, max(k_min, len(X) // k_div)))
    topk = np.zeros(len(gain_score), dtype=bool)
    topk[np.argsort(gain_score)[::-1][:k]] = True
    keep = gate | topk
    cols = list(X.columns)

    if method == "hybrid":
        rf = relieff(X, y, seed=seed).reindex(cols).fillna(0.0).to_numpy()
        ig = information_gain(X, y, seed=seed).reindex(cols).fillna(-np.inf)
        ig_protected = (ig.rank(ascending=False, method="min") <= ig_protect_k).to_numpy()
        cat_like = set(cat_cols or [])
        cat_protected = np.array([c in cat_like for c in cols])
        keep = keep & ((rf >= 0.0) | ig_protected | cat_protected)

    table = pd.DataFrame({"feature": cols, "gain_score": gain_score, "keep": keep})
    return table.sort_values("gain_score", ascending=False).reset_index(drop=True)
