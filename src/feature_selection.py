"""Null-importance feature selection (per panel).

Method (Grellier-style, the plan's gold standard): fit LightGBM on the real
labels to get gain importances, then fit many times on SHUFFLED labels to build
a null distribution per feature. A feature is kept when its real importance
clearly exceeds what it earns by chance. With 411 features and as few as ~111
rows, this is the right filter against false-discovery noise.

Importance is label-dependent, so unlike `features.build_features` this MUST be
run on training data only when used inside a fold. Here it is applied per panel
for selection, and the result is validated by re-running CV on the kept subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier

from src import config


def _imp_estimator(y, seed) -> LGBMClassifier:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = n_neg / n_pos if n_pos else 1.0
    return LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
        random_state=seed, n_jobs=1, verbosity=-1, importance_type="gain",
    )


def _fit_gain(X, y_values, cat_cols, seed) -> np.ndarray:
    model = _imp_estimator(pd.Series(y_values), seed)
    model.fit(X, y_values, categorical_feature=list(cat_cols))
    return model.booster_.feature_importance(importance_type="gain")


def null_importance_selection(X, y, cat_cols, n_actual=5, n_null=50,
                              seed=config.SEED, percentile=95,
                              k_div=12, k_cap=50, k_min=10):
    """Return a per-feature DataFrame with a `keep` flag.

    keep = (real importance > `percentile` of the null distribution) OR
           (feature is among the top-K by gain_score).

    The top-K floor makes selection strictly more inclusive than the bare
    null-importance gate and panel-size bounded (K = clip(n_rows/k_div, k_min,
    k_cap)). It guards against the gate's instability: when the candidate pool
    grows, the gate can otherwise drop a feature that a smaller pool had kept,
    silently regressing a panel. The floor never removes a gate-passer.
    """
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    null_seeds = rng.integers(0, 2**31 - 1, size=n_null)

    actual = np.mean(
        [_fit_gain(X, y.to_numpy(), cat_cols, seed + i) for i in range(n_actual)],
        axis=0,
    )

    def _null_run(i):
        y_shuf = y.sample(frac=1.0, random_state=int(null_seeds[i])).to_numpy()
        return _fit_gain(X, y_shuf, cat_cols, 1000 + i)

    nulls = Parallel(n_jobs=-1)(delayed(_null_run)(i) for i in range(n_null))
    nulls = np.vstack(nulls)  # (n_null, n_features)

    null_pct = np.percentile(nulls, percentile, axis=0)
    null_p75 = np.percentile(nulls, 75, axis=0)
    gain_score = np.log((1.0 + actual) / (1.0 + null_p75))
    gate = actual > null_pct

    k = int(min(k_cap, max(k_min, len(X) // k_div)))
    topk = np.zeros(len(gain_score), dtype=bool)
    topk[np.argsort(gain_score)[::-1][:k]] = True
    keep = gate | topk

    table = pd.DataFrame({
        "feature": list(X.columns),
        "actual_gain": actual,
        f"null_p{percentile}": null_pct,
        "gain_score": gain_score,
        "gate": gate,
        "keep": keep,
    })
    return table.sort_values("gain_score", ascending=False).reset_index(drop=True)
