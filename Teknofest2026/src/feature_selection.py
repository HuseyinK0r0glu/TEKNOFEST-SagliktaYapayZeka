"""
feature_selection.py
=====================
Phase 5 of the plan -- prune the ~522 engineered features down to the true signals
*before* the expensive Optuna search (Phase 4b). With 522 features and ~3,500 rows
the false-discovery rate on raw importance is high, so we combine three filters:

  * **5.1 Importance-based selection** -- gain importance of a default GBM; build
    cumulative-importance "elbow" sets (``top_k``) for CV comparison.
  * **5.2 Null Importance Test (gold standard)** -- compare each feature's real
    importance against the 95th percentile of a *null* distribution built from
    >=50 label shuffles. Parallelised with ``joblib``. A feature survives only if
    its real importance exceeds the null 95th percentile (it is doing more than
    fitting noise).
  * **5.4 Correlation pruning** -- for |corr| > 0.95 pairs, keep the one more
    point-biserially correlated with the label.
  * **5.5 SHAP importance** -- mean |SHAP| over the fold models' validation rows
    (more reliable than ``gain`` for correlated features); used as the final
    arbiter (intersection with the null-survivors).

Everything is model-agnostic: pass an ``importance_fn(X, y, seed) -> np.ndarray``
(aligned to ``X.columns``). ``make_xgb_importance_fn`` / ``make_lgbm_importance_fn``
build NaN-native gain extractors for the real run; SHAP uses ``shap.TreeExplainer``
on the fitted fold models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import FEATURE_SELECTION_DIR, SEED, SELECTED_FEATURE_LIST_PATH

ImportanceFn = Callable[[pd.DataFrame, np.ndarray, int], np.ndarray]


# =========================================================================== #
# Importance extractors (NaN-native gain) -- used for actual + null importance
# =========================================================================== #
def make_xgb_importance_fn(params: Optional[Dict] = None, n_estimators: int = 400,
                           n_jobs: int = 1) -> ImportanceFn:
    """Return ``fn(X, y, seed) -> gain importances`` backed by a CPU XGBoost.

    CPU (``device='cpu'``) so ``joblib`` can fan the 50 shuffles across processes
    without contending for a single GPU. Fixed ``n_estimators`` (no early stopping
    -- there is no val fold during importance extraction). NaN preserved.
    """
    base = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                device="cpu", max_depth=6, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.6, importance_type="gain", n_jobs=n_jobs)
    if params:
        base.update(params)

    def fn(X: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
        from xgboost import XGBClassifier
        m = XGBClassifier(**base, n_estimators=n_estimators, random_state=seed)
        m.fit(X, y, verbose=False)
        return np.asarray(m.feature_importances_, dtype=float)

    return fn


def make_lgbm_importance_fn(params: Optional[Dict] = None, n_estimators: int = 400,
                            n_jobs: int = 1) -> ImportanceFn:
    """Return ``fn(X, y, seed) -> gain importances`` backed by a CPU LightGBM."""
    base = dict(objective="binary", n_jobs=n_jobs, num_leaves=63, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.6, importance_type="gain", verbosity=-1)
    if params:
        base.update(params)

    def fn(X: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(**base, n_estimators=n_estimators, random_state=seed)
        m.fit(X, y)
        return np.asarray(m.feature_importances_, dtype=float)

    return fn


# =========================================================================== #
# 5.1  Importance-based selection
# =========================================================================== #
def gain_importance(importance_fn: ImportanceFn, X: pd.DataFrame, y: Sequence[int],
                    seed: int = SEED) -> pd.Series:
    """Single default-model gain importance, as a Series indexed by feature name."""
    imp = importance_fn(X, np.asarray(y).astype(int), seed)
    return pd.Series(imp, index=list(X.columns)).sort_values(ascending=False)


def cumulative_importance_sets(importance: pd.Series,
                               ks: Sequence[int] = (50, 100, 150),
                               coverage: float = 0.95) -> Dict[str, List[str]]:
    """``top_50/100/150`` plus the cumulative-importance ``elbow`` set covering
    ``coverage`` (default 95%) of total gain, and ``all_features``."""
    ranked = importance.sort_values(ascending=False)
    sets: Dict[str, List[str]] = {f"top_{k}": ranked.index[:k].tolist() for k in ks
                                  if k <= len(ranked)}
    total = ranked.sum()
    if total > 0:
        cum = (ranked.cumsum() / total)
        elbow_n = int((cum < coverage).sum()) + 1
        sets["elbow_95"] = ranked.index[:elbow_n].tolist()
    sets["all_features"] = ranked.index.tolist()
    return sets


# =========================================================================== #
# 5.2  Null Importance Test (>=50 shuffles, joblib-parallel)
# =========================================================================== #
def _null_importance_once(importance_fn: ImportanceFn, X: pd.DataFrame,
                          y: np.ndarray, seed: int) -> np.ndarray:
    """One null run: permute labels, refit, return importances (worker fn)."""
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y)
    return importance_fn(X, y_shuffled, seed)


def null_importance_test(
    importance_fn: ImportanceFn,
    X: pd.DataFrame,
    y: Sequence[int],
    *,
    n_shuffles: int = 50,
    n_jobs: int = -1,
    percentile: float = 95.0,
    seed: int = SEED,
    save_csv: bool = True,
    out_dir: str = FEATURE_SELECTION_DIR,
) -> pd.DataFrame:
    """Run the gold-standard null importance test.

    Returns a per-feature DataFrame with the real importance, the null
    distribution summary, a gain ``null_score`` (= log scaling of real vs null),
    and the ``keep`` flag (real > null ``percentile``). Persists
    ``null_importance_results.csv``.
    """
    from joblib import Parallel, delayed

    y = np.asarray(y).astype(int)
    feats = list(X.columns)
    if n_shuffles < 50:
        print(f"[null-imp] WARNING: n_shuffles={n_shuffles} < 50 "
              "(plan requires >=50 for a reliable 95th-percentile estimate).")

    actual = importance_fn(X, y, seed)  # importance on the REAL labels

    null_runs = Parallel(n_jobs=n_jobs)(
        delayed(_null_importance_once)(importance_fn, X, y, seed + 1 + i)
        for i in range(n_shuffles)
    )
    null = np.vstack(null_runs)                      # (n_shuffles, n_features)
    null_mean = null.mean(axis=0)
    null_pct = np.percentile(null, percentile, axis=0)
    null_std = null.std(axis=0)

    # Kaggle "null importance gain score": how far real sits above the null mass.
    null_score = np.log1p(actual) - np.log1p(np.maximum(null_pct, 0.0))

    df = pd.DataFrame({
        "feature": feats,
        "actual_importance": actual,
        "null_mean": null_mean,
        f"null_p{int(percentile)}": null_pct,
        "null_std": null_std,
        "null_score": null_score,
        "keep": actual > null_pct,
    }).sort_values("null_score", ascending=False).reset_index(drop=True)

    if save_csv:
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "null_importance_results.csv"), index=False)
    print(f"[null-imp] {int(df['keep'].sum())}/{len(df)} features beat the null "
          f"{int(percentile)}th pct over {n_shuffles} shuffles.")
    return df


# =========================================================================== #
# 5.4  Correlation-based pruning
# =========================================================================== #
def point_biserial(X: pd.DataFrame, y: Sequence[int], cols: Sequence[str]) -> pd.Series:
    """|point-biserial correlation| of each feature with the binary label."""
    y = pd.Series(np.asarray(y).astype(float))
    out = {}
    for c in cols:
        v = X[c]
        mask = v.notna()
        if mask.sum() > 2 and v[mask].nunique() > 1:
            out[c] = abs(np.corrcoef(v[mask], y[mask])[0, 1])
        else:
            out[c] = 0.0
    return pd.Series(out).fillna(0.0)


def correlation_prune(X: pd.DataFrame, y: Sequence[int], features: Sequence[str],
                      threshold: float = 0.95) -> List[str]:
    """Drop one of each |Pearson| > ``threshold`` pair -- keep the feature more
    point-biserially correlated with the label (Phase 5.4)."""
    feats = list(features)
    if len(feats) < 2:
        return feats
    pb = point_biserial(X, y, feats)
    corr = X[feats].corr().abs()
    dropped: set = set()
    for i, a in enumerate(feats):
        if a in dropped:
            continue
        for b in feats[i + 1:]:
            if b in dropped:
                continue
            if corr.loc[a, b] > threshold:
                weaker = a if pb[a] < pb[b] else b
                dropped.add(weaker)
                if weaker == a:
                    break  # 'a' is gone -> stop comparing it
    kept = [f for f in feats if f not in dropped]
    print(f"[corr-prune] dropped {len(dropped)} of {len(feats)} (|r|>{threshold}); "
          f"{len(kept)} remain.")
    return kept


# =========================================================================== #
# 5.5  SHAP importance (mean |SHAP| over fold-model validation rows)
# =========================================================================== #
def _shap_mean_abs(model, X: pd.DataFrame, max_samples: int = 2000,
                   seed: int = SEED) -> np.ndarray:
    """Mean |SHAP| per feature for one fitted tree model on rows ``X``."""
    import shap
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=seed)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):            # some explainers return [neg, pos]
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                    # (n, features, classes)
        sv = sv[..., -1]
    return np.abs(sv).mean(axis=0)


def shap_importance_from_cv(cv_result, X: pd.DataFrame, folds: List[dict],
                            max_samples: int = 2000, seed: int = SEED) -> pd.Series:
    """Average mean|SHAP| across the fold models, each explained on its own
    validation rows (Phase 5.5 -- SHAP on OOF). ``cv_result.models[k]`` are the
    learner wrappers from ``cv_runner.run_cv`` (their ``.model`` is the booster)."""
    cols = list(cv_result.feature_cols)
    acc = np.zeros(len(cols), dtype=float)
    n = 0
    for k, fold in enumerate(folds):
        if k >= len(cv_result.models):
            break
        learner = cv_result.models[k]
        model = getattr(learner, "model", learner)
        va = np.asarray(fold["val"], dtype=int)
        acc += _shap_mean_abs(model, X[cols].iloc[va], max_samples=max_samples, seed=seed)
        n += 1
    imp = acc / max(n, 1)
    return pd.Series(imp, index=cols).sort_values(ascending=False)


# =========================================================================== #
# Orchestrator -- combine null test + SHAP + correlation pruning
# =========================================================================== #
@dataclass
class FeatureSelectionResult:
    selected: List[str]
    null_importance: pd.DataFrame
    shap_importance: Optional[pd.Series] = None
    gain_importance: Optional[pd.Series] = None
    dropped_correlated: int = 0
    meta: Dict = field(default_factory=dict)

    def save(self, path: str = SELECTED_FEATURE_LIST_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"selected_features": self.selected,
                       "n_selected": len(self.selected), "meta": self.meta}, f, indent=2)
        txt = os.path.join(os.path.dirname(path), "final_feature_list.txt")
        with open(txt, "w") as f:
            f.write("\n".join(self.selected) + "\n")
        if self.shap_importance is not None:
            self.shap_importance.rename("mean_abs_shap").to_csv(
                os.path.join(os.path.dirname(path), "shap_importance_ranking.csv"))
        print(f"[feature-selection] saved {len(self.selected)} features -> {path}")


def run_feature_selection(
    importance_fn: ImportanceFn,
    X: pd.DataFrame,
    y: Sequence[int],
    *,
    n_shuffles: int = 50,
    n_jobs: int = -1,
    correlation_threshold: float = 0.95,
    shap_importance: Optional[pd.Series] = None,
    shap_top_n: Optional[int] = None,
    seed: int = SEED,
    save: bool = True,
) -> FeatureSelectionResult:
    """Full Phase-5 funnel:

    1. null importance test (>=50 shuffles) -> survivors,
    2. (optional) intersect with the top-``shap_top_n`` by mean|SHAP| (Phase 5.5),
    3. correlation pruning of the remaining set.

    ``shap_importance`` (from ``shap_importance_from_cv``) is optional so the
    funnel can run before any full CV, then be re-run with SHAP as the final
    arbiter once a CV pass exists.
    """
    y = np.asarray(y).astype(int)
    null_df = null_importance_test(importance_fn, X, y, n_shuffles=n_shuffles,
                                   n_jobs=n_jobs, seed=seed, save_csv=save)
    survivors = null_df.loc[null_df["keep"], "feature"].tolist()

    if shap_importance is not None and shap_top_n:
        top_shap = set(shap_importance.head(shap_top_n).index)
        intersected = [f for f in survivors if f in top_shap]
        # keep at least the null-survivors if the SHAP intersection is too thin
        survivors = intersected if len(intersected) >= 10 else survivors

    selected = correlation_prune(X, y, survivors, threshold=correlation_threshold)
    gi = gain_importance(importance_fn, X[selected] if selected else X, y, seed) \
        if selected else None

    res = FeatureSelectionResult(
        selected=selected, null_importance=null_df, shap_importance=shap_importance,
        gain_importance=gi, dropped_correlated=len(survivors) - len(selected),
        meta={"n_shuffles": n_shuffles, "correlation_threshold": correlation_threshold,
              "n_null_survivors": len(survivors), "n_selected": len(selected),
              "shap_top_n": shap_top_n},
    )
    if save:
        res.save()
    return res
