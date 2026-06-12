"""Structurally diverse base models for stacking (ExtraTrees, L1-LogReg).

The three boosted-tree bases (xgb/lgbm/cat) are highly correlated on the small
panels, so the earlier stacking pass only beat the best single model on PAH. These
two models bring a genuinely different inductive bias — bagged extremely-randomised
trees and a sparse linear model — to decorrelate the base OOF columns and give the
logistic meta-learner something worth combining.

Unlike the boosters, neither model consumes NaN or raw category codes natively, so
each is a self-contained sklearn Pipeline:

  - categorical code columns -> TargetEncoder (cross-fitted inside fit, leak-free)
  - numeric AL/EK columns    -> median impute (+ standardise for the linear model)

Because the preprocessing lives inside the Pipeline, it is refit per CV fold
automatically and the target encoder never sees validation rows. sklearn's
TargetEncoder uses internal K-fold cross-fitting during `fit_transform` (training)
and the full-train encoding during `transform` (prediction), so there is no target
leakage even though the encoding is label-dependent.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from src import config

SEED = config.SEED


def _split_cols(feature_cols, cat_cols):
    """Partition the feature list into (categorical-code, numeric) name lists."""
    cat_set = set(cat_cols or [])
    cat = [c for c in feature_cols if c in cat_set]
    num = [c for c in feature_cols if c not in cat_set]
    return cat, num


def _target_encoder() -> TargetEncoder:
    # cv=5 internal cross-fitting prevents target leakage during fit_transform.
    return TargetEncoder(target_type="binary", smooth="auto", cv=5)


def _preprocessor(feature_cols, cat_cols, *, scale: bool) -> ColumnTransformer:
    """Build a leak-free column transformer.

    scale=False (trees): target-encode cats, median-impute numerics.
    scale=True  (linear): additionally standardise every column.
    """
    cat, num = _split_cols(feature_cols, cat_cols)
    transformers = []
    if cat:
        cat_steps = [("te", _target_encoder())]
        if scale:
            cat_steps.append(("sc", StandardScaler()))
        transformers.append(("cat", Pipeline(cat_steps), cat))
    if num:
        num_steps = [("imp", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("sc", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num))
    return ColumnTransformer(transformers, remainder="drop")


def make_extratrees(params, y_tr, feature_cols, cat_cols) -> Pipeline:
    """Bagged extremely-randomised trees over target-encoded + imputed features."""
    p = dict(params)
    pre = _preprocessor(feature_cols, cat_cols, scale=False)
    clf = ExtraTreesClassifier(
        n_estimators=p.pop("n_estimators", 400),
        max_depth=p.pop("max_depth", None),
        min_samples_leaf=p.pop("min_samples_leaf", 1),
        max_features=p.pop("max_features", "sqrt"),
        class_weight=p.pop("class_weight", "balanced_subsample"),
        random_state=SEED, n_jobs=-1, **p,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def make_logreg_l1(params, y_tr, feature_cols, cat_cols) -> Pipeline:
    """Sparse L1-penalised logistic regression over standardised features.

    Uses the non-deprecated `l1_ratio=1.0` form (equivalent to the old
    `penalty="l1"`) so liblinear still performs pure L1 without the sklearn
    1.8+ `penalty` deprecation warning.
    """
    p = dict(params)
    pre = _preprocessor(feature_cols, cat_cols, scale=True)
    clf = LogisticRegression(
        solver="liblinear", l1_ratio=1.0,
        C=p.pop("C", 1.0),
        class_weight=p.pop("class_weight", "balanced"),
        max_iter=2000, random_state=SEED, **p,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# --- Optuna search spaces (kept small; these models are cheap to fit) ---

def extratrees_space(trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_depth=trial.suggest_int("max_depth", 4, 24),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_float("max_features", 0.3, 1.0),
        class_weight=trial.suggest_categorical(
            "class_weight", ["balanced_subsample", "balanced", None]
        ),
    )


def logreg_l1_space(trial) -> dict:
    return dict(
        C=trial.suggest_float("C", 1e-3, 20.0, log=True),
        class_weight=trial.suggest_categorical("class_weight", ["balanced", None]),
    )
