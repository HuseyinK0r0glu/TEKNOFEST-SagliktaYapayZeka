"""Shared per-panel model factories and CV evaluation.

Centralises the XGBoost / LightGBM / CatBoost configurations so the baseline
runner and the feature-selection runner stay in sync. Class weights are set
from each fold's own training labels (Pathogenic is the majority class, so
scale_pos_weight < 1).
"""
from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src import config, cv

SEED = config.SEED


def scale_pos_weight(y_tr) -> float:
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    return n_neg / n_pos if n_pos else 1.0


def make_xgb(y_tr):
    # min_child_weight=1: a higher sum-of-hessians floor combined with
    # scale_pos_weight<1 blocks all splits on small panels (degenerate model).
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=1,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=scale_pos_weight(y_tr),
        tree_method="hist", eval_metric="logloss",
        random_state=SEED, n_jobs=-1,
    )


def make_lgbm(y_tr):
    return LGBMClassifier(
        n_estimators=400, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=scale_pos_weight(y_tr),
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


def make_cat(y_tr):
    return CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
        auto_class_weights="Balanced", random_seed=SEED, verbose=False,
        allow_writing_files=False,
    )


# name -> (factory, fit-kwarg name carrying the categorical column list, or None)
MODELS = {
    "xgboost": (make_xgb, None),
    "lightgbm": (make_lgbm, "categorical_feature"),
    "catboost": (make_cat, "cat_features"),
}


def _fit_kwargs_fn(cat_kw, cat_cols):
    if cat_kw is None or not cat_cols:
        return None
    return lambda X_tr, y_tr: {cat_kw: list(cat_cols)}


def evaluate_models(X, y, cat_cols, model_names=None):
    """Run F1-tuned CV for each model. Returns {name: cv.run_cv result dict}."""
    names = model_names or list(MODELS)
    out = {}
    for name in names:
        make, cat_kw = MODELS[name]
        out[name] = cv.run_cv(
            make, X, y, cat_cols=cat_cols,
            fit_kwargs_fn=_fit_kwargs_fn(cat_kw, cat_cols),
        )
    return out
