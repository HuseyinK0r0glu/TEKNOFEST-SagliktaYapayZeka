"""Per-panel Optuna hyperparameter tuning, optimised for binary F1.

The objective is the same F1 the CV harness already reports (best-threshold F1
on the Pathogenic class), so tuning directly targets the competition metric.
Folds are held fixed across trials (same seed) for a fair comparison. Each
study is bounded by both a trial count and a wall-clock timeout.
"""
from __future__ import annotations

import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src import config, cv
from src import diverse_models as dm
from src import models as M

optuna.logging.set_verbosity(optuna.logging.WARNING)

# All factories share one signature: make(params, y_tr, feature_cols, cat_cols).
# The boosted-tree factories ignore feature_cols/cat_cols (they receive the
# categorical column list as a fit kwarg instead); the sklearn-pipeline factories
# in `diverse_models` need the column names to wire their ColumnTransformer.


def _xgb_space(trial):
    return dict(
        max_depth=trial.suggest_int("max_depth", 2, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def _make_xgb(params, y_tr, feature_cols=None, cat_cols=None):
    return XGBClassifier(
        n_estimators=400, tree_method="hist", eval_metric="logloss",
        scale_pos_weight=M.scale_pos_weight(y_tr),
        random_state=config.SEED, n_jobs=-1, **params,
    )


def _lgbm_space(trial):
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
    )


def _make_lgbm(params, y_tr, feature_cols=None, cat_cols=None):
    return LGBMClassifier(
        n_estimators=400, subsample_freq=1,
        scale_pos_weight=M.scale_pos_weight(y_tr),
        random_state=config.SEED, n_jobs=-1, verbosity=-1, **params,
    )


def _cat_space(trial):
    return dict(
        depth=trial.suggest_int("depth", 3, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
        random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
    )


def _make_cat(params, y_tr, feature_cols=None, cat_cols=None):
    return CatBoostClassifier(
        iterations=400, auto_class_weights="Balanced",
        random_seed=config.SEED, verbose=False, allow_writing_files=False, **params,
    )


# kind -> (search-space fn, factory, fit-kwarg name carrying the cat column list).
# The two sklearn-pipeline kinds carry None because they encode categoricals
# inside their own ColumnTransformer rather than via a fit kwarg.
SPACES = {
    "xgboost": (_xgb_space, _make_xgb, None),
    "lightgbm": (_lgbm_space, _make_lgbm, "categorical_feature"),
    "catboost": (_cat_space, _make_cat, "cat_features"),
    "extratrees": (dm.extratrees_space, dm.make_extratrees, None),
    "logreg_l1": (dm.logreg_l1_space, dm.make_logreg_l1, None),
}

# Canonical ordered list of every base model, used by tuning + stacking runners.
MODEL_KINDS = tuple(SPACES)


def tune_model(model_kind, X, y, cat_cols, n_trials=40, timeout=150, seed=config.SEED):
    """Return (best_params, best_f1, n_completed_trials) for one model on one panel."""
    space, make, cat_kw = SPACES[model_kind]
    use_cat = bool(cat_kw and cat_cols)
    fit_fn = (lambda X_tr, y_tr: {cat_kw: list(cat_cols)}) if use_cat else None
    feature_cols = list(X.columns)

    def objective(trial):
        params = space(trial)
        res = cv.run_cv(
            lambda y_tr: make(params, y_tr, feature_cols, cat_cols), X, y,
            cat_cols=cat_cols, fit_kwargs_fn=fit_fn, seed=seed,
        )
        return res["f1"]

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params, float(study.best_value), len(study.trials)


def build_estimator(model_kind, params, y_tr, feature_cols=None, cat_cols=None):
    """Rebuild an estimator from a saved Optuna param dict (for stacking)."""
    return SPACES[model_kind][1](params, y_tr, feature_cols, cat_cols)


def cat_kwarg(model_kind):
    """Name of the fit kwarg carrying categorical columns, or None."""
    return SPACES[model_kind][2]
