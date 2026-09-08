"""Optuna hyperparameter search spaces and the per-model-kind tuning entry
point (`tune_model`)."""
from __future__ import annotations

import optuna

from search.reference.config import CV_REPEATS, OBJECTIVE, SEED, SEED_JITTER, SPW_MULT_HI
from search.reference.modeling import MAKERS, CAT_KW, run_cv

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _small(n): return n is not None and n < 400


def _space(kind, trial, n_rows):
    s = _small(n_rows)
    if kind == "xgboost":
        return dict(max_depth=trial.suggest_int("max_depth", 2, 5 if s else 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            spw_mult=trial.suggest_float("spw_mult", 0.5, SPW_MULT_HI, log=True))
    if kind == "lightgbm":
        return dict(num_leaves=trial.suggest_int("num_leaves", 15, 63 if s else 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            spw_mult=trial.suggest_float("spw_mult", 0.5, SPW_MULT_HI, log=True))
    if kind == "catboost":
        return dict(depth=trial.suggest_int("depth", 3, 6 if s else 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0))
    if kind == "extratrees":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 12 if s else 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            class_weight=trial.suggest_categorical("class_weight",
                ["balanced_subsample", "balanced", None]))
    return dict(C=trial.suggest_float("C", 1e-3, 20.0, log=True),
        class_weight=trial.suggest_categorical("class_weight", ["balanced", None]))


DEFAULTS = {
    "xgboost": dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
        min_child_weight=1, gamma=0.0, reg_alpha=0.1, reg_lambda=1.0, spw_mult=1.0),
    "lightgbm": dict(num_leaves=31, learning_rate=0.05, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
        min_split_gain=0.0, spw_mult=1.0),
    "catboost": dict(depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
        bagging_temperature=1.0, random_strength=1.0),
    "extratrees": dict(n_estimators=400, max_depth=12, min_samples_leaf=1,
        max_features=0.5, class_weight="balanced_subsample"),
    "logreg_l1": dict(C=1.0, class_weight="balanced"),
}


def _fit_fn(kind, cat_cols):
    kw = CAT_KW.get(kind)
    if kw and cat_cols:
        return lambda X_tr, y_tr: {kw: list(cat_cols)}
    return None


def tune_model(kind, X, y, cat_cols, n_trials, timeout, seed=SEED, aug=None):
    feature_cols = list(X.columns)
    fit_fn = _fit_fn(kind, cat_cols)
    if aug is not None and aug[0] is not None:
        aug = (aug[0].reindex(columns=feature_cols), aug[1])
    n_rows = len(y)

    def objective(trial):
        params = _space(kind, trial, n_rows)
        tseed = seed + trial.number if SEED_JITTER else seed
        res = run_cv(lambda y_tr: MAKERS[kind](params, y_tr, feature_cols, cat_cols),
                     X, y, cat_cols=cat_cols, fit_kwargs_fn=fit_fn, seed=tseed,
                     aug=aug, n_repeats=max(1, CV_REPEATS - 1))
        return res.get(OBJECTIVE, res["f1"])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(DEFAULTS[kind])
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params, float(study.best_value)
