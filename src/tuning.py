"""Per-panel Optuna hyperparameter tuning, optimised for binary F1.

The objective is the same F1 the CV harness already reports (best-threshold F1
on the Pathogenic class), so tuning directly targets the competition metric.
Folds are held fixed across trials (same seed) for a fair comparison. Each
study is bounded by both a trial count and a wall-clock timeout.
"""
from __future__ import annotations

import os

import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src import config, cv
from src import diverse_models as dm
from src import models as M

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


# --- Anti-overfitting knobs (off by default; profiles in main.py set them) ---
# EARLY_STOP: boosted trees use a large iteration cap + early stopping on a
#   fold-internal validation slice instead of a fixed 400, picking the right
#   number of trees per fold. ES_ROUNDS is the patience.
EARLY_STOP = _flag("EARLY_STOP")
ES_ROUNDS = int(os.environ.get("EARLY_STOP_ROUNDS", "50"))
_N_EST = 2000 if EARLY_STOP else 400
# OPTUNA_OBJECTIVE: metric Optuna maximises. "auc"/"ap" are threshold-free and
#   decouple the hyperparameter search from threshold optimism; "f1" is the
#   original best-threshold objective.
OBJECTIVE = os.environ.get("OPTUNA_OBJECTIVE", "f1").strip().lower()
# OPTUNA_SEED_JITTER: perturb the CV fold seed per trial so hyperparameters that
#   only win on one fixed split stop winning (the enqueued defaults trial #0
#   stays on the canonical seed, preserving the baseline floor).
SEED_JITTER = _flag("OPTUNA_SEED_JITTER")


def _small(n_rows) -> bool:
    """Small-panel flag — tightens model capacity to curb small-n overfit."""
    return n_rows is not None and n_rows < 400


# Upper bound on the scale_pos_weight MULTIPLIER. The base weight is the raw
# n_neg/n_pos ratio (e.g. PAH ~0.20, CFTR ~0.23): with the old 2.0 cap the
# effective weight on the imbalanced panels topped out at ~0.4, so the search
# could never reach the equal-weight (1.0) regime the strategy doc recommends for
# PAH. 6.0 lets the search span from the raw ratio up through equal weights to
# benign-up-weighting; the enqueued default (spw_mult=1.0) keeps the floor.
_SPW_MULT_HI = float(os.environ.get("SPW_MULT_HI", "6.0"))

# All factories share one signature: make(params, y_tr, feature_cols, cat_cols).
# The boosted-tree factories ignore feature_cols/cat_cols (they receive the
# categorical column list as a fit kwarg instead); the sklearn-pipeline factories
# in `diverse_models` need the column names to wire their ColumnTransformer.


def _xgb_space(trial, n_rows=None):
    # Small panels: cap tree DEPTH only. The min_child_weight floor stays at 1 —
    # raising it with scale_pos_weight<1 blocks all splits on small panels
    # (degenerate model; see src/models.make_xgb) and would also collide with the
    # enqueued defaults.
    small = _small(n_rows)
    return dict(
        max_depth=trial.suggest_int("max_depth", 2, 5 if small else 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # Multiplier on the natural n_neg/n_pos class weight. Lets the search
        # trade recall vs precision instead of being locked to the raw ratio —
        # the main lever on the imbalanced panels (PAH 5:1, CFTR 4:1). 1.0 is
        # the current behaviour (and the enqueued default keeps the floor); the
        # widened upper bound lets PAH/CFTR reach the equal-weight regime.
        spw_mult=trial.suggest_float("spw_mult", 0.5, _SPW_MULT_HI, log=True),
    )


def _make_xgb(params, y_tr, feature_cols=None, cat_cols=None):
    es = dict(early_stopping_rounds=ES_ROUNDS) if EARLY_STOP else {}
    params = dict(params)
    spw_mult = params.pop("spw_mult", 1.0)
    return XGBClassifier(
        n_estimators=_N_EST, tree_method="hist", eval_metric="logloss",
        scale_pos_weight=M.scale_pos_weight(y_tr) * spw_mult,
        random_state=config.SEED, n_jobs=-1, **es, **params,
    )


def _lgbm_space(trial, n_rows=None):
    # Small panels: cap num_leaves (capacity) only; leave the min_child_samples
    # range as-is so the enqueued default (20) stays valid and Optuna can still
    # reach the high-regularization end (100).
    small = _small(n_rows)
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 63 if small else 255),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
        spw_mult=trial.suggest_float("spw_mult", 0.5, _SPW_MULT_HI, log=True),
    )


def _make_lgbm(params, y_tr, feature_cols=None, cat_cols=None):
    params = dict(params)
    spw_mult = params.pop("spw_mult", 1.0)
    return LGBMClassifier(
        n_estimators=_N_EST, subsample_freq=1,
        scale_pos_weight=M.scale_pos_weight(y_tr) * spw_mult,
        random_state=config.SEED, n_jobs=-1, verbosity=-1, **params,
    )


def _cat_space(trial, n_rows=None):
    small = _small(n_rows)
    return dict(
        depth=trial.suggest_int("depth", 3, 6 if small else 9),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
        random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
    )


def _make_cat(params, y_tr, feature_cols=None, cat_cols=None):
    es = dict(od_type="Iter", od_wait=ES_ROUNDS) if EARLY_STOP else {}
    return CatBoostClassifier(
        iterations=_N_EST, auto_class_weights="Balanced",
        random_seed=config.SEED, verbose=False, allow_writing_files=False,
        **es, **params,
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

# Known-good hand-tuned defaults (mirror src.models / library defaults). Enqueued
# as the first Optuna trial per study so a search can never score BELOW the
# baseline config — fixes observed regressions (e.g. PAH/CFTR xgboost) where 40
# blind trials failed to rediscover the default. Keys must match the space's
# suggest_* names and stay inside their ranges.
DEFAULTS = {
    "xgboost": dict(max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.6, min_child_weight=1, gamma=0.0,
                    reg_alpha=0.1, reg_lambda=1.0, spw_mult=1.0),
    "lightgbm": dict(num_leaves=31, learning_rate=0.05, min_child_samples=20,
                     subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1,
                     reg_lambda=1.0, min_split_gain=0.0, spw_mult=1.0),
    "catboost": dict(depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
                     bagging_temperature=1.0, random_strength=1.0),
    "extratrees": dict(n_estimators=400, max_depth=12, min_samples_leaf=1,
                       max_features=0.5, class_weight="balanced_subsample"),
    "logreg_l1": dict(C=1.0, class_weight="balanced"),
}


def eval_kwargs_fn(model_kind):
    """Early-stopping fit-kwargs builder for one model kind, or None.

    Returns f(X_es, y_es) -> dict of model-specific eval-set kwargs, or None when
    EARLY_STOP is off or the model does not iterate (extratrees/logreg). Passed to
    cv.run_cv so each fold early-stops on an internal validation slice.
    """
    if not EARLY_STOP:
        return None
    if model_kind == "xgboost":
        return lambda Xe, ye: {"eval_set": [(Xe, ye)], "verbose": False}
    if model_kind == "lightgbm":
        import lightgbm as lgb
        return lambda Xe, ye: {
            "eval_set": [(Xe, ye)],
            "callbacks": [lgb.early_stopping(ES_ROUNDS, verbose=False)],
        }
    if model_kind == "catboost":
        return lambda Xe, ye: {"eval_set": (Xe, ye)}
    return None


def fit_full(model, X, y, model_kind, fit_kwargs=None, seed=config.SEED,
             es_frac=0.15):
    """Fit a model on the full data, carving an internal early-stopping slice
    when EARLY_STOP is on (so XGB/CatBoost, which require an eval_set once
    early stopping is configured, can refit for prediction). No-op wrapper around
    plain .fit when early stopping is off.
    """
    fit_kwargs = dict(fit_kwargs or {})
    eval_fn = eval_kwargs_fn(model_kind)
    if eval_fn is None:
        model.fit(X, y, **fit_kwargs)
        return model
    X_fit, X_es, y_fit, y_es = train_test_split(
        X, y, test_size=es_frac, stratify=y, random_state=seed
    )
    fit_kwargs.update(eval_fn(X_es, y_es))
    model.fit(X_fit, y_fit, **fit_kwargs)
    return model


def tune_model(model_kind, X, y, cat_cols, n_trials=40, timeout=150, seed=config.SEED,
               aug=None):
    """Return (best_params, best_score, n_completed_trials) for one model on one panel.

    best_score is the OPTUNA_OBJECTIVE metric (f1/auc/ap) maximised over trials.
    aug: optional (X_aug, y_aug) external training rows passed through to cv.run_cv
        so tuning evaluates models in the same augmented regime they ship in.
    """
    space, make, cat_kw = SPACES[model_kind]
    use_cat = bool(cat_kw and cat_cols)
    fit_fn = (lambda X_tr, y_tr: {cat_kw: list(cat_cols)}) if use_cat else None
    eval_fn = eval_kwargs_fn(model_kind)
    feature_cols = list(X.columns)
    n_rows = len(y)
    if aug is not None and aug[0] is not None:
        aug = (aug[0].reindex(columns=feature_cols), aug[1])

    def objective(trial):
        params = space(trial, n_rows)
        # Jitter the fold seed per trial so winners generalise across splits; the
        # enqueued defaults (trial #0) stay on the canonical seed.
        trial_seed = seed + trial.number if SEED_JITTER else seed
        res = cv.run_cv(
            lambda y_tr: make(params, y_tr, feature_cols, cat_cols), X, y,
            cat_cols=cat_cols, fit_kwargs_fn=fit_fn, eval_kwargs_fn=eval_fn,
            seed=trial_seed, aug=aug,
        )
        return res.get(OBJECTIVE, res["f1"])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    if model_kind in DEFAULTS:
        study.enqueue_trial(DEFAULTS[model_kind])  # evaluate the baseline config first
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params, float(study.best_value), len(study.trials)


def build_estimator(model_kind, params, y_tr, feature_cols=None, cat_cols=None):
    """Rebuild an estimator from a saved Optuna param dict (for stacking)."""
    return SPACES[model_kind][1](params, y_tr, feature_cols, cat_cols)


def cat_kwarg(model_kind):
    """Name of the fit kwarg carrying categorical columns, or None."""
    return SPACES[model_kind][2]
