"""Model factories (GPU-aware, wide class-weight range) + F1-tuned CV harness
with optional GENEL augmentation appended to TRAINING folds only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from pathugen.model import scale_pos_weight
from search.reference.config import CV_REPEATS, POS, SEED, USE_GPU

_XGB_GPU = {"device": "cuda"} if USE_GPU else {}
_CAT_GPU = {"task_type": "GPU"} if USE_GPU else {}


def _split_cols(feature_cols, cat_cols):
    cat_set = set(cat_cols or [])
    return ([c for c in feature_cols if c in cat_set],
            [c for c in feature_cols if c not in cat_set])


def _preprocessor(feature_cols, cat_cols, scale):
    cat, num = _split_cols(feature_cols, cat_cols)
    tr = []
    if cat:
        steps = [("te", TargetEncoder(target_type="binary", smooth="auto", cv=5))]
        if scale:
            steps.append(("sc", StandardScaler()))
        tr.append(("cat", Pipeline(steps), cat))
    if num:
        steps = [("imp", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("sc", StandardScaler()))
        tr.append(("num", Pipeline(steps), num))
    return ColumnTransformer(tr, remainder="drop")


def make_xgb(params, y_tr, feature_cols=None, cat_cols=None):
    p = dict(params)
    spw = p.pop("spw_mult", 1.0)
    return XGBClassifier(n_estimators=p.pop("n_estimators", 400), tree_method="hist",
        eval_metric="logloss", scale_pos_weight=scale_pos_weight(y_tr) * spw,
        random_state=SEED, n_jobs=-1, **_XGB_GPU, **p)


def make_lgbm(params, y_tr, feature_cols=None, cat_cols=None):
    p = dict(params)
    spw = p.pop("spw_mult", 1.0)
    return LGBMClassifier(n_estimators=p.pop("n_estimators", 400), subsample_freq=1,
        scale_pos_weight=scale_pos_weight(y_tr) * spw, random_state=SEED, n_jobs=-1,
        verbosity=-1, **p)


def make_cat(params, y_tr, feature_cols=None, cat_cols=None):
    p = dict(params)
    return CatBoostClassifier(iterations=p.pop("iterations", 500),
        auto_class_weights="Balanced", random_seed=SEED, verbose=False,
        allow_writing_files=False, **_CAT_GPU, **p)


def make_extratrees(params, y_tr, feature_cols, cat_cols):
    p = dict(params)
    pre = _preprocessor(feature_cols, cat_cols, scale=False)
    clf = ExtraTreesClassifier(n_estimators=p.pop("n_estimators", 400),
        max_depth=p.pop("max_depth", None), min_samples_leaf=p.pop("min_samples_leaf", 1),
        max_features=p.pop("max_features", "sqrt"),
        class_weight=p.pop("class_weight", "balanced_subsample"),
        random_state=SEED, n_jobs=-1, **p)
    return Pipeline([("pre", pre), ("clf", clf)])


def make_logreg_l1(params, y_tr, feature_cols, cat_cols):
    p = dict(params)
    pre = _preprocessor(feature_cols, cat_cols, scale=True)
    clf = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=p.pop("C", 1.0),
        class_weight=p.pop("class_weight", "balanced"), max_iter=2000,
        random_state=SEED, **p)
    return Pipeline([("pre", pre), ("clf", clf)])


MAKERS = {"xgboost": make_xgb, "lightgbm": make_lgbm, "catboost": make_cat,
          "extratrees": make_extratrees, "logreg_l1": make_logreg_l1}
CAT_KW = {"lightgbm": "categorical_feature", "catboost": "cat_features"}
KINDS = tuple(MAKERS)
TREE_KINDS = ("xgboost", "lightgbm", "catboost")


def tune_threshold(y, proba, lo=0.05, hi=0.95, step=0.005):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        f1 = f1_score(y, (proba >= t).astype(int), pos_label=POS, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def _append_aug(X_tr, y_tr, aug):
    if aug is None or aug[0] is None:
        return X_tr, y_tr
    Xa, ya = aug
    Xc = pd.concat([X_tr, Xa[X_tr.columns]], axis=0, ignore_index=True)
    yc = pd.concat([pd.Series(np.asarray(y_tr)), pd.Series(np.asarray(ya))],
                   ignore_index=True)
    return Xc, yc


def run_cv(make_model, X, y, cat_cols=None, n_splits=5, seed=SEED, fit_kwargs_fn=None,
           n_repeats=None, aug=None):
    """StratifiedKFold OOF + F1 metrics. aug rows go to TRAIN folds only."""
    n_repeats = CV_REPEATS if n_repeats is None else n_repeats
    y = pd.Series(np.asarray(y), index=X.index)
    fold_thresholds = []

    def _fit(tr_idx):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_fit, y_fit = _append_aug(X_tr, y_tr, aug)
        m = make_model(y_fit)
        fk = fit_kwargs_fn(X_fit, y_fit) if fit_kwargs_fn else {}
        m.fit(X_fit, y_fit, **fk)
        return m

    if n_repeats > 1:
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                           random_state=seed)
        oof_sum, oof_cnt = np.zeros(len(y)), np.zeros(len(y))
        for tr, va in splitter.split(X, y):
            p = _fit(tr).predict_proba(X.iloc[va])[:, 1]
            np.add.at(oof_sum, va, p)
            np.add.at(oof_cnt, va, 1)
            ft, _ = tune_threshold(y.iloc[va].to_numpy(), p)
            fold_thresholds.append(ft)
        oof = oof_sum / np.maximum(oof_cnt, 1)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, va in skf.split(X, y):
            p = _fit(tr).predict_proba(X.iloc[va])[:, 1]
            oof[va] = p
            ft, _ = tune_threshold(y.iloc[va].to_numpy(), p)
            fold_thresholds.append(ft)

    pooled_t, pooled_f1 = tune_threshold(y.to_numpy(), oof)
    # fold-median threshold: robust to one lucky fold on the small panels.
    threshold = float(np.median(fold_thresholds)) if fold_thresholds else pooled_t
    f1 = f1_score(y, (oof >= threshold).astype(int), pos_label=POS, zero_division=0)
    return {"oof": oof, "threshold": threshold, "f1": float(f1),
            "f1_pooled": pooled_f1, "auc": float(roc_auc_score(y, oof)),
            "ap": float(average_precision_score(y, oof))}
