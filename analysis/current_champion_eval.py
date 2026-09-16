"""Fresh, honest evaluation of the CURRENT champion_config.json recipe.

Unlike report_figs/repro_metrics.json (frozen from the pre-refactor 5-model
reference run, see WARNING.md and Champion_Config.md), this reads the live
champion_config.json and pathugen_model/ bundles and measures the recipe
that TRAIN and DELIVER actually ship today: one tuned XGBoost per panel,
3-seed ensemble, frozen feature list and threshold.

Three things, all computed against the exact production code
(pathugen.features / pathugen.model / pathugen.augment):

1. Honest generalisation: repeated 5-fold outer CV (5 splits x 3 repeats).
   The recipe (features, xgb_params, threshold) is NOT re-derived per fold
   -- it is already frozen in champion_config.json -- only refit and scored
   on unseen rows. This carries the same small bias WARNING.md documents
   for hyperparameters, now also covering feature selection and threshold
   (both already fixed upstream by the sweep's own nested-CV audit).
2. Feature importance: gain-based `feature_importances_` averaged across
   the 3 shipped seeds per panel (pathugen_model/<PANEL>/v2/seed*.joblib).
3. Inference latency: wall-clock batch predict_proba on each panel's own
   rows with the shipped 3-seed ensemble.

Run from the project root:
    PYTHONPATH=. .venv/bin/python analysis/current_champion_eval.py
Writes report_figs/current_champion_metrics.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from pathugen.augment import build_aug_pool
from pathugen.champion_config import load_champion_config
from pathugen.config import MODEL_DIR, PANELS, SEED
from pathugen.data_io import load_panel
from pathugen.features import build_features, get_cat_maps
from pathugen.model import fit_seed_ensemble, predict_proba_ensemble

N_SPLITS, N_REPEATS = 5, 3
OUT_PATH = Path(__file__).resolve().parents[1] / "report_figs" / "current_champion_metrics.json"


def honest_cv(panel, cfg):
    df = load_panel(panel)
    cat_maps = get_cat_maps(df)
    X, y, _ = build_features(df, cat_maps=cat_maps)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    feats = cfg["features"]
    Xs = X[feats].reset_index(drop=True)
    thr = cfg["threshold"]

    aug_X, aug_y = build_aug_pool(panel, cfg["aug_frac"], columns=feats)
    aug = (aug_X, aug_y) if aug_X is not None else None

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    oof_sum = np.zeros(len(y))
    oof_cnt = np.zeros(len(y))
    fold_f1, fold_auc = [], []

    for tr_idx, va_idx in rskf.split(Xs, y):
        X_tr, X_va = Xs.iloc[tr_idx], Xs.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        if aug is not None:
            X_tr = pd.concat([X_tr, aug[0]], axis=0, ignore_index=True)
            y_tr = pd.concat([y_tr, aug[1]], axis=0, ignore_index=True)
        models = fit_seed_ensemble(X_tr, y_tr, cfg["xgb_params"])
        proba = predict_proba_ensemble(models, X_va)
        np.add.at(oof_sum, va_idx.astype(int), proba)
        np.add.at(oof_cnt, va_idx.astype(int), 1)
        pred = (proba >= thr).astype(int)
        fold_f1.append(f1_score(y_va, pred, pos_label=1, zero_division=0))
        fold_auc.append(roc_auc_score(y_va, proba))

    oof = oof_sum / np.maximum(oof_cnt, 1)
    pred_pooled = (oof >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_pooled, labels=[0, 1]).ravel()
    n = len(y)
    naive_f1 = f1_score(y, np.ones(n), pos_label=1, zero_division=0)

    return dict(
        n=n, n_features=len(feats), threshold=thr, threshold_policy=cfg["threshold_policy"],
        selection_method=cfg["selection_method"], aug_frac=cfg["aug_frac"],
        aug_rows=int(len(aug_y)) if aug is not None else 0,
        honest_f1_mean=float(np.mean(fold_f1)), honest_f1_std=float(np.std(fold_f1)),
        honest_auc_mean=float(np.mean(fold_auc)), honest_auc_std=float(np.std(fold_auc)),
        pooled_f1_at_thr=float(f1_score(y, pred_pooled, pos_label=1, zero_division=0)),
        pooled_auc=float(roc_auc_score(y, oof)),
        naive_f1_all_pathogenic=float(naive_f1),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        accuracy=float((tn + tp) / n), error_rate=float((fp + fn) / n),
        sensitivity=float(tp / (tp + fn)) if (tp + fn) else None,
        specificity=float(tn / (tn + fp)) if (tn + fp) else None,
        precision=float(tp / (tp + fp)) if (tp + fp) else None,
    )


def feature_importance(panel, cfg, model_dir=MODEL_DIR, version=None):
    from pathugen.versioning import _panel_versions
    v = version or max(_panel_versions(panel, model_dir))
    d = model_dir / panel / f"v{v}"
    feats = json.loads((d / "selected_features.json").read_text())
    models = [joblib.load(d / f"seed{i}.joblib") for i in range(3)]
    avg = np.mean([m.feature_importances_ for m in models], axis=0)
    order = np.argsort(-avg)
    return [(feats[j], float(avg[j])) for j in order], v


def latency(panel, cfg, version):
    d = MODEL_DIR / panel / f"v{version}"
    cat_maps = joblib.load(d / "cat_maps.joblib")
    feats = json.loads((d / "selected_features.json").read_text())
    models = [joblib.load(d / f"seed{i}.joblib") for i in range(3)]
    df = load_panel(panel)
    X, _, _ = build_features(df, cat_maps=cat_maps)
    Xs = X[feats]
    for m in models:
        m.predict_proba(Xs.iloc[:5])[:, 1]
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        predict_proba_ensemble(models, Xs)
    dt = (time.perf_counter() - t0) / reps
    return dict(n=len(Xs), batch_ms=dt * 1000, per_variant_us=dt / len(Xs) * 1e6)


def main():
    cfg_all = load_champion_config()
    out = {}
    for panel in PANELS:
        cfg = cfg_all[panel]
        cv = honest_cv(panel, cfg)
        feats, version = feature_importance(panel, cfg)
        lat = latency(panel, cfg, version)
        out[panel] = dict(cv=cv, top_features=feats[:10], model_version=version, latency=lat)
        print(f"{panel:7s} honest_F1={cv['honest_f1_mean']:.4f}+-{cv['honest_f1_std']:.4f} "
              f"honest_AUC={cv['honest_auc_mean']:.4f} acc={cv['accuracy']:.4f} "
              f"lat={lat['per_variant_us']:.1f}us/variant top1={feats[0][0]}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved ->", OUT_PATH)


if __name__ == "__main__":
    main()
