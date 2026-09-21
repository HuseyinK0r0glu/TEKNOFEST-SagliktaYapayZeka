"""TRAIN — fit the champion configuration for every panel from the raw
training CSVs. Every call creates a NEW version (v1, v2, ...); previous
versions are kept.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from pathugen.augment import build_aug_pool
from pathugen.champion_config import load_champion_config
from pathugen.config import DATA_DIR, MODEL_DIR, PANELS, SEED, SEED_ENSEMBLE
from pathugen.data_io import load_panel
from pathugen.features import binary_al_columns, build_features, get_cat_maps
from pathugen.model import fit_seed_ensemble, make_xgb
from pathugen.versioning import _panel_versions, next_model_version

CALIBRATION_SPLITS = 5


def fit_calibrator(Xs, y, aug, params):
    """Isotonic calibrator, fit on leak-free out-of-fold probabilities.

    This is used by NOTHING on the default decision path — `deliver` only
    touches it when `--prior-shift` is passed, to move the threshold onto a
    scale where the Bayes prior-shift formula is actually valid. The raw
    probabilities the models emit are deliberately NOT calibrated (they are
    fit with scale_pos_weight != 1, which distorts the probability scale), so
    shifting a threshold on them is only approximate.

    Fit on OUT-OF-FOLD predictions, never in-sample: the model is far too
    confident on rows it was trained on, and an in-sample isotonic fit would
    collapse into a near-step function that calibrates nothing. Validation
    folds stay on the panel's own rows; augmented rows go into the training
    folds only, exactly as in the search pipeline that chose this threshold.

    Returns None when a panel is too small to split, in which case the bundle
    simply ships without a calibrator and `--prior-shift` falls back to its
    previous (approximate) behaviour.
    """
    y_arr = np.asarray(y)
    n_splits = min(CALIBRATION_SPLITS, int(np.bincount(y_arr).min()))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_arr), dtype=float)
    for tr_idx, va_idx in skf.split(Xs, y_arr):
        X_tr, y_tr = Xs.iloc[tr_idx], y_arr[tr_idx]
        if aug is not None:
            X_tr = pd.concat([X_tr, aug[0]], axis=0, ignore_index=True)
            y_tr = np.concatenate([y_tr, np.asarray(aug[1])])
        model = make_xgb(params, y_tr, seed=SEED)
        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict_proba(Xs.iloc[va_idx])[:, 1]
    return IsotonicRegression(out_of_bounds="clip").fit(oof, y_arr)


def train(panels=None, model_dir: Path = MODEL_DIR) -> None:
    panel_config = load_champion_config()
    panels = panels or PANELS
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training PatHUgen — champion configuration, {SEED_ENSEMBLE} seeds/panel")
    print(f"Data dir  : {DATA_DIR.resolve()}")
    print(f"Model dir : {model_dir.resolve()}\n")

    for panel in panels:
        cfg = panel_config[panel]
        t0 = datetime.now()
        df = load_panel(panel)
        cat_maps = get_cat_maps(df)
        bin_al_cols = binary_al_columns(df)
        X, y, _ = build_features(df, cat_maps=cat_maps, bin_cols=bin_al_cols)
        y = pd.Series(np.asarray(y))
        train_prior = float(y.mean())

        missing = [f for f in cfg["features"] if f not in X.columns]
        if missing:
            raise RuntimeError(
                f"{panel}: champion feature list references columns not "
                f"produced by build_features(): {missing}")
        Xs = X[cfg["features"]]

        aug_X, aug_y = build_aug_pool(panel, cfg["aug_frac"], columns=Xs.columns,
                                      bin_cols=bin_al_cols)
        if aug_X is not None:
            Xs_full = pd.concat([Xs, aug_X], axis=0, ignore_index=True)
            y_full = pd.concat([y, aug_y], axis=0, ignore_index=True)
            print(f"  {panel}: {len(X)} rows + {len(aug_X)} augmented GENEL rows "
                  f"= {len(Xs_full)} training rows, {len(cfg['features'])} features")
        else:
            Xs_full, y_full = Xs, y
            print(f"  {panel}: {len(X)} rows (no augmentation), "
                  f"{len(cfg['features'])} features")

        models = fit_seed_ensemble(Xs_full, y_full, cfg["xgb_params"])
        aug_pair = (aug_X, aug_y) if aug_X is not None else None
        calibrator = fit_calibrator(Xs, y, aug_pair, cfg["xgb_params"])
        # The audited operating point, expressed on the calibrated scale.
        # Isotonic is monotonic, so this is the same decision boundary — it is
        # recorded here only so `--prior-shift` has something valid to move.
        calibrated_threshold = (
            float(calibrator.predict(np.array([cfg["threshold"]]))[0])
            if calibrator is not None else None)

        version = next_model_version(panel, model_dir)
        panel_dir = model_dir / panel / f"v{version}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(models):
            joblib.dump(m, panel_dir / f"seed{i}.joblib")
        joblib.dump(cat_maps, panel_dir / "cat_maps.joblib")
        if calibrator is not None:
            joblib.dump(calibrator, panel_dir / "calibrator.joblib")
        (panel_dir / "selected_features.json").write_text(
            json.dumps(cfg["features"], indent=2))
        (panel_dir / "bin_al_columns.json").write_text(
            json.dumps(bin_al_cols, indent=2))
        (panel_dir / "config.json").write_text(json.dumps({
            "panel": panel, "version": version, "n_train_rows": int(len(X)),
            "n_augmented_rows": int(len(aug_X)) if aug_X is not None else 0,
            "n_features": len(cfg["features"]), "seed_ensemble": SEED_ENSEMBLE,
            "threshold": cfg["threshold"], "threshold_policy": cfg["threshold_policy"],
            "has_calibrator": calibrator is not None,
            "calibrated_threshold": calibrated_threshold,
            "train_prior": train_prior, "trained_at": t0.isoformat(),
        }, indent=2))
        prior_versions = [v for v in _panel_versions(panel, model_dir) if v != version]
        kept = f" (previous kept: {', '.join('v' + str(v) for v in prior_versions)})" \
               if prior_versions else ""
        print(f"    -> {panel_dir}{kept}  ({(datetime.now()-t0).total_seconds():.0f}s)")

    print(f"\nDone. Model bundle ready at {model_dir.resolve()}")
    print("Every panel now has a new version; nothing was overwritten. "
          "`deliver` uses the newest version of each panel automatically — "
          "run `python PatHUgen.py versions` to see everything you have "
          "trained, or pin an older one with `deliver --model-version N`.")
    print("The resolved version directory is what Section 6 of the delivery "
          "guide calls the model package — it ships as-is inside every "
          "delivery folder.")
