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

from pathugen.augment import build_aug_pool
from pathugen.champion_config import load_champion_config
from pathugen.config import DATA_DIR, MODEL_DIR, PANELS, SEED_ENSEMBLE
from pathugen.data_io import load_panel
from pathugen.features import build_features, get_cat_maps
from pathugen.model import fit_seed_ensemble
from pathugen.versioning import _panel_versions, next_model_version


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
        X, y, _ = build_features(df, cat_maps=cat_maps)
        y = pd.Series(np.asarray(y))
        train_prior = float(y.mean())

        missing = [f for f in cfg["features"] if f not in X.columns]
        if missing:
            raise RuntimeError(
                f"{panel}: champion feature list references columns not "
                f"produced by build_features(): {missing}")
        Xs = X[cfg["features"]]

        aug_X, aug_y = build_aug_pool(panel, cfg["aug_frac"], columns=Xs.columns)
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

        version = next_model_version(panel, model_dir)
        panel_dir = model_dir / panel / f"v{version}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(models):
            joblib.dump(m, panel_dir / f"seed{i}.joblib")
        joblib.dump(cat_maps, panel_dir / "cat_maps.joblib")
        (panel_dir / "selected_features.json").write_text(
            json.dumps(cfg["features"], indent=2))
        (panel_dir / "config.json").write_text(json.dumps({
            "panel": panel, "version": version, "n_train_rows": int(len(X)),
            "n_augmented_rows": int(len(aug_X)) if aug_X is not None else 0,
            "n_features": len(cfg["features"]), "seed_ensemble": SEED_ENSEMBLE,
            "threshold": cfg["threshold"], "threshold_policy": cfg["threshold_policy"],
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
