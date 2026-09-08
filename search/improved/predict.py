"""Inference using a saved improved-pipeline bundle (seed ensemble +
prior-aware threshold). Competition-day inference goes through
`pathugen.predict`, not this module."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from pathugen.config import ID_COL
from pathugen.data_io import load_panel
from pathugen.features import build_features, get_cat_maps
from search.improved.threshold import prior_shift_threshold
from search.reference.config import OUTPUT_DIR


def predict_panel_plus(panel, df, target_prior=None):
    """Labels rows with the saved improved bundle.

    If target_prior is given, the threshold is moved to that class ratio via
    the isotonic-calibrated probability (the model is not retrained).
    """
    bundle = OUTPUT_DIR / "models" / panel
    champ = json.loads((bundle / "champion.json").read_text())
    selected = json.loads((bundle / "selected_features.json").read_text())
    cat_maps = get_cat_maps(load_panel(panel))
    X, _, _ = build_features(df, cat_maps=cat_maps)
    X = X[selected]
    view = champ["model_view"]

    def base_raw(col):
        return joblib.load(bundle / f"base_{col}.joblib").predict_proba(X[view[col]])[:, 1]

    strategy = champ["strategy"]
    if strategy == "single":
        n_seed = champ.get("seed_ensemble", 0)
        paths = [bundle / f"champion_seed{i}.joblib" for i in range(n_seed)]
        if n_seed > 1 and all(p.exists() for p in paths):
            feats = view[champ["model"]]
            proba = np.mean([joblib.load(p).predict_proba(X[feats])[:, 1] for p in paths],
                            axis=0)
        else:
            proba = base_raw(champ["model"])
    elif strategy == "mean":
        proba = np.mean([base_raw(c) for c in champ["base_order"]], axis=0)
    else:
        calibrators = joblib.load(bundle / f"calibrators_{strategy}.joblib")
        meta = joblib.load(bundle / f"meta_{strategy}.joblib")
        order = json.loads((bundle / f"{strategy}_order.json").read_text())
        z = np.column_stack([calibrators[c].predict(base_raw(c)) for c in order])
        proba = meta.predict_proba(z)[:, 1]

    threshold = champ["threshold"]
    if target_prior is not None:
        iso = joblib.load(bundle / "champion_calibrator.joblib")
        proba = iso.predict(proba)
        threshold = prior_shift_threshold(champ["calibrated_threshold"],
                                          champ["train_prior"], float(target_prior))
    return pd.DataFrame({ID_COL: df[ID_COL].to_numpy(), "proba": proba,
                         "prediction": (proba >= threshold).astype(int)})
