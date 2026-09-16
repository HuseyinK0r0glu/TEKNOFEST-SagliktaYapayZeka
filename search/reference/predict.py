"""Inference using a saved reference-pipeline champion bundle (in-sample
wiring checks only — competition-day inference goes through
`pathugen.predict`, not this module)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from pathugen.config import ID_COL
from pathugen.data_io import load_panel
from pathugen.features import build_features, get_cat_maps
from search.reference.config import OUTPUT_DIR


def predict_panel(panel, df):
    """Label rows in `df` with the saved champion bundle for `panel`."""
    bundle = OUTPUT_DIR / "models" / panel
    champion = json.loads((bundle / "champion.json").read_text())
    selected = json.loads((bundle / "selected_features.json").read_text())
    # Encode categoricals with the TRAIN-time vocabulary (rebuilt from the panel
    # CSV) so codes match those the bundle was trained on.
    cat_maps = get_cat_maps(load_panel(panel))
    X, _, _ = build_features(df, cat_maps=cat_maps)
    X = X[selected]
    strategy, order = champion["strategy"], champion["base_order"]

    def base_raw(kind):
        return joblib.load(bundle / f"base_{kind}.joblib").predict_proba(X)[:, 1]

    if strategy == "single":
        proba = base_raw(champion["model"])
    elif strategy == "mean":
        proba = np.mean([base_raw(k) for k in order], axis=0)
    else:
        calibrators = joblib.load(bundle / f"calibrators_{strategy}.joblib")
        meta = joblib.load(bundle / f"meta_{strategy}.joblib")
        z = np.column_stack([calibrators[k].predict(base_raw(k)) for k in order])
        proba = meta.predict_proba(z)[:, 1]
    pred = (proba >= champion["threshold"]).astype(int)
    return pd.DataFrame({ID_COL: df[ID_COL].to_numpy(), "proba": proba, "prediction": pred})
