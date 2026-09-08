"""PREDICT — load a trained panel bundle and score a test CSV."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pathugen.config import ID_COL, MODEL_DIR, SEED_ENSEMBLE
from pathugen.data_io import load_test_csv
from pathugen.features import build_features
from pathugen.model import predict_proba_ensemble
from pathugen.versioning import resolve_model_version_dir


def load_bundle(panel: str, model_dir: Path = MODEL_DIR, version: int | None = None):
    """Load the requested (or, by default, most recently trained) snapshot
    for `panel`. Returns (models, cat_maps, features, config, panel_dir)."""
    panel_dir, resolved_version = resolve_model_version_dir(panel, model_dir, version)
    models = [joblib.load(panel_dir / f"seed{i}.joblib") for i in range(SEED_ENSEMBLE)]
    cat_maps = joblib.load(panel_dir / "cat_maps.joblib")
    features = json.loads((panel_dir / "selected_features.json").read_text())
    config = json.loads((panel_dir / "config.json").read_text())
    return models, cat_maps, features, config, panel_dir


def prior_shift_threshold(t_ref: float, prior_train: float, prior_target: float) -> float:
    """Move a threshold fit under `prior_train` to what it should be under
    `prior_target`, using calibrated-probability Bayes rescaling. Used only
    when --prior-shift is passed; the panel's model is NOT retrained."""
    r = prior_target / prior_train
    s = (1.0 - prior_target) / (1.0 - prior_train)
    return float(t_ref * s / (t_ref * s + (1.0 - t_ref) * r))


def predict_panel(panel: str, csv_path: Path, model_dir: Path = MODEL_DIR,
                   prior_shift: float | None = None,
                   model_version: int | None = None) -> tuple[list[dict], Path]:
    """Return (prediction records, model snapshot dir actually used)."""
    models, cat_maps, features, config, panel_dir = load_bundle(
        panel, model_dir, version=model_version)
    df = load_test_csv(csv_path)
    X, _, _ = build_features(df, cat_maps=cat_maps)

    missing = [f for f in features if f not in X.columns]
    if missing:
        raise RuntimeError(f"{panel}: test data is missing engineered columns "
                           f"{missing} — check the input CSV matches the "
                           f"expected 353-column raw schema (351 features).")
    Xs = X[features]

    proba = predict_proba_ensemble(models, Xs)
    proba = np.clip(proba, 0.0, 1.0)   # defensive: guide forbids out-of-range probabilities

    threshold = config["threshold"]
    if prior_shift is not None:
        threshold = prior_shift_threshold(threshold, config["train_prior"], prior_shift)

    pred = (proba >= threshold).astype(int)
    ids = df[ID_COL].astype(str).tolist()
    if len(set(ids)) != len(ids):
        dupes = pd.Series(ids)[pd.Series(ids).duplicated()].unique().tolist()[:5]
        raise RuntimeError(f"{panel}: duplicate {ID_COL} values in the input "
                           f"file (e.g. {dupes}) — cannot produce one "
                           f"prediction per id as the delivery guide requires.")

    records = [
        {"id": vid, "panel": panel, "predicted_class": str(int(p)),
         "predicted_prob": round(float(prob), 6)}
        for vid, p, prob in zip(ids, pred, proba)
    ]
    return records, panel_dir
