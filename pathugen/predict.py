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
    for `panel`. Returns (models, cat_maps, features, config, calibrator,
    bin_al_cols, panel_dir); `calibrator` is None for snapshots trained
    before calibration was added, and `bin_al_cols` is None for snapshots
    trained before this fix — both cases stay fully loadable, falling back
    to their previous (approximate, for the calibrator) or per-batch (for
    bin_al_cols) behaviour."""
    panel_dir, resolved_version = resolve_model_version_dir(panel, model_dir, version)
    models = [joblib.load(panel_dir / f"seed{i}.joblib") for i in range(SEED_ENSEMBLE)]
    cat_maps = joblib.load(panel_dir / "cat_maps.joblib")
    features = json.loads((panel_dir / "selected_features.json").read_text())
    config = json.loads((panel_dir / "config.json").read_text())
    calibrator_path = panel_dir / "calibrator.joblib"
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None
    bin_cols_path = panel_dir / "bin_al_columns.json"
    bin_al_cols = json.loads(bin_cols_path.read_text()) if bin_cols_path.exists() else None
    return models, cat_maps, features, config, calibrator, bin_al_cols, panel_dir


def prior_shift_threshold(t_ref: float, prior_train: float, prior_target: float) -> float:
    """Move a threshold fit under `prior_train` to what it should be under
    `prior_target`, using calibrated-probability Bayes rescaling. Used only
    when --prior-shift is passed; the panel's model is NOT retrained."""
    r = prior_target / prior_train
    s = (1.0 - prior_target) / (1.0 - prior_train)
    return float(t_ref * s / (t_ref * s + (1.0 - t_ref) * r))


def _invert_calibration(calibrator, level: float, tol: float = 1e-9) -> float:
    """Smallest raw probability whose calibrated value reaches `level`.

    Isotonic regression is non-decreasing but flat over wide stretches, so it
    has no closed-form inverse — a bisection is both simplest and exact enough
    here. Used to bring a shifted threshold back to the raw scale, so the
    decision itself never leaves the scale the audited threshold lives on.
    """
    lo, hi = 0.0, 1.0
    if float(calibrator.predict(np.array([lo]))[0]) >= level:
        return lo
    if float(calibrator.predict(np.array([hi]))[0]) < level:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if float(calibrator.predict(np.array([mid]))[0]) >= level:
            hi = mid
        else:
            lo = mid
    return hi


def predict_panel(panel: str, csv_path: Path, model_dir: Path = MODEL_DIR,
                   prior_shift: float | None = None,
                   model_version: int | None = None) -> tuple[list[dict], Path]:
    """Return (prediction records, model snapshot dir actually used)."""
    models, cat_maps, features, config, calibrator, bin_al_cols, panel_dir = load_bundle(
        panel, model_dir, version=model_version)
    df = load_test_csv(csv_path)
    X, _, _ = build_features(df, cat_maps=cat_maps, bin_cols=bin_al_cols)

    missing = [f for f in features if f not in X.columns]
    if missing:
        raise RuntimeError(f"{panel}: test data is missing engineered columns "
                           f"{missing} — check the input CSV matches the "
                           f"expected 353-column raw schema (351 features).")
    Xs = X[features]

    proba = predict_proba_ensemble(models, Xs)
    proba = np.clip(proba, 0.0, 1.0)   # defensive: guide forbids out-of-range probabilities

    # The decision always happens on the raw probability, against the audited
    # threshold. The calibrator is never in this path — it only computes how
    # far to MOVE that threshold when --prior-shift asks for it.
    threshold = config["threshold"]
    if prior_shift is not None and calibrator is not None:
        # The Bayes prior-shift formula is only valid on calibrated
        # probabilities, so: lift the audited threshold onto the calibrated
        # scale, shift it there, then bring it back. When the target prior
        # equals the training prior the shift is exactly the identity, and
        # the audited threshold is returned untouched.
        t_cal = float(calibrator.predict(np.array([threshold]))[0])
        t_cal_shifted = prior_shift_threshold(
            t_cal, config["train_prior"], prior_shift)
        if t_cal_shifted != t_cal:
            threshold = _invert_calibration(calibrator, t_cal_shifted)
    elif prior_shift is not None:
        print(f"  [warning] {panel}: this snapshot has no calibrator, so "
              f"--prior-shift is applied to uncalibrated probabilities and is "
              f"only approximate. Re-run `train` for a calibrated snapshot.")
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
