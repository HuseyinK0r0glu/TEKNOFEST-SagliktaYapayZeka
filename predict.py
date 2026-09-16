"""End-to-end per-panel inference for the TEKNOFEST variant classifier.

Loads a panel's champion bundle (saved by run_stacking + finalize_champions) and
labels variants in an input CSV that follows the 353-column panel schema.

    PYTHONPATH=. .venv/bin/python predict.py <PANEL> <input_csv> [output_csv]
    PYTHONPATH=. .venv/bin/python predict.py CFTR              # wiring check on train CSV

Output columns: Variant_ID, proba (Pathogenic probability), prediction (0/1).
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd

from src import config, data_io, features


def _bundle_dir(panel: str):
    return config.OUTPUT_DIR / "models" / panel


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _base_proba(bundle_dir, kind, X):
    """Raw base-model Pathogenic probability (used by the stack path, which then
    applies the stack's own calibrators)."""
    model = joblib.load(bundle_dir / f"base_{kind}.joblib")
    return model.predict_proba(X)[:, 1]


def _cal_base_proba(bundle_dir, kind, X):
    """Calibrated base-model probability for the single / mean paths: raw model
    output passed through the saved per-model single calibrator when present
    (backward compatible — older bundles without one fall back to raw)."""
    p = _base_proba(bundle_dir, kind, X)
    cal_path = bundle_dir / f"calibrator_single_{kind}.joblib"
    if cal_path.exists():
        p = joblib.load(cal_path).predict(p)
    return p


def predict_panel(panel: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame [Variant_ID, proba, prediction] for the given rows."""
    bundle_dir = _bundle_dir(panel)
    champion = _load_json(bundle_dir / "champion.json")
    selected = _load_json(bundle_dir / "selected_features.json")

    # Encode categoricals with the TRAIN-time vocabulary so codes are consistent.
    cat_maps = features.get_cat_maps(data_io.load_panel(panel))
    X, _, _ = features.build_features(df, cat_maps=cat_maps)
    X = X[selected]

    strategy = champion["strategy"]
    order = champion["base_order"]
    if strategy == "single":
        proba = _cal_base_proba(bundle_dir, champion["model"], X)
    elif strategy == "mean":
        proba = np.mean([_cal_base_proba(bundle_dir, k, X) for k in order], axis=0)
    else:  # stacked_all / stacked_trees: calibrate each base, then logistic meta
        calibrators = joblib.load(bundle_dir / f"calibrators_{strategy}.joblib")
        meta = joblib.load(bundle_dir / f"meta_{strategy}.joblib")
        z = np.column_stack([
            calibrators[k].predict(_base_proba(bundle_dir, k, X)) for k in order
        ])
        proba = meta.predict_proba(z)[:, 1]

    prediction = (proba >= champion["threshold"]).astype(int)
    return pd.DataFrame({
        config.ID_COL: df[config.ID_COL].to_numpy(),
        "proba": proba,
        "prediction": prediction,
    })


def main(argv):
    if not argv:
        print(__doc__)
        return
    panel = argv[0].upper()
    input_csv = argv[1] if len(argv) > 1 else None
    output_csv = argv[2] if len(argv) > 2 else None

    if input_csv:
        df = pd.read_csv(input_csv)
    else:
        df = data_io.load_panel(panel)  # wiring check on the training CSV
        print(f"[no input csv given] running wiring check on training {panel}")

    out = predict_panel(panel, df)
    if output_csv:
        out.to_csv(output_csv, index=False)
        print(f"[written] {output_csv}")

    print(f"panel={panel}  n={len(out)}  predicted positive rate={out['prediction'].mean():.3f}")
    if config.LABEL_COL in df.columns:
        from sklearn.metrics import f1_score
        f1 = f1_score(df[config.LABEL_COL].to_numpy(), out["prediction"].to_numpy())
        print(f"[labeled input] F1={f1:.4f} "
              f"(in-sample if run on training data — wiring check only, not an honest score)")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
