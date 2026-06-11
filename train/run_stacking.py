"""Per-panel calibration + stacking: honest F1 and saved prediction bundle.

For each panel: build features, restrict to selected columns, generate leak-free
base OOF from the tuned models, evaluate the calibrated stacking ensemble, and
compare it to the best single model and the mean ensemble. Persists the full
per-panel bundle (base models + calibrators + meta + threshold + feature list)
under outputs/models/{panel}/ for test-time prediction.

    PYTHONPATH=. .venv/bin/python train/run_stacking.py           # all panels
    PYTHONPATH=. .venv/bin/python train/run_stacking.py CFTR      # one panel
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src import config, cv, data_io, features, stacking


def _load_selected(panel: str):
    path = config.OUTPUT_DIR / "feature_selection" / f"{panel}_selected.txt"
    if not path.exists():
        return None
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_panel(panel: str) -> dict:
    df = data_io.load_panel(panel)
    X, y, cat_cols = features.build_features(df)
    selected = _load_selected(panel)
    if selected:
        X = X[selected]
        cat_cols = [c for c in cat_cols if c in set(selected)]
    y_arr = np.asarray(y)

    base_oof, singles, full_models = stacking.generate_base_oof(panel, X, y, cat_cols)

    best_kind = max(singles, key=lambda k: singles[k]["f1"])
    best_single_f1 = singles[best_kind]["f1"]
    mean_t, mean_f1 = cv.tune_threshold(y_arr, base_oof.mean(axis=1).to_numpy())
    meta_oof, meta_t, meta_f1 = stacking.evaluate_stack(base_oof, y)
    brier = float(brier_score_loss(y_arr, meta_oof))

    print(f"\n=== Panel {panel} (n={len(y)}, features={X.shape[1]}) ===")
    for kind in stacking.KINDS:
        print(f"  base {kind:9s} F1={singles[kind]['f1']:.4f} (AUC {singles[kind]['auc']:.4f})")
    print(f"  best single   : {best_kind} F1={best_single_f1:.4f}")
    print(f"  mean ensemble : F1={mean_f1:.4f} @ t={mean_t:.3f}")
    print(f"  STACKED       : F1={meta_f1:.4f} @ t={meta_t:.3f} "
          f"(Δ vs best single {meta_f1 - best_single_f1:+.4f}) | Brier={brier:.4f}")

    # --- persist prediction bundle ---
    calibrators, meta = stacking.fit_final_stack(base_oof, y)
    model_dir = config.OUTPUT_DIR / "models" / panel
    model_dir.mkdir(parents=True, exist_ok=True)
    for kind, model in full_models.items():
        joblib.dump(model, model_dir / f"base_{kind}.joblib")
    joblib.dump(calibrators, model_dir / "calibrators.joblib")
    joblib.dump(meta, model_dir / "meta_logreg.joblib")
    (model_dir / "selected_features.json").write_text(
        json.dumps(list(X.columns), indent=2), encoding="utf-8")
    (model_dir / "threshold.json").write_text(
        json.dumps({"threshold": meta_t, "base_order": list(base_oof.columns)}, indent=2),
        encoding="utf-8")

    stack_dir = config.OUTPUT_DIR / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    oof_df = base_oof.copy()
    oof_df["meta"] = meta_oof
    oof_df["y"] = y_arr
    oof_df.to_csv(stack_dir / f"{panel}_oof.csv", index=False)

    return {
        "panel": panel, "best_single_model": best_kind, "f1_best_single": best_single_f1,
        "f1_mean_ensemble": mean_f1, "f1_stacked": meta_f1, "stacked_threshold": meta_t,
        "brier": brier,
    }


def main(panels):
    rows = [run_panel(p) for p in panels]
    summary = pd.DataFrame(rows)
    stack_dir = config.OUTPUT_DIR / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    out_path = stack_dir / "stacking_summary.csv"
    summary.to_csv(out_path, index=False)

    print("\n=== Final per-panel F1 (stacked) ===")
    for _, r in summary.iterrows():
        print(f"  {r['panel']:7s} stacked F1={r['f1_stacked']:.4f} "
              f"@ t={r['stacked_threshold']:.3f}  (best single {r['f1_best_single']:.4f})")
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    requested = [p.upper() for p in sys.argv[1:]] or config.PANELS
    main(requested)
