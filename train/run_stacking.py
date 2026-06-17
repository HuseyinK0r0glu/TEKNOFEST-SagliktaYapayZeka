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
from sklearn.metrics import brier_score_loss, roc_auc_score

from src import config, cv, data_io, features, stacking


def _trivial_f1(y) -> float:
    """F1 of the trivial 'predict all Pathogenic' baseline = 2p/(1+p)."""
    p = float(np.mean(y))
    return 2 * p / (1 + p) if (1 + p) else 0.0


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

    # Calibrate each base model's OOF probabilities (monotonic: AUC and best-F1
    # are unchanged, but the emitted probability becomes meaningful and the
    # threshold is robust to a class-balance shift). The stack keeps the RAW base
    # OOF (it runs its own per-fold isotonic internally); only the single-model
    # and mean candidates that finalize_champions reads use the calibrated column.
    single_cal = {
        k: stacking.fit_single_calibrator(base_oof[k], y_arr, n_rows=len(y_arr))
        for k in stacking.KINDS
    }
    cal_oof = {k: single_cal[k].predict(base_oof[k].to_numpy()) for k in stacking.KINDS}

    best_kind = max(singles, key=lambda k: singles[k]["f1"])
    best_single_f1 = singles[best_kind]["f1"]
    mean_t, mean_f1 = cv.tune_threshold(y_arr, base_oof.mean(axis=1).to_numpy())

    # Two stack variants: trees-only (validated core) and all bases (with the
    # diverse models). Champion selection keeps whichever wins on honest OOF.
    variants = {}
    for vname, vkinds in stacking.STACK_VARIANTS.items():
        vcols = [c for c in vkinds if c in base_oof.columns]
        m_oof, m_t, m_f1 = stacking.evaluate_stack(base_oof, y, cols=vcols)
        variants[vname] = {"oof": m_oof, "threshold": m_t, "f1": m_f1, "cols": vcols}

    trivial = _trivial_f1(y_arr)
    print(f"\n=== Panel {panel} (n={len(y)}, features={X.shape[1]}, "
          f"trivial-F1={trivial:.4f}) ===")
    for kind in stacking.KINDS:
        print(f"  base {kind:10s} F1={singles[kind]['f1']:.4f} (AUC {singles[kind]['auc']:.4f})")
    print(f"  best single    : {best_kind} F1={best_single_f1:.4f} "
          f"(lift {best_single_f1 - trivial:+.4f})")
    print(f"  mean ensemble  : F1={mean_f1:.4f} @ t={mean_t:.3f}")
    for vname, v in variants.items():
        print(f"  {vname:14s}: F1={v['f1']:.4f} @ t={v['threshold']:.3f} "
              f"(Δ vs best single {v['f1'] - best_single_f1:+.4f})")

    # Persist robust (THRESHOLD_AGG-aware) stack thresholds so finalize_champions
    # reuses them. Single/mean thresholds are intentionally NOT saved: their OOF
    # columns are written calibrated below, so finalize tunes their threshold in
    # the calibrated space that predict.py also operates in (kept consistent).
    thresholds = {}
    for vname, v in variants.items():
        thresholds[vname] = float(v["threshold"])

    # --- persist prediction bundle ---
    model_dir = config.OUTPUT_DIR / "models" / panel
    model_dir.mkdir(parents=True, exist_ok=True)
    for kind, model in full_models.items():
        joblib.dump(model, model_dir / f"base_{kind}.joblib")
    for kind, cal in single_cal.items():
        joblib.dump(cal, model_dir / f"calibrator_single_{kind}.joblib")

    stack_dir = config.OUTPUT_DIR / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    # Write CALIBRATED base columns: finalize_champions reads these as the single
    # and mean candidates, matching predict.py (which applies the saved single
    # calibrators). The stack (meta_*) columns below stay from the raw base OOF.
    oof_df = base_oof.copy()
    for kind in stacking.KINDS:
        oof_df[kind] = cal_oof[kind]
    best_variant = max(variants, key=lambda v: variants[v]["f1"])
    for vname, v in variants.items():
        calibrators, meta, order = stacking.fit_final_stack(base_oof, y, cols=v["cols"])
        joblib.dump(calibrators, model_dir / f"calibrators_{vname}.joblib")
        joblib.dump(meta, model_dir / f"meta_{vname}.joblib")
        (model_dir / f"{vname}_order.json").write_text(
            json.dumps(order, indent=2), encoding="utf-8")
        oof_df[f"meta_{vname}"] = v["oof"]
    (model_dir / "selected_features.json").write_text(
        json.dumps(list(X.columns), indent=2), encoding="utf-8")

    oof_df["y"] = y_arr
    oof_df.to_csv(stack_dir / f"{panel}_oof.csv", index=False)
    (stack_dir / f"{panel}_thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8")

    best_v = variants[best_variant]
    brier = float(brier_score_loss(y_arr, best_v["oof"]))
    return {
        "panel": panel, "best_single_model": best_kind, "f1_best_single": best_single_f1,
        "f1_mean_ensemble": mean_f1,
        "f1_stacked_all": variants["stacked_all"]["f1"],
        "f1_stacked_trees": variants["stacked_trees"]["f1"],
        "best_stack_variant": best_variant, "f1_stacked": best_v["f1"],
        "stacked_threshold": best_v["threshold"],
        "auc_stacked": float(roc_auc_score(y_arr, best_v["oof"])),
        "trivial_f1": trivial, "lift_stacked": best_v["f1"] - trivial,
        "brier": brier,
    }


def main(panels):
    rows = [run_panel(p) for p in panels]
    summary = pd.DataFrame(rows)
    stack_dir = config.OUTPUT_DIR / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    out_path = stack_dir / "stacking_summary.csv"
    summary.to_csv(out_path, index=False)

    print("\n=== Final per-panel F1 (best stack variant) ===")
    for _, r in summary.iterrows():
        print(f"  {r['panel']:7s} {r['best_stack_variant']:13s} F1={r['f1_stacked']:.4f} "
              f"@ t={r['stacked_threshold']:.3f}  (best single {r['f1_best_single']:.4f})")
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    requested = [p.upper() for p in sys.argv[1:]] or config.PANELS
    main(requested)
