"""Per-panel Optuna tuning on the selected feature sets.

For each panel and each model, tunes hyperparameters to maximise F1, saves the
best params as JSON, and compares the tuned F1 to the selected-feature baseline.

Config via env vars: OPTUNA_TRIALS (default 40), OPTUNA_TIMEOUT seconds per
study (default 150). Panels via argv (default all).

    PYTHONPATH=. OPTUNA_TRIALS=40 .venv/bin/python train/run_optuna.py
    PYTHONPATH=. OPTUNA_TRIALS=15 OPTUNA_TIMEOUT=60 .venv/bin/python train/run_optuna.py CFTR
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

from src import config, data_io, features, tuning

N_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "40"))
TIMEOUT = int(os.environ.get("OPTUNA_TIMEOUT", "150"))


def _load_selected(panel: str):
    path = config.OUTPUT_DIR / "feature_selection" / f"{panel}_selected.txt"
    if not path.exists():
        return None
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _selected_f1(panel: str) -> dict:
    path = config.OUTPUT_DIR / "feature_selection" / "selection_summary.csv"
    if not path.exists():
        return {}
    s = pd.read_csv(path)
    s = s[s["panel"] == panel]
    return dict(zip(s["model"], s["f1_selected"]))


def run_panel(panel: str, out_dir) -> list[dict]:
    df = data_io.load_panel(panel)
    X, y, cat_cols = features.build_features(df)
    selected = _load_selected(panel)
    if selected:
        X = X[selected]
        cat_cols = [c for c in cat_cols if c in set(selected)]
    prev = _selected_f1(panel)

    print(f"\n=== Panel {panel} (n={len(y)}, features={X.shape[1]}, "
          f"trials<={N_TRIALS}, timeout={TIMEOUT}s) ===", flush=True)
    rows = []
    for kind in ("xgboost", "lightgbm", "catboost"):
        best_params, best_f1, n_done = tuning.tune_model(
            kind, X, y, cat_cols, n_trials=N_TRIALS, timeout=TIMEOUT
        )
        (out_dir / f"{panel}_{kind}_best.json").write_text(
            json.dumps(best_params, indent=2), encoding="utf-8"
        )
        before = prev.get(kind)
        delta = f"{best_f1 - before:+.4f}" if before is not None else "  n/a "
        before_s = f"{before:.4f}" if before is not None else " n/a  "
        print(f"  {kind:9s} selected={before_s} -> tuned F1={best_f1:.4f} "
              f"(Δ{delta})  [{n_done} trials]", flush=True)
        rows.append({
            "panel": panel, "model": kind, "f1_selected": before,
            "f1_tuned": best_f1, "n_trials": n_done,
        })
    return rows


def main(panels):
    out_dir = config.OUTPUT_DIR / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for panel in panels:
        all_rows.extend(run_panel(panel, out_dir))
    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "optuna_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[written] {summary_path}", flush=True)


if __name__ == "__main__":
    requested = [p.upper() for p in sys.argv[1:]] or config.PANELS
    main(requested)
