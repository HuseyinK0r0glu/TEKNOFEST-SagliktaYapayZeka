"""Per-panel baseline tri-model leaderboard (XGBoost / LightGBM / CatBoost).

Builds features per panel, runs F1-tuned 5-fold CV for each model (factories
live in src.models), prints a per-panel leaderboard, and writes
outputs/baseline/baseline_leaderboard.csv.

Run from the project root:
    PYTHONPATH=. .venv/bin/python train/run_baseline.py            # all panels
    PYTHONPATH=. .venv/bin/python train/run_baseline.py CFTR       # one panel
"""
from __future__ import annotations

import sys

import pandas as pd

from src import config, data_io, features, models


def run_panel(panel: str) -> list[dict]:
    df = data_io.load_panel(panel)
    X, y, cat_cols = features.build_features(df)
    print(f"\n=== Panel {panel}  (n={len(y)}, features={X.shape[1]}, "
          f"cat={len(cat_cols)}) ===")
    results = models.evaluate_models(X, y, cat_cols)
    rows = []
    for name, res in results.items():
        print(f"  {name:9s} F1={res['f1']:.4f} @ t={res['threshold']:.3f} | "
              f"AUC={res['auc']:.4f} (fold {res['auc_fold_mean']:.4f}"
              f"±{res['auc_fold_std']:.4f}) | MCC={res['mcc']:.4f} | "
              f"P={res['precision']:.3f} R={res['recall']:.3f}")
        rows.append({
            "panel": panel, "model": name, "n": len(y), "n_features": X.shape[1],
            "f1": res["f1"], "threshold": res["threshold"], "auc": res["auc"],
            "auc_fold_mean": res["auc_fold_mean"], "auc_fold_std": res["auc_fold_std"],
            "mcc": res["mcc"], "precision": res["precision"], "recall": res["recall"],
        })
    return rows


def main(panels):
    out_dir = config.OUTPUT_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for panel in panels:
        all_rows.extend(run_panel(panel))

    board = pd.DataFrame(all_rows)
    out_path = out_dir / "baseline_leaderboard.csv"
    board.to_csv(out_path, index=False)

    print("\n=== Best model per panel (by F1) ===")
    best = board.loc[board.groupby("panel")["f1"].idxmax()]
    for _, r in best.iterrows():
        print(f"  {r['panel']:7s} -> {r['model']:9s} F1={r['f1']:.4f} "
              f"@ t={r['threshold']:.3f}")
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    requested = [p.upper() for p in sys.argv[1:]] or config.PANELS
    main(requested)
