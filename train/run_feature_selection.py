"""Per-panel null-importance feature selection + validation.

For each panel: build features, run the null-importance test, save the ranked
table and the kept-feature list, then validate by re-running F1 CV on the kept
subset and comparing to the saved baseline (all-feature) F1.

Run from the project root:
    PYTHONPATH=. .venv/bin/python train/run_feature_selection.py          # all
    PYTHONPATH=. .venv/bin/python train/run_feature_selection.py CFTR     # one
"""
from __future__ import annotations

import sys

import pandas as pd

from src import config, data_io, feature_selection, features, models


def _baseline_f1(panel: str) -> dict:
    path = config.OUTPUT_DIR / "baseline" / "baseline_leaderboard.csv"
    if not path.exists():
        return {}
    board = pd.read_csv(path)
    sub = board[board["panel"] == panel]
    return dict(zip(sub["model"], sub["f1"]))


def run_panel(panel: str) -> list[dict]:
    df = data_io.load_panel(panel)
    X, y, cat_cols = features.build_features(df)

    table = feature_selection.null_importance_selection(X, y, cat_cols)
    out_dir = config.OUTPUT_DIR / "feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{panel}_null_importance.csv", index=False)

    selected = table.loc[table["keep"], "feature"].tolist()
    (out_dir / f"{panel}_selected.txt").write_text("\n".join(selected), encoding="utf-8")
    sel_cat = [c for c in cat_cols if c in set(selected)]

    print(f"\n=== Panel {panel}: kept {len(selected)}/{X.shape[1]} features "
          f"({len(sel_cat)} categorical) ===")
    print("  top 12 by gain_score: "
          + ", ".join(table.head(12)["feature"].tolist()))

    base_f1 = _baseline_f1(panel)
    sub_results = models.evaluate_models(X[selected], y, sel_cat)
    rows = []
    for name, res in sub_results.items():
        prev = base_f1.get(name)
        delta = f"{res['f1'] - prev:+.4f}" if prev is not None else "  n/a "
        prev_s = f"{prev:.4f}" if prev is not None else " n/a  "
        print(f"  {name:9s} all={prev_s} -> selected F1={res['f1']:.4f} "
              f"(Δ{delta}) @ t={res['threshold']:.3f} | AUC={res['auc']:.4f}")
        rows.append({
            "panel": panel, "model": name, "n_selected": len(selected),
            "f1_all": prev, "f1_selected": res["f1"], "threshold": res["threshold"],
            "auc": res["auc"], "mcc": res["mcc"],
        })
    return rows


def main(panels):
    out_dir = config.OUTPUT_DIR / "feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for panel in panels:
        all_rows.extend(run_panel(panel))
    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "selection_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[written] {summary_path}")


if __name__ == "__main__":
    requested = [p.upper() for p in sys.argv[1:]] or config.PANELS
    main(requested)
