"""Per-panel null-importance feature selection + validation.

For each panel: build features, run the null-importance test, save the ranked
table and the kept-feature list, then validate by re-running F1 CV on the kept
subset and comparing to the saved baseline (all-feature) F1.

Run from the project root:
    PYTHONPATH=. .venv/bin/python train/run_feature_selection.py          # all
    PYTHONPATH=. .venv/bin/python train/run_feature_selection.py CFTR     # one
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from src import config, data_io, feature_selection, features, models

# Which feature set to SHIP as {panel}_selected.txt:
#   "null"          -> the null-importance keep-set (current default behaviour).
#   "union"         -> null-importance + ReliefF-guarded consensus adds.
#   "relief_pruned" -> null-importance minus locally-misleading features.
# Adopt a non-default method only after run_holdout_compare shows it wins on the
# held-out test set. Selection runs on the FULL panel here (that is how the
# shipped set is built); the honest estimate still comes from nested/holdout eval.
SELECTION_METHOD = os.environ.get("SELECTION_METHOD", "null").strip().lower()


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

    null_selected = table.loc[table["keep"], "feature"].tolist()

    # Default ships the null-importance set unchanged. SELECTION_METHOD lets a
    # holdout-validated variant (union / relief_pruned) be shipped instead.
    if SELECTION_METHOD == "null":
        selected = null_selected
    else:
        from src import multi_filter_selection
        variants, _ = multi_filter_selection.build_variants(
            X, y, cat_cols, null_selected=null_selected, verbose=True)
        if SELECTION_METHOD not in variants:
            raise SystemExit(
                f"SELECTION_METHOD={SELECTION_METHOD!r} unknown; "
                f"expected one of {list(variants)}")
        selected = variants[SELECTION_METHOD]
        print(f"  [SELECTION_METHOD={SELECTION_METHOD}] shipping {len(selected)} "
              f"features (null had {len(null_selected)})")

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
