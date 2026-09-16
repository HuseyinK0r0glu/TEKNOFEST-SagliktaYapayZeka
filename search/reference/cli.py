"""CLI entry point for the reference pipeline."""
from __future__ import annotations

import sys

from sklearn.metrics import f1_score

from pathugen.config import LABEL_COL
from pathugen.data_io import load_panel
from search.reference.config import OUTPUT_DIR, PANEL_FILES, PANELS
from search.reference.pipeline import run_all
from search.reference.predict import predict_panel


def _print_summary(res: "pd.DataFrame") -> None:
    cols = [c for c in ["panel", "n", "champion", "strategy", "augmented",
                        "n_features", "threshold", "oof_f1", "oof_auc", "honest_f1",
                        "honest_auc", "honest_balanced_f1", "optimism_gap",
                        "trivial_f1"] if c in res.columns]
    print("\n=== Per-panel results ===")
    print(res[cols].to_string(index=False))
    if "honest_f1" in res.columns:
        print(f"\nMean honest (nested) F1 across panels: {res['honest_f1'].mean():.4f}")
    print(f"Mean OOF F1 across panels:             {res['oof_f1'].mean():.4f}")


def _inference_check(panels) -> None:
    """Verify each saved bundle reloads and scores."""
    print("\n=== Inference wiring check (in-sample, saved bundles) ===")
    for p in panels:
        df = load_panel(p)
        out = predict_panel(p, df)
        f1 = f1_score(df[LABEL_COL], out["prediction"], zero_division=0)
        print(f"  {p:7s} in-sample F1={f1:.4f}  "
              f"predicted pos rate={out['prediction'].mean():.3f}")


def main() -> None:
    from search import runlog

    log_path = runlog.start("reference",
                            note="Reference (baseline) pipeline — the exact run behind the submitted PDR")

    panels = [a.upper() for a in sys.argv[1:]] or PANELS
    unknown = [p for p in panels if p not in PANEL_FILES]
    if unknown:
        raise SystemExit(f"Unknown panel(s): {unknown}. Choose from {PANELS}.")
    print(f"Panels   : {panels}")
    print(f"Output   : {OUTPUT_DIR.resolve()}")
    summary = run_all(panels)
    _print_summary(summary)
    _inference_check(panels)
    print(f"\nWrote {OUTPUT_DIR / 'results_summary.csv'} and "
          f"per-panel bundles under {OUTPUT_DIR / 'models'}/")
    print(f"Log      : {log_path}")


if __name__ == "__main__":
    main()
