"""HISTORICAL / SUPERSEDED — do not run.

This originally generated a flat `run_final.py` from the Colab notebook's
code cells. That flat file has since been hand-split into the
`search.reference` package (see `search/reference/*.py`), which is now the
maintained copy — running this script would recreate a stale flat file that
no longer matches the current package layout. Kept only so the notebook
cell numbers this once extracted stay documented.
"""
from __future__ import annotations
import json
from pathlib import Path

NB = Path("TEKNOFEST_pipeline_colab.ipynb")
OUT = Path("run_final.py")

# Pipeline code cells, in notebook order. Skipped: 2 (%pip), 4 (Colab/Drive env),
# 33/35/37/39 (notebook drivers — replaced by the CLI main below).
CELLS = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

HEADER = '''"""TEKNOFEST 2026 — final pipeline, local runner.

AUTO-GENERATED from `TEKNOFEST_pipeline_colab.ipynb` (code cells
7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 — verbatim, in order).
This is the exact pipeline that produced `outputs_colab/results_summary.csv`
and the numbers in the PDR; only the Colab/Drive plumbing was replaced by a CLI.

Run (full report-grade budget, all four panels):

    QUICK=0 USE_GPU=0 RUN_NESTED=1 AUGMENT_PANELS=PAH,CFTR \\
    SELECTION_METHOD=null OPTUNA_TRIALS=80 OPTUNA_TIMEOUT=600 \\
    TEKNOFEST_OUT_DIR=outputs_final \\
    .venv/bin/python run_final.py 2>&1 | tee run_final.log

Every run also writes to logs/run_final_<timestamp>.txt (logs/run_final_latest.txt).

Single panel:  .venv/bin/python run_final.py CFTR
Regenerate:    python scripts/gen_runner.py   (see repo notes)
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))  # runlog import edilebilsin

'''

FOOTER = '''

# ---------------------------------------------------------------------------
# CLI entry point (replaces notebook Steps 4-7)
# ---------------------------------------------------------------------------
def _print_summary(res: "pd.DataFrame") -> None:
    cols = [c for c in ["panel", "n", "champion", "strategy", "augmented",
                        "n_features", "threshold", "oof_f1", "oof_auc", "honest_f1",
                        "honest_auc", "honest_balanced_f1", "optimism_gap",
                        "trivial_f1"] if c in res.columns]
    print("\\n=== Per-panel results ===")
    print(res[cols].to_string(index=False))
    if "honest_f1" in res.columns:
        print(f"\\nMean honest (nested) F1 across panels: {res['honest_f1'].mean():.4f}")
    print(f"Mean OOF F1 across panels:             {res['oof_f1'].mean():.4f}")


def _inference_check(panels) -> None:
    """Notebook Step 6 — verify each saved bundle reloads and scores."""
    print("\\n=== Inference wiring check (in-sample, saved bundles) ===")
    for p in panels:
        df = load_panel(p)
        out = predict_panel(p, df)
        f1 = f1_score(df[LABEL_COL], out["prediction"], zero_division=0)
        print(f"  {p:7s} in-sample F1={f1:.4f}  "
              f"predicted pos rate={out['prediction'].mean():.3f}")


def main() -> None:
    import runlog
    log_path = runlog.start("run_final",
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
    print(f"\\nWrote {OUTPUT_DIR / 'results_summary.csv'} and "
          f"per-panel bundles under {OUTPUT_DIR / 'models'}/")
    print(f"Log      : {log_path}")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    nb = json.loads(NB.read_text())
    parts = [HEADER]
    for i in CELLS:
        src = "".join(nb["cells"][i]["source"]).rstrip()
        parts.append(f"# {'=' * 74}\n# notebook cell [{i}]\n# {'=' * 74}\n{src}\n")
    parts.append(FOOTER)
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()
