"""Main flow for the improved pipeline."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from pathugen.config import ID_COL, LABEL_COL
from pathugen.data_io import load_panel
from search import runlog
from search._project_root import PROJECT_ROOT
from search.improved.config import (
    AUG_FRAC, COMPARE_TO, DIVERSE_STACK, PAIRED_BOOTSTRAP, POLICIES,
    REPORT_PRIORS, SEED_ENSEMBLE, SHORTCUT_AUDIT, THRESHOLD_POLICY, THR_BOOT,
)
from search.improved.pipeline import run_panel, selection_method_for
from search.improved.predict import predict_panel_plus
from search.reference.config import NESTED_REPEATS, PANEL_FILES, PANELS, OUTPUT_DIR


def _print_summary(res):
    cols = [c for c in ["panel", "n", "champion", "strategy", "selection", "n_features",
                        "diverse_stack", "augmented", "threshold_policy", "threshold",
                        "oof_f1", "oof_f1_balanced", "oof_mcc", "oof_auc",
                        "honest_f1", "honest_auc", "honest_balanced_f1",
                        "optimism_gap", "trivial_f1"] if c in res.columns]
    runlog.section("PER-PANEL RESULTS (improved)")
    print(res[cols].to_string(index=False))
    if "honest_f1" in res.columns:
        print(f"\nMean honest (nested) F1 : {res['honest_f1'].mean():.4f}")
        print(f"Mean honest AUC         : {res['honest_auc'].mean():.4f}")
        print(f"Mean balanced F1 (1:1)  : {res['honest_balanced_f1'].mean():.4f}")
    print(f"Mean OOF F1             : {res['oof_f1'].mean():.4f}")

    if "f1_if_policy_f1" in res.columns:
        runlog.section("COST / GAIN OF THE THRESHOLD POLICY")
        print("Comparison of the selected policy against the old (F1-only-maximizing) "
              "policy.\nThe '1:1' column is the expected F1 if the test set turns out balanced.\n")
        t = res[["panel", "threshold_policy", "oof_f1", "oof_f1_balanced",
                 "f1_if_policy_f1", "balf1_if_policy_f1"]].copy()
        t.columns = ["panel", "policy", "F1 (new)", "1:1 F1 (new)",
                     "F1 (old)", "1:1 F1 (old)"]
        t["Delta F1"] = (t["F1 (new)"] - t["F1 (old)"]).round(4)
        t["Delta 1:1 F1"] = (t["1:1 F1 (new)"] - t["1:1 F1 (old)"]).round(4)
        print(t.to_string(index=False))

    if "f1_without_missingness" in res.columns:
        runlog.section("SHORTCUT ABLATION — without the missingness features")
        t = res[["panel", "oof_f1", "f1_without_missingness",
                 "oof_auc", "auc_without_missingness"]].copy()
        t["Delta F1"] = (t["f1_without_missingness"] - t["oof_f1"]).round(4)
        t["Delta AUC"] = (t["auc_without_missingness"] - t["oof_auc"]).round(4)
        print(t.to_string(index=False))
        print("\nA large delta means the model relies on annotation completeness; if the "
              "\nannotation profile shifts in the test set, expect a drop of about this size.")


def _compare_baseline(res):
    path = Path(COMPARE_TO)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"\n[comparison] reference summary not found ({path}). "
              f"Run the reference pipeline first, or point COMPARE_TO at a different path.")
        return
    base = pd.read_csv(path)
    runlog.section(f"REFERENCE COMPARISON  ({path.name})")
    keys = [("oof_f1", "OOF F1"), ("oof_auc", "OOF AUC"),
            ("honest_f1", "Honest F1"), ("honest_auc", "Honest AUC"),
            ("honest_balanced_f1", "Honest balanced F1")]
    rows = []
    for _, r in res.iterrows():
        b = base[base["panel"] == r["panel"]]
        if b.empty:
            continue
        b = b.iloc[0]
        row = {"panel": r["panel"]}
        for k, label in keys:
            if k in res.columns and k in base.columns:
                row[f"{label} (ref)"] = round(float(b[k]), 4)
                row[f"{label} (new)"] = round(float(r[k]), 4)
                row[f"Delta {label}"] = round(float(r[k]) - float(b[k]), 4)
        rows.append(row)
    if rows:
        comp = pd.DataFrame(rows)
        for _, label in keys:
            sub = ["panel"] + [c for c in (f"{label} (ref)", f"{label} (new)",
                                           f"Delta {label}") if c in comp.columns]
            if len(sub) > 1:
                print("\n" + comp[sub].to_string(index=False))
        deltas = {label: comp[f"Delta {label}"].mean() for _, label in keys
                  if f"Delta {label}" in comp.columns}
        print("\nMean difference: " + "   ".join(f"{k} {v:+.4f}" for k, v in deltas.items()))
        print("\nFor the decision, look at 'Honest F1' and 'Honest balanced F1': the first is "
              "\nexpected performance at the training distribution, the second at a balanced test set.")


def _inference_check(panels):
    runlog.section("INFERENCE WIRING CHECK (saved bundles)")
    for p in panels:
        df = load_panel(p)
        out = predict_panel_plus(p, df)
        pred = out["prediction"].to_numpy()
        y = df[LABEL_COL].to_numpy()
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1v = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        out5 = predict_panel_plus(p, df, target_prior=0.5)
        print(f"  {p:7s} in-sample F1={f1v:.4f}  predicted positive rate="
              f"{pred.mean():.3f}   |  positive rate at target prior 1:1="
              f"{out5['prediction'].mean():.3f}")


def main():
    panels = [a.upper() for a in sys.argv[1:]] or PANELS
    unknown = [p for p in panels if p not in PANEL_FILES]
    if unknown:
        raise SystemExit(f"Unknown panel(s): {unknown}. Choose from {PANELS}.")

    log_path = runlog.start(
        "improved",
        note="Improvement trials — threshold policy, paired One-SE, "
             "diverse-view stacking, shortcut audit")

    print(f"\nPanels     : {panels}")
    print(f"Output     : {OUTPUT_DIR.resolve()}")
    print(f"Reference  : {COMPARE_TO}")
    print("\nActive improvements")
    print(f"  1. Threshold policy       : {THRESHOLD_POLICY} "
          f"(compared: {', '.join(POLICIES)}; bootstrap median n={THR_BOOT})")
    print(f"  2. Paired One-SE          : {PAIRED_BOOTSTRAP} bootstrap draws")
    print(f"  3. Diverse-view stack     : {'on' if DIVERSE_STACK else 'off'}")
    print(f"  4. Shortcut audit         : {'on' if SHORTCUT_AUDIT else 'off'}")
    print(f"  5. Per-panel selector     : " +
          ", ".join(f"{p}={selection_method_for(p)}" for p in panels))
    print(f"  6. Seed ensemble          : {SEED_ENSEMBLE}")
    print(f"  7. NESTED_REPEATS={NESTED_REPEATS} (should match the reference)  "
          f"AUG_FRAC={AUG_FRAC}")
    if DIVERSE_STACK:
        print("\n  Note: the diverse view trains 7 base models instead of 5; this run will "
              "take roughly\n        30-40% longer than the reference pipeline. Turn it off "
              "with DIVERSE_STACK=0.")
    print(f"  Prior report              : {REPORT_PRIORS}")
    print("\n  No cached or inherited hyperparameters are used — every panel's Optuna "
          "search runs fresh.")

    t0 = time.time()
    rows, done = [], []
    out_csv = OUTPUT_DIR / "results_summary_improved.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        rows.append(run_panel(panel))
        done.append(panel)
        # Write after every panel: a partial result survives an interrupted run.
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    elapsed = int(time.time() - t0)
    runlog.section(f"ALL PANELS COMPLETE — {elapsed // 60}m {elapsed % 60}s")

    res = pd.DataFrame(rows)
    _print_summary(res)
    _compare_baseline(res)
    _inference_check(done)

    print(f"\nSummary CSV : {out_csv}")
    print(f"Bundles     : {OUTPUT_DIR / 'models'}/")
    print(f"Log         : {log_path}")


if __name__ == "__main__":
    main()
