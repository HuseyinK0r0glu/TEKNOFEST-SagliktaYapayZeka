"""Main flow for the sweep."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from pathugen.champion_config import save_champion_config
from search import runlog
from search.reference.config import PANEL_FILES, PANELS
from search.sweep.audit import choose_best, overfit_verdict
from search.sweep.config import (
    DECISION_GRID, GAP_LIMIT, POOLS, RULES, SELECT_OBJECTIVE, STABILITY_MIN,
    SWEEP_PROFILE, tc_key, train_grid,
)
from search.improved.config import POLICIES
from search.sweep.panel_sweep import sweep_panel
from search.sweep.persist import persist_winner
from search.sweep.report import compare_previous, report_panel, write_best_config_md
from search.reference.config import NESTED_REPEATS, NESTED_SPLITS, OPTUNA_TIMEOUT, OPTUNA_TRIALS, OUTPUT_DIR


def main(panels=None):
    """`panels` lets callers (e.g. PatHUgen.py's `run` command) invoke the
    sweep programmatically without going through sys.argv; the CLI entry
    point below still works exactly as before when panels is omitted.

    NOTE: this used to also fall back to `sys.argv[1:]` when `panels` was
    None, so that `python search/sweep/cli.py PANEL...` worked standalone.
    That collided with the real caller: `pathugen/cli.py`'s `run` subcommand
    invokes this as `sweep_main(panels=None)` after argparse has already
    consumed argv, but `sys.argv` itself still contains the outer script's
    own args (e.g. ["run"]) — so the fallback picked up "RUN" as if it were
    a panel name and crashed with "Unknown panel(s): ['RUN']". Nothing in
    this repo actually invokes this module standalone (the documented entry
    point is always `python PatHUgen.py run [PANEL...]`), so the sys.argv
    branch is dropped rather than reconciled with it.
    """
    panels = panels or PANELS
    unknown = [p for p in panels if p not in PANEL_FILES]
    if unknown:
        raise SystemExit(f"Unknown panel(s): {unknown}. Choose from {PANELS}")

    log_path = runlog.start(
        "sweep",
        note="Sweeps ALL options, entirely from scratch (no cached or "
             "inherited hyperparameters), picks the best among those that "
             "pass the overfitting audit")

    n_train = sum(len(train_grid(p)) for p in panels)
    print(f"\nPanels     : {panels}")
    print(f"Output     : {OUTPUT_DIR.resolve()}")
    print(f"Profile    : {SWEEP_PROFILE}")
    print(f"\nAxes to sweep")
    print(f"  EXPENSIVE (retraining, every cell tuned fresh — no reuse):")
    for p in panels:
        g = train_grid(p)
        print(f"    {p:<7s} {len(g)} configurations -> " +
              ", ".join(tc_key(s, a) for s, a in g))
    print(f"  CHEAP (post-training, free):")
    print(f"    rule    : {', '.join(RULES)}")
    print(f"    pool    : {', '.join(POOLS)}")
    print(f"    policy  : {', '.join(POLICIES)}")
    print(f"    -> {len(DECISION_GRID)} decision combinations per training configuration")
    print(f"\nTotal candidates to evaluate: {n_train * len(DECISION_GRID)}")
    print(f"\nOverfitting audit")
    print(f"  optimism gap <= {GAP_LIMIT:.3f}   champion stability >= "
          f"%{100*STABILITY_MIN:.0f}   honest F1 > naive baseline")
    print(f"Selection objective : {SELECT_OBJECTIVE}")
    print(f"Optuna       : {OPTUNA_TRIALS} trials / {OPTUNA_TIMEOUT} s  "
          f"(fresh for every training configuration, no cache, no inheritance)")
    print(f"Nested CV    : {NESTED_SPLITS} folds x {NESTED_REPEATS} repeats")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_rows, winners, fallbacks, champion_config = [], {}, {}, {}
    sweep_csv = OUTPUT_DIR / "sweep_all_results.csv"

    for panel in panels:
        pdf, artifacts, ctx = sweep_panel(panel)
        scored, win, fallback = choose_best(pdf)
        all_rows.append(scored)
        pd.concat(all_rows, ignore_index=True).to_csv(sweep_csv, index=False)
        report_panel(panel, scored, win, fallback)
        champ, cc_entry = persist_winner(panel, win, artifacts, ctx)
        winners[panel] = win
        fallbacks[panel] = fallback
        if cc_entry is not None:
            champion_config[panel] = cc_entry
        else:
            print(f"    [note] {panel}'s champion is a '{win['strategy']}' strategy, "
                  f"not a single XGBoost model — it cannot be written to "
                  f"champion_config.json (see champion.json for the full detail).")
        print(f"    bundle -> {OUTPUT_DIR / 'models' / panel}")
        del artifacts        # don't keep large model objects in memory

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(sweep_csv, index=False)

    elapsed = int(time.time() - t0)
    runlog.section(f"SWEEP COMPLETE — {elapsed//3600}h {(elapsed%3600)//60}m")

    summary = pd.DataFrame([{
        "panel": p, "champion": w["champion"], "strategy": w["strategy"],
        "selection": w["selection"], "aug_frac": w["aug_frac"],
        "n_features": w["n_features"], "rule": w["rule"], "pool": w["pool"],
        "policy": w["policy"], "threshold": round(w["threshold"], 3),
        "oof_f1": round(w["oof_f1"], 4),
        "honest_f1": round(w["honest_f1"], 4),
        "honest_balanced_f1": round(w["honest_balanced_f1"], 4),
        "honest_auc": round(w["honest_auc"], 4),
        "optimism_gap": round(w["gap"], 4),
        "stability": round(w["stability"], 2),
        "overfit_clean": overfit_verdict(w)[0],
        "trivial_f1": round(w["trivial_f1"], 4),
    } for p, w in winners.items()])
    summary.to_csv(OUTPUT_DIR / "results_summary_best.csv", index=False)

    runlog.section("FINAL DECISION — SELECTED CONFIGURATION PER PANEL")
    print(summary.to_string(index=False))
    print(f"\nMean honest F1              : {summary['honest_f1'].mean():.4f}")
    print(f"Mean honest balanced F1     : {summary['honest_balanced_f1'].mean():.4f}")
    print(f"Mean honest AUC             : {summary['honest_auc'].mean():.4f}")
    print(f"Mean optimism gap           : {summary['optimism_gap'].mean():+.4f}")

    risky = [p for p, w in winners.items() if not overfit_verdict(w)[0]]
    print()
    if risky:
        print(f"! OVERFITTING RISK: no configuration passed the audit for "
              f"{', '.join(risky)}.\n  Treat these panels' results with caution "
              f"(see BEST_CONFIG.md for details).")
    else:
        print("v ALL PANELS PASSED THE OVERFITTING AUDIT.")

    compare_previous(winners)

    md = OUTPUT_DIR / "BEST_CONFIG.md"
    write_best_config_md(winners, all_df, md)
    (OUTPUT_DIR / "best_config.json").write_text(json.dumps(
        {p: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in w.items() if k not in ("fail_reasons", "warn_reasons")}
         for p, w in winners.items()}, indent=2, ensure_ascii=False, default=str))

    cc_path = OUTPUT_DIR / "champion_config.json"
    if len(champion_config) == len(panels):
        save_champion_config(champion_config, path=cc_path)
        print(f"\nchampion_config.json written for all {len(panels)} panels "
              f"({cc_path}) — copy it over the project's champion_config.json "
              f"to adopt this result (see Champion_Config.md).")
    else:
        missing = [p for p in panels if p not in champion_config]
        print(f"\nchampion_config.json NOT written: panel(s) {missing} did not "
              f"produce a single-XGBoost champion this run. Run with just the "
              f"other panels' results already adopted, or extend pathugen to "
              f"support this panel's winning strategy.")

    runlog.section("OUTPUTS")
    print(f"  HUMAN-READABLE DECISION : {md}          <- READ THIS FIRST")
    print(f"  Machine-readable        : {OUTPUT_DIR / 'best_config.json'}")
    if len(champion_config) == len(panels):
        print(f"  Adoptable champion config: {cc_path}")
    print(f"  Per-panel summary       : {OUTPUT_DIR / 'results_summary_best.csv'}")
    print(f"  Full sweep              : {sweep_csv}  ({len(all_df)} rows)")
    print(f"  Model bundles           : {OUTPUT_DIR / 'models'}/")
    print(f"  Log                     : {log_path}")


if __name__ == "__main__":
    main()
