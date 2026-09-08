"""Panel sweep — the expensive (training) grid. Every (feature-selection,
augmentation-ratio) cell gets its OWN fresh Optuna search: no cached
`tuned_params*.json` is ever read back, and nothing is inherited from an
earlier run. This is slower than sharing tuning across the augmentation
axis, but it is what "try every hyperparameter, from scratch, with no
hints" requires.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from pathugen.data_io import load_panel
from pathugen.features import build_features
from search import runlog
from search.improved.config import DIVERSE_SPECS, STD_SPECS
from search.improved.threshold import pick_threshold
from search.improved.views import generate_base_oof_plus, relief_pruned_view
from search.reference.champion import _trivial_f1
from search.reference.config import (
    NESTED_REPEATS, NESTED_SPLITS, OPTUNA_TIMEOUT, OPTUNA_TRIALS, OUTPUT_DIR,
)
from search.reference.modeling import KINDS
from search.reference.selection import null_importance_selection
from search.reference.tuning import tune_model
from search.sweep.augpool import aug_for
from search.sweep.candidates import build_candidates, bootstrap_aucs, pick_champion, score_at
from search.sweep.config import DECISION_GRID, DIVERSE_STACK, THR_BOOT, dc_key, tc_key, train_grid
from search.sweep.nested_sweep import nested_sweep


def sweep_panel(panel):
    t_panel = time.time()
    runlog.section(f"PANEL {panel} — CONFIGURATION SWEEP")

    df = load_panel(panel)
    X, y, cat_cols = build_features(df)
    y = pd.Series(np.asarray(y))
    trivial = _trivial_f1(y)
    grid = train_grid(panel)
    print(f"  rows={len(y)}  features={X.shape[1]}  positive rate=%{100*y.mean():.1f}  "
          f"naive baseline F1={trivial:.4f}")
    print(f"  training configurations: {len(grid)}   decision configurations: "
          f"{len(DECISION_GRID)}   -> {len(grid)*len(DECISION_GRID)} candidates total")

    bundle_root = OUTPUT_DIR / "models" / panel
    bundle_root.mkdir(parents=True, exist_ok=True)
    rows, artifacts = [], {}

    for gi, (selection, aug_frac) in enumerate(grid, 1):
        t_cfg = time.time()
        tk = tc_key(selection, aug_frac)
        print(f"\n  -- [{gi}/{len(grid)}] {tk} " + "-" * 40)

        tbl = null_importance_selection(X, y, cat_cols, method=selection)
        selected = tbl.loc[tbl["keep"], "feature"].tolist()
        Xs = X[selected]
        sel_cat = [c for c in cat_cols if c in set(selected)]
        print(f"     selection:{selection} -> {len(selected)}/{X.shape[1]} features "
              f"({len(sel_cat)} categorical)")

        aug = aug_for(panel, aug_frac, Xs.columns)
        if aug is not None:
            print(f"     augmentation: +{len(aug[0])} GENEL rows "
                  f"({int((aug[1] == 1).sum())} pathogenic / {int((aug[1] == 0).sum())} benign)"
                  f"  [ratio {aug_frac:g}]")
        else:
            print("     augmentation: none")

        # Optuna — a fresh search for THIS exact (selection, aug_frac) cell.
        params = {}
        for kind in KINDS:
            best, sc_ = tune_model(kind, Xs, y, sel_cat, OPTUNA_TRIALS,
                                   OPTUNA_TIMEOUT, aug=aug)
            params[kind] = best
            print(f"       tuned {kind:11s} AUC={sc_:.4f}", flush=True)
        cell_tag = tk.replace("|", "_").replace("=", "-")
        (bundle_root / f"tuned_params_{cell_tag}.json").write_text(
            json.dumps(params, indent=2))

        views, specs = {"full": selected}, list(STD_SPECS)
        if DIVERSE_STACK:
            rp = relief_pruned_view(Xs, y, sel_cat)
            if 3 <= len(rp) < len(selected):
                views["rp"] = rp
                specs += list(DIVERSE_SPECS)
                print(f"     diverse view: {len(rp)}/{len(selected)} features -> +2 columns")

        base_oof, _, full_models, model_view = generate_base_oof_plus(
            Xs, y, sel_cat, params, views, specs, aug=aug)
        cand_oof, _, strategies, cols_of = build_candidates(
            base_oof, y, full_models, model_view, specs)
        aucs, boot = bootstrap_aucs(y, cand_oof)

        print(f"     nested CV ({NESTED_SPLITS}x{NESTED_REPEATS} folds) — "
              f"scoring {len(DECISION_GRID)} decision configurations on the same folds...",
              flush=True)
        nested = nested_sweep(panel, X, y, cat_cols, params, selection, aug_frac,
                              "rp" in views)

        thr_cache = {}
        for dc in DECISION_GRID:
            champ, best_key = pick_champion(aucs, boot, strategies,
                                            dc["rule"], dc["pool"])
            ck = (champ, dc["policy"])
            if ck not in thr_cache:
                thr_cache[ck] = pick_threshold(y.to_numpy(), cand_oof[champ],
                                               dc["policy"], n_boot=THR_BOOT)
            t = thr_cache[ck]
            oof = score_at(y.to_numpy(), cand_oof[champ], t)
            nk = nested[dc_key(dc)]
            rows.append({
                "panel": panel, "selection": selection, "aug_frac": aug_frac,
                "n_features": len(selected), "diverse": "rp" in views,
                "rule": dc["rule"], "pool": dc["pool"], "policy": dc["policy"],
                "champion": champ, "strategy": strategies[champ], "threshold": t,
                "oof_f1": oof["f1"], "oof_balanced_f1": oof["balanced_f1"],
                "oof_mcc": oof["mcc"], "oof_auc": aucs[champ],
                "tn": oof["tn"], "fp": oof["fp"], "fn": oof["fn"], "tp": oof["tp"],
                "honest_f1": nk["honest_f1"], "honest_f1_std": nk["honest_f1_std"],
                "honest_balanced_f1": nk["honest_balanced_f1"],
                "honest_auc": nk["honest_auc"],
                "stability": nk["stability"], "modal_champion": nk["modal_champion"],
                "n_distinct_champions": nk["n_distinct_champions"],
                "gap": oof["f1"] - nk["honest_f1"], "trivial_f1": trivial,
            })
        artifacts[tk] = dict(selected=selected, sel_cat=sel_cat, params=params,
                             views=views, specs=specs, base_oof=base_oof,
                             full_models=full_models, model_view=model_view,
                             cand_oof=cand_oof, strategies=strategies,
                             cols_of=cols_of, aucs=aucs, boot=boot, aug=aug, Xs=Xs)
        print(f"     [{tk} done — {int(time.time()-t_cfg)} s]", flush=True)

    print(f"\n  [PANEL {panel} sweep done — {int(time.time()-t_panel)} s, "
          f"{len(rows)} candidates evaluated]")
    return pd.DataFrame(rows), artifacts, dict(X=X, y=y, cat_cols=cat_cols,
                                               trivial=trivial, df=df)
