"""Per-panel pipeline: select → tune → OOF champion → honest nested estimate →
fit + persist the full-data champion bundle."""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from pathugen.data_io import load_panel
from pathugen.features import build_features
from search.reference.augment import build_aug_pool
from search.reference.champion import STACK_VARIANTS, _trivial_f1, select_champion
from search.reference.config import OBJECTIVE, OPTUNA_TIMEOUT, OPTUNA_TRIALS, OUTPUT_DIR, PANELS, RUN_NESTED
from search.reference.modeling import KINDS
from search.reference.nested import nested_eval
from search.reference.selection import null_importance_selection
from search.reference.stacking import fit_final_stack, generate_base_oof
from search.reference.tuning import tune_model

SELECTION_METHOD = os.environ.get("SELECTION_METHOD", "null").strip().lower()


def run_panel(panel):
    t0 = time.time()
    print("\n" + "=" * 70)
    print(f"  PANEL {panel}")
    print("=" * 70, flush=True)
    df = load_panel(panel)
    X, y, cat_cols = build_features(df)
    y = pd.Series(np.asarray(y))
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"  rows={len(y)}  features={X.shape[1]}  "
          f"pathogenic={n_pos} benign={n_neg} (pos rate {y.mean():.1%})")

    # 1) Feature selection (full panel) ---------------------------------------
    tbl = null_importance_selection(X, y, cat_cols, method=SELECTION_METHOD)
    selected = tbl.loc[tbl["keep"], "feature"].tolist()
    Xs = X[selected]
    sel_cat = [c for c in cat_cols if c in set(selected)]
    print(f"  [select:{SELECTION_METHOD}] kept {len(selected)}/{X.shape[1]} features "
          f"({len(sel_cat)} categorical)")

    # 2) GENEL augmentation pool (PAH/CFTR) -----------------------------------
    aug_X, aug_y = build_aug_pool(panel, columns=list(Xs.columns))
    aug = (aug_X, aug_y) if aug_X is not None else None
    if aug is not None:
        print(f"  [augment] += {len(aug_X)} GENEL rows in training folds "
              f"({int((aug_y == 1).sum())} pathogenic / {int((aug_y == 0).sum())} benign)")

    # 3) Optuna tuning per base model -----------------------------------------
    params_by_kind = {}
    for kind in KINDS:
        best, score = tune_model(kind, Xs, y, sel_cat, OPTUNA_TRIALS, OPTUNA_TIMEOUT, aug=aug)
        params_by_kind[kind] = best
        print(f"    tuned {kind:11s} {OBJECTIVE.upper()}={score:.4f}", flush=True)

    # 4) Base OOF + champion selection (mildly optimistic OOF) ----------------
    base_oof, singles, full_models = generate_base_oof(Xs, y, sel_cat, params_by_kind, aug=aug)
    ch = select_champion(base_oof, singles, y)
    label = ch["model"] if ch["strategy"] == "single" else ch["strategy"]
    trivial = _trivial_f1(y)
    print(f"  [OOF champion] {label}  F1={ch['oof_f1']:.4f}±{ch['f1_std']:.4f}  "
          f"AUC={ch['auc']:.4f}±{ch['auc_std']:.4f}  @ t={ch['threshold']:.3f}  "
          f"(trivial F1={trivial:.4f}, lift {ch['oof_f1']-trivial:+.4f})")

    # 5) Honest nested-CV estimate (the number to trust) ----------------------
    nested = {}
    if RUN_NESTED:
        nested = nested_eval(panel, X, y, cat_cols, params_by_kind, SELECTION_METHOD)
        print(f"  [HONEST nested] F1={nested['nested_f1']:.4f}±{nested['nested_f1_std']:.4f}  "
              f"AUC={nested['nested_auc']:.4f}  balF1={nested['nested_balanced_f1']:.4f}")

    # 6) Persist the full-data champion bundle --------------------------------
    bundle = OUTPUT_DIR / "models" / panel
    bundle.mkdir(parents=True, exist_ok=True)
    for kind, model in full_models.items():
        joblib.dump(model, bundle / f"base_{kind}.joblib")
    for vname, vkinds in STACK_VARIANTS.items():
        cols = [c for c in vkinds if c in base_oof.columns]
        calibrators, meta, order = fit_final_stack(base_oof, y, cols=cols)
        joblib.dump(calibrators, bundle / f"calibrators_{vname}.joblib")
        joblib.dump(meta, bundle / f"meta_{vname}.joblib")
        (bundle / f"{vname}_order.json").write_text(json.dumps(order))
    base_order = ch["cols"] if ch["strategy"].startswith("stacked") else list(KINDS)
    champion = {"panel": panel, "strategy": ch["strategy"], "model": ch["model"],
                "threshold": ch["threshold"], "base_order": base_order,
                "augmented": aug is not None, "selection_method": SELECTION_METHOD}
    (bundle / "champion.json").write_text(json.dumps(champion, indent=2))
    (bundle / "selected_features.json").write_text(json.dumps(list(Xs.columns)))

    print(f"  [done in {int(time.time()-t0)}s]  bundle → {bundle}")
    row = {"panel": panel, "n": len(y), "champion": label, "strategy": ch["strategy"],
           "n_features": len(selected), "augmented": aug is not None,
           "threshold": round(ch["threshold"], 3),
           "oof_f1": round(ch["oof_f1"], 4), "oof_auc": round(ch["auc"], 4),
           "trivial_f1": round(trivial, 4)}
    if nested:
        row.update({"honest_f1": round(nested["nested_f1"], 4),
                    "honest_auc": round(nested["nested_auc"], 4),
                    "honest_balanced_f1": round(nested["nested_balanced_f1"], 4),
                    "optimism_gap": round(ch["oof_f1"] - nested["nested_f1"], 4)})
    return row


def run_all(panels=None):
    panels = panels or PANELS
    t0 = time.time()
    rows = [run_panel(p) for p in panels]
    summary = pd.DataFrame(rows)
    OUTPUT_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
    print("\n" + "=" * 70)
    print(f"  ALL PANELS COMPLETE — {int(time.time()-t0)//60}m {int(time.time()-t0)%60}s")
    print("=" * 70)
    return summary
