"""Panel driver for the improved pipeline."""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from pathugen.config import SEED
from pathugen.data_io import load_panel
from pathugen.features import build_features, get_cat_maps
from search import runlog
from search.improved.config import (
    AUG_FRAC, DIVERSE_SPECS, DIVERSE_STACK, PANEL_SELECTION, POLICIES,
    REPORT_PRIORS, SEED_ENSEMBLE, SHORTCUT_AUDIT, STD_SPECS, THRESHOLD_POLICY,
)
from search.improved.paired_selection import select_champion_plus
from search.improved.shortcut_audit import shortcut_audit
from search.improved.threshold import metrics_at, pick_threshold, prior_shift_threshold
from search.improved.views import generate_base_oof_plus, relief_pruned_view, stack_variants
from search.reference.augment import build_aug_pool
from search.reference.champion import _trivial_f1
from search.reference.config import NESTED_REPEATS, NESTED_SPLITS, OBJECTIVE, OPTUNA_TIMEOUT, OPTUNA_TRIALS, OUTPUT_DIR, RUN_NESTED
from search.reference.modeling import KINDS, MAKERS
from search.reference.selection import null_importance_selection
from search.reference.stacking import fit_final_stack
from search.reference.tuning import _fit_fn, tune_model
from search.improved.nested import nested_eval_plus


def selection_method_for(panel):
    return os.environ.get(
        f"SELECTION_METHOD_{panel}",
        PANEL_SELECTION.get(panel, os.environ.get("SELECTION_METHOD", "null"))
    ).strip().lower()


def subsample_aug(aug_X, aug_y, frac, seed=SEED):
    """A stratified subset of the borrowed GENEL rows (a plumbing-free
    stand-in for down-weighting)."""
    if aug_X is None or frac >= 1.0:
        return aug_X, aug_y
    rng = np.random.default_rng(seed)
    ya = np.asarray(aug_y)
    keep = []
    for cls in np.unique(ya):
        idx = np.flatnonzero(ya == cls)
        k = max(1, int(round(len(idx) * frac)))
        keep.append(rng.choice(idx, size=k, replace=False))
    keep = np.sort(np.concatenate(keep))
    return (aug_X.iloc[keep].reset_index(drop=True),
            pd.Series(ya[keep]).reset_index(drop=True))


def fit_seed_ensemble(kind, params, X, y, cat_cols, n_seeds):
    """Refits the champion with several seeds (predictions are averaged later)."""
    import search.reference.modeling as modeling_module

    models, orig = [], modeling_module.SEED
    fit_fn = _fit_fn(kind, cat_cols)
    try:
        for s in range(orig, orig + max(1, n_seeds)):
            modeling_module.SEED = s
            m = MAKERS[kind](params, pd.Series(np.asarray(y)), list(X.columns), cat_cols)
            m.fit(X, np.asarray(y), **(fit_fn(X, np.asarray(y)) if fit_fn else {}))
            models.append(m)
    finally:
        modeling_module.SEED = orig
    return models


def run_panel(panel):
    t0 = time.time()
    runlog.section(f"PANEL {panel}")

    df = load_panel(panel)
    X, y, cat_cols = build_features(df)
    y = pd.Series(np.asarray(y))
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    train_prior = float(y.mean())
    print(f"  rows={len(y)}  features={X.shape[1]}  pathogenic={n_pos} benign={n_neg} "
          f"(positive rate {100*train_prior:.1f}%)")

    # 1) Feature selection (per-panel method) --------------------------------
    method = selection_method_for(panel)
    tbl = null_importance_selection(X, y, cat_cols, method=method)
    selected = tbl.loc[tbl["keep"], "feature"].tolist()
    Xs = X[selected]
    sel_cat = [c for c in cat_cols if c in set(selected)]
    print(f"  [selection:{method}] kept {len(selected)}/{X.shape[1]} features "
          f"({len(sel_cat)} categorical)")

    # 2) GENEL augmentation pool -----------------------------------------------
    aug_X, aug_y = build_aug_pool(panel, columns=list(Xs.columns))
    aug_X, aug_y = subsample_aug(aug_X, aug_y, AUG_FRAC)
    aug = (aug_X, aug_y) if aug_X is not None else None
    if aug is not None:
        print(f"  [augmentation] +{len(aug_X)} GENEL rows into the training folds "
              f"({int((aug_y == 1).sum())} pathogenic / {int((aug_y == 0).sum())} benign)"
              + (f"  [AUG_FRAC={AUG_FRAC}]" if AUG_FRAC < 1.0 else ""))

    # 3) Optuna — always tuned fresh; no cached or inherited hyperparameters --
    bundle = OUTPUT_DIR / "models" / panel
    bundle.mkdir(parents=True, exist_ok=True)
    params_by_kind = {}
    for kind in KINDS:
        best, score = tune_model(kind, Xs, y, sel_cat, OPTUNA_TRIALS,
                                 OPTUNA_TIMEOUT, aug=aug)
        params_by_kind[kind] = best
        print(f"    tuned {kind:11s} {OBJECTIVE.upper()}={score:.4f}", flush=True)
    (bundle / "tuned_params.json").write_text(json.dumps(params_by_kind, indent=2))

    # 4) Views + base model OOF ----------------------------------------
    views, specs = {"full": selected}, list(STD_SPECS)
    if DIVERSE_STACK:
        rp = relief_pruned_view(Xs, y, sel_cat)
        if 3 <= len(rp) < len(selected):
            views["rp"] = rp
            specs += list(DIVERSE_SPECS)
            print(f"  [diverse view] ReliefF-pruned second view: "
                  f"{len(rp)}/{len(selected)} features -> +2 base columns")
        else:
            print("  [diverse view] ReliefF pruning did not yield a meaningful subset, skipped")

    base_oof, singles, full_models, model_view = generate_base_oof_plus(
        Xs, y, sel_cat, params_by_kind, views, specs, aug=aug)

    # 5) Champion — paired bootstrap One-SE ----------------------------
    print("  [champion selection] paired bootstrap One-SE over AUC:")
    ch = select_champion_plus(base_oof, singles, y, specs, THRESHOLD_POLICY,
                              verbose=True)
    label = ch["col"] if ch["strategy"] == "single" else ch["strategy"]
    trivial = _trivial_f1(y)

    # 6) Threshold-policy comparison --------------------------------------
    proba = ch["proba"]
    print(f"\n  [threshold policies] champion = {label}; selected policy = "
          f"{THRESHOLD_POLICY}")
    hdr = (f"    {'policy':<14s} {'thresh':>6s} {'F1':>7s} {'F1@1:1':>8s} {'MCC':>7s} "
           f"{'precision':>9s} {'recall':>11s}")
    print(hdr)
    policy_rows = {}
    for pol in POLICIES:
        t = pick_threshold(y.to_numpy(), proba, pol)
        m = metrics_at(y.to_numpy(), proba, t, priors=(0.5,))
        policy_rows[pol] = m
        mark = "  <-shipped" if pol == THRESHOLD_POLICY else ""
        print(f"    {pol:<14s} {t:6.3f} {m['f1']:7.4f} {m['f1@0.5']:8.4f} "
              f"{m['mcc']:7.4f} {m['precision']:9.4f} {m['recall']:11.4f}{mark}")

    threshold = policy_rows[THRESHOLD_POLICY]["threshold"]
    chosen = policy_rows[THRESHOLD_POLICY]
    print(f"    naive baseline F1={trivial:.4f}   lift {chosen['f1']-trivial:+.4f}   "
          f"TN/FP/FN/TP = {chosen['tn']}/{chosen['fp']}/{chosen['fn']}/{chosen['tp']}")

    # 7) Threshold table under a prior shift -----------------------------------
    print("  [prior correction] threshold to use if the test class ratio turns out "
          "different (on the isotonic-calibrated probability):")
    iso = IsotonicRegression(out_of_bounds="clip").fit(proba, np.asarray(y))
    p_cal = iso.predict(proba)
    t_cal = pick_threshold(y.to_numpy(), p_cal, THRESHOLD_POLICY)
    shift = {}
    for tp_ in REPORT_PRIORS:
        t_shift = prior_shift_threshold(t_cal, train_prior, tp_)
        shift[f"{tp_:g}"] = t_shift
        m = metrics_at(y.to_numpy(), p_cal, t_shift, priors=(tp_,))
        print(f"      target prior {tp_:.2f} -> calibrated threshold {t_shift:.3f}   "
              f"(expected F1 at this prior {m[f'f1@{tp_:g}']:.4f})")

    # 8) Shortcut audit -----------------------------------------------------
    audit = {}
    if SHORTCUT_AUDIT:
        print()
        audit = shortcut_audit(panel, X, y, selected, cat_cols,
                               ch["col"] or "xgboost", params_by_kind,
                               (aug_X, aug_y) if aug is not None else None)

    # 9) Honest nested evaluation ------------------------------------------
    nested = {}
    if RUN_NESTED:
        print(f"\n  [honest nested CV] {NESTED_SPLITS} folds x {NESTED_REPEATS} repeats "
              f"— selection, champion, and threshold are re-decided in every outer fold...", flush=True)
        aug_nested = build_aug_pool(panel, columns=list(X.columns))
        aug_nested = subsample_aug(*aug_nested, AUG_FRAC)
        nested = nested_eval_plus(panel, X, y, cat_cols, params_by_kind, method,
                                  DIVERSE_STACK, THRESHOLD_POLICY,
                                  aug_nested if aug_nested[0] is not None else None)
        print(f"  [honest nested CV] F1={nested['nested_f1']:.4f}"
              f"±{nested['nested_f1_std']:.4f}  AUC={nested['nested_auc']:.4f}  "
              f"balancedF1={nested['nested_balanced_f1']:.4f}")
        print(f"      candidates selected across outer folds: {', '.join(nested['strategies'])}")

    # 10) Save the bundle -------------------------------------------------------
    for col, model in full_models.items():
        joblib.dump(model, bundle / f"base_{col}.joblib")
    for vname, vcols in stack_variants(specs).items():
        cols = [c for c in vcols if c in base_oof.columns]
        if len(cols) < 2:
            continue
        calibrators, meta, order = fit_final_stack(base_oof, y, cols=cols)
        joblib.dump(calibrators, bundle / f"calibrators_{vname}.joblib")
        joblib.dump(meta, bundle / f"meta_{vname}.joblib")
        (bundle / f"{vname}_order.json").write_text(json.dumps(order))
    joblib.dump(iso, bundle / "champion_calibrator.joblib")

    seed_models = []
    if ch["strategy"] == "single" and SEED_ENSEMBLE > 1:
        kind = dict((c, k) for c, k, _ in specs)[ch["col"]]
        feats = model_view[ch["col"]]
        if aug is not None:
            X_full = pd.concat([Xs[feats], aug_X[feats]], axis=0, ignore_index=True)
            y_full = np.concatenate([np.asarray(y), np.asarray(aug_y)])
        else:
            X_full, y_full = Xs[feats], np.asarray(y)
        cvc = [c for c in sel_cat if c in set(feats)]
        seed_models = fit_seed_ensemble(kind, params_by_kind.get(kind) or {},
                                        X_full, y_full, cvc, SEED_ENSEMBLE)
        for i, m in enumerate(seed_models):
            joblib.dump(m, bundle / f"champion_seed{i}.joblib")
        print(f"  [seed ensemble] champion refit with {SEED_ENSEMBLE} seeds "
              f"-> champion_seed0..{len(seed_models)-1}.joblib")

    champion = {
        "panel": panel, "strategy": ch["strategy"], "model": ch["col"],
        "base_order": ch["cols"] if ch["strategy"].startswith("stacked")
                      else [c for c, _, _ in specs],
        "model_view": model_view, "threshold": threshold,
        "threshold_policy": THRESHOLD_POLICY,
        "threshold_by_policy": {k: v["threshold"] for k, v in policy_rows.items()},
        "calibrated_threshold": t_cal, "prior_shifted_thresholds": shift,
        "train_prior": train_prior, "augmented": aug is not None,
        "aug_frac": AUG_FRAC, "selection_method": method,
        "seed_ensemble": len(seed_models), "diverse_stack": "rp" in views,
    }
    (bundle / "champion.json").write_text(json.dumps(champion, indent=2, ensure_ascii=False))
    (bundle / "selected_features.json").write_text(json.dumps(selected))
    (bundle / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    print(f"\n  [done — {int(time.time()-t0)}s]  bundle -> {bundle}")

    row = {"panel": panel, "n": len(y), "champion": label, "strategy": ch["strategy"],
           "selection": method, "n_features": len(selected),
           "diverse_stack": "rp" in views, "augmented": aug is not None,
           "threshold_policy": THRESHOLD_POLICY, "threshold": round(threshold, 3),
           "oof_f1": round(chosen["f1"], 4),
           "oof_f1_balanced": round(chosen["f1@0.5"], 4),
           "oof_mcc": round(chosen["mcc"], 4),
           "oof_auc": round(ch["auc"], 4),
           "oof_ap": round(float(average_precision_score(y, proba)), 4),
           "trivial_f1": round(trivial, 4),
           "f1_if_policy_f1": round(policy_rows["f1"]["f1"], 4),
           "balf1_if_policy_f1": round(policy_rows["f1"]["f1@0.5"], 4),
           "tn": chosen["tn"], "fp": chosen["fp"], "fn": chosen["fn"], "tp": chosen["tp"]}
    if nested:
        row.update({"honest_f1": round(nested["nested_f1"], 4),
                    "honest_auc": round(nested["nested_auc"], 4),
                    "honest_balanced_f1": round(nested["nested_balanced_f1"], 4),
                    "optimism_gap": round(chosen["f1"] - nested["nested_f1"], 4)})
    if audit.get("ablation"):
        row["f1_without_missingness"] = round(audit["ablation"]["f1"], 4)
        row["auc_without_missingness"] = round(audit["ablation"]["auc"], 4)
    return row
