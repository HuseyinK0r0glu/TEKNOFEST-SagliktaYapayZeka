"""Persist the winning configuration's bundle, and — when the winner is a
single tuned XGBoost model (the only shape `pathugen.train()` understands)
— its `champion_config.json` entry."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from pathugen.config import POS
from search.improved.pipeline import fit_seed_ensemble
from search.improved.threshold import pick_threshold, prior_shift_threshold
from search.reference.config import OUTPUT_DIR
from search.reference.stacking import fit_final_stack
from search.sweep.config import SEED_ENSEMBLE, THR_BOOT, tc_key


def persist_winner(panel, win, art, ctx):
    bundle = OUTPUT_DIR / "models" / panel
    bundle.mkdir(parents=True, exist_ok=True)
    tk = tc_key(win["selection"], win["aug_frac"])
    a = art[tk]
    y = ctx["y"]

    for col, model in a["full_models"].items():
        joblib.dump(model, bundle / f"base_{col}.joblib")
    for vname, cols in a["cols_of"].items():
        calibrators, meta, order = fit_final_stack(a["base_oof"], y, cols=cols)
        joblib.dump(calibrators, bundle / f"calibrators_{vname}.joblib")
        joblib.dump(meta, bundle / f"meta_{vname}.joblib")
        (bundle / f"{vname}_order.json").write_text(json.dumps(order))

    champ_proba = a["cand_oof"][win["champion"]]
    iso = IsotonicRegression(out_of_bounds="clip").fit(champ_proba, np.asarray(y))
    joblib.dump(iso, bundle / "champion_calibrator.joblib")
    train_prior = float(np.mean(np.asarray(y) == POS))
    t_cal = pick_threshold(np.asarray(y), iso.predict(champ_proba),
                           win["policy"], n_boot=THR_BOOT)

    seeds = 0
    champion_kind = None
    if win["strategy"] == "single" and SEED_ENSEMBLE > 1:
        col = win["champion"][len("single_"):]
        champion_kind = dict((c, k) for c, k, _ in a["specs"])[col]
        feats = a["model_view"][col]
        if a["aug"] is not None:
            X_full = pd.concat([a["Xs"][feats], a["aug"][0][feats]], axis=0,
                               ignore_index=True)
            y_full = np.concatenate([np.asarray(y), np.asarray(a["aug"][1])])
        else:
            X_full, y_full = a["Xs"][feats], np.asarray(y)
        cvc = [c for c in a["sel_cat"] if c in set(feats)]
        models = fit_seed_ensemble(champion_kind, a["params"].get(champion_kind) or {},
                                   X_full, y_full, cvc, SEED_ENSEMBLE)
        for i, m in enumerate(models):
            joblib.dump(m, bundle / f"champion_seed{i}.joblib")
        seeds = len(models)
    elif win["strategy"] == "single":
        col = win["champion"][len("single_"):]
        champion_kind = dict((c, k) for c, k, _ in a["specs"])[col]

    champion = {
        "panel": panel,
        "strategy": win["strategy"],
        "model": win["champion"][len("single_"):] if win["strategy"] == "single" else None,
        "base_order": a["cols_of"].get(win["strategy"], list(a["base_oof"].columns)),
        "model_view": a["model_view"],
        "threshold": float(win["threshold"]),
        "threshold_policy": win["policy"],
        "selection_rule": win["rule"],
        "candidate_pool": win["pool"],
        "selection_method": win["selection"],
        "aug_frac": float(win["aug_frac"]),
        "augmented": a["aug"] is not None,
        "diverse_stack": bool(win["diverse"]),
        "n_features": int(win["n_features"]),
        "seed_ensemble": seeds,
        "train_prior": train_prior,
        "calibrated_threshold": t_cal,
        "prior_shifted_thresholds": {
            f"{p:g}": prior_shift_threshold(t_cal, train_prior, p)
            for p in (0.5, 0.6, 0.74)},
        "honest_f1": float(win["honest_f1"]),
        "honest_balanced_f1": float(win["honest_balanced_f1"]),
        "honest_auc": float(win["honest_auc"]),
        "oof_f1": float(win["oof_f1"]),
        "optimism_gap": float(win["gap"]),
        "champion_stability": float(win["stability"]),
    }
    (bundle / "champion.json").write_text(json.dumps(champion, indent=2,
                                                     ensure_ascii=False))
    (bundle / "selected_features.json").write_text(json.dumps(a["selected"]))

    champion_config_entry = None
    if win["strategy"] == "single" and champion_kind == "xgboost":
        col = win["champion"][len("single_"):]
        champion_config_entry = {
            "selection_method": win["selection"],
            "aug_frac": float(win["aug_frac"]),
            "threshold": round(float(win["threshold"]), 3),
            "threshold_policy": win["policy"],
            "xgb_params": a["params"]["xgboost"],
            "features": list(a["model_view"][col]),
        }
    return champion, champion_config_entry
