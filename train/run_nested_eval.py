"""Leak-free nested cross-validation — the honest generalization estimate.

This is a DIAGNOSTIC only: it never feeds the shipped bundle. Its sole job is to
report the true per-panel F1 and the optimism gap versus the pooled-OOF numbers
the rest of the pipeline reports.

Why the rest of the pipeline is optimistic: feature selection, hyperparameters,
the threshold and the champion strategy are all chosen while touching every row.
Here, for each OUTER fold, all of that happens on the outer-train rows only and
is then locked and evaluated once on the untouched outer-test rows:

    Outer StratifiedKFold (NESTED_OUTER_SPLITS; repeated NESTED_REPEATS times)
      for each outer fold:
        outer_train, outer_test = split
        selected = null_importance_selection(outer_train)        # FOLD-AWARE
        params   = saved Optuna params  |  re-tune on outer_train (NESTED_RETUNE)
        base_oof = inner CV on outer_train  ->  champion + threshold (inner only)
        refit on full outer_train, predict outer_test, score at locked threshold
      report mean +/- std F1 / AUC across outer folds

    optimism_gap = pooled_OOF_F1 (champions_summary) - nested_f1_mean

Config via env:
    NESTED_OUTER_SPLITS (default 5), NESTED_REPEATS (default 1),
    NESTED_RETUNE (0/1; re-tune hyperparameters per outer fold, slow),
    OPTUNA_TRIALS / OPTUNA_TIMEOUT (used when NESTED_RETUNE=1).

    PYTHONPATH=. .venv/bin/python train/run_nested_eval.py            # all panels
    PYTHONPATH=. NESTED_RETUNE=1 .venv/bin/python train/run_nested_eval.py CFTR
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import train.finalize_champions as fc
from src import (
    augment,
    config,
    cv,
    data_io,
    feature_selection,
    features,
    stacking,
    tuning,
)

POS = config.POSITIVE_CLASS
OUTER_SPLITS = int(os.environ.get("NESTED_OUTER_SPLITS", "5"))
REPEATS = int(os.environ.get("NESTED_REPEATS", "1"))
RETUNE = os.environ.get("NESTED_RETUNE", "0").strip().lower() in ("1", "true", "yes")
N_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "40"))
TIMEOUT = int(os.environ.get("OPTUNA_TIMEOUT", "150"))


def _balanced_f1(y_true, y_pred) -> float:
    """F1 the locked model+threshold would score if the test set were 50/50.

    Robustness diagnostic for the unknown competition balance: positives and
    negatives are reweighted to equal mass, so the score no longer rides on the
    majority-positive base rate. The threshold stays the (train-prior) locked one
    on purpose — this is the downside if the test prior shifts under us.
    """
    y_true = np.asarray(y_true)
    n_pos = int((y_true == POS).sum())
    n_neg = int((y_true != POS).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    w = np.where(y_true == POS, 0.5 / n_pos, 0.5 / n_neg)
    return float(f1_score(y_true, y_pred, pos_label=POS, sample_weight=w,
                          zero_division=0))


def _outer_splitter(seed):
    if REPEATS > 1:
        return RepeatedStratifiedKFold(
            n_splits=OUTER_SPLITS, n_repeats=REPEATS, random_state=seed)
    return StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)


def _params_for_fold(panel, X_tr, y_tr, cat_cols, aug=None):
    """Per-outer-fold hyperparameters: re-tuned (honest) or saved-global (fast)."""
    if RETUNE:
        out = {}
        for kind in stacking.KINDS:
            best, _, _ = tuning.tune_model(
                kind, X_tr, y_tr, cat_cols, n_trials=N_TRIALS, timeout=TIMEOUT,
                aug=aug)
            out[kind] = best
        return out
    return {kind: stacking._tuned_params(panel, kind) for kind in stacking.KINDS}


def _inner_champion(base_oof, y_tr):
    """Replicate production champion selection on the inner OOF only.

    Returns (strategy, model, threshold, cols) where cols is the stack column
    subset (or None for single/mean).
    """
    probas, strategies = {}, {}
    for kind in base_oof.columns:
        probas[f"single_{kind}"] = base_oof[kind].to_numpy()
        strategies[f"single_{kind}"] = "single"
    probas["mean"] = base_oof.mean(axis=1).to_numpy()
    strategies["mean"] = "mean"
    variant_cols = {}
    for vname, vkinds in stacking.STACK_VARIANTS.items():
        vcols = [c for c in vkinds if c in base_oof.columns]
        m_oof, _, _ = stacking.evaluate_stack(base_oof, y_tr, cols=vcols, n_repeats=1)
        probas[vname] = m_oof
        strategies[vname] = vname
        variant_cols[vname] = vcols

    candidates = fc.score_candidates(y_tr, probas, strategies)
    champ_key, _, _ = fc.pick_champion(candidates)
    champ = candidates[champ_key]
    cols = variant_cols.get(champ["strategy"])
    return champ["strategy"], champ["model"], champ["threshold"], cols


def _predict_test(strategy, model, cols, full_models, base_oof_tr, y_tr, X_te):
    """Locked-strategy probability on the untouched outer-test rows."""
    base_te = {k: m.predict_proba(X_te)[:, 1] for k, m in full_models.items()}
    if strategy == "single":
        return base_te[model]
    if strategy == "mean":
        return np.mean([base_te[k] for k in full_models], axis=0)
    # stacked_*: refit calibrators + meta on inner OOF, apply to test base probas.
    calibrators, meta, order = stacking.fit_final_stack(base_oof_tr, y_tr, cols=cols)
    z = np.column_stack([calibrators[k].predict(base_te[k]) for k in order])
    return meta.predict_proba(z)[:, 1]


def run_panel(panel: str) -> dict:
    df = data_io.load_panel(panel)
    X, y, cat_cols = features.build_features(df)        # label-independent: safe
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)

    # GENEL augmentation pool (non-PAH rows): never present in any PAH outer fold,
    # so adding it to outer-train only is leak-free and the outer-test stays a
    # genuine PAH-only honest estimate of the augmented model.
    aug_X_full, aug_y_full = augment.build_aug_pool(panel, columns=list(X.columns))

    fold_f1, fold_auc, fold_balf1, strategies = [], [], [], []
    print(f"\n=== Nested eval: {panel} (n={len(y)}, outer={OUTER_SPLITS}x{REPEATS}, "
          f"retune={RETUNE}{', +GENEL aug' if aug_X_full is not None else ''}) ===",
          flush=True)
    for i, (tr, te) in enumerate(_outer_splitter(config.SEED).split(X, y)):
        X_tr_full, y_tr = X.iloc[tr], y.iloc[tr]
        X_te_full, y_te = X.iloc[te], y.iloc[te]

        # Fold-aware feature selection: outer-test labels never seen here.
        table = feature_selection.null_importance_selection(X_tr_full, y_tr, cat_cols)
        selected = table.loc[table["keep"], "feature"].tolist()
        sel_cat = [c for c in cat_cols if c in set(selected)]
        X_tr, X_te = X_tr_full[selected], X_te_full[selected]

        aug = ((aug_X_full[selected], aug_y_full)
               if aug_X_full is not None else None)
        params = _params_for_fold(panel, X_tr, y_tr, sel_cat, aug=aug)
        # Inner CV is single-pass (n_repeats=1): the outer folds already give the
        # variance estimate, so repeating the inner loop only burns time.
        base_oof_tr, _, full_models = stacking.generate_base_oof(
            panel, X_tr, y_tr, sel_cat, params_override=params, n_repeats=1, aug=aug)

        strategy, model, threshold, cols = _inner_champion(base_oof_tr, y_tr)
        proba_te = _predict_test(strategy, model, cols, full_models,
                                 base_oof_tr, y_tr, X_te)

        pred_te = (proba_te >= threshold).astype(int)
        f1 = float(f1_score(y_te, pred_te, pos_label=POS, zero_division=0))
        auc = float(roc_auc_score(y_te, proba_te))
        balf1 = _balanced_f1(y_te, pred_te)
        fold_f1.append(f1)
        fold_auc.append(auc)
        fold_balf1.append(balf1)
        strategies.append(strategy if strategy != "single" else f"single:{model}")
        print(f"  outer fold {i + 1:2d}: {strategy:13s} F1={f1:.4f} AUC={auc:.4f} "
              f"balF1={balf1:.4f} @ t={threshold:.3f} (kept {len(selected)} feats)",
              flush=True)

    return {
        "panel": panel,
        "nested_f1_mean": float(np.mean(fold_f1)),
        "nested_f1_std": float(np.std(fold_f1)),
        "nested_auc_mean": float(np.mean(fold_auc)),
        "nested_balanced_f1_mean": float(np.nanmean(fold_balf1)),
        "strategies": ";".join(sorted(set(strategies))),
    }


def _pooled_f1(panel: str):
    path = config.OUTPUT_DIR / "models" / "champions_summary.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path)
    s = s[s["panel"] == panel]
    return float(s["oof_f1"].iloc[0]) if len(s) else None


def main(panels):
    rows = []
    for panel in panels:
        r = run_panel(panel)
        pooled = _pooled_f1(panel)
        r["pooled_oof_f1"] = pooled
        r["optimism_gap"] = (pooled - r["nested_f1_mean"]) if pooled is not None else None
        rows.append(r)

    out_dir = config.OUTPUT_DIR / "nested_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["panel", "nested_f1_mean", "nested_f1_std", "nested_auc_mean",
            "nested_balanced_f1_mean", "pooled_oof_f1", "optimism_gap",
            "strategies"]
    summary = pd.DataFrame(rows)[cols]
    out_path = out_dir / "nested_summary.csv"
    summary.to_csv(out_path, index=False)

    print("\n=== Nested (honest) vs pooled OOF ===")
    for _, r in summary.iterrows():
        gap = f"{r['optimism_gap']:+.4f}" if pd.notna(r["optimism_gap"]) else "  n/a "
        print(f"  {r['panel']:7s} nested F1={r['nested_f1_mean']:.4f}"
              f"±{r['nested_f1_std']:.4f}  AUC={r['nested_auc_mean']:.4f}"
              f"  balF1={r['nested_balanced_f1_mean']:.4f}"
              f"  pooled={r['pooled_oof_f1']}  gap={gap}")
    print("  (balF1 = F1 if the test set were 50/50 at the locked threshold — "
          "robustness to an unknown competition balance)")
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    requested = [p.upper() for p in sys.argv[1:]] or config.PANELS
    main(requested)
