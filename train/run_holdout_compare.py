"""Holdout A/B comparison of feature-selection methods (the train/test split).

THE QUESTION THIS ANSWERS: the Orange exploration suggested some features might
be missing from (or harmful in) the current null-importance selection. Does
acting on that actually improve a score on data the model has NEVER seen? Naive
"add the high-IG features" already backfired once on KANSER, so we let a real
held-out test set decide instead of trusting the ranking.

HOW IT STAYS HONEST (leak-free):
  For each panel we cut ONE stratified split:
      train (80%)  -- everything below is fit on these rows only
      test  (20%)  -- untouched until the final score
  Then, for each selection method (null / union / relief_pruned):
      1. select features on TRAIN ONLY            (labels of train only)
      2. tune base models (reuse saved Optuna params) on train
      3. build the calibrated stack, pick the champion strategy + threshold
         on the train out-of-fold predictions  (same one-SE rule as production)
      4. lock it, predict the TEST rows once, score F1 / AUC / precision / recall
  Every method is judged on the SAME held-out rows -> a fair apples-to-apples
  comparison. The "train OOF F1" we also print is the optimistic number; the
  gap to the test F1 is how much that optimism was worth.

This is a DIAGNOSTIC. It does not change the shipped bundle. If a method wins
clearly, adopt it for real by re-running feature selection with that method:
      SELECTION_METHOD=union PYTHONPATH=. .venv/bin/python train/run_feature_selection.py
  (then run_optuna -> run_stacking -> finalize_champions as usual).

Config via env:
    HOLDOUT_TEST_SIZE   (default 0.20)   held-out fraction
    HOLDOUT_METHODS     (default "null,union,relief_pruned")
    HOLDOUT_ADD_K       (default 25)     top-K cutoff for the union consensus
    CV_REPEATS / THRESHOLD_AGG / SELECT_METRIC  -- honoured as in the main pipeline

    PYTHONPATH=. .venv/bin/python train/run_holdout_compare.py          # all panels
    PYTHONPATH=. .venv/bin/python train/run_holdout_compare.py PAH      # one panel
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Reuse the EXACT production champion-selection + locked-prediction logic so the
# only thing that varies between methods is the feature set.
from train.run_nested_eval import _balanced_f1, _inner_champion, _predict_test
from src import (
    config,
    cv,
    data_io,
    feature_filters,
    features,
    multi_filter_selection,
    stacking,
)

POS = config.POSITIVE_CLASS

TEST_SIZE = float(os.environ.get("HOLDOUT_TEST_SIZE", "0.20"))
METHODS = [m.strip() for m in os.environ.get(
    "HOLDOUT_METHODS", "null,union,relief_pruned").split(",") if m.strip()]
ADD_K = int(os.environ.get("HOLDOUT_ADD_K", "25"))


def _trivial_f1(y) -> float:
    p = float(np.mean(np.asarray(y) == POS))
    return 2 * p / (1 + p) if (1 + p) else 0.0


def _params(panel: str) -> dict:
    """Saved per-panel Optuna params for every base model (no per-method retune).

    Reusing one shared parameter set keeps the comparison fair: every selection
    method is given the same model budget, so a win comes from the FEATURES, not
    from one method getting luckier hyperparameters.
    """
    return {kind: stacking._tuned_params(panel, kind) for kind in stacking.KINDS}


def _print_filter_tables(table: pd.DataFrame, n: int = 15) -> None:
    """Echo the Orange 'Rank' widget so the terminal shows what Orange showed."""
    print(f"\n  Orange-style feature ranking on the TRAIN split (top {n}):")
    print("    (Info Gain = global predictiveness; ReliefF = local neighbour "
          "separation; negative ReliefF = locally misleading)")
    show = table.head(n)
    print(f"      {'feature':<22} {'InfoGain':>9} {'ANOVA_F':>9} {'ReliefF':>9}")
    for _, r in show.iterrows():
        print(f"      {r['feature']:<22} {r['info_gain']:>9.4f} "
              f"{r['anova_f']:>9.2f} {r['relieff']:>9.4f}")


def _eval_method(panel, method, selected, sel_cat, X_tr, y_tr, X_te, y_te):
    """Train the full stack on a feature set, score it on the held-out test."""
    Xtr_s = X_tr[selected].reset_index(drop=True)
    Xte_s = X_te[selected].reset_index(drop=True)

    base_oof_tr, _, full_models = stacking.generate_base_oof(
        panel, Xtr_s, y_tr, sel_cat, params_override=_params(panel), n_repeats=1)

    strategy, model, threshold, cols = _inner_champion(base_oof_tr, y_tr)

    # Train (optimistic) F1 at the locked threshold on the inner OOF.
    champ_oof = base_oof_tr.mean(axis=1).to_numpy() if strategy == "mean" else (
        base_oof_tr[model].to_numpy() if strategy == "single" else
        stacking.evaluate_stack(base_oof_tr, y_tr, cols=cols, n_repeats=1)[0])
    train_f1 = float(f1_score(y_tr, (champ_oof >= threshold).astype(int),
                              pos_label=POS, zero_division=0))

    # Honest held-out score.
    proba_te = _predict_test(strategy, model, cols, full_models,
                             base_oof_tr, y_tr, X_te[selected].reset_index(drop=True))
    pred_te = (proba_te >= threshold).astype(int)
    label = model if strategy == "single" else strategy
    return {
        "panel": panel,
        "method": method,
        "n_features": len(selected),
        "strategy": label,
        "threshold": float(threshold),
        "train_oof_f1": train_f1,
        "test_f1": float(f1_score(y_te, pred_te, pos_label=POS, zero_division=0)),
        "test_auc": float(roc_auc_score(y_te, proba_te)),
        "test_precision": float(precision_score(y_te, pred_te, pos_label=POS, zero_division=0)),
        "test_recall": float(recall_score(y_te, pred_te, pos_label=POS, zero_division=0)),
        "test_balanced_f1": _balanced_f1(y_te.to_numpy(), pred_te),
        "optimism_gap": train_f1 - float(f1_score(y_te, pred_te, pos_label=POS, zero_division=0)),
    }


def run_panel(panel: str) -> list[dict]:
    print("\n" + "#" * 72)
    print(f"#  PANEL {panel}")
    print("#" * 72)

    df = data_io.load_panel(panel)
    X, y, cat_cols = features.build_features(df)           # label-independent: safe
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)

    # ── ONE stratified holdout split (the train/test split you asked for) ──
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=config.SEED)
    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = y.iloc[tr_idx].reset_index(drop=True), y.iloc[te_idx].reset_index(drop=True)

    n_tr_pos = int((y_tr == POS).sum())
    n_te_pos = int((y_te == POS).sum())
    print(f"\n  Split: train={len(y_tr)} rows ({n_tr_pos} pathogenic / "
          f"{len(y_tr) - n_tr_pos} benign) | "
          f"test={len(y_te)} rows ({n_te_pos} pathogenic / {len(y_te) - n_te_pos} benign)")
    print(f"  Held-out test trivial F1 (predict-all-pathogenic) = "
          f"{_trivial_f1(y_te):.4f}  <- any real model must clearly beat this")

    # ── Build the feature-set variants on TRAIN ONLY ──
    print("\n  Selecting features on the TRAIN split (each method, leak-free):")
    variants, table = multi_filter_selection.build_variants(
        X_tr, y_tr, cat_cols, add_k=ADD_K, verbose=True)
    _print_filter_tables(table)

    # ── Evaluate each requested method on the SAME held-out test ──
    rows = []
    null_set = set(variants["null"])
    print("\n  Training + scoring each method on the held-out test set:")
    for method in METHODS:
        if method not in variants:
            print(f"    ! unknown method '{method}' — skipping")
            continue
        selected = variants[method]
        sel_cat = [c for c in cat_cols if c in set(selected)]
        same = set(selected) == null_set
        tag = "  (identical to null — no change to evaluate)" if same and method != "null" else ""
        print(f"\n    >> method='{method}': {len(selected)} features{tag}", flush=True)
        rows.append(_eval_method(panel, method, selected, sel_cat,
                                 X_tr, y_tr, X_te, y_te))

    # ── Per-panel verdict ──
    _print_panel_verdict(panel, rows)
    return rows


def _print_panel_verdict(panel: str, rows: list[dict]) -> None:
    if not rows:
        return
    print(f"\n  --- {panel}: held-out test results (honest, apples-to-apples) ---")
    print(f"      {'method':<14} {'#feat':>5} {'strategy':<13} "
          f"{'TEST_F1':>8} {'TEST_AUC':>8} {'prec':>6} {'rec':>6} {'trainF1':>8} {'gap':>7}")
    base = next((r for r in rows if r["method"] == "null"), rows[0])
    for r in rows:
        d = r["test_f1"] - base["test_f1"]
        delta = f"({d:+.4f})" if r["method"] != "null" else "(base)  "
        print(f"      {r['method']:<14} {r['n_features']:>5} {r['strategy']:<13} "
              f"{r['test_f1']:>8.4f} {r['test_auc']:>8.4f} {r['test_precision']:>6.3f} "
              f"{r['test_recall']:>6.3f} {r['train_oof_f1']:>8.4f} {r['optimism_gap']:>+7.4f} {delta}")

    best = max(rows, key=lambda r: r["test_f1"])
    margin = best["test_f1"] - base["test_f1"]
    if best["method"] == "null" or margin <= 0:
        print(f"\n  VERDICT [{panel}]: keep the CURRENT selection (null). No method "
              f"beat it on held-out F1.")
    elif margin < 0.005:
        print(f"\n  VERDICT [{panel}]: '{best['method']}' edged null by only "
              f"{margin:+.4f} — within noise on {sum(1 for _ in rows)} methods. "
              f"Treat as a TIE; prefer the simpler 'null' unless it repeats.")
    else:
        print(f"\n  VERDICT [{panel}]: '{best['method']}' WINS by {margin:+.4f} "
              f"held-out F1 ({base['test_f1']:.4f} -> {best['test_f1']:.4f}). "
              f"Worth shipping: SELECTION_METHOD={best['method']}")


def main(panels):
    print("=" * 72)
    print("  HOLDOUT FEATURE-SELECTION COMPARISON")
    print(f"  test_size={TEST_SIZE}  methods={METHODS}  add_k={ADD_K}  "
          f"seed={config.SEED}")
    print(f"  CV_REPEATS={cv.cv_repeats()}  THRESHOLD_AGG={cv.threshold_agg()}")
    print("=" * 72)

    all_rows = []
    for panel in panels:
        all_rows.extend(run_panel(panel))

    out_dir = config.OUTPUT_DIR / "holdout_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "holdout_compare_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Per-panel recommendation (best held-out F1; ties resolved toward 'null').
    recommended = {}
    for panel in summary["panel"].unique():
        sub = summary[summary["panel"] == panel]
        base = sub[sub["method"] == "null"]["test_f1"]
        base_f1 = float(base.iloc[0]) if len(base) else sub["test_f1"].max()
        best_row = sub.loc[sub["test_f1"].idxmax()]
        win = best_row["test_f1"] - base_f1
        recommended[panel] = best_row["method"] if win > 0.005 else "null"
    (out_dir / "recommended_selection.json").write_text(
        json.dumps(recommended, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("  OVERALL — held-out test F1 by panel and method")
    print("=" * 72)
    pivot = summary.pivot_table(index="panel", columns="method", values="test_f1")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n  Recommended selection per panel: {recommended}")
    print("  (a method is only recommended over 'null' when it wins by > 0.005 "
          "held-out F1 — otherwise keep the simpler current method)")
    print(f"\n[written] {summary_path}")
    print(f"[written] {out_dir / 'recommended_selection.json'}")


if __name__ == "__main__":
    requested = [p.upper() for p in sys.argv[1:]] or config.PANELS
    main(requested)
