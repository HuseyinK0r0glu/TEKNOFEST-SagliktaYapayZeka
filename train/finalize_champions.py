"""Pick and persist the champion strategy per panel from saved OOF.

Reads outputs/stacking/{panel}_oof.csv (base-model + meta OOF + y), scores each
candidate (single base models, the mean ensemble, each stack variant), and writes
outputs/models/{panel}/champion.json describing the winner + its threshold. No
model retraining — purely a selection over existing OOF.

Champion selection uses a one-standard-error rule: instead of blindly taking the
single best OOF F1 (which, picked from ~8 candidates on the same data, is inflated
by luck), we estimate each candidate's F1 uncertainty by bootstrapping the OOF
rows and keep the SIMPLEST candidate whose F1 is within 1 SE of the best. A stack
must beat a single model by more than noise to be chosen.

Thresholds: when run_stacking saved robust (THRESHOLD_AGG-aware) thresholds, they
are reused; otherwise each candidate's threshold is tuned on the pooled OOF.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src import config, cv

POS = config.POSITIVE_CLASS

# Lower = simpler/preferred when the selection metric is within one SE of the best.
COMPLEXITY = {"single": 0, "mean": 1, "stacked_trees": 2, "stacked_all": 3}
N_BOOT = 200

# SELECT_METRIC: the metric the one-SE rule selects on. Default "auc" because the
# competition test-set balance is unknown — AUC ranks models independently of the
# (possibly shifting) class prior, whereas F1 is tied to the train base rate.
SELECT_METRIC = os.environ.get("SELECT_METRIC", "auc").strip().lower()
if SELECT_METRIC not in ("auc", "f1"):
    SELECT_METRIC = "auc"


def _variant_order(panel: str, variant: str, fallback: list[str]) -> list[str]:
    """Base-column order the variant's meta-learner was fit on."""
    path = config.OUTPUT_DIR / "models" / panel / f"{variant}_order.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return fallback


def _saved_thresholds(panel: str) -> dict:
    path = config.OUTPUT_DIR / "stacking" / f"{panel}_thresholds.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _f1_at(y, proba, t) -> float:
    return float(f1_score(y, (proba >= t).astype(int), pos_label=POS, zero_division=0))


def _bootstrap_std(y, proba, metric_fn, n=N_BOOT, seed=config.SEED) -> float:
    """SE of an OOF metric via row resampling (how much luck is in this score).

    metric_fn(y_resampled, proba_resampled) -> float. Resamples that end up with
    a single class are skipped (AUC is undefined there).
    """
    rng = np.random.default_rng(seed)
    m = len(y)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        try:
            vals[i] = metric_fn(y[idx], proba[idx])
        except ValueError:
            vals[i] = np.nan
    return float(np.nanstd(vals))


def _trivial_f1(y) -> float:
    """F1 of the trivial 'predict all Pathogenic' baseline = 2p/(1+p)."""
    p = float(np.mean(y))
    return 2 * p / (1 + p) if (1 + p) else 0.0


def score_candidates(y, probas_by_key, strategies_by_key, thresholds=None):
    """Build the candidate table from {key: proba} + {key: strategy}.

    thresholds: optional {key: t}; missing keys are tuned on the pooled OOF.
    Returns {key: {strategy, model, proba, threshold, f1, f1_std, complexity}}.
    Shared by finalize() and the nested-eval harness so both use the identical
    one-SE selection rule.
    """
    thresholds = thresholds or {}
    y = np.asarray(y)  # positional bootstrap indexing below needs an ndarray
    candidates = {}
    for key, proba in probas_by_key.items():
        strategy = strategies_by_key[key]
        t = thresholds.get(key)
        if t is None:
            t, _ = cv.tune_threshold(y, proba)
        def _f1_metric(yy, pp, tt=t):
            return f1_score(yy, (pp >= tt).astype(int), pos_label=POS, zero_division=0)
        candidates[key] = {
            "strategy": strategy,
            "model": key[len("single_"):] if strategy == "single" else None,
            "proba": proba,
            "threshold": float(t),
            "f1": _f1_at(y, proba, t),
            "f1_std": _bootstrap_std(y, proba, _f1_metric),
            "auc": float(roc_auc_score(y, proba)),
            "auc_std": _bootstrap_std(y, proba, roc_auc_score),
            "complexity": COMPLEXITY[strategy],
        }
    return candidates


def pick_champion(candidates, metric=None):
    """One-SE rule: among candidates within 1 SE of the best `metric`, keep the
    simplest (tie-break on higher `metric`). Returns (champ_key, best_key,
    runner_key). `metric` defaults to SELECT_METRIC ("auc").
    """
    metric = metric or SELECT_METRIC
    std_key = f"{metric}_std"
    best_key = max(candidates, key=lambda k: candidates[k][metric])
    floor = candidates[best_key][metric] - candidates[best_key][std_key]
    eligible = [k for k, c in candidates.items() if c[metric] >= floor]
    champ_key = min(eligible, key=lambda k: (candidates[k]["complexity"], -candidates[k][metric]))
    others = [k for k in candidates if k != champ_key]
    runner_key = max(others, key=lambda k: candidates[k][metric]) if others else None
    return champ_key, best_key, runner_key


def finalize(panel: str) -> dict:
    oof_path = config.OUTPUT_DIR / "stacking" / f"{panel}_oof.csv"
    oof = pd.read_csv(oof_path)
    y = oof["y"].to_numpy()
    meta_cols = [c for c in oof.columns if c.startswith("meta")]
    base_cols = [c for c in oof.columns if c != "y" and c not in meta_cols]
    saved_t = _saved_thresholds(panel)

    # Candidate set: each single base, the mean ensemble, every stack variant.
    probas, strategies = {}, {}
    for col in base_cols:
        probas[f"single_{col}"] = oof[col].to_numpy()
        strategies[f"single_{col}"] = "single"
    probas["mean"] = oof[base_cols].mean(axis=1).to_numpy()
    strategies["mean"] = "mean"
    for mc in meta_cols:
        strategy = mc[len("meta_"):]  # stacked_all / stacked_trees
        probas[strategy] = oof[mc].to_numpy()
        strategies[strategy] = strategy

    candidates = score_candidates(y, probas, strategies, saved_t)
    champ_key, best_key, runner_key = pick_champion(candidates)
    champ = candidates[champ_key]

    strategy, model = champ["strategy"], champ["model"]
    if strategy.startswith("stacked"):
        base_order = _variant_order(panel, strategy, base_cols)
    else:
        base_order = base_cols
    champion = {
        "panel": panel, "strategy": strategy, "model": model,
        "threshold": champ["threshold"], "oof_f1": champ["f1"],
        "f1_std": champ["f1_std"],
        "auc": champ["auc"], "auc_std": champ["auc_std"],
        "select_metric": SELECT_METRIC,
        "trivial_f1": _trivial_f1(y),
        "lift": champ["f1"] - _trivial_f1(y),
        "base_order": base_order,
        "runner_up": (candidates[runner_key]["strategy"] if runner_key else None),
        "runner_up_f1": (candidates[runner_key]["f1"] if runner_key else None),
        "runner_up_auc": (candidates[runner_key]["auc"] if runner_key else None),
        "selected_over_best": best_key != champ_key,
    }
    out_path = config.OUTPUT_DIR / "models" / panel / "champion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(champion, indent=2), encoding="utf-8")
    label = model if strategy == "single" else strategy
    note = f"  (simpler pick over best {SELECT_METRIC})" if champion["selected_over_best"] else ""
    print(f"  {panel:7s} champion = {label:13s} AUC={champ['auc']:.4f}±{champ['auc_std']:.4f} "
          f"| F1={champ['f1']:.4f}±{champ['f1_std']:.4f} @ t={champ['threshold']:.3f} "
          f"lift={champion['lift']:+.4f}{note}")
    return champion


def main(panels):
    print(f"=== Champion per panel (one-SE rule on {SELECT_METRIC}) ===")
    rows = [finalize(p) for p in panels]
    cols = ["panel", "strategy", "model", "threshold", "oof_f1", "f1_std",
            "auc", "auc_std", "trivial_f1", "lift", "runner_up",
            "runner_up_f1", "runner_up_auc"]
    summary = pd.DataFrame(rows)[cols]
    out_path = config.OUTPUT_DIR / "models" / "champions_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    requested = [p.upper() for p in sys.argv[1:]] or config.PANELS
    main(requested)
