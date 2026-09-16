"""Per-panel champion selection: build the candidate table (singles + mean +
stacks) and apply the one-SE rule on AUC (marginal SE — see
`search.improved.paired_selection` for the paired-bootstrap variant)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from search.reference.config import POS, SELECT_METRIC, SEED
from search.reference.modeling import KINDS, TREE_KINDS, tune_threshold
from search.reference.stacking import evaluate_stack

COMPLEXITY = {"single": 0, "mean": 1, "stacked_trees": 2, "stacked_all": 3}
STACK_VARIANTS = {"stacked_all": KINDS, "stacked_trees": TREE_KINDS}
N_BOOT = 200


def _trivial_f1(y):
    p = float(np.mean(np.asarray(y) == POS))
    return 2 * p / (1 + p) if (1 + p) else 0.0


def _boot_std(y, proba, fn, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    m, vals = len(y), np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        try:
            vals[i] = fn(y[idx], proba[idx])
        except ValueError:
            vals[i] = np.nan
    return float(np.nanstd(vals))


def score_candidates(y, probas, strategies, thresholds=None):
    thresholds = thresholds or {}
    y = np.asarray(y)
    out = {}
    for key, proba in probas.items():
        strat = strategies[key]
        t = thresholds.get(key)
        if t is None:
            t, _ = tune_threshold(y, proba)
        f1m = lambda yy, pp, tt=t: f1_score(yy, (pp >= tt).astype(int), pos_label=POS,
                                            zero_division=0)
        out[key] = {"strategy": strat,
            "model": key[len("single_"):] if strat == "single" else None,
            "proba": proba, "threshold": float(t),
            "f1": float(f1_score(y, (proba >= t).astype(int), pos_label=POS, zero_division=0)),
            "f1_std": _boot_std(y, proba, f1m),
            "auc": float(roc_auc_score(y, proba)),
            "auc_std": _boot_std(y, proba, roc_auc_score),
            "complexity": COMPLEXITY[strat]}
    return out


def pick_champion(candidates, metric=SELECT_METRIC):
    std_key = f"{metric}_std"
    best = max(candidates, key=lambda k: candidates[k][metric])
    floor = candidates[best][metric] - candidates[best][std_key]
    eligible = [k for k, c in candidates.items() if c[metric] >= floor]
    champ = min(eligible, key=lambda k: (candidates[k]["complexity"], -candidates[k][metric]))
    return champ, best


def select_champion(base_oof, singles, y):
    """Build candidate table (singles + mean + stacks) and apply the one-SE rule."""
    probas, strategies, saved_t = {}, {}, {}
    for kind in base_oof.columns:
        probas[f"single_{kind}"] = base_oof[kind].to_numpy()
        strategies[f"single_{kind}"] = "single"
        saved_t[f"single_{kind}"] = singles[kind]["threshold"]
    probas["mean"] = base_oof.mean(axis=1).to_numpy()
    strategies["mean"] = "mean"
    variant_cols = {}
    for vname, vkinds in STACK_VARIANTS.items():
        cols = [c for c in vkinds if c in base_oof.columns]
        m_oof, m_t, _ = evaluate_stack(base_oof, y, cols=cols)
        probas[vname] = m_oof
        strategies[vname] = vname
        saved_t[vname] = m_t
        variant_cols[vname] = cols
    cand = score_candidates(y, probas, strategies, saved_t)
    champ_key, best_key = pick_champion(cand)
    c = cand[champ_key]
    return {"strategy": c["strategy"], "model": c["model"], "threshold": c["threshold"],
            "oof_f1": c["f1"], "f1_std": c["f1_std"], "auc": c["auc"], "auc_std": c["auc_std"],
            "cols": variant_cols.get(c["strategy"]), "candidates": cand,
            "selected_over_best": champ_key != best_key}
