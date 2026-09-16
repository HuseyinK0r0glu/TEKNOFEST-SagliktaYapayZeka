"""Champion selection — One-SE with a PAIRED bootstrap over AUC.

The reference pipeline's rule compares a candidate against the best one's
*marginal* SE; at n=111 that band is so wide the simplest candidate always
wins. Here the paired bootstrap distribution of the (best − candidate)
difference is used instead — a much narrower band, so a genuinely better
ensemble can actually win.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from pathugen.config import SEED
from search.improved.config import COMPLEXITY, PAIRED_BOOTSTRAP
from search.improved.threshold import pick_threshold
from search.improved.views import stack_variants
from search.reference.stacking import evaluate_stack


def paired_one_se(y, probas, n_boot=PAIRED_BOOTSTRAP, seed=SEED):
    """The paired bootstrap distribution of the (best - candidate) AUC difference.

    The old rule compared a candidate against the best one's *marginal* SE;
    the difference's own distribution is much narrower, so this rule can
    actually tell a genuinely better candidate apart.
    Returns: (best_key, {candidate: (mean_diff, se_diff, eligible)}, {candidate: auc}).
    """
    y = np.asarray(y)
    keys = list(probas)
    aucs = {k: float(roc_auc_score(y, probas[k])) for k in keys}
    best = max(aucs, key=aucs.get)

    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy = y[i]
        if yy.min() == yy.max():
            continue
        for k in keys:
            boot[k].append(roc_auc_score(yy, probas[k][i]))

    info = {}
    for k in keys:
        if k == best or not boot[k]:
            info[k] = (0.0, 0.0, True)
            continue
        d = np.asarray(boot[best]) - np.asarray(boot[k])
        mean_d, se_d = float(np.mean(d)), float(np.std(d))
        info[k] = (mean_d, se_d, mean_d <= se_d)   # difference within 1 paired SE
    return best, info, aucs


def select_champion_plus(base_oof, singles, y, specs, policy, n_boot=PAIRED_BOOTSTRAP,
                         n_repeats=None, verbose=False):
    """Builds the candidates (singles + mean + stacks), picks one with paired One-SE."""
    y = np.asarray(y)
    bo = base_oof.reset_index(drop=True)
    probas, strategies, cols_of = {}, {}, {}

    for c in bo.columns:
        probas[f"single_{c}"] = bo[c].to_numpy()
        strategies[f"single_{c}"] = "single"
    probas["mean"] = bo.mean(axis=1).to_numpy()
    strategies["mean"] = "mean"

    for vname, vcols in stack_variants(specs).items():
        cols = [c for c in vcols if c in bo.columns]
        if len(cols) < 2:
            continue
        m_oof, _, _ = evaluate_stack(bo, y, cols=cols, n_repeats=n_repeats)
        probas[vname] = m_oof
        strategies[vname] = vname
        cols_of[vname] = cols

    best, info, aucs = paired_one_se(y, probas, n_boot=n_boot)
    eligible = [k for k in probas if info[k][2]] or [best]
    champ = min(eligible, key=lambda k: (COMPLEXITY[strategies[k]], -aucs[k]))

    if verbose:
        print(f"    {'candidate':<20s} {'AUC':>7s} {'Δ(best-cand)':>13s} {'SE':>7s} {'ok':>5s}")
        for k in sorted(probas, key=lambda x: -aucs[x]):
            d, se, ok = info[k]
            mark = " <-champion" if k == champ else (" (best AUC)" if k == best else "")
            print(f"    {k:<20s} {aucs[k]:7.4f} {d:13.4f} {se:7.4f} "
                  f"{'yes' if ok else 'no':>5s}{mark}")

    proba = probas[champ]
    threshold = pick_threshold(y, proba, policy)
    return {"key": champ, "strategy": strategies[champ],
            "col": champ[len("single_"):] if strategies[champ] == "single" else None,
            "cols": cols_of.get(strategies[champ]), "proba": proba,
            "auc": aucs[champ], "best_key": best, "best_auc": aucs[best],
            "threshold": threshold}
