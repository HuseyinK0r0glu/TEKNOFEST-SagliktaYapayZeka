"""Candidate pool: both the training-OOF and the test probability for every
candidate, plus the champion-selection rule and threshold-independent
scoring used throughout the sweep."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from pathugen.config import POS, SEED
from search.improved.config import COMPLEXITY
from search.improved.threshold import prior_weights, scan
from search.improved.views import stack_variants
from search.reference.stacking import evaluate_stack, fit_final_stack
from search.sweep.config import BOOT_N


def build_candidates(base_oof, y_tr, full_models, model_view, specs, X_te=None,
                     n_repeats=None):
    """Returns (cand_oof, cand_te, strategies, cols_of).

    cand_oof : candidate -> OOF probability on the training rows (for champion selection)
    cand_te  : candidate -> probability on the test rows (when X_te is given; for scoring)
    Stack candidates have their calibrator+meta fit on the training rows and
    then applied to the test rows.
    """
    bo = base_oof.reset_index(drop=True)
    y_tr = np.asarray(y_tr)
    cand_oof, strategies, cols_of = {}, {}, {}

    for c in bo.columns:
        cand_oof[f"single_{c}"] = bo[c].to_numpy()
        strategies[f"single_{c}"] = "single"
    cand_oof["mean"] = bo.mean(axis=1).to_numpy()
    strategies["mean"] = "mean"
    for vname, vcols in stack_variants(specs).items():
        cols = [c for c in vcols if c in bo.columns]
        if len(cols) < 2:
            continue
        m_oof, _, _ = evaluate_stack(bo, y_tr, cols=cols, n_repeats=n_repeats)
        cand_oof[vname] = m_oof
        strategies[vname] = vname
        cols_of[vname] = cols

    cand_te = {}
    if X_te is not None:
        base_te = {c: m.predict_proba(X_te[model_view[c]])[:, 1]
                   for c, m in full_models.items()}
        for c in bo.columns:
            cand_te[f"single_{c}"] = base_te[c]
        cand_te["mean"] = np.mean([base_te[c] for c in bo.columns], axis=0)
        for vname, cols in cols_of.items():
            calibrators, meta, order = fit_final_stack(bo, y_tr, cols=cols)
            z = np.column_stack([calibrators[c].predict(base_te[c]) for c in order])
            cand_te[vname] = meta.predict_proba(z)[:, 1]
    return cand_oof, cand_te, strategies, cols_of


def bootstrap_aucs(y, cand_oof, n_boot=BOOT_N, seed=SEED):
    """A single round of bootstrapping yields both the marginal and the paired SE."""
    y = np.asarray(y)
    keys = list(cand_oof)
    aucs = {k: float(roc_auc_score(y, cand_oof[k])) for k in keys}
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy = y[i]
        if yy.min() == yy.max():
            continue
        for k in keys:
            boot[k].append(roc_auc_score(yy, cand_oof[k][i]))
    return aucs, {k: np.asarray(v) for k, v in boot.items()}


def pick_champion(aucs, boot, strategies, rule, pool):
    """One-SE rule: `marginal` = the reference pipeline's rule, `paired` =
    the improved pipeline's rule."""
    keys = [k for k in aucs if pool == "all" or strategies[k] == "single"]
    if not keys:
        keys = list(aucs)
    best = max(keys, key=lambda k: aucs[k])

    if rule == "marginal":
        se = float(np.std(boot[best])) if len(boot[best]) else 0.0
        floor = aucs[best] - se
        eligible = [k for k in keys if aucs[k] >= floor]
    else:                                        # paired
        eligible = []
        for k in keys:
            if k == best:
                eligible.append(k)
                continue
            if not len(boot[k]) or not len(boot[best]):
                continue
            d = boot[best] - boot[k]
            if float(np.mean(d)) <= float(np.std(d)):
                eligible.append(k)
    eligible = eligible or [best]
    return min(eligible, key=lambda k: (COMPLEXITY[strategies[k]], -aucs[k])), best


def score_at(y, proba, t):
    """F1, balanced F1 (1:1), MCC and the confusion matrix at a given threshold."""
    y = np.asarray(y)
    f1, mcc = scan(y, proba, np.array([t]))
    f1b, _ = scan(y, proba, np.array([t]), prior_weights(y, 0.5))
    pred = (np.asarray(proba) >= t).astype(int)
    pos = y == POS
    return {"f1": float(f1[0]), "balanced_f1": float(f1b[0]), "mcc": float(mcc[0]),
            "tp": int(((pred == 1) & pos).sum()), "fp": int(((pred == 1) & ~pos).sum()),
            "fn": int(((pred == 0) & pos).sum()), "tn": int(((pred == 0) & ~pos).sum())}
