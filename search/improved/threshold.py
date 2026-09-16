"""Threshold-scanning tools — computes weighted F1/MCC for every threshold
at once, and the prior-insensitive threshold policies."""
from __future__ import annotations

import numpy as np

from pathugen.config import POS, SEED
from search.improved.config import THR_BOOT, THR_GRID


def prior_weights(y, prior):
    """Sample weights that make the data look like a test set whose positive
    rate is `prior`.

    Positives get total weight `prior`, negatives get `1 - prior`; the
    resulting weighted F1 is the F1 you would expect "if the test set had
    this class balance".
    """
    y = np.asarray(y)
    pos = y == POS
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if prior is None or n_pos == 0 or n_neg == 0:
        return None
    return np.where(pos, prior / n_pos, (1.0 - prior) / n_neg)


def scan(y, proba, ts=THR_GRID, w=None):
    """Returns (weighted) F1 and MCC arrays, one value per threshold."""
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    pos = y == POS

    order = np.argsort(-proba)
    p_s, w_s, pos_s = proba[order], w[order], pos[order]
    cum_w = np.cumsum(w_s)
    cum_tp = np.cumsum(w_s * pos_s)

    # number of samples with proba >= t
    k = np.searchsorted(-p_s, -np.asarray(ts, dtype=float), side="right")
    take = np.clip(k - 1, 0, len(p_s) - 1)
    Pw = np.where(k > 0, cum_w[take], 0.0)
    TPw = np.where(k > 0, cum_tp[take], 0.0)

    APw, ANw = w[pos].sum(), w[~pos].sum()
    FPw = Pw - TPw
    FNw = APw - TPw
    TNw = ANw - FPw

    f1 = np.where(Pw + APw > 0, 2.0 * TPw / (Pw + APw), 0.0)
    den = np.sqrt((TPw + FPw) * (TPw + FNw) * (TNw + FPw) * (TNw + FNw))
    mcc = np.where(den > 0, (TPw * TNw - FPw * FNw) / den, 0.0)
    return f1, mcc


def _policy_scores(y, proba, policy, ts=THR_GRID):
    if policy == "f1":
        return scan(y, proba, ts)[0]
    if policy == "balanced_f1":
        return scan(y, proba, ts, prior_weights(y, 0.5))[0]
    if policy == "mcc":
        return scan(y, proba, ts)[1]
    if policy == "minimax":
        f1_train = scan(y, proba, ts)[0]
        f1_bal = scan(y, proba, ts, prior_weights(y, 0.5))[0]
        return np.minimum(f1_train, f1_bal)
    raise ValueError(policy)


def pick_threshold(y, proba, policy, n_boot=THR_BOOT, seed=SEED):
    """The policy's optimal threshold, pinned with a bootstrap median.

    The counterpart to the reference pipeline's fold-median threshold:
    prevents overfitting to a single sample's noise.
    """
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    picks = [float(THR_GRID[int(np.argmax(_policy_scores(y, proba, policy)))])]
    n = len(y)
    for _ in range(max(0, n_boot - 1)):
        i = rng.integers(0, n, n)
        yy = y[i]
        if yy.min() == yy.max():
            continue
        picks.append(float(THR_GRID[int(np.argmax(_policy_scores(yy, proba[i], policy)))]))
    return float(np.median(picks))


def metrics_at(y, proba, t, priors=()):
    """All reported metrics at the given threshold."""
    y = np.asarray(y)
    pred = (np.asarray(proba) >= t).astype(int)
    pos = y == POS
    tp = int(((pred == 1) & pos).sum())
    fp = int(((pred == 1) & ~pos).sum())
    fn = int(((pred == 0) & pos).sum())
    tn = int(((pred == 0) & ~pos).sum())
    f1_arr, mcc_arr = scan(y, proba, np.array([t]))
    f1_t, mcc_t = float(f1_arr[0]), float(mcc_arr[0])
    out = {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": tp / (tp + fp) if tp + fp else 0.0,
           "recall": tp / (tp + fn) if tp + fn else 0.0,
           "f1": f1_t, "mcc": mcc_t}
    for p in priors:
        out[f"f1@{p:g}"] = float(scan(y, proba, np.array([t]), prior_weights(y, p))[0][0])
    return out


def prior_shift_threshold(t_ref, prior_train, prior_target):
    """The threshold moved to a target prior, for calibrated probabilities.

    p' = p*r / (p*r + (1-p)*s),  r = target_prior/train_prior,
    s = (1-target_prior)/(1-train_prior), so the p' >= t_ref threshold
    corresponds to the following threshold on the raw p.
    """
    r = prior_target / prior_train
    s = (1.0 - prior_target) / (1.0 - prior_train)
    return float(t_ref * s / (t_ref * s + (1.0 - t_ref) * r))
