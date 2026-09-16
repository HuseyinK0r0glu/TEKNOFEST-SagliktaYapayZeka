"""Orange-style univariate feature filters: Information Gain, ANOVA F, ReliefF.

These reproduce the three rankings the Orange "Rank" widget produces, so the
selection logic can combine the neighbour-based ReliefF signal (which the
null-importance gain test does NOT capture) with the global filters.

Why this exists — the t-SNE / Orange exploration found:
  * Information Gain alone is misleading. The highest-IG features for KANSER
    (al_log_range, al_log_max, al_cont_max) HURT held-out F1 when added blindly,
    because they are redundant with features the model already had.
  * ReliefF is the discriminating signal. It scores a feature by whether knowing
    its value makes a sample's nearest neighbours easier to classify. On GENEL
    most AL columns had NEGATIVE ReliefF (locally anti-informative); on the
    well-separated panels the genuinely useful features had positive ReliefF.

So `union`/`relief_pruned` selection (see multi_filter_selection) use ReliefF as
a GUARD on top of IG/ANOVA agreement, not IG by itself.

All scores are label-dependent and so MUST be computed on the training split
only when used inside a holdout / fold (the callers honour this).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif

from src import config


def impute_for_filters(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute then zero-fill — univariate filters need complete data.

    The model itself keeps the raw NaNs (trees split on missingness); this
    imputation is ONLY for computing the ranking scores, never for training.
    """
    Xi = X.apply(pd.to_numeric, errors="coerce")
    Xi = Xi.fillna(Xi.median(numeric_only=True))
    return Xi.fillna(0.0)


def information_gain(X: pd.DataFrame, y, seed: int = config.SEED) -> pd.Series:
    """Mutual information between each feature and the label (Orange 'Inf. Gain')."""
    Xi = impute_for_filters(X)
    mi = mutual_info_classif(Xi.to_numpy(), np.asarray(y),
                             discrete_features=False, random_state=seed)
    return pd.Series(mi, index=X.columns, name="info_gain")


def anova_f(X: pd.DataFrame, y) -> pd.Series:
    """ANOVA F-statistic per feature (Orange 'ANOVA'). NaN/inf -> 0.

    Constant columns make f_classif emit a divide-by-zero warning and return
    NaN; both are expected here (those columns just score 0), so we silence them.
    """
    import warnings

    Xi = impute_for_filters(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_vals, _ = f_classif(Xi.to_numpy(), np.asarray(y))
    return pd.Series(np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0),
                     index=X.columns, name="anova_f")


def relieff(X: pd.DataFrame, y, n_neighbors: int = 10,
            sample_size: int | None = None, seed: int = config.SEED) -> pd.Series:
    """ReliefF feature weights (Orange 'ReliefF'), binary-class, numeric.

    For each sampled instance we find its k nearest HITS (same class) and k
    nearest MISSES (other class) in standardised feature space, then move each
    feature's weight DOWN by how much it differs across hits (it should be
    stable within a class) and UP by how much it differs across misses (it
    should change across classes). A feature whose value mostly separates the
    classes locally earns a positive weight; one that adds local noise goes
    negative — exactly the GENEL pattern the Orange widget showed.

    Returns a Series of raw weights (sign is what the selection guard uses);
    larger = more locally discriminative.
    """
    Xi = impute_for_filters(X).to_numpy(dtype=float)
    y = np.asarray(y)
    n, n_feat = Xi.shape

    # Standardise so every feature contributes on a comparable scale; constant
    # columns (std 0) collapse to 0 and correctly earn ~0 weight.
    mu = Xi.mean(axis=0)
    sd = Xi.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (Xi - mu) / sd

    rng = np.random.default_rng(seed)
    m = min(sample_size or 400, n)
    sample_idx = rng.choice(n, size=m, replace=False)

    # k bounded by the rarer class so both hit/miss pools always have k members.
    classes, counts = np.unique(y, return_counts=True)
    k = max(1, min(n_neighbors, int(counts.min()) - 1))

    weights = np.zeros(n_feat)
    used = 0
    for i in sample_idx:
        diff = np.abs(Xz - Xz[i])                 # (n, n_feat) per-feature distance
        dist = np.sqrt((diff ** 2).sum(axis=1))   # (n,) overall distance
        dist[i] = np.inf                          # never pick the instance itself

        same = y == y[i]
        same[i] = False
        other = ~same
        if same.sum() < k or other.sum() < k:
            continue                              # too few neighbours to score

        hit = np.flatnonzero(same)[np.argsort(dist[same])[:k]]
        miss = np.flatnonzero(other)[np.argsort(dist[other])[:k]]
        weights += diff[miss].mean(axis=0) - diff[hit].mean(axis=0)
        used += 1

    if used:
        weights /= used
    return pd.Series(weights, index=X.columns, name="relieff")


def all_filters(X: pd.DataFrame, y, seed: int = config.SEED) -> pd.DataFrame:
    """Compute all three rankings, returned as one tidy DataFrame.

    Columns: feature, info_gain, anova_f, relieff, plus a 0-based rank for each
    (rank 0 = best). Sorted by info_gain descending to mirror Orange's default.
    """
    ig = information_gain(X, y, seed=seed)
    fa = anova_f(X, y)
    rf = relieff(X, y, seed=seed)
    table = pd.DataFrame({"info_gain": ig, "anova_f": fa, "relieff": rf})
    table.index.name = "feature"
    table = table.reset_index()
    for col in ("info_gain", "anova_f", "relieff"):
        table[f"rank_{col}"] = table[col].rank(ascending=False, method="min").astype(int) - 1
    return table.sort_values("info_gain", ascending=False).reset_index(drop=True)
