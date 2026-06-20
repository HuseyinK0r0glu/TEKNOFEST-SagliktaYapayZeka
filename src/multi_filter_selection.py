"""Multi-filter consensus feature selection (Orange-informed).

Produces alternative feature SETS that the holdout harness scores against the
current null-importance selection on held-out data. The harness — not this
module — decides whether any variant is worth shipping.

Three variants (all start from the validated null-importance keep-set so they
can only differ from the baseline in controlled ways):

  null          The current null-importance set. Baseline; never modified here.

  union         null  PLUS  features that agree across filters:
                  rank top-K in ReliefF  AND  top-K in (Info Gain OR ANOVA),
                  AND have ReliefF >= 0.
                The ReliefF >= 0 guard is the lesson from the KANSER experiment:
                adding the highest-IG features blindly REGRESSED held-out F1
                because they were locally anti-informative. Requiring positive
                ReliefF only adds features that genuinely help nearest-neighbour
                separation. Tests: "did null-importance miss useful features?"

  relief_pruned null  MINUS  features whose ReliefF is clearly negative
                (<= -relief_drop). The GENEL t-SNE showed many AL columns are
                locally misleading; this asks whether dropping them helps.
                Guarded: never prunes a feature that is also top-K by Info Gain
                (a strong global signal the trees may combine non-linearly).

Categorical code columns (CAT_*, AA_*_code) are never added or pruned — they are
carried through unchanged so the train/predict schema and the cat_cols list
stay consistent with features.build_features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, feature_filters, feature_selection

# Variants this module knows how to build, in a stable display order.
VARIANTS = ("null", "union", "relief_pruned")

_CAT_LIKE = set(config.CAT_COLS) | {"AA_1_code", "AA_2_code"}


def _topk_set(series: pd.Series, k: int) -> set:
    """Names of the k highest-scoring features."""
    return set(series.sort_values(ascending=False).head(k).index)


def build_variants(X, y, cat_cols, *, null_selected=None, seed=config.SEED,
                   add_k=25, prune_relief_drop=0.0, verbose=True):
    """Return ({variant: feature_list}, filter_table).

    X, y         : TRAIN-split features / labels (label-dependent — keep it train-only).
    null_selected: precomputed null-importance keep-list; if None it is computed.
    add_k        : top-K cutoff used for the union consensus (per filter).
    prune_relief_drop: prune features with ReliefF <= -prune_relief_drop in the
        relief_pruned variant (0.0 = prune anything strictly negative).
    """
    cat_set = set(cat_cols)

    # --- Baseline: null-importance keep-set (the current shipped method) ---
    if null_selected is None:
        ni = feature_selection.null_importance_selection(X, y, cat_cols)
        null_selected = ni.loc[ni["keep"], "feature"].tolist()
    null_set = list(dict.fromkeys(null_selected))  # de-dup, preserve order

    # --- Orange-style filters (Info Gain, ANOVA, ReliefF) on the train split ---
    table = feature_filters.all_filters(X, y, seed=seed)
    ig = table.set_index("feature")["info_gain"]
    fa = table.set_index("feature")["anova_f"]
    rf = table.set_index("feature")["relieff"]

    top_ig = _topk_set(ig, add_k)
    top_fa = _topk_set(fa, add_k)
    top_rf = _topk_set(rf, add_k)

    # --- union: consensus ADD, ReliefF-guarded ---
    null_lookup = set(null_set)
    consensus_add = []
    for feat in table["feature"]:
        if feat in null_lookup or feat in cat_set:
            continue
        ranks_in_relief = feat in top_rf
        ranks_in_global = feat in top_ig or feat in top_fa
        relief_ok = rf.get(feat, 0.0) >= 0.0
        if ranks_in_relief and ranks_in_global and relief_ok:
            consensus_add.append(feat)
    union_set = null_set + consensus_add

    # --- relief_pruned: drop locally-misleading features (guarded) ---
    pruned, pruned_out = [], []
    for feat in null_set:
        if feat in cat_set:
            pruned.append(feat)
            continue
        relief_bad = rf.get(feat, 0.0) <= -abs(prune_relief_drop) and rf.get(feat, 0.0) < 0.0
        protected = feat in top_ig                 # strong global signal: keep
        if relief_bad and not protected:
            pruned_out.append(feat)
        else:
            pruned.append(feat)

    variants = {
        "null": null_set,
        "union": union_set,
        "relief_pruned": pruned,
    }

    if verbose:
        print(f"    [filters] null-importance kept {len(null_set)} features")
        print(f"    [union]   +{len(consensus_add)} consensus features "
              f"(top-{add_k} ReliefF AND IG/ANOVA, ReliefF>=0): "
              f"{consensus_add[:8]}{' ...' if len(consensus_add) > 8 else ''}")
        print(f"    [pruned]  -{len(pruned_out)} locally-misleading features "
              f"(ReliefF<0, not top-{add_k} IG): "
              f"{pruned_out[:8]}{' ...' if len(pruned_out) > 8 else ''}")

    return variants, table
