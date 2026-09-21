"""GENEL -> PAH/CFTR training-only augmentation pool (leak-free: rows NOT in
the target panel, used only in training folds; validation/test always stays
on the target panel's own rows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathugen.config import AUG_SOURCE, ID_COL
from pathugen.data_io import load_panel
from pathugen.features import build_features, get_cat_maps


def build_aug_pool(panel: str, aug_frac: float, columns=None, bin_cols=None):
    """GENEL rows NOT present in `panel`, engineered with `panel`'s cat_maps.

    `aug_frac` is the caller's decision (from the champion configuration) of
    whether this panel uses augmentation at all — this module has no config
    of its own to fall back on. `bin_cols` should be the target panel's own
    `binary_al_columns()` result (from its training data), so the borrowed
    GENEL rows get engineered with the exact same binary/continuous column
    split as the panel's own rows before the two are concatenated — without
    it, the pool's own (possibly different) column split could make columns
    like `n_tools_available` mean something subtly different for augmented
    rows than for the panel's own rows in the same training matrix.
    """
    if aug_frac <= 0.0 or panel == AUG_SOURCE:
        return None, None
    panel_df = load_panel(panel)
    source_df = load_panel(AUG_SOURCE)
    pool = source_df[~source_df[ID_COL].isin(set(panel_df[ID_COL]))].copy()
    if pool.empty:
        return None, None
    cat_maps = get_cat_maps(panel_df)
    X_aug, y_aug, _ = build_features(pool, cat_maps=cat_maps, bin_cols=bin_cols)
    if columns is not None:
        X_aug = X_aug.reindex(columns=list(columns))
    return X_aug.reset_index(drop=True), pd.Series(np.asarray(y_aug)).reset_index(drop=True)
