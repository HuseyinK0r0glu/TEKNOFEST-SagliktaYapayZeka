"""Augmentation pool — bypasses the reference pipeline's AUGMENT_PANELS gate
(so KANSER can be tried too)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathugen.config import AUG_SOURCE, ID_COL
from pathugen.data_io import load_panel
from pathugen.features import build_features, get_cat_maps
from search.improved.pipeline import subsample_aug

_POOL_CACHE: dict[str, tuple] = {}


def full_aug_pool(panel):
    """GENEL rows that are NOT in the target panel (442 engineered features,
    encoded with the panel's own encoding).

    The reference pipeline's `build_aug_pool` only runs for panels listed in
    AUGMENT_PANELS; the sweep wants to try augmentation on KANSER too, so
    that gate is deliberately bypassed here. Leak-safety is preserved
    exactly the same way: only Variant_IDs that are not already in the
    panel are taken.
    """
    panel = panel.upper()
    if panel in _POOL_CACHE:
        return _POOL_CACHE[panel]
    if panel == AUG_SOURCE:
        _POOL_CACHE[panel] = (None, None)
        return _POOL_CACHE[panel]
    panel_df = load_panel(panel)
    source_df = load_panel(AUG_SOURCE)
    pool = source_df[~source_df[ID_COL].isin(set(panel_df[ID_COL]))].copy()
    if pool.empty:
        _POOL_CACHE[panel] = (None, None)
        return _POOL_CACHE[panel]
    X_aug, y_aug, _ = build_features(pool, cat_maps=get_cat_maps(panel_df))
    _POOL_CACHE[panel] = (X_aug.reset_index(drop=True),
                          pd.Series(np.asarray(y_aug)).reset_index(drop=True))
    return _POOL_CACHE[panel]


def aug_for(panel, frac, columns):
    """The augmentation pool at the requested ratio, restricted to the given columns."""
    if frac <= 0.0:
        return None
    X_aug, y_aug = full_aug_pool(panel)
    if X_aug is None:
        return None
    X_aug, y_aug = subsample_aug(X_aug, y_aug, frac)
    return (X_aug.reindex(columns=list(columns)), y_aug)
