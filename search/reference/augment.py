"""GENEL -> panel training-only augmentation, gated by AUGMENT_PANELS (unlike
`search.sweep`, which deliberately bypasses this gate to try augmentation on
every panel)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathugen.config import ID_COL
from pathugen.data_io import load_panel
from pathugen.features import build_features, get_cat_maps
from search.reference.config import AUG_SOURCE, AUGMENT_PANELS


def augment_enabled(panel):
    return panel.upper() in AUGMENT_PANELS and panel.upper() != AUG_SOURCE


def build_aug_pool(panel, columns=None):
    """GENEL rows NOT in `panel` (leak-free extra training rows), or (None,None)."""
    if not augment_enabled(panel):
        return None, None
    panel_df = load_panel(panel)
    source_df = load_panel(AUG_SOURCE)
    pool = source_df[~source_df[ID_COL].isin(set(panel_df[ID_COL]))].copy()
    if pool.empty:
        return None, None
    cat_maps = get_cat_maps(panel_df)
    X_aug, y_aug, _ = build_features(pool, cat_maps=cat_maps)
    if columns is not None:
        X_aug = X_aug.reindex(columns=list(columns))
    return X_aug.reset_index(drop=True), pd.Series(y_aug).reset_index(drop=True)
