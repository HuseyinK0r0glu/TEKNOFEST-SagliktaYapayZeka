"""Cross-panel training augmentation (the PAH-from-GENEL strategy).

The architecture directive keeps the four panels INDEPENDENT for evaluation, and
this module never breaks that: a panel is always scored on its own rows only.
What it adds is an optional pool of EXTRA TRAINING rows borrowed from the GENEL
(MASTER) panel for a small, hard panel — currently PAH.

Why it is safe (no leakage):
  * MASTER contains 255/372 of PAH's variants with 100% label agreement, so the
    overlap is genuine duplication, not contradiction.
  * The pool is restricted to MASTER rows whose Variant_ID is NOT in the target
    panel. Those rows therefore never appear in the panel's CV folds, so adding
    them to a training fold cannot leak a validation/test row.

Why it should help:
  PAH has only 62 benign rows (83% positive); MASTER carries 782 benign. The
  extra benign (and pathogenic) examples give the model a far denser view of the
  benign decision boundary, the exact weakness the strategy doc flagged (low AUC
  0.79, balanced-F1 0.72). The augmentation rows go into TRAINING folds only and
  are reused as-is by the CV harness via `cv.run_cv(..., aug=(X_aug, y_aug))`.
"""
from __future__ import annotations

import os

import pandas as pd

from src import config, data_io, features

# Which target panels borrow rows, and from which source panel.
AUG_SOURCE = "GENEL"  # MASTER CSV
# PAH + CFTR by default: held-out checks show GENEL augmentation lifts PAH F1 and
# sharply lifts CFTR ranking/AUC (robustness to an unknown test balance), while it
# HURTS KANSER (cancer-gene signal is too specific to dilute) — so KANSER and the
# GENEL source itself are excluded. Override via env, e.g. AUGMENT_PANELS="PAH".
AUG_PANELS = tuple(
    p.strip().upper()
    for p in os.environ.get("AUGMENT_PANELS", "PAH,CFTR").split(",")
    if p.strip()
)


def augment_enabled(panel: str) -> bool:
    """True when `panel` should receive GENEL augmentation (env AUGMENT=0 disables)."""
    if os.environ.get("AUGMENT", "1").strip().lower() in ("0", "false", "no"):
        return False
    return panel.upper() in AUG_PANELS and panel.upper() != AUG_SOURCE


def build_aug_pool(panel: str, columns=None):
    """Return (X_aug, y_aug) of GENEL rows not present in `panel`, or (None, None).

    Features are built with the TARGET panel's categorical vocabulary so the codes
    line up with the panel's own matrix, then reindexed to `columns` (the panel's
    selected feature list) so the augmentation matrix is schema-identical to the
    training matrix. Missing columns become NaN, which the tree models consume.
    """
    if not augment_enabled(panel):
        return None, None

    panel_df = data_io.load_panel(panel)
    source_df = data_io.load_panel(AUG_SOURCE)

    panel_ids = set(panel_df[config.ID_COL])
    pool = source_df[~source_df[config.ID_COL].isin(panel_ids)].copy()
    if pool.empty:
        return None, None

    # Encode the pool with the panel's train-time categorical maps for code parity.
    cat_maps = features.get_cat_maps(panel_df)
    X_aug, y_aug, _ = features.build_features(pool, cat_maps=cat_maps)

    if columns is not None:
        X_aug = X_aug.reindex(columns=list(columns))
    return X_aug.reset_index(drop=True), pd.Series(y_aug).reset_index(drop=True)
