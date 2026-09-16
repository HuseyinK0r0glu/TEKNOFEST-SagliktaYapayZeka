"""Per-panel CSV loading.

Panels are loaded independently and are NEVER concatenated/pooled
(see the architecture directive in `config`).
"""
from __future__ import annotations

import pandas as pd

from src import config


def panel_path(panel: str):
    if panel not in config.PANEL_FILES:
        raise KeyError(f"Unknown panel {panel!r}; expected one of {config.PANELS}")
    return config.DATA_DIR / config.PANEL_FILES[panel]


def load_panel(panel: str) -> pd.DataFrame:
    """Load a single panel CSV with its native 353-column schema."""
    df = pd.read_csv(panel_path(panel))
    expected = (
        1
        + len(config.AL_COLS)
        + len(config.CAT_COLS)
        + len(config.EK_COLS)
        + len(config.AA_COLS)
        + 1
    )
    if df.shape[1] != expected:
        raise ValueError(f"{panel}: expected {expected} columns, got {df.shape[1]}")
    return df


def load_all_panels() -> dict[str, pd.DataFrame]:
    """Return {panel_name: DataFrame} for all four panels, kept separate."""
    return {panel: load_panel(panel) for panel in config.PANELS}
