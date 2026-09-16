"""The per-panel champion configuration: which features, which XGBoost
hyperparameters, and which decision threshold `train()` uses for each panel.

This is deliberately NOT a Python literal in this codebase. It is data,
produced by the search pipeline (`search/sweep`, i.e. `python PatHUgen.py
run`) and stored as `champion_config.json` at the project root. `train()`
reads it; nothing in this package writes it — writing a new one is the
search's job, and adopting it (copying the search's output over this file)
is a deliberate, reviewable action, exactly like a paper updating its
"final hyperparameters" table after a new experiment.

See `Champion_Config.md` for the current values in human-readable form and
instructions for regenerating this file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pathugen.config import PANELS, ROOT

CONFIG_PATH = Path(os.environ.get(
    "PATHUGEN_CHAMPION_CONFIG", str(ROOT / "champion_config.json")))

_REQUIRED_PANEL_KEYS = (
    "selection_method", "aug_frac", "threshold", "threshold_policy",
    "xgb_params", "features",
)


class ChampionConfigError(RuntimeError):
    """Raised when `champion_config.json` is missing or malformed."""


def load_champion_config(path: Path = CONFIG_PATH) -> dict:
    """Load and validate the champion configuration. Raises
    `ChampionConfigError` with an actionable message if it is missing,
    unreadable, or incomplete — this is the only way `train()` learns what
    to fit, so a clear failure here matters more than a clever fallback."""
    if not path.exists():
        raise ChampionConfigError(
            f"No champion configuration found at {path}. This file is not "
            f"shipped as a hardcoded default — it must come from a search. "
            f"Run `python PatHUgen.py run` to discover one (multi-hour), or "
            f"copy a previous search's `outputs_best/champion_config.json` "
            f"here. See Champion_Config.md.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ChampionConfigError(f"Could not read {path}: {exc}") from exc

    missing_panels = [p for p in PANELS if p not in data]
    if missing_panels:
        raise ChampionConfigError(
            f"{path} is missing panel(s) {missing_panels}. A champion "
            f"configuration must cover all of {PANELS}.")
    for panel in PANELS:
        cfg = data[panel]
        missing_keys = [k for k in _REQUIRED_PANEL_KEYS if k not in cfg]
        if missing_keys:
            raise ChampionConfigError(
                f"{path}: panel {panel} is missing field(s) {missing_keys}.")
    return data


def save_champion_config(data: dict, path: Path = CONFIG_PATH) -> None:
    """Write a champion configuration. Used only by the search pipeline
    (`search/sweep`) — nothing in the delivery package itself calls this."""
    missing_panels = [p for p in PANELS if p not in data]
    if missing_panels:
        raise ChampionConfigError(
            f"Refusing to write an incomplete champion configuration: "
            f"missing panel(s) {missing_panels}.")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
