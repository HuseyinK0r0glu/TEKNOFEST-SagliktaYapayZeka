"""Model versioning — every `train` run creates a new numbered snapshot
(v1, v2, ...) per panel; previous versions are kept, never overwritten."""
from __future__ import annotations

import json
import re
from pathlib import Path

from pathugen.config import MODEL_DIR, PANELS

_VERSION_RE = re.compile(r"v(\d+)")


def _panel_versions(panel: str, model_dir: Path) -> list[int]:
    panel_dir = model_dir / panel
    if not panel_dir.exists():
        return []
    return sorted(int(m.group(1)) for d in panel_dir.iterdir() if d.is_dir()
                  for m in [_VERSION_RE.fullmatch(d.name)] if m)


def next_model_version(panel: str, model_dir: Path = MODEL_DIR) -> int:
    versions = _panel_versions(panel, model_dir)
    return (max(versions) + 1) if versions else 1


def resolve_model_version_dir(panel: str, model_dir: Path = MODEL_DIR,
                              version: int | None = None) -> tuple[Path, int]:
    """Return (path, version_number) for the requested — or, if omitted, the
    most recently trained — snapshot of `panel`."""
    versions = _panel_versions(panel, model_dir)
    if not versions:
        raise FileNotFoundError(
            f"No trained model found for {panel} under {model_dir}. "
            f"Run `python PatHUgen.py train` first.")
    if version is None:
        version = max(versions)
    elif version not in versions:
        raise FileNotFoundError(
            f"{panel}: no such trained version v{version} under {model_dir} "
            f"(available: {['v' + str(v) for v in versions]}).")
    return model_dir / panel / f"v{version}", version


def list_model_versions(model_dir: Path = MODEL_DIR) -> None:
    """Print every trained snapshot per panel, oldest to newest, marking the
    one `deliver` will use by default (the latest)."""
    print(f"Model dir: {model_dir.resolve()}\n")
    for panel in PANELS:
        versions = _panel_versions(panel, model_dir)
        if not versions:
            print(f"  {panel:<7s} (no trained versions — run `train` first)")
            continue
        latest = max(versions)
        print(f"  {panel}:")
        for v in versions:
            cfg_path = model_dir / panel / f"v{v}" / "config.json"
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            mark = "  <- latest, used by `deliver` unless --model-version is given" if v == latest else ""
            print(f"    v{v}: trained {cfg.get('trained_at', '?')}  "
                  f"({cfg.get('n_train_rows', '?')} rows, threshold "
                  f"{cfg.get('threshold', '?')}){mark}")
