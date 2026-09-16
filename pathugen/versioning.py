"""Model versioning — every `train` run creates a new numbered snapshot
(v1, v2, ...) per panel; previous versions are kept, never overwritten.

Two on-disk layouts are readable:

    VERSIONED — what `train` writes:    <model_dir>/<PANEL>/v3/seed0.joblib
    FLAT      — what `deliver` ships:   <model_dir>/<PANEL>/seed0.joblib

`deliver` drops the version level when it copies the snapshot it used into a
delivery folder, so the package matches the flat layout Section 6 of the
delivery guide expects (the exact version is recorded in the delivery
README instead). That package is the delivered model, so it has to be
loadable on its own — hence resolution falls back to the flat layout when a
panel directory has no v<N> subdirectory:

    python code/PatHUgen.py deliver --input-dir <csvs> --model-dir model
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pathugen.config import MODEL_DIR, PANELS

_VERSION_RE = re.compile(r"v(\d+)")
# Files every snapshot directory holds, versioned or flat — used to tell a
# packaged bundle apart from a panel directory that merely exists.
_BUNDLE_FILES = ("config.json", "seed0.joblib")


def _is_bundle_dir(path: Path) -> bool:
    """True if `path` holds a trained snapshot directly, with no v<N> level."""
    return all((path / name).is_file() for name in _BUNDLE_FILES)


def _bundle_config(panel_dir: Path) -> dict:
    """The snapshot's `config.json`, or `{}` if it is missing or unreadable —
    this is only ever used for display and for recovering the version number,
    never for a decision, so a broken file must not block loading the model."""
    try:
        return json.loads((panel_dir / "config.json").read_text())
    except (json.JSONDecodeError, OSError):
        return {}


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
    most recently trained — snapshot of `panel`. Handles both layouts
    described in the module docstring."""
    panel_dir = model_dir / panel
    versions = _panel_versions(panel, model_dir)

    if not versions:
        if _is_bundle_dir(panel_dir):
            flat_version = int(_bundle_config(panel_dir).get("version", 0))
            if version is not None and version != flat_version:
                raise FileNotFoundError(
                    f"{panel}: {panel_dir} is a packaged, un-versioned model "
                    f"bundle containing v{flat_version}, so --model-version "
                    f"{version} cannot be satisfied from it — a delivery "
                    f"folder ships only the one snapshot that produced its "
                    f"result. Omit --model-version to use it.")
            return panel_dir, flat_version
        raise FileNotFoundError(
            f"No trained model found for {panel} under {model_dir} — expected "
            f"either {panel_dir}/v<N>/ (as `train` writes) or the snapshot "
            f"files directly in {panel_dir} (as `deliver` packages them). "
            f"Run `python PatHUgen.py train` first, or point --model-dir at a "
            f"delivery folder's model/ directory.")

    if version is None:
        version = max(versions)
    elif version not in versions:
        raise FileNotFoundError(
            f"{panel}: no such trained version v{version} under {model_dir} "
            f"(available: {['v' + str(v) for v in versions]}).")
    return panel_dir / f"v{version}", version


def _print_snapshot(label: str, cfg: dict, mark: str = "") -> None:
    print(f"    {label}: trained {cfg.get('trained_at', '?')}  "
          f"({cfg.get('n_train_rows', '?')} rows, threshold "
          f"{cfg.get('threshold', '?')}){mark}")


def list_model_versions(model_dir: Path = MODEL_DIR) -> None:
    """Print every trained snapshot per panel, oldest to newest, marking the
    one `deliver` will use by default (the latest)."""
    print(f"Model dir: {model_dir.resolve()}\n")
    for panel in PANELS:
        versions = _panel_versions(panel, model_dir)
        panel_dir = model_dir / panel
        if not versions:
            if _is_bundle_dir(panel_dir):
                cfg = _bundle_config(panel_dir)
                print(f"  {panel}:")
                _print_snapshot(
                    f"v{cfg.get('version', '?')} (packaged bundle, no version "
                    f"directory)", cfg, "  <- used by `deliver`")
            else:
                print(f"  {panel:<7s} (no trained versions — run `train` first)")
            continue
        latest = max(versions)
        print(f"  {panel}:")
        for v in versions:
            cfg_path = model_dir / panel / f"v{v}" / "config.json"
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            mark = "  <- latest, used by `deliver` unless --model-version is given" if v == latest else ""
            _print_snapshot(f"v{v}", cfg, mark)


def snapshot_label(panel_dir: Path) -> str:
    """Human label for a snapshot directory — "v3" in either layout. The
    directory name carries the version only in the versioned layout, so for a
    packaged (flat) bundle fall back to what its own config.json records."""
    if _VERSION_RE.fullmatch(panel_dir.name):
        return panel_dir.name
    version = _bundle_config(panel_dir).get("version")
    return f"v{version}" if version is not None else panel_dir.name
