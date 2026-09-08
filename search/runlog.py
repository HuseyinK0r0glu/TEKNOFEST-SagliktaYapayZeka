"""Run logging — tees stdout/stderr to both the screen and a timestamped
.txt file at the same time.

Usage (at the top of a runner's main() function):

    from search import runlog
    runlog.start("reference")

Files produced:
    logs/reference_YYYYMMDD-HHMMSS.txt   -> the full log for this run
    logs/reference_latest.txt            -> symlink to the most recent run

The log directory can be overridden with the TEKNOFEST_LOG_DIR environment
variable. If the script crashes, the traceback is written to the log too and
the footer is marked as an ERROR.
"""
from __future__ import annotations

import atexit
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from search._project_root import PROJECT_ROOT

ROOT = PROJECT_ROOT
LOG_DIR = Path(os.environ.get("TEKNOFEST_LOG_DIR", str(ROOT / "logs")))

# Environment variables shown in the log header.
KNOBS = [
    "TEKNOFEST_DATA_DIR", "TEKNOFEST_OUT_DIR", "QUICK", "USE_GPU", "RUN_NESTED",
    "OPTUNA_TRIALS", "OPTUNA_TIMEOUT", "OPTUNA_OBJECTIVE", "CV_REPEATS",
    "NESTED_REPEATS", "NESTED_SPLITS", "SELECT_METRIC", "SELECTION_METHOD",
    "AUGMENT_PANELS", "SPW_MULT_HI",
    # search.improved-specific knobs
    "THRESHOLD_POLICY", "REPORT_PRIORS", "PAIRED_BOOTSTRAP", "DIVERSE_STACK",
    "SHORTCUT_AUDIT", "SEED_ENSEMBLE", "AUG_FRAC", "COMPARE_TO",
]
_LIBS = ["numpy", "pandas", "sklearn", "xgboost", "lightgbm", "catboost", "optuna",
         "shap", "joblib"]

_state: dict[str, object] = {}
_BAR = "=" * 78
_SEP = "-" * 78


class _Tee:
    """Writes a stream to both its original target and the log file
    (flushing line by line)."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._handle.write(data)
        self._handle.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _run(cmd):
    try:
        out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def _git_info():
    sha = _run(["git", "rev-parse", "--short", "HEAD"]) or "?"
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "?"
    dirty = _run(["git", "status", "--porcelain"])
    n = len([l for l in dirty.splitlines() if l.strip()])
    return f"{sha} ({branch}) — {'clean' if n == 0 else f'{n} changed/untracked file(s)'}"


def _hardware():
    cores = os.cpu_count() or 0
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        ram_s = f"{ram:.1f} GB"
    except (ValueError, OSError, AttributeError):
        ram_s = "?"
    gpu = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]) or "none"
    return f"{cores} logical core(s) / {ram_s} RAM", gpu.replace("\n", ", ")


def _libs():
    import importlib
    parts = []
    for m in _LIBS:
        try:
            parts.append(f"{m} {getattr(importlib.import_module(m), '__version__', '?')}")
        except Exception:
            parts.append(f"{m} (not installed)")
    return ", ".join(parts)


def _fmt_duration(seconds):
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{h}h {m}m {s}s" if h else
            f"{m}m {s}s" if m else f"{s}s")


def _footer():
    handle = _state.get("handle")
    if handle is None or handle.closed:
        return
    start = _state["start"]
    status = _state.get("status", "SUCCESS")
    print("\n" + _BAR)
    print(f" Finished   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" Duration   : {_fmt_duration(datetime.now().timestamp() - start.timestamp())}")
    print(f" Status     : {status}")
    print(f" Log        : {_state['path']}")
    print(_BAR)
    sys.stdout = _state["orig_out"]
    sys.stderr = _state["orig_err"]
    handle.close()


def _excepthook(exc_type, exc, tb):
    _state["status"] = f"ERROR — {exc_type.__name__}: {exc}"
    handle = _state.get("handle")
    if handle is not None and not handle.closed:
        handle.write("\n" + _SEP + "\n TRACEBACK\n" + _SEP + "\n")
        traceback.print_exception(exc_type, exc, tb, file=handle)
        handle.flush()
    traceback.print_exception(exc_type, exc, tb, file=_state.get("orig_err", sys.__stderr__))


def start(name: str, note: str | None = None) -> Path:
    """Starts the log and returns its file path. A second call is a no-op."""
    if _state.get("handle") is not None:
        return _state["path"]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now()
    path = LOG_DIR / f"{name}_{stamp:%Y%m%d-%H%M%S}.txt"
    handle = open(path, "w", encoding="utf-8", buffering=1)

    _state.update(handle=handle, path=path, start=stamp,
                  orig_out=sys.stdout, orig_err=sys.stderr)
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    sys.excepthook = _excepthook
    atexit.register(_footer)

    hw, gpu = _hardware()
    print(_BAR)
    print(" TEKNOFEST 2026 — Variant Pathogenicity Classifier · run log")
    print(_SEP)
    print(f" Script     : {name}")
    print(f" Command    : {' '.join([Path(sys.argv[0]).name] + sys.argv[1:])}")
    print(f" Started    : {stamp:%Y-%m-%d %H:%M:%S}")
    print(f" Log        : {path}")
    if note:
        print(f" Note       : {note}")
    print(_SEP)
    print(f" Python     : {platform.python_version()} ({platform.python_implementation()})"
          f" — {platform.system()} {platform.release()}")
    print(f" Hardware   : {hw}")
    print(f" GPU        : {gpu}")
    print(f" Git        : {_git_info()}")
    print(f" Libraries  : {_libs()}")
    print(_SEP)
    print(" Environment variables:")
    shown = [(k, os.environ[k]) for k in KNOBS if k in os.environ]
    if shown:
        for i in range(0, len(shown), 3):
            print("   " + "   ".join(f"{k}={v}" for k, v in shown[i:i + 3]))
    else:
        print("   (defaults)")
    print(_BAR, flush=True)

    # "latest" shortcut
    latest = LOG_DIR / f"{name}_latest.txt"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(path.name)
    except OSError:
        pass
    return path


def section(title: str) -> None:
    """Prints a visually separated header in the log."""
    print("\n" + _BAR)
    print(f"  {title}")
    print(_BAR, flush=True)
