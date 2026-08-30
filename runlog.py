# -*- coding: utf-8 -*-
"""Koşu günlüğü — stdout/stderr'i aynı anda ekrana ve zaman damgalı bir .txt
dosyasına yazar.

Kullanım (betiğin main() fonksiyonunun ilk satırlarında):

    import runlog
    runlog.start("run_final")

Üretilen dosyalar:
    logs/run_final_YYYYMMDD-HHMMSS.txt   → bu koşunun tam günlüğü
    logs/run_final_latest.txt            → son koşuya sembolik bağ

Günlük klasörü TEKNOFEST_LOG_DIR ortam değişkeniyle değiştirilebilir.
Betik çökerse traceback de günlüğe yazılır ve altbilgide HATA olarak işaretlenir.
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

ROOT = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("TEKNOFEST_LOG_DIR", str(ROOT / "logs")))

# Günlük başlığında gösterilecek ortam değişkenleri.
KNOBS = [
    "TEKNOFEST_DATA_DIR", "TEKNOFEST_OUT_DIR", "QUICK", "USE_GPU", "RUN_NESTED",
    "OPTUNA_TRIALS", "OPTUNA_TIMEOUT", "OPTUNA_OBJECTIVE", "CV_REPEATS",
    "NESTED_REPEATS", "NESTED_SPLITS", "SELECT_METRIC", "SELECTION_METHOD",
    "AUGMENT_PANELS", "SPW_MULT_HI",
    # run_improved.py'ye özgü knob'lar
    "THRESHOLD_POLICY", "REPORT_PRIORS", "PAIRED_BOOTSTRAP", "DIVERSE_STACK",
    "SHORTCUT_AUDIT", "SEED_ENSEMBLE", "AUG_FRAC", "REUSE_PARAMS", "COMPARE_TO",
]
_LIBS = ["numpy", "pandas", "sklearn", "xgboost", "lightgbm", "catboost", "optuna",
         "shap", "joblib"]

_state: dict[str, object] = {}
_BAR = "=" * 78
_SEP = "-" * 78


class _Tee:
    """Bir akışı hem asıl hedefe hem günlük dosyasına yazar (satır satır flush)."""

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
    return f"{sha} ({branch}) — {'temiz' if n == 0 else f'{n} değişiklik/izlenmeyen dosya'}"


def _hardware():
    cores = os.cpu_count() or 0
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        ram_s = f"{ram:.1f} GB"
    except (ValueError, OSError, AttributeError):
        ram_s = "?"
    gpu = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]) or "yok"
    return f"{cores} mantıksal çekirdek / {ram_s} RAM", gpu.replace("\n", ", ")


def _libs():
    import importlib
    parts = []
    for m in _LIBS:
        try:
            parts.append(f"{m} {getattr(importlib.import_module(m), '__version__', '?')}")
        except Exception:
            parts.append(f"{m} (yok)")
    return ", ".join(parts)


def _fmt_duration(seconds):
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{h} sa {m} dk {s} sn" if h else
            f"{m} dk {s} sn" if m else f"{s} sn")


def _footer():
    handle = _state.get("handle")
    if handle is None or handle.closed:
        return
    start = _state["start"]
    status = _state.get("status", "BAŞARILI")
    print("\n" + _BAR)
    print(f" Bitiş      : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" Süre       : {_fmt_duration(datetime.now().timestamp() - start.timestamp())}")
    print(f" Durum      : {status}")
    print(f" Günlük     : {_state['path']}")
    print(_BAR)
    sys.stdout = _state["orig_out"]
    sys.stderr = _state["orig_err"]
    handle.close()


def _excepthook(exc_type, exc, tb):
    _state["status"] = f"HATA — {exc_type.__name__}: {exc}"
    handle = _state.get("handle")
    if handle is not None and not handle.closed:
        handle.write("\n" + _SEP + "\n TRACEBACK\n" + _SEP + "\n")
        traceback.print_exception(exc_type, exc, tb, file=handle)
        handle.flush()
    traceback.print_exception(exc_type, exc, tb, file=_state.get("orig_err", sys.__stderr__))


def start(name: str, note: str | None = None) -> Path:
    """Günlüğü başlatır ve dosya yolunu döndürür. İki kez çağrılırsa yok sayılır."""
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
    print(" TEKNOFEST 2026 — Varyant Patojenite Sınıflandırıcısı · koşu günlüğü")
    print(_SEP)
    print(f" Betik      : {name}.py")
    print(f" Komut      : {' '.join([Path(sys.argv[0]).name] + sys.argv[1:])}")
    print(f" Başlangıç  : {stamp:%Y-%m-%d %H:%M:%S}")
    print(f" Günlük     : {path}")
    if note:
        print(f" Not        : {note}")
    print(_SEP)
    print(f" Python     : {platform.python_version()} ({platform.python_implementation()})"
          f" — {platform.system()} {platform.release()}")
    print(f" Donanım    : {hw}")
    print(f" GPU        : {gpu}")
    print(f" Git        : {_git_info()}")
    print(f" Kütüphane  : {_libs()}")
    print(_SEP)
    print(" Ortam değişkenleri:")
    shown = [(k, os.environ[k]) for k in KNOBS if k in os.environ]
    if shown:
        for i in range(0, len(shown), 3):
            print("   " + "   ".join(f"{k}={v}" for k, v in shown[i:i + 3]))
    else:
        print("   (varsayılanlar)")
    print(_BAR, flush=True)

    # "latest" kısayolu
    latest = LOG_DIR / f"{name}_latest.txt"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(path.name)
    except OSError:
        pass
    return path


def section(title: str) -> None:
    """Günlükte görsel olarak ayrılmış bir başlık basar."""
    print("\n" + _BAR)
    print(f"  {title}")
    print(_BAR, flush=True)
