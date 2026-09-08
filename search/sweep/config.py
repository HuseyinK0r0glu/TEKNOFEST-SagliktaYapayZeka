"""Sweep settings, overfitting thresholds, the final selection objective,
and the decision/training grids.
"""
from __future__ import annotations

import os
from itertools import product

from pathugen.config import AUG_SOURCE
from search.improved.config import POLICIES


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


# --- Sweep settings ----------------------------------------------------------
SWEEP_PROFILE = os.environ.get("SWEEP_PROFILE", "full").strip().lower()
DIVERSE_STACK = _flag("DIVERSE_STACK", "1")
BOOT_N = int(os.environ.get("SWEEP_BOOTSTRAP", "300"))
THR_BOOT = int(os.environ.get("THRESHOLD_BOOTSTRAP", "31"))
SEED_ENSEMBLE = int(os.environ.get("SEED_ENSEMBLE", "3"))

# --- Overfitting thresholds --------------------------------------------------
GAP_LIMIT = float(os.environ.get("GAP_LIMIT", "0.040"))
STABILITY_MIN = float(os.environ.get("STABILITY_MIN", "0.50"))
FOLD_STD_WARN = float(os.environ.get("FOLD_STD_WARN", "0.060"))   # warning only

# --- Final selection objective -----------------------------------------------
#   robust             : min(honest F1, honest balanced F1) — test distribution unknown
#   honest_f1          : assumes the test distribution matches training
#   honest_balanced_f1 : assumes a 1:1 test distribution (what the spec implies)
SELECT_OBJECTIVE = os.environ.get("SELECT_OBJECTIVE", "robust").strip().lower()
OBJECTIVES = ("robust", "honest_f1", "honest_balanced_f1")
if SELECT_OBJECTIVE not in OBJECTIVES:
    raise SystemExit(f"SELECT_OBJECTIVE '{SELECT_OBJECTIVE}' is invalid; choose from {OBJECTIVES}")

# --- Cheap (decision-layer) grid ---------------------------------------------
RULES = ("paired", "marginal")
POOLS = ("all", "singles")
DECISION_GRID = [dict(rule=r, pool=p, policy=t)
                 for r, p, t in product(RULES, POOLS, POLICIES)]

# --- Expensive (training) grid ------------------------------------------------
_AUG_BY_PROFILE = {"full": (0.0, 0.5, 1.0), "wide": (0.0, 0.25, 0.5, 0.75, 1.0),
                   "fast": (0.0, 1.0), "decision": (1.0,)}
_SEL_BY_PROFILE = {"full": ("null", "hybrid"), "wide": ("null", "hybrid"),
                   "fast": ("null", "hybrid"), "decision": ("null",)}


def train_grid(panel):
    """(feature selection, augmentation ratio) configurations for a panel."""
    sels = _SEL_BY_PROFILE.get(SWEEP_PROFILE, _SEL_BY_PROFILE["full"])
    augs = _AUG_BY_PROFILE.get(SWEEP_PROFILE, _AUG_BY_PROFILE["full"])
    if panel.upper() == AUG_SOURCE:      # GENEL is the augmentation source, can't be augmented
        augs = (0.0,)
    if SWEEP_PROFILE == "decision":
        augs = (0.0,) if panel.upper() == AUG_SOURCE else (1.0,)
    return [(s, a) for s in sels for a in augs]


def dc_key(dc):
    return f"{dc['rule']}|{dc['pool']}|{dc['policy']}"


def tc_key(sel, aug):
    return f"sel={sel}|aug={aug:g}"
