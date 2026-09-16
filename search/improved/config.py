"""Knobs specific to the improved pipeline."""
from __future__ import annotations

import os

import numpy as np

from search.reference.modeling import KINDS


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


THRESHOLD_POLICY = os.environ.get("THRESHOLD_POLICY", "minimax").strip().lower()
PAIRED_BOOTSTRAP = int(os.environ.get("PAIRED_BOOTSTRAP", "400"))
DIVERSE_STACK = _flag("DIVERSE_STACK", "1")
SHORTCUT_AUDIT = _flag("SHORTCUT_AUDIT", "1")
SEED_ENSEMBLE = int(os.environ.get("SEED_ENSEMBLE", "3"))
AUG_FRAC = float(os.environ.get("AUG_FRAC", "1.0"))
COMPARE_TO = os.environ.get("COMPARE_TO", "outputs_final/results_summary.csv")
THR_BOOT = int(os.environ.get("THRESHOLD_BOOTSTRAP", "31"))
REPORT_PRIORS = [float(x) for x in
                 os.environ.get("REPORT_PRIORS", "0.5,0.6,0.74").split(",") if x.strip()]

POLICIES = ("f1", "balanced_f1", "mcc", "minimax")
if THRESHOLD_POLICY not in POLICIES:
    raise SystemExit(f"THRESHOLD_POLICY '{THRESHOLD_POLICY}' is invalid; choose from {POLICIES}")

# Per-panel feature selector (only KANSER beat the noise floor in the held-out A/B).
PANEL_SELECTION = {"KANSER": "hybrid"}

# Candidate complexity — the simplest one always wins a One-SE tie.
COMPLEXITY = {"single": 0, "mean": 1, "stacked_trees": 2, "stacked_all": 3,
              "stacked_diverse": 4}

# (column name, model kind, feature view)
STD_SPECS = tuple((k, k, "full") for k in KINDS)
DIVERSE_SPECS = (("xgboost_rp", "xgboost", "rp"), ("lightgbm_rp", "lightgbm", "rp"))

# Missingness ("annotation completeness") features — for the shortcut audit.
MISS_KEYS = ("n_missing", "pct_missing", "missingness_tier", "is_skeleton", "is_dense",
             "all_ek_missing", "population_block_missing", "genotype_block_missing",
             "n_tools_available", "missingtier_x_ekcount", "ek_mean_x_completeness")

THR_GRID = np.arange(0.05, 0.95 + 1e-9, 0.005)
