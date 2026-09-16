"""The reference pipeline plus explicit, hypothesis-driven improvements
(prior-insensitive threshold policy, paired-bootstrap One-SE, diverse-view
stacking, a shortcut/missingness audit, per-panel feature selection, and a
seed ensemble). Does not modify `search.reference` — imports it as a
library and layers on top, so the two runs are directly comparable.
"""
from __future__ import annotations

import os as _os

# `search.reference.config` reads these env vars at import time, so they
# must be set BEFORE anything in this package first imports it (this
# __init__.py runs before any of this package's submodules do). setdefault
# means an explicit env var from the caller always wins.
_DEFAULTS = {
    "QUICK": "0", "USE_GPU": "0", "RUN_NESTED": "1", "AUGMENT_PANELS": "PAH,CFTR",
    "SELECTION_METHOD": "null", "OPTUNA_TRIALS": "80", "OPTUNA_TIMEOUT": "600",
    "CV_REPEATS": "3", "NESTED_REPEATS": "2", "TEKNOFEST_OUT_DIR": "outputs_improved",
}
for _k, _v in _DEFAULTS.items():
    _os.environ.setdefault(_k, _v)
