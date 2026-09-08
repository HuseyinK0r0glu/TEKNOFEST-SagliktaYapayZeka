"""The full cross-product sweep: every combination the reference and
improved pipelines offer, scored under an overfitting audit, with no
cached or inherited hyperparameters anywhere. This is what
`python PatHUgen.py run` calls.
"""
from __future__ import annotations

import os as _os

# `search.reference.config` (and `search.improved`) read these env vars at
# import time, so they must be set BEFORE anything in this package first
# imports them (this __init__.py runs before any of this package's
# submodules do). setdefault means an explicit env var from the caller
# always wins.
_DEFAULTS = {
    "QUICK": "0", "USE_GPU": "0", "RUN_NESTED": "1", "OPTUNA_TRIALS": "80",
    "OPTUNA_TIMEOUT": "600", "CV_REPEATS": "3", "NESTED_REPEATS": "2",
    "AUGMENT_PANELS": "PAH,CFTR",          # the sweep itself already bypasses this gate
    "TEKNOFEST_OUT_DIR": "outputs_best",
}
for _k, _v in _DEFAULTS.items():
    _os.environ.setdefault(_k, _v)
