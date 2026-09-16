"""The development-only research pipeline: three runners that each try more
of the search space than the last, plus a shared run logger.

    search.reference  -- the baseline pipeline (single configuration)
    search.improved   -- the baseline plus explicit, hypothesis-driven
                         improvements (still a single configuration)
    search.sweep      -- the cross-product of every option either of the
                         above offers, scored under an overfitting audit
                         (this is what `python PatHUgen.py run` calls)
    search.runlog     -- tees stdout/stderr to a timestamped log file

Unlike `pathugen/`, this package is NOT self-contained and is never meant
to run during the timed competition — it requires the full project (the
training CSVs, hours of compute) and exists purely to discover and audit
the configuration that `pathugen/champion_config.py` later loads.
"""
from __future__ import annotations
