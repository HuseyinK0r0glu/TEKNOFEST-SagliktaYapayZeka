"""The reference (baseline) pipeline — a single configuration, marginal
one-SE champion selection, F1-tuned decision threshold. This is the exact
pipeline that produced the numbers in the PDR; `search.improved` layers
explicit improvements on top of it without editing it, and `search.sweep`
sweeps the cross-product of both.

Run: `python -c "from search.reference.cli import main; main()"` or, more
conveniently, through the project's documented commands (see HOW_TO_RUN.md).
"""
from __future__ import annotations
