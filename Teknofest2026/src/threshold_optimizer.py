"""
threshold_optimizer.py
======================
Phase 7 of the plan -- turn the stacked OOF *probabilities* into decision
*thresholds*.

Why this phase exists
---------------------
With a ~2.7:1 class imbalance (and PAH at ~5:1) the optimal decision boundary is
**not** 0.5; the competition metric decides it. This module:

  * **7.2 Multi-metric search** -- sweep a fine grid and, for every threshold,
    record F1, MCC, Youden's J, precision and recall. Report the optimum for each
    of F1 / MCC / Youden's J *separately*, plus their mean as an ensemble
    candidate (the plan's ``t_final = mean(t_f1, t_mcc, t_yj)``).
  * **7.3 Per-dataset thresholds (not optional)** -- a single global threshold
    systematically underperforms on PAH. We compute and store a separate optimal
    threshold for ``master`` / ``kanser`` / ``pah`` / ``cftr`` and an ``apply``
    lookup the inference code uses at scoring time, with the global threshold as
    the fallback for any unseen source.

The whole result is a single JSON-serialisable config saved to
``outputs/models/thresholds.json`` (``config.THRESHOLDS_PATH``), so test-time
inference is a pure lookup -- no re-optimisation, no leakage.

Everything here is pure ``numpy`` / ``scikit-learn`` (no model fitting), so it is
cheap and deterministic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from .config import SOURCES, THRESHOLDS_PATH

# The plan's grid: np.arange(0.05, 0.95, 0.005) -> 180 candidate thresholds.
GRID_START, GRID_STOP, GRID_STEP = 0.05, 0.95, 0.005
DEFAULT_GRID: np.ndarray = np.round(np.arange(GRID_START, GRID_STOP, GRID_STEP), 4)

# The three metrics the plan searches simultaneously (7.2).
SEARCH_METRICS = ("f1", "mcc", "youden_j")
PRIMARY_METRIC = "f1"  # the metric whose per-source optimum becomes `chosen`/`apply`


# =========================================================================== #
# Single-threshold metric profile
# =========================================================================== #
def metric_at_threshold(y_true: Sequence[int], proba: Sequence[float],
                        threshold: float) -> Dict[str, float]:
    """All decision metrics at one threshold (F1, MCC, Youden's J, P, R).

    Youden's J = sensitivity + specificity - 1 = recall(+) + recall(-) - 1.
    ``zero_division=0`` keeps degenerate slices (all-positive predictions etc.)
    from raising; MCC of a constant prediction is defined here as 0.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)

    recall_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(mcc),
        "youden_j": float(recall_pos + recall_neg - 1.0),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_pos),
    }


def threshold_curve(y_true: Sequence[int], proba: Sequence[float],
                    grid: np.ndarray = DEFAULT_GRID) -> pd.DataFrame:
    """Metric profile across the whole grid -> DataFrame indexed by threshold.

    Columns: f1, mcc, youden_j, precision, recall. Useful for plotting the
    metric-vs-threshold curves and for auditing the chosen operating point.
    """
    rows = {t: metric_at_threshold(y_true, proba, t) for t in grid}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "threshold"
    return df


# =========================================================================== #
# 7.2  Multi-metric optimum (F1, MCC, Youden's J + ensemble mean)
# =========================================================================== #
def best_threshold_for_metric(y_true: Sequence[int], proba: Sequence[float],
                              metric: str, grid: np.ndarray = DEFAULT_GRID,
                              ) -> Tuple[float, float, Dict[str, float]]:
    """``(best_threshold, best_score, full_metric_profile)`` for one metric.

    Ties are broken toward the *lower* threshold (favours recall on the
    majority-pathogenic problem), matching a ``>`` improvement test over the grid.
    """
    curve = threshold_curve(y_true, proba, grid)
    if metric not in curve.columns:
        raise ValueError(f"Unknown metric {metric!r}; choose from {list(curve.columns)}")
    best_t = float(curve[metric].idxmax())
    best_score = float(curve.loc[best_t, metric])
    return best_t, best_score, curve.loc[best_t].to_dict()


def multi_metric_optimum(y_true: Sequence[int], proba: Sequence[float],
                         grid: np.ndarray = DEFAULT_GRID) -> Dict[str, dict]:
    """Optimum threshold for F1, MCC and Youden's J *separately*, plus the mean
    of the three as the ensemble candidate (plan 7.2 ``t_final``)."""
    out: Dict[str, dict] = {}
    chosen_ts = []
    for m in SEARCH_METRICS:
        t, score, profile = best_threshold_for_metric(y_true, proba, m, grid)
        out[m] = {"threshold": round(t, 4), "score": round(score, 6), "profile": profile}
        chosen_ts.append(t)
    out["ensemble_mean"] = {"threshold": round(float(np.mean(chosen_ts)), 4),
                            "components": {m: out[m]["threshold"] for m in SEARCH_METRICS}}
    return out


# =========================================================================== #
# 7.3  Per-dataset thresholds (master / kanser / pah / cftr)
# =========================================================================== #
def per_source_thresholds(
    y_true: Sequence[int],
    proba: Sequence[float],
    source: Sequence[str],
    *,
    primary_metric: str = PRIMARY_METRIC,
    grid: np.ndarray = DEFAULT_GRID,
    sources: Sequence[str] = SOURCES,
    global_fallback: Optional[float] = None,
) -> Dict[str, dict]:
    """Compute a separate optimal threshold per sub-dataset.

    For every source we report the F1 / MCC / Youden's J optima (so the report
    can show each), and set ``chosen`` to the ``primary_metric`` optimum. A source
    slice that is single-class (degenerate -- e.g. an all-pathogenic mini-panel)
    cannot be optimised and falls back to ``global_fallback`` with a flag.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    source = np.asarray(source).astype(str)

    out: Dict[str, dict] = {}
    for s in sources:
        mask = source == s
        n = int(mask.sum())
        if n == 0:
            continue
        ys, ps = y_true[mask], proba[mask]
        entry: Dict[str, object] = {"n": n, "pos_rate": round(float(ys.mean()), 4)}
        if len(np.unique(ys)) < 2:
            # Degenerate slice -> cannot optimise; use the global threshold.
            entry.update({"chosen": (round(float(global_fallback), 4)
                                     if global_fallback is not None else 0.5),
                          "degenerate": True})
            out[s] = entry
            continue
        for m in SEARCH_METRICS:
            t, score, _ = best_threshold_for_metric(ys, ps, m, grid)
            entry[m] = {"threshold": round(t, 4), "score": round(score, 6)}
        entry["chosen"] = entry[primary_metric]["threshold"]
        entry["degenerate"] = False
        out[s] = entry
    return out


# =========================================================================== #
# Full config assembly + persistence
# =========================================================================== #
@dataclass
class ThresholdConfig:
    """Structured, JSON-serialisable threshold bundle (saved to thresholds.json)."""

    primary_metric: str
    grid: Dict[str, float]
    global_: Dict[str, dict]
    per_source: Dict[str, dict]
    apply: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "primary_metric": self.primary_metric,
            "grid": self.grid,
            "global": self.global_,
            "per_source": self.per_source,
            "apply": self.apply,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThresholdConfig":
        return cls(primary_metric=d["primary_metric"], grid=d["grid"],
                   global_=d["global"], per_source=d["per_source"], apply=d["apply"])


def build_threshold_config(
    y_true: Sequence[int],
    proba: Sequence[float],
    source: Sequence[str],
    *,
    primary_metric: str = PRIMARY_METRIC,
    grid: np.ndarray = DEFAULT_GRID,
    sources: Sequence[str] = SOURCES,
) -> ThresholdConfig:
    """Assemble the global multi-metric optimum + per-source thresholds + the
    ``apply`` lookup (per-source ``chosen`` with the global ``chosen`` fallback)."""
    global_block = multi_metric_optimum(y_true, proba, grid)
    global_chosen = float(global_block[primary_metric]["threshold"])

    per_src = per_source_thresholds(
        y_true, proba, source, primary_metric=primary_metric, grid=grid,
        sources=sources, global_fallback=global_chosen)

    # Inference-time lookup: source -> threshold, with global as the catch-all.
    apply = {"global": round(global_chosen, 4)}
    for s, entry in per_src.items():
        apply[s] = round(float(entry["chosen"]), 4)

    global_block["chosen"] = round(global_chosen, 4)
    return ThresholdConfig(
        primary_metric=primary_metric,
        grid={"start": GRID_START, "stop": GRID_STOP, "step": GRID_STEP,
              "n_points": int(len(grid))},
        global_=global_block,
        per_source=per_src,
        apply=apply,
    )


def save_thresholds(config: ThresholdConfig, path: str = THRESHOLDS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"[threshold] saved -> {path}")


def load_thresholds(path: str = THRESHOLDS_PATH) -> ThresholdConfig:
    with open(path) as f:
        return ThresholdConfig.from_dict(json.load(f))


# =========================================================================== #
# Application + evaluation
# =========================================================================== #
def apply_per_source_thresholds(
    proba: Sequence[float],
    source: Sequence[str],
    config: ThresholdConfig,
) -> np.ndarray:
    """Vectorised per-source thresholding -> hard 0/1 predictions.

    Each row uses ``config.apply[source]`` if its source is known, else the
    global threshold -- exactly the rule the test-set inference code applies.
    """
    proba = np.asarray(proba, dtype=float)
    source = np.asarray(source).astype(str)
    global_t = float(config.apply["global"])
    thr = np.array([config.apply.get(s, global_t) for s in source], dtype=float)
    return (proba >= thr).astype(int)


def evaluate_thresholding(
    y_true: Sequence[int],
    proba: Sequence[float],
    source: Sequence[str],
    config: ThresholdConfig,
    *,
    sources: Sequence[str] = SOURCES,
) -> pd.DataFrame:
    """Quantify the per-source threshold gain vs a single global threshold.

    Returns a tidy table (global row + per-source rows) with F1/MCC under the
    global-only threshold and under the per-source ``apply`` thresholds -- the
    evidence that per-source tuning helps (especially PAH).
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    source = np.asarray(source).astype(str)
    global_t = float(config.apply["global"])

    y_pred_global = (proba >= global_t).astype(int)
    y_pred_persrc = apply_per_source_thresholds(proba, source, config)

    def _row(name, mask):
        ys = y_true[mask]
        if len(ys) == 0:
            return None
        g, p = y_pred_global[mask], y_pred_persrc[mask]
        return {
            "scope": name, "n": int(mask.sum()),
            "threshold_global": round(global_t, 4),
            "threshold_applied": round(float(config.apply.get(name, global_t)), 4)
            if name != "overall" else global_t,
            "f1_global": round(float(f1_score(ys, g, zero_division=0)), 4),
            "f1_per_source": round(float(f1_score(ys, p, zero_division=0)), 4),
            "mcc_global": round(float(matthews_corrcoef(ys, g)) if len(np.unique(g)) > 1 else 0.0, 4),
            "mcc_per_source": round(float(matthews_corrcoef(ys, p)) if len(np.unique(p)) > 1 else 0.0, 4),
        }

    rows = [_row("overall", np.ones(len(y_true), bool))]
    for s in sources:
        r = _row(s, source == s)
        if r is not None:
            rows.append(r)
    return pd.DataFrame([r for r in rows if r is not None])


# =========================================================================== #
# Orchestrator
# =========================================================================== #
def optimize_thresholds(
    y_true: Sequence[int],
    proba: Sequence[float],
    source: Sequence[str],
    *,
    primary_metric: str = PRIMARY_METRIC,
    grid: np.ndarray = DEFAULT_GRID,
    sources: Sequence[str] = SOURCES,
    save: bool = True,
    path: str = THRESHOLDS_PATH,
    verbose: bool = True,
) -> Tuple[ThresholdConfig, pd.DataFrame]:
    """End-to-end Phase 7 on the stacked OOF predictions.

    Returns ``(config, evaluation_table)`` and (by default) persists
    ``thresholds.json``. ``proba`` should be the **stacked OOF probabilities**
    aligned to the development labels/sources.
    """
    config = build_threshold_config(
        y_true, proba, source, primary_metric=primary_metric, grid=grid, sources=sources)
    evaluation = evaluate_thresholding(y_true, proba, source, config, sources=sources)
    if save:
        save_thresholds(config, path)
    if verbose:
        g = config.global_
        print(f"[threshold] global optima -> F1@{g['f1']['threshold']} "
              f"MCC@{g['mcc']['threshold']} J@{g['youden_j']['threshold']} "
              f"| ensemble-mean {g['ensemble_mean']['threshold']} "
              f"| chosen ({primary_metric}) {g['chosen']}")
        print(f"[threshold] per-source apply thresholds: "
              f"{ {k: v for k, v in config.apply.items()} }")
    return config, evaluation
