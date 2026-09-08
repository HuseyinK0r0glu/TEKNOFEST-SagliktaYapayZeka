"""
shap_analysis.py
================
Phase 8 of the plan -- SHAP interpretability + calibration diagnostics, producing
the report-ready visuals for the TEKNOFEST technical write-up.

What it produces (all saved to ``outputs/plots/`` -- ``config.PLOTS_DIR``)
-------------------------------------------------------------------------
* ``shap_summary_global.png``     -- beeswarm of the top features (Phase 8.1)
* ``shap_summary_per_dataset.png``-- per-source mean|SHAP| panels, with PAH & CFTR
                                     highlighted to expose the biology differences
                                     the plan calls out (Phase 8.4)
* ``shap_dependence_top5.png``    -- dependence plots for the 5 strongest features
                                     with automatic interaction colouring (Phase 8.2)
* ``calibration_curves.png``      -- reliability diagram per base model, before vs
                                     after Isotonic Regression (Phase 8.5)
* ``shap_importance_ranking.csv`` -- mean|SHAP| table for the report appendix

Design
------
* **Model-agnostic.** Functions accept already-fitted tree models (XGBoost /
  LightGBM / CatBoost boosters *or* the ``cv_runner`` learner wrappers, whose
  ``.model`` is unwrapped automatically) plus the data. The driver notebook owns
  *which* models to explain (the "best base models").
* **OOF SHAP (leakage-free).** ``collect_oof_shap`` explains each fold model on
  its **own validation rows** and concatenates -- SHAP values that mirror the OOF
  predictions, so the global/per-dataset/dependence plots describe genuinely
  out-of-fold behaviour (the same principle as ``feature_selection`` Phase 5.5).
* **Lazy heavy imports.** ``shap`` / ``matplotlib`` are imported inside the
  functions so importing this module stays cheap on a CPU box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import PLOTS_DIR, SEED, SOURCES


# =========================================================================== #
# SHAP value container + computation
# =========================================================================== #
@dataclass
class ShapBundle:
    """A computed SHAP matrix aligned to the rows it explains.

    ``shap_values`` is ``(n_explained, n_features)`` for the **pathogenic** class;
    ``X`` are the matching feature rows; ``source`` (optional) is the per-row
    sub-dataset tag aligned to the same rows (for per-dataset plots).
    """

    shap_values: np.ndarray
    X: pd.DataFrame
    feature_names: List[str]
    expected_value: float = 0.0
    source: Optional[np.ndarray] = None
    row_index: Optional[np.ndarray] = None


def _unwrap(model):
    """Return the underlying booster from a ``cv_runner`` learner wrapper (or the
    object itself if it is already a fitted estimator)."""
    return getattr(model, "model", model)


def _normalize_shap(sv) -> np.ndarray:
    """Coerce any TreeExplainer output to a 2-D positive-class array.

    Handles: a plain ``(n, p)`` margin array; a ``[neg, pos]`` list; and a 3-D
    ``(n, p, classes)`` array -- always returning the last (pathogenic) class.
    """
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 3:
        sv = sv[..., -1]
    return sv


def _expected_value(explainer) -> float:
    ev = getattr(explainer, "expected_value", 0.0)
    ev = np.asarray(ev).ravel()
    return float(ev[-1]) if ev.size else 0.0


def explain_model(model, X: pd.DataFrame, *, max_samples: Optional[int] = None,
                  seed: int = SEED) -> ShapBundle:
    """SHAP values for one fitted tree model over rows ``X`` (optionally sampled)."""
    import shap

    if max_samples is not None and len(X) > max_samples:
        X = X.sample(max_samples, random_state=seed)
    explainer = shap.TreeExplainer(_unwrap(model))
    sv = _normalize_shap(explainer.shap_values(X))
    return ShapBundle(shap_values=sv, X=X.reset_index(drop=True),
                      feature_names=list(X.columns), expected_value=_expected_value(explainer),
                      row_index=np.asarray(X.index))


def collect_oof_shap(
    models: Sequence,
    X: pd.DataFrame,
    folds: List[dict],
    feature_cols: Sequence[str],
    *,
    source: Optional[Sequence[str]] = None,
    max_samples: int = 4000,
    seed: int = SEED,
) -> ShapBundle:
    """Out-of-fold SHAP: explain each fold model on its validation rows, concatenate.

    ``models[k]`` is the learner fitted on fold ``k`` (``cv_runner.CVResult.models``).
    The result is an OOF-aligned SHAP matrix capped at ``max_samples`` rows (a
    deterministic subsample for tractable plotting).
    """
    import shap

    cols = list(feature_cols)
    Xc = X[cols]
    sv_parts, idx_parts = [], []
    for k, fold in enumerate(folds):
        if k >= len(models):
            break
        va = np.asarray(fold["val"], dtype=int)
        explainer = shap.TreeExplainer(_unwrap(models[k]))
        sv_parts.append(_normalize_shap(explainer.shap_values(Xc.iloc[va])))
        idx_parts.append(va)

    sv = np.vstack(sv_parts)
    idx = np.concatenate(idx_parts)
    if len(idx) > max_samples:                      # deterministic subsample
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(idx), size=max_samples, replace=False)
        sel.sort()
        sv, idx = sv[sel], idx[sel]

    src = np.asarray(source).astype(str)[idx] if source is not None else None
    return ShapBundle(shap_values=sv, X=Xc.iloc[idx].reset_index(drop=True),
                      feature_names=cols, source=src, row_index=idx)


# =========================================================================== #
# Importance ranking
# =========================================================================== #
def mean_abs_importance(bundle: ShapBundle) -> pd.Series:
    """Mean |SHAP| per feature, descending (the report's importance ranking)."""
    imp = np.abs(bundle.shap_values).mean(axis=0)
    return pd.Series(imp, index=bundle.feature_names).sort_values(ascending=False)


def per_source_mean_abs(bundle: ShapBundle, sources: Sequence[str] = SOURCES,
                        ) -> Dict[str, pd.Series]:
    """Mean |SHAP| per feature computed *within each sub-dataset's* rows."""
    assert bundle.source is not None, "ShapBundle has no aligned source array."
    out: Dict[str, pd.Series] = {}
    for s in sources:
        m = bundle.source == s
        if m.sum() == 0:
            continue
        imp = np.abs(bundle.shap_values[m]).mean(axis=0)
        out[s] = pd.Series(imp, index=bundle.feature_names).sort_values(ascending=False)
    return out


# =========================================================================== #
# 8.1  Global SHAP summary (beeswarm)
# =========================================================================== #
def plot_summary_global(bundle: ShapBundle, out_path: str, *, max_display: int = 30) -> str:
    """Beeswarm summary of the top ``max_display`` features (Phase 8.1)."""
    import matplotlib.pyplot as plt
    import shap

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    shap.summary_plot(bundle.shap_values, bundle.X, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle("Global SHAP summary (pathogenic class)", y=1.02, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================================================================== #
# 8.4  Per-dataset SHAP (PAH vs CFTR emphasis)
# =========================================================================== #
def plot_summary_per_dataset(bundle: ShapBundle, out_path: str, *,
                             sources: Sequence[str] = SOURCES, top_n: int = 15) -> str:
    """One mean|SHAP| bar panel per sub-dataset (Phase 8.4).

    PAH and CFTR panels are highlighted because the plan expects their drivers to
    differ (PAH: conservation/EK; CFTR: AA physicochemical disruption). A grouped
    bar chart of the per-source top features makes those differences explicit and
    is far more robust than stuffing beeswarms into subplots.
    """
    import matplotlib.pyplot as plt

    per_src = per_source_mean_abs(bundle, sources)
    present = [s for s in sources if s in per_src]
    if not present:
        raise ValueError("No source slices available for per-dataset SHAP.")

    highlight = {"pah": "#c0392b", "cftr": "#2980b9"}  # draw the eye to PAH/CFTR
    ncols = 2
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, s in zip(axes, present):
        top = per_src[s].head(top_n)[::-1]                 # smallest at bottom
        color = highlight.get(s, "#7f8c8d")
        ax.barh(range(len(top)), top.values, color=color)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=8)
        emph = "  <<" if s in highlight else ""
        ax.set_title(f"{s.upper()} (n={int((bundle.source == s).sum())}){emph}", fontsize=11)
        ax.set_xlabel("mean |SHAP|")
    for ax in axes[len(present):]:
        ax.axis("off")

    fig.suptitle("Per-dataset SHAP importance (PAH vs CFTR drivers differ)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================================================================== #
# 8.2  Dependence plots for the top-5 features
# =========================================================================== #
def plot_dependence_top5(bundle: ShapBundle, out_path: str, *,
                         top_features: Optional[Sequence[str]] = None, n: int = 5) -> str:
    """Dependence plots for the ``n`` strongest features, coloured by the
    automatically-selected strongest interacting feature (Phase 8.2)."""
    import matplotlib.pyplot as plt
    import shap

    feats = list(top_features) if top_features is not None \
        else mean_abs_importance(bundle).head(n).index.tolist()
    feats = feats[:n]

    ncols = min(3, len(feats))
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(feats):
        shap.dependence_plot(feat, bundle.shap_values, bundle.X,
                             interaction_index="auto", ax=axes[i], show=False)
        axes[i].set_title(feat, fontsize=10)
    for ax in axes[len(feats):]:
        ax.axis("off")

    fig.suptitle("SHAP dependence -- top features (interaction-coloured)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================================================================== #
# 8.5  Calibration curves (before / after Isotonic Regression)
# =========================================================================== #
def _isotonic_calibrate(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    return iso.fit_transform(np.asarray(p, dtype=float), np.asarray(y, dtype=float))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    from sklearn.calibration import calibration_curve
    frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return float(np.mean(np.abs(frac - mean)))


def plot_calibration_curves(
    oof_uncalibrated: Dict[str, np.ndarray],
    y: Sequence[int],
    out_path: str,
    *,
    oof_calibrated: Optional[Dict[str, np.ndarray]] = None,
    n_bins: int = 10,
) -> str:
    """Reliability diagram per base model: before (raw) vs after Isotonic (8.5).

    If ``oof_calibrated`` is not supplied, it is produced here by fitting an
    ``IsotonicRegression`` per model on its own OOF -- so the figure always shows
    the pre/post comparison the plan asks for.
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    y = np.asarray(y).astype(int)
    names = list(oof_uncalibrated.keys())
    if oof_calibrated is None:
        oof_calibrated = {n: _isotonic_calibrate(oof_uncalibrated[n], y) for n in names}

    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, name in enumerate(names):
        ax = axes[i]
        pu, pc = np.asarray(oof_uncalibrated[name], float), np.asarray(oof_calibrated[name], float)
        fu, mu = calibration_curve(y, pu, n_bins=n_bins, strategy="quantile")
        fc, mc = calibration_curve(y, pc, n_bins=n_bins, strategy="quantile")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(mu, fu, "o-", color="#e67e22",
                label=f"before (ECE={_ece(y, pu, n_bins):.3f})")
        ax.plot(mc, fc, "s-", color="#27ae60",
                label=f"after  (ECE={_ece(y, pc, n_bins):.3f})")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("mean predicted prob"); ax.set_ylabel("fraction positive")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[len(names):]:
        ax.axis("off")

    fig.suptitle("Calibration curves -- before vs after Isotonic Regression", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================================================================== #
# Orchestrator
# =========================================================================== #
@dataclass
class ShapReport:
    paths: Dict[str, str]
    importance: pd.Series
    per_source_importance: Dict[str, pd.Series]


def run_shap_report(
    bundle: ShapBundle,
    *,
    oof_uncalibrated: Optional[Dict[str, np.ndarray]] = None,
    y: Optional[Sequence[int]] = None,
    oof_calibrated: Optional[Dict[str, np.ndarray]] = None,
    out_dir: str = PLOTS_DIR,
    sources: Sequence[str] = SOURCES,
    max_display: int = 30,
    top_n_per_source: int = 15,
    dependence_n: int = 5,
    n_bins: int = 10,
    save_importance_csv: bool = True,
    verbose: bool = True,
) -> ShapReport:
    """Produce every Phase-8 report visual from a computed ``ShapBundle``.

    The calibration panel is only drawn when ``oof_uncalibrated`` + ``y`` are
    given (it concerns the base-model probabilities, not the SHAP bundle).
    Returns the output paths and the mean|SHAP| importance tables.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    paths["global"] = plot_summary_global(
        bundle, os.path.join(out_dir, "shap_summary_global.png"), max_display=max_display)
    if bundle.source is not None:
        paths["per_dataset"] = plot_summary_per_dataset(
            bundle, os.path.join(out_dir, "shap_summary_per_dataset.png"),
            sources=sources, top_n=top_n_per_source)
    paths["dependence"] = plot_dependence_top5(
        bundle, os.path.join(out_dir, "shap_dependence_top5.png"), n=dependence_n)

    if oof_uncalibrated is not None and y is not None:
        paths["calibration"] = plot_calibration_curves(
            oof_uncalibrated, y, os.path.join(out_dir, "calibration_curves.png"),
            oof_calibrated=oof_calibrated, n_bins=n_bins)

    importance = mean_abs_importance(bundle)
    per_src = per_source_mean_abs(bundle, sources) if bundle.source is not None else {}
    if save_importance_csv:
        csv_path = os.path.join(out_dir, "shap_importance_ranking.csv")
        importance.rename("mean_abs_shap").to_csv(csv_path)
        paths["importance_csv"] = csv_path

    if verbose:
        print(f"[shap] wrote {len(paths)} artefacts -> {out_dir}")
        for k, p in paths.items():
            print(f"  [{k}] {os.path.basename(p)}")
        print("[shap] top-10 features by mean|SHAP|:")
        print(importance.head(10).round(4).to_string())

    return ShapReport(paths=paths, importance=importance, per_source_importance=per_src)
