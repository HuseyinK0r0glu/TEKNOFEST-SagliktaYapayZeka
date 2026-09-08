"""
experiment_tracking.py
=======================
Phases 9 & 10 of the plan -- ablation bookkeeping, experiment tracking, and the
final inference-ready packaging.

Three responsibilities
-----------------------
1. **Structured experiment log (10.1).** ``log_experiment`` appends one row to
   ``outputs/experiment_log.csv`` (``config.EXPERIMENT_LOG_PATH``). The schema is
   a superset of the plan's table -- timestamp, git commit, model/feature/encoding
   descriptors, **per-source CV AUC**, F1/MCC, the **chosen global + per-source
   thresholds**, and the one-time **canary AUC**. New columns can appear over time;
   the writer unions them so old rows stay valid.
2. **Feature-group ablation (9.1).** ``feature_group_ablation`` runs the shared
   ``cv_runner`` CV loop over cumulative engineered-feature groups (EK -> +AA ->
   +missingness -> +population -> +consensus -> +interactions -> all), tracking
   **per-source** AUC for each and logging every run -- the evidence base for the
   Phase-9 ablation tables.
3. **Pipeline packaging (10.2).** ``package_pipeline`` writes a manifest that
   points at every artifact test-time inference needs (feature schema, EK
   normaliser, target-encoding maps, calibrators, meta-learner, thresholds),
   verifies they exist, and records the exact ordered inference recipe -- so the
   final model "can be loaded and re-run on the test set identically".

Heavy modelling code (``cv_runner``) is imported lazily inside the ablation
helper, so this module stays import-light.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    CALIBRATORS_DIR,
    EK_DIRECTION_PATH,
    EK_RANK_THRESHOLDS_PATH,
    EXPERIMENT_LOG_PATH,
    FEATURE_SCHEMA_PATH,
    FINAL_PIPELINE_DIR,
    PIPELINE_MANIFEST_PATH,
    SEED,
    SELECTED_FEATURE_LIST_PATH,
    SOURCES,
    STACKING_DIR,
    TE_MAPPINGS_DIR,
    THRESHOLDS_PATH,
)

# Column order the experiment log prefers (extra columns are appended after).
_PREFERRED_COLS = [
    "experiment_id", "timestamp", "git_commit", "model_type", "feature_set",
    "n_features", "encoding", "imbalance_method", "ek_direction_corrected",
    "blosum_versions", "cv_auc_mean", "cv_auc_std",
    "cv_auc_master", "cv_auc_kanser", "cv_auc_pah", "cv_auc_cftr",
    "cv_f1_mean", "cv_mcc_mean",
    "threshold_global", "threshold_master", "threshold_kanser",
    "threshold_pah", "threshold_cftr",
    "canary_auc", "cat_leakage_risk", "pseudo_labels_used", "notes",
]


# =========================================================================== #
# Small helpers
# =========================================================================== #
def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def git_commit_short() -> str:
    """Best-effort short git SHA (empty string outside a repo)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def next_experiment_id(path: str = EXPERIMENT_LOG_PATH) -> str:
    """``exp_NNN`` one above the current max in the log (``exp_001`` if empty)."""
    if not os.path.exists(path):
        return "exp_001"
    try:
        df = pd.read_csv(path)
        ids = df.get("experiment_id", pd.Series(dtype=str)).dropna().astype(str)
        nums = [int(x.split("_")[-1]) for x in ids if x.startswith("exp_") and x.split("_")[-1].isdigit()]
        return f"exp_{(max(nums) + 1) if nums else 1:03d}"
    except Exception:
        return "exp_001"


def _round(v, nd=4):
    return round(float(v), nd) if (v is not None and np.isfinite(float(v))) else v


# =========================================================================== #
# 10.1  Experiment record + structured CSV log
# =========================================================================== #
def build_experiment_record(
    *,
    model_type: str,
    experiment_id: Optional[str] = None,
    feature_set: str = "",
    n_features: Optional[int] = None,
    encoding: str = "target_enc + native_nan",
    imbalance_method: str = "scale_pos_weight",
    cv_summary: Optional[Dict[str, float]] = None,
    per_source_auc: Optional[Dict[str, float]] = None,
    cv_f1_mean: Optional[float] = None,
    cv_mcc_mean: Optional[float] = None,
    thresholds: Optional[object] = None,
    canary_auc: Optional[float] = None,
    ek_direction_corrected: bool = True,
    blosum_versions: str = "45,62,80",
    cat_leakage_risk: str = "low",
    pseudo_labels_used: bool = False,
    notes: str = "",
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Assemble one experiment-log row from CV / stacking / threshold artefacts.

    ``cv_summary`` accepts the dict from ``CVResult.summary()`` (keys
    ``cv_auc_mean/std``, ``cv_f1_mean``, ``cv_mcc_mean``) *or* the stacking
    ``info`` dict (``stacked_oof_auc`` etc.). ``thresholds`` may be a
    ``ThresholdConfig`` or its ``.apply`` dict; the chosen global + per-source
    thresholds are flattened into dedicated columns.
    """
    cv_summary = cv_summary or {}
    rec: Dict[str, object] = {
        "experiment_id": experiment_id or next_experiment_id(),
        "timestamp": now_timestamp(),
        "git_commit": git_commit_short(),
        "model_type": model_type,
        "feature_set": feature_set,
        "n_features": n_features,
        "encoding": encoding,
        "imbalance_method": imbalance_method,
        "ek_direction_corrected": ek_direction_corrected,
        "blosum_versions": blosum_versions,
        "cv_auc_mean": _round(cv_summary.get("cv_auc_mean", cv_summary.get("stacked_oof_auc"))),
        "cv_auc_std": _round(cv_summary.get("cv_auc_std")),
        "cv_f1_mean": _round(cv_f1_mean if cv_f1_mean is not None else cv_summary.get("cv_f1_mean")),
        "cv_mcc_mean": _round(cv_mcc_mean if cv_mcc_mean is not None else cv_summary.get("cv_mcc_mean")),
        "canary_auc": _round(canary_auc),
        "cat_leakage_risk": cat_leakage_risk,
        "pseudo_labels_used": pseudo_labels_used,
        "notes": notes,
    }

    # per-source CV AUC -> cv_auc_<source>
    for s in SOURCES:
        rec[f"cv_auc_{s}"] = _round((per_source_auc or {}).get(s))

    # chosen thresholds -> threshold_<scope>
    apply = None
    if thresholds is not None:
        apply = getattr(thresholds, "apply", thresholds)
    if isinstance(apply, dict):
        rec["threshold_global"] = _round(apply.get("global"))
        for s in SOURCES:
            rec[f"threshold_{s}"] = _round(apply.get(s))

    if extra:
        rec.update(extra)
    return rec


def log_experiment(record: Dict[str, object], path: str = EXPERIMENT_LOG_PATH,
                   verbose: bool = True) -> pd.DataFrame:
    """Append ``record`` to the CSV log (union columns), return the full table.

    Built from a list of record dicts (not ``pd.concat``) so growing the schema
    over time naturally unions columns without dtype-coercion warnings.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    records.append(dict(record))
    df = pd.DataFrame(records)

    ordered = [c for c in _PREFERRED_COLS if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]
    df.to_csv(path, index=False)
    if verbose:
        print(f"[track] logged {record.get('experiment_id')} "
              f"({record.get('model_type')}) -> {path} | rows={len(df)}")
    return df


# =========================================================================== #
# 9.1  Feature-group definitions + cumulative ablation sets
# =========================================================================== #
# Prefix / exact-name rules mapping engineered columns to plan feature groups.
_GROUP_PREFIXES: Dict[str, tuple] = {
    "ek": ("EK_", "ek_"),
    "aa": ("blosum", "grantham", "aa1_", "aa2_", "delta_", "abs_delta_", "aa_"),
    "missing_meta": ("n_missing", "pct_missing", "missingness_tier", "block_",
                     "n_blocks", "misspat_", "has_", "is_skeleton", "is_dense"),
    "pop_freq": ("af_",),
    "consensus": ("damaging_vote", "benign_vote", "n_tools", "unanimous_",
                  "tools_available"),
    "interactions": ("int_", "stop_gain_severity"),
    "source": ("is_master", "is_kanser", "is_pah", "is_cftr", "source_"),
    "raw_al": ("AL_", "CAT_"),
}
_GROUP_EXACT: Dict[str, set] = {
    "aa": {"max_abs_delta", "group_change", "is_same_group", "charge_direction_change",
           "is_stop_gain", "is_stop_loss", "is_synonymous", "aa1_is_nonstandard",
           "aa2_is_nonstandard", "transition_compatible"},
    "pop_freq": {"is_rare", "is_very_rare", "is_common"},
    "consensus": {"conflicting_predictions"},
}


def standard_feature_groups(all_cols: Sequence[str]) -> Dict[str, List[str]]:
    """Partition ``all_cols`` into the plan's engineered feature groups (9.1)."""
    groups: Dict[str, List[str]] = {g: [] for g in _GROUP_PREFIXES}
    groups["other"] = []
    for c in all_cols:
        placed = False
        for g, prefixes in _GROUP_PREFIXES.items():
            if c in _GROUP_EXACT.get(g, set()) or c.startswith(prefixes):
                groups[g].append(c)
                placed = True
                break
        if not placed:
            groups["other"].append(c)
    return {g: cols for g, cols in groups.items() if cols}


def cumulative_ablation_sets(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Cumulative feature sets in the plan's 9.1 order (each adds one group)."""
    order = [
        ("EK_only", ["ek"]),
        ("EK+AA", ["ek", "aa"]),
        ("EK+AA+missing_meta", ["ek", "aa", "missing_meta"]),
        ("EK+AA+missing_meta+pop_freq", ["ek", "aa", "missing_meta", "pop_freq"]),
        ("+consensus", ["ek", "aa", "missing_meta", "pop_freq", "consensus"]),
        ("all_engineered+interactions",
         ["ek", "aa", "missing_meta", "pop_freq", "consensus", "interactions", "source"]),
    ]
    sets: Dict[str, List[str]] = {}
    for name, gnames in order:
        cols: List[str] = []
        for g in gnames:
            cols += groups.get(g, [])
        if cols:
            sets[name] = sorted(set(cols))
    # full set = every available column (engineered + raw + other)
    sets["all_features"] = sorted({c for cols in groups.values() for c in cols})
    return sets


def feature_group_ablation(
    factory,
    X: pd.DataFrame,
    y: Sequence[int],
    source: np.ndarray,
    folds: List[dict],
    *,
    all_cols: Optional[Sequence[str]] = None,
    sets: Optional[Dict[str, List[str]]] = None,
    model_type: str = "xgb_ablation",
    log: bool = True,
    log_path: str = EXPERIMENT_LOG_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the cumulative feature-group ablation (Phase 9.1) and log each run.

    ``factory`` is a zero-arg ``cv_runner`` learner factory (e.g.
    ``make_xgb_learner(y=y, use_gpu=USE_GPU)``). Returns a tidy DataFrame with
    per-set global + per-source OOF AUC; when ``log`` is set, every set is also
    appended to the experiment log with its threshold columns left blank.
    """
    from . import cv_runner as cvr  # lazy: avoids heavy import at module load

    all_cols = list(all_cols) if all_cols is not None else list(X.columns)
    sets = sets or cumulative_ablation_sets(standard_feature_groups(all_cols))

    rows = []
    for name, cols in sets.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        res = cvr.run_cv(factory, X, y, source, folds, name=f"abl_{name}",
                         feature_cols=cols, keep_models=False, verbose=False)
        summ = res.summary()
        ps = res.per_source_oof_auc
        row = {"experiment": name, "n_features": len(cols),
               "cv_auc_mean": _round(summ["cv_auc_mean"]),
               "cv_auc_std": _round(summ["cv_auc_std"]),
               "oof_auc": _round(res.oof_metrics["auc"]),
               "cv_f1_mean": _round(summ["cv_f1_mean"]),
               "cv_mcc_mean": _round(summ["cv_mcc_mean"])}
        for s in SOURCES:
            row[f"auc_{s}"] = _round(ps.get(s))
        rows.append(row)
        if verbose:
            print(f"[ablation] {name:32s} n={len(cols):3d} "
                  f"OOF AUC={row['oof_auc']} per-source="
                  f"{ {s: row.get(f'auc_{s}') for s in SOURCES} }")
        if log:
            log_experiment(build_experiment_record(
                model_type=model_type, feature_set=name, n_features=len(cols),
                cv_summary=summ, per_source_auc=ps,
                notes=f"Phase 9.1 feature-group ablation: {name}"),
                path=log_path, verbose=False)

    return pd.DataFrame(rows)


# =========================================================================== #
# 10.2  Final pipeline packaging (inference-ready)
# =========================================================================== #
def _library_versions() -> Dict[str, str]:
    """Best-effort version snapshot for reproducibility (missing libs omitted)."""
    versions: Dict[str, str] = {}
    for lib in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm",
                "catboost", "shap"):
        try:
            versions[lib] = __import__(lib).__version__
        except Exception:
            pass
    return versions


def _artifact(path: str) -> Dict[str, object]:
    return {"path": path, "exists": bool(os.path.exists(path))}


# Ordered, human-readable recipe for test-set inference (Phase 10.3 checklist).
_INFERENCE_RECIPE = [
    "Load raw test CSV(s); tag `source`; build FeatureSchema (or load feature_schema.json).",
    "engineer_features_stateless(test_df, schema)  # row-level Phase-2 features.",
    "Load EKRankNormalizer (ek_rank_thresholds.json) + per-fold target-encoding maps.",
    "transform_split(test_stateless, fitted_state)  # EK ranks + aggregates + interactions.",
    "encoders.assemble_tree_matrix(...) restricted to selected_features (NaN preserved).",
    "For each base model: average the per-fold predict_proba over the 5 saved folds.",
    "Apply each base model's Isotonic calibrator (calibrators/<model>_isotonic.pkl).",
    "Rank-normalise calibrated columns; build the meta-matrix (+ source flags + tier).",
    "meta_learner.predict_proba -> stacked P(pathogenic).",
    "threshold_optimizer.apply_per_source_thresholds(proba, source, thresholds.json).",
]


@dataclass
class PipelinePackage:
    manifest: Dict[str, object]

    def save(self, manifest_path: str = PIPELINE_MANIFEST_PATH,
             state: Optional[Dict[str, object]] = None) -> str:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)
        if state is not None:
            import joblib
            joblib.dump(state, os.path.join(os.path.dirname(manifest_path),
                                            "pipeline_state.joblib"))
        return manifest_path


def package_pipeline(
    *,
    selected_features: Sequence[str],
    base_order: Sequence[str],
    meta_cols: Optional[Sequence[str]] = None,
    thresholds: Optional[object] = None,
    metrics: Optional[Dict[str, object]] = None,
    out_dir: str = FINAL_PIPELINE_DIR,
    extra_artifacts: Optional[Dict[str, str]] = None,
    save_state: bool = True,
    verbose: bool = True,
) -> PipelinePackage:
    """Bundle the final pipeline state into an inference-ready manifest (10.2).

    Records every artifact test-time scoring depends on (with existence checks),
    the ordered base-model + meta-learner + threshold recipe, the selected feature
    list, key metrics, the RNG seed and the library versions. A small
    ``pipeline_state.joblib`` holds the lightweight python objects for programmatic
    loading. Returns the :class:`PipelinePackage` (already saved).
    """
    os.makedirs(out_dir, exist_ok=True)
    apply_thresholds = getattr(thresholds, "apply", thresholds) if thresholds is not None else None

    artifacts = {
        "feature_schema": _artifact(FEATURE_SCHEMA_PATH),
        "ek_direction_map": _artifact(EK_DIRECTION_PATH),
        "ek_rank_thresholds": _artifact(EK_RANK_THRESHOLDS_PATH),
        "target_encoding_mappings": _artifact(TE_MAPPINGS_DIR),
        "selected_features": _artifact(SELECTED_FEATURE_LIST_PATH),
        "calibrators": _artifact(CALIBRATORS_DIR),
        "meta_learner": _artifact(os.path.join(STACKING_DIR, "meta_learner.pkl")),
        "stacking_config": _artifact(os.path.join(STACKING_DIR, "stacking_config.json")),
        "thresholds": _artifact(THRESHOLDS_PATH),
    }
    if extra_artifacts:
        artifacts.update({k: _artifact(v) for k, v in extra_artifacts.items()})

    missing = [k for k, a in artifacts.items() if not a["exists"]]
    manifest = {
        "created": now_timestamp(),
        "git_commit": git_commit_short(),
        "seed": SEED,
        "library_versions": _library_versions(),
        "n_selected_features": len(selected_features),
        "selected_features": list(selected_features),
        "base_models": list(base_order),
        "meta_cols": list(meta_cols) if meta_cols is not None else None,
        "apply_thresholds": apply_thresholds,
        "metrics": metrics or {},
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "inference_ready": len(missing) == 0,
        "inference_recipe": _INFERENCE_RECIPE,
    }

    pkg = PipelinePackage(manifest=manifest)
    state = {
        "selected_features": list(selected_features),
        "base_order": list(base_order),
        "meta_cols": list(meta_cols) if meta_cols is not None else None,
        "apply_thresholds": apply_thresholds,
        "metrics": metrics or {},
        "seed": SEED,
    } if save_state else None
    pkg.save(os.path.join(out_dir, "pipeline_manifest.json"), state=state)

    if verbose:
        status = "READY" if manifest["inference_ready"] else f"MISSING {missing}"
        print(f"[package] manifest -> {os.path.join(out_dir, 'pipeline_manifest.json')}")
        print(f"[package] {len(selected_features)} features | base={list(base_order)} | {status}")
    return pkg


def load_pipeline_manifest(path: str = PIPELINE_MANIFEST_PATH) -> Dict[str, object]:
    with open(path) as f:
        return json.load(f)


def verify_inference_ready(manifest: Optional[Dict[str, object]] = None,
                           path: str = PIPELINE_MANIFEST_PATH) -> List[str]:
    """Return the list of missing artifacts (empty list == ready to score test)."""
    manifest = manifest or load_pipeline_manifest(path)
    return [k for k, a in manifest.get("artifacts", {}).items() if not a.get("exists")]
