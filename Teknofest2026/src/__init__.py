"""
TEKNOFEST 2026 - Variant Pathogenicity Classifier
==================================================
Source package for the competition pipeline.

Modules
-------
config              : paths, seed, column conventions, encoding strategy, GPU params
aa_tables           : amino-acid reference matrices & physicochemical properties
data_setup          : Phase 0/1 -- load, merge, fingerprint, canary, folds
feature_engineering : Phase 2 -- stateless + fit/transform feature builders
encoders            : Phase 3 -- target encoding, one-hot, native-NaN, calibration
feature_selection   : Phase 5 -- null importance, SHAP importance, correlation prune
cv_runner           : Phase 4 -- GPU/native-NaN learners, CV loop, tier heads, Optuna
stacking            : Phase 6 -- isotonic calibration, rank-norm, meta-learner
threshold_optimizer : Phase 7 -- multi-metric + per-dataset decision thresholds
shap_analysis       : Phase 8 -- global/per-dataset/dependence SHAP + calibration plots
experiment_tracking : Phase 9/10 -- ablation, experiment_log.csv, pipeline packaging
"""

from . import (  # noqa: F401
    config,
    aa_tables,
    data_setup,
    feature_engineering,
    encoders,
    feature_selection,
    cv_runner,
    stacking,
    threshold_optimizer,
    shap_analysis,
    experiment_tracking,
)

__all__ = [
    "config", "aa_tables", "data_setup", "feature_engineering", "encoders",
    "feature_selection", "cv_runner", "stacking",
    "threshold_optimizer", "shap_analysis", "experiment_tracking",
]
