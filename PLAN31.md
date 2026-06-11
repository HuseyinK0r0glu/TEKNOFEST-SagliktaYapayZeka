# Teknofest University Variant Pathogenicity Plan

## Summary
- Build four fully independent binary classification pipelines: `MASTER`, `KANSER`, `PAH`, and `CFTR`.
- Each panel gets its own data split, preprocessing fit, model training, hyperparameter tuning, validation report, and saved artifacts.
- `Variant_ID` is only a row identifier inside its own panel; never merge, join, deduplicate, group, pretrain, tune, or validate across panels using IDs or rows.
- Official university task: predict `Patojenik` vs `Benign`; final technical metric is F1 on hidden test labels, with reproducible code/report expected.

## Dataset Analysis
| Panel | Rows | Features | Categorical | Label Balance | Notes |
|---|---:|---:|---:|---|---|
| `MASTER` | 2931 | 351 | 8 | `1`: 2149, `0`: 782 | Actual data is imbalanced despite spec’s balanced target count. |
| `KANSER` | 388 | 351 | 8 | `1`: 268, `0`: 120 | 249 features have >=50% missingness. |
| `PAH` | 372 | 351 | 7 usable | `1`: 310, `0`: 62 | `CAT_6` is entirely missing/constant; drop for this panel. |
| `CFTR` | 111 | 351 | 8 | `1`: 90, `0`: 21 | Very small sample; use conservative CV and avoid complex ensembles unless validated. |

- All CSVs use `Variant_ID`, anonymized feature columns such as `AL_*`, `CAT_*`, `EK_*`, `AA_*`, and binary `Label`.
- `Variant_ID` is unique within each inspected panel and has no missing values.
- Drop `Variant_ID` from training features and preserve it only for prediction output ordering.
- Treat the actual CSVs as source of truth; their counts differ from the specification’s expected balanced train counts.

## Implementation Plan
- Create one shared training framework with strict panel isolation:
  - Config has four panel entries, each pointing to exactly one CSV.
  - Every fitted object is namespaced by panel: encoder, feature dropper, model, thresholds, metrics, predictions.
  - Shared code is allowed; fitted data artifacts are not shared.
- Environment setup:
  - Install/add `xgboost`, `lightgbm`, `catboost`, and optionally `optuna`; `scikit-learn` is already present.
  - Pin versions in a reproducible environment file.
- Preprocessing defaults:
  - Remove `Variant_ID` and `Label`.
  - Drop only all-missing or single-value features per panel/fold.
  - Keep high-missing numeric features by default because tree boosters can learn missing-value routing.
  - Fill categorical missing values with `__MISSING__`.
  - XGBoost: one-hot encode categorical columns with unknown handling.
  - LightGBM: use categorical-aware encoding/category dtype.
  - CatBoost: use native categorical feature handling.
- Validation:
  - Use panel-local `RepeatedStratifiedKFold`, fixed random seed, primary metric F1 with positive label `1`.
  - For `CFTR`, avoid a single holdout split; use repeated stratified CV because only 21 label-`0` rows exist.
  - Report per-panel F1, precision, recall, confusion matrix, PR-AUC, ROC-AUC, and threshold sensitivity.
- Modeling:
  - For each panel train XGBoost, LightGBM, and CatBoost baselines.
  - Tune each model family separately per panel with bounded search spaces.
  - Select the winning model per panel by mean CV F1; use tie-breakers in this order: lower F1 variance, higher recall for label `1`, simpler model.
  - Optional per-panel soft-voting ensemble only if it improves CV F1 and does not increase fold variance materially.
- Deliverables:
  - Four saved final models trained on full panel-specific training data.
  - Four validation reports and one consolidated comparison table.
  - Four prediction generators that preserve test row order and output `Variant_ID`, predicted class, and probability.
  - Project Detail Report material by the official deadline: 2026-06-29 17:00.

## Test Plan
- Data contract tests:
  - Each panel loads independently and has `Variant_ID` plus `Label`.
  - No pipeline accepts multiple panel CSVs as one training input.
  - No fitted encoder/dropper/model from one panel can be reused by another panel.
- Modeling tests:
  - CV folds preserve class representation.
  - Prediction output length equals input length and keeps original `Variant_ID` order.
  - Pipelines handle unseen categorical levels and missing values.
- Reproducibility tests:
  - Same seed reruns produce identical splits and near-identical metrics.
  - A clean environment can train one panel end-to-end from raw CSV to predictions.

## Assumptions
- `Label=1` is the positive class for F1 optimization unless organizer documentation later states otherwise.
- No existing `plans/` content is used.
- External row-level label lookup or genomic-coordinate enrichment is out of scope; use only the provided anonymized variant profiles for model training.
