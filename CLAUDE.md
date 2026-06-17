# TEKNOFEST 2026 — Variant Pathogenicity Classifier

Binary classification (Pathogenic=1 vs Benign=0) on genetic variants. **4 panels, always independent.**

## Panels

| Panel  | CSV                        |
|--------|----------------------------|
| GENEL  | `YARISMA_TRAIN_MASTER.csv` |
| KANSER | `YARISMA_TRAIN_KANSER.csv` |
| PAH    | `YARISMA_TRAIN_PAH.csv`    |
| CFTR   | `YARISMA_TRAIN_CFTR.csv`   |

Metric: binary F1 (Pathogenic), per-panel threshold tuning. Schema: 353 cols (`Variant_ID`, `Label`, 351 features: `AL_*`, `CAT_*`, `EK_*`, `AA_1/2`). Seed: 42.

## Data

Default: `universite-veri-seti/EĞİTİM (TRAIN) SETLERİ/` — Override: `export TEKNOFEST_DATA_DIR=/path`

## Commands (run from project root with `PYTHONPATH=.`)

```bash
PYTHONPATH=. .venv/bin/python train/run_baseline.py [PANEL]
PYTHONPATH=. .venv/bin/python train/run_feature_selection.py
PYTHONPATH=. .venv/bin/python train/run_optuna.py
PYTHONPATH=. .venv/bin/python train/run_stacking.py
PYTHONPATH=. .venv/bin/python train/finalize_champions.py
PYTHONPATH=. .venv/bin/python predict.py PANEL input.csv out.csv
PYTHONPATH=. .venv/bin/python eda/run_eda.py
```

Outputs → `outputs/`

## Layout

```
src/config.py           # paths, panels, column groups
src/data_io.py          # load CSVs
src/features.py         # feature engineering + encoding (build_features)
src/models.py           # XGB/LGBM/CatBoost + CV eval
src/cv.py               # 5-fold CV, threshold tuning
src/tuning.py           # Optuna spaces
src/feature_selection.py
src/diverse_models.py   # ExtraTrees, L1-LogReg (stacking bases)
src/stacking.py
src/aa_constants.py     # BLOSUM, Grantham, AA tables
train/run_*.py          # pipeline entry points
predict.py              # inference from champion bundle
plans/                  # strategy docs
```

## Pipeline order

1. `run_baseline.py` → leaderboard
2. `run_feature_selection.py` → selected features
3. `run_optuna.py` → tuned params
4. `run_stacking.py` → calibrated stack + OOF
5. `finalize_champions.py` → save champion per panel
6. `predict.py` → inference

## Rules

- Each panel trains/evaluates separately — no pooling, no cross-panel GroupKFold
- Always use `features.build_features()` for training and inference
- CatBoost gets `cat_cols`; other tree models get numeric matrix
- CAT column leakage risk: high-purity categories are suspect (`CAT_PURITY_FLAG=0.95`)
- Style: `from __future__ import annotations`, pathlib, type hints

## Dependencies

`requirements.txt` — Python 3.11+, scikit-learn, xgboost, lightgbm, catboost, optuna, shap, imbalanced-learn.
