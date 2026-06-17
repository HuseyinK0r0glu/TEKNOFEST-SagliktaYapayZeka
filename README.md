# TEKNOFEST 2026 — Variant Pathogenicity Classifier

Binary classification of genetic variants — **Pathogenic (1)** vs **Benign (0)** —
trained **independently** on four panels. Competition metric: **binary F1 on the
Pathogenic class**, with a per-panel decision threshold.

| Panel  | CSV                        | n (rows) | Notes |
|--------|----------------------------|----------|-------|
| GENEL  | `YARISMA_TRAIN_MASTER.csv` | 2931     | large, realistic |
| KANSER | `YARISMA_TRAIN_KANSER.csv` | 388      | strong discrimination |
| PAH    | `YARISMA_TRAIN_PAH.csv`    | 372      | most real headroom |
| CFTR   | `YARISMA_TRAIN_CFTR.csv`   | 111      | small — treat scores cautiously |

Each CSV has 353 columns: `Variant_ID`, `Label`, and 351 features
(`AL_*`, `CAT_*`, `EK_*`, `AA_1/2`). Seed is fixed at 42 everywhere.

---

## Quick start

```bash
# 1. create / activate the virtualenv and install deps
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. run the whole pipeline (one command, runs all steps, then leaves outputs)
.venv/bin/python main.py                       # 'fast' profile, all 4 panels
PIPELINE_PROFILE=thorough .venv/bin/python main.py   # full rigor (6-10h+)

# specific panels only
.venv/bin/python main.py CFTR PAH
```

Data location defaults to `universite-veri-seti/EĞİTİM (TRAIN) SETLERİ/`.
Override with `export TEKNOFEST_DATA_DIR=/path/to/csvs`.

All artifacts land in `outputs/`.

---

## How results are measured — read this

The pipeline reports **two** numbers per panel, and prints both at the end:

1. **Cross-validated (OOF) F1** — from 5-fold cross-validation. Every row is
   predicted by a model that did **not** train on it. This is the number the model
   selection is based on. It is *mildly optimistic*, because feature selection,
   hyperparameters, the threshold and the champion strategy all peek at every row
   at some point.

2. **Held-out test F1 (nested CV)** — the **honest** estimate. The data is split
   into outer folds; for each outer fold, feature selection + tuning + threshold +
   champion choice happen on the *outer-train* rows only, then are locked and
   scored once on the *untouched outer-test* rows. **This is the number to trust**
   for how the model will do on the competition's unseen test set.

There is **no single fixed train/test split** — that would waste rows on the tiny
CFTR panel (n=111). Nested cross-validation uses all the data and still gives a
leak-free test estimate.

The nested summary also reports **`nested_balanced_f1_mean`** — the F1 the locked
model+threshold would score if the test set were balanced (50/50). Because the
competition balance is unknown, this is the downside-risk number: a panel whose
ordinary F1 is high but balanced-F1 is much lower is leaning on the base rate.
Probabilities are **calibrated** (sigmoid on small panels, isotonic on GENEL) so
the emitted scores are meaningful and the threshold is more stable under a shift.

```
optimism_gap = OOF F1  -  honest (nested) F1
```

A larger gap means more overfitting. Expect it largest on the small panels
(CFTR, PAH) and smallest on GENEL. Example end-to-end CFTR result:

```
[1] CROSS-VALIDATED (OOF):  F1 = 0.973
[2] HELD-OUT TEST (nested): F1 = 0.937 ± 0.011   (optimism_gap = +0.036)
```

---

## Profiles & knobs

`PIPELINE_PROFILE` (default `fast`) presets everything below. Any variable you set
explicitly in the environment overrides the profile (it uses `setdefault`).

| Knob                 | `fast`       | `thorough`   | What it does |
|----------------------|--------------|--------------|--------------|
| `CV_REPEATS`         | 3            | 5            | repeats of 5-fold CV; averages OOF for lower-variance F1/threshold |
| `OPTUNA_TRIALS`      | 40           | 80           | hyperparameter search trials per model |
| `OPTUNA_TIMEOUT`     | 300          | 900          | wall-clock seconds per Optuna study |
| `OPTUNA_OBJECTIVE`   | `auc`        | `auc`        | search metric; `auc` is balance-independent and decouples search from threshold optimism |
| `OPTUNA_SEED_JITTER` | 0            | 1            | vary CV fold seed per trial so winners generalize across splits |
| `THRESHOLD_AGG`      | `fold_median`| `fold_median`| `fold_median` = median of per-fold optimal thresholds (more robust) |
| `SELECT_METRIC`      | `auc`        | `auc`        | metric the champion one-SE rule selects on (AUC = robust to test-set balance) |
| `NESTED_EVAL`        | 1            | 1            | run the honest held-out test estimate (step 6) |
| `NESTED_RETUNE`      | 0            | 0            | **opt-in only** — re-tunes 5 models on 15 folds; adds many hours, not for a fixed overnight window |
| `NESTED_REPEATS`     | 1            | 3            | outer-CV repeats; >1 stabilises the small-panel (PAH/CFTR) estimate |

Why AUC, not F1, drives selection: the **competition test-set class balance is
unknown**. AUC ranks variants independently of the class prior, while F1 is tied
to the (majority-positive) train base rate. Tuning, champion choice, and the
acceptance gate all use AUC; F1 is still reported. Also opt-in: `spw_mult` is
searched per model (a multiplier on the `n_neg/n_pos` class weight) so the
imbalanced panels can trade recall vs precision.

Extra (opt-in, both profiles): `EARLY_STOP=1` makes the boosted trees use a large
iteration cap + early stopping on a fold-internal validation slice instead of a
fixed 400 trees. `NESTED_OUTER_SPLITS` (default 5), `NESTED_REPEATS` (default 1).

To skip the honest estimate for a faster run: `NESTED_EVAL=0 .venv/bin/python main.py`.

---

## Pipeline steps

`main.py` runs these in order (each is also runnable standalone from `train/`):

1. **Baseline** (`run_baseline.py`) — XGB/LGBM/CatBoost 5-fold CV leaderboard.
2. **Feature selection** (`run_feature_selection.py`) — null-importance pruning:
   keep a feature only when its real importance beats what it earns on shuffled
   labels. Cuts noise, important on the small panels.
3. **Optuna tuning** (`run_optuna.py`) — per-panel, per-model hyperparameter
   search. The known-good default config is always evaluated first (trial #0) so a
   search can never score below the baseline.
4. **Stacking** (`run_stacking.py`) — 5 base models (XGB, LGBM, CatBoost,
   ExtraTrees, L1-LogReg) → isotonic calibration → logistic meta-learner. Two stack
   variants (trees-only / all-bases) are evaluated. Saves the per-panel bundle.
5. **Finalize champions** (`finalize_champions.py`) — picks the champion strategy
   per panel using a **one-standard-error rule**: among candidates within 1 SE of
   the best OOF F1, it keeps the *simplest* one, so a stack must beat a single
   model by more than noise to be chosen. Writes `champion.json` per panel.
6. **Nested eval** (`run_nested_eval.py`) — the leak-free honest test estimate.

---

## Anti-overfitting design

Why the OOF number is optimistic and what mitigates each risk:

| Risk | Mitigation |
|------|------------|
| Feature selection peeks at all rows | measured by **fold-aware** selection inside the nested estimate (step 6) |
| Threshold tuned on the same OOF | `THRESHOLD_AGG=fold_median` + `CV_REPEATS>1` |
| Champion picked from ~8 candidates | **one-SE rule on AUC** prefers the simpler model |
| Unknown test-set balance | select + tune + gate on **AUC**; calibrated probabilities; `nested_balanced_f1` diagnostic |
| Optuna overfits one fixed split | `OPTUNA_SEED_JITTER=1`, `CV_REPEATS>1`, threshold-free objective |
| Deep trees memorize small panels | panel-size-aware depth caps + optional `EARLY_STOP=1` |
| No true held-out test | **nested CV** (step 6) gives the honest number |

Judge improvements by the **nested AUC** (primary, balance-independent), then the
**nested F1**, the **`nested_balanced_f1`**, and the **lift over the trivial
all-positive baseline** (`2p/(1+p)`) — never by raw OOF F1. The positive class is
the majority, so a trivial model already scores 0.82–0.91 F1.

---

## Inference

```bash
.venv/bin/python predict.py PANEL input.csv output.csv
.venv/bin/python predict.py CFTR        # wiring check on the training CSV
```

Loads the panel's champion bundle and writes `Variant_ID, proba, prediction`.
Categoricals are encoded with the train-time vocabulary so codes stay consistent.

---

## Outputs

```
outputs/
  baseline/baseline_leaderboard.csv
  feature_selection/{PANEL}_selected.txt, {PANEL}_null_importance.csv
  optuna/{PANEL}_{model}_best.json
  stacking/{PANEL}_oof.csv, {PANEL}_thresholds.json, stacking_summary.csv
  models/{PANEL}/        champion.json + saved base models, calibrators, meta
  models/champions_summary.csv     # [1] cross-validated result
  nested_eval/nested_summary.csv   # [2] honest held-out result + optimism_gap
```

## Project layout

```
src/config.py            paths, panels, column groups, seed
src/data_io.py           load panel CSVs
src/features.py          feature engineering (build_features) — label-independent
src/models.py            XGB/LGBM/CatBoost factories + CV eval
src/cv.py                5-fold CV, threshold tuning, repeated-CV, early-stop hook
src/tuning.py            Optuna spaces, objective, panel-size-aware capacity
src/feature_selection.py null-importance selection
src/diverse_models.py    ExtraTrees, L1-LogReg (stacking bases)
src/stacking.py          calibration + logistic stacking
src/aa_constants.py      BLOSUM / Grantham / amino-acid tables
train/run_*.py           pipeline entry points
predict.py               inference from a champion bundle
main.py                  full pipeline orchestrator (this is what you run)
```

## Requirements

Python 3.11+ (tested on 3.12). See `requirements.txt` — scikit-learn, xgboost,
lightgbm, catboost, optuna, shap, imbalanced-learn.
```bash
.venv/bin/pip install -r requirements.txt
```
