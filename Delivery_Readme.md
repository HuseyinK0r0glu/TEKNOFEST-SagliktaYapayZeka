# PatHUgen — Delivery {{DELIVERY_NUMBER}}

Team: {{TEAM_NAME}} (team_id {{TEAM_ID}}, application_id {{APPLICATION_ID}})
Generated: {{GENERATED_AT}}

This folder is a complete, self-contained delivery package produced by
`PatHUgen.py deliver`, matching Section 6 ("Delivering Model Files via USB
and Reproducibility") of
`TEKNOFEST_2026_Saglikta_Yapay_Zeka_Final_Uygulama_ve_Sonuc_Teslim_Kilavuzu.pdf`.

## Run command

The exact command that produced this delivery:

```
{{RUN_COMMAND}}
```

## Input

Test data directory used for this run:

```
{{INPUT_DIR}}
```

Panel -> file mapping used:

{{PANEL_FILES}}

Predictions produced per panel:

{{PREDICTION_COUNTS}}

**Note:** the competition test data itself is intentionally NOT included in
this folder (the delivery guide explicitly forbids shipping test data as
part of the model package).

## Output

The result JSON for this run was written to:

- `{{OUTPUT_JSON}}` — upload this file to the results system.

Numbered per team convention (`_FINAL{{DELIVERY_NUMBER}}.json`) so every
run's output is preserved instead of overwritten.

## How to run this delivery's model on test data

Everything needed is in this folder — the trained weights under `model/` are
the delivered model, so no training data and no re-training are required.

Run both commands from inside this folder.

1. Create an isolated environment and install the pinned dependencies
   (`requirements.txt` here is an exact `pip freeze` of the environment that
   produced this delivery; Python {{PYTHON_VERSION}} was used):
   ```
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
   Windows: `py -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`.
   A virtual environment is used rather than a plain `pip install` because
   recent Linux distributions refuse to install into the system Python
   (PEP 668). If you already have an environment with these packages, use its
   interpreter in step 2 instead of `.venv/bin/python`.

2. Score any test CSVs with the shipped weights — `--model-dir model` points
   at this folder's model package, so nothing is re-trained:
   ```
   .venv/bin/python code/PatHUgen.py deliver --input-dir <path-to-test-csvs> --model-dir model
   ```
   Windows: `.venv\Scripts\python code\PatHUgen.py deliver --input-dir <csvs> --model-dir model`

   The test CSVs may sit anywhere, including inside this folder (e.g. a
   `test_data/` subfolder — then pass `--input-dir test_data`). The run prints
   the path of the result JSON it writes, under `code/delivery<N>/`. Given the
   same input files, it reproduces the result JSON above exactly.

## How to reproduce the weights themselves from scratch

Optional — only needed to verify that `model/` really is what the recorded
configuration produces. This step needs the official training CSVs, which the
delivery guide forbids shipping here:

```
.venv/bin/python code/PatHUgen.py train
```

It refits the champion configuration recorded in `code/champion_config.json`
(data, not a hardcoded literal) and writes a fresh numbered snapshot, leaving
this folder's `model/` untouched.

## How to redo the search itself (full pipeline, from scratch)

Optional, and by far the longest step. This is the research pipeline that
*produced* `champion_config.json` in the first place — it rediscovers the
recipe rather than merely refitting it, and it is not a competition-day
command. Reach for it when the training data changes materially, not to
reproduce this delivery.

```
.venv/bin/python code/PatHUgen.py run
```

Add panel names to search only some of them — `... run KANSER PAH` — any of
`GENEL`, `KANSER`, `PAH`, `CFTR`; the default is all four.

For each panel it sweeps the full cross-product of feature selection x
augmentation ratio, retraining and re-tuning every cell from scratch (no
cached or inherited hyperparameters anywhere), and then scores each of those
against the free post-training grid of threshold rule x pool x policy. Every
candidate is validated with leak-free nested cross-validation and screened by
an overfitting audit (optimism gap <= 0.040, champion stability >= 50%, must
beat a naive baseline). Under the default `full` profile that is **20 trained
configurations and 320 scored candidates**:

| Panel  | Trained configurations                                       |
|--------|--------------------------------------------------------------|
| GENEL  | 2 — it is the augmentation source, so it is never augmented   |
| KANSER | 6                                                            |
| PAH    | 6                                                            |
| CFTR   | 6                                                            |

### What it needs that `deliver` does not

- **The `search/` package**, which must sit next to `PatHUgen.py` — it is
  included under `code/` here. (`train` and `deliver` never need it; `run`
  *is* the search.)
- **The official training CSVs**, which the delivery guide forbids shipping in
  this folder. Put them at `universite-veri-seti/EĞİTİM (TRAIN) SETLERİ/`
  relative to the directory you run the command from, or point at them
  directly — the path is resolved against the working directory, not against
  this folder:
  ```
  TEKNOFEST_DATA_DIR=/path/to/train/csvs .venv/bin/python code/PatHUgen.py run
  ```
- **Time — budget about a day.** The boosters run on CPU by default (`USE_GPU=1`
  exists, but CPU is faster at this data size). For calibration: this project's
  last full-scale run used the larger `wide` profile (32 trained configurations,
  512 candidates) and took **39 h 14 m** wall-clock — GENEL 5.9 h, KANSER 17.1 h,
  PAH 9.3 h, CFTR 6.9 h. The default `full` profile is roughly 62% of that grid.
  Hardware changes this a lot.

Progress is only checkpointed **between** panels, so an interrupt mid-panel
loses that panel's work and a restart redoes it from the beginning. For a run
this long, start it detached (`tmux`, or `nohup ... &`) rather than in a
terminal you might close, and disable sleep/suspend — on a laptop a suspend
counts against the run's own elapsed-time budget. Running one panel at a time
is the safer pattern.

### What it writes

Everything lands in `outputs_best/`, created in the directory you run from:

- `BEST_CONFIG.md` — the human-readable decision per panel. **Read this first.**
- `champion_config.json` — the adoptable recipe, in the same format as
  `code/champion_config.json`
- `best_config.json`, `results_summary_best.csv`, `sweep_all_results.csv` —
  the machine-readable detail behind that decision
- `models/<PANEL>/` — the fitted bundle for each winning configuration

The full run log is teed to `code/logs/sweep_<timestamp>.txt`, with
`code/logs/sweep_latest.txt` pointing at the newest.

`run` never edits `code/champion_config.json` on its own. Adopting a result is
a deliberate, separate step: review `outputs_best/BEST_CONFIG.md`, copy
`outputs_best/champion_config.json` over `code/champion_config.json`, then run
`train` to fit it. (If a panel's winner is not a single XGBoost model, no
`champion_config.json` is written for that run — the full detail still lands in
`champion.json` under that panel's bundle.)

### Tuning breadth vs. time

No environment variables need setting: `run` already configures itself for the
full-rigor search (`QUICK=0`, 80 Optuna trials capped at 600 s per
configuration, 3 CV repeats, 5-fold nested CV x 2 repeats). `SWEEP_PROFILE`
trades breadth against runtime:

| `SWEEP_PROFILE`  | Trained configurations | Candidates |
|------------------|------------------------|------------|
| `decision`       | 4                      | 64         |
| `fast`           | 14                     | 224        |
| `full` (default) | 20                     | 320        |
| `wide`           | 32                     | 512        |

```
SWEEP_PROFILE=fast .venv/bin/python code/PatHUgen.py run
```

## Contents of this folder

```
delivery{{DELIVERY_NUMBER}}/
  README.md                          <- this file
  requirements.txt                   <- exact pinned dependency versions
  {{OUTPUT_JSON_NAME}}      <- result JSON — upload this
  code/                              <- PatHUgen.py + pathugen/ (what train/deliver need)
                                        + champion_config.json (the fitted recipe, not a
                                        hardcoded literal) + this README's template,
                                        plus search/ — the from-scratch search pipeline
                                        behind `PatHUgen.py run`, included because
                                        PatHUgen.py can invoke it too. Only pathugen/ is
                                        actually needed to reproduce this delivery's result.
  model/                             <- trained model weights + configuration
                                        (one subfolder per panel; exact snapshot used
                                        for this delivery — see "Input" above)
```

## Environment

- Python: {{PYTHON_VERSION}}
- Key libraries: {{LIBRARY_VERSIONS}}
- Seed: 42 (fixed throughout)
