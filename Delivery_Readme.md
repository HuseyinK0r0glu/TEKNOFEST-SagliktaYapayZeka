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

## Contents of this folder

```
delivery{{DELIVERY_NUMBER}}/
  README.md                          <- this file
  requirements.txt                   <- exact pinned dependency versions
  {{OUTPUT_JSON_NAME}}      <- result JSON — upload this
  code/                              <- PatHUgen.py + pathugen/ (what train/deliver need)
                                        + champion_config.json (the fitted recipe, not a
                                        hardcoded literal) + this README's template.
                                        May also include search/ — the from-scratch search
                                        pipeline behind `python PatHUgen.py run` — since
                                        PatHUgen.py can invoke it too; only pathugen/ is
                                        actually needed to reproduce this delivery's result.
  model/                             <- trained model weights + configuration
                                        (one subfolder per panel; exact snapshot used
                                        for this delivery — see "Input" above)
```

## Environment

- Python: {{PYTHON_VERSION}}
- Key libraries: {{LIBRARY_VERSIONS}}
- Seed: 42 (fixed throughout)
