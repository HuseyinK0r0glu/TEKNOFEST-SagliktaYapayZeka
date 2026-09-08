# PatHUgen — Delivery {{DELIVERY_NUMBER}}

Team: {{TEAM_NAME}} (team_id {{TEAM_ID}}, application_id {{APPLICATION_ID}})
Generated: {{GENERATED_AT}}

This folder is a complete, self-contained delivery package produced by
`PatHUgen.py deliver`, matching Section 6 ("Model Dosyalarının USB ile
Teslimi ve Yeniden Üretilebilirlik") of
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

## How to reproduce this result from scratch

1. Install dependencies (`requirements.txt` in this folder pins the exact
   versions used):
   ```
   pip install -r requirements.txt
   ```
2. Re-train the locked model configuration (needs the official training
   CSVs, not included here — see `code/` for `PatHUgen.py`):
   ```
   python code/PatHUgen.py train
   ```
   This reproduces the model files under `model/`.
3. Re-run inference on the same test data:
   ```
   python code/PatHUgen.py deliver --input-dir <path-to-test-csvs>
   ```

## Contents of this folder

```
delivery{{DELIVERY_NUMBER}}/
  README.md                          <- this file
  requirements.txt                   <- exact pinned dependency versions
  {{OUTPUT_JSON_NAME}}      <- result JSON — upload this
  code/                              <- PatHUgen.py + its delivery README template
                                        (everything this script needs to run — nothing else)
  model/                             <- trained model weights + configuration
                                        (one subfolder per panel; exact snapshot used
                                        for this delivery — see "Input" above)
```

## Environment

- Python: {{PYTHON_VERSION}}
- Key libraries: {{LIBRARY_VERSIONS}}
- Seed: 42 (fixed throughout)
