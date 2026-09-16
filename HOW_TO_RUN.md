# How to Run PatHUgen

Operational guide for `PatHUgen.py`. For what the system is and why it is
built this way, see `PatHUgen_Explanation.md` first.

---

## 0. One-time setup

```bash
cd ~/Desktop/TEKNOFEST
.venv/bin/pip install -r requirements.txt      # if not already installed
```

Confirm the competition identity fields match your KYS (application system)
registration — these are filled with this project's existing values as
defaults, but **verify them**:

```bash
grep -A3 "Competition identity" pathugen/config.py
```

If anything is wrong, either edit the defaults in `pathugen/config.py`
directly, or override at run time with environment variables (shown below)
— no code change needed for the override.

---

## 1. Train (run once, ahead of the competition)

This fits the champion configuration (see `Champion_Config.md` and
`PatHUgen_Explanation.md` §5) — read from `champion_config.json`, not a
literal in the code — on the official training CSVs, and writes the model
bundle to `pathugen_model/`.

```bash
.venv/bin/python PatHUgen.py train
```

Takes under a minute (no hyperparameter search happens here — those
decisions were already found and recorded in `champion_config.json`; `train`
fits that already-chosen configuration, it doesn't search for it — if
`champion_config.json` is missing, `train` fails with a clear message
telling you to run `search` first or copy one in). Train a single panel
while iterating:

```bash
.venv/bin/python PatHUgen.py train CFTR
```

**Data location:** by default this reads from
`universite-veri-seti/EĞİTİM (TRAIN) SETLERİ/`. Override with:

```bash
TEKNOFEST_DATA_DIR=/path/to/training/csvs .venv/bin/python PatHUgen.py train
```

**Re-running `train` never overwrites or deletes anything.** Every run
writes a new numbered snapshot — `pathugen_model/<PANEL>/v1/`, then `v2/`,
`v3/`, ... — the same way `deliver` never overwrites a previous
`deliveryN/` folder. `deliver` always uses each panel's most recently
trained snapshot unless you tell it otherwise (see §4 below). See what
you've trained so far:

```bash
.venv/bin/python PatHUgen.py versions
```

```
Model dir: /home/.../pathugen_model

  GENEL:
    v1: trained 2026-09-08T21:11:47  (2931 rows, threshold 0.67)  <- latest, used by `deliver` unless --model-version is given
  CFTR:
    v1: trained 2026-09-08T21:11:23  (111 rows, threshold 0.78)
    v2: trained 2026-09-08T21:11:33  (111 rows, threshold 0.78)  <- latest, used by `deliver` unless --model-version is given
```

---

## 2. Redoing the search (advanced, rare — not a competition-day command)

`train` fits the hyperparameters, feature set, and threshold that were
already found and recorded in `champion_config.json`. If you get new or
materially different training data and want to re-derive those decisions
from scratch — the same multi-hour search, cross-validation, and
overfitting audit that produced the current `champion_config.json` in the
first place — that's what `run` does:

```bash
.venv/bin/python PatHUgen.py run
```

**This is not the same kind of command as `train`/`deliver`.** Those two are
fully self-contained — verified to work with nothing but `PatHUgen.py` and
`pathugen/` present. `run` is the opposite: it delegates to this project's
search pipeline (the `search/` package), so it only works when that sits
alongside the rest of the project, and it takes hours, not seconds — it
sweeps many combinations of feature set, model, and decision threshold per
panel, from scratch, with no cached or inherited hyperparameters, and keeps
only the ones that survive an overfitting audit. Never run this during the
actual timed competition test.

Search a single panel instead of all four:

```bash
.venv/bin/python PatHUgen.py run CFTR
```

**`run` never edits `champion_config.json` for you.** It writes its findings
to `outputs_best/best_config.json`, `outputs_best/champion_config.json` (if
every searched panel's champion is a single XGBoost model), and a
human-readable `outputs_best/BEST_CONFIG.md` explaining the reasoning per
panel. To actually adopt a new result, open `BEST_CONFIG.md`, then copy
`outputs_best/champion_config.json` over this project's own
`champion_config.json`, then run `python PatHUgen.py train` to fit it. This
is a deliberate, reviewed step — not automatic — so the champion
configuration always reflects a decision someone actually looked at. See
`Champion_Config.md` for the full workflow.

`run` accepts the sweep's environment-variable knobs (`SWEEP_PROFILE`,
`SELECT_OBJECTIVE`, `GAP_LIMIT`, `STABILITY_MIN`, ...) — see `RUNNERS.md`
for the full list. For example, a faster (but less thorough) search:

```bash
SWEEP_PROFILE=fast .venv/bin/python PatHUgen.py run
```

---

## 3. When a test CSV arrives — competition day

This is the command you run **every time** you have test data to score,
including the organizers' practice/sample-data check and the real timed
test. Every run creates a new, numbered delivery folder — nothing is
overwritten, so every attempt during the test window is preserved.

```bash
.venv/bin/python PatHUgen.py deliver --input-dir /path/to/test/csvs
```

`--input-dir` should point at the folder holding the four test CSVs (one
per panel — GENEL, KANSER, PAH, CFTR). The script auto-detects which file
belongs to which panel by filename keyword (case-insensitive: `master` or
`genel`, `kanser`, `pah`, `cftr`). If the organizers use filenames that
don't contain any of these keywords, or auto-detection reports an ambiguous
match, pass the four paths explicitly instead:

```bash
.venv/bin/python PatHUgen.py deliver \
  --genel  /path/to/GENEL_test.csv \
  --kanser /path/to/KANSER_test.csv \
  --pah    /path/to/PAH_test.csv \
  --cftr   /path/to/CFTR_test.csv \
  --input-dir /path/to/test/csvs
```

(`--input-dir` is still required even when passing explicit paths — it is
recorded in the delivery README for the audit trail.)

### What happens

1. Predicts on all four panels using each panel's most recently trained
   snapshot in `pathugen_model/` (see §4 to pin an older one instead).
2. Validates the result locally against the official JSON schema (unique
   ids per panel, valid `"0"`/`"1"` classes, probabilities in range, no
   NaN/Infinity) — **before** anything is written or uploaded. If validation
   fails, nothing is written and the problems are printed.
3. Creates the next `deliveryN/` folder (`delivery1`, `delivery2`, ... —
   auto-incrementing, scanned from what already exists) containing:
   - `TEAM_<TEAM_ID>_FINALN.json` — **upload this file**. Numbered so every
     run's output is preserved instead of overwritten.
   - The exact trained model snapshot used, `PatHUgen.py` itself (the only
     code needed to reproduce the result — not the rest of the project), a
     frozen `requirements.txt`, and a rendered `README.md` documenting the
     exact command, input paths, and output location for that run.

### Uploading

Upload `deliveryN/TEAM_<TEAM_ID>_FINALN.json` to the results system (`N`
is whatever number that run produced — check the console output or the
delivery folder itself). The guide allows multiple uploads within the time
window — the last one the system validates successfully before closing is
the official submission. If you run `deliver` again to try something
different, upload the new `deliveryN+1/TEAM_<TEAM_ID>_FINAL{N+1}.json`
instead.

> The competition guide's suggested filename is the unnumbered
> `TEAM_<TAKIM_ID>_FINAL.json`; that's a recommendation, not a strict
> requirement — the upload system validates JSON content, not filename.
> This project keeps each run's file numbered instead so nothing gets
> silently overwritten between attempts.

### Fast iteration (optional)

During the organizers' sample-data / format-check phase, if you want to
try several input paths quickly without repackaging the full model+code
bundle each time, add `--skip-copy` — it still creates a numbered delivery
folder and writes the JSON, just skips the code/model/README packaging:

```bash
.venv/bin/python PatHUgen.py deliver --input-dir /path/to/sample/csvs --skip-copy
```

**Do a full (non-`--skip-copy`) run for your actual official submission** —
that is the one that produces a complete, jury-reproducible package.

---

## 4. Using an older trained snapshot

`deliver` always uses each panel's most recently trained snapshot by
default. If you retrained and want to go back to an earlier one for a
specific delivery (e.g. to compare, or because the latest retrain looks
off), pin it explicitly — this applies to all four panels for that run:

```bash
.venv/bin/python PatHUgen.py versions                     # see what's available
.venv/bin/python PatHUgen.py deliver --input-dir /path/to/test/csvs --model-version 1
```

The delivery's `README.md` always records exactly which version was used
per panel, so every past delivery folder is self-documenting even after
you've trained many more snapshots since.

---

## 5. If the true test class balance looks different from training

Training-set positive rates are GENEL 73%, KANSER 69%, PAH 83%, CFTR 81%.
Each panel's threshold was chosen with this in mind (see
`PatHUgen_Explanation.md` §5), but if you get any signal that the real test
set is closer to balanced (1:1), you can recompute the decision threshold
from the already-fitted, already-calibrated model — **no retraining** — with:

```bash
.venv/bin/python PatHUgen.py deliver --input-dir /path/to/test/csvs --prior-shift 0.5
```

`--prior-shift` accepts any target positive rate in (0, 1). This only moves
the classification threshold (`predicted_class`); `predicted_prob` is
unaffected since it is not calibration-dependent for the metrics the guide
describes (AUROC-style, rank-based).

**How the move is computed.** The Bayes prior-shift formula is only valid on
calibrated probabilities, and the raw model output deliberately is not
calibrated (it is fit with `scale_pos_weight != 1`). So `train` also fits an
isotonic calibrator per panel — on leak-free out-of-fold predictions, never
in-sample — and saves it as `calibrator.joblib` in the model bundle. With
`--prior-shift`, the audited threshold is lifted onto the calibrated scale,
shifted there, and brought back to the raw scale; the decision itself is
always made on the raw probability against a raw threshold, exactly as
without the flag.

Two consequences worth knowing:

- **Passing the panel's own training prior changes nothing.** The shift is
  then the exact identity and the audited threshold is returned untouched
  (verified per panel against the un-shifted output).
- **Snapshots trained before calibration existed still work.** They have no
  `calibrator.joblib`; `--prior-shift` falls back to the older, approximate
  behaviour and prints a warning naming the panel. Re-run `train` to get a
  calibrated snapshot.

The threshold move is not linear in the target prior — on some panels a
moderate shift barely moves any row across the boundary, on others it moves
many. Run `deliver --prior-shift ... --skip-copy` on sample data first and
read the printed per-panel "predicted pathogenic rate" before deciding.

---

## 6. Overriding competition identity fields

If team_name / team_id / application_id ever need to change without editing
`PatHUgen.py`:

```bash
TEAM_NAME="PatHUgen" TEAM_ID="761197" APPLICATION_ID="4882800" \
  .venv/bin/python PatHUgen.py deliver --input-dir /path/to/test/csvs
```

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `No trained model found for <PANEL>` | Run `python PatHUgen.py train` first. |
| `no such trained version vN under ...` | That snapshot number doesn't exist for that panel — run `python PatHUgen.py versions` to see what's actually available. |
| ``run` needs the full project, not just PatHUgen.py` | You tried `run` from a copy that only has `PatHUgen.py` (e.g. inside a `deliveryN/code/` folder) — that's expected, `run` is not self-contained. Use the full project checkout instead, and never during a real submission. |
| `Could not auto-detect a test file for panel(s) [...]` | Filenames don't contain a recognizable panel keyword — pass `--genel/--kanser/--pah/--cftr` explicitly. |
| `expected N columns, got M` | The test CSV's column count doesn't match the trained schema (352 columns without Label, 353 with) — confirm you were given the right file for that panel. |
| `duplicate <ID> values in the input file` | The guide requires exactly one prediction per id; the input file itself has repeated ids — check with the organizers before proceeding. |
| Validation fails after `deliver` prints per-panel counts | Nothing was written; read the printed list of problems — each line names the exact record and field at fault. |

Every run (both `train` and `deliver` failures) prints a clean one-line
error message, not a stack trace — if you see a Python traceback instead,
that indicates a bug, not an expected input problem; capture the full output
for debugging.

---

## 8. Full command reference

```bash
python PatHUgen.py train [PANEL ...]            # default: all four panels
    [--model-dir DIR]                           # default: pathugen_model/
    # creates a NEW numbered snapshot per panel (v1, v2, ...); never overwrites

python PatHUgen.py versions
    [--model-dir DIR]                           # default: pathugen_model/
    # lists every trained snapshot per panel, marks which one is "latest"

python PatHUgen.py run [PANEL ...]              # default: all four panels
    # multi-hour search + overfitting audit; requires the full project
    # (the `search/` package), NOT a competition-day command. Every
    # training configuration is tuned from scratch — no cache, no
    # inheritance. Writes to outputs_best/, never edits champion_config.json
    # automatically. Respects the sweep's env vars (SWEEP_PROFILE,
    # SELECT_OBJECTIVE, GAP_LIMIT, ... — see RUNNERS.md).

python PatHUgen.py deliver --input-dir DIR
    [--genel PATH --kanser PATH --pah PATH --cftr PATH]   # override auto-detection
    [--model-dir DIR]                           # default: pathugen_model/
    [--model-version N]                         # pin a specific snapshot (default: latest)
    [--prior-shift FLOAT]                       # e.g. 0.5 for an assumed balanced test set
    [--skip-copy]                               # JSON only, skip code/model/README packaging
```
