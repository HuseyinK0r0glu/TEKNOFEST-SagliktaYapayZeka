# Three search runners: `search.reference`, `search.improved`, `search.sweep`

All three live under the `search/` package (not shipped in a delivery — see
`pathugen/` for the self-contained delivery code) and each writes its own
`.txt` log per run via `search.runlog`.

---

## 1. `search.reference` — the baseline pipeline

The exact run reported in the PDR. Originally auto-generated from
`TEKNOFEST_pipeline_colab.ipynb`'s code cells, since hand-split into modules
(`search/reference/config.py`, `selection.py`, `modeling.py`, `tuning.py`,
`stacking.py`, `champion.py`, `nested.py`, `pipeline.py`, `predict.py`,
`cli.py`) — nothing about the logic changed, only the organization. Feature
engineering, the amino-acid tables, and raw data loading are shared with
`pathugen/` rather than duplicated (verified byte-identical before merging).

```bash
cd ~/Desktop/TEKNOFEST && \
QUICK=0 USE_GPU=0 RUN_NESTED=1 AUGMENT_PANELS=PAH,CFTR SELECTION_METHOD=null \
OPTUNA_TRIALS=80 OPTUNA_TIMEOUT=600 CV_REPEATS=3 NESTED_REPEATS=2 \
TEKNOFEST_OUT_DIR=outputs_final \
nohup .venv/bin/python -c "from search.reference.cli import main; main()" > /dev/null 2>&1 &
```

Outputs: `outputs_final/results_summary.csv` + `outputs_final/models/<PANEL>/`
Log:     `logs/reference_<timestamp>.txt`  (shortcut: `logs/reference_latest.txt`)

---

## 2. `search.improved` — improvements layered on top

Imports `search.reference` as a library and layers on top; does not modify it.

```bash
cd ~/Desktop/TEKNOFEST && \
nohup .venv/bin/python -c "from search.improved.cli import main; main()" > /dev/null 2>&1 &
```

Outputs: `outputs_improved/results_summary_improved.csv` + `outputs_improved/models/<PANEL>/`
Log:     `logs/improved_<timestamp>.txt`  (shortcut: `logs/improved_latest.txt`)

### Improvements added

| # | Improvement | Knob (default) | What it does |
|---|---|---|---|
| 1 | Prior-insensitive decision threshold | `THRESHOLD_POLICY=minimax` | Computes `f1` / `balanced_f1` / `mcc` / `minimax` together, logs the table, ships the selected one. Pinned with a bootstrap median. |
| 2 | Prior-shift-ready threshold | `REPORT_PRIORS=0.5,0.6,0.74` | Isotonic calibrator + training prior are saved in the bundle; if the test ratio differs, the threshold moves without retraining. |
| 3 | Paired-bootstrap One-SE | `PAIRED_BOOTSTRAP=400` | Uses the distribution of the (best − candidate) AUC **difference** instead of the marginal SE, which at n=111 was so wide the simplest candidate always won. |
| 4 | Diverse-view stacking | `DIVERSE_STACK=1` | A second, ReliefF-pruned feature view for XGBoost + LightGBM -> the meta layer sees 7 columns (no new Optuna search). |
| 5 | Shortcut audit + ablation | `SHORTCUT_AUDIT=1` | Standalone AUC of the missingness signal alone; re-scores the champion without missingness features. |
| 6 | Per-panel feature selector | `SELECTION_METHOD_<PANEL>` | KANSER defaults to `hybrid`, others to `null` (held-out A/B: only KANSER beat the noise floor with it). |
| 7 | Seed ensemble | `SEED_ENSEMBLE=3` | The final champion is refit with several seeds and predictions averaged. |
| 8 | Borrowed-row ratio | `AUG_FRAC=1.0` | Plumbing-free stand-in for down-weighting: a stratified subset of the augmentation pool. |

Every improvement is repeated **inside** the nested evaluation too — the
reported honest score is genuinely that pipeline's score. (See `WARNING.md`
for the one part that is *not* re-derived per fold: hyperparameters.)

Every panel's Optuna search runs fresh — there is no `REUSE_PARAMS` flag or
cross-run cache. `tuned_params.json` is still written per panel as a record
of what was found, but it is never read back.

### Useful knobs

```bash
THRESHOLD_POLICY=balanced_f1 .venv/bin/python -c "from search.improved.cli import main; main()"
DIVERSE_STACK=0 SHORTCUT_AUDIT=0 .venv/bin/python -c "from search.improved.cli import main; main()"
```
Or pass panels via `sys.argv` when invoking `main()` from a small wrapper script.

---

## 3. `search.sweep` — sweeps every option, entirely from scratch

The cross-product of everything `search.reference` and `search.improved`
offer: instead of trying each option one at a time, it sweeps **every
combination**, scores each with leak-free nested CV, and keeps the best
among those that pass an overfitting audit.

```bash
cd ~/Desktop/TEKNOFEST && \
nohup .venv/bin/python -c "from search.sweep.cli import main; main()" > /dev/null 2>&1 &
tail -f logs/sweep_latest.txt
```

`python PatHUgen.py run` calls this programmatically (same knobs, same
output); the only difference is it's reachable from PatHUgen.py's own CLI.
It never automatically applies its results to `champion_config.json` — see
`HOW_TO_RUN.md` §2 and `Champion_Config.md`.

**No cached or inherited hyperparameters, anywhere.** Every training
configuration (every combination of feature-selection method and
augmentation ratio) gets its own from-scratch Optuna search — there used to
be a `REUSE_PARAMS` flag and an `_inherit_params()` cross-run cache; both
have been removed entirely, because they made `python PatHUgen.py run`
silently skip all real tuning whenever cache files already existed on disk.
This makes the sweep slower (every cell tunes independently) but genuinely
honest.

### Swept axes

**Expensive** (requires retraining): feature selection `null|hybrid` x
augmentation ratio `0.0|0.5|1.0` (skipped for GENEL, the augmentation
source; tried on **KANSER too** — full augmentation hurt KANSER in the
reference run, but partial augmentation was never tried there).

**Cheap** (from the same base models, free): champion rule `marginal|paired`
x candidate pool `all|singles` x threshold policy `f1|balanced_f1|mcc|minimax`
= 16 combinations per training configuration.

The cheap axes are re-evaluated on **every outer fold** of the nested CV;
the reported honest score is genuinely that configuration's score.

### Overfitting audit (a configuration that fails is ELIMINATED)

| # | Test | Default limit | What it catches |
|---|---|---|---|
| 1 | Optimism gap (OOF F1 − honest F1) | <= 0.040 | Classic overfitting |
| 2 | Champion stability | >= 50% of outer folds pick the same candidate | Model selection fitting noise |
| 3 | Beats the naive baseline | honest F1 > baseline | A fake "high score" |

### Final selection objective

`SELECT_OBJECTIVE` (default `robust`):

* `robust` = min(honest F1, honest balanced F1) — hedges against an unknown test class ratio
* `honest_f1` — assumes the test distribution matches training
* `honest_balanced_f1` — assumes a 1:1 test distribution (what the spec implies)

All three objectives' winners are printed side by side; if they differ, you see it immediately.

### Outputs

```
outputs_best/BEST_CONFIG.md            <- READ THIS FIRST (human-readable final decision)
outputs_best/best_config.json          <- machine-readable
outputs_best/champion_config.json      <- adoptable recipe (only if every panel's champion
                                           is a single XGBoost model — see Champion_Config.md)
outputs_best/sweep_all_results.csv     <- score of EVERY configuration swept
outputs_best/results_summary_best.csv  <- winner per panel
outputs_best/models/<PANEL>/           <- the winning configuration's model bundle
logs/sweep_<timestamp>.txt
```

### Useful knobs

```bash
SWEEP_PROFILE=fast      # augmentation {0,1} — roughly half the cells
SWEEP_PROFILE=wide      # augmentation {0,.25,.5,.75,1} — more cells, longer
SWEEP_PROFILE=decision  # only 1 training cell per panel (still tuned fresh — see WARNING.md)
SELECT_OBJECTIVE=honest_balanced_f1
GAP_LIMIT=0.03 STABILITY_MIN=0.6      # a stricter overfitting filter
```

**WARNING:** do not run two of these three runners at the same time — each
saturates all available threads.

---

## Log files

All three write to `logs/`. A log contains:

* **Header** — date, command, Python/library versions, CPU/RAM/GPU, git commit, every environment variable active for that run.
* **Body** — everything printed to the screen (panel progress, Optuna scores, candidate tables, threshold-policy comparison, error analysis).
* **Footer** — finish time, total duration, and `SUCCESS` / `ERROR — <exception>`. If the script crashes, the traceback is written to the log too.

```bash
tail -f logs/improved_latest.txt      # watch a run live
grep "Status" logs/*.txt              # which runs finished successfully
```

The log directory can be overridden with `TEKNOFEST_LOG_DIR`.

---

## Comparison

`search.improved` automatically prints a side-by-side diff against the
reference pipeline at the end, if `outputs_final/results_summary.csv`
exists (override the path with `COMPARE_TO`). The two rows to look at:

* **Honest F1** — expected performance at the training distribution.
* **Honest balanced F1** — expected performance if the test set turns out balanced.

If the second rises noticeably while the first barely drops, the
improvements worked: robustness against an unknown test distribution was
bought cheaply.
