# Champion configuration

`champion_config.json` (project root) is the per-panel recipe `python PatHUgen.py
train` fits: which features, which XGBoost hyperparameters, and which decision
threshold. It is **data**, not code — nothing in `pathugen/` hardcodes these
numbers as Python literals.

## Where it comes from

`python PatHUgen.py run` (the `search.sweep` package) tries many candidate
configurations per panel — combinations of feature-selection method, GENEL
augmentation ratio, champion-selection rule, and threshold policy — scores each
one with leak-free nested cross-validation, and discards any that show signs of
overfitting. The survivor with the best worst-case honest F1 is written to
`outputs_best/champion_config.json`.

Every Optuna search inside `run` starts from scratch: there is no cache file it
reads instead of tuning, and no earlier run's parameters are inherited. This is
by design — see `WARNING.md` for the one caveat that *is* still true about the
nested honest-F1 estimate.

## How to adopt a new result

1. Run `python PatHUgen.py run [PANEL...]` (multi-hour; needs the `search/`
   package, not just `pathugen/`).
2. Read `outputs_best/BEST_CONFIG.md` — it explains, per panel, exactly why
   that configuration won and whether it passed the overfitting audit.
3. If you're satisfied, copy `outputs_best/champion_config.json` over this
   project's `champion_config.json`.
4. Run `python PatHUgen.py train` to fit real model weights from the new
   recipe (creates a new version; nothing is overwritten).

Adopting a result is always a deliberate, manual copy — `run` never edits
`champion_config.json` itself.

## Current values (human-readable)

| Panel | Selection | Augmentation | Threshold | Policy | Features |
|---|---|---|---|---|---|
| GENEL | hybrid | none (GENEL is the augmentation source) | 0.670 | balanced_f1 | 38 |
| KANSER | null | none (full augmentation measurably hurts this panel) | 0.705 | f1 | 35 |
| PAH | null | full GENEL augmentation | 0.595 | mcc | 37 |
| CFTR | null | full GENEL augmentation | 0.780 | balanced_f1 | 13 |

Every panel's champion is currently a single tuned XGBoost model — no
averaging or stacking beat a single well-tuned model once overfitting was
accounted for. The exact hyperparameters and feature names are in
`champion_config.json` itself (it's small and human-readable — just open it).

## If a future search picks a non-XGBoost champion

`pathugen/train.py` only knows how to fit a single XGBoost model per panel.
If `python PatHUgen.py run` ever finds that a mean-of-models or stacked
ensemble beats a single model for some panel, `search.sweep` will say so
clearly in its console output and will skip writing that panel into
`outputs_best/champion_config.json` — adopting such a result requires
extending `pathugen/train.py` first, which is a deliberate design decision,
not something to do silently.
