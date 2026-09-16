# Known limitation — review before final submission

## The "honest" nested-CV estimate has a mild optimistic bias from hyperparameters

Every `honest_f1` / `honest_auc` / `honest_balanced_f1` number produced by
`search.reference`, `search.improved`, and `search.sweep` comes from nested
cross-validation: on every outer fold, **feature selection**, **champion
model choice**, and **decision threshold** are all re-derived using only that
fold's training rows, then measured once on the untouched test rows. That
part is genuinely leak-free.

**What is not re-derived per fold: the XGBoost/LightGBM/CatBoost/etc.
hyperparameters themselves.** Optuna tunes each model kind once per training
configuration (per panel, on the full panel's data), and those same tuned
hyperparameters are then used inside every outer fold of the nested
evaluation. Since the outer test fold's rows were part of the data Optuna
saw while tuning (just not while fitting the final model), the reported
honest F1/AUC carries a mild optimistic bias from this one step.

## Why this hasn't been fixed

Fixing it properly means re-tuning hyperparameters inside every outer fold
instead of once per training configuration. At current settings (10 outer
folds x up to 6 grid cells per panel x 5 model kinds x up to 600s Optuna
budget each), that multiplies the sweep's runtime from roughly 6 hours to an
estimated 150-200+ hours across all four panels — not viable on the current
timeline. See the conversation/design notes for the considered alternatives
(a much smaller per-fold budget, a reduced grid, or re-tuning only the
already-selected champion kind); none were adopted yet.

## What IS fixed

- The search itself now genuinely searches: no `REUSE_PARAMS` flag, no
  `_inherit_params()` cross-run cache, no sharing of tuned parameters across
  augmentation ratios. Every training configuration in every sweep gets its
  own from-scratch Optuna search. (Previously, `python PatHUgen.py run` ran
  **zero** Optuna trials against the cached files that existed on disk —
  see git history for the audit that found this.)
- `champion_config.json` (the per-panel recipe `pathugen/train.py` fits) is
  no longer a hardcoded Python literal — it's a data file produced by
  `search.sweep` and adopted by a deliberate copy step. See
  `Champion_Config.md`.

## What to do about this

Before treating the currently reported honest F1/AUC numbers as final:

1. Decide whether the bias is acceptable for this competition (it is
   probably small — feature selection and model/threshold choice, the parts
   most prone to overfitting on small panels like CFTR, ARE fully re-derived
   per fold; only the hyperparameters are shared).
2. If not, budget the runtime for one of the fixes above and re-run
   `python PatHUgen.py run` before the final submission.
3. Either way, if this project is written up again (a report, a README), say
   precisely what was and wasn't re-derived per fold — don't call the nested
   estimate unqualified "leak-free honest F1" without this caveat.
