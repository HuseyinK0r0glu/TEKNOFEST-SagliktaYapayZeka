# `champion_config.json` — field reference

What every key in `champion_config.json` means, who writes it, who reads it, and
why editing it does nothing until you retrain.

> Companion document. `Champion_Config.md` covers *where the file comes from*
> (the search), *how to adopt a new one*, and the current per-panel values table.
> This file covers *what the fields mean* and *how the values reach a prediction*.

---

## What this file is

The per-panel recipe `python PatHUgen.py train` fits: the exact feature list,
XGBoost hyperparameters and decision threshold for each of the four panels.

It is **data, not a hardcoded default**. There is no fallback anywhere in the
code — if the file is missing or incomplete, `load_champion_config()` raises
`ChampionConfigError` rather than quietly fitting something else
(`pathugen/champion_config.py:41-58`).

## Who reads it

Only two places:

| Caller | Purpose |
|---|---|
| `pathugen/train.py:66` | `train()` — the only thing that fits models from it |
| `analysis/current_champion_eval.py:134` | honest re-evaluation of the recipe |

Set `PATHUGEN_CHAMPION_CONFIG=/path/to/other.json` to train from an alternate
config without touching the checked-in one (`champion_config.py:23`).

## It is **not** used at inference

Prediction never opens this file. `PatHUgen.py deliver` → `predict_panel()`
reads the trained snapshot instead:

| Needed at test time | Comes from |
|---|---|
| feature list | `models/<PANEL>/v<N>/selected_features.json` |
| decision threshold, `train_prior` | `.../config.json` |
| fitted trees (hyperparameters baked in) | `.../seed{0..N}.joblib` |
| categorical encodings | `.../cat_maps.joblib` |
| prior-shift calibrator | `.../calibrator.joblib` |

`train()` **copies** the values below into that snapshot, and the snapshot's
copy is what actually decides predictions.

**Consequence:** editing `champion_config.json` changes nothing until you
retrain. If you edit it inside a delivery package without retraining, the
shipped `.joblib` weights win silently. That fails in the safe direction — a
stale config cannot corrupt competition-day output — but don't read a
delivery's JSON as a description of its shipped model unless you retrained it.

---

## Field reference

Each panel entry has exactly six keys. All are required;
`load_champion_config()` rejects a panel that is missing any of them.

### `selection_method`

How the feature list was **discovered** during the search.

- `"null"` — null-importance selection: true gain vs. a shuffled-label null
  distribution (`search/reference/selection.py`).
- `"hybrid"` — the same, plus a ReliefF / information-gain guard that
  additionally protects categorical and high-IG columns.

**Provenance only.** `train()` never re-runs selection; it uses `features`
verbatim. Changing this string alone has no effect.

### `aug_frac`

- `0.0` — train on this panel's own rows only.
- `> 0.0` — *also* use GENEL rows **not** present in this panel as extra
  **training** rows (`pathugen/augment.py`).

Leak-free: borrowed rows are never scored. Validated to help PAH and CFTR;
KANSER measurably regresses, so it stays at `0.0`. GENEL is the augmentation
source and never augments itself (`augment.py:22`).

### `threshold`

The decision cut on the **raw** predicted probability:

```
proba >= threshold  =>  Pathogenic (class 1)
```

The isotonic calibrator is *not* in this path. It exists only to compute how
far to **move** this threshold when `deliver --prior-shift` is passed —
the decision itself always happens on the scale the audited threshold lives on
(`pathugen/predict.py`, `predict_panel`).

### `threshold_policy`

Which metric the search maximized when it picked `threshold` — `"f1"`,
`"balanced_f1"` or `"mcc"`.

**Provenance only.** The chosen number is already baked into `threshold`, so
editing this string alone does nothing.

### `xgb_params`

Passed to `XGBClassifier` (`pathugen/model.py:17-23`). Two things are not
obvious:

- **`spw_mult` is not an XGBoost parameter.** It is popped out and multiplied
  into the computed `scale_pos_weight` (`n_neg / n_pos`) to tune
  class-imbalance weighting.
- **`n_estimators` is not listed** and defaults to `400`. Also fixed:
  `tree_method="hist"`, `eval_metric="logloss"`, `n_jobs=-1`.

Each panel is fit with `SEED_ENSEMBLE` different random seeds and the
predictions averaged — cheap variance reduction that matters most on the
smallest panel (CFTR, ~111 rows).

### `features`

The exact engineered column names, in the order the search ranked them. Every
one must be produced by `pathugen/features.build_features()` or `train()`
raises with the offending names listed (`train.py:82-86`).

Mixes raw competition columns (`AL_*`, `CAT_*`, `EK_*`) with engineered ones
(`blosum62`, `grantham`, `pct_missing`, `al_n_ultrarare`, ...). Panels
deliberately differ in length — 38 / 35 / 37 / 13 for GENEL / KANSER / PAH /
CFTR — because each panel is an independent model.

---

## Why there are no comments in the JSON file

Strict JSON has no comment syntax, and `pathugen/champion_config.py` parses it
with the standard library's `json.loads`. Keeping the file as plain, valid JSON
means any tool can read it; this document carries the prose instead.

## Regenerating

`save_champion_config()` rewrites the file wholesale from the search's output
(`search/sweep/cli.py:148`), so nothing you add inside the JSON survives a
re-adoption. This document does — keep it in sync by hand when the champion
changes. See `Champion_Config.md` for the adoption procedure.

## Related

- `Champion_Config.md` — provenance, adoption steps, current values table
- `pathugen/champion_config.py` — loader, validator, env override
- `pathugen/train.py` — the only consumer that fits models from this
- `search/sweep/cli.py` — the producer
- `WARNING.md` — caveat on the nested honest-F1 estimate
