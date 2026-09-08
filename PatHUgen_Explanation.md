# PatHUgen — Full Architecture Explanation

TEKNOFEST 2026 Health AI Competition | Team PatHUgen | University-and-above level
Task: classify missense genetic variants as **Pathogenic (1)** or **Benign (0)**
across four independent panels: **GENEL, KANSER, PAH, CFTR**.

This document explains the whole system from zero, end to end: the problem,
the data, every stage the project went through, why the final code looks the
way it does, and how the pieces fit together. If you are looking at this
project after time away, read this file first.

---

## 1. The problem

Missense variants are point mutations that change a single amino acid in a
protein. Most are harmless (benign); some disrupt protein function and cause
disease (pathogenic). Given a variant's precomputed annotation profile —
population allele frequencies, in-silico pathogenicity scores, evolutionary
conservation scores, and the amino acid substitution itself — predict which
class it belongs to.

The competition provides four **independent** panels, each scored separately:

| Panel | Source file | Rows | Pathogenic rate |
|---|---|---|---|
| GENEL (general, multi-gene) | `YARISMA_TRAIN_MASTER.csv` | 2,931 | 73.3% |
| KANSER (hereditary cancer) | `YARISMA_TRAIN_KANSER.csv` | 388 | 69.1% |
| PAH (phenylketonuria) | `YARISMA_TRAIN_PAH.csv` | 372 | 83.3% |
| CFTR (cystic fibrosis) | `YARISMA_TRAIN_CFTR.csv` | 111 | 81.1% |

Each row has 351 anonymized features (`AL_1..AL_334` allele frequencies and
in-silico scores, `CAT_1..CAT_6` categorical context, `EK_1..EK_9`
conservation scores, `AA_1`/`AA_2` reference/alternate amino acid) plus
`Variant_ID` and `Label`. Column names and genomic coordinates are
deliberately anonymized so no team can look up the answer externally — the
only signal available is what these numbers encode.

**Why four independent models, not one:** the panels differ enormously in
size (111 to 2,931 rows) and class balance, and the feature sets that
matter for each turned out to be almost entirely disjoint (see Section 5).
Forcing one model across all four would mean whichever panel dominates by
row count drowns out the others.

---

## 2. The journey — four scripts, each answering a different question

This project went through four generations of pipeline, each built to close
a specific gap the previous one exposed. **`PatHUgen.py` is the only one
meant to run during the actual competition** — the other three exist so that
`PatHUgen.py`'s hard-coded configuration has a fully audited paper trail.

### 2.1 `run_final.py` — the reference pipeline

The baseline, unmodified pipeline (originally developed and validated on
Google Colab, later ported to run locally). For each panel, independently:

1. **Feature engineering** (`build_features`) — turns 351 raw columns into
   442 engineered features (see Section 4).
2. **Feature selection** (`null_importance_selection`) — LightGBM gain
   importance, real vs. label-shuffled null distribution, keeps only
   features that clear the 95th percentile of noise or land in the top-K by
   score. Cuts 442 features down to 13–52 per panel.
3. **Cross-panel augmentation** — GENEL rows not present in PAH/CFTR are
   added to those panels' **training folds only** (never validation/test),
   because PAH has only 62 benign rows and CFTR only 21 — far too few to
   learn a stable decision boundary alone.
4. **Hyperparameter tuning** (Optuna, ROC-AUC objective — not F1, because
   the competition's true test-set class balance is unknown and F1 would
   overfit to the training panel's balance).
5. **Calibrated stacking** — five base models (XGBoost, LightGBM, CatBoost,
   ExtraTrees, L1-logistic regression), isotonic-calibrated, fed to a
   logistic meta-learner.
6. **Champion selection** — a One-Standard-Error rule on bootstrapped AUC:
   the simplest candidate within one SE of the best AUC wins. In every run,
   on every panel, a **single tuned model beat every stacked ensemble**.
7. **Nested cross-validation** — the honest, leak-free generalization
   estimate: feature selection and model choice are redone inside every
   outer fold, using only that fold's training rows.

This is the pipeline that produced the numbers in the submitted PDR
(Proje Detay Raporu), scored 96.5.

### 2.2 `run_improved.py` — testing specific hypotheses

Built to test four concrete improvement ideas against the reference, without
touching `run_final.py`'s code (it imports it as a library):

1. **Prior-aware decision thresholds** — instead of only maximizing F1 at
   the training class balance, compute the threshold under four policies
   (`f1`, `balanced_f1`, `mcc`, `minimax`) and ship whichever is most robust
   if the true test balance turns out different from training.
2. **Paired-bootstrap champion selection** — the original One-SE rule
   compared each candidate's AUC against the *marginal* uncertainty of the
   best candidate. On small panels that band is so wide it can't
   distinguish anything. Comparing the **paired difference** between two
   candidates' bootstrap distributions is a much sharper test.
3. **A second, ReliefF-pruned feature view** fed to XGBoost/LightGBM
   alongside the original view, so a stack has genuinely diverse inputs
   rather than five models looking at the same columns.
4. **A "shortcut audit"** — measuring how much predictive power comes from
   annotation-completeness features alone (missingness patterns correlate
   strongly with pathogenicity in this data, which is a known circularity
   risk in the variant-effect-prediction literature) and re-scoring the
   model without them, as an ablation.

**Finding:** three of four panels improved or held steady. **PAH broke** —
the paired-bootstrap rule picked a 7-model averaged ensemble as champion,
but which candidate "won" changed on almost every cross-validation fold
(4 different winners across 10 folds). The averaged ensemble's optimism gap
(OOF score minus honest nested score) more than tripled versus the
reference. This was a real, measured overfitting failure, not a hypothetical
one — and it is exactly the failure `run_best_search.py` was built to catch
systematically instead of by hand.

### 2.3 `run_best_search.py` — sweep everything, audit everything

Rather than trying one configuration at a time, this script is the
cross-product of every option `run_final.py` and `run_improved.py` each
fixed to a single choice:

- **Expensive axis** (needs retraining): feature-selection method
  (`null`/`hybrid`) × augmentation fraction (`0.0`/`0.5`/`1.0` — including,
  deliberately, trying **partial** augmentation on KANSER, which the
  reference pipeline had only ever tested at full strength before rejecting).
- **Cheap axis** (computed once per trained model, free): champion rule
  (`marginal`/`paired`) × candidate pool (`all` candidates / `singles`
  only) × threshold policy (`f1`/`balanced_f1`/`mcc`/`minimax`) — 16
  combinations, each re-evaluated **inside every nested cross-validation
  fold**, so the reported score is never a post-hoc cherry-pick.

Every one of the ~320 resulting configurations across the four panels was
run through an explicit **overfitting audit**, and any configuration that
failed was discarded outright:

1. **Optimism gap** (OOF F1 − honest nested F1) must be ≤ 0.040.
2. **Champion stability** — the same model must win in at least half of the
   outer folds. *(This is precisely the check that would have caught the
   PAH failure in `run_improved.py` before it shipped.)*
3. **Beats the naive baseline** (always-predict-pathogenic).

The surviving configurations were ranked by a **robustness objective**
(`min(honest F1, honest balanced-F1)` — a hedge against not knowing the true
test-set class balance), and the winner per panel was written to
`outputs_best/best_config.json` with a full human-readable justification in
`outputs_best/BEST_CONFIG.md`.

**Result: every panel's audited champion is a single tuned XGBoost model.**
No stacking, no averaging survived the audit anywhere — a genuinely
cross-validated argument for simplicity, not an assumption.

### 2.4 `PatHUgen.py` — the locked, final delivery script

This file takes the answer `run_best_search.py` found and **hard-codes it**.
There is no Optuna, no sweeping, no nested cross-validation here — those
decisions are closed. What remains is exactly what a competition submission
needs: a `train` command (fit the locked configuration from the training
CSVs) and a `deliver` command (score new test data, validate the result
against the official JSON schema, and package a complete, reproducible
delivery folder). See `HOW_TO_RUN.md` for the operational guide.

---

## 3. Why this order of scripts, not fewer

It would have been possible to skip straight to sweeping everything
(`run_best_search.py`) from the start. The staged approach mattered in
practice: `run_final.py` established a trustworthy baseline and the honest
nested-CV methodology; `run_improved.py` tested specific, reasoned
hypotheses cheaply and **surfaced a real overfitting failure on PAH** that
would otherwise have gone unnoticed; `run_best_search.py` then generalized
the fix (an explicit, automated overfitting audit) instead of patching PAH
by hand. `PatHUgen.py` is deliberately the simplest file in the project,
because every decision it encodes was already earned by the three before it.

---

## 4. Feature engineering (`build_features`)

Column names are anonymized, so features had to be engineered from
statistical signatures rather than domain-labeled columns:

- **AL_1–AL_26**: bounded in [0, 1], heavily right-skewed (median ~1e-4) →
  interpreted as population allele frequencies.
- **Remaining AL_ columns**: some binary {0,1} → in-silico "damaging" flags;
  the rest continuous → in-silico pathogenicity scores.
- **EK_1–EK_9**: continuous, range [-20, +11.93] → conservation/composite
  scores.
- **CAT_1–CAT_6**: low-cardinality categorical context.
- **AA_1 / AA_2**: reference/alternate amino acid letters.

91 features were engineered on top of the 351 raw columns (442 total),
falling into five groups:

1. **Missingness meta-features** — row-wise missing counts per group,
   `pct_missing`, a 5-tier `missingness_tier`, `is_skeleton`/`is_dense`
   flags. Missingness is *informative*, not noise: well-annotated (common,
   often benign) variants and sparsely-annotated (rare, often pathogenic)
   variants form two visibly different populations in this data.
2. **Amino-acid biochemistry** — BLOSUM45/62/80 substitution scores,
   Grantham distance, and eight physicochemical scales (hydropathy,
   molecular weight, volume, polarity, isoelectric point, charge,
   helix/sheet/turn propensity), each split into reference value, alternate
   value, signed difference, and absolute difference.
3. **In-silico voting statistics** — how many binary damaging-flag tools
   are available, how many voted "damaging", unanimity/conflict flags.
4. **AL/EK aggregates** — mean, max, std, log-transformed distribution
   shape, rare/ultra-rare frequency counters.
5. **Interaction features** — e.g. `missingtier_x_ekcount`,
   `ek_max_x_grantham` — designed from EDA/t-SNE/Orange Rank findings (see
   `plans/` and `report_figs/` for the exploratory work behind these).

All transforms are row-wise and label-independent, so the identical function
runs at training time and at inference time with no special-casing.

---

## 5. What each panel's model actually looks at

| Panel | Champion | Features | Augmented | Threshold | Policy |
|---|---|---|---|---|---|
| GENEL | XGBoost | 38 (hybrid selection) | No (source panel) | 0.670 | balanced_f1 |
| KANSER | XGBoost | 35 (null selection) | No (measurably hurts this panel) | 0.705 | f1 |
| PAH | XGBoost | 37 (null selection) | Yes, full | 0.595 | mcc |
| CFTR | XGBoost | 13 (null selection) | Yes, full | 0.780 | balanced_f1 |

Across all four panels' selected features, only four columns are shared by
three or more panels (`AL_7`, `AL_26`, `ek_max`, `al_n_ultrarare`) — the
panels are biologically distinct enough that almost nothing generalizes
across them, reinforcing why they are modeled independently rather than
pooled.

**Threshold policy differs per panel because it was audited per panel**, not
because of an arbitrary choice: `f1` maximizes F1 assuming the test set
matches the training class balance; `balanced_f1` and `mcc` hedge toward
robustness if the true test set turns out closer to 1:1. Whichever policy
survived the overfitting audit with the best robustness score for that
specific panel is what is locked into `PatHUgen.py`.

**Every champion is re-fit with 3 random seeds and the predictions
averaged** (`SEED_ENSEMBLE = 3` in `PatHUgen.py`) — this costs almost
nothing and reduces variance from a single lucky/unlucky tree-building draw,
which matters most on the smallest panel (CFTR, 111 rows).

---

## 6. Categorical encoding and why it is saved, not recomputed

`CAT_1..CAT_6` and the amino-acid columns are integer-coded for XGBoost
(`get_cat_maps` / `_encode_categorical`). The mapping from category value to
integer code is derived from the **training** data and must be applied
identically at inference time — if a code meant something different at
predict time than it meant during training, the model's learned splits
would silently misinterpret the input.

`run_final.py` re-derives this mapping from the training CSV every time it
predicts. **`PatHUgen.py` instead saves the mapping into the model bundle**
(`cat_maps.joblib`) at training time and reloads it at inference time. This
is a deliberate change for the competition environment: the delivery guide
requires the system to work with **no internet access**, using only files
present in the model package — it should not depend on the raw training
CSVs still being at a known path on competition day.

---

## 7. Honest evaluation and the overfitting audit

Two numbers are reported everywhere in this project, and they mean
different things:

- **OOF (out-of-fold) F1** — used to *choose* the model. Optimistic: feature
  selection and model choice were both allowed to see the whole panel.
- **Honest (nested) F1** — the number to *trust*. Feature selection, model
  choice, and threshold are re-decided inside every held-out outer fold, on
  training rows only, then scored once on the untouched fold.

The gap between them (`optimism_gap`) is the direct measure of overfitting.
`run_best_search.py`'s audit requires this gap to stay under 0.040 **and**
requires the winning model to actually win consistently across folds (not
just once, on the full panel) **and** requires it to beat the naive
always-pathogenic baseline. All four of `PatHUgen.py`'s locked
configurations passed all three checks — see `outputs_best/BEST_CONFIG.md`
for the full per-panel numbers and reasoning.

---

## 8. The delivery system

Per `TEKNOFEST_2026_Saglikta_Yapay_Zeka_Final_Uygulama_ve_Sonuc_Teslim_Kilavuzu.pdf`
(the official final-round guide), a submission has two independent parts:

1. **A result JSON**, uploaded to the competition's web system, structured
   per section 3.2 of the guide (one record per test variant: `id`, `panel`,
   `predicted_class` — the string `"0"` or `"1"` — and `predicted_prob`, the
   model's probability of the Pathogenic class).
2. **A USB-delivered model package**, sufficient for the jury to re-run the
   model and reproduce the submitted result, per section 6 of the guide.

`PatHUgen.py deliver` produces both from one command: it predicts on all
four panels, validates the result against the schema locally (catching
malformed output before it ever reaches the upload system), and builds a
numbered `deliveryN/` folder containing the JSON, the exact trained model
snapshot used, `PatHUgen.py` itself (the only code the guide's reproducibility
requirement actually calls for — this project's search and analysis scripts
played no part in producing the delivered result, so they are not included),
a frozen `requirements.txt`, and a rendered README documenting exactly how
that delivery was produced. Competition test data is never copied into a
delivery folder — the guide explicitly forbids this. Full operational detail
is in `HOW_TO_RUN.md`.

---

## 9. Project file map

```
PatHUgen.py                    <- FINAL delivery script (train + deliver)
Delivery_Readme.md             <- rendered into every delivery{N}/README.md
pathugen_model/<PANEL>/v1,v2.. <- trained snapshots; `train` adds, never overwrites (git-ignored)
delivery1/, delivery2/, ...    <- one per `deliver` run (git-ignored)

run_final.py                   <- reference pipeline (generation 1)
run_improved.py                <- hypothesis-testing pipeline (generation 2)
run_best_search.py             <- full sweep + overfitting audit (generation 3)
runlog.py                      <- shared run-logging (used by all three above)
outputs_best/BEST_CONFIG.md    <- the audited reasoning PatHUgen.py's config comes from

src/, train/                   <- earlier (pre-notebook) modular pipeline version
eda/, analysis/, report_build/ <- exploratory analysis + PSR report generation scripts
plans/, report_figs/           <- exploratory write-ups and figures behind the feature design

universite-veri-seti/          <- official training data (not delivered — organizer-provided)
```

For how to actually run any of this, see **`HOW_TO_RUN.md`**.
