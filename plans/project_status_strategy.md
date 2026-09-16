# TEKNOFEST 2026 — Project Status & Strategy

> Binary classification of genetic variants (Pathogenic / Benign) across four independent panels.
> After the Orange feature-ranking investigation and an overnight thorough-profile run (80 Optuna trials, 5 CV repeats), here is what the numbers say and where the real leverage is.

---

## What was done

| Step | Name | Summary |
|------|------|---------|
| 01 | Orange Visualization | Exported all 4 panels as 441-feature Orange-compatible CSVs. Ran Rank widget (Info Gain, ANOVA, ReliefF) and t-SNE per panel. KANSER: best-separated. GENEL: worst. Detected **negative-ReliefF pattern** in GENEL's AL features. |
| 02 | Feature Filters (in-code) | Built `src/feature_filters.py` reproducing all three Orange rankings inside the pipeline so selection logic can gate on ReliefF sign, not just Info Gain. No new dependencies. |
| 03 | Multi-Filter Variants | Three feature-set candidates: `null` (current baseline), `union` (consensus adds), `relief_pruned` (null minus negative-ReliefF features). Applied as post-hoc filter on the null-importance keep-set. |
| 04 | Holdout A/B | Per-panel 80/20 stratified split. Every method trained on train rows only; scored once on the same held-out test. All three methods compared apples-to-apples (threshold: +0.005 to call a winner). |
| 05 | Overnight run | Full 7-step pipeline: baseline → feature selection → Optuna → stacking → finalize → nested eval → holdout A/B. All 5 base models tuned per panel. |

---

## Honest performance — trust the nested F1, not the OOF

OOF F1 is optimistic: feature selection, hyperparameters, threshold, and champion strategy all peeked at every row.  
Nested CV is the leak-free estimate: for each outer fold, selection and tuning run on training rows only, scored once on untouched test rows.

| Panel | Rows | Nested F1 | OOF F1 | AUC | Balanced F1 | Trivial baseline | Champion | Optimism gap |
|-------|------|-----------|--------|-----|-------------|-----------------|----------|-------------|
| GENEL | 2931 | **0.8943** | 0.8994 | 0.851 | 0.778 | 0.846 | catboost | +0.005 ✓ reliable |
| KANSER | 388 | **0.9153** | 0.9348 | 0.923 | 0.856 | 0.817 | xgboost | +0.020 moderate |
| PAH | 372 | **0.9078** | 0.9325 | **0.793 ⚠** | **0.715 ⚠** | 0.909 | lightgbm | +0.025 — AUC is low, test-balance risk |
| CFTR | 111 | **0.9196** | 0.9780 | 0.879 | 0.779 | 0.896 | lightgbm | **+0.058 ⚠ OOF is a mirage** |

**Every champion was a single model — stacking never won** the one-standard-error selection on any panel this run. Tuned individual boosted trees outperform the calibrated meta-ensemble on all four panels.

---

## The Orange / ReliefF investigation — what it found

> **Info Gain misleads. ReliefF discriminates.**

Adding KANSER's top-4 IG features (`al_log_range`, `al_log_max`, `al_cont_max`, `n_missing_cat`) regressed held-out F1 from **0.9326 → 0.9314** because they were correlated redundancies of features already in the model. ReliefF caught this.

The *sign* of ReliefF is the guard: features with **negative ReliefF** are locally anti-informative — knowing their value makes nearest-neighbour classification harder, not easier. GENEL had 10+ such features in its null-importance set.

### Holdout A/B results — three methods, same test rows

80/20 stratified split, selection on train only.

| Panel | Method | # feat | Test F1 | vs null | Test AUC |
|-------|--------|--------|---------|---------|---------|
| GENEL | null (current) | 52 | 0.8926 | — | 0.865 |
| GENEL | union | 52 | 0.8926 | ±0.0000 | 0.865 |
| GENEL | **relief_pruned** | 42 | **0.8948** | **+0.0022** | 0.867 |
| KANSER | null (current) | 36 | 0.8909 | — | 0.869 |
| KANSER | union | 39 | 0.8785 | −0.0124 | 0.870 |
| KANSER | **relief_pruned** | 22 | **0.9107** | **+0.0198 ✓** | 0.874 |
| PAH | null (current) | 28 | 0.9104 | — | 0.759 |
| PAH | union | 34 | 0.9008 | −0.0096 | 0.741 |
| PAH | **relief_pruned** | 23 | **0.9147** | **+0.0043** | 0.715 |
| CFTR | null (current) | 22 | 0.8947 | — | 0.645 |
| CFTR | union | 33 | 0.8947 | ±0.0000 | 0.632 |
| CFTR | relief_pruned | 21 | 0.8947 | ±0.0000 | 0.645 |

**Pattern:** `relief_pruned ≥ null` everywhere · `union ≤ null` everywhere.  
Adding high-IG features consistently hurts or is neutral. Removing negative-ReliefF features consistently helps or is neutral.  
The KANSER win (+0.020) is the only margin large enough to recommend unconditionally. PAH and GENEL are borderline (± noise on a single split).

---

## Strategies to boost the score

### [YOUR IDEA] Hybrid selector — most promising architectural change

Right now, null-importance runs first and produces a keep-set, then `relief_pruned` removes negative-ReliefF features as a post-hoc filter. The issue: the two stages use different LightGBM models on the same data, and the post-hoc filter may drop features that the null-importance model valued because of cross-feature interactions.

**Running them together would be stronger**: let null-importance produce its `gain_score` ranking, then intersect with ReliefF sign as a joint condition at decision time. This prevents a feature from surviving purely because it is top-K by gain_score if its ReliefF is negative.

```
CURRENT:   null_importance → keep_set  →  [post-hoc] relief_pruned removes negatives

PROPOSED:  null_importance gain_score  ×  (ReliefF ≥ 0 OR top-K by IG)  =  joint keep_set
           ↑                                              ↑
       gate: actual > null_p95                    NEW: also require local validity
       topk: floor always on
           ↓
       keep = (gate | topk) AND (relieff_positive OR ig_protected)
```

**Implementation:** pass the filter table from `feature_filters.all_filters()` into `null_importance_selection()` and add a `relief_mask` parameter that gates the topk floor. A feature only survives the topk floor if it also has non-negative ReliefF — unless it is strongly protected by Info Gain (`ig_protected`), so globally predictive features are never silently dropped.

Test it through `run_holdout_compare.py` as a fourth method before shipping:
```bash
HOLDOUT_METHODS=null,hybrid PYTHONPATH=. .venv/bin/python train/run_holdout_compare.py
```

---

### 1 · relief_pruned for KANSER — confirmed winner, ship now

The only A/B win above noise (+0.020, 22 features vs 36). 14 negative-ReliefF AL features dropped.

Before shipping, confirm with a full nested eval:
```bash
SELECTION_METHOD=relief_pruned PIPELINE_PROFILE=thorough python main.py KANSER 2>&1 | tee relief_pruned_kanser.log
```
If nested F1 holds ≥ 0.915, ship it. If it drops, the single-split win was lucky.

---

### 2 · Hybrid selector — next experiment, high value

Implement the joint keep condition in `src/feature_selection.py`, add it as method `"hybrid"` in `multi_filter_selection.py`, then test with `run_holdout_compare.py`. Expected to pick up KANSER-style gains for the other panels too, since the post-hoc filter stage is the current weak point. The hybrid test does not cost a full pipeline run.

---

### 3 · Symmetric class weights for PAH

PAH: 310 pathogenic / 62 benign (83% positive rate), AUC 0.793, balanced F1 only 0.715. The model currently uses `scale_pos_weight = n_neg/n_pos = 0.2` which down-weights pathogenic predictions. Try `scale_pos_weight=1.0` (equal weights) or even up-weighting benign detection. With AUC this low, the model is barely ranking classes — equal weights may help it learn an actual decision boundary rather than defaulting toward the majority.

---

### 4 · Tighten the meta-learner or swap it

The calibrated stack with logistic meta-learner has never beaten a tuned single model on the one-SE rule in any run. Two options:

- **(a)** The meta-LR may be under-regularized — try `C=0.01` instead of default 1.0 in `fit_final_stack()`.
- **(b)** Ensemble from *different feature sets*: base model A on null features, base model B on relief_pruned features, meta over both. Diversity in the input space may unlock the stack.

---

### 5 · Missingness as a stratification signal (experimental — PAH / GENEL)

Orange's Rank widget for PAH showed `n_missing_al`, `pct_missing`, `n_missing_total` in the top 10 by Info Gain. The t-SNE "branches" for GENEL and distinct clusters for CFTR both suggest variants group by which annotation tools covered them. Consider:

- A `missingness_tier × ek_count` interaction feature
- A hard flag: "tool profile group" (e.g. 0 EK scores AND 0–5 AL scores = skeleton variant) used as a split-point feature or routing condition to a sub-model

---

### 6 · Threshold sensitivity and balanced F1 floor

The competition test set class balance is unknown. PAH's balanced F1 is 0.715 — if the test set is 50/50 instead of 83% pathogenic, PAH's real score collapses. After locking a champion, sweep thresholds in [0.3, 0.7] and plot F1 vs threshold. For panels where the F1 curve is flat (KANSER), the threshold is robust. For panels where it is sharp (PAH), consider setting the threshold slightly more conservatively toward the center.

---

## Recommended next actions (priority order)

1. `SELECTION_METHOD=relief_pruned PIPELINE_PROFILE=thorough python main.py KANSER` — confirm before shipping
2. Implement hybrid selector in `src/feature_selection.py` + test via `run_holdout_compare.py`
3. Try `scale_pos_weight=1.0` for PAH via Optuna space or direct override
4. If stacking still never wins: try meta-LR `C=0.01` or multi-feature-set ensemble diversity
