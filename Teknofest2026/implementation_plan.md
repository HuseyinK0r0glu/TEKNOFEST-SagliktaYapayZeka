# Competition-Grade Implementation Plan
## TEKNOFEST 2026 — Variant Pathogenicity Classifier
### Revised After Expert Panel Review (Kaggle Grandmaster · Clinical Bioinformatician · ML Engineer)

> **Goal**: Maximize ROC-AUC / F1 on hidden test set (4 sub-datasets: General, Cancer, PAH, CFTR)

---

## Expert Panel Findings Summary

Before diving into the revised plan, the three reviewers identified the following critical issues in the original version:

**Kaggle Grandmaster Flags:**
- No pseudo-labeling strategy for the unlabeled test set (a top-5 standard move)
- No test-time augmentation or prediction averaging with Monte Carlo Dropout
- Meta-learner input is too thin (only 3 OOF columns) — risks underfitting the stacker
- No adversarial validation to detect train/test distribution shift
- Optuna budget of 200 trials is likely insufficient for the full hyperparameter space; recommend pruning with MedianPruner
- No rank-averaging of OOF predictions before stacking (normalizes scale differences between models)
- Missing: calibrated probability outputs (Platt/Isotonic) before stacking

**Clinical Bioinformatician Flags:**
- BLOSUM62 is correct but BLOSUM45/80 capture different evolutionary distances — test all three
- Missing: PolyPhen-2-style physicochemical feature set (volume change, polarity change, charge change)
- Missing: Nonsense-mediated decay (NMD) proxy: stop-gain variants near exon 3′ end are NMD-escaped and may be less severe — the plan treats all stop-gains as equally pathogenic, which is wrong
- CAT columns likely encode clinical database membership (ClinVar submitter, gnomAD filter status, inheritance mode) — target-encoding these without investigation is a leakage time bomb
- The plan never considers that PATHOGENIC missingness pattern may encode prior curation status, not biological signal — this requires explicit EDA validation
- AA physicochemical properties listed are incomplete: missing `secondary_structure_propensity` (helix/sheet/coil), `solvent_accessibility`, `conformational_flexibility` (B-factor proxy)
- Missing: codon change type proxy (transition vs transversion) even if DNA sequence is hidden — can be inferred from AA pair in some cases
- Missing: conservation score aggregation strategy — EK columns may include PhyloP/GERP; these have different directionality conventions (higher PhyloP = more conserved = more likely damaging)

**ML Engineer Flags:**
- No data versioning or checksums — if you re-run EDA, you cannot guarantee the CSV files haven't changed
- GroupKFold is not considered: if the same gene variant appears in both General and Cancer CSVs (after dedup by variant ID, not position), they may be biologically linked and should be grouped
- Reproducibility checklist is incomplete: no mention of OS/Python/library version locking (conda environment.yml), no Docker image for finals submission
- Target encoding noise (σ=0.01) is applied uniformly — too high for rare categories, too low for common ones; use adaptive noise proportional to 1/√count
- No monitoring of OOF calibration — stacking with poorly calibrated base probabilities degrades meta-learner performance
- Missing: a held-out "canary" validation set (10% stratified, never touched during development) to catch overfit to CV folds
- Threshold optimization grid should be applied to F1, MCC, AND Youden's J simultaneously, then ensemble the three threshold candidates

---

## Critical Data Insights (Updated)

| Finding | Impact on Strategy |
|---------|-------------------|
| **Bimodal missingness**: rows cluster at ~1-20 missing (well-annotated) vs ~300-351 missing (sparse) | Use `missing_count` as a top feature; train separate model heads per missingness tier |
| **Missingness is label-correlated**: Pathogenic avg 212 missing, Benign avg 145 missing | Missingness encodes curation status, not biology — never impute blindly; investigate before treating as a raw signal |
| **767 rows have ≥337/351 features missing** (only EK + AA available) | These "skeleton rows" need EK and AA features to carry all the weight; consider a dedicated sub-model |
| **Class ratio ~2.7:1 overall (PAH at 5:1)** | Use `scale_pos_weight`, class weights, and per-dataset threshold tuning |
| **Features come in triplets**: (score, binary_flag, p-value) pattern in AL columns | High collinearity — tree models handle this, but feature selection helps |
| **AA_1/AA_2**: Single-letter amino acid codes with `*` = stop codon | Grantham + BLOSUM45/62/80 + full physicochemical suite; NMD proxy for stop-gains |
| **4 sub-datasets with different class ratios** | Track per-dataset CV scores separately; never optimise only on global mean AUC |
| **Same variant may appear in multiple CSVs** | Deduplicate on (AA_1, AA_2, available EK/AL fingerprint) before any split; GroupKFold by variant fingerprint |
| **EK columns likely span different scales** (CADD 0-60, REVEL 0-1, etc.) | Rank-normalise per-column on training set only; save thresholds for test |
| **Missingness PATTERN, not just count, encodes biology** | Engineer pattern-based features: which tool groups are absent together |
| **Unlabeled test set available during finals** | Use pseudo-labeling and adversarial validation |

---

## Phase 0 — Environment Setup & Data Integrity

> This phase is new. Do NOT skip it. A corrupted or version-drifted input file invalidates all downstream work.

### 0.1 Reproducible Environment

```
# environment.yml (pin every version)
name: teknofest2026
channels: [conda-forge, defaults]
dependencies:
  - python=3.11.9
  - numpy=1.26.4
  - pandas=2.2.2
  - scikit-learn=1.5.0
  - xgboost=2.0.3
  - lightgbm=4.3.0
  - catboost=1.2.5
  - shap=0.45.1
  - optuna=3.6.1
  - imbalanced-learn=0.12.3
  - mlflow=2.13.0
  - joblib=1.4.2
  - scipy=1.13.0
```

Use `conda env export --no-builds > environment.yml` after installing. Commit to version control.

### 0.2 Data Integrity Checksums

```python
import hashlib

for csv_path in ['general.csv', 'kanser.csv', 'pah.csv', 'cftr.csv']:
    sha256 = hashlib.sha256(open(csv_path, 'rb').read()).hexdigest()
    print(f"{csv_path}: {sha256}")
# Save output to checksums.txt — rerun at start of every session to detect drift
```

### 0.3 Canary Validation Set

Before ANY modelling, extract a 10% stratified (label + source) hold-out set and save it to disk. This set is NEVER used in any training, CV, feature selection, threshold tuning, or encoding. It is touched exactly once: final model evaluation before submission.

```python
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
strat_key = df['label'].astype(str) + '_' + df['source']
for train_idx, canary_idx in sss.split(df, strat_key):
    canary = df.iloc[canary_idx]
    development = df.iloc[train_idx]

canary.to_parquet('data/canary_set.parquet')
development.to_parquet('data/development_set.parquet')
# All subsequent phases operate ONLY on development_set.parquet
```

### 0.4 Fold Index Persistence

Generate and save all CV fold indices before any modelling. Every subsequent script loads the same fold file. Never regenerate folds mid-project.

```python
from sklearn.model_selection import StratifiedKFold
import json

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_key = development['label'].astype(str) + '_' + development['source']

folds = []
for fold_idx, (tr, va) in enumerate(skf.split(development, strat_key)):
    folds.append({'fold': fold_idx, 'train': tr.tolist(), 'val': va.tolist()})

with open('data/fold_indices.json', 'w') as f:
    json.dump(folds, f)
```

### 0.5 Project Directory Structure

```
teknofest2026/
├── data/
│   ├── raw/                  # original CSVs (read-only after checksum)
│   ├── development_set.parquet
│   ├── canary_set.parquet
│   ├── fold_indices.json
│   └── checksums.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering_dev.ipynb
│   └── 03_adversarial_validation.ipynb
├── src/
│   ├── feature_engineering.py
│   ├── encoders.py           # all encoding logic with fit/transform separation
│   ├── cv_runner.py
│   ├── stacking.py
│   └── threshold_optimizer.py
├── outputs/
│   ├── models/
│   ├── oof_predictions/
│   ├── shap_plots/
│   └── experiment_log.csv
├── environment.yml
├── checksums.txt
└── README.md
```

---

## Phase 1 — EDA (`notebooks/01_eda.ipynb`)

### 1.1 Load & Merge

- Load all 4 CSVs, add `source` column (`master/kanser/pah/cftr`)
- Verify schema alignment: all 353 columns identical, same dtypes
- **Duplicate row check** (Critical): compute a fingerprint for each row using (AA_1, AA_2, all non-null EK values rounded to 4 dp). Flag rows sharing the same fingerprint across different CSVs. Decision options:
  - Option A: Keep only one copy (from most specific dataset — CFTR/PAH > Kanser > General)
  - Option B: Keep all copies with a `is_duplicated_across_source` flag
  - Option C: Use GroupKFold on variant fingerprint to ensure duplicates never appear in both train and val
  - **Recommended**: Option A + GroupKFold verification
- Check for within-dataset duplicates (exact row matches)

### 1.2 Missingness Profiling

- Per-column: `df.isnull().mean()` → identify always-null (drop), always-present (keep as-is), and partial columns
- Per-row: `df.isnull().sum(axis=1)` → histogram; bucket into tiers (0-20, 21-100, 101-200, 201-300, 301+)
- **Missingness × Label crosstab** — confirm Pathogenic variants have more missing data
- **Block missingness analysis**: use `missingno.matrix()` and hierarchical clustering of column null patterns. Identify which columns are always missing together. These blocks likely correspond to specific in-silico tools (e.g., SpliceAI, SIFT, PolyPhen) — each block's presence/absence is a binary feature
- **Missingness × Source crosstab** — check whether certain sources are systematically sparser
- **Critical investigation**: Is pathogenic missingness signal truly biological, or does it reflect curation status? Pathogenic variants that are well-studied will be in ClinVar and therefore will have MORE annotation (less missing). Variants that are pathogenic but rare/novel may have LESS annotation. Both patterns exist in genomic data. Your EDA must determine which dominates — this changes whether `n_missing_total` is a leakage risk or a legitimate signal.

### 1.3 CAT Column Semantic Investigation (Critical — Leakage Risk)

> This is a new subsection not in the original plan. It is critical.

Each CAT column must be investigated before any encoding decision:

- Count unique values and their frequencies
- Cross-tabulate each CAT value with Label: `pd.crosstab(df['CAT_1'], df['label'], normalize='index')`
- **Red flag**: If any CAT column has categories that are 95%+ Pathogenic or 95%+ Benign, it may be encoding ClinVar classification, database membership, or clinical interpretation status — effectively a proxy label
- Such columns must be handled with extreme caution:
  - If they encode prior pathogenicity assessment → **exclude from training** (they ARE the label, not a predictor)
  - If they encode population group membership → safe to use with target encoding
  - If unclear → include with a flag and run ablation with/without
- Document your decision for each CAT column in the technical report

### 1.4 Feature Type Classification

- Scan each AL column: if unique non-null values ∈ {0.0, 1.0} → tag as `binary_flag`
- Remaining AL columns: tag as `continuous_score`
- Count: ~how many binary flags vs continuous scores
- **EK column scale investigation**: For each EK column, check the range of non-null values. Known scales:
  - CADD: typically 0-60, with Phred-scaled values above 20 being damaging
  - REVEL: 0-1, above 0.5 is damaging
  - SIFT: 0-1, BELOW 0.05 is damaging (inverted directionality — critical for raw aggregation)
  - PhyloP: can be negative (accelerated evolution) to positive (conserved); higher = more damaging
  - GERP: 0-6+, higher = more conserved = more likely damaging
  - Note directionality differences before creating any consensus features

### 1.5 Adversarial Validation (`notebooks/03_adversarial_validation.ipynb`)

> New. A top-5 Kaggle move. Run this before building any model.

Train a binary classifier to distinguish training samples from test samples. If the classifier achieves AUC > 0.55, there is meaningful distribution shift between train and test.

```python
# Create adversarial labels: 0 = train, 1 = test
adv_df = pd.concat([development.assign(is_test=0), test_set.assign(is_test=1)])
# Train XGBoost to classify is_test
# Inspect feature importances — features that are most different between train and test
# Exclude or weight-down heavily shifting features
```

**Actions based on result:**
- AUC < 0.55: No significant shift. Proceed normally.
- AUC 0.55-0.65: Mild shift. Note which features differ most. Consider dropping or down-weighting.
- AUC > 0.65: Significant shift. Use importance-weighted sampling in CV. Reconsider which features to include.

### 1.6 Target-Correlated Features

- Point-biserial correlation of every feature vs Label
- Rank top-50 most correlated features
- EK features boxplot by label (expected to be top discriminators)
- **Check EK directionality**: for each EK column, does higher value correlate with Pathogenic (1) or Benign (0)? SIFT-like scores are inverted — document this

### 1.7 AA Substitution Analysis

- Pivot table: `AA_1` × `AA_2` × `mean(Label)` → which substitutions are most pathogenic
- `*` (stop codon) in AA_2 should be near-100% pathogenic — verify AND check position effect (early stop = more severe; late stop may escape NMD)
- `*` in AA_1: biologically invalid for missense variants — may indicate indels or frameshifts hidden as AA substitutions; flag and investigate
- `AA_1 == AA_2`: synonymous variants — expected near-100% benign; verify. Any synonymous variant labeled Pathogenic likely hits a splice site — note for report
- Non-standard codes (X, B, Z, U): note frequency; plan a `nonstandard_aa` flag
- **Transition vs transversion proxy**: certain AA pairs are only reachable via transitions (single nucleotide) vs transversions (requires two changes or specific mutations). This correlates with mutational probability. Encode as `mutation_class_proxy` = {transition_compatible, transversion_only, ambiguous}

### 1.8 Per-Dataset Analysis

- Repeat label distribution and missingness stats per source
- Check if feature distributions shift across sources (domain shift risk)
- Per-source class ratios: General ~2.7:1, PAH ~5:1 — note precisely
- PAH and CFTR are gene-specific panels; features relevant to ion channel function (CFTR) vs vascular biology (PAH) may differ — SHAP analysis later will confirm this

---

## Phase 2 — Feature Engineering (`src/feature_engineering.py`)

> All feature engineering must be implemented as a stateless transformation for inputs known at row level, and as a fit/transform pair for anything requiring training data statistics (EK rank normalisation, target encoding). The fit always happens on training fold data only.

### 2.1 Missingness Meta-Features (HIGH IMPACT)

| Feature | Formula |
|---------|---------|
| `n_missing_total` | Count NaN across all 351 features |
| `n_missing_al` | Count NaN in AL_* only |
| `n_missing_ek` | Count NaN in EK_* only |
| `pct_missing` | `n_missing_total / 351` |
| `missingness_tier` | Categorical bucket: sparse (≥301) / partial (101-300) / dense (0-100) |
| `has_population_data` | `CAT_1.notna()` (if CAT_1 confirmed as population label) |
| `has_allofus_data` | `CAT_2.notna()` (verify semantics in Phase 1) |
| `has_genotype` | `CAT_3.notna()` (verify semantics in Phase 1) |

**Missingness Block Features** (new — based on Phase 1.2 block analysis):

For each identified block of co-missing columns (e.g., SpliceAI tools always missing together, PolyPhen tools always missing together):

| Feature | Formula |
|---------|---------|
| `block_A_present` | Binary: the block-A tool group has any non-null values |
| `block_B_present` | Binary: the block-B tool group has any non-null values |
| `n_blocks_present` | Count of present tool blocks |
| `tool_coverage_profile` | String encoding of which blocks are present (e.g., "1011" → for CatBoost) |

**Missingness Pattern Similarity** (medium impact):

```python
# Encode missingness as a binary vector per row
missing_vec = df[AL_columns].isnull().astype(int)

# For sparse rows (≥301 missing), compute cosine similarity to the centroid
# of each label class's missingness pattern
# sparse_rows with missingness profile similar to typical-pathogenic-sparse → higher prior
```

### 2.2 Amino Acid Features (HIGH IMPACT)

**BLOSUM Family** — use all three matrices, not just BLOSUM62:

| Matrix | Evolutionary Distance | Best For |
|--------|-----------------------|---------|
| BLOSUM45 | Distant (30% identity) | Ancient conserved positions |
| BLOSUM62 | Moderate (62% identity) | Standard choice |
| BLOSUM80 | Close (80% identity) | Functionally constrained recent changes |

Features: `blosum45_score`, `blosum62_score`, `blosum80_score`, `blosum_mean`, `blosum_min`

**Grantham Distance Matrix** — physicochemical dissimilarity (0-215 scale):
- `grantham_score`: lookup `GRANTHAM[AA_1][AA_2]`
- High Grantham = radical substitution = more likely pathogenic
- `grantham_category`: {conservative <50, moderate 50-100, radical >100}

**Amino Acid Properties** — expanded set (for both AA_1 and AA_2):

| Property | AA_1 | AA_2 | Delta |
|----------|------|------|-------|
| Molecular weight | ✓ | ✓ | ✓ |
| Hydrophobicity (Kyte-Doolittle) | ✓ | ✓ | ✓ |
| Hydrophobicity (Eisenberg) | ✓ | ✓ | ✓ (second scale cross-check) |
| Isoelectric point (pI) | ✓ | ✓ | ✓ |
| Charge at pH 7.4 | ✓ | ✓ | ✓ |
| Van der Waals volume | ✓ | ✓ | ✓ |
| Polarity (Zimmermann) | ✓ | ✓ | ✓ |
| Secondary structure propensity (helix) | ✓ | ✓ | ✓ |
| Secondary structure propensity (sheet) | ✓ | ✓ | ✓ |
| Secondary structure propensity (coil) | ✓ | ✓ | ✓ |
| Solvent accessible surface area | ✓ | ✓ | ✓ |
| Conformational flexibility index | ✓ | ✓ | ✓ |

**Chemical Group Membership** (for AA_1 and AA_2):
- `is_polar`, `is_nonpolar`, `is_aromatic`, `is_aliphatic`, `is_charged_pos`, `is_charged_neg`, `is_tiny` (G, A, S)
- `group_change` = 1 if AA_1 and AA_2 belong to different chemical groups (strong pathogenicity signal)
- `charge_direction_change` = 1 if charge sign flips (e.g., positive → negative or neutral → charged)

**Delta Features** (absolute + signed):
- `delta_hydrophobicity`, `delta_mw`, `delta_pI`, `delta_charge`, `delta_volume`, `delta_polarity`
- `abs_delta_*` versions for each
- `max_abs_delta` across all properties (worst-case physicochemical disruption)

**Special Flags**:
- `is_stop_gain = (AA_2 == '*')` — nonsense mutation; DO NOT assume all are equally pathogenic
  - `stop_gain_severity_proxy`: encoded as `n_missing_total` rank within stop-gains (well-annotated stop-gains may be NMD-escaped)
- `is_stop_loss = (AA_1 == '*')` — biologically invalid for standard missense; flag as data quality and use as a feature cautiously
- `is_synonymous = (AA_1 == AA_2)` — expected near-100% benign; strong negative signal
- `aa1_is_nonstandard` / `aa2_is_nonstandard` — covers X, B, Z, U codes
- `is_same_group` — AA_1 and AA_2 belong to same chemical group?
- `transition_compatible` — is this AA substitution reachable via a single transition mutation?
- `aa_pair` — `AA_1 + "_" + AA_2` as categorical (for CatBoost native; target-encoded for others)

### 2.3 Population Frequency Aggregations

AL_1 through AL_26 are likely allele frequencies across populations.

> Population frequencies are extremely right-skewed. **Log-transform before aggregating** — raw `af_mean` is dominated by a handful of common variants and is nearly meaningless. Compute all aggregations on log-scale values.

| Feature | Formula |
|---------|---------|
| `af_log_max` | `log10(max(AL_1:AL_26) + 1e-10)` |
| `af_log_mean` | `mean(log10(AL_1:AL_26 + 1e-10))` ignoring NaN |
| `af_log_std` | `std(log10(AL_1:AL_26 + 1e-10))` |
| `af_log_min` | `log10(min(AL_1:AL_26) + 1e-10)` — rarest population frequency |
| `af_max_raw` | `max(AL_1:AL_26)` raw — keep for interpretability |
| `af_count_nonzero` | Count of non-null AL_1:AL_26 with value > 0 |
| `af_population_spread` | `af_log_max - af_log_min` — how much frequency varies across populations |
| `is_rare` | `af_max_raw < 0.001` |
| `is_very_rare` | `af_max_raw < 0.0001` |
| `is_common` | `af_max_raw > 0.01` |
| `af_missing_fraction` | Fraction of AL_1:AL_26 that are NaN |

> Rare variants (low population frequency) are much more likely to be pathogenic. `af_log_max` and `is_rare` will be top features.

### 2.4 In-Silico Score Consensus

Binary flag columns represent "tool X says damaging":

| Feature | Formula |
|---------|---------|
| `damaging_vote_count` | Sum of all binary flag columns (ignoring NaN) |
| `damaging_vote_ratio` | `damaging_vote_count / n_binary_flags_present` |
| `unanimous_damaging` | All non-null binary flags == 1 |
| `unanimous_benign` | All non-null binary flags == 0 |
| `n_tools_available` | Count of non-null binary flags |
| `benign_vote_count` | Count of binary flags == 0 |
| `conflicting_predictions` | `damaging_vote_count > 0 AND benign_vote_count > 0` |
| `tools_available_ratio` | `n_tools_available / total_binary_flag_columns` |

### 2.5 EK Score Features

EK_1 through EK_9 are likely composite pathogenicity/conservation scores with different scales and directionality.

**Critical pre-processing — investigate directionality in Phase 1.4 before computing:**

For each EK column `i`:
- If higher value = more pathogenic (CADD, REVEL, GERP style): direction = +1
- If lower value = more pathogenic (SIFT style): direction = -1, multiply values by -1 before rank-normalisation
- Save `ek_direction` mapping to `outputs/ek_direction_map.json`

**Rank-normalise each EK column to [0,1] percentile after direction correction:**

```python
# Fit on training fold only — save thresholds
ek_rank_thresholds = {}
for col in EK_columns:
    vals = X_train[col].dropna().values
    direction = ek_direction[col]
    vals_adj = vals * direction  # flip if needed
    ek_rank_thresholds[col] = {'sorted_vals': np.sort(vals_adj), 'direction': direction}
    X_train[col + '_ranked'] = scipy.stats.rankdata(vals_adj, method='average') / len(vals_adj)

# Apply to val/test using training ranks (no leakage)
for col in EK_columns:
    vals_test = X_test[col].values * ek_direction[col]
    X_test[col + '_ranked'] = np.searchsorted(ek_rank_thresholds[col]['sorted_vals'], vals_test) / len(ek_rank_thresholds[col]['sorted_vals'])
```

| Feature | Formula |
|---------|---------|
| `ek_mean_ranked` | Mean of direction-corrected percentile-ranked EK scores |
| `ek_max_ranked` | Max of ranked EK scores |
| `ek_min_ranked` | Min of ranked EK scores (weakest tool agreement) |
| `ek_range_ranked` | `ek_max_ranked - ek_min_ranked` (tool disagreement metric) |
| `ek_count` | Number of non-null EK values |
| `ek_unanimity` | Count of ranked EK columns above 0.5 (consensus above-median pathogenicity) |
| `ek_strong_damaging` | Count of ranked EK columns above 0.75 (strong damaging signal) |
| `ek_conflicting` | `ek_unanimity > 0 AND (ek_count - ek_unanimity) > 0` — tools disagree |
| `ek_std_ranked` | Standard deviation of ranked EK scores (disagreement magnitude) |

> Rank normalisation must be fit on the training set only. Save `ek_rank_thresholds` to disk for test-set application and reproducibility.

### 2.6 Interaction Features (MEDIUM IMPACT)

| Feature | Rationale |
|---------|-----------|
| `ek_max_ranked * grantham_score / 215` | High pathogenicity score + radical AA change (normalised) |
| `af_log_max * damaging_vote_ratio` | Frequency × damage consensus |
| `ek_max_ranked / (af_max_raw + 1e-10)` | Score-to-frequency ratio (high score + low freq = very suspicious) |
| `grantham_score * damaging_vote_ratio` | AA severity × tool consensus |
| `ek_mean_ranked * (1 - pct_missing)` | Pathogenicity score discounted by annotation completeness |
| `is_rare * ek_max_ranked` | Rare AND predicted damaging (strong combined signal) |
| `is_stop_gain * ek_max_ranked` | Stop-gain with high functional score |
| `n_tools_available * damaging_vote_ratio` | Weighted vote: more tools + higher ratio = more reliable |
| `missingness_tier_encoded * ek_mean_ranked` | Interaction: annotation density × score quality |

### 2.7 Missingness Tier Sub-Model Flag

- `is_skeleton_row` = 1 if `n_missing_total >= 337` (only EK + AA available)
- `is_dense_row` = 1 if `n_missing_total <= 20` (fully annotated)
- These flags will be used in Phase 4.6 (tier-specific model heads)

### 2.8 Dataset Source Features

- `is_master`, `is_kanser`, `is_pah`, `is_cftr`: binary flags — always safe
- `source_target_encoded`: K-fold target encoding (leak-proof, see Phase 3.3)
- Priority: binary flags are the safe baseline; target-encoded version is an experiment

---

## Phase 3 — Encoding & Missing Value Strategy

### 3.1 Missing Values — Dual Strategy

**Strategy A (Primary — for XGBoost/LightGBM):**
- **Do NOT impute.** Pass NaN directly — both libraries learn optimal split directions for missing values natively.
- This preserves the missingness signal without corrupting it.
- Verified: XGBoost `tree_method='hist'` and LightGBM both handle NaN natively.

**Strategy B (Secondary experiment):**
- Impute with `-999` sentinel (clearly out-of-distribution; tree models split on it separately)
- Also test: median imputation ON THE TRAINING FOLD ONLY + explicit `{col}_was_missing` binary indicator columns
- Do NOT use global median — recompute median per fold to avoid leakage

### 3.2 Categorical Encoding

| Column | Strategy | Rationale |
|--------|----------|-----------|
| `CAT_1` (~30 categories) | Target encoding (5-fold) IF not a leakage risk; else binary-flag top categories | Investigate semantic in Phase 1.3 first |
| `CAT_2` (~10 categories) | Target encoding (5-fold) IF safe | Same |
| `CAT_3`, `CAT_4` (4 genotypes) | One-hot encoding | Small cardinality; genotype labels are safe |
| `CAT_5` | Target encoding (5-fold) | Similar to CAT_1 |
| `CAT_6` | Investigate in Phase 1.3; likely label or drop | May be a string encoding of pathogenicity |
| `AA_1`, `AA_2` (20 AAs) | Label encoding + full property feature suite | Properties carry more signal than raw encoding |
| `aa_pair` (400 combos) | Target encoding (5-fold) | High cardinality; target encoding is appropriate |
| All categoricals for CatBoost | Pass raw strings as `cat_features` | CatBoost handles natively without manual encoding |

> **Leakage Warning on CAT columns**: Before target-encoding any CAT column, perform the crosstab analysis from Phase 1.3. If any category maps to >95% one class, that column is a proxy label — target encoding it is valid computationally but may give an overly optimistic CV score that collapses on the test set if the column encodes prior expert assessment not available for new variants.

### 3.3 Target Encoding Protocol (Leak-Proof)

```python
def target_encode_fold(X_train, X_val, col, y_train, n_folds=5, smoothing=10, noise_sigma=None):
    """
    Perform K-fold target encoding on col using training fold only.
    smoothing: additive smoothing parameter (tunable; higher = more shrinkage toward global mean)
    noise_sigma: if None, compute adaptively as 0.01 / sqrt(category_count)
    """
    global_mean = y_train.mean()
    
    # Inner K-fold on training data
    inner_kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    encoded_train = np.zeros(len(X_train))
    
    for inner_tr, inner_va in inner_kf.split(X_train, y_train):
        # Compute category stats on inner_tr
        cat_stats = y_train.iloc[inner_tr].groupby(X_train[col].iloc[inner_tr]).agg(['sum', 'count'])
        cat_stats['mean'] = cat_stats['sum'] / cat_stats['count']
        
        # Smoothing formula: weight toward global mean for rare categories
        cat_stats['smoothed'] = (
            (cat_stats['count'] * cat_stats['mean'] + smoothing * global_mean) /
            (cat_stats['count'] + smoothing)
        )
        
        # Apply to inner validation fold with adaptive noise
        mapping = cat_stats['smoothed'].to_dict()
        vals = X_train[col].iloc[inner_va].map(mapping).fillna(global_mean)
        
        if noise_sigma is None:
            # Adaptive noise: larger for rare categories
            counts = X_train[col].iloc[inner_va].map(cat_stats['count']).fillna(1)
            adaptive_sigma = 0.01 / np.sqrt(np.maximum(counts, 1))
            vals += np.random.normal(0, adaptive_sigma)
        else:
            vals += np.random.normal(0, noise_sigma, size=len(vals))
        
        encoded_train[inner_va] = vals
    
    # For validation/test: compute stats on full training fold, apply smoothing, no noise
    cat_stats_full = y_train.groupby(X_train[col]).agg(['sum', 'count'])
    cat_stats_full['mean'] = cat_stats_full['sum'] / cat_stats_full['count']
    cat_stats_full['smoothed'] = (
        (cat_stats_full['count'] * cat_stats_full['mean'] + smoothing * global_mean) /
        (cat_stats_full['count'] + smoothing)
    )
    mapping_full = cat_stats_full['smoothed'].to_dict()
    encoded_val = X_val[col].map(mapping_full).fillna(global_mean).values
    
    return encoded_train, encoded_val, mapping_full  # save mapping_full for test encoding
```

> The `count` in smoothing always refers to per-category count within the current inner training fold, not the global dataset. The `smoothing` parameter (default 10) is tunable — test {5, 10, 20}.

### 3.4 Probability Calibration Strategy

Before stacking, base model probabilities must be calibrated. Raw XGBoost/LightGBM/CatBoost outputs are not guaranteed to be well-calibrated, especially with class imbalance.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# After training each base model on each fold, calibrate OOF probabilities:
# Option A: Isotonic regression (non-parametric, better for large samples)
# Option B: Platt scaling / sigmoid (better for small samples)

# Calibrate OOF predictions before passing to meta-learner
for model_name, oof_proba in oof_predictions.items():
    fraction_of_positives, mean_predicted_value = calibration_curve(y_train, oof_proba, n_bins=10)
    # Plot calibration curve — if far from diagonal, apply calibration
    # Use CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
```

---

## Phase 4 — Model Training

### 4.1 Cross-Validation Framework

```
- Load fold indices from data/fold_indices.json (generated in Phase 0.4)
- Stratify on (label + source) combined — ensures PAH is proportionally represented in every fold
- For EVERY experiment: report mean ± std of AUC-ROC, F1, MCC across folds
- Also report per-source AUC for each fold (master / kanser / pah / cftr)
- Save OOF predictions (probabilities, not classes) for stacking
- Save per-fold models for test-time averaging
- For final model selection: use RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```

> **Stratify on label AND source combined** — With 4 sub-datasets at very different class ratios (PAH 5:1 vs overall 2.7:1), label-only stratification can produce folds with unrepresentative PAH ratios.

> **GroupKFold consideration**: If duplicate variants were found across datasets in Phase 1.1 and you chose to keep all copies, you MUST use GroupKFold where the group = variant fingerprint. This prevents the same variant appearing in both train and val, which would massively inflate CV scores.

### 4.2 Class Imbalance Handling

Test each independently; track which works best per sub-dataset:

| Method | Implementation | Notes |
|--------|---------------|-------|
| `scale_pos_weight` | `count_benign / count_pathogenic` ≈ **0.37** (NOT 2.7 — Pathogenic is majority class at 73%) | Verify your class ratio direction each time |
| `is_unbalance=True` | LightGBM built-in | Automatically sets weights |
| `auto_class_weights='Balanced'` | CatBoost built-in | |
| **SMOTE** (inside CV loop) | Fit on training fold only; never touch val fold | Synthetic minority = Benign here |
| **Borderline-SMOTE** | Better for overlapping classes | |
| **No resampling** | Test as baseline | Often best for tree models |

> **Class ratio correction**: With 73% Pathogenic and 27% Benign, the MINORITY class is Benign (0). `scale_pos_weight` in XGBoost weights the POSITIVE class (1 = Pathogenic = majority here). Since Pathogenic is majority, `scale_pos_weight = count_benign / count_pathogenic ≈ 0.37` DOWN-weights the majority. Verify your data's actual ratios at runtime — do not hardcode.

> **SMOTE leakage**: SMOTE must be applied INSIDE each CV fold — fit and transform only the training portion of that fold, then predict on the untouched validation fold. Applying SMOTE before splitting inflates CV scores.

### 4.3 XGBoost Configuration

**Initial strong defaults:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `objective` | `binary:logistic` | |
| `eval_metric` | `auc` | |
| `tree_method` | `hist` | Fast; GPU support with `device='cuda'` |
| `max_depth` | 6 | |
| `learning_rate` | 0.05 | |
| `n_estimators` | 2000 | With early stopping |
| `subsample` | 0.8 | |
| `colsample_bytree` | 0.6 | |
| `min_child_weight` | 5 | |
| `reg_alpha` | 0.1 | |
| `reg_lambda` | 1.0 | |
| `scale_pos_weight` | ~0.37 | count_benign / count_pathogenic; verify at runtime |
| `early_stopping_rounds` | 100 | |
| `random_state` | 42 | |

**Optuna search space (run AFTER feature selection — Phase 5):**

| Parameter | Range | Scale |
|-----------|-------|-------|
| `max_depth` | [3, 10] | int |
| `learning_rate` | [0.005, 0.3] | log |
| `subsample` | [0.5, 1.0] | uniform |
| `colsample_bytree` | [0.3, 1.0] | uniform |
| `colsample_bylevel` | [0.3, 1.0] | uniform (additional regularisation) |
| `min_child_weight` | [1, 20] | int |
| `gamma` | [0, 5] | uniform |
| `reg_alpha` | [1e-8, 10] | log |
| `reg_lambda` | [1e-8, 10] | log |
| `max_delta_step` | [0, 10] | uniform (helps with imbalance) |

Run **300 trials** minimum with `MedianPruner(n_startup_trials=20, n_warmup_steps=10)` to prune unpromising trials early. Objective: mean CV AUC-ROC on selected feature set.

### 4.4 LightGBM Configuration

**Initial defaults:**

| Parameter | Value |
|-----------|-------|
| `objective` | `binary` |
| `metric` | `auc` |
| `boosting_type` | `gbdt` |
| `num_leaves` | 63 |
| `learning_rate` | 0.05 |
| `n_estimators` | 2000 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.6 |
| `min_child_samples` | 20 |
| `is_unbalance` | True |
| `early_stopping_rounds` | 100 |
| `verbosity` | -1 |

**Optuna search space:**

| Parameter | Range | Scale |
|-----------|-------|-------|
| `num_leaves` | [15, 255] | int |
| `learning_rate` | [0.005, 0.3] | log |
| `min_child_samples` | [5, 100] | int |
| `subsample` | [0.5, 1.0] | uniform |
| `colsample_bytree` | [0.3, 1.0] | uniform |
| `reg_alpha` | [1e-8, 10] | log |
| `reg_lambda` | [1e-8, 10] | log |
| `min_split_gain` | [0, 1] | uniform |
| `path_smooth` | [0, 10] | uniform (underused; helps small datasets) |

Also test `boosting_type='dart'` as a separate experiment — DART uses dropout regularization and often gives different error profiles, improving ensemble diversity.

### 4.5 CatBoost Configuration

**Initial defaults:**

| Parameter | Value |
|-----------|-------|
| `loss_function` | `Logloss` |
| `eval_metric` | `AUC` |
| `depth` | 6 |
| `learning_rate` | 0.05 |
| `iterations` | 2000 |
| `l2_leaf_reg` | 3 |
| `auto_class_weights` | `Balanced` |
| `cat_features` | [CAT_1..6, AA_1, AA_2, aa_pair, tool_coverage_profile] |
| `early_stopping_rounds` | 100 |
| `random_seed` | 42 |

**Optuna search space:**

| Parameter | Range |
|-----------|-------|
| `depth` | [4, 10] |
| `learning_rate` | [0.01, 0.3] |
| `l2_leaf_reg` | [1, 10] |
| `bagging_temperature` | [0, 10] |
| `random_strength` | [0, 10] |
| `border_count` | [32, 255] |

Run two CatBoost variants:
- (a) Raw categoricals via `cat_features` (CatBoost native)
- (b) Fully encoded features with Grantham/BLOSUM/physicochemical suite
Compare OOF AUC and also OOF prediction correlation — use whichever is less correlated with XGBoost as the "more diverse" ensemble member.

### 4.6 Missingness-Tier Model Heads (Medium-High Impact)

Train separate base models for skeleton rows (≥337 missing):

```python
skeleton_mask = X['n_missing_total'] >= 337   # only EK + AA available
dense_mask    = X['n_missing_total'] <= 100   # well-annotated rows

# For skeleton rows: use only EK and AA features + missingness meta-features
# For dense rows: use all features
# Meta-learner sees tier-specific OOF predictions as additional base model outputs
```

This avoids the dense-row model being distorted by skeleton-row gradients, and vice versa.

### 4.7 Pseudo-Labeling (Advanced — Run After Initial Stacked Model)

> Top-5 Kaggle technique for situations where unlabeled test data is available.

```python
# Step 1: Train best stacked model on full development set
# Step 2: Predict on test set → get probabilities
# Step 3: Add high-confidence test predictions as pseudo-labels
#   - Keep test samples with prob > 0.90 (Pathogenic) or prob < 0.10 (Benign)
#   - Add to training set with pseudo-labels
# Step 4: Retrain model on (development + pseudo-labeled test) set
# Step 5: Re-predict on remaining unlabeled test samples
# Step 6: Iterate 1-2 times maximum (more iterations → confirmation bias)

# Key: use threshold 0.90/0.10, not 0.5. Only high-confidence pseudo-labels.
# Track how many pseudo-labels are added per class to avoid ratio shift.
```

Only apply pseudo-labeling if adversarial validation (Phase 1.5) shows meaningful train/test distribution shift, OR if it demonstrably improves canary set performance.

---

## Phase 5 — Feature Selection

> **Run Phase 5 BEFORE full Optuna tuning (Phase 4b).** Train quick default models first to identify the feature set, then run the expensive 300-trial Optuna search on selected features only. Doing Optuna on all features wastes compute budget.

**Revised execution order:**
1. Train quick default XGBoost/LightGBM (no Optuna) on all features → get importances
2. Run feature selection (steps 5.1–5.5 below)
3. Return to Phase 4 for full Optuna tuning on the selected feature set

### 5.1 Importance-Based Selection

1. Train default XGBoost on all features
2. Extract `gain`-based feature importance
3. Plot cumulative importance — identify the "elbow" (expect ~80-120 features carry 95% importance)
4. Create feature sets: `top_50`, `top_100`, `top_150`, `all_features`
5. Run CV on each set — diminishing returns beyond some point; pick the set where marginal gain < 0.001 AUC per 20 added features

### 5.2 Null Importance Test (Gold Standard)

1. Train model on real labels → get real importance scores
2. Shuffle labels randomly, retrain → get null importance scores
3. Repeat null **50 times** minimum (50 shuffles gives reliable 95th percentile estimates; 10 is insufficient)
4. Use `joblib.Parallel(n_jobs=-1)` — 50 shuffles is fast in parallel
5. Keep only features whose real importance > 95th percentile of null importance distribution
6. This eliminates features that appear important by chance (especially critical with 350+ features and 3,800 rows — high false discovery rate risk)

### 5.3 Boruta / Recursive Feature Elimination

- Run Boruta with Random Forest as estimator (n_estimators=200, max_iter=100)
- Use as a cross-check against importance-based selection
- If Boruta and importance ranking agree on a feature, high confidence to include
- If they disagree, test both with and without in ablation (Phase 9)

### 5.4 Correlation-Based Pruning

- For pairs with |Pearson correlation| > 0.95: keep the one with higher point-biserial correlation to Label
- Reduces multicollinearity without losing information
- Check both AL-AL pairs and engineered feature pairs (Grantham vs delta_hydrophobicity can correlate)

### 5.5 SHAP-Based Importance (After First Full CV Run)

- Once any model completes a full 5-fold CV, compute SHAP values on OOF predictions
- Use mean |SHAP| as feature importance ranking (more reliable than `gain` for correlated features)
- Cross-check against null importance result
- Final feature set = intersection of top-N by SHAP and top-N by null importance test

---

## Phase 6 — Stacking Ensemble

### 6.1 Level-0: Base Model OOF Predictions

```
For each base model M in {XGB_tuned, LGBM_tuned, LGBM_dart, CatBoost_raw_cats, CatBoost_encoded, XGB_skeleton, XGB_dense}:
    For each fold k in 5-fold:
        Train M on folds != k
        Predict probabilities on fold k → store as OOF[k]
        Predict probabilities on test set → store as TEST_PRED[k]
    
    OOF_M = concatenate all OOF[k]               # shape: (n_train,)
    TEST_M = mean of all TEST_PRED[k]             # shape: (n_test,)
```

> Before stacking: verify (1) OOF length = training set length, (2) no NaN in OOF predictions, (3) test predictions averaged across all 5 folds.

### 6.2 Calibrate OOF Predictions

Before building the stacking input matrix, calibrate each model's OOF probabilities using the OOF data itself:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# For each base model's OOF predictions:
iso_calibrator = IsotonicRegression(out_of_bounds='clip')
iso_calibrator.fit(oof_proba_uncalibrated, y_train)
oof_proba_calibrated = iso_calibrator.predict(oof_proba_uncalibrated)
test_proba_calibrated = iso_calibrator.predict(test_proba_uncalibrated)

# Save calibrator per model for test-time use
```

Plot calibration curves (fraction of positives vs mean predicted value) for each model before and after calibration. A well-calibrated model has points near the diagonal.

### 6.3 Diversity Check

Before stacking, verify models are diverse:
- Pearson correlation of calibrated OOF predictions: want < 0.95 between any pair
- Disagreement rate: `mean(pred_A != pred_B)` at threshold 0.5
- If two models are >0.97 correlated: drop the lower-CV one, OR add a diverse model

**Rank-normalize OOF predictions** before stacking to remove scale differences:
```python
from scipy.stats import rankdata
for model_name in oof_dict:
    oof_dict[model_name] = rankdata(oof_dict[model_name]) / len(oof_dict[model_name])
```

### 6.4 Level-1: Meta-Learner

**Stack input features:**
```python
# Primary: calibrated + rank-normalized OOF probabilities from all base models
# Extended: also include top-10 raw features by SHAP importance (standardized)
# Source flags: is_master, is_kanser, is_pah, is_cftr
# Missingness tier: missingness_tier_encoded

meta_features_train = np.column_stack([
    oof_xgb_cal, oof_lgbm_cal, oof_lgbm_dart_cal, 
    oof_catboost_raw_cal, oof_catboost_enc_cal,
    oof_xgb_skeleton_cal, oof_xgb_dense_cal,
    X_train[top10_shap_features].values,
    X_train[['is_pah', 'is_cftr', 'is_kanser', 'missingness_tier_encoded']].values
])
```

**Meta-learner options (test all, pick best by inner CV):**

| Meta-Learner | Why |
|-------------|-----|
| Logistic Regression (C tuned 0.01–100) | Simple, low overfit risk; good calibration |
| Ridge Classifier (alpha tuned) | L2 regularization; handles correlated OOF inputs |
| LightGBM (depth=2, 50 trees, min_child_samples=50) | Can learn non-linear model interactions; constrain heavily |
| Weighted average (Nelder-Mead) | Optimize ensemble weights directly on OOF; no learning |
| Rank averaging | Average rank-normalized predictions; parameter-free |

Train meta-learner with its own inner 5-fold CV on the stacked features (nested CV).

### 6.5 Extended Stacking (If Diversity Is Low)

If any two base models have OOF correlation > 0.97, add diverse models:

- **ExtraTreesClassifier** (high randomness, different error profile from boosting)
- **Logistic Regression with L1** on top-50 features (very different inductive bias)
- **Random Forest** (bagging vs boosting diversity)
- XGBoost trained on **feature subsets** (top-50 features only → different learned boundaries)

---

## Phase 7 — Threshold Optimization

### 7.1 Why Not 0.5?

With 2.7:1 class imbalance, the optimal decision boundary is NOT at 0.5. The competition metric determines the optimal threshold.

### 7.2 Multi-Metric Threshold Search

```python
for threshold in np.arange(0.05, 0.95, 0.005):
    y_pred = (oof_proba >= threshold).astype(int)
    results[threshold] = {
        'f1':       f1_score(y_true, y_pred),
        'mcc':      matthews_corrcoef(y_true, y_pred),
        'youden_j': recall_score(y_true, y_pred) + recall_score(y_true, y_pred, pos_label=0) - 1,
        'precision': precision_score(y_true, y_pred),
        'recall':   recall_score(y_true, y_pred),
    }

# Identify optimal threshold for each metric separately
t_optimal_f1   = max(results, key=lambda t: results[t]['f1'])
t_optimal_mcc  = max(results, key=lambda t: results[t]['mcc'])
t_optimal_yj   = max(results, key=lambda t: results[t]['youden_j'])

# Ensemble threshold candidates: use the mean of the three, or test all three
t_final = np.mean([t_optimal_f1, t_optimal_mcc, t_optimal_yj])
```

Report all three thresholds and their metric profiles. Select based on the confirmed competition metric.

### 7.3 Per-Dataset Thresholds (Standard Step — Not Optional)

Different sub-datasets have meaningfully different class ratios (PAH 5:1 vs overall 2.7:1). A single global threshold systematically underperforms on PAH.

```python
thresholds = {'global': t_final}

for source in ['master', 'kanser', 'pah', 'cftr']:
    mask = (source_labels == source)
    source_oof = oof_proba[mask]
    source_y   = y_true[mask]
    
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.005):
        f1 = f1_score(source_y, (source_oof >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    
    thresholds[source] = best_t

# Apply at prediction time:
# y_pred = np.where(source == 'pah', (proba >= thresholds['pah']).astype(int), ...)
```

Save `thresholds.json` for test-time use. At minimum, always test a PAH-specific threshold given its 5:1 ratio.

---

## Phase 8 — SHAP Analysis & Interpretability

### 8.1 Global SHAP

```python
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val, max_display=30)
```

**Expected top features:**
1. EK scores (direction-corrected, rank-normalised)
2. `n_missing_total` / `pct_missing` (if confirmed not leakage)
3. `grantham_score` / `blosum62_score` / `blosum_min`
4. `af_log_max` / `is_rare`
5. `damaging_vote_ratio`
6. `ek_std_ranked` (tool disagreement)
7. `delta_charge` / `charge_direction_change`

### 8.2 SHAP Dependence Plots

For top-5 features: dependence plot showing feature value vs SHAP value, colored by interaction feature. Reveals:
- Non-linear thresholds (e.g., "pathogenic when EK_1_ranked > 0.75")
- Interaction effects (e.g., "Grantham matters more when variant is rare AND ek_mean_ranked > 0.6")
- Directionality confirmations (EK columns should be monotonically positive in SHAP)

### 8.3 Per-Class SHAP

- What drives Pathogenic predictions?
- What drives Benign predictions?
- Cluster SHAP profiles to identify "types" of pathogenic variants (e.g., rare + radical AA change vs common + all tools damaging)

### 8.4 Per-Dataset SHAP

- Run SHAP separately on each source subset (master / kanser / pah / cftr)
- Check whether top features differ between sub-datasets
- PAH panel: expect conservation scores (EK) to dominate over population frequency (AF), because PAH variants are rare by definition
- CFTR panel: expect AA physicochemical features to be prominent (CFTR is highly sensitive to fold disruption)
- This analysis justifies source-specific model branches in future iterations and explains per-source performance gaps

### 8.5 Calibration Plots (New)

For the final stacked model:
- Plot calibration curve (fraction of positives vs mean predicted value, 10 bins)
- Plot reliability diagram per source
- If calibration is poor (ECE > 0.05), apply post-hoc calibration (Isotonic Regression on canary set)

### 8.6 Report-Ready Outputs

Save for TEKNOFEST report:
- `shap_summary_global.png` — beeswarm plot, top 30 features
- `shap_summary_per_dataset.png` — one panel per source (PAH especially important)
- `shap_dependence_top5.png` — 5 dependence plots with interaction coloring
- `shap_force_plot_examples.png` — 3 example predictions explained (one TP, one TN, one FP or FN)
- `calibration_curves.png` — before/after calibration per model
- Feature importance table (SHAP mean absolute value ranking, with 95% CI across folds)
- EK directionality map with biological interpretation

---

## Phase 9 — Ablation Studies

### 9.1 Feature Group Ablation

| Experiment | Features Used | Track |
|------------|--------------|-------|
| `EK_only` | EK_1..EK_9 direction-corrected | Baseline |
| `EK + AA_basic` | EK + Grantham + BLOSUM62 | +? |
| `EK + AA_full` | EK + full physicochemical suite | +? |
| `EK + AA_full + missing_meta` | Add missingness features | +? |
| `EK + AA_full + missing_meta + pop_freq` | Add population frequencies | +? |
| `all_engineered` | All engineered features | +? |
| `all_engineered + raw_AL` | Add all raw AL columns | +? (diminishing?) |
| `all_engineered + interactions` | Add interaction terms | +? |

Track **per-source AUC** for each experiment, not just global AUC — some feature groups may help overall while hurting PAH specifically.

### 9.2 BLOSUM Variant Ablation (New)

| Experiment | BLOSUM Version |
|------------|----------------|
| `BLOSUM45 only` | Distant evolution |
| `BLOSUM62 only` | Standard |
| `BLOSUM80 only` | Close evolution |
| `BLOSUM all three + mean + min` | Combined signal |

### 9.3 Model Architecture Ablation

| Experiment | Setup |
|------------|-------|
| Single XGBoost (tuned) | Best single model baseline |
| Single CatBoost (tuned) | Compare |
| Single LightGBM (tuned) | Compare |
| XGB skeleton + XGB dense | Tier-specific model heads |
| Soft voting (equal weights) | Simple ensemble |
| Soft voting (Nelder-Mead weights) | Optimized weights |
| Stacking (LR meta) | Linear stacking |
| Stacking (LGBM meta) | Non-linear stacking |
| Stacking + pseudo-labels | Advanced |

### 9.4 Encoding Ablation

| Experiment | Encoding |
|------------|----------|
| Raw NaN (no impute) | Tree-native missing |
| Sentinel -999 | Out-of-distribution marker |
| Median impute (fold-level) | Standard impute |
| Median impute + indicator cols | Impute + missingness flags |

### 9.5 Imbalance Strategy Ablation

| Experiment | Method |
|------------|--------|
| No correction | Raw imbalanced |
| `scale_pos_weight` (verify direction) | Reweight loss |
| SMOTE inside CV loop | Oversample minority (Benign) |
| Borderline-SMOTE | Smarter oversampling |
| Class-weighted loss | Per-sample weights |

### 9.6 EK Directionality Ablation (New)

| Experiment | EK Treatment |
|------------|--------------|
| Raw EK, no correction | Naive baseline |
| Rank-normalised, no direction correction | Normalise only |
| Direction-corrected + rank-normalised | Full treatment |
| Direction-corrected only | No normalisation |

This ablation validates that EK direction correction genuinely helps and is not just adding noise.

### 9.7 CAT Column Leakage Ablation (New)

| Experiment | CAT columns |
|------------|-------------|
| All CAT columns included | Maximum signal |
| CAT columns with <80% label purity only | Safe subset |
| No CAT columns | Lower bound |

If "All CAT" gives drastically higher CV AUC but "No CAT" collapses, the suspicious CAT columns are proxy labels. Verify on canary set.

---

## Phase 10 — Experiment Tracking

### 10.1 Tracking System

Use **MLflow** or a structured CSV:

| Column | Example |
|--------|---------|
| `experiment_id` | `exp_042` |
| `timestamp` | `2026-06-10 14:30` |
| `git_commit` | `a3f9b12` |
| `model_type` | `xgboost_v3` |
| `feature_set` | `all_engineered_v2` |
| `n_features` | `187` |
| `encoding` | `target_enc + native_nan` |
| `imbalance_method` | `scale_pos_weight` |
| `ek_direction_corrected` | `True` |
| `blosum_versions` | `45,62,80` |
| `cv_auc_mean` | `0.9342` |
| `cv_auc_std` | `0.0081` |
| `cv_auc_master` | `0.941` |
| `cv_auc_kanser` | `0.928` |
| `cv_auc_pah` | `0.887` |
| `cv_auc_cftr` | `0.912` |
| `canary_auc` | `0.931` (ONLY for final model) |
| `cv_f1_mean` | `0.8876` |
| `cv_mcc_mean` | `0.7234` |
| `threshold_global` | `0.42` |
| `threshold_pah` | `0.35` |
| `threshold_cftr` | `0.40` |
| `cat_leakage_risk` | `low / medium / high` |
| `pseudo_labels_used` | `False` |
| `notes` | `Added BLOSUM45/80; EK direction corrected` |

> Per-source AUC and per-source threshold columns are essential. A model with 0.93 mean AUC but 0.85 on PAH may lose to a model with 0.925 mean but 0.91 on PAH.

> The `canary_auc` column is filled ONLY for the final candidate model. If canary AUC diverges from CV AUC by > 0.02, you have overfit to CV folds — investigate before submitting.

### 10.2 Model Versioning

```
outputs/models/
├── xgb_v1_default/
│   ├── fold_0.json ... fold_4.json
│   ├── params.yaml
│   ├── oof_predictions.npy
│   ├── test_predictions_per_fold.npy
│   ├── calibrator_isotonic.pkl
│   └── metrics.json
├── lgbm_v3_tuned/
│   └── ...
├── ensemble_v2_stack/
│   ├── base_models/
│   ├── oof_stacked_input.npy        # rank-normalized + calibrated OOF
│   ├── meta_learner.pkl
│   ├── calibration_curves.png
│   └── metrics.json
├── encoders/
│   ├── ek_rank_thresholds.json       # EK normalisation (train-set percentiles)
│   ├── target_encoding_mappings/     # per-fold, per-column
│   │   ├── fold_0_CAT_1.json
│   │   └── ...
│   └── thresholds.json               # global + per-source optimal thresholds
└── feature_selection/
    ├── null_importance_results.csv
    ├── boruta_results.csv
    ├── shap_importance_ranking.csv
    └── final_feature_list.txt
```

### 10.3 Reproducibility Checklist

- [ ] All random seeds set: `SEED=42` in numpy, random, sklearn, xgboost, lightgbm, catboost, torch (if used)
- [ ] `environment.yml` with pinned versions committed to git
- [ ] Data checksums verified at session start (checksums.txt)
- [ ] CV fold indices loaded from `data/fold_indices.json` (never regenerated)
- [ ] Feature engineering is deterministic (no randomness except target encoding noise)
- [ ] EK direction map saved to `outputs/encoders/ek_direction_map.json`
- [ ] EK rank normalisation thresholds saved from training set; applied to test using saved values
- [ ] Target encoding mappings saved per fold in `outputs/encoders/target_encoding_mappings/`
- [ ] Calibration isotonic models saved per base model in `outputs/models/{model}/calibrator.pkl`
- [ ] OOF predictions saved before and after calibration
- [ ] Canary set never touched until final evaluation
- [ ] Duplicate-row check completed and documented in EDA notebook
- [ ] CAT column semantic investigation documented with leakage assessment
- [ ] EK directionality investigation documented with per-column findings
- [ ] Adversarial validation completed; AUC recorded
- [ ] Best model can be loaded and re-run on test set identically (write a `predict_test.py` script)
- [ ] Pseudo-labeling (if used) is documented with number of pseudo-labels per class

---

## Execution Priority

```mermaid
graph LR
    A["Phase 0: Setup + Canary<br/>0.5 day"] --> B["Phase 1: EDA + CAT Audit + AdvVal<br/>1.5 days"]
    B --> C["Phase 2: Feature Eng<br/>2 days"]
    C --> D["Phase 3: Encoding<br/>0.5 day"]
    D --> E["Phase 4a: Default Models<br/>0.5 day"]
    E --> F["Phase 5: Feature Selection<br/>1 day"]
    F --> G["Phase 4b: Optuna Tuning<br/>2.5 days"]
    G --> H["Phase 6: Calibration + Stacking<br/>1.5 days"]
    H --> I["Phase 7: Threshold Opt<br/>0.5 day"]
    I --> J["Phase 8: SHAP<br/>1 day"]
    J --> K["Phase 9: Ablation<br/>2 days"]
    K --> L["Phase 7b: Pseudo-Labels<br/>0.5 day"]
    L --> M["Phase 10: Final Sub<br/>0.5 day"]
```

**Critical path**: Phases 0→1→2→3→4a→5→4b→6→7 (minimum viable competition submission)

**Total estimated time**: ~15 days for full pipeline

---

## Top 10 Priority Ranking (ROI per hour invested)

| Rank | Action | Expected AUC Impact | Phase |
|------|--------|---------------------|-------|
| 1 | **CAT column semantic audit** — Identify and exclude any CAT columns that encode prior pathogenicity assessment. A leakage-contaminated CV score will mislead all downstream decisions. | Can prevent catastrophic test-set collapse | Phase 1.3 |
| 2 | **EK directionality correction** — SIFT-like columns are inverted; raw aggregation actively hurts. Correcting direction before rank-normalisation is free and guaranteed to improve EK-based features. | +0.01–0.03 AUC | Phase 2.5 |
| 3 | **Adversarial validation** — If train and test distributions differ meaningfully, your CV score is lying to you. Know this before training 300 Optuna trials. | Prevents wasted compute; +? AUC indirectly | Phase 1.5 |
| 4 | **Full physicochemical AA suite + all 3 BLOSUM matrices** — Beyond Grantham/BLOSUM62, the additional properties (secondary structure propensity, solvent accessibility, volume, charge direction) encode known pathogenicity mechanisms. Free signal. | +0.005–0.02 AUC | Phase 2.2 |
| 5 | **Missingness block features** — The PATTERN of which tools are absent (not just the count) likely encodes variant type. Rare/novel variants will be missing specific tool groups systematically. | +0.005–0.015 AUC | Phase 2.1 |
| 6 | **Probability calibration before stacking** — Poorly calibrated base probabilities degrade the meta-learner. This is a 30-minute addition that prevents a silent performance drain. | +0.003–0.01 AUC | Phase 6.2 |
| 7 | **Per-source threshold optimization** — PAH at 5:1 class ratio guarantees the global threshold is suboptimal for PAH. Separate thresholds are not optional given the competition evaluates on all sub-datasets. | +0.01–0.03 F1 | Phase 7.3 |
| 8 | **Null importance test (50 shuffles)** — With 350+ features and 3,800 rows, the false discovery rate for feature importance is high. Null importance test is the gold-standard filter. | +0.005–0.02 AUC (by removing noise features) | Phase 5.2 |
| 9 | **Canary set** — Without a held-out set that's never touched, you cannot detect overfit to CV folds before submission. One check on the canary at the end prevents a silent overfit disaster. | Prevents test-set surprise | Phase 0.3 |
| 10 | **Pseudo-labeling (high-confidence only)** — If adversarial validation reveals distribution shift, pseudo-labeling bridges the gap. If no shift, the gain is modest but non-negative. | +0.003–0.015 AUC | Phase 4.7 |

---

## Summary of All Changes vs Original Plan

| Area | Change | Source |
|------|--------|--------|
| **Phase 0 (new)** | Environment setup, data checksums, canary set, fold persistence | ML Engineer |
| Phase 1.1 | Stronger duplicate check with GroupKFold guidance | Kaggle GM |
| **Phase 1.3 (new)** | CAT column semantic audit for leakage risk | Clinical Bioinformatician |
| **Phase 1.5 (new)** | Adversarial validation before any modelling | Kaggle GM |
| Phase 1.4 | EK directionality investigation (SIFT is inverted) | Clinical Bioinformatician |
| Phase 1.7 | Added transition/transversion proxy; NMD context for stop-gains | Clinical Bioinformatician |
| Phase 2.1 | Missingness BLOCK features (pattern, not just count) | Clinical Bioinformatician + Kaggle GM |
| Phase 2.2 | BLOSUM45/80 added alongside BLOSUM62 | Clinical Bioinformatician |
| Phase 2.2 | Expanded physicochemical suite: secondary structure propensity, solvent accessibility, volume, conformational flexibility | Clinical Bioinformatician |
| Phase 2.2 | NMD proxy for stop-gains; transition/transversion proxy | Clinical Bioinformatician |
| Phase 2.5 | EK directionality correction before rank-normalisation | Clinical Bioinformatician |
| Phase 2.5 | Added `ek_conflicting`, `ek_std_ranked`, `ek_strong_damaging` | Kaggle GM |
| **Phase 2.7 (new)** | Missingness-tier sub-model flags for skeleton rows | Kaggle GM |
| Phase 3.2 | Strengthened CAT column leakage warning with explicit investigation reference | ML Engineer |
| Phase 3.3 | Adaptive noise (σ ∝ 1/√count) instead of uniform σ=0.01 | ML Engineer |
| **Phase 3.4 (new)** | Probability calibration strategy (Isotonic before stacking) | ML Engineer |
| Phase 4.1 | Load fold indices from disk; GroupKFold when duplicates present | ML Engineer |
| Phase 4.2 | Corrected scale_pos_weight direction (Benign is minority at 27%) | All three |
| Phase 4.3 | Increased Optuna to 300 trials with MedianPruner; added colsample_bylevel | Kaggle GM |
| **Phase 4.6 (new)** | Missingness-tier model heads (separate models for skeleton vs dense rows) | Kaggle GM |
| **Phase 4.7 (new)** | Pseudo-labeling protocol | Kaggle GM |
| Phase 5.2 | Null importance parallelism; 50 shuffles minimum | ML Engineer |
| **Phase 5.5 (new)** | SHAP-based importance as final arbiter for feature selection | Kaggle GM |
| Phase 6.1 | Extended base model set (LGBM DART, CatBoost raw vs encoded, tier-specific) | Kaggle GM |
| **Phase 6.2 (new)** | OOF calibration with Isotonic Regression before stacking | ML Engineer |
| Phase 6.3 | Added rank-normalization of OOF predictions before stacking | Kaggle GM |
| Phase 6.4 | Extended meta-features: source flags, missingness tier | Kaggle GM |
| Phase 7.2 | Multi-metric threshold (F1 + MCC + Youden's J simultaneously) | ML Engineer |
| Phase 8.5 | Added calibration plots as report output | ML Engineer |
| **Phase 9.2 (new)** | BLOSUM variant ablation | Clinical Bioinformatician |
| **Phase 9.6 (new)** | EK directionality ablation | Clinical Bioinformatician |
| **Phase 9.7 (new)** | CAT column leakage ablation | Clinical Bioinformatician + ML Engineer |
| Phase 10.1 | Added git_commit, canary_auc, cat_leakage_risk, pseudo_labels_used columns | ML Engineer |
| Phase 10.2 | Added encoder artifacts to model versioning | ML Engineer |
| Phase 10.3 | Added EK direction map, calibration models, CAT audit to reproducibility checklist | ML Engineer |
| Top-10 Priority | Reranked based on expert consensus; CAT audit and EK directionality moved to #1 and #2 | All three |
