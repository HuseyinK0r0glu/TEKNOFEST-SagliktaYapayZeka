"""
data_setup.py
=============
Phase 0 (Environment / Data Integrity) + the data-prep half of Phase 1.

Responsibilities
----------------
* ``load_raw_data``        : read the 4 CSVs, tag each with its ``source``, merge.
* ``compute_checksums``    : SHA-256 of every raw CSV -> ``outputs/checksums.txt``
                             (re-run each session to detect input drift).
* ``add_variant_fingerprint`` : stable per-variant grouping key + cross-source
                             duplicate flag.
* ``build_canary_split``   : 10% stratified (label + source) hold-out, with an
                             optional group-disjoint guard so the canary stays
                             truly leak-free. Saves canary + development parquets.
* ``build_fold_indices``   : 5-fold ``StratifiedGroupKFold`` (group = fingerprint,
                             stratify = label + source) -> ``fold_indices.json``.
* ``run_phase0``           : orchestrates the whole Phase-0 pipeline end to end.

Why ``variant_fingerprint == Variant_ID``
-----------------------------------------
The plan proposes a content hash (AA_1, AA_2, rounded EK) as the fingerprint.
Phase-1 EDA showed that hash is the *wrong* grouping key for this data:
  * it splits 567 rows that share a ``Variant_ID`` into different groups, and
  * it collides ~370 sparse/skeleton rows (identical missing EK + AA) into ONE
    giant group spanning many distinct variants.
``Variant_ID`` is clean (3,224 unique, 0 within-ID label conflicts), so it is
used as the fingerprint. The content hash is still provided
(``compute_content_fingerprint``) for ablation.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from .config import (
    CANARY_FRACTION,
    CANARY_PATH,
    CHECKSUMS_PATH,
    DEVELOPMENT_PATH,
    DUP_FLAG_COL,
    FINGERPRINT_COL,
    FOLD_INDICES_PATH,
    ID_COL,
    LABEL_COL,
    MERGED_PATH,
    N_FOLDS,
    RAW_DIR,
    RAW_FILES,
    SEED,
    SOURCE_COL,
    ensure_dirs,
    set_global_seed,
)


# =========================================================================== #
# 0.2  Data integrity
# =========================================================================== #
def compute_checksums(raw_dir: str = RAW_DIR, out_path: str = CHECKSUMS_PATH) -> Dict[str, str]:
    """SHA-256 each raw CSV and persist to ``checksums.txt``. Re-run every session."""
    ensure_dirs()
    checksums: Dict[str, str] = {}
    lines = []
    for src, fname in RAW_FILES.items():
        path = os.path.join(raw_dir, fname)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        checksums[fname] = digest
        lines.append(f"{fname}\t{digest}")
        print(f"[checksum] {fname}: {digest}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return checksums


# =========================================================================== #
# 1.1  Load & merge
# =========================================================================== #
def load_raw_data(raw_dir: str = RAW_DIR, verify_schema: bool = True) -> pd.DataFrame:
    """Load the 4 CSVs, append a ``source`` column, and concatenate.

    All four share an identical 353-column schema (verified in EDA); we assert it
    so a silently changed file is caught immediately.
    """
    frames, ref_cols = [], None
    for src, fname in RAW_FILES.items():
        df = pd.read_csv(os.path.join(raw_dir, fname))
        if verify_schema:
            if ref_cols is None:
                ref_cols = list(df.columns)
            elif list(df.columns) != ref_cols:
                raise ValueError(f"Schema mismatch in {fname} vs reference columns.")
        df[SOURCE_COL] = src
        frames.append(df)
        print(f"[load] {src:8s} {fname:26s} shape={df.shape}")
    merged = pd.concat(frames, ignore_index=True)
    print(f"[load] merged shape={merged.shape} "
          f"| pos_rate={merged[LABEL_COL].mean():.4f}")
    return merged


# =========================================================================== #
# 1.1  Variant fingerprint & duplicate flag
# =========================================================================== #
def compute_content_fingerprint(df: pd.DataFrame, n_round: int = 4) -> pd.Series:
    """Plan's content-based hash (AA_1, AA_2, sorted non-null EK rounded).

    Provided for ablation only -- NOT the default grouping key (see module docs).
    """
    ek_cols = sorted([c for c in df.columns if c.startswith("EK_")],
                     key=lambda c: int(c.split("_")[1]))

    def _fp(row) -> str:
        aa = f"{row.get('AA_1')}>{row.get('AA_2')}"
        parts = [f"{c}:{round(float(row[c]), n_round)}" for c in ek_cols if pd.notna(row[c])]
        return hashlib.md5((aa + "|" + "|".join(parts)).encode()).hexdigest()[:16]

    return df.apply(_fp, axis=1)


def add_variant_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``variant_fingerprint`` (== Variant_ID) and a cross-source dup flag."""
    out = df.copy()
    out[FINGERPRINT_COL] = out[ID_COL].astype(str)
    src_per_id = out.groupby(FINGERPRINT_COL)[SOURCE_COL].transform("nunique")
    out[DUP_FLAG_COL] = (src_per_id > 1).astype("int8")
    n_dup_ids = int((out.groupby(FINGERPRINT_COL)[SOURCE_COL].nunique() > 1).sum())
    print(f"[fingerprint] unique={out[FINGERPRINT_COL].nunique()} "
          f"| variants spanning >1 source={n_dup_ids} "
          f"| rows flagged duplicated={int(out[DUP_FLAG_COL].sum())}")
    return out


def deduplicate_keep_one(df: pd.DataFrame,
                         priority: Tuple[str, ...] = ("cftr", "pah", "kanser", "master")) -> pd.DataFrame:
    """Plan Option A: keep one copy per fingerprint, preferring the most specific
    panel (CFTR/PAH > Kanser > General). NOT applied by default -- we keep all
    rows and rely on group-aware CV (Option C). Provided for ablation."""
    order = {s: i for i, s in enumerate(priority)}
    out = df.copy()
    out["_prio"] = out[SOURCE_COL].map(order).fillna(len(order))
    out = out.sort_values("_prio").drop_duplicates(FINGERPRINT_COL, keep="first")
    out = out.drop(columns="_prio").reset_index(drop=True)
    print(f"[dedup] kept {len(out)} of {len(df)} rows (Option A)")
    return out


def _strat_key(df: pd.DataFrame) -> pd.Series:
    """Combined stratification key: label + source (plan 0.3 / 0.4)."""
    return df[LABEL_COL].astype(str) + "_" + df[SOURCE_COL].astype(str)


# =========================================================================== #
# 0.3  Canary hold-out split
# =========================================================================== #
def build_canary_split(
    df: pd.DataFrame,
    test_size: float = CANARY_FRACTION,
    enforce_group_disjoint: bool = True,
    seed: int = SEED,
    canary_path: str = CANARY_PATH,
    development_path: str = DEVELOPMENT_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract a 10% stratified (label + source) canary set and save both splits.

    Uses ``StratifiedShuffleSplit`` as specified. ``enforce_group_disjoint``
    (default True) then moves any canary rows whose ``variant_fingerprint`` also
    appears in development back into development, so no variant is split across
    the two sets -- otherwise the cross-source duplicates would silently leak
    into the supposedly-untouched canary. Set False for a pure stratified split.
    """
    ensure_dirs()
    assert FINGERPRINT_COL in df.columns, "Run add_variant_fingerprint() first."
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_idx, can_idx = next(sss.split(df, _strat_key(df)))
    development = df.iloc[dev_idx].copy()
    canary = df.iloc[can_idx].copy()

    if enforce_group_disjoint:
        dev_fps = set(development[FINGERPRINT_COL])
        leak_mask = canary[FINGERPRINT_COL].isin(dev_fps)
        n_leak = int(leak_mask.sum())
        if n_leak:
            development = pd.concat([development, canary[leak_mask]], ignore_index=True)
            canary = canary[~leak_mask]
            print(f"[canary] moved {n_leak} group-overlapping rows back to development")

    development = development.reset_index(drop=True)
    canary = canary.reset_index(drop=True)
    development.to_parquet(development_path, index=False)
    canary.to_parquet(canary_path, index=False)

    print(f"[canary] canary n={len(canary)} pos_rate={canary[LABEL_COL].mean():.4f} -> {canary_path}")
    print(f"[canary] development n={len(development)} pos_rate={development[LABEL_COL].mean():.4f} -> {development_path}")
    overlap = set(canary[FINGERPRINT_COL]) & set(development[FINGERPRINT_COL])
    print(f"[canary] fingerprint overlap canary<->development = {len(overlap)} (target 0)")
    return development, canary


# =========================================================================== #
# 0.4  Fold index persistence (StratifiedGroupKFold)
# =========================================================================== #
def build_fold_indices(
    development: pd.DataFrame,
    n_splits: int = N_FOLDS,
    seed: int = SEED,
    out_path: str = FOLD_INDICES_PATH,
) -> List[dict]:
    """Generate 5 StratifiedGroupKFold folds and persist positional indices.

    * groups   = ``variant_fingerprint`` (no variant in both train & val)
    * stratify = label + source (PAH proportionally represented in every fold)

    Indices are positional into ``development`` (which must be ``reset_index``-ed,
    as ``build_canary_split`` guarantees). Every later phase loads this exact file.
    """
    ensure_dirs()
    dev = development.reset_index(drop=True)
    y = _strat_key(dev)
    groups = dev[FINGERPRINT_COL].astype(str).values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, va) in enumerate(sgkf.split(dev, y, groups=groups)):
        tr_fp = set(groups[tr])
        va_fp = set(groups[va])
        leak = tr_fp & va_fp
        assert not leak, f"Fold {k} has {len(leak)} leaking groups!"
        folds.append({"fold": k, "train": tr.tolist(), "val": va.tolist()})
        va_pos = dev.iloc[va][LABEL_COL].mean()
        print(f"[folds] fold {k}: train={len(tr)} val={len(va)} "
              f"val_pos_rate={va_pos:.3f} groups(tr/va)={len(tr_fp)}/{len(va_fp)} leak={len(leak)}")

    with open(out_path, "w") as f:
        json.dump(folds, f)
    print(f"[folds] saved -> {out_path}")
    return folds


def load_fold_indices(path: str = FOLD_INDICES_PATH) -> List[dict]:
    with open(path) as f:
        return json.load(f)


# =========================================================================== #
# Orchestrator
# =========================================================================== #
def run_phase0(
    raw_dir: str = RAW_DIR,
    enforce_group_disjoint: bool = True,
    save_merged: bool = True,
) -> Dict[str, object]:
    """Run Phase 0 end-to-end: seed -> dirs -> checksums -> load/merge ->
    fingerprint -> canary split -> fold indices. Returns the key artefacts."""
    set_global_seed(SEED)
    ensure_dirs()

    checksums = compute_checksums(raw_dir)
    merged = load_raw_data(raw_dir)
    merged = add_variant_fingerprint(merged)
    if save_merged:
        merged.to_parquet(MERGED_PATH, index=False)

    development, canary = build_canary_split(
        merged, enforce_group_disjoint=enforce_group_disjoint)
    folds = build_fold_indices(development)

    return {
        "checksums": checksums,
        "merged": merged,
        "development": development,
        "canary": canary,
        "folds": folds,
    }
