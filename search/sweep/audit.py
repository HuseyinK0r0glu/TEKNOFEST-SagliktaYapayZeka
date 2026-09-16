"""Overfitting audit and final selection."""
from __future__ import annotations

from search.sweep.config import (
    FOLD_STD_WARN, GAP_LIMIT, OBJECTIVES, SELECT_OBJECTIVE, STABILITY_MIN,
)


def overfit_verdict(row):
    """The three overfitting tests for a configuration. Returns (passed, [reasons])."""
    fails, warns = [], []
    if row["gap"] > GAP_LIMIT:
        fails.append(f"optimism gap {row['gap']:+.4f} > {GAP_LIMIT:.3f} "
                     f"(OOF {row['oof_f1']:.4f} vs honest {row['honest_f1']:.4f})")
    if row["stability"] < STABILITY_MIN:
        fails.append(f"unstable champion: the same candidate won in only "
                     f"%{100*row['stability']:.0f} of outer folds "
                     f"({row['n_distinct_champions']} distinct winners)")
    if row["honest_f1"] <= row["trivial_f1"]:
        fails.append(f"honest F1 {row['honest_f1']:.4f} did not beat the naive "
                     f"baseline ({row['trivial_f1']:.4f})")
    if row["honest_f1_std"] > FOLD_STD_WARN:
        warns.append(f"high cross-fold volatility (std {row['honest_f1_std']:.4f})")
    return (len(fails) == 0), fails, warns


def objective_value(row, objective):
    if objective == "honest_f1":
        return row["honest_f1"]
    if objective == "honest_balanced_f1":
        return row["honest_balanced_f1"]
    return min(row["honest_f1"], row["honest_balanced_f1"])       # robust


def choose_best(panel_df):
    """Pick the best configuration by objective among those that passed the audit."""
    df = panel_df.copy()
    verdicts = df.apply(lambda r: overfit_verdict(r), axis=1)
    df["passes"] = [v[0] for v in verdicts]
    df["fail_reasons"] = [" ; ".join(v[1]) for v in verdicts]
    df["warn_reasons"] = [" ; ".join(v[2]) for v in verdicts]
    for obj in OBJECTIVES:
        df[f"obj_{obj}"] = df.apply(lambda r: objective_value(r, obj), axis=1)

    clean = df[df["passes"]]
    fallback = clean.empty
    pool = df if fallback else clean
    # Tie-break: smaller gap, then simpler strategy, then fewer features
    pool = pool.sort_values(
        by=[f"obj_{SELECT_OBJECTIVE}", "gap", "n_features"],
        ascending=[False, True, True])
    return df, pool.iloc[0], fallback
