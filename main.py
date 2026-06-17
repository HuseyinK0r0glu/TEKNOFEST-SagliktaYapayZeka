"""Full TEKNOFEST pipeline — run this from the project root inside the venv.

    python main.py                          # all 4 panels, 'fast' profile
    PIPELINE_PROFILE=thorough python main.py   # full rigor (6-10h+)
    python main.py CFTR PAH                  # specific panels

Profiles (PIPELINE_PROFILE, default 'fast') preset the anti-overfit knobs below;
anything you set explicitly in the environment still wins (setdefault):

    knob              fast        thorough
    CV_REPEATS         3            5
    OPTUNA_TRIALS      40           80
    OPTUNA_TIMEOUT     300          900
    OPTUNA_OBJECTIVE   auc          auc        (balance-independent search)
    OPTUNA_SEED_JITTER 0            1          (per-trial fold seed)
    THRESHOLD_AGG      fold_median  fold_median
    SELECT_METRIC      auc          auc        (champion picked on AUC, one-SE rule)
    NESTED_EVAL        1            1          (leak-free honest estimate)
    NESTED_RETUNE      0            0          (opt-in only; re-tune per fold = many extra hours)
    NESTED_REPEATS     1            3          (outer-CV repeats; stabilises small n)

Opt-in anywhere: EARLY_STOP=1 (boosted trees early-stop on a fold-internal slice).

Steps: 1 Baseline -> 2 Feature selection -> 3 Optuna -> 4 Stacking ->
       5 Finalize -> [6 Nested eval, if NESTED_EVAL]. Outputs land in outputs/.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Allow `python main.py` without setting PYTHONPATH manually.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- Profile resolution MUST run before importing the train/src modules, since
#     several capture their env knobs at import time (tuning, run_optuna). ---
PROFILES = {
    "fast": {
        "CV_REPEATS": "3", "OPTUNA_TRIALS": "40", "OPTUNA_TIMEOUT": "300",
        # AUC objective + AUC champion selection + median threshold: the test-set
        # class balance is unknown, so optimise the balance-independent ranking
        # and pick a robust threshold rather than overfit F1 to the train prior.
        "OPTUNA_OBJECTIVE": "auc", "OPTUNA_SEED_JITTER": "0",
        "THRESHOLD_AGG": "fold_median", "SELECT_METRIC": "auc",
        # Honest held-out estimate ON in both profiles so the CV (OOF) number and
        # the true test number always print together. fast reuses the global
        # hyperparameters per outer fold (no per-fold re-tune) to stay quick.
        "NESTED_EVAL": "1", "NESTED_RETUNE": "0", "NESTED_REPEATS": "1",
    },
    "thorough": {
        "CV_REPEATS": "5", "OPTUNA_TRIALS": "80", "OPTUNA_TIMEOUT": "900",
        "OPTUNA_OBJECTIVE": "auc", "OPTUNA_SEED_JITTER": "1",
        "THRESHOLD_AGG": "fold_median", "SELECT_METRIC": "auc",
        # Nested eval REUSES the global hyperparameters per outer fold and runs 3
        # outer-CV repeats: this keeps the honest estimate stable (small panels)
        # without exploding runtime. Per-outer-fold re-tuning (NESTED_RETUNE=1) is
        # an explicit opt-in only — it re-tunes 5 models on 15 folds and adds many
        # hours; do NOT enable it for a fixed overnight window.
        "NESTED_EVAL": "1", "NESTED_RETUNE": "0", "NESTED_REPEATS": "3",
    },
}


def _resolve_profile() -> str:
    profile = os.environ.get("PIPELINE_PROFILE", "fast").strip().lower()
    if profile not in PROFILES:
        profile = "fast"
    for key, val in PROFILES[profile].items():
        os.environ.setdefault(key, val)  # explicit env wins
    return profile


PROFILE = _resolve_profile()

from src import config  # noqa: E402

import train.run_baseline as run_baseline  # noqa: E402
import train.run_feature_selection as run_feature_selection  # noqa: E402
import train.run_optuna as run_optuna  # noqa: E402
import train.run_stacking as run_stacking  # noqa: E402
import train.finalize_champions as finalize_champions  # noqa: E402

_KNOBS = ["CV_REPEATS", "OPTUNA_TRIALS", "OPTUNA_TIMEOUT", "OPTUNA_OBJECTIVE",
          "OPTUNA_SEED_JITTER", "THRESHOLD_AGG", "SELECT_METRIC", "EARLY_STOP",
          "NESTED_EVAL", "NESTED_RETUNE", "NESTED_REPEATS"]


def _nested_on() -> bool:
    return os.environ.get("NESTED_EVAL", "0").strip().lower() in ("1", "true", "yes")


def _banner(step: int, total: int, name: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  STEP {step}/{total} — {name}")
    print(f"{sep}\n", flush=True)


def _elapsed(t0: float) -> str:
    secs = int(time.time() - t0)
    return f"{secs // 60}m {secs % 60}s"


def main() -> None:
    panels = [p.upper() for p in sys.argv[1:]] or config.PANELS
    total = 6 if _nested_on() else 5
    print(f"\nProfile: {PROFILE}")
    print("Knobs:   " + "  ".join(
        f"{k}={os.environ.get(k)}" for k in _KNOBS if os.environ.get(k) is not None))
    print(f"Panels:  {panels}")
    print(f"Output:  {config.OUTPUT_DIR}\n")

    total_start = time.time()

    # ── Step 1: Baseline ──────────────────────────────────────────────────
    _banner(1, total, "Baseline (XGB / LGBM / CatBoost)")
    t0 = time.time()
    run_baseline.main(panels)
    print(f"\n[step 1 done in {_elapsed(t0)}]")

    # ── Step 2: Feature selection ─────────────────────────────────────────
    _banner(2, total, "Feature Selection (null-importance)")
    t0 = time.time()
    run_feature_selection.main(panels)
    print(f"\n[step 2 done in {_elapsed(t0)}]")

    # ── Step 3: Optuna tuning ─────────────────────────────────────────────
    _banner(3, total, "Optuna Hyperparameter Tuning")
    t0 = time.time()
    run_optuna.main(panels)
    print(f"\n[step 3 done in {_elapsed(t0)}]")

    # ── Step 4: Stacking ──────────────────────────────────────────────────
    _banner(4, total, "Stacking (calibrated meta-learner + OOF)")
    t0 = time.time()
    run_stacking.main(panels)
    print(f"\n[step 4 done in {_elapsed(t0)}]")

    # ── Step 5: Finalize champions ────────────────────────────────────────
    _banner(5, total, "Finalize Champions (one-SE rule)")
    t0 = time.time()
    finalize_champions.main(panels)
    print(f"\n[step 5 done in {_elapsed(t0)}]")

    # ── Step 6: Nested eval (honest estimate) — optional ──────────────────
    if _nested_on():
        _banner(6, total, "Nested CV (honest generalization estimate)")
        t0 = time.time()
        import train.run_nested_eval as run_nested_eval
        run_nested_eval.main(panels)
        print(f"\n[step 6 done in {_elapsed(t0)}]")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE — total time: {_elapsed(total_start)}")
    print(f"  Results in: {config.OUTPUT_DIR}")
    print("=" * 60)

    import pandas as pd

    champ_path = config.OUTPUT_DIR / "models" / "champions_summary.csv"
    if champ_path.exists():
        print("\n[1] CROSS-VALIDATED (OOF) — champion per panel "
              "(model selection; mildly optimistic):\n")
        print(pd.read_csv(champ_path).to_string(index=False))

    nested_path = config.OUTPUT_DIR / "nested_eval" / "nested_summary.csv"
    if _nested_on() and nested_path.exists():
        print("\n[2] HELD-OUT TEST (nested CV) — honest generalization estimate "
              "(trust this number):\n")
        print(pd.read_csv(nested_path).to_string(index=False))
        print("\n  optimism_gap = OOF F1 - honest F1.  Larger gap = more "
              "overfitting (expected worst on the small panels).")
    print()


if __name__ == "__main__":
    main()
