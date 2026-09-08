#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an offline, PatHUgen-shaped test set from the real training CSVs.

Purpose: let you run `PatHUgen.py deliver` end-to-end with NO internet and
NO competition data, before the real thing arrives. This is a MECHANICAL
test (does auto-detection, prediction, JSON validation, and delivery
packaging all work?) — not a fair accuracy test, since these rows were seen
during training. A held-out answer key is written alongside so you can still
sanity-check the predictions look reasonable.

Usage:
    .venv/bin/python scripts/make_sample_test_data.py [--frac 0.2]

Output:
    sample_test_data/GENEL_sample.csv    <- pass this folder to `deliver`
    sample_test_data/KANSER_sample.csv      (no Label column - real test
    sample_test_data/PAH_sample.csv          data won't have one either)
    sample_test_data/CFTR_sample.csv

    sample_test_data_answers/<PANEL>_answers.csv  <- id + true Label, kept
    OUTSIDE the folder you point `deliver` at, so it can never be mistaken
    for part of the test input.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import PatHUgen as PH  # noqa: E402

OUT_DIR = ROOT / "sample_test_data"
ANSWERS_DIR = ROOT / "sample_test_data_answers"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2,
                    help="Fraction of each panel to sample (stratified by "
                    "label, default 0.2).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    ANSWERS_DIR.mkdir(exist_ok=True)
    print(f"Sampling {args.frac:.0%} of each panel (stratified by label, seed={args.seed})\n")

    for panel in PH.PANELS:
        df = PH.load_panel(panel)
        parts = [g.sample(frac=args.frac, random_state=args.seed)
                 for _, g in df.groupby(PH.LABEL_COL)]
        sample = pd.concat(parts, ignore_index=True)
        sample = sample.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        answers = sample[[PH.ID_COL, PH.LABEL_COL]].reset_index(drop=True)
        test_rows = sample.drop(columns=[PH.LABEL_COL]).reset_index(drop=True)

        test_path = OUT_DIR / f"{panel}_sample.csv"
        answers_path = ANSWERS_DIR / f"{panel}_answers.csv"
        test_rows.to_csv(test_path, index=False)
        answers.to_csv(answers_path, index=False)

        pos_rate = answers[PH.LABEL_COL].mean()
        print(f"  {panel:<7s} {len(test_rows):4d} rows  "
              f"(true pathogenic rate {pos_rate:.1%})  -> {test_path.relative_to(ROOT)}")

    print(f"\nTest input folder : {OUT_DIR.relative_to(ROOT)}/   <- point --input-dir here")
    print(f"Answer key folder : {ANSWERS_DIR.relative_to(ROOT)}/   <- NOT test input, for your own sanity check only")
    print("\nNote: these rows were part of the training data, so any accuracy")
    print("you compute against the answer key is optimistic in-sample")
    print("performance, not a real generalization estimate. It only tells")
    print("you the pipeline mechanically works end to end.")


if __name__ == "__main__":
    main()
