"""Pick and persist the champion strategy per panel from saved OOF.

Reads outputs/stacking/{panel}_oof.csv (base-model + meta OOF + y), computes the
best-threshold F1 for each single base model, the mean ensemble, and the stacked
meta, then writes outputs/models/{panel}/champion.json describing the winner and
its threshold. No model retraining — purely a selection over existing OOF.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from src import config, cv


def finalize(panel: str) -> dict:
    oof_path = config.OUTPUT_DIR / "stacking" / f"{panel}_oof.csv"
    oof = pd.read_csv(oof_path)
    y = oof["y"].to_numpy()
    base_cols = [c for c in oof.columns if c not in ("y", "meta")]

    candidates = {}
    for col in base_cols:
        t, f1 = cv.tune_threshold(y, oof[col].to_numpy())
        candidates[("single", col)] = (f1, t)
    t_mean, f1_mean = cv.tune_threshold(y, oof[base_cols].mean(axis=1).to_numpy())
    candidates[("mean", None)] = (f1_mean, t_mean)
    t_stack, f1_stack = cv.tune_threshold(y, oof["meta"].to_numpy())
    candidates[("stacked", None)] = (f1_stack, t_stack)

    (strategy, model), (f1, threshold) = max(candidates.items(), key=lambda kv: kv[1][0])
    champion = {
        "panel": panel, "strategy": strategy, "model": model,
        "threshold": threshold, "oof_f1": f1, "base_order": base_cols,
    }
    out_path = config.OUTPUT_DIR / "models" / panel / "champion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(champion, indent=2), encoding="utf-8")
    label = model if strategy == "single" else strategy
    print(f"  {panel:7s} champion = {label:9s} F1={f1:.4f} @ t={threshold:.3f}")
    return champion


def main(panels):
    print("=== Champion per panel ===")
    rows = [finalize(p) for p in panels]
    summary = pd.DataFrame(rows)[["panel", "strategy", "model", "threshold", "oof_f1"]]
    out_path = config.OUTPUT_DIR / "models" / "champions_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    requested = [p.upper() for p in sys.argv[1:]] or config.PANELS
    main(requested)
