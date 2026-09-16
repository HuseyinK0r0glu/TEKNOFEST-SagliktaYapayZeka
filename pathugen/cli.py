"""CLI — `train` / `versions` / `run` / `deliver`."""
from __future__ import annotations

import sys
from pathlib import Path

from pathugen.config import MODEL_DIR, PANELS
from pathugen.delivery import deliver
from pathugen.train import train
from pathugen.versioning import list_model_versions


def run_search(panels=None) -> None:
    try:
        from search.sweep.cli import main as sweep_main
    except ImportError as exc:
        raise RuntimeError(
            "`run` needs the full project, not just PatHUgen.py and "
            "pathugen/: the `search/` package must sit in the same folder "
            "as this file. (train/deliver never need it — only `run` does, "
            "because it IS the search.) "
            f"Original import error: {exc}") from exc

    print("Searching for the best configuration per panel — this is a "
          "genuinely from-scratch, multi-hour operation (no cached or "
          "inherited hyperparameters are used).")
    print("Results will be written to outputs_best/ "
          "(champion_config.json, BEST_CONFIG.md, models/) and will NOT "
          "change this project's champion_config.json automatically.\n")
    sweep_main(panels=panels)
    print("\nSearch complete. To adopt a new result: open outputs_best/"
          "BEST_CONFIG.md, review the audited winner per panel, then copy "
          "outputs_best/champion_config.json over this project's "
          "champion_config.json — then run `python PatHUgen.py train` to "
          "fit it. See Champion_Config.md.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="PatHUgen.py",
        description="TEKNOFEST 2026 Health AI — PatHUgen final delivery system.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the champion configuration "
                             "(read from champion_config.json) from the official "
                             "training CSVs. Every run creates a NEW numbered "
                             "snapshot (v1, v2, ...) — previous ones are kept, "
                             "never overwritten.")
    p_train.add_argument("panels", nargs="*", choices=PANELS,
                         default=None, help="Panels to train (default: all four).")
    p_train.add_argument("--model-dir", type=Path, default=MODEL_DIR)

    p_versions = sub.add_parser("versions", help="List every trained snapshot "
                                "per panel (which one `deliver` will use by "
                                "default is marked).")
    p_versions.add_argument("--model-dir", type=Path, default=MODEL_DIR)

    p_run = sub.add_parser("run", help="Redo the full search for the best "
                           "configuration per panel (feature set, model, "
                           "threshold) under an overfitting audit, entirely "
                           "from scratch — no cached or inherited "
                           "hyperparameters. Multi-hour, development-only — "
                           "NOT a competition-day command, and requires the "
                           "`search/` package alongside this file (unlike "
                           "train/deliver, which are self-contained). Writes "
                           "results to outputs_best/ and does not modify this "
                           "project's champion_config.json.")
    p_run.add_argument("panels", nargs="*", choices=PANELS, default=None,
                       help="Panels to search (default: all four).")

    p_deliver = sub.add_parser("deliver", help="Run inference on test data, "
                               "validate the result JSON, and build a full, "
                               "numbered delivery folder (competition day).")
    p_deliver.add_argument("--input-dir", type=Path, required=True,
                           help="Folder containing the four test CSVs.")
    p_deliver.add_argument("--genel", type=Path, default=None)
    p_deliver.add_argument("--kanser", type=Path, default=None)
    p_deliver.add_argument("--pah", type=Path, default=None)
    p_deliver.add_argument("--cftr", type=Path, default=None)
    p_deliver.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p_deliver.add_argument("--model-version", type=int, default=None,
                           help="Use a specific trained snapshot number "
                           "(e.g. 2 for v2) instead of the most recently "
                           "trained one. Applies to all four panels. See "
                           "`python PatHUgen.py versions` for what's available. "
                           "Omit to always use the latest.")
    p_deliver.add_argument("--prior-shift", type=float, default=None,
                           help="Optional: recompute the decision threshold as "
                           "if the test set's positive rate were this value "
                           "(e.g. 0.5 for a balanced test set), using the "
                           "saved calibration. Does not retrain anything. "
                           "Omit to use the audited default threshold.")
    p_deliver.add_argument("--skip-copy", action="store_true",
                           help="Write only the JSON (fast). Skips copying "
                           "code/model/README into the delivery folder — use "
                           "for quick iteration, not for the final submission.")

    args = parser.parse_args()
    run_command = "python " + " ".join([Path(sys.argv[0]).name] + sys.argv[1:])

    try:
        if args.command == "train":
            train(panels=args.panels or None, model_dir=args.model_dir)
        elif args.command == "versions":
            list_model_versions(model_dir=args.model_dir)
        elif args.command == "run":
            run_search(panels=args.panels or None)
        elif args.command == "deliver":
            explicit = {"GENEL": args.genel, "KANSER": args.kanser,
                       "PAH": args.pah, "CFTR": args.cftr}
            explicit = {k: v for k, v in explicit.items() if v is not None}
            panel_files = None
            if explicit:
                missing = [p for p in PANELS if p not in explicit]
                if missing:
                    raise SystemExit(f"--genel/--kanser/--pah/--cftr must all be given "
                                     f"together if any is used; missing: {missing}")
                panel_files = explicit
            deliver(args.input_dir, panel_files=panel_files, model_dir=args.model_dir,
                   prior_shift=args.prior_shift, model_version=args.model_version,
                   skip_copy=args.skip_copy, run_command=run_command)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # Expected, actionable failures (bad input path, schema mismatch,
        # missing model, duplicate ids, ...) — print a clean message instead
        # of a stack trace. This runs live during the timed competition test;
        # a scary traceback wastes time a jury-facing error message would not.
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
