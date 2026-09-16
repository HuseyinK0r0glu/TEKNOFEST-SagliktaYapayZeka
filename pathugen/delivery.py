"""Delivery packaging — Section 6 of the delivery guide, fully automated."""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pathugen.config import PANELS, ROOT, TEAM_ID, TEAM_NAME, APPLICATION_ID
from pathugen.predict import predict_panel
from pathugen.versioning import snapshot_label
from pathugen.result_json import build_result_json, validate_result_json
from pathugen.config import MODEL_DIR

_PANEL_KEYWORDS = {
    "GENEL": ("master", "genel"),
    "KANSER": ("kanser",),
    "PAH": ("pah",),
    "CFTR": ("cftr",),
}


def discover_panel_files(input_dir: Path) -> dict[str, Path]:
    """Match every CSV in `input_dir` to a panel by filename keyword.

    Robust to whatever exact filenames the organization hands out on
    competition day (training used YARISMA_TRAIN_<PANEL>.csv; test files may
    follow a different convention) — matching is by keyword, not by exact
    name. If this fails, pass explicit --genel/--kanser/--pah/--cftr paths.
    """
    csvs = sorted(Path(input_dir).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv files found in {input_dir}")
    found: dict[str, Path] = {}
    for csv in csvs:
        name = csv.name.lower()
        for panel, keywords in _PANEL_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                if panel in found:
                    raise RuntimeError(
                        f"Ambiguous input: both {found[panel].name!r} and "
                        f"{csv.name!r} match panel {panel}. Use explicit "
                        f"--{panel.lower()} PATH flags instead.")
                found[panel] = csv
    missing = [p for p in PANELS if p not in found]
    if missing:
        raise FileNotFoundError(
            f"Could not auto-detect a test file for panel(s) {missing} in "
            f"{input_dir} (files seen: {[c.name for c in csvs]}). "
            f"Pass explicit paths: --genel PATH --kanser PATH --pah PATH --cftr PATH")
    return found


def next_delivery_dir(root: Path = ROOT) -> tuple[Path, int]:
    existing = [int(m.group(1)) for d in root.iterdir() if d.is_dir()
                for m in [re.fullmatch(r"delivery(\d+)", d.name)] if m]
    n = (max(existing) + 1) if existing else 1
    return root / f"delivery{n}", n


def freeze_requirements(dest: Path) -> None:
    """Snapshot exact installed package versions for reproducibility.
    Falls back to copying the repo's requirements.txt if `pip freeze` fails
    (e.g. running outside the intended virtualenv)."""
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=60, check=True)
        dest.write_text(out.stdout)
        return
    except Exception as exc:
        print(f"  [warning] pip freeze failed ({exc}); copying requirements.txt instead")
    src = ROOT / "requirements.txt"
    if src.exists():
        shutil.copy2(src, dest)
    else:
        dest.write_text("# pip freeze unavailable and no requirements.txt found\n")


# Dependency discovery is done by STATIC ANALYSIS, not a hand-maintained
# list. A fixed file list silently goes stale the moment a command gains a
# new dependency — instead, every local module PatHUgen.py imports, and
# every local module *those* import, is discovered transitively by parsing
# import statements with `ast`. When a dependency is a PACKAGE (like
# `pathugen` itself), every `.py` file under that package directory is
# included, not just `__init__.py` — a package's submodules are real code
# the delivery needs even when `__init__.py` never imports them itself.
# Non-import file reads (the delivery README template, the requirements.txt
# fallback) are discovered the same way, by scanning for `ROOT / "<name>.
# <ext>"` patterns in the source. This never needs to be updated by hand
# when the project's dependencies change — it is recomputed from the actual
# code every time. requirements.txt is deliberately excluded here even
# though the code reads it (as a pip-freeze fallback): freeze_requirements()
# already writes the authoritative, exact-versions file at the delivery
# root, and copying the repo's separate, hand-maintained requirements.txt
# into code/ as well would put two DIFFERENT requirements.txt files in the
# same delivery — confusing, not helpful.
_LOCAL_FILE_REF_RE = re.compile(r'ROOT\s*/\s*"([^"/]+\.(?:md|json))"')


def _local_dependency_paths(module_name: str, root: Path) -> list[Path]:
    """Local `.py` file(s) `module_name` (e.g. "pathugen") resolves to: a
    single file for a flat module, or every `.py` file (recursively) under
    the package directory for a package. Empty list if `module_name` is not
    local (stdlib / third-party)."""
    candidate = root / f"{module_name}.py"
    if candidate.is_file():
        return [candidate]
    package_dir = root / module_name
    if (package_dir / "__init__.py").is_file():
        return sorted(package_dir.rglob("*.py"))
    return []


def _direct_local_dependencies(py_file: Path, root: Path) -> list[Path]:
    """Local files `py_file` needs: imported local modules/packages
    (anywhere in the file, including inside function bodies — `ast.walk`
    finds both) plus any `ROOT / "<file>"` reads for non-Python assets."""
    source = py_file.read_text(encoding="utf-8")
    deps: list[Path] = []
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError as exc:
        raise RuntimeError(f"Could not parse {py_file} while resolving "
                           f"delivery dependencies: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".")[0]]
        else:
            continue
        for name in names:
            deps.extend(_local_dependency_paths(name, root))
    for match in _LOCAL_FILE_REF_RE.finditer(source):
        candidate = root / match.group(1)
        if candidate.is_file():
            deps.append(candidate)
    return deps


def resolve_dependency_closure(entry: Path, root: Path = ROOT) -> list[Path]:
    """Every local file `entry` needs, directly or transitively, computed by
    following imports and file reads outward from `entry` — not a hand-kept
    list. `entry` itself is always included."""
    seen: set[Path] = set()
    queue = [entry.resolve()]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        if current.suffix == ".py":
            for dep in _direct_local_dependencies(current, root):
                resolved = dep.resolve()
                if resolved not in seen:
                    queue.append(resolved)
    return sorted(seen)


def copy_source_code(dest: Path, root: Path = ROOT) -> list[str]:
    """Copy PatHUgen.py and every local file it (transitively) depends on —
    discovered fresh each run via `resolve_dependency_closure`, so the
    delivery package always matches what the code actually needs, including
    for commands like `run` that pull in the rest of the project."""
    dest.mkdir(parents=True, exist_ok=True)
    deps = resolve_dependency_closure(root / "PatHUgen.py", root)
    copied = []
    for src in deps:
        rel = src.relative_to(root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(str(rel))
    return copied


def render_delivery_readme(template_path: Path, *, delivery_n: int, run_command: str,
                           input_dir: Path, output_json: Path,
                           panel_files: dict[str, Path], counts: dict[str, int],
                           version_dirs: dict[str, Path] | None = None) -> str:
    text = template_path.read_text(encoding="utf-8")
    panel_files_md = "\n".join(f"- `{p}` -> `{f}`" for p, f in panel_files.items())
    version_dirs = version_dirs or {}
    counts_md = "\n".join(
        f"- {p}: {counts.get(p, 0)}"
        + (f"  (model {snapshot_label(version_dirs[p])})" if p in version_dirs else "")
        for p in PANELS)
    try:
        lib_versions = subprocess.run(
            [sys.executable, "-c",
             "import numpy,pandas,sklearn,xgboost,joblib;"
             "print(f'numpy {numpy.__version__}, pandas {pandas.__version__}, "
             "scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}, "
             "joblib {joblib.__version__}')"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except Exception:
        lib_versions = "(unavailable — see requirements.txt)"

    replacements = {
        "{{DELIVERY_NUMBER}}": str(delivery_n),
        "{{TEAM_NAME}}": TEAM_NAME, "{{TEAM_ID}}": TEAM_ID,
        "{{APPLICATION_ID}}": APPLICATION_ID,
        "{{GENERATED_AT}}": datetime.now().isoformat(timespec="seconds"),
        "{{RUN_COMMAND}}": run_command,
        "{{INPUT_DIR}}": str(input_dir.resolve()),
        "{{PANEL_FILES}}": panel_files_md,
        "{{PREDICTION_COUNTS}}": counts_md,
        "{{OUTPUT_JSON}}": f"delivery{delivery_n}/{output_json.name}",
        "{{OUTPUT_JSON_NAME}}": output_json.name,
        "{{PYTHON_VERSION}}": sys.version.split()[0],
        "{{LIBRARY_VERSIONS}}": lib_versions,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def deliver(input_dir: Path, *, panel_files: dict[str, Path] | None = None,
            model_dir: Path = MODEL_DIR, prior_shift: float | None = None,
            model_version: int | None = None, skip_copy: bool = False,
            run_command: str = "") -> Path:
    input_dir = Path(input_dir)
    files = panel_files or discover_panel_files(input_dir)
    print("Panel -> input file mapping:")
    for p in PANELS:
        print(f"  {p:<7s} {files[p]}")

    predictions_by_panel, counts, version_dirs = {}, {}, {}
    for panel in PANELS:
        recs, panel_dir = predict_panel(panel, files[panel], model_dir=model_dir,
                                        prior_shift=prior_shift,
                                        model_version=model_version)
        predictions_by_panel[panel] = recs
        counts[panel] = len(recs)
        version_dirs[panel] = panel_dir
        n_pos = sum(1 for r in recs if r["predicted_class"] == "1")
        print(f"  {panel:<7s} {len(recs):5d} predictions  "
              f"(predicted pathogenic rate {n_pos/len(recs):.1%})  "
              f"[model {snapshot_label(panel_dir)}]")

    result = build_result_json(predictions_by_panel)
    problems = validate_result_json(result, counts)
    if problems:
        print("\nVALIDATION FAILED — result JSON was NOT written:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nValidation passed: schema, ids, classes, and probabilities all OK.")

    delivery_dir, n = next_delivery_dir()
    delivery_dir.mkdir(parents=True)
    numbered_name = f"TEAM_{TEAM_ID}_FINAL{n}.json"
    numbered_path = delivery_dir / numbered_name
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    numbered_path.write_text(payload, encoding="utf-8")
    print(f"\nResult JSON written:")
    print(f"  {numbered_path}   <- upload this file")

    if not skip_copy:
        req_path = delivery_dir / "requirements.txt"
        freeze_requirements(req_path)
        copied_files = copy_source_code(delivery_dir / "code")
        # Copy only the exact model snapshot used for THIS delivery's
        # predictions (not the whole version history) — the delivery guide
        # wants the package sufficient to reproduce this result, and shipping
        # every old snapshot would only obscure which one actually ran.
        # The version number is dropped from the path (model/<PANEL>/, not
        # model/<PANEL>/v3/) so the guide's expected flat layout is
        # unaffected; the exact version used is recorded in README.md.
        model_dest = delivery_dir / "model"
        for panel, panel_dir in version_dirs.items():
            shutil.copytree(panel_dir, model_dest / panel)
        readme_text = render_delivery_readme(
            ROOT / "Delivery_Readme.md", delivery_n=n,
            run_command=run_command or f"python PatHUgen.py deliver --input-dir {input_dir}",
            input_dir=input_dir, output_json=numbered_path,
            panel_files=files, counts=counts, version_dirs=version_dirs)
        (delivery_dir / "README.md").write_text(readme_text, encoding="utf-8")
        print(f"\nPackaged: {', '.join(copied_files)}, model bundle, "
              f"requirements.txt, README.md -> {delivery_dir}")
    else:
        print("\n--skip-copy set: only the JSON file was written (no code/model/"
              "README packaging). Use this for fast iteration; do a full "
              "(non-skip) run before the final upload.")

    print(f"\nDelivery folder ready: {delivery_dir}")
    return delivery_dir
