"""Per-panel EDA-as-code for the TEKNOFEST variant-pathogenicity task.

Validates the dataset facts that drive modelling decisions, treating each
panel INDEPENDENTLY (no pooling). Produces a markdown report at
outputs/eda/eda_report.md and prints a console summary.

Run from the project root:
    PYTHONPATH=. .venv/bin/python eda/run_eda.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config, data_io

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MISSING_TIERS = [(0, 20, "dense(0-20)"), (21, 100, "well(21-100)"),
                 (101, 200, "partial(101-200)"), (201, 300, "sparse(201-300)"),
                 (301, 351, "skeleton(301+)")]
MIN_CAT_SUPPORT = 10  # categories rarer than this are ignored in the leakage flag


def _fmt(x, nd=3):
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def label_summary(df: pd.DataFrame) -> dict:
    y = df[config.LABEL_COL]
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_rate = n_pos / n
    # scale_pos_weight weights the positive (Pathogenic) class; Pathogenic is the
    # MAJORITY here, so this value is < 1 (down-weights the majority).
    spw = (n_neg / n_pos) if n_pos else float("nan")
    return {"n": n, "n_pos": n_pos, "n_neg": n_neg, "pos_rate": pos_rate,
            "imbalance": n_pos / n_neg if n_neg else float("inf"), "scale_pos_weight": spw}


def missingness_profile(df: pd.DataFrame) -> dict:
    feats = df[config.FEATURE_COLS]
    per_row = feats.isnull().sum(axis=1)
    tiers = {}
    for lo, hi, name in MISSING_TIERS:
        tiers[name] = int(((per_row >= lo) & (per_row <= hi)).sum())
    col_null = feats.isnull().mean()
    return {
        "row_min": int(per_row.min()), "row_med": float(per_row.median()),
        "row_q25": float(per_row.quantile(0.25)), "row_q75": float(per_row.quantile(0.75)),
        "row_max": int(per_row.max()), "tiers": tiers,
        "cols_always_null": int((col_null == 1.0).sum()),
        "cols_always_present": int((col_null == 0.0).sum()),
        "cols_dense_ge50pct": int((col_null <= 0.5).sum()),
    }


def al_binary_vs_continuous(df: pd.DataFrame) -> dict:
    n_binary = 0
    n_continuous = 0
    n_empty = 0
    for col in config.AL_COLS:
        vals = df[col].dropna().unique()
        if len(vals) == 0:
            n_empty += 1
        elif set(np.unique(vals)).issubset({0.0, 1.0}):
            n_binary += 1
        else:
            n_continuous += 1
    return {"binary_flags": n_binary, "continuous": n_continuous, "all_null": n_empty}


def ek_directionality(df: pd.DataFrame) -> list[dict]:
    y = df[config.LABEL_COL]
    rows = []
    for col in config.EK_COLS:
        x = df[col]
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < 20 or x[mask].nunique() < 2:
            rows.append({"col": col, "n": n, "r": None, "vmin": None, "vmax": None, "dir": "?"})
            continue
        r, _ = stats.pointbiserialr(y[mask].astype(int), x[mask].astype(float))
        rows.append({"col": col, "n": n, "r": float(r),
                     "vmin": float(x[mask].min()), "vmax": float(x[mask].max()),
                     "dir": "higher=PATHO" if r > 0 else "higher=BENIGN"})
    return rows


def cat_leakage_audit(df: pd.DataFrame) -> list[dict]:
    y = df[config.LABEL_COL]
    rows = []
    for col in config.CAT_COLS:
        s = df[col]
        present = s.notna()
        n_present = int(present.sum())
        miss_frac = 1 - n_present / len(s)
        nunique = int(s.nunique(dropna=True))
        if n_present == 0:
            rows.append({"col": col, "miss_frac": miss_frac, "nunique": 0,
                         "weighted_purity": None, "pure_cat_coverage": None})
            continue
        grp = pd.DataFrame({"cat": s[present].astype(str), "y": y[present].astype(int)})
        agg = grp.groupby("cat")["y"].agg(["count", "mean"])
        agg["purity"] = agg["mean"].clip(lower=0).combine(1 - agg["mean"], max)
        weighted_purity = float((agg["count"] * agg["purity"]).sum() / agg["count"].sum())
        pure = agg[(agg["count"] >= MIN_CAT_SUPPORT) & (agg["purity"] >= config.CAT_PURITY_FLAG)]
        pure_coverage = float(pure["count"].sum() / agg["count"].sum())
        rows.append({"col": col, "miss_frac": miss_frac, "nunique": nunique,
                     "weighted_purity": weighted_purity, "pure_cat_coverage": pure_coverage,
                     "n_pure_cats": int(len(pure))})
    return rows


def aa_analysis(df: pd.DataFrame) -> dict:
    y = df[config.LABEL_COL]
    a1, a2 = df["AA_1"].astype("string"), df["AA_2"].astype("string")
    both_present = a1.notna() & a2.notna()
    aa_missing = int((~both_present).sum())
    stop_gain = (a2 == "*")
    stop_loss = (a1 == "*")
    synonymous = both_present & (a1 == a2)
    nonstd = both_present & (
        ~a1.isin(STANDARD_AA | {"*"}) | ~a2.isin(STANDARD_AA | {"*"})
    )

    def rate(mask):
        m = mask & y.notna()
        return (float(y[m].mean()), int(m.sum())) if m.sum() else (float("nan"), 0)

    sg_rate, sg_n = rate(stop_gain)
    sl_rate, sl_n = rate(stop_loss)
    syn_rate, syn_n = rate(synonymous)
    return {
        "aa_missing": aa_missing, "aa_missing_frac": aa_missing / len(df),
        "stop_gain_n": sg_n, "stop_gain_patho_rate": sg_rate,
        "stop_loss_n": sl_n, "stop_loss_patho_rate": sl_rate,
        "synonymous_n": syn_n, "synonymous_patho_rate": syn_rate,
        "nonstandard_n": int(nonstd.sum()),
    }


def cross_panel_overlap(panels: dict[str, pd.DataFrame]) -> list[str]:
    """Document Variant_ID overlap and whether shared IDs are independently
    annotated (matching AA/Label but differing EK) — validates 'no pooling'."""
    lines = ["", "## Cross-panel Variant_ID overlap (validates independence)", ""]
    ids = {p: set(df[config.ID_COL]) for p, df in panels.items()}
    lines.append("| A | B | shared IDs |")
    lines.append("|---|---|-----------:|")
    names = config.PANELS
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lines.append(f"| {a} | {b} | {len(ids[a] & ids[b])} |")

    # For GENEL ∩ each sub-panel: do AA/Label agree while EK differs?
    lines += ["", "For IDs shared with GENEL — agreement of AA/Label vs divergence of EK_1:", ""]
    lines.append("| sub-panel | shared w/ GENEL | AA+Label match % | EK_1 differs % |")
    lines.append("|---|--:|--:|--:|")
    genel = panels["GENEL"].set_index(config.ID_COL)
    genel = genel[~genel.index.duplicated()]
    for p in ["KANSER", "PAH", "CFTR"]:
        sub = panels[p].set_index(config.ID_COL)
        sub = sub[~sub.index.duplicated()]
        common = sorted(set(genel.index) & set(sub.index))
        if not common:
            lines.append(f"| {p} | 0 | NA | NA |")
            continue
        g, s = genel.loc[common], sub.loc[common]
        aa_label_match = (
            (g["AA_1"].astype("string") == s["AA_1"].astype("string"))
            & (g["AA_2"].astype("string") == s["AA_2"].astype("string"))
            & (g[config.LABEL_COL] == s[config.LABEL_COL])
        ).mean()
        ek_diff = (~np.isclose(g["EK_1"].fillna(-9e9), s["EK_1"].fillna(-9e9))).mean()
        lines.append(f"| {p} | {len(common)} | {aa_label_match*100:.1f}% | {ek_diff*100:.1f}% |")
    return lines


def build_report(panels: dict[str, pd.DataFrame]) -> str:
    out = [
        "# TEKNOFEST Variant Pathogenicity — Per-Panel EDA Report",
        "",
        "Architecture: **4 independent per-panel models** (no pooling). "
        "Metric: **binary F1** on Pathogenic (positive) class.",
        f"Generated from CSVs in `{config.DATA_DIR.name}`.",
        "",
    ]
    for panel, df in panels.items():
        ls = label_summary(df)
        ms = missingness_profile(df)
        al = al_binary_vs_continuous(df)
        ek = ek_directionality(df)
        cat = cat_leakage_audit(df)
        aa = aa_analysis(df)

        out += [
            f"## Panel: {panel}  ({config.PANEL_FILES[panel]})", "",
            f"- Rows: **{ls['n']}** | Pathogenic: {ls['n_pos']} "
            f"({ls['pos_rate']*100:.1f}%) | Benign: {ls['n_neg']} | "
            f"imbalance {ls['imbalance']:.2f}:1 | suggested scale_pos_weight ≈ "
            f"**{_fmt(ls['scale_pos_weight'])}**",
            f"- Missing/row (of 351): min {ms['row_min']}, q25 {ms['row_q25']:.0f}, "
            f"median **{ms['row_med']:.0f}**, q75 {ms['row_q75']:.0f}, max {ms['row_max']}",
            f"- Columns: {ms['cols_always_present']} always-present, "
            f"{ms['cols_always_null']} always-null, {ms['cols_dense_ge50pct']} present in ≥50% rows",
            f"- Missingness tiers: " + ", ".join(f"{k}={v}" for k, v in ms["tiers"].items()),
            f"- AL columns: {al['binary_flags']} binary flags, {al['continuous']} continuous, "
            f"{al['all_null']} all-null",
            "",
            "**EK directionality** (point-biserial r vs Label; sign = direction):", "",
            "| EK | n | r | range | direction |",
            "|---|--:|--:|---|---|",
        ]
        for e in ek:
            rng = f"[{_fmt(e['vmin'],2)}, {_fmt(e['vmax'],2)}]" if e["vmin"] is not None else "NA"
            out.append(f"| {e['col']} | {e['n']} | {_fmt(e['r'])} | {rng} | {e['dir']} |")

        out += ["", "**CAT leakage audit** (weighted purity → proxy-label risk):", "",
                "| CAT | miss% | n_uniq | weighted_purity | pure-cat coverage |",
                "|---|--:|--:|--:|--:|"]
        for c in cat:
            wp = _fmt(c["weighted_purity"]) if c["weighted_purity"] is not None else "NA"
            pc = (f"{c['pure_cat_coverage']*100:.1f}% ({c.get('n_pure_cats',0)} cats)"
                  if c["pure_cat_coverage"] is not None else "NA")
            out.append(f"| {c['col']} | {c['miss_frac']*100:.0f}% | {c['nunique']} | {wp} | {pc} |")

        out += [
            "",
            f"**AA**: missing {aa['aa_missing']} ({aa['aa_missing_frac']*100:.1f}%) | "
            f"stop-gain n={aa['stop_gain_n']} (patho {_fmt(aa['stop_gain_patho_rate'])}) | "
            f"stop-loss n={aa['stop_loss_n']} (patho {_fmt(aa['stop_loss_patho_rate'])}) | "
            f"synonymous n={aa['synonymous_n']} (patho {_fmt(aa['synonymous_patho_rate'])}) | "
            f"nonstandard n={aa['nonstandard_n']}",
            "",
        ]

    out += cross_panel_overlap(panels)
    return "\n".join(out)


def main():
    config.EDA_DIR.mkdir(parents=True, exist_ok=True)
    panels = data_io.load_all_panels()
    report = build_report(panels)
    report_path = config.EDA_DIR / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[written] {report_path}")


if __name__ == "__main__":
    main()
