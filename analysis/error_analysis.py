import json, os, sys
import numpy as np, pandas as pd
os.environ.setdefault("TEKNOFEST_OUT_DIR", "/tmp/x_unused_out")
sys.path.insert(0, ".")
from pathugen.data_io import load_panel
from pathugen.features import build_features

repro = json.load(open("report_figs/repro_metrics.json"))
KEYS = ["n_missing_total","pct_missing","ek_mean","ek_max","ek_count","al_log_max",
        "al_n_ultrarare","al_neglog10_min_pos","grantham","blosum62",
        "damaging_vote_ratio","n_tools_available","missingness_tier"]

for p in ["GENEL","KANSER","PAH","CFTR"]:
    d = np.load(f"report_figs/oof_{p}.npz", allow_pickle=True)
    oof, y, thr = d["oof"], d["y"], float(d["thr"])
    df = load_panel(p)
    X, yy, _ = build_features(df)
    assert len(X) == len(y), (len(X), len(y))
    print(f"\n{'='*72}\n{p}  n={len(y)} thr={thr:.3f} kind={d['kind']}  label-match={bool((np.asarray(yy)==y).all())}")
    pred = (oof >= thr).astype(int)
    tn = int(((y==0)&(pred==0)).sum()); fp = int(((y==0)&(pred==1)).sum())
    fn = int(((y==1)&(pred==0)).sum()); tp = int(((y==1)&(pred==1)).sum())
    print(f"  recomputed tn={tn} fp={fp} fn={fn} tp={tp}  | repro tn={repro[p]['tn']} fp={repro[p]['fp']} fn={repro[p]['fn']} tp={repro[p]['tp']}")
    grp = pd.DataFrame(index=X.index)
    for k in KEYS:
        if k in X.columns: grp[k] = pd.to_numeric(X[k], errors="coerce")
    grp["y"], grp["pred"] = y, pred
    grp["cls"] = np.select(
        [(y==0)&(pred==1), (y==0)&(pred==0), (y==1)&(pred==0), (y==1)&(pred==1)],
        ["FP","TN","FN","TP"], default="?")
    med = grp.groupby("cls")[list(grp.columns[:-3])].median().round(3)
    cnt = grp["cls"].value_counts()
    print("  counts:", cnt.to_dict())
    print(med.to_string())
    # probability band of errors
    for name, mask in [("FP", (y==0)&(pred==1)), ("FN", (y==1)&(pred==0))]:
        if mask.sum():
            pr = oof[mask]
            print(f"  {name} proba: min={pr.min():.3f} med={np.median(pr):.3f} max={pr.max():.3f}"
                  f"  | within 0.10 of thr: {int((np.abs(pr-thr)<=0.10).sum())}/{int(mask.sum())}")
