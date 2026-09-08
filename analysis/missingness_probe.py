import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
os.environ.setdefault("TEKNOFEST_OUT_DIR", "/tmp/x_unused_out")
from pathugen.data_io import load_panel
from pathugen.features import build_features
print(f"{'panel':8s} {'n':>5s} {'AUC(pct_missing)':>17s} {'AUC(ek_mean)':>13s} {'AUC(n_tools)':>13s} {'AUC(al_n_ultrarare)':>20s}")
for p in ["GENEL","KANSER","PAH","CFTR"]:
    df = load_panel(p); X,y,_ = build_features(df); y=np.asarray(y)
    row=[]
    for f in ["pct_missing","ek_mean","n_tools_available","al_n_ultrarare"]:
        v = pd.to_numeric(X[f], errors="coerce").fillna(pd.to_numeric(X[f],errors="coerce").median())
        row.append(roc_auc_score(y, v))
    print(f"{p:8s} {len(y):5d} {row[0]:17.3f} {row[1]:13.3f} {row[2]:13.3f} {row[3]:20.3f}")
