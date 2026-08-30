import os, time, importlib.util, warnings
warnings.filterwarnings("ignore")
SCR="/tmp/claude-1000/-home-huseyink-Desktop-TEKNOFEST/c398d6bd-6ea3-4e42-b66e-e6eb8b13a05d/scratchpad"
os.environ["TEKNOFEST_OUT_DIR"]=f"{SCR}/colab_full/outputs_colab"
spec = importlib.util.spec_from_file_location("rf", "run_final.py")
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
import numpy as np
from sklearn.metrics import f1_score
print(f"{'panel':8s} {'n':>5s} {'batch_s':>9s} {'per_row_ms':>11s} {'single_row_ms':>14s} {'F1(in-sample)':>14s}")
for p in ["GENEL","KANSER","PAH","CFTR"]:
    df = rf.load_panel(p)
    t0=time.perf_counter(); out = rf.predict_panel(p, df); t1=time.perf_counter()
    one = df.head(1)
    rf.predict_panel(p, one)  # warm
    t2=time.perf_counter(); rf.predict_panel(p, one); t3=time.perf_counter()
    f1 = f1_score(df[rf.LABEL_COL], out["prediction"], zero_division=0)
    print(f"{p:8s} {len(df):5d} {t1-t0:9.2f} {1000*(t1-t0)/len(df):11.2f} {1000*(t3-t2):14.1f} {f1:14.4f}")
