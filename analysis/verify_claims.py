import os, json, warnings, collections
warnings.filterwarnings("ignore")
os.environ.setdefault("TEKNOFEST_OUT_DIR","/tmp/x_unused_out")
import numpy as np, pandas as pd
from pathugen.config import PANELS, FEATURE_COLS
from pathugen.data_io import load_panel
SCR="/tmp/claude-1000/-home-huseyink-Desktop-TEKNOFEST/c398d6bd-6ea3-4e42-b66e-e6eb8b13a05d/scratchpad"
sel={p: json.load(open(f"{SCR}/colab_full/outputs_colab/models/{p}/selected_features.json")) for p in PANELS}
print("seçilen toplam:", sum(len(v) for v in sel.values()))
cnt=collections.Counter(f for v in sel.values() for f in v)
shared={f:c for f,c in cnt.items() if c>=2}
print("birden fazla panelde paylaşılan:", dict(sorted(shared.items(), key=lambda x:-x[1])))
tot=0; n=0
for p in PANELS:
    df=load_panel(p); m=df[FEATURE_COLS].isnull().sum(axis=1)
    tot+=m.sum(); n+=len(m)
print(f"satır başına ortalama eksik: {tot/n:.1f} / 351  = %{100*tot/n/351:.1f}")
