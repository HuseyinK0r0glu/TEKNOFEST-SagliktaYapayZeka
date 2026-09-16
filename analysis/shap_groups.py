import os, json, warnings, re
warnings.filterwarnings("ignore")
SCR="/tmp/claude-1000/-home-huseyink-Desktop-TEKNOFEST/c398d6bd-6ea3-4e42-b66e-e6eb8b13a05d/scratchpad"
BUNDLE=f"{SCR}/colab_full/outputs_colab/models"
os.environ["TEKNOFEST_OUT_DIR"]=f"{SCR}/colab_full/outputs_colab"
import numpy as np, pandas as pd, joblib, shap
from pathugen.config import ID_COL
from pathugen.data_io import load_panel
from pathugen.features import build_features, get_cat_maps

def group_of(f):
    if re.fullmatch(r"AL_([1-9]|1\d|2[0-6])", f): return "Popülasyon frekansı (AL ham)"
    if f.startswith("al_"): return "Popülasyon frekansı (AL türetilmiş)"
    if f.startswith("AL_"): return "In-silico risk skorları (AL)"
    if f.startswith("EK_") or f.startswith("ek_"): return "Evrimsel korunmuşluk (EK)"
    if f.startswith("CAT_") or f.startswith("AA_"): return "Kategorik bağlam (CAT/AA)"
    if any(f.startswith(k) for k in ("n_missing","pct_missing","missing","is_skeleton",
        "is_dense","all_ek_missing","population_block","genotype_block","n_tools")): return "Anotasyon eksiksizliği"
    if "_x_" in f or f.startswith("votes_x"): return "Etkileşim öznitelikleri"
    if any(f.startswith(k) for k in ("blosum","grantham","hydropathy","weight","volume",
        "polarity","pi","charge","helix","sheet","turn","d_","absd_","group_change","aa")): return "Biyokimyasal değişim"
    if "vote" in f or "unanimous" in f or "conflict" in f: return "In-silico oy birliği"
    return "Diğer"

for p in ["GENEL","KANSER","PAH","CFTR"]:
    ch=json.loads(open(f"{BUNDLE}/{p}/champion.json").read())
    sel=json.loads(open(f"{BUNDLE}/{p}/selected_features.json").read())
    df=load_panel(p); cm=get_cat_maps(df)
    X,y,_=build_features(df, cat_maps=cm); X=X[sel]
    model=joblib.load(f"{BUNDLE}/{p}/base_{ch['model']}.joblib")
    ex=shap.TreeExplainer(model)
    sv=ex.shap_values(X)
    sv=np.asarray(sv)
    if sv.ndim==3: sv=sv[...,1]
    absmean=np.abs(sv).mean(0)
    s=pd.Series(absmean,index=sel)
    g=s.groupby([group_of(f) for f in sel]).sum().sort_values(ascending=False)
    gp=(100*g/g.sum()).round(1)
    print(f"\n=== {p}  ({ch['model']}, {len(sel)} öznitelik, eşik {ch['threshold']:.3f}) ===")
    print("  Grup katkısı (% |SHAP| toplamı):")
    for k,v in gp.items(): print(f"    {k:38s} {v:5.1f}%")
    print("  En yüksek 6 öznitelik:", [(f, round(v,3)) for f,v in s.sort_values(ascending=False).head(6).items()])
    if p=="KANSER":
        # one concrete example: highest-confidence TP and a FP
        proba=model.predict_proba(X)[:,1]; yy=np.asarray(y)
        i=int(np.argmax(np.where(yy==1, proba, -1)))
        row=pd.Series(sv[i],index=sel).sort_values(key=np.abs,ascending=False).head(6)
        print(f"  [ÖRNEK] Variant_ID={df[ID_COL].iloc[i]} y={yy[i]} p={proba[i]:.3f}")
        for f,v in row.items(): print(f"      {f:28s} SHAP={v:+.3f}  değer={X[f].iloc[i]}")
