import os, importlib.util, warnings, itertools
warnings.filterwarnings("ignore")
os.environ.setdefault("TEKNOFEST_OUT_DIR","/tmp/x_unused_out")
spec=importlib.util.spec_from_file_location("rf","run_final.py")
rf=importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
import numpy as np, pandas as pd

D={p: rf.load_panel(p) for p in rf.PANELS}
print("\n--- 1. Boyut / sınıf dağılımı ---")
for p,df in D.items():
    y=df[rf.LABEL_COL]
    print(f"{p:7s} n={len(df):5d} cols={df.shape[1]} pos={int((y==1).sum()):5d} neg={int((y==0).sum()):4d} "
          f"pos%={100*y.mean():.1f}  medyan_eksik={int(df[rf.FEATURE_COLS].isnull().sum(axis=1).median())}")

print("\n--- 2. Yinelenen Variant_ID (panel içi) ---")
for p,df in D.items():
    print(f"{p:7s} dup_id={int(df[rf.ID_COL].duplicated().sum())}")

print("\n--- 3. Yinelenen tam öznitelik profili (panel içi) ---")
for p,df in D.items():
    dup=df.duplicated(subset=rf.FEATURE_COLS, keep=False)
    n=int(dup.sum())
    conflict=0
    if n:
        g=df[dup].groupby(list(rf.FEATURE_COLS), dropna=False)[rf.LABEL_COL].nunique()
        conflict=int((g>1).sum())
    print(f"{p:7s} ayni_profil_satir={n}  celiskili_etiketli_grup={conflict}")

print("\n--- 4. Paneller arası Variant_ID örtüşmesi ve etiket tutarlılığı ---")
for a,b in itertools.combinations(rf.PANELS,2):
    m=D[a].merge(D[b][[rf.ID_COL,rf.LABEL_COL]], on=rf.ID_COL, suffixes=("_a","_b"))
    if len(m):
        agree=int((m[rf.LABEL_COL+"_a"]==m[rf.LABEL_COL+"_b"]).sum())
        print(f"{a:7s}∩{b:7s} ortak={len(m):4d} etiket_uyum={agree}/{len(m)} ({100*agree/len(m):.1f}%)")

print("\n--- 5. Sabit / tamamen boş kolonlar ---")
for p,df in D.items():
    allna=[c for c in rf.FEATURE_COLS if df[c].isnull().all()]
    const=[c for c in rf.FEATURE_COLS if df[c].nunique(dropna=True)==1]
    print(f"{p:7s} tamamen_bos={len(allna)} tek_degerli={len(const)}")

print("\n--- 6. Aykırı değer taraması: AL_1..AL_26 (allel frekansı) aralık ihlali ---")
AF=[f"AL_{i}" for i in range(1,27)]
for p,df in D.items():
    v=df[AF]
    print(f"{p:7s} <0: {int((v<0).sum().sum())}  >1: {int((v>1).sum().sum())}  max={np.nanmax(v.to_numpy()):.4g}")

print("\n--- 7. EK skorları aralık ---")
for p,df in D.items():
    v=df[rf.EK_COLS]
    print(f"{p:7s} min={np.nanmin(v.to_numpy()):.3f} max={np.nanmax(v.to_numpy()):.3f}")

print("\n--- 8. CAT kolonlarında saflık (>%95 tek etiket) — sızıntı proxy taraması ---")
for p,df in D.items():
    flags=[]
    for c in rf.CAT_COLS:
        g=df.groupby(c, dropna=False)[rf.LABEL_COL].agg(["count","mean"])
        g=g[g["count"]>=20]
        hits=g[(g["mean"]>=0.95)|(g["mean"]<=0.05)]
        if len(hits): flags.append(f"{c}:{len(hits)}kat/{int(hits['count'].sum())}satir")
    print(f"{p:7s} yuksek_saflik_kategori: {flags if flags else 'yok'}")

print("\n--- 9. AA_1/AA_2 standart dışı ---")
for p,df in D.items():
    a1=df["AA_1"].astype("string"); a2=df["AA_2"].astype("string")
    ns=int(((~a1.isin(rf.STD_AA))&a1.notna()).sum()+((~a2.isin(rf.STD_AA|{"*"}))&a2.notna()).sum())
    print(f"{p:7s} standart_disi_AA={ns}  eksik_AA={int((a1.isna()|a2.isna()).sum())}")

print("\n--- 10. Eksiklik iki-modluluğu ---")
for p,df in D.items():
    nm=df[rf.FEATURE_COLS].isnull().sum(axis=1)
    print(f"{p:7s} <=20 eksik: {int((nm<=20).sum()):5d}  >=301 eksik: {int((nm>=301).sum()):5d}  arada: {int(((nm>20)&(nm<301)).sum()):5d}")
