#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TEKNOFEST 2026 — TÜM SEÇENEKLERİ SÜPÜREN "EN İYİYİ BUL" KOŞUCUSU.

`run_final.py` (referans) ve `run_improved.py` (iyileştirmeler) betiklerinin
melezidir: ikisinin de sunduğu seçenekleri tek tek denemek yerine **hepsini bir
arada süpürür**, her yapılandırmayı sızıntısız (iç içe / nested) çapraz
doğrulamayla ölçer, AŞIRI UYUM DENETİMİNDEN geçenler arasından en iyisini seçer
ve sonunda "hangi yapılandırma, neden" diye açıkça yazar.

  run_final.py        → tek yapılandırma, marjinal One-SE, F1 eşiği
  run_improved.py     → tek yapılandırma, eşleştirilmiş One-SE, minimax eşiği
  run_best_search.py  → HEPSİNİN ÇAPRAZ KOMBİNASYONU + aşırı uyum filtresi  ← bu dosya

Süpürülen eksenler
------------------
PAHALI (yeniden eğitim gerektirir — panel başına ayrı Optuna/OOF/nested):
  * Öznitelik seçimi     : null | hybrid (ReliefF kapılı)
  * Veri artırımı oranı  : 0.0 | 0.5 | 1.0   (GENEL→panel ödünç satır oranı;
                           KANSER dâhil TÜM panellerde denenir — referans koşuda
                           KANSER'de tam artırım zarar vermişti, kısmi artırım
                           denenmemişti)

UCUZ (eğitim sonrası; aynı temel modellerden hesaplanır — bedelsiz):
  * Şampiyon seçim kuralı: marginal One-SE (run_final) | paired One-SE (run_improved)
  * Aday havuzu          : all (tekil+ortalama+stack) | singles (yalnız tekil model)
  * Karar eşiği politikası: f1 | balanced_f1 | mcc | minimax

Ucuz eksenler iç içe CV'nin HER dış katında yeniden değerlendirilir; yani
raporlanan dürüst skor gerçekten o yapılandırmanın skorudur, sonradan seçilmiş
bir kırpma değildir.

Aşırı uyum denetimi (bir yapılandırma bunlardan birini geçemezse ELENİR)
-----------------------------------------------------------------------
  1. İyimserlik sapması  : OOF F1 − dürüst F1  ≤  GAP_LIMIT (varsayılan 0.040)
  2. Şampiyon kararlılığı: aynı aday, dış katların ≥ STABILITY_MIN'inde (0.50)
                           kazanmalı. PAH'ta 10 dış katta 4 farklı kazanan çıkması
                           tam da bu testin yakaladığı aşırı uyum imzasıdır.
  3. Naif tabanı geçme   : dürüst F1 > "hepsini patojenik de" taban çizgisi

Çıktılar
--------
  outputs_best/BEST_CONFIG.md              ← insan okunur nihai karar (ÖNCE BUNU OKUYUN)
  outputs_best/best_config.json            ← makine okunur nihai karar
  outputs_best/sweep_all_results.csv       ← süpürülen HER yapılandırmanın skoru
  outputs_best/results_summary_best.csv    ← panel başına kazanan
  outputs_best/models/<PANEL>/             ← kazanan yapılandırmanın model paketi
  logs/run_best_search_<zaman>.txt         ← tam koşu günlüğü (+ _latest.txt)

Çalıştırma
----------
    cd ~/Desktop/TEKNOFEST
    nohup .venv/bin/python run_best_search.py > /dev/null 2>&1 &
    tail -f logs/run_best_search_latest.txt

    SWEEP_PROFILE=fast  ...   # daha az yapılandırma (yaklaşık yarı süre)
    SWEEP_PROFILE=decision ...# yeniden eğitim yok; yalnız ucuz eksenler (dakikalar)
    REUSE_PARAMS=1      ...   # Optuna'yı önbellekten oku
    .venv/bin/python run_best_search.py CFTR KANSER   # tek/çift panel

UYARI: run_final.py veya run_improved.py ile AYNI ANDA çalıştırmayın; ikisi de
16 iş parçacığını doldurur ve birbirini yavaşlatır.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# run_improved (ve dolaylı olarak run_final) modül düzeyinde env okuduğu için
# İÇE AKTARMADAN ÖNCE ayarlanmalı.
_DEFAULTS = {
    "QUICK": "0", "USE_GPU": "0", "RUN_NESTED": "1",
    "OPTUNA_TRIALS": "80", "OPTUNA_TIMEOUT": "600",
    "CV_REPEATS": "3", "NESTED_REPEATS": "2",
    "AUGMENT_PANELS": "PAH,CFTR",          # süpürme bu kapıyı zaten atlıyor
    "TEKNOFEST_OUT_DIR": "outputs_best",
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import joblib                                                        # noqa: E402
from sklearn.isotonic import IsotonicRegression                      # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score   # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold  # noqa: E402

import runlog                                                        # noqa: E402
import run_final as RF                                               # noqa: E402
import run_improved as RI                                            # noqa: E402


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


# --- Süpürme ayarları -------------------------------------------------------
SWEEP_PROFILE = os.environ.get("SWEEP_PROFILE", "full").strip().lower()
REUSE_PARAMS = _flag("REUSE_PARAMS", "1")     # varsayılan AÇIK: saatlerce Optuna'yı tekrarlama
DIVERSE_STACK = _flag("DIVERSE_STACK", "1")
BOOT_N = int(os.environ.get("SWEEP_BOOTSTRAP", "300"))
THR_BOOT = int(os.environ.get("THRESHOLD_BOOTSTRAP", "31"))
SEED_ENSEMBLE = int(os.environ.get("SEED_ENSEMBLE", "3"))

# --- Aşırı uyum eşikleri ----------------------------------------------------
GAP_LIMIT = float(os.environ.get("GAP_LIMIT", "0.040"))
STABILITY_MIN = float(os.environ.get("STABILITY_MIN", "0.50"))
FOLD_STD_WARN = float(os.environ.get("FOLD_STD_WARN", "0.060"))   # yalnız uyarı

# --- Nihai seçim ölçütü -----------------------------------------------------
#   robust             : min(dürüst F1, dürüst dengeli F1) — test dağılımı bilinmiyor
#   honest_f1          : eğitim dağılımı varsayımı
#   honest_balanced_f1 : 1:1 test dağılımı varsayımı (şartnamenin öngördüğü)
SELECT_OBJECTIVE = os.environ.get("SELECT_OBJECTIVE", "robust").strip().lower()
OBJECTIVES = ("robust", "honest_f1", "honest_balanced_f1")
if SELECT_OBJECTIVE not in OBJECTIVES:
    raise SystemExit(f"SELECT_OBJECTIVE '{SELECT_OBJECTIVE}' geçersiz; {OBJECTIVES}")

# --- Ucuz (karar katmanı) ızgarası ------------------------------------------
RULES = ("paired", "marginal")
POOLS = ("all", "singles")
POLICIES = RI.POLICIES                                   # f1 / balanced_f1 / mcc / minimax
DECISION_GRID = [dict(rule=r, pool=p, policy=t)
                 for r, p, t in product(RULES, POOLS, POLICIES)]

# --- Pahalı (eğitim) ızgarası ------------------------------------------------
_AUG_BY_PROFILE = {"full": (0.0, 0.5, 1.0), "wide": (0.0, 0.25, 0.5, 0.75, 1.0),
                   "fast": (0.0, 1.0), "decision": (1.0,)}
_SEL_BY_PROFILE = {"full": ("null", "hybrid"), "wide": ("null", "hybrid"),
                   "fast": ("null", "hybrid"), "decision": ("null",)}


def train_grid(panel):
    """Panel için (öznitelik seçimi, artırım oranı) yapılandırmaları."""
    sels = _SEL_BY_PROFILE.get(SWEEP_PROFILE, _SEL_BY_PROFILE["full"])
    augs = _AUG_BY_PROFILE.get(SWEEP_PROFILE, _AUG_BY_PROFILE["full"])
    if panel.upper() == RF.AUG_SOURCE:      # GENEL ödünç kaynağıdır, artırılamaz
        augs = (0.0,)
    if SWEEP_PROFILE == "decision":
        augs = (0.0,) if panel.upper() == RF.AUG_SOURCE else (1.0,)
    return [(s, a) for s in sels for a in augs]


def dc_key(dc):
    return f"{dc['rule']}|{dc['pool']}|{dc['policy']}"


def tc_key(sel, aug):
    return f"sel={sel}|aug={aug:g}"


# ===========================================================================
# Artırım havuzu — RF'nin AUGMENT_PANELS kapısını atlar (KANSER'i de denemek için)
# ===========================================================================
_POOL_CACHE: dict[str, tuple] = {}


def full_aug_pool(panel):
    """GENEL'in hedef panelde BULUNMAYAN satırları (442 öznitelikli, panel kodlamasıyla).

    RF.build_aug_pool yalnızca AUGMENT_PANELS listesindekiler için çalışır; süpürme
    KANSER'de de artırımı denemek istediğinden kapı burada bilinçli olarak atlanır.
    Sızıntı güvenliği aynen korunur: yalnızca panelde olmayan Variant_ID'ler alınır.
    """
    panel = panel.upper()
    if panel in _POOL_CACHE:
        return _POOL_CACHE[panel]
    if panel == RF.AUG_SOURCE:
        _POOL_CACHE[panel] = (None, None)
        return _POOL_CACHE[panel]
    panel_df = RF.load_panel(panel)
    source_df = RF.load_panel(RF.AUG_SOURCE)
    pool = source_df[~source_df[RF.ID_COL].isin(set(panel_df[RF.ID_COL]))].copy()
    if pool.empty:
        _POOL_CACHE[panel] = (None, None)
        return _POOL_CACHE[panel]
    X_aug, y_aug, _ = RF.build_features(pool, cat_maps=RF.get_cat_maps(panel_df))
    _POOL_CACHE[panel] = (X_aug.reset_index(drop=True),
                          pd.Series(np.asarray(y_aug)).reset_index(drop=True))
    return _POOL_CACHE[panel]


def aug_for(panel, frac, columns):
    """İstenen orandaki artırım havuzu, verilen kolonlara indirgenmiş."""
    if frac <= 0.0:
        return None
    X_aug, y_aug = full_aug_pool(panel)
    if X_aug is None:
        return None
    X_aug, y_aug = RI.subsample_aug(X_aug, y_aug, frac)
    return (X_aug.reindex(columns=list(columns)), y_aug)


# ===========================================================================
# Aday havuzu: her adayın hem eğitim-OOF hem test olasılığı
# ===========================================================================
def build_candidates(base_oof, y_tr, full_models, model_view, specs, X_te=None,
                     n_repeats=None):
    """(cand_oof, cand_te, strategies, cols_of) döndürür.

    cand_oof : aday → eğitim satırlarındaki OOF olasılığı (şampiyon seçimi için)
    cand_te  : aday → test satırlarındaki olasılık (X_te verilmişse; skorlama için)
    Stack adayları için kalibratör+meta eğitim satırlarına fit edilir, teste uygulanır.
    """
    bo = base_oof.reset_index(drop=True)
    y_tr = np.asarray(y_tr)
    cand_oof, strategies, cols_of = {}, {}, {}

    for c in bo.columns:
        cand_oof[f"single_{c}"] = bo[c].to_numpy()
        strategies[f"single_{c}"] = "single"
    cand_oof["mean"] = bo.mean(axis=1).to_numpy()
    strategies["mean"] = "mean"
    for vname, vcols in RI.stack_variants(specs).items():
        cols = [c for c in vcols if c in bo.columns]
        if len(cols) < 2:
            continue
        m_oof, _, _ = RF.evaluate_stack(bo, y_tr, cols=cols, n_repeats=n_repeats)
        cand_oof[vname] = m_oof
        strategies[vname] = vname
        cols_of[vname] = cols

    cand_te = {}
    if X_te is not None:
        base_te = {c: m.predict_proba(X_te[model_view[c]])[:, 1]
                   for c, m in full_models.items()}
        for c in bo.columns:
            cand_te[f"single_{c}"] = base_te[c]
        cand_te["mean"] = np.mean([base_te[c] for c in bo.columns], axis=0)
        for vname, cols in cols_of.items():
            calibrators, meta, order = RF.fit_final_stack(bo, y_tr, cols=cols)
            z = np.column_stack([calibrators[c].predict(base_te[c]) for c in order])
            cand_te[vname] = meta.predict_proba(z)[:, 1]
    return cand_oof, cand_te, strategies, cols_of


def bootstrap_aucs(y, cand_oof, n_boot=BOOT_N, seed=RF.SEED):
    """Tek bootstrap turundan hem marjinal hem eşleştirilmiş SE türetilir."""
    y = np.asarray(y)
    keys = list(cand_oof)
    aucs = {k: float(roc_auc_score(y, cand_oof[k])) for k in keys}
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy = y[i]
        if yy.min() == yy.max():
            continue
        for k in keys:
            boot[k].append(roc_auc_score(yy, cand_oof[k][i]))
    return aucs, {k: np.asarray(v) for k, v in boot.items()}


def pick_champion(aucs, boot, strategies, rule, pool):
    """One-SE kuralı: `marginal` = run_final'ın kuralı, `paired` = run_improved'ın."""
    keys = [k for k in aucs if pool == "all" or strategies[k] == "single"]
    if not keys:
        keys = list(aucs)
    best = max(keys, key=lambda k: aucs[k])

    if rule == "marginal":
        se = float(np.std(boot[best])) if len(boot[best]) else 0.0
        floor = aucs[best] - se
        eligible = [k for k in keys if aucs[k] >= floor]
    else:                                        # paired
        eligible = []
        for k in keys:
            if k == best:
                eligible.append(k)
                continue
            if not len(boot[k]) or not len(boot[best]):
                continue
            d = boot[best] - boot[k]
            if float(np.mean(d)) <= float(np.std(d)):
                eligible.append(k)
    eligible = eligible or [best]
    return min(eligible, key=lambda k: (RI.COMPLEXITY[strategies[k]], -aucs[k])), best


def score_at(y, proba, t):
    """Eşikte F1, dengeli F1 (1:1), MCC ve karmaşıklık matrisi."""
    y = np.asarray(y)
    f1, mcc = RI.scan(y, proba, np.array([t]))
    f1b, _ = RI.scan(y, proba, np.array([t]), RI.prior_weights(y, 0.5))
    pred = (np.asarray(proba) >= t).astype(int)
    pos = y == RF.POS
    return {"f1": float(f1[0]), "balanced_f1": float(f1b[0]), "mcc": float(mcc[0]),
            "tp": int(((pred == 1) & pos).sum()), "fp": int(((pred == 1) & ~pos).sum()),
            "fn": int(((pred == 0) & pos).sum()), "tn": int(((pred == 0) & ~pos).sum())}


# ===========================================================================
# İç içe (nested) süpürme — TÜM karar yapılandırmaları aynı katlarda ölçülür
# ===========================================================================
def nested_sweep(panel, X, y, cat_cols, params, selection, aug_frac, diverse):
    """Her dış katta seçim+model yeniden kurulur; 16 karar yapılandırması aynı
    katlarda skorlanır. Böylece yapılandırmalar arası fark yalnızca karar
    katmanından gelir, kat şansından değil."""
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    pool_X, pool_y = full_aug_pool(panel)
    if pool_X is not None and aug_frac > 0:
        pool_X, pool_y = RI.subsample_aug(pool_X, pool_y, aug_frac)
    else:
        pool_X = None

    if RF.NESTED_REPEATS > 1:
        splitter = RepeatedStratifiedKFold(n_splits=RF.NESTED_SPLITS,
                                           n_repeats=RF.NESTED_REPEATS,
                                           random_state=RF.SEED)
    else:
        splitter = StratifiedKFold(n_splits=RF.NESTED_SPLITS, shuffle=True,
                                   random_state=RF.SEED)

    acc = {dc_key(dc): {"f1": [], "balanced_f1": [], "auc": [], "champs": []}
           for dc in DECISION_GRID}

    for tr, te in splitter.split(X, y):
        X_tr_f, y_tr = X.iloc[tr], y.iloc[tr]
        X_te_f, y_te = X.iloc[te], y.iloc[te]

        tbl = RF.null_importance_selection(X_tr_f, y_tr, cat_cols,
                                           method=selection, n_null=25)
        sel = tbl.loc[tbl["keep"], "feature"].tolist()
        sc = [c for c in cat_cols if c in set(sel)]
        X_tr, X_te = X_tr_f[sel], X_te_f[sel]

        views, specs = {"full": sel}, list(RI.STD_SPECS)
        if diverse:
            rp = RI.relief_pruned_view(X_tr, y_tr, sc)
            if 3 <= len(rp) < len(sel):
                views["rp"] = rp
                specs += list(RI.DIVERSE_SPECS)

        aug = (pool_X[sel], pool_y) if pool_X is not None else None
        bo, _, full_models, model_view = RI.generate_base_oof_plus(
            X_tr, y_tr, sc, params, views, specs, aug=aug, n_repeats=1)
        cand_oof, cand_te, strategies, _ = build_candidates(
            bo, y_tr, full_models, model_view, specs, X_te=X_te, n_repeats=1)

        aucs, boot = bootstrap_aucs(y_tr, cand_oof)
        thr_cache: dict[tuple, float] = {}
        for dc in DECISION_GRID:
            champ, _ = pick_champion(aucs, boot, strategies, dc["rule"], dc["pool"])
            ck = (champ, dc["policy"])
            if ck not in thr_cache:
                thr_cache[ck] = RI.pick_threshold(y_tr.to_numpy(), cand_oof[champ],
                                                  dc["policy"], n_boot=THR_BOOT)
            s = score_at(y_te.to_numpy(), cand_te[champ], thr_cache[ck])
            a = acc[dc_key(dc)]
            a["f1"].append(s["f1"])
            a["balanced_f1"].append(s["balanced_f1"])
            a["auc"].append(roc_auc_score(y_te, cand_te[champ]))
            a["champs"].append(champ)

    out = {}
    for k, a in acc.items():
        counts = Counter(a["champs"])
        top, top_n = counts.most_common(1)[0]
        out[k] = {"honest_f1": float(np.mean(a["f1"])),
                  "honest_f1_std": float(np.std(a["f1"])),
                  "honest_balanced_f1": float(np.mean(a["balanced_f1"])),
                  "honest_auc": float(np.mean(a["auc"])),
                  "stability": top_n / len(a["champs"]),
                  "modal_champion": top,
                  "n_distinct_champions": len(counts)}
    return out


# ===========================================================================
# Panel süpürmesi
# ===========================================================================
def sweep_panel(panel):
    t_panel = time.time()
    runlog.section(f"PANEL {panel} — YAPILANDIRMA SÜPÜRMESİ")

    df = RF.load_panel(panel)
    X, y, cat_cols = RF.build_features(df)
    y = pd.Series(np.asarray(y))
    trivial = RF._trivial_f1(y)
    grid = train_grid(panel)
    print(f"  satır={len(y)}  öznitelik={X.shape[1]}  pozitif oran=%{100*y.mean():.1f}  "
          f"naif taban F1={trivial:.4f}")
    print(f"  eğitim yapılandırması: {len(grid)}   karar yapılandırması: "
          f"{len(DECISION_GRID)}   → toplam {len(grid)*len(DECISION_GRID)} aday")

    bundle_root = RF.OUTPUT_DIR / "models" / panel
    bundle_root.mkdir(parents=True, exist_ok=True)
    rows, artifacts = [], {}

    for gi, (selection, aug_frac) in enumerate(grid, 1):
        t_cfg = time.time()
        tk = tc_key(selection, aug_frac)
        print(f"\n  ── [{gi}/{len(grid)}] {tk} " + "─" * 40)

        tbl = RF.null_importance_selection(X, y, cat_cols, method=selection)
        selected = tbl.loc[tbl["keep"], "feature"].tolist()
        Xs = X[selected]
        sel_cat = [c for c in cat_cols if c in set(selected)]
        print(f"     seçim:{selection} → {len(selected)}/{X.shape[1]} öznitelik "
              f"({len(sel_cat)} kategorik)")

        aug = aug_for(panel, aug_frac, Xs.columns)
        if aug is not None:
            print(f"     artırım: +{len(aug[0])} GENEL satırı "
                  f"({int((aug[1] == 1).sum())} pat / {int((aug[1] == 0).sum())} ben)"
                  f"  [oran {aug_frac:g}]")
        else:
            print("     artırım: yok")

        # Optuna — (panel, seçim yöntemi) başına bir kez, artırım oranları arasında paylaşılır
        pfile = bundle_root / f"tuned_params_{selection}.json"
        if REUSE_PARAMS and pfile.exists():
            params = json.loads(pfile.read_text())
            print(f"     optuna: önbellekten ({pfile.name})")
        else:
            inherited = _inherit_params(panel, selection)
            if inherited is not None:
                params = inherited
                pfile.write_text(json.dumps(params, indent=2))
                print(f"     optuna: önceki koşudan devralındı ({panel}/{selection})")
            else:
                params = {}
                for kind in RF.KINDS:
                    best, sc_ = RF.tune_model(kind, Xs, y, sel_cat, RF.OPTUNA_TRIALS,
                                              RF.OPTUNA_TIMEOUT, aug=aug)
                    params[kind] = best
                    print(f"       ayarlandı {kind:11s} AUC={sc_:.4f}", flush=True)
                pfile.write_text(json.dumps(params, indent=2))

        views, specs = {"full": selected}, list(RI.STD_SPECS)
        if DIVERSE_STACK:
            rp = RI.relief_pruned_view(Xs, y, sel_cat)
            if 3 <= len(rp) < len(selected):
                views["rp"] = rp
                specs += list(RI.DIVERSE_SPECS)
                print(f"     çeşitli görünüm: {len(rp)}/{len(selected)} öznitelik → +2 sütun")

        base_oof, _, full_models, model_view = RI.generate_base_oof_plus(
            Xs, y, sel_cat, params, views, specs, aug=aug)
        cand_oof, _, strategies, cols_of = build_candidates(
            base_oof, y, full_models, model_view, specs)
        aucs, boot = bootstrap_aucs(y, cand_oof)

        print(f"     iç içe CV ({RF.NESTED_SPLITS}×{RF.NESTED_REPEATS} kat) — "
              f"{len(DECISION_GRID)} karar yapılandırması aynı katlarda ölçülüyor...",
              flush=True)
        nested = nested_sweep(panel, X, y, cat_cols, params, selection, aug_frac,
                              "rp" in views)

        thr_cache = {}
        for dc in DECISION_GRID:
            champ, best_key = pick_champion(aucs, boot, strategies,
                                            dc["rule"], dc["pool"])
            ck = (champ, dc["policy"])
            if ck not in thr_cache:
                thr_cache[ck] = RI.pick_threshold(y.to_numpy(), cand_oof[champ],
                                                  dc["policy"], n_boot=THR_BOOT)
            t = thr_cache[ck]
            oof = score_at(y.to_numpy(), cand_oof[champ], t)
            nk = nested[dc_key(dc)]
            rows.append({
                "panel": panel, "selection": selection, "aug_frac": aug_frac,
                "n_features": len(selected), "diverse": "rp" in views,
                "rule": dc["rule"], "pool": dc["pool"], "policy": dc["policy"],
                "champion": champ, "strategy": strategies[champ], "threshold": t,
                "oof_f1": oof["f1"], "oof_balanced_f1": oof["balanced_f1"],
                "oof_mcc": oof["mcc"], "oof_auc": aucs[champ],
                "tn": oof["tn"], "fp": oof["fp"], "fn": oof["fn"], "tp": oof["tp"],
                "honest_f1": nk["honest_f1"], "honest_f1_std": nk["honest_f1_std"],
                "honest_balanced_f1": nk["honest_balanced_f1"],
                "honest_auc": nk["honest_auc"],
                "stability": nk["stability"], "modal_champion": nk["modal_champion"],
                "n_distinct_champions": nk["n_distinct_champions"],
                "gap": oof["f1"] - nk["honest_f1"], "trivial_f1": trivial,
            })
        artifacts[tk] = dict(selected=selected, sel_cat=sel_cat, params=params,
                             views=views, specs=specs, base_oof=base_oof,
                             full_models=full_models, model_view=model_view,
                             cand_oof=cand_oof, strategies=strategies,
                             cols_of=cols_of, aucs=aucs, boot=boot, aug=aug, Xs=Xs)
        print(f"     [{tk} bitti — {int(time.time()-t_cfg)} sn]", flush=True)

    print(f"\n  [PANEL {panel} süpürmesi bitti — {int(time.time()-t_panel)} sn, "
          f"{len(rows)} aday değerlendirildi]")
    return pd.DataFrame(rows), artifacts, dict(X=X, y=y, cat_cols=cat_cols,
                                               trivial=trivial, df=df)


def _inherit_params(panel, selection):
    """Daha önceki koşuların ayarladığı hiperparametreleri devral (saatler kazandırır)."""
    for cand in (RF.OUTPUT_DIR / "models" / panel / f"tuned_params_{selection}.json",
                 ROOT / "outputs_improved" / "models" / panel / "tuned_params.json"):
        if cand.exists():
            try:
                data = json.loads(cand.read_text())
                # outputs_improved yalnız kendi seçim yöntemiyle ayarlandı
                if cand.name == "tuned_params.json" and \
                        selection != RI.selection_method_for(panel):
                    continue
                if set(data) >= set(RF.KINDS):
                    return data
            except (json.JSONDecodeError, OSError):
                continue
    return None


# ===========================================================================
# Aşırı uyum denetimi ve nihai seçim
# ===========================================================================
def overfit_verdict(row):
    """Bir yapılandırma için üç aşırı uyum testi. (geçti_mi, [neden...]) döndürür."""
    fails, warns = [], []
    if row["gap"] > GAP_LIMIT:
        fails.append(f"iyimserlik sapması {row['gap']:+.4f} > {GAP_LIMIT:.3f} "
                     f"(OOF {row['oof_f1']:.4f} vs dürüst {row['honest_f1']:.4f})")
    if row["stability"] < STABILITY_MIN:
        fails.append(f"şampiyon kararsız: dış katların yalnız %{100*row['stability']:.0f}"
                     f"'inde aynı aday kazandı ({row['n_distinct_champions']} farklı kazanan)")
    if row["honest_f1"] <= row["trivial_f1"]:
        fails.append(f"dürüst F1 {row['honest_f1']:.4f} naif tabanı "
                     f"({row['trivial_f1']:.4f}) geçemedi")
    if row["honest_f1_std"] > FOLD_STD_WARN:
        warns.append(f"katlar arası oynaklık yüksek (std {row['honest_f1_std']:.4f})")
    return (len(fails) == 0), fails, warns


def objective_value(row, objective):
    if objective == "honest_f1":
        return row["honest_f1"]
    if objective == "honest_balanced_f1":
        return row["honest_balanced_f1"]
    return min(row["honest_f1"], row["honest_balanced_f1"])       # robust


def choose_best(panel_df):
    """Aşırı uyum denetiminden geçenler arasından hedefe göre en iyisini seç."""
    df = panel_df.copy()
    verdicts = df.apply(lambda r: overfit_verdict(r), axis=1)
    df["passes"] = [v[0] for v in verdicts]
    df["fail_reasons"] = [" ; ".join(v[1]) for v in verdicts]
    df["warn_reasons"] = [" ; ".join(v[2]) for v in verdicts]
    for obj in OBJECTIVES:
        df[f"obj_{obj}"] = df.apply(lambda r: objective_value(r, obj), axis=1)

    clean = df[df["passes"]]
    fallback = clean.empty
    pool = df if fallback else clean
    # Eşitlikte: daha küçük sapma, sonra daha yalın strateji, sonra daha az öznitelik
    pool = pool.sort_values(
        by=[f"obj_{SELECT_OBJECTIVE}", "gap", "n_features"],
        ascending=[False, True, True])
    return df, pool.iloc[0], fallback


# ===========================================================================
# Kazanan yapılandırmanın paketini kaydet
# ===========================================================================
def persist_winner(panel, win, art, ctx):
    bundle = RF.OUTPUT_DIR / "models" / panel
    bundle.mkdir(parents=True, exist_ok=True)
    tk = tc_key(win["selection"], win["aug_frac"])
    a = art[tk]
    y = ctx["y"]

    for col, model in a["full_models"].items():
        joblib.dump(model, bundle / f"base_{col}.joblib")
    for vname, cols in a["cols_of"].items():
        calibrators, meta, order = RF.fit_final_stack(a["base_oof"], y, cols=cols)
        joblib.dump(calibrators, bundle / f"calibrators_{vname}.joblib")
        joblib.dump(meta, bundle / f"meta_{vname}.joblib")
        (bundle / f"{vname}_order.json").write_text(json.dumps(order))

    champ_proba = a["cand_oof"][win["champion"]]
    iso = IsotonicRegression(out_of_bounds="clip").fit(champ_proba, np.asarray(y))
    joblib.dump(iso, bundle / "champion_calibrator.joblib")
    train_prior = float(np.mean(np.asarray(y) == RF.POS))
    t_cal = RI.pick_threshold(np.asarray(y), iso.predict(champ_proba),
                              win["policy"], n_boot=THR_BOOT)

    seeds = 0
    if win["strategy"] == "single" and SEED_ENSEMBLE > 1:
        col = win["champion"][len("single_"):]
        kind = dict((c, k) for c, k, _ in a["specs"])[col]
        feats = a["model_view"][col]
        if a["aug"] is not None:
            X_full = pd.concat([a["Xs"][feats], a["aug"][0][feats]], axis=0,
                               ignore_index=True)
            y_full = np.concatenate([np.asarray(y), np.asarray(a["aug"][1])])
        else:
            X_full, y_full = a["Xs"][feats], np.asarray(y)
        cvc = [c for c in a["sel_cat"] if c in set(feats)]
        models = RI.fit_seed_ensemble(kind, a["params"].get(kind) or {},
                                      X_full, y_full, cvc, SEED_ENSEMBLE)
        for i, m in enumerate(models):
            joblib.dump(m, bundle / f"champion_seed{i}.joblib")
        seeds = len(models)

    champion = {
        "panel": panel,
        "strategy": win["strategy"],
        "model": win["champion"][len("single_"):] if win["strategy"] == "single" else None,
        "base_order": a["cols_of"].get(win["strategy"], list(a["base_oof"].columns)),
        "model_view": a["model_view"],
        "threshold": float(win["threshold"]),
        "threshold_policy": win["policy"],
        "selection_rule": win["rule"],
        "candidate_pool": win["pool"],
        "selection_method": win["selection"],
        "aug_frac": float(win["aug_frac"]),
        "augmented": a["aug"] is not None,
        "diverse_stack": bool(win["diverse"]),
        "n_features": int(win["n_features"]),
        "seed_ensemble": seeds,
        "train_prior": train_prior,
        "calibrated_threshold": t_cal,
        "prior_shifted_thresholds": {
            f"{p:g}": RI.prior_shift_threshold(t_cal, train_prior, p)
            for p in (0.5, 0.6, 0.74)},
        "honest_f1": float(win["honest_f1"]),
        "honest_balanced_f1": float(win["honest_balanced_f1"]),
        "honest_auc": float(win["honest_auc"]),
        "oof_f1": float(win["oof_f1"]),
        "optimism_gap": float(win["gap"]),
        "champion_stability": float(win["stability"]),
    }
    (bundle / "champion.json").write_text(json.dumps(champion, indent=2,
                                                     ensure_ascii=False))
    (bundle / "selected_features.json").write_text(json.dumps(a["selected"]))
    return champion


# ===========================================================================
# Raporlama
# ===========================================================================
def report_panel(panel, all_df, win, fallback):
    """Konsola panel kararını okunur biçimde bas."""
    passed = int(all_df["passes"].sum())
    ok, fails, warns = overfit_verdict(win)
    print()
    print("  " + "═" * 74)
    print(f"  ★ {panel} — SEÇİLEN YAPILANDIRMA")
    print("  " + "═" * 74)
    print(f"    Öznitelik seçimi     : {win['selection']}   ({win['n_features']} öznitelik)")
    print(f"    Veri artırımı        : " +
          (f"GENEL'den, oran {win['aug_frac']:g}" if win["aug_frac"] > 0 else "yok"))
    print(f"    Çeşitli görünüm      : {'açık' if win['diverse'] else 'kapalı'}")
    print(f"    Aday havuzu          : {win['pool']}"
          f"   ({'tekil+ortalama+stack' if win['pool']=='all' else 'yalnız tekil model'})")
    print(f"    Şampiyon seçim kuralı: {win['rule']} One-SE")
    print(f"    SEÇİLEN MODEL        : {win['champion']}  ({win['strategy']})")
    print(f"    KARAR EŞİĞİ          : {win['threshold']:.3f}   "
          f"(politika: {win['policy']})")
    print()
    print(f"    ── BAŞARIM ──")
    print(f"    Dürüst (iç içe) F1   : {win['honest_f1']:.4f} ± {win['honest_f1_std']:.4f}"
          f"   ← GÜVENİLECEK SAYI")
    print(f"    Dürüst dengeli F1    : {win['honest_balanced_f1']:.4f}   (test 1:1 ise)")
    print(f"    Dürüst ROC-AUC       : {win['honest_auc']:.4f}")
    print(f"    OOF F1 (iyimser)     : {win['oof_f1']:.4f}")
    print(f"    Naif taban F1        : {win['trivial_f1']:.4f}   "
          f"(kazanç {win['honest_f1']-win['trivial_f1']:+.4f})")
    print(f"    Karmaşıklık matrisi  : TN={win['tn']} FP={win['fp']} "
          f"FN={win['fn']} TP={win['tp']}")
    print()
    print(f"    ── AŞIRI UYUM DENETİMİ ──")
    g = "GEÇTİ" if win["gap"] <= GAP_LIMIT else "KALDI"
    s = "GEÇTİ" if win["stability"] >= STABILITY_MIN else "KALDI"
    b = "GEÇTİ" if win["honest_f1"] > win["trivial_f1"] else "KALDI"
    print(f"    1) İyimserlik sapması : {win['gap']:+.4f}  (sınır {GAP_LIMIT:.3f})"
          f"        → {g}")
    print(f"    2) Şampiyon kararlılığı: %{100*win['stability']:.0f} dış katta aynı aday"
          f"  (sınır %{100*STABILITY_MIN:.0f})  → {s}")
    print(f"    3) Naif tabanı geçme  : {win['honest_f1']:.4f} > {win['trivial_f1']:.4f}"
          f"          → {b}")
    if warns:
        for w in warns:
            print(f"    ⚠ uyarı: {w}")
    if fallback:
        print(f"    ✖ SONUÇ: HİÇBİR yapılandırma denetimi geçemedi — en az sapmalı olan "
              f"seçildi.\n         Bu panelin sonucuna TEMKİNLİ yaklaşın.")
    elif ok:
        print(f"    ✔ SONUÇ: AŞIRI UYUM BELİRTİSİ YOK")
    print()
    print(f"    ── NEDEN BU? ──")
    print(f"    {len(all_df)} yapılandırma denendi; {passed} tanesi aşırı uyum "
          f"denetiminden geçti, {len(all_df)-passed} tanesi elendi.")
    print(f"    Seçim ölçütü: {SELECT_OBJECTIVE}"
          + ("  = min(dürüst F1, dürüst dengeli F1)" if SELECT_OBJECTIVE == "robust" else ""))
    for obj in OBJECTIVES:
        pool = all_df[all_df["passes"]] if passed else all_df
        top = pool.sort_values(f"obj_{obj}", ascending=False).iloc[0]
        same = "  ← seçilen" if (top["rule"], top["pool"], top["policy"],
                                 top["selection"], top["aug_frac"]) == \
                                (win["rule"], win["pool"], win["policy"],
                                 win["selection"], win["aug_frac"]) else ""
        print(f"      {obj:<20s} en iyi → {top['champion']:<20s} "
              f"eşik {top['threshold']:.3f}  {top[f'obj_{obj}']:.4f}{same}")


def write_best_config_md(winners, all_rows, path):
    """İnsan okunur nihai karar dosyası — aylar sonra bakınca anlaşılsın diye."""
    L = []
    A = L.append
    A("# EN İYİ YAPILANDIRMA — nihai karar")
    A("")
    A(f"Üretim tarihi : {datetime.now():%Y-%m-%d %H:%M}")
    A(f"Üreten betik  : `run_best_search.py`  (profil: `{SWEEP_PROFILE}`)")
    A(f"Seçim ölçütü  : `{SELECT_OBJECTIVE}`"
      + ("  = min(dürüst F1, dürüst dengeli F1) — test sınıf oranı bilinmediği için"
         if SELECT_OBJECTIVE == "robust" else ""))
    A(f"Aşırı uyum sınırları : iyimserlik sapması ≤ {GAP_LIMIT:.3f} · "
      f"şampiyon kararlılığı ≥ %{100*STABILITY_MIN:.0f} · dürüst F1 > naif taban")
    A("")
    A("**Hangi sayıya bakmalı?** `Dürüst F1`, iç içe (nested) çapraz doğrulamadan gelir: "
      "öznitelik seçimi, model seçimi ve karar eşiği her dış katta yalnız eğitim "
      "satırlarında belirlenip dokunulmamış test katında bir kez ölçülür. OOF F1 "
      "iyimserdir, model seçimi için kullanılır; genelleme tahmini DEĞİLDİR.")
    A("")
    A("## Özet")
    A("")
    A("| Panel | Model | Eşik | Politika | Öznitelik | Artırım | Dürüst F1 | Dürüst dengeli F1 | Dürüst AUC | Sapma | Kararlılık | Aşırı uyum |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p, w in winners.items():
        ok, fails, _ = overfit_verdict(w)
        A(f"| {p} | {w['champion']} | {w['threshold']:.3f} | {w['policy']} | "
          f"{w['selection']}/{w['n_features']} | {w['aug_frac']:g} | "
          f"**{w['honest_f1']:.4f}** | {w['honest_balanced_f1']:.4f} | "
          f"{w['honest_auc']:.4f} | {w['gap']:+.4f} | %{100*w['stability']:.0f} | "
          f"{'temiz' if ok else 'RİSKLİ'} |")
    A("")
    hf = np.mean([w["honest_f1"] for w in winners.values()])
    hb = np.mean([w["honest_balanced_f1"] for w in winners.values()])
    ha = np.mean([w["honest_auc"] for w in winners.values()])
    A(f"**Ortalama dürüst F1 = {hf:.4f}** · ortalama dürüst dengeli F1 = {hb:.4f} · "
      f"ortalama dürüst AUC = {ha:.4f}")
    A("")
    for p, w in winners.items():
        ok, fails, warns = overfit_verdict(w)
        A(f"## {p}")
        A("")
        A("| Ayar | Değer |")
        A("|---|---|")
        A(f"| Öznitelik seçimi | `{w['selection']}` ({w['n_features']} öznitelik) |")
        A(f"| Veri artırımı | " +
          (f"GENEL'den, oran {w['aug_frac']:g}" if w["aug_frac"] > 0 else "yok") + " |")
        A(f"| Çeşitli görünüm (ReliefF) | {'açık' if w['diverse'] else 'kapalı'} |")
        A(f"| Aday havuzu | `{w['pool']}` |")
        A(f"| Şampiyon seçim kuralı | `{w['rule']}` One-SE |")
        A(f"| **Seçilen model** | **{w['champion']}** ({w['strategy']}) |")
        A(f"| **Karar eşiği** | **{w['threshold']:.3f}** (politika `{w['policy']}`) |")
        A(f"| Dürüst F1 | **{w['honest_f1']:.4f} ± {w['honest_f1_std']:.4f}** |")
        A(f"| Dürüst dengeli F1 (1:1) | {w['honest_balanced_f1']:.4f} |")
        A(f"| Dürüst ROC-AUC | {w['honest_auc']:.4f} |")
        A(f"| OOF F1 (iyimser) | {w['oof_f1']:.4f} |")
        A(f"| Naif taban F1 | {w['trivial_f1']:.4f} |")
        A(f"| TN / FP / FN / TP | {w['tn']} / {w['fp']} / {w['fn']} / {w['tp']} |")
        A("")
        A("**Aşırı uyum denetimi**")
        A("")
        A(f"- İyimserlik sapması `{w['gap']:+.4f}` (sınır {GAP_LIMIT:.3f}) — "
          f"{'GEÇTİ' if w['gap'] <= GAP_LIMIT else 'KALDI'}")
        A(f"- Şampiyon kararlılığı `%{100*w['stability']:.0f}` "
          f"({w['n_distinct_champions']} farklı kazanan, sınır %{100*STABILITY_MIN:.0f}) — "
          f"{'GEÇTİ' if w['stability'] >= STABILITY_MIN else 'KALDI'}")
        A(f"- Naif tabanı geçme — "
          f"{'GEÇTİ' if w['honest_f1'] > w['trivial_f1'] else 'KALDI'}")
        for wr in warns:
            A(f"- ⚠ {wr}")
        A("")
        A(f"**Sonuç: {'AŞIRI UYUM BELİRTİSİ YOK' if ok else 'RİSKLİ — ' + '; '.join(fails)}**")
        A("")
        sub = all_rows[all_rows["panel"] == p]
        A(f"Bu panelde {len(sub)} yapılandırma denendi, {int(sub['passes'].sum())} tanesi "
          f"denetimi geçti. En iyi 5 (geçenler arasında, `{SELECT_OBJECTIVE}` ölçütüne göre):")
        A("")
        A("| # | seçim | artırım | kural | havuz | politika | model | eşik | Dürüst F1 | Dengeli F1 | Sapma |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        pool = sub[sub["passes"]] if sub["passes"].any() else sub
        for i, (_, r) in enumerate(pool.sort_values(f"obj_{SELECT_OBJECTIVE}",
                                                    ascending=False).head(5).iterrows(), 1):
            A(f"| {i} | {r['selection']} | {r['aug_frac']:g} | {r['rule']} | {r['pool']} | "
              f"{r['policy']} | {r['champion']} | {r['threshold']:.3f} | "
              f"{r['honest_f1']:.4f} | {r['honest_balanced_f1']:.4f} | {r['gap']:+.4f} |")
        A("")
    A("## Bu paket nasıl kullanılır?")
    A("")
    A("Her panelin kazanan modeli `outputs_best/models/<PANEL>/` altındadır; "
      "`champion.json` seçilen modeli, eşiği ve tüm ayarları taşır. Çıkarım için "
      "`run_improved.py` içindeki `predict_panel_plus()` kullanılabilir "
      "(`TEKNOFEST_OUT_DIR=outputs_best` ile).")
    A("")
    A("Test setinin sınıf oranı eğitimden farklı çıkarsa modeli yeniden eğitmeye gerek "
      "yoktur: `champion.json` içindeki `prior_shifted_thresholds` hazır eşikleri verir "
      "(izotonik kalibre olasılıklar üzerinde).")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def compare_previous(winners):
    """Varsa run_final ve run_improved sonuçlarıyla yan yana."""
    refs = {"run_final (referans)": ROOT / "outputs_final" / "results_summary.csv",
            "run_improved": ROOT / "outputs_improved" / "results_summary_improved.csv",
            "PDR (Colab)": None}
    colab = None
    zpath = ROOT / "outputs_colab.zip"
    if zpath.exists():
        import zipfile
        try:
            with zipfile.ZipFile(zpath) as z:
                import io
                colab = pd.read_csv(io.BytesIO(
                    z.read("outputs_colab/results_summary.csv")))
        except Exception:
            colab = None

    runlog.section("ÖNCEKİ KOŞULARLA KARŞILAŞTIRMA (dürüst / iç içe F1)")
    rows = []
    for p, w in winners.items():
        row = {"panel": p, "run_best_search": round(w["honest_f1"], 4)}
        for label, path in refs.items():
            if path is None:
                continue
            if path.exists():
                d = pd.read_csv(path)
                m = d[d["panel"] == p]
                if not m.empty and "honest_f1" in d.columns:
                    row[label] = round(float(m.iloc[0]["honest_f1"]), 4)
        if colab is not None:
            m = colab[colab["panel"] == p]
            if not m.empty:
                row["PDR (Colab)"] = round(float(m.iloc[0]["honest_f1"]), 4)
        rows.append(row)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    for c in t.columns:
        if c in ("panel", "run_best_search"):
            continue
        d = (t["run_best_search"] - t[c]).mean()
        print(f"  run_best_search − {c:<22s} ortalama fark: {d:+.4f}")
    print("\nNot: yalnız 'dürüst (iç içe) F1' karşılaştırılır — OOF skorları iyimserdir "
          "ve\nkoşular arasında karşılaştırılabilir değildir.")


# ===========================================================================
# Ana akış
# ===========================================================================
def main():
    panels = [a.upper() for a in sys.argv[1:]] or RF.PANELS
    unknown = [p for p in panels if p not in RF.PANEL_FILES]
    if unknown:
        raise SystemExit(f"Bilinmeyen panel: {unknown}. Seçenekler: {RF.PANELS}")

    log_path = runlog.start(
        "run_best_search",
        note="TÜM seçenekleri süpürür, aşırı uyum denetiminden geçenler arasından "
             "en iyisini seçer")

    n_train = sum(len(train_grid(p)) for p in panels)
    print(f"\nPaneller   : {panels}")
    print(f"Çıktı      : {RF.OUTPUT_DIR.resolve()}")
    print(f"Profil     : {SWEEP_PROFILE}")
    print(f"\nSüpürülecek eksenler")
    print(f"  PAHALI (yeniden eğitim):")
    for p in panels:
        g = train_grid(p)
        print(f"    {p:<7s} {len(g)} yapılandırma → " +
              ", ".join(tc_key(s, a) for s, a in g))
    print(f"  UCUZ (eğitim sonrası, bedelsiz):")
    print(f"    kural   : {', '.join(RULES)}")
    print(f"    havuz   : {', '.join(POOLS)}")
    print(f"    politika: {', '.join(POLICIES)}")
    print(f"    → yapılandırma başına {len(DECISION_GRID)} karar kombinasyonu")
    print(f"\nToplam değerlendirilecek aday: {n_train * len(DECISION_GRID)}")
    print(f"\nAşırı uyum denetimi")
    print(f"  iyimserlik sapması ≤ {GAP_LIMIT:.3f}   şampiyon kararlılığı ≥ "
          f"%{100*STABILITY_MIN:.0f}   dürüst F1 > naif taban")
    print(f"Seçim ölçütü : {SELECT_OBJECTIVE}")
    print(f"Optuna       : {RF.OPTUNA_TRIALS} deneme / {RF.OPTUNA_TIMEOUT} s  "
          f"(REUSE_PARAMS={int(REUSE_PARAMS)}, seçim yöntemi başına bir kez)")
    print(f"İç içe CV    : {RF.NESTED_SPLITS} kat × {RF.NESTED_REPEATS} tekrar")

    RF.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_rows, winners, fallbacks = [], {}, {}
    sweep_csv = RF.OUTPUT_DIR / "sweep_all_results.csv"

    for panel in panels:
        pdf, artifacts, ctx = sweep_panel(panel)
        scored, win, fallback = choose_best(pdf)
        all_rows.append(scored)
        pd.concat(all_rows, ignore_index=True).to_csv(sweep_csv, index=False)
        report_panel(panel, scored, win, fallback)
        champ = persist_winner(panel, win, artifacts, ctx)
        winners[panel] = win
        fallbacks[panel] = fallback
        print(f"    paket → {RF.OUTPUT_DIR / 'models' / panel}")
        del artifacts        # büyük model nesnelerini bellekte tutma

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(sweep_csv, index=False)

    elapsed = int(time.time() - t0)
    runlog.section(f"SÜPÜRME TAMAMLANDI — {elapsed//3600} sa {(elapsed%3600)//60} dk")

    summary = pd.DataFrame([{
        "panel": p, "champion": w["champion"], "strategy": w["strategy"],
        "selection": w["selection"], "aug_frac": w["aug_frac"],
        "n_features": w["n_features"], "rule": w["rule"], "pool": w["pool"],
        "policy": w["policy"], "threshold": round(w["threshold"], 3),
        "oof_f1": round(w["oof_f1"], 4),
        "honest_f1": round(w["honest_f1"], 4),
        "honest_balanced_f1": round(w["honest_balanced_f1"], 4),
        "honest_auc": round(w["honest_auc"], 4),
        "optimism_gap": round(w["gap"], 4),
        "stability": round(w["stability"], 2),
        "overfit_clean": overfit_verdict(w)[0],
        "trivial_f1": round(w["trivial_f1"], 4),
    } for p, w in winners.items()])
    summary.to_csv(RF.OUTPUT_DIR / "results_summary_best.csv", index=False)

    runlog.section("NİHAİ KARAR — PANEL BAŞINA SEÇİLEN YAPILANDIRMA")
    print(summary.to_string(index=False))
    print(f"\nOrtalama dürüst F1          : {summary['honest_f1'].mean():.4f}")
    print(f"Ortalama dürüst dengeli F1  : {summary['honest_balanced_f1'].mean():.4f}")
    print(f"Ortalama dürüst AUC         : {summary['honest_auc'].mean():.4f}")
    print(f"Ortalama iyimserlik sapması : {summary['optimism_gap'].mean():+.4f}")

    risky = [p for p, w in winners.items() if not overfit_verdict(w)[0]]
    print()
    if risky:
        print(f"⚠ AŞIRI UYUM RİSKİ: {', '.join(risky)} panelinde hiçbir yapılandırma "
              f"denetimden geçemedi.\n  Bu panellerin sonucuna temkinli yaklaşın "
              f"(ayrıntı: BEST_CONFIG.md).")
    else:
        print("✔ TÜM PANELLER AŞIRI UYUM DENETİMİNDEN GEÇTİ.")

    compare_previous(winners)

    md = RF.OUTPUT_DIR / "BEST_CONFIG.md"
    write_best_config_md(winners, all_df, md)
    (RF.OUTPUT_DIR / "best_config.json").write_text(json.dumps(
        {p: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in w.items() if k not in ("fail_reasons", "warn_reasons")}
         for p, w in winners.items()}, indent=2, ensure_ascii=False, default=str))

    runlog.section("ÇIKTILAR")
    print(f"  İNSAN OKUNUR KARAR : {md}          ← ÖNCE BUNU OKUYUN")
    print(f"  Makine okunur      : {RF.OUTPUT_DIR / 'best_config.json'}")
    print(f"  Panel özeti        : {RF.OUTPUT_DIR / 'results_summary_best.csv'}")
    print(f"  Tüm süpürme        : {sweep_csv}  ({len(all_df)} satır)")
    print(f"  Model paketleri    : {RF.OUTPUT_DIR / 'models'}/")
    print(f"  Günlük             : {log_path}")


if __name__ == "__main__":
    main()
