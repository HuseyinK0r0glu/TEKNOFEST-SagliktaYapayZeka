#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TEKNOFEST 2026 — iyileştirme denemeleri koşucusu (run_final.py'nin üzerine).

Bu betik `run_final.py`'yi DEĞİŞTİRMEZ; onu bir kütüphane gibi içe aktarır ve
öneri listesindeki iyileştirmeleri onun üzerine ekler. Böylece iki koşu
doğrudan karşılaştırılabilir:

    outputs_final/results_summary.csv        ← run_final.py   (referans)
    outputs_improved/results_summary_improved.csv  ← bu betik (iyileştirmeli)

Eklenen iyileştirmeler (hepsi ortam değişkeniyle kapatılabilir)
--------------------------------------------------------------
1. ÖNCELE DUYARSIZ KARAR EŞİĞİ  (THRESHOLD_POLICY, varsayılan "minimax")
   Eşik artık yalnızca eğitim önceline göre F1'i maksimize etmiyor. Dört politika
   birlikte hesaplanıp günlüğe tablo olarak basılıyor; gönderilen eşik seçilen
   politikadır:
     f1           — mevcut davranış (referans)
     balanced_f1  — 1:1 test dağılımı varsayımıyla yeniden ağırlıklandırılmış F1
     mcc          — MCC'yi maksimize eden eşik
     minimax      — min(F1@eğitim önceli, F1@1:1) en yüksek olan eşik  ← varsayılan
   Eşik, tek bir bölünmeye aşırı uymasın diye bootstrap-medyanıyla sabitlenir.
   Ayrıca kalibre olasılıklar ve eğitim önceli paketle saklanır; böylece test
   dağılımı farklı çıkarsa eşik yeniden eğitim olmadan taşınabilir.

2. EŞLEŞTİRİLMİŞ BOOTSTRAP ile One-SE  (PAIRED_BOOTSTRAP, varsayılan 400)
   Eski kural adayı "en iyinin marjinal SE'si" ile karşılaştırıyordu; n=111'de bu
   bant o kadar geniş ki her zaman en yalın aday kazanıyordu. Artık (en iyi − aday)
   FARKININ eşleştirilmiş bootstrap dağılımı kullanılıyor — çok daha dar bir bant,
   yani gerçekten daha iyi bir topluluk kazanabiliyor.

3. ÇEŞİTLİ GÖRÜNÜMLÜ STACKING  (DIVERSE_STACK, varsayılan 1)
   Beş temel model aynı öznitelik kümesini görüyordu. XGBoost ve LightGBM için
   ReliefF ile budanmış ikinci bir öznitelik görünümü eklenir (yeni Optuna yok,
   ayarlı parametreler yeniden kullanılır) → meta katman 7 sütun görür.

4. KISAYOL DENETİMİ + ABLASYON  (SHORTCUT_AUDIT, varsayılan 1)
   Anotasyon eksiksizliğinin tek başına ROC-AUC'si ölçülür ve şampiyon, eksiklik
   öznitelikleri çıkarılmış hâlde yeniden değerlendirilir → dağılım kaymasına karşı
   ne kadar kırılgan olduğumuzu sayıyla gösterir.

5. PANEL BAZLI ÖZNİTELİK SEÇİCİ
   KANSER varsayılan olarak `hybrid` (ReliefF kapılı), diğerleri `null`.
   SELECTION_METHOD_<PANEL> ile panel bazında ezilebilir.

6. TOHUM TOPLULUĞU  (SEED_ENSEMBLE, varsayılan 3)
   Nihai şampiyon birden çok tohumla yeniden eğitilir, tahminler ortalanır.

7. AUG_FRAC ile ödünç alınan GENEL satırlarının oranı ayarlanabilir (ağırlık
   düşürmenin muadili). NESTED_REPEATS referansla aynı (2) tutulur ki karşılaştırma
   adil olsun; daha kararlı dürüst tahmin için NESTED_REPEATS=3 veya 5 verin.

Çalıştırma
----------
    QUICK=0 USE_GPU=0 .venv/bin/python run_improved.py            # dört panel
    .venv/bin/python run_improved.py CFTR KANSER                  # tek/çift panel
    REUSE_PARAMS=1 .venv/bin/python run_improved.py               # Optuna'yı atla

Her koşu logs/run_improved_<zaman>.txt dosyasına da yazılır
(logs/run_improved_latest.txt son koşuya işaret eder).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# --- run_final.py knob'larını İÇE AKTARMADAN ÖNCE ayarla (modül düzeyinde okunuyor) ---
_DEFAULTS = {
    "QUICK": "0",
    "USE_GPU": "0",
    "RUN_NESTED": "1",
    "AUGMENT_PANELS": "PAH,CFTR",
    "SELECTION_METHOD": "null",
    "OPTUNA_TRIALS": "80",
    "OPTUNA_TIMEOUT": "600",
    "CV_REPEATS": "3",
    # run_final ile AYNI (2): "Δ Dürüst F1" farkı tekrar sayısından değil,
    # iyileştirmelerden gelsin. Daha kararlı tahmin için NESTED_REPEATS=3/5 verin.
    "NESTED_REPEATS": "2",
    "TEKNOFEST_OUT_DIR": "outputs_improved",
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)

import numpy as np                                                      # noqa: E402
import pandas as pd                                                     # noqa: E402
import joblib                                                           # noqa: E402
from sklearn.isotonic import IsotonicRegression                         # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score      # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold  # noqa: E402

import runlog                                                           # noqa: E402
import run_final as RF                                                  # noqa: E402


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


# --- Bu betiğe özgü knob'lar ------------------------------------------------
THRESHOLD_POLICY = os.environ.get("THRESHOLD_POLICY", "minimax").strip().lower()
PAIRED_BOOTSTRAP = int(os.environ.get("PAIRED_BOOTSTRAP", "400"))
DIVERSE_STACK = _flag("DIVERSE_STACK", "1")
SHORTCUT_AUDIT = _flag("SHORTCUT_AUDIT", "1")
SEED_ENSEMBLE = int(os.environ.get("SEED_ENSEMBLE", "3"))
AUG_FRAC = float(os.environ.get("AUG_FRAC", "1.0"))
REUSE_PARAMS = _flag("REUSE_PARAMS", "0")
COMPARE_TO = os.environ.get("COMPARE_TO", "outputs_final/results_summary.csv")
THR_BOOT = int(os.environ.get("THRESHOLD_BOOTSTRAP", "31"))
REPORT_PRIORS = [float(x) for x in
                 os.environ.get("REPORT_PRIORS", "0.5,0.6,0.74").split(",") if x.strip()]

POLICIES = ("f1", "balanced_f1", "mcc", "minimax")
if THRESHOLD_POLICY not in POLICIES:
    raise SystemExit(f"THRESHOLD_POLICY '{THRESHOLD_POLICY}' geçersiz; {POLICIES}")

# Panel bazlı öznitelik seçici (held-out A/B'de yalnız KANSER gürültüyü aştı).
PANEL_SELECTION = {"KANSER": "hybrid"}

# Aday karmaşıklığı — One-SE eşitliğinde daima en yalın olan kazanır.
COMPLEXITY = {"single": 0, "mean": 1, "stacked_trees": 2, "stacked_all": 3,
              "stacked_diverse": 4}

# (sütun adı, model türü, öznitelik görünümü)
STD_SPECS = tuple((k, k, "full") for k in RF.KINDS)
DIVERSE_SPECS = (("xgboost_rp", "xgboost", "rp"), ("lightgbm_rp", "lightgbm", "rp"))

# Eksiklik ("anotasyon eksiksizliği") öznitelikleri — kısayol denetimi için.
MISS_KEYS = ("n_missing", "pct_missing", "missingness_tier", "is_skeleton", "is_dense",
             "all_ek_missing", "population_block_missing", "genotype_block_missing",
             "n_tools_available", "missingtier_x_ekcount", "ek_mean_x_completeness")

THR_GRID = np.arange(0.05, 0.95 + 1e-9, 0.005)


# ===========================================================================
# Eşik tarama araçları — ağırlıklı F1/MCC'yi tüm eşikler için vektörel hesaplar
# ===========================================================================
def prior_weights(y, prior):
    """Hedef sınıf oranı `prior` olan bir test setini taklit eden örnek ağırlıkları.

    Pozitiflerin toplam ağırlığı `prior`, negatiflerinki `1 - prior` olur; böylece
    ağırlıklı F1, "test seti bu oranda olsaydı" beklenen F1'i verir.
    """
    y = np.asarray(y)
    pos = y == RF.POS
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if prior is None or n_pos == 0 or n_neg == 0:
        return None
    return np.where(pos, prior / n_pos, (1.0 - prior) / n_neg)


def scan(y, proba, ts=THR_GRID, w=None):
    """Her eşik için (ağırlıklı) F1 ve MCC dizilerini döndürür."""
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    pos = y == RF.POS

    order = np.argsort(-proba)
    p_s, w_s, pos_s = proba[order], w[order], pos[order]
    cum_w = np.cumsum(w_s)
    cum_tp = np.cumsum(w_s * pos_s)

    # proba >= t olan örnek sayısı
    k = np.searchsorted(-p_s, -np.asarray(ts, dtype=float), side="right")
    take = np.clip(k - 1, 0, len(p_s) - 1)
    Pw = np.where(k > 0, cum_w[take], 0.0)
    TPw = np.where(k > 0, cum_tp[take], 0.0)

    APw, ANw = w[pos].sum(), w[~pos].sum()
    FPw = Pw - TPw
    FNw = APw - TPw
    TNw = ANw - FPw

    f1 = np.where(Pw + APw > 0, 2.0 * TPw / (Pw + APw), 0.0)
    den = np.sqrt((TPw + FPw) * (TPw + FNw) * (TNw + FPw) * (TNw + FNw))
    mcc = np.where(den > 0, (TPw * TNw - FPw * FNw) / den, 0.0)
    return f1, mcc


def _policy_scores(y, proba, policy, ts=THR_GRID):
    if policy == "f1":
        return scan(y, proba, ts)[0]
    if policy == "balanced_f1":
        return scan(y, proba, ts, prior_weights(y, 0.5))[0]
    if policy == "mcc":
        return scan(y, proba, ts)[1]
    if policy == "minimax":
        f1_train = scan(y, proba, ts)[0]
        f1_bal = scan(y, proba, ts, prior_weights(y, 0.5))[0]
        return np.minimum(f1_train, f1_bal)
    raise ValueError(policy)


def pick_threshold(y, proba, policy, n_boot=THR_BOOT, seed=RF.SEED):
    """Politikanın optimum eşiği; bootstrap-medyanıyla sabitlenir.

    run_final'daki fold-medyan eşiğin muadili: tek bir örneklemin gürültüsüne
    aşırı uymayı engeller.
    """
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    picks = [float(THR_GRID[int(np.argmax(_policy_scores(y, proba, policy)))])]
    n = len(y)
    for _ in range(max(0, n_boot - 1)):
        i = rng.integers(0, n, n)
        yy = y[i]
        if yy.min() == yy.max():
            continue
        picks.append(float(THR_GRID[int(np.argmax(_policy_scores(yy, proba[i], policy)))]))
    return float(np.median(picks))


def metrics_at(y, proba, t, priors=()):
    """Verilen eşikte tüm raporlanan metrikler."""
    y = np.asarray(y)
    pred = (np.asarray(proba) >= t).astype(int)
    pos = y == RF.POS
    tp = int(((pred == 1) & pos).sum())
    fp = int(((pred == 1) & ~pos).sum())
    fn = int(((pred == 0) & pos).sum())
    tn = int(((pred == 0) & ~pos).sum())
    f1_arr, mcc_arr = scan(y, proba, np.array([t]))
    f1_t, mcc_t = float(f1_arr[0]), float(mcc_arr[0])
    out = {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": tp / (tp + fp) if tp + fp else 0.0,
           "recall": tp / (tp + fn) if tp + fn else 0.0,
           "f1": f1_t, "mcc": mcc_t}
    for p in priors:
        out[f"f1@{p:g}"] = float(scan(y, proba, np.array([t]), prior_weights(y, p))[0][0])
    return out


def prior_shift_threshold(t_ref, prior_train, prior_target):
    """Kalibre olasılıklar için eşiğin hedef öncele taşınmış hâli.

    p' = p·r / (p·r + (1−p)·s),  r = π*/π_eğitim,  s = (1−π*)/(1−π_eğitim)
    olduğundan p' >= t_ref eşiği, ham p üzerinde şu eşiğe denktir.
    """
    r = prior_target / prior_train
    s = (1.0 - prior_target) / (1.0 - prior_train)
    return float(t_ref * s / (t_ref * s + (1.0 - t_ref) * r))


# ===========================================================================
# Temel model OOF üretimi — her sütunun kendi öznitelik görünümü olabilir
# ===========================================================================
def relief_pruned_view(X, y, cat_cols, ig_protect_k=None):
    """Negatif ReliefF ağırlıklı öznitelikleri düşüren ikinci görünüm.

    Bilgi kazancında ilk `ig_protect_k` içinde olanlar ve kategorikler korunur —
    küresel olarak güçlü bir öznitelik yerel ölçüt yüzünden sessizce düşmesin.
    Koruma sayısı küme boyutuna göre ölçeklenir: CFTR gibi 13 öznitelikli
    panellerde sabit 15 tüm kümeyi koruyup budamayı anlamsız kılıyordu.
    """
    cols = list(X.columns)
    if ig_protect_k is None:
        ig_protect_k = min(15, max(3, len(cols) // 3))
    try:
        rfw = RF.relieff(X, y).reindex(cols).fillna(0.0)
        ig = RF.information_gain(X, y).reindex(cols).fillna(-np.inf)
    except Exception as exc:                       # filtreler çökerse tam görünüme dön
        print(f"    [uyarı] ReliefF görünümü hesaplanamadı ({exc}); tam görünüm kullanılıyor")
        return cols
    protected = set(ig.sort_values(ascending=False).head(ig_protect_k).index) | set(cat_cols or [])
    keep = [c for c in cols if rfw[c] >= 0.0 or c in protected]
    return keep


def generate_base_oof_plus(X, y, cat_cols, params_by_kind, views, specs,
                           aug=None, n_repeats=None):
    """RF.generate_base_oof'un görünüm farkındalıklı sürümü.

    Döndürür: (base_oof DataFrame, singles, tam-veri modelleri, sütun→görünüm).
    Artırım satırları yalnızca eğitim fold'larına ve nihai yeniden eğitime girer.
    """
    base_oof, singles, full_models, model_view = {}, {}, {}, {}
    aug_X, aug_y = aug if aug is not None else (None, None)

    for col, kind, vkey in specs:
        feats = [c for c in views[vkey] if c in X.columns]
        if not feats:
            continue
        Xv = X[feats]
        cv_cat = [c for c in (cat_cols or []) if c in set(feats)]
        fit_fn = RF._fit_fn(kind, cv_cat)
        params = params_by_kind.get(kind) or {}

        def factory(y_tr, _k=kind, _p=params, _f=feats, _c=tuple(cv_cat)):
            return RF.MAKERS[_k](_p, y_tr, list(_f), list(_c))

        augv = (aug_X[feats], aug_y) if aug_X is not None else None
        res = RF.run_cv(factory, Xv, y, cat_cols=cv_cat, fit_kwargs_fn=fit_fn,
                        aug=augv, n_repeats=n_repeats)
        base_oof[col] = res["oof"]
        singles[col] = {"f1": res["f1"], "threshold": res["threshold"], "auc": res["auc"]}

        if aug_X is not None:
            X_full = pd.concat([Xv, aug_X[feats]], axis=0, ignore_index=True)
            y_full = np.concatenate([np.asarray(y), np.asarray(aug_y)])
        else:
            X_full, y_full = Xv, np.asarray(y)
        model = factory(pd.Series(y_full))
        model.fit(X_full, y_full, **(fit_fn(X_full, y_full) if fit_fn else {}))
        full_models[col] = model
        model_view[col] = feats

    return pd.DataFrame(base_oof, index=X.index), singles, full_models, model_view


def stack_variants(specs):
    cols = [c for c, _, _ in specs]
    trees = [c for c, k, _ in specs if k in RF.TREE_KINDS and c in cols]
    out = {"stacked_all": [c for c, _, v in specs if v == "full"],
           "stacked_trees": [c for c in trees if c in
                             {s[0] for s in specs if s[2] == "full"}]}
    if any(v != "full" for _, _, v in specs):
        out["stacked_diverse"] = cols
    return out


def predict_with(strategy, col, cols, full_models, model_view, base_oof_tr, y_tr, X_te):
    """Şampiyon stratejisiyle test satırlarını skorlar (her sütun kendi görünümüyle)."""
    base_te = {c: m.predict_proba(X_te[model_view[c]])[:, 1] for c, m in full_models.items()}
    if strategy == "single":
        return base_te[col]
    if strategy == "mean":
        return np.mean([base_te[c] for c in full_models], axis=0)
    calibrators, meta, order = RF.fit_final_stack(base_oof_tr, y_tr, cols=cols)
    z = np.column_stack([calibrators[c].predict(base_te[c]) for c in order])
    return meta.predict_proba(z)[:, 1]


# ===========================================================================
# Şampiyon seçimi — AUC üzerinde EŞLEŞTİRİLMİŞ bootstrap ile One-SE
# ===========================================================================
def paired_one_se(y, probas, n_boot=PAIRED_BOOTSTRAP, seed=RF.SEED):
    """(en iyi − aday) AUC farkının eşleştirilmiş bootstrap dağılımı.

    Eski kural adayı en iyinin *marjinal* SE'siyle karşılaştırıyordu; farkın kendi
    dağılımı çok daha dar olduğundan bu kural gerçekten iyi bir adayı ayırt eder.
    Döndürür: (best_key, {aday: (fark_ortalaması, fark_SE, uygun_mu)}, {aday: auc}).
    """
    y = np.asarray(y)
    keys = list(probas)
    aucs = {k: float(roc_auc_score(y, probas[k])) for k in keys}
    best = max(aucs, key=aucs.get)

    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy = y[i]
        if yy.min() == yy.max():
            continue
        for k in keys:
            boot[k].append(roc_auc_score(yy, probas[k][i]))

    info = {}
    for k in keys:
        if k == best or not boot[k]:
            info[k] = (0.0, 0.0, True)
            continue
        d = np.asarray(boot[best]) - np.asarray(boot[k])
        mean_d, se_d = float(np.mean(d)), float(np.std(d))
        info[k] = (mean_d, se_d, mean_d <= se_d)   # fark 1 eşleştirilmiş SE içinde
    return best, info, aucs


def select_champion_plus(base_oof, singles, y, specs, policy, n_boot=PAIRED_BOOTSTRAP,
                         n_repeats=None, verbose=False):
    """Adayları kurar (tekil + ortalama + stack'ler), eşleştirilmiş One-SE ile seçer."""
    y = np.asarray(y)
    bo = base_oof.reset_index(drop=True)
    probas, strategies, cols_of = {}, {}, {}

    for c in bo.columns:
        probas[f"single_{c}"] = bo[c].to_numpy()
        strategies[f"single_{c}"] = "single"
    probas["mean"] = bo.mean(axis=1).to_numpy()
    strategies["mean"] = "mean"

    for vname, vcols in stack_variants(specs).items():
        cols = [c for c in vcols if c in bo.columns]
        if len(cols) < 2:
            continue
        m_oof, _, _ = RF.evaluate_stack(bo, y, cols=cols, n_repeats=n_repeats)
        probas[vname] = m_oof
        strategies[vname] = vname
        cols_of[vname] = cols

    best, info, aucs = paired_one_se(y, probas, n_boot=n_boot)
    eligible = [k for k in probas if info[k][2]] or [best]
    champ = min(eligible, key=lambda k: (COMPLEXITY[strategies[k]], -aucs[k]))

    if verbose:
        print(f"    {'aday':<20s} {'AUC':>7s} {'Δ(en iyi−aday)':>15s} {'SE':>7s} {'uygun':>6s}")
        for k in sorted(probas, key=lambda x: -aucs[x]):
            d, se, ok = info[k]
            mark = " ←şampiyon" if k == champ else (" (en iyi AUC)" if k == best else "")
            print(f"    {k:<20s} {aucs[k]:7.4f} {d:15.4f} {se:7.4f} "
                  f"{'evet' if ok else 'hayır':>6s}{mark}")

    proba = probas[champ]
    threshold = pick_threshold(y, proba, policy)
    return {"key": champ, "strategy": strategies[champ],
            "col": champ[len("single_"):] if strategies[champ] == "single" else None,
            "cols": cols_of.get(strategies[champ]), "proba": proba,
            "auc": aucs[champ], "best_key": best, "best_auc": aucs[best],
            "threshold": threshold}


# ===========================================================================
# Kısayol (anotasyon eksiksizliği) denetimi
# ===========================================================================
def is_missingness(feat):
    return any(feat == k or feat.startswith(k) for k in MISS_KEYS)


def shortcut_audit(panel, X_all, y, selected, cat_cols, champ_kind, params, aug):
    """Eksiklik sinyalinin tek başına gücü + eksikliksiz ablasyon."""
    y_arr = np.asarray(y)
    print("  [kısayol denetimi] tek öznitelik ROC-AUC (yalnız bu sütun kullanılsaydı):")
    solo = {}
    for feat in ("pct_missing", "n_missing_total", "n_tools_available", "ek_mean"):
        if feat not in X_all.columns:
            continue
        v = pd.to_numeric(X_all[feat], errors="coerce")
        v = v.fillna(v.median())
        try:
            auc = float(roc_auc_score(y_arr, v))
        except ValueError:
            continue
        solo[feat] = auc
        flip = " (ters yönde)" if auc < 0.5 else ""
        print(f"      {feat:<20s} AUC={auc:.3f}{flip}   → yönden bağımsız {max(auc, 1-auc):.3f}")

    miss_sel = [f for f in selected if is_missingness(f)]
    print(f"  [kısayol denetimi] seçilen {len(selected)} öznitelikten {len(miss_sel)} tanesi "
          f"eksiklik temelli: {miss_sel if miss_sel else '—'}")

    ablation = {}
    kept = [f for f in selected if not is_missingness(f)]
    if miss_sel and len(kept) >= 3:
        kind = champ_kind if champ_kind in RF.MAKERS else "xgboost"
        cv_cat = [c for c in cat_cols if c in set(kept)]
        fit_fn = RF._fit_fn(kind, cv_cat)
        p = params.get(kind) or {}
        augk = (aug[0][kept], aug[1]) if aug is not None else None
        res = RF.run_cv(lambda y_tr: RF.MAKERS[kind](p, y_tr, kept, cv_cat),
                        X_all[kept], y, cat_cols=cv_cat, fit_kwargs_fn=fit_fn, aug=augk)
        ablation = {"kind": kind, "n_features": len(kept),
                    "f1": res["f1"], "auc": res["auc"]}
        print(f"  [kısayol ablasyonu] {kind}, eksiklik öznitelikleri çıkarıldı "
              f"({len(selected)}→{len(kept)}): F1={res['f1']:.4f}  AUC={res['auc']:.4f}")
    return {"solo_auc": solo, "missingness_selected": miss_sel, "ablation": ablation}


# ===========================================================================
# Dürüst (iç içe) değerlendirme — iyileştirmelerin tamamı fold içinde tekrarlanır
# ===========================================================================
def nested_eval_plus(panel, X, y, cat_cols, params_by_kind, selection_method,
                     diverse, policy, aug_full):
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    aug_X_full, aug_y_full = aug_full if aug_full is not None else (None, None)

    if RF.NESTED_REPEATS > 1:
        splitter = RepeatedStratifiedKFold(n_splits=RF.NESTED_SPLITS,
                                           n_repeats=RF.NESTED_REPEATS,
                                           random_state=RF.SEED)
    else:
        splitter = StratifiedKFold(n_splits=RF.NESTED_SPLITS, shuffle=True,
                                   random_state=RF.SEED)

    f1s, aucs, balf1s, chosen = [], [], [], []
    n_boot = max(100, PAIRED_BOOTSTRAP // 4)          # dış fold başına daha ucuz
    for tr, te in splitter.split(X, y):
        X_tr_f, y_tr = X.iloc[tr], y.iloc[tr]
        X_te_f, y_te = X.iloc[te], y.iloc[te]

        tbl = RF.null_importance_selection(X_tr_f, y_tr, cat_cols,
                                           method=selection_method, n_null=25)
        sel = tbl.loc[tbl["keep"], "feature"].tolist()
        sc = [c for c in cat_cols if c in set(sel)]
        X_tr, X_te = X_tr_f[sel], X_te_f[sel]

        views, specs = {"full": sel}, list(STD_SPECS)
        if diverse:
            rp = relief_pruned_view(X_tr, y_tr, sc)
            if 3 <= len(rp) < len(sel):
                views["rp"] = rp
                specs += list(DIVERSE_SPECS)

        aug = (aug_X_full[sel], aug_y_full) if aug_X_full is not None else None
        bo, singles, full_models, model_view = generate_base_oof_plus(
            X_tr, y_tr, sc, params_by_kind, views, specs, aug=aug, n_repeats=1)
        ch = select_champion_plus(bo, singles, y_tr, specs, policy,
                                  n_boot=n_boot, n_repeats=1)
        proba_te = predict_with(ch["strategy"], ch["col"], ch["cols"],
                                full_models, model_view, bo, y_tr, X_te)
        pred = (proba_te >= ch["threshold"]).astype(int)
        f1s.append(scan(y_te.to_numpy(), proba_te, np.array([ch["threshold"]]))[0][0])
        aucs.append(roc_auc_score(y_te, proba_te))
        balf1s.append(RF._balanced_f1(y_te.to_numpy(), pred))
        chosen.append(ch["key"])

    return {"nested_f1": float(np.mean(f1s)), "nested_f1_std": float(np.std(f1s)),
            "nested_auc": float(np.mean(aucs)),
            "nested_balanced_f1": float(np.nanmean(balf1s)),
            "strategies": sorted(set(chosen))}


# ===========================================================================
# Panel sürücüsü
# ===========================================================================
def selection_method_for(panel):
    return os.environ.get(
        f"SELECTION_METHOD_{panel}",
        PANEL_SELECTION.get(panel, os.environ.get("SELECTION_METHOD", "null"))
    ).strip().lower()


def subsample_aug(aug_X, aug_y, frac, seed=RF.SEED):
    """Ödünç alınan GENEL satırlarının tabakalı bir alt kümesi (ağırlık düşürmenin
    plumbing gerektirmeyen muadili)."""
    if aug_X is None or frac >= 1.0:
        return aug_X, aug_y
    rng = np.random.default_rng(seed)
    ya = np.asarray(aug_y)
    keep = []
    for cls in np.unique(ya):
        idx = np.flatnonzero(ya == cls)
        k = max(1, int(round(len(idx) * frac)))
        keep.append(rng.choice(idx, size=k, replace=False))
    keep = np.sort(np.concatenate(keep))
    return (aug_X.iloc[keep].reset_index(drop=True),
            pd.Series(ya[keep]).reset_index(drop=True))


def fit_seed_ensemble(kind, params, X, y, cat_cols, n_seeds):
    """Şampiyonu birden çok tohumla yeniden eğitir (tahminler sonra ortalanır)."""
    models, orig = [], RF.SEED
    fit_fn = RF._fit_fn(kind, cat_cols)
    try:
        for s in range(orig, orig + max(1, n_seeds)):
            RF.SEED = s
            m = RF.MAKERS[kind](params, pd.Series(np.asarray(y)), list(X.columns), cat_cols)
            m.fit(X, np.asarray(y), **(fit_fn(X, np.asarray(y)) if fit_fn else {}))
            models.append(m)
    finally:
        RF.SEED = orig
    return models


def run_panel(panel):
    t0 = time.time()
    runlog.section(f"PANEL {panel}")

    df = RF.load_panel(panel)
    X, y, cat_cols = RF.build_features(df)
    y = pd.Series(np.asarray(y))
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    train_prior = float(y.mean())
    print(f"  satır={len(y)}  öznitelik={X.shape[1]}  patojenik={n_pos} benign={n_neg} "
          f"(pozitif oran %{100*train_prior:.1f})")

    # 1) Öznitelik seçimi (panel bazlı yöntem) --------------------------------
    method = selection_method_for(panel)
    tbl = RF.null_importance_selection(X, y, cat_cols, method=method)
    selected = tbl.loc[tbl["keep"], "feature"].tolist()
    Xs = X[selected]
    sel_cat = [c for c in cat_cols if c in set(selected)]
    print(f"  [seçim:{method}] {len(selected)}/{X.shape[1]} öznitelik korundu "
          f"({len(sel_cat)} kategorik)")

    # 2) GENEL artırım havuzu -------------------------------------------------
    aug_X, aug_y = RF.build_aug_pool(panel, columns=list(Xs.columns))
    aug_X, aug_y = subsample_aug(aug_X, aug_y, AUG_FRAC)
    aug = (aug_X, aug_y) if aug_X is not None else None
    if aug is not None:
        print(f"  [artırım] eğitim fold'larına +{len(aug_X)} GENEL satırı "
              f"({int((aug_y == 1).sum())} patojenik / {int((aug_y == 0).sum())} benign)"
              + (f"  [AUG_FRAC={AUG_FRAC}]" if AUG_FRAC < 1.0 else ""))

    # 3) Optuna (veya önbellekten) -------------------------------------------
    bundle = RF.OUTPUT_DIR / "models" / panel
    bundle.mkdir(parents=True, exist_ok=True)
    pfile = bundle / "tuned_params.json"
    if REUSE_PARAMS and pfile.exists():
        params_by_kind = json.loads(pfile.read_text())
        print(f"  [optuna] önbellekten okundu → {pfile}")
    else:
        params_by_kind = {}
        for kind in RF.KINDS:
            best, score = RF.tune_model(kind, Xs, y, sel_cat, RF.OPTUNA_TRIALS,
                                        RF.OPTUNA_TIMEOUT, aug=aug)
            params_by_kind[kind] = best
            print(f"    ayarlandı {kind:11s} {RF.OBJECTIVE.upper()}={score:.4f}", flush=True)
        pfile.write_text(json.dumps(params_by_kind, indent=2))

    # 4) Görünümler + temel model OOF ----------------------------------------
    views, specs = {"full": selected}, list(STD_SPECS)
    if DIVERSE_STACK:
        rp = relief_pruned_view(Xs, y, sel_cat)
        if 3 <= len(rp) < len(selected):
            views["rp"] = rp
            specs += list(DIVERSE_SPECS)
            print(f"  [çeşitli görünüm] ReliefF ile budanmış ikinci görünüm: "
                  f"{len(rp)}/{len(selected)} öznitelik → +2 temel sütun")
        else:
            print("  [çeşitli görünüm] ReliefF budaması anlamlı bir alt küme vermedi, atlandı")

    base_oof, singles, full_models, model_view = generate_base_oof_plus(
        Xs, y, sel_cat, params_by_kind, views, specs, aug=aug)

    # 5) Şampiyon — eşleştirilmiş bootstrap One-SE ----------------------------
    print("  [şampiyon seçimi] AUC üzerinde eşleştirilmiş bootstrap One-SE:")
    ch = select_champion_plus(base_oof, singles, y, specs, THRESHOLD_POLICY,
                              verbose=True)
    label = ch["col"] if ch["strategy"] == "single" else ch["strategy"]
    trivial = RF._trivial_f1(y)

    # 6) Eşik politikası karşılaştırması --------------------------------------
    proba = ch["proba"]
    print(f"\n  [eşik politikaları] şampiyon = {label}; seçilen politika = "
          f"{THRESHOLD_POLICY}")
    hdr = (f"    {'politika':<14s} {'eşik':>6s} {'F1':>7s} {'F1@1:1':>8s} {'MCC':>7s} "
           f"{'kesinlik':>9s} {'duyarlılık':>11s}")
    print(hdr)
    policy_rows = {}
    for pol in POLICIES:
        t = pick_threshold(y.to_numpy(), proba, pol)
        m = metrics_at(y.to_numpy(), proba, t, priors=(0.5,))
        policy_rows[pol] = m
        mark = "  ←gönderilen" if pol == THRESHOLD_POLICY else ""
        print(f"    {pol:<14s} {t:6.3f} {m['f1']:7.4f} {m['f1@0.5']:8.4f} "
              f"{m['mcc']:7.4f} {m['precision']:9.4f} {m['recall']:11.4f}{mark}")

    threshold = policy_rows[THRESHOLD_POLICY]["threshold"]
    chosen = policy_rows[THRESHOLD_POLICY]
    print(f"    naif taban F1={trivial:.4f}   kazanç {chosen['f1']-trivial:+.4f}   "
          f"TN/FP/FN/TP = {chosen['tn']}/{chosen['fp']}/{chosen['fn']}/{chosen['tp']}")

    # 7) Öncel kayması altında eşik tablosu -----------------------------------
    print("  [öncel düzeltmesi] test sınıf oranı farklı çıkarsa kullanılacak eşik "
          "(izotonik kalibre olasılık üzerinde):")
    iso = IsotonicRegression(out_of_bounds="clip").fit(proba, np.asarray(y))
    p_cal = iso.predict(proba)
    t_cal = pick_threshold(y.to_numpy(), p_cal, THRESHOLD_POLICY)
    shift = {}
    for tp_ in REPORT_PRIORS:
        t_shift = prior_shift_threshold(t_cal, train_prior, tp_)
        shift[f"{tp_:g}"] = t_shift
        m = metrics_at(y.to_numpy(), p_cal, t_shift, priors=(tp_,))
        print(f"      hedef öncel {tp_:.2f} → kalibre eşik {t_shift:.3f}   "
              f"(bu öncelde beklenen F1 {m[f'f1@{tp_:g}']:.4f})")

    # 8) Kısayol denetimi -----------------------------------------------------
    audit = {}
    if SHORTCUT_AUDIT:
        print()
        audit = shortcut_audit(panel, X, y, selected, cat_cols,
                               ch["col"] or "xgboost", params_by_kind,
                               (aug_X, aug_y) if aug is not None else None)

    # 9) Dürüst iç içe değerlendirme ------------------------------------------
    nested = {}
    if RF.RUN_NESTED:
        print(f"\n  [dürüst iç içe CV] {RF.NESTED_SPLITS} kat × {RF.NESTED_REPEATS} tekrar "
              f"— seçim, şampiyon ve eşik her dış katta yeniden belirleniyor...", flush=True)
        aug_nested = RF.build_aug_pool(panel, columns=list(X.columns))
        aug_nested = subsample_aug(*aug_nested, AUG_FRAC)
        nested = nested_eval_plus(panel, X, y, cat_cols, params_by_kind, method,
                                  DIVERSE_STACK, THRESHOLD_POLICY,
                                  aug_nested if aug_nested[0] is not None else None)
        print(f"  [dürüst iç içe CV] F1={nested['nested_f1']:.4f}"
              f"±{nested['nested_f1_std']:.4f}  AUC={nested['nested_auc']:.4f}  "
              f"dengeliF1={nested['nested_balanced_f1']:.4f}")
        print(f"      dış katlarda seçilen adaylar: {', '.join(nested['strategies'])}")

    # 10) Paketi kaydet -------------------------------------------------------
    for col, model in full_models.items():
        joblib.dump(model, bundle / f"base_{col}.joblib")
    for vname, vcols in stack_variants(specs).items():
        cols = [c for c in vcols if c in base_oof.columns]
        if len(cols) < 2:
            continue
        calibrators, meta, order = RF.fit_final_stack(base_oof, y, cols=cols)
        joblib.dump(calibrators, bundle / f"calibrators_{vname}.joblib")
        joblib.dump(meta, bundle / f"meta_{vname}.joblib")
        (bundle / f"{vname}_order.json").write_text(json.dumps(order))
    joblib.dump(iso, bundle / "champion_calibrator.joblib")

    seed_models = []
    if ch["strategy"] == "single" and SEED_ENSEMBLE > 1:
        kind = dict((c, k) for c, k, _ in specs)[ch["col"]]
        feats = model_view[ch["col"]]
        if aug is not None:
            X_full = pd.concat([Xs[feats], aug_X[feats]], axis=0, ignore_index=True)
            y_full = np.concatenate([np.asarray(y), np.asarray(aug_y)])
        else:
            X_full, y_full = Xs[feats], np.asarray(y)
        cvc = [c for c in sel_cat if c in set(feats)]
        seed_models = fit_seed_ensemble(kind, params_by_kind.get(kind) or {},
                                        X_full, y_full, cvc, SEED_ENSEMBLE)
        for i, m in enumerate(seed_models):
            joblib.dump(m, bundle / f"champion_seed{i}.joblib")
        print(f"  [tohum topluluğu] şampiyon {SEED_ENSEMBLE} tohumla yeniden eğitildi "
              f"→ champion_seed0..{len(seed_models)-1}.joblib")

    champion = {
        "panel": panel, "strategy": ch["strategy"], "model": ch["col"],
        "base_order": ch["cols"] if ch["strategy"].startswith("stacked")
                      else [c for c, _, _ in specs],
        "model_view": model_view, "threshold": threshold,
        "threshold_policy": THRESHOLD_POLICY,
        "threshold_by_policy": {k: v["threshold"] for k, v in policy_rows.items()},
        "calibrated_threshold": t_cal, "prior_shifted_thresholds": shift,
        "train_prior": train_prior, "augmented": aug is not None,
        "aug_frac": AUG_FRAC, "selection_method": method,
        "seed_ensemble": len(seed_models), "diverse_stack": "rp" in views,
    }
    (bundle / "champion.json").write_text(json.dumps(champion, indent=2, ensure_ascii=False))
    (bundle / "selected_features.json").write_text(json.dumps(selected))
    (bundle / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    print(f"\n  [bitti — {int(time.time()-t0)} sn]  paket → {bundle}")

    row = {"panel": panel, "n": len(y), "champion": label, "strategy": ch["strategy"],
           "selection": method, "n_features": len(selected),
           "diverse_stack": "rp" in views, "augmented": aug is not None,
           "threshold_policy": THRESHOLD_POLICY, "threshold": round(threshold, 3),
           "oof_f1": round(chosen["f1"], 4),
           "oof_f1_balanced": round(chosen["f1@0.5"], 4),
           "oof_mcc": round(chosen["mcc"], 4),
           "oof_auc": round(ch["auc"], 4),
           "oof_ap": round(float(average_precision_score(y, proba)), 4),
           "trivial_f1": round(trivial, 4),
           "f1_if_policy_f1": round(policy_rows["f1"]["f1"], 4),
           "balf1_if_policy_f1": round(policy_rows["f1"]["f1@0.5"], 4),
           "tn": chosen["tn"], "fp": chosen["fp"], "fn": chosen["fn"], "tp": chosen["tp"]}
    if nested:
        row.update({"honest_f1": round(nested["nested_f1"], 4),
                    "honest_auc": round(nested["nested_auc"], 4),
                    "honest_balanced_f1": round(nested["nested_balanced_f1"], 4),
                    "optimism_gap": round(chosen["f1"] - nested["nested_f1"], 4)})
    if audit.get("ablation"):
        row["f1_without_missingness"] = round(audit["ablation"]["f1"], 4)
        row["auc_without_missingness"] = round(audit["ablation"]["auc"], 4)
    return row


# ===========================================================================
# Çıkarım (tohum topluluğu + öncel farkındalıklı eşik)
# ===========================================================================
def predict_panel_plus(panel, df, target_prior=None):
    """Kaydedilmiş iyileştirilmiş paketle satırları etiketler.

    target_prior verilirse eşik, izotonik kalibre olasılık üzerinden o sınıf
    oranına taşınır (model yeniden eğitilmez).
    """
    bundle = RF.OUTPUT_DIR / "models" / panel
    champ = json.loads((bundle / "champion.json").read_text())
    selected = json.loads((bundle / "selected_features.json").read_text())
    cat_maps = RF.get_cat_maps(RF.load_panel(panel))
    X, _, _ = RF.build_features(df, cat_maps=cat_maps)
    X = X[selected]
    view = champ["model_view"]

    def base_raw(col):
        return joblib.load(bundle / f"base_{col}.joblib").predict_proba(X[view[col]])[:, 1]

    strategy = champ["strategy"]
    if strategy == "single":
        n_seed = champ.get("seed_ensemble", 0)
        paths = [bundle / f"champion_seed{i}.joblib" for i in range(n_seed)]
        if n_seed > 1 and all(p.exists() for p in paths):
            feats = view[champ["model"]]
            proba = np.mean([joblib.load(p).predict_proba(X[feats])[:, 1] for p in paths],
                            axis=0)
        else:
            proba = base_raw(champ["model"])
    elif strategy == "mean":
        proba = np.mean([base_raw(c) for c in champ["base_order"]], axis=0)
    else:
        calibrators = joblib.load(bundle / f"calibrators_{strategy}.joblib")
        meta = joblib.load(bundle / f"meta_{strategy}.joblib")
        order = json.loads((bundle / f"{strategy}_order.json").read_text())
        z = np.column_stack([calibrators[c].predict(base_raw(c)) for c in order])
        proba = meta.predict_proba(z)[:, 1]

    threshold = champ["threshold"]
    if target_prior is not None:
        iso = joblib.load(bundle / "champion_calibrator.joblib")
        proba = iso.predict(proba)
        threshold = prior_shift_threshold(champ["calibrated_threshold"],
                                          champ["train_prior"], float(target_prior))
    return pd.DataFrame({RF.ID_COL: df[RF.ID_COL].to_numpy(), "proba": proba,
                         "prediction": (proba >= threshold).astype(int)})


# ===========================================================================
# Ana akış
# ===========================================================================
def _print_summary(res):
    cols = [c for c in ["panel", "n", "champion", "strategy", "selection", "n_features",
                        "diverse_stack", "augmented", "threshold_policy", "threshold",
                        "oof_f1", "oof_f1_balanced", "oof_mcc", "oof_auc",
                        "honest_f1", "honest_auc", "honest_balanced_f1",
                        "optimism_gap", "trivial_f1"] if c in res.columns]
    runlog.section("PANEL BAZLI SONUÇLAR (iyileştirilmiş)")
    print(res[cols].to_string(index=False))
    if "honest_f1" in res.columns:
        print(f"\nOrtalama dürüst (iç içe) F1 : {res['honest_f1'].mean():.4f}")
        print(f"Ortalama dürüst AUC         : {res['honest_auc'].mean():.4f}")
        print(f"Ortalama dengeli F1 (1:1)   : {res['honest_balanced_f1'].mean():.4f}")
    print(f"Ortalama OOF F1             : {res['oof_f1'].mean():.4f}")

    if "f1_if_policy_f1" in res.columns:
        runlog.section("EŞİK POLİTİKASININ BEDELİ / KAZANCI")
        print("Seçilen politika ile eski (yalnız F1'i maksimize eden) politikanın "
              "karşılaştırması.\n'1:1' sütunu, test seti dengeli çıkarsa beklenen F1'dir.\n")
        t = res[["panel", "threshold_policy", "oof_f1", "oof_f1_balanced",
                 "f1_if_policy_f1", "balf1_if_policy_f1"]].copy()
        t.columns = ["panel", "politika", "F1 (yeni)", "1:1 F1 (yeni)",
                     "F1 (eski)", "1:1 F1 (eski)"]
        t["Δ F1"] = (t["F1 (yeni)"] - t["F1 (eski)"]).round(4)
        t["Δ 1:1 F1"] = (t["1:1 F1 (yeni)"] - t["1:1 F1 (eski)"]).round(4)
        print(t.to_string(index=False))

    if "f1_without_missingness" in res.columns:
        runlog.section("KISAYOL ABLASYONU — eksiklik öznitelikleri olmadan")
        t = res[["panel", "oof_f1", "f1_without_missingness",
                 "oof_auc", "auc_without_missingness"]].copy()
        t["Δ F1"] = (t["f1_without_missingness"] - t["oof_f1"]).round(4)
        t["Δ AUC"] = (t["auc_without_missingness"] - t["oof_auc"]).round(4)
        print(t.to_string(index=False))
        print("\nΔ büyükse model anotasyon eksiksizliğine dayanıyor demektir; test setinde "
              "\nanotasyon profili değişirse bu kadarlık bir düşüş riski vardır.")


def _compare_baseline(res):
    path = Path(COMPARE_TO)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"\n[karşılaştırma] referans özet bulunamadı ({path}). "
              f"Önce run_final.py'yi çalıştırın veya COMPARE_TO ile yol verin.")
        return
    base = pd.read_csv(path)
    runlog.section(f"REFERANS KARŞILAŞTIRMASI  ({path.name})")
    keys = [("oof_f1", "OOF F1"), ("oof_auc", "OOF AUC"),
            ("honest_f1", "Dürüst F1"), ("honest_auc", "Dürüst AUC"),
            ("honest_balanced_f1", "Dürüst dengeli F1")]
    rows = []
    for _, r in res.iterrows():
        b = base[base["panel"] == r["panel"]]
        if b.empty:
            continue
        b = b.iloc[0]
        row = {"panel": r["panel"]}
        for k, label in keys:
            if k in res.columns and k in base.columns:
                row[f"{label} (ref)"] = round(float(b[k]), 4)
                row[f"{label} (yeni)"] = round(float(r[k]), 4)
                row[f"Δ {label}"] = round(float(r[k]) - float(b[k]), 4)
        rows.append(row)
    if rows:
        comp = pd.DataFrame(rows)
        for _, label in keys:
            sub = ["panel"] + [c for c in (f"{label} (ref)", f"{label} (yeni)",
                                           f"Δ {label}") if c in comp.columns]
            if len(sub) > 1:
                print("\n" + comp[sub].to_string(index=False))
        deltas = {label: comp[f"Δ {label}"].mean() for _, label in keys
                  if f"Δ {label}" in comp.columns}
        print("\nOrtalama fark: " + "   ".join(f"{k} {v:+.4f}" for k, v in deltas.items()))
        print("\nKarar için 'Dürüst F1' ve 'Dürüst dengeli F1' sütunlarına bakın: birincisi "
              "\neğitim dağılımındaki, ikincisi dengeli bir test setindeki beklenen başarımdır.")


def _inference_check(panels):
    runlog.section("ÇIKARIM BAĞLANTI KONTROLÜ (kaydedilmiş paketler)")
    for p in panels:
        df = RF.load_panel(p)
        out = predict_panel_plus(p, df)
        pred = out["prediction"].to_numpy()
        y = df[RF.LABEL_COL].to_numpy()
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1v = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        out5 = predict_panel_plus(p, df, target_prior=0.5)
        print(f"  {p:7s} örneklem içi F1={f1v:.4f}  tahmini pozitif oran="
              f"{pred.mean():.3f}   |  hedef öncel 1:1 ile pozitif oran="
              f"{out5['prediction'].mean():.3f}")


def main():
    panels = [a.upper() for a in sys.argv[1:]] or RF.PANELS
    unknown = [p for p in panels if p not in RF.PANEL_FILES]
    if unknown:
        raise SystemExit(f"Bilinmeyen panel: {unknown}. Seçenekler: {RF.PANELS}")

    log_path = runlog.start(
        "run_improved",
        note="İyileştirme denemeleri — eşik politikası, eşleştirilmiş One-SE, "
             "çeşitli görünümlü stacking, kısayol denetimi")

    print(f"\nPaneller   : {panels}")
    print(f"Çıktı      : {RF.OUTPUT_DIR.resolve()}")
    print(f"Referans   : {COMPARE_TO}")
    print("\nAktif iyileştirmeler")
    print(f"  1. Eşik politikası        : {THRESHOLD_POLICY} "
          f"(karşılaştırılanlar: {', '.join(POLICIES)}; bootstrap-medyan n={THR_BOOT})")
    print(f"  2. Eşleştirilmiş One-SE   : {PAIRED_BOOTSTRAP} bootstrap")
    print(f"  3. Çeşitli görünüm stack  : {'açık' if DIVERSE_STACK else 'kapalı'}")
    print(f"  4. Kısayol denetimi       : {'açık' if SHORTCUT_AUDIT else 'kapalı'}")
    print(f"  5. Panel bazlı seçici     : " +
          ", ".join(f"{p}={selection_method_for(p)}" for p in panels))
    print(f"  6. Tohum topluluğu        : {SEED_ENSEMBLE}")
    print(f"  7. NESTED_REPEATS={RF.NESTED_REPEATS} (referansla aynı olmalı)  "
          f"AUG_FRAC={AUG_FRAC}  REUSE_PARAMS={int(REUSE_PARAMS)}")
    if DIVERSE_STACK:
        print("\n  Not: çeşitli görünüm 5 yerine 7 temel model eğitir; bu koşu "
              "run_final.py'den\n       yaklaşık %30-40 daha uzun sürer. "
              "DIVERSE_STACK=0 ile kapatabilirsiniz.")
    print(f"  Öncel raporu              : {REPORT_PRIORS}")

    t0 = time.time()
    rows, done = [], []
    out_csv = RF.OUTPUT_DIR / "results_summary_improved.csv"
    RF.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        rows.append(run_panel(panel))
        done.append(panel)
        # Her panelden sonra yaz: koşu yarıda kalırsa da elde sonuç kalır.
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    elapsed = int(time.time() - t0)
    runlog.section(f"TÜM PANELLER TAMAMLANDI — {elapsed // 60} dk {elapsed % 60} sn")

    res = pd.DataFrame(rows)
    _print_summary(res)
    _compare_baseline(res)
    _inference_check(done)

    print(f"\nÖzet CSV   : {out_csv}")
    print(f"Paketler   : {RF.OUTPUT_DIR / 'models'}/")
    print(f"Günlük     : {log_path}")


if __name__ == "__main__":
    main()
