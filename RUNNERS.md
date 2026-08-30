# Üç koşucu: `run_final.py`, `run_improved.py`, `run_best_search.py`

Üçü de proje kökünden, sanal ortamla çalıştırılır ve **her koşu için ayrı bir
`.txt` günlük dosyası** üretir.

---

## 1. `run_final.py` — referans (baseline)

PDR'de raporlanan koşunun birebir aynısı. `TEKNOFEST_pipeline_colab.ipynb`
kod hücrelerinden otomatik üretilmiştir (`scripts/gen_runner.py`); pipeline'a
hiçbir şey eklenmemiş, yalnızca Colab/Drive bağlantısı bir CLI ile değiştirilmiştir.

```bash
cd ~/Desktop/TEKNOFEST && \
QUICK=0 USE_GPU=0 RUN_NESTED=1 AUGMENT_PANELS=PAH,CFTR SELECTION_METHOD=null \
OPTUNA_TRIALS=80 OPTUNA_TIMEOUT=600 CV_REPEATS=3 NESTED_REPEATS=2 \
TEKNOFEST_OUT_DIR=outputs_final \
nohup .venv/bin/python run_final.py > /dev/null 2>&1 &
```

Çıktılar: `outputs_final/results_summary.csv` + `outputs_final/models/<PANEL>/`
Günlük:   `logs/run_final_<zaman>.txt`  (kısayol: `logs/run_final_latest.txt`)

---

## 2. `run_improved.py` — öneriler eklenmiş sürüm

`run_final.py`'yi **değiştirmez**, kütüphane olarak içe aktarır ve üzerine ekler.

```bash
cd ~/Desktop/TEKNOFEST && \
nohup .venv/bin/python run_improved.py > /dev/null 2>&1 &
```

Çıktılar: `outputs_improved/results_summary_improved.csv` + `outputs_improved/models/<PANEL>/`
Günlük:   `logs/run_improved_<zaman>.txt`  (kısayol: `logs/run_improved_latest.txt`)

### Eklenen iyileştirmeler

| # | İyileştirme | Knob (varsayılan) | Ne yapar |
|---|---|---|---|
| 1 | Öncele duyarsız karar eşiği | `THRESHOLD_POLICY=minimax` | `f1` / `balanced_f1` / `mcc` / `minimax` politikalarını birlikte hesaplar, tabloyu günlüğe basar, seçileni gönderir. Eşik bootstrap-medyanıyla sabitlenir. |
| 2 | Öncel kaymasına hazır eşik | `REPORT_PRIORS=0.5,0.6,0.74` | İzotonik kalibratör + eğitim önceli paketle saklanır; test oranı farklı çıkarsa eşik yeniden eğitim olmadan taşınır. |
| 3 | Eşleştirilmiş bootstrap One-SE | `PAIRED_BOOTSTRAP=400` | (en iyi − aday) AUC **farkının** dağılımını kullanır. Eski marjinal-SE bandı n=111'de o kadar genişti ki daima en yalın aday kazanıyordu. |
| 4 | Çeşitli görünümlü stacking | `DIVERSE_STACK=1` | XGBoost + LightGBM için ReliefF ile budanmış ikinci öznitelik görünümü → meta katman 7 sütun görür (yeni Optuna yok). |
| 5 | Kısayol denetimi + ablasyon | `SHORTCUT_AUDIT=1` | Eksiklik sinyalinin tek başına AUC'si; şampiyonu eksiklik öznitelikleri olmadan yeniden değerlendirir. |
| 6 | Panel bazlı öznitelik seçici | `SELECTION_METHOD_<PANEL>` | KANSER varsayılan `hybrid`, diğerleri `null` (held-out A/B'de yalnız KANSER gürültüyü aşmıştı). |
| 7 | Tohum topluluğu | `SEED_ENSEMBLE=3` | Nihai şampiyon birden çok tohumla yeniden eğitilir, tahminler ortalanır. |
| 8 | Ödünç satır oranı | `AUG_FRAC=1.0` | Ağırlık düşürmenin plumbing gerektirmeyen muadili: artırım havuzunun tabakalı alt kümesi. `NESTED_REPEATS` bilinçli olarak referansla **aynı** (2) tutulur ki fark tekrar sayısından değil iyileştirmelerden gelsin; daha kararlı tahmin için 3/5 verin. |

Tüm iyileştirmeler iç içe (nested) değerlendirmenin **içinde de** tekrarlanır —
yani raporlanan dürüst skor gerçekten bu pipeline'ın skorudur.

### Faydalı knob'lar

```bash
REUSE_PARAMS=1  .venv/bin/python run_improved.py   # Optuna'yı atla (önbellekten oku)
THRESHOLD_POLICY=balanced_f1 .venv/bin/python run_improved.py
DIVERSE_STACK=0 SHORTCUT_AUDIT=0 .venv/bin/python run_improved.py   # sadece eşik denemesi
.venv/bin/python run_improved.py CFTR KANSER       # tek/çift panel
```

---

## 3. `run_best_search.py` — TÜM seçenekleri süpürüp en iyisini seçer

`run_final.py` ile `run_improved.py`'nin melezi: ikisinin sunduğu seçenekleri tek tek
denemek yerine **çapraz kombinasyonlarını** süpürür, her yapılandırmayı sızıntısız
iç içe CV ile ölçer, **aşırı uyum denetiminden geçenler** arasından en iyisini seçer.

```bash
cd ~/Desktop/TEKNOFEST && \
nohup .venv/bin/python run_best_search.py > /dev/null 2>&1 &
tail -f logs/run_best_search_latest.txt
```

### Süpürülen eksenler

**Pahalı** (yeniden eğitim gerektirir): öznitelik seçimi `null|hybrid` ×
artırım oranı `0.0|0.5|1.0` (GENEL hariç; **KANSER dâhil** — referansta tam artırım
KANSER'e zarar vermişti, kısmi artırım hiç denenmemişti).

**Ucuz** (aynı temel modellerden, bedelsiz): şampiyon kuralı `marginal|paired` ×
aday havuzu `all|singles` × eşik politikası `f1|balanced_f1|mcc|minimax`
= yapılandırma başına 16 kombinasyon.

Ucuz eksenler iç içe CV'nin **her dış katında** yeniden değerlendirilir; raporlanan
dürüst skor gerçekten o yapılandırmanın skorudur.

### Aşırı uyum denetimi (geçemeyen ELENİR)

| # | Test | Varsayılan sınır | Neyi yakalar |
|---|---|---|---|
| 1 | İyimserlik sapması (OOF F1 − dürüst F1) | ≤ 0.040 | Klasik aşırı uyum |
| 2 | Şampiyon kararlılığı | ≥ %50 dış katta aynı aday | Model seçiminin gürültüye uyması — PAH'ta 10 katta 4 farklı kazanan çıkması tam bu testin yakaladığı şey |
| 3 | Naif tabanı geçme | dürüst F1 > taban | Sahte "yüksek skor" |

### Nihai seçim ölçütü

`SELECT_OBJECTIVE` (varsayılan `robust`):

* `robust` = min(dürüst F1, dürüst dengeli F1) — test sınıf oranı bilinmediği için hedge
* `honest_f1` — test dağılımı eğitimle aynı varsayımı
* `honest_balanced_f1` — test 1:1 varsayımı (şartnamenin öngördüğü)

Üçünün de kazananı raporda yan yana basılır; farklıysa hemen görürsünüz.

### Çıktılar

```
outputs_best/BEST_CONFIG.md            ← ÖNCE BUNU OKUYUN (insan okunur nihai karar)
outputs_best/best_config.json          ← makine okunur
outputs_best/sweep_all_results.csv     ← süpürülen HER adayın skoru
outputs_best/results_summary_best.csv  ← panel başına kazanan
outputs_best/models/<PANEL>/           ← kazananın model paketi
logs/run_best_search_<zaman>.txt
```

### Faydalı knob'lar

```bash
SWEEP_PROFILE=fast      # artırım {0,1} — yaklaşık yarı süre
SWEEP_PROFILE=wide      # artırım {0,.25,.5,.75,1} — daha uzun
SWEEP_PROFILE=decision  # yeniden eğitim yok, yalnız ucuz eksenler (dakikalar)
SELECT_OBJECTIVE=honest_balanced_f1
GAP_LIMIT=0.03 STABILITY_MIN=0.6      # daha sıkı aşırı uyum filtresi
REUSE_PARAMS=0                        # Optuna'yı sıfırdan çalıştır (varsayılan 1)
```

Optuna sonuçları `outputs_improved/` ve `outputs_best/` içinden **devralınır**, bu
yüzden ilk koşu beklenenden hızlı başlar.

**UYARI:** üç betikten ikisini aynı anda çalıştırmayın — her biri 16 iş parçacığını
doldurur.

---

## Günlük dosyaları

Her iki betik de `logs/` altına yazar. Günlük şunları içerir:

* **Başlık** — tarih, komut, Python/kütüphane sürümleri, CPU/RAM/GPU, git commit
  ve o koşuda etkin olan tüm ortam değişkenleri.
* **Gövde** — ekrana basılan her şey (panel ilerlemesi, Optuna skorları,
  aday tabloları, eşik politikası karşılaştırması, hata analizi).
* **Altbilgi** — bitiş saati, toplam süre ve `BAŞARILI` / `HATA — <istisna>`.
  Betik çökerse traceback de günlüğe yazılır.

```bash
tail -f logs/run_improved_latest.txt      # koşuyu canlı izle
grep "Durum" logs/*.txt                   # hangi koşular başarılı bitti
```

Günlük klasörü `TEKNOFEST_LOG_DIR` ile değiştirilebilir.

---

## Karşılaştırma

`run_improved.py`, bitiminde `outputs_final/results_summary.csv` varsa referansla
otomatik yan yana bir fark tablosu basar (`COMPARE_TO` ile yol değiştirilebilir).
Kararı verirken bakılacak iki satır:

* **Dürüst F1** — eğitim dağılımındaki beklenen başarım,
* **Dürüst dengeli F1** — test seti 1:1 çıkarsa beklenen başarım.

İkincisi belirgin biçimde yükselirken birincisi çok az düşüyorsa, iyileştirmeler
işe yaramış demektir: bilinmeyen test dağılımına karşı dayanıklılık satın alınmıştır.
