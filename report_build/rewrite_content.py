# -*- coding: utf-8 -*-
"""psr_content.py'yi bellekteki BODY/TOC/REFERENCES verisinden yeniden yazar.

Kaynak dosyadaki örtük string birleştirmeleri metin eşlemeyi zorlaştırdığı için
kısaltmalar modül içeriği üzerinde uygulanır, dosya sonra yeniden serileştirilir.
"""
from __future__ import annotations

import sys, textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import psr_content as C

# (eski alt dize, yeni alt dize) — paragraf metinleri üzerinde uygulanır
SUBS = [
 ("Her rolün çıktısı depoda ayrı bir modül olarak izlenebilir; görevin gerektirdiği "
  "biyoinformatik yorum, istatistik/makine öğrenmesi ve yazılım geliştirme bileşenleri bu "
  "roller üzerinden karşılanır.",
  "Her rolün çıktısı depoda ayrı bir modül olarak izlenebilir; görevin gerektirdiği "
  "biyoinformatik, istatistik/ML ve yazılım bileşenleri bu roller üzerinden karşılanır."),
 ("Üçüncü referans Grimm ve ark. [11]’dir: missense araçları, aynı varyantların hem eğitimde "
  "hem testte bulunması ve bir genin varyantlarının tek sınıfta toplanması nedeniyle şişmiş "
  "skorlar üretir.",
  "Üçüncü referans Grimm ve ark. [11]’dir: missense araçları, aynı varyantların hem eğitimde "
  "hem testte bulunmasından doğan dairesellik nedeniyle şişmiş skorlar üretir."),
 ("(iii) Fold içi sızıntı: medyan doldurma, target encoding ve izotonik kalibrasyon yalnızca "
  "eğitim fold’unda öğrenilir. (iv) Öznitelik seçimi de raporlanan genelleme skorunda her dış "
  "fold içinde yeniden çalıştırılır.",
  "(iii) Fold içi sızıntı: medyan doldurma, target encoding ve izotonik kalibrasyon yalnızca "
  "eğitim fold’unda öğrenilir; (iv) öznitelik seçimi de genelleme skorunda her dış fold içinde "
  "yeniden çalıştırılır."),
 ("Boyut indirgemede PCA yerine öznitelik seçimi tercih edilmiştir; bileşen uzayında SHAP "
  "tabanlı grup açıklaması ve ACMG kanıt kodlarıyla eşleme imkânı kaybolurdu (4.4).",
  "Boyut indirgemede PCA yerine öznitelik seçimi tercih edilmiştir; bileşen uzayında SHAP "
  "tabanlı grup açıklaması imkânı kaybolurdu (4.4)."),
 ("(iii) Gri bölge: FP grubunun hem korunmuşluk hem Grantham medyanı her panelde TN ile TP "
  "arasındadır; model orta şiddetli ve orta düzeyde korunmuş ikamelerde zorlanır — klinikte "
  "VUS olarak kalan varyantların bölgesi tam olarak burasıdır.",
  "(iii) Gri bölge: FP grubunun hem korunmuşluk hem Grantham medyanı TN ile TP arasındadır — "
  "model orta şiddetli ikamelerde, yani klinikte VUS olarak kalan varyantların bölgesinde "
  "zorlanır."),
 ("(4) SMOTE ve benzeri sentetik örnekleme: %55’i eksik olan 442 boyutlu bir uzayda komşuluk "
  "temelli interpolasyon gerçekte var olmayan profiller üretir; yerine gerçek GENEL "
  "satırlarıyla artırım tercih edilmiştir.",
  "(4) SMOTE: %55’i eksik olan 442 boyutlu bir uzayda komşuluk temelli interpolasyon var "
  "olmayan profiller üretir; yerine gerçek GENEL satırlarıyla artırım tercih edilmiştir."),
 ("türetilen 91 öznitelik bu yorumlara dayandırılmış ve SHAP katkıları grup düzeyinde "
  "toplanarak ACMG kanıt kodlarıyla eşlenmiştir (Tablo 5). Bu çerçeve, anonim veride "
  "açıklanabilirliği yalnızca teknik değil klinik olarak da anlamlı kılar.",
  "türetilen 91 öznitelik bu yorumlara dayandırılmış ve SHAP katkıları grup düzeyinde "
  "toplanarak ACMG kanıt kodlarıyla eşlenmiştir (Tablo 5) — bu çerçeve anonim veride "
  "açıklanabilirliği klinik olarak anlamlı kılar."),
 ("Kısa vadede skoru bir miktar düşüren bu muhafazakârlık, bilinmeyen bir test dağılımı "
  "karşısında beklenen başarımı yükselten bilinçli bir tercihtir.",
  "Kısa vadede skoru bir miktar düşüren bu muhafazakârlık, bilinmeyen bir test dağılımı "
  "karşısında bilinçli bir tercihtir."),
 ("hiçbir panelde naif taban çizgisinin altına düşmemiştir; bu, mimarinin hem 2.931 satırlı "
  "hem 111 satırlı panelde tutarlı çalıştığını gösterir. Gürültüye dayanıklılık, Optuna "
  "denemeleri arasında kat tohumunun sarsılması ve fold-medyan eşikle sağlanır: tek bir şanslı "
  "bölünmede iyi olan bir yapılandırma seçilemez.",
  "hiçbir panelde naif tabanın altına düşmemiştir; mimari hem 2.931 satırlı hem 111 satırlı "
  "panelde tutarlı çalışır. Gürültüye dayanıklılık, kat tohumunun denemeler arası sarsılması "
  "ve fold-medyan eşikle sağlanır."),
 ("Determinizm sabit tohumla sağlanmıştır. Özetle yüksek başarım, klinik bir karar destek "
  "akışına gömülmesini engellemeyecek bir maliyetle elde edilmiştir.",
  "Determinizm sabit tohumla sağlanmıştır. Özetle eğitim tek seferlik ve birkaç saatlik, "
  "çıkarım pratik olarak anlıktır."),
 ("Bu süre, panel başına beş modelin 80 denemeye kadar aranmasını, 3 tekrarlı OOF üretimini ve "
  "10 dış katlı iç içe değerlendirmeyi kapsar. Gradyan artırma kullanıldığından epoch/batch "
  "yerine ağaç sayısı raporlanır: varsayılan 400 ağaç, CatBoost’ta 500 iterasyon.",
  "Bu süre, panel başına beş modelin 80 denemeye kadar aranmasını, 3 tekrarlı OOF üretimini ve "
  "10 dış katlı iç içe değerlendirmeyi kapsar; gradyan artırma kullanıldığından epoch/batch "
  "yerine ağaç sayısı raporlanır (varsayılan 400 ağaç, CatBoost’ta 500 iterasyon)."),
 ("Rastlantısallığa karşı dört ek önlem vardır: CV tekrarları, denemeler arası kat tohumu "
  "sarsması, 200 örneklemli bootstrap standart hatası ve fold-medyan eşik. Şeffaflık gereği "
  "belirtilmelidir ki dış katlarda hiperparametreler yeniden aranmamış, global parametreler "
  "kullanılmıştır; bu, iç içe skorda küçük bir iyimserlik payı bırakır.",
  "Rastlantısallığa karşı dört ek önlem vardır: CV tekrarları, denemeler arası kat tohumu "
  "sarsması, 200 örneklemli bootstrap standart hatası ve fold-medyan eşik. Şeffaflık gereği: "
  "dış katlarda hiperparametreler yeniden aranmamış, global parametreler kullanılmıştır; bu, "
  "iç içe skorda küçük bir iyimserlik payı bırakır."),
 ("çalışmalar hâlinde yürütülmüştür (model başına en fazla 80 deneme, 600 s bütçe sınırı); "
  "her çalışmanın ilk denemesi olarak elle ayarlanmış makul bir varsayılan kuyruğa alınır, "
  "böylece bütçe erken tükense bile makul bir çözüm garanti edilir.",
  "çalışmalar hâlinde yürütülmüştür (model başına en fazla 80 deneme, 600 s bütçe sınırı); "
  "ilk deneme olarak elle ayarlanmış bir varsayılan kuyruğa alınır, böylece bütçe erken "
  "tükense bile makul bir çözüm garanti edilir."),
]

SUBS += [
 ("Örnek: KANSER panelinden VAR_003700 (gerçek etiket patojenik, model olasılığı 1,000). En "
  "büyük SHAP katkıları AL_9 (popülasyon frekans grubu, 0,0305; +1,14), ek_std (korunmuşluk "
  "skorlarının satır içi standart sapması, 3,95; +1,10), ek_max × Grantham etkileşimi "
  "(1.167,8; +0,55), helix_prop_1 (0,57; +0,49) ve ek_max (11,93 — panelin üst sınırı; "
  "+0,44). Karar tek bir özniteliğe değil, korunmuşluğun hem seviyesine hem iç "
  "değişkenliğine ve bunun şiddetli bir biyokimyasal ikameyle (Grantham 97,9) birleşimine "
  "dayanır. AL_9 tek başına ayırıcı değildir (marjinal ROC-AUC 0,489); katkısı yalnızca "
  "korunmuşluk sinyaliyle etkileşim üzerinden ortaya çıkar — bu, tekil sıralama "
  "ölçütlerinin neden yanıltıcı olabildiğini somutlaştırır.",
  "Örnek: KANSER panelinden VAR_003700 (gerçek etiket patojenik, model olasılığı 1,000). En "
  "büyük SHAP katkıları AL_9 (popülasyon frekans grubu, 0,0305; +1,14), ek_std (korunmuşluk "
  "skorlarının satır içi standart sapması, 3,95; +1,10), ek_max × Grantham etkileşimi "
  "(1.167,8; +0,55) ve ek_max (11,93 — panelin üst sınırı; +0,44) olmuştur. Karar tek bir "
  "özniteliğe değil, korunmuşluğun hem seviyesine hem iç değişkenliğine ve bunun şiddetli bir "
  "biyokimyasal ikameyle (Grantham 97,9) birleşimine dayanır. AL_9 tek başına ayırıcı değildir "
  "(marjinal ROC-AUC 0,489); katkısı yalnızca etkileşim üzerinden ortaya çıkar — bu, tekil "
  "sıralama ölçütlerinin neden yanıltıcı olabildiğini somutlaştırır."),
 ("(4) Bilgisiz kolonlar: tamamen boş kolon PAH’ta 1, diğerlerinde 0; tek değerli kolon 57–90 "
  "— null-importance aşamasında sıfır kazanç ürettiklerinden otomatik elenirler. (5) Aralık "
  "taraması: allel frekanslarında [0, 1] ihlali yok; standart dışı amino asit 8 (GENEL) ve 2 "
  "(KANSER); eksik amino asit 350 / 22 / 9 / 0 — satır silmek yerine aa_missing, aa1_nonstd ve "
  "aa2_nonstd bayraklarıyla bildirilmiştir.",
  "(4) Bilgisiz kolonlar: tamamen boş kolon PAH’ta 1, diğerlerinde 0; tek değerli kolon 57–90 "
  "— null-importance aşamasında otomatik elenirler. (5) Aralık taraması: allel frekanslarında "
  "[0, 1] ihlali yok; standart dışı amino asit 8 (GENEL) ve 2 (KANSER); eksik amino asit "
  "350 / 22 / 9 / 0 — satır silmek yerine aa_missing, aa1_nonstd ve aa2_nonstd bayraklarıyla "
  "bildirilmiştir."),
 ("Arama uzayı: XGBoost’ta max_depth 2–8, gamma 0–5, min_child_weight 1–10; LightGBM’de "
  "num_leaves 15–255, min_child_samples 5–100, min_split_gain 0–1; her ikisinde learning_rate "
  "0,01–0,3 (log), subsample 0,5–1,0, colsample_bytree 0,3–1,0, reg_alpha ve reg_lambda "
  "10⁻⁸–10 (log), sınıf ağırlığı çarpanı 0,5–6,0 (log). CatBoost’ta depth 3–9 ve l2_leaf_reg "
  "1–10; ExtraTrees’te n_estimators 200–800, max_depth 4–24, min_samples_leaf 1–20; lojistik "
  "regresyonda C 10⁻³–20 (log). 400’den az satırlı panellerde derinlik ve yaprak sınırları "
  "daraltılır.",
  "Arama uzayı: XGBoost’ta max_depth 2–8, gamma 0–5, min_child_weight 1–10; LightGBM’de "
  "num_leaves 15–255, min_child_samples 5–100, min_split_gain 0–1; her ikisinde learning_rate "
  "0,01–0,3 (log), subsample 0,5–1,0, colsample_bytree 0,3–1,0, reg_alpha/reg_lambda 10⁻⁸–10 "
  "(log) ve sınıf ağırlığı çarpanı 0,5–6,0 (log). CatBoost’ta depth 3–9, l2_leaf_reg 1–10; "
  "ExtraTrees’te n_estimators 200–800, max_depth 4–24, min_samples_leaf 1–20; lojistik "
  "regresyonda C 10⁻³–20 (log); 400’den az satırlı panellerde sınırlar daraltılır."),
 ("Bu çeşitlilik, meta katmanın birbirini tamamlayan hatalardan yararlanabilmesi için "
  "kasıtlıdır. Dengesizlik model düzeyinde yönetilir:",
  "Bu çeşitlilik, meta katmanın tamamlayıcı hatalardan yararlanabilmesi için kasıtlıdır. "
  "Dengesizlik model düzeyinde yönetilir:"),
]

SUBS += [
 ("(3) A/B katmanı: %80/20 tabakalı held-out bölmeyle "
  "öznitelik seçim yöntemleri aynı test satırlarında karşılaştırılır. Hiperparametre seçimi "
  "yalnızca birinci katmanda yapılır.",
  "(3) A/B katmanı: %80/20 tabakalı held-out bölmeyle öznitelik seçim yöntemleri aynı test "
  "satırlarında karşılaştırılır. Hiperparametreler yalnızca birinci katmanda seçilir."),
 ("(4) Dağılım kaymasına dayanıklı karar mimarisi: sınıf oranından bağımsız ROC-AUC hedefi, "
  "AUC üzerinde One-SE ile yalınlık tercihi [22], fold-medyan eşik ve izotonik "
  "kalibratörlerin paketle saklanması birlikte çalışır. Sonuncusu, test setinin sınıf oranı "
  "eğitimden farklı çıkarsa modeli yeniden eğitmeden eşiği Bayes-optimal noktaya taşıma "
  "imkânı verir. Dört panelde de One-SE kuralı stacking yerine tekil modeli seçmiş, "
  "iyimserlik sapması +0,005…+0,034 bandında kalmıştır.",
  "(4) Dağılım kaymasına dayanıklı karar mimarisi: sınıf oranından bağımsız ROC-AUC hedefi, "
  "AUC üzerinde One-SE ile yalınlık tercihi [22], fold-medyan eşik ve izotonik kalibratörlerin "
  "paketle saklanması birlikte çalışır — sonuncusu, test setinin sınıf oranı farklı çıkarsa "
  "modeli yeniden eğitmeden eşiği Bayes-optimal noktaya taşıma imkânı verir. Dört panelde de "
  "One-SE kuralı tekil modeli seçmiş, iyimserlik sapması +0,005…+0,034 bandında kalmıştır."),
 ("Metrik seçimi 3.5’te gerekçelendirilmiştir; sonuçlar dört panel için ayrı hesaplanmıştır "
  "(Tablo 3). Eşik stratejisi: her katta 0,05–0,95 aralığı 0,005 adımla taranır, F1’i "
  "maksimize eden değer bulunur ve katların medyanı nihai eşik olur — fold-medyanı tek bir "
  "bölünmeye aşırı uyumu engeller.",
  "Metrik seçimi 3.5’te gerekçelendirilmiştir; sonuçlar dört panel için ayrı hesaplanmıştır "
  "(Tablo 3). Eşik stratejisi: her katta 0,05–0,95 aralığı 0,005 adımla taranır ve katların "
  "F1-optimal eşiklerinin medyanı alınır; bu, tek bir bölünmeye aşırı uyumu engeller."),
 ("Eksik değer yönetimi bu verinin merkezi kararıdır. Satır başına medyan eksik öznitelik "
  "202–247 (CFTR’de 37) düzeyindedir ve dağılım iki modludur: GENEL’de 575 satırda 20’den az, "
  "1.027 satırda 301’den fazla eksik değer bulunur; örüntü rastgele değildir, bilgi taşır.",
  "Eksik değer yönetimi bu verinin merkezi kararıdır. Satır başına medyan eksik öznitelik "
  "202–247 (CFTR’de 37) düzeyindedir ve dağılım iki modludur: GENEL’de 575 satırda 20’den az, "
  "1.027 satırda 301’den fazla eksik değer vardır; örüntü rastgele değildir, bilgi taşır."),
 ("Sinyali açık hâle "
  "getirmek için eksik sayıları, pct_missing, missingness_tier, is_skeleton, is_dense ve "
  "blok göstergeleri üretilmiştir. Aykırı değerler silinmemiştir: allel frekans "
  "kolonlarında aralık ihlali yoktur, çarpıklık log1p ve −log₁₀(en küçük pozitif frekans) "
  "dönüşümleriyle ele alınmıştır. Ölçekleme yalnızca L1-lojistik regresyon ve ReliefF "
  "hesabında uygulanır.",
  "Sinyali açık hâle getirmek için eksik sayıları, pct_missing, missingness_tier, "
  "is_skeleton, is_dense ve blok göstergeleri üretilmiştir. Aykırı değerler silinmemiştir: "
  "allel frekans kolonlarında aralık ihlali yoktur ve çarpıklık log1p ile −log₁₀(en küçük "
  "pozitif frekans) dönüşümleriyle ele alınmıştır. Ölçekleme yalnızca L1-lojistik regresyon ve "
  "ReliefF hesabında uygulanır."),
]

SUBS += [
 ("(1) Sızıntısız çapraz-panel veri artırımı: küçük panellerin azınlık sınıfı sorunu "
  "sentetik örnekleme yerine gerçek verilerle çözülmüştür. GENEL panelinin hedef panelde "
  "bulunmayan satırları yalnızca eğitim fold’larına eklenir, değerlendirme her zaman hedef "
  "panelin kendi satırlarında yapılır; ön koşul olan etiket tutarlılığı 578 ortak varyantta "
  "%100 ölçülmüştür. Ablasyon: CFTR iç içe ROC-AUC ≈ 0,88 → 0,948, benign eğitim yoğunluğu "
  "21 → 799; KANSER’de aynı müdahale performansı düşürdüğü için uygulanmamıştır.",
  "(1) Sızıntısız çapraz-panel veri artırımı: küçük panellerin azınlık sınıfı sorunu sentetik "
  "örnekleme yerine gerçek verilerle çözülmüştür. GENEL panelinin hedef panelde bulunmayan "
  "satırları yalnızca eğitim fold’larına eklenir, değerlendirme her zaman hedef panelin kendi "
  "satırlarında yapılır; ön koşul olan etiket tutarlılığı 578 ortak varyantta %100 "
  "ölçülmüştür. Ablasyon sonuçları 4.5’tedir (CFTR iç içe AUC ≈ 0,88 → 0,948)."),
 ("(3) ReliefF kapılı hibrit öznitelik seçici: null-importance seçimine, yerel ayrım gücünü "
  "ölçen ReliefF [21] işaretinin ikinci bir kapı olarak eklenmesi. Ablasyon (aynı %80/20 "
  "held-out bölmesi, seçim yalnızca eğitim satırlarında): KANSER test F1 null seçicide "
  "0,891 (36 öznitelik), bilgi kazancı temelli genişletmede 0,879 (39 öznitelik), ReliefF "
  "ile budanmış kümede 0,911 (22 öznitelik) — bilgi kazancına göre öznitelik eklemek zarar "
  "verir, negatif ReliefF ağırlıklıları çıkarmak yardım eder.",
  "(3) ReliefF kapılı hibrit öznitelik seçici: null-importance seçimine, yerel ayrım gücünü "
  "ölçen ReliefF [21] işaretinin ikinci bir kapı olarak eklenmesi. Ablasyon (aynı %80/20 "
  "held-out bölmesi, seçim yalnızca eğitim satırlarında): KANSER test F1 null seçicide 0,891 "
  "(36 öznitelik), bilgi kazancı temelli genişletmede 0,879 (39 öznitelik), ReliefF ile "
  "budanmış kümede 0,911 (22 öznitelik) — bilgi kazancına göre öznitelik eklemek zarar verir, "
  "çıkarmak yardım eder."),
 ("(5) Kısayol sinyalinin ölçülmesi ve sınırlanması: 4.3’te gösterildiği gibi anotasyon "
  "eksiksizliği tek başına GENEL’de 0,653, KANSER’de 0,769 ROC-AUC üretmektedir; "
  "literatürde dairesellik olarak tartışılan bu olgunun [11] bu veri kümesi için ölçülüp "
  "raporlanması çalışmanın ayırt edici katkılarındandır ve nihai modellerde bu grubun SHAP "
  "katkısı %0–6,9 ile sınırlı kalmıştır. Buna ek olarak raporlama disiplini de bilinçli bir "
  "tercihtir: her tabloda OOF ve iç içe CV değerleri yan yana verilmiş, iyimserlik sapması "
  "açıkça hesaplanmış ve naif taban çizgisi hiçbir panelde gizlenmemiştir.",
  "(5) Kısayol sinyalinin ölçülmesi ve sınırlanması: 4.3’te gösterildiği gibi anotasyon "
  "eksiksizliği tek başına GENEL’de 0,653, KANSER’de 0,769 ROC-AUC üretmektedir; literatürde "
  "dairesellik olarak tartışılan bu olgunun [11] bu veri kümesi için ölçülüp raporlanması "
  "çalışmanın ayırt edici katkılarındandır ve nihai modellerde bu grubun SHAP katkısı %0–6,9 "
  "ile sınırlı kalmıştır. Raporlama disiplini de bilinçli bir tercihtir: her tabloda OOF ve iç "
  "içe CV değerleri yan yana verilmiş ve naif taban hiçbir panelde gizlenmemiştir."),
 ("Bu asimetri nedeniyle eşik duyarlılık lehine seçilmiş, dört panelde de duyarlılık "
  "0,955–0,994 aralığında tutulmuştur. Tercih körü körüne uygulanmamıştır: eşiklerin dengeli "
  "F1 üzerindeki bedeli ölçülmüş (Tablo 3) ve izotonik kalibratörler paketle saklanmıştır; "
  "böylece test dağılımı farklı çıkarsa model yeniden eğitilmeden eşik Bayes-optimal noktaya "
  "taşınabilir.",
  "Bu asimetri nedeniyle eşik duyarlılık lehine seçilmiş, dört panelde de duyarlılık "
  "0,955–0,994 aralığında tutulmuştur; bedeli dengeli F1 üzerinden ölçülmüştür (Tablo 3). "
  "İzotonik kalibratörler paketle saklandığından, test dağılımı farklı çıkarsa model yeniden "
  "eğitilmeden eşik Bayes-optimal noktaya taşınabilir."),
]


def apply_subs():
    n = 0
    for i, (kind, payload) in enumerate(C.BODY):
        if kind != "p":
            continue
        new = payload
        for old, rep in SUBS:
            if old in new:
                new = new.replace(old, rep)
                n += 1
        C.BODY[i] = (kind, new)
    return n


def fmt_str(value, indent):
    """Uzun metni satırlara bölerek Python literali üretir."""
    if not isinstance(value, str) or len(value) <= 88:
        return repr(value).replace("'", '"', 1)[0:0] or _q(value)
    parts = textwrap.wrap(value, width=88 - indent, break_long_words=False,
                          break_on_hyphens=False)
    pad = " " * indent
    out = []
    for j, part in enumerate(parts):
        suffix = " " if j < len(parts) - 1 else ""
        out.append(f'{pad}{_q(part + suffix)}')
    return "\n".join(out).lstrip()


def _q(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def serialize():
    L = ['# -*- coding: utf-8 -*-',
         '"""PSR (Proje Sunuş Raporu) içeriği — TEKNOFEST 2026, PatHUgen.',
         '',
         'Sayısal değerler nihai tam bütçeli koşudan gelir (outputs_colab/results_summary.csv,',
         'report_figs/repro_metrics.json, outputs/holdout_compare/, kalite-kontrol ve SHAP',
         'betikleri). Şartname sınırı: kapak ve içindekiler hariç en fazla 10 sayfa.',
         '',
         'Bu dosya report_build/rewrite_content.py tarafından yeniden üretilebilir.',
         '"""',
         'from __future__ import annotations',
         '']
    L.append("COVER = {")
    for k, v in C.COVER.items():
        L.append(f"    {_q(k)}: {_q(v)},")
    L.append("}")
    L.append("")
    L.append("TOC = [")
    for t, pg in C.TOC:
        L.append(f"    ({_q(t)}, {pg}),")
    L.append("]")
    L.append("")
    L.append("BODY = [")
    for kind, payload in C.BODY:
        if kind == "table":
            L.append('    ("table", {')
            L.append(f'        "caption": {fmt_str(payload["caption"], 12)},')
            L.append(f'        "widths": {payload["widths"]},')
            if "fontsize" in payload:
                L.append(f'        "fontsize": {payload["fontsize"]},')
            L.append('        "rows": [')
            for row in payload["rows"]:
                cells = ", ".join(_q(str(c)) for c in row)
                L.append(f"            [{cells}],")
            L.append("        ]}),")
        else:
            L.append(f'    ({_q(kind)}, {fmt_str(payload, 8)}),')
    L.append("]")
    L.append("")
    L.append("REFERENCES = [")
    for ref in C.REFERENCES:
        L.append(f"    {fmt_str(ref, 4)},")
    L.append("]")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print("uygulanan kısaltma:", apply_subs())
    (HERE / "psr_content.py").write_text(serialize(), encoding="utf-8")
    print("psr_content.py yeniden yazıldı")
