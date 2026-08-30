# -*- coding: utf-8 -*-
"""PSR (Proje Sunuş Raporu) içeriği — TEKNOFEST 2026, PatHUgen.

Sayısal değerler nihai tam bütçeli koşudan gelir (outputs_colab/results_summary.csv,
report_figs/repro_metrics.json, outputs/holdout_compare/, kalite-kontrol ve SHAP
betikleri). Şartname sınırı: kapak ve içindekiler hariç en fazla 10 sayfa.

Bu dosya report_build/rewrite_content.py tarafından yeniden üretilebilir.
"""
from __future__ import annotations

COVER = {
    "Proje Adı:": "PatHUgen",
    "Takım Adı:": " PatHUgen",
    "Takım ID:": " #761197",
    "Başvuru ID:": " #4882800",
}

TOC = [
    ("1. TAKIM ŞEMASI", 2),
    ("2. PROBLEME EN YAKIN ÇÖZÜM SUNAN ULUSLARARASI MAKALELERİN ÖZETİ", 2),
    ("3. VERİ VE YÖNTEM", 3),
    ("3.1. Kullanılan Veri Seti ve Etiketler", 3),
    ("3.2. Veri Kısıtları ve Etikete Doğrudan Erişimi Engelleme", 3),
    ("3.3. Veri Ön İşleme ve Temsilleme Stratejisi", 4),
    ("3.4. Etiket Güvenilirliği ve Veri Kalitesi Kontrolü", 4),
    ("3.5. Sınıf Dengesi ve Risk Perspektifi", 5),
    ("3.6. Seçilen Algoritmalar ve Gerekçe", 5),
    ("4. DENEY TASARIMI, SONUÇLAR VE İNCELEME", 5),
    ("4.1. Deney Protokolü ve Veri Bölme", 5),
    ("4.2. Performans Metrikleri ve Panel Bazlı Raporlama", 6),
    ("4.3. Hata Analizi ve Model Davranışı", 6),
    ("4.4. Açıklanabilirlik: Model Neden Böyle Karar Verdi?", 7),
    ("4.5. Öğrenme Süreci ve Teknik Evrim", 8),
    ("5. YAKLAŞIMIN GEREKÇESİ, KAYNAK KULLANIMI VE ÖZGÜNLÜK", 9),
    ("5.1. Neden Bu Algoritma / Mimari?", 9),
    ("5.2. Alternatifler Neden Elendi?", 9),
    ("5.3. Parametre Seçimi ve Model Ayarları", 9),
    ("5.4. Hesaplama Kaynakları ve Çalıştırılabilirlik", 9),
    ("5.5. Özgünlük", 10),
    ("6. REFERANSLAR", 11),
]

BODY = [
    ("h1", "1. TAKIM ŞEMASI"),
    ("p", "Takım, tek bir sürüm kontrollü depo (Git) üzerinde çalışır ve kişilere değil "
        "teknik sorumluluk alanlarına göre tanımlanmış beş rol etrafında örgütlenir: (1) "
        "veri analizi ve EDA — panel bazlı dağılım/eksiklik profilleri, t-SNE ayrışma "
        "analizi, Orange [18] Rank sıralamaları (Info Gain / ANOVA / ReliefF) ve veri "
        "kalitesi kontrolleri; (2) öznitelik mühendisliği ve seçim — 351 ham kolondan 91 "
        "türetilmiş öznitelik, null-importance seçici ve ReliefF kapılı hibrit varyantı; "
        "(3) modelleme ve optimizasyon — beş temel model, panel × model bazında Optuna "
        "çalışmaları, izotonik kalibreli stacking, One-SE şampiyon kuralı; (4) deney "
        "tasarımı ve doğrulama — iç içe çapraz doğrulama, held-out A/B karşılaştırmaları, "
        "çok-seed artırım denetimi; (5) yazılım/MLOps ve raporlama — tek dosyada çalışan "
        "tekrarlanabilir pipeline, model paketlerinin sürümlenmesi, çıkarım arayüzü. Her "
        "rolün çıktısı depoda ayrı bir modül olarak izlenebilir; görevin gerektirdiği "
        "biyoinformatik, istatistik/ML ve yazılım bileşenleri bu roller üzerinden "
        "karşılanır."),
    ("p", "Karar alma ve kalite kontrol üç kurala dayanır. (i) Kod inceleme: her değişiklik "
        "ayrı bir dalda geliştirilip ana dala birleştirilmeden önce gözden geçirilir. "
        "(ii) Deney kaydı: her koşu sabit tohum (42), ortam değişkenleriyle kaydedilen "
        "bütçe parametreleri ve CSV özetleriyle kayıt altına alınır. (iii) Kabul kuralı: "
        "bir yöntem ancak bağımsız bir held-out testinde veya çok-seed denemesinde "
        "iyileşme gösterirse ana hatta alınır — bu kural sayesinde çapraz-panel artırım "
        "PAH ve CFTR’de kabul, KANSER’de reddedilmiştir."),
    ("h1", "2. PROBLEME EN YAKIN ÇÖZÜM SUNAN ULUSLARARASI MAKALELERİN ÖZETİ"),
    ("p", "Seçim ölçütümüz: çalışma (a) missense varyant patojenite sınıflandırmasını "
        "hedeflemeli, (b) tablo biçimindeki anotasyon profillerine karşılık gelmeli ya da "
        "farklı bir veri modalitesine dayanarak çözüm uzayımızın sınırını göstermeli, (c) "
        "ölçülebilir başarım raporlamalıdır."),
    ("table", {
        "caption": "Tablo 1. Seçilen yedi çalışma. Tümünde problem tanımı missense varyant "
            "patojenitesi, veri türü ise varyant anotasyon profilleridir.",
        "widths": [2.3, 3.4, 2.2, 2.1, 5.8],
        "fontsize": 8,
        "rows": [
            ["Çalışma", "Yaklaşım", "Veri kaynağı", "Metrik", "Sınırlılık"],
            ["Jain ve ark. [4]", "Yedi klasik ML algoritması; en iyi XGBoost", "ClinVar [2] türevli tablolar", "F1 = 0,88", "Tek birleşik model; panel bazlı genelleme ve kalibrasyon ele alınmamış"],
            ["Ioannidis ve ark., REVEL [5]", "13 in-silico skoru birleştiren rastgele orman meta-tahmincisi", "ClinVar, ESP, ARIC", "AUC 0,91–0,96", "Girdi skorlarının eğitim setleriyle örtüşmesinden doğan dairesellik; eşik tasarımı yok"],
            ["Quang ve ark., DANN [6]", "CADD öznitelikleri üzerinde derin sinir ağı", "ClinVar, 1000 Genomes", "AUC’de %14 iyileşme", "Genom ölçekli, missense’e özgü değil; küçük panel rejiminde denenmemiş"],
            ["Lai ve ark., LEAP [7]", "Klinik laboratuvar akışına gömülü gradyan artırma", "Kurumsal klinik veritabanı", "AUROC ≈ 0,98", "Kuruma özgü, halka açık olmayan öznitelikler; dış geçerlilik sınırlı"],
            ["Cheng ve ark., AlphaMissense [8]", "Yapı farkındalıklı protein dil modeli", "UniProt, ClinVar", "auROC = 0,94", "Protein dizisi ve rezidü pozisyonu zorunlu — verimizde gizlenmiştir"],
            ["Brandes ve ark., ESM1b [9]", "650M parametreli protein dil modeli", "UniRef, ClinVar", "AUC 0,90–0,93", "Dizi bağımlı; yüksek hesaplama maliyeti, düşük yorumlanabilirlik"],
            ["Ruzicka ve ark., DiagAI [10]", "Klinik yorumlamada XGBoost tabanlı karar desteği", "Klinik genetik test kayıtları", "Duyarlılık %94,9", "Duyarlılık öncelikli; özgüllük ve yanlış pozitif yükü ikincil planda"],
        ]}),
    ("p", "Konumlanma: ilk grup ([4], [5], [7], [10]) tablo biçimindeki anotasyon "
        "profillerinden tahmin üretir ve istikrarlı biçimde gradyan artırmalı karar "
        "ağaçlarında (GBDT) birleşir — model ailesi seçimimizin doğrudan gerekçesi budur "
        "(3.6). İkinci grup ([8], [9]) en güçlü sonuçları protein dizisinden üretir; "
        "ancak bu görevde kromozom/pozisyon ve kolon adları gizlendiğinden dizi tabanlı "
        "hiçbir yöntem uygulanabilir değildir. İkisi birlikte sınırımızı çizer: rekabetçi "
        "başarım, dış bilgi eklemekle değil, mevcut profilden daha iyi öznitelik çıkarmak "
        "ve daha disiplinli bir karar katmanı kurmakla elde edilebilir. Üçüncü referans "
        "Grimm ve ark. [11]’dir: missense araçları, aynı varyantların hem eğitimde hem "
        "testte bulunmasından doğan dairesellik nedeniyle şişmiş skorlar üretir. Bu "
        "olgunun veri kümemizde ölçülebilir bir karşılığı vardır (4.3) ve katkımızın "
        "merkezinde onu ölçmek ve sınırlamak yer alır."),
    ("h1", "3. VERİ VE YÖNTEM"),
    ("h2", "3.1. Kullanılan Veri Seti ve Etiketler"),
    ("p", "Eğitim verisi yarışmanın sağladığı dört CSV dosyasından oluşur: genel çok-genli "
        "set (GENEL/MASTER) ve üç gen paneli — herediter kanser (KANSER), fenilketonüri "
        "(PAH), kistik fibrozis (CFTR). Toplam 3.802 varyant ve dosya başına 353 kolon "
        "vardır: Variant_ID, Label ve 351 öznitelik (334 AL_, 9 EK_, 6 CAT_, 2 AA_). "
        "Harici hiçbir kaynak kullanılmamıştır. Etiketleme ACMG/AMP kılavuzuna [1] "
        "dayanır ve ground truth kabul edilmiştir: “Pathogenic” + “Likely Pathogenic” → "
        "Patojenik (1); “Benign” + “Likely Benign” → Benign (0); VUS kaydı yoktur. "
        "Birleştirmenin sınırı açıktır — “Likely” önekli kayıtlar kanıt gücü bakımından "
        "daha zayıftır, yani model iki güven seviyesini tek sınıf olarak öğrenir — ancak "
        "azınlık benign sınıfının örnek sayısını korumak için gereklidir. Yarışma "
        "yalnızca eğitim setlerini paylaşmış, test seti hakemde tutulmuştur; tüm başarım "
        "tahminleri kendi iç doğrulama düzenimizden gelir (4.1). Dört panel her aşamada "
        "bağımsız modellenmiştir."),
    ("table", {
        "caption": "Tablo 2. Panel bazlı kapsam, sınıf dağılımı ve eksik veri yoğunluğu.",
        "widths": [3.2, 2.3, 2.4, 2.0, 2.7, 3.2],
        "fontsize": 9,
        "rows": [
            ["Panel", "Varyant", "Patojenik", "Benign", "Patojenik oranı", "Medyan eksik / 351"],
            ["GENEL (MASTER)", "2.931", "2.149", "782", "%73,3", "212"],
            ["KANSER", "388", "268", "120", "%69,1", "247"],
            ["PAH", "372", "310", "62", "%83,3", "202"],
            ["CFTR", "111", "90", "21", "%81,1", "37"],
            ["Toplam", "3.802", "2.817", "985", "%74,1", "—"],
        ]}),
    ("h2", "3.2. Veri Kısıtları ve Etikete Doğrudan Erişimi Engelleme"),
    ("p", "Yarışma kısıtına tam uyum sağlanmıştır: hiçbir dış veri tabanı (ClinVar [2], "
        "gnomAD [3] vb.) sorgulanmamış, hiçbir dış öznitelik veya önceden eğitilmiş model "
        "çıktısı eklenmemiştir; 442 özniteliğin tamamı sağlanan 351 anonim kolondan "
        "türetilmiştir. Genomik adres veride bulunmadığından harici etiket eşleştirmesi "
        "teknik olarak da mümkün değildir. Dolaylı sızıntı beş başlıkta denetlenmiştir. "
        "(i) Panel kimliği hiçbir aşamada öznitelik olarak kullanılmamış, paneller tek "
        "modelde birleştirilmemiştir. (ii) Aynı varyantın farklı bölümlere düşmesi: "
        "paneller arası Variant_ID örtüşmesi ölçülmüştür (GENEL∩KANSER 246, GENEL∩PAH "
        "255, GENEL∩CFTR 77) ve çapraz-panel artırımda yalnızca hedef panelde bulunmayan "
        "GENEL satırları kullanılmıştır. (iii) Fold içi sızıntı: medyan doldurma, target "
        "encoding ve izotonik kalibrasyon yalnızca eğitim fold’unda öğrenilir; (iv) "
        "öznitelik seçimi de genelleme skorunda her dış fold içinde yeniden çalıştırılır. "
        "(v) Kategorik proxy taraması: en az 20 satırlık her CAT kategorisinde etiket "
        "saflığı ölçülmüş, %95 üzerindekiler işaretlenmiştir (GENEL’de CAT_1’de bir "
        "kategori / 51 satır; CFTR’de CAT_3–CAT_5’te birer kategori / 35 satır); tek "
        "başına etiketi belirleyecek büyüklükte olmadıklarından çıkarılmamış, "
        "izlenmiştir."),
    ("h2", "3.3. Veri Ön İşleme ve Temsilleme Stratejisi"),
    ("p", "Kolon adları gizli olduğundan gruplar istatistiksel imzalarından "
        "anlamlandırılmıştır. AL_1–AL_26 kolonları [0, 1] aralığında ve şiddetle sağa "
        "çarpıktır (maksimum 0,929; medyanlar 10⁻⁴ mertebesinde) → popülasyon allel "
        "frekansı. Kalan AL_ kolonlarının bir bölümü yalnızca {0, 1} alır → ikili "
        "“zararlı” bayrağı; geri kalanı sürekli in-silico risk skorudur. EK_ kolonları "
        "[−20, +11,93] aralığında süreklidir → korunmuşluk skoru. CAT_ düşük "
        "kardinaliteli kategorik, AA_1/AA_2 amino asit harfleridir."),
    ("p", "Eksik değer yönetimi bu verinin merkezi kararıdır. Satır başına medyan eksik "
        "öznitelik 202–247 (CFTR’de 37) düzeyindedir ve dağılım iki modludur: GENEL’de "
        "575 satırda 20’den az, 1.027 satırda 301’den fazla eksik değer vardır; örüntü "
        "rastgele değildir, bilgi taşır. Ağaç modelleri eksik değerin gideceği dal yönünü "
        "öğrenebildiğinden NaN korunmuş; yalnızca ExtraTrees ve L1-lojistik regresyon "
        "için sayısal kolonlar eğitim fold’u medyanıyla, kategorik kolonlar target "
        "encoding ile doldurulmuştur. Sinyali açık hâle getirmek için eksik sayıları, "
        "pct_missing, missingness_tier, is_skeleton, is_dense ve blok göstergeleri "
        "üretilmiştir. Aykırı değerler silinmemiştir: allel frekans kolonlarında aralık "
        "ihlali yoktur ve çarpıklık log1p ile −log₁₀(en küçük pozitif frekans) "
        "dönüşümleriyle ele alınmıştır. Ölçekleme yalnızca L1-lojistik regresyon ve "
        "ReliefF hesabında uygulanır. Boyut indirgemede PCA yerine öznitelik seçimi "
        "tercih edilmiştir; bileşen uzayında SHAP tabanlı grup açıklaması imkânı "
        "kaybolurdu (4.4). Türetilen 91 öznitelikle sayı 442’ye çıkar: eksiklik "
        "meta-öznitelikleri; amino asit biyokimyası (BLOSUM45/62/80 [12], Grantham "
        "uzaklığı [13] ve sekiz fizikokimyasal skala için referans, alternatif, fark ve "
        "mutlak fark); in-silico oy istatistikleri; AL/EK agregatları; altı etkileşim "
        "özniteliği. Tüm dönüşümler satır bazlı ve etiketten bağımsızdır."),
    ("h2", "3.4. Etiket Güvenilirliği ve Veri Kalitesi Kontrolü"),
    ("p", "Amaç etiketleri yeniden yargılamak değil, gürültünün etkisini sınırlayacak "
        "sistematik bir denetim göstermektir. (1) Yinelenen kayıt: dört panelde de 0 "
        "yinelenen Variant_ID. (2) Yinelenen tam öznitelik profili: GENEL 116, KANSER 3, "
        "PAH 5, CFTR 0 satır; içlerinde çelişkili etiketli grup yalnızca 2’dir. Bu "
        "satırlar silinmemiş, işaretlenmiştir; oran %0,1’in altındadır ve kaynağı 300’den "
        "fazla eksik değere sahip “iskelet” varyantlardır — etiket hatası değil temsil "
        "yetersizliği — ve is_skeleton bayrağıyla bildirilmiştir. (3) Paneller arası "
        "etiket tutarlılığı: 578 ortak varyantta uyum %100; bu doğrulama olmadan "
        "çapraz-panel artırım savunulabilir olmazdı. (4) Bilgisiz kolonlar: tamamen boş "
        "kolon PAH’ta 1, diğerlerinde 0; tek değerli kolon 57–90 — null-importance "
        "aşamasında otomatik elenirler. (5) Aralık taraması: allel frekanslarında [0, 1] "
        "ihlali yok; standart dışı amino asit 8 (GENEL) ve 2 (KANSER); eksik amino asit "
        "350 / 22 / 9 / 0 — satır silmek yerine aa_missing, aa1_nonstd ve aa2_nonstd "
        "bayraklarıyla bildirilmiştir. (6) Anormal örnekler: yüksek güvenle yanlış "
        "sınıflanan örnekler ayrıştırılıp hata analizinde incelenmiştir (4.3)."),
    ("h2", "3.5. Sınıf Dengesi ve Risk Perspektifi"),
    ("p", "Patojenik sınıf baskındır (%74,1; PAH %83,3, CFTR %81,1) ve bu iki risk üretir. "
        "Ölçüm riski: tüm örnekleri patojenik sayan naif bir sınıflandırıcı bile GENEL’de "
        "0,846, PAH’ta 0,909 F1 alır; bu nedenle her panelde naif taban çizgisi birlikte "
        "raporlanmıştır. Genelleme riski: test setinin sınıf oranı bilinmemekte ve "
        "dengeli (1:1) olması muhtemeldir; eğitim önceliğine aşırı uyan bir model bu "
        "durumda çöker. Bu yüzden arama ve model seçimi, eşikten ve sınıf oranından "
        "bağımsız olan ROC-AUC üzerinden yürütülmüş; F1 (yarışma ölçütü), AUPRC (azınlık "
        "sınıfı baskınken ROC’un düzeltmesi), MCC (dört hücreyi birden dikkate alan özet) "
        "ve “dengeli F1” (1:1 varsayımındaki beklenen F1) birlikte raporlanmıştır. Küçük "
        "panel oynaklığı da ele alınmıştır: CFTR’de 21 benign vardır ve tek bir hata "
        "özgüllüğü ~4,8 puan değiştirir; bu nedenle iç içe CV standart sapması ile 200 "
        "örneklemli bootstrap standart hatası raporlanır ve model seçimi bu bandı dikkate "
        "alan One-SE kuralıyla yapılır."),
    ("p", "Hata türlerinin klinik maliyeti simetrik değildir: yanlış negatif tanı "
        "gecikmesine ve hatalı üreme danışmanlığına yol açabilirken, yanlış pozitif "
        "gereksiz doğrulama testi ve aile taraması yükü üretir — çoğunlukla geri "
        "döndürülebilir bir maliyet. Bu asimetri nedeniyle eşik duyarlılık lehine "
        "seçilmiş, dört panelde de duyarlılık 0,955–0,994 aralığında tutulmuştur; bedeli "
        "dengeli F1 üzerinden ölçülmüştür (Tablo 3). İzotonik kalibratörler paketle "
        "saklandığından, test dağılımı farklı çıkarsa model yeniden eğitilmeden eşik "
        "Bayes-optimal noktaya taşınabilir."),
    ("h2", "3.6. Seçilen Algoritmalar ve Gerekçe"),
    ("p", "Veri yapısı — heterojen ölçekli tablo verisi, satır başına ortalama 192 eksik "
        "değer, 111–2.931 satır ve yüksek öznitelik sayısı — GBDT ailesini doğal seçim "
        "yapar: eksik değeri silmeden işler, monoton olmayan etkileşimleri yakalar, "
        "ölçeklemeye ihtiyaç duymaz, SHAP açıklamasına elverişlidir; en yakın "
        "çalışmaların aynı ailede birleşmesi ([4], [10]) bu seçimi destekler. Temel "
        "katmanda beş model kullanılır: XGBoost [14], LightGBM [15] ve CatBoost — "
        "sırasıyla seviye bazlı, yaprak bazlı ve sıralı artırma ile farklı yanlılık "
        "profilleri; ExtraTrees — rastgele bölme eşikleriyle farklı bir varyans profili; "
        "L1-düzenlileştirilmiş lojistik regresyon — ağaç ailesine dik, doğrusal ve seyrek "
        "bir sinyal. Bu çeşitlilik, meta katmanın tamamlayıcı hatalardan yararlanabilmesi "
        "için kasıtlıdır. Dengesizlik model düzeyinde yönetilir: XGBoost ve LightGBM’de "
        "scale_pos_weight log ölçekli 0,5–6,0 aralığında aranan bir çarpanla ölçeklenir, "
        "CatBoost’ta auto_class_weights = “Balanced”, diğerlerinde class_weight "
        "kullanılır; geniş aralık bilinçlidir, çünkü CFTR’nin nihai modeli ~5,96’ya "
        "yakınsamıştır. Olasılık kalibrasyonu izotonik regresyonla [16] fold içinde "
        "öğrenilir ve stacking meta-öğrenicisinin girdisi olur; tekil modeller ham "
        "olasılıkla eşiklenir — izotonik dönüşüm monoton olduğundan ayarlanmış eşikte "
        "F1’i değiştirmez — kalibratörler yine de pakette saklanır. Düzenlileştirme "
        "mekanizmaları 5.3’te ayrıntılandırılmıştır."),
    ("h1", "4. DENEY TASARIMI, SONUÇLAR VE İNCELEME"),
    ("h2", "4.1. Deney Protokolü ve Veri Bölme"),
    ("p", "Nihai test seti hakemde tutulduğundan, “rastlantısal iyi sonuç” riskini "
        "azaltacak üç katmanlı bir iç değerlendirme düzeni kurulmuştur. (1) Model seçim "
        "katmanı: tabakalı 5-katlı çapraz doğrulama, 3 tekrarla (seed = 42); katlar arası "
        "ortalamayla out-of-fold (OOF) olasılıkları üretilir — Optuna denemelerinin "
        "içinde bütçe nedeniyle 2 tekrar. (2) Genelleme katmanı, iç içe çapraz doğrulama: "
        "5 dış kat × 2 tekrar = 10 dış kat; her dış katta öznitelik seçimi, temel model "
        "OOF üretimi, şampiyon seçimi ve eşik yalnızca eğitim satırlarında belirlenir ve "
        "dokunulmamış test katında bir kez skorlanır. Raporun ana genelleme sayıları "
        "buradan gelir. (3) A/B katmanı: %80/20 tabakalı held-out bölmeyle öznitelik "
        "seçim yöntemleri aynı test satırlarında karşılaştırılır. Hiperparametreler "
        "yalnızca birinci katmanda seçilir. Rastlantısallığa karşı dört ek önlem vardır: "
        "CV tekrarları, denemeler arası kat tohumu sarsması, 200 örneklemli bootstrap "
        "standart hatası ve fold-medyan eşik. Şeffaflık gereği: dış katlarda "
        "hiperparametreler yeniden aranmamış, global parametreler kullanılmıştır; bu, iç "
        "içe skorda küçük bir iyimserlik payı bırakır."),
    ("h2", "4.2. Performans Metrikleri ve Panel Bazlı Raporlama"),
    ("p", "Metrik seçimi 3.5’te gerekçelendirilmiştir; sonuçlar dört panel için ayrı "
        "hesaplanmıştır (Tablo 3). Eşik stratejisi: her katta 0,05–0,95 aralığı 0,005 "
        "adımla taranır ve katların F1-optimal eşiklerinin medyanı alınır; bu, tek bir "
        "bölünmeye aşırı uyumu engeller."),
    ("table", {
        "caption": "Tablo 3. Panel bazlı başarım. Üst blok model seçiminde kullanılan 5-katlı "
            "OOF değerleri, alt blok iç içe CV ile elde edilen sızıntısız genelleme "
            "tahminidir. Karmaşıklık matrisi ve kesinlik/duyarlılık, şampiyonun tek "
            "geçişli yeniden üretiminden gelir ve tekrarlı-CV OOF F1’inden en fazla 0,004 "
            "sapabilir.",
        "widths": [6.2, 2.4, 2.4, 2.4, 2.4],
        "fontsize": 9,
        "rows": [
            ["Ölçüt", "GENEL", "KANSER", "PAH", "CFTR"],
            ["Varyant sayısı", "2.931", "388", "372", "111"],
            ["Nihai model (One-SE)", "CatBoost", "XGBoost", "XGBoost", "XGBoost"],
            ["Seçilen öznitelik sayısı", "52", "35", "37", "13"],
            ["GENEL’den veri artırımı", "—", "Hayır", "Evet", "Evet"],
            ["Karar eşiği", "0,280", "0,515", "0,280", "0,810"],
            ["OOF F1", "0,899", "0,933", "0,941", "0,967"],
            ["OOF ROC-AUC", "0,865", "0,947", "0,842", "0,977"],
            ["OOF AUPRC", "0,933", "0,973", "0,944", "0,995"],
            ["OOF kesinlik / duyarlılık", "0,843 / 0,962", "0,905 / 0,955", "0,888 / 0,994", "0,967 / 0,967"],
            ["OOF MCC", "0,563", "0,760", "0,543", "0,824"],
            ["TN / FP / FN / TP", "397/385/81/2068", "93/27/12/256", "23/39/2/308", "18/3/3/87"],
            ["İç içe F1 (genelleme)", "0,894", "0,915", "0,925", "0,933"],
            ["İç içe ROC-AUC", "0,851", "0,924", "0,795", "0,948"],
            ["İç içe dengeli F1 (1:1)", "0,781", "0,852", "0,735", "0,801"],
            ["Naif taban F1 (hepsi patojenik)", "0,846", "0,817", "0,909", "0,896"],
            ["İyimserlik sapması (OOF − iç içe)", "+0,005", "+0,018", "+0,015", "+0,034"],
        ]}),
    ("p", "Eşikler panel karakterini yansıtır: ayrım gücü düşük olan PAH ile GENEL’de düşük "
        "eşik (0,280) duyarlılığı korurken, olasılıkları keskin ayrışan CFTR’de en iyi "
        "denge 0,810’da oluşur. Sonuçlar naif tabanla birlikte okunmalıdır: göreli "
        "iyileşmenin en belirgin olduğu panel 0,817’den 0,933’e çıkan KANSER’dir "
        "(+0,116). PAH’ta F1 yüksek görünse de (0,941) taban zaten 0,909’dur ve iç içe "
        "AUC 0,795 ile en düşük değerdir. Dört panelin ortalama iç içe F1 değeri "
        "0,917’dir."),
    ("h2", "4.3. Hata Analizi ve Model Davranışı"),
    ("p", "Yanlış sınıflanan örnekler OOF olasılıkları ve panele özgü eşiklerle dört gruba "
        "ayrılmış ve grup medyanları karşılaştırılmıştır (Tablo 4)."),
    ("table", {
        "caption": "Tablo 4. Hata gruplarının medyan profilleri; hücreler TN / FP / TP sırasıyla "
            "verilmiştir. Yanlış pozitifler her göstergede TN ile TP arasında konumlanır.",
        "widths": [4.6, 3.7, 3.7, 3.6],
        "fontsize": 9,
        "rows": [
            ["Gösterge (medyan)", "GENEL (TN/FP/TP)", "KANSER (TN/FP/TP)", "PAH (TN/FP/TP)"],
            ["Eksik öznitelik sayısı", "19 / 201 / 253", "19 / 213 / 300", "212 / 177 / 202"],
            ["ek_mean (korunmuşluk)", "2,16 / 3,46 / 3,70", "1,76 / 3,44 / 3,83", "2,64 / 3,44 / 3,78"],
            ["Grantham uzaklığı", "56,9 / 77,7 / 97,8", "62,2 / 77,7 / 97,8", "55,3 / 59,5 / 81,2"],
            ["Mevcut in-silico araç sayısı", "103 / 47 / 24", "156 / 66 / 14", "43 / 53 / 43"],
            ["Grup büyüklüğü (n)", "397 / 385 / 2.068", "93 / 27 / 256", "23 / 39 / 308"],
        ]}),
    ("p", "(i) Anotasyon eksiksizliği bir kısayol oluşturur: doğru sınıflanan patojenik "
        "varyantların medyan eksik öznitelik sayısı, benign olanlarınkinin çok "
        "üzerindedir (GENEL 253’e karşı 19). Tek başına pct_missing GENEL’de 0,653, "
        "KANSER’de 0,769, CFTR’de 0,616 ROC-AUC üretir; mevcut in-silico araç sayısı ters "
        "yönde aynı gücü taşır (0,641 ve 0,752). Biyolojik açıklaması makuldür — yaygın, "
        "çoğunlukla benign varyantlar popülasyon veri tabanlarında iyi kapsanırken nadir "
        "patojenik varyantlar seyrek anote edilir — ancak modelin bu eksene yaslanması "
        "Grimm ve ark. [11]’nın daireselliğinin bu veri kümesindeki karşılığıdır. (ii) "
        "Yanlış pozitifler tam bu eksende yoğunlaşır: FP grubunun medyan eksiklik profili "
        "(GENEL 201, KANSER 213) TN grubundan (19) çok uzak, TP grubuna yakındır. (iii) "
        "Gri bölge: FP grubunun hem korunmuşluk hem Grantham medyanı TN ile TP "
        "arasındadır — model orta şiddetli ikamelerde, yani klinikte VUS olarak kalan "
        "varyantların bölgesinde zorlanır."),
    ("p", "(iv) Hataların eşiğe uzaklığı panelden panele değişir ve farklı müdahale "
        "gerektirir. GENEL’de 385 yanlış pozitifin 77’si ve 81 yanlış negatifin 43’ü "
        "eşiğin ±0,10 bandındadır; bunlar eşik ayarıyla kısmen geri kazanılabilir. "
        "KANSER’de 27 yanlış pozitifin yalnızca 6’sı bu banttadır ve 12 yanlış negatifin "
        "hiçbiri değildir (olasılıkları 0,017–0,382): bunlar “kendinden emin” hatalardır, "
        "eşikle çözülemez ve öznitelik düzeyinde bir eksikliğe işaret eder. CFTR’de üç "
        "yanlış pozitifin üçü de 0,816–0,843 aralığındadır (eşik 0,810) — tamamen "
        "sınırdadır. (v) PAH’ın yapısal zorluğu ayrı bir nedene dayanır: bu panelde "
        "pct_missing’in tek başına AUC’si 0,519’dur, yani eksiklik kısayolu PAH’ta "
        "yoktur; model yalnızca in-silico skorlara dayanmak zorunda kalır ve bu skorlar "
        "PAH benign varyantlarını ayırmakta zayıftır (62 benign’in 39’u yanlış pozitif). "
        "Bu, PAH’ın yüksek F1’inin neredeyse tamamen taban etkisinden geldiğini doğrular."),
    ("h2", "4.4. Açıklanabilirlik: Model Neden Böyle Karar Verdi?"),
    ("p", "Açıklanabilirlik iki yöntemle ele alınmıştır: nihai şampiyon modeller üzerinde "
        "tüm panel satırları için hesaplanan TreeSHAP değerleri [17] ve bunlarla çapraz "
        "kontrol edilen kazanç tabanlı öznitelik önemi. Kolon adları paylaşılmadığı için "
        "açıklama dili öznitelik grupları üzerinden kurulmuş; her grubun katkısı, gruba "
        "ait özniteliklerin ortalama mutlak SHAP değerlerinin toplamıdır."),
    ("table", {
        "caption": "Tablo 5. Öznitelik gruplarının karara katkısı (ortalama |SHAP| toplamının "
            "yüzdesi). Tire, grubun o panelin seçilmiş kümesinde bulunmadığını gösterir.",
        "widths": [6.2, 2.4, 2.4, 2.4, 2.4],
        "fontsize": 9,
        "rows": [
            ["Öznitelik grubu", "GENEL", "KANSER", "PAH", "CFTR"],
            ["In-silico risk skorları", "%32,9", "%29,8", "%42,5", "%33,5"],
            ["Evrimsel korunmuşluk (EK)", "%19,1", "%28,0", "%11,0", "%28,0"],
            ["Popülasyon frekansları (ham + türetilmiş)", "%22,9", "%20,5", "%34,0", "%25,5"],
            ["Biyokimyasal / yapısal etki", "%15,0", "%13,4", "%11,2", "%13,0"],
            ["Anotasyon eksiksizliği", "%6,9", "%6,0", "—", "—"],
            ["Kategorik bağlam, oy birliği, etkileşim", "%3,3", "%2,2", "%1,2", "—"],
        ]}),
    ("p", "Dört panelde de in-silico risk skorları ile evrimsel korunmuşluk toplam katkının "
        "%52–62’sini oluşturur; popülasyon frekansı %20–34, biyokimyasal etki %11–15 pay "
        "alır. Bu sıralama ACMG/AMP kanıt kodlarıyla [1] uyumludur: hesaplamalı kanıt "
        "(PP3/BP4), popülasyon nadirliği (PM2/BA1) ve biyokimyasal bağlam (PP2/BP1) aynı "
        "öncelik sırasındadır. Panel farkları da yorumlanabilirdir: PAH’ta in-silico "
        "skorlar %42,5 ile ezici üstünlüktedir ve korunmuşluk yalnızca %11 pay alır — "
        "4.3’teki “PAH’ta eksiklik kısayolu yok” bulgusuyla örtüşür. KANSER ve CFTR’de "
        "korunmuşluk %28 ile in-silico skorlarla eşit ağırlıktadır ve bu iki panel en "
        "yüksek iç içe AUC değerlerine sahiptir (0,924 ve 0,948). Anotasyon "
        "eksiksizliğinin katkısı %0–6,9 ile sınırlıdır: model, 4.3’te ölçülen kısayolu "
        "kullanabilecekken büyük ölçüde kullanmamaktadır."),
    ("p", "Örnek: KANSER panelinden VAR_003700 (gerçek etiket patojenik, model olasılığı "
        "1,000). En büyük SHAP katkıları AL_9 (popülasyon frekans grubu, 0,0305; +1,14), "
        "ek_std (korunmuşluk skorlarının satır içi standart sapması, 3,95; +1,10), ek_max "
        "× Grantham etkileşimi (1.167,8; +0,55) ve ek_max (11,93 — panelin üst sınırı; "
        "+0,44) olmuştur. Karar tek bir özniteliğe değil, korunmuşluğun hem seviyesine "
        "hem iç değişkenliğine ve bunun şiddetli bir biyokimyasal ikameyle (Grantham "
        "97,9) birleşimine dayanır. AL_9 tek başına ayırıcı değildir (marjinal ROC-AUC "
        "0,489); katkısı yalnızca etkileşim üzerinden ortaya çıkar — bu, tekil sıralama "
        "ölçütlerinin neden yanıltıcı olabildiğini somutlaştırır."),
    ("h2", "4.5. Öğrenme Süreci ve Teknik Evrim"),
    ("p", "Beş sorun ve müdahalesi. (1) 442 öznitelik / 111–388 satır oranı aşırı uyum "
        "üretiyordu → null-importance seçimi (5 gerçek kazanç koşusunun ortalaması, 40 "
        "karıştırılmış etiket koşusundan üretilen null dağılımın %95 yüzdeliğiyle "
        "karşılaştırılır, ek olarak top-K tabanı [18]) → 442’den panel başına 13–52 "
        "özniteliğe inildi; seçim sonrası OOF F1 CFTR’de 0,951 → 0,962, KANSER’de 0,922 → "
        "0,929. (2) F1’i doğrudan optimize etmek çoğunluk sınıfına kayıyordu → Optuna "
        "hedefi ve şampiyon ölçütü ROC-AUC yapıldı, eşik ayrı bir adım olarak "
        "fold-medyanıyla belirlendi → iyimserlik sapması +0,005…+0,034 bandına indi. (3) "
        "PAH (62 benign) ve CFTR (21 benign) panellerinde azınlık sınıfı karar sınırını "
        "kuramıyordu → GENEL panelinden, hedef panelde bulunmayan satırların yalnızca "
        "eğitim fold’larına eklenmesi → benign eğitim yoğunluğu PAH’ta 62 → 831, CFTR’de "
        "21 → 799; CFTR iç içe AUC ≈ 0,88 → 0,948. KANSER’de regresyon görüldüğünden "
        "uygulanmadı. (4) Bilgi kazancına göre öznitelik eklemek KANSER held-out F1’ini "
        "0,9326 → 0,9314 düşürdü → ReliefF işaretiyle kapılanan hibrit seçici → KANSER "
        "held-out F1 0,891 → 0,911 (36 yerine 22 öznitelik). (5) OOF skorları iyimserdi "
        "(CFTR’de 0,978’e karşı 0,933) → iç içe çapraz doğrulama standart raporlama "
        "katmanı hâline getirildi. Ayrıca kalibreli stacking hiçbir panelde tekil modeli "
        "geçemediğinden One-SE kuralıyla yalınlık tercih edilmiş, ensemble altyapısı yine "
        "de paketlenmiştir."),
    ("p", "En önemli ders, tek bir bölünmede gözlenen iyileşmelere güvenmemek olmuştur: "
        "hibrit seçici KANSER’de +0,020 kazanç göstermesine rağmen diğer panellerde "
        "etkisi gürültü bandında kaldığı için varsayılan yapılmamıştır. Kısa vadede skoru "
        "bir miktar düşüren bu muhafazakârlık, bilinmeyen bir test dağılımı karşısında "
        "bilinçli bir tercihtir."),
    ("h1", "5. YAKLAŞIMIN GEREKÇESİ, KAYNAK KULLANIMI VE ÖZGÜNLÜK"),
    ("h2", "5.1. Neden Bu Algoritma / Mimari?"),
    ("p", "Model ailesinin veri yapısına uygunluğu 3.6’da gerekçelendirilmiştir; burada "
        "mimarinin genelleme ve tekrarlanabilirlik boyutu ele alınmaktadır. Panel bazlı "
        "bağımsız pipeline tercihi, dört panelin iç içe F1 değerlerini 0,894–0,933 gibi "
        "dar bir banda oturtmuş ve hiçbir panelde naif tabanın altına düşmemiştir; mimari "
        "hem 2.931 satırlı hem 111 satırlı panelde tutarlı çalışır. Gürültüye "
        "dayanıklılık, kat tohumunun denemeler arası sarsılması ve fold-medyan eşikle "
        "sağlanır. Tekrarlanabilirlik açısından tüm akış sabit tohumla (42) tek bir "
        "betikten uçtan uca çalışır; model paketleri seçilmiş öznitelik listesi ve karar "
        "eşiğiyle diske yazılır ve çıkarım aynı öznitelik üretim kodunu kullanır, böylece "
        "eğitim–çıkarım tutarsızlığı yapısal olarak imkânsızdır."),
    ("h2", "5.2. Alternatifler Neden Elendi?"),
    ("p", "(1) Tek birleşik model: panellerin seçilmiş öznitelik kümeleri büyük ölçüde "
        "ayrıktır — 137 seçim yalnızca 106 farklı özniteliğe karşılık gelir, hiçbir "
        "öznitelik dört panelin tamamında seçilmemiştir ve yalnızca dördü (AL_7, AL_26, "
        "ek_max, al_n_ultrarare) üç panelde ortaktır — ayrıca optimal eşikler 0,280 ile "
        "0,810 arasında dağılır; ayrıca panel kimliğinin etiketle ilişkilenmesi dolaylı "
        "sızıntı riski doğururdu. (2) Protein dil modelleri ([8], [9]): protein dizisi ve "
        "rezidü pozisyonu gerektirir, bunlar veride gizlenmiştir. (3) Tablo verisi için "
        "derin ağlar (TabNet / FT-Transformer türü): 111–388 satırda aşırı uyum riski "
        "yüksek, açıklanabilirlik zayıftır. (4) SMOTE: %55’i eksik olan 442 boyutlu bir "
        "uzayda komşuluk temelli interpolasyon var olmayan profiller üretir; yerine "
        "gerçek GENEL satırlarıyla artırım tercih edilmiştir. (5) Kalibreli stacking: her "
        "panelde değerlendirilmiş, ancak One-SE kuralında tekil modeli aşamamıştır."),
    ("h2", "5.3. Parametre Seçimi ve Model Ayarları"),
    ("p", "Hiperparametre araması Optuna [19] TPE örnekleyicisiyle, panel × model başına "
        "ayrı çalışmalar hâlinde yürütülmüştür (model başına en fazla 80 deneme, 600 s "
        "bütçe sınırı); ilk deneme olarak elle ayarlanmış bir varsayılan kuyruğa alınır, "
        "böylece bütçe erken tükense bile makul bir çözüm garanti edilir. Arama uzayı: "
        "XGBoost’ta max_depth 2–8, gamma 0–5, min_child_weight 1–10; LightGBM’de "
        "num_leaves 15–255, min_child_samples 5–100, min_split_gain 0–1; her ikisinde "
        "learning_rate 0,01–0,3 (log), subsample 0,5–1,0, colsample_bytree 0,3–1,0, "
        "reg_alpha/reg_lambda 10⁻⁸–10 (log) ve sınıf ağırlığı çarpanı 0,5–6,0 (log). "
        "CatBoost’ta depth 3–9, l2_leaf_reg 1–10; ExtraTrees’te n_estimators 200–800, "
        "max_depth 4–24, min_samples_leaf 1–20; lojistik regresyonda C 10⁻³–20 (log); "
        "400’den az satırlı panellerde sınırlar daraltılır. Doğrulama metriği "
        "ROC-AUC’dir; her deneme 2 tekrarlı 5-katlı çapraz doğrulamayla ve denemeye özgü "
        "kat tohumuyla değerlendirilir. Karar eşiği kasıtlı olarak aramanın dışında "
        "tutulmuştur: eşik sınıf oranına doğrudan bağlı olduğundan arama içinde optimize "
        "edilseydi eğitim önceliğine aşırı uyum kaçınılmaz olurdu; bunun yerine eşik, "
        "model seçildikten sonra OOF olasılıkları üzerinde fold-medyanıyla belirlenir."),
    ("h2", "5.4. Hesaplama Kaynakları ve Çalıştırılabilirlik"),
    ("p", "Nihai tam bütçeli eğitim koşusu Google Colab’da GPU hızlandırmalı çalışma "
        "zamanında yürütülmüştür (XGBoost device = “cuda”, CatBoost task_type = “GPU”; "
        "ExtraTrees ve lojistik regresyon CPU). Yazılım yığını: Python 3, scikit-learn "
        "[20], XGBoost, LightGBM, CatBoost, Optuna, SHAP, joblib. Ölçülen panel başına "
        "duvar saati: GENEL 7.343 s (2 sa 2 dk), PAH 4.441 s (1 sa 14 dk), KANSER 3.974 s "
        "(1 sa 6 dk) — üç panelin toplamı 15.758 s (≈ 4 sa 23 dk); CFTR en küçük panel "
        "olarak en kısa süreyi almıştır. Bu süre, panel başına beş modelin 80 denemeye "
        "kadar aranmasını, 3 tekrarlı OOF üretimini ve 10 dış katlı iç içe "
        "değerlendirmeyi kapsar; gradyan artırma kullanıldığından epoch/batch yerine ağaç "
        "sayısı raporlanır (varsayılan 400 ağaç, CatBoost’ta 500 iterasyon)."),
    ("p", "Çözüm GPU olmadan da doğrulanmıştır: yerel referans ortamda (AMD Ryzen 7 7730U, "
        "8 çekirdek / 16 iş parçacığı, 37 GB RAM, Linux, Python 3.12.3; numpy 2.4.6, "
        "pandas 3.0.3, scikit-learn 1.9.0, xgboost 3.2.0, lightgbm 4.6.0, catboost "
        "1.2.10, optuna 4.9.0, shap 0.52.0) pipeline yalnızca CPU ile uçtan uca "
        "çalıştırılmış ve aynı öznitelik seçimini üretmiştir. En büyük eğitim matrisi "
        "(GENEL, 2.931 × 442) 10,4 MB, ölçülen tepe yerleşik bellek 305 MB’dır. Model "
        "paketleri panel başına 4,6–29,8 MB yer kaplar; şampiyon modelin kendisi "
        "0,35–0,45 MB’dır. Çıkarım: toplu tahmin GENEL’in 2.931 satırını 0,67 s’de (satır "
        "başına 0,23 ms), diğer panelleri 0,18–0,23 s’de tamamlar; tek varyant için uçtan "
        "uca gecikme — paket yükleme ve öznitelik üretimi dâhil — 65–124 ms’dir ve paket "
        "bellekte tutulduğunda milisaniye altına iner. Determinizm sabit tohumla "
        "sağlanmıştır. Özetle eğitim tek seferlik ve birkaç saatlik, çıkarım pratik "
        "olarak anlıktır."),
    ("h2", "5.5. Özgünlük"),
    ("p", "(1) Sızıntısız çapraz-panel veri artırımı: küçük panellerin azınlık sınıfı "
        "sorunu sentetik örnekleme yerine gerçek verilerle çözülmüştür. GENEL panelinin "
        "hedef panelde bulunmayan satırları yalnızca eğitim fold’larına eklenir, "
        "değerlendirme her zaman hedef panelin kendi satırlarında yapılır; ön koşul olan "
        "etiket tutarlılığı 578 ortak varyantta %100 ölçülmüştür. Ablasyon sonuçları "
        "4.5’tedir (CFTR iç içe AUC ≈ 0,88 → 0,948)."),
    ("p", "(2) Kolon adları olmadan öznitelik grubu anlamlandırma: gruplar istatistiksel "
        "imzalarından (değer aralığı, çarpıklık, ikili/sürekli olma, kardinalite) "
        "çıkarılmış, türetilen 91 öznitelik bu yorumlara dayandırılmış ve SHAP katkıları "
        "grup düzeyinde toplanarak ACMG kanıt kodlarıyla eşlenmiştir (Tablo 5) — bu "
        "çerçeve anonim veride açıklanabilirliği klinik olarak anlamlı kılar."),
    ("p", "(3) ReliefF kapılı hibrit öznitelik seçici: null-importance seçimine, yerel "
        "ayrım gücünü ölçen ReliefF [21] işaretinin ikinci bir kapı olarak eklenmesi. "
        "Ablasyon (aynı %80/20 held-out bölmesi, seçim yalnızca eğitim satırlarında): "
        "KANSER test F1 null seçicide 0,891 (36 öznitelik), bilgi kazancı temelli "
        "genişletmede 0,879 (39 öznitelik), ReliefF ile budanmış kümede 0,911 (22 "
        "öznitelik) — bilgi kazancına göre öznitelik eklemek zarar verir, çıkarmak yardım "
        "eder."),
    ("p", "(4) Dağılım kaymasına dayanıklı karar mimarisi: sınıf oranından bağımsız ROC-AUC "
        "hedefi, AUC üzerinde One-SE ile yalınlık tercihi [22], fold-medyan eşik ve "
        "izotonik kalibratörlerin paketle saklanması birlikte çalışır — sonuncusu, test "
        "setinin sınıf oranı farklı çıkarsa modeli yeniden eğitmeden eşiği Bayes-optimal "
        "noktaya taşıma imkânı verir. Dört panelde de One-SE kuralı tekil modeli seçmiş, "
        "iyimserlik sapması +0,005…+0,034 bandında kalmıştır."),
    ("p", "(5) Kısayol sinyalinin ölçülmesi ve sınırlanması: 4.3’te gösterildiği gibi "
        "anotasyon eksiksizliği tek başına GENEL’de 0,653, KANSER’de 0,769 ROC-AUC "
        "üretmektedir; literatürde dairesellik olarak tartışılan bu olgunun [11] bu veri "
        "kümesi için ölçülüp raporlanması çalışmanın ayırt edici katkılarındandır ve "
        "nihai modellerde bu grubun SHAP katkısı %0–6,9 ile sınırlı kalmıştır. Raporlama "
        "disiplini de bilinçli bir tercihtir: her tabloda OOF ve iç içe CV değerleri yan "
        "yana verilmiş ve naif taban hiçbir panelde gizlenmemiştir."),
]

REFERENCES = [
    "S. Richards, N. Aziz, S. Bale, D. Bick, S. Das, J. Gastier-Foster ve ark., "
    "“Standards and guidelines for the interpretation of sequence variants: a joint "
    "consensus recommendation of the American College of Medical Genetics and Genomics "
    "and the Association for Molecular Pathology,” Genetics in Medicine, c. 17, s. 5, ss. "
    "405–424, 2015.",
    "M. J. Landrum, J. M. Lee, M. Benson, G. R. Brown, C. Chao, S. Chitipiralla ve ark., "
    "“ClinVar: improving access to variant interpretations and supporting evidence,” "
    "Nucleic Acids Research, c. 46, s. D1, ss. D1062–D1067, 2018.",
    "K. J. Karczewski, L. C. Francioli, G. Tiao, B. B. Cummings, J. Alföldi, Q. Wang ve "
    "ark., “The mutational constraint spectrum quantified from variation in 141,456 "
    "humans,” Nature, c. 581, ss. 434–443, 2020.",
    "A. Jain, G. Slabaugh ve D. Gurdasani, “Classification of genetic variants using "
    "machine learning,” arXiv:2112.05154, 2021.",
    "N. M. Ioannidis, J. H. Rothstein, V. Pejaver, S. Middha, S. K. McDonnell, S. Baheti "
    "ve ark., “REVEL: An ensemble method for predicting the pathogenicity of rare "
    "missense variants,” The American Journal of Human Genetics, c. 99, s. 4, ss. "
    "877–885, 2016.",
    "D. Quang, Y. Chen ve X. Xie, “DANN: a deep learning approach for annotating the "
    "pathogenicity of genetic variants,” Bioinformatics, c. 31, s. 5, ss. 761–763, 2015.",
    "C. Lai, M. K. Zimmer, R. O’Connor, S. Kim, R. Chan, K. van den Akker ve ark., “LEAP: "
    "Using machine learning to support variant classification in a clinical setting,” "
    "Human Mutation, c. 41, s. 12, ss. 2049–2060, 2020.",
    "J. Cheng, G. Novati, J. Pan, C. Bycroft, A. Žemgulytė, T. Applebaum ve ark., "
    "“Accurate proteome-wide missense variant effect prediction with AlphaMissense,” "
    "Science, c. 381, s. 6664, eadg7492, 2023.",
    "N. Brandes, G. Goldman, C. H. Wang, C. J. Ye ve V. Ntranos, “Genome-wide prediction "
    "of disease variant effects with a deep protein language model,” Nature Genetics, c. "
    "55, ss. 1512–1522, 2023.",
    "J. Ruzicka, S. Aradhya ve K. Nykamp, “Clinical evaluation of an artificial "
    "intelligence system for streamlined variant interpretation in genetic testing,” "
    "medRxiv, 2025.",
    "D. G. Grimm, C.-A. Azencott, F. Aicheler, U. Gieraths, D. G. MacArthur, K. E. "
    "Samocha ve ark., “The evaluation of tools used to predict the impact of missense "
    "variants is hindered by two types of circularity,” Human Mutation, c. 36, s. 5, ss. "
    "513–523, 2015.",
    "S. Henikoff ve J. G. Henikoff, “Amino acid substitution matrices from protein "
    "blocks,” Proceedings of the National Academy of Sciences USA, c. 89, s. 22, ss. "
    "10915–10919, 1992.",
    "R. Grantham, “Amino acid difference formula to help explain protein evolution,” "
    "Science, c. 185, s. 4154, ss. 862–864, 1974.",
    "T. Chen ve C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. 22nd "
    "ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, "
    "ss. 785–794.",
    "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma ve ark., “LightGBM: A highly "
    "efficient gradient boosting decision tree,” in Advances in Neural Information "
    "Processing Systems (NeurIPS), 2017, ss. 3146–3154.",
    "B. Zadrozny ve C. Elkan, “Transforming classifier scores into accurate multiclass "
    "probability estimates,” in Proc. 8th ACM SIGKDD International Conference on "
    "Knowledge Discovery and Data Mining, 2002, ss. 694–699.",
    "S. M. Lundberg ve S.-I. Lee, “A unified approach to interpreting model predictions,” "
    "in Advances in Neural Information Processing Systems (NeurIPS), 2017, ss. 4765–4774.",
    "A. Altmann, L. Toloşi, O. Sander ve T. Lengauer, “Permutation importance: a "
    "corrected feature importance measure,” Bioinformatics, c. 26, s. 10, ss. 1340–1347, "
    "2010.",
    "T. Akiba, S. Sano, T. Yanase, T. Ohta ve M. Koyama, “Optuna: A next-generation "
    "hyperparameter optimization framework,” in Proc. 25th ACM SIGKDD International "
    "Conference on Knowledge Discovery and Data Mining, 2019, ss. 2623–2631.",
    "F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel ve ark., "
    "“Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, c. "
    "12, ss. 2825–2830, 2011.",
    "I. Kononenko, “Estimating attributes: Analysis and extensions of RELIEF,” in Proc. "
    "European Conference on Machine Learning (ECML), 1994, ss. 171–182.",
    "L. Breiman, J. H. Friedman, R. A. Olshen ve C. J. Stone, Classification and "
    "Regression Trees. Belmont, CA, ABD: Wadsworth, 1984.",
]
