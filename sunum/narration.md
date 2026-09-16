# PatHUgen — Sunum Anlatım Metni (narration)

**Kaynak:** `sunum/PatHUgen.pptx` (18 slayt: kapak + 17 içerik)
**Kural (Final Uygulama Kılavuzu §5):** Sunum **10 dakika**, soru-cevap **3 dakika**. Sunum 100 puan üzerinden, nihai sıralamanın **%10**'u.
**Nihai metrik (Üniversite ve Üzeri):** panel başına **F1 ve MCC'nin aritmetik ortalaması**, her panel eşit ağırlık; eşitlikte **PR-AUC**.

> Bu dosya yalnızca anlatım metnidir — sunum dosyasında hiçbir değişiklik yapılmadı.
> Metinler kelime sayısı ~2,4 kelime/saniye üzerinden zamanlandı. Aynen ezberlemek zorunda
> değilsin; **her slaytın "Tek cümlelik omurga"sı** ezberlenirse gerisi doğaçlama taşınır.

---

## Süre planı

| # (pptx) | Alt bilgi no | Slayt | Süre | Kümülatif |
|---|---|---|---|---|
| 1 | — | Kapak | 0:08 | 0:08 |
| 2 | 1 | Takım Tanıtımı | 0:20 | 0:28 |
| 3 | 2 | Sunum İçeriği | 0:10 | 0:38 |
| 4 | 3 | Problem ve Klinik Önem | 0:50 | 1:28 |
| 5 | 4 | Yöntem — 7 adım, iki hat | 0:45 | 2:13 |
| 6 | 5 | Arama hattı mimarisi | 0:30 | 2:43 |
| 7 | 6 | Öznitelik mühendisliği | 0:40 | 3:23 |
| 8 | 7 | Öznitelik seçimi | 0:30 | 3:53 |
| 9 | 8 | Model + karar eşiği | 0:40 | 4:33 |
| 10 | 9 | **Sonuçlar** | 1:00 | 5:33 |
| 11 | 10 | Hata analizi + öznitelik önemi | 0:40 | 6:13 |
| 12 | 11 | Ek veri kullanımı ve açıklık | 0:30 | 6:43 |
| 13 | 12 | **Cross-panel augmentation** | 0:50 | 7:33 |
| 14 | 13 | Yenilikçilik + riskler | 0:35 | 8:08 |
| 15 | 14 | Rekabet + iş modeli | 0:35 | 8:43 |
| 16 | 15 | Klinik entegrasyon | 0:30 | 9:13 |
| 17 | 16 | Karar kuralı / veri sözleşmesi / emniyet | 0:30 | 9:43 |
| 18 | 17 | Teşekkür | 0:12 | **9:55** |

**Kontrol noktası:** Slayt 10 (Sonuçlar) bitiminde saat **5:40'ı geçmemeli**.
Geçtiyse → "Zaman aşımı planı" bölümüne bak.

**Önerilen ikili paylaşım:** Konuşmacı A → slayt 1–4 ve 12–18 (problem, açıklık, iş/klinik).
Konuşmacı B → slayt 5–11 (teknik yöntem ve sonuçlar). Geçiş slayt 5'in başında ve slayt 12'nin başında.

---

## Slayt 1 — Kapak · 0:08

**Omurga:** Kim olduğumuz + hangi problem.

> İyi günler. Biz PatHUgen takımıyız. Üniversite ve üzeri seviyesinde, missense genetik
> varyantların patojenik–benign sınıflandırması üzerine çalıştık.

---

## Slayt 2 (alt bilgi 1) — Takım Tanıtımı · 0:20

**Omurga:** Takım yapısı ve roller — hızlı geç, jüri buradan puan vermez, zaman kaybı en pahalı hatadır.

> Takımımız [N] kişiden oluşuyor, danışmanımız [AD SOYAD]. Takım kaptanımız [AD SOYAD].
> Takım numaramız 761197, başvuru numaramız 4882800.
> Ekipte biyoinformatik ve makine öğrenmesi tarafı birlikte çalıştı; veri analizi, model arama
> ve teslim altyapısı ayrı ayrı sahiplenildi.

⚠️ **Sunumdan önce:** slayttaki "AD SOYAD" ve "Bölüm / Üniversite" alanları doldurulmuş olmalı.

---

## Slayt 3 (2) — Sunum İçeriği · 0:10

**Omurga:** Beş başlık, doğrudan probleme geç.

> Sunumda beş başlık var: problemin klinik önemi, teknik yöntem, ek veri kullanımı ve açıklık,
> yenilikçilik ve klinik entegrasyon. Doğrudan problemle başlayalım.

---

## Slayt 4 (3) — Problemin Tanımı ve Klinik Önemi · 0:50

**Omurga:** Darboğaz sekanslama değil, **yorumlama**. Ve veri dengesiz — naif taban 0,85–0,91.

> Bugün bir hastanın genomunu okumak artık darboğaz değil. Darboğaz, okunan varyantın ne
> anlama geldiğine karar vermek. Şemadaki dört adımdan üçüncüsü — yorumlama — hâlâ uzmanın
> masasında ve büyük ölçüde elle yapılıyor.
>
> Türkiye'de nadir hastalığı olan bir birey, doğru tanıya ulaşana kadar ortalama sekiz uzman
> hekimden geçiyor; süre yaklaşık yedi yıl.
>
> Bizim görevimiz tam bu adımı hızlandırmak: dört panelde, verilen bir varyantın patojenik mi
> benign mi olduğunu sınıflandırmak. Değerlendirme panel başına F1 ve MCC ortalamasıyla yapılıyor,
> karar eşiği de panel başına ayrı.
>
> Burada altını çizmek istediğim bir tuzak var: veri dengesiz, patojenik sınıf çoğunlukta —
> PAH panelinde yüzde 83. Yani hiçbir şey öğrenmeyip "hepsi patojenik" diyen naif bir model bile
> 0,85 ile 0,91 arası F1 alır. Bu yüzden bütün tablolarımızda naif taban çizgisi yanımızda duruyor.

💡 Slaytta "metrik binary F1" yazıyor; **sözlü olarak "F1 ve MCC ortalaması" de** — kılavuzdaki
resmî tanım bu, jüri bunu bilerek dinliyor.

---

## Slayt 5 (4) — Teknik Yöntem: 7 adım, iki hat · 0:45

**Omurga:** Arama hattı saatler sürer, teslim hattı saniyeler; ikisini **tek bir tarif dosyası** bağlar.

> Sistemimiz yedi adımdan oluşuyor: veri, öznitelik dönüşümü, öznitelik seçimi, veri artırımı,
> model, karar eşiği, çıktı.
>
> Ama asıl anlatmak istediğim mimari ayrım şu: iki ayrı hat var. **Arama hattı** yüzlerce
> kombinasyonu deneyip kazanan tarifi bulur — saatler sürer. **Teslim hattı** yalnızca o kazanan
> tarifi okuyup çalıştırır — saniyeler sürer.
>
> Tarif, kodun içine gömülü bir sayı değil; ayrı bir yapılandırma dosyasında duruyor ve bir insan
> bakıp onaylamadan değişmiyor. Eğitim ve tahmin aynı öznitelik dönüşümünü çağırıyor; bu yüzden
> ikisi arasında sapma yapısal olarak imkânsız.

---

## Slayt 6 (5) — Arama hattı mimarisi · 0:30

**Omurga:** Beş model ailesi + stacking denendi; **teslim edilen şey tek XGBoost.**

> Ekrandaki şema arama hattının kendisi. Beş taban model deniyoruz: XGBoost, LightGBM, CatBoost,
> ExtraTrees ve L1-lojistik regresyon; ayrıca izotonik kalibrasyonlu stacking kombinasyonları.
> Optuna'nın hedefi ROC-AUC.
>
> Kazananı seçerken One-SE kuralını kullanıyoruz: istatistiksel olarak birbirinden ayırt
> edilemeyen adaylar arasında **en yalın olanı** seçiyoruz. Dört panelde de kazanan tek bir
> XGBoost oldu. Yani teslim ettiğimiz sistemde stack yok — sahada çalışan şey, panel başına
> tek ve ayarlanmış bir model.

🔴 **Bu cümleyi atlama:** "teslim ettiğimiz sistemde stack yok". Atlarsan jüri şemaya bakıp
"5 modeli sahada mı koşturuyorsunuz?" diye sorar ve 30 saniye kaybedersin.

---

## Slayt 7 (6) — Öznitelik mühendisliği · 0:40

**Omurga:** 351 → 442 kolon, 91 türetilmiş; hepsi **etiketten bağımsız**; eksiklik bir imza.

> 351 ham kolonu 442 kolonluk bir temsile çeviriyoruz; bunun 91'i türetilmiş öznitelik.
> En büyük aile amino asit biyokimyası: 46 öznitelik BLOSUM62, 45 ve 80 skorlarını, Grantham
> uzaklığını ve dokuz fizikokimyasal ölçeği taşıyor — yani değişimin biyolojik şiddetini.
>
> Ham allel frekansları 10⁻³'ten 2,6 çarpı 10⁵'e uzanıyor; log ölçeğe alıyoruz.
>
> Sağdaki histogram önemli: eksiklik çift tepeli. Bir tarafta yoğun anote edilmiş varyantlar,
> diğer tarafta iskelet varyantlar. Bu gürültü değil, bir imza — onu da öznitelik olarak kodluyoruz.
>
> Kritik nokta: bütün dönüşüm durumsuz ve etiketten bağımsız. Bu yüzden matrisi bir kez kurup
> katlara bölmek sızıntı üretmiyor.

---

## Slayt 8 (7) — Öznitelik seçimi · 0:30

**Omurga:** Şansa karşı duramayan öznitelik modele giremez. 442 → 13–38.

> 442 aday öznitelik iki bağımsız istatistiksel filtreden geçiyor.
> Birincisi null-importance: etiketi rastgele karıştırıp önemi yeniden ölçüyoruz; şans
> dağılımını geçemeyen öznitelik eleniyor.
> İkincisi ReliefF: en yakın komşular üzerinden yerel ayrımı ölçüyor — global testin kaçırdığı
> ayrımı yakalıyor.
> En büyük ve en heterojen panel olan GENEL'de ikisinin **kesişimini** istiyoruz.
>
> Sonuç: panel başına 13 ile 38 arasında öznitelik. Yani modele giren her kolon, şansa karşı
> ayakta kalabildiğini kanıtlamış durumda.

💡 Slayt tablosunda aday havuzu **441** yazıyor, slayt 5 ve 7'de **442** geçiyor.
Sözlü olarak **442 de**; sorulursa doğru cevap: canlı kod 442 öznitelik üretiyor ve seçim
havuzu bu 442'nin tamamı.

---

## Slayt 9 (8) — Model ve karar eşiği · 0:40

**Omurga:** Tek model ailesi, dört ayrı tarif; eşik bilinçli olarak recall'a eğimli.

> Model ailesi tek: XGBoost. Ama panel başına dört ayrı tarif var.
> Üstteki tabloda GENEL derinlik 8 ile çalışıyor, diğer üç panel derinlik 3 ile — çünkü onlar
> küçük paneller, derin ağaç orada ezberler. Sınıf ağırlığı çarpanı CFTR'de en yüksek;
> benign sınıf orada en seyrek.
>
> Alt tabloda karar eşikleri var: 0,595'ten 0,78'e kadar değişiyor ve her panelde farklı bir
> ölçütle seçildi — F1, dengeli F1 ya da MCC.
>
> Sonuç bilinçli olarak recall'a eğimli: 0,86 ile 0,99 arası. Çünkü kaçan tanının maliyeti,
> fazladan bir testin maliyetinden yüksek.

---

## Slayt 10 (9) — Sonuçlar · 1:00 ⭐ *Sunumun en önemli slaytı*

**Omurga:** Sayılar dürüst CV'den; asıl kanıt F1 değil, **naif tabana göre MCC farkı**.

> Buradaki sayıların hepsi dürüst çapraz doğrulamadan geliyor: 5 dış kat, 3 tekrar.
> Öznitelik listesi, hiperparametreler ve eşik önceden kilitli; tek soru şu — hiç görmediği
> satırlarda ne alıyor.
>
> Dürüst F1: GENEL 0,876; KANSER 0,924; PAH 0,938; CFTR 0,952. Ortalama 0,92.
>
> Ama asıl bakılması gereken satır naif tabanla aramızdaki fark. Naif modelin MCC'si sıfır;
> bizim MCC'miz 0,56 ile 0,79 arasında. Dengeli doğruluk naifte 0,50, bizde 0,72 ile 0,90 arası.
> Yani kazanç F1'in birkaç puanından ibaret değil — model gerçekten ayrım öğreniyor.
>
> Şeffaf olmak adına iki sınırı da söylüyoruz: hiperparametreler her katta yeniden aranmıyor,
> bu küçük bir iyimserlik payı bırakıyor. Ve CFTR en yüksek skoru veriyor ama 111 satırla
> en oynak panel.

🔴 Son paragrafı **atlama**. Zayıflığı kendin söylersen jüri onu soru olarak sormaz; söylemezsen
soru olarak gelir ve savunmada kalırsın.

---

## Slayt 11 (10) — Hata analizi ve öznitelik önemi · 0:40

**Omurga:** Hatalar nereye düşüyor + eksiklik imzasını risk olarak kendimiz işaretliyoruz.

> Hatalar nereye düşüyor? GENEL'de 223 yanlış pozitif, 301 yanlış negatif; PAH'ta 34 yanlış
> pozitif. Bu, recall öncelikli eşiğin faturası ve bilinçli bir tercih: yanlış negatif kaçan tanı
> demek, yanlış pozitif ise geri döndürülebilir bir ek test.
>
> Altta panel başına en etkili yedi öznitelik var. Dikkat çeken şey: GENEL ve KANSER'de eksiklik
> imzası en tepede. Bunu bir başarı olarak değil, **bir risk olarak** işaretliyoruz — gerçek bir
> biyolojik sinyal de olabilir, veri toplama artefaktı da. Eksiklik grubunu tamamen çıkarıp F1'i
> ayrıca ölçmeyi planlıyoruz.

---

## Slayt 12 (11) — Ek veri kullanımı ve açıklık · 0:30

**Omurga:** Dış veri yok — ve bu bir tercih değil, doğrulanabilir bir kısıt.

> Ek veri sorusuna net cevap: TEKNOFEST verisi dışında hiçbir kaynak kullanılmadı.
> ClinVar, gnomAD, dbNSFP sorgulanmadı; AlphaMissense, ESM1b, REVEL gibi hazır model çıktıları
> eklenmedi.
>
> Bu bir tercih değil, **doğrulanabilir bir kısıt**: kromozom ve pozisyon gizli, kolon adları
> anonim. Bu ikisi dış kaynakla eşleştirmeyi teknik olarak imkânsız kılıyor — aynı sınır
> AlphaMissense'i de eler, çünkü o da dizi ve pozisyon ister.
>
> Kullandığımız tek dış bilgi sabit tablolar: BLOSUM, Grantham, fizikokimyasal ölçekler.
> Bunlar veri değil, ders kitabı sabiti.

---

## Slayt 13 (12) — Cross-panel augmentation · 0:50 ⭐ *En çok soru gelecek slayt*

**Omurga:** Ödünç satırlar hedef panelde **hiç yok** + 578 ortak varyantta etiketler %100 uyumlu
+ CFTR için bu bir süs değil, ön koşul.

> Peki dış veri yoksa küçük panellerde ne yaptık? Cross-panel augmentation.
> GENEL panelinden, hedef panelde **hiç bulunmayan** varyantları alıp yalnızca eğitim katına
> ekliyoruz. Doğrulama her zaman panelin kendi satırlarıyla yapılıyor — ödünç satır hiçbir katta
> test tarafına geçmiyor.
>
> Meşruiyeti iki şeye dayanıyor. Birincisi: ödünç satırlar tanım gereği hedef panelde yok, yani
> test edilecek satırın kopyası eğitime giremez. İkincisi: paneller arasında ortak olan 578
> varyantta etiketler yüzde 100 uyuşuyor — GENEL'in etiket tanımı panellerle aynı.
>
> CFTR'de bu bir iyileştirme değil, ön koşul: artırım kapalıyken AUC 0,50 — model hiçbir şey
> öğrenemiyor. Açıkken 0,976.
>
> KANSER'de ise denedik ve bağımsız kontrolde kaybetti; o yüzden dışarıda. Kuralımız net —
> bir yöntem bağımsız denemede kazanırsa giriyor, kaybederse girmiyor.

---

## Slayt 14 (13) — Yenilikçilik ve riskler · 0:35

**Omurga:** Katkı yeni mimari değil, **dürüst karar katmanı**; riskleri sormadan söylüyoruz.

> Özgün katkımız yeni bir mimari değil; daha dürüst bir karar katmanı. Dört başlık: sızıntısız
> cross-panel augmentation, adsız kolondan anlam çıkarımı, ReliefF kapılı hibrit seçici ve
> denetimden geçmiş sabit karar paketi. Dördü de küçük kohortlu klinik tablo problemlerine
> taşınabilir.
>
> Ve sormadan söylüyoruz: 3.802 varyant kavram kanıtıdır, klinik kanıt değil. CFTR satırlarının
> yaklaşık yüzde 96'sı ödünç — bu saf bir CFTR modeli değil, GENEL'den transfer. Eğitim verimizde
> VUS yok; laboratuvarın en çok vakit kaybettiği yeri çözmüş gibi göstermiyoruz.

---

## Slayt 15 (14) — Rekabet ve iş modeli · 0:35

**Omurga:** Farkımız kapsam değil, **konuşlanma** — veri kurumdan çıkmıyor.

> Rekabet tablosunda VarSome, Franklin, Emedgene ve Fabric Genomics var. Hepsi bulut tabanlı ve
> küratörlü dış veritabanlarına dayanıyor.
>
> Kapsam eşitliği iddia etmiyoruz — onlar tüm ekzomu kanıt kodlarıyla raporluyor, biz dört
> panelde missense önceliklendirmesi yapıyoruz. Farkımız kapsam değil, konuşlanma: dış çağrı yok,
> genomik veri kurumdan çıkmıyor, paket yaklaşık 1 megabayt, GPU gerekmiyor.
>
> İş modeli de labın örgütlenmesiyle aynı: satış birimi labın zaten çalıştırdığı panel, geliri
> panel başına yıllık lisans. Yol haritamız dört aşamalı: pilot, ileriye dönük doğrulama,
> regülasyon, ölçekleme.

---

## Slayt 16 (15) — Klinik entegrasyon · 0:30

**Omurga:** Anotasyondan sonra, uzmandan önce. Otonom tanı değil, triyaj.

> Sistem klinik akışın neresinde? Anotasyondan sonra, uzmandan önce. Otonom tanı değil, karar
> desteği.
>
> Uzmanın eline üç şey geçiyor: kalibre olasılık — 0,8 gerçekten "yüzde 80 patojenik" demek;
> panel başına dürüst F1 ve AUC, güven aralığıyla; ve ayarlanabilir çalışma noktası — laboratuvar
> eşiği kendi kohortuna kapalı formülle taşıyabiliyor, yeniden eğitim gerekmeden.
>
> Mimarimiz zaten klinik biçimde: laboratuvarlar zaten panel çalıştırıyor — kanser paneli,
> kistik fibrozis paneli, PKU paneli. Bizim dört bağımsız panelimiz bununla birebir örtüşüyor.

---

## Slayt 17 (16) — Karar kuralı, veri sözleşmesi, emniyet sınırı · 0:30

**Omurga:** İki eşik–üç bant; her satır izlenebilir; neyi **yapmadığımız** yazılı.

> Klinikte çalışması için üç şeyin yazılı olması gerekir.
> Bir, karar kuralı: iki eşik, üç bant. Üst bant otomatik kabul, alt bant otomatik ret, arada
> kalan her varyant uzmana gider — kararsızlık da bir çıktıdır.
> İki, veri sözleşmesi: giriş laboratuvarın kendi anotasyon çıktısı; çıkış mevcut tabloya eklenen
> dört kolon. Her satır hangi tarif ve hangi eşikten geldiğini taşıyor; denetim izi burada kurulur.
> Üç, emniyet sınırı: ACMG kanıt kodu atamıyoruz, VUS için sınıf iddia etmiyoruz.
> Bant genişlikleri şu an elimizde yok — pilotun ilk çıktısı tam bu tablo olacak.

---

## Slayt 18 (17) — Teşekkür · 0:12

> Biyoinformatik bilimiyle kliniğin pratiğini birleştiren dürüst bir köprü kurmaya çalıştık.
> Teşekkür ederiz; sorularınızı alabiliriz.

---

## Zaman aşımı planı

Slayt 10 bitiminde **5:40'ı geçtiysen**, şu sırayla kıs:

1. **Slayt 15** (rekabet/iş modeli) → 35 sn yerine 15 sn: *"Rekabette VarSome, Franklin, Emedgene var; hepsi bulut ve küratörlü veritabanı bağımlısı. Farkımız kapsam değil konuşlanma: dış çağrı yok, veri kurumdan çıkmıyor."* → geç.
2. **Slayt 8** (öznitelik seçimi) → 30 sn yerine 12 sn: *"442 aday öznitelik permütasyon bazlı bir şans testinden geçiyor; geçemeyen eleniyor. Panel başına 13–38 öznitelik kalıyor."*
3. **Slayt 6** (arama şeması) → 30 sn yerine 12 sn: *"Bu şema arama hattı: beş model ailesi ve stacking denendi, dört panelde de en yalın aday — tek XGBoost — kazandı."*

**Asla kısma:** slayt 4 (problem), slayt 10 (sonuçlar), slayt 13 (augmentation).
Bu üçü sunumun puan taşıyan omurgası.

---

## Sayı kartı (ezberlenecek)

| Konu | Sayı |
|---|---|
| Toplam varyant | **3.802** (GENEL 2.931 · KANSER 388 · PAH 372 · CFTR 111) |
| Kolon | 353 ham (351 öznitelik) → **442** temsil (91 türetilmiş) |
| Seçilen öznitelik | GENEL 38 · KANSER 35 · PAH 37 · CFTR 13 |
| Dürüst F1 (5×3 CV) | 0,876 · 0,924 · 0,938 · 0,952 → **ort. 0,92** |
| Dürüst ROC-AUC | 0,859 · 0,947 · 0,847 · 0,976 |
| MCC (naif = 0,00) | 0,56 · 0,79 · 0,58 · 0,77 → **ort. 0,68** |
| Dengeli doğruluk (naif = 0,50) | 0,79 · 0,89 · 0,72 · 0,90 |
| Naif taban F1 | 0,846 · 0,817 · 0,909 · 0,896 |
| Recall / Özgüllük | 0,86/0,71 · 0,94/0,85 · 0,99/0,45 · 0,94/0,86 |
| Karar eşiği | 0,670 · 0,705 · 0,595 · 0,780 |
| Eşik ölçütü | balanced_f1 · f1 · mcc · balanced_f1 |
| Artırım | PAH +2.676 · CFTR +2.854 (GENEL/KANSER: yok) |
| Ortak varyantta etiket uyumu | **578 varyant, %100** (GENEL∩PAH 255, GENEL∩CFTR 77) |
| CFTR artırımsız | AUC **0,50** → artırımla **0,976** |
| Çıkarım hızı | **6–58 µs/varyant**, paket ~1 MB, GPU yok |
| Seed | 42 (her yerde), model başına 3 seed ortalaması |
| Resmî skor tahmini | (ort. F1 0,92 + ort. MCC 0,68) / 2 ≈ **0,80** |

---

## Soru-Cevap bankası (3 dakika)

**S1 — "Cross-panel augmentation veri sızıntısı değil mi?"** *(en olası soru)*
> Hayır, çünkü ödünç aldığımız satırlar hedef panelde hiç bulunmuyor — filtre bunun üzerine
> kurulu. Bir satır PAH'ta varsa GENEL'den ödünç alınmaz. Doğrulama katları her zaman panelin
> kendi satırlarından oluşur; ödünç satırlar yalnızca eğitim katına girer ve bu her katta
> yeniden yapılır. Sızıntı, test edilecek bilginin eğitime sızmasıdır; burada test edilecek satır
> zaten havuzun dışında. Ayrıca etiket tutarlılığını da ölçtük: paneller arasında ortak olan
> 578 varyantta etiketler yüzde 100 uyuşuyor.

**S2 — "CFTR'de 111 gerçek satırın yanına 2.854 satır koydunuz. Bu hâlâ bir CFTR modeli mi?"**
> Haklısınız, saf bir CFTR modeli değil — bunu slaytta da risk olarak yazıyoruz. Bu bir
> GENEL→CFTR transfer modeli ve CFTR için tek çalışan seçenek: artırım kapalıyken AUC 0,50,
> yani 111 satırın 21'i benign olduğu için model ayrımı hiç öğrenemiyor. Değerlendirme yine de
> yalnızca CFTR'nin kendi satırlarında yapılıyor, dolayısıyla rapor ettiğimiz 0,952 CFTR
> performansıdır — ama arkasındaki bilginin çoğunun GENEL'den geldiğini açıkça söylüyoruz.

**S3 — "Eşiği eğitim verisinde ayarladınız; bu iyimserlik yaratmaz mı?"**
> Eşik iç katlarda seçiliyor, dış katta dondurulmuş olarak uygulanıyor — yani raporladığımız
> skoru üreten satırlar eşik seçimini hiç görmüyor. Ayrıca bir aşırı öğrenme denetimimiz var:
> iyimser skor ile dürüst skor arasındaki fark 0,04'ü aşan hiçbir aday seçilemiyor, aday dış
> katların en az yarısında kazanmak zorunda ve naif tabanı geçmek zorunda. Bıraktığımız tek
> iyimserlik payı, hiperparametrelerin her katta yeniden aranmaması — bunu sunumda da söylüyoruz.

**S4 — "Naif model zaten 0,90 alıyor; sizin katkınız ne?"**
> F1 dengesiz veride yanıltıcı, bu yüzden MCC'ye bakıyoruz. Naif modelin MCC'si tanım gereği
> sıfır — hiçbir ayrım yok. Bizim MCC'miz 0,56 ile 0,79 arasında, dengeli doğruluğumuz 0,50
> yerine 0,72–0,90. Resmî metrik de zaten F1 ile MCC'nin ortalaması; katkı tam olarak orada
> görünüyor.

**S5 — "Neden derin öğrenme kullanmadınız?"**
> Üç sebep. Bir, veri boyutu: en küçük panel 111, en büyüğü 2.931 satır — bu ölçekte
> gradyan artırmalı ağaçlar tablo verisinde tutarlı biçimde daha iyi. İki, veri tablo ve
> yüzde 55'i eksik; ağaçlar eksikliği doğal olarak işliyor. Üç, konuşlanma: 1 megabayt,
> tek CPU, GPU yok — hastane sunucusunda çalışması gereken bir araç için bu bir tasarım kararı.

**S6 — "Neden tek model? Ensemble/stacking denediniz mi?"**
> Denedik. Arama hattı beş model ailesini ve izotonik kalibrasyonlu stack varyantlarını
> deniyor. One-SE kuralıyla — istatistiksel olarak ayırt edilemeyen adaylar arasında en yalını
> seç — dört panelde de tek XGBoost kazandı. Ayrıca daha geniş profilli, 39 saat süren bir
> yeniden taramayla bunu doğruladık; sonuç değişmedi.

**S7 — "En önemli özniteliğiniz 'eksiklik' — bu bir artefakt değil mi?"**
> Olabilir ve bunu biz de risk olarak işaretliyoruz, slaytta yazılı. İki gözlem var: eksiklik
> imzası yalnızca GENEL ve KANSER'de öne çıkıyor, PAH ve CFTR'de çıkmıyor. Ve eksiklik gerçekte
> anotasyon tamlığını ölçüyor — klinikte de iyi çalışılmış varyantlar daha çok anote edilir,
> yani sinyal olma ihtimali gerçek. Ayrımı yapmak için eksiklik grubunu tamamen çıkarıp F1'i
> yeniden ölçeceğiz; bu bir sonraki adımımız.

**S8 — "Test setinin sınıf dağılımı eğitimden farklıysa ne olur?"**
> Buna hazırız. Her panel için dışarıda bırakılmış tahminler üzerinde izotonik kalibratör
> saklıyoruz. Test dağılımı farklıysa eşiği Bayes prior-shift formülüyle yeni dağılıma taşıyoruz
> — yeniden eğitim gerekmiyor, tek parametre. Bu, teslim komutumuzda hazır bir seçenek.

**S9 — "SMOTE gibi bir örnekleme yöntemi neden kullanmadınız?"**
> 442 boyutlu ve yüzde 55'i eksik bir uzayda SMOTE'un üreteceği ara profiller biyolojik olarak
> var olamaz — komşular arası enterpolasyon burada anlamlı bir varyant üretmez. Biz sentetik
> satır yerine gerçek, etiketli, aynı şemadan gelen satır ödünç aldık.

**S10 — "Sonuçlarınız tekrar üretilebilir mi?"**
> Evet. Seed her yerde 42. Kazanan tarif koda gömülü değil, ayrı bir yapılandırma dosyasında —
> gizli elle yazılmış sayı yok. Her eğitim yeni numaralı bir sürüm yazıyor, öncekiler silinmiyor.
> Teslim klasörü kodu, modeli, dondurulmuş bağımlılık listesini ve README'yi birlikte
> paketliyor; sunumdaki her sayı üretim modelinin kendisinden ölçüldü.

**S11 — "PAH'ta özgüllük 0,45 — bu çok düşük değil mi?"**
> Doğru ve bilinçli. PAH'ta eşik MCC ölçütüyle seçildi ve sonuç recall 0,99'a çıkan bir nokta.
> Panelde negatif sınıf yalnızca 62 satır; orada kaçan tanı ile gereksiz test arasındaki maliyet
> dengesini kaçan tanı lehine kurduk. Klinik üründe bu zaten üç bantlı triyajla yönetiliyor:
> belirsiz aralık otomatik karara bağlanmadan uzmana gidiyor.

**S12 — "Panelleri neden tek modelde birleştirmediniz?"**
> Değerlendirme panel bazlı ve klinik de panel bazlı çalışıyor — laboratuvar CFTR paneli
> koşuyorsa CFTR kararı istiyor. Ayrıca paneller arasında sinyal farkı gerçek: KANSER'de
> GENEL'den satır ödünç almak performansı düşürdü, PAH ve CFTR'de yükseltti. Yani birleştirme
> tek yönlü bir kazanç değil; biz onu panel başına ölçüp kararı ölçüme bıraktık.

**S13 — "Model kararını nasıl açıklıyorsunuz?"**
> İki katman var. Model seviyesinde, üç seed ortalaması üzerinden kazanç bazlı öznitelik önemi —
> slayt 10'daki listeler. Ürün seviyesinde ise çıktıya en katkılı üç öznitelik grubunu ekliyoruz:
> allel frekansı, korunmuşluk, biyokimyasal değişim. Bilinçli olarak yapmadığımız şey ACMG kanıt
> kodu atamak — o klinik sorumluluk uzmanda kalmalı.

**S14 — "Ürünleşme için regülasyon tarafı ne durumda?"**
> Tanı amaçlı yazılım tıbbi cihaz kapsamında: TİTCK ve IVDR/CE-IVD. Bu, kalite sistemi ve denetim
> izi demek. Yol haritamızda regülasyon 12–30. ay aralığında ve ondan önce ileriye dönük
> doğrulama var. Bugün elimizdeki şey kavram kanıtı; bunu klinik kanıt gibi sunmuyoruz.

**S15 — "Yarışma günü sistemi nasıl çalıştırdınız?"**
> Tek komutla. Model yarışmadan önce eğitilip sürümlendi; yarışma gününde çalışan komut test
> CSV'lerini okuyup tahmin üretiyor, sonuç JSON'unu resmî şemaya karşı doğruluyor ve kod, model,
> bağımlılıklar ve README'den oluşan teslim klasörünü numaralandırarak paketliyor. Arama hiç
> devreye girmiyor; yarışma günü yeni bir karar alınmıyor.

---

## Sunum öncesi kontrol listesi

- [ ] Slayt 2'deki "AD SOYAD" / "Bölüm / Üniversite" alanları dolduruldu.
- [ ] Slayt 4'te sözlü metrik ifadesi: **"F1 ve MCC ortalaması"** (slaytta "binary F1" yazıyor).
- [ ] Slayt 6'da "**teslim edilen sistemde stack yok, tek XGBoost**" cümlesi söylendi.
- [ ] Slayt 8'de aday havuzu sözlü olarak **442** (slayt tablosunda 441 yazıyor).
- [ ] Slayt 10'da iyimserlik payı ve CFTR'nin oynaklığı **kendimiz** söylendi.
- [ ] Slayt 14'te CFTR'nin **%96 ödünç** olduğu **kendimiz** söylendi.
- [ ] Prova ile toplam süre ≤ 9:55; slayt 10 bitişi ≤ 5:40.
