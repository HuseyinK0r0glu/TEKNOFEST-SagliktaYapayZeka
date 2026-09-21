# PatHUgen

### TEKNOFEST 2026 Sağlıkta Yapay Zeka — Proje Genel Bakışı

*Bir genetik varyant zararlı mı, yoksa masum mu? PatHUgen, dört farklı hastalık panelinde bu soruyu yanıtlamak için eğitildi — ve her yanıtın ne kadar güvenilir olduğunu da ölçtü.*

---

## Problem

Bir genom dizileme raporunda onlarca, bazen yüzlerce varyant çıkar; bunlardan hangisinin hastalığa yol açtığını, hangisinin zararsız bir polimorfizm olduğunu ayırmak klinik genetikte gerçek ve zaman alan bir darboğazdır. PatHUgen, bu ayrımı dört **birbirinden bağımsız** panelde otomatikleştiriyor:

| Panel | Ne olduğu | Örnek sayısı |
|---|---|---|
| **GENEL** | Karma / genel varyant havuzu | 2.931 |
| **KANSER** | Kansere yatkınlık genleri | 388 |
| **PAH** | Pulmoner arteriyel hipertansiyon | 372 |
| **CFTR** | Kistik fibrozis geni | 111 |

Panellerin verisi anonimleştirilmiş sütun adlarıyla geldi (`AL_*`, `EK_*`, `CAT_*`) ve büyüklükleri 30 kata varan bir farkla değişiyor — CFTR'de 111 satırla genellenebilir bir model kurmak, GENEL'deki 2.931 satırdan çok daha zor bir problem. Bu yüzden tek bir "evrensel" model yerine, her panel kendi verisiyle, kendi öznitelik setiyle ve kendi karar eşiğiyle ayrı ayrı eğitildi.

---

## Sonuçlar

| Panel | Öznitelik | **Dürüst F1** | Dürüst AUC | Naif taban | Kazanç |
|---|---|---|---|---|---|
| GENEL | 38 | **0.876** ± 0.008 | 0.859 | 0.846 | +0.030 |
| KANSER | 35 | **0.924** ± 0.030 | 0.947 | 0.817 | +0.107 |
| PAH | 37 | **0.938** ± 0.016 | 0.847 | 0.909 | +0.029 |
| CFTR | 13 | **0.952** ± 0.029 | 0.976 | 0.896 | +0.057 |

**"Naif taban"**, modelin hiçbir şey öğrenmeden sadece "her varyant patojeniktir" dese elde edeceği F1 skorudur — panellerin çoğunda pozitif sınıf zaten baskın olduğu için bu taban şaşırtıcı derecede yüksektir. **"Kazanç"** sütunu, modelin bu naif tahminin üzerine gerçekten ne kattığını gösterir; KANSER panelinde bu fark en belirgin olanı.

---

## Bu sayılara neden güvenebilirsiniz

Bir modelin skorunu yükseltmenin en kolay yolu, onu ölçme şeklini gevşetmektir. Bunu yapmadık:

- **Dürüst (nested) çapraz doğrulama.** Yukarıdaki her F1 değeri, öznitelik seçimi, model seçimi ve karar eşiğinin **her dış kat için sıfırdan yeniden belirlendiği** iç içe çapraz doğrulamadan geliyor — modelin daha önce hiç görmediği satırlar üzerinde, tek seferlik bir ölçüm. Bu, çoğu hızlı prototipte atlanan ama en çok fark yaratan adım.
- **Aşırı uyum denetimi.** Her aday yapılandırma otomatik olarak üç testten geçirildi: iyimserlik sapması yeterince küçük mü, kazanan model dış katlar arasında tutarlı mı, ve naif tabanı gerçekten geçiyor mu? Bu testlerden birini geçemeyen hiçbir yapılandırma şampiyon olarak seçilmedi.
- **Sıfırdan, ipuçsuz arama.** Panel başına en iyi öznitelik seti, hiperparametreler ve eşik; önbelleğe alınmış veya bir önceki koşudan miras alınmış hiçbir değer kullanılmadan, otomatik bir aramayla bulundu. Bu arama sürecinin kendisi de teslimatın içinde — jüri isterse baştan çalıştırıp aynı sonuca ulaşıldığını doğrulayabilir.

---

## Teslimat ne içeriyor

- **Tek komutla çalışan, kendi kendine yeten bir teslim sistemi** (`PatHUgen.py train` / `deliver`) — yalnızca eğitilmiş model ağırlıklarını ve kilitlenmiş yapılandırmayı kullanır, dışarıdan hiçbir bağımlılığa ihtiyaç duymaz.
- **Şeffaf yapılandırma.** Her panelin öznitelik listesi, hiperparametreleri ve eşiği kodun içine gömülü bir "sihirli sayı" değil, `champion_config.json` içinde okunabilir veri olarak duruyor.
- **Uçtan uca yeniden üretilebilirlik.** Aynı komut, aynı eğitim verisiyle çalıştırıldığında bit-bit aynı tahminleri üretir; arama sürecinden nihai teslimata kadar her adım kayıt altında.

---

**Takım PatHUgen**
