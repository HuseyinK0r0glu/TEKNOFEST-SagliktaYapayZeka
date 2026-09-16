# PatHUgen — A0 afiş kaynağı

`../PatHUgen_Afis_A0.pdf` dosyasını üreten kaynak. Format: **A0 dikey, 841 × 1189 mm**.

## Yeniden üretmek

```bash
google-chrome --headless --disable-gpu --no-sandbox \
  --no-pdf-header-footer --virtual-time-budget=20000 \
  --print-to-pdf=PatHUgen_Afis_A0.pdf poster.html
```

Sayfa boyutu `poster.css` içindeki `@page { size: 841mm 1189mm; margin: 0 }` kuralından gelir.

## Baskı çözünürlüğü

Afişteki metin, tablolar, grafikler ve çizgilerin tamamı **vektör**dür; çözünürlük
kaybı yaşamazlar. Yalnızca fotoğraf/logo varlıkları raster:

| Varlık                  | PDF içindeki çözünürlük | Not                                 |
|-------------------------|-------------------------|-------------------------------------|
| `nebula-300ppi.jpg`     | 300 ppi                 | masthead fotoğrafı, tam sayfa eni    |
| `teknofest.png`         | 430 ppi                 |                                      |
| `t3.png` · `tuseb*.png` | 503–970 ppi             |                                      |
| `mth.png`               | 573 ppi                 |                                      |
| `grain.png`             | 191 ppi                 | kasıtlı: gren dokusu, döşeme deseni  |
| gölge maskeleri         | 300 ppi                 | Chrome'un blur rasterleştirme sınırı |

Kontrol: `pdfimages -list ../PatHUgen_Afis_A0.pdf` — `x-ppi` sütunu 300'ün altına
düşmemeli (gren dokusu hariç; o `pdfimages` listesinde döşeme deseni olduğu için
görünmez).

### Varlık üretimi

- `tools/upscale_nebula.py` — `assets/nebula.jpg` (4792 px = 145 ppi) dosyasını
  Lanczos + unsharp ile `assets/nebula-300ppi.jpg` (9933 px = 300 ppi) üretir.
  `assets/nebula.jpg` dokunulmamış kaynaktır, silmeyin.
- `tools/otf2ttf.py` — Linux Libertine O'nun sistemdeki `.otf` (CFF) sürümünü
  TrueType'a çevirip `assets/fonts/` altına yazar. Sebebi: Chrome CFF fontları
  PDF'e **Type 3** olarak gömüyor, bazı baskı RIP'leri Type 3'ü düşük
  çözünürlükte tarıyor. TTF sürümü normal CID TrueType olarak gömülüyor.
  Kontrol: `pdffonts ../PatHUgen_Afis_A0.pdf` çıktısında `Type 3` olmamalı.

## Künye

Afişteki isimler `sunum/PatHUgen.pptx` slayt 2'den (Takım Tanıtımı) alındı:
kaptan Hasan Can Zorlu; üyeler Mehmed Akif Arslan, Hüseyin Köroğlu, Fikret Karakuzu,
Efe Gürdamar; danışman Assoc. Prof. Hacer Yalım Keleş; Bilgisayar Mühendisliği,
Hacettepe Üniversitesi. Değişirse `poster.html` içindeki `<p class="byline">` bloğunu düzenle.

## Notlar

- Yükseklik bütçesi sıkı: içerik 1.177 mm, sayfa 1.189 mm. Metin eklerseniz
  afiş iki sayfaya taşabilir — `pdfinfo` ile `Pages: 1` olduğunu doğrulayın.
- Fontlar (Lato, Linux Libertine O, Aptos Mono) PDF'e gömülür; matbaa ayrıca font istemez.
  Birkaç özel karakter (→, ∩, µ, üst simgeler) Lato/Aptos Mono'da olmadığı için
  Liberation Sans ve DejaVu Sans Mono'ya düşüyor — bunlar da gömülü ve vektör.
- Renkler RGB'dir. Matbaa CMYK isterse dönüşümü baskı öncesi onlar yapsın —
  lacivert zemin CMYK'de kasıtlı olarak güvenli aralıkta seçildi.
