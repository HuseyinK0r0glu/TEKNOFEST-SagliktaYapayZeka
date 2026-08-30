# Rapor sayılarının üretildiği analiz betikleri

PSR'deki (PatHUgen_Proje_Sunus_Raporu.docx) her sayı bu betiklerden veya
`outputs_colab.zip` / `report_figs/` içindeki koşu çıktılarından gelir.
Hepsi proje kökünden çalıştırılır ve `run_final.py`'yi modül olarak yükler.

| Betik | Rapordaki karşılığı |
|---|---|
| `data_quality.py` | 3.4 — yinelenen kayıt, çelişkili etiket, panel örtüşmesi, aralık taraması |
| `error_analysis.py` | 4.3, Tablo 4 — TN/FP/TP medyan profilleri, eşiğe uzaklık |
| `missingness_probe.py` | 4.3 — pct_missing ve araç sayısının tek başına ROC-AUC'si |
| `shap_groups.py` | 4.4, Tablo 5 — grup bazlı SHAP katkıları ve VAR_003700 örneği |
| `latency.py` | 5.4 — toplu ve tek satır çıkarım süreleri |
| `verify_claims.py` | 5.2 — paylaşılan öznitelik sayısı, ortalama eksik değer |

`shap_groups.py` ve `latency.py`, `outputs_colab.zip` içindeki model paketlerinin
açılmış olmasını bekler; betiğin başındaki `BUNDLE` yolunu güncelleyin.
