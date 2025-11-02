---
title: "Hamming Loss: tingkat kesalahan per label"
linkTitle: "Hamming Loss"
seo_title: "Hamming Loss | Mengukur kesalahan per label pada tugas multi-label"
title_suffix: "Mengukur kesalahan per label pada tugas multi-label"
pre: "4.3.13 "
weight: 13
---

{{< lead >}}
Hamming Loss menghitung proporsi label yang diprediksi keliru. Metrik ini sangat berguna pada klasifikasi multi-label karena menunjukkan rata-rata jumlah label yang salah per sampel.
{{< /lead >}}

---

## 1. Definisi
Misalkan \\(Y_i\\) adalah himpunan label benar, \\(\hat{Y}_i\\) himpunan label prediksi, dan \\(L\\) jumlah label keseluruhan:

$$
\mathrm{Hamming\ Loss} = \frac{1}{nL} \sum_{i=1}^n \lvert Y_i \triangle \hat{Y}_i \rvert
$$

Di sini \\(Y \triangle \hat{Y}\\) merupakan beda simetris (label yang hanya muncul di salah satu himpunan). Untuk tugas multi-label, nilai ini setara dengan rata-rata jumlah label yang salah per sampel.

---

## 2. Perhitungan di Python 3.13
```bash
python --version  # contoh: Python 3.13.0
pip install scikit-learn
```

```python
from sklearn.metrics import hamming_loss

print("Hamming Loss:", hamming_loss(y_true, y_pred))
```

`y_true` dan `y_pred` berupa matriks indikator 0/1 untuk setiap label (output `MultiLabelBinarizer` sangat berguna untuk ini).

---

## 3. Cara membaca nilainya
- Semakin mendekati 0 semakin baik; model sempurna menghasilkan 0.
- Nilai 0.05 berarti “rata-rata 5% label diprediksi salah”.
- Bila tiap label punya dampak bisnis berbeda, gunakan Hamming Loss berbobot.

---

## 4. Perbandingan dengan metrik lain
| Metrik          | Apa yang diukur              | Kapan digunakan                                 |
| --------------- | --------------------------- | ------------------------------------------------ |
| Exact Match     | Akurasi sempurna per sampel | Memerlukan kesesuaian label secara keseluruhan  |
| **Hamming Loss**| Tingkat kesalahan per label | Memantau jumlah rata-rata kesalahan             |
| Micro F1        | Keseimbangan precision/recall | Memperhitungkan ketidakseimbangan positif/negatif |
| Indeks Jaccard  | Tumpang tindih himpunan      | Menilai kemiripan himpunan label                |

Hamming Loss tidak setegas Exact Match sehingga memberi sinyal yang lebih halus saat memperbaiki model.

---

## 5. Tips praktis
- **Rekomendasi tag**: ketahui berapa banyak tag yang salah per item.
- **Sistem peringatan**: ukur seberapa sering alarm multi-label memberi sinyal keliru.
- **Evaluasi berbobot**: terapkan bobot per label jika biaya kesalahan berbeda-beda.

---

## Ringkasan
- Hamming Loss menangkap tingkat kesalahan per label dan ideal untuk memantau peningkatan pada klasifikasi multi-label.
- Fungsi `hamming_loss` di scikit-learn mudah dipakai serta melengkapi Exact Match dan F1.
- Gabungkan metrik ini dengan analisis per label untuk memprioritaskan perbaikan di area yang paling merugikan.
