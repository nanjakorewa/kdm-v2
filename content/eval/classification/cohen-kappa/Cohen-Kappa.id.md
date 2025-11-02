---
title: "Kappa Cohen: mengukur kesepakatan di luar kebetulan"
linkTitle: "Kappa Cohen"
seo_title: "Kappa Cohen | Mengukur kesepakatan di luar kebetulan"
title_suffix: "Mengukur kesepakatan di luar kebetulan"
pre: "4.3.11 "
weight: 11
---

{{< lead >}}
Koefisien Kappa Cohen mengurangi porsi kesepakatan yang terjadi karena kebetulan. Metode ini umum dipakai untuk mengevaluasi kualitas anotasi dan kinerja model klasifikasi pada data yang tidak seimbang.
{{< /lead >}}

---

## 1. Definisi
Misalkan \\(p_o\\) adalah tingkat kesepakatan yang diamati dan \\(p_e\\) adalah kesepakatan yang diharapkan terjadi secara acak. Maka

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- \\(\kappa = 1\\): kesepakatan sempurna  
- \\(\kappa = 0\\): setara dengan kebetulan  
- \\(\kappa < 0\\): lebih buruk daripada kebetulan

---

## 2. Perhitungan di Python 3.13
```bash
python --version  # contoh: Python 3.13.0
pip install scikit-learn
```

```python
from sklearn.metrics import cohen_kappa_score, confusion_matrix

print("Kappa Cohen:", cohen_kappa_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

Fungsi ini mendukung klasifikasi multikelas. Gunakan `weights="quadratic"` untuk menghitung Kappa berbobot pada label ordinal.

---

## 3. Panduan interpretasi
Landis & Koch (1977) memberikan panduan berikut. Sesuaikan ambang batas tersebut dengan standar yang berlaku di domain Anda.

| κ       | Interpretasi          |
| ------- | --------------------- |
| < 0     | Hampir tidak ada kesepakatan |
| 0.0–0.2 | Kesepakatan lemah     |
| 0.2–0.4 | Kesepakatan wajar     |
| 0.4–0.6 | Kesepakatan sedang    |
| 0.6–0.8 | Kesepakatan kuat      |
| 0.8–1.0 | Hampir sempurna       |

---

## 4. Kelebihan untuk evaluasi model
- **Tahan terhadap ketidakseimbangan**: Model yang hanya menebak kelas mayoritas akan memperoleh κ rendah sehingga Accuracy tidak menipu.
- **Audit kualitas anotasi**: Bandingkan prediksi model dengan label manusia atau antar anotator secara objektif.
- **Kappa berbobot**: Pada label ordinal (misalnya penilaian 5 tingkat) kita dapat memberi bobot sesuai jauhnya kesalahan prediksi.

---

## 5. Tips praktis
- Accuracy tinggi tetapi κ rendah menandakan model bergantung pada kebetulan. Periksa matriks kebingungan untuk memahami kesalahan.
- Beberapa industri yang diawasi regulator mewajibkan pelaporan κ; dokumentasikan pipeline perhitungannya.
- Gunakan κ ketika mengaudit data latih untuk menemukan anotator atau subset yang konsisten rendah.

---

## Ringkasan
- Kappa Cohen mengurangi kesepakatan kebetulan sehingga cocok untuk data tidak seimbang dan evaluasi anotasi.
- `cohen_kappa_score` di scikit-learn mudah digunakan, termasuk versi berbobot.
- Gabungkan κ dengan Accuracy, F1, dan metrik lainnya agar penilaian kinerja model dan kualitas label lebih menyeluruh.
