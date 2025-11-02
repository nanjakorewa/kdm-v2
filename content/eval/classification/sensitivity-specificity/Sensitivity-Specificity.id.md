---
title: "Sensitivitas dan spesifisitas: menyeimbangkan false negative dan false positive"
linkTitle: "Sensitivitas / Spesifisitas"
seo_title: "Sensitivitas vs. spesifisitas | Menyeimbangkan false negative dan false positive"
title_suffix: "Menyeimbangkan false negative dan false positive"
pre: "4.3.7 "
weight: 7
---

{{< lead >}}
Sensitivitas mengukur kemampuan model menangkap kelas positif, sedangkan spesifisitas mengukur kemampuannya menolak alarm palsu pada kelas negatif. Keduanya penting ketika biaya false negative dan false positive sangat berbeda.
{{< /lead >}}

---

## 1. Definisi
Dari matriks kebingungan kita peroleh:

$$
\mathrm{Sensitivity} = \frac{TP}{TP + FN}, \qquad
\mathrm{Specificity} = \frac{TN}{TN + FP}
$$

- Sensitivitas tinggi berarti sedikit kasus positif yang terlewat.  
- Spesifisitas tinggi berarti sedikit kasus negatif yang salah diprediksi sebagai positif.

---

## 2. Perhitungan di Python 3.13
```bash
python --version  # contoh: Python 3.13.0
pip install numpy scikit-learn
```

```python
import numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)  # [[TN, FP], [FN, TP]]
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Sensitivitas:", round(sensitivity, 3))
print("Spesifisitas:", round(specificity, 3))
```

Dalam scikit-learn, sensitivitas adalah `recall`. Spesifisitas dapat dihitung dengan `recall_score(y_true, y_pred, pos_label=0)` atau secara manual seperti contoh di atas.

---

## 3. Trade-off lewat penyesuaian ambang
Pada model yang menghasilkan probabilitas, mengubah ambang keputusan akan memengaruhi sensitivitas dan spesifisitas.

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, probas)
specificities = 1 - fpr
```

- Setiap titik pada kurva ROC mewakili pasangan sensitivitas/spesifisitas tertentu.
- Jika biaya kesalahan diketahui, optimalkan ambang menggunakan indeks Youden atau fungsi biaya yang relevan.

---

## 4. Indeks Youden untuk kompromi seimbang
Indeks Youden mendefinisikan kompromi sederhana antara keduanya:

$$
J = \mathrm{Sensitivity} + \mathrm{Specificity} - 1
$$

Ambang yang memaksimalkan \\(J\\) biasanya memberikan keseimbangan yang baik ketika kedua jenis kesalahan penting.

---

## 5. Panduan praktis
- **Fokus sensitivitas**: Skrining penyakit serius memprioritaskan sensitivitas agar kasus positif tidak terlewat.
- **Fokus spesifisitas**: Pada deteksi penipuan, spesifisitas tinggi mencegah transaksi sah diblokir secara keliru.
- **Pelaporan metrik**: Sajikan sensitivitas dan spesifisitas berdampingan dengan Accuracy supaya pemangku kepentingan bisa menilai risiko secara eksplisit.

---

## Ringkasan
- Sensitivitas memantau kasus positif yang terlewat, sedangkan spesifisitas memantau alarm palsu pada negatif.
- Mengatur ambang keputusan mengubah keseimbangan keduanya—sesuaikan dengan biaya bisnis.
- Gunakan indeks Youden dan metrik pelengkap untuk mengungkap risiko yang tidak terlihat hanya dengan Accuracy.
