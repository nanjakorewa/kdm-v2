---
title: "Top-k Accuracy dan Recall@k"
linkTitle: "Top-k Accuracy"
seo_title: "Top-k Accuracy & Recall@k | Mengevaluasi daftar kandidat"
title_suffix: "Menilai apakah label benar muncul di antara kandidat teratas"
pre: "4.3.12 "
weight: 12
---

{{< lead >}}
Top-k Accuracy (atau Recall@k) menghitung seberapa sering label benar muncul di antara k kandidat teratas yang diberikan model. Metrik ini sangat berguna pada klasifikasi multikelas dan sistem rekomendasi ketika yang penting adalah memasukkan jawaban benar ke dalam daftar pendek.
{{< /lead >}}

---

## 1. Definisi
Jika model menghasilkan skor per kelas dan \\(S_k(x)\\) adalah himpunan k kandidat teratas untuk masukan \\(x\\), maka

$$
\mathrm{Top\text{-}k\ Accuracy} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{ y_i \in S_k(x_i) \}
$$

Metrik ini identik dengan `Recall@k`: kita mendapatkan “hit” apabila label sebenarnya termasuk dalam k prediksi teratas.

---

## 2. Perhitungan di Python 3.13
```bash
python --version  # contoh: Python 3.13.0
pip install scikit-learn
```

```python
import numpy as np
from sklearn.metrics import top_k_accuracy_score

proba = model.predict_proba(X_test)  # shape: (n_samples, n_classes)
top3 = top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)
top5 = top_k_accuracy_score(y_test, proba, k=5, labels=model.classes_)

print("Top-3 Accuracy:", round(top3, 3))
print("Top-5 Accuracy:", round(top5, 3))
```

`top_k_accuracy_score` menerima probabilitas kelas dan mengembalikan metrik untuk nilai `k` yang kita pilih. Sertakan `labels` agar evaluasi tetap benar meski urutan kelas berbeda.

---

## 3. Hal yang perlu diperhatikan
- **Menentukan k**: Sesuaikan dengan jumlah kandidat yang bisa ditampilkan UI atau alur produk.
- **Mengatasi skor seri**: Jika banyak kandidat memiliki skor sama, ranking dapat tidak stabil; gunakan pengurutan stabil atau tambahkan tie-breaker.
- **Banyak label benar**: Pada multi-label asli, lengkapi evaluasi dengan Precision@k dan Recall@k.

---

## 4. Contoh penggunaan
- **Sistem rekomendasi**: Pastikan item yang akhirnya dipilih pengguna muncul dalam daftar rekomendasi.
- **Klasifikasi gambar**: Dataset besar seperti ImageNet lazim melaporkan Top-5 Accuracy.
- **Pencarian informasi**: Nilai apakah dokumen relevan muncul dalam 10 hasil teratas.

---

## 5. Perbandingan dengan metrik ranking lain
| Metrik           | Apa yang diukur                       | Kapan digunakan                                  |
| ---------------- | ------------------------------------- | ------------------------------------------------ |
| Top-k Accuracy   | Apakah label benar ada di k teratas   | Daftar kandidat di mana satu hit sudah cukup     |
| NDCG             | Relevansi dengan bobot posisi         | Saat kualitas urutan juga penting                |
| MAP              | Rata-rata precision@k                 | Ranking dengan banyak item relevan               |
| Hit Rate         | Sinonim Top-k Accuracy                | Umum di literatur sistem rekomendasi             |

---

## Ringkasan
- Top-k Accuracy memeriksa apakah jawaban benar muncul di antara kandidat—sinyal sederhana namun efektif.
- `top_k_accuracy_score` dari scikit-learn memudahkan evaluasi untuk berbagai nilai k.
- Gunakan bersama NDCG atau MAP untuk mempertimbangkan kualitas urutan dan situasi dengan banyak jawaban relevan.
