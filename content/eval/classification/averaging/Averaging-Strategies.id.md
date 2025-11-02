---
title: "Memilih strategi averaging untuk metrik klasifikasi"
linkTitle: "Strategi averaging"
seo_title: "Strategi averaging | Evaluasi multikelas dan multi-label"
pre: "4.3.14 "
weight: 14
---

{{< lead >}}
Saat menghitung Precision, Recall, F1, dan metrik lain pada tugas multikelas atau multi-label, argumen `average` di scikit-learn menentukan cara skor digabungkan. Halaman ini membandingkan empat strategi yang paling umum dengan contoh Python 3.13.
{{< /lead >}}

---

## 1. Pilihan averaging utama
| average    | Cara perhitungan                                             | Kapan digunakan                                                   |
| ---------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| `micro`    | Menjumlahkan TP/FP/FN seluruh sampel lalu menghitung metrik | Menekankan akurasi total tanpa memedulikan distribusi kelas       |
| `macro`    | Menghitung metrik per kelas lalu mengambil rata-rata sederhana | Memberi bobot sama untuk setiap kelas; menyorot kelas minoritas   |
| `weighted` | Menghitung metrik per kelas lalu merata-ratakan dengan bobot jumlah sampel | Mempertahankan proporsi kelas asli; perilakunya mirip Accuracy |
| `samples`  | Khusus multi-label. Merata-ratakan metrik per sampel        | Untuk kasus ketika satu sampel memiliki banyak label              |

---

## 2. Perbandingan di Python 3.13
```bash
python --version  # contoh: Python 3.13.0
pip install scikit-learn matplotlib
```

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=30_000,
    n_features=20,
    n_informative=6,
    weights=[0.85, 0.1, 0.05],  # Distribusi kelas tidak seimbang
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, multi_class="ovr"),
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
for avg in ["micro", "macro", "weighted"]:
    print(f"F1 ({avg}):", f1_score(y_test, y_pred, average=avg))
```

`classification_report` menampilkan metrik per kelas sekaligus `macro avg`, `weighted avg`, dan `micro avg`, sehingga kita dapat membandingkan strategi averaging dengan cepat.

---

## 3. Cara memilih strategi
- **micro** – Gunakan bila Anda mengutamakan performa keseluruhan dan setiap prediksi dianggap sama penting.
- **macro** – Cocok saat kelas minoritas sangat penting; sensitif terhadap recall yang rendah pada label langka.
- **weighted** – Berguna jika ingin mempertahankan proporsi kelas nyata sekaligus melaporkan Precision/Recall/F1.
- **samples** – Pilihan standar untuk tugas multi-label di mana satu sampel dapat memiliki beberapa label ground truth.

---

## Catatan akhir
- Parameter `average` dapat mengubah interpretasi metrik secara drastis; sesuaikan dengan kebutuhan bisnis dan sifat data.
- Ingat: `macro` memandang kelas secara adil, `micro` fokus pada rasio global, `weighted` mempertahankan distribusi, dan `samples` dirancang untuk multi-label.
- Manfaatkan kemampuan scikit-learn untuk menghitung beberapa average sekaligus agar tidak salah menilai kualitas model.
