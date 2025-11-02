---

title: "Accuracy | Dasar dan jebakan dalam Python 3.13"

linkTitle: "Accuracy"

seo_title: "Accuracy | Dasar dan jebakan dalam Python 3.13"

pre: "4.3.1 "

weight: 1

---



{{< lead >}}

Accuracy memberi tahu berapa banyak sampel yang diprediksi dengan benar oleh model, namun tidak membedakan kesalahan pada kelas minoritas. Panduan ini menunjukkan cara menghitung dan memvisualisasikannya di Python 3.13 serta metrik pendukung yang sebaiknya Anda laporkan.

{{< /lead >}}



---



## 1. Definisi



Dengan menggunakan elemen matriks kebingungan (true positive TP, false positive FP, false negative FN, true negative TN), accuracy didefinisikan sebagai:



$$

\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

$$



Metrik ini menggambarkan rasio prediksi benar secara keseluruhan. Namun ketika data tidak seimbang, nilai ini bisa menyesatkan karena tidak menyoroti kinerja pada kelas minoritas.



---



## 2. Implementasi dan visualisasi di Python 3.13



Periksa versi Python dan instal dependensi:



```bash

python --version        # contoh: Python 3.13.0

pip install scikit-learn matplotlib

```



Script berikut melatih Random Forest pada dataset Breast Cancer, menghitung Accuracy dan Balanced Accuracy, lalu menampilkannya dalam diagram batang. Penggunaan `Pipeline` + `StandardScaler` memastikan proses yang stabil dan mudah diulang. Gambar disimpan di `static/images/eval/...` sehingga dapat digenerasi ulang dengan `generate_eval_assets.py`.



```python

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=42, stratify=y

)



pipeline = make_pipeline(

    StandardScaler(),

    RandomForestClassifier(random_state=42, n_estimators=300),

)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)



acc = accuracy_score(y_test, y_pred)

bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f}, Balanced Accuracy: {bal_acc:.3f}")



fig, ax = plt.subplots(figsize=(5, 4))

scores = np.array([acc, bal_acc])

labels = ["Accuracy", "Balanced Accuracy"]

colors = ["#2563eb", "#f97316"]

bars = ax.bar(labels, scores, color=colors)

ax.set_ylim(0, 1.05)

for bar, score in zip(bars, scores):

    ax.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.3f}", ha="center", va="bottom")

ax.set_ylabel("Skor")

ax.set_title("Accuracy vs. Balanced Accuracy (Breast Cancer Dataset)")

ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

output_dir = Path("static/images/eval/classification/accuracy")

output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "accuracy_vs_balanced.png", dpi=150)

plt.close(fig)

```



{{< figure src="/images/eval/classification/accuracy/accuracy_vs_balanced.png" alt="Perbandingan Accuracy dan Balanced Accuracy" caption="Balanced Accuracy mengungkap kesalahan yang tersembunyi ketika data tidak seimbang." >}}



---



## 3. Menangani ketidakseimbangan kelas



Accuracy tidak membedakan biaya false negative dan false positive. Untuk gambaran yang lebih lengkap:



- **Precision / Recall / F1** – mengukur tingkat false alarm dan missed detection.

- **Balanced Accuracy** – rata-rata recall per kelas sehingga kelas minoritas tetap terlihat.

- **Confusion Matrix** – menunjukkan kelas mana yang paling sering salah.

- **Kurva ROC-AUC / PR** – mengevaluasi perubahan skor saat mengubah ambang keputusan.



Balanced Accuracy setara dengan rerata recall antar kelas, dan sering digunakan sebagai metrik utama ketika fokus pada deteksi kelas minoritas.



---



## 4. Checklist praktis



1. **Selaras dengan biaya bisnis** – pastikan accuracy tinggi tidak menyembunyikan kegagalan pada kasus kritis.

2. **Evaluasi ambang keputusan** – gunakan kurva ROC-AUC atau PR untuk melihat dampak penyesuaian threshold terhadap accuracy.

3. **Laporkan metrik majemuk** – tampilkan Precision, Recall, F1, dan Balanced Accuracy bersama Accuracy agar pemangku kepentingan paham trade-off yang terjadi.

4. **Notebook yang dapat diulang** – simpan evaluasi dalam notebook Python 3.13 supaya mudah dijalankan ulang ketika model diperbarui.



---



## Ringkasan



- Accuracy adalah indikator cepat untuk kinerja keseluruhan, tetapi bisa menipu pada data tidak seimbang.

- Gunakan pipeline scikit-learn dengan penskalaan agar perhitungan stabil dan mudah direproduksi di Python 3.13.

- Kombinasikan dengan Balanced Accuracy dan metrik per kelas untuk memberikan gambaran yang dapat ditindaklanjuti.

