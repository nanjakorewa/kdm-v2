---
title: "Balanced Accuracy | Evaluasi ketika data tidak seimbang"
linkTitle: "Balanced Accuracy"
seo_title: "Balanced Accuracy | Evaluasi ketika data tidak seimbang"
pre: "4.3.6 "
weight: 6
---

{{< lead >}}
Balanced Accuracy menghitung rata-rata recall setiap kelas sehingga tetap informatif ketika data sangat tidak seimbang. Contoh Python 3.13 berikut menunjukkan perbedaannya dibanding Accuracy biasa dan kapan sebaiknya Anda menggunakannya.
{{< /lead >}}

---

## 1. Definisi

Balanced Accuracy merupakan rata-rata dari true positive rate (TPR) dan true negative rate (TNR):


\mathrm{Balanced\ Accuracy} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)


Pada kasus multikelas, recall tiap kelas dihitung lalu dirata-rata dengan cara yang sama.

---

## 2. Implementasi di Python 3.13

`ash
python --version        # contoh: Python 3.13.0
pip install scikit-learn matplotlib
`

Kita menggunakan model Random Forest yang sama dengan contoh Accuracy dan menampilkan kedua metrik secara berdampingan. Diagram batang disimpan di static/images/eval/classification/accuracy/accuracy_vs_balanced.png sehingga bisa dibuat ulang melalui generate_eval_assets.py.

`python
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
`

{{< figure src="/images/eval/classification/accuracy/accuracy_vs_balanced.png" alt="Perbandingan Accuracy vs Balanced Accuracy" caption="Balanced Accuracy memperlakukan recall setiap kelas secara seimbang sehingga minoritas tidak tersembunyi." >}}

---

## 3. Kapan memilih Balanced Accuracy

- **Data sangat tidak seimbang** – Accuracy biasa bisa tinggi meski kelas minoritas tidak pernah terdeteksi. Balanced Accuracy menyoroti hal itu.
- **Membandingkan model** – ketika beberapa model diuji pada data yang miring, Balanced Accuracy membuat perbedaan performa lebih jujur.
- **Menentukan ambang** – gunakan bersamaan dengan precision/recall untuk memastikan kedua kelas tetap terdeteksi pada ambang pilihan Anda.

---

## 4. Metrik pelengkap

| Metrik | Yang diukur | Catatan pada data tidak seimbang |
| --- | --- | --- |
| Accuracy | Persentase keseluruhan yang benar | Didominasi oleh kelas mayoritas |
| Recall / Sensitivity | Tingkat deteksi per kelas | Perlu dilaporkan per kelas |
| **Balanced Accuracy** | Rata-rata recall per kelas | Menjaga visibilitas kelas minoritas |
| Macro F1 | Rata-rata harmonik precision & recall per kelas | Cocok saat precision juga penting |

---

## Ringkasan

- Balanced Accuracy adalah rata-rata recall per kelas dan cocok untuk mengevaluasi data yang miring.
- Dengan alanced_accuracy_score di Python 3.13, Anda dapat menghitungnya dengan mudah dan membandingkannya dengan Accuracy.
- Sertakan precision, recall, dan F1 agar seluruh pemangku kepentingan memahami kualitas model secara menyeluruh.
---
