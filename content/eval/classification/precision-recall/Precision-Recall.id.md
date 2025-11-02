---

title: "Precision, Recall, dan F1 | Menyetel ambang dengan Python 3.13"

linkTitle: "Precision-Recall"

seo_title: "Precision, Recall, dan F1 | Menyetel ambang dengan Python 3.13"

pre: "4.3.2 "

weight: 2

---



{{< lead >}}

Precision menunjukkan seberapa banyak prediksi positif yang benar, recall menunjukkan seberapa banyak kasus positif yang berhasil ditangkap, dan F1 merangkum keduanya. Dengan kode Python 3.13 yang dapat diulang, kita bisa melihat kurva precision–recall dan memilih ambang keputusan yang paling sesuai.

{{< /lead >}}



---



## 1. Definisi singkat



Dari matriks kebingungan dengan true positive (TP), false positive (FP), dan false negative (FN):





\text{Precision} = \frac{TP}{TP + FP}, \qquad

\text{Recall} = \frac{TP}{TP + FN}





- **Precision** – seberapa akurat prediksi positif. Penting ketika false positive harus ditekan.

- **Recall** – seberapa banyak positif nyata yang berhasil ditemukan. Penting ketika false negative sangat berisiko.

- **F1** – rata-rata harmonik precision dan recall, berguna untuk ringkasan satu angka.





F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}





---



## 2. Implementasi dan kurva PR di Python 3.13



Pastikan interpreter dan paket sudah siap:



```bash

python --version        # contoh: Python 3.13.0

pip install scikit-learn matplotlib

```



Contoh berikut membuat dataset tidak seimbang (kelas positif 5%), melatih regresi logistik dengan class_weight="balanced", lalu menggambar kurva precision–recall bersama average precision (AP). Gambar disimpan di static/images/eval/classification/precision-recall/pr_curve.png dan bisa dibuat ulang melalui generate_eval_assets.py.



```python

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    ConfusionMatrixDisplay,

    classification_report,

    precision_recall_curve,

    precision_score,

    recall_score,

    f1_score,

    average_precision_score,

)

from sklearn.model_selection import train_test_split



X, y = make_classification(

    n_samples=40_000,

    n_features=20,

    n_informative=6,

    weights=[0.95, 0.05],

    random_state=42,

)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, stratify=y, random_state=42

)



clf = LogisticRegression(max_iter=2000, class_weight="balanced")

clf.fit(X_train, y_train)



proba = clf.predict_proba(X_test)[:, 1]

y_pred = (proba >= 0.5).astype(int)

print(classification_report(y_test, y_pred, digits=3))



precision, recall, thresholds = precision_recall_curve(y_test, proba)

ap = average_precision_score(y_test, proba)



fig, ax = plt.subplots(figsize=(5, 4))

ax.step(recall, precision, where="post", label=f"Kurva PR (AP={ap:.3f})")

ax.set_xlabel("Recall")

ax.set_ylabel("Precision")

ax.set_xlim(0, 1)

ax.set_ylim(0, 1.05)

ax.grid(alpha=0.3)

ax.legend()

fig.tight_layout()

output_dir = Path("static/images/eval/classification/precision-recall")

output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "pr_curve.png", dpi=150)

plt.close(fig)

```



{{< figure src="/images/eval/classification/precision-recall/pr_curve.png" alt="Kurva precision–recall" caption="Average precision (AP) adalah luas di bawah kurva PR. Mengubah ambang akan menggeser titik pada kurva." >}}



---



## 3. Menentukan ambang keputusan



- Ambang default 0.5 bisa membuat recall rendah jika kelas positif langka.

- Setiap titik pada kurva PR memiliki ambang yang bersesuaian pada array 	hresholds dari precision_recall_curve.

- Menurunkan ambang meningkatkan recall namun menurunkan precision; keputusan akhir tergantung biaya bisnis.



```python

threshold = 0.3

custom_pred = (proba >= threshold).astype(int)

print(

    "threshold=0.30",

    "Precision=", precision_score(y_test, custom_pred),

    "Recall=", recall_score(y_test, custom_pred),

    "F1=", f1_score(y_test, custom_pred),

)

```



---



## 4. Strategi rata-rata untuk multikelas



Parameter verage pada scikit-learn menyediakan beberapa opsi:



- macro — rata-rata sederhana antar kelas; setiap kelas bobotnya sama.

- weighted — rata-rata berbobot jumlah sampel; menjaga keseimbangan global.

- micro — menghitung ulang dari gabungan semua sampel; bisa menutupi kinerja kelas minoritas.



```python

precision_score(y_test, y_pred, average="macro")

recall_score(y_test, y_pred, average="weighted")

f1_score(y_test, y_pred, average="micro")

```



---



## Ringkasan



- Precision menekan false positive, recall menekan false negative; F1 menyeimbangkan keduanya.

- Kurva precision–recall menunjukkan perubahan trade-off ketika ambang diubah; average precision merangkum informasi tersebut.

- Dengan scikit-learn di Python 3.13, pembuatan kurva dan analisis ambang bisa dilakukan hanya dengan beberapa baris kode.

---

