---

title: "Confusion matrix | Membaca performa model klasifikasi"

linkTitle: "Confusion matrix"

seo_title: "Confusion matrix | Membaca performa model klasifikasi"

pre: "4.3.0 "

weight: 0

---



{{< lead >}}

Confusion matrix merangkum bagaimana model klasifikasi memberi label pada setiap kelas. Dengan melihatnya bersama akurasi, presisi, recall, dan F1, kita bisa menemukan sumber kesalahan secara cepat.

{{< /lead >}}



---



## 1. Struktur confusion matrix



Untuk klasifikasi biner, confusion matrix berupa tabel 2×2:



|                 | Prediksi: Negatif | Prediksi: Positif |

| --------------- | ----------------- | ----------------- |

| **Fakta: Negatif** | True Negative (TN)   | False Positive (FP)  |

| **Fakta: Positif** | False Negative (FN)  | True Positive (TP)   |



- Baris menunjukkan label sebenarnya, kolom menunjukkan hasil prediksi.

- Meninjau TP / FP / FN / TN membantu mengecek apakah model berat sebelah terhadap kelas tertentu.



---



## 2. Contoh lengkap dengan Python 3.13



Pastikan lingkungan Anda menggunakan **Python 3.13** dan instal dependensi berikut:



```bash

python --version  # contoh: Python 3.13.0

pip install scikit-learn matplotlib

```



Script di bawah melatih regresi logistik pada dataset Breast Cancer, lalu mencetak dan memvisualisasikan confusion matrix. `Pipeline` dengan `StandardScaler` membuat proses optimasi stabil dan menghindari peringatan konvergensi.



```python

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)



pipeline = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=1000, solver="lbfgs"),

)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)



cm = confusion_matrix(y_test, y_pred)

print(cm)



disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues", colorbar=False)

plt.tight_layout()

plt.show()

```



{{< figure src="/images/eval/confusion-matrix/binary_matrix.png" alt="Confusion matrix untuk dataset Breast Cancer" caption="Visualisasi confusion matrix menggunakan scikit-learn (Python 3.13)" >}}



---



## 3. Normalisasi untuk melihat proporsi



Jika dataset tidak seimbang, normalisasi per baris memudahkan perbandingan tingkat kesalahan.



```python

cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

print(cm_norm)



disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)

disp_norm.plot(cmap="Blues", values_format=".2f", colorbar=False)

plt.tight_layout()

plt.show()

```



- `normalize="true"`: rasio terhadap setiap kelas aktual  

- `normalize="pred"`: rasio terhadap setiap kelas prediksi  

- `normalize="all"`: rasio terhadap seluruh observasi  



---



## 4. Klasifikasi multikelas



Gunakan `ConfusionMatrixDisplay.from_predictions` untuk membuat confusion matrix multikelas beserta label sumbu secara otomatis.



```python

ConfusionMatrixDisplay.from_predictions(

    y_true=label_sebenarnya,

    y_pred=label_model,

    normalize="true",

    values_format=".2f",

    cmap="Blues",

)

plt.tight_layout()

plt.show()

```



---



## 5. Checklist praktis



- **False negative vs. false positive**: tentukan kesalahan mana yang paling kritis (misal kesehatan atau fraud) dan pantau nilai sel terkait.

- **Gunakan heatmap**: visualisasi membantu mendeteksi kelas yang kurang baik dan memudahkan diskusi tim.

- **Turunkan metrik lain**: akurasi, presisi, recall, dan F1 bisa dihitung dari matriks yang sama; bandingkan pula dengan ROC-AUC atau PR curve.

- **Simpan notebook yang dapat diulang**: dokumentasikan analisis di notebook Python 3.13 agar iterasi tuning dan retraining lebih cepat.



---



## Ringkasan



- Confusion matrix merangkum TP / FP / FN / TN dan mengungkap pola kesalahan model.

- Normalisasi membantu membaca rasio kesalahan ketika kelas tidak seimbang.

- Kombinasikan confusion matrix dengan metrik turunan dan kebutuhan bisnis untuk menentukan kriteria evaluasi yang tepat sasaran.

