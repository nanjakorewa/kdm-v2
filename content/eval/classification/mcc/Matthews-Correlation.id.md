---

title: "Matthews Correlation Coefficient (MCC) | Metrik biner yang seimbang"

linkTitle: "MCC"

seo_title: "Matthews Correlation Coefficient (MCC) | Metrik biner yang seimbang"

pre: "4.3.5 "

weight: 5

---



{{< lead >}}

MCC menggabungkan semua elemen matriks kebingungan—TP, FP, FN, TN—ke dalam satu nilai antara −1 dan 1. Metrik ini tetap informatif ketika data sangat tidak seimbang, sehingga layak dikomunikasikan bersama Accuracy dan F1.

{{< /lead >}}



---



## 1. Definisi



Untuk klasifikasi biner:





\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}.





- **1** → prediksi sempurna

- **0** → setara dengan tebakan acak

- **−1** → prediksi kebalikan dari kenyataan



Versi multikelas diperoleh dari matriks kebingungan penuh.



---



## 2. Perhitungan di Python 3.13



```bash

python --version        # contoh: Python 3.13.0

pip install scikit-learn matplotlib

```



```python

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import matthews_corrcoef, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



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



model = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, class_weight="balanced"),

)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(confusion_matrix(y_test, y_pred))

print("MCC:", matthews_corrcoef(y_test, y_pred))

```



class_weight="balanced" memastikan kelas minoritas berkontribusi terhadap skor.



---



## 3. MCC terhadap ambang



{{< figure src="/images/eval/classification/mcc/mcc_vs_threshold.png" alt="MCC terhadap ambang" caption="Carilah ambang yang memaksimalkan MCC untuk mendapatkan keseimbangan terbaik antar kelas." >}}



Berbeda dengan F1, MCC juga menghargai prediksi negatif yang benar. Evaluasi di sepanjang ambang membantu memilih titik operasi paling stabil.



---



## 4. Penggunaan praktis



- **Memvalidasi Accuracy** – jika Accuracy tinggi tetapi MCC rendah, kemungkinan ada kelas yang diabaikan.

- **Seleksi model** – gunakan make_scorer(matthews_corrcoef) pada GridSearchCV untuk mengoptimalkan MCC secara langsung.

- **Lengkapi ROC/PR** – MCC memberikan pandangan global, sementara ROC-AUC atau PR-AUC fokus pada trade-off recall/precision.



---



## Ringkasan



- MCC menyediakan evaluasi yang seimbang antara −1 dan 1 dengan memanfaatkan seluruh matriks kebingungan.

- Di Python 3.13, matthews_corrcoef menghitungnya dengan mudah; visualisasi terhadap ambang menunjukkan titik operasi terbaik.

- Laporkan MCC bersama Accuracy, F1, dan metrik PR untuk menghindari kesimpulan keliru di data yang tidak seimbang.

---

