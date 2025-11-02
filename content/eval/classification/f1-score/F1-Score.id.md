---

title: "F1 Score | Rata-rata harmonik precision dan recall"

linkTitle: "F1 Score"

seo_title: "F1 Score | Rata-rata harmonik precision dan recall"

pre: "4.3.8 "

weight: 8

---



{{< lead >}}

F1 score menggabungkan precision dan recall melalui rata-rata harmonik. Metrik ini cocok ketika false positive dan false negative sama-sama krusial. Berikut cara menghitungnya di Python 3.13, melihat pengaruh ambang keputusan, serta memanfaatkan varian Fβ.

{{< /lead >}}



---



## 1. Definisi



Dengan precision \(P\) dan recall \(R\):





F_1 = 2 \cdot \frac{P \cdot R}{P + R}.





Versi umumnya, Fβ, menambah bobot ke recall (\(\beta > 1\)) atau precision (\(\beta < 1\)):





F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}.





---



## 2. Perhitungan di Python 3.13



```bash

python --version        # contoh: Python 3.13.0

pip install scikit-learn matplotlib

```



```python

import numpy as np

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, f1_score, fbeta_score

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



print(classification_report(y_test, y_pred, digits=3))

print("F1:", f1_score(y_test, y_pred))

print("F0.5:", fbeta_score(y_test, y_pred, beta=0.5))

print("F2:", fbeta_score(y_test, y_pred, beta=2.0))

```



classification_report merangkum precision, recall, dan F1 per kelas.



---



## 3. Dampak ambang terhadap F1



Dengan probabilitas keluaran, kita bisa melihat bagaimana nilai F1 berubah ketika ambang keputusan digeser.



```python

from sklearn.metrics import f1_score, precision_recall_curve



proba = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, proba)

thresholds = np.append(thresholds, 1.0)

f1_scores = [

    f1_score(y_test, (proba >= t).astype(int))

    for t in thresholds

]

```



{{< figure src="/images/eval/classification/f1-score/f1_vs_threshold.png" alt="F1 terhadap ambang" caption="Temukan titik maksimum untuk mendapatkan keseimbangan terbaik antara precision dan recall." >}}



- Titik puncak menunjukkan trade-off terbaik ketika precision dan recall dianggap sama penting.

- Gunakan F0.5 atau F2 jika ingin menekankan precision atau recall sesuai kebutuhan bisnis.



---



## 4. Strategi rata-rata pada multikelas



Argumen verage di scikit-learn memungkinkan agregasi F1 pada masalah multikelas/multilabel:



- macro — rata-rata sederhana atas F1 per kelas.

- weighted — rata-rata berbobot berdasarkan jumlah sampel tiap kelas.

- micro — menghitung ulang dari matriks kebingungan gabungan; bisa menutupi kinerja kelas minoritas.



```python

from sklearn.metrics import f1_score



f1_macro = f1_score(y_test, y_pred, average="macro")

f1_weighted = f1_score(y_test, y_pred, average="weighted")

```



Untuk multilabel, verage="samples" memberikan rata-rata per sampel.



---



## Ringkasan



- F1 menyeimbangkan precision dan recall; grafikan terhadap ambang untuk memilih titik operasi yang sesuai.

- Fβ membantu ketika perlu lebih menekankan recall (β>1) atau precision (β<1).

- Pada multikelas, nyatakan strategi averaging dan analisis F1 bersama precision, recall, dan kurva PR untuk memahami perilaku model secara menyeluruh.

---

