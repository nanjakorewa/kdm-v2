---
title: "Regresi Komponen Utama (PCR)"
pre: "2.1.8 "
weight: 8
title_suffix: "Mengurangi multikolinearitas lewat reduksi dimensi"
---

{{% summary %}}
- PCR menerapkan PCA untuk memampatkan fitur sebelum menjalankan regresi linear, sehingga mengurangi ketidakstabilan akibat multikolinearitas.
- Komponen utama memprioritaskan arah dengan varians besar, menyaring sumbu yang didominasi noise sambil mempertahankan struktur informatif.
- Menentukan jumlah komponen yang disimpan menyeimbangkan risiko overfitting dan biaya komputasi.
- Praproses yang baik—standarisasi dan penanganan nilai hilang—menjadi fondasi akurasi dan interpretabilitas.
{{% /summary %}}

## Intuisi
Ketika prediktor saling berkorelasi kuat, metode kuadrat terkecil dapat menghasilkan koefisien yang sangat berfluktuasi. PCR terlebih dahulu melakukan PCA untuk menggabungkan arah yang berkorelasi, kemudian melakukan regresi pada komponen utama dengan urutan varians terbesar. Dengan fokus pada variasi dominan, koefisien yang diperoleh menjadi lebih stabil.

## Formulasi matematis
PCA diterapkan pada matriks rancangan \(\mathbf{X}\) yang telah distandarkan, lalu \(k\) autovektor teratas dipertahankan. Dengan skor komponen utama \(\mathbf{Z} = \mathbf{X} \mathbf{W}_k\), model regresi

$$
y = \boldsymbol{\gamma}^\top \mathbf{Z} + b
$$

dipelajari. Koefisien dalam ruang fitur asli dipulihkan melalui \(\boldsymbol{\beta} = \mathbf{W}_k \boldsymbol{\gamma}\). Jumlah komponen \(k\) dipilih menggunakan varian kumulatif atau cross-validation.

## Eksperimen dengan Python
Kami menghitung skor validasi silang PCR pada dataset diabetes sambil memvariasikan jumlah komponen.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import japanize_matplotlib

X, y = load_diabetes(return_X_y=True)

def build_pcr(n_components):
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=0)),
        ("reg", LinearRegression()),
    ])

components = range(1, X.shape[1] + 1)
cv_scores = []
for k in components:
    model = build_pcr(k)
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    cv_scores.append(score.mean())

best_k = components[int(np.argmax(cv_scores))]
print("Jumlah komponen terbaik:", best_k)

best_model = build_pcr(best_k).fit(X, y)
print("Rasio varians yang dijelaskan:", best_model["pca"].explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(components, [-s for s in cv_scores], marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"terbaik={best_k}")
plt.xlabel("Jumlah komponen k")
plt.ylabel("CV MSE (semakin kecil semakin baik)")
plt.legend()
plt.tight_layout()
plt.show()
```

![principal-component-regression block 1](/images/basic/regression/principal-component-regression_block01_id.png)

### Cara membaca hasil
- Semakin banyak komponen, kecocokan terhadap data latih meningkat, tetapi MSE validasi silang mencapai minimum pada jumlah menengah.
- Rasio varians yang dijelaskan memperlihatkan seberapa besar variabilitas yang ditangkap tiap komponen.
- Loading komponen menunjukkan fitur asli mana yang paling berkontribusi pada setiap arah utama.

## Referensi
{{% references %}}
<li>Jolliffe, I. T. (2002). <i>Principal Component Analysis</i> (2nd ed.). Springer.</li>
<li>Massy, W. F. (1965). Principal Components Regression in Exploratory Statistical Research. <i>Journal of the American Statistical Association</i>, 60(309), 234–256.</li>
{{% /references %}}
