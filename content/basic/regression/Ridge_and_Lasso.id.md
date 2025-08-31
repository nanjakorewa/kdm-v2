---
title: "Regresi Ridge dan Lasso"
pre: "2.1.2 "
weight: 2
searchtitle: "Regresi Ridge dan Lasso dalam Python"
---

<div class="pagetop-box">
    <p>Regresi Ridge dan Lasso adalah model regresi linier yang didesain untuk mencegah model dari overfitting dengan cara menghukum koefisien model. Teknik-teknik seperti ini yang mencoba untuk mencegah overfitting dan meningkatkan generalisasi disebut regularisasi. Pada halaman ini, kita akan menggunakan python untuk menjalankan regresi kuadrat terkecil, ridge, dan lasso dan membandingkan koefisiennya.</p>
</div>

## Impor modul yang diperlukan

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

## Membuat data regresi untuk eksperimen

{{% notice document %}}
[sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)
{{% /notice %}}

Membuat data regresi secara artifisial menggunakan [sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html). Tentukan dua fitur (`n_informatif` = 2) yang diperlukan untuk prediksi. Jika tidak, mereka adalah fitur-fitur berlebihan yang tidak berguna untuk prediksi.


```python
n_features = 5
n_informative = 2
X, y = make_regression(
    n_samples=500,
    n_features=n_features,
    n_informative=n_informative,
    noise=0.5,
    random_state=777,
)
X = np.concatenate([X, np.log(X + 100)], 1)  # Tambahkan fitur yang berlebihan
y_mean = y.mean(keepdims=True)
y_std = np.std(y, keepdims=True)
y = (y - y_mean) / y_std
```

## Membandingkan model regresi kuadrat terkecil, regresi ridge, dan regresi lasso
Mari kita latih setiap model dengan data yang sama dan lihat perbedaannya.
Biasanya, koefisien alpha dari istilah regularisasi harus diubah, tetapi dalam kasus ini, koefisien alpha tetap.

{{% notice document %}}
- [sklearn.pipeline.make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)
- [sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)
{{% /notice %}}


```python
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
rid_r = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2)).fit(X, y)
las_r = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=0.1)).fit(X, y)
```

## Bandingkan nilai-nilai koefisien untuk setiap model
Plot nilai absolut dari setiap koefisien. Dapat dilihat dari grafik bahwa untuk regresi linier, koefisien hampir tidak pernah nol. Kita juga dapat melihat bahwa regresi Lasso memiliki koefisien = 0 untuk semua kecuali fitur yang dibutuhkan untuk prediksi.

Karena ada dua fitur yang berguna (`n_informative = 2`), kita dapat melihat bahwa regresi linier memiliki koefisien tanpa syarat bahkan pada fitur yang tidak berguna, sementara regresi Lasso-Ridge tidak dapat dipersempit menjadi dua fitur. Tetap saja, regresi ini menghasilkan banyak fitur yang tidak diinginkan yang memiliki koefisien nol.

```python
feat_index = np.array([i for i in range(X.shape[1])])

plt.figure(figsize=(12, 4))
plt.bar(
    feat_index - 0.2,
    np.abs(lin_r.steps[1][1].coef_),
    width=0.2,
    label="regresi kuadrat terkecil",
)
plt.bar(feat_index, np.abs(rid_r.steps[1][1].coef_), width=0.2, label="regresi ridge")
plt.bar(
    feat_index + 0.2,
    np.abs(las_r.steps[1][1].coef_),
    width=0.2,
    label="regresi lasso",
)

plt.xlabel(r"$\beta$")
plt.ylabel(r"$|\beta|$")
plt.grid()
plt.legend()
```

    
![png](/images/basic/regression/02_Ridge_and_Lasso_files/02_Ridge_and_Lasso_10_1.png)
    

