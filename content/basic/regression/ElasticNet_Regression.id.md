---
title: "Regresi Elastic Net"
pre: "2.1.5 "
weight: 5
title_suffix: "Menggabungkan kekuatan L1 dan L2"
---

{{% summary %}}
- Elastic Net menggabungkan penalti L1 (lasso) dan L2 (ridge) untuk menyeimbangkan sparsitas dan stabilitas.
- Kelompok fitur yang sangat berkorelasi dapat dipertahankan sekaligus menyesuaikan pentingnya secara kolektif.
- Menyetel \(\alpha\) dan `l1_ratio` dengan cross-validation memudahkan pencarian keseimbangan bias–varian.
- Menstandarkan fitur dan memberi iterasi yang cukup meningkatkan stabilitas numerik optimisasi.
{{% /summary %}}

## Intuisi
Lasso dapat membuat koefisien tepat nol dan melakukan seleksi fitur, tetapi jika fitur berkorelasi kuat ia mungkin hanya memilih satu dan membuang yang lain. Ridge mengecilkan koefisien secara halus dan stabil, namun tidak pernah membuatnya nol. Elastic Net menggabungkan kedua penalti sehingga fitur yang berkorelasi dapat dipertahankan bersama sementara koefisien yang tidak penting didorong mendekati nol.

## Formulasi matematis
Elastic Net meminimalkan

$$
\min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left( y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b) \right)^2 + \alpha \left( \rho \lVert \boldsymbol\beta \rVert_1 + (1 - \rho) \lVert \boldsymbol\beta \rVert_2^2 \right),
$$

dengan \(\alpha > 0\) sebagai kekuatan regularisasi dan \(\rho \in [0,1]\) (`l1_ratio`) mengontrol perbandingan L1/L2. Mengubah \(\rho\) dari 0 ke 1 memungkinkan kita menelusuri spektrum antara ridge dan lasso.

## Eksperimen dengan Python
Berikut penggunaan `ElasticNetCV` untuk memilih \(\alpha\) dan `l1_ratio` secara bersamaan, kemudian meninjau koefisien serta kinerjanya.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Membuat data dengan korelasi fitur yang kuat
X, y, coef = make_regression(
    n_samples=500,
    n_features=30,
    n_informative=10,
    noise=15.0,
    coef=True,
    random_state=123,
)

# Duplikasi beberapa fitur untuk memperkuat korelasi
X = np.hstack([X, X[:, :5] + np.random.normal(scale=0.1, size=(X.shape[0], 5))])
feature_names = [f"x{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Menyetel alpha dan l1_ratio secara bersamaan
enet_cv = ElasticNetCV(
    l1_ratio=[0.2, 0.5, 0.7, 0.9, 0.95, 1.0],
    alphas=np.logspace(-3, 1, 30),
    cv=5,
    random_state=42,
    max_iter=5000,
)
enet_cv.fit(X_train, y_train)

print("Alpha terbaik:", enet_cv.alpha_)
print("l1_ratio terbaik:", enet_cv.l1_ratio_)

# Melatih model dengan hiperparameter terbaik dan mengevaluasi kinerja
enet = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_, max_iter=5000)
enet.fit(X_train, y_train)

train_pred = enet.predict(X_train)
test_pred = enet.predict(X_test)

print("Train R^2:", r2_score(y_train, train_pred))
print("Test R^2:", r2_score(y_test, test_pred))
print("Test RMSE:", mean_squared_error(y_test, test_pred, squared=False))

# Mengurutkan koefisien terbesar untuk inspeksi
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": enet.coef_,
}).sort_values("coef", key=lambda s: s.abs(), ascending=False)
print(coef_df.head(10))
```

### Cara membaca hasil
- `ElasticNetCV` mengevaluasi berbagai kombinasi L1/L2 secara otomatis dan memilih keseimbangan yang baik.
- Ketika fitur yang berkorelasi bertahan bersama, koefisiennya cenderung memiliki besar yang serupa sehingga lebih mudah ditafsirkan.
- Jika konvergensi lambat, standarkan input atau tingkatkan `max_iter`.

## Referensi
{{% references %}}
<li>Zou, H., &amp; Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320.</li>
<li>Friedman, J., Hastie, T., &amp; Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. <i>Journal of Statistical Software</i>, 33(1), 1–22.</li>
{{% /references %}}
