---
title: "Regresi Ridge dan Lasso"
pre: "2.1.2 "
weight: 2
title_suffix: "Meningkatkan generalisasi dengan regularisasi"
---

{{% summary %}}
- Ridge mengecilkan koefisien secara halus dengan penalti L2 dan tetap stabil meskipun terjadi multikolinearitas.
- Lasso menggunakan penalti L1 yang dapat membuat sebagian koefisien menjadi nol sehingga fitur penting tetap tersisa dan model mudah ditafsirkan.
- Menyetel kekuatan regularisasi \(\alpha\) mengendalikan kompromi antara ketepatan pada data latih dan kemampuan generalisasi.
- Kombinasi standardisasi dan validasi silang membantu memilih hiperparameter yang mencegah overfitting tanpa mengorbankan kinerja.
{{% /summary %}}

## Intuisi
Kuadrat terkecil dapat menghasilkan koefisien ekstrem saat ada pencilan atau ketika fitur saling berkorelasi kuat. Ridge dan Lasso menambahkan penalti agar koefisien tidak tumbuh tak terbatas, sehingga keseimbangan antara akurasi dan interpretabilitas terjaga. Ridge “menyusutkan semuanya sedikit E, sedangkan Lasso “mematikan Esebagian koefisien dengan membuat nilainya tepat nol.

## Formulasi matematis
Kedua metode meminimalkan loss kuadrat biasa ditambah term regularisasi:

- **Ridge**
  $$
  \min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2 + \alpha \lVert \boldsymbol\beta \rVert_2^2
  $$
- **Lasso**
  $$
  \min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2 + \alpha \lVert \boldsymbol\beta \rVert_1
  $$

Semakin besar \(\alpha\), semakin kuat kontraksinya. Pada Lasso, jika \(\alpha\) melewati ambang tertentu maka beberapa koefisien menjadi nol sehingga model menjadi jarang (sparse).

## Eksperimen dengan Python
Contoh berikut menerapkan ridge, lasso, dan regresi linier biasa pada dataset regresi sintetis yang sama serta membandingkan besar koefisien dan skor generalisasi.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Parameter
n_samples = 200
n_features = 10
n_informative = 3
rng = np.random.default_rng(42)

# Koefisien sebenarnya (hanya tiga fitur informatif)
coef = np.zeros(n_features)
coef[:n_informative] = rng.normal(loc=3.0, scale=1.0, size=n_informative)

X = rng.normal(size=(n_samples, n_features))
y = X @ coef + rng.normal(scale=5.0, size=n_samples)

linear = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
ridge = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
).fit(X, y)
lasso = make_pipeline(
    StandardScaler(with_mean=False),
    LassoCV(alphas=np.logspace(-3, 1, 9), cv=5, max_iter=50_000)
).fit(X, y)

models = {
    "Linear": linear,
    "Ridge": ridge,
    "Lasso": lasso,
}

# Visualisasi koefisien
indices = np.arange(n_features)
width = 0.25
plt.figure(figsize=(10, 4))
plt.bar(indices - width, np.abs(linear[-1].coef_), width=width, label="Linear")
plt.bar(indices, np.abs(ridge[-1].coef_), width=width, label="Ridge")
plt.bar(indices + width, np.abs(lasso[-1].coef_), width=width, label="Lasso")
plt.xlabel("indeks fitur")
plt.ylabel("|koefisien|")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Skor R^2 dengan validasi silang
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"{name:>6}: R^2 = {scores.mean():.3f} ± {scores.std():.3f}")
```

### Membaca hasil
- Ridge mengecilkan semua koefisien dan tetap stabil meski ada multikolinearitas.
- Lasso membuat sebagian koefisien bernilai nol sehingga hanya fitur terpenting yang tersisa.
- Pilih \(\alpha\) dengan validasi silang untuk menyeimbangkan bias dan varians, serta lakukan standardisasi agar tiap fitur berada pada skala yang sebanding.

## Referensi
{{% references %}}
<li>Hoerl, A. E., &amp; Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. <i>Technometrics</i>, 12(1), 55–67.</li>
<li>Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. <i>Journal of the Royal Statistical Society: Series B</i>, 58(1), 267–288.</li>
<li>Zou, H., &amp; Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320.</li>
{{% /references %}}
