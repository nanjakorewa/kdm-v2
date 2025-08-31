---
title: "Regresi Ridge dan Lasso"
pre: "2.1.2 "
weight: 2
title_suffix: "Intuisi dan praktik"
---

{{% youtube "rhGYOBrxPXA" %}}


<div class="pagetop-box">
  <p><b>Ridge</b> (L2) dan <b>Lasso</b> (L1) menambahkan penalti pada besaran koefisien untuk <b>mengurangi overfitting</b> dan meningkatkan <b>generalization</b>. Ini disebut <b>regularisasi</b>. Kita akan melatih OLS, Ridge, dan Lasso dengan kondisi yang sama dan membandingkan perilakunya.</p>
  </div>

---

## 1. Intuisi dan rumus

OLS meminimalkan jumlah kuadrat residual:
$$
\text{RSS}(\boldsymbol\beta, b) = \sum_{i=1}^n \big(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\big)^2.
$$
Dengan banyak fitur, kolinearitas kuat, atau data berisik, koefisien bisa terlalu besar dan overfit. Regularisasi menambah penalti untuk <b>mengecilkan</b> koefisien.

- <b>Ridge (L2)</b>
  $$
  \min_{\boldsymbol\beta, b}\; \text{RSS}(\boldsymbol\beta,b) + \alpha \lVert \boldsymbol\beta \rVert_2^2
  $$
  Mengecilkan semua koefisien secara halus (jarang tepat nol).

- <b>Lasso (L1)</b>
  $$
  \min_{\boldsymbol\beta, b}\; \text{RSS}(\boldsymbol\beta,b) + \alpha \lVert \boldsymbol\beta \rVert_1
  $$
  Membuat sebagian koefisien tepat nol (seleksi fitur).

> Heuristik:
> - Ridge “mengecilkan semuanya sedikit” → stabil, tidak sparse.
> - Lasso “membuat sebagian nol” → model sparse, bisa tidak stabil saat kolinearitas kuat.

---

## 2. Persiapan

{{% notice tip %}}
Dalam regularisasi, <b>skala fitur penting</b>. Gunakan <b>StandardScaler</b> agar penalti berlaku adil.
{{% /notice %}}

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

---

## 3. Data: hanya 2 fitur informatif

{{% notice document %}}
- [sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)
{{% /notice %}}

Kita buat data dengan tepat dua fitur informatif (<code>n_informative=2</code>) lalu menambah transformasi redundan untuk mensimulasikan banyak fitur tak berguna.

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

# Tambahkan transformasi nonlinier redundan
X = np.concatenate([X, np.log(X + 100)], axis=1)

# Standarkan y untuk skala yang adil
y = (y - y.mean()) / np.std(y)
```

---

## 4. Latih tiga model dengan pipeline yang sama

{{% notice document %}}
- [make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)  
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)  
- [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
{{% /notice %}}

```python
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
rid_r = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2)).fit(X, y)
las_r = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=0.1, max_iter=10_000)).fit(X, y)
```

> Lasso bisa lambat konvergen; praktiknya, naikkan <code>max_iter</code> bila perlu.

---

## 5. Bandingkan besar koefisien (plot)

- OLS: koefisien jarang mendekati nol.  
- Ridge: mengecilkan koefisien secara umum.  
- Lasso: sebagian koefisien menjadi tepat nol (seleksi fitur).

```python
feat_index = np.arange(X.shape[1])

plt.figure(figsize=(12, 4))
plt.bar(feat_index - 0.25, np.abs(lin_r.steps[1][1].coef_), width=0.25, label="Linear")
plt.bar(feat_index,         np.abs(rid_r.steps[1][1].coef_), width=0.25, label="Ridge")
plt.bar(feat_index + 0.25,  np.abs(las_r.steps[1][1].coef_), width=0.25, label="Lasso")

plt.xlabel(r"indeks koefisien $\beta$")
plt.ylabel(r"$|\beta|$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

![png](/images/basic/regression/02_Ridge_and_Lasso_files/02_Ridge_and_Lasso_10_1.png)

{{% notice tip %}}
Di sini hanya dua fitur yang informatif (<code>n_informative=2</code>). Lasso cenderung men-nol-kan banyak fitur tak berguna (model sparse), sedangkan Ridge mengecilkan semua koefisien secara halus untuk menstabilkan solusi.
{{% /notice %}}

---

## 6. Cek generalisasi (CV)

<code>alpha</code> tetap bisa menyesatkan; bandingkan dengan cross-validation.

```python
from sklearn.metrics import make_scorer, mean_squared_error

scorer = make_scorer(mean_squared_error, greater_is_better=False)

models = {
    "Linear": lin_r,
    "Ridge (α=2)": rid_r,
    "Lasso (α=0.1)": las_r,
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"{name:12s}  R^2 (CV mean±std): {scores.mean():.3f} ± {scores.std():.3f}")
```

{{% notice tip %}}
Tidak ada satu jawaban untuk semua. Dengan kolinearitas kuat atau sampel kecil, Ridge/Lasso sering lebih stabil daripada OLS. <b>Atur α</b> via CV.
{{% /notice %}}

---

## 7. Memilih α otomatis

Dalam praktik, gunakan <code>RidgeCV</code>/<code>LassoCV</code> (atau <code>GridSearchCV</code>).

```python
from sklearn.linear_model import RidgeCV, LassoCV

ridge_cv = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
).fit(X, y)

lasso_cv = make_pipeline(
    StandardScaler(with_mean=False),
    LassoCV(alphas=np.logspace(-3, 1, 9), cv=5, max_iter=50_000)
).fit(X, y)

print("α terbaik (RidgeCV):", ridge_cv.steps[1][1].alpha_)
print("α terbaik (LassoCV):", lasso_cv.steps[1][1].alpha_)
```

---

## 8. Pilih yang mana?

- Banyak fitur / kolinearitas kuat → coba <b>Ridge</b> dulu (menstabilkan).
- Butuh seleksi fitur / interpretabilitas → <b>Lasso</b>.
- Kolinearitas kuat dan Lasso tidak stabil → <b>Elastic Net</b> (L1+L2).

---

## 9. Ringkasan

- Regularisasi menambah <b>penalti pada besaran koefisien</b> untuk mengurangi overfitting.
- <b>Ridge (L2)</b> mengecilkan secara halus; jarang tepat nol.
- <b>Lasso (L1)</b> membuat sebagian koefisien nol; seleksi fitur.
- <b>Standarisasi penting</b>; <b>atur α via CV</b>.

---

