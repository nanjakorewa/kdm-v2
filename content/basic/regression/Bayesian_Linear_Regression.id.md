---
title: "Regresi Linear Bayesian"
pre: "2.1.6 "
weight: 6
title_suffix: "Mengukur ketidakpastian prediksi"
---

{{% summary %}}
- Regresi linear Bayesian memperlakukan koefisien sebagai variabel acak sehingga dapat memperkirakan prediksi beserta ketidakpastiannya.
- Distribusi posterior diperoleh secara analitik dari prior dan likelihood sehingga tetap andal pada data yang sedikit atau bising.
- Distribusi prediktif berbentuk Gaussian, sehingga mean dan variansinya mudah divisualisasikan untuk mendukung pengambilan keputusan.
- `BayesianRidge` di scikit-learn menyesuaikan varians noise secara otomatis sehingga implementasi praktis menjadi sederhana.
{{% /summary %}}

## Intuisi
Mínim kuadrat biasa mencari satu set koefisien “terbaik”, tetapi pada data nyata estimasi tersebut tetap mengandung ketidakpastian. Regresi linear Bayesian memodelkan ketidakpastian itu dengan memperlakukan koefisien secara probabilistik dan menggabungkan pengetahuan awal dengan data yang diamati. Bahkan dengan sedikit observasi kita memperoleh prediksi yang diharapkan beserta sebarannya.

## Formulasi matematis
Misalkan vektor koefisien \(\boldsymbol\beta\) memiliki prior Gaussian multivariat dengan mean 0 dan varians \(\tau^{-1}\), serta noise observasi \(\epsilon_i \sim \mathcal{N}(0, \alpha^{-1})\). Distribusi posteriornya adalah

$$
p(\boldsymbol\beta \mid \mathbf{X}, \mathbf{y}) = \mathcal{N}(\boldsymbol\beta \mid \boldsymbol\mu, \mathbf{\Sigma})
$$

dengan

$$
\mathbf{\Sigma} = (\alpha \mathbf{X}^\top \mathbf{X} + \tau \mathbf{I})^{-1}, \qquad
\boldsymbol\mu = \alpha \mathbf{\Sigma} \mathbf{X}^\top \mathbf{y}.
$$

Distribusi prediktif untuk masukan baru \(\mathbf{x}_*\) juga Gaussian, \(\mathcal{N}(\hat{y}_*, \sigma_*^2)\). `BayesianRidge` mengestimasi \(\alpha\) dan \(\tau\) langsung dari data sehingga tidak perlu disetel manual.

## Eksperimen dengan Python
Contoh berikut membandingkan mínimos kuadrat biasa dan regresi linear Bayesian pada data yang mengandung outlier.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error

# Membuat tren linear dengan noise dan menambahkan beberapa outlier
rng = np.random.default_rng(0)
X = np.linspace(-4, 4, 120)
y = 1.8 * X - 0.5 + rng.normal(scale=1.0, size=X.shape)
outlier_idx = rng.choice(len(X), size=6, replace=False)
y[outlier_idx] += rng.normal(scale=8.0, size=outlier_idx.shape)
X = X[:, None]

# Melatih kedua model
ols = LinearRegression().fit(X, y)
bayes = BayesianRidge(compute_score=True).fit(X, y)

grid = np.linspace(-6, 6, 200)[:, None]
ols_mean = ols.predict(grid)
bayes_mean, bayes_std = bayes.predict(grid, return_std=True)

print("MSE OLS:", mean_squared_error(y, ols.predict(X)))
print("MSE Bayesian:", mean_squared_error(y, bayes.predict(X)))
print("Rata-rata posterior koefisien:", bayes.coef_)
print("Diagonal varians posterior:", np.diag(bayes.sigma_))

upper = bayes_mean + 1.96 * bayes_std
lower = bayes_mean - 1.96 * bayes_std

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="#ff7f0e", alpha=0.6, label="observasi")
plt.plot(grid, ols_mean, color="#1f77b4", linestyle="--", label="OLS")
plt.plot(grid, bayes_mean, color="#2ca02c", linewidth=2, label="rata-rata Bayesian")
plt.fill_between(grid.ravel(), lower, upper, color="#2ca02c", alpha=0.2, label="CI 95%")
plt.xlabel("input $x$")
plt.ylabel("output $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![bayesian-linear-regression block 1](/images/basic/regression/bayesian-linear-regression_block01.svg)

### Cara membaca hasil
- OLS mudah terpengaruh oleh outlier, sedangkan regresi Bayesian menjaga prediksi rata-rata tetap stabil.
- `return_std=True` memberikan simpangan baku prediktif sehingga interval kredibel dapat digambar dengan mudah.
- Memeriksa varians posterior membantu mengidentifikasi koefisien mana yang masih memiliki ketidakpastian besar.

## Referensi
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
