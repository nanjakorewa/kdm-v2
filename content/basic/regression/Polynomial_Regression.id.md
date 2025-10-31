---
title: "Regresi polinomial"
pre: "2.1.4 "
weight: 4
title_suffix: "Menangkap pola nonlinier dengan model linear"
---

{{% summary %}}
- Regresi polinomial menambahkan fitur berupa pangkat sehingga model linear dapat menyesuaikan hubungan nonlinier.
- Model tetap berbentuk kombinasi linear dari koefisien, sehingga solusi tertutup dan interpretabilitas tetap terjaga.
- Semakin tinggi derajat polinomial semakin ekspresif, tetapi risiko overfitting juga meningkat; regularisasi dan validasi silang menjadi penting.
- Standardisasi fitur dan penyetelan derajat beserta kekuatan penalti memberi prediksi yang stabil.
{{% /summary %}}

## Intuisi
Garis lurus tidak mampu menggambarkan kurva halus atau pola berbukit. Dengan menambahkan istilah polinomial—\(x, x^2, x^3, \dots\) untuk kasus univariat atau pangkat dan interaksi pada kasus multivariat—kita dapat mengekspresikan perilaku nonlinier sambil tetap berada dalam kerangka model linear.

## Formulasi matematis
Untuk \(\mathbf{x} = (x_1, \dots, x_m)\) kita bentuk vektor fitur polinomial \(\phi(\mathbf{x})\) hingga derajat \(d\). Contohnya, jika \(m = 2\) dan \(d = 2\),

$$
\phi(\mathbf{x}) = (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2),
$$

sehingga modelnya

$$
y = \mathbf{w}^\top \phi(\mathbf{x}).
$$

Jumlah istilah akan cepat bertambah saat derajat meningkat, karenanya dalam praktik biasanya mulai dari derajat 2 atau 3 dan dipadukan dengan regularisasi (misalnya Ridge).

## Eksperimen dengan Python
Contoh berikut menambahkan fitur hingga derajat tiga dan menyesuaikan kurva pada data yang berasal dari fungsi kubik ditambah noise.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 200)
y_true = 0.5 * X**3 - 1.2 * X**2 + 2.0 * X + 1.5
y = y_true + rng.normal(scale=2.0, size=X.shape)

X = X[:, None]

linear_model = LinearRegression().fit(X, y)
poly_model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression()
).fit(X, y)

grid = np.linspace(-3.5, 3.5, 300)[:, None]
linear_pred = linear_model.predict(grid)
poly_pred = poly_model.predict(grid)
true_curve = 0.5 * grid.ravel()**3 - 1.2 * grid.ravel()**2 + 2.0 * grid.ravel() + 1.5

print("MSE regresi linear:", mean_squared_error(y, linear_model.predict(X)))
print("MSE regresi polinomial derajat 3:", mean_squared_error(y, poly_model.predict(X)))

plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=20, color="#ff7f0e", alpha=0.6, label="observasi")
plt.plot(grid, true_curve, color="#2ca02c", linewidth=2, label="kurva asli")
plt.plot(grid, linear_pred, color="#1f77b4", linestyle="--", linewidth=2, label="regresi linear")
plt.plot(grid, poly_pred, color="#d62728", linewidth=2, label="polinomial derajat 3")
plt.xlabel("input $x$")
plt.ylabel("output $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![polynomial-regression block 1](/images/basic/regression/polynomial-regression_block01_id.png)

### Membaca hasil
- Regresi linear biasa gagal mengikuti kelengkungan (terutama di tengah), sedangkan model kubik mengikuti kurva sebenarnya dengan baik.
- Derajat yang lebih tinggi meningkatkan kecocokan pada data latih, tetapi dapat membuat prediksi di luar rentang menjadi tidak stabil.
- Menggabungkan fitur polinomial dengan regresi ter-regularisasi (misalnya Ridge) di dalam pipeline membantu menekan overfitting.

## Referensi
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
