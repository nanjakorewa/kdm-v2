---
title: "Weighted Least Squares (WLS)"
pre: "2.1.11 "
weight: 11
title_suffix: "Menangani observasi dengan varians berbeda"
---

{{% summary %}}
- WLS memberikan bobot khusus pada tiap observasi sehingga pengukuran yang lebih dapat dipercaya memiliki pengaruh lebih besar pada garis regresi.
- Mengalikan error kuadrat dengan bobot membuat observasi ber-varians tinggi kurang berpengaruh dan menjaga estimasi tetap dekat dengan data yang andal.
- WLS dapat dijalankan menggunakan `LinearRegression` di scikit-learn dengan menyediakan `sample_weight`.
- Bobot dapat berasal dari varians yang diketahui, analisis residual, atau pengetahuan domain; perancangannya sangat krusial.
{{% /summary %}}

## Intuisi
Kuadrat terkecil biasa mengasumsikan semua observasi sama reliabel, padahal sensor di dunia nyata sering memiliki ketelitian berbeda. WLS “mendengarkan” lebih banyak ke titik data yang dipercaya dengan menekankan kontribusinya saat fitting, sehingga regresi linear dapat menangani data heteroskedastik.

## Formulasi matematis
Dengan bobot positif \(w_i\) kita meminimalkan

$$
L(\boldsymbol\beta, b) = \sum_{i=1}^{n} w_i \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2.
$$

Pilihan idealnya adalah \(w_i \propto 1/\sigma_i^2\) (kebalikan varians) sehingga observasi yang presisi memiliki pengaruh lebih besar.

## Eksperimen dengan Python
Kami membandingkan OLS dan WLS pada data dengan tingkat noise berbeda di setiap wilayah.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(7)
n_samples = 200
X = np.linspace(0, 10, n_samples)
true_y = 1.2 * X + 3

# Tingkat noise berbeda per wilayah (error heteroskedastik)
noise_scale = np.where(X < 5, 0.5, 2.5)
y = true_y + rng.normal(scale=noise_scale)

weights = 1.0 / (noise_scale ** 2)  # Gunakan kebalikan varians sebagai bobot
X = X[:, None]

ols = LinearRegression().fit(X, y)
wls = LinearRegression().fit(X, y, sample_weight=weights)

grid = np.linspace(0, 10, 200)[:, None]
ols_pred = ols.predict(grid)
wls_pred = wls.predict(grid)

print("Kemiringan OLS:", ols.coef_[0], " Intersep:", ols.intercept_)
print("Kemiringan WLS:", wls.coef_[0], " Intersep:", wls.intercept_)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, c=noise_scale, cmap="coolwarm", s=25, label="observasi (warna=noise)")
plt.plot(grid, 1.2 * grid.ravel() + 3, color="#2ca02c", label="garis sebenarnya")
plt.plot(grid, ols_pred, color="#1f77b4", linestyle="--", linewidth=2, label="OLS")
plt.plot(grid, wls_pred, color="#d62728", linewidth=2, label="WLS")
plt.xlabel("input $x$")
plt.ylabel("output $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![weighted-least-squares block 1](/images/basic/regression/weighted-least-squares_block01.svg)

### Cara membaca hasil
- Pemberian bobot menarik garis hasil fitting ke wilayah dengan noise rendah sehingga mendekati garis sebenarnya.
- OLS terdistorsi oleh wilayah dengan noise tinggi dan cenderung meremehkan kemiringan.
- Kinerja sangat bergantung pada pemilihan bobot yang tepat; diagnosa dan intuisi domain sangat membantu.

## Referensi
{{% references %}}
<li>Carroll, R. J., &amp; Ruppert, D. (1988). <i>Transformation and Weighting in Regression</i>. Chapman &amp; Hall.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
