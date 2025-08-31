---
title: "Metode Kuadrat Terkecil"
pre: "2.1.1 "
weight: 1
title_suffix: "Konsep dan Implementasi"
---

{{% youtube "KKuAxQbuJpk" %}}



<div class="pagetop-box">
  <p><b>Kuadrat terkecil</b> mencari koefisien fungsi yang paling pas terhadap pasangan observasi <code>(x_i, y_i)</code> dengan meminimalkan jumlah kuadrat residual. Kita fokus pada kasus paling sederhana, garis lurus <code>y = wx + b</code>, dan meninjau intuisi serta implementasi praktisnya.</p>
  </div>

{{% notice tip %}}
Rumus ditampilkan dengan KaTeX. <code>\(\hat y\)</code> adalah prediksi model dan <code>\(\epsilon\)</code> adalah noise.
{{% /notice %}}

## Tujuan
- Mempelajari garis <code>\(\hat y = wx + b\)</code> yang paling pas dengan data.
- “Paling pas” berarti meminimalkan jumlah kuadrat error (SSE):
  <code>\(\displaystyle L(w,b) = \sum_{i=1}^n (y_i - (w x_i + b))^2\)</code>

## Buat dataset sederhana
Kita buat garis lurus dengan noise dan tetapkan seed agar hasil dapat direproduksi.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # opsional untuk label Jepang

rng = np.random.RandomState(42)
n_samples = 200

# Garis sebenarnya (kemiringan 0.8, intersep 0.5) dengan noise
X = np.linspace(-10, 10, n_samples)
epsilon = rng.normal(loc=0.0, scale=1.0, size=n_samples)
y = 0.8 * X + 0.5 + epsilon

# Ubah ke 2D untuk scikit-learn: (n_samples, 1)
X_2d = X.reshape(-1, 1)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="observasi", c="orange")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_6_0.png)

{{% notice info %}}
Di scikit-learn, fitur selalu berbentuk array 2D: baris = sampel, kolom = fitur. Gunakan <code>X.reshape(-1, 1)</code> untuk satu fitur.
{{% /notice %}}

## Periksa noise
Mari lihat distribusi <code>epsilon</code>.

```python
plt.figure(figsize=(10, 5))
plt.hist(epsilon, bins=30)
plt.xlabel("$\\epsilon$")
plt.ylabel("jumlah")
plt.show()
```

![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_8_0.png)

## Regresi linear (kuadrat terkecil) dengan scikit-learn
Kita gunakan <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html" target="_blank" rel="noopener">sklearn.linear_model.LinearRegression</a>.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()  # fit_intercept=True secara default
model.fit(X_2d, y)

print("kemiringan w:", model.coef_[0])
print("intersep b:", model.intercept_)

y_pred = model.predict(X_2d)

# Metrik
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("MSE:", mse)
print("R^2:", r2)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="observasi", c="orange")
plt.plot(X, y_pred, label="garis terpasang", c="C0")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_10_0.png)

{{% notice tip %}}
Penskalaan tidak wajib untuk OLS, tetapi membantu pada kasus multivariat dan regularisasi.
{{% /notice %}}

## Solusi bentuk tertutup (referensi)
Untuk <code>\(\hat y = wx + b\)</code>:

- <code>\(\displaystyle w = \frac{\operatorname{Cov}(x,y)}{\operatorname{Var}(x)}\)</code>
- <code>\(\displaystyle b = \bar y - w\,\bar x\)</code>

Verifikasi dengan NumPy:

```python
x_mean, y_mean = X.mean(), y.mean()
w_hat = ((X - x_mean) * (y - y_mean)).sum() / ((X - x_mean) ** 2).sum()
b_hat = y_mean - w_hat * x_mean
print(w_hat, b_hat)
```

## Jebakan umum
- Bentuk array: <code>X</code> harus <code>(n_samples, n_features)</code>. Untuk satu fitur, gunakan <code>reshape(-1, 1)</code>.
- Bentuk target: <code>y</code> bisa <code>(n_samples,)</code>. <code>(n,1)</code> juga berfungsi; perhatikan broadcasting.
- Intersep: default <code>fit_intercept=True</code>. Jika fitur dan target sudah dicenter, <code>False</code> juga baik.
- Reproducibility: gunakan seed tetap via <code>np.random.RandomState</code> atau <code>np.random.default_rng</code>.

## Lanjut (multivariat)
Untuk banyak fitur, pertahankan <code>X</code> sebagai <code>(n_samples, n_features)</code>. Pipeline memadukan pra-pemrosesan dan estimator.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_multi = rng.normal(size=(n_samples, 2))
y_multi = 1.0 * X_multi[:, 0] - 2.0 * X_multi[:, 1] + 0.3 + rng.normal(size=n_samples)

pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_multi, y_multi)
```

{{% notice note %}}
Kode untuk pembelajaran; gambar sudah dirender sebelumnya untuk situs.
{{% /notice %}}

