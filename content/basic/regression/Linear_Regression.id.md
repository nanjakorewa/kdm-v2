---
title: "Metode kuadrat terkecil"
pre: "2.1.1 "
weight: 1
searchtitle: "Regresi Kuadrat Terkecil dalam Python"
---

<div class="pagetop-box">
    <p>Metode kuadrat terkecil mengacu pada pencarian koefisien fungsi untuk meminimalkan jumlah kuadrat residual ketika menyesuaikan fungsi ke kumpulan pasangan angka $(x_i, y_i)$ untuk mengetahui hubungan mereka. Pada halaman ini, kita akan mencoba melakukan metode least-squares pada data sampel menggunakan scikit-learn.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

{{% notice tip %}}
`japanize_matplotlib` diimpor untuk menampilkan bahasa Jepang dalam bagan.
{{% /notice %}}

## Membuat data regresi untuk eksperimen
Gunakan `np.linspace` untuk membuat data. Ini menciptakan daftar nilai dengan jarak yang sama di antara nilai yang Anda tentukan. Kode berikut ini membuat 500 sampel data untuk regresi linier.

```python
# Data pelatihan
n_samples = 500
X = np.linspace(-10, 10, n_samples)[:, np.newaxis]
epsolon = np.random.normal(size=n_samples)
y = np.linspace(-2, 2, n_samples) + epsolon

# Memvisualisasikan garis lurus
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="Target", c="orange")
plt.xlabel("$x_1$")
plt.xlabel("$y$")
plt.legend()
plt.show()
```

![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_6_0.png)


## Konfirmasi kebisingan pada y

Tentang `y = np.linspace(-2, 2, n_samples) + epsolon`, saya memplot histogram untuk `epsolon`.
Konfirmasikan bahwa noise dengan distribusi yang mendekati distribusi normal ada pada variabel target.

```python
plt.figure(figsize=(10, 5))
plt.hist(epsolon)
plt.xlabel("$\epsilon$")
plt.ylabel("#data")
plt.show()
```


    
![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_8_0.png)
    


## Mencocokkan garis lurus dengan metode kuadrat terkecil

{{% notice document %}}
[sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
{{% /notice %}}


```python
# pelatihan
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
y_pred = lin_r.predict(X)

# Memvisualisasikan garis lurus
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="target", c="orange")
plt.plot(X, y_pred, label="Garis lurus yang dipasang oleh regresi linier")
plt.xlabel("$x_1$")
plt.xlabel("$y$")
plt.legend()
plt.show()
```


    
![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_10_0.png)
    

