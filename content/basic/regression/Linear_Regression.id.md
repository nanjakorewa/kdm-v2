---
title: "Regresi linear dan metode kuadrat terkecil"
pre: "2.1.1 "
weight: 1
title_suffix: "Memahami dari prinsip dasar"
---

{{% summary %}}
- Regresi linear memodelkan hubungan linier antara masukan dan keluaran dan menjadi landasan untuk prediksi maupun interpretasi.
- Metode kuadrat terkecil (ordinary least squares) menghitung koefisien dengan meminimalkan jumlah kuadrat residu sehingga menghasilkan solusi tertutup.
- Koefisien kemiringan menunjukkan seberapa besar keluaran berubah ketika masukan bertambah satu satuan, sedangkan intercept memberi nilai harapan saat masukan bernilai nol.
- Jika noise atau pencilan besar, kombinasikan standardisasi dan pendekatan robust agar praproses dan evaluasi tetap andal.
{{% /summary %}}

## Intuisi
Ketika plot sebar \((x_i, y_i)\) membentuk garis lurus, memperpanjang garis tersebut adalah cara alami untuk menginterpolasi masukan baru. Metode kuadrat terkecil menarik satu garis yang sedekat mungkin dengan seluruh titik dengan meminimalkan penyimpangan total antara observasi dan garis.

## Formulasi matematis
Model linear satu variabel ditulis sebagai

$$
y = w x + b.
$$

Dengan meminimalkan jumlah kuadrat residu \(\epsilon_i = y_i - (w x_i + b)\)

$$
L(w, b) = \sum_{i=1}^{n} \big(y_i - (w x_i + b)\big)^2,
$$

kita memperoleh solusi analitik

$$
w = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}, \qquad b = \bar{y} - w \bar{x},
$$

dengan \(\bar{x}\) dan \(\bar{y}\) adalah rata-rata \(x\) dan \(y\). Ide yang sama berlaku untuk regresi multivariat menggunakan vektor dan matriks.

## Eksperimen dengan Python
Contoh berikut menyesuaikan garis regresi menggunakan `scikit-learn` dan menampilkan hasilnya. Kodenya sama seperti versi bahasa Jepang sehingga gambar yang dihasilkan konsisten.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # opsional; tetap menampilkan label Jepang bila notebook dijalankan
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Membuat data
n_samples = 100
X = np.linspace(-5, 5, n_samples)[:, np.newaxis]
epsilon = np.random.normal(scale=2, size=n_samples)
y = 2 * X.ravel() + 1 + epsilon  # hubungan sebenarnya y = 2x + 1 + noise

# Melatih model dengan standardisasi opsional
lin_reg = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
y_pred = lin_reg.predict(X)

# Visualisasi
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="observasi", c="orange")
plt.plot(X, y_pred, label="regresi linear (OLS)")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

![linear-regression block 1](/images/basic/regression/linear-regression_block01.svg)

### Membaca hasil
- **Kemiringan \(w\)**: menunjukkan seberapa besar keluaran naik atau turun ketika masukan bertambah satu satuan; estimasinya seharusnya mendekati nilai sebenarnya.
- **Intercept \(b\)**: memberikan nilai harapan saat masukan bernilai 0 dan menggeser garis secara vertikal.
- Menstandardisasi fitur dengan `StandardScaler` membantu stabilitas ketika skala masukan berbeda-beda.

## Referensi
{{% references %}}
<li>Draper, N. R., &amp; Smith, H. (1998). <i>Applied Regression Analysis</i> (3rd ed.). John Wiley &amp; Sons.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
