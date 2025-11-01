---
title: "Regresi robust | Menangani outlier dengan loss Huber"
linkTitle: "Regresi robust"
seo_title: "Regresi robust | Menangani outlier dengan loss Huber"
pre: "2.1.3 "
weight: 3
title_suffix: "Menangani outlier dengan loss Huber"
---

{{% summary %}}
- Metode kuadrat terkecil (OLS) sangat sensitif terhadap outlier karena residu dikuadratkan sehingga satu kesalahan besar dapat menyeret garis hasil.
- Loss Huber mempertahankan loss kuadrat untuk residu kecil dan beralih ke penalti linear untuk residu besar sehingga pengaruh titik ekstrem berkurang.
- Menyetel ambang \\(\delta\\) (epsilon di scikit-learn) serta penalti L2 opsional \\(\alpha\\) membantu menyeimbangkan ketahanan dan varians.
- Kombinasi penskalaan dan validasi silang menghasilkan model yang stabil pada data nyata yang sering mencampurkan titik normal dan anomali.
{{% /summary %}}

## Intuisi
Outlier dapat muncul akibat gangguan sensor, kesalahan input, atau perubahan pola. Dalam OLS, residu outlier yang dikuadratkan menjadi sangat besar sehingga garis regresi tertarik ke arah titik tersebut. Regresi robust memperlakukan residu besar dengan lebih lunak agar model mengikuti tren utama sambil mengabaikan observasi yang meragukan. Loss Huber adalah pilihan klasik: perilakunya kuadrat di sekitar nol dan absolut pada ekor distribusi.

## Formulasi matematis
Dengan residu \\(r = y - \hat{y}\\) dan ambang \\(\delta > 0\\), loss Huber didefinisikan sebagai

$$
\ell_\delta(r) =
\begin{cases}
\dfrac{1}{2} r^2, & |r| \le \delta, \\
\delta \bigl(|r| - \dfrac{1}{2}\delta\bigr), & |r| > \delta.
\end{cases}
$$

Residu kecil tetap berbentuk kuadrat, sedangkan residu besar tumbuh secara linear. Fungsi pengaruhnya pun jenuh:

$$
\psi_\delta(r) =
\begin{cases}
r, & |r| \le \delta, \\
\delta\,\mathrm{sign}(r), & |r| > \delta.
\end{cases}
$$

Dalam scikit-learn parameter `epsilon` setara dengan \\(\delta\\). Kita juga dapat menambahkan penalti L2 \\(\alpha \lVert \boldsymbol\beta \rVert_2^2\\) untuk menstabilkan koefisien saat fitur saling berkorelasi.

## Eksperimen dengan Python
Berikut visualisasi bentuk loss serta perbandingan OLS, Ridge, dan Huber pada dataset kecil yang mengandung satu outlier ekstrem.

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

### Loss Huber dibandingkan loss kuadrat dan absolut

```python
def huber_loss(r: np.ndarray, delta: float = 1.5):
    half_sq = 0.5 * np.square(r)
    lin = delta * (np.abs(r) - 0.5 * delta)
    return np.where(np.abs(r) <= delta, half_sq, lin)

delta = 1.5
r_vals = np.arange(-2, 2, 0.01)
h_vals = huber_loss(r_vals, delta=delta)

plt.figure(figsize=(8, 6))
plt.plot(r_vals, np.square(r_vals), "red",   label=r"kuadrat $r^2$")
plt.plot(r_vals, np.abs(r_vals),    "orange",label=r"absolut $|r|$")
plt.plot(r_vals, h_vals,            "green", label=fr"Huber ($\delta={delta}$)")
plt.axhline(0, color="k", linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel("residu $r$")
plt.ylabel("loss")
plt.title("Perbandingan loss kuadrat, absolut, dan Huber")
plt.show()
```

### Dataset sederhana dengan satu outlier

```python
np.random.seed(42)

N = 30
x1 = np.arange(N)
x2 = np.arange(N)
X = np.c_[x1, x2]
epsilon = np.random.rand(N)
y = 5 * x1 + 10 * x2 + epsilon * 10

y[5] = 500  # tambahkan satu outlier ekstrem

plt.figure(figsize=(8, 6))
plt.plot(x1, y, "ko", label="data")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Data dengan outlier")
plt.show()
```

### Membandingkan OLS, Ridge, dan Huber

```python
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression

plt.figure(figsize=(8, 6))

huber = HuberRegressor(alpha=0.0, epsilon=3.0)
huber.fit(X, y)
plt.plot(x1, huber.predict(X), "green", label="Huber")

ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(X, y)
plt.plot(x1, ridge.predict(X), "orange", label="Ridge (α=1.0)")

ols = LinearRegression()
ols.fit(X, y)
plt.plot(x1, ols.predict(X), "r-", label="OLS")

plt.plot(x1, y, "kx", alpha=0.7)
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Pengaruh outlier pada berbagai regresor")
plt.grid(alpha=0.3)
plt.show()
```

### Membaca hasil
- OLS (merah) sangat tertarik oleh outlier.
- Ridge (oranye) sedikit lebih stabil berkat penalti L2 tetapi masih terdampak.
- Huber (hijau) membatasi pengaruh outlier dan lebih mengikuti tren utama.

## Referensi
{{% references %}}
<li>Huber, P. J. (1964). Robust Estimation of a Location Parameter. <i>The Annals of Mathematical Statistics</i>, 35(1), 73–101.</li>
<li>Hampel, F. R. et al. (1986). <i>Robust Statistics: The Approach Based on Influence Functions</i>. Wiley.</li>
<li>Huber, P. J., &amp; Ronchetti, E. M. (2009). <i>Robust Statistics</i> (2nd ed.). Wiley.</li>
{{% /references %}}
