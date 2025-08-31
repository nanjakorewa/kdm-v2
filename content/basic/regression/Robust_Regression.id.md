---
title: "Outlier dan Robustness"
pre: "2.1.3 "
weight: 3
title_suffix: "Menangani dengan Regresi Huber"
---

{{% youtube "CrN5Si0379g" %}}


<div class="pagetop-box">
  <p><b>Outlier</b> adalah observasi yang sangat menyimpang dari mayoritas data. Apa itu outlier tergantung masalah, distribusi, dan skala target. Di sini kita bandingkan OLS (loss kuadrat) dengan <b>loss Huber</b> pada data yang mengandung titik ekstrem.</p>
  </div>

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

---

## 1. Mengapa OLS sensitif terhadap outlier

OLS meminimalkan jumlah kuadrat residual
$$
\text{RSS} = \sum_{i=1}^n (y_i - \hat y_i)^2.
$$
Karena residual di<b>kuadrat</b>kan, satu titik ekstrem saja dapat mendominasi loss dan <b>menyeret garis</b> hasil fitting menuju outlier.

---

## 2. Loss Huber: kompromi antara kuadrat dan absolut

<b>Loss Huber</b> memakai kuadrat untuk residual kecil dan absolut untuk residual besar. Untuk <code>r = y - \hat y</code> dan ambang <code>$\delta > 0$</code>:

$$
\ell_\delta(r) = \begin{cases}
\dfrac{1}{2}r^2, & |r| \le \delta \\
\delta\left(|r| - \dfrac{1}{2}\delta\right), & |r| > \delta.
\end{cases}
$$

Turunannya (pengaruh)
$$
\psi_\delta(r) = \frac{d}{dr}\ell_\delta(r) = \begin{cases}
r, & |r| \le \delta \\
\delta\,\mathrm{sign}(r), & |r| > \delta,
\end{cases}
$$
sehingga gradien <b>terklip</b> untuk residual besar (outlier).

> Catatan: di <code>HuberRegressor</code> scikit-learn, ambang disebut <code>epsilon</code> (sesuai <code>$\delta$</code> di atas).

---

## 3. Visualisasi bentuk loss

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
plt.xlabel("residual $r$")
plt.ylabel("loss")
plt.title("Kuadrat vs absolut vs Huber")
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_5_0.png)

---

## 4. Apa yang terjadi bila ada outlier? (data)

Buat masalah linear 2 fitur dan suntikkan <b>satu outlier ekstrem</b> pada <code>y</code>.

```python
np.random.seed(42)

N = 30
x1 = np.arange(N)
x2 = np.arange(N)
X = np.c_[x1, x2]                      # (N, 2)
epsilon = np.random.rand(N)            # noise [0, 1)
y = 5 * x1 + 10 * x2 + epsilon * 10

y[5] = 500  # satu outlier sangat besar

plt.figure(figsize=(8, 6))
plt.plot(x1, y, "ko", label="data")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Dataset dengan outlier")
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_7_0.png)

---

## 5. Bandingkan OLS vs Ridge vs Huber

- <b>OLS</b>: sangat sensitif terhadap outlier.  
- <b>Ridge</b> (L2): mengecilkan koefisien; sedikit lebih stabil, tetap terpengaruh.  
- <b>Huber</b>: mengklip pengaruh outlier; garis kurang terseret.

{{% notice document %}}
- [HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)
{{% /notice %}}

{{% notice seealso %}}
[Pendekatan praproses: label anomali (JP)](https://k-dm.work/ja/prep/numerical/add_label_to_anomaly/)
{{% /notice %}}

```python
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression

plt.figure(figsize=(8, 6))

# Huber: epsilon=3 untuk mengurangi pengaruh outlier
huber = HuberRegressor(alpha=0.0, epsilon=3.0)
huber.fit(X, y)
plt.plot(x1, huber.predict(X), "green", label="Huber")

# Ridge (L2). Dengan alpha≈0, mirip OLS
ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(X, y)
plt.plot(x1, ridge.predict(X), "orange", label="Ridge (α=1.0)")

# OLS
lr = LinearRegression()
lr.fit(X, y)
plt.plot(x1, lr.predict(X), "r-", label="OLS")

# data mentah
plt.plot(x1, y, "kx", alpha=0.7)

plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Dampak outlier pada garis hasil fitting")
plt.grid(alpha=0.3)
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_9_0.png)

Interpretasi:
- OLS (merah) sangat terseret oleh outlier.
- Ridge (oranye) sedikit meredakan namun tetap terpengaruh.
- Huber (hijau) mengurangi pengaruh outlier dan mengikuti tren keseluruhan lebih baik.

---

## 6. Parameter: epsilon dan alpha

- <code>epsilon</code> (ambang <code>$\delta$</code>):
  - Lebih besar → mendekati OLS; lebih kecil → mendekati loss absolut.
  - Bergantung skala residual; lakukan standardisasi atau robust scaling.
- <code>alpha</code> (penalti L2):
  - Menstabilkan koefisien; berguna pada kolinearitas.

Sensitivitas terhadap <code>epsilon</code>:

```python
from sklearn.metrics import mean_squared_error

for eps in [1.2, 1.5, 2.0, 3.0]:
    h = HuberRegressor(alpha=0.0, epsilon=eps).fit(X, y)
    mse = mean_squared_error(y, h.predict(X))
    print(f"epsilon={eps:>3}: MSE={mse:.3f}")
```

---

## 7. Catatan praktis

- <b>Penskalaan</b>: jika skala fitur/target berbeda, makna <code>epsilon</code> berubah; lakukan standardisasi atau robust scaling.
- <b>Titik leverage tinggi</b>: Huber robust untuk outlier vertikal di <code>y</code>, tidak selalu untuk titik ekstrem di <code>X</code>.
- <b>Pemilihan ambang</b>: atur <code>epsilon</code> dan <code>alpha</code> (misalnya <code>GridSearchCV</code>).
- <b>Evaluasi dengan CV</b>: jangan hanya melihat kecocokan pada data latih.

---

## 8. Ringkasan

- OLS sensitif terhadap outlier; garis hasil fitting dapat terseret.
- Huber memakai kuadrat untuk error kecil dan absolut untuk error besar, <b>mengklip gradien</b> untuk outlier.
- Menyetel <code>epsilon</code> dan <code>alpha</code> menyeimbangkan robustness dan kecocokan.
- Waspada titik leverage; kombinasikan dengan inspeksi dan praproses bila perlu.

---

