---
title: "Regresi Kuantil"
pre: "2.1.7 "
weight: 7
title_suffix: "Memperkirakan kontur distribusi kondisional"
---

{{% summary %}}
- Regresi kuantil mengestimasi kuantil arbitrer—misalnya median atau persentil ke-10—secara langsung alih-alih hanya rata-rata.
- Dengan meminimalkan loss pinball, model menjadi tahan terhadap outlier dan mampu menangani noise yang tidak simetris.
- Model untuk berbagai kuantil dapat dilatih secara terpisah dan digabungkan menjadi interval prediksi.
- Penyetaraan skala fitur dan regularisasi membantu menstabilkan konvergensi serta menjaga kemampuan generalisasi.
{{% /summary %}}

## Intuisi
Jika mínimos kuadrat memusatkan perhatian pada perilaku rata-rata, regresi kuantil bertanya “seberapa sering respons berada di bawah nilai ini?”. Pendekatan ini cocok untuk merencanakan skenario pesimistis, median, dan optimistis pada peramalan permintaan, atau menghitung Value at Risk, ketika rata-rata saja tidak cukup untuk mengambil keputusan.

## Formulasi matematis
Dengan residual \(r = y - \hat{y}\) dan level kuantil \(\tau \in (0, 1)\), loss pinball didefinisikan sebagai

$$
L_\tau(r) =
\begin{cases}
\tau r & (r \ge 0) \\
(\tau - 1) r & (r < 0)
\end{cases}
$$

Meminimalkan loss ini menghasilkan prediktor linear untuk kuantil \(\tau\). Ketika \(\tau = 0.5\) kita memperoleh median, identik dengan regresi deviasi absolut.

## Eksperimen dengan Python
Kita menggunakan `QuantileRegressor` untuk mengestimasi kuantil 0.1, 0.5, dan 0.9, lalu membandingkannya dengan mínimos kuadrat biasa.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(123)
n_samples = 400
X = np.linspace(0, 10, n_samples)
# Menambahkan noise yang tidak simetris
noise = rng.gamma(shape=2.0, scale=1.0, size=n_samples) - 2.0
y = 1.5 * X + 5 + noise
X = X[:, None]

taus = [0.1, 0.5, 0.9]
models = {}
for tau in taus:
    model = make_pipeline(
        StandardScaler(with_mean=True),
        QuantileRegressor(alpha=0.001, quantile=tau, solver="highs"),
    )
    model.fit(X, y)
    models[tau] = model

ols = LinearRegression().fit(X, y)

grid = np.linspace(0, 10, 200)[:, None]
preds = {tau: m.predict(grid) for tau, m in models.items()}
ols_pred = ols.predict(grid)

for tau, pred in preds.items():
    print(f"tau={tau:.1f}, prediksi minimum {pred.min():.2f}, maksimum {pred.max():.2f}")

plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=15, alpha=0.4, label="observasi")
colors = {0.1: "#1f77b4", 0.5: "#2ca02c", 0.9: "#d62728"}
for tau, pred in preds.items():
    plt.plot(grid, pred, color=colors[tau], linewidth=2, label=f"kuantil τ={tau}")
plt.plot(grid, ols_pred, color="#9467bd", linestyle="--", label="rata-rata (OLS)")
plt.xlabel("input X")
plt.ylabel("output y")
plt.legend()
plt.tight_layout()
plt.show()
```

![quantile-regression block 1](/images/basic/regression/quantile-regression_block01.svg)

### Cara membaca hasil
- Setiap kuantil menghasilkan garis berbeda yang menggambarkan sebaran vertikal data.
- Dibandingkan model rata-rata (OLS), regresi kuantil menyesuaikan diri terhadap noise yang miring.
- Menggabungkan beberapa kuantil menghasilkan interval prediksi yang menyampaikan ketidakpastian penting untuk pengambilan keputusan.

## Referensi
{{% references %}}
<li>Koenker, R., &amp; Bassett, G. (1978). Regression Quantiles. <i>Econometrica</i>, 46(1), 33–50.</li>
<li>Koenker, R. (2005). <i>Quantile Regression</i>. Cambridge University Press.</li>
{{% /references %}}
