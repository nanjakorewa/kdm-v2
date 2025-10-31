---
title: "Regresi Partial Least Squares (PLS)"
pre: "2.1.9 "
weight: 9
title_suffix: "Faktor laten yang selaras dengan target"
---

{{% summary %}}
- PLS mengekstrak faktor laten yang memaksimalkan kovarians antara prediktor dan target sebelum melakukan regresi.
- Berbeda dengan PCA, sumbu yang dipelajari memasukkan informasi target sehingga performa prediksi tetap terjaga saat dimensi dikurangi.
- Menyetel jumlah faktor laten menstabilkan model ketika terjadi multikolinearitas yang kuat.
- Melihat loading membantu mengidentifikasi kombinasi fitur yang paling berkaitan dengan target.
{{% /summary %}}

## Intuisi
Regresi komponen utama hanya memperhatikan varians fitur sehingga arah yang penting bagi target bisa terbuang. PLS membangun faktor laten sambil mempertimbangkan prediktor dan respon secara bersamaan, sehingga informasi yang paling berguna untuk prediksi tetap dipertahankan. Dengan demikian kita dapat menggunakan representasi ringkas tanpa kehilangan akurasi.

## Formulasi matematis
Dengan matriks prediktor \(\mathbf{X}\) dan vektor target \(\mathbf{y}\), PLS memperbarui skor laten \(\mathbf{t} = \mathbf{X} \mathbf{w}\) dan \(\mathbf{u} = \mathbf{y} c\) secara bergantian agar kovarians \(\mathbf{t}^\top \mathbf{u}\) maksimum. Pengulangan prosedur menghasilkan faktor laten tempat model linear

$$
\hat{y} = \mathbf{t} \boldsymbol{b} + b_0
$$

dipasang. Jumlah faktor \(k\) biasanya dipilih melalui cross-validation.

## Eksperimen dengan Python
Berikut perbandingan kinerja PLS untuk berbagai jumlah faktor laten menggunakan dataset kebugaran Linnerud.

```python
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_linnerud
import matplotlib.pyplot as plt
import japanize_matplotlib

data = load_linnerud()
X = data["data"]          # Rutinitas latihan dan ukuran tubuh
y = data["target"][:, 0]  # Di sini kita prediksi jumlah pull-up saja

components = range(1, min(X.shape[1], 6) + 1)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

scores = []
for k in components:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("pls", PLSRegression(n_components=k)),
    ])
    score = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error").mean()
    scores.append(score)

best_k = components[int(np.argmax(scores))]
print("Jumlah faktor laten terbaik:", best_k)

best_model = Pipeline([
    ("scale", StandardScaler()),
    ("pls", PLSRegression(n_components=best_k)),
]).fit(X, y)

print("Loading X:\n", best_model["pls"].x_loadings_)
print("Loading Y:\n", best_model["pls"].y_loadings_)

plt.figure(figsize=(8, 4))
plt.plot(components, [-s for s in scores], marker="o")
plt.axvline(best_k, color="red", linestyle="--")
plt.xlabel("Jumlah faktor laten")
plt.ylabel("CV MSE (lebih kecil lebih baik)")
plt.tight_layout()
plt.show()
```

![partial-least-squares block 1](/images/basic/regression/partial-least-squares_block01_id.png)

### Cara membaca hasil
- MSE hasil cross-validation menurun seiring penambahan faktor, mencapai minimum, lalu memburuk jika faktor ditambah terus.
- Meninjau `x_loadings_` dan `y_loadings_` menunjukkan fitur mana yang paling berkontribusi pada tiap faktor laten.
- Menstandarkan masukan memastikan fitur dengan skala berbeda tetap berkontribusi secara seimbang.

## Referensi
{{% references %}}
<li>Wold, H. (1975). Soft Modelling by Latent Variables: The Non-Linear Iterative Partial Least Squares (NIPALS) Approach. Dalam <i>Perspectives in Probability and Statistics</i>. Academic Press.</li>
<li>Geladi, P., &amp; Kowalski, B. R. (1986). Partial Least-Squares Regression: A Tutorial. <i>Analytica Chimica Acta</i>, 185, 1–17.</li>
{{% /references %}}
