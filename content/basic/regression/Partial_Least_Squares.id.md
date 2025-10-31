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
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_linnerud
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_pls_latent_factors(
    cv_splits: int = 5,
    xlabel: str = "Number of latent factors",
    ylabel: str = "CV MSE (lower is better)",
    label_best: str = "best={k}",
    title: str | None = None,
) -> dict[str, object]:
    """Cross-validate PLS regression for different latent factor counts.

    Args:
        cv_splits: Number of folds for cross-validation.
        xlabel: Label for the number-of-factors axis.
        ylabel: Label for the cross-validation error axis.
        label_best: Format string for the best-factor annotation.
        title: Optional plot title.

    Returns:
        Dictionary with the selected factor count, CV score, and loadings.
    """
    japanize_matplotlib.japanize()
    data = load_linnerud()
    X = data["data"]
    y = data["target"][:, 0]

    max_components = min(X.shape[1], 6)
    components = np.arange(1, max_components + 1)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=0)

    scores = []
    pipelines = []
    for k in components:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("pls", PLSRegression(n_components=int(k))),
        ])
        cv_score = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
        ).mean()
        scores.append(cv_score)
        pipelines.append(model)

    scores_arr = np.array(scores)
    best_idx = int(np.argmax(scores_arr))
    best_k = int(components[best_idx])
    best_mse = float(-scores_arr[best_idx])

    best_model = pipelines[best_idx].fit(X, y)
    x_loadings = best_model["pls"].x_loadings_
    y_loadings = best_model["pls"].y_loadings_

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(components, -scores_arr, marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=label_best.format(k=best_k))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "best_k": best_k,
        "best_mse": best_mse,
        "x_loadings": x_loadings,
        "y_loadings": y_loadings,
    }



metrics = evaluate_pls_latent_factors(
    xlabel="Jumlah faktor laten",
    ylabel="MSE CV (semakin kecil semakin baik)",
    label_best="k terbaik={k}",
    title="Pemilihan faktor laten PLS",
)
print(f"Jumlah faktor laten terbaik: {metrics['best_k']}")
print(f"MSE CV terbaik: {metrics['best_mse']:.3f}")
print("Loading X:
", metrics['x_loadings'])
print("Loading Y:
", metrics['y_loadings'])

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
