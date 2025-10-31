---
title: "Support Vector Regression (SVR)"
pre: "2.1.10 "
weight: 10
title_suffix: "Prediksi tangguh dengan tabung ε-insensitive"
---

{{% summary %}}
- SVR memperluas SVM ke regresi dengan memperlakukan kesalahan di dalam tabung ε-insensitive sebagai nol sehingga dampak outlier berkurang.
- Metode kernel memungkinkan hubungan nonlinier yang fleksibel sambil menjaga model tetap ringkas melalui support vector.
- Hiperparameter `C`, `epsilon`, dan `gamma` mengatur keseimbangan antara kemampuan generalisasi dan kehalusan model.
- Penyetaraan skala fitur sangat penting; membungkus praproses dan pembelajaran dalam pipeline memastikan transformasi yang konsisten.
{{% /summary %}}

## Intuisi
SVR menyesuaikan sebuah fungsi yang dikelilingi tabung selebar ε: titik yang berada di dalam tabung tidak dikenai penalti, sedangkan titik di luar tabung dibebani penalti. Hanya titik yang menyentuh atau keluar dari tabung —support vector— yang memengaruhi model akhir, sehingga menghasilkan aproksimasi halus yang tahan terhadap noise.

## Formulasi matematis
Masalah optimisasinya adalah

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)
$$

dengan kendala

$$
\begin{aligned}
y_i - (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) &\le \epsilon + \xi_i, \\
(\mathbf{w}^\top \phi(\mathbf{x}_i) + b) - y_i &\le \epsilon + \xi_i^*, \\
\xi_i, \xi_i^* &\ge 0,
\end{aligned}
$$

di mana \(\phi\) memetakan input ke ruang fitur melalui kernel yang dipilih. Menyelesaikan bentuk dual menghasilkan support vector dan koefisiennya.

## Eksperimen dengan Python
Contoh berikut menunjukkan SVR yang dikombinasikan dengan `StandardScaler` dalam sebuah pipeline.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def run_svr_demo(
    *,
    n_samples: int = 300,
    train_size: float = 0.75,
    xlabel: str = "input x",
    ylabel: str = "output y",
    label_train: str = "train samples",
    label_test: str = "test samples",
    label_pred: str = "SVR prediction",
    label_truth: str = "ground truth",
    title: str | None = None,
) -> dict[str, float]:
    """Train SVR on synthetic nonlinear data, plot fit, and report metrics.

    Args:
        n_samples: Number of data points sampled from the underlying function.
        train_size: Fraction of data used for training.
        xlabel: X-axis label for the plot.
        ylabel: Y-axis label for the plot.
        label_train: Legend label for training samples.
        label_test: Legend label for test samples.
        label_pred: Legend label for the SVR prediction line.
        label_truth: Legend label for the ground-truth curve.
        title: Optional title for the plot.

    Returns:
        Dictionary containing training and test RMSE values.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(seed=42)

    X = np.linspace(0.0, 6.0, n_samples, dtype=float)
    y_true = np.sin(X) * 1.5 + 0.3 * np.cos(2 * X)
    y_noisy = y_true + rng.normal(scale=0.2, size=X.shape)

    X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(
        X[:, np.newaxis],
        y_noisy,
        y_true,
        train_size=train_size,
        random_state=42,
        shuffle=True,
    )

    svr = make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
    )
    svr.fit(X_train, y_train)

    train_pred = svr.predict(X_train)
    test_pred = svr.predict(X_test)

    grid = np.linspace(0.0, 6.0, 400, dtype=float)[:, np.newaxis]
    grid_truth = np.sin(grid.ravel()) * 1.5 + 0.3 * np.cos(2 * grid.ravel())
    grid_pred = svr.predict(grid)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_train, y_train, color="#1f77b4", alpha=0.6, label=label_train)
    ax.scatter(X_test, y_test, color="#ff7f0e", alpha=0.6, label=label_test)
    ax.plot(grid, grid_truth, color="#2ca02c", linewidth=2, label=label_truth)
    ax.plot(grid, grid_pred, color="#d62728", linewidth=2, linestyle="--", label=label_pred)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "train_rmse": float(mean_squared_error(y_train, train_pred, squared=False)),
        "test_rmse": float(mean_squared_error(y_test, test_pred, squared=False)),
    }



metrics = run_svr_demo(
    xlabel="masukan x",
    ylabel="keluaran y",
    label_train="data latih",
    label_test="data uji",
    label_pred="prediksi SVR",
    label_truth="fungsi asli",
    title="Regresi SVR pada data non-linear",
)
print(f"RMSE pelatihan: {metrics['train_rmse']:.3f}")
print(f"RMSE pengujian: {metrics['test_rmse']:.3f}")

```

### Cara membaca hasil
- Pipeline menstandarkan data latih menggunakan mean dan variansnya, lalu menerapkan transformasi yang sama ke data uji.
- `pred` berisi prediksi; pengaturan `epsilon` dan `C` mengubah trade-off antara overfitting dan underfitting.
- Nilai `gamma` yang lebih besar pada kernel RBF menekankan pola lokal, sedangkan nilai kecil menghasilkan fungsi yang lebih halus.

## Referensi
{{% references %}}
<li>Smola, A. J., &amp; Schölkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199–222.</li>
<li>Vapnik, V. (1995). <i>The Nature of Statistical Learning Theory</i>. Springer.</li>
{{% /references %}}
