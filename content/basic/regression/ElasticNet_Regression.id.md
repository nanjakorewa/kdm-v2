---
title: "Regresi Elastic Net"
pre: "2.1.5 "
weight: 5
title_suffix: "Menggabungkan kekuatan L1 dan L2"
---

{{% summary %}}
- Elastic Net menggabungkan penalti L1 (lasso) dan L2 (ridge) untuk menyeimbangkan sparsitas dan stabilitas.
- Kelompok fitur yang sangat berkorelasi dapat dipertahankan sekaligus menyesuaikan pentingnya secara kolektif.
- Menyetel \(\alpha\) dan `l1_ratio` dengan cross-validation memudahkan pencarian keseimbangan bias–varian.
- Menstandarkan fitur dan memberi iterasi yang cukup meningkatkan stabilitas numerik optimisasi.
{{% /summary %}}

## Intuisi
Lasso dapat membuat koefisien tepat nol dan melakukan seleksi fitur, tetapi jika fitur berkorelasi kuat ia mungkin hanya memilih satu dan membuang yang lain. Ridge mengecilkan koefisien secara halus dan stabil, namun tidak pernah membuatnya nol. Elastic Net menggabungkan kedua penalti sehingga fitur yang berkorelasi dapat dipertahankan bersama sementara koefisien yang tidak penting didorong mendekati nol.

## Formulasi matematis
Elastic Net meminimalkan

$$
\min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left( y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b) \right)^2 + \alpha \left( \rho \lVert \boldsymbol\beta \rVert_1 + (1 - \rho) \lVert \boldsymbol\beta \rVert_2^2 \right),
$$

dengan \(\alpha > 0\) sebagai kekuatan regularisasi dan \(\rho \in [0,1]\) (`l1_ratio`) mengontrol perbandingan L1/L2. Mengubah \(\rho\) dari 0 ke 1 memungkinkan kita menelusuri spektrum antara ridge dan lasso.

## Eksperimen dengan Python
Berikut penggunaan `ElasticNetCV` untuk memilih \(\alpha\) dan `l1_ratio` secara bersamaan, kemudian meninjau koefisien serta kinerjanya.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def run_elastic_net_demo(
    n_samples: int = 500,
    n_features: int = 30,
    n_informative: int = 10,
    noise: float = 15.0,
    duplicate_features: int = 5,
    label_scatter_x: str = "predicted",
    label_scatter_y: str = "actual",
    label_scatter_title: str = "Predicted vs. actual",
    label_bar_title: str = "Top coefficients",
    label_bar_ylabel: str = "coefficient",
    top_n: int = 10,
) -> dict[str, float]:
    """Fit Elastic Net with CV, report metrics, and plot predictions/coefs.

    Args:
        n_samples: Number of generated samples.
        n_features: Total features before duplication.
        n_informative: Features with non-zero weights in the generator.
        noise: Standard deviation of noise added to targets.
        duplicate_features: Number of leading features to duplicate for correlation.
        label_scatter_x: Label for the scatter plot x-axis.
        label_scatter_y: Label for the scatter plot y-axis.
        label_scatter_title: Title for the scatter plot.
        label_bar_title: Title for the coefficient bar plot.
        label_bar_ylabel: Y-axis label for the coefficient plot.
        top_n: Number of largest-magnitude coefficients to display.

    Returns:
        Dictionary with training/test metrics for inspection.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(seed=123)

    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        coef=True,
        random_state=123,
    )

    correlated = X[:, :duplicate_features] + rng.normal(
        scale=0.1, size=(X.shape[0], duplicate_features)
    )
    X = np.hstack([X, correlated])
    feature_names = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    enet_cv = ElasticNetCV(
        l1_ratio=[0.2, 0.5, 0.7, 0.9, 0.95, 1.0],
        alphas=np.logspace(-3, 1, 30),
        cv=5,
        random_state=42,
        max_iter=5000,
    )
    enet_cv.fit(X_train, y_train)

    enet = ElasticNet(
        alpha=float(enet_cv.alpha_),
        l1_ratio=float(enet_cv.l1_ratio_),
        max_iter=5000,
        random_state=42,
    )
    enet.fit(X_train, y_train)

    train_pred = enet.predict(X_train)
    test_pred = enet.predict(X_test)

    metrics = {
        "best_alpha": float(enet_cv.alpha_),
        "best_l1_ratio": float(enet_cv.l1_ratio_),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "test_rmse": float(mean_squared_error(y_test, test_pred, squared=False)),
    }

    top_idx = np.argsort(np.abs(enet.coef_))[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_scatter, ax_bar = axes

    ax_scatter.scatter(test_pred, y_test, alpha=0.6, color="#1f77b4")
    ax_scatter.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
    )
    ax_scatter.set_title(label_scatter_title)
    ax_scatter.set_xlabel(label_scatter_x)
    ax_scatter.set_ylabel(label_scatter_y)

    ax_bar.bar(
        np.arange(top_n),
        enet.coef_[top_idx],
        color="#ff7f0e",
    )
    ax_bar.set_xticks(np.arange(top_n))
    ax_bar.set_xticklabels(feature_names[top_idx], rotation=45, ha="right")
    ax_bar.set_title(label_bar_title)
    ax_bar.set_ylabel(label_bar_ylabel)

    fig.tight_layout()
    plt.show()

    return metrics



metrics = run_elastic_net_demo(
    label_scatter_x="prediksi",
    label_scatter_y="nilai aktual",
    label_scatter_title="Perbandingan prediksi vs aktual",
    label_bar_title="Koefisien terbesar",
    label_bar_ylabel="nilai koefisien",
)
print(f"Alpha terbaik: {metrics['best_alpha']:.4f}")
print(f"l1_ratio terbaik: {metrics['best_l1_ratio']:.2f}")
print(f"R^2 pelatihan: {metrics['train_r2']:.3f}")
print(f"R^2 pengujian: {metrics['test_r2']:.3f}")
print(f"RMSE pengujian: {metrics['test_rmse']:.3f}")

```

### Cara membaca hasil
- `ElasticNetCV` mengevaluasi berbagai kombinasi L1/L2 secara otomatis dan memilih keseimbangan yang baik.
- Ketika fitur yang berkorelasi bertahan bersama, koefisiennya cenderung memiliki besar yang serupa sehingga lebih mudah ditafsirkan.
- Jika konvergensi lambat, standarkan input atau tingkatkan `max_iter`.

## Referensi
{{% references %}}
<li>Zou, H., &amp; Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320.</li>
<li>Friedman, J., Hastie, T., &amp; Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. <i>Journal of Statistical Software</i>, 33(1), 1–22.</li>
{{% /references %}}
